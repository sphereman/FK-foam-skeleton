"""Shape-first FK foam: keep only the part of the FK graph that lies inside an
arbitrary closed mesh, and end the network exactly on that surface.

No boolean, no voxel remesh:
  * the FK graph (nodes + edges from FK_Foam_Edges) is filtered by an inside
    test against the shape (parity ray casting on a BVH tree);
  * nodes outside the shape are dropped;
  * an edge with one endpoint inside gets a new degree-1 vertex exactly on the
    surface - the generator then extrudes its outer triangle into a leg that
    stops on the shape;
  * the rest of the pipeline (tetra nodes, chamfers, bridges, edge bevel,
    end-face inset, subdivision) is unchanged.

The result is a foam skeleton whose own topology conforms to the shape, not a
lattice that was cut afterwards.
"""
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import fk_nodes


def _bvh(obj, depsgraph=None):
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    verts = [obj.matrix_world @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys, all_triangles=False, epsilon=0.0)


def _inside(bvh, p, eps=1e-6):
    """Parity test: cast a ray in +X and count the hits."""
    o = Vector((p[0], p[1], p[2]))
    d = Vector((1.0, 0.0, 0.0))
    hits, origin = 0, o
    for _ in range(64):
        loc, nor, idx, dist = bvh.ray_cast(origin, d)
        if loc is None:
            break
        hits += 1
        origin = loc + d * 1e-4
    return hits % 2 == 1


def _surface_hit(bvh, a, b):
    """First hit along a -> b (a outside, b inside)."""
    a = Vector(a)
    b = Vector(b)
    d = b - a
    L = d.length
    if L < 1e-9:
        return None
    d = d / L
    loc, nor, idx, dist = bvh.ray_cast(a, d, L)
    return None if loc is None else np.array(loc[:])


def clip_graph(src_obj, domain_obj, outside_margin=0.0):
    """Return (verts, edges) of the FK graph restricted to `domain_obj`."""
    bvh = _bvh(domain_obj)
    V = np.array([v.co[:] for v in src_obj.data.vertices])
    E = [tuple(e.vertices) for e in src_obj.data.edges]

    inside = np.array([_inside(bvh, p) for p in V])

    verts = [tuple(p) for p in V]
    src_index = {i: i for i in range(len(V))}
    edges = []
    new_verts = []

    def tip_vertex(p):
        i = len(verts) + len(new_verts)
        new_verts.append(tuple(p))
        return i

    for a, b in E:
        ia, ib = inside[a], inside[b]
        if ia and ib:
            edges.append((a, b))
        elif ia and not ib:
            hit = _surface_hit(bvh, V[a], V[b])
            if hit is not None:
                edges.append((a, tip_vertex(hit)))
        elif ib and not ia:
            hit = _surface_hit(bvh, V[b], V[a])
            if hit is not None:
                edges.append((b, tip_vertex(hit)))
        # both outside -> drop

    verts.extend(new_verts)
    return np.array(verts, float), edges


def build(ob, src_obj, domain_obj, **params):
    """Rebuild the foam skeleton so that it fills `domain_obj` exactly."""
    verts, edges = clip_graph(src_obj, domain_obj)
    # write the clipped graph into a temporary edge object, then reuse the
    # normal generator (it only needs vertices + edges)
    tmp = bpy.data.objects.get("_fk_graph_tmp")
    if tmp is not None:
        bpy.data.objects.remove(tmp, do_unlink=True)
    me = bpy.data.meshes.new("_fk_graph_tmp")
    me.from_pydata([tuple(v) for v in verts], edges, [])
    me.update()
    tmp = bpy.data.objects.new("_fk_graph_tmp", me)
    bpy.context.scene.collection.objects.link(tmp)
    try:
        st = fk_nodes.rebuild_object(ob, tmp, **params)
    finally:
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.meshes.remove(me)
    return st


def _proxy_tris(obj, target=400):
    """Low-poly proxy of the domain (Decimate, no remesh) for fast tests."""
    me = obj.data.copy()
    tmp = bpy.data.objects.new("_fk_proxy", me)
    bpy.context.scene.collection.objects.link(tmp)
    tmp.matrix_world = obj.matrix_world
    nt = sum(max(0, len(p_.vertices) - 2) for p_ in me.polygons)
    if nt > target:
        mod = tmp.modifiers.new("Dec", 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = max(0.02, float(target) / float(nt))
    bpy.context.view_layer.update()
    tris = _tris_world(tmp)
    bpy.data.objects.remove(tmp, do_unlink=True)
    return tris


def complete_cell_graph(src_obj, domain_obj, N=3, delta=0.0024802625, scale=10.0,
                        keep_ratio=0.6, proxy_tris=400):
    """Keep the A15 cells that fill `domain_obj`, every kept cell complete.

    A cell is kept when
      * its centre is inside the domain, and
      * at least `keep_ratio` of its vertices are inside
    (the domain is tested against a low-poly proxy, so this stays fast).
    """
    from . import fk_geom

    if delta is None:
        delta = fk_geom.solve_delta()
    cells = fk_geom.build_cells(delta)
    tris = _proxy_tris(domain_obj, proxy_tris)

    # cover the whole bounding box of the domain (no fixed 3x3x3 limit)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = domain_obj.evaluated_get(dg)
    me = ev.to_mesh()
    M = domain_obj.matrix_world
    bb = np.array([(M @ v.co)[:] for v in me.vertices], float)
    ev.to_mesh_clear()
    lo = bb.min(axis=0) / scale
    hi = bb.max(axis=0) / scale
    rng = [range(int(np.floor(lo[d])) - 1, int(np.ceil(hi[d])) + 2) for d in range(3)]

    inst = []
    for i in rng[0]:
        for j in rng[1]:
            for k in rng[2]:
                off = np.array([i, j, k], float)
                for cv, cf in cells:
                    pts = (np.asarray(cv, float) + off) * scale
                    inst.append((pts, cf))

    centres = np.array([pts.mean(axis=0) for pts, cf in inst])
    inside_centres = inside_many(centres, tris)

    allpts = np.vstack([pts for pts, cf in inst])
    ins = inside_many(allpts, tris)
    pos = 0
    keep = []
    for (pts, cf), c in zip(inst, inside_centres):
        n = len(pts)
        frac = float(ins[pos:pos + n].mean())
        pos += n
        keep.append(bool(c) and frac >= keep_ratio)

    verts, index, edges = [], {}, set()

    def vid(k, pos_):
        e = index.get(k)
        if e is None:
            e = len(verts)
            index[k] = e
            verts.append(tuple(pos_))
        return e

    kept = 0
    for (pts, cf), ok in zip(inst, keep):
        if not ok:
            continue
        kept += 1
        for f in cf:
            for a, b in zip(f, f[1:] + f[:1]):
                pa, pb = pts[a], pts[b]
                ia = vid(tuple(np.round(pa, 6)), pa)
                ib = vid(tuple(np.round(pb, 6)), pb)
                edges.add((min(ia, ib), max(ia, ib)))
    return np.array(verts, float), sorted(edges), kept


def _tris_world(obj):
    """World-space triangles of an object as an (T,3,3) array."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    M = obj.matrix_world
    vs = np.array([(M @ v.co)[:] for v in me.vertices], float)
    tris = []
    for poly in me.polygons:
        vi = list(poly.vertices)
        for a in range(1, len(vi) - 1):
            tris.append((vs[vi[0]], vs[vi[a]], vs[vi[a + 1]]))
    ev.to_mesh_clear()
    return np.array(tris, float)


def inside_many(points, tris, direction=(1.0, 0.0, 0.0)):
    """Vectorised parity ray test: True where the point is inside the mesh."""
    P = np.asarray(points, float).reshape(-1, 3)
    if len(tris) == 0:
        return np.zeros(len(P), bool)
    d = np.array(direction, float)
    V0 = tris[:, 0, :]
    E1 = tris[:, 1, :] - V0
    E2 = tris[:, 2, :] - V0
    h = np.cross(d[None, :], E2)
    a = np.einsum("tj,tj->t", E1, h)
    ok = np.abs(a) > 1e-12
    f = np.zeros_like(a)
    f[ok] = 1.0 / a[ok]
    s = P[:, None, :] - V0[None, :, :]
    u = f[None, :] * np.einsum("ntj,tj->nt", s, h)
    q = np.cross(s, E1[None, :, :])
    v = f[None, :] * np.einsum("ntj,j->nt", q, d)
    t = f[None, :] * np.einsum("ntj,tj->nt", q, E2)
    hit = ok[None, :] & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return (hit.sum(axis=1) % 2) == 1
