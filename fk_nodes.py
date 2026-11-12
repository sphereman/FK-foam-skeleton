"""Truncated-tetrahedron node network built on the FK foam edge graph.

For every four-valent node of the edge graph a tetrahedron is placed with its
four vertices on the four incident edges (distance `node_size` from the node).
Each tetrahedron vertex is chamfered by `bevel` (measured along the tetra
edges), which turns the tetrahedron into a truncated tetrahedron: 4 hexagons +
4 triangles.  The triangular faces of two neighbouring nodes that face each
other are then bridged with three quads, so the nodes and the bridges form one
continuous surface.

`master` multiplies both sizes, so a single value drives the whole node/strut
proportion; `node_size` and `bevel` tune the two independent ratios.
"""
import numpy as np
from collections import defaultdict


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _orient(pts, ref):
    """Return `pts` ordered so the polygon normal points along `ref`."""
    p = np.asarray(pts, float)
    c = p.mean(axis=0)
    n = np.zeros(3)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        n += np.cross(a - c, b - c)
    if np.dot(n, ref) < 0:
        return list(p[::-1])
    return list(p)


def build(V, edges, node_size=0.80, bevel=0.80, master=1.0, legs=True,
          clamp=0.45, sharp=0.35, leg_scale=1.0, cap_bevel=0.03,
          cap_cut=True, leg_chamfer=0.05, cap_inset=0.04, complete_nodes=False):
    """Return (verts Nx3, faces list, stats dict)."""
    V = np.asarray(V, float)
    edges = [tuple(e) for e in edges]
    if complete_nodes:
        # give every 2- or 3-connected vertex the missing arm(s) so that each
        # tetrahedron node is complete and every arm is extruded
        adj0 = defaultdict(list)
        for a, b in edges:
            adj0[int(a)].append(int(b))
            adj0[int(b)].append(int(a))
        extra = []
        for v, nb in list(adj0.items()):
            d = len(nb)
            if not (2 <= d <= 3):
                continue
            dirs = []
            lens = []
            for u in nb:
                dv = V[u] - V[v]
                L = float(np.linalg.norm(dv))
                if L > 1e-9:
                    dirs.append(dv / L)
                    lens.append(L)
            if not dirs:
                continue
            s = -np.sum(dirs, axis=0)
            if np.linalg.norm(s) < 1e-6:
                s = np.cross(dirs[0], [0.3, 0.5, 0.8])
            s = s / np.linalg.norm(s)
            Lm = float(np.mean(lens))
            for _ in range(4 - d):
                extra.append(V[v] + s * Lm)
                edges.append((v, len(V) + len(extra) - 1))
        if extra:
            V = np.vstack([V, np.array(extra, float)])
        adj = defaultdict(list)
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    deg = {i: len(v) for i, v in adj.items()}

    verts = []
    index = {}

    def vid(key, pos):
        i = index.get(key)
        if i is None:
            i = len(verts)
            index[key] = i
            verts.append(np.asarray(pos, float))
        return i

    s0 = node_size * master
    b0 = bevel * master

    nodes = {}
    for i, nb in adj.items():
        if deg[i] != 4:
            continue
        P = V[i]
        dirs = [_unit(V[j] - P) for j in nb]
        lens = [np.linalg.norm(V[j] - P) for j in nb]
        s_arm = [min(s0, clamp * L) for L in lens]
        s = min(s_arm)
        p = [P + sk * d for sk, d in zip(s_arm, dirs)]
        q = {}
        for k in range(4):
            q[k] = {}
            for l in range(4):
                if l == k:
                    continue
                e = p[l] - p[k]
                le = np.linalg.norm(e)
                bb = min(b0, clamp * le)
                q[k][l] = p[k] + bb * (e / le)
        nodes[i] = dict(P=P, p=p, q=q, s=s, nb=nb,
                        C=np.mean(np.array(p), axis=0))

    faces = []
    n_tri = n_hex = n_bridge = n_leg = 0
    rim = set()          # edges of the chamfered triangular faces: stay sharp
    cap_faces = []       # indices of the outer end faces
    leg_tips = []        # (tip point, axis) of every leg

    for i, nd in nodes.items():
        p, q, C = nd["p"], nd["q"], nd["C"]
        # --- four truncated corners -> triangles (bridged later, legs capped)
        for k in range(4):
            others = [l for l in range(4) if l != k]
            idx3 = [vid((i, k, l), q[k][l]) for l in others]
            for a, b in zip(idx3, idx3[1:] + idx3[:1]):
                rim.add((min(a, b), max(a, b)))
            if not legs and deg.get(nd["nb"][k], 0) != 4:
                faces.append(idx3)
                n_tri += 1
        # --- four hexagonal faces
        for a in range(4):
            for b_ in range(a + 1, 4):
                for c in range(b_ + 1, 4):
                    loop = [q[a][b_], q[b_][a], q[b_][c], q[c][b_], q[c][a], q[a][c]]
                    ref = np.mean(np.array(loop), axis=0) - C
                    idx = [vid((i, x, y), pt) for (x, y), pt in
                           zip([(a, b_), (b_, a), (b_, c), (c, b_), (c, a), (a, c)], loop)]
                    faces.append(idx)
                    n_hex += 1

    # --- bridges between neighbouring four-valent nodes
    done = set()
    for i, nd in nodes.items():
        for k, j in enumerate(nd["nb"]):
            if j not in nodes:
                continue
            e = (min(i, j), max(i, j))
            if e in done:
                continue
            done.add(e)
            nd2 = nodes[j]
            l = nd2["nb"].index(i)
            ko = [x for x in range(4) if x != k]
            lo = [x for x in range(4) if x != l]
            ra = [nd["q"][k][x] for x in ko]
            rb = [nd2["q"][l][x] for x in lo]
            # best rotational pairing, allowing the mirrored order
            best, bo, flip = None, 0, False
            for rev in (False, True):
                rr = rb[::-1] if rev else rb
                for o in range(3):
                    d = sum(np.linalg.norm(ra[t] - rr[(t + o) % 3]) for t in range(3))
                    if best is None or d < best:
                        best, bo, flip = d, o, rev
            rb2 = rb[::-1] if flip else rb
            lo2 = lo[::-1] if flip else lo
            ia = [vid((i, k, ko[t]), ra[t]) for t in range(3)]
            ib = [vid((j, l, lo2[(t + bo) % 3]), rb2[(t + bo) % 3]) for t in range(3)]
            axis = _unit(nd2["P"] - nd["P"])
            for t in range(3):
                u, v_ = t, (t + 1) % 3
                quad = [ia[u], ia[v_], ib[v_], ib[u]]
                mid = np.mean([verts[x] for x in quad], axis=0)
                ref = mid - (nd["P"] + np.dot(mid - nd["P"], axis) * axis)
                pts = [verts[x] for x in quad]
                faces.append([ia[u], ia[v_], ib[v_], ib[u]] if np.dot(
                    np.cross(pts[1] - pts[0], pts[2] - pts[1]), ref) >= 0
                    else [ia[v_], ia[u], ib[u], ib[v_]])
                n_bridge += 1

    # --- dangling edges: prism out to the tip, capped
    if legs:
        for i, nd in nodes.items():
            for k, j in enumerate(nd["nb"]):
                if deg.get(j, 0) == 4:
                    continue
                ko = [x for x in range(4) if x != k]
                ra = [nd["q"][k][x] for x in ko]
                off = leg_scale * (V[j] - nd["p"][k])
                ia = [vid((i, k, ko[t]), ra[t]) for t in range(3)]
                ib = [vid(("legE", j, t), ra[t] + off) for t in range(3)]
                for t in range(3):
                    u = (t + 1) % 3
                    faces.append([ia[t], ia[u], ib[u], ib[t]])
                faces.append([ib[2], ib[1], ib[0]])
                cap_faces.append(len(faces) - 1)
                for a_, b_ in ((ib[0], ib[1]), (ib[1], ib[2]), (ib[2], ib[0])):
                    rim.add((min(a_, b_), max(a_, b_)))
                axis = off / np.linalg.norm(off)
                leg_tips.append((np.mean(np.array(ra, float), axis=0) + off, axis))
                n_leg += 1
                n_tri += 1

    allpairs = set()
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            allpairs.add((min(a, b), max(a, b)))
    bevel_pairs = sorted(allpairs - rim)

    stats = dict(nodes=len(nodes), tris=n_tri, hexes=n_hex,
                 bridges=n_bridge // 3, legs=n_leg,
                 verts=len(verts), faces=len(faces),
                 node_size=s0, bevel=b0, bevel_edges=bevel_pairs,
                 cap_faces=cap_faces, leg_tips=leg_tips)
    return np.array(verts), faces, stats

def _apply_bevel(me, width, segments, want):
    import bpy
    scn = bpy.context.scene
    tmp = bpy.data.objects.new("_tmp_bevel", me.copy())
    scn.collection.objects.link(tmp)
    tm = tmp.data
    attr = tm.attributes.get("bevel_weight_edge")
    if attr is None:
        attr = tm.attributes.new("bevel_weight_edge", 'FLOAT', 'EDGE')
    attr.data.foreach_set("value",
                          [1.0 if tuple(sorted(e.vertices)) in want else 0.0
                           for e in tm.edges])
    m = tmp.modifiers.new("B", 'BEVEL')
    m.width = max(0.0, float(width))
    m.segments = max(1, int(segments))
    m.limit_method = 'WEIGHT'
    m.use_clamp_overlap = False
    m.miter_outer = 'MITER_ARC'
    bpy.context.view_layer.update()
    out = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(tmp, do_unlink=True)
    return out


def _apply_bevel(me, width, segments, want):
    import bpy
    scn = bpy.context.scene
    tmp = bpy.data.objects.new("_tmp_bevel", me.copy())
    scn.collection.objects.link(tmp)
    tm = tmp.data
    attr = tm.attributes.get("bevel_weight_edge")
    if attr is None:
        attr = tm.attributes.new("bevel_weight_edge", 'FLOAT', 'EDGE')
    attr.data.foreach_set("value",
                          [1.0 if tuple(sorted(e.vertices)) in want else 0.0
                           for e in tm.edges])
    m = tmp.modifiers.new("B", 'BEVEL')
    m.width = max(0.0, float(width))
    m.segments = max(1, int(segments))
    m.limit_method = 'WEIGHT'
    m.use_clamp_overlap = False
    m.miter_outer = 'MITER_ARC'
    bpy.context.view_layer.update()
    out = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(tmp, do_unlink=True)
    return out


def rebuild_object(ob, source, node_size=0.80, bevel=0.80, master=1.0,
                   legs=True, leg_scale=1.0, smooth_angle=30.0,
                   edge_bevel=0.05, bevel_segments=1, cap_inset=0.0,
                   subsurf=0, weld=None, leg_chamfer=None,
                   cap_bevel=0.0, cap_cut=True, complete_nodes=False):
    """Build the network, apply the edge bevel and (optionally) inset the
    hexagonal end faces.

    cap_inset = 0  -> the Edge Bevel stays a live modifier (step 2)
    cap_inset > 0  -> the bevel is baked, the end hexagons are inset (step 3)
    """
    import bpy
    import bmesh
    import math
    from mathutils.kdtree import KDTree

    V = np.array([v.co[:] for v in source.data.vertices])
    E = [tuple(e.vertices) for e in source.data.edges]
    verts, faces, st = build(V, E, node_size=node_size, bevel=bevel,
                             master=master, legs=legs, leg_scale=leg_scale,
                             leg_chamfer=leg_chamfer, complete_nodes=complete_nodes)
    me = ob.data
    me.clear_geometry()
    me.from_pydata([tuple(map(float, x)) for x in verts], [], faces)
    me.update()
    me.validate(verbose=False)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = False

    for a in list(me.attributes):
        if a.name == "bevel_weight_edge":
            me.attributes.remove(a)
    want = {tuple(sorted(x)) for x in st["bevel_edges"]}
    attr = me.attributes.new("bevel_weight_edge", 'FLOAT', 'EDGE')
    attr.data.foreach_set("value",
                          [1.0 if tuple(sorted(e.vertices)) in want else 0.0
                           for e in me.edges])
    me.update()

    for m in list(ob.modifiers):
        ob.modifiers.remove(m)
    bev = None
    if edge_bevel > 1e-6:
        bev = ob.modifiers.new("Edge Bevel", 'BEVEL')
        bev.limit_method = 'WEIGHT'
        bev.width = float(edge_bevel)
        bev.segments = max(1, int(bevel_segments))
        bev.profile = 0.5
        bev.use_clamp_overlap = True
        bev.miter_outer = 'MITER_ARC'

    if cap_inset > 1e-6 and st.get("leg_tips"):
        # bake the bevel, then inset the hexagonal end faces
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        baked = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        old = ob.data
        baked.name = "FK_Node_Network"
        ob.data = baked
        if old.users == 0:
            bpy.data.meshes.remove(old)
        for m in list(ob.modifiers):
            ob.modifiers.remove(m)
        me = baked
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        kd = KDTree(len(bm.faces))
        for f in bm.faces:
            kd.insert(f.calc_center_median(), f.index)
        kd.balance()
        targets = []
        for T, ax in st["leg_tips"]:
            co, idx, dist = kd.find(tuple(np.asarray(T, float).ravel()))
            f = bm.faces[idx]
            if f.normal.dot(tuple(np.asarray(ax, float).ravel())) > 0.35 and dist < 1.0:
                targets.append(f)
        for f in targets:
            vs = list(f.verts)
            m = f.calc_center_median()
            inner = []
            for v in vs:
                d = (v.co - m)
                r = d.length
                k = max(0.0, 1.0 - float(cap_inset) / r) if r > 1e-9 else 0.0
                inner.append(bm.verts.new(m + d * k))
            for q in range(len(vs)):
                u = vs[(q + 1) % len(vs)]
                iu = inner[(q + 1) % len(vs)]
                bm.faces.new([vs[q], u, iu, inner[q]])
            bm.faces.new(inner)
            bm.faces.remove(f)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        print("   inset %d end faces (%.3f mm)" % (len(targets), cap_inset))

    if subsurf and subsurf > 0:
        sub = ob.modifiers.new("Subdivision", 'SUBSURF')
        sub.levels = int(subsurf)
        sub.render_levels = int(subsurf)

    print("FK node network: base %d verts / %d faces, bevel %.3f (seg %d), inset %.3f"
          % (len(me.vertices), len(me.polygons), edge_bevel, bevel_segments, cap_inset))
    return st
