"""Frank-Kasper (A15 / Weaire-Phelan topology) periodic foam - geometry core.

Everything here is pure numpy so it can be tested outside Blender.

Idealisation used:
  * cell centres sit on the A15 (Cr3Si, Pm-3n) Frank-Kasper lattice:
      2a : (0,0,0), (1/2,1/2,1/2)
      6c : (1/4,0,1/2), (1/2,1/4,0), (0,1/2,1/4),
           (3/4,0,1/2), (1/2,3/4,0), (0,1/2,3/4)
  * the foam is the periodic LAGUERRE (power) tessellation of those
    centres; one free weight (2a vs 6c) is solved so that all eight
    cells have the same volume, i.e. a monodisperse dry foam.
"""
import numpy as np
from itertools import combinations

# ----------------------------------------------------------------- A15 ----
A15_SITES = np.array([
    [0.00, 0.00, 0.00],   # 2a
    [0.50, 0.50, 0.50],   # 2a
    [0.25, 0.00, 0.50],   # 6c
    [0.50, 0.25, 0.00],   # 6c
    [0.00, 0.50, 0.25],   # 6c
    [0.75, 0.00, 0.50],   # 6c
    [0.50, 0.75, 0.00],   # 6c
    [0.00, 0.50, 0.75],   # 6c
], float)
ORBIT = np.array([0, 0, 1, 1, 1, 1, 1, 1])   # 0 = 2a, 1 = 6c


def _images(p, cutoff=1.0, span=2):
    """All lattice images of the A15 sites within `cutoff` of point p."""
    out = []
    rng = range(-span, span + 1)
    for i in rng:
        for j in rng:
            for k in rng:
                shift = np.array([i, j, k], float)
                for si, s in enumerate(A15_SITES):
                    q = s + shift
                    d = np.linalg.norm(q - p)
                    if 1e-8 < d < cutoff:
                        out.append((d, si, q))
    out.sort(key=lambda t: t[0])
    return out


def _polytope(p, w_self, weights, cutoff=1.0):
    """Laguerre cell of site p as a closed polyhedron.

    Returns (verts, faces) with faces as lists of indices, wound CCW as
    seen from outside.
    """
    planes = []
    for _, si, q in _images(p, cutoff):
        n = q - p
        d = 0.5 * (q @ q - p @ p + w_self - weights[si])
        planes.append((n, d))
    N = len(planes)
    A = np.array([pl[0] for pl in planes])
    b = np.array([pl[1] for pl in planes])

    # vertex enumeration: intersections of plane triples that satisfy all
    cand = []
    idx = np.arange(N)
    for i, j, k in combinations(range(N), 3):
        M = A[[i, j, k]]
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        x = np.linalg.solve(M, b[[i, j, k]])
        cand.append(x)
    if not cand:
        raise RuntimeError("no candidate vertices")
    X = np.array(cand)

    # keep only vertices inside every half-space
    slack = X @ A.T - b[None, :]
    keep = np.max(slack, axis=1) < 1e-7
    X = X[keep]

    # de-duplicate
    key = np.round(X, 7)
    _, uniq = np.unique(key, axis=0, return_index=True)
    X = X[np.sort(uniq)]
    V = len(X)

    verts = [X[i] for i in range(V)]
    faces = []
    for pi in range(N):
        n, d = planes[pi]
        on = np.where(np.abs(X @ n - d) < 1e-6)[0]
        if len(on) < 3:
            continue
        pts = X[on]
        c = pts.mean(axis=0)
        # in-plane basis with e1 x e2 = n  -> increasing angle = CCW seen from +n
        e1 = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, [0.0, 1.0, 0.0])
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        e2 = e2 / np.linalg.norm(e2)
        ang = np.arctan2((pts - c) @ e2, (pts - c) @ e1)
        order = np.argsort(ang)
        loop = [int(on[i]) for i in order]
        if len(loop) >= 3:
            faces.append(loop)
    return verts, faces


def _volume(verts, faces):
    vol = 0.0
    for f in faces:
        p0 = verts[f[0]]
        for a, b in zip(f[1:-1], f[2:]):
            vol += np.dot(p0, np.cross(verts[a], verts[b])) / 6.0
    return vol


def _area(verts, faces):
    tot = 0.0
    for f in faces:
        p0 = verts[f[0]]
        for a, b in zip(f[1:-1], f[2:]):
            tot += 0.5 * np.linalg.norm(np.cross(verts[a] - p0, verts[b] - p0))
    return tot


def _stats(verts, faces):
    sides = {}
    for f in faces:
        sides[len(f)] = sides.get(len(f), 0) + 1
    edges = set()
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges.add((min(a, b), max(a, b)))
    return {
        "faces": len(faces),
        "verts": len(verts),
        "edges": len(edges),
        "sides": dict(sorted(sides.items())),
        "volume": _volume(verts, faces),
        "area": _area(verts, faces),
    }


def build_cells(delta):
    """All eight Laguerre cells of the A15 cell; delta = weight(2a)-weight(6c)."""
    weights = np.where(ORBIT == 0, float(delta), 0.0)
    cells = []
    for i, p in enumerate(A15_SITES):
        v, f = _polytope(p, weights[i], weights)
        cells.append((v, f))
    return cells


def cell_stats(delta, verbose=True):
    cells = build_cells(delta)
    out = []
    for i, (v, f) in enumerate(cells):
        st = _stats(v, f)
        st["site"] = i
        st["pos"] = A15_SITES[i].copy()
        out.append(st)
        if verbose:
            print("site %d  %-14s  F=%2d  V=%2d  E=%2d  sides=%s  vol=%.6f  area=%.5f"
                  % (i, np.round(A15_SITES[i], 2), st["faces"], st["verts"],
                     st["edges"], st["sides"], st["volume"], st["area"]))
    tv = sum(s["volume"] for s in out)
    print("total volume = %.8f  (should be 1.0)" % tv)
    return out


def solve_delta(lo=-0.4, hi=0.4, tol=1e-10):
    """Bisection so that the 2a and 6c cells have equal volume."""
    def f(d):
        cells = build_cells(d)
        return _volume(*cells[0]) - _volume(*cells[2])
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise RuntimeError("no bracket for delta")
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if fm * flo < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)
