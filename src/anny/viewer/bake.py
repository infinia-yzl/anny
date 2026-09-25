# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Ray-traced bakes of the fine body: ambient occlusion, thickness (for light through thin parts)
and curvature. Ported from the legacy 3D Model build (build/bake.py). They need Embree
(``embreex``).
"""

import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector


def vnormals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, T[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def edges(T):
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    return np.unique(np.sort(E, 1), axis=0)


def smooth_values(X, E, nV, iters=10, lam=0.5, mask=None):
    deg = np.bincount(E.ravel(), minlength=nV).astype(float)
    X = X.copy()
    for i in range(iters):
        s = np.zeros_like(X)
        np.add.at(s, E[:, 0], X[E[:, 1]])
        np.add.at(s, E[:, 1], X[E[:, 0]])
        avg = s / np.maximum(deg, 1).reshape((-1,) + (1,) * (X.ndim - 1))
        if mask is None:
            X = X + lam * (avg - X)
        else:
            X = X + lam * (avg - X) * mask.reshape((-1,) + (1,) * (X.ndim - 1))
    return X


def taubin(V, E, iters=20, lam=0.5, mu=-0.53):
    nV = len(V)
    for i in range(iters):
        V = smooth_values(V, E, nV, 1, lam)
        V = smooth_values(V, E, nV, 1, mu)
    return V


def cos_hemisphere(n, rng):
    u1 = rng.random(n)
    u2 = rng.random(n)
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], -1)


def frame(N):
    a = np.where(np.abs(N[:, 0:1]) < 0.9, np.array([[1, 0, 0]]), np.array([[0, 1, 0]]))
    t = np.cross(N, a)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(N, t)
    return t, b


def bake_ao(V, N, scene_mesh, nrays=48, maxdist=0.06, eps=0.0002, seed=1, chunk=200000):
    rng = np.random.default_rng(seed)
    inter = RayMeshIntersector(scene_mesh)
    t, b = frame(N)
    nV = len(V)
    occl = np.zeros(nV)
    dirs_local = cos_hemisphere(nrays, rng)
    for k in range(nrays):
        d = dirs_local[k]
        # rotate per-vertex with random twist to decorrelate
        ang = rng.random(nV) * 2 * np.pi
        c, s = np.cos(ang), np.sin(ang)
        x = d[0] * c - d[1] * s
        y = d[0] * s + d[1] * c
        D = t * x[:, None] + b * y[:, None] + N * d[2]
        origins = V + N * eps
        dist = np.full(nV, np.inf)
        for s0 in range(0, nV, chunk):
            sl = slice(s0, min(nV, s0 + chunk))
            locs, ri, ti = inter.intersects_location(
                origins[sl], D[sl], multiple_hits=False
            )
            if len(ri):
                dd = np.linalg.norm(locs - origins[sl][ri], axis=1)
                dist[s0 + ri] = dd
        w = np.clip(1 - dist / maxdist, 0, 1) ** 0.5
        w[~np.isfinite(dist)] = 0
        occl += w
    return 1 - occl / nrays


def bake_thickness(V, N, head_mesh, nrays=12, maxd=0.03, seed=2, cone=0.5):
    rng = np.random.default_rng(seed)
    inter = RayMeshIntersector(head_mesh)
    t, b = frame(N)
    nV = len(V)
    acc = np.zeros(nV)
    for k in range(nrays):
        u = rng.random(nV) * cone
        phi = rng.random(nV) * 2 * np.pi
        st = np.sqrt(u)
        D = (
            -(N * np.sqrt(1 - u)[:, None])
            + t * (st * np.cos(phi))[:, None]
            + b * (st * np.sin(phi))[:, None]
        )
        origins = V - N * 0.0002
        locs, ri, ti = inter.intersects_location(origins, D, multiple_hits=False)
        dist = np.full(nV, maxd)
        if len(ri):
            dist[ri] = np.minimum(np.linalg.norm(locs - origins[ri], axis=1), maxd)
        acc += dist
    return acc / nrays


def curvature(V, T, N, E=None):
    if E is None:
        E = edges(T)
    dp = V[E[:, 1]] - V[E[:, 0]]
    dn = N[E[:, 1]] - N[E[:, 0]]
    k = (dp * dn).sum(1) / np.maximum((dp * dp).sum(1), 1e-20)
    nV = len(V)
    s = np.zeros(nV)
    c = np.zeros(nV)
    np.add.at(s, E[:, 0], k)
    np.add.at(s, E[:, 1], k)
    np.add.at(c, E[:, 0], 1)
    np.add.at(c, E[:, 1], 1)
    return s / np.maximum(c, 1)


def bake_thickness_open(V, N, mesh, nrays=12, maxd=0.03, cont=0.04, seed=2, cone=0.5):
    """average distance through the surface along rays opposite the normal, counting only paths that come out
    into open space (a ray that exits into a closed cavity, such as the mouth or an eye socket, counts as maxd)"""
    rng = np.random.default_rng(seed)
    inter = RayMeshIntersector(mesh)
    t, b = frame(N)
    nV = len(V)
    acc = np.zeros(nV)
    for k in range(nrays):
        u = rng.random(nV) * cone
        phi = rng.random(nV) * 2 * np.pi
        st = np.sqrt(u)
        D = (
            -(N * np.sqrt(1 - u)[:, None])
            + t * (st * np.cos(phi))[:, None]
            + b * (st * np.sin(phi))[:, None]
        )
        origins = V - N * 0.0002
        locs, ri, ti = inter.intersects_location(origins, D, multiple_hits=False)
        dist = np.full(nV, maxd)
        if len(ri):
            d1 = np.linalg.norm(locs - origins[ri], axis=1)
            near = d1 < maxd
            ri, locs, d1 = ri[near], locs[near], d1[near]
            # continue from the exit point: the path is open when nothing else lies within `cont`
            O2 = locs + D[ri] * 0.0003
            l2, r2, _ = inter.intersects_location(O2, D[ri], multiple_hits=False)
            blocked = np.zeros(len(ri), bool)
            if len(r2):
                d2 = np.linalg.norm(l2 - O2[r2], axis=1)
                blocked[r2[d2 < cont]] = True
            dist[ri[~blocked]] = d1[~blocked]
        acc += dist
    return acc / nrays


# d'Eon & Luebke sum-of-gaussians skin diffusion profile (variance mm^2, rgb weights); the
# lookup table of the fallback skin shading (?sss=lut). Ported from build/lut.py.
LUT_VAR = np.array([0.0064, 0.0484, 0.187, 0.567, 1.99, 7.41])
LUT_W = np.array(
    [
        [0.233, 0.455, 0.649],
        [0.100, 0.336, 0.344],
        [0.118, 0.198, 0.0],
        [0.113, 0.007, 0.007],
        [0.358, 0.004, 0.0],
        [0.078, 0.0, 0.0],
    ]
)


def diffusion_profile(d):
    """d in mm, returns (..., 3)"""
    g = np.exp(-(d[..., None] ** 2) / (2 * LUT_VAR)) / (2 * np.pi * LUT_VAR)
    return g @ LUT_W


def make_lut(nu=256, nv=64, cmax=0.5):
    """u: NdotL in [-1, 1]; v: curvature c = 1/r in [0, cmax] (1/mm) mapped v = sqrt(c/cmax)"""
    lut = np.zeros((nv, nu, 3))
    for j in range(nv):
        v = (j + 0.5) / nv
        c = cmax * v * v
        r = 1.0 / max(c, 1e-4)
        xm = min(np.pi, 20.0 / r)
        x = np.linspace(-xm, xm, 4001)
        d = np.abs(2 * r * np.sin(x / 2))
        prof = diffusion_profile(d)
        norm = prof.sum(0)
        th = np.arccos(np.clip(-1 + 2 * (np.arange(nu) + 0.5) / nu, -1, 1))
        cosv = np.clip(np.cos(th[:, None] + x[None, :]), 0, None)
        lut[j] = (cosv @ prof) / norm
    return lut
