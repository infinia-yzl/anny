# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Medium tousled haircut: guide strands relaxed under gravity against the head, then interpolated,
cut and clumped. Ported from the legacy 3D Model build (build/hair2.py). The groom grows on the
fine body of anny's default body (``anny.viewer.geometry``) in the legacy frame of the
authoring rig, where the eyes and the floor sit where the legacy head had them.

Pipeline
  1. signed distance grid of the head (collision and scalp offsets)
  2. guide roots (Poisson disk) with a comb direction that flows from the crown whorl
  3. follow-the-leader relaxation: gravity, bending stiffness, collision and a layered volume shell
  4. render strands interpolated from guides, cut along a perimeter line (fringe, ears, nape) and a layer length map
  5. clumping, point-cut texture, noise and flyaways
"""

import time

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


MM = 0.001
C = np.array([0.0, 0.515, 0.035])  # cranium centre

# hairline: minimum elevation (deg) as a function of |azimuth| (deg; 0 = front, 90 = side, 180 = back)
HL_PHI = np.array(
    [
        0,
        20,
        32,
        45,
        56,
        62,
        65,
        67,
        70,
        73,
        76,
        79,
        84,
        90,
        96,
        100,
        104,
        110,
        114,
        120,
        132,
        150,
        165,
        180,
    ]
)
HL_EL = np.array(
    [
        33.5,
        33,
        32,
        29,
        23,
        16,
        8,
        -8,
        -16,
        -17,
        -4,
        6,
        6.5,
        5.5,
        3,
        0,
        -3,
        -9,
        -14,
        -22,
        -31,
        -35,
        -37.5,
        -38.5,
    ]
)


def sph(P):
    d = P - C
    r = np.linalg.norm(d, axis=1)
    phi = np.degrees(np.arctan2(d[:, 0], d[:, 2]))  # 0 front, +90 model's left (x > 0)
    el = np.degrees(np.arcsin(np.clip(d[:, 1] / r, -1, 1)))
    return phi, el, r


def hairline_el(phi):
    return np.interp(np.abs(phi), HL_PHI, HL_EL)


def on_ear(P):
    """points on the part of the ear that stands out from the head"""
    return (
        (np.abs(P[:, 0]) > 0.0712)
        & (P[:, 1] < 0.535)
        & (P[:, 2] > -0.01)
        & (P[:, 2] < 0.075)
    )


def tangent_of(v, N):
    v = np.asarray(v, float)
    if v.ndim == 1:
        v = np.broadcast_to(v / np.linalg.norm(v), N.shape)
    t = v - (v * N).sum(1, keepdims=True) * N
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)


def nrm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------------------------- signed distance
class SDF:
    def __init__(self, V, N, h=0.0015, lo=(-0.11, 0.39, -0.10), hi=(0.11, 0.67, 0.18)):
        self.h = h
        self.lo = np.array(lo)
        dims = np.ceil((np.array(hi) - self.lo) / h).astype(int) + 1
        self.dims = dims
        g = [self.lo[k] + np.arange(dims[k]) * h for k in range(3)]
        X, Y, Z = np.meshgrid(*g, indexing="ij")
        Q = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        tree = cKDTree(V)
        dist, idx = tree.query(Q, k=1, workers=-1)
        dv = Q - V[idx]
        s = (dv * N[idx]).sum(1)
        sd = np.where(dist < 0.004, s, dist * np.sign(s))
        self.grid = sd.reshape(dims).astype(np.float32)
        gx, gy, gz = np.gradient(gaussian_filter(self.grid, 0.7), h)
        self.gx, self.gy, self.gz = (
            gx.astype(np.float32),
            gy.astype(np.float32),
            gz.astype(np.float32),
        )

    @classmethod
    def load(cls, fn):
        z = np.load(fn)
        o = cls.__new__(cls)
        o.grid, o.gx, o.gy, o.gz = z["grid"], z["gx"], z["gy"], z["gz"]
        o.lo, o.h = z["lo"], float(z["h"])
        o.dims = np.array(o.grid.shape)
        return o

    def __call__(self, P, grad=True):
        c = ((P - self.lo) / self.h).T
        sd = map_coordinates(self.grid, c, order=1, mode="nearest")
        if not grad:
            return sd
        g = np.stack(
            [
                map_coordinates(a, c, order=1, mode="nearest")
                for a in (self.gx, self.gy, self.gz)
            ],
            1,
        )
        return sd, nrm(g)


# ---------------------------------------------------------------------------------------------- scalp sampling
def scalp_triangles(V, T):
    A = V[T[:, 0]]
    B = V[T[:, 1]]
    Cc = V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, Cc - A), axis=1)
    cen = (A + B + Cc) / 3
    phi, el, _ = sph(cen)
    ok = (el > hairline_el(phi) - 1.5) & (cen[:, 1] > 0.43) & ~on_ear(cen)
    return area * ok


def sample_on_mesh(V, T, N, prob, n, rng):
    p = prob / prob.sum()
    fi = rng.choice(len(T), n, p=p)
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    A = V[T[fi, 0]]
    B = V[T[fi, 1]]
    Cc = V[T[fi, 2]]
    P = A + (B - A) * u[:, None] + (Cc - A) * v[:, None]
    Nn = (
        N[T[fi, 0]] * (1 - u - v)[:, None]
        + N[T[fi, 1]] * u[:, None]
        + N[T[fi, 2]] * v[:, None]
    )
    return P, nrm(Nn)


def poisson_filter(P, r, rng):
    """greedy dart throwing on a random order: keep points at least r apart"""
    order = rng.permutation(len(P))
    tree = cKDTree(P)
    keep = np.zeros(len(P), bool)
    dead = np.zeros(len(P), bool)
    for i in order:
        if dead[i]:
            continue
        keep[i] = True
        dead[tree.query_ball_point(P[i], r)] = True
    return keep


def edge_density(P):
    """hair density ramps up over the first few degrees above the hairline"""
    phi, el, _ = sph(P)
    return smoothstep(-0.8, 3.0, el - hairline_el(phi))


# ---------------------------------------------------------------------------------------------- style fields
WHORL = None


def comb_dir(P, N):
    """direction the hair is combed at scalp points: away from the crown whorl with a clockwise spiral near it"""
    d = P - WHORL
    t = d - (d * N).sum(1, keepdims=True) * N
    tn = np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    t = t / tn
    sp = np.cross(N, t)
    near = np.exp(-((tn[:, 0] / 0.028) ** 2))
    t = nrm(t + sp * (1.9 * near + 0.12)[:, None])
    # sides and back fall downward; the front top keeps its forward flow toward the fringe
    phi, el, _ = sph(P)
    dn = tangent_of([0, -1.0, 0], N)
    low = smoothstep(55, 15, el) * smoothstep(35, 70, np.abs(phi))
    t = nrm(t + dn * (0.9 * low)[:, None])
    return t


def length_max(P):
    """layer length (m) before the perimeter cut"""
    phi, el, _ = sph(P)
    a = np.abs(phi)
    h = el - hairline_el(phi)
    L = np.full(len(P), 70.0)
    # fringe roots: long enough that the perimeter cut shapes the fringe
    front = smoothstep(52, 30, a) * smoothstep(48, 30, h)
    L = L + 25.0 * front
    # sides and back grow a little shorter toward the hairline (the perimeter cut does the rest)
    side = smoothstep(40, 60, a)
    L = L - side * 18.0 * smoothstep(30, 0, h)
    return L * MM


def perimeter_y(phi):
    """world height of the perimeter cut as a function of |azimuth|"""
    return np.interp(
        np.abs(phi),
        [0, 22, 36, 48, 58, 66, 74, 82, 92, 98, 106, 116, 128, 145, 162, 180],
        [
            0.5372,
            0.5375,
            0.5362,
            0.5320,
            0.5260,
            0.5200,
            0.5170,
            0.5165,
            0.5160,
            0.5060,
            0.4900,
            0.4760,
            0.4660,
            0.4600,
            0.4575,
            0.4570,
        ],
    )


# ---------------------------------------------------------------------------------------------- guides
def make_guides(sdf, V, T, N, rng, spacing=0.0042, M=25):
    prob = scalp_triangles(V, T)
    P, Nn = sample_on_mesh(V, T, N, prob, 60000, rng)
    keep = poisson_filter(P, spacing, rng)
    P, Nn = P[keep], Nn[keep]
    G = len(P)
    comb = comb_dir(P, Nn)
    # per-guide tousle: comb rotation, lift and length jitter
    ang = rng.normal(0, 0.28, G)
    b = np.cross(Nn, comb)
    comb = nrm(comb * np.cos(ang)[:, None] + b * np.sin(ang)[:, None])
    phi, el, _ = sph(P)
    h = el - hairline_el(phi)
    crown = np.exp(-((np.linalg.norm(P - WHORL, axis=1) / 0.035) ** 2))
    front = smoothstep(50, 25, np.abs(phi)) * smoothstep(20, 4, h)
    top = smoothstep(38, 55, el) * (1 - crown)
    alpha = np.radians(
        np.clip(
            58
            + 8 * rng.normal(0, 1, G)
            - 12 * top
            + 10 * crown
            - 10 * front
            + 12 * smoothstep(8, 0, h),
            25,
            82,
        )
    )
    d0 = nrm(Nn * np.cos(alpha)[:, None] + comb * np.sin(alpha)[:, None])
    L = length_max(P) * rng.uniform(0.94, 1.06, G) + 10 * MM
    # short hair below the perimeter (sideburns, nape): clipper length
    below = P[:, 1] < perimeter_y(phi) + 0.002
    L[below] = rng.uniform(9, 13, below.sum()) * MM
    info = dict(roots=P, normals=Nn, comb=comb, d0=d0, L=L, below=below)
    # initial shape: march along the comb direction hugging the scalp at a small offset
    X = np.zeros((G, M, 3))
    X[:, 0] = P
    seg = L / (M - 1)
    X[:, 1] = P + d0 * seg[:, None]
    d = d0.copy()
    for i in range(2, M):
        p = X[:, i - 1]
        sd, g = sdf(p)
        c = comb_dir(p - g * sd[:, None], g)
        c = nrm(c * np.cos(ang)[:, None] + np.cross(g, c) * np.sin(ang)[:, None])
        d = nrm(d + (c - d) * 0.35)
        X[:, i] = p + d * seg[:, None]
        sd, g = sdf(X[:, i])
        lowp = sd < 1.0 * MM
        X[lowp, i] += g[lowp] * (1.0 * MM - sd[lowp])[:, None]
    info["ang"] = ang
    return X, info


def relax(
    sdf,
    X,
    info,
    offs,
    band,
    iters=160,
    gravity=0.00022,
    stiff0=2.2,
    stiff1=0.25,
    comb_k=0.00016,
    rng=None,
):
    """follow-the-leader relaxation with gravity, stiffness toward the tip, collision and a scalp band"""
    G, M, _ = X.shape
    seg = info["L"] / (M - 1)
    d0 = info["d0"]
    root = X[:, 0].copy()
    X = X.copy()
    X[:, 1] = root + d0 * seg[:, None]
    Xp = X.copy()
    s = np.linspace(0, 1, M)
    stiff = stiff0 + (stiff1 - stiff0) * s**0.7
    gvec = np.array([0, -1.0, 0])
    for it in range(iters):
        vel = (X - Xp) * 0.6
        Xp = X.copy()
        X[:, 2:] += vel[:, 2:] + gvec * gravity * (seg[:, None, None] / 0.004)
        if comb_k > 0 and it % 2 == 0:
            flat = X[:, 2:].reshape(-1, 3)
            sd, g = sdf(flat)
            c = comb_dir(flat - g * sd[:, None], g)
            a = np.repeat(info["ang"], M - 2)
            c = nrm(c * np.cos(a)[:, None] + np.cross(g, c) * np.sin(a)[:, None])
            wgt = np.tile(1.0 - 0.65 * s[2:], G)
            X[:, 2:] += (c * (comb_k * 2 * wgt)[:, None]).reshape(G, M - 2, 3) * (
                seg[:, None, None] / 0.004
            )
        for i in range(2, M):
            dcur = X[:, i] - X[:, i - 1]
            dprev = nrm(X[:, i - 1] - X[:, i - 2])
            dnew = nrm(nrm(dcur) + stiff[i] * dprev)
            X[:, i] = X[:, i - 1] + dnew * seg[:, None]
        flat = X[:, 2:].reshape(-1, 3)
        o = offs[:, 2:].reshape(-1)
        bd = band[:, 2:].reshape(-1)
        sd, g = sdf(flat)
        lo_ = sd < o
        flat[lo_] += g[lo_] * (o[lo_] - sd[lo_])[:, None]
        hi_ = sd > o + bd
        flat[hi_] -= g[hi_] * ((sd[hi_] - o[hi_] - bd[hi_]) * 0.35)[:, None]
        X[:, 2:] = flat.reshape(G, M - 2, 3)
    # final hard collision + length fix
    for i in range(2, M):
        dnew = nrm(X[:, i] - X[:, i - 1])
        X[:, i] = X[:, i - 1] + dnew * seg[:, None]
    return X


def sway(sdf, X, info, offs, rng, amount=0.18):
    """rotate the far part of each guide about its root normal, then settle it back onto its layer"""
    G, M, _ = X.shape
    th = rng.normal(0, amount, G)
    s = np.linspace(0, 1, M)
    n0 = info["normals"]
    root = X[:, :1]
    rel = X - root
    out = X.copy()
    for i in range(1, M):
        a = th * s[i] ** 1.6
        c, sn = np.cos(a)[:, None], np.sin(a)[:, None]
        v = rel[:, i]
        out[:, i] = (
            root[:, 0]
            + v * c
            + np.cross(n0, v) * sn
            + n0 * (n0 * v).sum(1, keepdims=True) * (1 - c)
        )
    seg = info["L"] / (M - 1)
    for it in range(3):
        for i in range(2, M):
            out[:, i] = out[:, i - 1] + nrm(out[:, i] - out[:, i - 1]) * seg[:, None]
        flat = out[:, 2:].reshape(-1, 3)
        o = offs[:, 2:].reshape(-1)
        sd, g = sdf(flat)
        lo_ = sd < o
        flat[lo_] += g[lo_] * (o[lo_] - sd[lo_])[:, None]
        out[:, 2:] = flat.reshape(G, M - 2, 3)
    return out


def layer_offsets(X, info, base=0.6 * MM, per_guide=1.25 * MM, cell=2.0):
    """volume: each guide rests on the hair that lies below it (guides further from the crown lie lower)"""
    G, M, _ = X.shape
    dist_w = np.linalg.norm(info["roots"] - WHORL, axis=1)
    order = np.argsort(-dist_w)  # farthest from the whorl first = bottom layer
    nphi, nel = int(360 / cell), int(180 / cell)
    Tk = np.zeros((nphi, nel))
    phi, el, _ = sph(X.reshape(-1, 3))
    ci = np.clip(((phi + 180) / cell).astype(int), 0, nphi - 1).reshape(G, M)
    cj = np.clip(((el + 90) / cell).astype(int), 0, nel - 1).reshape(G, M)
    offs = np.zeros((G, M))
    ker = np.array([[0.05, 0.12, 0.05], [0.12, 0.32, 0.12], [0.05, 0.12, 0.05]])
    ker /= ker.sum()
    s = np.linspace(0, 1, M)
    for g in order:
        offs[g] = base + Tk[ci[g], cj[g]]
        # splat this guide's ribbon thickness along its path (skip the root segment)
        for j in range(1, M):
            a, b = ci[g, j], cj[g, j]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    Tk[(a + di) % nphi, min(max(b + dj, 0), nel - 1)] += (
                        per_guide * ker[di + 1, dj + 1]
                    )
    # root keeps its own emergence, grow the offset smoothly from the root
    offs = offs * smoothstep(0.0, 0.25, s)[None, :] + base
    return offs


def hair_ao(strands):
    """density-based occlusion: how much hair lies outward of each point"""
    pts = strands.reshape(-1, 3)
    lo = pts.min(0) - 0.01
    hi = pts.max(0) + 0.01
    h = 0.0015
    dims = np.ceil((hi - lo) / h).astype(int) + 1
    idx = np.floor((pts - lo) / h).astype(int)
    grid = np.zeros(dims, np.float32)
    np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    grid = gaussian_filter(grid, 1.0)
    dirv = nrm(pts - C)
    acc = np.zeros(len(pts))
    for st in range(1, 14):
        q = pts + dirv * (st * h)
        qi = np.floor((q - lo) / h).astype(int)
        ok = np.all((qi >= 0) & (qi < dims), axis=1)
        v = np.zeros(len(pts))
        v[ok] = grid[qi[ok, 0], qi[ok, 1], qi[ok, 2]]
        acc += v
    return acc.reshape(strands.shape[:2])


def arclen(S):
    seg = np.linalg.norm(np.diff(S, axis=1), axis=2)
    return np.concatenate([np.zeros((len(S), 1)), np.cumsum(seg, 1)], 1)


def resample(S, s_end, M_out):
    """resample each polyline from arc length 0 to s_end into M_out points"""
    cum = arclen(S)
    n, M, _ = S.shape
    u = np.linspace(0, 1, M_out)[None, :] * s_end[:, None]
    out = np.zeros((n, M_out, 3))
    j = np.clip(np.array([np.searchsorted(cum[i], u[i]) for i in range(n)]), 1, M - 1)
    r = np.arange(n)[:, None]
    c0 = cum[r, j - 1]
    c1 = cum[r, j]
    f = np.clip((u - c0) / np.maximum(c1 - c0, 1e-9), 0, 1)[..., None]
    out = S[r, j - 1] * (1 - f) + S[r, j] * f
    return out


def clump_pull(S, P, n_clumps, strength, power, rng, weights=None):
    n, M, _ = S.shape
    cidx = rng.choice(n, n_clumps, replace=False)
    _, cl = cKDTree(P[cidx]).query(P)
    cnt = np.bincount(cl, minlength=n_clumps).astype(float)
    cm = np.zeros((n_clumps, M, 3))
    for k in range(3):
        for i in range(M):
            cm[:, i, k] = np.bincount(
                cl, weights=S[:, i, k], minlength=n_clumps
            ) / np.maximum(cnt, 1)
    s = np.linspace(0, 1, M)
    cs = rng.uniform(*strength, n_clumps)[cl]
    pw = rng.uniform(*power, n_clumps)[cl]
    w = cs[:, None] * s[None, :] ** pw[:, None]
    if weights is not None:
        w = w * weights[:, None]
    S = S + (cm[cl] - S) * w[..., None]
    return S, cl, cm


def interp_from_guides(P, X, info, offs, use):
    """shape of each render strand from the nearest guides of the same class that flow the same way"""
    G, M, _ = X.shape
    gi = np.where(use)[0]
    dist, loc = cKDTree(info["roots"][gi]).query(P, k=4)
    idx = gi[loc]
    gdir = nrm(X[:, M // 3] - X[:, 0])
    sim = (gdir[idx] * gdir[idx[:, :1]]).sum(2)
    w = (1.0 / (dist + 0.0012) ** 2) * (sim > 0.55)
    w /= w.sum(1, keepdims=True)
    n = len(P)
    S = np.zeros((n, M, 3))
    Os = np.zeros((n, M))
    for k in range(4):
        S += w[:, k, None, None] * (X[idx[:, k]] - X[idx[:, k], :1])
        Os += w[:, k, None] * offs[idx[:, k]]
    S += P[:, None, :]
    Lg = (w * info["L"][idx]).sum(1)
    return S, Os, Lg


def settle(sdf, S, Os, iters=2):
    n, M, _ = S.shape
    for it in range(iters):
        flat = S[:, 1:].reshape(-1, 3)
        o = Os[:, 1:].reshape(-1)
        sd, g = sdf(flat)
        lo_ = sd < o
        flat[lo_] += g[lo_] * (o[lo_] - sd[lo_])[:, None]
        S[:, 1:] = flat.reshape(n, M - 1, 3)
    return S


def long_strands(sdf, P, Nn, S, Os, Lg, rng):
    """pieces, clumps and cuts for the hair above the perimeter"""
    n, M, _ = S.shape
    sM = np.linspace(0, 1, M)
    # pieces: groups of strands that swing, lift, clump and get cut together (the tousled texture)
    n_piece = n // 60
    pidx = rng.choice(n, n_piece, replace=False)
    _, piece = cKDTree(P[pidx]).query(P)
    th = rng.normal(0, 0.26, n_piece)[piece]
    lift = (rng.random(n_piece) ** 2 * 5.0 * MM)[piece]
    root = S[:, :1]
    rel = S - root
    for i in range(1, M):
        a = th * sM[i] ** 1.4
        c, sn = np.cos(a)[:, None], np.sin(a)[:, None]
        v = rel[:, i]
        S[:, i] = (
            root[:, 0]
            + v * c
            + np.cross(Nn, v) * sn
            + Nn * (Nn * v).sum(1, keepdims=True) * (1 - c)
        )
    Os = Os + lift[:, None] * smoothstep(0.15, 0.7, sM)[None, :]
    S = settle(sdf, S, Os, 3)
    # piece clumping: strands gather early into a piece and meet at a point near the tip
    cnt = np.bincount(piece, minlength=n_piece).astype(float)
    cmP = np.zeros((n_piece, M, 3))
    for k in range(3):
        for i in range(M):
            cmP[:, i, k] = np.bincount(
                piece, weights=S[:, i, k], minlength=n_piece
            ) / np.maximum(cnt, 1)
    cs = rng.uniform(0.5, 0.82, n_piece)[piece]
    prof = (
        cs[:, None] * smoothstep(0.0, 0.5, sM)[None, :] ** 0.9
        + (1 - cs[:, None]) * 0.55 * sM[None, :] ** 2.2
    )
    S = S + (cmP[piece] - S) * prof[..., None]
    S, cl, cm = clump_pull(S, P, n // 12, (0.3, 0.65), (1.0, 1.5), rng)
    # cuts: layer length per piece, thinning, and a point-cut perimeter
    cum = arclen(S)
    total = cum[:, -1]
    L_layer = np.minimum(
        length_max(P)
        * rng.uniform(0.72, 1.05, n_piece)[piece]
        * rng.uniform(0.95, 1.03, n),
        Lg,
    )
    thin = rng.random(n) < 0.28
    L_layer[thin] *= rng.uniform(0.55, 0.85, thin.sum())
    a_p = np.abs(sph(P[pidx])[0])
    amp_p = np.interp(a_p, [0, 40, 60, 100, 130, 180], [7.0, 8.0, 7.0, 7.0, 13.0, 14.0])
    tex_p = rng.uniform(0, 1, n_piece) * amp_p * MM * (rng.random(n_piece) < 0.75)
    longer = (rng.random(n_piece) < 0.2) & (a_p > 50)
    tex_p[longer] = -rng.uniform(1.5, 4.0, longer.sum()) * MM
    r_tip = np.linalg.norm(S[:, -1] - cm[cl][:, -1], axis=1)
    tex = tex_p[piece] + np.abs(rng.normal(0, 1.5, n)) * MM + 1.2 * r_tip
    tex[thin] += rng.uniform(4, 14, thin.sum()) * MM
    phiS = sph(S.reshape(-1, 3))[0].reshape(n, M)
    ycut = perimeter_y(phiS) + tex[:, None]
    under = S[..., 1] < ycut
    under[:, 0] = False
    has = under.any(1)
    j = np.argmax(under, axis=1)
    r = np.arange(n)
    jj = np.clip(j, 1, M - 1)
    y0 = S[r, jj - 1, 1] - ycut[r, jj - 1]
    y1 = S[r, jj, 1] - ycut[r, jj]
    f = np.clip(y0 / np.maximum(y0 - y1, 1e-9), 0, 1)
    s_per = cum[r, jj - 1] * (1 - f) + cum[r, jj] * f
    s_end = np.where(has, np.minimum(s_per, L_layer), L_layer)
    s_end = np.clip(np.minimum(s_end, total), 2.5 * MM, None)
    at_edge = has & (s_per <= L_layer + 1e-6)
    flick = (rng.random(n_piece) < 0.45)[piece] & at_edge & (np.abs(sph(P)[0]) > 50)
    famp = rng.uniform(1.5, 4.5, n_piece)[piece] * MM
    return S, s_end, cl, flick, famp


def short_strands(sdf, P, S, Os, Lg, rng):
    """clipper-length hair below the perimeter: sideburns, behind the ears and the nape"""
    n, M, _ = S.shape
    S, cl, cm = clump_pull(S, P, max(n // 10, 1), (0.15, 0.4), (1.0, 1.5), rng)
    total = arclen(S)[:, -1]
    s_end = np.clip(np.minimum(Lg * rng.uniform(0.8, 1.1, n), total), 2.0 * MM, None)
    return S, s_end, cl


def render_strands(
    sdf, V, T, N, X, info, offs, rng, n_target=56000, M_out=24, sdf_full=None
):
    G, M, _ = X.shape
    # --- roots (a little denser around the crown whorl)
    prob = scalp_triangles(V, T)
    cen = V[T].mean(1)
    prob = prob * (
        1.0 + 0.7 * np.exp(-((np.linalg.norm(cen - WHORL, axis=1) / 0.022) ** 2))
    )
    P, Nn = sample_on_mesh(V, T, N, prob, int(n_target * 2.2), rng)
    keep = rng.random(len(P)) < edge_density(P)
    P, Nn = P[keep], Nn[keep]
    pairs = cKDTree(P).query_pairs(0.00030, output_type="ndarray")
    drop = np.zeros(len(P), bool)
    drop[pairs[:, 1]] = True
    P, Nn = P[~drop], Nn[~drop]
    if len(P) > n_target:
        sel = rng.choice(len(P), n_target, replace=False)
        P, Nn = P[sel], Nn[sel]
    n = len(P)
    phi_r = sph(P)[0]
    short = P[:, 1] < perimeter_y(phi_r) + 0.002
    # --- two classes, each interpolated only from its own guides
    S_all = np.zeros((n, M_out, 3))
    L_all = np.zeros(n)
    cl_all = np.zeros(n, int)
    for cls in (False, True):
        m = short == cls
        if not m.any():
            continue
        S, Os, Lg = interp_from_guides(P[m], X, info, offs, info["below"] == cls)
        S = settle(sdf, S, Os, 2)
        if cls:
            S, s_end, cl = short_strands(sdf, P[m], S, Os, Lg, rng)
            cl = cl + 10_000_000
        else:
            S, s_end, cl, flick, famp = long_strands(sdf, P[m], Nn[m], S, Os, Lg, rng)
        S = resample(S, s_end, M_out)
        if not cls:
            # flicks: the ends of some perimeter pieces turn outward, which breaks up the helmet outline
            fi = np.where(flick)[0]
            if len(fi):
                sO = np.linspace(0, 1, M_out)
                _, gF = sdf(S[fi].reshape(-1, 3))
                gF = gF.reshape(len(fi), M_out, 3)
                bend = smoothstep(0.6, 1.0, sO) ** 2
                S[fi] += gF * (famp[fi][:, None] * bend[None, :])[..., None]
        S_all[m] = S
        L_all[m] = s_end
        cl_all[m] = cl
    S = S_all
    # --- small-scale waviness and flyaways
    s = np.linspace(0, 1, M_out)
    tng = nrm(np.gradient(S, axis=1))
    ref = np.where(
        np.abs(tng[..., 1:2]) < 0.9, np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
    )
    b1 = nrm(np.cross(tng, ref))
    b2 = nrm(np.cross(tng, b1))
    ph1, ph2 = rng.uniform(0, 6.3, (2, n))
    fr = rng.uniform(1.0, 2.6, n)
    amp = rng.uniform(0.15, 0.55, n) * MM * np.clip(L_all / (30 * MM), 0.2, 1.0)
    wav = (
        np.sin(2 * np.pi * fr[:, None] * s[None] + ph1[:, None])[..., None] * b1
        + np.sin(2 * np.pi * fr[:, None] * 0.7 * s[None] + ph2[:, None])[..., None] * b2
    )
    S = S + wav * (amp[:, None] * s[None] ** 1.3)[..., None]
    longs = np.where(~short & (L_all > 25 * MM))[0]
    nf = int(n * 0.005)
    fi = rng.choice(longs, nf, replace=False)
    _, g0 = sdf(S[fi, 0])
    drift = (
        g0 * rng.uniform(1.5, 4.5, nf)[:, None] * MM + rng.normal(0, 1.8, (nf, 3)) * MM
    )
    S[fi] += drift[:, None, :] * (s**2)[None, :, None]
    # --- final collision against the full head (ear included)
    flat = S[:, 1:].reshape(-1, 3)
    sd, g = (sdf_full or sdf)(flat)
    lo_ = sd < 0.25 * MM
    flat[lo_] += g[lo_] * (0.25 * MM - sd[lo_])[:, None]
    S[:, 1:] = flat.reshape(n, M_out - 1, 3)
    edge = edge_density(P)
    return S, L_all, edge, cl_all


def grow(V, T, N, seed=11, verbose=True):
    """the groom on a head mesh (V, T, N in the legacy frame): strands, lengths, edge density,
    occlusion, clump ids and the crown whorl"""
    global WHORL
    t0 = time.time()
    rng = np.random.default_rng(seed)
    sdf = SDF(V, N)
    keepv = ~on_ear(V)
    sdf_ne = SDF(V[keepv], N[keepv])
    wdir = np.array(
        [
            np.sin(np.radians(168)) * np.cos(np.radians(60)),
            np.sin(np.radians(60)),
            np.cos(np.radians(168)) * np.cos(np.radians(60)),
        ]
    )
    wp = C + wdir * 0.12
    sd, g = sdf(wp[None])
    WHORL = wp - g[0] * sd[0]
    X, info = make_guides(sdf_ne, V, T, N, rng)
    G, M, _ = X.shape
    offs = np.full((G, M), 1.0 * MM)
    el_r = sph(info["roots"])[1]
    band = np.full((G, M), 2.5 * MM) + (3.5 * MM * smoothstep(35, 55, el_r))[:, None]
    X1 = relax(sdf_ne, X, info, offs, band)
    offs = layer_offsets(X1, info)
    # messy pieces: some guides rest higher on the hair below them, and the far part of each
    # guide sways sideways
    lift = (rng.random(G) ** 3) * 5.0 * MM
    s_ = np.linspace(0, 1, M)
    offs = offs + lift[:, None] * smoothstep(0.1, 0.6, s_)[None, :]
    band = band + lift[:, None] * 0.5
    X2 = relax(sdf_ne, X1, info, offs, band)
    X2 = sway(sdf_ne, X2, info, offs, rng)
    S, L, edge, cl = render_strands(sdf_ne, V, T, N, X2, info, offs, rng, sdf_full=sdf)
    ao = hair_ao(S)
    if verbose:
        print(f"groom: {len(X2)} guides, {len(S)} strands, {time.time() - t0:.0f} s")
    return dict(P=S, L=L, edge=edge, ao=ao, clump=cl, whorl=WHORL)
