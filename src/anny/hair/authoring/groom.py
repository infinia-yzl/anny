# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The guides of a hairstyle, grown on anny's default body at the guide roots of the scalp layout
(:mod:`anny.hair.authoring.layout`), in the legacy frame of the authoring rig. A style spec
(``anny.data/hair/styles/<name>.json``, key ``groom``) sets every step; the medium tousled cut
of the legacy 3D Model build (build/hair2.py) is the spec ``medium_tousled``.

Pipeline
  1. signed distance grids of the head (collision and scalp offsets), and of the shoulders for
     long styles; they are cached, since every style uses them
  2. a comb direction at each root from the flow terms of the spec (whorl, directions, parts)
  3. follow-the-leader relaxation: gravity, bending stiffness, collision and a layered volume shell
  4. layers: each guide rests on the hair below it; then lift, sway and a second relaxation
  5. cuts: a default length per guide from its layer length and the perimeter line, with a
     point-cut texture, and the flicks of the perimeter pieces

The page draws the render strands from these guides (``anny.hair.styles``): the guides keep the
shape, and the strands add the clumps, the waves and the flyaways.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from anny.hair.chart import CRANIUM_CENTRE as C
from anny.hair.chart import MM, chart, hairline, resample_uniform, smoothstep

# ---------------------------------------------------------------------------------------------- helpers


def nrm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def tangent_of(v, N):
    v = np.asarray(v, float)
    if v.ndim == 1:
        v = np.broadcast_to(v / np.linalg.norm(v), N.shape)
    t = v - (v * N).sum(1, keepdims=True) * N
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)


def window(x, lohi):
    """smoothstep window of a spec: [a, b] rises from a to b (falls when a > b); None is 1"""
    if lohi is None:
        return np.ones(np.shape(x))
    return smoothstep(lohi[0], lohi[1], x)


def table2(spec, phi, h):
    """bilinear value of a table over (|azimuth|, height above the hairline)"""
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator(
        (np.asarray(spec["phi"], float), np.asarray(spec["h"], float)),
        np.asarray(spec["values"], float),
        bounds_error=False,
        fill_value=None,
    )
    a = np.clip(np.abs(phi), spec["phi"][0], spec["phi"][-1])
    hh = np.clip(h, spec["h"][0], spec["h"][-1])
    return f(np.stack([a, hh], -1))


# ---------------------------------------------------------------------------------------------- signed distance
class SDF:
    """
    Signed distance grid of a surface (positive outside) in a box, with its smoothed gradient.
    Cells within ``band`` of the surface query the nearest vertex (the distance to its tangent
    plane within 4 mm, the signed distance beyond); farther cells take the value of a grid four
    times coarser, whose sign is safe that far from the surface.
    """

    def __init__(self, V, N, h, lo, hi, band=0.02):
        self.h = h
        self.lo = np.array(lo, float)
        hi = np.array(hi, float)
        dims = np.ceil((hi - self.lo) / h).astype(int) + 1
        self.dims = dims
        near = np.all((V > self.lo - band - h) & (V < hi + band + h), 1)
        tree = cKDTree(V[near])
        Vn, Nn = V[near], N[near]

        def signed(Q, dist, idx):
            ok = np.isfinite(dist)
            out = np.full(len(Q), np.nan)
            dv = Q[ok] - Vn[idx[ok]]
            s = (dv * Nn[idx[ok]]).sum(1)
            out[ok] = np.where(dist[ok] < 0.004, s, dist[ok] * np.sign(s))
            return out

        # coarse grid over the box, every cell (vertices within 12 cm of the box)
        hc = 4 * h
        dc = np.ceil((hi - self.lo) / hc).astype(int) + 2
        gc = [self.lo[k] + np.arange(dc[k]) * hc for k in range(3)]
        Qc = np.stack(np.meshgrid(*gc, indexing="ij"), -1).reshape(-1, 3)
        wide = np.all((V > self.lo - 0.12) & (V < hi + 0.12), 1)
        tw = cKDTree(V[wide])
        dist, idx = tw.query(Qc, workers=-1)
        s = ((Qc - V[wide][idx]) * N[wide][idx]).sum(1)
        coarse = (dist * np.sign(s)).reshape(dc)
        # fine grid: the band from the nearest vertex, the rest from the coarse grid
        g = [self.lo[k] + np.arange(dims[k]) * h for k in range(3)]
        Q = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        dist, idx = tree.query(Q, distance_upper_bound=band, workers=-1)
        idx = np.minimum(idx, len(Vn) - 1)
        sd = signed(Q, dist, idx)
        far = np.isnan(sd)
        sd[far] = map_coordinates(
            coarse, ((Q[far] - self.lo) / hc).T, order=1, mode="nearest"
        )
        self.grid = sd.reshape(dims).astype(np.float32)
        gx, gy, gz = np.gradient(gaussian_filter(self.grid, 0.7), h)
        self.gx, self.gy, self.gz = (
            gx.astype(np.float32),
            gy.astype(np.float32),
            gz.astype(np.float32),
        )

    def arrays(self):
        return dict(
            grid=self.grid, gx=self.gx, gy=self.gy, gz=self.gz, lo=self.lo, h=self.h
        )

    @classmethod
    def from_arrays(cls, z):
        o = cls.__new__(cls)
        o.grid, o.gx, o.gy, o.gz = z["grid"], z["gx"], z["gy"], z["gz"]
        o.lo, o.h = np.asarray(z["lo"], float), float(z["h"])
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


class SDFUnion:
    """
    The union of several grids: the smallest distance, with the gradient of that grid. A grid
    counts only inside its own box (outside, its clamped values mean nothing); a point outside
    every box takes the first grid.
    """

    def __init__(self, *sdfs):
        self.sdfs = sdfs

    def __call__(self, P, grad=True):
        res = [s(P, True) for s in self.sdfs]
        sd = np.stack([r[0] for r in res])
        for i, s in enumerate(self.sdfs):
            hi = s.lo + (s.dims - 1) * s.h
            out = ~np.all((P >= s.lo) & (P <= hi), axis=1)
            sd[i, out] = np.inf
        none = np.isinf(sd).all(0)
        sd[0, none] = res[0][0][none]
        k = np.argmin(sd, axis=0)
        r = np.arange(len(P))
        if not grad:
            return sd[k, r]
        g = np.stack([x[1] for x in res])[k, r]
        return sd[k, r], g


HEAD_BOX = ((-0.11, 0.39, -0.10), (0.11, 0.67, 0.18))
BODY_BOX = ((-0.26, 0.12, -0.16), (0.26, 0.44, 0.20))
# m: the gap between hair and the neck, the shoulders and the chest. The render strands are
# straight between the points of a guide (1.4 cm apart on long hair), so on a shoulder they dip
# about 0.5 mm below the points; the clumps and waves of the page move them by as much again.
BODY_CLEARANCE = 2.5 * MM


def clear_body(sdf_body, X, clearance=BODY_CLEARANCE, iters=4):
    """
    Push the points of the guides X (G, M, 3) that lie on the body (inside BODY_BOX) out to
    ``clearance`` from it, and keep the length of each segment (the roots stay).
    """
    X = X.copy()
    seg = np.linalg.norm(np.diff(X, axis=1), axis=2)
    lo, hi = np.array(BODY_BOX[0]), np.array(BODY_BOX[1])
    for _ in range(iters):
        flat = X[:, 1:].reshape(-1, 3)
        inside = np.all((flat >= lo) & (flat <= hi), axis=1)
        sd, g = sdf_body(flat)
        low = inside & (sd < clearance)
        flat[low] += g[low] * (clearance - sd[low])[:, None]
        X[:, 1:] = flat.reshape(X.shape[0], -1, 3)
        for i in range(1, X.shape[1]):
            X[:, i] = X[:, i - 1] + nrm(X[:, i] - X[:, i - 1]) * seg[:, i - 1, None]
    return X


def body_sdfs(V, N, T=None, with_body=False, cache=True):
    """
    The grids of anny's default body: ``full`` (the head with its ears), ``head`` (without the
    ears, for the relaxation) and, with ``with_body``, ``body`` (neck, shoulders and chest at
    3 mm). Cached under ``ANNY_CACHE_DIR/hair``.
    """
    from anny.hair.authoring.layout import on_ear
    from anny.paths import get_anny_cache_path

    key = hashlib.sha1(np.ascontiguousarray(V, np.float32).tobytes()).hexdigest()[:12]
    d = get_anny_cache_path() / "hair"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    names = ["full", "head"] + (["body"] if with_body else [])
    for name in names:
        path = d / f"sdf_{name}_{key}.npz"
        if cache and path.exists():
            with np.load(path) as z:
                out[name] = SDF.from_arrays(z)
            continue
        t0 = time.time()
        if name == "full":
            s = SDF(V, N, 0.0015, *HEAD_BOX)
        elif name == "head":
            keep = ~on_ear(V)
            s = SDF(V[keep], N[keep], 0.0015, *HEAD_BOX)
        else:
            s = SDF(V, N, 0.003, *BODY_BOX, band=0.03)
        print(f"sdf {name}: {time.time() - t0:.0f} s")
        if cache:
            np.savez(path, **s.arrays())
        out[name] = s
    return out


# ---------------------------------------------------------------------------------------------- style fields
class Fields:
    """the comb direction, the length map and the perimeter of a spec"""

    def __init__(self, spec, sdf):
        self.spec = spec
        w = spec["whorl"]
        az, el = np.radians(w["azimuth"]), np.radians(w["elevation"])
        wdir = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        wp = C + wdir * 0.12
        sd, g = sdf(wp[None])
        self.whorl = wp - g[0] * sd[0]

    def comb(self, P, N):
        """the direction the hair is combed at scalp points (unit tangents)"""
        phi, el, _ = chart(P)
        a = np.abs(phi)
        t = np.zeros_like(P)
        for term in self.spec["flow"]:
            w = (
                term.get("weight", 1.0)
                * window(el, term.get("el"))
                * window(a, term.get("phi"))
            )
            kind = term["type"]
            if kind == "whorl":
                d = P - self.whorl
                v = d - (d * N).sum(1, keepdims=True) * N
                vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
                v = v / vn
                near = np.exp(-((vn[:, 0] / term.get("radius", 0.028)) ** 2))
                spiral = term.get("spiral", 1.9) * near + term.get("twist", 0.12)
                v = nrm(v + np.cross(N, v) * spiral[:, None])
            elif kind == "direction":
                v = tangent_of(term["vector"], N)
            elif kind == "part":
                # away from the part plane x = x0, strongest next to it
                side = np.where(P[:, 0] >= term["x"], 1.0, -1.0)
                v = tangent_of([1.0, 0, 0], N) * side[:, None]
                w = w * np.exp(
                    -(((P[:, 0] - term["x"]) / term.get("falloff", 0.04)) ** 2)
                )
            else:
                raise ValueError(f"unknown flow term {kind}")
            t = t + v * w[:, None]
        return tangent_of(nrm(t), N)

    def length(self, P):
        """layer length (m) before the cuts"""
        phi, el, _ = chart(P)
        return table2(self.spec["length"], phi, el - hairline(phi)) * MM

    def perimeter(self, phi):
        """world height of the perimeter cut at the azimuth ``phi`` (None: no perimeter)"""
        p = self.spec.get("perimeter")
        if p is None:
            return np.full(np.shape(phi), -np.inf)
        return np.interp(np.abs(phi), p["phi"], p["y"])


# ---------------------------------------------------------------------------------------------- guides
def rotate_about(v, axis, ang):
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return (
        v * c
        + np.cross(axis, v) * s
        + axis * (axis * v).sum(1, keepdims=True) * (1 - c)
    )


def make_guides(sdf, fields, roots, normals, rng, M):
    spec = fields.spec
    G = len(roots)
    P, Nn = roots, normals
    comb = fields.comb(P, Nn)
    # per-guide tousle: comb rotation, lift and length jitter
    ang = rng.normal(0, spec.get("tousle", 0.28), G)
    comb = nrm(rotate_about(comb, Nn, ang))
    phi, el, _ = chart(P)
    h = el - hairline(phi)
    crown = np.exp(-((np.linalg.norm(P - fields.whorl, axis=1) / 0.035) ** 2))
    front = smoothstep(50, 25, np.abs(phi)) * smoothstep(20, 4, h)
    top = smoothstep(38, 55, el) * (1 - crown)
    lf = spec["lift"]
    alpha = np.radians(
        np.clip(
            lf["base"]
            + lf["sd"] * rng.normal(0, 1, G)
            + lf["top"] * top
            + lf["crown"] * crown
            + lf["front"] * front
            + lf["edge"] * smoothstep(8, 0, h),
            lf["min"],
            lf["max"],
        )
    )
    d0 = nrm(Nn * np.cos(alpha)[:, None] + comb * np.sin(alpha)[:, None])
    jit = spec.get("length_jitter", [0.94, 1.06])
    # the guides grow to the longest length of the page's slider; the default cut trims them
    longest = spec.get("longest", 1.0)
    L = (
        fields.length(P) * rng.uniform(*jit, G) * longest
        + spec.get("length_extra_mm", 10) * MM
    )
    # short hair below the perimeter (sideburns, nape): clipper length
    below = P[:, 1] < fields.perimeter(phi) + 0.002
    sb = spec.get("below_perimeter_mm", [9, 13])
    L[below] = rng.uniform(*sb, below.sum()) * MM
    info = dict(roots=P, normals=Nn, comb=comb, d0=d0, L=L, below=below, ang=ang)
    # initial shape: march along the comb direction hugging the scalp at a small offset
    X = np.zeros((G, M, 3))
    X[:, 0] = P
    seg = L / (M - 1)
    X[:, 1] = P + d0 * seg[:, None]
    d = d0.copy()
    for i in range(2, M):
        p = X[:, i - 1]
        sd, g = sdf(p)
        c = fields.comb(p - g * sd[:, None], g)
        c = nrm(rotate_about(c, g, ang))
        d = nrm(d + (c - d) * 0.35)
        X[:, i] = p + d * seg[:, None]
        sd, g = sdf(X[:, i])
        lowp = sd < 1.0 * MM
        X[lowp, i] += g[lowp] * (1.0 * MM - sd[lowp])[:, None]
    return X, info


def relax(
    sdf,
    fields,
    X,
    info,
    offs,
    band,
    iters=160,
    gravity=0.00022,
    stiff0=2.2,
    stiff1=0.25,
    comb_k=0.00016,
    band_floor=-np.inf,
    stiff_ref_mm=None,
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
    stiff = np.broadcast_to(stiff0 + (stiff1 - stiff0) * s**0.7, (G, M))
    if stiff_ref_mm:
        # the bending stiffness per segment, for segments of stiff_ref_mm: longer segments bend more
        stiff = stiff * np.clip(stiff_ref_mm * MM / seg, 0.1, 1.0)[:, None]
    gvec = np.array([0, -1.0, 0])
    scale = seg[:, None, None] / 0.004
    a = np.repeat(info["ang"], M - 2)
    wgt = np.tile(1.0 - 0.65 * s[2:], G)
    o = offs[:, 2:].reshape(-1)
    bd = band[:, 2:].reshape(-1)
    for it in range(iters):
        vel = (X - Xp) * 0.6
        Xp = X.copy()
        X[:, 2:] += vel[:, 2:] + gvec * gravity * scale
        if comb_k > 0 and it % 2 == 0:
            flat = X[:, 2:].reshape(-1, 3)
            sd, g = sdf(flat)
            c = nrm(rotate_about(fields.comb(flat - g * sd[:, None], g), g, a))
            # the comb holds the hair that lies on the head; below band_floor it hangs
            on_head = (flat[:, 1] > band_floor).astype(float)
            X[:, 2:] += (c * (comb_k * 2 * wgt * on_head)[:, None]).reshape(
                G, M - 2, 3
            ) * scale
        for i in range(2, M):
            dcur = X[:, i] - X[:, i - 1]
            dprev = nrm(X[:, i - 1] - X[:, i - 2])
            dnew = nrm(nrm(dcur) + stiff[:, i, None] * dprev)
            X[:, i] = X[:, i - 1] + dnew * seg[:, None]
        flat = X[:, 2:].reshape(-1, 3)
        sd, g = sdf(flat)
        lo_ = sd < o
        flat[lo_] += g[lo_] * (o[lo_] - sd[lo_])[:, None]
        # the band holds the hair to the head; below band_floor the hair hangs free
        hi_ = (sd > o + bd) & (flat[:, 1] > band_floor)
        flat[hi_] -= g[hi_] * ((sd[hi_] - o[hi_] - bd[hi_]) * 0.35)[:, None]
        X[:, 2:] = flat.reshape(G, M - 2, 3)
    # final length fix
    for i in range(2, M):
        X[:, i] = X[:, i - 1] + nrm(X[:, i] - X[:, i - 1]) * seg[:, None]
    return X


def sway(sdf, X, info, offs, rng, amount=0.18, power=1.6):
    """rotate the far part of each guide about its root normal, then settle it back onto its layer"""
    G, M, _ = X.shape
    th = rng.normal(0, amount, G)
    s = np.linspace(0, 1, M)
    n0 = info["normals"]
    root = X[:, :1]
    rel = X - root
    out = X.copy()
    for i in range(1, M):
        out[:, i] = root[:, 0] + rotate_about(rel[:, i], n0, th * s[i] ** power)
    seg = info["L"] / (M - 1)
    o = offs[:, 2:].reshape(-1)
    for it in range(3):
        for i in range(2, M):
            out[:, i] = out[:, i - 1] + nrm(out[:, i] - out[:, i - 1]) * seg[:, None]
        flat = out[:, 2:].reshape(-1, 3)
        sd, g = sdf(flat)
        lo_ = sd < o
        flat[lo_] += g[lo_] * (o[lo_] - sd[lo_])[:, None]
        out[:, 2:] = flat.reshape(G, M - 2, 3)
    return out


def smooth(sdf, X, offs, seg, iters=6, k=0.5):
    """
    Laplacian smoothing of the guides after the relaxation: it removes the kinks that the
    collisions leave, then restores the segment lengths and pushes the points back out of the skin.
    """
    G, M, _ = X.shape
    X = X.copy()
    o = offs[:, 2:].reshape(-1)
    for _ in range(iters):
        X[:, 2:-1] += k * (0.5 * (X[:, 1:-2] + X[:, 3:]) - X[:, 2:-1])
        for i in range(2, M):
            X[:, i] = X[:, i - 1] + nrm(X[:, i] - X[:, i - 1]) * seg[:, None]
        flat = X[:, 2:].reshape(-1, 3)
        sd, g = sdf(flat)
        lo_ = sd < o
        flat[lo_] += g[lo_] * (o[lo_] - sd[lo_])[:, None]
        X[:, 2:] = flat.reshape(G, M - 2, 3)
    for i in range(2, M):
        X[:, i] = X[:, i - 1] + nrm(X[:, i] - X[:, i - 1]) * seg[:, None]
    return X


LAYER_KERNEL = np.array([[0.05, 0.12, 0.05], [0.12, 0.32, 0.12], [0.05, 0.12, 0.05]])


def layer_offsets(
    X,
    info,
    whorl,
    base=0.6 * MM,
    per_guide=1.25 * MM,
    cell=2.0,
    buckets=64,
    floor=-np.inf,
    cap=np.inf,
):
    """
    Volume: each guide rests on the hair that lies below it (guides farther from the crown lie
    lower). The guides go down in buckets of distance from the whorl; a bucket rests on the
    buckets below it, and its guides splat their thickness together.
    """
    G, M, _ = X.shape
    dist_w = np.linalg.norm(info["roots"] - whorl, axis=1)
    order = np.argsort(-dist_w)  # farthest from the whorl first = bottom layer
    nphi, nel = int(360 / cell), int(180 / cell)
    Tk = np.zeros((nphi, nel))
    phi, el, _ = chart(X.reshape(-1, 3))
    ci = np.clip(((phi + 180) / cell).astype(int), 0, nphi - 1).reshape(G, M)
    cj = np.clip(((el + 90) / cell).astype(int), 0, nel - 1).reshape(G, M)
    offs = np.zeros((G, M))
    ker = LAYER_KERNEL / LAYER_KERNEL.sum()
    s = np.linspace(0, 1, M)
    # the layers stack on the head alone: below ``floor`` the hair hangs and takes the base offset
    on_head = X[..., 1] > floor
    for group in np.array_split(order, min(buckets, G)):
        offs[group] = base + np.minimum(Tk[ci[group], cj[group]], cap) * on_head[group]
        # splat the ribbon thickness of the bucket along its paths (skip the root segment)
        keep = on_head[group, 1:].ravel()
        a = ci[group, 1:].ravel()[keep]
        b = cj[group, 1:].ravel()[keep]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                np.add.at(
                    Tk,
                    ((a + di) % nphi, np.clip(b + dj, 0, nel - 1)),
                    per_guide * ker[di + 1, dj + 1],
                )
    # the root keeps its own emergence: grow the offset smoothly from the root
    return offs * smoothstep(0.0, 0.25, s)[None, :] + base


def cut(sdf, fields, X, info, rng):
    """
    Default length of each guide: its layer length, and for the guides above the perimeter the
    arc length where they cross the perimeter line, with a point-cut texture. Returns the
    lengths and the flick amplitude (m) of the guides whose end forms the perimeter.
    """
    spec = fields.spec
    ct = spec["cuts"]
    G, M, _ = X.shape
    seg = np.linalg.norm(np.diff(X, axis=1), axis=2)
    cum = np.concatenate([np.zeros((G, 1)), np.cumsum(seg, 1)], 1)
    total = cum[:, -1]
    P = info["roots"]
    phi_r = chart(P)[0]
    layer = np.minimum(fields.length(P) * rng.uniform(*ct["layer_jitter"], G), total)
    below = info["below"]
    a = np.abs(phi_r)
    amp = np.interp(a, [0, 40, 60, 100, 130, 180], ct["texture_mm"]) * MM
    tex = (
        rng.uniform(0, 1, G) * amp * (rng.random(G) < 0.75)
        + np.abs(rng.normal(0, 1.5, G)) * MM
    )
    longer = (rng.random(G) < 0.2) & (a > 50)
    tex[longer] = -rng.uniform(1.5, 4.0, longer.sum()) * MM
    phiS = chart(X.reshape(-1, 3))[0].reshape(G, M)
    ycut = fields.perimeter(phiS) + tex[:, None]
    ycut = np.where(np.isfinite(ycut), ycut, -10.0)
    under = X[..., 1] < ycut
    under[:, 0] = False
    has = under.any(1) & ~below
    j = np.clip(np.argmax(under, axis=1), 1, M - 1)
    r = np.arange(G)
    y0 = X[r, j - 1, 1] - ycut[r, j - 1]
    y1 = X[r, j, 1] - ycut[r, j]
    f = np.clip(y0 / np.maximum(y0 - y1, 1e-9), 0, 1)
    s_per = cum[r, j - 1] * (1 - f) + cum[r, j] * f
    length = np.where(has, np.minimum(s_per, layer), layer)
    sb = spec.get("below_perimeter_mm", [9, 13])
    length[below] = np.minimum(
        total[below] * rng.uniform(0.8, 1.1, below.sum()), sb[1] * MM
    )
    length = np.clip(np.minimum(length, total), 2.0 * MM, None)
    at_edge = has & (s_per <= layer + 1e-6)
    fl = ct.get("flick")
    flick = np.zeros(G)
    if fl:
        on = (rng.random(G) < fl["share"]) & at_edge & (a > fl["min_azimuth"])
        flick[on] = rng.uniform(*fl["mm"], on.sum()) * MM
    return length, flick


def root_normals(P, V, T, N):
    from anny.hair import closest_triangles

    tri, bary, _ = closest_triangles(P, V, T)
    return nrm(np.einsum("nk,nkd->nd", bary, N[T[tri]]))


def grow(V, T, N, spec, layout, points=24, verbose=True, sdfs=None):
    """
    The guides of a style on anny's default body (legacy frame) at the guide roots of the layout.
    Returns ``points`` (G, points, 3) with uniform segments over the length each guide has,
    ``length`` (G,) the default length, ``group`` (G,) 0 above the perimeter and 1 below it,
    and ``flick`` (G,).
    """
    t0 = time.time()
    g = spec["groom"]
    G = layout.guides
    if g.get("bald"):
        # no hair: every guide has no length
        pts = np.repeat(layout.guide_position[:, None], points, 1)
        return dict(
            points=pts,
            length=np.zeros(G),
            group=np.zeros(G, np.uint8),
            flick=np.zeros(G),
            pivot=np.zeros(G),
        )
    if g.get("tie"):
        return grow_tied(
            V,
            T,
            N,
            spec,
            layout,
            points,
            sdfs or body_sdfs(V, N, with_body=True),
            verbose,
        )
    rng = np.random.default_rng(g.get("seed", 11))
    sdfs = sdfs or body_sdfs(V, N, with_body=bool(g.get("body_collision")))
    sdf, sdf_full = sdfs["head"], sdfs["full"]
    if g.get("ears"):
        # hair that covers the ears lies on them
        sdf = sdf_full
    if g.get("body_collision"):
        # long hair: the neck, the shoulders and the chest as well
        sdf, sdf_full = SDFUnion(sdf, sdfs["body"]), SDFUnion(sdf_full, sdfs["body"])
    fields = Fields(g, sdf)
    roots = layout.guide_position
    normals = root_normals(roots, V, T, N)
    M = g.get("relax_points", 25)
    X, info = make_guides(sdf, fields, roots, normals, rng, M)
    G = len(X)
    rl = g.get("relax", {})
    offs = np.full((G, M), 1.0 * MM)
    el_r = chart(roots)[1]
    # the band above the scalp that holds the hair: thicker on top, and at the front for volume
    bd = g.get("band_mm", [2.5, 3.5, 0.0])
    phi_r = chart(roots)[0]
    front = smoothstep(55, 25, np.abs(phi_r)) * smoothstep(55, 20, el_r)
    band = (
        np.full((G, M), bd[0] * MM)
        + (bd[1] * MM * smoothstep(35, 55, el_r) + bd[2] * MM * front)[:, None]
    )
    X1 = relax(sdf, fields, X, info, offs, band, **rl)
    ly = g.get("layers", {})
    offs = layer_offsets(X1, info, fields.whorl, **ly)
    # messy pieces: some guides rest higher on the hair below them, and the far part of each
    # guide sways sideways
    s_ = np.linspace(0, 1, M)
    lift = (rng.random(G) ** 3) * g.get("lift_mm", 5.0) * MM
    offs = offs + lift[:, None] * smoothstep(0.1, 0.6, s_)[None, :]
    band = band + lift[:, None] * 0.5
    X2 = relax(sdf, fields, X1, info, offs, band, **rl)
    for amount, power in g.get("sway", [[0.18, 1.6]]):
        X2 = sway(sdf, X2, info, offs, rng, amount, power)
    if g.get("smooth"):
        X2 = smooth(sdf, X2, offs, info["L"] / (M - 1), iters=g["smooth"])
    # final collision against the whole head (ears included)
    flat = X2[:, 1:].reshape(-1, 3)
    sd, grad = sdf_full(flat)
    low = sd < 0.25 * MM
    flat[low] += grad[low] * (0.25 * MM - sd[low])[:, None]
    X2[:, 1:] = flat.reshape(G, M - 1, 3)
    if g.get("body_collision"):
        X2 = clear_body(sdfs["body"], X2)
    length, flick = cut(sdf, fields, X2, info, rng)
    pts, avail = resample_uniform(X2, points)
    if verbose:
        print(f"groom {spec['name']}: {G} guides, {time.time() - t0:.0f} s")
    return dict(
        points=pts,
        length=np.minimum(length, avail),
        group=info["below"].astype(np.uint8),
        flick=flick,
        pivot=pivots(X2, g.get("hang_below")),
    )


NO_PIVOT = 1000.0  # m: the guide never leaves the head


def pivots(X, hang_below=None):
    """
    The arc length where each guide leaves the head (the pivot): the first point below the height
    ``hang_below``. Before it the strands between guides turn with the scalp; after it they hang
    parallel, and the physics lets them swing.
    """
    G, M, _ = X.shape
    out = np.full(G, NO_PIVOT)
    if hang_below is None:
        return out
    cum = np.concatenate(
        [np.zeros((G, 1)), np.cumsum(np.linalg.norm(np.diff(X, axis=1), axis=2), 1)], 1
    )
    below = X[..., 1] < hang_below
    below[:, 0] = False
    has = below.any(1)
    j = np.argmax(below, axis=1)
    out[has] = cum[np.nonzero(has)[0], j[has]]
    return out


# ---------------------------------------------------------------------------------------------- ties
def tie_point(sdf, tie):
    """the point of a tie: over the head at a chart direction, lifted from the scalp"""
    az, el = np.radians(tie["azimuth"]), np.radians(tie["elevation"])
    d = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    p = C + d * 0.12
    for _ in range(3):
        sd, g = sdf(p[None])
        p = p - g[0] * sd[0]
    _, n = sdf(p[None])
    return p + n[0] * tie["offset_mm"] * MM, n[0]


def hang(sdf, start, direction, length, M, iters=240):
    """one strand of ``length`` from ``start``, first along ``direction``, that settles under
    gravity against the body (follow-the-leader)"""
    seg = length / (M - 1)
    X = start + np.linspace(0, length, M)[:, None] * direction[None]
    Xp = X.copy()
    stiff = np.linspace(1.5, 0.15, M)
    for _ in range(iters):
        vel = (X - Xp) * 0.7
        Xp = X.copy()
        X[1:] += vel[1:] + np.array([0, -0.0006, 0]) * (seg / 0.01)
        for i in range(1, M):
            prev = direction if i == 1 else nrm(X[i - 1] - X[i - 2])
            X[i] = X[i - 1] + nrm(nrm(X[i] - X[i - 1]) + stiff[i] * prev) * seg
        sd, g = sdf(X[1:])
        low = sd < 4 * MM
        X[1:][low] += g[low] * (4 * MM - sd[low])[:, None]
    for i in range(1, M):
        X[i] = X[i - 1] + nrm(X[i] - X[i - 1]) * seg
    return X


def transport_frames(X):
    """unit tangents and a frame (a, b) carried along a polyline by parallel transport"""
    t = np.gradient(X, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    a = np.zeros_like(X)
    ref = np.array([1.0, 0, 0]) if abs(t[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    a[0] = nrm(ref - (ref @ t[0]) * t[0])
    for i in range(1, len(X)):
        v = a[i - 1] - (a[i - 1] @ t[i]) * t[i]
        a[i] = nrm(v)
    return t, a, np.cross(t, a)


EARS = [np.array([0.075, 0.500, 0.035]), np.array([-0.075, 0.500, 0.035])]


def grow_tied(V, T, N, spec, layout, points, sdfs, verbose=True):
    """
    Tied hair (ponytail, bun): each guide runs over the scalp to the tie, then into the tail. The
    scalp paths follow the tangent toward the tie at a small height; at the tie the guides sit in
    a bundle whose disk keeps the angle at which they arrive; the tail follows one hanging (or
    coiled) centre line with the bundle's offsets, widening toward the end. The pivot of each
    guide is its arc length at the tie.
    """
    t0 = time.time()
    g = spec["groom"]
    tie = g["tie"]
    rng = np.random.default_rng(g.get("seed", 11))
    sdf = sdfs["head"]
    body = SDFUnion(sdfs["full"], sdfs["body"]) if "body" in sdfs else sdf
    Tp, nT = tie_point(sdf, tie)
    roots = layout.guide_position
    G = len(roots)
    # the tie's frame: its normal, and the direction down the head from it
    e1 = nrm(
        np.cross(nT, [0, 1.0, 0]) if abs(nT[1]) < 0.95 else np.cross(nT, [1.0, 0, 0])
    )
    e2 = np.cross(nT, e1)
    height = (1.0 + 3.0 * rng.random(G)) * MM
    step = 2.0 * MM
    capture = tie["radius_mm"] * MM
    # every guide walks over the scalp toward the tie at once: the tangent of the direction to the tie
    x = roots.copy()
    walk = [x.copy()]
    active = np.ones(G, bool)
    ends = np.zeros(G, int)
    for it in range(400):
        dT = Tp - x
        dist = np.linalg.norm(dT, axis=1)
        arrived = active & (dist < capture)
        ends[arrived] = it
        active &= ~arrived
        if not active.any():
            break
        sd, gr = sdf(x)
        # toward the tie, over the top of the ears
        over = np.zeros(len(x))
        for ear in EARS:
            de = np.linalg.norm(x - ear, axis=1)
            below = smoothstep(ear[1] + 0.03, ear[1] + 0.01, x[:, 1])
            over = np.maximum(over, smoothstep(0.045, 0.025, de) * below)
        dT = nrm(dT) + 1.5 * over[:, None] * np.array([0, 1.0, 0])
        tng = dT - (dT * gr).sum(1, keepdims=True) * gr
        nx = x + nrm(tng) * step
        sd, gr = sdf(nx)
        near = smoothstep(4 * capture, capture, np.linalg.norm(Tp - nx, axis=1))
        h = height * (1 - near) + tie["offset_mm"] * MM * near
        nx = nx + gr * (h - sd)[:, None]
        x = np.where(active[:, None], nx, x)
        walk.append(x.copy())
    ends[active] = len(walk) - 1
    walk = np.array(walk)  # (steps, G, 3)
    flat = walk.reshape(-1, 3)
    sd, gr = sdfs["full"](flat)
    low = sd < 0.8 * MM
    flat[low] += gr[low] * (0.8 * MM - sd[low])[:, None]
    walk = flat.reshape(walk.shape)
    paths, s_tie, disk = [], np.zeros(G), np.zeros((G, 3))
    rad = tie["bundle_mm"] * MM * np.sqrt(rng.random(G))
    for k in range(G):
        P = walk[: ends[k] + 1, k]
        # the place in the bundle: the angle of arrival, a random radius
        arrive = P[-1] - Tp
        ang = np.arctan2(arrive @ e2, arrive @ e1)
        disk[k] = rad[k] * (np.cos(ang) * e1 + np.sin(ang) * e2)
        P = np.concatenate([P, (Tp + disk[k])[None]])
        paths.append(P)
        s_tie[k] = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    # the tail's centre line
    L_tail = tie["tail_mm"] * MM * g.get("longest", 1.0)
    Mt = 48
    if tie.get("bun"):
        # a coil around the bun's centre, winding inward and outward from the head
        r0, r1 = tie["bun_mm"][0] * MM, tie["bun_mm"][1] * MM
        s_ = np.linspace(0, L_tail, Mt)
        r = r0 + (r1 - r0) * s_ / L_tail
        th = np.cumsum(np.r_[0, np.diff(s_) / np.maximum(r[:-1], 1e-3)])
        centre = Tp + nT * (0.6 * r0)
        lift = nT[None] * ((0.8 * r0) * np.sin(np.pi * s_ / L_tail))[:, None]
        line = (
            centre[None]
            + (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2) * r[:, None]
            + lift
        )
        line = np.concatenate([Tp[None], line[1:]])
    else:
        d0 = nrm(nT + np.array([0, -tie.get("droop", 0.8), 0]))
        line = hang(body, Tp, d0, L_tail, Mt)
    tng, a, b = transport_frames(line)
    s_line = np.linspace(0, 1, Mt)
    spread = 1.0 + (tie.get("spread", 1.6) - 1.0) * s_line
    out = []
    for k in range(G):
        # the offset in the bundle, carried along the centre line and widening toward the end
        u, v = disk[k] @ a[0], disk[k] @ b[0]
        tail = line + (u * a + v * b) * spread[:, None]
        tail = tail + rng.normal(0, 0.6 * MM, tail.shape) * s_line[:, None]
        full = np.concatenate([paths[k], tail[1:]])
        out.append(resample_uniform(full[None], points)[0][0])
    pts = np.array(out)
    if verbose:
        print(f"groom {spec['name']}: {G} guides, {time.time() - t0:.0f} s")
    return dict(
        points=pts,
        # the default cut: the scalp path and the default tail (the curve holds the longest tail)
        length=s_tie + tie["tail_mm"] * MM,
        group=np.zeros(G, np.uint8),
        flick=np.zeros(G),
        pivot=s_tie,
    )
