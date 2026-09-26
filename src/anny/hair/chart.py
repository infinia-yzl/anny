# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The scalp chart and the compact form of hair curves.

**Chart.** A scalp point has an azimuth and an elevation (degrees) about the cranium centre of
anny's default body, in the legacy frame of the authoring rig (Y up, X toward the figure's left,
Z forward). The azimuth is 0 at the front and +90 on the figure's left. The render roots and the
fine head vertices keep the chart coordinates of anny's default body, so the chart rides on
vertex identity and follows every phenotype. The style fields (hairline, fade) are curves over
the absolute azimuth: :func:`coverage` and :func:`fade_length` evaluate them here, and the page
repeats them in GLSL.

**Curves.** A hair curve of ``P`` points has uniform segments. It is stored as its segment
length and one octahedral direction per segment, 8 bits per coordinate. The encoder quantises
each direction from the rebuilt point toward the true next point (error feedback), so the error
stays within about one quantum and does not grow along the curve.
"""

from __future__ import annotations

import numpy as np

MM = 0.001
CRANIUM_CENTRE = np.array([0.0, 0.515, 0.035])

# anny's default hairline: minimum elevation (degrees) against |azimuth| (degrees)
HAIRLINE_PHI = np.array(
    [0, 20, 32, 45, 56, 62, 65, 67, 70, 73, 76, 79, 84, 90]
    + [96, 100, 104, 110, 114, 120, 132, 150, 165, 180],
    dtype=np.float64,
)
HAIRLINE_EL = np.array(
    [33.5, 33, 32, 29, 23, 16, 8, -8, -16, -17, -4, 6, 6.5, 5.5]
    + [3, 0, -3, -9, -14, -22, -31, -35, -37.5, -38.5]
)
# the style curves are sampled at these |azimuth| values (degrees), for the page's uniforms
CURVE_PHI = np.linspace(0.0, 180.0, 19)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def chart(P: np.ndarray, centre: np.ndarray = CRANIUM_CENTRE):
    """azimuth and elevation (degrees) and distance of points about the cranium centre"""
    d = np.asarray(P, dtype=np.float64) - centre
    r = np.linalg.norm(d, axis=-1)
    phi = np.degrees(np.arctan2(d[..., 0], d[..., 2]))
    el = np.degrees(np.arcsin(np.clip(d[..., 1] / np.maximum(r, 1e-12), -1, 1)))
    return phi, el, r


def hairline(phi, offset=None):
    """elevation of the hairline at the azimuth ``phi``; ``offset`` (degrees at CURVE_PHI) raises it"""
    a = np.abs(phi)
    el = np.interp(a, HAIRLINE_PHI, HAIRLINE_EL)
    if offset is not None:
        el = el + np.interp(a, CURVE_PHI, offset)
    return el


def coverage(phi, el, offset=None):
    """share of the roots that carry hair: it ramps up over the first degrees above the hairline"""
    return smoothstep(-0.8, 3.0, el - hairline(phi, offset))


def fade_length(phi, el, fade, shift=0.0, offset=None):
    """
    The length (m) a fade allows at a chart point. Below the band it is the clipper length; over
    the band it grows geometrically, as the guard numbers of a clipper do, and reaches ``top`` at
    the top of the band. Above the band it keeps growing, so the guides soon decide the length.
    ``fade`` holds ``start`` (degrees above the hairline, at CURVE_PHI; a large negative value
    means no fade), ``width`` (degrees), ``clipper`` and ``top`` (m); ``shift`` moves the band.
    """
    if not fade:
        return np.full(np.shape(phi), np.inf)
    h = el - hairline(phi, offset)
    lo = np.interp(np.abs(phi), CURVE_PHI, fade["start"]) + shift
    u = np.clip((h - lo) / fade["width"], 0.0, 30.0)
    return fade["clipper"] * (fade["top"] / fade["clipper"]) ** u


# ---------------------------------------------------------------------------------------------- codec
def oct_encode(d: np.ndarray) -> np.ndarray:
    """octahedral coordinates in [-1, 1]^2 of unit vectors (..., 3)"""
    d = d / np.maximum(np.abs(d).sum(-1, keepdims=True), 1e-30)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    sx = np.where(x >= 0, 1.0, -1.0)
    sy = np.where(y >= 0, 1.0, -1.0)
    ox = np.where(z >= 0, x, (1 - np.abs(y)) * sx)
    oy = np.where(z >= 0, y, (1 - np.abs(x)) * sy)
    return np.stack([ox, oy], -1)


def oct_decode(e: np.ndarray) -> np.ndarray:
    """unit vectors of octahedral coordinates (..., 2)"""
    x, y = e[..., 0], e[..., 1]
    z = 1.0 - np.abs(x) - np.abs(y)
    t = np.maximum(-z, 0.0)
    x = x + np.where(x >= 0, -t, t)
    y = y + np.where(y >= 0, -t, t)
    d = np.stack([x, y, z], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def code_to_oct(c: np.ndarray) -> np.ndarray:
    """octahedral coordinates of 8-bit codes (0..254 map to -1..1; the page does the same)"""
    return np.asarray(c, dtype=np.float64) / 127.0 - 1.0


def _nearest_codes(d: np.ndarray) -> np.ndarray:
    """the 8-bit code pair whose direction is nearest to each unit vector (..., 3)"""
    e = oct_encode(d)
    base = np.floor((e + 1.0) * 127.0)
    best = np.zeros(e.shape, np.int32)
    best_dot = np.full(e.shape[:-1], -np.inf)
    for i in (0, 1):
        for j in (0, 1):
            c = np.clip(base + np.array([i, j]), 0, 254)
            dot = (oct_decode(code_to_oct(c)) * d).sum(-1)
            better = dot > best_dot
            best[better] = c[better]
            best_dot = np.where(better, dot, best_dot)
    return best.astype(np.uint8)


def resample_uniform(S: np.ndarray, P: int, length: np.ndarray | None = None):
    """
    Curves (n, M, 3) resampled to P points with uniform segments, from the root to the arc length
    ``length`` (the whole curve by default). Returns the points (n, P, 3) and the lengths.
    """
    S = np.asarray(S, dtype=np.float64)
    n, M, _ = S.shape
    seg = np.linalg.norm(np.diff(S, axis=1), axis=2)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, 1)], 1)
    L = cum[:, -1] if length is None else np.minimum(length, cum[:, -1])
    u = np.linspace(0.0, 1.0, P)[None, :] * L[:, None]
    # one searchsorted over all curves: offset each curve's arc lengths past the previous ones
    off = np.arange(n)[:, None] * (cum[:, -1].max() + 1.0)
    j = np.searchsorted((cum + off).ravel(), (u + off).ravel(), side="right")
    j = j.reshape(n, P) - np.arange(n)[:, None] * M
    j = np.clip(j, 1, M - 1)
    r = np.arange(n)[:, None]
    c0, c1 = cum[r, j - 1], cum[r, j]
    f = np.clip((u - c0) / np.maximum(c1 - c0, 1e-12), 0, 1)[..., None]
    return S[r, j - 1] * (1 - f) + S[r, j] * f, L


def encode_curves(S: np.ndarray):
    """
    Curves (n, P, 3) with uniform segments as (roots, segment lengths, codes (n, P-1, 2) uint8).
    The segment length is the mean of each curve's segments.
    """
    S = np.asarray(S, dtype=np.float64)
    n, P, _ = S.shape
    seg = np.linalg.norm(np.diff(S, axis=1), axis=2).mean(1)
    codes = np.zeros((n, P - 1, 2), np.uint8)
    cur = S[:, 0].copy()
    for j in range(1, P):
        d = S[:, j] - cur
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(norm > 1e-15, d / np.maximum(norm, 1e-30), np.array([0, 1.0, 0]))
        c = _nearest_codes(d)
        codes[:, j - 1] = c
        cur = cur + seg[:, None] * oct_decode(code_to_oct(c))
    return S[:, 0].copy(), seg, codes


def decode_curves(roots: np.ndarray, seg: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """points (n, P, 3) of encoded curves"""
    steps = oct_decode(code_to_oct(codes)) * np.asarray(seg)[:, None, None]
    pts = np.concatenate([np.asarray(roots)[:, None, :], steps], 1)
    return np.cumsum(pts, axis=1)
