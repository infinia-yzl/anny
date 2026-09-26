# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Hairstyles on the scalp layout, and the reference of the page's hair passes.

A style is a spec (``data/hair/styles/<name>.json``) and its guides
(``data/hair/styles.safetensors``): one curve per guide root of the layout, stored as segment
directions (``anny.hair.chart``), with its default length, its group and its flick. The page
draws the render strands in two passes, which this module repeats in NumPy:

- **pass A** (:func:`follow_guides`, :func:`pose_guides`): the guides follow the body, as
  :class:`anny.hair.StrandBinding` with tips does, then take the pose of their root and the motion
  of the simulated guides;
- **pass B** (:func:`strands`): each render strand blends the offsets of its four guides, each one
  turned from the normal of its guide root to the normal of the strand root, then takes its length,
  its clumps, curl, waves, flyaways, flick and volume.

The random values of a strand come from :func:`rnd`, an integer hash that the page repeats in GLSL.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from anny.hair import closest_triangles, scalp_size, triangle_frames
from anny.hair.chart import (
    CRANIUM_CENTRE,
    MM,
    coverage,
    decode_curves,
    fade_length,
    smoothstep,
)
from anny.hair.layout import DATA_DIR, Layout, load_layout

STYLE_DIR = DATA_DIR / "styles"
GUIDES_PATH = DATA_DIR / "styles.safetensors"
SIMILARITY = 0.55  # guides that flow another way than the nearest one do not blend in
SECTOR_RADIUS = 0.6 * 0.0042 * 0.5  # centre of a fine clump from its guide root
# a guide tip within 1 cm of the skin follows the skin under it
TIP_BIND_DISTANCE = 0.01
PIVOT_BLEND = (
    0.02  # m: the turn to the root normal fades out over this arc before the pivot
)
DEFAULT_PARAMS = dict(length=1.0, curl=1.0, volume=1.0, density=1.0, fade=0.0)
# the waves that the curl slider adds to a straight style (radius, period in mm)
DEFAULT_CURL = (4.0, 40.0)


def style_params(spec: dict) -> dict:
    """the page's parameters at their defaults for a style: a straight style starts without curl"""
    cu = spec["render"].get("curl") or {}
    return dict(DEFAULT_PARAMS, curl=1.0 if cu.get("radius_mm", 0) > 0 else 0.0)


def curl_shape(spec: dict):
    """radius and period (mm) of the curl at the parameter 1: the style's own, or waves"""
    cu = spec["render"].get("curl") or {}
    ramp = cu.get("ramp", 0.15)
    if cu.get("radius_mm", 0) > 0:
        return cu["radius_mm"], cu["period_mm"], ramp
    return DEFAULT_CURL[0], DEFAULT_CURL[1], ramp


# ---------------------------------------------------------------------------------------------- random values
def _pcg(v):
    with np.errstate(over="ignore"):
        state = v * np.uint32(747796405) + np.uint32(2891336453)
        word = (
            (state >> ((state >> np.uint32(28)) + np.uint32(4))) ^ state
        ) * np.uint32(277803737)
        return (word >> np.uint32(22)) ^ word


def rnd(key, stream):
    """uniform values in [0, 1) for integer keys and a stream number (GLSL: hairRnd)"""
    key = np.asarray(key, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = _pcg(key * np.uint32(0x9E3779B9) ^ _pcg(np.uint32(stream)))
    return (h >> np.uint32(8)).astype(np.float64) / 16777216.0


STRAND, GUIDE, SECTOR = 0, 1, 2


def strand_key(r):
    return np.asarray(r, np.uint32) * np.uint32(4) + np.uint32(STRAND)


def guide_key(k):
    return np.asarray(k, np.uint32) * np.uint32(4) + np.uint32(GUIDE)


def sector_key(k, q, sectors):
    return (
        np.asarray(k, np.uint32) * np.uint32(sectors) + np.asarray(q, np.uint32)
    ) * np.uint32(4) + np.uint32(SECTOR)


def lerp_range(u, lohi):
    return lohi[0] + (lohi[1] - lohi[0]) * u


# ---------------------------------------------------------------------------------------------- styles
@dataclasses.dataclass
class HairStyle:
    spec: dict
    points: np.ndarray  # (G, P, 3) guide points on anny's default body (legacy frame)
    length: np.ndarray  # (G,) default length (m)
    group: np.ndarray  # (G,) uint8
    flick: np.ndarray  # (G,) flick amplitude (m)
    pivot: (
        np.ndarray
    )  # (G,) arc length (m) where the guide leaves the head (a tie, or where it hangs)

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def available(self) -> np.ndarray:
        """the length of each guide curve (m): the longest strand it can carry"""
        return np.linalg.norm(np.diff(self.points, axis=1), axis=2).sum(1)

    def mirrored(self, layout: Layout) -> "HairStyle":
        """the style on the other side (a side part on the right): guide k takes guide mirror[k]"""
        m = layout.guide_mirror
        pts = self.points[m] * np.array([-1.0, 1.0, 1.0])
        return HairStyle(
            self.spec, pts, self.length[m], self.group[m], self.flick[m], self.pivot[m]
        )


def style_names() -> list[str]:
    return sorted(p.stem for p in STYLE_DIR.glob("*.json"))


def load_spec(name: str) -> dict:
    return json.loads((STYLE_DIR / f"{name}.json").read_text())


def load_style(name: str, layout: Layout | None = None, path=GUIDES_PATH) -> HairStyle:
    from safetensors import safe_open

    layout = layout or load_layout()
    spec = load_spec(name)
    with safe_open(str(path), framework="numpy") as f:
        codes = f.get_tensor(f"{name}.codes")
        seg = f.get_tensor(f"{name}.segment").astype(np.float64)
        length = f.get_tensor(f"{name}.length").astype(np.float64)
        group = f.get_tensor(f"{name}.group")
        flick = f.get_tensor(f"{name}.flick").astype(np.float64)
        pivot = f.get_tensor(f"{name}.pivot").astype(np.float64)
    points = decode_curves(layout.guide_position, seg, codes)
    return HairStyle(spec, points, length, group, flick, pivot)


# ---------------------------------------------------------------------------------------------- binding
@dataclasses.dataclass
class Binding:
    """the triangles under the guide roots, guide tips and render roots, on anny's default body"""

    guide_tri: np.ndarray
    guide_bary: np.ndarray
    root_tri: np.ndarray
    root_bary: np.ndarray
    reference_size: float

    @classmethod
    def build(cls, layout: Layout, V, T) -> "Binding":
        gt, gb, _ = closest_triangles(layout.guide_position, V, T)
        rt, rb, _ = closest_triangles(layout.root_position, V, T)
        roots = np.einsum("nk,nkd->nd", gb, V[T[gt]])
        return cls(gt, gb, rt, rb, scalp_size(roots))


def tip_binding(style: HairStyle, V, T):
    """the triangle under each guide tip, and whether the tip is near enough to follow it"""
    tri, bary, dist = closest_triangles(style.points[:, -1], V, T)
    return tri, bary, dist < TIP_BIND_DISTANCE


def _anchor(V, T, tri, bary):
    return np.einsum("nk,nkd->nd", bary, V[T[tri]])


class GuideRest:
    """
    The guide curves of a style bound to anny's default body (V0, T): the points in the frame of
    the root triangle, and for the tips near the skin in the frame of the tip triangle, divided by
    the scalp size (as :class:`anny.hair.StrandBinding` with ``tips=True``).
    """

    def __init__(self, style: HairStyle, binding: Binding, V0, T):
        self.T = T
        self.binding = binding
        pts = style.points
        P = pts.shape[1]
        roots = _anchor(V0, T, binding.guide_tri, binding.guide_bary)
        F = triangle_frames(V0, T, binding.guide_tri)
        size = binding.reference_size
        self.local = np.einsum("nij,nmj->nmi", F, pts - roots[:, None]) / size
        self.tip_tri, self.tip_bary, self.tip_on = tip_binding(style, V0, T)
        Ft = triangle_frames(V0, T, self.tip_tri)
        anchors = _anchor(V0, T, self.tip_tri, self.tip_bary)
        self.tip_local = np.einsum("nij,nmj->nmi", Ft, pts - anchors[:, None]) / size
        self.weights = np.linspace(0.0, 1.0, P) ** 2

    def follow(self, V, size=None):
        """the guide points on a body with the same triangles and new vertices V"""
        b = self.binding
        roots = _anchor(V, self.T, b.guide_tri, b.guide_bary)
        if size is None:
            size = scalp_size(roots)
        F = triangle_frames(V, self.T, b.guide_tri)
        out = roots[:, None] + size * np.einsum("nji,nmj->nmi", F, self.local)
        Ft = triangle_frames(V, self.T, self.tip_tri)
        anchors = _anchor(V, self.T, self.tip_tri, self.tip_bary)
        tips = anchors[:, None] + size * np.einsum("nji,nmj->nmi", Ft, self.tip_local)
        w = self.weights[None, :, None] * self.tip_on[:, None, None]
        return (1.0 - w) * out + w * tips


def follow_guides(style, binding, V0, V, T):
    """pass A without a pose: the guide points on the body V"""
    return GuideRest(style, binding, V0, T).follow(V)


def pose_guides(rest, matrices=None, offsets=None):
    """
    Pass A: the guides with the pose of their roots, ``matrices`` (G, 4, 4) blended from the bones,
    plus ``offsets`` (G, P, 3), the motion of the simulated guides.
    """
    out = rest
    if matrices is not None:
        out = (
            np.einsum("gij,gpj->gpi", matrices[:, :3, :3], rest)
            + matrices[:, None, :3, 3]
        )
    if offsets is not None:
        out = out + offsets
    return out


def sim_offsets(layout, sim_motion, rest_available):
    """
    The motion of every guide from the motion of the simulated guides (S, P_sim, 3), each taken
    at the same arc fraction and blended with the layout's weights.
    """
    S, Ps, _ = sim_motion.shape
    w = layout.guide_sim_weights.astype(np.float64) / 255.0
    return np.einsum("gk,gkpd->gpd", w, sim_motion[layout.guide_sim])


# ---------------------------------------------------------------------------------------------- pass B
def rotate_between(u, v, x):
    """x turned by the smallest rotation that takes the unit vector u to the unit vector v"""
    c = (u * v).sum(-1, keepdims=True)
    a = np.cross(u, v)
    return x * c + np.cross(a, x) + a * (a * x).sum(-1, keepdims=True) / (1.0 + c)


def tangent_frame(n):
    """a tangent frame (a, b) at unit normals n, the same one the page builds"""
    ref = np.where(
        np.abs(n[..., 1:2]) < 0.9, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    )
    a = np.cross(ref, n)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    return a, np.cross(n, a)


def sample_curve(G, seg, s):
    """points and segment directions of uniform curves G (n, P, 3) with segment lengths seg (n,)
    at arc lengths s (n, m); past the end the last segment continues"""
    n, P, _ = G.shape
    u = s / np.maximum(seg, 1e-12)[:, None]
    i0 = np.clip(np.floor(u).astype(int), 0, P - 2)
    f = u - i0
    r = np.arange(n)[:, None]
    step = G[r, i0 + 1] - G[r, i0]
    tangent = step / np.maximum(np.linalg.norm(step, axis=-1, keepdims=True), 1e-12)
    return G[r, i0] + f[..., None] * step, tangent


@dataclasses.dataclass
class Roots:
    """the render roots on the current body: rest positions and normals"""

    position: np.ndarray
    normal: np.ndarray

    @classmethod
    def on(cls, binding: Binding, V, T) -> "Roots":
        F = triangle_frames(V, T, binding.root_tri)
        return cls(_anchor(V, T, binding.root_tri, binding.root_bary), F[:, 2])


def guide_normals(binding: Binding, V, T):
    return triangle_frames(V, T, binding.guide_tri)[:, 2]


def strands(
    style: HairStyle,
    layout: Layout,
    guides: np.ndarray,
    guide_normal: np.ndarray,
    roots: Roots,
    params: dict | None = None,
    count: int | None = None,
    points: int | None = None,
    head_centre=CRANIUM_CENTRE,
    scale: float = 1.0,
):
    """
    Pass B: the render strands (count, points, 3) of a style from its posed guides (G, P, 3), the
    posed guide root normals, and the render roots, for the page's parameters ``params``
    (length, curl, volume, fade shift; see DEFAULT_PARAMS). Also returns the strand lengths.

    ``scale`` is the size of the head against anny's default head (the guides follow the body
    with it): every length of the strands scales with it.
    """
    prm = dict(style_params(style.spec), **(params or {}))
    rs = style.spec["render"]
    n = count or layout.roots
    P = points or guides.shape[1]
    r = np.arange(n)
    gid = layout.root_guides[:n]
    w = layout.root_weights[:n].astype(np.float64) / 255.0
    k0 = gid[:, 0]
    # guides of another group, or that flow another way, do not blend in
    third = guides.shape[1] // 3
    gdir = guides[:, third] - guides[:, 0]
    gdir /= np.maximum(np.linalg.norm(gdir, axis=-1, keepdims=True), 1e-12)
    same = (style.group[gid] == style.group[k0][:, None]) & (
        (gdir[gid] * gdir[k0][:, None]).sum(-1) > SIMILARITY
    )
    same[:, 0] = True
    w = w * same
    w /= w.sum(1, keepdims=True)
    # length
    key = strand_key(r)
    jit = lerp_range(rnd(key, 1), rs["strand_jitter"])
    th = rs.get("thinning")
    if th:
        thin = rnd(key, 2) < th["share"]
        jit = np.where(thin, jit * lerp_range(rnd(key, 3), th["range"]), jit)
    avail = style.available
    seg = avail / (guides.shape[1] - 1) * scale
    ell = prm["length"] * (w * style.length[gid]).sum(1) * jit * scale
    ell = np.minimum(ell, (w * avail[gid]).sum(1) * scale)
    phi, el = layout.root_chart[:n, 0], layout.root_chart[:n, 1]
    fade = fade_length(phi, el, rs.get("fade"), prm["fade"], rs.get("hairline"))
    ell = np.minimum(ell, fade * scale)
    alive = rnd(key, 4) < coverage(phi, el, rs.get("hairline"))
    ell = np.where(alive, ell, 0.0)
    # base: the blended offsets of the guides, turned to the strand root
    t = np.linspace(0.0, 1.0, P)
    s = ell[:, None] * t[None, :]
    nr = roots.normal[:n]
    xr = roots.position[:n]
    base = np.repeat(xr[:, None], P, 1).copy()
    tng = np.zeros_like(base)
    # the height of the blended guides over their roots, from the head centre, and the weight of
    # the turn (before the pivots)
    lift = np.zeros((n, P))
    turn = np.zeros((n, P))
    # the turn to the root normal ends at the pivot of each guide, where the hair leaves the head:
    # hanging and tied hair stays parallel
    for j in range(gid.shape[1]):
        k = gid[:, j]
        piv = style.pivot[k][:, None] * scale
        rho = (1.0 - smoothstep(piv - PIVOT_BLEND * scale, piv, s))[..., None]
        pts, dirs = sample_curve(guides[k], seg[k], s)
        off = pts - guides[k, 0][:, None]
        gn = guide_normal[k][:, None]
        off = off + rho * (rotate_between(gn, nr[:, None], off) - off)
        dirs = dirs + rho * (rotate_between(gn, nr[:, None], dirs) - dirs)
        base += w[:, j, None, None] * off
        tng += w[:, j, None, None] * dirs
        radius = np.linalg.norm(pts - head_centre, axis=-1)
        lift += w[:, j, None] * (
            radius - np.linalg.norm(guides[k, 0] - head_centre, axis=-1)[:, None]
        )
        turn += w[:, j, None] * rho[..., 0]
    tng /= np.maximum(np.linalg.norm(tng, axis=-1, keepdims=True), 1e-12)
    # on the head, each point keeps the height of the guides over the root: blending offsets that
    # turn over a curved scalp would lift the strands off it or sink them into it
    rel = base - head_centre
    rad = np.linalg.norm(rel, axis=-1)
    want = np.linalg.norm(xr - head_centre, axis=-1)[:, None] + lift
    base = (
        head_centre
        + rel * (1.0 + turn * (want / np.maximum(rad, 1e-9) - 1.0))[..., None]
    )
    # clumps: the lateral offset of the root from its guide shrinks along the strand, first toward
    # the centre of its sector (fine clumps), then toward the guide (coarse clumps)
    cl = rs["clump"]
    g0 = guides[k0, 0]
    d = xr - g0
    n0 = guide_normal[k0]
    d = d - (d * n0).sum(1, keepdims=True) * n0
    a, b = tangent_frame(n0)
    S = cl["sectors"]
    ang = np.arctan2((d * b).sum(1), (d * a).sum(1))
    q = np.clip(np.floor((ang / (2 * np.pi) + 0.5) * S), 0, S - 1).astype(np.uint32)
    qa = (q + 0.5) / S * 2 * np.pi - np.pi
    centre = (np.cos(qa)[:, None] * a + np.sin(qa)[:, None] * b) * SECTOR_RADIUS * scale
    skey = sector_key(k0, q, S)
    fs = lerp_range(rnd(skey, 1), cl["fine"])
    fp = lerp_range(rnd(skey, 2), cl["fine_power"])
    cs = lerp_range(rnd(guide_key(k0), 1), cl["coarse"])
    prof_f = fs[:, None] * t[None, :] ** fp[:, None]
    tip = cl.get("coarse_tip", 0.55)
    prof_c = (
        cs[:, None] * smoothstep(0.0, 0.5, t)[None, :] ** 0.9
        + (1 - cs[:, None]) * tip * t[None, :] ** 2.2
    )
    lateral = centre[:, None] + (d - centre)[:, None] * (1 - prof_f)[..., None]
    lateral = lateral * (1 - prof_c)[..., None]
    out = base - (d[:, None] - lateral)
    # a frame along the strand: the blended guide tangent and the outward direction
    outward = out - head_centre
    outward /= np.maximum(np.linalg.norm(outward, axis=-1, keepdims=True), 1e-12)
    b1 = np.cross(tng, outward)
    b1 /= np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-12)
    b2 = np.cross(tng, b1)
    # curl: a helix about the strand, in phase within a fine clump
    c_rad, c_per, c_ramp = curl_shape(style.spec)
    if prm["curl"] > 0:
        rad = prm["curl"] * c_rad * MM * scale * lerp_range(rnd(key, 5), [0.8, 1.2])
        per = c_per * MM * scale * lerp_range(rnd(skey, 3), [0.85, 1.15])
        phase = 2 * np.pi * (rnd(skey, 4) + 0.08 * rnd(key, 6))
        th_ = 2 * np.pi * s / per[:, None] + phase[:, None]
        ramp = smoothstep(0.0, c_ramp, t)[None, :] * rad[:, None]
        out = out + ramp[..., None] * (
            np.cos(th_)[..., None] * b1 + np.sin(th_)[..., None] * b2
        )
    # small waves
    fr = rs.get("frizz")
    if fr:
        amp = (
            lerp_range(rnd(key, 7), fr["mm"])
            * MM
            * scale
            * np.clip(ell / (30 * MM * scale), 0.2, 1.0)
        )
        cyc = lerp_range(rnd(key, 8), fr["cycles"])
        p1, p2 = 2 * np.pi * rnd(key, 9), 2 * np.pi * rnd(key, 10)
        wav = (
            np.sin(2 * np.pi * cyc[:, None] * t + p1[:, None])[..., None] * b1
            + np.sin(2 * np.pi * cyc[:, None] * 0.7 * t + p2[:, None])[..., None] * b2
        )
        out = out + wav * (amp[:, None] * t[None, :] ** 1.3)[..., None]
    # flyaways
    fl = rs.get("flyaways")
    if fl:
        fly = (rnd(key, 11) < fl["share"]) & (ell > 25 * MM * scale)
        root_out = xr - head_centre
        root_out /= np.maximum(np.linalg.norm(root_out, axis=-1, keepdims=True), 1e-12)
        drift = root_out * lerp_range(rnd(key, 12), fl["mm"])[:, None] * MM
        rand3 = np.stack([rnd(key, 13 + i) for i in range(3)], 1)
        drift = (drift + (rand3 - 0.5) * 3.6 * MM) * scale
        out = out + (fly[:, None] * drift)[:, None] * (t**2)[None, :, None]
    # flick: the ends of the perimeter pieces turn outward
    fk = style.flick[k0] * scale
    out = (
        out + outward * (fk[:, None] * smoothstep(0.6, 1.0, t)[None, :] ** 2)[..., None]
    )
    # volume: points move away from the head centre in proportion to their height above the root
    vol = prm["volume"]
    if vol != 1.0:
        rel = out - head_centre
        rho = np.linalg.norm(rel, axis=-1)
        rho_r = np.linalg.norm(xr - head_centre, axis=-1)[:, None]
        k = 1.0 + (vol - 1.0) * np.maximum(rho - rho_r, 0.0) / np.maximum(rho, 1e-9)
        out = head_centre + rel * k[..., None]
    return out, ell


# ---------------------------------------------------------------------------------------------- density
VOLUME_STEP = 0.003  # voxel size (m)
VOLUME_SAMPLE = 0.0034  # spacing of the samples along the guides (m)
VOLUME_RAY = 7  # voxels walked outward for the occlusion
VOLUME_K = (
    0.044  # occlusion per unit of density walked (matches the legacy groom's occlusion)
)
VOLUME_NEAR = 60.0  # the density at which the scalp tint is full


def guide_strands(style: HairStyle, layout: Layout, count: int | None = None):
    """the number of drawn render strands whose nearest guide is each guide (with the hairline)"""
    n = count or layout.roots
    rs = style.spec["render"]
    phi, el = layout.root_chart[:n, 0], layout.root_chart[:n, 1]
    cov = coverage(phi, el, rs.get("hairline"))
    return np.bincount(layout.root_guides[:n, 0], weights=cov, minlength=layout.guides)


def density_volume(
    style: HairStyle,
    layout: Layout,
    params: dict | None = None,
    count: int | None = None,
):
    """
    A grid over the hair of a style on anny's default body: in channel 0 the occlusion of a point
    by the hair outward of it (``exp(-k * density walked outward from the head centre)``), in
    channel 1 the local density against VOLUME_NEAR (the scalp tint). Both are uint8. Returns the
    grid (nx, ny, nz, 2), its lower corner and its step. The page builds the same grid
    (``viewer/src/hair/data.ts``) when the style or its parameters change.
    """
    prm = dict(style_params(style.spec), **(params or {}))
    rs = style.spec["render"]
    n_g = guide_strands(style, layout, count)
    avail = style.available
    D = np.minimum(style.length * prm["length"], avail)
    phi, el = layout.guide_chart[:, 0], layout.guide_chart[:, 1]
    D = np.minimum(
        D, fade_length(phi, el, rs.get("fade"), prm["fade"], rs.get("hairline"))
    )
    live = (n_g > 0) & (D > 0)
    pts, wts = [], []
    P = style.points.shape[1]
    seg = avail / (P - 1)
    for g in np.nonzero(live)[0]:
        m = int(np.ceil(D[g] / VOLUME_SAMPLE))
        s = (np.arange(m) + 0.5) * D[g] / m
        p, _ = sample_curve(style.points[g : g + 1], seg[g : g + 1], s[None, :])
        pts.append(p[0])
        wts.append(np.full(m, n_g[g] * (D[g] / m) / VOLUME_SAMPLE))
    if not pts:
        empty = np.zeros((2, 2, 2, 2), np.uint8)
        empty[..., 0] = 255  # no hair: no occlusion and no scalp tint
        return empty, CRANIUM_CENTRE - VOLUME_STEP, VOLUME_STEP
    pts, wts = np.concatenate(pts), np.concatenate(wts)
    h = VOLUME_STEP
    lo = np.floor((pts.min(0) - 0.01) / h) * h
    dims = (np.ceil((pts.max(0) + 0.01 - lo) / h)).astype(int) + 1
    grid = np.zeros(dims)
    # trilinear splat
    u = (pts - lo) / h
    i0 = np.floor(u).astype(int)
    f = u - i0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = (
                    (f[:, 0] if dx else 1 - f[:, 0])
                    * (f[:, 1] if dy else 1 - f[:, 1])
                    * (f[:, 2] if dz else 1 - f[:, 2])
                )
                np.add.at(
                    grid, (i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz), wts * wgt
                )
    # two passes of a [1, 2, 1] / 4 blur along each axis
    for _ in range(2):
        for ax in range(3):
            pad = np.pad(grid, [(1, 1) if a == ax else (0, 0) for a in range(3)])
            sl = [slice(None)] * 3
            a_, b_, c_ = list(sl), list(sl), list(sl)
            a_[ax], b_[ax], c_[ax] = slice(0, -2), slice(1, -1), slice(2, None)
            grid = 0.25 * pad[tuple(a_)] + 0.5 * pad[tuple(b_)] + 0.25 * pad[tuple(c_)]
    # occlusion: the density walked outward from each voxel (nearest voxel at each step)
    idx = np.stack(
        np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1
    ).reshape(-1, 3)
    centre = lo + idx * h
    d = centre - CRANIUM_CENTRE
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    acc = np.zeros(len(idx))
    for st in range(1, VOLUME_RAY + 1):
        q = np.floor((centre + d * (st * h) - lo) / h + 0.5).astype(int)
        ok = np.all((q >= 0) & (q < dims), axis=1)
        acc[ok] += grid[q[ok, 0], q[ok, 1], q[ok, 2]]
    occ = np.exp(-VOLUME_K * acc).reshape(dims)
    near = np.clip(grid / VOLUME_NEAR, 0, 1)
    out = np.stack([occ, near], -1)
    return np.round(out * 255).astype(np.uint8), lo, h


def sample_volume(volume, lo, h, P, channel=0):
    """trilinear value in [0, 1] of a channel of the density volume at points (n, 3)"""
    from scipy.ndimage import map_coordinates

    c = ((np.asarray(P) - lo) / h).T
    return map_coordinates(
        volume[..., channel].astype(np.float64) / 255.0, c, order=1, mode="nearest"
    )
