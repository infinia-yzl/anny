# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The fine body of the viewer on anny's default body, in the legacy frame of the authoring rig.

The body surface of anny (``topology="anny-quads"``) goes through :class:`MixedSubdivision`:
three levels of Catmull-Clark on the head and the neck, two elsewhere. Three detail layers then
follow the legacy 3D Model build (build/build_body.py):

- the eye sockets fit a sphere around each eyeball (``sculpt.fit_socket``); the eyeball sits on
  the sphere fitted to anny's own eye vertices;
- the lips close along the line of the mouth (``mouth.detect_stomion``, ``sculpt.close_lips``);
- the collarbones and the biceps get a relief (``relief.py``).

The detail layers are stored as displacements of the fine vertices, split into the normal and
the two tangent directions of the smooth surface. The viewer adds them after it subdivides the
body for the current phenotype, scaled with the local size of the body.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np

from anny.poses.authoring import posing as P
from anny.utils.subdivision import MixedSubdivision, faces_weighted_to_bones

from . import eyes

EYE_BASE_RANGES = {"l": (14598, 14670), "r": (14670, 14742)}
EYE_OFFSET = np.array(
    [0.0, -0.0005, 0.0012]
)  # legacy frame, mirrored in x for the right eye


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def vertex_normals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    length = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.where(length > 0, length, 1)


def tangent_frames(V, T, N):
    """an orthonormal frame (t1, t2, n) per vertex: t1 along the first edge, projected"""
    edge = np.zeros_like(V)
    np.add.at(edge, T[:, 0], V[T[:, 1]] - V[T[:, 0]])
    np.add.at(edge, T[:, 1], V[T[:, 2]] - V[T[:, 1]])
    np.add.at(edge, T[:, 2], V[T[:, 0]] - V[T[:, 2]])
    t1 = edge - (edge * N).sum(1, keepdims=True) * N
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(N, t1)
    return np.stack([t1, t2, N], 1)  # (n, 3, 3), rows


def sphere_fit(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P**2).sum(1)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    center = c[:3]
    return center, float(np.sqrt(c[3] + (center**2).sum()))


# ------------------------------------------------------------------ sculpt (build/sculpt.py)
def fit_socket(V, T, center, R, inset=0.00015, reach=0.0045):
    """pull socket-facing vertices onto the sphere (center, R - inset)"""
    N = vertex_normals(V, T)
    d = V - center
    dist = np.linalg.norm(d, axis=1)
    dirn = d / dist[:, None]
    facing = -(N * dirn).sum(1)  # > 0 when the normal points toward the eye centre
    near = dist < R + reach
    w = (
        smoothstep(0.05, 0.45, facing)
        * near
        * smoothstep(R + reach, R + reach * 0.5, dist)
    )
    target = center + dirn * (R - inset)
    return V + (target - V) * w[:, None]


def detect_stomion(V, T, y0=0.430, y1=0.462):
    """the line where the lips meet, seen from the front: (x, y, front z) per step of x"""
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    inter = RayMeshIntersector(trimesh.Trimesh(V, T, process=False))
    xs = np.arange(0, 0.026, 0.0005)
    ys = np.arange(y0, y1, 0.00002)
    out_x, out_y, zfront = [], [], []
    for x in xs:
        origins = np.stack([np.full_like(ys, x), ys, np.full_like(ys, 0.3)], 1)
        D = np.tile([0, 0, -1.0], (len(ys), 1))
        locs, ri, _ = inter.intersects_location(origins, D, multiple_hits=False)
        z = np.full(len(ys), np.nan)
        z[ri] = locs[:, 2]
        zf = np.nanmax(z)
        deep = z < zf - 0.03
        if not deep.any():
            break
        out_x.append(x)
        out_y.append(ys[np.where(deep)[0]].mean())
        zfront.append(zf)
    return np.array(out_x), np.array(out_y), np.array(zfront)


def close_lips(V, xs, ys, xc, zf):
    ax = np.abs(V[:, 0])
    ysx = np.interp(ax, xs, ys)
    dy = V[:, 1] - ysx
    w = np.exp(-((dy / 0.0011) ** 2))
    w *= smoothstep(xc + 0.0015, xc - 0.0015, ax) * smoothstep(
        zf - 0.03, zf - 0.014, V[:, 2]
    )
    V = V.copy()
    V[:, 1] = ysx + dy * (1 - 1.25 * w)
    return V


@dataclasses.dataclass
class FineBody:
    subdivision: MixedSubdivision
    T: np.ndarray  # fine triangles
    V_smooth: np.ndarray  # subdivided default body, legacy frame
    V: np.ndarray  # with the detail layers
    N: np.ndarray  # normals of V
    detail: np.ndarray  # V - V_smooth
    detail_local: np.ndarray  # the detail in the (t1, t2, n) frames of V_smooth
    relief_height: (
        np.ndarray
    )  # the part of detail_local[:, 2] that comes from the relief
    eye_centers: dict  # 'l', 'r' -> centre of the eyeball (legacy frame)
    eye_radius: float
    eye_sphere: (
        dict  # 'l', 'r' -> (centre, radius) of anny's eye vertices (legacy frame)
    )
    mouth: dict
    body_faces: np.ndarray  # mask of anny's quads that the fine body uses
    dense_weights: np.ndarray  # anny's skinning weights (coarse vertices x bones)


@functools.lru_cache(maxsize=1)
def fine_body() -> FineBody:
    from . import relief

    rig = P.RIG
    coarse = rig.preview["coarse"]
    base_index = coarse["base_index"]
    quads_all = _all_quads()
    body_faces = (base_index[quads_all] < 13380).all(1)
    subdivision = _subdivision(quads_all, body_faces)
    V = subdivision(coarse["V"])
    T = subdivision.triangles
    V_smooth = V.copy()

    # eyes: the sphere of anny's eye vertices gives the centre, as head.eye_sphere did
    eye_sphere, centers = {}, {}
    for side, (lo, hi) in EYE_BASE_RANGES.items():
        sel = (base_index >= lo) & (base_index < hi)
        c, r = sphere_fit(coarse["V"][sel])
        eye_sphere[side] = (c, r)
        sgn = 1.0 if c[0] > 0 else -1.0
        centers[side] = c + EYE_OFFSET * np.array([sgn, 1.0, 1.0])
    R = eyes.R_SCLERA
    for side in "lr":
        V = fit_socket(V, T, centers[side], R)
    mx, my, mz = detect_stomion(V, T)
    mouth = dict(xs=mx.tolist(), ys=my.tolist(), zf=mz.tolist())
    if len(mx) > 3:
        ysm = np.convolve(np.pad(my, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        V = close_lips(V, mx, ysm, mx[-1] + 0.0008, float(np.median(mz)))
    N = vertex_normals(V, T)
    arm_w = relief.arm_weights(subdivision, coarse["W"])
    bump = relief.relief(V, N, arm_w)
    V = V + bump
    detail = V - V_smooth
    frames = tangent_frames(V_smooth, T, vertex_normals(V_smooth, T))
    detail_local = np.einsum("nij,nj->ni", frames, detail)
    # the relief alone, along the normal of the smooth surface (the page fades it with the sliders)
    relief_height = np.einsum("nj,nj->n", frames[:, 2], bump)
    return FineBody(
        subdivision=subdivision,
        T=T,
        V_smooth=V_smooth,
        V=V,
        N=vertex_normals(V, T),
        detail=detail,
        detail_local=detail_local,
        relief_height=relief_height,
        eye_centers=centers,
        eye_radius=R,
        eye_sphere=eye_sphere,
        mouth=mouth,
        body_faces=body_faces,
        dense_weights=coarse["W"],
    )


def _all_quads():
    import anny

    model = anny.Anny(topology="anny-quads")
    return model.faces.numpy()


def _subdivision(quads_all, body_faces):
    import anny

    model = anny.Anny(topology="anny-quads")
    faces = quads_all[body_faces]
    region = faces_weighted_to_bones(model, faces)
    return MixedSubdivision(model.template_vertices.shape[0], faces, region)
