# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Hair for anny: strands bound to the skin, so that they follow every phenotype.

The groom grows once on anny's default body (``anny.hair.authoring``). :class:`StrandBinding`
then ties each strand to the skin: its root sits on a triangle of the fine body mesh
(barycentric coordinates), and its points are stored in the frame of that triangle, divided by
the size of the scalp. For another body, the root follows the triangle, and the strand turns
with the triangle and scales with the size of the scalp. The same binding carries the brows and
the lashes.

Example::

    binding = StrandBinding(strands, V_default, triangles)
    moved = binding.follow(V_other)   # (n, points, 3)
"""

from __future__ import annotations

import numpy as np


def triangle_frames(V, T, tri):
    """orthonormal frames (rows: edge direction, in-plane normal, face normal) of triangles"""
    A, B, C = V[T[tri, 0]], V[T[tri, 1]], V[T[tri, 2]]
    e1 = B - A
    e1 = e1 / np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-18)
    n = np.cross(B - A, C - A)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-18)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n], -2)


def closest_triangles(points, V, T):
    """
    Closest triangle of a mesh for each point, with barycentric coordinates of the projected
    point (clamped to the triangle). The search looks at the triangles around the nearest
    vertices, which is enough for points on or near the surface.
    """
    from scipy.spatial import cKDTree

    n_v = len(V)
    order = np.argsort(T.ravel(), kind="stable")
    counts = np.bincount(T.ravel(), minlength=n_v)
    starts = np.concatenate([[0], np.cumsum(counts)])
    _, nearest = cKDTree(V).query(points, k=4)
    best_tri = np.zeros(len(points), np.int64)
    best_bary = np.zeros((len(points), 3))
    best_d = np.full(len(points), np.inf)
    for k in range(nearest.shape[1]):
        v = nearest[:, k]
        for j in range(int(counts.max())):
            has = counts[v] > j
            if not has.any():
                break
            rows = np.nonzero(has)[0]
            tri = order[starts[v[rows]] + j] // 3
            bary = _project(points[rows], V, T, tri)
            proj = np.einsum("nk,nkd->nd", bary, V[T[tri]])
            d = np.linalg.norm(points[rows] - proj, axis=1)
            better = d < best_d[rows]
            r = rows[better]
            best_d[r] = d[better]
            best_tri[r] = tri[better]
            best_bary[r] = bary[better]
    return best_tri, best_bary, best_d


def _project(P, V, T, tri):
    """barycentric coordinates of the closest point of each triangle (clamped to it)"""
    A, B, C = V[T[tri, 0]], V[T[tri, 1]], V[T[tri, 2]]
    v0, v1, v2 = B - A, C - A, P - A
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bary = np.stack([1 - v - w, v, w], 1)
    bary = np.clip(bary, 0, None)
    return bary / np.maximum(bary.sum(1, keepdims=True), 1e-12)


def scalp_size(roots: np.ndarray) -> float:
    """size of a set of root points: the root mean square distance to their centre"""
    return float(np.sqrt(((roots - roots.mean(0)) ** 2).sum(1).mean()))


class StrandBinding:
    """
    Polylines (n, points, 3) bound to a triangle mesh (V, T). The first point of each polyline
    is its root.
    """

    def __init__(self, strands: np.ndarray, V: np.ndarray, T: np.ndarray):
        strands = np.asarray(strands, dtype=np.float64)
        self.T = np.asarray(T)
        self.triangles, self.barycentric, self.root_distance = closest_triangles(
            strands[:, 0], V, self.T
        )
        roots = self.roots(V)
        self.reference_size = scalp_size(roots)
        frames = triangle_frames(V, self.T, self.triangles)
        rel = strands - roots[:, None, :]
        self.local = np.einsum("nij,nmj->nmi", frames, rel) / self.reference_size

    def roots(self, V: np.ndarray) -> np.ndarray:
        return np.einsum("nk,nkd->nd", self.barycentric, V[self.T[self.triangles]])

    def follow(self, V: np.ndarray, scale: float | None = None) -> np.ndarray:
        """the strands on a mesh with the same triangles and new vertices V"""
        roots = self.roots(V)
        size = scalp_size(roots) if scale is None else scale * self.reference_size
        frames = triangle_frames(V, self.T, self.triangles)
        return roots[:, None, :] + size * np.einsum("nji,nmj->nmi", frames, self.local)
