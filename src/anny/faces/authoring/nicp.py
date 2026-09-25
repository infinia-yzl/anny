# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Registration tools: rigid alignment from landmark pairs (Umeyama 1991) and a non-rigid
refinement in the spirit of optimal-step non-rigid ICP (Amberg et al., CVPR 2007): the source
surface moves toward its closest points on the target, while a Laplacian term keeps the
displacement smooth, with a stiffness that decreases over the rounds.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from anny.hair import closest_triangles


def umeyama(src: np.ndarray, dst: np.ndarray, scale: bool = False):
    """rotation R, translation t and scale s with dst ~ s R src + t (least squares)"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / (S**2).sum(1).mean() if scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def uniform_laplacian(n: int, triangles: np.ndarray) -> scipy.sparse.csr_matrix:
    """L = D - A of the edge graph of the triangles"""
    i = np.concatenate([triangles[:, k] for k in (0, 1, 2, 1, 2, 0)])
    j = np.concatenate([triangles[:, k] for k in (1, 2, 0, 0, 1, 2)])
    A = scipy.sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return (scipy.sparse.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


def refine(
    V: np.ndarray,
    T: np.ndarray,
    target_V: np.ndarray,
    target_T: np.ndarray,
    stiffness=(30.0, 10.0, 3.0, 1.0),
    iterations: int = 3,
    max_distance: float = 0.01,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """move the vertices V (with triangles T) onto the surface of (target_V, target_T)"""
    n = len(V)
    L = uniform_laplacian(n, T)
    LtL = (L.T @ L).tocsr()
    X = V.copy()
    base = np.ones(n) if weights is None else weights
    for alpha in stiffness:
        start = X.copy()
        for _ in range(iterations):
            tri, bary, d = closest_triangles(X, target_V, target_T)
            C = np.einsum("nk, nkd -> nd", bary, target_V[target_T[tri]])
            w = base * (d < max_distance)
            W = scipy.sparse.diags(w)
            A = (W + alpha * LtL).tocsc()
            solve = scipy.sparse.linalg.factorized(A)
            rhs = W @ C + alpha * (LtL @ start)
            X = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
    return X


def correspondences(
    V: np.ndarray, target_V: np.ndarray, target_T: np.ndarray, max_distance: float
):
    """for each vertex of V: the closest target triangle, its barycentric coordinates, and
    whether the distance stays below max_distance"""
    tri, bary, d = closest_triangles(V, target_V, target_T)
    return tri, bary, d < max_distance, d
