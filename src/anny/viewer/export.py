# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The shape space of anny for the viewer.

The page computes the blend-shape coefficients of anny's phenotype sliders with the same rule
as ``Anny._get_phenotype_blendshape_coefficients``: piecewise linear weights between the anchors
of each phenotype, normalised race weights, and one product over the phenotype mask per blend
shape. :func:`phenotype_tables` exports the anchors, the order of the variations and the mask.

The 624 blend shapes of anny are too large for the page, so :func:`compress_shapes` keeps the
principal components of the shape offsets that the sliders can reach. With ``c`` the
coefficients, the offsets are ``components.T @ (projection @ c)``. The export picks the number
of components that keeps every vertex within ``max_error`` of anny over random slider settings.

The bone heads are linear in the coefficients, so :func:`joint_tables` exports them in full.
"""

from __future__ import annotations

import numpy as np
import torch

from anny.models.model_data import PHENOTYPE_VARIATIONS


def phenotype_tables(model) -> dict:
    """the rule that turns anny's phenotype sliders into blend-shape coefficients"""
    anchors = {
        k: model.anchors[k].detach().cpu().tolist()
        for k in PHENOTYPE_VARIATIONS
        if k != "race"
    }
    mask = (
        model.stacked_phenotype_blend_shapes_mask.detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    return dict(
        labels=list(model.phenotype_labels),
        variations=[[k, list(v)] for k, v in PHENOTYPE_VARIATIONS.items()],
        anchors=anchors,
        extrapolate=bool(model.extrapolate_phenotypes),
        mask=["".join(str(int(x)) for x in row) for row in mask],
        default=0.5,
    )


def sample_coefficients(model, n: int, seed: int = 0, sliders=None) -> torch.Tensor:
    """coefficients for random settings of the given sliders (all by default; the others stay at
    0.5): half uniform, half pushed toward the ends"""
    rng = np.random.default_rng(seed)
    k = len(model.phenotype_labels)
    uniform = rng.random((n // 2, k))
    ends = rng.beta(0.35, 0.35, (n - n // 2, k))
    values = np.concatenate([uniform, ends])
    if sliders is not None:
        fixed = [
            i for i, label in enumerate(model.phenotype_labels) if label not in sliders
        ]
        values[:, fixed] = 0.5
    params = torch.tensor(values, dtype=model.template_vertices.dtype)
    empty = params.new_zeros(n, 0)
    with torch.no_grad():
        coeffs = model._get_phenotype_blendshape_coefficients(params, empty, empty)
    return coeffs[:, : model.stacked_phenotype_blend_shapes_mask.shape[0]]


def compress_shapes(
    model,
    vertex_ids: np.ndarray,
    max_error: float = 1e-3,
    samples: int = 6000,
    seed: int = 0,
    sliders=None,
) -> dict:
    """
    Principal components of anny's phenotype offsets on the given vertices.

    Returns components (K, n, 3), projection (K, N) and a report of the largest and the mean
    vertex error on held-out settings.
    """
    n_shapes = model.stacked_phenotype_blend_shapes_mask.shape[0]
    B = (
        model.blendshapes[:n_shapes, vertex_ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    B = B.reshape(n_shapes, -1)
    C = sample_coefficients(model, samples, seed, sliders).numpy()
    G = C.T @ C / len(C)
    w, U = np.linalg.eigh(G)
    sqrtG = (U * np.sqrt(np.clip(w, 0, None))) @ U.T
    _, _, Vt = np.linalg.svd(sqrtG @ B, full_matrices=False)
    C_test = sample_coefficients(model, 2000, seed + 1, sliders).numpy()
    exact = (C_test @ B).reshape(len(C_test), -1, 3)
    report = None
    for K in list(range(16, 400, 8)) + [len(Vt)]:
        comp = Vt[:K]
        approx = ((C_test @ B @ comp.T) @ comp).reshape(len(C_test), -1, 3)
        err = np.linalg.norm(approx - exact, axis=-1)
        report = dict(
            components=K, max_error=float(err.max()), mean_error=float(err.mean())
        )
        if err.max() <= max_error:
            break
    comp = Vt[: report["components"]]
    return dict(
        components=comp.reshape(len(comp), -1, 3),
        projection=comp @ B.T,
        report=report,
    )


def joint_tables(model) -> dict:
    """rest bone heads: template (B, 3) and their blend shapes (N, B, 3)"""
    n_shapes = model.stacked_phenotype_blend_shapes_mask.shape[0]
    return dict(
        bones=list(model.bone_labels),
        parents=[int(p) for p in model.bone_parents],
        template=model.template_bone_heads.detach().cpu().numpy(),
        blendshapes=model.bone_heads_blendshapes[:n_shapes].detach().cpu().numpy(),
    )


def coefficients(model, phenotype_kwargs: dict) -> np.ndarray:
    """anny's coefficients for one slider setting (for checks)"""
    params = torch.tensor(
        [[float(phenotype_kwargs.get(k, 0.5)) for k in model.phenotype_labels]],
        dtype=model.template_vertices.dtype,
    )
    empty = params.new_zeros(1, 0)
    with torch.no_grad():
        c = model._get_phenotype_blendshape_coefficients(params, empty, empty)
    return c[0, : model.stacked_phenotype_blend_shapes_mask.shape[0]].numpy()
