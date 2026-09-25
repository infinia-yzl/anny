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


# ------------------------------------------------------------------ face shapes
def face_tables(model, prior=None) -> dict:
    """
    The rule of anny's face shapes for the page: the parameters (names, groups, ranges), the
    blend-shape rows (parameter, direction, scale group), the landmark pairs that size each scale
    group with their sizes on the default body, and the face-shape distribution
    (:mod:`anny.faces.distribution`) as means and low-rank factors at its anchors.
    """
    from anny.faces.measurements import SCALE_GROUP_MEASUREMENTS
    from anny.models.face_shapes import GROUP_TO_SCALE, SCALE_GROUPS, face_shape_spec

    names = list(model.face_shape_labels)
    rows = [x for x in model.blendshape_labels if x.startswith("face_shape:")]
    landmarks = sorted(
        {k for pairs in SCALE_GROUP_MEASUREMENTS.values() for p in pairs for k in p}
    )
    out = dict(
        names=names,
        groups=[model.face_shape_groups[n] for n in names],
        group_order=[
            g
            for g in dict.fromkeys(p.group for p in face_shape_spec())
            if g in set(model.face_shape_groups.values())
        ],
        ranges=[list(model.face_shape_ranges[n]) for n in names],
        ends=[_face_ends(n) for n in names],
        row_param=[int(i) for i in model.face_shape_row_parameter],
        row_sign=[int(s) for s in model.face_shape_row_sign],
        row_scale=[int(i) for i in model.face_shape_row_scale_group],
        scale_groups=list(SCALE_GROUPS),
        scale_group_of=[
            SCALE_GROUPS.index(GROUP_TO_SCALE[g])
            for g in (model.face_shape_groups[n] for n in names)
        ],
        landmarks=landmarks,
        scale_pairs=[
            [
                [landmarks.index(a), landmarks.index(b)]
                for a, b in SCALE_GROUP_MEASUREMENTS[g]
            ]
            for g in SCALE_GROUPS
        ],
        reference_sizes=model.face_shape_reference_sizes.detach().cpu().tolist(),
        first_row=int(model.stacked_phenotype_blend_shapes_mask.shape[0])
        + len(model.facial_action_labels)
        + 2 * len(model.local_change_labels),
        rows=len(rows),
    )
    if prior is not None:
        out["prior"] = prior
    return out


ENDS = {
    "decr": "Less",
    "incr": "More",
    "backward": "Back",
    "compress": "Compressed",
    "uncompress": "Uncompressed",
}


def _face_ends(name: str) -> list[str]:
    """the words at the two ends of a face-shape slider, from its MakeHuman targets"""
    from anny.models.face_shapes import face_shape_parameter

    p = face_shape_parameter(name)
    if p.source == "ict":
        return ["-", "+"]
    if not p.negative:
        return ["None", "Full"]
    words = [t.split("/")[-1].split("-")[-1] for t in (p.negative[0], p.positive[0])]
    return [ENDS.get(w, w.capitalize()) for w in words]


def face_landmark_tables(model, labels) -> dict:
    """the named landmarks of the template (K, 3) and of the phenotype blend shapes (N, K, 3)"""
    idx = [model.craniofacial_landmark_labels.index(k) for k in labels]
    n = model.stacked_phenotype_blend_shapes_mask.shape[0]
    return dict(
        template=model.craniofacial_landmarks_template[idx].detach().cpu().numpy(),
        blendshapes=model.craniofacial_landmarks_blendshapes[:n][:, idx]
        .detach()
        .cpu()
        .numpy(),
    )


def face_offsets(model, vertex_ids: np.ndarray, threshold: float = 1e-6) -> dict:
    """
    The face-shape rows on the given vertices, as sparse offsets: for each row, the positions (in
    vertex_ids) and the offsets of the vertices that it moves; and the bone-head deltas of every
    row on the bones that any row moves.
    """
    first = face_tables(model)["first_row"]
    B = model.blendshapes[first:, vertex_ids].detach().cpu().numpy()
    J = model.bone_heads_blendshapes[first:].detach().cpu().numpy()
    starts, counts, ids, offsets = [], [], [], []
    for r in range(len(B)):
        moved = np.nonzero(np.linalg.norm(B[r], axis=1) > threshold)[0]
        starts.append(sum(counts))
        counts.append(len(moved))
        ids.append(moved)
        offsets.append(B[r, moved])
    bones = np.nonzero(np.abs(J).max(axis=(0, 2)) > threshold)[0]
    return dict(
        starts=starts,
        counts=counts,
        ids=np.concatenate(ids).astype(np.uint32),
        offsets=np.concatenate(offsets),
        bones=bones.tolist(),
        bone_deltas=J[:, bones],
    )


def face_prior_tables(model, rank: int = 24) -> tuple[dict, np.ndarray]:
    """the face-shape distribution for the page: anchors and means in the manifest, and low-rank
    factors (A, 2, F, rank) that keep the largest directions of each covariance"""
    from anny.faces.distribution import FaceShapeDistribution

    d = FaceShapeDistribution(model)
    means = d.means.detach().cpu().numpy()
    trils = d.scale_trils.detach().cpu().numpy()
    A, G, F, _ = trils.shape
    factors = np.zeros((A, G, F, rank), np.float32)
    kept = []
    for a in range(A):
        for g in range(G):
            C = trils[a, g] @ trils[a, g].T
            w, V = np.linalg.eigh(C)
            order = np.argsort(w)[::-1][:rank]
            factors[a, g] = V[:, order] * np.sqrt(np.clip(w[order], 0, None))
            kept.append(float(w[order].sum() / max(w.sum(), 1e-30)))
    meta = dict(
        age_anchors=d.age_anchors.tolist(),
        gender_anchors=d.gender_anchors.tolist(),
        means=np.round(means, 5).tolist(),
        weight_muscle_regression=np.round(
            d.weight_muscle_regression.cpu().numpy(), 5
        ).tolist(),
        weight_muscle_centre=d.weight_muscle_centre.tolist(),
        rank=rank,
        variance_kept=min(kept),
    )
    return meta, factors
