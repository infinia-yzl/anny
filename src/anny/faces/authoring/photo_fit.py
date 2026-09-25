# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Fit anny to the MediaPipe landmarks of photographs, to measure how well anny's faces reach
real ones.

Each near-frontal, neutral FairFace photo is fitted with anny's phenotype (the age within the
photo's age group, the gender on the side of its label, weight and muscle), anny's facial
actions (started from MediaPipe's expression scores, which use the same ARKit names), a
weak-perspective camera, and either anny's face shapes (with the calibrated distribution of
:mod:`anny.faces.distribution` as prior) or none. The landmarks of anny come from
``data/keypoints/mediapipe.json`` (:mod:`anny.faces.authoring.mediapipe_map`).

The error of a fit is the normalised mean error (NME): the mean distance between the projected
and the detected landmarks, divided by the distance between the outer eye corners.
"""

from __future__ import annotations

import numpy as np
import roma
import torch

import anny
from anny.faces.authoring.mediapipe_map import mediapipe_regressor
from anny.faces.authoring.photos import AGE_YEARS, MP
from anny.faces.distribution import FaceShapeDistribution
from anny.shape_distribution import SimpleShapeDistribution

INDICES = list(range(468)) + [468, 473]


class LandmarkModel(torch.nn.Module):
    """MediaPipe landmarks of anny's rest body for phenotype, face and expression values"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        reg, _ = mediapipe_regressor(model, INDICES)
        idx, w = reg.regression_indices, reg.regression_weights
        used, inverse = torch.unique(idx, return_inverse=True)
        self.register_buffer("template", model.template_vertices[used])
        B = model.blendshapes[:, used]
        self.register_buffer("B", B.reshape(len(B), -1))
        self.register_buffer("idx", inverse)
        self.register_buffer("w", w.to(model.dtype))

    def forward(self, phenotype, face, expression):
        n = phenotype.shape[0]
        local = phenotype.new_zeros((n, 0))
        coeffs = self.model._get_phenotype_blendshape_coefficients(
            phenotype, local, expression, face
        )
        V = self.template[None] + (coeffs @ self.B).view(n, -1, 3)
        return torch.einsum("ks, bksd -> bkd", self.w, V[:, self.idx])


def fit_photos(
    landmarks: np.ndarray,
    scores: np.ndarray,
    score_names: list[str],
    age_group: np.ndarray,
    gender: np.ndarray,
    with_face: bool,
    steps: int = 300,
    batch: int = 256,
    prior_weight: float = 1.0,
) -> dict:
    """fits of photos: landmarks (n, 478, 3) in pixels; returns NME (n,) and the fitted values"""
    model = anny.Anny(face_shapes="all", facial_actions="all").to(dtype=torch.float64)
    lm_model = LandmarkModel(model)
    prior = FaceShapeDistribution(model)
    mapping = SimpleShapeDistribution(model).morphological_age_mapping
    labels = model.phenotype_labels
    F = len(model.face_shape_labels)
    actions = model.facial_action_labels
    score_idx = [score_names.index(a) if a in score_names else -1 for a in actions]
    out = dict(nme=[], face=[], phenotype=[])
    for start in range(0, len(landmarks), batch):
        sl = slice(start, start + batch)
        obs = torch.tensor(landmarks[sl][:, INDICES, :2])
        n = len(obs)
        iod = (
            obs[:, INDICES.index(MP["ex_r"])] - obs[:, INDICES.index(MP["ex_l"])]
        ).norm(dim=-1)
        # bounds of the phenotype: the age group and the side of the gender label
        years = torch.tensor([AGE_YEARS[g] for g in age_group[sl]], dtype=torch.float64)
        age_lo = mapping.morphological_to_anny_age(years[:, 0])
        age_hi = mapping.morphological_to_anny_age(years[:, 1])
        g = torch.tensor(gender[sl], dtype=torch.float64)
        lo = torch.stack([0.5 * g, age_lo, torch.zeros(n), torch.zeros(n)], -1)
        hi = torch.stack([0.5 + 0.5 * g, age_hi, torch.ones(n), torch.ones(n)], -1)
        free = [labels.index(k) for k in ("gender", "age", "weight", "muscle")]
        raw = torch.zeros((n, 4), dtype=torch.float64, requires_grad=True)
        face = torch.zeros((n, F), dtype=torch.float64, requires_grad=with_face)
        s = np.nan_to_num(scores[sl])
        e0 = torch.tensor(
            np.stack([s[:, i] if i >= 0 else np.zeros(n) for i in score_idx], -1)
        ).clamp(1e-3, 1 - 1e-3)
        e_raw = torch.log(e0 / (1 - e0)).clone().requires_grad_(True)
        rotvec = torch.zeros((n, 3), dtype=torch.float64, requires_grad=True)
        log_scale = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        shift = torch.zeros((n, 2), dtype=torch.float64, requires_grad=True)

        def phenotypes():
            p = torch.full((n, len(labels)), 0.5, dtype=torch.float64)
            p[:, free] = lo + (hi - lo) * torch.sigmoid(raw)
            return p

        # the camera from the eye corners and the centroid
        with torch.no_grad():
            L0 = lm_model(phenotypes(), face.detach(), torch.sigmoid(e_raw))
            a_iod = (
                L0[:, INDICES.index(MP["ex_r"])] - L0[:, INDICES.index(MP["ex_l"])]
            ).norm(dim=-1)
            scale0 = iod / a_iod
            c_obs = obs.mean(1)
        if with_face:
            # the calibrated prior at the middle of the bounds: mean and precision (with a floor)
            with torch.no_grad():
                mean, tril = prior.parameters_for(phenotypes())
                C = tril @ tril.transpose(1, 2) + 1e-3 * torch.eye(
                    F, dtype=torch.float64
                )
                precision = torch.linalg.inv(C)
                face.data.copy_(mean)
        params = [raw, e_raw, rotvec, log_scale, shift] + ([face] if with_face else [])
        opt = torch.optim.Adam(params, lr=0.03)
        P0 = torch.tensor([[1.0, 0, 0], [0, 0, -1.0]], dtype=torch.float64)

        def project(L):
            R = roma.rotvec_to_rotmat(rotvec)
            X = torch.einsum("bij, bkj -> bki", R, L - L.mean(1, keepdim=True))
            uv = torch.einsum("ij, bkj -> bki", P0, X)
            return (scale0 * torch.exp(log_scale))[:, None, None] * uv + (
                c_obs + shift * iod[:, None]
            )[:, None]

        for _ in range(steps):
            opt.zero_grad()
            phen = phenotypes()
            L = lm_model(phen, face, torch.sigmoid(e_raw))
            err = ((project(L) - obs) ** 2).sum(-1).mean(-1) / iod**2
            loss = err.sum() + 0.01 * ((torch.sigmoid(e_raw) - e0) ** 2).sum()
            if with_face:
                # MAP: the landmarks with a noise of about 2 % of the eye width, and the
                # Mahalanobis distance under the calibrated prior
                d = face - mean
                maha = torch.einsum("bf, bfg, bg -> b", d, precision, d)
                loss = loss + prior_weight * 1e-6 * maha.sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            phen = phenotypes()
            L = lm_model(phen, face, torch.sigmoid(e_raw))
            nme = ((project(L) - obs).norm(dim=-1).mean(-1) / iod).numpy()
        out["nme"].append(nme)
        out["face"].append(face.detach().numpy())
        out["phenotype"].append(phen.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
