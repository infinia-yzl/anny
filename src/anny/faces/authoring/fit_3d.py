# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Fit anny's face shapes to the ICT-FaceKit identity space (see :mod:`anny.faces.authoring.ict`).

1. **Registration.** Anny's adult body (gender 0.5) and the ICT neutral head are aligned on 15
   landmark pairs. Anny's face shapes, gender, weight and muscle are then fitted to the ICT
   surface (closest points and landmarks), and a non-rigid refinement
   (:mod:`anny.faces.authoring.nicp`) lays anny's head on the ICT surface. The closest ICT
   point of each vertex of anny's head then corresponds to it. The ears, the eye and mouth
   cavities and the eyes stay out: closest points are unreliable in the folds of the ear, and
   ANSUR II calibrates the ears.
2. **Fits.** For each random ICT identity (weights from a standard normal over the 100 modes),
   anny's face values, gender, weight, muscle, adult age and a rigid transform are fitted to the
   corresponding points by least squares with a small ridge penalty (Adam in PyTorch).
3. **Report.** On held-out identities, the RMS residual and the share of the ICT identity
   variation that anny's face space explains.

Usage::

    python -m anny.faces.authoring.fit_3d [--samples 10000] [--held-out 1000]

The results go to ``ANNY_CACHE_DIR/faces/ict_fits.npz``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import roma
import torch

import anny
from anny.faces.authoring import ict, nicp
from anny.faces.authoring.base_mesh import base_mesh, read_target
from anny.faces.authoring.sources import cache_dir
from anny.faces.measurements import craniofacial_landmark_vertices
from anny.shape_distribution import SimpleShapeDistribution

FIT_PHENOTYPES = ("gender", "age", "muscle", "weight")
ADULT_YEARS = (18.0, 67.0)


def adult_age_range(model) -> tuple[float, float]:
    """anny ages of 18 and 67 years (the ICT subjects) on anny's morphological age scale"""
    mapping = SimpleShapeDistribution(model).morphological_age_mapping
    years = torch.tensor(ADULT_YEARS, dtype=model.dtype)
    lo, hi = mapping.morphological_to_anny_age(years).tolist()
    return lo, hi


def fitting_model() -> anny.Anny:
    return anny.Anny(face_shapes="all").to(dtype=torch.float64)


def head_vertices(model) -> np.ndarray:
    """vertices of the model that take part in the fit: the skin of the head, without the ears,
    the eye and mouth cavities and the eyes"""
    mesh = base_mesh()
    base = model.base_mesh_vertex_indices.numpy()
    V = mesh.V
    sm = craniofacial_landmark_vertices()["sm"][0]
    head = V[base, 2] > V[sm, 2] - 0.02
    body = np.zeros(len(V), bool)
    body[mesh.body_vertices] = True
    cavities = np.zeros(len(V), bool)
    for label in ("eye_cavity.L", "eye_cavity.R", "mouth_cavity"):
        cavities[np.unique(mesh.quads[mesh.face_labels == label])] = True
    ears = np.zeros(len(V), bool)
    for side in ("l", "r"):
        idx, d = read_target(f"ears/{side}-ear-scale-incr")
        mag = np.linalg.norm(d, axis=1)
        ears[idx[mag > 0.05 * mag.max()]] = True
    keep = head & body[base] & ~cavities[base] & ~ears[base]
    return np.nonzero(keep)[0]


class HeadModel(torch.nn.Module):
    """rest positions of a subset of vertices for phenotype and face values"""

    def __init__(self, model, vertex_ids):
        super().__init__()
        self.model = model
        ids = torch.as_tensor(vertex_ids)
        self.register_buffer("template", model.template_vertices[ids])
        B = model.blendshapes[:, ids]
        self.register_buffer("B", B.reshape(len(B), -1))

    def forward(self, phenotype: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        n = phenotype.shape[0]
        empty = phenotype.new_zeros((n, 0))
        coeffs = self.model._get_phenotype_blendshape_coefficients(
            phenotype, empty, empty, face
        )
        return self.template[None] + (coeffs @ self.B).view(n, -1, 3)


@dataclasses.dataclass
class Fit:
    phenotype: torch.Tensor  # (B, P) all the phenotype labels of the model
    face: torch.Tensor  # (B, F)
    rotation: torch.Tensor  # (B, 3, 3), applied to anny: R x + t ~ target
    translation: torch.Tensor  # (B, 3)
    rms: torch.Tensor  # (B,) weighted RMS distance (m)


def fit(
    head: HeadModel,
    targets: torch.Tensor,
    weights: torch.Tensor,
    age_range: tuple[float, float],
    steps: int = 400,
    ridge: float = 1e-8,
    init: Fit | None = None,
    extra_loss=None,
) -> Fit:
    """
    Least-squares fit of phenotype (gender, age, muscle, weight), face values and a rigid
    transform so that anny's head vertices meet the targets (B, H, 3).
    """
    model = head.model
    labels = model.phenotype_labels
    n = targets.shape[0]
    dtype = targets.dtype
    free = [labels.index(k) for k in FIT_PHENOTYPES]
    lo = torch.tensor([0.0, age_range[0], 0.0, 0.0], dtype=dtype)
    hi = torch.tensor([1.0, age_range[1], 1.0, 1.0], dtype=dtype)

    def to_raw(values):
        u = ((values - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        return torch.log(u / (1 - u))

    if init is None:
        raw = to_raw(0.5 * (lo + hi)).expand(n, -1).clone()
        face = torch.zeros((n, len(model.face_shape_labels)), dtype=dtype)
        rotvec = torch.zeros((n, 3), dtype=dtype)
        trans = torch.zeros((n, 3), dtype=dtype)
    else:
        raw = to_raw(init.phenotype[:, free]).expand(n, -1).clone()
        face = init.face.expand(n, -1).clone()
        rotvec = roma.rotmat_to_rotvec(init.rotation).expand(n, -1).clone()
        trans = init.translation.expand(n, -1).clone()
    params = [raw, face, rotvec, trans]
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            dict(params=[raw], lr=0.05),
            dict(params=[face], lr=0.03),
            dict(params=[rotvec], lr=0.003),
            dict(params=[trans], lr=0.0005),
        ]
    )
    w = weights / weights.sum()
    base = torch.full((n, len(labels)), 0.5, dtype=dtype)

    def evaluate():
        phenotype = base.clone()
        phenotype[:, free] = lo + (hi - lo) * torch.sigmoid(raw)
        X = head(phenotype, face)
        R = roma.rotvec_to_rotmat(rotvec)
        Y = torch.einsum("bij, bhj -> bhi", R, X) + trans[:, None]
        sq = ((Y - targets) ** 2).sum(-1)
        return phenotype, R, Y, sq

    for step in range(steps):
        optimizer.zero_grad()
        phenotype, R, Y, sq = evaluate()
        loss = (sq * w).sum(-1).sum() + ridge * (face**2).sum()
        if extra_loss is not None:
            loss = loss + extra_loss(phenotype, face, R, trans)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        phenotype, R, Y, sq = evaluate()
        rms = torch.sqrt((sq * w).sum(-1))
    return Fit(
        phenotype=phenotype.detach(),
        face=face.detach(),
        rotation=R.detach(),
        translation=trans.detach(),
        rms=rms,
    )


# ------------------------------------------------------------------ registration
def register(model, target_V, target_T, target_landmarks: dict, max_distance=0.006):
    """
    Correspondences between anny's head vertices and a target head surface: for each vertex, a
    target triangle and barycentric coordinates, and whether the pair is valid.
    """
    ids = head_vertices(model)
    head = HeadModel(model, ids)
    age_range = adult_age_range(model)
    labels = model.craniofacial_landmark_labels
    names = [k for k in target_landmarks if k in labels]
    # anny's adult body, and the rigid alignment of the target onto it from the landmarks
    adult = {"age": sum(age_range) / 2}
    params = model._parse_parameter_kwargs(adult, model.phenotype_labels, 0.5, "")
    anny_landmarks = model.phenotype_craniofacial_landmarks(params)[0].numpy()
    src = np.stack([target_landmarks[k] for k in names])
    dst = np.stack([anny_landmarks[labels.index(k)] for k in names])
    R, t, _ = nicp.umeyama(src, dst)
    TV = target_V @ R.T + t
    target_lm = {k: target_landmarks[k] @ R.T + t for k in names}

    # anny's face shapes fitted to the target surface (closest points, refreshed in rounds)
    weights = torch.ones(len(ids), dtype=torch.float64)
    init = None
    lm_ids = torch.tensor([labels.index(k) for k in names])
    lm_target = torch.tensor(np.stack([target_lm[k] for k in names]))

    def landmark_loss(phenotype, face, R_, trans):
        n = phenotype.shape[0]
        empty = phenotype.new_zeros((n, 0))
        coeffs = model._get_phenotype_blendshape_coefficients(
            phenotype, empty, empty, face
        )
        L = model.craniofacial_landmarks_template[None] + torch.einsum(
            "bn, nkd -> bkd", coeffs, model.craniofacial_landmarks_blendshapes
        )
        L = torch.einsum("bij, bkj -> bki", R_, L[:, lm_ids]) + trans[:, None]
        return 0.2 * ((L - lm_target[None]) ** 2).sum(-1).mean()

    for round_ in range(6):
        with torch.no_grad():
            phenotype = params if init is None else init.phenotype
            face = (
                torch.zeros((1, len(model.face_shape_labels)), dtype=torch.float64)
                if init is None
                else init.face
            )
            X = head(phenotype, face)[0]
            if init is not None:
                X = X @ init.rotation[0].T + init.translation[0]
        tri, bary, d = nicp.closest_triangles(X.numpy(), TV, target_T)
        C = np.einsum("nk, nkd -> nd", bary, TV[target_T[tri]])
        valid = torch.tensor(d < 0.02, dtype=torch.float64)
        init = fit(
            head,
            torch.tensor(C)[None],
            weights * valid,
            age_range,
            steps=300,
            ridge=1e-8,
            init=init,
            extra_loss=landmark_loss,
        )
        print(f"registration round {round_}: rms {1000 * init.rms.item():.2f} mm")

    # non-rigid refinement of anny's head onto the target, then the correspondences
    with torch.no_grad():
        X = head(init.phenotype, init.face)[0]
        X = (X @ init.rotation[0].T + init.translation[0]).numpy()
    faces = model.faces.numpy()
    local = -np.ones(len(model.template_vertices), np.int64)
    local[ids] = np.arange(len(ids))
    tri_local = local[faces[(local[faces] >= 0).all(1)]]
    X = nicp.refine(X, tri_local, TV, target_T)
    tri, bary, ok, d = nicp.correspondences(X, TV, target_T, max_distance)
    print(
        f"correspondences: {ok.sum()} of {len(ok)} vertices within {1000 * max_distance:.0f} mm"
    )
    return dict(
        vertex_ids=ids,
        triangles=tri,
        bary=bary,
        valid=ok,
        rotation=R,
        translation=t,
        neutral_fit=init,
    )


# ------------------------------------------------------------------ fits of ICT identities
def targets_for(ict_model, reg, weights: np.ndarray) -> torch.Tensor:
    """corresponding points (B, H, 3) of ICT identities, in anny's frame"""
    meshes = ict_model.sample(weights) @ reg["rotation"].T + reg["translation"]
    T = ict_model.triangles[reg["triangles"]]
    pts = np.einsum("hk, bhkd -> bhd", reg["bary"], meshes[:, T])
    return torch.tensor(pts)


def run(samples=10000, held_out=1000, batch=500, seed=0, steps=300):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = fitting_model()
    ict_model = ict.load()
    t0 = time.time()
    reg = register(model, ict_model.V, ict_model.triangles, ict_model.landmarks())
    print(f"registration: {time.time() - t0:.0f} s")
    ids = reg["vertex_ids"]
    head = HeadModel(model, ids)
    age_range = adult_age_range(model)
    weights = torch.tensor(reg["valid"], dtype=torch.float64)
    rng = np.random.default_rng(seed)
    total = samples + held_out
    beta = rng.standard_normal((total, ict_model.modes.shape[0]))
    beta[0] = 0.0  # the ICT mean identity first
    out = {k: [] for k in ("phenotype", "face", "rms", "rotation", "translation")}
    for start in range(0, total, batch):
        targets = targets_for(ict_model, reg, beta[start : start + batch])
        f = fit(
            head,
            targets,
            weights,
            age_range,
            steps=steps,
            init=reg["neutral_fit"],
        )
        for k in out:
            out[k].append(getattr(f, k).numpy())
        print(
            f"{start + len(targets)}/{total}: median rms {1000 * np.median(f.rms.numpy()):.2f} mm"
            f" ({time.time() - t0:.0f} s)"
        )
    result = {k: np.concatenate(v) for k, v in out.items()}
    result.update(
        beta=beta,
        vertex_ids=ids,
        corr_triangles=reg["triangles"],
        corr_bary=reg["bary"],
        corr_valid=reg["valid"],
        align_rotation=reg["rotation"],
        align_translation=reg["translation"],
        held_out=np.arange(total) >= samples,
        phenotype_labels=np.array(model.phenotype_labels),
        face_labels=np.array(model.face_shape_labels),
        age_range=np.array(age_range),
    )
    np.savez(cache_dir() / "ict_fits.npz", **result)
    report(result, model, ict_model, head)
    return result


def report(result, model, ict_model, head) -> dict:
    """RMS residual and explained identity variation on the held-out identities"""
    held = np.nonzero(result["held_out"])[0]
    reg = dict(
        rotation=result["align_rotation"],
        translation=result["align_translation"],
        triangles=result["corr_triangles"],
        bary=result["corr_bary"],
    )
    w = result["corr_valid"].astype(np.float64)
    w /= w.sum()
    targets = targets_for(ict_model, reg, result["beta"][held]).numpy()
    target0 = targets_for(ict_model, reg, result["beta"][:1]).numpy()[0]
    with torch.no_grad():
        X = head(
            torch.tensor(result["phenotype"][held]), torch.tensor(result["face"][held])
        ).numpy()
        X0 = head(
            torch.tensor(result["phenotype"][:1]), torch.tensor(result["face"][:1])
        ).numpy()[0]
    R, t = result["rotation"][held], result["translation"][held]
    fitted = np.einsum("bij, bhj -> bhi", R, X) + t[:, None]
    fitted0 = X0 @ result["rotation"][0].T + result["translation"][0]
    variation = ((targets - target0[None]) ** 2).sum(-1) @ w
    unexplained = (((targets - fitted) - (target0 - fitted0)[None]) ** 2).sum(-1) @ w
    info = dict(
        held_out=len(held),
        rms_mm_median=float(1000 * np.median(result["rms"][held])),
        rms_mm_p90=float(1000 * np.percentile(result["rms"][held], 90)),
        ict_variation_rms_mm=float(1000 * np.sqrt(variation.mean())),
        explained_variance=float(1 - unexplained.sum() / variation.sum()),
        face_values_beyond_range=float((np.abs(result["face"][held]) > 1).mean()),
    )
    print(json.dumps(info, indent=1))
    with open(cache_dir() / "ict_fit_report.json", "w") as f:
        json.dump(info, f, indent=1)
    return info


def main():
    parser = argparse.ArgumentParser(
        description="fit anny's face shapes to ICT identities"
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--held-out", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    run(args.samples, args.held_out, args.batch, steps=args.steps)


if __name__ == "__main__":
    main()
