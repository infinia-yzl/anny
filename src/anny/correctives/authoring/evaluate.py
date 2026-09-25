# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Compare plain skinning and the corrective shapes with a full simulation of library poses.

    ANNY_AUTHORING_PHENOTYPE='{"age": 0.3333}' \
        python -m anny.correctives.authoring.evaluate t_pose reach_up seated crouch

The body is the authoring body (``anny.poses.authoring.rig.authoring_phenotype``), so a run
with other slider values checks how the shapes, fitted on anny's default body and scaled with
the size of their support, carry over to that body. For each pose the simulation moves the
whole body, and the report counts the skin vertices where plain skinning lies more than 2 mm
from the simulated surface. It gives the mean distance of these vertices from the simulated
surface for plain skinning and for the corrected body, in anny's millimetres. The results go
to the cache folder of the correctives as JSON.
"""

import json
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

from anny.paths import get_anny_cache_path
from anny.poses import library
from anny.poses.authoring import posing as P
from anny.poses.authoring.rig import ANNY_TO_LEGACY, authoring_phenotype

from . import sim, train

THRESHOLD = 0.002  # anny metres


def library_pose(name):
    """a library pose on the authoring body: its rotations, and its root offset scaled with the hip height"""
    rotations, root = library().frames(name)
    M = ANNY_TO_LEGACY
    pose = P.Pose()
    pose.q = np.stack(
        [P.mat2q(M @ P.qmat(q) @ M.T) for q in rotations[0].double().numpy()]
    )
    pose.root = M @ root[0].double().numpy() * P.RIG.scale * P.RIG.hip_height
    return pose


def shapes_on_body():
    """the stored shapes in the legacy frame on the authoring body, with their scale factors"""
    spec = json.loads((train.DATA_DIR / "soft_tissue.json").read_text())
    tensors = load_file(str(train.DATA_DIR / "soft_tissue.safetensors"))
    coarse = P.preview_mesh()["coarse"]
    used = coarse["used"]
    lookup = -np.ones(int(coarse["base_index"].max()) + 1, np.int64)
    lookup[coarse["base_index"][used]] = np.arange(len(used))
    rest_anny = P.RIG.to_anny(coarse["V"][used])
    out = {}
    for s in spec["shapes"]:
        idx = lookup[tensors[s["name"] + ".indices"].numpy()]
        assert (idx >= 0).all()
        D = (
            P.RIG.scale
            * tensors[s["name"] + ".offsets"].double().numpy()
            @ ANNY_TO_LEGACY.T
        )
        support = rest_anny[idx]
        radius = np.sqrt(((support - support.mean(0)) ** 2).sum(1).mean())
        out[s["name"]] = (idx, D, radius / s["reference_radius"])
    return out


def corrected_rest(F, Vs, shapes):
    """the rest surface with the weighted and scaled shapes of both sides"""
    V = Vs.copy()
    for side in ("L", "R"):
        for j, w in train.weights(F, side).items():
            for name, wk in zip(train.shape_names(j), w):
                key = f"{name}.{side}"
                if wk <= 0 or key not in shapes:
                    continue
                idx, D, k = shapes[key]
                V[idx] += wk * k * D
    return V


def evaluate(names):
    t0 = time.time()
    S = sim.Sim(verbose=False)
    coarse = P.preview_mesh()["coarse"]
    used = coarse["used"]
    Vs = coarse["V"][used]
    si, sw = coarse["si"][used], coarse["sw"][used]
    shapes = shapes_on_body()
    rows = []
    for name in names:
        pose = library_pose(name)
        F = P.fk(pose)
        X = S.solve(pose, region="all")[: S.ns]
        plain = P.skin(F, Vs, si, sw)
        corrected = P.skin(F, corrected_rest(F, Vs, shapes), si, sw)
        # distances in anny's units
        d_plain = np.linalg.norm(plain - X, axis=1) / P.RIG.scale
        d_corr = np.linalg.norm(corrected - X, axis=1) / P.RIG.scale
        far = d_plain > THRESHOLD
        rows.append(
            dict(
                pose=name,
                vertices=int(far.sum()),
                plain_mm=float(d_plain[far].mean() * 1000) if far.any() else 0.0,
                corrected_mm=float(d_corr[far].mean() * 1000) if far.any() else 0.0,
                corrected_max_mm=float(d_corr.max() * 1000),
                iterations=S.last["iters"],
                seconds=round(S.last["time"], 1),
            )
        )
        r = rows[-1]
        print(
            f"{name:>12}: {r['vertices']} vertices, plain {r['plain_mm']:.1f} mm, "
            f"corrected {r['corrected_mm']:.1f} mm ({r['seconds']} s)",
            flush=True,
        )
    phenotype = authoring_phenotype()
    report = dict(phenotype=phenotype, rows=rows, seconds=round(time.time() - t0))
    tag = "_".join(f"{k}{v:g}" for k, v in sorted(phenotype.items())) or "default"
    out = get_anny_cache_path() / "correctives" / f"evaluation_{tag}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    evaluate(sys.argv[1:] or ["t_pose", "reach_up", "seated", "crouch"])
