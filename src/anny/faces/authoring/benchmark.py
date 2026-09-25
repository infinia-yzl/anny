# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Benchmarks of anny's faces against measured and photographed faces.

- **B1, 3D coverage**: the share of the ICT-FaceKit identity variation that anny's face shapes
  explain (:mod:`anny.faces.authoring.fit_3d`).
- **B2, measurements**: anny's simulated populations against the calibration targets (ANSUR II,
  3D Facial Norms, CDC), as standardised mean differences and SD ratios.
- **B4, photographs**: the face proportions of near-frontal, neutral FairFace photos against
  anny's populations of the same age group and sex, with three sources of face shapes: none
  (all values 0), uniform over the slider ranges, and the calibrated distribution. anny's
  proportions come from its MediaPipe landmarks projected in a frontal view; a set of renders
  of the viewer page, measured by MediaPipe itself, checks that the two agree.
- **B5, photo fits**: the error of fits of anny to the landmarks of held-out photos, with and
  without face shapes (:mod:`anny.faces.authoring.photo_fit`).

Usage::

    python -m anny.faces.authoring.benchmark [--per-group 400] [--renders 12]

The command writes ``ANNY_CACHE_DIR/faces/benchmark.json`` and ``benchmark.html``.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import time

import numpy as np
import torch

from anny.faces.authoring import photos
from anny.faces.authoring.calibrate import Simulator
from anny.faces.authoring.mediapipe_map import mediapipe_regressor
from anny.faces.authoring.sources import cache_dir
from anny.faces.distribution import FaceShapeDistribution

VARIANTS = ("none", "uniform", "calibrated")


# ------------------------------------------------------------------ anny's populations
def population(sim: Simulator, group: int, sex: str, n: int, variant: str, seed: int):
    """phenotypes (n, P) and face values (n, F) of anny's bodies of a FairFace group"""
    rng = np.random.default_rng(seed)
    lo, hi = photos.AGE_YEARS[group]
    years = rng.uniform(lo, hi, n)
    params = []
    for k, y in enumerate(years):
        params.append(sim.phenotypes(sex, float(y), 1, seed * 1000 + k)[0])
    phen = torch.stack(params)
    F = len(sim.model.face_shape_labels)
    if variant == "none":
        face = torch.zeros((n, F), dtype=torch.float64)
    elif variant == "uniform":
        lo_r = np.array(
            [sim.model.face_shape_ranges[k][0] for k in sim.model.face_shape_labels]
        )
        hi_r = np.array(
            [sim.model.face_shape_ranges[k][1] for k in sim.model.face_shape_labels]
        )
        face = torch.tensor(rng.uniform(lo_r, hi_r, (n, F)))
    else:
        dist = FaceShapeDistribution(sim.model)
        g = torch.Generator().manual_seed(seed)
        face = dist.sample(phen, generator=g)
    return phen, face


def mesh_ratios(sim: Simulator, regressor, phen, face, distance: float = 1.0) -> dict:
    """face proportions from anny's MediaPipe landmarks, projected by a camera in front of the face"""
    out = []
    for start in range(0, len(phen), 64):
        with torch.no_grad():
            o = sim.model(
                phenotype_kwargs=phen[start : start + 64],
                face_shape_kwargs=face[start : start + 64],
            )
            L = regressor({"vertices": o["rest_vertices"]}).numpy()
        centre = L.mean(1, keepdims=True)
        depth = distance + (L[..., 1] - centre[..., 1])  # the face looks toward -Y
        u = (L[..., 0] - centre[..., 0]) / depth
        v = -(L[..., 2] - centre[..., 2]) / depth
        uv = np.stack([u, v], -1)
        P = np.zeros((len(L), 478, 2))
        P[:, :468] = uv[:, :468]
        P[:, photos.MP["iris_r"]] = uv[:, 468]
        P[:, photos.MP["iris_l"]] = uv[:, 469]
        out.append(P)
    P = np.concatenate(out)
    return photos.ratios(P)


def _stack(r: dict) -> np.ndarray:
    return np.stack([r[k] for k in r], 1)


# ------------------------------------------------------------------ B2
def measurement_check(sim: Simulator, samples: int = 300) -> list[dict]:
    """anny's simulated measurements at each calibration anchor against its targets"""
    with open(cache_dir() / "calibration_report.json") as f:
        report = json.load(f)
    dist = FaceShapeDistribution(sim.model)
    rows = []
    for a in report["anchors"]:
        phen = sim.phenotypes(a["sex"], a["years"], samples, 7)
        g = torch.Generator().manual_seed(11)
        face = dist.sample(phen, generator=g)
        base = sim(phen, torch.zeros_like(face), a["names"])
        cal = sim(phen, face, a["names"])
        for k, name in enumerate(a["names"]):
            m, s = a["target_mean"][k], a["target_sd"][k]
            rows.append(
                dict(
                    sex=a["sex"],
                    years=a["years"],
                    measurement=name,
                    target_mean=m,
                    target_sd=s,
                    smd_default=float((base[:, k].mean() - m) / s),
                    smd_calibrated=float((cal[:, k].mean() - m) / s),
                    sd_ratio_default=float(base[:, k].std() / s),
                    sd_ratio_calibrated=float(cal[:, k].std() / s),
                )
            )
    return rows


# ------------------------------------------------------------------ B4
def photo_comparison(sim, per_group: int, renders: int, seed: int = 0) -> dict:
    regressor, _ = mediapipe_regressor(sim.model, list(range(468)) + [468, 473])
    regressor = regressor.to(dtype=torch.float64)
    real = {}
    for split in ("train", "validation"):
        det = photos.fairface_detections(split)
        ok = photos.keep(det)
        real[split] = (det, ok)
    det, ok = real["train"]
    R_real = photos.ratios(det["landmarks"])
    names = list(R_real)
    X_real = _stack(R_real)
    groups = []
    render_check = []
    for g in range(len(photos.AGE_GROUPS)):
        for si, sex in enumerate(photos.GENDERS):
            sel = ok & (det["age"] == g) & (det["gender"] == si)
            if sel.sum() < 20:
                continue
            real_x = X_real[sel]
            entry = dict(group=photos.AGE_GROUPS[g], sex=sex, n_photos=int(sel.sum()))
            for variant in VARIANTS:
                phen, face = population(
                    sim, g, sex, per_group, variant, seed + 17 * g + si
                )
                sim_x = _stack(mesh_ratios(sim, regressor, phen, face))
                entry[variant] = photos.compare(real_x, sim_x, names)
                if variant == "calibrated" and renders > 0:
                    render_check.append(
                        (g, sex, phen[:renders], face[:renders], sim_x[:renders])
                    )
            groups.append(entry)
            print(
                f"B4 {entry['group']:>5s} {sex:6s}: C2ST "
                + ", ".join(f"{v} {entry[v]['c2st_auc']:.2f}" for v in VARIANTS)
            )
    out = dict(ratios=names, groups=groups)
    if render_check:
        out["render_check"] = render_bias(sim, render_check, names)
    return out


def render_bias(sim, items, names) -> dict:
    """MediaPipe on renders of the viewer against the projected landmarks, for the same faces"""
    looks, mesh = [], []
    for g, sex, phen, face, sim_x in items:
        for p, f, x in zip(phen, face, sim_x):
            ph = {
                k: float(p[sim.labels.index(k)])
                for k in ("gender", "age", "muscle", "weight", "height", "proportions")
            }
            fv = {
                k: float(v)
                for k, v in zip(sim.model.face_shape_labels, f)
                if float(v) != 0
            }
            looks.append(
                dict(format="anny-viewer/look@2", name="bench", phenotype=ph, face=fv)
            )
            mesh.append(x)
    shots = cache_dir() / "benchmark_renders"
    shots.mkdir(exist_ok=True)
    t0 = time.time()
    images = photos.render_anny(looks, out_dir=shots)
    print(f"renders: {len(images)} in {time.time() - t0:.0f} s")
    det = photos.detect_all(images)
    ok = photos.keep(det, max_angle=15)
    rendered = _stack(photos.ratios(det["landmarks"]))
    mesh = np.array(mesh)
    diff = rendered[ok] - mesh[ok]
    sd = mesh.std(0) + 1e-12
    return dict(
        renders=len(images),
        detected=int(det["found"].sum()),
        kept=int(ok.sum()),
        bias_in_sd=dict(zip(names, np.round(diff.mean(0) / sd, 3).tolist())),
        correlation=dict(
            zip(
                names,
                [
                    round(
                        float(np.corrcoef(rendered[ok][:, k], mesh[ok][:, k])[0, 1]), 3
                    )
                    for k in range(len(names))
                ],
            )
        ),
    )


# ------------------------------------------------------------------ B5
def photo_fits(limit: int | None = None) -> dict:
    from anny.faces.authoring.photo_fit import fit_photos

    det = photos.fairface_detections("validation")
    ok = np.nonzero(photos.keep(det))[0]
    if limit:
        ok = ok[:limit]
    names = list(det["score_names"])
    out = dict(n=int(len(ok)))
    for with_face in (False, True):
        f = fit_photos(
            det["landmarks"][ok],
            det["scores"][ok],
            names,
            det["age"][ok],
            det["gender"][ok],
            with_face,
        )
        key = "with_face_shapes" if with_face else "without_face_shapes"
        out[key] = dict(
            median_nme=float(np.median(f["nme"])),
            by_age=[
                float(np.median(f["nme"][det["age"][ok] == g]))
                if (det["age"][ok] == g).any()
                else None
                for g in range(len(photos.AGE_GROUPS))
            ],
            by_race=[
                float(np.median(f["nme"][det["race"][ok] == r]))
                if (det["race"][ok] == r).any()
                else None
                for r in range(len(photos.RACES))
            ],
        )
        print(f"B5 {key}: median NME {out[key]['median_nme']:.4f}")
    return out


# ------------------------------------------------------------------ report
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def html_report(result: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = [
        "<!doctype html><meta charset='utf-8'><title>Anny face benchmark</title>",
        "<style>body{font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}"
        "table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}"
        "th{background:#f3f3f3}td:first-child,th:first-child{text-align:left}</style>",
        "<h1>Anny face benchmark</h1>",
    ]
    b1 = result.get("B1")
    if b1:
        parts.append("<h2>B1. 3D coverage (ICT-FaceKit identities)</h2><table>")
        for k, v in b1.items():
            parts.append(f"<tr><td>{html.escape(k)}</td><td>{v:.4g}</td></tr>")
        parts.append("</table>")
    b2 = result.get("B2")
    if b2:
        parts.append("<h2>B2. Measurements against the calibration targets</h2>")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key, title in (
            (axes[0], "smd", "standardised mean difference"),
            (axes[1], "sd_ratio", "SD ratio"),
        ):
            ax.hist(
                [r[key + "_default"] for r in b2],
                bins=40,
                alpha=0.6,
                label="face values 0",
            )
            ax.hist(
                [r[key + "_calibrated"] for r in b2],
                bins=40,
                alpha=0.6,
                label="calibrated",
            )
            ax.set_title(title)
            ax.legend()
        parts.append(f"<img src='{_png(fig)}' width='100%'>")
        plt.close(fig)
        adult = [r for r in b2 if r["years"] == 28.0]
        parts.append(
            "<h3>Adults (28 years)</h3><table><tr><th>sex</th><th>measurement</th><th>target</th>"
            "<th>SMD, values 0</th><th>SMD, calibrated</th><th>SD ratio, calibrated</th></tr>"
        )
        for r in adult:
            parts.append(
                f"<tr><td>{r['sex']}</td><td>{r['measurement']}</td>"
                f"<td>{r['target_mean']:.1f} ± {r['target_sd']:.1f}</td>"
                f"<td>{r['smd_default']:+.2f}</td><td>{r['smd_calibrated']:+.2f}</td><td>{r['sd_ratio_calibrated']:.2f}</td></tr>"
            )
        parts.append("</table>")
    b4 = result.get("B4")
    if b4:
        parts.append("<h2>B4. Face proportions against FairFace photographs</h2>")
        parts.append(
            "<table><tr><th>age group</th><th>sex</th><th>photos</th>"
            + "".join(
                f"<th>C2ST AUC, {v}</th><th>median |SMD|, {v}</th>" for v in VARIANTS
            )
            + "</tr>"
        )
        for g in b4["groups"]:
            parts.append(
                f"<tr><td>{g['group']}</td><td>{g['sex']}</td><td>{g['n_photos']}</td>"
                + "".join(
                    f"<td>{g[v]['c2st_auc']:.2f}</td><td>{g[v]['median_abs_smd']:.2f}</td>"
                    for v in VARIANTS
                )
                + "</tr>"
            )
        parts.append("</table>")
        rc = b4.get("render_check")
        if rc:
            parts.append(
                f"<h3>Render check</h3><p>{rc['kept']} of {rc['renders']} renders kept. Mean difference of "
                "MediaPipe on renders minus the projected landmarks, in SDs of anny's population, and their "
                "correlation:</p><table><tr><th>ratio</th><th>bias (SD)</th><th>correlation</th></tr>"
            )
            for k in rc["bias_in_sd"]:
                parts.append(
                    f"<tr><td>{k}</td><td>{rc['bias_in_sd'][k]:+.2f}</td><td>{rc['correlation'][k]:.2f}</td></tr>"
                )
            parts.append("</table>")
    b5 = result.get("B5")
    if b5:
        parts.append(
            f"<h2>B5. Fits to {b5['n']} held-out photos</h2><table><tr><th></th><th>median NME</th>"
            + "".join(f"<th>{a}</th>" for a in photos.AGE_GROUPS)
            + "</tr>"
        )
        for key in ("without_face_shapes", "with_face_shapes"):
            r = b5[key]
            parts.append(
                f"<tr><td>{key.replace('_', ' ')}</td><td>{r['median_nme']:.4f}</td>"
                + "".join(
                    f"<td>{v:.4f}</td>" if v is not None else "<td></td>"
                    for v in r["by_age"]
                )
                + "</tr>"
            )
        parts.append("</table>")
    return "\n".join(parts)


def run(per_group: int = 400, renders: int = 12, fits: int | None = None):
    sim = Simulator()
    result = {}
    report = cache_dir() / "ict_fit_report.json"
    if report.exists():
        result["B1"] = json.load(open(report))
    result["B2"] = measurement_check(sim)
    print("B2: done")
    result["B4"] = photo_comparison(sim, per_group, renders)
    result["B5"] = photo_fits(fits)
    with open(cache_dir() / "benchmark.json", "w") as f:
        json.dump(result, f, indent=1)
    with open(cache_dir() / "benchmark.html", "w") as f:
        f.write(html_report(result))
    print(f"benchmark -> {cache_dir() / 'benchmark.html'}")
    return result


def main():
    parser = argparse.ArgumentParser(description="benchmark anny's faces")
    parser.add_argument("--per-group", type=int, default=400)
    parser.add_argument("--renders", type=int, default=12)
    parser.add_argument("--fits", type=int, default=None)
    args = parser.parse_args()
    run(args.per_group, args.renders, args.fits)


if __name__ == "__main__":
    main()
