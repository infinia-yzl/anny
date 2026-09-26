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
  renders of anny's populations of the same age group and sex, both measured by MediaPipe, with
  three sources of face shapes: none (all values 0), uniform over the slider ranges, and the
  calibrated distribution. The renders come from the viewer page; the MediaPipe landmarks that
  ``data/keypoints/mediapipe.json`` places on anny's mesh differ from those that MediaPipe
  detects on renders by up to 3 SD of some ratios, so the projected landmarks do not serve here.
- **B5, photo fits**: the error of fits of anny to the landmarks of held-out photos, with and
  without face shapes (:mod:`anny.faces.authoring.photo_fit`).

Usage::

    python -m anny.faces.authoring.benchmark [--per-group 60] [--uniform 30]

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
def look_of(sim: Simulator, phenotype, face) -> dict:
    """the viewer look (anny-viewer/look@2) of one body and face"""
    ph = {
        k: float(phenotype[sim.labels.index(k)])
        for k in ("gender", "age", "muscle", "weight", "height", "proportions")
    }
    fv = {
        k: float(v) for k, v in zip(sim.model.face_shape_labels, face) if float(v) != 0
    }
    return dict(format="anny-viewer/look@2", name="bench", phenotype=ph, face=fv)


def photo_comparison(
    sim, per_group: int = 60, uniform_per_group: int = 30, seed: int = 0
) -> dict:
    """face proportions of the photos against those of renders of anny's populations"""
    det = photos.real_photos()
    ok = photos.keep(det)
    R_real = photos.ratios(det["landmarks"])
    names = list(R_real)
    X_real = _stack(R_real)
    items, looks = [], []
    for g in range(len(photos.AGE_GROUPS)):
        for si, sex in enumerate(photos.GENDERS):
            for variant in VARIANTS:
                n = uniform_per_group if variant == "uniform" else per_group
                phen, face = population(sim, g, sex, n, variant, seed + 17 * g + si)
                for p, f in zip(phen, face):
                    items.append((g, sex, variant))
                    looks.append(look_of(sim, p, f))
    detector = photos.Detector()
    found = [None] * len(looks)

    def on_image(i, image):
        found[i] = detector(image)

    shots = cache_dir() / "benchmark_renders"
    shots.mkdir(exist_ok=True)
    t0 = time.time()
    photos.render_anny(
        looks, size=320, accumulation=1, out_dir=shots, on_image=on_image
    )
    print(f"B4 renders: {len(looks)} in {time.time() - t0:.0f} s")
    nan = (np.full((478, 3), np.nan), np.full(52, np.nan), np.full((4, 4), np.nan))
    rdet = dict(
        landmarks=np.array([(r or nan)[0] for r in found]),
        scores=np.array([(r or nan)[1] for r in found]),
        pose=np.array([(r or nan)[2] for r in found]),
        found=np.array([r is not None for r in found]),
        score_names=np.array(detector.blend_names or []),
    )
    rok = photos.keep(rdet, max_angle=15)
    X_sim = _stack(photos.ratios(rdet["landmarks"]))
    groups = []
    for g in range(len(photos.AGE_GROUPS)):
        for si, sex in enumerate(photos.GENDERS):
            sel = ok & (det["age"] == g) & (det["gender"] == si)
            entry = dict(group=photos.AGE_GROUPS[g], sex=sex, n_photos=int(sel.sum()))
            for variant in VARIANTS:
                idx = [
                    i
                    for i, it in enumerate(items)
                    if it == (g, sex, variant) and rok[i]
                ]
                entry[variant] = photos.compare(X_real[sel], X_sim[idx], names)
                entry[variant]["n_renders"] = len(idx)
            groups.append(entry)
            print(
                f"B4 {entry['group']:>5s} {sex:6s}: C2ST "
                + ", ".join(f"{v} {entry[v]['c2st_auc']:.2f}" for v in VARIANTS)
                + "; median |SMD| "
                + ", ".join(f"{v} {entry[v]['median_abs_smd']:.2f}" for v in VARIANTS)
            )
    return dict(ratios=names, groups=groups, renders=len(looks), kept=int(rok.sum()))


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
        parts.append(
            f"<p>{b4['kept']} of {b4['renders']} renders kept as near-frontal and neutral.</p>"
        )
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


def run(per_group: int = 60, uniform: int = 30, fits: int | None = None):
    sim = Simulator()
    result = {}
    report = cache_dir() / "ict_fit_report.json"
    if report.exists():
        result["B1"] = json.load(open(report))
    result["B2"] = measurement_check(sim)
    print("B2: done")
    result["B4"] = photo_comparison(sim, per_group, uniform)
    result["B5"] = photo_fits(fits)
    with open(cache_dir() / "benchmark.json", "w") as f:
        json.dump(result, f, indent=1)
    with open(cache_dir() / "benchmark.html", "w") as f:
        f.write(html_report(result))
    print(f"benchmark -> {cache_dir() / 'benchmark.html'}")
    return result


def main():
    parser = argparse.ArgumentParser(description="benchmark anny's faces")
    parser.add_argument("--per-group", type=int, default=60)
    parser.add_argument("--uniform", type=int, default=30)
    parser.add_argument("--fits", type=int, default=None)
    args = parser.parse_args()
    run(args.per_group, args.uniform, args.fits)


if __name__ == "__main__":
    main()
