# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Face proportions of photographs and of anny's renders, measured the same way.

MediaPipe's face landmarker (Apache 2.0) finds 478 landmarks, 52 expression scores and the pose
of the head in each image. The benchmark keeps near-frontal faces (yaw and pitch within 10
degrees) with a neutral expression, and turns their landmarks into scale-free ratios of 2D
distances (:data:`RATIOS`). The same detector measures FairFace photographs (CC BY 4.0) and
renders of anny from the viewer page, so the bias of the detector cancels.

- :func:`fairface_detections` measures the FairFace validation photos (cached);
- :func:`render_anny` renders portraits of anny with the viewer page (Playwright);
- :func:`compare` compares two sets of ratios (standardised mean difference, SD ratio, sliced
  Wasserstein distance, classifier two-sample test).
"""

from __future__ import annotations

import io
import json
import pathlib

import numpy as np

from anny.faces.authoring.sources import cache_dir, fetch, fetch_fairface

AGE_GROUPS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
AGE_YEARS = [(0.5, 2.9), (3, 9.9), (10, 19.9), (20, 29.9), (30, 39.9), (40, 49.9),
             (50, 59.9), (60, 69.9), (70, 85)]  # fmt: skip
GENDERS = ["male", "female"]
RACES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic",
         "Southeast Asian"]  # fmt: skip

# MediaPipe face mesh indices (subject's right has negative x in the canonical model)
MP = dict(
    ex_r=33, en_r=133, en_l=362, ex_l=263,
    lid_up_r=159, lid_lo_r=145, lid_up_l=386, lid_lo_l=374,
    iris_r=468, iris_l=473,
    n=168, prn=4, sn=2, ls=0, sto_up=13, sto_lo=14, li=17, gn=152, top=10,
    al_r=102, al_l=331, ch_r=61, ch_l=291,
    face_r=234, face_l=454, jaw_r=172, jaw_l=397,
)  # fmt: skip


def _d(P, a, b):
    return np.linalg.norm(P[..., MP[a], :2] - P[..., MP[b], :2], axis=-1)


def _mid(P, a, b):
    return 0.5 * (P[..., MP[a], :2] + P[..., MP[b], :2])


def ratios(P: np.ndarray) -> dict[str, np.ndarray]:
    """scale-free proportions of frontal faces from landmarks (n, 478, >=2) in pixels"""
    outer = _d(P, "ex_r", "ex_l")
    inner = _d(P, "en_r", "en_l")
    eye_w = 0.5 * (_d(P, "ex_r", "en_r") + _d(P, "ex_l", "en_l"))
    eye_h = 0.5 * (_d(P, "lid_up_r", "lid_lo_r") + _d(P, "lid_up_l", "lid_lo_l"))
    face_w = _d(P, "face_r", "face_l")
    sto = _mid(P, "sto_up", "sto_lo")
    n2 = P[..., MP["n"], :2]
    sn2 = P[..., MP["sn"], :2]
    gn2 = P[..., MP["gn"], :2]
    face_h = np.linalg.norm(n2 - gn2, axis=-1)
    lower = np.linalg.norm(sn2 - gn2, axis=-1)
    return {
        "intercanthal/outercanthal": inner / outer,
        "eye width/outercanthal": eye_w / outer,
        "eye height/eye width": eye_h / eye_w,
        "nose width/intercanthal": _d(P, "al_r", "al_l") / inner,
        "mouth width/face width": _d(P, "ch_r", "ch_l") / face_w,
        "face width/face height": face_w / face_h,
        "jaw width/face width": _d(P, "jaw_r", "jaw_l") / face_w,
        "interpupillary/face width": _d(P, "iris_r", "iris_l") / face_w,
        "nose length/face height": np.linalg.norm(n2 - sn2, axis=-1) / face_h,
        "lower face/face height": lower / face_h,
        "upper lip/lower face": np.linalg.norm(sn2 - sto, axis=-1) / lower,
        "chin/lower face": np.linalg.norm(P[..., MP["li"], :2] - gn2, axis=-1) / lower,
        "red lips/mouth width": (_d(P, "ls", "sto_up") + _d(P, "sto_lo", "li"))
        / _d(P, "ch_r", "ch_l"),
    }


# ------------------------------------------------------------------ detection
class Detector:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(fetch("mediapipe_face_landmarker"))
            ),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.blend_names = None

    def __call__(self, image: np.ndarray):
        """landmarks (478, 3) in pixels, expression scores (52,) and pose (4, 4), or None"""
        h, w = image.shape[:2]
        result = self.landmarker.detect(
            self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)
            )
        )
        if not result.face_landmarks:
            return None
        lm = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
        cats = result.face_blendshapes[0]
        if self.blend_names is None:
            self.blend_names = [c.category_name for c in cats]
        scores = np.array([c.score for c in cats])
        pose = np.array(result.facial_transformation_matrixes[0])
        return lm, scores, pose


def head_angles(pose: np.ndarray) -> np.ndarray:
    """yaw, pitch and roll (degrees) of MediaPipe's facial transformation matrices (n, 4, 4)"""
    R = pose[..., :3, :3]
    yaw = np.degrees(np.arctan2(R[..., 0, 2], R[..., 2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-R[..., 1, 2], -1, 1)))
    roll = np.degrees(np.arctan2(R[..., 1, 0], R[..., 1, 1]))
    return np.stack([yaw, pitch, roll], -1)


def detect_all(images, detector=None) -> dict:
    detector = detector or Detector()
    lms, scores, poses, found = [], [], [], []
    for image in images:
        r = detector(image)
        found.append(r is not None)
        if r is None:
            lms.append(np.full((478, 3), np.nan))
            scores.append(np.full(52, np.nan))
            poses.append(np.full((4, 4), np.nan))
        else:
            lms.append(r[0])
            scores.append(r[1])
            poses.append(r[2])
    return dict(
        landmarks=np.array(lms),
        scores=np.array(scores),
        pose=np.array(poses),
        found=np.array(found),
        score_names=np.array(detector.blend_names or []),
    )


def keep(
    det: dict, max_angle: float = 10.0, jaw: float = 0.1, smile: float = 0.25
) -> np.ndarray:
    """near-frontal faces with a neutral expression"""
    ok = det["found"].copy()
    angles = head_angles(np.nan_to_num(det["pose"]))
    ok &= (np.abs(angles[:, 0]) < max_angle) & (np.abs(angles[:, 1]) < max_angle)
    names = list(det["score_names"])
    if names:
        s = np.nan_to_num(det["scores"])
        ok &= s[:, names.index("jawOpen")] < jaw
        ok &= s[:, names.index("mouthSmileLeft")] < smile
        ok &= s[:, names.index("mouthSmileRight")] < smile
    return ok


def fairface_detections(
    split: str = "validation", ages: list[int] | None = None
) -> dict:
    """
    MediaPipe on the FairFace photos of a split, with their labels (cached). ``ages`` limits
    the detection to some age groups (indices of ``AGE_YEARS``), for the groups that the
    validation split leaves short.
    """
    tag = "" if ages is None else "_ages" + "".join(str(a) for a in sorted(ages))
    path = cache_dir() / f"fairface_{split}{tag}_mediapipe.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    import pyarrow.parquet as pq
    from PIL import Image

    detector, parts = Detector(), []
    for file in fetch_fairface(split):
        table = pq.read_table(file)
        if ages is not None:
            rows = np.flatnonzero(np.isin(table.column("age").to_numpy(), ages))
            table = table.take(rows)
        images = (
            np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB"))
            for b in table.column("image").to_pylist()
        )
        det = detect_all(images, detector)
        det.update(
            age=np.array(table.column("age").to_pylist()),
            gender=np.array(table.column("gender").to_pylist()),
            race=np.array(table.column("race").to_pylist()),
        )
        parts.append(det)
    det = {
        k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "score_names"
    }
    det["score_names"] = parts[0]["score_names"]
    np.savez(path, **det)
    return det


# the age groups (0-2, 50-59, 60-69 and 70+) with fewer than about 100 kept photos per sex in
# the validation split; the benchmark adds their photos from the train split
SHORT_AGE_GROUPS = [0, 6, 7, 8]


def real_photos() -> dict:
    """detections of the validation split and of the short age groups of the train split"""
    parts = [
        fairface_detections("validation"),
        fairface_detections("train", SHORT_AGE_GROUPS),
    ]
    det = {
        k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "score_names"
    }
    det["score_names"] = parts[0]["score_names"]
    return det


# ------------------------------------------------------------------ anny's portraits
def render_anny(
    looks: list[dict],
    size: int = 448,
    page: pathlib.Path | None = None,
    accumulation: int = 4,
    margin: float = 1.25,
    out_dir: pathlib.Path | None = None,
):
    """
    Portraits (n, size, size, 3) of anny from the viewer page, one per look (the page's preset
    format: phenotype, face, skin, hair and eyes). The page renders with its own shading and
    hair; ``margin`` widens the face framing like FairFace's crops.
    """
    from PIL import Image
    from playwright.sync_api import sync_playwright

    from anny.viewer.build import REPO

    page_path = page or REPO / "viewer" / "dist" / "anny_viewer.html"
    url = page_path.resolve().as_uri() + f"?shot=1&acc={accumulation}"
    images = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/opt/pw-browsers/chromium",
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        tab = browser.new_page(viewport=dict(width=size, height=size))
        tab.goto(url)
        tab.wait_for_function(
            "window.__BODY && window.__BODY.ready && window.__RIG && window.__RIG.ready",
            timeout=600000,
        )
        # the canvas alone, and the head without its hair (the brows and the lashes stay)
        tab.add_style_tag(
            content="body * { visibility: hidden !important; } "
            "canvas { visibility: visible !important; }"
        )
        tab.evaluate("() => window.setHair(false)")
        for i, look in enumerate(looks):
            tab.evaluate("(l) => window.setLook(l)", look)
            tab.evaluate(f"() => window.setPortrait(24, {margin})")
            png = tab.screenshot()
            image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            if out_dir is not None and i < 64:
                Image.fromarray(image).save(out_dir / f"anny_{i:03d}.png")
            images.append(image)
        browser.close()
    return images


# ------------------------------------------------------------------ comparisons
def _standardise(real, sim):
    mu, sd = real.mean(0), real.std(0) + 1e-12
    return (real - mu) / sd, (sim - mu) / sd


def sliced_wasserstein(
    a: np.ndarray, b: np.ndarray, projections: int = 256, seed: int = 0
):
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((projections, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    q = np.linspace(0.01, 0.99, 99)
    pa = np.quantile(a @ dirs.T, q, axis=0)
    pb = np.quantile(b @ dirs.T, q, axis=0)
    return float(np.sqrt(((pa - pb) ** 2).mean()))


def c2st_auc(a: np.ndarray, b: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """cross-validated AUC of a logistic regression that tells a from b (0.5: indistinguishable)"""
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    X = np.concatenate([a[rng.permutation(len(a))[:n]], b[rng.permutation(len(b))[:n]]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    scores = np.zeros(len(X))
    for k in range(folds):
        test = np.arange(len(X)) % folds == k
        w = np.zeros(X.shape[1])
        for _ in range(200):  # Newton steps with an L2 penalty
            p = 1 / (1 + np.exp(-X[~test] @ w))
            g = X[~test].T @ (p - y[~test]) + 1e-2 * w
            H = (X[~test] * (p * (1 - p))[:, None]).T @ X[~test] + 1e-2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        scores[test] = X[test] @ w
    pos, neg = scores[y == 1], scores[y == 0]
    return float(
        (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
    )


def compare(real: np.ndarray, sim: np.ndarray, names: list[str]) -> dict:
    """compare ratio vectors (n, R) of photographs and of simulated faces"""
    a, b = _standardise(real, sim)
    smd = b.mean(0) - a.mean(0)
    sd_ratio = sim.std(0) / (real.std(0) + 1e-12)
    return dict(
        n_real=int(len(real)),
        n_sim=int(len(sim)),
        smd=dict(zip(names, np.round(smd, 3).tolist())),
        sd_ratio=dict(zip(names, np.round(sd_ratio, 3).tolist())),
        max_abs_smd=float(np.abs(smd).max()),
        median_abs_smd=float(np.median(np.abs(smd))),
        sliced_wasserstein=sliced_wasserstein(a, b),
        c2st_auc=c2st_auc(a, b),
    )


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
