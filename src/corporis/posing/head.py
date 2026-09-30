# Corporis
# Apache License, Version 2.0
"""
The head's rotation from points that may be wrong.

Pictures made by AI models often draw anatomy that no body has, and any picture can be noisy,
so MediaPipe's head points can land in the wrong place: on a stylised face, the hidden ear can
land on the cheek. The head therefore comes from a robust fit of the same points on Anny's rest
head: every triple of points proposes a rotation and a scale, the proposal that most points
agree with wins, and the fit is then repeated on the points that agree (the inliers). A free
scale lets a drawn head be larger than Anny's.

The fit uses MediaPipe's face mesh when the face landmarker finds a face (468 points, placed on
Anny by ``data/keypoints/mediapipe.json``), and the pose landmarker's 11 head points otherwise:
in 3D (:func:`robust_fit`), or by their positions in the picture alone
(:func:`robust_fit_image`), since MediaPipe only guesses their depth.
:meth:`corporis.posing.retarget.Retargeter.head_fit` picks the source.
"""

from __future__ import annotations

import dataclasses
import itertools
import json

import numpy as np

# the pose landmarker's head points: the nose, the eyes, the ears and the mouth (BODY[:11])
HEAD_POINTS = 11
# a point agrees with a proposal when it lies within this share of the head's size
INLIER = 0.12
# the head may be this much larger or smaller than the body's scale suggests
SCALE_RANGE = (0.5, 2.2)
# the fit needs this many agreeing points, and points that do not lie along one line
MIN_INLIERS = 4
MIN_SPREAD = 0.2


@dataclasses.dataclass
class HeadFit:
    """A fitted head: its world rotation from the rest pose, its scale against Anny's head,
    the weight of each point (1 for the points that agree, 0 for the others) and the RMS
    error of the agreeing points relative to the head's size."""

    rotation: np.ndarray
    scale: float
    inliers: np.ndarray
    error: float


def similarity(rest: np.ndarray, points: np.ndarray, weights: np.ndarray):
    """the rotation R and scale s (and offset t) with s R rest + t closest to ``points`` in
    weighted least squares (Umeyama)"""
    w = weights / weights.sum()
    ca, cb = w @ rest, w @ points
    A, B = rest - ca, points - cb
    H = (A * w[:, None]).T @ B
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ D @ U.T
    var = (w * (A**2).sum(1)).sum()
    s = float((S * np.diag(D)).sum() / max(var, 1e-12))
    return R, s, cb - s * R @ ca


def _spread(points: np.ndarray) -> float:
    """how far points spread off one line: the second singular value over the first"""
    sv = np.linalg.svd(points - points.mean(0), compute_uv=False)
    return float(sv[1] / max(sv[0], 1e-12))


def robust_fit(
    rest: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray | None = None,
    samples: int = 200,
) -> HeadFit | None:
    """the rotation and scale that the most points agree with, refitted on those points;
    None when too few points agree or the agreeing points lie along one line"""
    rest = np.asarray(rest, np.float64)
    points = np.asarray(points, np.float64)
    n = len(rest)
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    usable = np.flatnonzero(w > 0)
    if len(usable) < MIN_INLIERS:
        return None
    size = np.sqrt(((rest[usable] - rest[usable].mean(0)) ** 2).sum(1).mean())
    triples = list(itertools.combinations(usable.tolist(), 3))
    if len(triples) > samples:
        # a large set (the face mesh): an even spread of triples, the same on every run
        pick = np.linspace(0, len(triples) - 1, samples).round().astype(int)
        triples = [triples[i] for i in pick]
    best, best_score = None, -1.0
    for t in triples:
        t = list(t)
        if _spread(rest[t]) < MIN_SPREAD:
            continue
        R, s, off = similarity(rest[t], points[t], np.ones(3))
        if s <= 0:
            continue
        r = np.linalg.norm(rest @ R.T * s + off - points, axis=1) / (s * size)
        inl = (r < INLIER) & (w > 0)
        score = float(w[inl].sum())
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < MIN_INLIERS:
        return None
    inliers = best
    for _ in range(3):
        R, s, off = similarity(rest[inliers], points[inliers], w[inliers])
        r = np.linalg.norm(rest @ R.T * s + off - points, axis=1) / (s * size)
        new = (r < INLIER) & (w > 0)
        if new.sum() < MIN_INLIERS or (new == inliers).all():
            break
        inliers = new
    if _spread(rest[inliers]) < MIN_SPREAD:
        return None
    error = float(np.sqrt((r[inliers] ** 2).mean()))
    return HeadFit(R, s, inliers.astype(np.float64), error)


def plausible(fit: HeadFit, body_scale: float, chest: np.ndarray, limit: float = 110.0):
    """whether a fitted head fits the body: its size against the body's, and its turn from
    the chest within ``limit`` degrees"""
    ratio = fit.scale / max(body_scale, 1e-12)
    if not SCALE_RANGE[0] <= ratio <= SCALE_RANGE[1]:
        return False
    rel = chest.T @ fit.rotation
    angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0)))
    return angle <= limit


def face_mesh_rest(model, vertices) -> np.ndarray | None:
    """MediaPipe's 468 face mesh points on Anny's ``vertices`` (the topology of ``model``), or
    None when the topology lacks the mesh's vertices"""
    from anny.paths import get_anny_root_dir

    with open(get_anny_root_dir() / "data" / "keypoints" / "mediapipe.json") as f:
        table = json.load(f)
    base = model.base_mesh_vertex_indices.detach().cpu().numpy()
    lookup = -np.ones(int(base.max()) + 1, np.int64)
    lookup[base] = np.arange(len(base))
    v = np.asarray(vertices, np.float64)
    out = np.zeros((468, 3))
    for i in range(468):
        ids, bary = table[str(i)]
        ids = np.asarray(ids)
        if ids.max() >= len(lookup) or (lookup[ids] < 0).any():
            return None
        out[i] = np.asarray(bary) @ v[lookup[ids]]
    return out


def _rotvec(v: np.ndarray) -> np.ndarray:
    t = float(np.linalg.norm(v))
    if t < 1e-12:
        return np.eye(3)
    k = v / t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _log(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    t = np.arccos(c)
    if t < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - t < 1e-6:
        i = int(np.argmax(np.diag(R)))
        col = R[:, i] + np.eye(3)[i]
        return col / np.linalg.norm(col) * t
    return w * t / (2 * np.sin(t))


# the picture fit: a point agrees when it lies within this share of the head's size, the
# robust loss levels off at ROBUST_IMAGE of it, and PRIOR pulls the head toward its start,
# which settles the depth that one picture leaves open
INLIER_IMAGE = 0.25
ROBUST_IMAGE = 0.15
PRIOR = 0.02
# the starts: the chest's rotation, tipped back and forward and turned both ways (degrees)
STARTS = [(0, 0), (40, 0), (-30, 0), (0, 40), (0, -40), (40, 40), (40, -40)]


def robust_fit_image(
    rest: np.ndarray,
    image: np.ndarray,
    weights: np.ndarray,
    chest: np.ndarray,
    pixel_scale: float,
    steps: int = 40,
) -> HeadFit | None:
    """the head's rotation from the picture positions of its points alone (``image``: pixels,
    x right, y up) seen by a camera along +y, with a scale and an offset in the picture.

    Levenberg-Marquardt with reweighting for a robust loss runs from each start in
    :data:`STARTS`; the lowest loss wins. The rotation's depth is weakly held near the start.
    """
    rest = np.asarray(rest, np.float64)
    image = np.asarray(image, np.float64)
    w = np.asarray(weights, np.float64)
    size = np.sqrt(((rest - rest.mean(0)) ** 2).sum(1).mean())
    best = None
    for tip, turn in STARTS:
        R0 = (
            chest
            @ _rotvec(np.radians([-tip, 0, 0]))
            @ _rotvec(np.radians([0, 0, turn]))
        )
        start = R0.copy()

        def residuals(p, R0=R0, start=start):
            R = _rotvec(p[:3]) @ R0
            s = np.exp(p[3])
            q = rest @ R.T
            e = s * q[:, [0, 2]] + p[4:6] - image
            prior = np.sqrt(PRIOR) * _log(R @ start.T)
            return e, s, prior

        q0 = rest @ R0.T
        s0 = pixel_scale
        t0 = (w @ image) / w.sum() - s0 * (w @ q0[:, [0, 2]]) / w.sum()
        p = np.concatenate([np.zeros(3), [np.log(s0)], t0])
        mu = 1e-2
        for _ in range(steps):
            e, s, prior = residuals(p)
            c2 = (ROBUST_IMAGE * s * size) ** 2
            e2 = (e**2).sum(1)
            irls = w * c2**2 / (e2 + c2) ** 2  # Geman-McClure weights
            scale_r = np.sqrt(irls)[:, None] / (s * size)
            r = np.concatenate([(e * scale_r).ravel(), prior])
            J = np.zeros((len(r), 6))
            for k in range(6):
                d = np.zeros(6)
                d[k] = 1e-6
                e_k, _, prior_k = residuals(p + d)
                J[:, k] = (
                    np.concatenate([(e_k * scale_r).ravel(), prior_k]) - r
                ) / 1e-6
            A = J.T @ J
            step = np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-9), -J.T @ r)
            e_new, s_new, prior_new = residuals(p + step)
            r_new = np.concatenate([(e_new * scale_r).ravel(), prior_new])
            if (r_new**2).sum() < (r**2).sum():
                p, mu = p + step, mu * 0.3
            else:
                mu *= 10.0
        e, s, prior = residuals(p)
        c2 = (ROBUST_IMAGE * s * size) ** 2
        e2 = (e**2).sum(1)
        loss = float(
            (w * c2 * e2 / (e2 + c2)).sum() / (s * size) ** 2 + (prior**2).sum()
        )
        if best is None or loss < best[0]:
            dist = np.sqrt(e2) / (s * size)
            best = (loss, _rotvec(p[:3]) @ R0, s, dist)
    _, R, s, dist = best
    inliers = (dist < INLIER_IMAGE) & (w > 0)
    if inliers.sum() < MIN_INLIERS or _spread(rest[inliers]) < MIN_SPREAD:
        return None
    return HeadFit(
        R,
        float(s),
        inliers.astype(np.float64),
        float(np.sqrt((dist[inliers] ** 2).mean())),
    )
