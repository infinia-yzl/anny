# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Anatomical relief on the fine body mesh: the collarbones and the biceps. Ported from the legacy
3D Model build (build/relief.py); it works in the legacy frame of the authoring rig.

The MakeHuman body at this build is smooth where a slim teenager shows bone and muscle. Two features read as
missing: the collarbones between the neck and the chest, and the belly of the biceps (the middle of the upper arm was
a plain tube, thinner than the forearm below the elbow). This module adds them as a displacement along the rest
normals of the final mesh.

The relief is a layer of detail on top of the simulated body: the soft-tissue simulation and the fit of the
corrective shapes use the mesh without it (geo_cur.npz keeps it in 'detail'), and skinning carries it with the bones.

    from anny.viewer import relief
    D = relief.relief(V, N, arm_w)     # arm_w: skin weight of the upper arm per side on the fine mesh
"""

import numpy as np

from anny.poses.authoring import posing as P

IX, H = P.IDX, P.HEADS

# collarbone: centre line in the front view (x, y) from the joint with the breastbone to the tip of the shoulder,
# left side; the right side mirrors it. The depth comes from the skin.
CLAV_XY = np.array(
    [
        [0.020, 0.3445],
        [0.035, 0.3405],
        [0.055, 0.3375],
        [0.075, 0.3365],
        [0.095, 0.337],
        [0.112, 0.338],
        [0.127, 0.3375],
    ]
)
CLAV = dict(
    ridge=0.003,
    ridge_width=0.0095,
    fossa=0.004,
    fossa_offset=0.019,
    fossa_width=0.012,
    below=0.001,
    below_offset=0.016,
    below_width=0.01,
    notch=0.003,
    notch_width=0.0095,
    head=0.001,
    scm=0.0016,
    scm_width=0.007,
)
ARM = dict(
    biceps=0.0065,
    biceps_at=0.53,
    biceps_len=0.24,
    triceps=0.0025,
    triceps_at=0.43,
    triceps_len=0.24,
)


def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _surface_point(V, N, x, y):
    """the front skin point at (x, y) in the front view"""
    sel = (
        (np.abs(V[:, 0] - x) < 0.003) & (np.abs(V[:, 1] - y) < 0.003) & (N[:, 2] > 0.0)
    )
    if not sel.any():
        sel = (
            (np.abs(V[:, 0] - x) < 0.006)
            & (np.abs(V[:, 1] - y) < 0.006)
            & (N[:, 2] > 0.0)
        )
    k = np.where(sel)[0]
    return V[k[np.argmax(V[k, 2])]]


def _polyline(V, N, side):
    sx = 1.0 if side == "L" else -1.0
    pts = []
    for x, y in CLAV_XY:
        # a denser line: resample between the control points
        pts.append(_surface_point(V, N, sx * x, y))
    pts = np.array(pts)
    # resample to 2 mm steps along the line and relax the depth so it follows the skin smoothly
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.linspace(0, s[-1], int(s[-1] / 0.002) + 2)
    line = np.stack([np.interp(ss, s, pts[:, k]) for k in range(3)], 1)
    return line, ss / s[-1]


def _closest_on_polyline(X, line, t):
    """closest point of a polyline for each row of X, with its parameter (from t) and distance"""
    A, B = line[:-1], line[1:]
    AB = B - A
    L2 = np.maximum((AB**2).sum(1), 1e-18)
    s = np.clip(
        np.einsum("nsd,sd->ns", X[:, None, :] - A[None], AB) / L2[None], 0.0, 1.0
    )
    P = A[None] + s[..., None] * AB[None]
    d2 = ((X[:, None, :] - P) ** 2).sum(2)
    k = np.argmin(d2, 1)
    r = np.arange(len(X))
    param = t[:-1][k] + s[r, k] * (t[1:][k] - t[:-1][k])
    return P[r, k], param, np.sqrt(d2[r, k])


def clavicles(V, N):
    D = np.zeros(len(V))
    c = CLAV
    for side in "LR":
        sx = 1.0 if side == "L" else -1.0
        line, t = _polyline(V, N, side)
        # vertices of the region: front and top of the upper chest near the line
        box = (
            (sx * V[:, 0] > -0.005)
            & (sx * V[:, 0] < 0.15)
            & (V[:, 1] > 0.29)
            & (V[:, 1] < 0.40)
            & (V[:, 2] > 0.0)
        )
        idx = np.where(box)[0]
        X = V[idx]
        # nearest point of the line, continuous along it (the legacy code took the nearest sample,
        # whose jumps showed on anny's chest)
        near, tj, dist = _closest_on_polyline(X, line, t)
        rel = X - near
        # above the line on the skin (toward the neck and the top of the shoulder) or below it (toward the chest)
        up = rel[:, 1] - 0.6 * rel[:, 2]
        # past the tips of the line: fade out there
        u0 = line[1] - line[0]
        u0 = u0 / np.linalg.norm(u0)
        u1 = line[-1] - line[-2]
        u1 = u1 / np.linalg.norm(u1)
        past = np.maximum(0.0, -(X - line[0]) @ u0) + np.maximum(
            0.0, (X - line[-1]) @ u1
        )
        ends = np.exp(-((past / 0.006) ** 2))
        # stronger toward the breastbone; the ridge eases in at the tip, where it meets the head of
        # the bone (on anny's mesh the full sum made a sharp knob there)
        along = (0.75 + 0.25 * np.cos(np.pi * tj)) * (0.6 + 0.4 * smooth01(tj / 0.12))
        ridge = c["ridge"] * along * np.exp(-((dist / c["ridge_width"]) ** 2)) * ends
        # the hollow above the collarbone, between the neck muscle and the trapezius (middle of the bone)
        mid = smooth01((tj - 0.12) / 0.2) * (1 - smooth01((tj - 0.72) / 0.22))
        above = smooth01(up / 0.004 + 0.5)
        fossa = (
            -c["fossa"]
            * mid
            * np.exp(-(((up - c["fossa_offset"]) / c["fossa_width"]) ** 2))
            * above
        )
        # a slight hollow below the outer third (between the chest and the shoulder muscle)
        outer = smooth01((tj - 0.55) / 0.2) * (1 - smooth01((tj - 0.95) / 0.1))
        below = (
            -c["below"]
            * outer
            * np.exp(-(((-up - c["below_offset"]) / c["below_width"]) ** 2))
            * (1 - above)
        )
        # the rounded head of the bone at the breastbone
        head = c["head"] * np.exp(-((np.linalg.norm(X - line[0], axis=1) / 0.008) ** 2))
        D[idx] += ridge + fossa + below + head
    # the neck muscles (sternocleidomastoid) that rise from the heads of the collarbones toward the ears: a soft ridge
    # on each side of the notch, fading out up the neck
    for side in "LR":
        sx = 1.0 if side == "L" else -1.0
        p0 = _surface_point(V, N, sx * 0.019, 0.347)
        p1 = _surface_point(V, N, sx * 0.034, 0.392)
        seg = p1 - p0
        L2 = seg @ seg
        box = (
            (np.abs(V[:, 0] - sx * 0.027) < 0.03)
            & (V[:, 1] > 0.335)
            & (V[:, 1] < 0.40)
            & (V[:, 2] > 0.02)
        )
        idx = np.where(box)[0]
        tt = np.clip(((V[idx] - p0) @ seg) / L2, 0, 1)
        dist = np.linalg.norm(V[idx] - (p0 + tt[:, None] * seg), axis=1)
        D[idx] += (
            c["scm"]
            * np.exp(-((dist / c["scm_width"]) ** 2))
            * (1 - smooth01((tt - 0.6) / 0.4))
            * smooth01(tt / 0.15 + 0.3)
        )
    # the notch at the top of the breastbone, between the two heads of the collarbones
    notch_c = _surface_point(V, N, 0.0, 0.352)
    r = np.linalg.norm(V - notch_c, axis=1)
    front = V[:, 2] > 0.02
    D += -c["notch"] * np.exp(-((r / c["notch_width"]) ** 2)) * front
    return D


def arm_weights(subdivision, dense_weights):
    """skin weight of the upper arm (upperarm01 + upperarm02) on each fine vertex, per side; the
    weights follow the topology only"""
    out = {}
    for side in "LR":
        cols = [IX[f"upperarm01.{side}"], IX[f"upperarm02.{side}"]]
        out[side] = subdivision(dense_weights[:, cols].sum(1, keepdims=True))[:, 0]
    return out


def biceps(V, N, arm_w):
    D = np.zeros(len(V))
    a = ARM
    for side in "LR":
        s0, s1, w0 = (
            IX[f"upperarm01.{side}"],
            IX[f"lowerarm01.{side}"],
            IX[f"wrist.{side}"],
        )
        A, B, W = H[s0], H[s1], H[w0]
        u = (B - A) / np.linalg.norm(B - A)
        f = (W - B) / np.linalg.norm(W - B)
        front = f - u * (f @ u)
        front /= np.linalg.norm(front)  # the side the forearm folds to: the biceps
        arm = arm_w[side]
        idx = np.where(arm > 0.2)[0]
        X = V[idx] - A
        t = (X @ u) / np.linalg.norm(B - A)
        radial = X - np.outer(X @ u, u)
        rn = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
        c = rn @ front
        bic = (
            a["biceps"]
            * np.exp(-(((t - a["biceps_at"]) / a["biceps_len"]) ** 2))
            * np.clip(c, 0, None) ** 1.2
        )
        tri = (
            a["triceps"]
            * np.exp(-(((t - a["triceps_at"]) / a["triceps_len"]) ** 2))
            * np.clip(-c, 0, None) ** 1.2
        )
        # only on the arm itself: fade out where the weights hand over to the shoulder and to the forearm
        k = (
            smooth01((arm[idx] - 0.2) / 0.5)
            * smooth01((t - 0.08) / 0.15)
            * (1 - smooth01((t - 0.9) / 0.12))
        )
        D[idx] += (bic + tri) * k
    return D


def relief(V, N, arm_w):
    """displacement vectors (rest) for the final mesh; arm_w from arm_weights(). The result is exactly symmetric:
    each vertex takes the mean of its own value and its mirror vertex's value."""
    from scipy.spatial import cKDTree

    d = clavicles(V, N) + biceps(V, N, arm_w)
    # keep it below the neck line of the head sculpt and off the head
    d *= np.clip((0.40 - V[:, 1]) / 0.02, 0, 1)
    D = N * d[:, None]
    dist, m = cKDTree(V).query(V * np.array([-1.0, 1.0, 1.0]))
    ok = dist < 2e-5
    Dm = D[m] * np.array([-1.0, 1.0, 1.0])
    D[ok] = 0.5 * (D[ok] + Dm[ok])
    return D
