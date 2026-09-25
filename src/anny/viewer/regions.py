# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Skin regions of the viewer: albedo, lips, oily areas, veins, vellus hair, pores, moles and body
masks. Ported from the legacy 3D Model build (build/regions.py). The landmarks are in the legacy
frame of the authoring rig, where anny's default body has the eyes and the floor of the legacy
body; the joint helpers come from anny's own MakeHuman base mesh.
"""

import numpy as np

MM = 0.001


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# landmarks (meters), left side (x>0); mirror for right
LM = dict(
    eye=np.array([0.0266, 0.5111, 0.1181]),
    stomion=np.array([0.0, 0.4463, 0.140]),
    nose_tip=np.array([0.0, 0.478, 0.155]),
    cheek=np.array([0.040, 0.474, 0.105]),
    ear=np.array([0.071, 0.497, 0.045]),
    chin=np.array([0.0, 0.412, 0.128]),
)
UP_X = np.array([0, 3.5, 8, 12, 15.5, 18, 19.5, 21]) * MM
UP_Y = np.array([452.6, 453.3, 452.3, 450.6, 448.6, 446.9, 446.3, 446.2]) * MM
LO_X = np.array([0, 5, 9, 13, 16, 18.3, 19.5, 21]) * MM
LO_Y = np.array([438.6, 438.8, 439.6, 441.2, 443.2, 445.2, 446.2, 446.3]) * MM
MOUTH = None


def set_mouth(info):
    global MOUTH
    MOUTH = info


def lip_mask(V, soft=0.55 * MM):
    ax = np.abs(V[:, 0])
    if MOUTH is None:
        yu = np.interp(ax, UP_X, UP_Y)
        yl = np.interp(ax, LO_X, LO_Y)
        xc = 20.2 * MM
    else:
        xs = np.array(MOUTH["xs"])
        ys = np.array(MOUTH["ys"])
        xc = xs[-1] + 1.6 * MM
        ysx = np.interp(ax, xs, ys, right=ys[-1])
        u = np.clip(ax / xc, 0, 1)
        hu = (
            np.interp(
                u,
                [0, 0.18, 0.41, 0.62, 0.79, 0.92, 1.0],
                [6.3, 7.0, 6.0, 4.3, 2.3, 0.7, 0.0],
            )
            * MM
        )
        hl = (
            np.interp(
                u,
                [0, 0.26, 0.46, 0.67, 0.82, 0.94, 1.0],
                [7.7, 7.5, 6.7, 5.1, 3.1, 1.1, 0.0],
            )
            * MM
        )
        yu = ysx + hu
        yl = ysx - hl
    m = smoothstep(yu + soft, yu - soft, V[:, 1]) * smoothstep(
        yl - soft, yl + soft, V[:, 1]
    )
    m *= smoothstep(xc + 0.6 * MM, xc - 0.6 * MM, ax)
    m *= V[:, 2] > 0.100
    return m


def blob(V, c, r, mirror=True, scale=(1, 1, 1)):
    s = np.array(scale)

    def g(cc):
        d = (V - cc) / (r * s)
        return np.exp(-0.5 * (d**2).sum(1))

    out = g(c)
    if mirror:
        c2 = c * np.array([-1, 1, 1])
        out = np.maximum(out, g(c2))
    return out


def vnoise3(P, freq, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.random(4096)
    Q = P * freq
    i = np.floor(Q).astype(np.int64)
    f = Q - i
    u = f * f * (3 - 2 * f)

    def h(ix, iy, iz):
        return perm[(ix * 73856093 ^ iy * 19349663 ^ iz * 83492791) & 4095]

    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (u[:, 0] if dx else 1 - u[:, 0])
                    * (u[:, 1] if dy else 1 - u[:, 1])
                    * (u[:, 2] if dz else 1 - u[:, 2])
                )
                out = out + w * h(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
    return out


def fbm3(P, freq, octaves=3, seed=0):
    s = 0
    a = 1
    tot = 0
    for o in range(octaves):
        s = s + a * vnoise3(P, freq * 2**o, seed + o)
        tot += a
        a *= 0.5
    return s / tot


def srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def compute_albedo(V, N, ao, eye_centers, R_eye, scalp=None, brow=None):
    nV = len(V)
    base = srgb2lin(np.array([0.78, 0.595, 0.462]))  # light-medium warm skin
    alb = np.tile(base, (nV, 1))
    # low-frequency mottling (hue + value)
    n1 = fbm3(V, 60.0, 3, 1) - 0.5
    n2 = fbm3(V, 180.0, 2, 7) - 0.5
    alb *= (1 + 0.10 * n1)[:, None]
    red_mottle = np.clip(n2 * 2, -1, 1)
    # redness zones
    red = 0.0
    red = red + 0.65 * blob(V, LM["cheek"], 0.015, scale=(1.0, 0.85, 1.2))
    red = red + 0.60 * blob(
        V,
        LM["nose_tip"] + np.array([0, -0.002, -0.004]),
        0.008,
        mirror=False,
        scale=(1.4, 1, 1),
    )
    red = red + 0.35 * blob(V, LM["chin"], 0.012, mirror=False)
    ear = ear_mask(V)
    red = red + 0.22 * ear
    # eyelids / periorbital
    lid_up = blob(
        V, LM["eye"] + np.array([0.0, 0.0085, 0.006]), 0.006, scale=(2.0, 0.8, 1.0)
    )
    lid_lo = blob(
        V, LM["eye"] + np.array([0.0, -0.0105, 0.004]), 0.006, scale=(2.0, 0.7, 1.0)
    )
    red = red + 0.35 * lid_up + 0.25 * lid_lo
    red = np.clip(red + 0.15 * red_mottle, 0, 1.2)
    red_col = srgb2lin(np.array([0.78, 0.47, 0.42])) / base
    alb *= 1 + (red_col - 1)[None, :] * red[:, None]
    # under-eye slightly darker/cooler
    dark = 0.6 * lid_lo + 0.35 * blob(
        V, LM["eye"] + np.array([0.004, -0.015, 0.0]), 0.007, scale=(1.6, 0.8, 1.0)
    )
    dark_col = srgb2lin(np.array([0.70, 0.55, 0.50])) / base
    alb *= 1 + (dark_col - 1)[None, :] * np.clip(dark, 0, 1)[:, None] * 0.6
    # lips
    lip = lip_mask(V)
    lip_col = srgb2lin(np.array([0.69, 0.385, 0.35]))
    alb = (
        alb * (1 - lip[:, None])
        + lip_col[None, :] * (1 + 0.06 * n1)[:, None] * lip[:, None]
    )
    # eye socket lining / conjunctiva / caruncle
    for c in eye_centers:
        d = np.linalg.norm(V - c, axis=1)
        sock = smoothstep(R_eye + 0.0016, R_eye + 0.0003, d)
        car_col = srgb2lin(np.array([0.80, 0.53, 0.50]))
        alb = alb * (1 - sock[:, None]) + car_col[None, :] * sock[:, None]
    # interior cavities (mouth, nostrils) darker red
    cav = smoothstep(0.35, 0.05, ao)
    alb *= (1 - 0.35 * cav)[:, None]
    alb[:, 1:] *= (1 - 0.25 * cav)[:, None]
    # neck slightly paler/yellower
    neck = smoothstep(0.415, 0.385, V[:, 1]) * smoothstep(0.08, 0.10, V[:, 2] + 0.1)
    alb *= 1 + np.array([0.0, 0.02, 0.0])[None, :] * neck[:, None]
    # moles, freckles, blemishes
    rng = np.random.default_rng(5)
    moles = [
        ((0.047, 0.468, None), 0.0009, 0.55),
        ((-0.031, 0.398, None), 0.0008, 0.5),
        ((0.021, 0.566, None), 0.0006, 0.4),
        ((-0.016, 0.487, None), 0.0005, 0.35),
        ((0.052, 0.432, None), 0.0007, 0.45),
        ((-0.052, 0.515, None), 0.0006, 0.35),
    ]
    mole_col = srgb2lin(np.array([0.42, 0.28, 0.22]))
    for (mx, my, _), rad, amp in moles:
        sel = (np.abs(V[:, 0] - mx) < 0.004) & (np.abs(V[:, 1] - my) < 0.004)
        if not sel.any():
            continue
        idx = np.where(sel)[0]
        front = idx[
            np.argmax(
                V[idx, 2]
                * np.sign(np.dot(N[idx], np.array([np.sign(mx) * 0.3, 0, 1])) + 1e-9)
            )
        ]
        c3 = V[front]
        dd = np.linalg.norm(V - c3, axis=1)
        w = smoothstep(rad * 1.25, rad * 0.55, dd) * amp
        alb = alb * (1 - w[:, None]) + mole_col[None, :] * w[:, None]
    # faint freckles over nose bridge and upper cheeks
    fr_n = 70
    fx = rng.normal(0, 0.018, fr_n)
    fy = 0.485 + rng.normal(0, 0.006, fr_n)
    fcol = srgb2lin(np.array([0.62, 0.42, 0.32]))
    region = np.abs(V[:, 0]) < 0.05
    for i in range(fr_n):
        sel = (
            region
            & (np.abs(V[:, 0] - fx[i]) < 0.003)
            & (np.abs(V[:, 1] - fy[i]) < 0.003)
            & (V[:, 2] > 0.08)
        )
        if not sel.any():
            continue
        idx = np.where(sel)[0]
        c3 = V[idx[np.argmax(V[idx, 2])]]
        dd = np.linalg.norm(V - c3, axis=1)
        w = smoothstep(0.0007, 0.0002, dd) * rng.uniform(0.08, 0.2)
        alb = alb * (1 - w[:, None]) + fcol[None, :] * w[:, None]
    # two small blemishes (teen skin)
    for (bx, by), rad in [((0.012, 0.552), 0.0011), ((-0.008, 0.418), 0.0012)]:
        sel = (
            (np.abs(V[:, 0] - bx) < 0.003)
            & (np.abs(V[:, 1] - by) < 0.003)
            & (V[:, 2] > 0.08)
        )
        if not sel.any():
            continue
        idx = np.where(sel)[0]
        c3 = V[idx[np.argmax(V[idx, 2])]]
        dd = np.linalg.norm(V - c3, axis=1)
        w = smoothstep(rad * 1.6, 0.0, dd) * 0.35
        bcol = srgb2lin(np.array([0.76, 0.42, 0.40]))
        alb = alb * (1 - w[:, None]) + bcol[None, :] * w[:, None]
    if scalp is not None:
        sc_col = srgb2lin(np.array([0.24, 0.18, 0.15]))
        alb = alb * (1 - scalp[:, None]) + sc_col[None, :] * scalp[:, None]
    if brow is not None:
        alb *= (1 - 0.35 * brow)[:, None]
    return np.clip(alb, 0, 1), dict(lip=lip, red=red)


def rough_mask(V):
    # 0..1 oiliness (lower roughness): T-zone forehead + nose
    t = blob(V, np.array([0, 0.545, 0.12]), 0.018, mirror=False, scale=(1.8, 1.0, 1.0))
    t = np.maximum(
        t,
        blob(V, np.array([0, 0.49, 0.145]), 0.012, mirror=False, scale=(0.8, 1.6, 1.0)),
    )
    t = np.maximum(t, 0.6 * blob(V, np.array([0, 0.418, 0.125]), 0.01, mirror=False))
    return np.clip(t, 0, 1)


def ear_mask(V):
    c = LM["ear"]
    d = (np.abs(V[:, 0:1]) - c[0]) / 0.013
    e = np.sqrt(
        d[:, 0] ** 2 + ((V[:, 1] - c[1]) / 0.029) ** 2 + ((V[:, 2] - c[2]) / 0.017) ** 2
    )
    return smoothstep(1.15, 0.85, e)


# ---------------------------------------------------------------------------------------------- body
def joint_positions():
    """joint helper centres of anny's default body (legacy frame), and the nipples"""
    from .landmarks import joint_helpers

    return dict(joint_helpers())


def body_masks(V, N, J):
    """soft masks for body skin regions: areolae, palms, soles, knuckles, knees, elbows, nails"""
    m = {}
    # areolae: most forward point of each side of the chest around the nipple line
    ar = np.zeros(len(V))
    for key in ("l-nipple", "r-nipple"):
        if key not in J:
            continue
        # snap the MakeHuman nipple landmark onto the surface
        k = np.argmin(np.linalg.norm(V - J[key], axis=1))
        c = V[k]
        dd = np.linalg.norm(V - c, axis=1)
        ar = np.maximum(ar, smoothstep(0.0112, 0.0072, dd) * (N[:, 2] > 0.2))
        ar = np.maximum(ar, 0.6 * smoothstep(0.0028, 0.0012, dd))
    m["areola"] = ar
    # hands: palms face down and toward the thighs in the A-pose
    palm = np.zeros(len(V))
    nail = np.zeros(len(V))
    knuck = np.zeros(len(V))
    for s, k in ((1, "l"), (-1, "r")):
        w = J[f"{k}-hand"]
        el = J[f"{k}-elbow"]
        axis = (w - el) / np.linalg.norm(w - el)
        along = (V - w) @ axis
        hand = smoothstep(-0.01, 0.015, along) * (V[:, 0] * s > 0.3)
        dpalm = np.array([-s * 0.55, -0.75, 0.35])
        dpalm /= np.linalg.norm(dpalm)
        pal = smoothstep(0.1, 0.55, N @ dpalm) * hand * smoothstep(0.17, 0.12, along)
        palm = np.maximum(palm, pal)
        tip = smoothstep(0.125, 0.155, along) * hand
        nail = np.maximum(nail, tip * smoothstep(0.0, 0.4, -(N @ dpalm)))
        knuck = np.maximum(
            knuck,
            smoothstep(0.075, 0.085, along)
            * smoothstep(0.1, 0.085, along)
            * hand
            * smoothstep(0.0, 0.5, -(N @ dpalm)),
        )
    m["palm"] = palm
    m["nail"] = nail
    m["knuckle"] = knuck
    # soles
    g = J["ground"][1]
    m["sole"] = smoothstep(g + 0.02, g + 0.004, V[:, 1]) * smoothstep(
        -0.2, -0.7, N[:, 1]
    )
    # knees (front of the kneecap) and elbows (back of the elbow)
    kn = np.zeros(len(V))
    eb = np.zeros(len(V))
    for k in ("l", "r"):
        kc = J[f"{k}-knee"] + np.array([0, 0.005, 0.045])
        kn = np.maximum(
            kn,
            np.exp(
                -0.5
                * (np.linalg.norm((V - kc) / np.array([0.028, 0.035, 0.02]), axis=1))
                ** 2
            ),
        )
        ec = J[f"{k}-elbow"] + np.array([0, 0.0, -0.035])
        eb = np.maximum(
            eb,
            np.exp(
                -0.5
                * (np.linalg.norm((V - ec) / np.array([0.022, 0.03, 0.02]), axis=1))
                ** 2
            ),
        )
    m["knee"] = kn
    m["elbow"] = eb
    return m


def body_albedo(V, N, alb, J):
    """add body skin regions on top of the face albedo; returns the new albedo and the masks"""
    m = body_masks(V, N, J)
    ar_col = srgb2lin(np.array([0.64, 0.44, 0.39]))
    a = m["areola"][:, None] * 0.85
    alb = alb * (1 - a) + ar_col[None, :] * a
    light = srgb2lin(np.array([0.86, 0.66, 0.56])) / srgb2lin(
        np.array([0.78, 0.595, 0.462])
    )
    ps = np.maximum(m["palm"], m["sole"])[:, None] * 0.8
    alb = alb * (1 + (light - 1)[None, :] * ps)
    nail_col = srgb2lin(np.array([0.86, 0.66, 0.62]))
    nl = m["nail"][:, None] * 0.7
    alb = alb * (1 - nl) + nail_col[None, :] * nl
    red_col = srgb2lin(np.array([0.76, 0.50, 0.44])) / srgb2lin(
        np.array([0.78, 0.595, 0.462])
    )
    rd = np.clip(0.45 * m["knee"] + 0.35 * m["elbow"] + 0.4 * m["knuckle"], 0, 1)[
        :, None
    ]
    alb = alb * (1 + (red_col - 1)[None, :] * rd)
    return np.clip(alb, 0, 1), m


# ---------------------------------------------------------------------------------------------- skin detail masks (v5)
# plain base layer (snug trunks painted on the skin); the viewer draws the same shape in the shader
BASE_LAYER = dict(
    waist_front=0.0, waist_back=0.012, leg_y=-0.235, half_x=0.2, band=0.026
)
BASE_COLOR = np.array([0.030, 0.032, 0.036])  # linear


def base_layer_dist(P, B=BASE_LAYER):
    """signed distance (m) to the edge of the base layer, positive inside"""
    waist = B["waist_back"] + (B["waist_front"] - B["waist_back"]) * smoothstep(
        -0.03, 0.06, P[:, 2]
    )
    return np.minimum(
        np.minimum(waist - P[:, 1], P[:, 1] - B["leg_y"]), B["half_x"] - np.abs(P[:, 0])
    )


def face_weight(V):
    return smoothstep(0.372, 0.41, V[:, 1])


def seg_param(V, a, b):
    """position along a->b (0 at a, 1 at b) and distance from the segment"""
    ab = b - a
    L = np.linalg.norm(ab)
    u = ab / L
    t = ((V - a) @ u) / L
    proj = a + np.clip(t, 0, 1)[:, None] * ab
    return t, np.linalg.norm(V - proj, axis=1)


def limb_masks(V, N, J):
    """soft masks for limbs: forearm, upper arm, hand, lower leg, thigh, foot, plus the palm direction per side"""
    m = {
        k: np.zeros(len(V))
        for k in (
            "forearm",
            "upperarm",
            "hand",
            "dorsal_hand",
            "fingertip",
            "shin",
            "thigh",
            "foot",
            "dorsal_foot",
            "toes",
            "heel",
            "ventral_arm",
            "armpit",
            "shoulder_top",
            "inner_ankle",
        )
    }
    for s, k in ((1, "l"), (-1, "r")):
        side = V[:, 0] * s > 0
        sh, el, wr = J[f"{k}-shoulder"], J[f"{k}-elbow"], J[f"{k}-hand"]
        kn, an = J[f"{k}-knee"], J[f"{k}-ankle"]
        hip = np.array([s * 0.085, -0.10, 0.01])
        dpalm = np.array([-s * 0.55, -0.75, 0.35])
        dpalm /= np.linalg.norm(dpalm)
        t, r = seg_param(V, el, wr)
        fa = (
            side
            * smoothstep(0.075, 0.05, r)
            * smoothstep(-0.08, 0.1, t)
            * smoothstep(1.08, 0.98, t)
        )
        m["forearm"] = np.maximum(m["forearm"], fa)
        t2, r2 = seg_param(V, sh, el)
        ua = (
            side
            * smoothstep(0.075, 0.055, r2)
            * smoothstep(0.05, 0.3, t2)
            * smoothstep(1.1, 0.95, t2)
        )
        m["upperarm"] = np.maximum(m["upperarm"], ua)
        axis = (wr - el) / np.linalg.norm(wr - el)
        along = (V - wr) @ axis
        hand = smoothstep(-0.01, 0.015, along) * (V[:, 0] * s > 0.3)
        m["hand"] = np.maximum(m["hand"], hand)
        m["dorsal_hand"] = np.maximum(
            m["dorsal_hand"],
            hand * smoothstep(0.0, -0.45, N @ dpalm) * smoothstep(0.1, 0.07, along),
        )
        m["fingertip"] = np.maximum(
            m["fingertip"], hand * smoothstep(0.115, 0.15, along)
        )
        ven = smoothstep(0.05, 0.45, N @ dpalm)
        m["ventral_arm"] = np.maximum(
            m["ventral_arm"], ven * np.maximum(fa, 0.6 * ua * smoothstep(0.5, 0.95, t2))
        )
        # armpit: under the shoulder joint, normals pointing down and toward the body
        dd = np.linalg.norm(V - (sh + np.array([-s * 0.01, -0.055, -0.005])), axis=1)
        m["armpit"] = np.maximum(
            m["armpit"],
            side * smoothstep(0.045, 0.02, dd) * smoothstep(-0.1, -0.5, N[:, 1]),
        )
        dd = np.linalg.norm(V - (sh + np.array([s * 0.01, 0.035, 0.0])), axis=1)
        m["shoulder_top"] = np.maximum(
            m["shoulder_top"],
            side * smoothstep(0.07, 0.02, dd) * smoothstep(0.2, 0.7, N[:, 1]),
        )
        t3, r3 = seg_param(V, kn, an)
        m["shin"] = np.maximum(
            m["shin"],
            side
            * smoothstep(0.08, 0.06, r3)
            * smoothstep(0.0, 0.25, t3)
            * smoothstep(1.1, 0.95, t3),
        )
        t4, r4 = seg_param(V, hip, kn)
        m["thigh"] = np.maximum(
            m["thigh"],
            side
            * smoothstep(0.11, 0.09, r4)
            * smoothstep(0.1, 0.4, t4)
            * smoothstep(1.05, 0.9, t4),
        )
        foot = side * smoothstep(an[1] + 0.03, an[1] + 0.005, V[:, 1])
        m["foot"] = np.maximum(m["foot"], foot)
        m["dorsal_foot"] = np.maximum(
            m["dorsal_foot"],
            foot
            * smoothstep(0.2, 0.6, N[:, 1])
            * smoothstep(an[2] - 0.01, an[2] + 0.03, V[:, 2])
            * smoothstep(0.125, 0.105, V[:, 2]),
        )
        m["toes"] = np.maximum(m["toes"], foot * smoothstep(0.11, 0.135, V[:, 2]))
        m["heel"] = np.maximum(
            m["heel"], foot * smoothstep(an[2] - 0.01, an[2] - 0.04, V[:, 2])
        )
        di = np.linalg.norm(
            (V - (an + np.array([-s * 0.022, 0.0, 0.0])))
            / np.array([0.02, 0.05, 0.035]),
            axis=1,
        )
        m["inner_ankle"] = np.maximum(m["inner_ankle"], side * smoothstep(1.0, 0.4, di))
    return m


def body_tone(V, N, alb, J):
    """regional colour of the body skin: warmer and redder hands and feet, a light tan on the forearms and shins,
    lighter inner arms and torso, darker armpits, and large soft blotches"""
    m = limb_masks(V, N, J)
    body = 1.0 - face_weight(V)

    def tint(a, col, w):
        return a * (1 + (np.asarray(col) - 1)[None, :] * np.clip(w, 0, 1)[:, None])

    tan = np.maximum(m["forearm"] * (1 - 0.5 * m["ventral_arm"]), m["shin"] * 0.9)
    alb = tint(alb, [0.95, 0.925, 0.905], tan * body)
    alb = tint(
        alb, [0.955, 0.885, 0.855], np.maximum(m["dorsal_hand"], 0.5 * m["hand"])
    )
    alb = tint(alb, [0.99, 0.855, 0.82], m["fingertip"])
    alb = tint(alb, [0.965, 0.90, 0.875], m["dorsal_foot"])
    alb = tint(alb, [0.985, 0.845, 0.805], np.maximum(m["toes"], m["heel"]))
    alb = tint(alb, [0.87, 0.85, 0.85], m["armpit"])
    alb = tint(alb, [0.975, 0.935, 0.915], m["shoulder_top"])
    inner = m["ventral_arm"] * (1 - m["forearm"] * 0.5)
    alb = tint(alb, [1.03, 1.03, 1.02], inner)
    torso = (
        smoothstep(0.2, 0.15, np.abs(V[:, 0]))
        * smoothstep(-0.12, -0.05, V[:, 1])
        * smoothstep(0.36, 0.3, V[:, 1])
    )
    alb = tint(alb, [1.015, 1.02, 1.015], torso)
    # large soft blotches (value) and hemoglobin blotches (hue), body only
    n1 = fbm3(V, 16.0, 3, 21) - 0.5
    n2 = fbm3(V, 45.0, 3, 33) - 0.5
    alb *= (1 + 0.11 * n1 * body)[:, None]
    alb = tint(alb, [1.0, 0.91, 0.90], np.clip(0.5 + 1.8 * n2, 0, 1) * 0.8 * body)
    return np.clip(alb, 0, 1), m


def vein_mask(V, N, J, lm):
    """where thin superficial veins show through the skin (0..1)"""
    w = 0.8 * lm["ventral_arm"] * np.maximum(lm["forearm"], 0.5)
    w = np.maximum(w, lm["dorsal_hand"])
    w = np.maximum(w, 0.8 * lm["dorsal_foot"])
    w = np.maximum(w, 0.7 * lm["inner_ankle"])
    w = np.maximum(
        w, 0.3 * lm["shin"] * smoothstep(0.3, 0.7, -(N[:, 0] * np.sign(V[:, 0])))
    )
    for s in (1, -1):
        c = np.array([s * 0.059, 0.532, 0.082])
        w = np.maximum(
            w, 0.45 * np.exp(-0.5 * (np.linalg.norm(V - c, axis=1) / 0.01) ** 2)
        )
    chest = (
        smoothstep(0.12, 0.06, np.abs(V[:, 0]))
        * smoothstep(0.2, 0.26, V[:, 1])
        * smoothstep(0.34, 0.3, V[:, 1])
        * (N[:, 2] > 0.3)
    )
    w = np.maximum(w, 0.22 * chest)
    return np.clip(w * (1 - lm["fingertip"]), 0, 1)


def fuzz_mask(V, N, J, lm, lip, scalp):
    """density of fine vellus hair (peach fuzz) that catches light at grazing angles"""
    f = face_weight(V)
    body = 1 - f
    w = body * 0.32
    w = np.maximum(w, 0.8 * lm["forearm"] * (1 - 0.4 * lm["ventral_arm"]))
    w = np.maximum(w, 0.5 * lm["upperarm"])
    w = np.maximum(w, 0.9 * lm["shin"])
    w = np.maximum(w, 0.55 * lm["thigh"])
    w = w * (1 - 0.7 * lm["hand"]) * (1 - 0.7 * lm["foot"])
    face = 0.55 * f
    face = np.maximum(
        face,
        0.95
        * blob(
            V,
            np.array([0.0, 0.4555, 0.137]),
            0.008,
            mirror=False,
            scale=(2.6, 0.8, 1.0),
        ),
    )
    face = np.maximum(
        face, 0.75 * blob(V, LM["cheek"] + np.array([0.012, -0.01, -0.02]), 0.016)
    )
    w = np.maximum(w, face)
    w *= (1 - lip) * (1 - scalp)
    return np.clip(w, 0, 1)


def pore_mask(V, lip, ear):
    """visibility of facial pores (0 on the body)"""
    f = face_weight(V)
    p = 0.35 * f
    p = np.maximum(
        p,
        1.0
        * blob(
            V,
            LM["nose_tip"] + np.array([0, 0.004, -0.006]),
            0.012,
            mirror=False,
            scale=(1.4, 1.3, 1.0),
        ),
    )
    p = np.maximum(
        p, 0.8 * blob(V, LM["cheek"] + np.array([-0.008, 0.004, 0.006]), 0.013)
    )
    p = np.maximum(p, 0.6 * blob(V, LM["chin"], 0.011, mirror=False))
    p = np.maximum(
        p,
        0.5
        * blob(
            V, np.array([0.0, 0.55, 0.115]), 0.02, mirror=False, scale=(1.8, 0.8, 1.0)
        ),
    )
    lids = blob(V, LM["eye"], 0.011, scale=(1.3, 0.9, 1.0))
    p *= (1 - 0.8 * lids) * (1 - lip) * (1 - 0.6 * ear)
    return np.clip(p, 0, 1)


def body_moles(V, N, lm, n=16, seed=11):
    """a few small moles on the body: list of (x, y, z, radius, strength)"""
    rng = np.random.default_rng(seed)
    ok = (
        (face_weight(V) < 0.01)
        & (base_layer_dist(V) < -0.01)
        & (lm["hand"] < 0.1)
        & (lm["foot"] < 0.1)
        & (lm["armpit"] < 0.1)
    )
    idx = np.where(ok)[0]
    out = []
    while len(out) < n:
        i = rng.choice(idx)
        p = V[i]
        if any(np.linalg.norm(p - np.array(q[:3])) < 0.05 for q in out):
            continue
        out.append(
            [
                float(p[0]),
                float(p[1]),
                float(p[2]),
                float(rng.uniform(0.0005, 0.0012)),
                float(rng.uniform(0.35, 0.7)),
            ]
        )
    return out


def knuckles(J):
    """finger joints for the wrinkle shader: [cx, cy, cz, kind, ax, ay, az, radius]
    kind 0: base knuckle, 1: middle joint, 2: end joint (thumb joints use 0 and 2)"""
    out = []
    for s, k in ((1, "l"), (-1, "r")):
        for f in range(1, 6):
            P = [J[f"{k}-finger-{f}-{i}"] for i in range(1, 5)]
            joints = [(1, 0), (2, 2)] if f == 1 else [(0, 0), (1, 1), (2, 2)]
            for i, kind in joints:
                a = P[i + 1] - P[i]
                a = a / np.linalg.norm(a)
                rad = (
                    0.0072
                    if f == 1
                    else (0.0066 if kind == 0 else (0.0058 if kind == 1 else 0.005))
                )
                out.append([*map(float, P[i]), float(kind), *map(float, a), rad])
    return out
