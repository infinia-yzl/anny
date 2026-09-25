# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""The animation clips: idle, walk, run, wave, nod, shrug and jump. Each clip is a function of time that builds a
posing.Pose from poselib controls; sample(clip) turns it into frames at 30 frames per second. All clips loop.

Ported from the legacy 3D Model build (build/anims.py).

Walk and run are in place, like on a treadmill: each foot moves back along the floor while it carries the weight
and swings forward in the air. The feet follow explicit paths and two-bone IK places the legs.
"""

import numpy as np
from . import posing as P
from . import poselib as L
from . import poses as PS
from .posing import IDX, HEADS

FPS = 30
TAU = 2 * np.pi


def smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def keys(t, pts, loop=None):
    """Catmull-Rom through (time, value) points; `loop` (period) wraps the curve"""
    ts = np.array([p[0] for p in pts], float)
    vs = np.array([p[1] for p in pts], float)
    if loop:
        t = t % loop
        ts = np.concatenate([ts[-2:] - loop, ts, ts[:2] + loop])
        vs = np.concatenate([vs[-2:], vs, vs[:2]])
    else:
        ts = np.concatenate([[ts[0] - 1], ts, [ts[-1] + 1]])
        vs = np.concatenate([[vs[0]], vs, [vs[-1]]])
        t = np.clip(t, ts[1], ts[-2])
    i = np.searchsorted(ts, t, side="right") - 1
    i = int(np.clip(i, 1, len(ts) - 3))
    t0, t1 = ts[i], ts[i + 1]
    u = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    p0, p1, p2, p3 = vs[i - 1], vs[i], vs[i + 1], vs[i + 2]
    # non-uniform tangents scaled to the segment
    m1 = (p2 - p0) / max(ts[i + 1] - ts[i - 1], 1e-6) * (t1 - t0)
    m2 = (p3 - p1) / max(ts[i + 2] - ts[i], 1e-6) * (t1 - t0)
    u2, u3 = u * u, u * u * u
    return (
        (2 * u3 - 3 * u2 + 1) * p1
        + (u3 - 2 * u2 + u) * m1
        + (-2 * u3 + 3 * u2) * p2
        + (u3 - u2) * m2
    )


def hold(t, pts):
    """eased steps between (time, value) points: each change eases in and out, flat between points"""
    v = pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t <= t0:
            return v0
        if t < t1:
            return v0 + (v1 - v0) * smooth01((t - t0) / (t1 - t0))
        v = v1
    return v


REST_ANKLE_Z = {s: HEADS[IDX[f"foot.{s}"]][2] for s in "LR"}
BASE, BASE_FEET = PS.base_stance()
BASE_Y = BASE.root[1]


# ------------------------------------------------------------------ locomotion
def gait_foot(s, ph, stance, stride, clear, pitch_keys, x, z0, swing_path=None):
    """ankle target and foot pitch for one foot at gait phase ph (0 = heel strike)"""
    ph = ph % 1.0
    if ph < stance:
        u = ph / stance
        z = z0 + stride * (0.5 - u)
        lift = 0.0
    else:
        u = (ph - stance) / (1 - stance)
        e = smooth01(u)
        z = z0 + stride * (-0.5 + e)
        lift = clear * np.sin(np.pi * u) ** 1.5 if swing_path is None else swing_path(u)
    pitch = keys(ph, pitch_keys, loop=1.0)
    return L.foot_on_floor(s, x, z, yaw=5, pitch=pitch, lift=lift), pitch


def walk(t, T=1.1):
    ph = (t / T) % 1.0
    p = P.Pose()
    stance, stride = 0.6, 0.5
    pk = [(0.0, 14), (0.1, 0), (0.42, 0), (0.6, -38), (0.7, -30), (0.85, 5), (0.97, 14)]
    L.pelvis(
        p,
        offset=(
            0.014 * np.cos(TAU * (ph - 0.3)),
            BASE_Y - 0.028 + 0.012 * np.cos(2 * TAU * (ph - 0.3)),
            0.0,
        ),
        pitch=4,
        roll=-3 * np.cos(TAU * (ph - 0.3)),
        yaw=-5 * np.cos(TAU * ph),
    )
    L.spine(p, bend=3, side=2.2 * np.cos(TAU * (ph - 0.3)), twist=8 * np.cos(TAU * ph))
    for s, off in (("L", 0.0), ("R", 0.5)):
        sg = L.side_sign(s)
        a, pitch = gait_foot(
            s, ph + off, stance, stride, 0.075, pk, sg * 0.085, REST_ANKLE_Z[s] + 0.01
        )
        L.leg(p, s, a, yaw=5, pitch=pitch)
    L.head(p, pitch=3, yaw=-3 * np.cos(TAU * ph))
    for s, off in (("L", 0.5), ("R", 0.0)):
        c = np.cos(TAU * (ph - off))
        L.arm_fk(
            p, s, abd=8, flex=4 + 17 * c, twist=-4, elbow=20 + 10 * (c + 1) / 2, pron=-6
        )
        L.hands(p, s, 0.32)
    return p


def run(t, T=0.72):
    ph = (t / T) % 1.0
    p = P.Pose()
    stance, stride = 0.38, 0.62
    pk = [
        (0.0, 6),
        (0.08, 0),
        (0.25, 0),
        (0.38, -45),
        (0.5, -60),
        (0.75, -25),
        (0.92, 0),
        (1.0, 6),
    ]

    def swing(u):  # the heel comes up behind, then the foot reaches forward and down
        return keys(u, [(0.0, 0.0), (0.3, 0.2), (0.55, 0.19), (0.85, 0.07), (1.0, 0.0)])

    bob = keys(
        ph % 0.5, [(0.0, -0.01), (0.19, -0.045), (0.36, 0.01), (0.5, -0.01)], loop=0.5
    )
    L.pelvis(
        p,
        offset=(0.008 * np.cos(TAU * (ph - 0.19)), BASE_Y - 0.05 + bob, 0.0),
        pitch=8,
        roll=-2 * np.cos(TAU * (ph - 0.19)),
        yaw=-7 * np.cos(TAU * ph),
    )
    L.spine(p, bend=5, twist=11 * np.cos(TAU * ph))
    for s, off in (("L", 0.0), ("R", 0.5)):
        sg = L.side_sign(s)
        a, pitch = gait_foot(
            s,
            ph + off,
            stance,
            stride,
            0.0,
            pk,
            sg * 0.075,
            REST_ANKLE_Z[s] + 0.03,
            swing_path=swing,
        )
        a = a + np.array([0, 0.008 * smooth01(-(pitch + 25) / 20), 0])
        L.leg(p, s, a, yaw=5, pitch=pitch)
    L.head(p, pitch=-4, yaw=-4 * np.cos(TAU * ph))
    for s, off in (("L", 0.5), ("R", 0.0)):
        c = np.cos(TAU * (ph - off))
        L.arm_fk(p, s, abd=12, flex=18 + 32 * c, twist=8, elbow=88 + 10 * c, pron=10)
        L.hands(p, s, 0.62, thumb=0.45)
    return p


# ------------------------------------------------------------------ standing clips
def standing(
    p=None, dx=0.0, dy=0.0, roll=0.0, yaw=0.0, pitch=0.0, knee_l=None, feet=None
):
    p = p or P.Pose()
    L.pelvis(p, offset=(dx, BASE_Y + dy, 0.0), roll=roll, yaw=yaw, pitch=pitch)
    return p


def plant(p, feet=None, yaws=None):
    feet = feet or BASE_FEET
    for s in "LR":
        L.leg(p, s, feet[s], yaw=(yaws or {}).get(s, 6))


def idle(t, T=6.0):
    u = t % T
    breath = 0.5 - 0.5 * np.cos(TAU * u / 3.0)
    sway = np.sin(TAU * u / T)
    p = standing(dx=0.012 * sway, dy=-0.004 * abs(sway), roll=-1.6 * sway)
    L.spine(p, bend=-1.2 * breath + 1, side=0.9 * sway)
    plant(p)
    for s in "LR":
        L.clavicle(p, s, elev=1.6 * breath)
    L.head(
        p,
        pitch=keys(u, [(0, 2), (1.6, -1), (3.0, 3), (4.6, 0), (6.0, 2)], loop=T),
        yaw=keys(
            u,
            [
                (0, 0),
                (0.8, 0),
                (1.9, 16),
                (2.9, 14),
                (3.7, 0),
                (4.4, -12),
                (5.2, -11),
                (6.0, 0),
            ],
            loop=T,
        ),
        roll=1.5 * sway,
    )
    for s in "LR":
        sg = L.side_sign(s)
        L.arm_fk(
            p,
            s,
            abd=7 + 0.8 * breath - sg * 0.6 * sway,
            flex=4,
            twist=-5,
            elbow=14 + 1.5 * breath,
            pron=-5,
        )
        L.hands(p, s, 0.28)
    return p


def arm_info(p, s):
    """wrist position, elbow pole and hand directions of one arm in pose p (to drive the same arm by IK)"""
    Rw, Pw = P.fk(p)
    sh, el, wr = (
        Pw[IDX[f"upperarm01.{s}"]],
        Pw[IDX[f"lowerarm01.{s}"]],
        Pw[IDX[f"wrist.{s}"]],
    )
    mid = 0.5 * (sh + wr)
    d0, t0, n0 = L.hand_axes(s)
    R = Rw[IDX[f"wrist.{s}"]]
    return dict(w=wr, pole=el + (el - mid) * 3.0, fingers=R @ d0, palm=R @ n0)


def arm_between(p, s, k, a, b, via=None):
    """drive one arm by IK between two arm states (from arm_info); `via` bends the wrist path through a point"""
    k = float(np.clip(k, 0, 1))
    if via is None:
        w = a["w"] * (1 - k) + b["w"] * k
    else:
        w = (1 - k) ** 2 * a["w"] + 2 * k * (1 - k) * np.asarray(via) + k * k * b["w"]
    pole = a["pole"] * (1 - k) + b["pole"] * k
    fingers = P.unit(a["fingers"] * (1 - k) + b["fingers"] * k)
    palm = P.unit(a["palm"] * (1 - k) + b["palm"] * k)
    L.arm_ik(p, s, w, pole=pole, fingers=fingers, palm=palm)


def relaxed_arms(p, sides="LR"):
    for s in sides:
        L.arm_fk(p, s, abd=7, flex=4, twist=-5, elbow=14, pron=-5)
        L.hands(p, s, 0.28)


def wave(t, T=3.2):
    u = t % T
    up = hold(u, [(0.1, 0), (0.75, 1), (2.4, 1), (3.0, 0)])
    p = standing(dx=-0.012 * up, roll=1.5 * up)
    L.spine(p, bend=1, side=-2.5 * up, twist=-4 * up)
    plant(p)
    L.head(p, pitch=2 - 3 * up, yaw=-5 * up, roll=-5 * up)
    relaxed_arms(p, "L")
    ref = p.copy()
    relaxed_arms(ref, "R")
    a = arm_info(ref, "R")
    sh = P.fk(p)[1][IDX["upperarm01.R"]]
    # the forearm swings about the elbow while the hand is up
    swing = (
        np.sin(TAU * 2.4 * (u - 0.75))
        * smooth01((u - 0.7) / 0.2)
        * (1 - smooth01((u - 2.25) / 0.2))
    )
    ang = np.radians(8 + 22 * swing)
    elbow = sh + np.array([-0.2, -0.04, 0.03])
    b = dict(
        w=elbow + np.array([-np.sin(ang) * 0.2, np.cos(ang) * 0.2, 0.03]),
        pole=elbow + np.array([-0.3, -0.25, -0.1]),
        fingers=np.array([-np.sin(ang) * 0.9, np.cos(ang), 0.15]),
        palm=np.array([0.0, 0.0, 1.0]),
    )
    arm_between(p, "R", up, a, b, via=sh + np.array([-0.12, -0.12, 0.22]))
    L.hands(p, "R", 0.28 - 0.23 * up, thumb=0.17 * (1 - up), spread=6 * up)
    return p


def nod(t, T=2.2):
    u = t % T
    p = standing()
    L.spine(p, bend=hold(u, [(0.2, 0), (0.45, 2.5), (1.1, 2.5), (1.4, 0)]))
    plant(p)
    L.head(
        p,
        pitch=keys(
            u,
            [(0, 1), (0.25, 1), (0.5, 16), (0.72, 3), (0.95, 13), (1.25, 1), (2.2, 1)],
            loop=T,
        ),
    )
    for s in "LR":
        L.arm_fk(p, s, abd=7, flex=4, twist=-5, elbow=14, pron=-5)
        L.hands(p, s, 0.28)
    return p


def shrug(t, T=2.6):
    u = t % T
    k = hold(u, [(0.15, 0), (0.55, 1), (1.3, 1), (1.8, 0)])
    p = standing(dy=0.004 * k)
    L.spine(p, bend=-1.5 * k)
    plant(p)
    L.head(p, pitch=-2 * k, roll=7 * k, yaw=3 * k)
    for s in "LR":
        L.clavicle(p, s, elev=15 * k, prot=3 * k)
    ref = p.copy()
    relaxed_arms(ref)
    for s in "LR":
        sg = L.side_sign(s)
        a = arm_info(ref, s)
        b = dict(
            w=np.array([sg * 0.27, 0.125, 0.17]),
            pole=np.array([sg * 0.35, -0.05, -0.3]),
            fingers=P.unit([sg * 0.55, 0.08, 1.0]),
            palm=np.array([0.0, 1.0, 0.0]),
        )
        arm_between(p, s, k, a, b)
        L.hands(p, s, 0.28 - 0.18 * k, spread=6 * k)
    return p


def jump(t, T=2.4):
    u = t % T
    # timing: crouch 0.1-0.45, push 0.45-0.6, air 0.6-0.98, land 0.98-1.15, recover 1.15-1.7, rest to T
    air = 0.6 <= u < 0.98
    h = 0.0
    if air:
        s = (u - 0.6) / 0.38
        h = 0.2 * 4 * s * (1 - s)
    crouch = keys(
        u,
        [
            (0, 0),
            (0.1, 0),
            (0.42, 1),
            (0.5, 0.75),
            (0.6, -0.05),
            (0.98, 0.05),
            (1.1, 0.8),
            (1.25, 0.75),
            (1.7, 0),
            (2.4, 0),
        ],
        loop=T,
    )
    heel = keys(
        u,
        [
            (0, 0),
            (0.45, 0),
            (0.58, -35),
            (0.66, -45),
            (0.9, -30),
            (0.98, -12),
            (1.05, 0),
            (2.4, 0),
        ],
        loop=T,
    )
    p = standing(dy=-0.13 * crouch + h, pitch=24 * max(crouch, 0))
    L.spine(
        p,
        bend=14 * max(crouch, 0)
        - 4 * smooth01((u - 0.5) / 0.1) * (1 - smooth01((u - 0.85) / 0.1)),
    )
    F = P.fk(p)
    for s in "LR":
        ground = L.foot_on_floor(
            s, L.side_sign(s) * 0.095, REST_ANKLE_Z[s], yaw=6, pitch=min(heel, 0)
        )
        a = ground
        if air:
            # in the air the legs hang from the hips with the knees a little drawn up; they blend from and back
            # to the floor targets at take-off and landing
            tuck = np.sin(np.pi * (u - 0.6) / 0.38)
            hip = F[1][IDX[f"upperleg01.{s}"]]
            hang = hip + np.array(
                [L.side_sign(s) * 0.012, -0.66 + 0.1 * tuck, 0.03 * tuck]
            )
            w = smooth01((u - 0.6) / 0.08) * (1 - smooth01((u - 0.88) / 0.1))
            a = ground * (1 - w) + hang * w
        L.leg(p, s, a, yaw=6, pitch=heel)
    L.head(p, pitch=-12 * max(crouch, 0) + 2)
    swing = keys(
        u,
        [
            (0, 4),
            (0.1, 4),
            (0.42, -45),
            (0.62, 150),
            (0.8, 120),
            (1.0, 40),
            (1.2, 25),
            (1.7, 4),
            (2.4, 4),
        ],
        loop=T,
    )
    for s in "LR":
        L.arm_fk(
            p,
            s,
            abd=8 + max(swing - 60, 0) * 0.15,
            flex=swing,
            twist=-5,
            elbow=14 + 12 * max(crouch, 0),
            pron=-5,
        )
        L.hands(p, s, 0.3)
    return p


CLIPS = dict(
    idle=(idle, 6.0),
    walk=(walk, 1.1),
    run=(run, 0.72),
    wave=(wave, 3.2),
    nod=(nod, 2.2),
    shrug=(shrug, 2.6),
    jump=(jump, 2.4),
)


def sample(name, fps=FPS):
    fn, T = CLIPS[name]
    n = int(round(T * fps))
    return [fn(i / fps) for i in range(n)], T
