# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The authored poses: static poses built from poselib controls. Each function returns a
posing.Pose (with .props). Ported from the legacy 3D Model build (build/poses.py).
"""

import numpy as np
from . import posing as P
from . import poselib as L
from .posing import IDX, HEADS


# leg lengths for fitting the pelvis height
def _leg_len(s):
    Lm = P.LIMBS["leg." + s]
    return Lm["la"], Lm["lb"]


def fit_pelvis(pose, ankles, bend=6.0):
    """lower or raise the pelvis so each knee bends by about `bend` degrees for the given ankle targets"""
    F = P.fk(pose)
    dy = []
    for s, a in ankles.items():
        h = F[1][IDX[f"upperleg01.{s}"]]
        la, lb = _leg_len(s)
        d = np.sqrt(la * la + lb * lb + 2 * la * lb * np.cos(np.radians(bend)))
        # |h + (0, y, 0) - a| = d
        dx2 = (h[0] - a[0]) ** 2 + (h[2] - a[2]) ** 2
        dy.append(a[1] - h[1] + np.sqrt(max(d * d - dx2, 0.0)))
    return min(dy)


def base_stance(width=0.095, yaw=6, bend=5, pelvis_kw=None, feet=None, spine_kw=None):
    """a standing base: pelvis fitted to the feet on the floor"""
    feet = feet or {
        s: L.foot_on_floor(
            s, L.side_sign(s) * width, HEADS[IDX[f"foot.{s}"]][2], yaw=yaw
        )
        for s in "LR"
    }
    p = P.Pose()
    L.pelvis(p, **(pelvis_kw or {}))
    p.root[1] += fit_pelvis(p, feet, bend)
    L.spine(p, **(spine_kw or {}))
    return p, feet


def place_legs(p, feet, yaws=None, poles=None, pitches=None):
    for s in "LR":
        L.leg(
            p,
            s,
            feet[s],
            pole=(poles or {}).get(s),
            yaw=(yaws or {}).get(s, 6),
            pitch=(pitches or {}).get(s, 0),
        )


# ------------------------------------------------------------------ reference
def a_pose():
    return P.Pose()


def t_pose():
    p = P.Pose()
    for s in "LR":
        L.arm_fk(p, s, abd=90, flex=0, twist=0, elbow=0, pron=35)
        L.hands(p, s, 0.05, thumb=0.0)
    return p


# ------------------------------------------------------------------ everyday standing
def relaxed():
    p, feet = base_stance()
    place_legs(p, feet)
    L.head(p, pitch=2)
    for s in "LR":
        L.arm_fk(p, s, abd=7, flex=4, twist=-5, elbow=14, pron=-5)
        L.hands(p, s, 0.28)
    return p


def weight_shift():
    """contrapposto: the weight on the right leg, the left knee relaxed"""
    feet = {
        "R": L.foot_on_floor("R", -0.075, HEADS[IDX["foot.R"]][2], yaw=8),
        "L": L.foot_on_floor("L", 0.125, HEADS[IDX["foot.L"]][2] + 0.04, yaw=14),
    }
    p = P.Pose()
    L.pelvis(p, offset=(-0.024, 0, 0), roll=4, yaw=-4)
    p.root[1] += fit_pelvis(p, {"R": feet["R"]}, 4)
    L.spine(p, bend=1, side=3.5, twist=4)
    place_legs(
        p,
        feet,
        yaws={"R": 8, "L": 14},
        poles={"L": feet["L"] + np.array([0.08, 0.35, 0.5])},
    )
    L.head(p, pitch=3, yaw=6, roll=-4)
    L.arm_fk(p, "L", abd=8, flex=6, twist=-4, elbow=18, pron=-8)
    L.arm_fk(p, "R", abd=10, flex=-2, twist=-6, elbow=10, pron=-4)
    L.hands(p, "L", 0.3)
    L.hands(p, "R", 0.24)
    return p


def hands_on_hips():
    p, feet = base_stance(width=0.115, yaw=10)
    place_legs(p, feet, yaws={"L": 10, "R": 10})
    L.spine(p, bend=-2)
    L.head(p, pitch=0, yaw=-4)
    for s in "LR":
        sg = L.side_sign(s)
        L.clavicle(p, s, prot=-2)
        fingers = P.unit([-0.15 * sg, -0.55, 0.85])
        palm = np.array([-sg, 0.0, 0.1])
        contact = np.array([sg * 0.113, -0.035, 0.005])
        wrist = contact + np.array([sg * 0.014, 0, 0]) - fingers * 0.045
        L.arm_ik(
            p,
            s,
            wrist,
            pole=np.array([sg * 0.6, 0.15, -0.35]),
            fingers=fingers,
            palm=palm,
        )
        L.hands(p, s, 0.12, thumb=0.0)
    return p


def arms_crossed():
    p, feet = base_stance(width=0.1, yaw=8)
    place_legs(p, feet)
    L.spine(p, bend=-1, twist=0)
    L.head(p, pitch=2, yaw=0)
    for s in "LR":
        L.clavicle(p, s, prot=3)
    # the left forearm lies under the right one; each hand holds the opposite upper arm
    L.arm_ik(
        p,
        "L",
        np.array([-0.07, 0.088, 0.128]),
        pole=np.array([0.5, -0.25, 0.05]),
        fingers=P.unit([-0.55, 0.1, -0.85]),
        palm=np.array([0.3, 0.3, -0.2]),
    )
    L.arm_ik(
        p,
        "R",
        np.array([0.095, 0.118, 0.172]),
        pole=np.array([-0.5, -0.25, 0.05]),
        fingers=P.unit([0.8, 0.1, -0.6]),
        palm=np.array([-0.2, 0.0, -1.0]),
    )
    L.hands(p, "L", 0.3, thumb=0.2)
    L.hands(p, "R", 0.45, thumb=0.3)
    return p


# ------------------------------------------------------------------ action
def ready():
    feet = {
        s: L.foot_on_floor(
            s, L.side_sign(s) * 0.15, HEADS[IDX[f"foot.{s}"]][2] + 0.01, yaw=14
        )
        for s in "LR"
    }
    p = P.Pose()
    L.pelvis(p, pitch=16)
    p.root[1] += fit_pelvis(p, feet, 55)
    L.spine(p, bend=6)
    place_legs(
        p,
        feet,
        yaws={"L": 14, "R": 14},
        poles={s: feet[s] + np.array([L.side_sign(s) * 0.25, 0.4, 0.6]) for s in "LR"},
    )
    L.head(p, pitch=-18)
    for s in "LR":
        L.arm_fk(p, s, abd=18, flex=38, twist=25, elbow=88, pron=25)
        L.hands(p, s, 0.45)
    return p


def reach_up():
    feet = {
        s: L.foot_on_floor(
            s, L.side_sign(s) * 0.09, HEADS[IDX[f"foot.{s}"]][2], yaw=6, pitch=-14
        )
        for s in "LR"
    }
    p = P.Pose()
    p.root[1] += fit_pelvis(p, feet, 3)
    L.spine(p, bend=-5, side=-4, twist=-6)
    place_legs(p, feet, pitches={"L": -14, "R": -14})
    L.head(p, pitch=-24, yaw=-8)
    # the right arm points almost straight up in the world, a little out and forward; the girdle follows it
    L.arm_fk(
        p,
        "R",
        aim=L.world_aim(p, "R", [-0.1, 1.0, 0.12]),
        twist=10,
        elbow=6,
        pron=0,
        wrist_flex=-8,
    )
    L.hands(p, "R", 0.08, thumb=0.05, spread=4)
    L.arm_fk(p, "L", abd=12, flex=6, elbow=16, pron=-6)
    L.hands(p, "L", 0.3)
    return p


def crouch():
    feet = {
        s: L.foot_on_floor(
            s, L.side_sign(s) * 0.13, HEADS[IDX[f"foot.{s}"]][2] - 0.02, yaw=16
        )
        for s in "LR"
    }
    p = P.Pose()
    L.pelvis(p, pitch=32)
    p.root[1] += fit_pelvis(p, feet, 118)
    L.spine(p, bend=14)
    place_legs(
        p,
        feet,
        yaws={"L": 16, "R": 16},
        poles={s: feet[s] + np.array([L.side_sign(s) * 0.3, 0.3, 0.6]) for s in "LR"},
    )
    L.head(p, pitch=-30)
    F = P.fk(p)
    for s in "LR":
        sg = L.side_sign(s)
        knee = F[1][IDX[f"lowerleg01.{s}"]]
        # forearms rest on the knees, the hands hang in front of them
        wrist = knee + np.array([-sg * 0.03, 0.02, 0.1])
        L.arm_ik(
            p,
            s,
            wrist,
            pole=np.array([sg * 0.5, 0.3, -0.1]),
            fingers=P.unit([-sg * 0.2, -0.9, 0.35]),
            palm=np.array([-sg, 0.0, -0.3]),
        )
        L.hands(p, s, 0.3)
    return p


# ------------------------------------------------------------------ seated
STOOL = dict(radius=0.145, thickness=0.032, legs=4, leg_radius=0.012)


def seated():
    feet = {s: L.foot_on_floor(s, L.side_sign(s) * 0.11, 0.23, yaw=8) for s in "LR"}
    p = P.Pose()
    L.pelvis(p, offset=(0, -0.33, -0.01), pitch=-8)
    L.spine(p, bend=10)
    place_legs(
        p,
        feet,
        yaws={"L": 8, "R": 8},
        poles={s: feet[s] + np.array([L.side_sign(s) * 0.1, 0.5, 0.6]) for s in "LR"},
    )
    L.head(p, pitch=4)
    F = P.fk(p)
    for s in "LR":
        sg = L.side_sign(s)
        knee = F[1][IDX[f"lowerleg01.{s}"]]
        hip = F[1][IDX[f"upperleg01.{s}"]]
        mid = hip + (knee - hip) * 0.62
        wrist = mid + np.array([sg * 0.005, 0.075, -0.045])
        L.arm_ik(
            p,
            s,
            wrist,
            pole=np.array([sg * 0.6, 0.1, -0.4]),
            fingers=P.unit([0.1 * sg, -0.25, 1.0]),
            palm=np.array([0.0, -1.0, 0.0]),
        )
        L.hands(p, s, 0.18, thumb=0.1)
    # the stool top meets the lowest point of the seat (the buttocks and the backs of the thighs)
    top = float(P.lowest(p, "seat")[1])
    h = top - P.FLOOR
    cz = HIP_Z_SEAT(p)
    p.props = [
        dict(
            type="cyl",
            size=[STOOL["radius"], STOOL["thickness"]],
            pos=[0, top - STOOL["thickness"] / 2, cz],
        )
    ]
    for k in range(STOOL["legs"]):
        a = np.pi / 4 + k * np.pi / 2
        r = STOOL["radius"] * 0.72
        p.props.append(
            dict(
                type="cyl",
                size=[STOOL["leg_radius"], h - STOOL["thickness"]],
                pos=[
                    float(r * np.cos(a)),
                    P.FLOOR + (h - STOOL["thickness"]) / 2,
                    cz + float(r * np.sin(a)),
                ],
            )
        )
    p.stool = dict(top=float(top), center=[0.0, float(cz)], **STOOL)
    return p


def HIP_Z_SEAT(p):
    F = P.fk(p)
    return float(
        0.5 * (F[1][IDX["upperleg01.L"]][2] + F[1][IDX["upperleg01.R"]][2]) - 0.035
    )


POSES = dict(
    a_pose=a_pose,
    t_pose=t_pose,
    relaxed=relaxed,
    weight_shift=weight_shift,
    hands_on_hips=hands_on_hips,
    arms_crossed=arms_crossed,
    ready=ready,
    reach_up=reach_up,
    crouch=crouch,
    seated=seated,
)
