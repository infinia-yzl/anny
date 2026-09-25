# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""High-level controls for posing (poses and animation frames are built from these).

Ported from the legacy 3D Model build (build/poselib.py).

Axes: X to the figure's left, Y up, Z forward. Angles in degrees.
- spine: forward bend (+ forward), side bend (+ toward the figure's right), twist (+ turns the chest to the left)
- head: pitch (+ down), yaw (+ to the left), roll (+ toward the right shoulder)
- arms (FK): abd (raise sideways from hanging down), flex (raise forward), twist (+ turns the elbow crease forward),
  elbow (0 straight), pron (+ turns the palm down/back), wrist_flex (+ toward the palm), wrist_dev (+ toward the thumb).
  For high arms, `aim=(elev, plane)` gives the upper-arm direction instead of abd and flex: elev 0 hangs down, 90 is
  level and 180 is straight up; plane 0 is out to the side, 90 forward, -90 back and 180 across the body.
- arms (IK): target (wrist position), pole (a point the elbow points to), hand (fingers direction, palm normal) optional
- arm directions are relative to the chest (spine01). The shoulder girdle (clavicle and shoulder01) follows the arm
  by itself (girdle_for); clavicle() adds a raise or a forward push on top, for example for a shrug.
- legs (IK): ankle target, knee pole, foot yaw (+ toes out), foot pitch (+ toes up); toes stay on the floor
- pelvis: offset (x, y, z), pitch (+ tips the top forward), roll (+ drops the figure's left hip),
  yaw (+ turns to the left)
"""

import numpy as np
from . import posing as P
from .posing import X, Y, Z, IDX, HEADS, TAILS, qaxis, qmul, qinv, qpow, unit

SPINE = ["spine05", "spine04", "spine03", "spine02", "spine01"]
SPINE_W = [0.16, 0.18, 0.2, 0.22, 0.24]
NECK = ["neck01", "neck02", "neck03", "head"]
NECK_W = [0.22, 0.24, 0.24, 0.30]
HIP_C = 0.5 * (HEADS[IDX["upperleg01.L"]] + HEADS[IDX["upperleg01.R"]])
REST_ARM_ABD = {}
for s, sg in (("L", 1), ("R", -1)):
    u = unit(HEADS[IDX[f"lowerarm01.{s}"]] - HEADS[IDX[f"upperarm01.{s}"]])
    REST_ARM_ABD[s] = np.degrees(np.arctan2(abs(u[0]), -u[1]))
ANKLE_H = (
    HEADS[IDX["foot.L"]][1] - P.FLOOR
)  # ankle height above the floor with the foot flat


def side_sign(s):
    return 1 if s == "L" else -1


def arm_rest(s):
    u0 = unit(HEADS[IDX[f"lowerarm01.{s}"]] - HEADS[IDX[f"upperarm01.{s}"]])
    f0 = unit(HEADS[IDX[f"wrist.{s}"]] - HEADS[IDX[f"lowerarm01.{s}"]])
    h0 = unit(np.cross(u0, f0))
    ang = np.degrees(np.arccos(np.clip(np.dot(u0, f0), -1, 1)))
    return u0, f0, h0, ang


def hand_axes(s):
    d = unit(HEADS[IDX[f"finger3-1.{s}"]] - HEADS[IDX[f"wrist.{s}"]])
    t = unit(HEADS[IDX[f"finger2-1.{s}"]] - HEADS[IDX[f"finger5-1.{s}"]])
    n = unit(np.cross(d, t)) if s == "L" else unit(np.cross(t, d))
    t = unit(np.cross(n, d)) if s == "L" else unit(np.cross(d, n))
    return d, t, n  # fingers, thumb side, out of the palm


# ------------------------------------------------------------------ building blocks
def pelvis(pose, offset=(0, 0, 0), pitch=0, roll=0, yaw=0):
    q = qmul(qaxis(Y, yaw), qmul(qaxis(Z, -roll), qaxis(X, pitch)))
    R = P.qmat(q)
    r0 = HEADS[IDX["root"]]
    pose.q[IDX["root"]] = qmul(q, pose.q[IDX["root"]])
    pose.root = pose.root + (HIP_C + R @ (r0 - HIP_C) - r0) + np.asarray(offset, float)


def spine(pose, bend=0, side=0, twist=0, upright=0):
    """bend, side bend and twist spread over the spine; `upright` (0..1) cancels the pelvis tilt first"""
    if upright:
        # undo part of the root rotation in the lowest spine bones
        qr = pose.q[IDX["root"]]
        pose.q[IDX["spine05"]] = qmul(qpow(qinv(qr), upright), pose.q[IDX["spine05"]])
    pose.spread(SPINE, X, bend, SPINE_W)
    pose.spread(SPINE, Z, -side, SPINE_W)
    pose.spread(SPINE, Y, twist, SPINE_W)


def head(pose, pitch=0, yaw=0, roll=0):
    pose.spread(NECK, X, pitch, NECK_W)
    pose.spread(NECK, Y, yaw, NECK_W)
    pose.spread(NECK, Z, -roll, NECK_W)


# ------------------------------------------------------------------ shoulder girdle
# Scapulohumeral rhythm: the girdle stays still for the first 20 degrees of arm elevation, then takes a growing share
# of the motion (34 degrees with the arm straight up). It comes forward when the arm reaches forward or across the body
# and goes back a little when the arm goes behind the body. The values are relative to the rest pose (the arms at about
# 42 degrees), so the girdle is at rest there and drops a little when the arms hang down. The corrective shapes are
# trained with the same rule.
GIRDLE = dict(start=20.0, soft=20.0, gain=0.26, forward=10.0, across=10.0, back=6.0)


def arm_angles(d, s):
    """elevation and plane (degrees, as for `aim`) of an upper-arm direction d given in the chest frame"""
    x = d[0] * side_sign(s)
    return float(np.degrees(np.arccos(np.clip(-d[1], -1, 1)))), float(
        np.degrees(np.arctan2(d[2], x))
    )


REST_AIM = arm_angles(
    arm_rest("L")[0], "L"
)  # the rest pose's arm direction (about 42 degrees out to the side)


def aim_dir(elev, plane, s):
    """unit direction in the chest frame for an elevation and a plane"""
    e, p = np.radians(elev), np.radians(plane)
    return np.array(
        [side_sign(s) * np.sin(e) * np.cos(p), -np.cos(e), np.sin(e) * np.sin(p)]
    )


def world_aim(pose, s, d_world):
    """(elev, plane) relative to the chest for an upper-arm direction given in world space (for the current spine)"""
    Rc = P.fk(pose)[0][IDX["spine01"]]
    return arm_angles(Rc.T @ unit(d_world), s)


def _girdle_raw(elev, plane):
    g = GIRDLE
    x = max(0.0, elev - g["start"])
    up = g["gain"] * x * x / (x + g["soft"])
    e, p = np.radians(elev), np.radians(plane)
    fwd, out = np.sin(e) * np.sin(p), np.sin(e) * np.cos(p)
    prot = (
        g["forward"] * max(fwd, 0.0)
        + g["across"] * max(-out, 0.0)
        - g["back"] * max(-fwd, 0.0)
    )
    return up, prot


def girdle_for(elev, plane):
    """girdle raise and forward push (degrees, as for clavicle()) for an arm elevation and plane, relative to rest"""
    up, prot = _girdle_raw(elev, plane)
    up0, prot0 = _girdle_raw(*REST_AIM)
    return float(up - up0), float(prot - prot0)


def _girdle(pose, s):
    if not hasattr(pose, "girdle"):
        pose.girdle = {}
    return pose.girdle.setdefault(s, dict(extra=(0.0, 0.0), auto=(0.0, 0.0)))


def _apply_girdle(pose, s):
    """set the clavicle and shoulder01 from the automatic and the extra girdle motion; the upper arm keeps its
    rotation relative to the chest"""
    sg = side_sign(s)
    g = _girdle(pose, s)
    elev = g["extra"][0] + g["auto"][0]
    prot = g["extra"][1] + g["auto"][1]
    ic, ish, ia = IDX[f"clavicle.{s}"], IDX[f"shoulder01.{s}"], IDX[f"upperarm01.{s}"]
    arm_chest = qmul(qmul(pose.q[ic], pose.q[ish]), pose.q[ia])
    pose.q[ic] = qmul(qaxis(Y, -sg * prot), qaxis(Z, sg * elev * 0.6))
    pose.q[ish] = qaxis(Z, sg * elev * 0.4)
    pose.q[ia] = qmul(qinv(qmul(pose.q[ic], pose.q[ish])), arm_chest)


def clavicle(pose, s, elev=0, prot=0):
    """raise (elev) or push forward (prot) the shoulder girdle, on top of the motion that follows the arm"""
    g = _girdle(pose, s)
    g["extra"] = (g["extra"][0] + elev, g["extra"][1] + prot)
    _apply_girdle(pose, s)


def arm_chest(pose, s):
    """rotation of the upper arm relative to the chest"""
    return qmul(
        qmul(pose.q[IDX[f"clavicle.{s}"]], pose.q[IDX[f"shoulder01.{s}"]]),
        pose.q[IDX[f"upperarm01.{s}"]],
    )


def follow(pose, s, d=None, auto=True):
    """let the girdle follow the arm: d is the upper-arm direction in the chest frame (measured when None)"""
    if d is None:
        F = P.fk(pose)
        d = F[0][IDX["spine01"]].T @ unit(
            F[1][IDX[f"lowerarm01.{s}"]] - F[1][IDX[f"upperarm01.{s}"]]
        )
    _girdle(pose, s)["auto"] = girdle_for(*arm_angles(d, s)) if auto else (0.0, 0.0)
    _apply_girdle(pose, s)


def set_arm(pose, s, q, auto=True):
    """give the upper arm the rotation q relative to the chest (from its rest pose); the girdle follows"""
    ia = IDX[f"upperarm01.{s}"]
    u0 = arm_rest(s)[0]
    ic, ish = IDX[f"clavicle.{s}"], IDX[f"shoulder01.{s}"]
    pose.q[ia] = qmul(qinv(qmul(pose.q[ic], pose.q[ish])), q)
    follow(pose, s, P.qmat(q) @ u0, auto)


def swing_q(u, d):
    """shortest rotation taking direction u to direction d"""
    u, d = unit(u), unit(d)
    ax = np.cross(u, d)
    sn, c = np.linalg.norm(ax), float(u @ d)
    if sn < 1e-9:
        return np.array([0, 0, 0, 1.0])
    return qaxis(ax / sn, np.degrees(np.arctan2(sn, c)))


def arm_fk(
    pose,
    s,
    abd=0,
    flex=0,
    twist=0,
    elbow=10,
    pron=0,
    wrist_flex=0,
    wrist_dev=0,
    aim=None,
    auto=True,
):
    sg = side_sign(s)
    u0, f0, h0, a0 = arm_rest(s)
    if aim is None:
        # upper arm: twist about its rest axis, bring it down to hanging, then abduct and flex
        q = qaxis(u0, sg * twist)
        q = qmul(qaxis(Z, sg * (abd - REST_ARM_ABD[s])), q)
        q = qmul(qaxis(X, -flex), q)
    else:
        # the shortest turn from the rest direction to the aim, then the twist about the arm
        q = qmul(swing_q(u0, aim_dir(*aim, s)), qaxis(u0, sg * twist))
    set_arm(pose, s, qmul(q, arm_chest(pose, s)), auto)
    # elbow hinge (the rest pose is already bent by a0)
    pose.rot(f"lowerarm01.{s}", h0, elbow - a0)
    # forearm turn shared by the forearm and the wrist
    pose.rot(f"lowerarm02.{s}", f0, -sg * pron * 0.6)
    d, t, n = hand_axes(s)
    wq = qaxis(f0, -sg * pron * 0.4)
    wq = qmul(qaxis(np.cross(d, n), wrist_flex), wq)
    wq = qmul(qaxis(n, sg * wrist_dev), wq)
    pose.q[IDX[f"wrist.{s}"]] = qmul(wq, pose.q[IDX[f"wrist.{s}"]])


def hand_world(pose, s, fingers, palm, forearm_share=0.6):
    """turn the hand so its fingers point along `fingers` and its palm faces `palm` (world directions);
    the turn about the forearm axis is shared with the forearm"""
    d0, t0, n0 = hand_axes(s)
    F0 = np.stack([d0, n0, np.cross(d0, n0)], 1)
    d1 = unit(fingers)
    n1 = unit(np.asarray(palm, float) - d1 * np.dot(palm, d1))
    F1 = np.stack([d1, n1, np.cross(d1, n1)], 1)
    Rw_target = F1 @ F0.T
    Rw, Pw = P.fk(pose)
    i2, iw = IDX[f"lowerarm02.{s}"], IDX[f"wrist.{s}"]
    Rp = Rw[P.PAR[i2]]  # forearm (lowerarm01) world rotation
    ql = P.mat2q(
        Rp.T @ Rw_target
    )  # wrist rotation relative to lowerarm01 (lowerarm02 at rest)
    f0 = unit(HEADS[iw] - HEADS[IDX[f"lowerarm01.{s}"]])
    sw, tw = P.swing_twist(ql, f0)
    t2 = qpow(tw, forearm_share)
    pose.q[i2] = t2
    pose.q[iw] = qmul(qinv(t2), ql)


def arm_ik(pose, s, target, pole, fingers=None, palm=None, twist_split=0.5, auto=True):
    """put the wrist at `target`; the girdle follows the arm, which moves the shoulder, so the solve
    repeats a few times"""
    for _ in range(4):
        P.ik(pose, "arm." + s, target, pole, twist_split)
        if not auto:
            break
        before = _girdle(pose, s)["auto"]
        follow(pose, s)
        after = _girdle(pose, s)["auto"]
        if abs(after[0] - before[0]) + abs(after[1] - before[1]) < 0.05:
            break
    P.ik(pose, "arm." + s, target, pole, twist_split)
    if fingers is not None:
        hand_world(pose, s, fingers, palm)


def leg(pose, s, ankle, pole=None, yaw=0, pitch=0, toe_bend=None):
    """put the ankle at `ankle` (world); the foot turns by yaw (toes out) and pitch (toes up, negative
    lifts the heel)"""
    sg = side_sign(s)
    a = np.asarray(ankle, float)
    if pole is None:
        pole = a + np.array([sg * 0.05, 0.3, 0.6])
    P.ik(pose, "leg." + s, a, pole, 0.5)
    q = qmul(qaxis(Y, -sg * yaw), qaxis(X, -pitch))
    F = P.fk(pose)
    P.set_world_rot(pose, f"foot.{s}", P.qmat(q), F)
    # toes: keep them flat on the floor when the heel is up
    tb = (-pitch if pitch < 0 else 0) if toe_bend is None else toe_bend
    for k in range(1, 6):
        pose.rot(f"toe{k}-1.{s}", X, -tb)


def foot_on_floor(s, x, z, yaw=0, pitch=0, lift=0.0):
    """ankle position for a foot at (x, z) on the floor; pitch < 0 raises the heel and pivots about the
    ball of the foot"""
    sg = side_sign(s)
    rest_ankle = HEADS[IDX[f"foot.{s}"]]
    ball = HEADS[IDX[f"toe3-1.{s}"]].copy()
    ball[1] = P.FLOOR
    heel = np.array([rest_ankle[0], P.FLOOR, rest_ankle[2] - 0.055])
    q = qmul(qaxis(Y, -sg * yaw), qaxis(X, -pitch))
    R = P.qmat(q)
    piv = ball if pitch < 0 else heel
    a_rel = R @ (rest_ankle - piv)
    piv_new = np.array(
        [x + (piv[0] - rest_ankle[0]), P.FLOOR + lift, z + (piv[2] - rest_ankle[2])]
    )
    return piv_new + a_rel


def hands(pose, s, amount=0.25, thumb=None, spread=0.0, cascade=0.35):
    """curl the fingers of one hand: 0 flat, 1 a fist; the ring and little fingers curl a little more (cascade)"""
    d, t, n = hand_axes(s)
    for f in range(2, 6):
        k_c = 1 + cascade * (f - 2) / 3
        for k, base in ((1, 50), (2, 85), (3, 55)):
            b = f"finger{f}-{k}.{s}"
            dk = unit(TAILS[IDX[b]] - HEADS[IDX[b]])
            ax = np.cross(dk, n)
            if np.linalg.norm(ax) < 1e-6:
                continue
            pose.rot(b, unit(ax), min(amount * base * k_c, base * 1.15))
        if spread:
            pose.rot(f"finger{f}-1.{s}", n, side_sign(s) * (f - 3.3) * spread)
    th = amount * 0.6 if thumb is None else thumb
    toward = unit(
        HEADS[IDX[f"finger5-1.{s}"]] + n * 0.01 - HEADS[IDX[f"finger1-2.{s}"]]
    )
    for k, base in ((1, 20), (2, 25), (3, 40)):
        b = f"finger1-{k}.{s}"
        dk = unit(TAILS[IDX[b]] - HEADS[IDX[b]])
        ax = np.cross(dk, toward)
        if np.linalg.norm(ax) < 1e-6:
            continue
        pose.rot(b, unit(ax), th * base)


def stance(pose, width=0.10, yaw=7, zoff=0.0, which="LR"):
    """both feet flat on the floor, ankles `width` from the middle"""
    for s in which:
        sg = side_sign(s)
        a = foot_on_floor(s, sg * width, HEADS[IDX[f"foot.{s}"]][2] + zoff, yaw=yaw)
        leg(pose, s, a, yaw=yaw)
