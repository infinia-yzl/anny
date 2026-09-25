# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Posing tools: quaternions, forward kinematics, skinning, two-bone IK and grounding on the
authoring rig (``rig.authoring_rig``). Ported from the legacy 3D Model build (build/posing.py).

Conventions (shared with the viewer):
- Every bone rests with the world axes at its head. A pose gives each bone a local rotation q (x, y, z, w) relative
  to its parent, and the root bone an offset. At rest the local axes are the world axes: X points to the figure's
  left, Y up and Z forward (the side the face looks to).
- World transform of bone i: R_i = R_parent(i) * q_i, P_i = P_parent(i) + R_parent(i) (head_i - head_parent(i)).
- Skinning: x' = sum_k w_k (R_k (x - head_k) + P_k).
Units are metres in the legacy frame of the authoring rig.
"""

import numpy as np

from .rig import authoring_rig

RIG = authoring_rig()
NAMES = list(RIG.names)
IDX = {n: i for i, n in enumerate(NAMES)}
PAR = RIG.parents.astype(int)
HEADS = RIG.heads.astype(np.float64)
TAILS = RIG.tails.astype(np.float64)
NB = len(NAMES)
FLOOR = RIG.floor
X, Y, Z = np.eye(3)


# ------------------------------------------------------------------ quaternions (x, y, z, w)
def qaxis(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    h = np.radians(deg) / 2
    return np.array([*(a * np.sin(h)), np.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qmat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat2q(M):
    t = np.trace(M)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [
            (M[2, 1] - M[1, 2]) / s,
            (M[0, 2] - M[2, 0]) / s,
            (M[1, 0] - M[0, 1]) / s,
            0.25 * s,
        ]
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        q = [
            0.25 * s,
            (M[0, 1] + M[1, 0]) / s,
            (M[0, 2] + M[2, 0]) / s,
            (M[2, 1] - M[1, 2]) / s,
        ]
    elif M[1, 1] > M[2, 2]:
        s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        q = [
            (M[0, 1] + M[1, 0]) / s,
            0.25 * s,
            (M[1, 2] + M[2, 1]) / s,
            (M[0, 2] - M[2, 0]) / s,
        ]
    else:
        s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        q = [
            (M[0, 2] + M[2, 0]) / s,
            (M[1, 2] + M[2, 1]) / s,
            0.25 * s,
            (M[1, 0] - M[0, 1]) / s,
        ]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def qslerp(a, b, t):
    d = np.dot(a, b)
    if d < 0:
        b = -b
        d = -d
    if d > 0.9995:
        r = a + (b - a) * t
        return r / np.linalg.norm(r)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)


def qpow(q, t):
    return qslerp(np.array([0, 0, 0, 1.0]), q, t)


def swing_twist(q, axis):
    """q = swing * twist, twist about `axis` (unit, in the frame q acts in)"""
    p = np.dot(q[:3], axis) * axis
    tw = np.array([*p, q[3]])
    n = np.linalg.norm(tw)
    tw = np.array([0, 0, 0, 1.0]) if n < 1e-9 else tw / n
    return qmul(q, qinv(tw)), tw


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ poses
class Pose:
    def __init__(self):
        self.q = np.tile([0.0, 0.0, 0.0, 1.0], (NB, 1))
        self.root = np.zeros(3)

    def copy(self):
        p = Pose()
        p.q = self.q.copy()
        p.root = self.root.copy()
        if hasattr(self, "girdle"):
            p.girdle = {s: dict(v) for s, v in self.girdle.items()}
        return p

    def rot(self, bone, axis, deg):
        """turn a bone about an axis of its rest-aligned local frame (applied after its current local rotation)"""
        i = IDX[bone]
        self.q[i] = qmul(qaxis(axis, deg), self.q[i])
        return self

    def spread(self, bones, axis, deg, weights=None):
        """share one turn over a chain of bones (for example the spine)"""
        w = np.ones(len(bones)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        for b, f in zip(bones, w):
            self.rot(b, axis, deg * f)
        return self


def mirror_name(n):
    return (
        n[:-2] + (".R" if n.endswith(".L") else ".L") if n.endswith((".L", ".R")) else n
    )


def fk(pose):
    Rw = np.zeros((NB, 3, 3))
    Pw = np.zeros((NB, 3))
    for i in range(NB):
        Rl = qmat(pose.q[i])
        p = PAR[i]
        if p < 0:
            Rw[i] = Rl
            Pw[i] = HEADS[i] + pose.root
        else:
            Rw[i] = Rw[p] @ Rl
            Pw[i] = Pw[p] + Rw[p] @ (HEADS[i] - HEADS[p])
    return Rw, Pw


def point(pose_fk, bone, x):
    """world position of a rest point x carried by `bone`"""
    Rw, Pw = pose_fk
    i = IDX[bone]
    return Rw[i] @ (np.asarray(x) - HEADS[i]) + Pw[i]


def skin(pose_fk, V, si, sw):
    Rw, Pw = pose_fk
    out = np.zeros_like(V)
    for k in range(si.shape[1]):
        b = si[:, k]
        loc = V - HEADS[b]
        out += sw[:, k : k + 1] * (np.einsum("nij,nj->ni", Rw[b], loc) + Pw[b])
    return out


def set_world_rot(pose, bone, Rworld, pose_fk=None):
    """give a bone a world rotation (its parent's pose stays as it is)"""
    Rw, _ = pose_fk if pose_fk is not None else fk(pose)
    p = PAR[IDX[bone]]
    Rp = Rw[p] if p >= 0 else np.eye(3)
    pose.q[IDX[bone]] = mat2q(Rp.T @ Rworld)


def frame(d, h):
    """right-handed frame with columns (d, f, h): d a direction, h the hinge axis, f = h x d"""
    d = unit(d)
    h = unit(h - d * np.dot(h, d))
    f = np.cross(h, d)
    return np.stack([d, f, h], 1)


# rest limb geometry for the IK solver
def _limb(upper, lower, end, rest_bend_dir):
    a, b, c = HEADS[IDX[upper]], HEADS[IDX[lower]], HEADS[IDX[end]]
    t0 = unit(c - a)
    n0 = b - a - t0 * np.dot(b - a, t0)
    n0 = (
        unit(n0)
        if np.linalg.norm(n0) > 1e-4
        else unit(rest_bend_dir - t0 * np.dot(rest_bend_dir, t0))
    )
    h0 = unit(np.cross(t0, n0))
    return dict(
        a=a,
        b=b,
        c=c,
        la=np.linalg.norm(b - a),
        lb=np.linalg.norm(c - b),
        F0u=frame(b - a, h0),
        F0l=frame(c - b, h0),
    )


LIMBS = {}
for s, sx in (("L", 1), ("R", -1)):
    LIMBS["leg." + s] = dict(
        bones=(
            f"upperleg01.{s}",
            f"upperleg02.{s}",
            f"lowerleg01.{s}",
            f"lowerleg02.{s}",
            f"foot.{s}",
        ),
        **_limb(f"upperleg01.{s}", f"lowerleg01.{s}", f"foot.{s}", Z),
    )
    LIMBS["arm." + s] = dict(
        bones=(
            f"upperarm01.{s}",
            f"upperarm02.{s}",
            f"lowerarm01.{s}",
            f"lowerarm02.{s}",
            f"wrist.{s}",
        ),
        **_limb(f"upperarm01.{s}", f"lowerarm01.{s}", f"wrist.{s}", -Z),
    )


def ik(pose, limb, target, pole, twist_split=0.5):
    """two-bone IK: put the end joint (ankle or wrist) at `target`, with the knee or elbow toward
    `pole` (a world point).
    The upper and lower segments turn as hinges about one axis; the second bone of each segment shares the twist."""
    L = LIMBS[limb]
    u1, u2, l1, l2, end = L["bones"]
    F = fk(pose)
    Rw, Pw = F
    a = Pw[IDX[u1]]
    d = np.asarray(target, float) - a
    dist = np.clip(
        np.linalg.norm(d), abs(L["la"] - L["lb"]) + 1e-4, L["la"] + L["lb"] - 1e-4
    )
    t = unit(d)
    n = np.asarray(pole, float) - a
    n = unit(n - t * np.dot(n, t))
    cosA = (L["la"] ** 2 + dist**2 - L["lb"] ** 2) / (2 * L["la"] * dist)
    knee = a + L["la"] * (cosA * t + np.sqrt(max(0.0, 1 - cosA**2)) * n)
    endp = a + t * dist
    h = unit(np.cross(t, n))
    Ru = frame(knee - a, h) @ L["F0u"].T
    Rl = frame(endp - knee, h) @ L["F0l"].T
    # upper segment: its world rotation split between the two bones as swing + shared twist about the bone axis
    p = PAR[IDX[u1]]
    Rp = Rw[p]
    qloc = mat2q(Rp.T @ Ru)
    axis = (
        unit(HEADS[IDX[u2]] - HEADS[IDX[u1]])
        if np.linalg.norm(HEADS[IDX[u2]] - HEADS[IDX[u1]]) > 1e-6
        else unit(L["b"] - L["a"])
    )
    sw, tw = swing_twist(qloc, axis)
    pose.q[IDX[u1]] = qmul(sw, qpow(tw, twist_split))
    pose.q[IDX[u2]] = qpow(tw, 1 - twist_split)
    # lower segment
    Rw_u2 = Rp @ qmat(pose.q[IDX[u1]]) @ qmat(pose.q[IDX[u2]])
    pose.q[IDX[l1]] = mat2q(Rw_u2.T @ Rl)
    pose.q[IDX[l2]] = np.array([0, 0, 0, 1.0])
    return knee


# ------------------------------------------------------------------ hands
def _hand_frame(s):
    w = HEADS[IDX[f"wrist.{s}"]]
    tip = HEADS[IDX[f"finger3-1.{s}"]]
    d = unit(tip - w)
    across = unit(HEADS[IDX[f"finger5-1.{s}"]] - HEADS[IDX[f"finger2-1.{s}"]])
    palm = unit(np.cross(across, d))
    # the palm faces the thumb side's opposite... pick the sign that points from the back of the hand to the palm
    return d, across, palm


def curl(pose, side, amount, thumb=None, spread=0.0):
    """close the fingers of one hand: amount 0 (flat) to 1 (fist); thumb 0..1 (defaults to half the amount)"""
    d, across, palm = _hand_frame(side)
    sgn = 1 if side == "L" else -1
    # flexion turns each phalanx about the across axis, toward the palm
    for f in range(2, 6):
        base = [0.55, 0.62, 0.66, 0.72][f - 2]
        for k, share in ((1, 0.85), (2, 1.0), (3, 0.7)):
            pose.rot(
                f"finger{f}-{k}.{side}",
                across,
                -sgn
                * 90
                * amount
                * share
                * (base if k == 1 else 1.0)
                * PALM_SIGN[side],
            )
        if spread:
            pose.rot(f"finger{f}-1.{side}", palm, (f - 3.5) * spread * sgn)
    th = amount * 0.5 if thumb is None else thumb
    tdir = unit(HEADS[IDX[f"finger1-2.{side}"]] - HEADS[IDX[f"finger1-1.{side}"]])
    tax = unit(np.cross(tdir, palm))
    for k, share in ((1, 0.5), (2, 0.8), (3, 0.9)):
        pose.rot(f"finger1-{k}.{side}", tax, -sgn * 60 * th * share * PALM_SIGN[side])
    return pose


PALM_SIGN = {"L": 1.0, "R": 1.0}  # tuned in the previews


# ------------------------------------------------------------------ preview mesh and grounding
def preview_mesh():
    """level-1 body with skin weights, and the sole points of the body (for contact and ground checks)"""
    return RIG.preview


def lowest(pose, part="sole"):
    """lowest point of the soles ('sole') or of the whole preview body ('all') in this pose"""
    M = preview_mesh()
    F = fk(pose)
    if part == "sole":
        P = skin(F, M["sole_V"], M["sole_si"], M["sole_sw"])
    elif part == "seat":
        V = M["V"]
        s = (V[:, 1] > -0.32) & (V[:, 1] < -0.04) & (np.abs(V[:, 0]) < 0.16)
        P = skin(F, V[s], M["si"][s], M["sw"][s])
    else:
        P = skin(F, M["V"], M["si"], M["sw"])
    return P[np.argmin(P[:, 1])]


def ground(pose, floor=FLOOR):
    """move the root up or down so that the lowest sole point touches the floor"""
    pose.root[1] += floor - lowest(pose)[1]
    return pose
