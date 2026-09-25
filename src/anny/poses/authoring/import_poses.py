# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Import MakeHuman BVH poses (CC0 pose packs) onto the rig.

The pose packs use the MakeHuman default skeleton, so every bone of the rig has a joint of the same name in the file.
The files differ in their axes (MakeHuman exports Y up, MPFB2 in Blender exports Z up), in their rest skeletons
(other body shapes) and in helper joints (names starting with '__'), so the import works from world rotations:

- the axes of a file come from its rest skeleton: left from the right hip to the left hip, up from the feet to the head;
- each bone takes the rotation of its joint in the file (the files rest in the same A-pose as the rig; their other
  proportions keep to the rig's own rest shape);
- the shoulder girdle follows the rule of the pose library (poselib.girdle_for); where a file raises it further,
  a share of the difference stays as an extra raise;
- the lowest point of the body touches the floor, and the hips sit over the middle of the floor;
- a figure sitting in the air (on a chair in the original pose) gets the stool of the pose library under its seat.

    from anny.poses.authoring import import_poses as IP
    p = IP.load_pose(IP.pose_files()["standing01"])   # posing.Pose with .props and .stool when seated

The pose packs are the zip files of the legacy 3D Model experiment
(legacy/3d_model/baseline_body/*_cc0.zip); ANNY_POSE_PACKS can name another folder of packs.
Ported from the legacy 3D Model build (build/import_poses.py).
"""

import glob
import json
import os
import pathlib
import zipfile

import numpy as np
from . import bvh
from . import posing as P
from . import poselib as L
from .posing import IDX, HEADS, TAILS, PAR, NAMES, NB


def pack_dir():
    """folder of the CC0 pose packs (zip files)"""
    env = os.getenv("ANNY_POSE_PACKS")
    if env:
        return pathlib.Path(env)
    return (
        pathlib.Path(__file__).resolve().parents[4]
        / "legacy"
        / "3d_model"
        / "baseline_body"
    )


def pack_files():
    return sorted(glob.glob(str(pack_dir() / "*_cc0.zip")))


# main child of each bone of the rig: the child whose head sits at the bone's tail
_KIDS = {}
for _i, _p in enumerate(PAR):
    if _p >= 0:
        _KIDS.setdefault(_p, []).append(_i)
MAIN_CHILD = {}
for _i in range(NB):
    _c = [c for c in _KIDS.get(_i, []) if np.linalg.norm(HEADS[c] - TAILS[_i]) < 1e-3]
    if _c and np.linalg.norm(TAILS[_i] - HEADS[_i]) > 0.03:
        MAIN_CHILD[_i] = _c[0]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def swing(u, d):
    u, d = unit(u), unit(d)
    ax = np.cross(u, d)
    s, c = np.linalg.norm(ax), float(u @ d)
    if s < 1e-9:
        return np.eye(3)
    return P.qmat(P.qaxis(ax / s, np.degrees(np.arctan2(s, c))))


def file_world(B, frame=0):
    """posed world rotations and rest positions of every joint, in the file's axes"""
    R, T = B.local(frame)
    n = len(B.names)
    W = np.zeros((n, 3, 3))
    rest = np.zeros((n, 3))
    for j in range(n):
        p = B.parents[j]
        if p < 0:
            W[j] = R[j]
            rest[j] = B.offsets[j]
        else:
            W[j] = W[p] @ R[j]
            rest[j] = rest[p] + B.offsets[j]
    return W, rest


def file_axes(B, rest):
    """rows: the file's left, up and forward directions (maps file vectors to the rig's axes)"""
    index = B.index
    left = unit(rest[index["upperleg01.L"]] - rest[index["upperleg01.R"]])
    up = rest[index["head"]] - 0.5 * (rest[index["foot.L"]] + rest[index["foot.R"]])
    up = unit(up - left * (left @ up))
    fwd = np.cross(left, up)
    C = np.stack([left, up, fwd])
    assert np.linalg.det(C) > 0.99
    return C


# girdle positions of the rig for a grid of raise and forward push (to read the girdle of a file)
_GRID = None


def _girdle_grid():
    global _GRID
    if _GRID is None:
        es, ps = np.arange(-12, 61, 1.0), np.arange(-25, 36, 1.0)
        pts = np.zeros((len(es), len(ps), 3))
        for a, e in enumerate(es):
            for b, pr in enumerate(ps):
                p = P.Pose()
                L.clavicle(p, "L", elev=e, prot=pr)
                F = P.fk(p)
                pts[a, b] = F[1][IDX["upperarm01.L"]] - HEADS[IDX["upperarm01.L"]]
        _GRID = (es, ps, pts)
    return _GRID


def read_girdle(pose, s):
    """raise and forward push of the girdle in a pose, from where the shoulder joint sits relative to the chest"""
    F = P.fk(pose)
    Rc = F[0][IDX["spine01"]]
    rel = Rc.T @ (F[1][IDX[f"upperarm01.{s}"]] - F[1][IDX["spine01"]]) - (
        HEADS[IDX[f"upperarm01.{s}"]] - HEADS[IDX["spine01"]]
    )
    if s == "R":
        rel = rel * np.array([-1.0, 1.0, 1.0])
    es, ps, pts = _girdle_grid()
    d = np.linalg.norm(pts - rel, axis=2)
    a, b = np.unravel_index(np.argmin(d), d.shape)
    return float(es[a]), float(ps[b]), float(d[a, b])


def to_pose(B, frame=0, girdle=True, keep_extra=True):
    W, rest = file_world(B, frame)
    C = file_axes(B, rest)
    Wo = np.einsum("ij,njk,lk->nil", C, W, C)  # C W C^T
    index = B.index
    Wr = np.zeros((NB, 3, 3))
    for i in range(NB):  # parents come before children in the rig
        n = NAMES[i]
        if n not in index:
            Wr[i] = Wr[PAR[i]] if PAR[i] >= 0 else np.eye(3)
            continue
        # the local rotations carry over as they are: the files rest in the same A-pose as the rig, and their other
        # proportions (a different curve of the spine, other shoulders) should not bend the rig's own rest shape
        Wr[i] = Wo[index[n]]
    p = P.Pose()
    for i in range(NB):
        Rp = Wr[PAR[i]] if PAR[i] >= 0 else np.eye(3)
        p.q[i] = P.mat2q(Rp.T @ Wr[i])
    p.source_girdle = {}
    for s in "LR":
        e_f, p_f, err = read_girdle(p, s)
        p.source_girdle[s] = (e_f, p_f, err)
        if not girdle:
            continue
        # the girdle follows the arm; a file that raises it further keeps the difference
        chest = L.arm_chest(p, s)
        F = P.fk(p)
        d = F[0][IDX["spine01"]].T @ unit(
            F[1][IDX[f"lowerarm01.{s}"]] - F[1][IDX[f"upperarm01.{s}"]]
        )
        auto = L.girdle_for(*L.arm_angles(d, s))
        # a file that raises the girdle further keeps a share of the difference (its authors posed adult bodies,
        # where the same raise looks smaller)
        extra_e = EXTRA_SHARE * max(0.0, e_f - auto[0]) if keep_extra else 0.0
        extra_p = (
            EXTRA_SHARE * (p_f - auto[1] if abs(p_f) > abs(auto[1]) + 2.0 else 0.0)
            if keep_extra
            else 0.0
        )
        g = L._girdle(p, s)
        g["auto"] = auto
        g["extra"] = (extra_e, extra_p)
        L._apply_girdle(p, s)
        assert np.allclose(L.arm_chest(p, s), chest, atol=1e-6) or np.allclose(
            L.arm_chest(p, s), -chest, atol=1e-6
        )
    return p


EXTRA_SHARE = 0.3
STOOL = dict(radius=0.145, thickness=0.032, legs=4, leg_radius=0.012)
_BUTT = None


def buttocks(p, F=None):
    """lowest point of the buttocks of the preview body in pose p"""
    global _BUTT
    M = P.preview_mesh()
    if _BUTT is None:
        V = M["V"]
        _BUTT = (
            (V[:, 1] > -0.24)
            & (V[:, 1] < -0.05)
            & (V[:, 2] < -0.02)
            & (np.abs(V[:, 0]) < 0.14)
        )
    F = F if F is not None else P.fk(p)
    X = P.skin(F, M["V"][_BUTT], M["si"][_BUTT], M["sw"][_BUTT])
    return X[np.argmin(X[:, 1])]


def turn(p, deg):
    """turn the whole pose about the vertical axis through the root joint (the root's position stays)"""
    p.q[IDX["root"]] = P.qmul(P.qaxis(P.Y, deg), p.q[IDX["root"]])


def face_front(p):
    """an upright figure turned far to one side faces the front; a figure lying face up or down lies along X with
    its head to the right of the picture (the viewer looks at it from the front and the left)"""
    W = P.fk(p)[0][IDX["root"]]
    fw = W @ P.Z
    up = W @ P.Y
    if up[1] > 0.5 and abs(fw[1]) < 0.7:
        yaw = np.degrees(np.arctan2(fw[0], fw[2]))
        if abs(yaw) > 60:
            turn(p, -yaw)
    elif abs(fw[1]) > 0.7 and abs(up[1]) < 0.7:
        yaw = np.degrees(
            np.arctan2(up[0], up[2])
        )  # direction of the spine on the floor
        turn(p, 90 - yaw)  # the spine points along +X, toward the picture's right


def place(p, stool=None):
    """hips over the middle of the floor, lowest point on the floor, a stool under a figure that sits in the air
    (stool=True or False overrides the test)"""
    face_front(p)
    F = P.fk(p)
    hip = F[1][IDX["root"]]
    p.root[0] -= hip[0]
    p.root[2] -= hip[2] - HEADS[IDX["root"]][2]
    low = P.lowest(p, "all")
    p.root[1] += P.FLOOR - low[1]
    # sitting in the air (on a chair in the original pose): the buttocks 30 cm or more above the floor, both thighs
    # close to level, and one leg with its knee near hip height, its shin close to upright and its foot on the floor
    F = P.fk(p)
    b = buttocks(p, F)
    legs = []
    for s_ in "LR":
        k = F[1][IDX[f"lowerleg01.{s_}"]]
        a = F[1][IDX[f"foot.{s_}"]]
        h = F[1][IDX[f"upperleg01.{s_}"]]
        legs.append(
            dict(
                level=np.hypot(k[0] - h[0], k[2] - h[2]) > 0.3,
                planted=(-0.15 < k[1] - h[1] < 0.03)
                and unit(a - k)[1] < -0.7
                and a[1] - P.FLOOR < 0.2,
            )
        )
    sits = (
        b[1] - P.FLOOR > 0.3
        and all(leg["level"] for leg in legs)
        and any(leg["planted"] for leg in legs)
    )
    p.props = []
    if stool is not None:
        sits = stool
    if sits:
        top = float(b[1])
        h = top - P.FLOOR
        cx, cz = float(b[0]), float(b[2]) + 0.06
        p.props = [
            dict(
                type="cyl",
                size=[STOOL["radius"], STOOL["thickness"]],
                pos=[cx, top - STOOL["thickness"] / 2, cz],
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
                        cx + float(r * np.cos(a)),
                        P.FLOOR + (h - STOOL["thickness"]) / 2,
                        cz + float(r * np.sin(a)),
                    ],
                )
            )
        p.stool = dict(top=top, center=[cx, cz], **STOOL)
    return p


def read_member(entry, suffix=".bvh"):
    """text of a file of a pose pack; `entry` is (zip path, member name of the .bvh file)"""
    zip_path, member = entry
    name = member[: -len(".bvh")] + suffix
    with zipfile.ZipFile(zip_path) as z:
        if name not in z.namelist():
            return None
        return z.read(name).decode("utf-8", errors="replace")


def load_pose(entry, frame=0, girdle=True, stool=None, keep_extra=True):
    B = bvh.load(text=read_member(entry))
    p = to_pose(B, frame, girdle, keep_extra)
    return place(p, stool)


def meta(entry):
    """name, author, license and tags from the .meta file next to a pose"""
    m = dict(name=os.path.basename(entry[1])[:-4], author="", license="", tags=[])
    text = read_member(entry, ".meta")
    for line in (text or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        k, v = parts
        if k == "tag":
            m["tags"].append(v)
        elif k in ("name", "author", "license", "copyright", "description"):
            m[k] = v
    return m


def pose_files():
    """every pose of the packs: file stem -> (zip path, member name)"""
    out = {}
    for zip_path in pack_files():
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.endswith(".bvh"):
                    out[os.path.basename(name)[:-4]] = (zip_path, name)
    return out


def pack_descriptions():
    """the pack descriptions (packs/*.json) of every pack, by file stem"""
    out = {}
    for zip_path in pack_files():
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.startswith("packs/") and name.endswith(".json"):
                    try:
                        out.update(json.loads(z.read(name)))
                    except ValueError:
                        pass
    return out
