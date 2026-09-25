# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Build the pose library of anny (``src/anny/data/poses``).

The poses and clips are built on the authoring rig (anny's default body in the legacy frame)
and stored in a form that fits every body of anny:

- one rotation per bone and frame, as a quaternion (x, y, z, w) in anny's axes, with the
  convention of the legacy viewer: at rest every bone has the world axes, and a rotation is
  relative to the parent bone;
- the offset of the root bone, in anny's axes, divided by the height of the root above the
  floor, so that it scales with the body;
- labels, groups, credits and the stool of the seated poses.

``anny.poses`` reads this data and turns it into pose parameters of an anny model.

    python -m anny.poses.authoring.build_library

Ported from the legacy 3D Model build (build/export_motion.py).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file

from . import anims as A
from . import import_poses as IP
from . import posing as P
from . import poses as PS
from . import selection as MH
from .rig import ANNY_TO_LEGACY

POSE_LABELS = dict(
    a_pose="A-pose",
    t_pose="T-pose",
    relaxed="Relaxed",
    weight_shift="Weight on one leg",
    hands_on_hips="Hands on hips",
    arms_crossed="Arms crossed",
    ready="Ready stance",
    reach_up="Reaching up",
    crouch="Crouch",
    seated="Seated",
)
# the library's own poses first in each group, then the poses from the MakeHuman community packs
POSE_GROUPS = [
    ("Reference", ["a_pose", "t_pose"]),
    ("Standing", ["relaxed", "weight_shift", "hands_on_hips", "arms_crossed"]),
    ("Gestures", []),
    ("Action", ["ready", "reach_up", "crouch"]),
    ("Martial arts", []),
    ("Sports and fitness", []),
    ("Sitting", ["seated"]),
    ("On the floor", []),
]
CLIP_LABELS = dict(
    idle="Idle",
    walk="Walk",
    run="Run",
    wave="Wave",
    nod="Nod",
    shrug="Shrug",
    jump="Jump",
)

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "poses"


def seat_vertices() -> list[int]:
    """anny vertex indices of the buttocks, whose lowest point carries the stool"""
    coarse = P.preview_mesh()["coarse"]
    V = coarse["V"]
    mask = np.zeros(len(V), bool)
    mask[coarse["used"]] = True
    mask &= (V[:, 1] > -0.24) & (V[:, 1] < -0.05) & (V[:, 2] < -0.02)
    mask &= np.abs(V[:, 0]) < 0.14
    return np.nonzero(mask)[0].tolist()


def to_anny(pose):
    """rotations (B, 4) and normalised root offset (3,) of a legacy-frame pose in anny's axes"""
    M = ANNY_TO_LEGACY
    R = np.stack([P.qmat(q) for q in pose.q])
    R = np.einsum("ji,njk,kl->nil", M, R, M)  # M^T R M
    q = np.stack([P.mat2q(r) for r in R])
    root = (M.T @ pose.root) / P.RIG.scale / P.RIG.hip_height
    return q, root


def build():
    for key, _stem, label, group in MH.SELECTION:
        POSE_LABELS[key] = label
        next(g for g in POSE_GROUPS if g[0] == group)[1].append(key)
    credits = {
        c["key"]: ("MakeHuman" if c["author"] == "makehuman_system" else c["author"])
        for c in MH.credits()
    }
    Q, ROOT, entries = [], [], []
    hip = (
        P.RIG.hip_height * P.RIG.scale
    )  # legacy-frame metres per unit of normalised offset

    def push(name, label, kind, frames, duration, extra=None):
        start = len(Q)
        prev = None
        for f in frames:
            q, root = to_anny(f)
            if prev is not None:
                flip = (q * prev).sum(1) < 0
                q[flip] *= -1
            Q.append(q)
            ROOT.append(root)
            prev = q
        entry = dict(
            name=name,
            label=label,
            kind=kind,
            start=start,
            count=len(frames),
            duration=float(duration),
            fps=A.FPS,
        )
        entry.update(extra or {})
        entries.append(entry)

    for group, names in POSE_GROUPS:
        for n in names:
            p = PS.POSES[n]() if n in PS.POSES else MH.build(n)
            extra = dict(group=group)
            if n in credits:
                extra["credit"] = credits[n]
            if hasattr(p, "stool"):
                s = p.stool
                extra["stool"] = {
                    k: float(s[k]) / hip for k in ("radius", "thickness", "leg_radius")
                } | dict(legs=int(s["legs"]))
            push(n, POSE_LABELS[n], "pose", [p], 0.0, extra)
    for n in A.CLIPS:
        frames, T = A.sample(n)
        push(n, CLIP_LABELS[n], "loop", frames, T)
        print(f"{n:>6}: {len(frames)} frames")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "rotations": torch.tensor(np.array(Q), dtype=torch.float32).contiguous(),
            "root_offsets": torch.tensor(np.array(ROOT), dtype=torch.float32),
        },
        str(DATA_DIR / "library.safetensors"),
    )
    meta = dict(
        format="anny-pose-library@1",
        description=(
            "Poses and clips for anny. rotations[f, b] is the rotation (x, y, z, w) of bone b "
            "relative to its parent, in anny's axes, with every bone at rest along the world "
            "axes. root_offsets[f] is the offset of the root bone divided by the height of the "
            "root above the floor at rest."
        ),
        bones=list(P.NAMES),
        seat_vertices=seat_vertices(),
        stool_forward=0.06 / hip,
        entries=entries,
        credits_note=(
            "Poses with a credit come from the MakeHuman community pose packs, which are "
            "published under CC0."
        ),
        pose_packs=[pathlib.Path(f).name for f in IP.pack_files()],
    )
    with open(DATA_DIR / "library.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("frames", len(Q), "bones", len(P.NAMES), "->", DATA_DIR)


if __name__ == "__main__":
    build()
