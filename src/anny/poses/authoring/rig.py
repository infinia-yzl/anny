# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The authoring rig: anny's default body in the frame of the legacy pose library.

The pose library of the legacy 3D Model experiment was written in its own frame: metres, Y up,
X toward the figure's left and Z forward, with the left eye at a fixed height and the floor at
a fixed level. Many poses place hands and feet at absolute positions in that frame. The
authoring rig puts anny's default body (every phenotype at 0.5) into the same frame, scaled
uniformly so that the eye stands at the same height above the floor. Rotations do not change
under a uniform scale, so the poses written in this frame carry over to anny as they are.

``authoring_rig()`` builds the rig once per process.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import torch

# The legacy frame: the centre of the left eye sits at this height and depth, and the floor
# at this level (metres).
EYE_HEIGHT = 0.5116
EYE_DEPTH = 0.1169
FLOOR = -0.8403

# Maps anny coordinates (Z up, the figure faces -Y) to the legacy frame (Y up, the figure
# faces +Z): legacy = ANNY_TO_LEGACY @ anny.
ANNY_TO_LEGACY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

# Vertices of anny's body surface in the MakeHuman base mesh (eyes and tongue follow).
BODY_BASE_VERTEX_COUNT = 13380

PREVIEW_WEIGHTS = 8


@dataclasses.dataclass
class AuthoringRig:
    names: list[str]
    parents: np.ndarray  # (B,) parent index, -1 for the root
    heads: np.ndarray  # (B, 3) rest heads in the legacy frame
    tails: np.ndarray  # (B, 3) rest tails in the legacy frame
    floor: float
    scale: float  # legacy = scale * ANNY_TO_LEGACY @ anny + offset
    offset: np.ndarray
    preview: dict  # light preview body: V, T, si, sw, sole_V, sole_si, sole_sw, coarse
    hip_height: float  # root head above the floor on anny's default body (anny metres)

    def to_legacy(self, points_anny):
        return self.scale * np.asarray(points_anny) @ ANNY_TO_LEGACY.T + self.offset

    def to_anny(self, points_legacy):
        return ((np.asarray(points_legacy) - self.offset) / self.scale) @ ANNY_TO_LEGACY


def body_faces_mask(model) -> np.ndarray:
    """Faces of the body surface of an anny model built from the MakeHuman base mesh."""
    base_index = model.base_mesh_vertex_indices.detach().cpu().numpy()
    faces = model.faces.detach().cpu().numpy()
    return (base_index[faces] < BODY_BASE_VERTEX_COUNT).all(1)


def top_weights(W: np.ndarray, k: int = PREVIEW_WEIGHTS):
    """The k largest weights of each row, normalised, sorted in decreasing order."""
    k = min(k, W.shape[1])
    idx = np.argpartition(-W, k - 1, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(W, idx, 1), 0, None)
    s = w.sum(1, keepdims=True)
    w = np.where(s > 0, w / np.maximum(s, 1e-12), np.eye(k)[0])
    order = np.argsort(-w, 1)
    return np.take_along_axis(idx, order, 1), np.take_along_axis(w, order, 1)


def dense_skinning_weights(model) -> np.ndarray:
    indices = model.vertex_bone_indices.detach().cpu().numpy()
    weights = model.vertex_bone_weights.detach().cpu().numpy().astype(np.float64)
    W = np.zeros((indices.shape[0], len(model.bone_labels)))
    np.add.at(W, (np.arange(indices.shape[0])[:, None], indices), weights)
    return W


def bone_tails(heads: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """
    Tails from the rig structure: the head of the only child, the mean of the heads of several
    children, and for a bone without children its own length continued from its parent.
    """
    children = {}
    for i, p in enumerate(parents):
        if p >= 0:
            children.setdefault(int(p), []).append(i)
    tails = heads.copy()
    for i in range(len(heads)):
        kids = children.get(i, [])
        if kids:
            tails[i] = heads[kids].mean(0)
        elif parents[i] >= 0:
            tails[i] = heads[i] + 0.7 * (heads[i] - heads[parents[i]])
    return tails


@functools.lru_cache(maxsize=1)
def authoring_rig() -> AuthoringRig:
    import anny
    from anny.utils.subdivision import catmull_clark

    model = anny.Anny(topology="anny-quads").to(dtype=torch.float64)
    with torch.no_grad():
        output = model(phenotype_kwargs={})
    rest = output["rest_vertices"][0].numpy()
    heads_anny = output["rest_bone_heads"][0].numpy()
    names = list(model.bone_labels)
    parents = np.array([int(p) for p in model.bone_parents])

    faces = model.faces.numpy()[body_faces_mask(model)]
    used = np.unique(faces)
    floor_anny = rest[used, 2].min()
    eye = heads_anny[names.index("eye.L")]
    scale = (EYE_HEIGHT - FLOOR) / (eye[2] - floor_anny)
    # the middle plane of the body stays at x = 0
    offset = np.array([0.0, EYE_HEIGHT, EYE_DEPTH]) - scale * ANNY_TO_LEGACY @ eye
    offset[0] = 0.0
    heads = scale * heads_anny @ ANNY_TO_LEGACY.T + offset

    # preview body: one level of subdivision with the skinning weights
    level = catmull_clark(len(rest), faces)
    keep = np.unique(level.quads)
    remap = -np.ones(level.operator.shape[0], np.int64)
    remap[keep] = np.arange(len(keep))
    op = level.operator.select_rows(keep)
    V1 = scale * op.apply(rest) @ ANNY_TO_LEGACY.T + offset
    si, sw = top_weights(op.apply(dense_skinning_weights(model)))
    quads = remap[level.quads]
    T = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    sole = np.nonzero(V1[:, 1] < V1[:, 1].min() + 0.012)[0]
    # the coarse body (anny's own vertices), used to find anny vertex indices of regions
    coarse_si, coarse_sw = top_weights(dense_skinning_weights(model))
    preview = dict(
        V=V1,
        T=T,
        si=si,
        sw=sw,
        sole_V=V1[sole],
        sole_si=si[sole],
        sole_sw=sw[sole],
        coarse=dict(
            V=scale * rest @ ANNY_TO_LEGACY.T + offset,
            used=used,
            si=coarse_si,
            sw=coarse_sw,
        ),
    )
    return AuthoringRig(
        names=names,
        parents=parents,
        heads=heads,
        tails=bone_tails(heads, parents),
        floor=FLOOR,
        scale=float(scale),
        offset=offset,
        preview=preview,
        hip_height=float(heads_anny[0, 2] - floor_anny),
    )
