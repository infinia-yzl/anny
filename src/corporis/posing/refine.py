# Corporis
# Apache License, Version 2.0
"""
Refining a retargeted pose against the landmarks.

The retarget (:mod:`corporis.posing.retarget`) aims each bone along its landmarks but leaves
the spine's curve and the collarbones at a guess. The refinement turns each bone a little
further, in the world frame, so that Anny's landmarks (:class:`AnnyLandmarks`) meet the
picture's: in 3D after the best scale and offset, and in the picture after the best scale and
offset of a camera that looks along y, when the landmarks carry pixel positions. A prior keeps
the turns small.

Pictures can be noisy, and pictures made by AI models can draw anatomy that no body has, so a
few landmarks can be far off. Each landmark's error passes through a robust loss (Geman and
McClure), which grows like the squared error for small errors and levels off beyond
``ROBUST`` of the body's size, so one bad landmark cannot pull the pose. The head points that
the head fit rejected (:meth:`Retargeter.head_fit`) take no part, in 3D or in the picture.
"""

from __future__ import annotations

import numpy as np
import roma
import torch

from corporis.posing.landmarks import Landmarks
from corporis.posing.retarget import Retargeter
from corporis.posing.head import HEAD_POINTS
from corporis.posing.skeleton import Rotations

# the error (a share of the body's size) beyond which a landmark's loss levels off
ROBUST = 0.08

# the bones the refinement turns; the torso, the collarbones and the limbs carry the pose
BONES = (
    ["spine05", "spine04", "spine03", "spine02", "spine01", "neck01", "head"]
    + [f"{b}{s}" for b in ("clavicle", "shoulder01") for s in (".L", ".R")]
    + [
        f"{b}{s}"
        for b in ("upperarm01", "lowerarm01", "upperleg01", "lowerleg01")
        for s in (".L", ".R")
    ]
)


def _align(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``pred`` moved and scaled (no turn) to meet ``target`` best, in weighted least squares"""
    w = weight[:, None] / weight.sum()
    pc = (pred * w).sum(0)
    tc = (target * w).sum(0)
    p, t = pred - pc, target - tc
    scale = (w * p * t).sum() / (w * p * p).sum().clamp_min(1e-9)
    return p * scale + tc


def refine(
    retargeter: Retargeter,
    W: Rotations,
    L: Landmarks,
    steps: int = 60,
    lr: float = 0.02,
    prior: float = 0.002,
    image_weight: float = 1.0,
) -> Rotations:
    """``W`` turned so that the model's landmarks meet ``L``"""
    sk = retargeter.skeleton
    anny_lm = retargeter.anny
    W = dict(W)
    for b in BONES:
        W.setdefault(b, W.get(sk.labels[sk.parents[sk.labels.index(b)]], torch.eye(3)))
    offsets = torch.zeros(len(BONES), 3, requires_grad=True)
    target = torch.as_tensor(L.body, dtype=torch.float32)
    vis = torch.as_tensor(
        L.visibility if L.visibility is not None else np.ones(len(L.body)),
        dtype=torch.float32,
    ).clamp(0.05, 1.0)
    head = retargeter.head_fit(L)
    vis_image = vis.clone()
    vis[:HEAD_POINTS] *= torch.as_tensor(head.in_3d, dtype=torch.float32)
    vis_image[:HEAD_POINTS] *= torch.as_tensor(head.in_image, dtype=torch.float32)
    image = None
    if L.image is not None and image_weight > 0:
        image = torch.as_tensor(L.image, dtype=torch.float32)
        image = torch.stack(
            [image[:, 0], -image[:, 1]], dim=1
        )  # y up, as the model's z
    size = (target.max(0).values - target.min(0).values).norm().clamp_min(1e-6)
    opt = torch.optim.Adam([offsets], lr=lr)
    base = {b: W[b] for b in BONES}
    for _ in range(steps):
        turned = dict(W)
        for i, b in enumerate(BONES):
            turned[b] = roma.rotvec_to_rotmat(offsets[i]) @ base[b]
        out = sk.model(
            pose_parameters=sk.params(turned).to(sk.model.dtype),
            phenotype_kwargs=sk.phenotype,
        )
        pred = _points(anny_lm, out)
        err = (_align(pred, target, vis) - target) / size
        loss = (vis * _robust((err**2).sum(1))).sum() / vis.sum()
        if image is not None:
            flat = pred[:, [0, 2]]
            loss = loss + image_weight * _image_loss(flat, image, vis_image)
        loss = loss + prior * (offsets**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        for i, b in enumerate(BONES):
            W[b] = roma.rotvec_to_rotmat(offsets[i]) @ base[b]
    return W


def _points(anny_lm, output) -> torch.Tensor:
    v = output["vertices"][0]
    bp = output["bone_poses"][0]
    labels = anny_lm.skeleton.labels
    return torch.stack(
        [
            bp[labels.index(ref), :3, 3] if kind == "joint" else v[ref]
            for kind, ref in anny_lm.body
        ]
    ).float()


def _image_loss(
    flat: torch.Tensor, image: torch.Tensor, vis: torch.Tensor
) -> torch.Tensor:
    """the picture's error of the model's landmarks seen along y, after the best scale and
    offset, relative to the figure's size in the picture"""
    fitted = _align(flat, image, vis)
    size = (image.max(0).values - image.min(0).values).norm().clamp_min(1e-6)
    return (vis * _robust((((fitted - image) / size) ** 2).sum(1))).sum() / vis.sum()


def _robust(e2: torch.Tensor) -> torch.Tensor:
    """Geman-McClure: about e2 for small errors, levelling off at ROBUST**2"""
    c2 = ROBUST**2
    return c2 * e2 / (e2 + c2)
