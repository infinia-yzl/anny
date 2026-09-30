# Corporis
# Apache License, Version 2.0
"""
Posing Anny by the world rotation of each bone.

Anny's ``local-ref`` pose parameters hold, for each bone, a rotation relative to its parent in
a frame where every bone rests along the world axes. A pose is easiest to build the other way
round: give each posed bone a world rotation ``W`` (the turn of the bone from the rest pose,
in world axes), and take each bone's parameter as ``W_parent^T W``, where the parent is the
nearest posed ancestor (:meth:`Skeleton.params`).

The model's frame: x toward the figure's left, -y forward, z up, in metres.
"""

from __future__ import annotations

import math
from typing import Mapping

import roma
import torch

import anny

Rotations = dict[
    str, torch.Tensor
]  # bone label -> 3x3 world rotation from the rest pose


def rot(axis: str, deg: float) -> torch.Tensor:
    """a turn of ``deg`` degrees about a world axis ("x", "y" or "z")"""
    v = torch.zeros(3)
    v["xyz".index(axis)] = math.radians(deg)
    return roma.rotvec_to_rotmat(v)


def bend(deg: float) -> torch.Tensor:
    """a backward bend of the torso, or a forward swing of a leg, in the body's side plane"""
    return rot("x", -deg)


def axis_angle(axis, deg: float) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float32)
    return roma.rotvec_to_rotmat(axis / axis.norm() * math.radians(deg))


def align(a, b) -> torch.Tensor:
    """the smallest rotation that takes direction a to direction b"""
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    a, b = a / a.norm(), b / b.norm()
    axis = torch.linalg.cross(a, b)
    s, c = axis.norm(), torch.dot(a, b)
    if s < 1e-8:
        if c > 0:
            return torch.eye(3)
        # half a turn about any axis at right angles to a
        other = torch.tensor([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else torch.eye(3)[1]
        axis = torch.linalg.cross(a, other)
        return roma.rotvec_to_rotmat(axis / axis.norm() * math.pi)
    return roma.rotvec_to_rotmat(axis / s * torch.atan2(s, c))


def frame(f, t) -> torch.Tensor:
    """an orthonormal frame (as columns) from a main direction f and a second direction t"""
    f = torch.as_tensor(f, dtype=torch.float32)
    t = torch.as_tensor(t, dtype=torch.float32)
    f = f / f.norm()
    t = t - torch.dot(t, f) * f
    t = t / t.norm()
    return torch.stack([f, t, torch.linalg.cross(f, t)], dim=1)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """the rotation a fraction ``t`` of the way from a to b"""
    return a @ roma.rotvec_to_rotmat(roma.rotmat_to_rotvec(a.T @ b) * t)


# the finger bones: the thumb hangs from the wrist, the other fingers from their metacarpals
FINGER_PARENTS = {
    2: "metacarpal1",
    3: "metacarpal2",
    4: "metacarpal3",
    5: "metacarpal4",
}


class Skeleton:
    """Anny's rig for one body, posed by world rotations."""

    def __init__(self, model=None, phenotype: Mapping[str, float] | None = None):
        self.model = model if model is not None else anny.Anny().to(torch.float32)
        self.phenotype = dict(phenotype or {})
        self.labels = list(self.model.bone_labels)
        self.parents = list(self.model.bone_parents)
        with torch.no_grad():
            out = self.model(phenotype_kwargs=self.phenotype)
        self.rest = out["bone_poses"][0].float()

    def joint(self, name: str) -> torch.Tensor:
        """a bone's head in the rest pose"""
        return self.rest[self.labels.index(name), :3, 3]

    def rest_direction(self, a: str, b: str) -> torch.Tensor:
        d = self.joint(b) - self.joint(a)
        return d / d.norm()

    def params(self, W: Rotations, root=None) -> torch.Tensor:
        """local-ref pose parameters (1, bones, 4, 4); ``root`` moves the root bone (m)"""
        P = torch.eye(4)[None, None].repeat(1, self.model.bone_count, 1, 1)
        for bone, Wb in W.items():
            i = self.labels.index(bone)
            j = self.parents[i]
            while j >= 0 and self.labels[j] not in W:
                j = self.parents[j]
            Wp = W[self.labels[j]] if j >= 0 else torch.eye(3)
            P[0, i, :3, :3] = Wp.T @ Wb
        if root is not None:
            P[0, 0, :3, 3] = torch.as_tensor(root, dtype=torch.float32)
        return P

    def output(self, W: Rotations, root=None, **kwargs) -> dict:
        """the model's output for the pose (``vertices``, ``bone_poses``, ...)"""
        with torch.no_grad():
            return self.model(
                pose_parameters=self.params(W, root).to(self.model.dtype),
                phenotype_kwargs=self.phenotype,
                **kwargs,
            )

    def posed_joints(self, W: Rotations, root=None) -> dict[str, torch.Tensor]:
        bp = self.output(W, root)["bone_poses"][0]
        return {n: bp[i, :3, 3].float() for i, n in enumerate(self.labels)}

    def aim(self, W: Rotations, bone: str, child: str, direction) -> None:
        """turn ``bone`` by the smallest rotation that points it (toward ``child``) along
        ``direction``"""
        W[bone] = align(self.rest_direction(bone, child), direction)

    def hinge(self, W: Rotations, chain, upper, lower, rest_axis, bent=8.0) -> None:
        """two bones that meet at a hinge (an elbow, a knee, a finger joint)

        ``chain`` names (upper bone, lower bone, the lower bone's child); ``upper`` and
        ``lower`` are the directions the two segments point along; ``rest_axis`` is the
        hinge's axis in the rest pose. Both bones keep the hinge's axis, so the lower bone
        turns about it alone. When the joint is nearly straight (under ``bent`` degrees), the
        axis follows the upper bone by the smallest turn.
        """
        a, b, c = chain
        u0, l0 = self.rest_direction(a, b), self.rest_direction(b, c)
        u = torch.as_tensor(upper, dtype=torch.float32)
        low = torch.as_tensor(lower, dtype=torch.float32)
        u, low = u / u.norm(), low / low.norm()
        n0 = torch.as_tensor(rest_axis, dtype=torch.float32)
        guess = align(u0, u) @ n0  # the axis after the smallest turn of the upper bone
        cross = torch.linalg.cross(u, low)
        if cross.norm() > math.sin(math.radians(bent)):
            n = cross / cross.norm()
            if torch.dot(n, guess) < 0:
                n = -n
        else:
            n = guess
        W[a] = frame(u, n) @ frame(u0, n0).T
        W[b] = frame(low, n) @ frame(l0, n0).T

    def hand(
        self,
        W: Rotations,
        side: str,
        point,
        index,
        curl=0.0,
        spread=(0, 0, 0, 0),
        thumb=0.0,
    ):
        """a hand set by a few values: the fingers along ``point``, the index side toward
        ``index``, each finger ``spread`` degrees apart and curled ``curl`` degrees at each
        joint, the thumb opened ``thumb`` degrees"""
        s = side
        target = frame(point, index)
        f0 = self.joint("finger3-1" + s) - self.joint("wrist" + s)
        t0 = self.joint("finger2-1" + s) - self.joint("finger5-1" + s)
        R = target @ frame(f0, t0).T
        W["wrist" + s] = R
        normal = torch.linalg.cross(target[:, 0], target[:, 1])  # across the palm
        for k, sp in zip((2, 3, 4, 5), spread):
            W[FINGER_PARENTS[k] + s] = R
            base = axis_angle(normal, sp) @ R
            rest = self.joint(f"finger{k}-2" + s) - self.joint(f"finger{k}-1" + s)
            axis = torch.linalg.cross(
                base @ rest, normal
            )  # the axis the finger curls about
            for i in (1, 2, 3):
                W[f"finger{k}-{i}" + s] = axis_angle(axis, i * curl) @ base
        opened = axis_angle(normal, thumb) @ R
        for i in (1, 2, 3):
            W[f"finger1-{i}" + s] = opened
