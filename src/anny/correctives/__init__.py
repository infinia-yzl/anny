# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Soft-tissue corrective shapes for anny.

Plain linear blend skinning loses volume where a joint bends far: an elbow or a knee thins out,
and a shoulder or a hip folds in. The corrective shapes restore this volume at the shoulders,
the elbows, the hips and the knees. A soft-tissue simulation of anny's default body made them
(``anny.correctives.authoring``), and drivers set their weights from the posed skeleton:

- hinge joints (elbow, knee): the bend angle between two bones, interpolated between keys;
- ball joints (shoulder, hip): the direction of the upper bone in the frame of the torso bone
  above it, as barycentric coordinates in a triangle of key directions.

At the rest pose every weight is 0. Each shape scales with the size of the body around it, so
the shapes follow anny's phenotype sliders. :class:`SoftTissueCorrectives` works on the output
of an anny model and leaves ``Anny.forward`` unchanged::

    model = anny.Anny()
    correctives = anny.correctives.SoftTissueCorrectives(model)
    output = model(pose_parameters=pose, phenotype_kwargs=phenotype)
    output = correctives(output)  # corrected "vertices"

The layer assumes linear blend skinning. It needs a model built from the MakeHuman base mesh
(the ``anny`` and ``makehuman`` topologies) with MakeHuman bone names (the ``anny`` and
``makehuman`` rigs).
"""

from __future__ import annotations

import json
import pathlib

import torch
from safetensors.torch import load_file

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "correctives"


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def hinge_key_weights(angle: torch.Tensor, key_angles: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation weights of ``angle`` (B,) between increasing ``key_angles`` (B, K).
    Outside the keys the first or last key takes the full weight.
    """
    k = key_angles.shape[-1]
    a = torch.minimum(torch.maximum(angle, key_angles[:, 0]), key_angles[:, -1])
    weights = torch.zeros_like(key_angles)
    assigned = torch.zeros_like(angle, dtype=torch.bool)
    for i in range(k - 1):
        x0, x1 = key_angles[:, i], key_angles[:, i + 1]
        inside = (a >= x0) & (a <= x1) & ~assigned
        t = ((a - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0, 1)
        weights[:, i] = weights[:, i] + torch.where(inside, 1 - t, torch.zeros_like(t))
        weights[:, i + 1] = weights[:, i + 1] + torch.where(
            inside, t, torch.zeros_like(t)
        )
        assigned = assigned | inside
    return weights


def cone_key_weights(
    direction: torch.Tensor, targets: torch.Tensor, triangles: torch.Tensor
) -> torch.Tensor:
    """
    Barycentric weights of ``direction`` (B, 3) among ``targets`` (B, K, 3). Each triangle of
    ``triangles`` (T, 3) gives coordinates that are divided by their sum; the triangle with the
    largest smallest coordinate wins, its negative coordinates become 0 and the others are
    divided by their sum. A direction outside every triangle (no triangle gives a positive sum)
    gets no weight, so the correction fades out there.
    """
    B, K = targets.shape[:2]
    M = targets[:, triangles].transpose(-1, -2)  # (B, T, 3, 3), columns are the targets
    rhs = direction[:, None, :].expand(-1, triangles.shape[0], -1)
    b = torch.linalg.solve(M, rhs)  # (B, T, 3)
    total = b.sum(-1, keepdim=True)
    valid = total[..., 0] > 0
    b = b / torch.where(total > 0, total, torch.ones_like(total))
    score = torch.where(valid, b.min(-1).values, torch.full_like(total[..., 0], -1e9))
    best = score.argmax(-1)  # (B,)
    bb = b[torch.arange(B), best].clamp_min(0)
    bb = bb / bb.sum(-1, keepdim=True).clamp_min(1e-12)
    bb = torch.where(valid.any(-1, keepdim=True), bb, torch.zeros_like(bb))
    weights = torch.zeros(B, K, dtype=targets.dtype, device=targets.device)
    weights.scatter_add_(1, triangles[best], bb)
    return weights


class SoftTissueCorrectives(torch.nn.Module):
    """Corrective shapes for the output of an anny model (see the module documentation)."""

    def __init__(self, model, data_dir: str | pathlib.Path = DATA_DIR):
        super().__init__()
        data_dir = pathlib.Path(data_dir)
        with open(data_dir / "soft_tissue.json") as f:
            self.spec = json.load(f)
        tensors = load_file(str(data_dir / "soft_tissue.safetensors"))
        dtype = model.template_vertices.dtype
        device = model.template_vertices.device
        self.bone_labels = list(model.bone_labels)
        base_index = model.base_mesh_vertex_indices.detach().cpu()
        lookup = torch.full((int(base_index.max()) + 1,), -1, dtype=torch.int64)
        lookup[base_index] = torch.arange(len(base_index))
        self.shape_names = [s["name"] for s in self.spec["shapes"]]
        self.reference_radius = torch.tensor(
            [s["reference_radius"] for s in self.spec["shapes"]], dtype=dtype
        )
        indices, offsets, owners = [], [], []
        for k, name in enumerate(self.shape_names):
            base = tensors[name + ".indices"]
            if int(base.max()) >= len(lookup) or (lookup[base] < 0).any():
                raise ValueError(
                    "SoftTissueCorrectives needs a model built from the MakeHuman base mesh "
                    "(the anny or makehuman topologies)."
                )
            indices.append(lookup[base])
            offsets.append(tensors[name + ".offsets"].to(dtype))
            owners.append(torch.full((len(base),), k, dtype=torch.int64))
        self.register_buffer("indices", torch.cat(indices).to(device))
        self.register_buffer("offsets", torch.cat(offsets).to(device))
        self.register_buffer("owners", torch.cat(owners).to(device))
        self.reference_radius = self.reference_radius.to(device)
        self.touched = torch.unique(self.indices)
        self.joints = []
        for joint in self.spec["joints"]:
            self.joints.append(self._parse_joint(joint, dtype, device))
        self.vertex_bone_indices = model.vertex_bone_indices
        self.vertex_bone_weights = model.vertex_bone_weights
        self.enabled = True

    def _bone(self, name):
        if name not in self.bone_labels:
            raise ValueError(
                f"The rig has no bone {name!r} for the corrective drivers."
            )
        return self.bone_labels.index(name)

    def _shape(self, name):
        return -1 if name is None else self.shape_names.index(name)

    def _parse_joint(self, joint, dtype, device):
        if joint["type"] == "hinge":
            return dict(
                type="hinge",
                bones=[self._bone(b) for b in joint["bones"]],
                angles=torch.tensor(
                    [k["angle"] for k in joint["keys"]], dtype=dtype, device=device
                ),
                rest=[k.get("rest", False) for k in joint["keys"]].index(True),
                shapes=[self._shape(k["shape"]) for k in joint["keys"]],
            )
        return dict(
            type="cone",
            bone=self._bone(joint["bone"]),
            end=self._bone(joint["end"]),
            frame=self._bone(joint["frame"]),
            dirs=torch.tensor(
                [t["dir"] for t in joint["targets"]], dtype=dtype, device=device
            ),
            rest=[t.get("rest", False) for t in joint["targets"]].index(True),
            shapes=[self._shape(t["shape"]) for t in joint["targets"]],
            triangles=torch.tensor(
                joint["triangles"], dtype=torch.int64, device=device
            ),
        )

    def shape_weights(self, output) -> torch.Tensor:
        """Weights (B, number of shapes) of the corrective shapes for a posed output."""
        bone_poses = output["bone_poses"]
        rest_poses = output["rest_bone_poses"].expand_as(bone_poses)
        G = bone_poses[..., :3, :3] @ rest_poses[..., :3, :3].transpose(-1, -2)
        posed = bone_poses[..., :3, 3]
        rest = rest_poses[..., :3, 3]
        B = bone_poses.shape[0]
        weights = torch.zeros(
            B, len(self.shape_names), dtype=bone_poses.dtype, device=bone_poses.device
        )
        for joint in self.joints:
            if joint["type"] == "hinge":
                a, b, c = joint["bones"]

                def bend(P):
                    u, f = _unit(P[:, b] - P[:, a]), _unit(P[:, c] - P[:, b])
                    cos = (u * f).sum(-1).clamp(-1 + 1e-9, 1 - 1e-9)
                    return torch.rad2deg(torch.acos(cos))

                angles = joint["angles"].expand(B, -1).clone()
                i = joint["rest"]
                rest_angle = bend(rest)
                # the rest key stays between its neighbours
                lo = angles[:, i - 1] + 1e-3 if i > 0 else rest_angle
                hi = angles[:, i + 1] - 1e-3 if i + 1 < angles.shape[1] else rest_angle
                angles[:, i] = torch.minimum(torch.maximum(rest_angle, lo), hi)
                key_weights = hinge_key_weights(bend(posed), angles)
            else:
                d = posed[:, joint["end"]] - posed[:, joint["bone"]]
                d = _unit(
                    (G[:, joint["frame"]].transpose(-1, -2) @ d[..., None])[..., 0]
                )
                dirs = joint["dirs"].expand(B, -1, -1).clone()
                dirs[:, joint["rest"]] = _unit(
                    rest[:, joint["end"]] - rest[:, joint["bone"]]
                )
                key_weights = cone_key_weights(d, dirs, joint["triangles"])
            for k, shape in enumerate(joint["shapes"]):
                if shape >= 0:
                    weights[:, shape] = weights[:, shape] + key_weights[:, k]
        return weights

    def shape_scales(self, rest_vertices: torch.Tensor) -> torch.Tensor:
        """Size of the body around each shape relative to anny's default body: (B, shapes)."""
        support = rest_vertices[:, self.indices]  # (B, N, 3)
        n = len(self.shape_names)
        B = rest_vertices.shape[0]
        count = torch.bincount(self.owners, minlength=n).to(rest_vertices.dtype)
        mean = torch.zeros(
            B, n, 3, dtype=rest_vertices.dtype, device=rest_vertices.device
        )
        mean.index_add_(1, self.owners, support)
        mean = mean / count[None, :, None]
        sq = ((support - mean[:, self.owners]) ** 2).sum(-1)
        var = torch.zeros(B, n, dtype=rest_vertices.dtype, device=rest_vertices.device)
        var.index_add_(1, self.owners, sq)
        return torch.sqrt(var / count[None]) / self.reference_radius.to(var.dtype)

    def rest_offsets(self, output, weights=None) -> torch.Tensor:
        """The corrective offsets of the rest vertices (B, V, 3) for a posed output."""
        rest_vertices = output["rest_vertices"]
        B = output["bone_poses"].shape[0]
        weights = self.shape_weights(output) if weights is None else weights
        scale = self.shape_scales(rest_vertices).expand(B, -1)
        factor = (weights * scale)[:, self.owners]  # (B, N)
        offsets = torch.zeros(
            B,
            rest_vertices.shape[1],
            3,
            dtype=rest_vertices.dtype,
            device=rest_vertices.device,
        )
        offsets.index_add_(1, self.indices, factor[..., None] * self.offsets[None])
        return offsets

    def forward(self, output: dict) -> dict:
        """A copy of ``output`` with corrected ``vertices`` and the ``corrective_weights``."""
        if not self.enabled:
            return output
        weights = self.shape_weights(output)
        offsets = self.rest_offsets(output, weights)[:, self.touched]
        bone_poses = output["bone_poses"]
        rest_poses = output["rest_bone_poses"].expand_as(bone_poses)
        G = bone_poses[..., :3, :3] @ rest_poses[..., :3, :3].transpose(-1, -2)
        bi = self.vertex_bone_indices[self.touched]
        bw = self.vertex_bone_weights[self.touched].to(G.dtype)
        A = (bw[None, :, :, None, None] * G[:, bi]).sum(2)  # (B, n, 3, 3)
        moved = (A @ offsets[..., None])[..., 0]
        vertices = output["vertices"].clone()
        vertices[:, self.touched] = vertices[:, self.touched] + moved
        result = dict(output)
        result["vertices"] = vertices
        result["corrective_weights"] = weights
        return result
