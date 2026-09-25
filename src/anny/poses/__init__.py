# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The pose library of anny: 50 poses and 7 looping clips that fit every body of anny.

The library stores one rotation per bone relative to its parent, with every bone at rest along
the world axes, and a root offset relative to the height of the hips. :func:`pose_parameters`
turns an entry into ``local-ref`` pose parameters for a given model and phenotype, and it puts
the lowest point of the body on the floor (``z = 0``). Seated poses also get a stool under the
seat.

Example::

    import anny
    import anny.poses

    model = anny.Anny()
    posed = anny.poses.pose_parameters(model, "hands_on_hips", phenotype_kwargs={"age": 0.2})
    output = model(
        pose_parameters=posed["pose_parameters"],
        phenotype_kwargs={"age": 0.2},
        pose_parameterization="local-ref",
    )

The poses of the MakeHuman community pose packs are CC0; each entry names its author.
``anny.poses.authoring`` holds the tools that build the library.
"""

from __future__ import annotations

import functools
import json
import pathlib

import roma
import torch
from safetensors.torch import load_file

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "poses"


class PoseLibrary:
    """The poses and clips of ``data/poses``."""

    def __init__(self, data_dir: pathlib.Path = DATA_DIR):
        with open(pathlib.Path(data_dir) / "library.json") as f:
            self.meta = json.load(f)
        tensors = load_file(str(pathlib.Path(data_dir) / "library.safetensors"))
        self.rotations = tensors["rotations"]  # (frames, bones, 4) x, y, z, w
        self.root_offsets = tensors["root_offsets"]  # (frames, 3)
        self.bones = list(self.meta["bones"])
        self.entries = {e["name"]: e for e in self.meta["entries"]}
        self.seat_vertices = list(self.meta["seat_vertices"])

    def names(self, kind: str | None = None) -> list[str]:
        """Entry names, in the order of the library; ``kind`` is ``"pose"`` or ``"loop"``."""
        return [
            e["name"] for e in self.meta["entries"] if kind is None or e["kind"] == kind
        ]

    def frames(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotations (F, B, 4) and normalised root offsets (F, 3) of an entry."""
        e = self.entries[name]
        s = slice(e["start"], e["start"] + e["count"])
        return self.rotations[s], self.root_offsets[s]


@functools.lru_cache(maxsize=1)
def library() -> PoseLibrary:
    return PoseLibrary()


def _rest(model, phenotype_kwargs, local_changes_kwargs):
    with torch.no_grad():
        return model(
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs,
            pose_parameterization="local-ref",
        )


def to_local_ref(
    model,
    rotations: torch.Tensor,
    root_offsets: torch.Tensor,
    bones: list[str],
    phenotype_kwargs=None,
    local_changes_kwargs=None,
    rest_output=None,
) -> torch.Tensor:
    """
    Turn library rotations into ``local-ref`` pose parameters of a model.

    Args:
        rotations: (F, len(bones), 4) rotations (x, y, z, w) of each bone relative to its
            parent, with every bone at rest along the world axes.
        root_offsets: (F, 3) offsets of the root bone, divided by the height of the root
            above the lowest rest vertex.
        bones: names of the bones of ``rotations``. Bones of the model that are missing
            stay at rest.

    Returns:
        (F, model.bone_count, 4, 4) pose parameters for ``pose_parameterization="local-ref"``.
        The root keeps its rest height plus the offset; see :func:`ground`.
    """
    dtype, device = model.template_vertices.dtype, model.template_vertices.device
    rest = rest_output or _rest(model, phenotype_kwargs, local_changes_kwargs)
    n_frames = rotations.shape[0]
    labels = list(model.bone_labels)
    source = {b: i for i, b in enumerate(bones)}
    R = torch.eye(3, dtype=dtype, device=device).repeat(n_frames, len(labels), 1, 1)
    for i, label in enumerate(labels):
        if label in source:
            q = rotations[:, source[label]].to(dtype=dtype, device=device)
            R[:, i] = roma.unitquat_to_rotmat(q)
    rest_rot = rest["rest_bone_poses"][0, :, :3, :3]
    if model.reference_bone_orientations is not None:
        C = model.reference_bone_orientations.to(dtype) @ rest_rot.transpose(-1, -2)
    else:
        C = torch.eye(3, dtype=dtype, device=device).repeat(len(labels), 1, 1)
    parents = [int(p) for p in model.bone_parents]
    C_parent = torch.stack(
        [C[p] if p >= 0 else torch.eye(3, dtype=dtype, device=device) for p in parents]
    )
    delta = torch.eye(4, dtype=dtype, device=device).repeat(n_frames, len(labels), 1, 1)
    delta[:, :, :3, :3] = C_parent[None] @ R @ C.transpose(-1, -2)[None]
    heads = rest["rest_bone_heads"][0]
    hip_height = heads[0, 2] - rest["rest_vertices"][0, :, 2].min()
    delta[:, 0, :3, 3] = (
        heads[0] + root_offsets.to(dtype=dtype, device=device) * hip_height
    )
    return delta


def ground(model, pose_parameters, rest_output=None, per_frame=True, **model_kwargs):
    """
    Move the root so that the lowest vertex touches the floor ``z = 0``.

    With ``per_frame=False`` (for clips), one vertical shift for all frames puts the lowest
    point over all frames on the floor, so that a jump stays in the air.
    """
    with torch.no_grad():
        out = model(
            pose_parameters=pose_parameters,
            pose_parameterization="local-ref",
            **model_kwargs,
        )
    low = out["vertices"][..., 2].min(dim=-1).values
    if not per_frame:
        low = low.min().expand_as(low)
    pose_parameters = pose_parameters.clone()
    pose_parameters[:, 0, 2, 3] -= low
    return pose_parameters, out["vertices"] - torch.stack(
        [torch.zeros_like(low), torch.zeros_like(low), low], -1
    )[:, None, :]


def pose_parameters(
    model,
    name: str,
    phenotype_kwargs=None,
    local_changes_kwargs=None,
    grounded: bool = True,
) -> dict:
    """
    ``local-ref`` pose parameters of a library entry for a model and a phenotype.

    Returns a dict with ``pose_parameters`` (F, B, 4, 4), one frame for a pose and all frames
    of a clip, ``fps`` and ``kind``. When ``grounded`` is True, the lowest point sits on the
    floor ``z = 0``, the dict holds the posed ``vertices``, and seated poses get a ``stool``
    (``top`` height, ``center`` (x, y), ``radius``, ``thickness``, ``legs``, ``leg_radius``).
    """
    lib = library()
    entry = lib.entries[name]
    rotations, root = lib.frames(name)
    model_kwargs = dict(
        phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes_kwargs
    )
    rest = _rest(model, phenotype_kwargs, local_changes_kwargs)
    params = to_local_ref(model, rotations, root, lib.bones, rest_output=rest)
    result = dict(pose_parameters=params, kind=entry["kind"], fps=entry["fps"])
    if not grounded:
        return result
    params, vertices = ground(
        model, params, per_frame=entry["kind"] == "pose", **model_kwargs
    )
    result.update(pose_parameters=params, vertices=vertices)
    if "stool" in entry and lib.seat_vertices:
        heads = rest["rest_bone_heads"][0]
        hip_height = float(heads[0, 2] - rest["rest_vertices"][0, :, 2].min())
        seat = vertices[0, lib.seat_vertices]
        lowest = seat[seat[:, 2].argmin()]
        stool = {k: v * hip_height for k, v in entry["stool"].items() if k != "legs"}
        stool.update(
            legs=entry["stool"]["legs"],
            top=float(lowest[2]),
            center=[
                float(lowest[0]),
                float(lowest[1]) - lib.meta["stool_forward"] * hip_height,
            ],
        )
        result["stool"] = stool
    return result


def names(kind: str | None = None) -> list[str]:
    """Names of the library entries; ``kind`` is ``"pose"``, ``"loop"`` or None for all."""
    return library().names(kind)
