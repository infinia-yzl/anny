# OpenSculptBoy
# Apache License, Version 2.0
"""A character: the settings of the Anny model that describe one humanoid character."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

SCHEMA_VERSION = 1


@dataclasses.dataclass
class Character:
    """
    The settings that make one character of the Anny model.

    Every mapping takes parameter names of the model: ``phenotype`` takes
    ``model.phenotype_labels`` (0 to 1, default 0.5), ``local_changes`` takes
    ``model.local_change_labels`` (-1 to 1), ``face_shapes`` takes ``model.face_shape_labels``
    (see ``model.face_shape_ranges``) and ``facial_actions`` takes
    ``model.facial_action_labels`` (0 to 1, the ARKit blend shape names). Missing names keep
    their defaults.

    ``pose`` holds an optional still pose in Anny's ``local-ref`` parameters: ``"bones"`` maps
    bone labels (``model.bone_labels``) to unit quaternions (x, y, z, w), each the bone's turn
    relative to its parent, and ``"root"`` moves the root bone (x, y, z in metres, Anny's
    frame). Missing bones keep the rest pose, and an empty ``pose`` is the rest pose.
    """

    name: str = "character"
    rig: str = "anny"
    topology: str = "anny"
    phenotypes: str = "default"
    phenotype: dict[str, float] = dataclasses.field(default_factory=dict)
    local_changes: dict[str, float] = dataclasses.field(default_factory=dict)
    face_shapes: dict[str, float] = dataclasses.field(default_factory=dict)
    facial_actions: dict[str, float] = dataclasses.field(default_factory=dict)
    pose: dict[str, Any] = dataclasses.field(default_factory=dict)

    def build_model(self, face_shapes: bool | None = None, dtype=torch.float32):
        """
        The Anny model for this character. ``face_shapes`` loads the face-shape parameters;
        by default they load when the character sets any of them.
        """
        import anny

        if face_shapes is None:
            face_shapes = bool(self.face_shapes)
        model = anny.Anny(
            rig=self.rig,
            topology=self.topology,
            phenotypes=self.phenotypes,
            local_changes="all" if self.local_changes else "none",
            facial_actions="all",
            face_shapes="all" if face_shapes else "none",
        )
        return model.to(dtype=dtype)

    def model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``model.forward`` for this character."""
        return dict(
            phenotype_kwargs=dict(self.phenotype) or None,
            local_changes_kwargs=dict(self.local_changes) or None,
            face_shape_kwargs=dict(self.face_shapes) or None,
            facial_actions=dict(self.facial_actions) or None,
        )

    def pose_parameters(self, model) -> torch.Tensor | None:
        """``local-ref`` pose parameters (1, bones, 4, 4) of ``pose`` for ``model``, or None
        when the character has no pose"""
        if not self.pose:
            return None
        import roma

        labels = list(model.bone_labels)
        bones = self.pose.get("bones", {})
        unknown = sorted(set(bones) - set(labels))
        if unknown:
            raise ValueError(f"Unknown bones in the pose: {unknown}.")
        dtype = model.template_vertices.dtype
        params = torch.eye(4, dtype=dtype).repeat(1, len(labels), 1, 1)
        for bone, quaternion in bones.items():
            q = torch.as_tensor(quaternion, dtype=torch.float64)
            params[0, labels.index(bone), :3, :3] = roma.unitquat_to_rotmat(
                q / q.norm()
            ).to(dtype)
        params[0, 0, :3, 3] = torch.as_tensor(
            self.pose.get("root", (0, 0, 0)), dtype=dtype
        )
        return params

    @staticmethod
    def pose_from_parameters(model, pose_parameters, tolerance: float = 1e-6) -> dict:
        """the ``pose`` field for ``local-ref`` pose parameters (bones, 4, 4) or (1, bones, 4,
        4); bones within ``tolerance`` of the rest pose are left out"""
        import roma

        params = (
            torch.as_tensor(pose_parameters).detach().double().cpu().reshape(-1, 4, 4)
        )
        quaternions = roma.rotmat_to_unitquat(params[:, :3, :3])
        bones = {}
        for label, q, R in zip(model.bone_labels, quaternions, params[:, :3, :3]):
            if (R - torch.eye(3, dtype=R.dtype)).abs().max() > tolerance:
                bones[label] = [round(float(x), 7) for x in q]
        return {
            "bones": bones,
            "root": [round(float(x), 6) for x in params[0, :3, 3]],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        if not isinstance(data, dict):
            raise ValueError(
                f"A character card must be a JSON object (got {type(data).__name__})."
            )
        schema = data.get("schema", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported character schema {schema}; this version reads schema {SCHEMA_VERSION}."
            )
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - fields - {"schema"})
        if unknown:
            raise ValueError(
                f"Unknown character fields {unknown}; the fields are {sorted(fields)}."
            )
        return cls(**{k: v for k, v in data.items() if k in fields})
