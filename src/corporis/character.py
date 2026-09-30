# Corporis
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
    """

    name: str = "character"
    rig: str = "anny"
    topology: str = "anny"
    phenotypes: str = "default"
    phenotype: dict[str, float] = dataclasses.field(default_factory=dict)
    local_changes: dict[str, float] = dataclasses.field(default_factory=dict)
    face_shapes: dict[str, float] = dataclasses.field(default_factory=dict)
    facial_actions: dict[str, float] = dataclasses.field(default_factory=dict)

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

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
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
