# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, final

import torch

from anny.torch_compat import make_buffer
from anny.models.rigged_model import (
    PoseParameterization,
    RiggedModelWithLinearBlendShapes,
)

if TYPE_CHECKING:
    from anny.models.model_data import ModelData
    from anny.typing import (
        LocalChanges,
        SkinningMethod,
        BoneOrientation,
        FaceShapes,
        FacialActions,
        Phenotypes,
    )
from anny.models.model_data import (
    AnnyModelConfig,
    PHENOTYPE_LABELS,
    PHENOTYPE_VARIATIONS,
    RigConfig,
    TopologyConfig,
    resolve_phenotypes,
)
import anny.utils.interpolation
from anny.models.face_shapes import (
    GROUP_TO_SCALE,
    SCALE_GROUPS,
    face_shape_parameter,
    parse_row_label,
)


class BufferDict(torch.nn.Module):
    def __init__(self, input_dict):
        super().__init__()
        for k, v in input_dict.items():
            self.register_buffer(k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def to_batched_tensor(value, device, dtype):
    """
    Helper function to accept float inputs
    """
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.dim() == 0:
        value = value.unsqueeze(dim=0)
    if value.dim() != 1:
        raise ValueError(
            f"Must be a scalar or a 1-D tensor, got shape {tuple(value.shape)}."
        )
    return value


@final
class Anny(RiggedModelWithLinearBlendShapes):
    """Phenotype-aware Anny model and public full-body constructor."""

    def __init__(
        self,
        rig: str | RigConfig = "anny",
        topology: str | TopologyConfig = "anny",
        local_changes: LocalChanges = "none",
        facial_actions: FacialActions = "none",
        phenotypes: Phenotypes = "default",
        extrapolate_phenotypes: bool = False,
        pose_parameterization: PoseParameterization = "local-ref",
        skinning_method: SkinningMethod | None = None,
        face_shapes: FaceShapes = "none",
        scale_face_shapes: bool = True,
    ) -> None:
        """
        face_shapes: the named face-shape parameters to load ("none", "all" or a list of names,
            see ``anny.models.face_shapes``); ``forward`` takes their values in
            ``face_shape_kwargs``.
        scale_face_shapes: scale each face-shape group with the size of the matching part of the
            head relative to anny's default body, so that the shapes stay in proportion at every
            age. With False, the MakeHuman offsets apply unchanged.
        """
        from anny.models import build_model_data

        rig_config = RigConfig.from_string(rig) if isinstance(rig, str) else rig
        topology_config = (
            TopologyConfig.from_string(topology)
            if isinstance(topology, str)
            else topology
        )
        self.config = AnnyModelConfig(
            rig=rig,
            topology=topology,
            local_changes=local_changes,
            facial_actions=facial_actions,
            extrapolate_phenotypes=extrapolate_phenotypes,
            phenotypes=phenotypes,
            pose_parameterization=pose_parameterization,
            skinning_method=skinning_method,
            face_shapes=face_shapes,
            scale_face_shapes=scale_face_shapes,
        )

        data = build_model_data(
            rig=rig_config,
            topology=topology_config,
            local_changes=local_changes,
            facial_actions=facial_actions,
            face_shapes=face_shapes,
        )
        self._init_from_model_data(
            data,
            pose_parameterization=self.config.pose_parameterization,
            skinning_method=skinning_method,
            bone_orientation=rig_config.bone_orientation,
            root_identity_orientation=rig_config.root_identity_orientation,
            phenotypes=phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            scale_face_shapes=scale_face_shapes,
        )

    @staticmethod
    def from_model_data(
        data: ModelData,
        skinning_method: SkinningMethod | None = None,
        pose_parameterization: PoseParameterization = "local-bone",
        bone_orientation: BoneOrientation = "blender",
        root_identity_orientation: bool = True,
        phenotypes: Phenotypes = "default",
        extrapolate_phenotypes: bool = False,
        scale_face_shapes: bool = True,
    ):
        """Construct an Anny model from a ModelData object."""
        model = Anny.__new__(Anny)
        model.config = None
        model._init_from_model_data(
            data,
            pose_parameterization=pose_parameterization,
            skinning_method=skinning_method,
            bone_orientation=bone_orientation,
            root_identity_orientation=root_identity_orientation,
            phenotypes=phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            scale_face_shapes=scale_face_shapes,
        )
        return model

    def _init_from_model_data(
        self,
        data: ModelData,
        skinning_method: SkinningMethod | None = None,
        pose_parameterization: PoseParameterization = "local-bone",
        bone_orientation: BoneOrientation = "blender",
        root_identity_orientation: bool = True,
        phenotypes: Phenotypes = "default",
        extrapolate_phenotypes: bool = False,
        scale_face_shapes: bool = True,
    ):
        super().__init__(
            data,
            pose_parameterization=pose_parameterization,
            skinning_method=skinning_method,
            bone_orientation=bone_orientation,
            root_identity_orientation=root_identity_orientation,
        )
        if data.stacked_phenotype_blend_shapes_mask is None:
            raise ValueError(
                "Model data does not contain stacked_phenotype_blend_shapes_mask, cannot initialize Anny model."
            )

        self._init_phenotype_parameters(
            stacked_phenotype_blend_shapes_mask=data.stacked_phenotype_blend_shapes_mask,
            local_change_labels=data.metadata.local_change_labels,
            facial_action_labels=data.metadata.facial_action_labels,
            base_mesh_vertex_indices=data.base_mesh_vertex_indices,
            extrapolate_phenotypes=extrapolate_phenotypes,
            phenotype_labels=resolve_phenotypes(phenotypes=phenotypes),
        )
        self._init_face_shapes(data, scale_face_shapes)

    def _init_phenotype_parameters(
        self,
        stacked_phenotype_blend_shapes_mask: torch.Tensor,
        local_change_labels: list[str],
        facial_action_labels: list[str],
        base_mesh_vertex_indices: torch.Tensor,
        extrapolate_phenotypes: bool,
        phenotype_labels: list[str],
    ):
        self.stacked_phenotype_blend_shapes_mask = make_buffer(
            self,
            "stacked_phenotype_blend_shapes_mask",
            stacked_phenotype_blend_shapes_mask,
            persistent=False,
        )
        self.local_change_labels = local_change_labels
        self.facial_action_labels = facial_action_labels
        self.base_mesh_vertex_indices = base_mesh_vertex_indices
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.phenotype_labels = phenotype_labels
        self.anchors = BufferDict(self._make_phenotype_anchors())

    def _make_phenotype_anchors(self) -> dict[str, torch.Tensor]:
        anchors = {
            "age": torch.linspace(
                -1 / 3,
                1.0,
                len(PHENOTYPE_VARIATIONS["age"]),
                dtype=self.dtype,
                device=self.device,
            )
        }
        for label in [
            "gender",
            "muscle",
            "weight",
            "height",
            "proportions",
            "cupsize",
            "firmness",
        ]:
            anchors[label] = torch.linspace(
                0.0,
                1.0,
                len(PHENOTYPE_VARIATIONS[label]),
                dtype=self.dtype,
                device=self.device,
            )
        return anchors

    def _init_face_shapes(self, data: ModelData, scale_face_shapes: bool):
        """Rows of the face-shape block, and the landmarks that measure the head for its scales."""
        self.face_shape_labels = data.metadata.face_shape_labels
        self.scale_face_shapes = scale_face_shapes
        row_labels = [
            x for x in data.metadata.blendshape_labels if x.startswith("face_shape:")
        ]
        rows = [parse_row_label(x) for x in row_labels]
        index = {name: i for i, name in enumerate(self.face_shape_labels)}
        params = [face_shape_parameter(name) for name in self.face_shape_labels]
        self.face_shape_groups = {p.name: p.group for p in params}
        self.face_shape_ranges = {p.name: p.range for p in params}
        self.face_shape_row_parameter = make_buffer(
            self,
            "face_shape_row_parameter",
            torch.tensor([index[name] for name, _ in rows], dtype=torch.int64),
            persistent=False,
        )
        self.face_shape_row_sign = make_buffer(
            self,
            "face_shape_row_sign",
            torch.tensor([sign for _, sign in rows], dtype=self.dtype),
            persistent=False,
        )
        self.face_shape_row_scale_group = make_buffer(
            self,
            "face_shape_row_scale_group",
            torch.tensor(
                [
                    SCALE_GROUPS.index(GROUP_TO_SCALE[self.face_shape_groups[name]])
                    for name, _ in rows
                ],
                dtype=torch.int64,
            ),
            persistent=False,
        )
        # Craniofacial landmarks on the phenotype blend shapes: they measure the head for the scales
        # of the face-shape groups (and for anny.faces.measurements)
        self.craniofacial_landmark_labels = list(
            data.metadata.craniofacial_landmark_labels
        )
        n_phenotype = self.stacked_phenotype_blend_shapes_mask.shape[0]
        if data.craniofacial_landmarks is not None:
            self.craniofacial_landmarks_template = make_buffer(
                self,
                "craniofacial_landmarks_template",
                data.craniofacial_landmarks,
                persistent=False,
            )
            self.craniofacial_landmarks_blendshapes = make_buffer(
                self,
                "craniofacial_landmarks_blendshapes",
                data.craniofacial_landmarks_blendshapes,
                persistent=False,
            )
        else:
            self.craniofacial_landmarks_template = None
            self.craniofacial_landmarks_blendshapes = None
        self._n_phenotype_blendshapes = n_phenotype
        reference_sizes = None
        if (
            self.face_shape_labels
            and scale_face_shapes
            and self.craniofacial_landmarks_template is None
        ):
            raise ValueError(
                "Model data has no craniofacial landmarks, so the face shapes cannot scale with "
                "the head; rebuild the model data or pass scale_face_shapes=False."
            )
        if self.face_shape_labels and scale_face_shapes:
            with torch.no_grad():
                default = self._parse_parameter_kwargs(
                    None, self.phenotype_labels, 0.5, "phenotype_kwargs"
                )
                reference_sizes = self._face_shape_group_sizes(default)[0]
        self.face_shape_reference_sizes = (
            None
            if reference_sizes is None
            else make_buffer(
                self, "face_shape_reference_sizes", reference_sizes, persistent=False
            )
        )

    def phenotype_craniofacial_landmarks(
        self, phenotype_parameters: torch.Tensor
    ) -> torch.Tensor:
        """craniofacial landmarks (B, K, 3) of the rest body for phenotype parameters (B, P) alone"""
        if self.craniofacial_landmarks_template is None:
            raise ValueError("Model data has no craniofacial landmarks.")
        coeffs = self._phenotype_block_coefficients(phenotype_parameters)
        n = self._n_phenotype_blendshapes
        return self.craniofacial_landmarks_template[None] + torch.einsum(
            "bn, nkd -> bkd", coeffs, self.craniofacial_landmarks_blendshapes[:n]
        )

    def _face_shape_group_sizes(self, phenotype_parameters: torch.Tensor):
        """reference size (B, G) of each face-shape scale group, see anny.faces.measurements"""
        from anny.faces.measurements import face_shape_group_sizes

        landmarks = self.phenotype_craniofacial_landmarks(phenotype_parameters)
        return face_shape_group_sizes(landmarks, self.craniofacial_landmark_labels)

    def face_shape_scales(
        self,
        phenotype_kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Scale (B, G) of each face-shape group (``anny.models.face_shapes.SCALE_GROUPS``): the size
        of the matching part of the head for these phenotypes, divided by its size on anny's
        default body.
        """
        phenotype_parameters = self._parse_parameter_kwargs(
            phenotype_kwargs, self.phenotype_labels, 0.5, "phenotype_kwargs"
        )
        return self._face_shape_scales(phenotype_parameters)

    def _face_shape_scales(self, phenotype_parameters: torch.Tensor) -> torch.Tensor:
        if self.face_shape_reference_sizes is None:
            return phenotype_parameters.new_ones(
                (phenotype_parameters.shape[0], len(SCALE_GROUPS))
            )
        sizes = self._face_shape_group_sizes(phenotype_parameters)
        return sizes / self.face_shape_reference_sizes[None]

    def _face_shape_coefficients(
        self, phenotype_parameters: torch.Tensor, face_shapes: torch.Tensor
    ) -> torch.Tensor:
        """coefficients of the face-shape rows: the rectified values times the group scales"""
        values = (
            face_shapes[:, self.face_shape_row_parameter] * self.face_shape_row_sign
        )
        # Rectified so that the gradient at zero is 1 rather than 0 (as for local changes)
        weights = values * (values >= 0).to(values.dtype)
        scales = self._face_shape_scales(phenotype_parameters)
        return weights * scales[:, self.face_shape_row_scale_group]

    def _parse_parameter_kwargs(
        self,
        kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None,
        labels: list[str],
        default: float,
        name: str,
    ) -> torch.Tensor:
        if kwargs is None:
            return default * torch.ones(
                (1, len(labels)), dtype=self.dtype, device=self.device
            )
        if isinstance(kwargs, dict):
            unknown = set(kwargs) - set(labels)
            if unknown:
                raise ValueError(f"Invalid {name}: {unknown}; available: {labels}")
        if isinstance(kwargs, torch.Tensor):
            if kwargs.dim() != 2 or kwargs.shape[1] != len(labels):
                raise ValueError(
                    f"{name} tensor must have shape [B, {len(labels)}], got {tuple(kwargs.shape)}."
                )
            return kwargs
        values = [
            to_batched_tensor(kwargs.get(label, default), self.device, self.dtype)
            for label in labels
        ]
        batch_size = max((value.shape[0] for value in values), default=1)
        if not values:
            return torch.empty((batch_size, 0), dtype=self.dtype, device=self.device)
        return torch.stack([value.expand(batch_size) for value in values], dim=1)

    def get_phenotype_blendshape_coefficients(
        self,
        gender: float | torch.Tensor = 0.5,
        age: float | torch.Tensor = 0.5,
        muscle: float | torch.Tensor = 0.5,
        weight: float | torch.Tensor = 0.5,
        height: float | torch.Tensor = 0.5,
        proportions: float | torch.Tensor = 0.5,
        cupsize: float | torch.Tensor = 0.5,
        firmness: float | torch.Tensor = 0.5,
        african: float | torch.Tensor = 0.5,
        asian: float | torch.Tensor = 0.5,
        caucasian: float | torch.Tensor = 0.5,
        local_changes: dict[str, float | torch.Tensor] | None = None,
    ):
        """
        Return blendshape coefficients corresponding to the input phenotype description.
        Deprecated but kept for compatibility with SOMA.
        """
        phenotype_parameters = self._parse_parameter_kwargs(
            {
                "gender": gender,
                "age": age,
                "muscle": muscle,
                "weight": weight,
                "height": height,
                "proportions": proportions,
                "cupsize": cupsize,
                "firmness": firmness,
                "african": african,
                "asian": asian,
                "caucasian": caucasian,
            },
            PHENOTYPE_LABELS,
            0.5,
            "phenotype_kwargs",
        )
        local_changes_parameters = self._parse_parameter_kwargs(
            local_changes, self.local_change_labels, 0.0, "local_changes_kwargs"
        )
        return self._get_phenotype_blendshape_coefficients(
            phenotype_parameters,
            local_changes_parameters,
            torch.zeros(
                phenotype_parameters.shape[0],
                len(self.facial_action_labels),
                dtype=phenotype_parameters.dtype,
                device=phenotype_parameters.device,
            ),
        )

    def _phenotype_block_coefficients(
        self, phenotype_parameters: torch.Tensor
    ) -> torch.Tensor:
        """coefficients (B, N) of the phenotype blend shapes (the first N rows)"""
        phenotype_labels = self.phenotype_labels
        batch_size = phenotype_parameters.shape[0]

        def phenotype_value(label: str) -> torch.Tensor:
            if label in phenotype_labels:
                return phenotype_parameters[:, phenotype_labels.index(label)]
            return phenotype_parameters.new_full((batch_size,), 0.5)

        weight_dicts = {}
        for feature in [
            "age",
            "gender",
            "muscle",
            "weight",
            "height",
            "proportions",
            "cupsize",
            "firmness",
        ]:
            interpolation_coeffs = (
                anny.utils.interpolation.linear_interpolation_coefficients(
                    phenotype_value(feature),
                    self.anchors[feature],
                    extrapolate=self.extrapolate_phenotypes,
                )
            )
            weight_dicts[feature] = {
                key: interpolation_coeffs[:, i]
                for i, key in enumerate(PHENOTYPE_VARIATIONS[feature])
            }

        race_values = torch.stack(
            [phenotype_value(key) for key in ("african", "asian", "caucasian")],
            dim=1,
        )
        race_weights = torch.nan_to_num(
            race_values / torch.sum(race_values, dim=1, keepdim=True),
            1 / 3,
            1 / 3,
            1 / 3,
        )

        dict_phens = {
            **weight_dicts["age"],
            **weight_dicts["gender"],
            **weight_dicts["muscle"],
            **weight_dicts["weight"],
            **weight_dicts["height"],
            **weight_dicts["proportions"],
            **weight_dicts["cupsize"],
            **weight_dicts["firmness"],
            "african": race_weights[:, 0],
            "asian": race_weights[:, 1],
            "caucasian": race_weights[:, 2],
        }
        phens = torch.stack(
            [
                dict_phens[key]
                for key_list in PHENOTYPE_VARIATIONS.values()
                for key in key_list
            ],
            dim=1,
        )

        mask = self.stacked_phenotype_blend_shapes_mask.unsqueeze(0)
        return torch.prod(phens.unsqueeze(1) * mask + (1 - mask), dim=-1)

    def _get_phenotype_blendshape_coefficients(
        self,
        phenotype_parameters: torch.Tensor,
        local_changes: torch.Tensor,
        facial_actions: torch.Tensor,
        face_shapes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = max(
            phenotype_parameters.shape[0],
            local_changes.shape[0],
            facial_actions.shape[0],
            1 if face_shapes is None else face_shapes.shape[0],
        )
        phenotype_parameters = phenotype_parameters.expand(batch_size, -1)
        local_changes = local_changes.expand(batch_size, -1)

        coefficient_groups = [self._phenotype_block_coefficients(phenotype_parameters)]
        if self.facial_action_labels:
            coefficient_groups.append(facial_actions.expand(batch_size, -1))

        if self.local_change_labels:
            local_weights = torch.zeros(
                (batch_size, 2 * len(self.local_change_labels)),
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(len(self.local_change_labels)):
                value = local_changes[:, i]
                # ReLU written so that the gradient at zero is 1 rather than 0,
                # to avoid dead local change parameters.
                local_weights[:, 2 * i] = value * (value >= 0).to(value.dtype)
                local_weights[:, 2 * i + 1] = -value * (-value >= 0).to(value.dtype)
            coefficient_groups.append(local_weights)

        if self.face_shape_labels:
            if face_shapes is None:
                face_shapes = phenotype_parameters.new_zeros(
                    (batch_size, len(self.face_shape_labels))
                )
            coefficient_groups.append(
                self._face_shape_coefficients(
                    phenotype_parameters, face_shapes.expand(batch_size, -1)
                )
            )

        return torch.cat(coefficient_groups, dim=1)

    def parse_phenotype_kwargs(
        self, phenotype_kwargs: dict[str, float | torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        For backward compatibility only. Used by SOMA.
        """
        tensor = self._parse_parameter_kwargs(
            phenotype_kwargs, self.phenotype_labels, 0.5, "phenotype_kwargs"
        )
        return {k: tensor[:, i] for (i, k) in enumerate(self.phenotype_labels)}

    def get_tensor_inputs(
        self,
        pose_parameters: dict[str, torch.Tensor]
        | torch.Tensor
        | tuple[torch.Tensor, ...]
        | None,
        phenotype_kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None,
        local_changes_kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None,
        facial_actions: dict[str, float | torch.Tensor] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_parameters = self.parse_delta_transforms_dict(pose_parameters)
        phenotype_parameters = self._parse_parameter_kwargs(
            phenotype_kwargs,
            self.phenotype_labels,
            0.5,
            "phenotype_kwargs",
        )
        local_change_parameters = self._parse_parameter_kwargs(
            local_changes_kwargs,
            self.local_change_labels,
            0.0,
            "local_changes_kwargs",
        )
        facial_action_parameters = self._parse_parameter_kwargs(
            facial_actions,
            self.facial_action_labels,
            0.0,
            "facial_actions",
        )
        return (
            pose_parameters,
            phenotype_parameters,
            local_change_parameters,
            facial_action_parameters,
        )

    def forward(
        self,
        pose_parameters: torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[torch.Tensor, ...]
        | None = None,
        phenotype_kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None = None,
        local_changes_kwargs: dict[str, float | torch.Tensor]
        | torch.Tensor
        | None = None,
        facial_actions: dict[str, float | torch.Tensor] | torch.Tensor | None = None,
        pose_parameterization: PoseParameterization | None = None,
        return_bone_ends: bool = False,
        face_shape_kwargs: dict[str, float | torch.Tensor] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        face_shape_kwargs: values of the face-shape parameters (a dict by name, or a [B, F]
            tensor in the order of ``face_shape_labels``); missing names take 0.
        """
        (
            pose_parameters,
            phenotype_parameters,
            local_change_parameters,
            facial_action_parameters,
        ) = self.get_tensor_inputs(
            pose_parameters,
            phenotype_kwargs,
            local_changes_kwargs,
            facial_actions,
        )
        face_shape_parameters = self._parse_parameter_kwargs(
            face_shape_kwargs, self.face_shape_labels, 0.0, "face_shape_kwargs"
        )
        blendshape_coeffs = self._get_phenotype_blendshape_coefficients(
            phenotype_parameters,
            local_change_parameters,
            facial_action_parameters,
            face_shape_parameters,
        )
        return super().forward(
            pose_parameters,
            blendshape_coeffs,
            pose_parameterization=pose_parameterization,
            return_bone_ends=return_bone_ends,
        )

    def to_model_data(self) -> "ModelData":
        model_data = super().to_model_data()
        return dataclasses.replace(
            model_data,
            metadata=dataclasses.replace(
                model_data.metadata,
                craniofacial_landmark_labels=list(self.craniofacial_landmark_labels),
            ),
            stacked_phenotype_blend_shapes_mask=self.stacked_phenotype_blend_shapes_mask,
            craniofacial_landmarks=self.craniofacial_landmarks_template,
            craniofacial_landmarks_blendshapes=self.craniofacial_landmarks_blendshapes,
        )
