# OpenSculptBoy
# Apache License, Version 2.0
"""
glTF 2.0 binary (GLB) export of an Anny character.

The file holds the character's mesh in its rest (bind) pose with normals and UVs, the bones of
the rig as joint nodes with inverse bind matrices, the skin weights (the strongest 4 or 8 bones
of each vertex), morph targets for the facial actions and, on request, for the face shapes, and
one animation for each pose or clip of ``anny.poses`` that the caller names, after an animation
named ``pose`` when the character holds its own pose. The phenotype, the
local changes and the face shapes that are not morph targets are baked into the mesh and the
skeleton. The settings of the character are stored in the ``extras`` of the scene, and
``read_character`` reads them back, from GLB and VRM files alike.

The exporter builds the file with :class:`~opensculptboy.export.document.GltfDocument` and the
body data of :mod:`opensculptboy.export.body`, which the VRM exporter shares.

Frame: Anny works in metres with Z up and the figure facing -Y. glTF works in metres with Y up
and the figure facing +Z. Every position p becomes C p, and every joint matrix M becomes
C M C^T, where C is the rotation (x, y, z) -> (x, z, -y). The skinned result is then C times
Anny's skinned vertices.
"""

from __future__ import annotations

import difflib
import pathlib
from typing import Iterable, Sequence

import numpy as np
import roma
import torch

from opensculptboy.character import Character
from opensculptboy.export.body import (
    ANNY_TO_GLTF,
    body_primitive,
    build_body,
    morph_target_rows,
    rest_rows,
)
from opensculptboy.export.document import GltfDocument, pbr_material, read_gltf_json

__all__ = ["C3", "C4", "export_glb", "read_character", "read_gltf_json"]

# (x, y, z) -> (x, z, -y): Anny's Z-up frame to glTF's Y-up frame.
C3 = ANNY_TO_GLTF.copy()
C4 = np.eye(4)
C4[:3, :3] = C3


def _check_names(names: Iterable[str], allowed: Sequence[str], what: str) -> None:
    for name in names:
        if name not in allowed:
            close = difflib.get_close_matches(name, allowed, n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise ValueError(f"Unknown {what} {name!r}.{hint}")


def _to_gltf_matrices(matrices: np.ndarray) -> np.ndarray:
    """Joint matrices (..., 4, 4) of Anny's frame in glTF's frame: C M C^T."""
    return C4 @ matrices @ C4.T


def _local_transforms(world: np.ndarray, parents: Sequence[int]) -> np.ndarray:
    """Local matrices (..., J, 4, 4) of joints from their world matrices."""
    local = world.copy()
    for j, parent in enumerate(parents):
        if parent >= 0:
            local[..., j, :, :] = (
                np.linalg.inv(world[..., parent, :, :]) @ world[..., j, :, :]
            )
    return local


def _quaternions(rotations: np.ndarray) -> np.ndarray:
    """Unit quaternions (x, y, z, w) of rotation matrices (..., 3, 3)."""
    return roma.rotmat_to_unitquat(
        torch.from_numpy(np.ascontiguousarray(rotations))
    ).numpy()


def export_glb(
    path: str | pathlib.Path,
    character: Character | None = None,
    model=None,
    animations: Sequence[str] = (),
    morph_targets: Sequence[str] | None = None,
    max_influences: int = 4,
    ground: bool = True,
    base_color: Sequence[float] = (0.80, 0.62, 0.52, 1.0),
) -> dict:
    """
    Write a character as a binary glTF 2.0 file.

    Args:
        path: the ``.glb`` file to write.
        character: the character; the default character when None.
        model: the Anny model to use; ``character.build_model()`` when None. It must have
            ``facial_actions="all"`` for facial-action morph targets, and face shapes for
            face-shape morph targets.
        animations: names of poses and clips of ``anny.poses`` (``anny.poses.names()``), one
            animation each. A pose is one keyframe; a clip plays at its own frame rate. A
            character with a ``pose`` also gets a one-keyframe animation named ``pose``, first.
        morph_targets: names of facial actions and face shapes to export as morph targets.
            None exports every facial action of the model. A facial action gives one target
            (weight 1 is the value 1). A face shape gives ``<name>.pos`` at +1 and, when the
            shape runs below 0, ``<name>.neg`` at -1. The character's values become the default
            weights, and the other face shapes are baked into the mesh.
        max_influences: bones per vertex, 4 (read by every engine) or 8 (a second
            JOINTS/WEIGHTS set). Anny uses up to 9; the weaker ones are dropped and the rest
            renormalised.
        ground: stand the rest pose on the floor (y = 0). The poses and clips of
            ``anny.poses`` stand on the floor in any case.
        base_color: the linear RGBA base colour of the material.

    Returns:
        A summary: counts, the file size and the largest dropped skin weight.
    """
    if max_influences not in (4, 8):
        raise ValueError(f"max_influences must be 4 or 8, got {max_influences}.")
    character = character or Character()
    if model is None:
        wants_face = morph_targets is not None and bool(
            set(morph_targets) & set(_face_shape_names())
        )
        model = character.build_model(
            face_shapes=bool(character.face_shapes) or wants_face
        )
    facial_labels = list(model.facial_action_labels)
    face_labels = list(model.face_shape_labels)
    if morph_targets is None:
        morph_targets = facial_labels
    _check_names(morph_targets, facial_labels + face_labels, "morph target")
    _check_names(character.phenotype, list(model.phenotype_labels), "phenotype")
    _check_names(
        character.local_changes, list(model.local_change_labels), "local change"
    )
    _check_names(character.face_shapes, face_labels, "face shape")
    _check_names(character.facial_actions, facial_labels, "facial action")
    if animations:
        import anny.poses

        _check_names(animations, anny.poses.names(), "pose or clip")

    # Rows of parameter values: row 0 is the base mesh, one more row per morph target.
    targets, facial_rows, face_rows = morph_target_rows(model, character, morph_targets)
    rest_vertices, rest_bone_poses = rest_rows(model, character, facial_rows, face_rows)
    base = rest_vertices[0]

    # Mesh: split the vertices at UV seams; normals come from the welded mesh.
    body = build_body(
        model, base, rest_vertices[1:] - base, ANNY_TO_GLTF, max_influences
    )
    doc = GltfDocument(f"OpenSculptBoy glTF exporter (anny {_anny_version()})")
    material = doc.add_material(pbr_material("skin", base_color, 0.0, 0.6))
    joint_type = np.uint8 if model.bone_count <= 256 else np.uint16
    primitive = body_primitive(doc, body, material, joint_type=joint_type)
    mesh = {"name": character.name, "primitives": [primitive]}
    if targets:
        mesh["weights"] = [t.weight for t in targets]
        mesh["extras"] = {"targetNames": [t.name for t in targets]}

    # Skeleton: the bind pose is the rest pose of the base character.
    parents = [int(p) for p in model.bone_parents]
    labels = list(model.bone_labels)
    bind = _to_gltf_matrices(rest_bone_poses)
    lift = np.eye(4)
    if ground:
        # the lowest vertex as the file stores it (float32), on the floor
        lift[1, 3] = -float(body.positions[:, 1].astype(np.float32).min())
    default_local = _local_transforms(lift @ bind, parents)
    for j, label in enumerate(labels):
        node = {
            "name": label,
            "translation": default_local[j, :3, 3].tolist(),
            "rotation": _quaternions(default_local[j, :3, :3]).tolist(),
        }
        children = [c for c, p in enumerate(parents) if p == j]
        if children:
            node["children"] = children
        doc.add_node(node)
    mesh_index = doc.add_mesh(mesh)
    roots = [j for j, p in enumerate(parents) if p < 0]
    inverse_bind = np.linalg.inv(bind).transpose(0, 2, 1).reshape(len(labels), 16)
    skin = doc.add_skin(
        {
            "name": character.rig,
            "joints": list(range(len(labels))),
            "skeleton": roots[0],
            "inverseBindMatrices": doc.accessor(inverse_bind.astype(np.float32)),
        }
    )
    mesh_node = doc.add_node({"name": character.name, "mesh": mesh_index, "skin": skin})

    if character.pose:
        doc.add_animation(
            _character_pose(doc, model, character, parents, default_local)
        )
    for name in animations:
        doc.add_animation(
            _animation(doc, model, character, name, parents, default_local)
        )

    doc.scene_name = character.name
    doc.scene_nodes = roots + [mesh_node]
    doc.scene_extras = {
        "opensculptboy": {
            "generator": "opensculptboy",
            "anny_version": _anny_version(),
            "character": character.to_dict(),
            "morph_targets": [t.name for t in targets],
            "animations": (["pose"] if character.pose else []) + list(animations),
            "max_influences": max_influences,
            "frame": "metres, Y up, the figure faces +Z",
        }
    }
    size = doc.write(path)
    return dict(
        path=str(path),
        bytes=size,
        vertices=int(len(body.source)),
        triangles=int(len(body.triangles)),
        joints=len(labels),
        morph_targets=len(targets),
        animations=len(doc.animations),
        max_influences=max_influences,
        largest_dropped_weight=body.dropped_weight,
    )


def _animation(
    doc: GltfDocument, model, character: Character, name: str, parents, default_local
) -> dict:
    """One glTF animation for a pose or a clip of ``anny.poses``."""
    import anny.poses

    entry = anny.poses.pose_parameters(
        model,
        name,
        phenotype_kwargs=dict(character.phenotype) or None,
        local_changes_kwargs=dict(character.local_changes) or None,
    )
    return _keyframes(
        doc,
        model,
        character,
        name,
        entry["pose_parameters"],
        float(entry["fps"]),
        parents,
        default_local,
    )


def _character_pose(
    doc: GltfDocument, model, character: Character, parents, default_local
) -> dict:
    """The character's own pose as a one-frame animation named ``pose``, on the floor."""
    import anny.poses

    params, _ = anny.poses.ground(
        model,
        character.pose_parameters(model),
        phenotype_kwargs=dict(character.phenotype) or None,
        local_changes_kwargs=dict(character.local_changes) or None,
    )
    return _keyframes(
        doc, model, character, "pose", params, 1.0, parents, default_local
    )


def _keyframes(
    doc: GltfDocument,
    model,
    character: Character,
    name: str,
    pose_parameters: torch.Tensor,
    fps: float,
    parents,
    default_local,
) -> dict:
    """One glTF animation for ``local-ref`` pose parameters (frames, bones, 4, 4)."""
    with torch.no_grad():
        out = model(
            pose_parameters=pose_parameters,
            pose_parameterization="local-ref",
            **character.model_kwargs(),
        )
    world = _to_gltf_matrices(out["bone_poses"].double().cpu().numpy())
    local = _local_transforms(world, parents)
    frames = local.shape[0]
    times = (np.arange(frames) / fps).astype(np.float32)
    time_accessor = doc.accessor(times, minmax=True)
    samplers, channels = [], []
    for j in range(len(parents)):
        quats = _quaternions(local[:, j, :3, :3])
        for f in range(1, frames):  # the shortest path between keyframes
            if np.dot(quats[f], quats[f - 1]) < 0:
                quats[f] = -quats[f]
        tracks = [("rotation", quats)]
        translations = local[:, j, :3, 3]
        if np.abs(translations - default_local[j, :3, 3]).max() > 1e-6:
            tracks.append(("translation", translations))
        for path, values in tracks:
            samplers.append(
                {
                    "input": time_accessor,
                    "output": doc.accessor(values.astype(np.float32)),
                    "interpolation": "LINEAR",
                }
            )
            channels.append(
                {"sampler": len(samplers) - 1, "target": {"node": j, "path": path}}
            )
    return {"name": name, "samplers": samplers, "channels": channels}


def read_character(path: str | pathlib.Path) -> Character:
    """
    The character stored in a GLB or VRM file written by OpenSculptBoy (``export_glb`` or
    ``export_vrm``): the ``opensculptboy`` entry of the extras of a scene, of the root object
    or of the asset.
    """
    gltf = read_gltf_json(path)
    holders = [scene.get("extras") for scene in gltf.get("scenes", [])]
    holders += [gltf.get("extras"), gltf.get("asset", {}).get("extras")]
    for extras in holders:
        metadata = extras.get("opensculptboy") if isinstance(extras, dict) else None
        if isinstance(metadata, dict) and "character" in metadata:
            return Character.from_dict(metadata["character"])
    raise ValueError(
        f"{path} holds no OpenSculptBoy character (no 'opensculptboy' entry in its extras)."
    )


def _face_shape_names() -> list[str]:
    from anny.models.face_shapes import face_shape_spec

    return [p.name for p in face_shape_spec()]


def _anny_version() -> str:
    import anny

    return anny.__version__
