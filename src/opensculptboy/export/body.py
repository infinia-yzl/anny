# OpenSculptBoy
# Apache License, Version 2.0
"""
The body mesh of an Anny character as glTF vertex data, shared by the GLB and VRM exporters.

:func:`morph_target_rows` lays out the parameter rows of a character and its morph targets,
and :func:`rest_rows` evaluates the rest mesh of each row. :func:`build_body` takes welded
vertices in Anny's frame (the rest pose, or the T-pose bind mesh of a VRM file) and the offsets
of the morph targets, and returns the glTF vertex data in an output frame: the vertices split
at UV seams, normals from the welded mesh, UVs, the strongest skin weights and the target
offsets. :func:`body_primitive` writes that data into a
:class:`~opensculptboy.export.document.GltfDocument`; several primitives of one body (parts
with their own materials) can share its vertex accessors and targets
(:func:`body_attributes`, :func:`body_targets`).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
import torch

from opensculptboy.export.document import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    GltfDocument,
)

# (x, y, z) -> (x, z, -y): Anny's Z-up frame to glTF's Y-up frame, the figure facing +Z.
ANNY_TO_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


@dataclasses.dataclass
class BodyMesh:
    """
    glTF vertex data of a body mesh, in the output frame.

    ``source`` maps each glTF vertex to its Anny vertex; ``triangles`` index the glTF vertices,
    and ``triangle_faces`` maps each triangle to its row of ``triangulated_faces(model)``.
    ``targets`` holds one (N, 3) offset array per morph target, in the order given to
    :func:`build_body`.
    """

    positions: np.ndarray  # (N, 3) float64
    normals: np.ndarray  # (N, 3) float64, unit length
    uvs: np.ndarray | None  # (N, 2) float64, glTF convention (v = 0 at the top)
    source: np.ndarray  # (N,) int64
    triangles: np.ndarray  # (T, 3) int64
    triangle_faces: np.ndarray  # (T,) int64
    joints: np.ndarray  # (N, k) int64
    weights: np.ndarray  # (N, k) float32, rows sum to 1
    targets: list[np.ndarray]  # each (N, 3) float64
    dropped_weight: (
        float  # the largest skin weight left out by the truncation to k bones
    )


@dataclasses.dataclass
class MorphTarget:
    """
    A morph target of a character's body: one parameter of the model set to ``value``.

    ``block`` is ``"facial_action"`` or ``"face_shape"`` and ``index`` the parameter's index in
    ``model.facial_action_labels`` or ``model.face_shape_labels``; ``weight`` is the default
    weight of the target in the file, from the character's own value.
    """

    name: str
    block: str
    index: int
    value: float
    weight: float


def morph_target_rows(
    model, character, names: Sequence[str]
) -> tuple[list[MorphTarget], np.ndarray, np.ndarray]:
    """
    The morph targets of a character for parameter names, and the parameter rows that give
    them: row 0 is the base mesh, row r + 1 the mesh of target r.

    A facial action gives one target (weight 1 is the value 1). A face shape gives
    ``<name>.pos`` at +1 and, when the shape runs below 0, ``<name>.neg`` at -1. The character's
    values become the default weights; the parameters that are targets are 0 in the base row,
    and the other face shapes and facial actions of the character stay baked into every row.

    Returns:
        The targets, the facial action rows (R + 1, A) and the face-shape rows (R + 1, F),
        both float64.
    """
    facial_labels = list(model.facial_action_labels)
    face_labels = list(model.face_shape_labels)
    base_actions = np.array(
        [character.facial_actions.get(n, 0.0) for n in facial_labels], dtype=np.float64
    )
    base_faces = np.array(
        [character.face_shapes.get(n, 0.0) for n in face_labels], dtype=np.float64
    )
    targets: list[MorphTarget] = []
    for name in names:
        if name in facial_labels:
            i = facial_labels.index(name)
            targets.append(
                MorphTarget(name, "facial_action", i, 1.0, float(base_actions[i]))
            )
            base_actions[i] = 0.0
        else:
            i = face_labels.index(name)
            value = float(base_faces[i])
            targets.append(
                MorphTarget(f"{name}.pos", "face_shape", i, 1.0, max(value, 0.0))
            )
            if model.face_shape_ranges[name][0] < 0:
                targets.append(
                    MorphTarget(f"{name}.neg", "face_shape", i, -1.0, max(-value, 0.0))
                )
            base_faces[i] = 0.0
    facial_rows = np.repeat(base_actions[None], len(targets) + 1, axis=0)
    face_rows = np.repeat(base_faces[None], len(targets) + 1, axis=0)
    for row, target in enumerate(targets, start=1):
        rows = facial_rows if target.block == "facial_action" else face_rows
        rows[row, target.index] = target.value
    return targets, facial_rows, face_rows


def rest_rows(
    model, character, facial_rows, face_rows, chunk: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rest vertices (B, V, 3) and the rest bone poses (J, 4, 4) of the first row, both float64,
    for rows of facial action values (B, A) and face-shape values (B, F), with the phenotype
    and the local changes of the character. The rows run through the model ``chunk`` at a time.
    """
    dtype = model.template_vertices.dtype
    facial_rows = torch.as_tensor(np.asarray(facial_rows), dtype=dtype)
    face_rows = torch.as_tensor(np.asarray(face_rows), dtype=dtype)
    kwargs = dict(
        phenotype_kwargs=dict(character.phenotype) or None,
        local_changes_kwargs=dict(character.local_changes) or None,
    )
    vertices, bone_poses = [], None
    with torch.no_grad():
        for start in range(0, len(facial_rows), chunk):
            stop = start + chunk
            call = dict(kwargs)
            if model.facial_action_labels:
                call["facial_actions"] = facial_rows[start:stop]
            if model.face_shape_labels:
                call["face_shape_kwargs"] = face_rows[start:stop]
            out = model(**call)
            vertices.append(out["rest_vertices"].double().cpu())
            if bone_poses is None:
                bone_poses = out["rest_bone_poses"][0].double().cpu().numpy()
    return torch.cat(vertices).numpy(), bone_poses


def triangulated_faces(model) -> tuple[np.ndarray, np.ndarray | None]:
    """Vertex and UV index triangles of a model; quads split along the same diagonal in both."""
    faces = model.faces.cpu().numpy()
    uv_faces = model.face_texture_coordinate_indices
    uv_faces = None if uv_faces is None else uv_faces.cpu().numpy()
    if faces.shape[1] == 3:
        return faces, uv_faces
    if faces.shape[1] != 4:
        raise ValueError(f"Faces with {faces.shape[1]} corners are not supported.")

    def split(f):
        return np.concatenate([f[:, [0, 1, 2]], f[:, [0, 2, 3]]])

    return split(faces), None if uv_faces is None else split(uv_faces)


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals of a welded triangle mesh."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    face_normals = np.cross(b - a, c - a)  # length is twice the area: area weighting
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def top_skin_weights(
    weights: np.ndarray, indices: np.ndarray, max_influences: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    The strongest ``max_influences`` bones of each vertex, renormalised, and the largest dropped
    weight. ``weights`` and ``indices`` are (V, K) arrays as in ``model.vertex_bone_weights``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    indices = np.asarray(indices)
    order = np.argsort(-weights, axis=1, kind="stable")
    weights = np.take_along_axis(weights, order, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    dropped = (
        float(weights[:, max_influences:].max())
        if weights.shape[1] > max_influences
        else 0.0
    )
    weights, indices = weights[:, :max_influences], indices[:, :max_influences]
    if weights.shape[1] < max_influences:
        pad = max_influences - weights.shape[1]
        weights = np.pad(weights, ((0, 0), (0, pad)))
        indices = np.pad(indices, ((0, 0), (0, pad)))
    weights = weights / weights.sum(axis=1, keepdims=True)
    indices = np.where(weights > 0, indices, 0)
    return weights.astype(np.float32), indices.astype(np.int64), dropped


def build_body(
    model,
    vertices: np.ndarray,
    target_offsets: Sequence[np.ndarray] | np.ndarray = (),
    frame: np.ndarray = ANNY_TO_GLTF,
    max_influences: int = 4,
    vertex_bone_weights: np.ndarray | None = None,
    vertex_bone_indices: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> BodyMesh:
    """
    glTF vertex data of a body mesh.

    Args:
        model: the Anny model (faces, UVs and, by default, skin weights).
        vertices: (V, 3) welded vertices in Anny's frame: the rest pose, or a bind mesh.
        target_offsets: (R, V, 3) morph target offsets in Anny's frame, one per target.
        frame: (3, 3) rotation from Anny's frame to the output frame; positions and offsets
            become ``p @ frame.T``. Translations (ground, centring) belong to the caller.
        max_influences: bones kept per vertex.
        vertex_bone_weights, vertex_bone_indices: (V, K) skin weights to use in place of the
            model's own (for example the VRM weights with the eyelids moved to the head).
        normals: (V, 3) welded unit normals in Anny's frame, in place of the normals of
            ``vertices`` (for example toon face normals).

    Returns:
        A :class:`BodyMesh`; normals come from the welded ``vertices`` unless given.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.float64)
    triangles, uv_triangles = triangulated_faces(model)
    if normals is None:
        normals = vertex_normals(vertices, triangles)
    else:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != vertices.shape:
            raise ValueError(
                f"normals of shape {normals.shape} for vertices of shape {vertices.shape}."
            )
    triangle_faces = np.arange(len(triangles))
    if uv_triangles is not None and model.texture_coordinates is not None:
        pairs = np.stack([triangles.reshape(-1), uv_triangles.reshape(-1)], axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        source, uv_source = unique[:, 0], unique[:, 1]
        split = inverse.reshape(-1, 3)
        uvs = model.texture_coordinates.detach().cpu().double().numpy()[uv_source]
        uvs = np.stack(
            [uvs[:, 0], 1.0 - uvs[:, 1]], axis=1
        )  # glTF puts v = 0 at the top
    else:
        source, uvs, split = np.arange(len(vertices)), None, triangles
    if vertex_bone_weights is None:
        vertex_bone_weights = model.vertex_bone_weights.detach().cpu().double().numpy()
    if vertex_bone_indices is None:
        vertex_bone_indices = model.vertex_bone_indices.cpu().numpy()
    weights, joints, dropped = top_skin_weights(
        vertex_bone_weights, vertex_bone_indices, max_influences
    )
    offsets = [np.asarray(t, dtype=np.float64) for t in target_offsets]
    return BodyMesh(
        positions=(vertices @ frame.T)[source],
        normals=(normals @ frame.T)[source],
        uvs=uvs,
        source=source.astype(np.int64),
        triangles=split.astype(np.int64),
        triangle_faces=triangle_faces,
        joints=joints[source],
        weights=weights[source],
        targets=[(t @ frame.T)[source] for t in offsets],
        dropped_weight=dropped,
    )


def body_attributes(
    doc: GltfDocument,
    body: BodyMesh,
    translation: np.ndarray | None = None,
    anny_vertex: bool = True,
    joint_type=None,
) -> dict:
    """
    Write the vertex data of a body mesh into a document and return the attributes of its
    primitives: positions (plus ``translation``), normals, UVs, ``_ANNY_VERTEX`` and the
    JOINTS/WEIGHTS sets, 4 bones each. ``joint_type`` is the integer type of the joint indices;
    by default uint8 when they all fit, uint16 otherwise.
    """
    positions = body.positions + (0.0 if translation is None else translation)
    attributes = {
        "POSITION": doc.accessor(
            positions.astype(np.float32), ARRAY_BUFFER, minmax=True
        ),
        "NORMAL": doc.accessor(body.normals.astype(np.float32), ARRAY_BUFFER),
    }
    if body.uvs is not None:
        attributes["TEXCOORD_0"] = doc.accessor(
            body.uvs.astype(np.float32), ARRAY_BUFFER
        )
    if anny_vertex:
        # The Anny vertex of each glTF vertex (float: glTF allows unsigned int for indices
        # only), so that keypoint regressors and other per-vertex data of Anny apply to the file.
        attributes["_ANNY_VERTEX"] = doc.accessor(
            body.source.astype(np.float32), ARRAY_BUFFER
        )
    if joint_type is None:
        joint_type = np.uint8 if int(body.joints.max(initial=0)) < 256 else np.uint16
    for s in range(body.joints.shape[1] // 4):
        cols = slice(4 * s, 4 * s + 4)
        attributes[f"JOINTS_{s}"] = doc.accessor(
            body.joints[:, cols].astype(joint_type), ARRAY_BUFFER
        )
        attributes[f"WEIGHTS_{s}"] = doc.accessor(body.weights[:, cols], ARRAY_BUFFER)
    return attributes


def body_targets(
    doc: GltfDocument, body: BodyMesh, sparse_targets: bool = True
) -> list[dict]:
    """Write the morph targets of a body mesh into a document; returns the primitive targets."""
    return [
        {"POSITION": doc.morph_accessor(t, sparse=sparse_targets)} for t in body.targets
    ]


def body_primitive(
    doc: GltfDocument,
    body: BodyMesh,
    material: int,
    translation: np.ndarray | None = None,
    triangles: np.ndarray | None = None,
    anny_vertex: bool = True,
    sparse_targets: bool = True,
    joint_type=None,
    attributes: dict | None = None,
    targets: list[dict] | None = None,
) -> dict:
    """
    Write a body mesh into a document and return its glTF primitive.

    ``translation`` is added to the positions (for example the ground and centring offset of a
    VRM file). ``triangles`` selects a subset of ``body.triangles`` (a part of the body); all
    vertices stay in every primitive, so the morph targets are the same in each. The
    ``_ANNY_VERTEX`` attribute maps each vertex to its Anny vertex. ``attributes`` and
    ``targets`` from an earlier primitive of the same body (its ``"attributes"`` and
    ``"targets"``, or :func:`body_attributes` and :func:`body_targets`) are shared instead of
    written again; the accessors are written in the order attributes, indices, targets.
    """
    if attributes is None:
        attributes = body_attributes(doc, body, translation, anny_vertex, joint_type)
    tris = body.triangles if triangles is None else triangles
    index_type = np.uint16 if len(body.source) < 65536 else np.uint32
    primitive = {
        "attributes": dict(attributes),
        "indices": doc.accessor(
            np.asarray(tris).reshape(-1).astype(index_type), ELEMENT_ARRAY_BUFFER
        ),
        "material": material,
        "mode": 4,
    }
    if targets is None and body.targets:
        targets = body_targets(doc, body, sparse_targets)
    if targets:
        primitive["targets"] = [dict(t) for t in targets]
    return primitive
