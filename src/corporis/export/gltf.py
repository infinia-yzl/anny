# Corporis
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
``read_character`` reads them back.

Frame: Anny works in metres with Z up and the figure facing -Y. glTF works in metres with Y up
and the figure facing +Z. Every position p becomes C p, and every joint matrix M becomes
C M C^T, where C is the rotation (x, y, z) -> (x, z, -y). The skinned result is then C times
Anny's skinned vertices.
"""

from __future__ import annotations

import difflib
import json
import pathlib
import struct
from typing import Iterable, Sequence

import numpy as np
import roma
import torch

from corporis.character import Character

# (x, y, z) -> (x, z, -y): Anny's Z-up frame to glTF's Y-up frame.
C3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
C4 = np.eye(4)
C4[:3, :3] = C3

_FLOAT, _UBYTE, _USHORT, _UINT = 5126, 5121, 5123, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
_COMPONENTS = {
    np.dtype(np.float32): _FLOAT,
    np.dtype(np.uint8): _UBYTE,
    np.dtype(np.uint16): _USHORT,
    np.dtype(np.uint32): _UINT,
}
# A morph target is stored sparse when it moves fewer than this share of the vertices.
_SPARSE_SHARE = 0.4
# Offsets at or below this length (metres) count as no movement: the rounding noise of two forward passes
# (about 1e-16, depending on the CPU's BLAS kernels) stays out of the file.
_MORPH_EPSILON = 1e-6
_GLB_MAGIC, _JSON_CHUNK, _BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942


class _Buffers:
    """The binary chunk of a GLB, with its buffer views and accessors."""

    def __init__(self):
        self.data = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, raw: bytes, target: int | None = None) -> int:
        self.data.extend(b"\0" * (-len(self.data) % 4))
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.data.extend(raw)
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def accessor(
        self,
        array: np.ndarray,
        target: int | None = None,
        minmax: bool = False,
        normalized: bool = False,
    ) -> int:
        array = np.ascontiguousarray(array)
        width = 1 if array.ndim == 1 else array.shape[1]
        accessor = {
            "bufferView": self.view(array.tobytes(), target),
            "componentType": _COMPONENTS[array.dtype],
            "count": int(array.shape[0]),
            "type": _TYPES[width],
        }
        if normalized:
            accessor["normalized"] = True
        if minmax:
            flat = array.reshape(array.shape[0], width)
            accessor["min"] = flat.min(axis=0).tolist()
            accessor["max"] = flat.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def morph_accessor(self, deltas: np.ndarray) -> int:
        """A VEC3 float accessor of position offsets, sparse when few vertices move."""
        deltas = np.ascontiguousarray(deltas, dtype=np.float32)
        still = np.linalg.norm(deltas, axis=1) <= _MORPH_EPSILON
        deltas[still] = 0.0
        moving = np.flatnonzero(~still)
        if len(moving) >= _SPARSE_SHARE * len(deltas):
            return self.accessor(deltas, minmax=True)
        if len(moving) == 0:
            moving = np.array([0])  # glTF needs at least one sparse element
        index_type = np.uint16 if len(deltas) < 65536 else np.uint32
        indices = moving.astype(index_type)
        accessor = {
            "componentType": _FLOAT,
            "count": int(len(deltas)),
            "type": "VEC3",
            "min": deltas.min(axis=0).tolist(),
            "max": deltas.max(axis=0).tolist(),
            "sparse": {
                "count": int(len(indices)),
                "indices": {
                    "bufferView": self.view(indices.tobytes()),
                    "componentType": _COMPONENTS[np.dtype(index_type)],
                },
                "values": {"bufferView": self.view(deltas[moving].tobytes())},
            },
        }
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _check_names(names: Iterable[str], allowed: Sequence[str], what: str) -> None:
    for name in names:
        if name not in allowed:
            close = difflib.get_close_matches(name, allowed, n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise ValueError(f"Unknown {what} {name!r}.{hint}")


def _triangles(model) -> tuple[np.ndarray, np.ndarray | None]:
    """Vertex and UV index triangles; quads split along the same diagonal in both."""
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


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    face_normals = np.cross(b - a, c - a)  # length is twice the area: area weighting
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def _skin_weights(model, max_influences: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The strongest ``max_influences`` bones of each vertex, renormalised, and the largest dropped weight."""
    weights = model.vertex_bone_weights.detach().cpu().double().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
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
    return weights.astype(np.float32), indices, dropped


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


def _rest_outputs(model, character: Character, facial_rows, face_rows, chunk: int = 32):
    """
    Rest vertices (B, V, 3) and rest bone poses of the first row, for rows of facial action
    values (B, A) and face-shape values (B, F), with the phenotype and the local changes of
    the character.
    """
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
    base_actions = np.array(
        [character.facial_actions.get(n, 0.0) for n in facial_labels]
    )
    base_faces = np.array([character.face_shapes.get(n, 0.0) for n in face_labels])
    targets: list[
        tuple[str, int, int, float, float]
    ] = []  # name, block, index, value, default weight
    for name in morph_targets:
        if name in facial_labels:
            i = facial_labels.index(name)
            targets.append((name, 0, i, 1.0, float(base_actions[i])))
            base_actions[i] = 0.0
        else:
            i = face_labels.index(name)
            value = float(base_faces[i])
            targets.append((f"{name}.pos", 1, i, 1.0, max(value, 0.0)))
            if model.face_shape_ranges[name][0] < 0:
                targets.append((f"{name}.neg", 1, i, -1.0, max(-value, 0.0)))
            base_faces[i] = 0.0
    facial_rows = np.repeat(base_actions[None], len(targets) + 1, axis=0)
    face_rows = np.repeat(base_faces[None], len(targets) + 1, axis=0)
    for row, (_, block, i, value, _) in enumerate(targets, start=1):
        (facial_rows if block == 0 else face_rows)[row, i] = value
    dtype = model.template_vertices.dtype
    rest_vertices, rest_bone_poses = _rest_outputs(
        model,
        character,
        torch.as_tensor(facial_rows, dtype=dtype),
        torch.as_tensor(face_rows, dtype=dtype),
    )
    base = rest_vertices[0]

    # Mesh: split the vertices at UV seams; normals come from the welded mesh.
    triangles, uv_triangles = _triangles(model)
    normals = _vertex_normals(base, triangles)
    if uv_triangles is not None and model.texture_coordinates is not None:
        pairs = np.stack([triangles.reshape(-1), uv_triangles.reshape(-1)], axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        source, uv_source = unique[:, 0], unique[:, 1]
        triangles = inverse.reshape(-1, 3)
        uvs = model.texture_coordinates.detach().cpu().double().numpy()[uv_source]
        uvs = np.stack(
            [uvs[:, 0], 1.0 - uvs[:, 1]], axis=1
        )  # glTF puts v = 0 at the top
    else:
        source, uvs = np.arange(len(base)), None

    buffers = _Buffers()
    positions = (base @ C3.T)[source].astype(np.float32)
    attributes = {
        "POSITION": buffers.accessor(positions, _ARRAY_BUFFER, minmax=True),
        "NORMAL": buffers.accessor(
            (normals @ C3.T)[source].astype(np.float32), _ARRAY_BUFFER
        ),
    }
    if uvs is not None:
        attributes["TEXCOORD_0"] = buffers.accessor(
            uvs.astype(np.float32), _ARRAY_BUFFER
        )
    # The Anny vertex of each glTF vertex (float: glTF allows unsigned int for indices only),
    # so that keypoint regressors and other per-vertex data of Anny apply to the file.
    attributes["_ANNY_VERTEX"] = buffers.accessor(
        source.astype(np.float32), _ARRAY_BUFFER
    )
    weights, joints, dropped = _skin_weights(model, max_influences)
    joint_type = np.uint8 if model.bone_count <= 256 else np.uint16
    for s in range(max_influences // 4):
        cols = slice(4 * s, 4 * s + 4)
        attributes[f"JOINTS_{s}"] = buffers.accessor(
            joints[source, cols].astype(joint_type), _ARRAY_BUFFER
        )
        attributes[f"WEIGHTS_{s}"] = buffers.accessor(
            weights[source, cols], _ARRAY_BUFFER
        )
    index_type = np.uint16 if len(source) < 65536 else np.uint32
    primitive = {
        "attributes": attributes,
        "indices": buffers.accessor(
            triangles.reshape(-1).astype(index_type), _ELEMENT_ARRAY_BUFFER
        ),
        "material": 0,
        "mode": 4,
    }
    mesh = {"name": character.name, "primitives": [primitive]}
    if targets:
        primitive["targets"] = [
            {
                "POSITION": buffers.morph_accessor(
                    ((rest_vertices[row] - base) @ C3.T)[source]
                )
            }
            for row in range(1, len(targets) + 1)
        ]
        mesh["weights"] = [t[4] for t in targets]
        mesh["extras"] = {"targetNames": [t[0] for t in targets]}

    # Skeleton: the bind pose is the rest pose of the base character.
    parents = [int(p) for p in model.bone_parents]
    labels = list(model.bone_labels)
    bind = _to_gltf_matrices(rest_bone_poses)
    lift = np.eye(4)
    if ground:
        lift[1, 3] = -float(positions[:, 1].min())
    default_local = _local_transforms(lift @ bind, parents)
    nodes = []
    for j, label in enumerate(labels):
        node = {
            "name": label,
            "translation": default_local[j, :3, 3].tolist(),
            "rotation": _quaternions(default_local[j, :3, :3]).tolist(),
        }
        children = [c for c, p in enumerate(parents) if p == j]
        if children:
            node["children"] = children
        nodes.append(node)
    mesh_node = len(nodes)
    nodes.append({"name": character.name, "mesh": 0, "skin": 0})
    roots = [j for j, p in enumerate(parents) if p < 0]
    inverse_bind = np.linalg.inv(bind).transpose(0, 2, 1).reshape(len(labels), 16)
    skin = {
        "name": character.rig,
        "joints": list(range(len(labels))),
        "skeleton": roots[0],
        "inverseBindMatrices": buffers.accessor(inverse_bind.astype(np.float32)),
    }

    gltf_animations = []
    if character.pose:
        gltf_animations.append(
            _character_pose(buffers, model, character, parents, default_local)
        )
    for name in animations:
        gltf_animations.append(
            _animation(buffers, model, character, name, parents, default_local)
        )

    metadata = {
        "generator": "corporis",
        "anny_version": _anny_version(),
        "character": character.to_dict(),
        "morph_targets": [t[0] for t in targets],
        "animations": (["pose"] if character.pose else []) + list(animations),
        "max_influences": max_influences,
        "frame": "metres, Y up, the figure faces +Z",
    }
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": f"Corporis glTF exporter (anny {_anny_version()})",
        },
        "scene": 0,
        "scenes": [
            {
                "name": character.name,
                "nodes": roots + [mesh_node],
                "extras": {"corporis": metadata},
            }
        ],
        "nodes": nodes,
        "meshes": [mesh],
        "skins": [skin],
        "materials": [
            {
                "name": "skin",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [float(c) for c in base_color],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                },
            }
        ],
        "buffers": [{"byteLength": 0}],
        "bufferViews": buffers.buffer_views,
        "accessors": buffers.accessors,
    }
    if gltf_animations:
        gltf["animations"] = gltf_animations
    size = _write_glb(pathlib.Path(path), gltf, buffers.data)
    return dict(
        path=str(path),
        bytes=size,
        vertices=int(len(source)),
        triangles=int(len(triangles)),
        joints=len(labels),
        morph_targets=len(targets),
        animations=len(gltf_animations),
        max_influences=max_influences,
        largest_dropped_weight=dropped,
    )


def _animation(
    buffers: _Buffers, model, character: Character, name: str, parents, default_local
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
        buffers,
        model,
        character,
        name,
        entry["pose_parameters"],
        float(entry["fps"]),
        parents,
        default_local,
    )


def _character_pose(
    buffers: _Buffers, model, character: Character, parents, default_local
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
        buffers, model, character, "pose", params, 1.0, parents, default_local
    )


def _keyframes(
    buffers: _Buffers,
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
    time_accessor = buffers.accessor(times, minmax=True)
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
                    "output": buffers.accessor(values.astype(np.float32)),
                    "interpolation": "LINEAR",
                }
            )
            channels.append(
                {"sampler": len(samplers) - 1, "target": {"node": j, "path": path}}
            )
    return {"name": name, "samplers": samplers, "channels": channels}


def _write_glb(path: pathlib.Path, gltf: dict, binary: bytearray) -> int:
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    gltf["buffers"][0]["byteLength"] = len(binary)
    text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(text), _JSON_CHUNK))
        f.write(text)
        f.write(struct.pack("<II", len(binary), _BIN_CHUNK))
        f.write(binary)
    return total


def read_gltf_json(path: str | pathlib.Path) -> dict:
    """The JSON chunk of a GLB file."""
    with open(path, "rb") as f:
        magic, version, _ = struct.unpack("<III", f.read(12))
        if magic != _GLB_MAGIC or version != 2:
            raise ValueError(f"{path} is not a glTF 2.0 binary file.")
        length, kind = struct.unpack("<II", f.read(8))
        if kind != _JSON_CHUNK:
            raise ValueError(f"{path} has no JSON chunk first.")
        return json.loads(f.read(length))


def read_character(path: str | pathlib.Path) -> Character:
    """The character stored in a GLB file written by ``export_glb``."""
    gltf = read_gltf_json(path)
    for scene in gltf.get("scenes", []):
        metadata = scene.get("extras", {}).get("corporis")
        if metadata:
            return Character.from_dict(metadata["character"])
    raise ValueError(
        f"{path} holds no Corporis character (no 'corporis' scene extras)."
    )


def _face_shape_names() -> list[str]:
    from anny.models.face_shapes import face_shape_spec

    return [p.name for p in face_shape_spec()]


def _anny_version() -> str:
    import anny

    return anny.__version__
