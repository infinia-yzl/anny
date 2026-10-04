# OpenSculptBoy
# Apache License, Version 2.0
"""
A minimal glTF 2.0 binary reader for the tests: it evaluates a file as a glTF engine does (node
transforms, animation keyframes, skinning and morph targets), so that the tests can compare the
result with Anny's own forward pass.

- :class:`GLB` reads the JSON and binary chunks of a ``.glb`` or ``.vrm`` file and decodes
  accessors (sparse, strided and normalised ones too) and embedded images.
- :func:`node_world_matrices` gives the world matrix of every node (TRS or ``matrix``), in the
  default pose or in one keyframe of an animation.
- :func:`evaluate_nodes` skins every primitive of every node that has a mesh and a skin (several
  skins and several JOINTS_n/WEIGHTS_n sets), with its morph targets; :func:`evaluate` keeps
  the old interface: the first primitive of the first skinned mesh node.
- :func:`structure_errors` lists the structural faults of a file (dangling indices, accessors
  out of their buffer views, inconsistent primitives and skins, undeclared extensions).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import struct

import numpy as np

_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class GLB:
    """A minimal glTF 2.0 binary reader, for a path or the bytes of a GLB file."""

    def __init__(self, path):
        if isinstance(path, (bytes, bytearray)):
            data = bytes(path)
        else:
            data = pathlib.Path(path).read_bytes()
        magic, version, total = struct.unpack_from("<III", data, 0)
        assert magic == 0x46546C67 and version == 2 and total == len(data)
        length, kind = struct.unpack_from("<II", data, 12)
        assert kind == 0x4E4F534A, "the first chunk of a GLB file is its JSON"
        self.json = json.loads(data[20 : 20 + length])
        offset = 20 + length
        self.bin = b""
        if offset < len(data):
            bin_length, kind = struct.unpack_from("<II", data, offset)
            assert kind == 0x004E4942, "the second chunk of a GLB file is its binary"
            self.bin = data[offset + 8 : offset + 8 + bin_length]

    def _view(self, index, dtype, count, width, byte_offset=0):
        view = self.json["bufferViews"][index]
        start = view.get("byteOffset", 0) + byte_offset
        dtype = np.dtype(dtype)
        stride = view.get("byteStride")
        if stride and stride != dtype.itemsize * width:
            array = np.ndarray(
                (count, width),
                dtype=dtype,
                buffer=self.bin,
                offset=start,
                strides=(stride, dtype.itemsize),
            )
            return array if width > 1 else array[:, 0]
        array = np.frombuffer(self.bin, dtype=dtype, count=count * width, offset=start)
        return array.reshape(count, width) if width > 1 else array

    def accessor(self, index):
        """
        The values of an accessor: (count,) or (count, width), with its sparse elements applied;
        normalised integers become floats in [0, 1] or [-1, 1].
        """
        a = self.json["accessors"][index]
        dtype, width = _DTYPES[a["componentType"]], _WIDTHS[a["type"]]
        if "bufferView" in a:
            array = self._view(
                a["bufferView"], dtype, a["count"], width, a.get("byteOffset", 0)
            ).copy()
        else:
            array = np.zeros(
                (a["count"], width) if width > 1 else a["count"], dtype=dtype
            )
        if "sparse" in a:
            s = a["sparse"]
            idx = self.sparse_indices(index)
            array[idx] = self._view(
                s["values"]["bufferView"],
                dtype,
                s["count"],
                width,
                s["values"].get("byteOffset", 0),
            )
        if a.get("normalized") and np.issubdtype(dtype, np.integer):
            limit = np.iinfo(dtype).max
            array = np.maximum(array.astype(np.float64) / limit, -1.0)
        return array

    def sparse_indices(self, index):
        """The indices of the sparse elements of an accessor, or None for a dense accessor."""
        a = self.json["accessors"][index]
        if "sparse" not in a:
            return None
        s = a["sparse"]
        return self._view(
            s["indices"]["bufferView"],
            _DTYPES[s["indices"]["componentType"]],
            s["count"],
            1,
            s["indices"].get("byteOffset", 0),
        ).copy()

    def image_bytes(self, index) -> bytes:
        """The bytes of an embedded image (a PNG or JPEG file)."""
        view = self.json["bufferViews"][self.json["images"][index]["bufferView"]]
        start = view.get("byteOffset", 0)
        return bytes(self.bin[start : start + view["byteLength"]])


def trs_matrix(translation, rotation, scale=None):
    """The 4x4 matrix of a translation, a unit quaternion (x, y, z, w) and a scale."""
    x, y, z, w = rotation
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    if scale is not None:
        m[:3, :3] = m[:3, :3] * np.asarray(scale, dtype=np.float64)[None, :]
    m[:3, 3] = translation
    return m


def node_local_matrix(node) -> np.ndarray:
    """The local matrix of a node: its ``matrix`` (column-major), or its TRS."""
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    return trs_matrix(
        node.get("translation", [0, 0, 0]),
        node.get("rotation", [0, 0, 0, 1]),
        node.get("scale"),
    )


def _keyframe(glb, sampler, frame, size):
    """The output of an animation sampler at one keyframe: ``size`` values."""
    values = np.asarray(glb.accessor(sampler["output"]), dtype=np.float64)
    values = values.reshape(-1, size)
    if sampler.get("interpolation") == "CUBICSPLINE":
        return values[3 * frame + 1]  # in-tangent, value, out-tangent
    return values[frame]


def animated_nodes(glb, animation=None, frame=0) -> list[dict]:
    """Copies of the nodes, with the TRS and weights of one keyframe of an animation."""
    g = glb.json
    nodes = [dict(n) for n in g["nodes"]]
    if animation is None:
        return nodes
    if isinstance(animation, int):
        anim = g["animations"][animation]
    else:
        anim = next(a for a in g["animations"] if a.get("name") == animation)
    for channel in anim["channels"]:
        target = channel["target"]
        if "node" not in target:
            continue  # a channel of an extension (KHR_animation_pointer)
        node, path = nodes[target["node"]], target["path"]
        sampler = anim["samplers"][channel["sampler"]]
        if path == "weights":
            size = len(g["meshes"][node["mesh"]]["primitives"][0].get("targets", []))
        else:
            size = {"translation": 3, "rotation": 4, "scale": 3}[path]
        node.pop("matrix", None)  # an animated node has no matrix
        node[path] = _keyframe(glb, sampler, frame, size)
    return nodes


def node_world_matrices(glb, animation=None, frame=0, nodes=None):
    """
    World matrices of the nodes in the default pose, or in one keyframe of an animation (its
    name or index); None for the nodes outside the scene.
    """
    g = glb.json
    if nodes is None:
        nodes = animated_nodes(glb, animation, frame)
    world = [None] * len(nodes)

    def visit(i, parent):
        world[i] = parent @ node_local_matrix(nodes[i])
        for c in nodes[i].get("children", []):
            visit(c, world[i])

    for root in g["scenes"][g.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return world


@dataclasses.dataclass
class EvaluatedPrimitive:
    """The vertices of one primitive of a mesh node, after morph targets and skinning."""

    node: int
    mesh: int
    primitive: int
    positions: np.ndarray  # (N, 3) float64, in the scene's frame
    anny_vertex: np.ndarray | None  # (N,) int, the _ANNY_VERTEX attribute
    triangles: np.ndarray | None  # (T, 3) int, for mode 4 with indices
    material: int | None


def _node_weights(glb, nodes, i, weights):
    """The morph weights of mesh node ``i``: the argument, the node's or the mesh's."""
    node = nodes[i]
    if isinstance(weights, dict):
        weights = weights.get(i, weights.get(node.get("name")))
    if weights is not None:
        return np.asarray(weights, dtype=np.float64)
    mesh = glb.json["meshes"][node["mesh"]]
    return np.asarray(node.get("weights", mesh.get("weights", [])), dtype=np.float64)


def _skin_matrices(glb, world, skin_index):
    skin = glb.json["skins"][skin_index]
    count = len(skin["joints"])
    if "inverseBindMatrices" in skin:
        ibm = glb.accessor(skin["inverseBindMatrices"]).astype(np.float64)
        ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)
    else:
        ibm = np.tile(np.eye(4), (count, 1, 1))
    return np.stack([world[j] for j in skin["joints"]]) @ ibm


def evaluate_nodes(glb, animation=None, frame=0, weights=None, skinned_only=True):
    """
    The vertices of every primitive of every node with a mesh and a skin, in the default pose
    or in one keyframe of an animation, with their morph targets.

    Args:
        glb: a :class:`GLB`.
        animation, frame: an animation (name or index) and its keyframe, or None.
        weights: morph weights for every mesh (a sequence), or a dict that maps node indices or
            node names to weights; by default the node's weights, else the mesh's (an animation
            of the weights takes their place). Missing weights count as 0.
        skinned_only: False adds the nodes with a mesh and no skin, placed by their world
            matrix.

    Returns:
        A dict from node index to the list of its :class:`EvaluatedPrimitive`. Primitives that
        share their vertex data are evaluated once and share the arrays.
    """
    g = glb.json
    nodes = animated_nodes(glb, animation, frame)
    world = node_world_matrices(glb, nodes=nodes)
    skins: dict[int, np.ndarray] = {}
    result: dict[int, list[EvaluatedPrimitive]] = {}
    for i, node in enumerate(nodes):
        if "mesh" not in node or world[i] is None:
            continue
        if skinned_only and "skin" not in node:
            continue
        node_weights = _node_weights(glb, nodes, i, weights)
        shared: dict[tuple, tuple] = {}
        prims = []
        for p, prim in enumerate(g["meshes"][node["mesh"]]["primitives"]):
            attrs = prim["attributes"]
            targets = tuple(t.get("POSITION") for t in prim.get("targets", []))
            sets = 0
            while f"JOINTS_{sets}" in attrs:
                sets += 1
            key = (
                attrs["POSITION"],
                targets,
                tuple(attrs.get(f"JOINTS_{s}") for s in range(sets)),
                tuple(attrs.get(f"WEIGHTS_{s}") for s in range(sets)),
                attrs.get("_ANNY_VERTEX"),
            )
            if key not in shared:
                positions = glb.accessor(attrs["POSITION"]).astype(np.float64)
                for target, w in zip(targets, node_weights):
                    if target is not None and w != 0:
                        positions = positions + w * glb.accessor(target)
                if "skin" in node:
                    if node["skin"] not in skins:
                        skins[node["skin"]] = _skin_matrices(glb, world, node["skin"])
                    matrices = skins[node["skin"]]
                    out = np.zeros_like(positions)
                    for s in range(sets):
                        joints = glb.accessor(attrs[f"JOINTS_{s}"]).astype(int)
                        w = glb.accessor(attrs[f"WEIGHTS_{s}"]).astype(np.float64)
                        for k in range(joints.shape[1]):
                            m = matrices[joints[:, k]]
                            out += w[:, k, None] * (
                                np.einsum("vij,vj->vi", m[:, :3, :3], positions)
                                + m[:, :3, 3]
                            )
                else:
                    m = world[i]
                    out = positions @ m[:3, :3].T + m[:3, 3]
                anny = attrs.get("_ANNY_VERTEX")
                anny = None if anny is None else glb.accessor(anny).astype(int)
                shared[key] = (out, anny)
            out, anny = shared[key]
            triangles = None
            if "indices" in prim and prim.get("mode", 4) == 4:
                triangles = glb.accessor(prim["indices"]).astype(int).reshape(-1, 3)
            prims.append(
                EvaluatedPrimitive(
                    i, node["mesh"], p, out, anny, triangles, prim.get("material")
                )
            )
        result[i] = prims
    return result


def evaluate(glb, animation=None, frame=0, weights=None):
    """
    Skinned glTF vertices of the default pose, or of one keyframe of an animation: the first
    primitive of the first skinned mesh node, and its ``_ANNY_VERTEX`` attribute (None when it
    has none).
    """
    nodes = evaluate_nodes(glb, animation, frame, weights)
    if not nodes:
        raise ValueError("The file has no skinned mesh node.")
    first = nodes[min(nodes)][0]
    return first.positions, first.anny_vertex


def _extension_names(value) -> set[str]:
    names = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "extras":
                continue
            if key == "extensions" and isinstance(item, dict):
                names.update(item)
            names |= _extension_names(item)
    elif isinstance(value, list):
        for item in value:
            names |= _extension_names(item)
    return names


def structure_errors(glb) -> list[str]:
    """
    The structural faults of a file, as messages (an empty list for a sound file): indices out
    of range, accessors out of their buffer views, primitives whose attributes, indices or
    targets disagree, skins whose joints or weights are wrong, nodes with two parents or in a
    cycle, and extensions missing from ``extensionsUsed``.
    """
    g = glb.json
    errors: list[str] = []
    count = {
        key: len(g.get(key, []))
        for key in (
            "nodes",
            "meshes",
            "skins",
            "materials",
            "textures",
            "images",
            "samplers",
            "accessors",
            "bufferViews",
        )
    }

    def check(kind, index, where):
        if not isinstance(index, int) or not 0 <= index < count[kind]:
            errors.append(f"{where}: {kind} index {index!r} out of range")
            return False
        return True

    for v, view in enumerate(g.get("bufferViews", [])):
        end = view.get("byteOffset", 0) + view["byteLength"]
        if end > len(glb.bin):
            errors.append(f"bufferViews[{v}] ends at {end}, past the binary chunk")
    for a, acc in enumerate(g.get("accessors", [])):
        where = f"accessors[{a}]"
        size = np.dtype(_DTYPES[acc["componentType"]]).itemsize * _WIDTHS[acc["type"]]
        if "bufferView" in acc and check("bufferViews", acc["bufferView"], where):
            view = g["bufferViews"][acc["bufferView"]]
            stride = view.get("byteStride", size)
            need = acc.get("byteOffset", 0) + stride * (acc["count"] - 1) + size
            if need > view["byteLength"]:
                errors.append(f"{where} needs {need} bytes of a {view['byteLength']}")
        if "sparse" in acc:
            idx = glb.sparse_indices(a)
            if len(idx) and (idx.max() >= acc["count"] or np.any(np.diff(idx) <= 0)):
                errors.append(f"{where}: sparse indices out of range or not increasing")
    for m, mesh in enumerate(g.get("meshes", [])):
        target_counts = set()
        for p, prim in enumerate(mesh["primitives"]):
            where = f"meshes[{m}].primitives[{p}]"
            attrs = prim["attributes"]
            counts = {
                name: g["accessors"][a]["count"]
                for name, a in attrs.items()
                if check("accessors", a, f"{where}.{name}")
            }
            if len(set(counts.values())) > 1:
                errors.append(f"{where}: attribute counts differ {counts}")
            vertices = counts.get("POSITION", 0)
            if "indices" in prim and check("accessors", prim["indices"], where):
                indices = glb.accessor(prim["indices"])
                if len(indices) and indices.max() >= vertices:
                    errors.append(f"{where}: an index passes the {vertices} vertices")
                if prim.get("mode", 4) == 4 and len(indices) % 3:
                    errors.append(f"{where}: {len(indices)} indices are not triangles")
            if "material" in prim:
                check("materials", prim["material"], where)
            for t, target in enumerate(prim.get("targets", [])):
                for name, a in target.items():
                    if check("accessors", a, f"{where}.targets[{t}]"):
                        if g["accessors"][a]["count"] != vertices:
                            errors.append(f"{where}.targets[{t}].{name}: wrong count")
            target_counts.add(len(prim.get("targets", [])))
            sets = sum(1 for name in attrs if name.startswith("JOINTS_"))
            for s in range(sets):
                if f"WEIGHTS_{s}" not in attrs:
                    errors.append(f"{where}: JOINTS_{s} without WEIGHTS_{s}")
            if sets and all(f"WEIGHTS_{s}" in attrs for s in range(sets)):
                total, tolerance = 0.0, 0.0
                for s in range(sets):
                    acc = g["accessors"][attrs[f"WEIGHTS_{s}"]]
                    dtype = _DTYPES[acc["componentType"]]
                    # float rounding, or half a step of each normalised integer
                    step = 1e-6 if dtype == np.float32 else 0.5 / np.iinfo(dtype).max
                    tolerance += 4 * step
                    weights = np.asarray(glb.accessor(attrs[f"WEIGHTS_{s}"]))
                    total = total + weights.astype(np.float64).sum(1)
                if np.size(total) and np.abs(total - 1).max() > tolerance:
                    errors.append(f"{where}: skin weights do not sum to 1")
        if len(target_counts) > 1:
            errors.append(f"meshes[{m}]: primitives have different target counts")
        if (
            "weights" in mesh
            and target_counts
            and {len(mesh["weights"])} != target_counts
        ):
            errors.append(f"meshes[{m}]: weights do not match the targets")
    parents: dict[int, int] = {}
    for n, node in enumerate(g.get("nodes", [])):
        where = f"nodes[{n}]"
        for c in node.get("children", []):
            if check("nodes", c, where):
                if c in parents:
                    errors.append(f"nodes[{c}] has two parents")
                parents[c] = n
        if "mesh" in node:
            check("meshes", node["mesh"], where)
        if "skin" in node and check("skins", node["skin"], where):
            joints = len(g["skins"][node["skin"]]["joints"])
            for prim in g["meshes"][node["mesh"]]["primitives"]:
                for name, a in prim["attributes"].items():
                    if name.startswith("JOINTS_") and len(glb.accessor(a)):
                        if int(glb.accessor(a).max()) >= joints:
                            errors.append(f"{where}: {name} passes the skin's joints")
        if "matrix" in node and any(
            k in node for k in ("translation", "rotation", "scale")
        ):
            errors.append(f"{where} has a matrix and a TRS")
    for n in range(count["nodes"]):  # no cycles: every chain of parents ends
        seen, i = set(), n
        while i in parents:
            if i in seen:
                errors.append(f"nodes[{n}] lies on a cycle")
                break
            seen.add(i)
            i = parents[i]
    for scene in g.get("scenes", []):
        for root in scene.get("nodes", []):
            if check("nodes", root, "scene") and root in parents:
                errors.append(f"scene root nodes[{root}] has a parent")
    for s, skin in enumerate(g.get("skins", [])):
        for j in skin["joints"]:
            check("nodes", j, f"skins[{s}].joints")
        if "inverseBindMatrices" in skin and check(
            "accessors", skin["inverseBindMatrices"], f"skins[{s}]"
        ):
            if g["accessors"][skin["inverseBindMatrices"]]["count"] < len(
                skin["joints"]
            ):
                errors.append(f"skins[{s}]: fewer inverse bind matrices than joints")
    for t, texture in enumerate(g.get("textures", [])):
        if "source" in texture:
            check("images", texture["source"], f"textures[{t}]")
        if "sampler" in texture:
            check("samplers", texture["sampler"], f"textures[{t}]")
    for i, image in enumerate(g.get("images", [])):
        if "bufferView" in image:
            check("bufferViews", image["bufferView"], f"images[{i}]")
            if "mimeType" not in image:
                errors.append(f"images[{i}] has a buffer view and no mimeType")

    def texture_refs(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "extras":
                    continue
                if (
                    key.endswith("Texture")
                    and isinstance(item, dict)
                    and "index" in item
                ):
                    check("textures", item["index"], f"{where}.{key}")
                texture_refs(item, f"{where}.{key}")
        elif isinstance(value, list):
            for k, item in enumerate(value):
                texture_refs(item, f"{where}[{k}]")

    for m, material in enumerate(g.get("materials", [])):
        texture_refs(material, f"materials[{m}]")
    used = set(g.get("extensionsUsed", []))
    for name in sorted(_extension_names(g) - used):
        errors.append(f"extension {name} is not in extensionsUsed")
    for name in sorted(set(g.get("extensionsRequired", [])) - used):
        errors.append(f"required extension {name} is not in extensionsUsed")
    return errors
