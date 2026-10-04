# OpenSculptBoy
# Apache License, Version 2.0
"""
A minimal glTF 2.0 binary reader for the tests: it evaluates a file as a glTF engine does (node
transforms, animation keyframes, skinning and morph targets), so that the tests can compare the
result with Anny's own forward pass.

CONTRACT (PR 1, stream A1 generalises ``evaluate`` to every skinned node and primitive).
"""

import json
import pathlib
import struct

import numpy as np

_DTYPES = {5126: np.float32, 5121: np.uint8, 5123: np.uint16, 5125: np.uint32}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GLB:
    """A minimal glTF 2.0 binary reader."""

    def __init__(self, path):
        data = pathlib.Path(path).read_bytes()
        magic, version, total = struct.unpack_from("<III", data, 0)
        assert magic == 0x46546C67 and version == 2 and total == len(data)
        length, _ = struct.unpack_from("<II", data, 12)
        self.json = json.loads(data[20 : 20 + length])
        offset = 20 + length
        bin_length, _ = struct.unpack_from("<II", data, offset)
        self.bin = data[offset + 8 : offset + 8 + bin_length]

    def _view(self, index, dtype, count, width, byte_offset=0):
        view = self.json["bufferViews"][index]
        start = view.get("byteOffset", 0) + byte_offset
        array = np.frombuffer(self.bin, dtype=dtype, count=count * width, offset=start)
        return array.reshape(count, width) if width > 1 else array

    def accessor(self, index):
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
            idx = self._view(
                s["indices"]["bufferView"],
                _DTYPES[s["indices"]["componentType"]],
                s["count"],
                1,
            )
            array[idx] = self._view(s["values"]["bufferView"], dtype, s["count"], width)
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


def trs_matrix(translation, rotation):
    x, y, z, w = rotation
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = translation
    return m


def node_world_matrices(glb, animation=None, frame=0):
    """World matrices of the nodes in the default pose, or in one keyframe of an animation."""
    g = glb.json
    nodes = [dict(n) for n in g["nodes"]]
    if animation is not None:
        anim = next(a for a in g["animations"] if a["name"] == animation)
        for channel in anim["channels"]:
            sampler = anim["samplers"][channel["sampler"]]
            nodes[channel["target"]["node"]][channel["target"]["path"]] = glb.accessor(
                sampler["output"]
            )[frame]
    world = [None] * len(nodes)

    def visit(i, parent):
        n = nodes[i]
        world[i] = parent @ trs_matrix(
            n.get("translation", [0, 0, 0]), n.get("rotation", [0, 0, 0, 1])
        )
        for c in n.get("children", []):
            visit(c, world[i])

    for root in g["scenes"][0]["nodes"]:
        visit(root, np.eye(4))
    return world


def evaluate(glb, animation=None, frame=0, weights=None):
    """Skinned glTF vertices of the default pose, or of one keyframe of an animation."""
    g = glb.json
    world = node_world_matrices(glb, animation, frame)
    skin = g["skins"][0]
    ibm = glb.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    joint_matrices = np.stack([world[j] for j in skin["joints"]]) @ ibm
    mesh = g["meshes"][0]
    prim = mesh["primitives"][0]
    attrs = prim["attributes"]
    positions = glb.accessor(attrs["POSITION"]).astype(np.float64)
    weights = mesh.get("weights", []) if weights is None else weights
    for target, w in zip(prim.get("targets", []), weights):
        positions = positions + w * glb.accessor(target["POSITION"])
    out = np.zeros_like(positions)
    s = 0
    while f"JOINTS_{s}" in attrs:
        joints = glb.accessor(attrs[f"JOINTS_{s}"]).astype(int)
        w = glb.accessor(attrs[f"WEIGHTS_{s}"]).astype(np.float64)
        for k in range(4):
            m = joint_matrices[joints[:, k]]
            out += w[:, k, None] * (
                np.einsum("vij,vj->vi", m[:, :3, :3], positions) + m[:, :3, 3]
            )
        s += 1
    return out, glb.accessor(attrs["_ANNY_VERTEX"]).astype(int)
