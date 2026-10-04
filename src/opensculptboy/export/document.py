# OpenSculptBoy
# Apache License, Version 2.0
"""
A glTF 2.0 document in memory, written as a binary file (GLB).

:class:`GltfDocument` holds the JSON parts of a glTF file (nodes, meshes, skins, materials,
textures, images, animations, extensions) and one binary buffer with its buffer views and
accessors. The GLB and VRM exporters build their files with it.
"""

from __future__ import annotations

import json
import pathlib
import struct
from typing import Any

import numpy as np

_FLOAT, _UBYTE, _USHORT, _UINT = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
_TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
_COMPONENTS = {
    np.dtype(np.float32): _FLOAT,
    np.dtype(np.uint8): _UBYTE,
    np.dtype(np.uint16): _USHORT,
    np.dtype(np.uint32): _UINT,
}
# A morph target is stored sparse when it moves fewer than this share of the vertices.
SPARSE_SHARE = 0.4
# Offsets at or below this length (metres) count as no movement: the rounding noise of two forward passes
# (about 1e-16, depending on the CPU's BLAS kernels) stays out of the file.
MORPH_EPSILON = 1e-6
GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942


class Buffers:
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

    def morph_accessor(self, deltas: np.ndarray, sparse: bool = True) -> int:
        """
        A VEC3 float accessor of position offsets. It is sparse when few vertices move and
        ``sparse`` is True; ``sparse=False`` always writes a dense accessor.
        """
        deltas = np.ascontiguousarray(deltas, dtype=np.float32)
        still = np.linalg.norm(deltas, axis=1) <= MORPH_EPSILON
        deltas[still] = 0.0
        moving = np.flatnonzero(~still)
        if not sparse or len(moving) >= SPARSE_SHARE * len(deltas):
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


class GltfDocument:
    """
    A glTF 2.0 document: JSON lists that grow with ``add_*`` calls, and one binary buffer.

    Each ``add_*`` method appends a JSON object and returns its index. ``extras`` of the scene,
    root ``extensions`` and ``extensionsUsed`` are set directly on the document. ``to_json``
    gives the JSON part and ``write`` the GLB file.
    """

    def __init__(self, generator: str):
        self.generator = generator
        self.buffers = Buffers()
        self.nodes: list[dict] = []
        self.meshes: list[dict] = []
        self.skins: list[dict] = []
        self.materials: list[dict] = []
        self.textures: list[dict] = []
        self.images: list[dict] = []
        self.samplers: list[dict] = []
        self.animations: list[dict] = []
        self.scene_nodes: list[int] = []
        self.scene_name: str | None = None
        self.scene_extras: dict[str, Any] = {}
        self.extensions: dict[str, Any] = {}
        self.extensions_used: list[str] = []
        self.extensions_required: list[str] = []
        self.asset_extras: dict[str, Any] = {}

    # binary data
    def accessor(self, array, target=None, minmax=False, normalized=False) -> int:
        return self.buffers.accessor(array, target, minmax, normalized)

    def morph_accessor(self, deltas, sparse: bool = True) -> int:
        return self.buffers.morph_accessor(deltas, sparse)

    # JSON objects
    def _add(self, items: list, item: dict) -> int:
        items.append(item)
        return len(items) - 1

    def add_node(self, node: dict) -> int:
        return self._add(self.nodes, node)

    def add_mesh(self, mesh: dict) -> int:
        return self._add(self.meshes, mesh)

    def add_skin(self, skin: dict) -> int:
        return self._add(self.skins, skin)

    def add_material(self, material: dict) -> int:
        return self._add(self.materials, material)

    def add_animation(self, animation: dict) -> int:
        return self._add(self.animations, animation)

    def add_sampler(self, sampler: dict) -> int:
        return self._add(self.samplers, sampler)

    def add_image(self, data: bytes, mime_type: str, name: str | None = None) -> int:
        """An image embedded in the binary buffer (``image/png`` or ``image/jpeg``)."""
        image = {"bufferView": self.buffers.view(bytes(data)), "mimeType": mime_type}
        if name:
            image["name"] = name
        return self._add(self.images, image)

    def add_texture(self, image: int, sampler: int | None = None) -> int:
        texture = {"source": image}
        if sampler is not None:
            texture["sampler"] = sampler
        return self._add(self.textures, texture)

    def use_extension(self, name: str, required: bool = False) -> None:
        if name not in self.extensions_used:
            self.extensions_used.append(name)
        if required and name not in self.extensions_required:
            self.extensions_required.append(name)

    def to_json(self) -> dict:
        """The JSON part of the file; ``buffers[0].byteLength`` is set by :meth:`write`."""
        scene: dict[str, Any] = {"nodes": list(self.scene_nodes)}
        if self.scene_name is not None:
            scene = {"name": self.scene_name, **scene}
        if self.scene_extras:
            scene["extras"] = self.scene_extras
        asset: dict[str, Any] = {"version": "2.0", "generator": self.generator}
        if self.asset_extras:
            asset["extras"] = self.asset_extras
        gltf: dict[str, Any] = {"asset": asset}
        if self.extensions_used:
            gltf["extensionsUsed"] = list(self.extensions_used)
        if self.extensions_required:
            gltf["extensionsRequired"] = list(self.extensions_required)
        gltf.update(scene=0, scenes=[scene], nodes=self.nodes)
        for key, items in (
            ("meshes", self.meshes),
            ("skins", self.skins),
            ("materials", self.materials),
            ("textures", self.textures),
            ("images", self.images),
            ("samplers", self.samplers),
        ):
            if items:
                gltf[key] = items
        gltf.update(
            buffers=[{"byteLength": 0}],
            bufferViews=self.buffers.buffer_views,
            accessors=self.buffers.accessors,
        )
        if self.animations:
            gltf["animations"] = self.animations
        if self.extensions:
            gltf["extensions"] = self.extensions
        return gltf

    def write(self, path: str | pathlib.Path) -> int:
        """Write the GLB file; returns its size in bytes."""
        return write_glb(pathlib.Path(path), self.to_json(), self.buffers.data)


def write_glb(path: pathlib.Path, gltf: dict, binary: bytearray) -> int:
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    gltf["buffers"][0]["byteLength"] = len(binary)
    text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(text), JSON_CHUNK))
        f.write(text)
        f.write(struct.pack("<II", len(binary), BIN_CHUNK))
        f.write(binary)
    return total


def read_gltf_json(path: str | pathlib.Path) -> dict:
    """The JSON chunk of a GLB file (``.glb`` or ``.vrm``)."""
    with open(path, "rb") as f:
        magic, version, _ = struct.unpack("<III", f.read(12))
        if magic != GLB_MAGIC or version != 2:
            raise ValueError(f"{path} is not a glTF 2.0 binary file.")
        length, kind = struct.unpack("<II", f.read(8))
        if kind != JSON_CHUNK:
            raise ValueError(f"{path} has no JSON chunk first.")
        return json.loads(f.read(length))
