# OpenSculptBoy
# Apache License, Version 2.0
"""
A glTF 2.0 document in memory, written as a binary file (GLB).

:class:`GltfDocument` holds the JSON parts of a glTF file (nodes, meshes, skins, materials,
textures, images, samplers, animations, extensions and extras) and one binary buffer with its
buffer views and accessors. The GLB and VRM exporters build their files with it.

A document may hold several meshes, each with several primitives, and several skins; primitives
may share accessors. Images are embedded in the binary buffer. Every extension that a JSON
object of the document names under ``extensions`` is listed in ``extensionsUsed`` when the file
is written, whether or not :meth:`GltfDocument.use_extension` named it; only the extensions
named with ``required=True`` go to ``extensionsRequired``. :func:`pbr_material` and
:func:`unlit_material` (``KHR_materials_unlit``) build the two plain materials of the exporters.
"""

from __future__ import annotations

import io
import json
import pathlib
import struct
from typing import Any, Sequence

import numpy as np

_BYTE, _UBYTE, _SHORT, _USHORT, _UINT, _FLOAT = 5120, 5121, 5122, 5123, 5125, 5126
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
_TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
_COMPONENTS = {
    np.dtype(np.float32): _FLOAT,
    np.dtype(np.int8): _BYTE,
    np.dtype(np.uint8): _UBYTE,
    np.dtype(np.int16): _SHORT,
    np.dtype(np.uint16): _USHORT,
    np.dtype(np.uint32): _UINT,
}
# A morph target is stored sparse when it moves fewer than this share of the vertices.
SPARSE_SHARE = 0.4
# Offsets at or below this length (metres) count as no movement: the rounding noise of two forward passes
# (about 1e-16, depending on the CPU's BLAS kernels) stays out of the file.
MORPH_EPSILON = 1e-6
GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
# Sampler filters and wrap modes of glTF (WebGL enums).
NEAREST, LINEAR, LINEAR_MIPMAP_LINEAR = 9728, 9729, 9987
CLAMP_TO_EDGE, MIRRORED_REPEAT, REPEAT = 33071, 33648, 10497


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
        """
        An accessor of a (count,) or (count, width) array: float32, int8, uint8, int16, uint16
        or uint32, with width 1, 2, 3, 4 or 16 (MAT4, column-major).
        """
        array = np.ascontiguousarray(array)
        if array.dtype not in _COMPONENTS:
            raise TypeError(
                f"glTF accessors hold float32, int8, uint8, int16, uint16 or uint32 values, "
                f"not {array.dtype}."
            )
        width = 1 if array.ndim == 1 else array.shape[1]
        if array.ndim > 2 or width not in _TYPES:
            raise ValueError(f"Cannot store an array of shape {array.shape}.")
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
        deltas = np.array(deltas, dtype=np.float32)  # a copy: the still rows are zeroed
        still = np.linalg.norm(deltas, axis=1) <= MORPH_EPSILON
        deltas[still] = 0.0
        moving = np.flatnonzero(~still)
        if not sparse or len(moving) >= SPARSE_SHARE * len(deltas):
            # morph targets are vertex attributes: their buffer view names ARRAY_BUFFER
            return self.accessor(deltas, ARRAY_BUFFER, minmax=True)
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
    of the asset and of the root object, root ``extensions`` and ``extensionsUsed`` are set
    directly on the document. ``to_json`` gives the JSON part, ``to_bytes`` the GLB file and
    ``write`` writes it.
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
        self.extras: dict[str, Any] = {}

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

    def add_png(self, pixels: np.ndarray, name: str | None = None) -> int:
        """
        An image of (H, W), (H, W, 3) or (H, W, 4) uint8 pixels, embedded as a PNG; the first
        row is the top of the image, as glTF's texture coordinates expect.
        """
        return self.add_image(png_bytes(pixels), "image/png", name)

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
        """The JSON part of the file, with the extensions it uses listed in ``extensionsUsed``."""
        scene: dict[str, Any] = {"nodes": list(self.scene_nodes)}
        if self.scene_name is not None:
            scene = {"name": self.scene_name, **scene}
        if self.scene_extras:
            scene["extras"] = self.scene_extras
        asset: dict[str, Any] = {"version": "2.0", "generator": self.generator}
        if self.asset_extras:
            asset["extras"] = self.asset_extras
        gltf: dict[str, Any] = {"asset": asset}
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
        size = len(self.buffers.data)
        size += -size % 4  # the binary chunk is padded to 4 bytes
        gltf.update(
            buffers=[{"byteLength": size}],
            bufferViews=self.buffers.buffer_views,
            accessors=self.buffers.accessors,
        )
        if self.animations:
            gltf["animations"] = self.animations
        if self.extensions:
            gltf["extensions"] = self.extensions
        if self.extras:
            gltf["extras"] = self.extras
        used = list(self.extensions_used)
        for name in extension_names(gltf):
            if name not in used:
                used.append(name)
        if used or self.extensions_required:
            head = {"asset": gltf.pop("asset")}
            if used:
                head["extensionsUsed"] = used
            if self.extensions_required:
                head["extensionsRequired"] = list(self.extensions_required)
            gltf = {**head, **gltf}
        return gltf

    def to_bytes(self) -> bytes:
        """The GLB file."""
        return glb_bytes(self.to_json(), self.buffers.data)

    def write(self, path: str | pathlib.Path) -> int:
        """Write the GLB file; returns its size in bytes."""
        return write_glb(pathlib.Path(path), self.to_json(), self.buffers.data)


def extension_names(value: Any) -> list[str]:
    """
    The names of the extensions that a glTF JSON value uses (the keys of every ``extensions``
    object, outside ``extras``), in the order in which they first appear.
    """
    names: list[str] = []

    def visit(v):
        if isinstance(v, dict):
            for key, item in v.items():
                if key == "extras":
                    continue
                if key == "extensions" and isinstance(item, dict):
                    names.extend(n for n in item if n not in names)
                visit(item)
        elif isinstance(v, list):
            for item in v:
                visit(item)

    visit(value)
    return names


def pbr_material(
    name: str,
    base_color: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    metallic: float = 0.0,
    roughness: float = 0.6,
    texture: int | None = None,
    alpha_mode: str | None = None,
    double_sided: bool = False,
) -> dict:
    """A glTF metallic-roughness material: a linear RGBA base colour, optionally textured."""
    pbr: dict[str, Any] = {"baseColorFactor": [float(c) for c in base_color]}
    if texture is not None:
        pbr["baseColorTexture"] = {"index": int(texture)}
    pbr.update(metallicFactor=float(metallic), roughnessFactor=float(roughness))
    material: dict[str, Any] = {"name": name, "pbrMetallicRoughness": pbr}
    if alpha_mode is not None:
        material["alphaMode"] = alpha_mode
    if double_sided:
        material["doubleSided"] = True
    return material


def unlit_material(
    name: str,
    base_color: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    texture: int | None = None,
    alpha_mode: str | None = None,
    double_sided: bool = False,
) -> dict:
    """
    An unlit material (``KHR_materials_unlit``): the base colour as it is, without lighting.
    Its metallic-roughness fallback is a rough dielectric, as the extension recommends.
    """
    material = pbr_material(
        name, base_color, 0.0, 1.0, texture, alpha_mode, double_sided
    )
    material["extensions"] = {"KHR_materials_unlit": {}}
    return material


def png_bytes(pixels: np.ndarray) -> bytes:
    """A PNG file of (H, W), (H, W, 3) or (H, W, 4) uint8 pixels."""
    from PIL import Image

    pixels = np.ascontiguousarray(pixels)
    if pixels.dtype != np.uint8:
        raise TypeError(f"PNG pixels must be uint8, not {pixels.dtype}.")
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def glb_bytes(gltf: dict, binary: bytes | bytearray) -> bytes:
    """A GLB file of a glTF JSON dict and its binary buffer; sets ``buffers[0].byteLength``."""
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    gltf["buffers"][0]["byteLength"] = len(binary)
    text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return b"".join(
        (
            struct.pack("<III", GLB_MAGIC, 2, total),
            struct.pack("<II", len(text), JSON_CHUNK),
            text,
            struct.pack("<II", len(binary), BIN_CHUNK),
            binary,
        )
    )


def write_glb(path: pathlib.Path, gltf: dict, binary: bytes | bytearray) -> int:
    """Write a GLB file; returns its size in bytes."""
    data = glb_bytes(gltf, binary)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


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
