# OpenSculptBoy
# Apache License, Version 2.0
"""
Budgets of VRM files: the triangles, joints, materials and texture sizes that VTuber apps and
platforms handle well, and the counts of a glTF document.

:func:`counts` counts what a glTF file draws: the triangles of every mesh node, the joints of
its skins, its materials, images, morph targets and meshes, and the size of its largest image,
read from the PNG or JPEG header. :func:`check` compares the counts with a budget of
:data:`BUDGETS`: VRM 1.0 apps (Warudo, VMagicMirror, three-vrm) take larger avatars than the
VRM 0.x apps (VSeeFace, 3tene).
"""

from __future__ import annotations

import base64
import struct

BUDGETS = {
    "vrm1": {"triangles": 70_000, "joints": 200, "materials": 16, "texture_size": 4096},
    "vrm0": {"triangles": 32_000, "joints": 128, "materials": 8, "texture_size": 2048},
}
# The names of the budgets, and the VRM versions that name them.
TARGETS = {"vrm1": "vrm1", "1.0": "vrm1", "vrm0": "vrm0", "0.x": "vrm0"}
LABELS = {"vrm1": "VRM 1.0", "vrm0": "VRM 0.x"}
MODES = ("strict", "warn", "off")
# The count that each budget limits.
_COUNTED = {
    "triangles": "triangles",
    "joints": "joints",
    "materials": "materials",
    "texture_size": "max_image_size",
}
_WHAT = {
    "triangles": "triangles",
    "joints": "joints",
    "materials": "materials",
    "texture_size": "largest texture (pixels)",
}


class BudgetError(ValueError):
    """A file over its budget, with ``--budget strict``."""


class BudgetWarning(UserWarning):
    """A file over its budget, with ``--budget warn``."""


def image_size(data: bytes) -> tuple[int, int] | None:
    """The width and height of a PNG or JPEG file from its header, or None for other data."""
    data = bytes(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # a fill byte
            i += 1
            continue
        if marker in (0x01, *range(0xD0, 0xD8)):  # markers without a length
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        # The start-of-frame markers; C4 (Huffman tables), C8 and CC (arithmetic coding
        # conditions) share the range but are not frames.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > len(data):
                return None
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return int(width), int(height)
        i += 2 + length
    return None


def _image_bytes(gltf: dict, image: dict, binary: bytes | None) -> bytes | None:
    """The bytes of an image of a glTF file: in the binary buffer or in a data URI."""
    if "bufferView" in image:
        if binary is None:
            return None
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        return bytes(binary[start : start + view["byteLength"]])
    uri = image.get("uri", "")
    if uri.startswith("data:") and ";base64," in uri:
        return base64.b64decode(uri.split(";base64,", 1)[1])
    return None  # an external file: its size is unknown here


def _primitive_triangles(gltf: dict, primitive: dict) -> int:
    """The triangles that one primitive draws (0 for points and lines)."""
    mode = primitive.get("mode", 4)
    if "indices" in primitive:
        count = gltf["accessors"][primitive["indices"]]["count"]
    else:
        count = gltf["accessors"][primitive["attributes"]["POSITION"]]["count"]
    if mode == 4:
        return count // 3
    if mode in (5, 6):  # triangle strips and fans
        return max(count - 2, 0)
    return 0


def counts(gltf: dict, binary: bytes | None = None) -> dict:
    """
    Triangles, vertices, joints, materials, images, morph targets and meshes of a glTF JSON
    dict, and the size of its largest image.

    ``triangles`` and ``vertices`` add up the primitives of every node with a mesh, so a mesh
    drawn by two nodes counts twice. ``joints`` counts the distinct joint nodes of all skins,
    and ``morph_targets`` the targets of each mesh (the most of its primitives), summed over
    the meshes. ``max_image_size`` is the larger side, in pixels, of the largest image, read
    from the PNG or JPEG header of the images in ``binary`` (the binary chunk of a GLB file)
    or in data URIs; 0 when the file has no image whose size can be read.
    """
    meshes = gltf.get("meshes", [])
    triangles = vertices = 0
    for node in gltf.get("nodes", []):
        if "mesh" not in node:
            continue
        for primitive in meshes[node["mesh"]]["primitives"]:
            triangles += _primitive_triangles(gltf, primitive)
            position = primitive["attributes"].get("POSITION")
            if position is not None:
                vertices += gltf["accessors"][position]["count"]
    joints = {j for skin in gltf.get("skins", []) for j in skin.get("joints", [])}
    morph_targets = sum(
        max((len(p.get("targets", [])) for p in mesh["primitives"]), default=0)
        for mesh in meshes
    )
    largest = 0
    for image in gltf.get("images", []):
        data = _image_bytes(gltf, image, binary)
        size = None if data is None else image_size(data)
        if size is not None:
            largest = max(largest, *size)
    return {
        "triangles": int(triangles),
        "vertices": int(vertices),
        "joints": len(joints),
        "materials": len(gltf.get("materials", [])),
        "images": len(gltf.get("images", [])),
        "max_image_size": int(largest),
        "morph_targets": int(morph_targets),
        "meshes": len(meshes),
    }


def check(found: dict, target: str, mode: str = "warn") -> list[str]:
    """
    Compare counts with ``BUDGETS[target]``. Returns the messages for each count over budget;
    ``mode="strict"`` raises :class:`BudgetError` instead, and ``mode="off"`` checks nothing.

    ``target`` is ``"vrm1"`` or ``"vrm0"``, or the VRM version ``"1.0"`` or ``"0.x"``;
    ``found`` holds the counts of :func:`counts`.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown budget mode {mode!r}; use one of {list(MODES)}.")
    if target not in TARGETS:
        raise ValueError(f"Unknown budget {target!r}; use one of {sorted(TARGETS)}.")
    if mode == "off":
        return []
    target = TARGETS[target]
    messages = []
    for key, limit in BUDGETS[target].items():
        value = int(found.get(_COUNTED[key], 0))
        if value > limit:
            messages.append(
                f"{_WHAT[key]}: {value:,} over the {LABELS[target]} budget of {limit:,}"
            )
    if messages and mode == "strict":
        raise BudgetError(
            "; ".join(messages)
            + ". The budget 'warn' (--budget warn) writes the file all the same."
        )
    return messages
