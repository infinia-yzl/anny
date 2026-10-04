# OpenSculptBoy
# Apache License, Version 2.0
"""
Tables of the VRM export for the ``anny`` rig: the humanoid bone map, the required bones of
each VRM version, the twist bones, the eyeball vertices, the VRM skin weights, the expressions
(``data/vrm/expressions.json``) and the look-at ranges.

CONTRACT (PR 1, stream A3 implements): the humanoid maps and the required bones are final;
``TWIST_BONES`` is a first guess that A3 may revise; ``vrm_skin_weights``, ``look_at_ranges`` and
the expression table are placeholders.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib

import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "vrm"

# VRM 1.0 humanoid bone -> anny bone. The right side mirrors the left (".L" -> ".R").
_LEFT = {
    "leftUpperLeg": "upperleg01.L",
    "leftLowerLeg": "lowerleg01.L",
    "leftFoot": "foot.L",
    "leftShoulder": "clavicle.L",
    "leftUpperArm": "upperarm01.L",
    "leftLowerArm": "lowerarm01.L",
    "leftHand": "wrist.L",
    "leftThumbMetacarpal": "finger1-1.L",
    "leftThumbProximal": "finger1-2.L",
    "leftThumbDistal": "finger1-3.L",
    "leftIndexProximal": "finger2-1.L",
    "leftIndexIntermediate": "finger2-2.L",
    "leftIndexDistal": "finger2-3.L",
    "leftMiddleProximal": "finger3-1.L",
    "leftMiddleIntermediate": "finger3-2.L",
    "leftMiddleDistal": "finger3-3.L",
    "leftRingProximal": "finger4-1.L",
    "leftRingIntermediate": "finger4-2.L",
    "leftRingDistal": "finger4-3.L",
    "leftLittleProximal": "finger5-1.L",
    "leftLittleIntermediate": "finger5-2.L",
    "leftLittleDistal": "finger5-3.L",
    "leftEye": "eye.L",
}
HUMANOID_1 = {
    "hips": "root",
    "spine": "spine04",
    "chest": "spine02",
    "upperChest": "spine01",
    "neck": "neck01",
    "head": "head",
    **_LEFT,
    **{
        "right" + k[len("left") :]: v[:-2] + ".R"
        for k, v in _LEFT.items()  # "leftFoot" -> "rightFoot", "foot.L" -> "foot.R"
    },
}
# VRM 0.x names the three thumb bones Proximal, Intermediate and Distal.
_THUMB_0 = {"ThumbMetacarpal": "ThumbProximal", "ThumbProximal": "ThumbIntermediate"}


def _name_0(name_1: str) -> str:
    for side in ("left", "right"):
        if name_1.startswith(side) and name_1[len(side) :] in _THUMB_0:
            return side + _THUMB_0[name_1[len(side) :]]
    return name_1


HUMANOID_0 = {_name_0(k): v for k, v in HUMANOID_1.items()}

REQUIRED_1 = {
    "hips",
    "spine",
    "head",
    *(
        side + part
        for side in ("left", "right")
        for part in (
            "UpperLeg",
            "LowerLeg",
            "Foot",
            "UpperArm",
            "LowerArm",
            "Hand",
        )
    ),
}
REQUIRED_0 = REQUIRED_1 | {"chest", "neck"}

# Twist bones: twist bone -> (the humanoid bone whose roll it follows, weight of the roll).
TWIST_BONES = {
    f"{bone}.{s}": (f"{source}.{s}", 0.5)
    for s in ("L", "R")
    for bone, source in (
        ("lowerarm02", "wrist"),
        ("upperarm02", "lowerarm01"),
        ("upperleg02", "lowerleg01"),
        ("lowerleg02", "foot"),
    )
}

# The eyeballs: MakeHuman base-mesh vertex ranges of helper-l-eye and helper-r-eye.
_EYE_RANGES = {"L": (14598, 14669), "R": (14670, 14741)}


def humanoid_bones(version: str) -> dict[str, str]:
    """VRM humanoid bone name -> anny bone label, for ``version`` "1.0" or "0.x"."""
    if version == "1.0":
        return dict(HUMANOID_1)
    if version == "0.x":
        return dict(HUMANOID_0)
    raise ValueError(f"Unknown VRM version {version!r}; use '1.0' or '0.x'.")


def required_bones(version: str) -> set[str]:
    return set(REQUIRED_1 if version == "1.0" else REQUIRED_0)


def eyeball_vertices(model) -> dict[str, np.ndarray]:
    """The model vertex indices of each eyeball (``"L"`` and ``"R"``)."""
    base = model.base_mesh_vertex_indices.cpu().numpy()
    return {
        side: np.flatnonzero((base >= lo) & (base <= hi))
        for side, (lo, hi) in _EYE_RANGES.items()
    }


def vrm_skin_weights(model, twist: str = "constraint") -> tuple[np.ndarray, np.ndarray]:
    """
    (V, K) skin weights and bone indices for a VRM file, before the truncation to 4 bones.

    The eyelid vertices weighted to an eye bone move that weight to ``head``, so that look-at
    turns the eyeballs alone. ``twist="merge"`` (the VRM 0.x default) moves the weights of the
    twist bones to their mapped parents; ``twist="constraint"`` keeps them (VRM 1.0 drives them
    with roll constraints).
    """
    if twist not in ("constraint", "merge"):
        raise ValueError(f"Unknown twist mode {twist!r}.")
    # placeholder: stream A3
    return (
        model.vertex_bone_weights.detach().cpu().double().numpy(),
        model.vertex_bone_indices.cpu().numpy(),
    )


@dataclasses.dataclass
class Expression:
    """A VRM expression as a mix of ARKit facial actions (``model.facial_action_labels``)."""

    name: str  # the VRM 1.0 preset name, or the custom name
    mix: dict[str, float]  # facial action -> weight (0 to 1)
    preset: bool  # a VRM 1.0 preset (aa, blink, happy, ...)
    is_binary: bool = False
    override_blink: str = "none"  # "none", "block" or "blend"
    override_look_at: str = "none"
    override_mouth: str = "none"
    vrm0_preset: str | None = (
        None  # the VRM 0.x presetName; None: "unknown" (a custom group)
    )
    vrm0_name: str | None = None  # the VRM 0.x group name; None: the same as ``name``
    versions: tuple[str, ...] = ("1.0", "0.x")  # the VRM versions that carry it


@functools.lru_cache(maxsize=1)
def _expression_table() -> dict:
    with open(DATA_DIR / "expressions.json") as f:
        return json.load(f)


def expressions(version: str, facial_action_labels) -> list[Expression]:
    """
    The expressions of a VRM file: the presets, then the 52 perfect-sync customs (PascalCase
    ARKit names), then, for VRM 0.x, the VSeeFace viseme customs.
    """
    table = _expression_table()
    result = []
    for name, entry in table["presets"].items():
        result.append(
            Expression(
                name=name,
                mix=dict(entry["mix"]),
                preset=True,
                is_binary=entry.get("isBinary", False),
                override_blink=entry.get("overrideBlink", "none"),
                override_look_at=entry.get("overrideLookAt", "none"),
                override_mouth=entry.get("overrideMouth", "none"),
                vrm0_preset=entry.get("vrm0Preset"),
                vrm0_name=entry.get("vrm0Name"),
                versions=tuple(entry.get("versions", ("1.0", "0.x"))),
            )
        )
    if table.get("perfectSync", False):
        for label in facial_action_labels:
            result.append(
                Expression(
                    name=label[0].upper() + label[1:], mix={label: 1.0}, preset=False
                )
            )
    for name, entry in table.get("customs", {}).items():
        result.append(
            Expression(
                name=name,
                mix=dict(entry["mix"]),
                preset=False,
                versions=tuple(entry.get("versions", ("1.0", "0.x"))),
            )
        )
    return [e for e in result if version in e.versions]


def look_at_ranges(model, rest_output=None) -> dict[str, float]:
    """
    The look-at range of each direction in degrees (``"lookUp"``, ``"lookDown"``, ``"lookIn"``,
    ``"lookOut"``): the eyeball rotation that the full ``eyeLook*`` facial action gives.
    """
    # placeholder: stream A3 (a Kabsch fit of the eyeball vertices)
    return {"lookUp": 10.0, "lookDown": 10.0, "lookIn": 10.0, "lookOut": 10.0}


def lid_only(offsets: np.ndarray, eyeballs: dict[str, np.ndarray]) -> np.ndarray:
    """Morph target offsets (V, 3) with the eyeball rows zeroed (the VRM ``eyeLook*`` targets)."""
    offsets = np.array(offsets, dtype=np.float64, copy=True)
    for rows in eyeballs.values():
        offsets[rows] = 0.0
    return offsets
