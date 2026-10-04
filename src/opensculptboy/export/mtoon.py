# OpenSculptBoy
# Apache License, Version 2.0
"""
MToon materials for VRM files: one version-neutral :class:`ToonMaterial` and two encoders.

``to_vrm1`` writes a glTF material with the ``VRMC_materials_mtoon`` extension (VRM 1.0, linear
colours) over a PBR fallback. ``to_vrm0`` writes the glTF material and the VRM 0.x
``materialProperties`` entry (shader ``VRM/MToon``, sRGB colours, the 0.x shade parameters).
``v0_to_v1`` repeats three-vrm's conversion of 0.x properties, for tests.

CONTRACT (PR 1, stream A5 implements): the encoders below are minimal placeholders.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ToonMaterial:
    """An MToon material in VRM 1.0 terms: linear RGB colours, MToon 1.0 factors."""

    name: str
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    base_texture: int | None = None  # glTF texture index
    shade_color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    shade_texture: int | None = None
    shading_shift: float = 0.0
    shading_toony: float = 0.9
    gi_equalization: float = 0.9
    rim_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rim_fresnel_power: float = 5.0
    rim_lift: float = 0.0
    rim_lighting_mix: float = 1.0
    matcap_texture: int | None = None
    matcap_factor: tuple[float, float, float] = (1.0, 1.0, 1.0)
    outline_width_mode: str = (
        "none"  # "none", "worldCoordinates" or "screenCoordinates"
    )
    outline_width: float = (
        0.0  # metres in worldCoordinates; a share of the screen height otherwise
    )
    outline_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    outline_lighting_mix: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha_mode: str = "OPAQUE"  # "OPAQUE", "MASK" or "BLEND"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    render_queue_offset: int = 0


def soft_material(
    name: str, base_color: tuple[float, float, float, float]
) -> ToonMaterial:
    """The Soft style (a gentle toon look) for a base colour; PR 2 takes it from the style presets."""
    r, g, b, a = base_color
    return ToonMaterial(
        name=name,
        base_color=(r, g, b, a),
        shade_color=(0.78 * r, 0.55 * g, 0.50 * b),
        shading_shift=-0.1,
        shading_toony=0.3,
        gi_equalization=0.6,
    )


def to_vrm1(material: ToonMaterial) -> dict:
    """A glTF material with ``VRMC_materials_mtoon`` (VRM 1.0)."""
    # placeholder: stream A5
    return {
        "name": material.name,
        "pbrMetallicRoughness": {
            "baseColorFactor": list(material.base_color),
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "alphaMode": material.alpha_mode,
        "doubleSided": material.double_sided,
        "extensions": {
            "VRMC_materials_mtoon": {
                "specVersion": "1.0",
                "shadeColorFactor": list(material.shade_color),
                "shadingShiftFactor": material.shading_shift,
                "shadingToonyFactor": material.shading_toony,
                "giEqualizationFactor": material.gi_equalization,
            }
        },
    }


def to_vrm0(material: ToonMaterial) -> tuple[dict, dict]:
    """The glTF material and its VRM 0.x ``materialProperties`` entry."""
    # placeholder: stream A5
    gltf = {
        "name": material.name,
        "pbrMetallicRoughness": {
            "baseColorFactor": list(material.base_color),
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "alphaMode": material.alpha_mode,
        "doubleSided": material.double_sided,
    }
    props = {
        "name": material.name,
        "shader": "VRM/MToon",
        "renderQueue": 2000,
        "floatProperties": {},
        "vectorProperties": {},
        "textureProperties": {},
        "keywordMap": {},
        "tagMap": {"RenderType": "Opaque"},
    }
    return gltf, props


def v0_to_v1(props: dict) -> dict:
    """three-vrm's conversion of a 0.x ``materialProperties`` entry to MToon 1.0 factors."""
    raise NotImplementedError("stream A5")
