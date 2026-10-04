# OpenSculptBoy
# Apache License, Version 2.0
"""
MToon materials for VRM files: one version-neutral :class:`ToonMaterial` and two encoders.

``to_vrm1`` writes a glTF material with the ``VRMC_materials_mtoon`` extension (VRM 1.0, linear
colours) over a PBR fallback (metallic 0, roughness 1) for viewers without MToon. ``to_vrm0``
writes the same glTF material without the extension, and the VRM 0.x ``materialProperties``
entry: shader ``VRM/MToon`` with every property of the MToon 0.x shader, its keywords, its render
queue and its tags. A VRM 0.x file lists one entry per glTF material, in the order of the glTF
materials, with the same names.

VRM 0.x cannot hold every MToon 1.0 material. :func:`representable` projects a material onto what
0.x holds and says what it changed, and ``to_vrm0`` writes that projection. ``v0_to_v1`` repeats
three-vrm's conversion of a 0.x entry to MToon 1.0, and ``from_vrm1`` reads a 1.0 material back,
so that the tests check that a 0.x reader sees the material of the 1.0 file.

Colours: MToon 1.0 holds linear colours, and MToon 0.x holds the sRGB-encoded colours of Unity's
material properties, which Unity decodes with the exact sRGB curve in a linear colour space (and
uses as they are in a gamma colour space). VRM 0.x exists for VSeeFace, 3tene and the other
UniVRM 0.x apps, so ``to_vrm0`` writes the exact sRGB encoding of each linear colour
(:func:`srgb_encode`, Unity's ``Mathf.LinearToGammaSpace``), which those apps decode exactly.
three-vrm decodes the same values with a 2.2 power curve (:func:`gamma_decode`, which
``v0_to_v1`` repeats) and reads them up to 0.009 off in linear units (0.0085 at a linear 0.52).
That difference holds for every colour of every 0.x file, so :func:`representable` does not
report it.

Sources, at the commits that the line numbers below refer to:

- MToon 1.0: vrm-c/vrm-specification@94e82dd, ``specification/VRMC_materials_mtoon-1.0/README.md``
  and ``schema/VRMC_materials_mtoon.schema.json``.
- MToon 0.x: Santarh/MToon@42b0316, ``MToon/Resources/Shaders/MToon.shader`` (the properties and
  their defaults), ``MToonCore.cginc`` (the shading) and ``MToon/Scripts/UtilsSetter.cs`` and
  ``Utils.cs`` (the keywords, blend states and render queues of each render mode).
- three-vrm: pixiv/three-vrm@1b4fc0c,
  ``packages/three-vrm-materials-v0compat/src/VRMMaterialsV0CompatPlugin.ts`` (its 0.x reader).
- UniVRM: vrm-c/UniVRM@011b895, ``Packages/VRM10/MToon10/Runtime/MToon10Migrator.cs`` and
  ``Packages/VRM10/Runtime/Migration/Materials/MigrationMToonMaterial.cs`` (the 0.x reader of the
  Unity apps that read VRM 1.0), and
  ``Packages/VRM/Runtime/IO/MaterialIO/BuiltInRP/Import/Materials/BuiltInVrmMToonMaterialImporter.cs``
  (the 0.x importer of VSeeFace and the other UniVRM 0.x apps, which renders with MToon 0.x).
"""

from __future__ import annotations

import dataclasses
import math
import numbers

ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")
OUTLINE_WIDTH_MODES = ("none", "worldCoordinates", "screenCoordinates")

# MToon 0.x (MToon.shader line 38, UtilsVersion.cs line 6): the shader version of the 0.x entry.
MTOON_0X_VERSION = 39
# three-vrm reads an _IndirectLightIntensity of 0 as unset and falls back to its default
# giEqualizationFactor of 0.9 (VRMMaterialsV0CompatPlugin.ts lines 159-160:
# ``giIntensityFactor ? 1.0 - giIntensityFactor : undefined``), so the 0.x encoder never writes
# less than this, and the highest GI equalisation that VRM 0.x holds is 1 minus it.
MIN_INDIRECT_LIGHT_INTENSITY = 1e-4
MAX_GI_EQUALIZATION_0X = 1.0 - MIN_INDIRECT_LIGHT_INTENSITY
# The power of the gamma curve with which three-vrm decodes 0.x colours (gammaEOTF:
# ``Math.pow(e, 2.2)``, packages/three-vrm-materials-v0compat/src/utils/gammaEOTF.ts).
GAMMA = 2.2
# The ST vector (offset x, offset y, scale x, scale y) of a 0.x texture slot without a transform.
IDENTITY_ST = (0.0, 0.0, 1.0, 1.0)

# The 0.x render modes (Enums.cs RenderMode) and what MToon's SetRenderMode gives each of them
# (UtilsSetter.cs lines 137-196): the RenderType tag, _SrcBlend, _DstBlend, _ZWrite, _AlphaToMask,
# the alpha keyword and the base render queue (Utils.cs lines 74-113: Unity's Geometry queue 2000,
# AlphaTest 2450, Transparent 3000, and 2501 for the first transparent queue with depth writes).
# Unity's BlendMode: One 1, Zero 0, SrcAlpha 5, OneMinusSrcAlpha 10.
_OPAQUE, _CUTOUT, _TRANSPARENT, _TRANSPARENT_ZWRITE = 0, 1, 2, 3
_RENDER_MODES = {
    _OPAQUE: ("Opaque", 1, 0, 1, 0, None, 2000),
    _CUTOUT: ("TransparentCutout", 1, 0, 1, 1, "_ALPHATEST_ON", 2450),
    _TRANSPARENT: ("Transparent", 5, 10, 0, 0, "_ALPHABLEND_ON", 3000),
    _TRANSPARENT_ZWRITE: ("Transparent", 5, 10, 1, 0, "_ALPHABLEND_ON", 2501),
}
# Unity's CullMode (Enums.cs): Off 0, Front 1, Back 2. Outlines always cull the front faces
# (UtilsSetter.cs lines 239-258), as MToon 1.0 does (README, "Double Sided").
_CULL_OFF, _CULL_FRONT, _CULL_BACK = 0, 1, 2

# The texture slots of MToon 0.x, in the order of the shader's properties (MToon.shader lines
# 5-34). Each slot carries an ST vector (offset x, offset y, scale x, scale y) in
# vectorProperties, as UniVRM's 0.x exporter writes for every texture property (lines 69-101 of
# Packages/VRM/Runtime/IO/MaterialIO/BuiltInRP/Export/BuiltInVrmExtensionMaterialPropertyExporter.cs,
# which also writes Unity's sRGB-encoded colour values as they are, lines 53-58).
V0_TEXTURE_SLOTS = (
    "_MainTex",
    "_ShadeTexture",
    "_BumpMap",
    "_ReceiveShadowTexture",
    "_ShadingGradeTexture",
    "_RimTexture",
    "_SphereAdd",
    "_EmissionMap",
    "_OutlineWidthTexture",
    "_UvAnimMaskTexture",
)
# The float properties of MToon 0.x, in the order of the shader (MToon.shader lines 5-48).
V0_FLOAT_PROPERTIES = (
    "_Cutoff",
    "_BumpScale",
    "_ReceiveShadowRate",
    "_ShadingGradeRate",
    "_ShadeShift",
    "_ShadeToony",
    "_LightColorAttenuation",
    "_IndirectLightIntensity",
    "_RimLightingMix",
    "_RimFresnelPower",
    "_RimLift",
    "_OutlineWidth",
    "_OutlineScaledMaxDistance",
    "_OutlineLightingMix",
    "_UvAnimScrollX",
    "_UvAnimScrollY",
    "_UvAnimRotation",
    "_MToonVersion",
    "_DebugMode",
    "_BlendMode",
    "_OutlineWidthMode",
    "_OutlineColorMode",
    "_CullMode",
    "_OutlineCullMode",
    "_SrcBlend",
    "_DstBlend",
    "_ZWrite",
    "_AlphaToMask",
)
# The vector properties of MToon 0.x, in the order of the shader: the colours and the ST vectors.
V0_VECTOR_PROPERTIES = (
    "_Color",
    "_ShadeColor",
    "_MainTex",
    "_ShadeTexture",
    "_BumpMap",
    "_ReceiveShadowTexture",
    "_ShadingGradeTexture",
    "_RimColor",
    "_RimTexture",
    "_SphereAdd",
    "_EmissionColor",
    "_EmissionMap",
    "_OutlineWidthTexture",
    "_OutlineColor",
    "_UvAnimMaskTexture",
)


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
    # A tangent-space normal map and its scale (glTF normalTexture).
    normal_texture: int | None = None
    normal_scale: float = 1.0
    # An sRGB texture that multiplies the emissive colour (glTF emissiveTexture).
    emissive_texture: int | None = None
    # An sRGB mask that multiplies the rim lighting: the matcap and the parametric rim in MToon
    # 1.0, the parametric rim alone in MToon 0.x.
    rim_multiply_texture: int | None = None
    # A linear mask of the outline width: MToon 1.0 reads its G channel and MToon 0.x its R
    # channel (MToonCore.cginc line 86), so the mask must be grey for both versions to agree.
    outline_width_texture: int | None = None
    # Write depth in BLEND mode; renderQueueOffsetNumber then runs from 0 to 9 instead of -9 to 0.
    transparent_with_zwrite: bool = False


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


# ---------------------------------------------------------------------------------------------
# Checks and the shared glTF part


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_integer(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def queue_offset_range(material: ToonMaterial) -> tuple[int, int]:
    """
    The range of ``render_queue_offset`` for the material's alpha mode (MToon 1.0 README,
    "Render Queue"): 0 for OPAQUE and MASK, 0 to 9 for BLEND with depth writes, -9 to 0 for BLEND.
    """
    if material.alpha_mode != "BLEND":
        return 0, 0
    return (0, 9) if material.transparent_with_zwrite else (-9, 0)


def check(material: ToonMaterial) -> None:
    """Raise ValueError when a value of the material lies outside the glTF and MToon 1.0 schemas."""
    problems = []

    def finite(label, values):
        if not all(_is_number(v) and math.isfinite(v) for v in values):
            problems.append(f"{label} must be finite numbers, not {values}")
            return False
        return True

    def within(label, values, low=0.0, high=1.0):
        values = list(values)
        if finite(label, values) and not all(low <= v <= high for v in values):
            problems.append(f"{label} must lie in [{low}, {high}], not {values}")

    def colour(label, values, size):
        if len(values) != size:
            problems.append(f"{label} needs {size} values, not {len(values)}")
        within(label, values)

    m = material
    if m.alpha_mode not in ALPHA_MODES:
        problems.append(
            f"alpha_mode must be one of {ALPHA_MODES}, not {m.alpha_mode!r}"
        )
    if m.outline_width_mode not in OUTLINE_WIDTH_MODES:
        problems.append(
            f"outline_width_mode must be one of {OUTLINE_WIDTH_MODES}, not {m.outline_width_mode!r}"
        )
    colour("base_color", m.base_color, 4)
    for label in (
        "shade_color",
        "rim_color",
        "matcap_factor",
        "outline_color",
        "emissive",
    ):
        colour(label, getattr(m, label), 3)
    for label in (
        "shading_toony",
        "gi_equalization",
        "rim_lighting_mix",
        "outline_lighting_mix",
    ):
        within(label, [getattr(m, label)])
    for label in ("rim_fresnel_power", "outline_width", "alpha_cutoff"):
        within(label, [getattr(m, label)], 0.0, math.inf)
    finite(
        "shading_shift, rim_lift, normal_scale",
        [m.shading_shift, m.rim_lift, m.normal_scale],
    )
    for label in (
        "base_texture",
        "shade_texture",
        "matcap_texture",
        "normal_texture",
        "emissive_texture",
        "rim_multiply_texture",
        "outline_width_texture",
    ):
        index = getattr(m, label)
        if index is not None and not (_is_integer(index) and index >= 0):
            problems.append(f"{label} must be a glTF texture index, not {index!r}")
    low, high = queue_offset_range(m) if m.alpha_mode in ALPHA_MODES else (-9, 9)
    offset = m.render_queue_offset
    if not (_is_integer(offset) and low <= offset <= high):
        zwrite = " with depth writes" if _zwrite(m) else ""
        problems.append(
            f"render_queue_offset must be an integer in [{low}, {high}] for "
            f"{m.alpha_mode}{zwrite}, not {offset!r}"
        )
    if problems:
        raise ValueError(f"MToon material {m.name!r}: " + "; ".join(problems))


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def _texture(index: int) -> dict:
    return {"index": int(index)}


def _gltf_material(m: ToonMaterial) -> dict:
    """The core glTF material: the lit colour, the fallback PBR factors, emission and alpha."""
    pbr = {"baseColorFactor": _floats(m.base_color)}
    if m.base_texture is not None:
        pbr["baseColorTexture"] = _texture(m.base_texture)
    pbr["metallicFactor"] = 0.0
    pbr["roughnessFactor"] = 1.0
    out = {"name": m.name, "pbrMetallicRoughness": pbr}
    if m.normal_texture is not None:
        out["normalTexture"] = {
            **_texture(m.normal_texture),
            "scale": float(m.normal_scale),
        }
    if m.emissive_texture is not None:
        out["emissiveTexture"] = _texture(m.emissive_texture)
    out["emissiveFactor"] = _floats(m.emissive)
    out["alphaMode"] = m.alpha_mode
    if m.alpha_mode == "MASK":  # glTF ignores alphaCutoff in the other modes
        out["alphaCutoff"] = float(m.alpha_cutoff)
    out["doubleSided"] = bool(m.double_sided)
    return out


def _zwrite(m: ToonMaterial) -> bool:
    return m.alpha_mode == "BLEND" and bool(m.transparent_with_zwrite)


# ---------------------------------------------------------------------------------------------
# VRM 1.0


def to_vrm1(material: ToonMaterial) -> dict:
    """
    A glTF material with ``VRMC_materials_mtoon`` (VRM 1.0).

    Every factor of the extension is written, so that no reader falls back to its own default
    (the README and the schema disagree on the default shade colour, for example). Colours are
    linear, and texture slots are ``{"index": i}`` texture infos. The writer of the file adds
    ``VRMC_materials_mtoon`` to ``extensionsUsed``.
    """
    check(material)
    m = material
    out = _gltf_material(m)
    ext = {
        "specVersion": "1.0",
        "transparentWithZWrite": _zwrite(m),
        "renderQueueOffsetNumber": int(m.render_queue_offset),
        "shadeColorFactor": _floats(m.shade_color),
    }
    if m.shade_texture is not None:
        ext["shadeMultiplyTexture"] = _texture(m.shade_texture)
    ext["shadingShiftFactor"] = float(m.shading_shift)
    ext["shadingToonyFactor"] = float(m.shading_toony)
    ext["giEqualizationFactor"] = float(m.gi_equalization)
    ext["matcapFactor"] = _floats(m.matcap_factor)
    if m.matcap_texture is not None:
        ext["matcapTexture"] = _texture(m.matcap_texture)
    ext["parametricRimColorFactor"] = _floats(m.rim_color)
    if m.rim_multiply_texture is not None:
        ext["rimMultiplyTexture"] = _texture(m.rim_multiply_texture)
    ext["rimLightingMixFactor"] = float(m.rim_lighting_mix)
    ext["parametricRimFresnelPowerFactor"] = float(m.rim_fresnel_power)
    ext["parametricRimLiftFactor"] = float(m.rim_lift)
    ext["outlineWidthMode"] = m.outline_width_mode
    ext["outlineWidthFactor"] = float(m.outline_width)
    if m.outline_width_texture is not None:
        ext["outlineWidthMultiplyTexture"] = _texture(m.outline_width_texture)
    ext["outlineColorFactor"] = _floats(m.outline_color)
    ext["outlineLightingMixFactor"] = float(m.outline_lighting_mix)
    out["extensions"] = {"VRMC_materials_mtoon": ext}
    return out


# Defaults of the MToon 1.0 README tables, which three-vrm's MToonMaterial also uses
# (MToonMaterial.ts lines 399-433). The schema gives [1, 1, 1] for shadeColorFactor instead of
# the README's [0, 0, 0], which is why the encoder always writes it.
_V1_DEFAULTS = {
    "transparentWithZWrite": False,
    "renderQueueOffsetNumber": 0,
    "shadeColorFactor": [0.0, 0.0, 0.0],
    "shadingShiftFactor": 0.0,
    "shadingToonyFactor": 0.9,
    "giEqualizationFactor": 0.9,
    "matcapFactor": [1.0, 1.0, 1.0],
    "parametricRimColorFactor": [0.0, 0.0, 0.0],
    "rimLightingMixFactor": 1.0,
    "parametricRimFresnelPowerFactor": 5.0,
    "parametricRimLiftFactor": 0.0,
    "outlineWidthMode": "none",
    "outlineWidthFactor": 0.0,
    "outlineColorFactor": [0.0, 0.0, 0.0],
    "outlineLightingMixFactor": 1.0,
}


def from_vrm1(material: dict) -> ToonMaterial:
    """
    Read a glTF material with ``VRMC_materials_mtoon`` back into a :class:`ToonMaterial`.

    Missing factors take the defaults of the MToon 1.0 README. Only the first 3 components of a
    colour factor count (three-vrm's 0.x reader gives 4). Texture transforms and the UV animation
    are not read.
    """
    pbr = material.get("pbrMetallicRoughness") or {}
    ext = {
        **_V1_DEFAULTS,
        **(material.get("extensions") or {}).get("VRMC_materials_mtoon", {}),
    }

    def index(info):
        return None if info is None else int(info["index"])

    def colour(values, size=3):
        return tuple(float(v) for v in list(values)[:size])

    normal = material.get("normalTexture")
    return ToonMaterial(
        name=material.get("name", ""),
        base_color=colour(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), 4),
        base_texture=index(pbr.get("baseColorTexture")),
        shade_color=colour(ext["shadeColorFactor"]),
        shade_texture=index(ext.get("shadeMultiplyTexture")),
        shading_shift=float(ext["shadingShiftFactor"]),
        shading_toony=float(ext["shadingToonyFactor"]),
        gi_equalization=float(ext["giEqualizationFactor"]),
        rim_color=colour(ext["parametricRimColorFactor"]),
        rim_fresnel_power=float(ext["parametricRimFresnelPowerFactor"]),
        rim_lift=float(ext["parametricRimLiftFactor"]),
        rim_lighting_mix=float(ext["rimLightingMixFactor"]),
        matcap_texture=index(ext.get("matcapTexture")),
        matcap_factor=colour(ext["matcapFactor"]),
        outline_width_mode=ext["outlineWidthMode"],
        outline_width=float(ext["outlineWidthFactor"]),
        outline_color=colour(ext["outlineColorFactor"]),
        outline_lighting_mix=float(ext["outlineLightingMixFactor"]),
        emissive=colour(material.get("emissiveFactor", [0.0, 0.0, 0.0])),
        alpha_mode=material.get("alphaMode", "OPAQUE"),
        alpha_cutoff=float(material.get("alphaCutoff", 0.5)),
        double_sided=bool(material.get("doubleSided", False)),
        render_queue_offset=int(ext["renderQueueOffsetNumber"]),
        normal_texture=index(normal),
        normal_scale=float(normal.get("scale", 1.0)) if normal else 1.0,
        emissive_texture=index(material.get("emissiveTexture")),
        rim_multiply_texture=index(ext.get("rimMultiplyTexture")),
        outline_width_texture=index(ext.get("outlineWidthMultiplyTexture")),
        transparent_with_zwrite=bool(ext["transparentWithZWrite"]),
    )


# ---------------------------------------------------------------------------------------------
# The shade ramp in both versions
#
# MToon 1.0 (README lines 297-304): shading = linearstep(-1 + toony, 1 - toony, N.L + shift),
# a ramp from the shade colour at N.L = -1 + toony - shift to the lit colour at 1 - toony - shift.
# MToon 0.x (MToonCore.cginc lines 183-191, without shadows): the ramp runs from N.L = _ShadeShift
# to lerp(1, _ShadeShift, _ShadeToony). three-vrm (VRMMaterialsV0CompatPlugin.ts lines 153-157)
# and UniVRM (MToon10Migrator.cs lines 13-46) map the 0.x pair to the 1.0 pair with the same ramp;
# the two formulas are algebraically equal. The 0.x edges lie within [-1, 1], so 0.x holds exactly
# the 1.0 ramps whose edges lie within [-1, 1]: |shift| <= toony <= 1.


def shade_to_v1(shade_toony: float, shade_shift: float) -> tuple[float, float]:
    """
    The MToon 1.0 (shadingToonyFactor, shadingShiftFactor) of a 0.x (_ShadeToony, _ShadeShift),
    as three-vrm computes them (VRMMaterialsV0CompatPlugin.ts lines 154-157).
    """
    t = 0.5 + 0.5 * shade_shift
    toony = (1.0 - t) * shade_toony + t * 1.0  # THREE.MathUtils.lerp(x, y, t)
    shift = -shade_shift - (1.0 - toony)
    return toony, shift


def shade_to_v0(toony: float, shift: float) -> tuple[float, float]:
    """
    The MToon 0.x (_ShadeToony, _ShadeShift) whose shade ramp equals the MToon 1.0 ramp of
    (toony, shift): the exact inverse of :func:`shade_to_v1` on |shift| <= toony <= 1.

    _ShadeShift is the lower edge of the ramp, -1 + toony - shift, and _ShadeToony follows from
    the upper edge, lerp(1, _ShadeShift, _ShadeToony) = 1 - toony - shift. At toony = 1 and
    shift = -1 the ramp is a step at N.L = 1 and every _ShadeToony gives it; this returns 1.
    """
    toony, shift = float(toony), float(shift)
    if not (0.0 <= toony <= 1.0 and -toony - 1e-12 <= shift <= toony + 1e-12):
        raise ValueError(
            f"VRM 0.x cannot hold shading toony {toony} with shift {shift}; "
            "project them with project_shading first"
        )
    shade_shift = -1.0 + toony - shift
    span = 2.0 - toony + shift  # 1 - _ShadeShift
    shade_toony = (toony + shift) / span if span > 0.0 else 1.0
    return min(max(shade_toony, 0.0), 1.0), min(max(shade_shift, -1.0), 1.0)


def project_shading(toony: float, shift: float) -> tuple[float, float]:
    """
    The nearest (toony, shift) that VRM 0.x holds: the pair itself when |shift| <= toony <= 1;
    otherwise the ramp whose edges are the original edges clamped to [-1, 1], the range of N.L.

    When one edge lies outside, this keeps the edge within the range and moves the other to the
    end of the range, which is the Euclidean projection onto |shift| <= toony.
    """
    toony = min(max(float(toony), 0.0), 1.0)
    shift = float(shift)
    if -toony <= shift <= toony:
        return toony, shift
    if shift > toony:
        # The lower edge lies below -1: it goes to -1, and the upper edge stays (or goes to -1).
        # With the edges (-1, upper), toony = shift = (1 - upper) / 2.
        upper = max(1.0 - toony - shift, -1.0)
        side = 0.5 * (1.0 - upper)
        return side, side
    # The upper edge lies above 1: it goes to 1, and the lower edge stays (or goes to 1).
    # With the edges (lower, 1), toony = -shift = (1 + lower) / 2.
    lower = min(-1.0 + toony - shift, 1.0)
    side = 0.5 * (1.0 + lower)
    return side, -side


# ---------------------------------------------------------------------------------------------
# VRM 0.x


def srgb_encode(linear) -> list[float]:
    """
    Linear colour components in [0, 1] as a 0.x file stores them: the exact sRGB encoding, as
    Unity's ``Mathf.LinearToGammaSpace`` computes it (12.92 c up to 0.0031308, then
    1.055 c^(1/2.4) - 0.055). MToon 0.x in Unity decodes it exactly (:func:`srgb_decode`).
    """
    out = []
    for c in linear:
        c = float(c)
        out.append(12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055)
    return out


def srgb_decode(encoded) -> list[float]:
    """
    The exact sRGB decoding of each component, Unity's ``Mathf.GammaToLinearSpace`` (e / 12.92 up
    to 0.04045, then ((e + 0.055) / 1.055)^2.4): the inverse of :func:`srgb_encode`, within
    3e-9 where the two pieces of the curve meet.
    """
    out = []
    for e in encoded:
        e = float(e)
        out.append(e / 12.92 if e <= 0.04045 else ((e + 0.055) / 1.055) ** 2.4)
    return out


def gamma_encode(linear) -> list[float]:
    """
    Linear colour components as three-vrm's 2.2 power curve encodes them: linear^(1/2.2), the
    inverse of :func:`gamma_decode`. The 0.x encoder writes :func:`srgb_encode` instead, for the
    Unity apps; this pair serves to compare three-vrm's reading with it.
    """
    return [float(c) ** (1.0 / GAMMA) for c in linear]


def gamma_decode(encoded) -> list[float]:
    """three-vrm's gammaEOTF: ``Math.pow(e, 2.2)`` for each component."""
    return [float(c) ** GAMMA for c in encoded]


def _tidy(value: float) -> float:
    """A derived 0.x value rounded to 12 decimals, so that the file reads 0 where float error
    would leave 4e-17."""
    return round(float(value), 12) + 0.0  # + 0.0 turns -0.0 into 0.0


def _render_mode(m: ToonMaterial) -> int:
    if m.alpha_mode == "MASK":
        return _CUTOUT
    if m.alpha_mode == "BLEND":
        return _TRANSPARENT_ZWRITE if m.transparent_with_zwrite else _TRANSPARENT
    return _OPAQUE


def representable(material: ToonMaterial) -> tuple[ToonMaterial, list[str]]:
    """
    The material as VRM 0.x holds it, and one message for each value that changed and for each
    value that VRM 0.x readers read differently.

    Changes:

    - (shading_toony, shading_shift) outside |shift| <= toony <= 1 go to :func:`project_shading`.
    - gi_equalization above ``MAX_GI_EQUALIZATION_0X`` goes down to it, since the 0.x
      _IndirectLightIntensity must stay above 0.
    - matcap_factor goes to (1, 1, 1) when there is a matcap texture: MToon 0.x adds _SphereAdd
      without a factor (MToonCore.cginc lines 242-244), so the factor must be baked into the
      texture of the 0.x file.

    Notes, without changes:

    - a matcap texture with a rim lighting mix above 0 or a rim multiply texture: MToon 0.x adds
      the matcap unlit and unmasked, since _RimLightingMix and _RimTexture apply to the
      parametric rim alone (MToonCore.cginc lines 232-234) and _SphereAdd is added raw (lines
      243-244), where MToon 1.0 multiplies the matcap by the rim mask and mixes in the
      lighting;
    - a non-black emissive colour (three-vrm decodes _EmissionColor with the 2.2 gamma curve,
      UniVRM's 1.0 migration reads it as linear);
    - an outline width texture (R channel in 0.x, G in 1.0);
    - a base texture without a shade texture (UniVRM's 1.0 migration then multiplies the shade
      colour by the base texture, MigrationMToonMaterial.cs lines 183-202, while MToon 0.x and
      three-vrm do not).
    """
    check(material)
    m = material
    messages = []
    changes = {}
    toony, shift = project_shading(m.shading_toony, m.shading_shift)
    if (toony, shift) != (m.shading_toony, m.shading_shift):
        changes.update(shading_toony=toony, shading_shift=shift)
        messages.append(
            f"{m.name}: VRM 0.x cannot hold shading toony {m.shading_toony:g} with shift "
            f"{m.shading_shift:g} (the shade ramp must lie within N.L in [-1, 1]); "
            f"written as toony {toony:g}, shift {shift:g}"
        )
    if m.gi_equalization > MAX_GI_EQUALIZATION_0X:
        changes["gi_equalization"] = MAX_GI_EQUALIZATION_0X
        messages.append(
            f"{m.name}: GI equalisation {m.gi_equalization:g} written as "
            f"{MAX_GI_EQUALIZATION_0X:g} (_IndirectLightIntensity "
            f"{MIN_INDIRECT_LIGHT_INTENSITY:g}), since three-vrm reads 0 as unset"
        )
    if m.matcap_texture is not None and tuple(m.matcap_factor) != (1.0, 1.0, 1.0):
        changes["matcap_factor"] = (1.0, 1.0, 1.0)
        messages.append(
            f"{m.name}: VRM 0.x has no matcap factor; bake {tuple(m.matcap_factor)} into the "
            "matcap texture of the 0.x file"
        )
    if m.matcap_texture is not None and (
        m.rim_lighting_mix > 0.0 or m.rim_multiply_texture is not None
    ):
        messages.append(
            f"{m.name}: MToon 0.x adds the matcap unlit and unmasked (_RimLightingMix and "
            "_RimTexture apply to the parametric rim alone), where MToon 1.0 multiplies it by "
            "the rim mask and mixes in the lighting; the matcap of the 0.x file looks "
            "different in VSeeFace and 3tene"
        )
    if any(c > 0.0 for c in m.emissive):
        messages.append(
            f"{m.name}: VRM 0.x readers differ on the emissive colour (three-vrm decodes it with "
            "the 2.2 gamma curve, UniVRM's 1.0 migration reads it as linear)"
        )
    if m.outline_width_texture is not None:
        messages.append(
            f"{m.name}: MToon 0.x reads the R channel of the outline width texture and MToon 1.0 "
            "its G channel; keep the mask grey"
        )
    if m.base_texture is not None and m.shade_texture is None:
        messages.append(
            f"{m.name}: a base texture without a shade texture; UniVRM's 1.0 migration of the 0.x "
            "file multiplies the shade colour by the base texture, MToon 0.x and three-vrm do not"
        )
    return dataclasses.replace(m, **changes), messages


def to_vrm0(
    material: ToonMaterial, messages: list[str] | None = None
) -> tuple[dict, dict]:
    """
    The glTF material and its VRM 0.x ``materialProperties`` entry.

    The entry holds the projection of :func:`representable`; its messages are appended to
    ``messages`` when the caller passes a list. Every float and vector property of the MToon 0.x
    shader is written, in the shader's order, because the 0.x defaults differ from the 1.0 ones
    (for example _RimLightingMix 0, _RimFresnelPower 1 and a pink _ShadeColor) and UniVRM's
    importer takes the keywords, the render queue and the blend states from the file as they
    are, without MToon's own validation (BuiltInVrmMToonMaterialImporter.cs lines 138-185).

    Mapping (0.x property from the 1.0 value):

    - _Color, _ShadeColor, _RimColor, _EmissionColor, _OutlineColor: :func:`srgb_encode` of
      the linear colour, which MToon 0.x in Unity decodes exactly (three-vrm reads it up to
      0.009 off, see the module docstring), the alpha of _Color linear
      (VRMMaterialsV0CompatPlugin.ts lines 98-100).
    - _ShadeToony, _ShadeShift: :func:`shade_to_v0`.
    - _IndirectLightIntensity: 1 - giEqualizationFactor, at least ``MIN_INDIRECT_LIGHT_INTENSITY``.
    - _OutlineWidth: centimetres in world coordinates (100 x metres). In screen coordinates the
      MToon 0.x shader moves the vertex by 0.01 x _OutlineWidth in normalised device coordinates
      (MToonCore.cginc lines 92-101), where the screen is 2 high, so _OutlineWidth is 200 x the
      share of the screen height, as UniVRM's migration reads it (MigrationMToonMaterial.cs line
      307). three-vrm reads 0.01 x _OutlineWidth (line 195), twice the share, but its MToon 1.0
      shader also draws a share of half the screen height, so it draws the 0.x file at the same
      width as MToon 0.x does.
    - _OutlineColorMode: 1 (mixed lighting) when outlineLightingMixFactor > 0, else 0 (fixed),
      with _OutlineLightingMix the factor (VRMMaterialsV0CompatPlugin.ts lines 211-213).
    - _BlendMode, _CullMode, _SrcBlend, _DstBlend, _ZWrite, _AlphaToMask, the keywords, the render
      queue and the RenderType tag: as MToon's SetRenderMode and SetCullMode set them for the
      alpha mode (UtilsSetter.cs lines 137-258). MASK gives _ALPHATEST_ON, _BlendMode 1, queue
      2450 and TransparentCutout; BLEND gives queue 3000 + offset (2501 + offset with depth
      writes); double-sided materials get _CullMode 0.
    - keywordMap lists the enabled keywords only: three-vrm treats a material as transparent
      when ``_ALPHABLEND_ON`` is present at all, even as false (lines 361-364).
    - The properties without a 1.0 counterpart keep the MToon 0.x defaults (MToon.shader lines
      5-48): _ReceiveShadowRate 1, _ShadingGradeRate 1, _LightColorAttenuation 0, no UV
      animation, and _OutlineScaledMaxDistance 1, except 10 (the slider's top) for screen
      outlines, which then keep their width up to 10 m from the camera as in MToon 1.0.
    """
    projected, notes = representable(material)
    if messages is not None:
        messages.extend(notes)
    m = projected
    gltf = _gltf_material(m)
    shade_toony, shade_shift = shade_to_v0(m.shading_toony, m.shading_shift)
    mode = _render_mode(m)
    tag, src_blend, dst_blend, zwrite, alpha_to_mask, alpha_keyword, base_queue = (
        _RENDER_MODES[mode]
    )
    outline_mode = OUTLINE_WIDTH_MODES.index(m.outline_width_mode)
    outline_colour_mode = 1 if m.outline_lighting_mix > 0.0 else 0
    if m.outline_width_mode == "screenCoordinates":
        outline_width, scaled_max_distance = 200.0 * m.outline_width, 10.0
    else:
        outline_width, scaled_max_distance = 100.0 * m.outline_width, 1.0

    floats = {
        "_Cutoff": float(m.alpha_cutoff),
        "_BumpScale": float(m.normal_scale),
        "_ReceiveShadowRate": 1.0,
        "_ShadingGradeRate": 1.0,
        "_ShadeShift": _tidy(shade_shift),
        "_ShadeToony": _tidy(shade_toony),
        "_LightColorAttenuation": 0.0,
        "_IndirectLightIntensity": _tidy(
            max(MIN_INDIRECT_LIGHT_INTENSITY, 1.0 - float(m.gi_equalization))
        ),
        "_RimLightingMix": float(m.rim_lighting_mix),
        "_RimFresnelPower": float(m.rim_fresnel_power),
        "_RimLift": float(m.rim_lift),
        "_OutlineWidth": _tidy(outline_width),
        "_OutlineScaledMaxDistance": scaled_max_distance,
        "_OutlineLightingMix": float(m.outline_lighting_mix),
        "_UvAnimScrollX": 0.0,
        "_UvAnimScrollY": 0.0,
        "_UvAnimRotation": 0.0,
        "_MToonVersion": float(MTOON_0X_VERSION),
        "_DebugMode": 0.0,
        "_BlendMode": float(mode),
        "_OutlineWidthMode": float(outline_mode),
        "_OutlineColorMode": float(outline_colour_mode),
        "_CullMode": float(_CULL_OFF if m.double_sided else _CULL_BACK),
        "_OutlineCullMode": float(_CULL_FRONT),
        "_SrcBlend": float(src_blend),
        "_DstBlend": float(dst_blend),
        "_ZWrite": float(zwrite),
        "_AlphaToMask": float(alpha_to_mask),
    }
    colours = {
        "_Color": srgb_encode(m.base_color[:3]) + [float(m.base_color[3])],
        "_ShadeColor": srgb_encode(m.shade_color) + [1.0],
        "_RimColor": srgb_encode(m.rim_color) + [1.0],
        "_EmissionColor": srgb_encode(m.emissive) + [1.0],
        "_OutlineColor": srgb_encode(m.outline_color) + [1.0],
    }
    vectors = {
        name: colours[name] if name in colours else list(IDENTITY_ST)
        for name in V0_VECTOR_PROPERTIES
    }
    slots = {
        "_MainTex": m.base_texture,
        "_ShadeTexture": m.shade_texture,
        "_BumpMap": m.normal_texture,
        "_RimTexture": m.rim_multiply_texture,
        "_SphereAdd": m.matcap_texture,
        "_EmissionMap": m.emissive_texture,
        "_OutlineWidthTexture": m.outline_width_texture,
    }
    textures = {
        name: int(slots[name])
        for name in V0_TEXTURE_SLOTS
        if slots.get(name) is not None
    }
    keywords = {}
    if alpha_keyword is not None:
        keywords[alpha_keyword] = True
    if m.normal_texture is not None:
        keywords["_NORMALMAP"] = True
    if m.outline_width_mode != "none":  # UtilsSetter.cs lines 199-229
        world = m.outline_width_mode == "worldCoordinates"
        keywords[
            "MTOON_OUTLINE_WIDTH_WORLD" if world else "MTOON_OUTLINE_WIDTH_SCREEN"
        ] = True
        mixed = outline_colour_mode == 1
        keywords[
            "MTOON_OUTLINE_COLOR_MIXED" if mixed else "MTOON_OUTLINE_COLOR_FIXED"
        ] = True
    # BLEND: 3000 + offset (2991 to 3000), or 2501 + offset (2501 to 2510) with depth writes,
    # within the queues that MToon allows each mode (Utils.cs lines 96-109).
    queue = base_queue + (int(m.render_queue_offset) if m.alpha_mode == "BLEND" else 0)
    props = {
        "name": m.name,
        "shader": "VRM/MToon",
        "renderQueue": int(queue),
        "floatProperties": floats,
        "vectorProperties": vectors,
        "textureProperties": textures,
        "keywordMap": keywords,
        "tagMap": {"RenderType": tag},
    }
    return gltf, props


# ---------------------------------------------------------------------------------------------
# three-vrm's 0.x reader, for the tests


def _given(mapping: dict, key: str, default):
    """``mapping?.[key] ?? default`` in JavaScript: the default for a missing key or null."""
    value = mapping.get(key) if mapping else None
    return default if value is None else value


def _at(values, i: int, default):
    """``values?.[i] ?? default``."""
    if values is None or i >= len(values) or values[i] is None:
        return default
    return values[i]


def _js_string(number) -> str:
    """The string of a number as JavaScript's default ``Array.prototype.sort`` compares it."""
    number = float(number)
    return str(int(number)) if number.is_integer() else repr(number)


def _render_queue_flags(props: dict) -> tuple[bool, bool]:
    """(isTransparent, enabledZWrite) as lines 360-365 and 401-406 compute them."""
    shader = props.get("shader")
    zwrite_shader = shader == "VRM/UnlitTransparentZWrite"
    transparent = (
        (props.get("keywordMap") or {}).get("_ALPHABLEND_ON")
        is not None  # `!= undefined`
        or shader == "VRM/UnlitTransparent"
        or zwrite_shader
    )
    zwrite = (props.get("floatProperties") or {}).get("_ZWrite") == 1 or zwrite_shader
    return transparent, zwrite


def _render_queue_maps(materials: list[dict]) -> tuple[dict, dict]:
    """_populateRenderQueueMap (lines 388-448): v0 render queue to v1 offset, per transparency."""
    transparent, transparent_zwrite = set(), set()
    for props in materials:
        is_transparent, zwrite = _render_queue_flags(props)
        queue = props.get("renderQueue")
        if is_transparent and queue is not None:
            (transparent_zwrite if zwrite else transparent).add(queue)
    by_queue = {}
    for i, queue in enumerate(sorted(transparent, key=_js_string)):
        by_queue[queue] = min(max(i - len(transparent) + 1, -9), 0)
    by_queue_zwrite = {}
    for i, queue in enumerate(sorted(transparent_zwrite, key=_js_string)):
        by_queue_zwrite[queue] = min(max(i, 0), 9)
    return by_queue, by_queue_zwrite


def v0_to_v1(
    props: dict, material: dict | None = None, materials: list[dict] | None = None
) -> dict:
    """
    three-vrm's conversion of a 0.x ``materialProperties`` entry to a glTF material with MToon 1.0
    factors: a port of ``_parseV0MToonProperties`` (VRMMaterialsV0CompatPlugin.ts lines 79-281),
    with JavaScript's ``undefined`` as a missing key.

    Args:
        props: the ``materialProperties`` entry (shader ``VRM/MToon``).
        material: the glTF material of the same index, whose other keys (the name) the result
            keeps, as three-vrm spreads it.
        materials: every ``materialProperties`` entry of the file, from which three-vrm ranks the
            render queues of the transparent materials; ``[props]`` when None.

    Returns:
        The glTF material, as three-vrm's loader then reads it. Its colour factors have 4
        components where the 0.x vector has 4 (three-vrm reads the first 3), decoded with
        three-vrm's 2.2 power curve, so the sRGB colours of ``to_vrm0`` come back up to 0.009
        off; the UV animation rotation stays in 0.x's turns per second, as in three-vrm (UniVRM
        multiplies it by 2 pi).
    """
    floats = props.get("floatProperties") or {}
    vectors = props.get("vectorProperties") or {}
    textures = props.get("textureProperties") or {}
    keywords = props.get("keywordMap") or {}
    by_queue, by_queue_zwrite = _render_queue_maps(
        materials if materials is not None else [props]
    )

    # lines 83-94
    is_transparent = _given(keywords, "_ALPHABLEND_ON", False)
    enabled_zwrite = floats.get("_ZWrite") == 1
    transparent_with_zwrite = bool(enabled_zwrite and is_transparent)
    # _v0ParseRenderQueue, lines 359-382
    queue_transparent, queue_zwrite = _render_queue_flags(props)
    render_queue_offset = 0
    if queue_transparent and props.get("renderQueue") is not None:
        queue_map = by_queue_zwrite if queue_zwrite else by_queue
        render_queue_offset = queue_map.get(props["renderQueue"])
    is_cutoff = _given(keywords, "_ALPHATEST_ON", False)
    alpha_mode = "BLEND" if is_transparent else "MASK" if is_cutoff else "OPAQUE"
    alpha_cutoff = _given(floats, "_Cutoff", 0.5) if is_cutoff else None
    double_sided = _given(floats, "_CullMode", 2) == 0

    # _portTextureTransform, lines 338-353
    transform = {}
    st = vectors.get("_MainTex")
    if st is not None:
        offset = [_at(st, 0, 0.0), _at(st, 1, 0.0)]
        scale = [_at(st, 2, 1.0), _at(st, 3, 1.0)]
        offset[1] = 1.0 - scale[1] - offset[1]
        transform = {"KHR_texture_transform": {"offset": offset, "scale": scale}}

    def texture(slot: str, scale: float | None = None, transformed: bool = True):
        index = textures.get(slot)
        if index is None:
            return None
        info = {"index": index}
        if scale is not None:
            info["scale"] = scale
        if transformed:
            info["extensions"] = dict(transform)
        return info

    # lines 98-151
    base_color = [
        v if i == 3 else v**GAMMA
        for i, v in enumerate(_given(vectors, "_Color", [1.0, 1.0, 1.0, 1.0]))
    ]
    normal_texture = texture("_BumpMap", scale=_given(floats, "_BumpScale", 1.0))
    emissive = gamma_decode(_given(vectors, "_EmissionColor", [0.0, 0.0, 0.0, 1.0]))
    shade_color = gamma_decode(_given(vectors, "_ShadeColor", [0.97, 0.81, 0.86, 1.0]))
    # lines 153-157
    shading_toony, shading_shift = shade_to_v1(
        _given(floats, "_ShadeToony", 0.9), _given(floats, "_ShadeShift", 0.0)
    )
    # lines 159-160: JavaScript's truthiness, so 0 (and NaN) give undefined
    gi_intensity = _given(floats, "_IndirectLightIntensity", 0.1)
    gi_equalization = (
        1.0 - gi_intensity if gi_intensity and not math.isnan(gi_intensity) else None
    )
    # lines 162-169
    has_matcap = textures.get("_SphereAdd") is not None
    # lines 189-195
    width_mode = _given(floats, "_OutlineWidthMode", 0)
    outline_width_mode = (
        OUTLINE_WIDTH_MODES[int(width_mode)]
        if float(width_mode).is_integer() and 0 <= width_mode < 3
        else None
    )
    # lines 211-213
    outline_colour_mode = _given(floats, "_OutlineColorMode", 0)
    outline_lighting_mix = (
        _given(floats, "_OutlineLightingMix", 1.0) if outline_colour_mode == 1 else 0.0
    )

    ext = {
        "specVersion": "1.0",
        "transparentWithZWrite": transparent_with_zwrite,
        "renderQueueOffsetNumber": render_queue_offset,
        "shadeColorFactor": shade_color,
        "shadeMultiplyTexture": texture("_ShadeTexture"),
        "shadingShiftFactor": shading_shift,
        "shadingToonyFactor": shading_toony,
        "giEqualizationFactor": gi_equalization,
        "matcapFactor": [1.0, 1.0, 1.0] if has_matcap else None,
        "matcapTexture": texture("_SphereAdd", transformed=False),
        "rimLightingMixFactor": _given(floats, "_RimLightingMix", 0.0),
        "rimMultiplyTexture": texture("_RimTexture"),
        "parametricRimColorFactor": gamma_decode(
            _given(vectors, "_RimColor", [0.0, 0.0, 0.0, 1.0])
        ),
        "parametricRimFresnelPowerFactor": _given(floats, "_RimFresnelPower", 1.0),
        "parametricRimLiftFactor": _given(floats, "_RimLift", 0.0),
        "outlineWidthMode": outline_width_mode,
        "outlineWidthFactor": 0.01 * _given(floats, "_OutlineWidth", 0.0),
        "outlineWidthMultiplyTexture": texture("_OutlineWidthTexture"),
        "outlineColorFactor": gamma_decode(
            _given(vectors, "_OutlineColor", [0.0, 0.0, 0.0])
        ),
        "outlineLightingMixFactor": outline_lighting_mix,
        "uvAnimationMaskTexture": texture("_UvAnimMaskTexture"),
        "uvAnimationScrollXSpeedFactor": _given(floats, "_UvAnimScrollX", 0.0),
        # lines 228-232: the Y scroll is opposite between 0.x and 1.0
        "uvAnimationScrollYSpeedFactor": -_given(floats, "_UvAnimScrollY", 0.0),
        "uvAnimationRotationSpeedFactor": _given(floats, "_UvAnimRotation", 0.0),
    }
    pbr = {
        "baseColorFactor": base_color,
        "baseColorTexture": texture("_MainTex"),
    }
    overrides = {
        "pbrMetallicRoughness": {k: v for k, v in pbr.items() if v is not None},
        "normalTexture": normal_texture,
        "emissiveTexture": texture("_EmissionMap"),
        "emissiveFactor": emissive,
        "alphaMode": alpha_mode,
        "alphaCutoff": alpha_cutoff,
        "doubleSided": double_sided,
        "extensions": {
            "VRMC_materials_mtoon": {k: v for k, v in ext.items() if v is not None}
        },
    }
    # lines 263-280: ``{...schemaMaterial, ...}``; an undefined value hides the original key
    out = dict(material or {})
    for key, value in overrides.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out
