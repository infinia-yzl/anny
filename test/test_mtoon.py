# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the MToon encoders (opensculptboy.export.mtoon): the VRM 1.0 material against the
structure of the ``VRMC_materials_mtoon`` schema, and the VRM 0.x entry through a port of
three-vrm's 0.x reader, which must give back the 1.0 material on the domain that 0.x holds.

The schema files of vrm-c/vrm-specification carry no licence, so the structure they define is
written out here (``MTOON1_PROPERTIES``) instead of being shipped.
"""

import dataclasses
import json
import math
import random
import unittest

import numpy as np

from opensculptboy.export import mtoon
from opensculptboy.export.mtoon import ToonMaterial

# VRMC_materials_mtoon.schema.json (vrm-c/vrm-specification@94e82dd): each property with its
# type and range. "colour" is an array of 3 numbers in [0, 1]; "texture" a textureInfo.
MTOON1_PROPERTIES = {
    "specVersion": ("string",),
    "transparentWithZWrite": ("boolean",),
    "renderQueueOffsetNumber": ("integer", -9, 9),
    "shadeColorFactor": ("colour",),
    "shadeMultiplyTexture": ("texture",),
    "shadingShiftFactor": ("number", -math.inf, math.inf),
    "shadingShiftTexture": ("texture",),
    "shadingToonyFactor": ("number", 0.0, 1.0),
    "giEqualizationFactor": ("number", 0.0, 1.0),
    "matcapFactor": ("colour",),
    "matcapTexture": ("texture",),
    "parametricRimColorFactor": ("colour",),
    "rimMultiplyTexture": ("texture",),
    "rimLightingMixFactor": ("number", 0.0, 1.0),
    "parametricRimFresnelPowerFactor": ("number", 0.0, math.inf),
    "parametricRimLiftFactor": ("number", -math.inf, math.inf),
    "outlineWidthMode": ("enum", ("none", "worldCoordinates", "screenCoordinates")),
    "outlineWidthFactor": ("number", 0.0, math.inf),
    "outlineWidthMultiplyTexture": ("texture",),
    "outlineColorFactor": ("colour",),
    "outlineLightingMixFactor": ("number", 0.0, 1.0),
    "uvAnimationMaskTexture": ("texture",),
    "uvAnimationScrollXSpeedFactor": ("number", -math.inf, math.inf),
    "uvAnimationScrollYSpeedFactor": ("number", -math.inf, math.inf),
    "uvAnimationRotationSpeedFactor": ("number", -math.inf, math.inf),
    "extensions": ("object",),
    "extras": ("object",),
}
MTOON1_REQUIRED = {"specVersion"}
# The factors that the encoder writes for every material, so that no reader's default applies.
MTOON1_WRITTEN = {
    "specVersion",
    "transparentWithZWrite",
    "renderQueueOffsetNumber",
    "shadeColorFactor",
    "shadingShiftFactor",
    "shadingToonyFactor",
    "giEqualizationFactor",
    "matcapFactor",
    "parametricRimColorFactor",
    "rimLightingMixFactor",
    "parametricRimFresnelPowerFactor",
    "parametricRimLiftFactor",
    "outlineWidthMode",
    "outlineWidthFactor",
    "outlineColorFactor",
    "outlineLightingMixFactor",
}
# glTF 2.0 material.schema.json and material.pbrMetallicRoughness.schema.json.
GLTF_MATERIAL_KEYS = {
    "name",
    "pbrMetallicRoughness",
    "normalTexture",
    "occlusionTexture",
    "emissiveTexture",
    "emissiveFactor",
    "alphaMode",
    "alphaCutoff",
    "doubleSided",
    "extensions",
    "extras",
}
GLTF_PBR_KEYS = {
    "baseColorFactor",
    "baseColorTexture",
    "metallicFactor",
    "roughnessFactor",
    "metallicRoughnessTexture",
}
# The 0.x properties that the task names, written out independently of the module's tables.
V0_FLOATS = [
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
]
V0_COLOURS = ["_Color", "_ShadeColor", "_RimColor", "_EmissionColor", "_OutlineColor"]
V0_TEXTURES = [
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
]
SOFT_COLOURS = [
    (0.80, 0.62, 0.52, 1.0),
    (0.95, 0.85, 0.80, 1.0),
    (0.30, 0.18, 0.12, 1.0),
    (0.10, 0.10, 0.12, 1.0),
    (1.0, 1.0, 1.0, 1.0),
]


def random_material(rng: random.Random, name: str, **overrides) -> ToonMaterial:
    """
    A random material within the domain that VRM 0.x holds (representable leaves it alone). The
    values that no file carries keep their defaults: the cutoff outside MASK, the normal scale
    without a normal texture and depth writes outside BLEND.
    """

    def colour(size=3):
        return tuple(rng.random() for _ in range(size))

    def texture():
        return rng.randrange(8) if rng.random() < 0.4 else None

    toony = rng.random()
    alpha_mode = rng.choice(mtoon.ALPHA_MODES)
    kwargs = dict(
        name=name,
        base_color=colour(4),
        base_texture=texture(),
        shade_color=colour(),
        shade_texture=texture(),
        shading_toony=toony,
        shading_shift=rng.uniform(-toony, toony),
        gi_equalization=rng.uniform(0.0, mtoon.MAX_GI_EQUALIZATION_0X),
        rim_color=colour(),
        rim_fresnel_power=rng.uniform(0.0, 20.0),
        rim_lift=rng.uniform(-0.5, 0.5),
        rim_lighting_mix=rng.random(),
        matcap_texture=texture(),
        outline_width_mode=rng.choice(["none", "worldCoordinates"]),
        outline_width=rng.uniform(0.0, 0.01),
        outline_color=colour(),
        outline_lighting_mix=rng.choice([0.0, rng.random()]),
        emissive=colour(),
        alpha_mode=alpha_mode,
        alpha_cutoff=rng.random(),
        double_sided=rng.random() < 0.5,
        normal_texture=texture(),
        normal_scale=rng.uniform(0.0, 2.0),
        emissive_texture=texture(),
        rim_multiply_texture=texture(),
        outline_width_texture=texture(),
        transparent_with_zwrite=rng.random() < 0.5,
    )
    kwargs.update(overrides)
    if kwargs["alpha_mode"] != "MASK":
        kwargs["alpha_cutoff"] = 0.5
    if kwargs["normal_texture"] is None:
        kwargs["normal_scale"] = 1.0
    if kwargs["alpha_mode"] != "BLEND":
        kwargs["transparent_with_zwrite"] = False
    return ToonMaterial(**kwargs)


def ramp_v1(toony, shift, x):
    """MToon 1.0 shading (README lines 297-304): linearstep(-1 + toony, 1 - toony, N.L + shift)."""
    a, b = -1.0 + toony, 1.0 - toony
    return np.clip((x + shift - a) / (b - a), 0.0, 1.0)


def ramp_v0(shade_toony, shade_shift, x):
    """MToon 0.x shading (MToonCore.cginc lines 183-191), without shadows and shading grade."""
    upper = 1.0 + shade_toony * (shade_shift - 1.0)  # lerp(1, _ShadeShift, _ShadeToony)
    lower = shade_shift
    return np.clip((x - lower) / max(1e-5, upper - lower), 0.0, 1.0)


def univrm_shading(shade_toony, shade_shift):
    """UniVRM's MToon10Migrator (lines 13-46) in float64: the 0.x pair as a 1.0 pair."""
    lower = shade_shift
    upper = shade_shift + (1.0 - shade_shift) * (1.0 - min(max(shade_toony, 0.0), 1.0))
    toony = min(max((2.0 - (upper - lower)) * 0.5, 0.0), 1.0)
    shift = min(max((upper + lower) * 0.5 * -1.0, -1.0), 1.0)
    return toony, shift


def srgb_eotf(c):
    """The exact sRGB transfer function with which Unity decodes colour properties."""
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# The colour fields of a material, which the 0.x file holds sRGB-encoded.
COLOUR_FIELDS = ("base_color", "shade_color", "rim_color", "emissive", "outline_color")


def unity_colours(material: ToonMaterial) -> ToonMaterial:
    """
    A material read by three-vrm with its colours decoded again as MToon 0.x in Unity decodes
    them: three-vrm's 2.2 power is undone, and the exact sRGB curve is applied. The alpha of the
    base colour is linear in the file and stays as it is.
    """
    changes = {}
    for name in COLOUR_FIELDS:
        value = tuple(getattr(material, name))
        rgb = mtoon.srgb_decode(mtoon.gamma_encode(value[:3]))
        changes[name] = tuple(rgb) + value[3:]
    return dataclasses.replace(material, **changes)


def through_v0(material: ToonMaterial, materials=None) -> ToonMaterial:
    """
    The material of the 0.x entry of ``to_vrm0``, read back by three-vrm's conversion, with the
    colours decoded as MToon 0.x in Unity decodes them (VSeeFace and 3tene, which the 0.x file
    serves). :class:`TestColours` checks three-vrm's own reading of the colours.
    """
    gltf, props = mtoon.to_vrm0(material)
    return unity_colours(mtoon.from_vrm1(mtoon.v0_to_v1(props, gltf, materials)))


class MaterialAssertions:
    def assertMaterialsClose(self, a: ToonMaterial, b: ToonMaterial, tol=1e-6, skip=()):
        for field in dataclasses.fields(ToonMaterial):
            if field.name in skip:
                continue
            x, y = getattr(a, field.name), getattr(b, field.name)
            if isinstance(x, (tuple, list)):
                self.assertEqual(len(x), len(y), field.name)
                np.testing.assert_allclose(y, x, rtol=0, atol=tol, err_msg=field.name)
            elif isinstance(x, float):
                self.assertAlmostEqual(x, y, delta=tol, msg=field.name)
            else:
                self.assertEqual(x, y, field.name)


class TestVrm1(unittest.TestCase, MaterialAssertions):
    """The VRM 1.0 encoder against the schema, and its inverse."""

    def materials(self):
        rng = random.Random(1)
        out = [mtoon.soft_material(f"soft{i}", c) for i, c in enumerate(SOFT_COLOURS)]
        out += [random_material(rng, f"random{i}") for i in range(100)]
        out += [
            ToonMaterial(
                "screen", outline_width_mode="screenCoordinates", outline_width=0.01
            ),
            ToonMaterial("blend", alpha_mode="BLEND", render_queue_offset=-9),
            ToonMaterial(
                "zwrite",
                alpha_mode="BLEND",
                transparent_with_zwrite=True,
                render_queue_offset=9,
            ),
            ToonMaterial(
                "flat", shading_toony=1.0, shading_shift=-3.0, gi_equalization=1.0
            ),
        ]
        return out

    def check_texture(self, info, where, scale=False):
        allowed = {"index", "texCoord", "extensions", "extras"} | (
            {"scale"} if scale else set()
        )
        self.assertIsInstance(info, dict, where)
        self.assertLessEqual(set(info), allowed, where)
        self.assertIsInstance(info["index"], int, where)
        self.assertGreaterEqual(info["index"], 0, where)

    def check_number(self, value, low, high, where, integer=False):
        types = (int,) if integer else (int, float)
        self.assertIsInstance(value, types, where)
        self.assertNotIsInstance(value, bool, where)
        self.assertTrue(math.isfinite(value), where)
        self.assertTrue(
            low <= value <= high, f"{where}: {value} outside [{low}, {high}]"
        )

    def check_colour(self, value, size, where):
        self.assertIsInstance(value, list, where)
        self.assertEqual(len(value), size, where)
        for v in value:
            self.check_number(v, 0.0, 1.0, where)

    def check_schema(self, material: dict):
        """The structure of glTF's material schema and of VRMC_materials_mtoon.schema.json."""
        name = material.get("name")
        self.assertLessEqual(set(material), GLTF_MATERIAL_KEYS, name)
        pbr = material["pbrMetallicRoughness"]
        self.assertLessEqual(set(pbr), GLTF_PBR_KEYS, name)
        self.check_colour(pbr["baseColorFactor"], 4, f"{name} baseColorFactor")
        self.assertEqual(pbr["metallicFactor"], 0.0)
        self.assertEqual(pbr["roughnessFactor"], 1.0)
        for key in ("baseColorTexture",):
            if key in pbr:
                self.check_texture(pbr[key], f"{name} {key}")
        if "normalTexture" in material:
            self.check_texture(
                material["normalTexture"], f"{name} normalTexture", scale=True
            )
        if "emissiveTexture" in material:
            self.check_texture(material["emissiveTexture"], f"{name} emissiveTexture")
        self.check_colour(material["emissiveFactor"], 3, f"{name} emissiveFactor")
        self.assertIn(material["alphaMode"], ("OPAQUE", "MASK", "BLEND"))
        if material["alphaMode"] == "MASK":
            self.check_number(
                material["alphaCutoff"], 0.0, math.inf, f"{name} alphaCutoff"
            )
        else:
            self.assertNotIn("alphaCutoff", material, name)
        self.assertIsInstance(material["doubleSided"], bool)
        self.assertEqual(list(material["extensions"]), ["VRMC_materials_mtoon"])

        ext = material["extensions"]["VRMC_materials_mtoon"]
        self.assertLessEqual(MTOON1_REQUIRED, set(ext), name)
        self.assertLessEqual(MTOON1_WRITTEN, set(ext), name)
        self.assertLessEqual(set(ext), set(MTOON1_PROPERTIES), name)
        self.assertEqual(ext["specVersion"], "1.0")
        for key, value in ext.items():
            kind, *limits = MTOON1_PROPERTIES[key]
            where = f"{name} {key}"
            if kind == "string":
                self.assertIsInstance(value, str, where)
            elif kind == "boolean":
                self.assertIsInstance(value, bool, where)
            elif kind == "integer":
                self.check_number(value, *limits, where, integer=True)
            elif kind == "number":
                self.check_number(value, *limits, where)
            elif kind == "colour":
                self.check_colour(value, 3, where)
            elif kind == "texture":
                self.check_texture(value, where)
            elif kind == "enum":
                self.assertIn(value, limits[0], where)
            else:
                self.assertIsInstance(value, dict, where)
        # README "Render Queue": the offset range of each alpha mode.
        offset = ext["renderQueueOffsetNumber"]
        if material["alphaMode"] != "BLEND":
            self.assertEqual(offset, 0, name)
            self.assertFalse(ext["transparentWithZWrite"], name)
        elif ext["transparentWithZWrite"]:
            self.assertTrue(0 <= offset <= 9, name)
        else:
            self.assertTrue(-9 <= offset <= 0, name)
        json.dumps(material, allow_nan=False)

    def test_schema(self):
        for material in self.materials():
            with self.subTest(material=material.name):
                self.check_schema(mtoon.to_vrm1(material))

    def test_soft_values(self):
        m = mtoon.to_vrm1(mtoon.soft_material("skin", (0.8, 0.62, 0.52, 1.0)))
        ext = m["extensions"]["VRMC_materials_mtoon"]
        self.assertEqual(
            m["pbrMetallicRoughness"]["baseColorFactor"], [0.8, 0.62, 0.52, 1.0]
        )
        np.testing.assert_allclose(ext["shadeColorFactor"], [0.624, 0.341, 0.26])
        self.assertEqual(
            (
                ext["shadingToonyFactor"],
                ext["shadingShiftFactor"],
                ext["giEqualizationFactor"],
            ),
            (0.3, -0.1, 0.6),
        )

    def test_textures(self):
        m = ToonMaterial(
            "textured",
            base_texture=0,
            shade_texture=1,
            matcap_texture=2,
            normal_texture=3,
            normal_scale=0.5,
            emissive_texture=4,
            rim_multiply_texture=5,
            outline_width_texture=6,
            alpha_mode="MASK",
            alpha_cutoff=0.4,
        )
        out = mtoon.to_vrm1(m)
        ext = out["extensions"]["VRMC_materials_mtoon"]
        self.assertEqual(out["pbrMetallicRoughness"]["baseColorTexture"], {"index": 0})
        self.assertEqual(ext["shadeMultiplyTexture"], {"index": 1})
        self.assertEqual(ext["matcapTexture"], {"index": 2})
        self.assertEqual(out["normalTexture"], {"index": 3, "scale": 0.5})
        self.assertEqual(out["emissiveTexture"], {"index": 4})
        self.assertEqual(ext["rimMultiplyTexture"], {"index": 5})
        self.assertEqual(ext["outlineWidthMultiplyTexture"], {"index": 6})
        self.assertEqual((out["alphaMode"], out["alphaCutoff"]), ("MASK", 0.4))

    def test_inverse(self):
        for material in self.materials():
            with self.subTest(material=material.name):
                self.assertEqual(mtoon.from_vrm1(mtoon.to_vrm1(material)), material)

    def test_invalid(self):
        bad = [
            dict(shading_toony=1.5),
            dict(gi_equalization=-0.1),
            dict(base_color=(1.2, 0.0, 0.0, 1.0)),
            dict(shade_color=(0.5, 0.5)),
            dict(alpha_mode="CUTOUT"),
            dict(outline_width_mode="world"),
            dict(outline_width=-0.001),
            dict(render_queue_offset=3),
            dict(alpha_mode="BLEND", render_queue_offset=2),
            dict(
                alpha_mode="BLEND", transparent_with_zwrite=True, render_queue_offset=-1
            ),
            dict(base_texture=-1),
            dict(shading_shift=math.nan),
        ]
        for change in bad:
            with self.subTest(**{k: str(v) for k, v in change.items()}):
                material = ToonMaterial("bad", **change)
                with self.assertRaises(ValueError):
                    mtoon.to_vrm1(material)
                with self.assertRaises(ValueError):
                    mtoon.to_vrm0(material)

    def test_numpy_values(self):
        # Values from NumPy (a style table, a bake) encode as plain JSON numbers.
        material = ToonMaterial(
            "np",
            base_color=tuple(np.float32([0.5, 0.25, 0.125, 1.0])),
            shading_toony=np.float32(0.5),
            base_texture=np.int64(2),
            render_queue_offset=np.int64(0),
        )
        out = mtoon.to_vrm1(material)
        self.check_schema(out)
        self.assertIs(
            type(out["pbrMetallicRoughness"]["baseColorTexture"]["index"]), int
        )
        _, props = mtoon.to_vrm0(material)
        self.assertIs(type(props["textureProperties"]["_MainTex"]), int)
        json.dumps(props, allow_nan=False)


class TestVrm0(unittest.TestCase, MaterialAssertions):
    """The VRM 0.x encoder, read back with three-vrm's conversion."""

    def test_soft_round_trip(self):
        for colour in SOFT_COLOURS:
            with self.subTest(colour=colour):
                material = mtoon.soft_material("skin", colour)
                projected, messages = mtoon.representable(material)
                self.assertEqual(projected, material)
                self.assertEqual(messages, [])
                self.assertMaterialsClose(material, through_v0(material))

    def test_random_round_trip(self):
        rng = random.Random(2)
        for i in range(300):
            material = random_material(rng, f"m{i}")
            with self.subTest(material=material.name):
                self.assertEqual(mtoon.representable(material)[0], material)
                back = through_v0(material)
                # A file with one transparent material gives it the offset 0
                # (test_render_queue_offsets checks the order of several); 0.x has no matcap
                # factor, so three-vrm's default stands when there is no matcap texture.
                skip = ["render_queue_offset"]
                if material.matcap_texture is None:
                    skip.append("matcap_factor")
                self.assertMaterialsClose(material, back, skip=skip)

    def test_shading_ramps_agree(self):
        # The 0.x ramp of the encoded pair is the 1.0 ramp of the material at every N.L.
        rng = random.Random(3)
        x = np.linspace(-1.0, 1.0, 401)
        pairs = [
            (0.3, -0.1),
            (0.95, 0.0),
            (0.0, 0.0),
            (0.5, 0.5),
            (0.5, -0.5),
            (0.999, 0.9),
        ]
        for _ in range(300):
            toony = rng.uniform(0.0, 0.999)
            pairs.append((toony, rng.uniform(-toony, toony)))
        for toony, shift in pairs:
            with self.subTest(toony=toony, shift=shift):
                shade_toony, shade_shift = mtoon.shade_to_v0(toony, shift)
                self.assertTrue(
                    0.0 <= shade_toony <= 1.0 and -1.0 <= shade_shift <= 1.0
                )
                np.testing.assert_allclose(
                    ramp_v0(shade_toony, shade_shift, x),
                    ramp_v1(toony, shift, x),
                    atol=1e-9,
                )
                back = mtoon.shade_to_v1(shade_toony, shade_shift)
                np.testing.assert_allclose(back, (toony, shift), atol=1e-12)
                # UniVRM's migration of the 0.x pair agrees with three-vrm's.
                np.testing.assert_allclose(
                    univrm_shading(shade_toony, shade_shift), (toony, shift), atol=1e-12
                )

    def test_shading_domain(self):
        # Every 0.x pair maps into |shift| <= toony <= 1, and the corners map onto its corners.
        for shade_toony in np.linspace(0.0, 1.0, 21):
            for shade_shift in np.linspace(-1.0, 1.0, 41):
                toony, shift = mtoon.shade_to_v1(shade_toony, shade_shift)
                self.assertTrue(-1e-12 <= toony <= 1.0 + 1e-12)
                self.assertTrue(abs(shift) <= toony + 1e-12)
        np.testing.assert_allclose(mtoon.shade_to_v1(0.0, -1.0), (0.0, 0.0))
        np.testing.assert_allclose(mtoon.shade_to_v1(1.0, -1.0), (1.0, 1.0))
        np.testing.assert_allclose(mtoon.shade_to_v1(0.0, 1.0), (1.0, -1.0))
        # The singular corner: a step at N.L = 1, which every _ShadeToony gives.
        self.assertEqual(mtoon.shade_to_v0(1.0, -1.0), (1.0, 1.0))
        with self.assertRaises(ValueError):
            mtoon.shade_to_v0(0.2, 0.5)

    def test_projection(self):
        cases = {
            (0.1, 0.5): (0.3, 0.3),
            (0.2, -0.7): (0.45, -0.45),
            (1.0, 5.0): (1.0, 1.0),
            (0.0, -3.0): (1.0, -1.0),
            (0.3, -0.1): (0.3, -0.1),
            (0.9, 0.9): (0.9, 0.9),
        }
        for (toony, shift), expected in cases.items():
            with self.subTest(toony=toony, shift=shift):
                np.testing.assert_allclose(
                    mtoon.project_shading(toony, shift), expected
                )
        rng = random.Random(4)
        for _ in range(300):
            toony, shift = rng.random(), rng.uniform(-3.0, 3.0)
            projected = mtoon.project_shading(toony, shift)
            self.assertLessEqual(abs(projected[1]), projected[0] + 1e-12)
            self.assertEqual(mtoon.project_shading(*projected), projected)
            # The projected ramp has the original edges clamped to [-1, 1], the range of N.L.
            edges = np.array([-1.0 + toony - shift, 1.0 - toony - shift])
            new_edges = np.array(
                [-1.0 + projected[0] - projected[1], 1.0 - projected[0] - projected[1]]
            )
            np.testing.assert_allclose(new_edges, np.clip(edges, -1.0, 1.0), atol=1e-12)

    def test_out_of_range_reported(self):
        rng = random.Random(5)
        for i in range(100):
            toony = rng.random()
            shift = rng.choice([1, -1]) * rng.uniform(toony + 1e-3, toony + 2.0)
            material = random_material(
                rng,
                f"out{i}",
                shading_toony=toony,
                shading_shift=shift,
                matcap_texture=None,
            )
            with self.subTest(toony=toony, shift=shift):
                projected, notes = mtoon.representable(material)
                self.assertTrue(
                    any("cannot hold shading toony" in n for n in notes), notes
                )
                expected = mtoon.project_shading(toony, shift)
                self.assertEqual(
                    (projected.shading_toony, projected.shading_shift), expected
                )
                messages = []
                gltf, props = mtoon.to_vrm0(material, messages=messages)
                self.assertEqual(messages, notes)
                back = unity_colours(mtoon.from_vrm1(mtoon.v0_to_v1(props, gltf)))
                self.assertMaterialsClose(
                    projected, back, skip=["render_queue_offset", "matcap_factor"]
                )

    def test_indirect_light_never_zero(self):
        material = ToonMaterial("flat", shading_toony=0.9, gi_equalization=1.0)
        messages = []
        _, props = mtoon.to_vrm0(material, messages=messages)
        self.assertEqual(
            props["floatProperties"]["_IndirectLightIntensity"],
            mtoon.MIN_INDIRECT_LIGHT_INTENSITY,
        )
        self.assertTrue(any("GI equalisation" in m for m in messages), messages)
        ext = mtoon.v0_to_v1(props)["extensions"]["VRMC_materials_mtoon"]
        self.assertAlmostEqual(ext["giEqualizationFactor"], 1.0 - 1e-4, delta=1e-12)
        # Why: three-vrm reads an intensity of 0 as unset, which gives its default of 0.9.
        zero = json.loads(json.dumps(props))
        zero["floatProperties"]["_IndirectLightIntensity"] = 0.0
        self.assertNotIn(
            "giEqualizationFactor",
            mtoon.v0_to_v1(zero)["extensions"]["VRMC_materials_mtoon"],
        )
        rng = random.Random(6)
        for i in range(200):
            m = random_material(rng, f"m{i}", gi_equalization=rng.uniform(0.99, 1.0))
            intensity = mtoon.to_vrm0(m)[1]["floatProperties"][
                "_IndirectLightIntensity"
            ]
            self.assertGreaterEqual(intensity, mtoon.MIN_INDIRECT_LIGHT_INTENSITY)

    def test_every_property_written(self):
        rng = random.Random(7)
        materials = [mtoon.soft_material("soft", SOFT_COLOURS[0])]
        materials += [random_material(rng, f"m{i}") for i in range(50)]
        materials.append(ToonMaterial("screen", outline_width_mode="screenCoordinates"))
        for material in materials:
            with self.subTest(material=material.name):
                gltf, props = mtoon.to_vrm0(material)
                self.assertNotIn("extensions", gltf)
                self.assertEqual(gltf["name"], props["name"])
                self.assertEqual(
                    list(props),
                    [
                        "name",
                        "shader",
                        "renderQueue",
                        "floatProperties",
                        "vectorProperties",
                        "textureProperties",
                        "keywordMap",
                        "tagMap",
                    ],
                )
                self.assertEqual(props["shader"], "VRM/MToon")
                self.assertIsInstance(props["renderQueue"], int)
                # Every float of the MToon 0.x shader, in its order.
                self.assertEqual(list(props["floatProperties"]), V0_FLOATS)
                self.assertEqual(
                    list(props["floatProperties"]), list(mtoon.V0_FLOAT_PROPERTIES)
                )
                for key, value in props["floatProperties"].items():
                    self.assertIsInstance(value, float, key)
                    self.assertTrue(math.isfinite(value), key)
                # Every colour and every texture ST vector.
                vectors = props["vectorProperties"]
                self.assertEqual(set(vectors), set(V0_COLOURS) | set(V0_TEXTURES))
                self.assertEqual(list(vectors), list(mtoon.V0_VECTOR_PROPERTIES))
                for key, value in vectors.items():
                    self.assertEqual(len(value), 4, key)
                    self.assertTrue(all(isinstance(v, float) for v in value), key)
                    self.assertTrue(all(0.0 <= v <= 1.0 for v in value), key)
                for key in V0_TEXTURES:
                    self.assertEqual(vectors[key], [0.0, 0.0, 1.0, 1.0], key)
                # vrm.material.schema.json: integer textures, boolean keywords, string tags.
                for key, value in props["textureProperties"].items():
                    self.assertIn(key, V0_TEXTURES)
                    self.assertIsInstance(value, int, key)
                # Only enabled keywords: three-vrm reads any _ALPHABLEND_ON entry as transparent.
                self.assertTrue(all(v is True for v in props["keywordMap"].values()))
                self.assertEqual(list(props["tagMap"]), ["RenderType"])
                json.dumps(props, allow_nan=False)

    def test_texture_slots(self):
        material = ToonMaterial(
            "textured",
            base_texture=0,
            shade_texture=1,
            matcap_texture=2,
            normal_texture=3,
            normal_scale=0.5,
            emissive_texture=4,
            rim_multiply_texture=5,
            outline_width_texture=6,
        )
        gltf, props = mtoon.to_vrm0(material)
        self.assertEqual(
            props["textureProperties"],
            {
                "_MainTex": 0,
                "_ShadeTexture": 1,
                "_BumpMap": 3,
                "_RimTexture": 5,
                "_SphereAdd": 2,
                "_EmissionMap": 4,
                "_OutlineWidthTexture": 6,
            },
        )
        self.assertEqual(props["keywordMap"], {"_NORMALMAP": True})
        self.assertEqual(props["floatProperties"]["_BumpScale"], 0.5)
        self.assertEqual(gltf["normalTexture"], {"index": 3, "scale": 0.5})
        self.assertMaterialsClose(material, through_v0(material))

    def test_opaque(self):
        _, props = mtoon.to_vrm0(ToonMaterial("opaque"))
        floats = props["floatProperties"]
        self.assertEqual(props["renderQueue"], 2000)
        self.assertEqual(props["keywordMap"], {})
        self.assertEqual(props["tagMap"], {"RenderType": "Opaque"})
        self.assertEqual(
            [floats[k] for k in ("_BlendMode", "_SrcBlend", "_DstBlend", "_ZWrite")],
            [0.0, 1.0, 0.0, 1.0],
        )
        self.assertEqual((floats["_CullMode"], floats["_OutlineCullMode"]), (2.0, 1.0))
        self.assertEqual(floats["_AlphaToMask"], 0.0)

    def test_cutout(self):
        material = ToonMaterial(
            "hair", alpha_mode="MASK", alpha_cutoff=0.4, double_sided=True
        )
        gltf, props = mtoon.to_vrm0(material)
        floats = props["floatProperties"]
        self.assertEqual(props["keywordMap"], {"_ALPHATEST_ON": True})
        self.assertEqual(props["renderQueue"], 2450)
        self.assertEqual(props["tagMap"], {"RenderType": "TransparentCutout"})
        self.assertEqual(floats["_BlendMode"], 1.0)
        self.assertEqual(floats["_Cutoff"], 0.4)
        self.assertEqual(floats["_CullMode"], 0.0)
        self.assertEqual(floats["_OutlineCullMode"], 1.0)
        self.assertEqual(
            [floats[k] for k in ("_SrcBlend", "_DstBlend", "_ZWrite", "_AlphaToMask")],
            [1.0, 0.0, 1.0, 1.0],
        )
        self.assertEqual(
            (gltf["alphaMode"], gltf["alphaCutoff"], gltf["doubleSided"]),
            ("MASK", 0.4, True),
        )
        back = mtoon.v0_to_v1(props, gltf)
        self.assertEqual((back["alphaMode"], back["alphaCutoff"]), ("MASK", 0.4))
        self.assertTrue(back["doubleSided"])

    def test_transparent(self):
        for zwrite, offset, queue, blend_mode in (
            (False, -3, 2997, 2.0),
            (True, 4, 2505, 3.0),
        ):
            with self.subTest(zwrite=zwrite):
                material = ToonMaterial(
                    "blend",
                    alpha_mode="BLEND",
                    transparent_with_zwrite=zwrite,
                    render_queue_offset=offset,
                )
                _, props = mtoon.to_vrm0(material)
                floats = props["floatProperties"]
                self.assertEqual(props["renderQueue"], queue)
                self.assertEqual(props["keywordMap"], {"_ALPHABLEND_ON": True})
                self.assertEqual(props["tagMap"], {"RenderType": "Transparent"})
                self.assertEqual(floats["_BlendMode"], blend_mode)
                self.assertEqual(
                    (floats["_SrcBlend"], floats["_DstBlend"]), (5.0, 10.0)
                )
                self.assertEqual(floats["_ZWrite"], 1.0 if zwrite else 0.0)
                back = mtoon.from_vrm1(mtoon.v0_to_v1(props))
                self.assertEqual(
                    (back.alpha_mode, back.transparent_with_zwrite), ("BLEND", zwrite)
                )

    def test_render_queue_offsets(self):
        # three-vrm ranks the queues of a file's transparent materials: when they use every
        # offset of their range the offsets come back, and otherwise their order holds.
        for zwrite, offsets in ((False, range(-9, 1)), (True, range(0, 10))):
            with self.subTest(zwrite=zwrite):
                materials = [
                    ToonMaterial(
                        f"m{o}",
                        alpha_mode="BLEND",
                        transparent_with_zwrite=zwrite,
                        render_queue_offset=o,
                    )
                    for o in offsets
                ]
                entries = [mtoon.to_vrm0(m)[1] for m in materials]
                back = [
                    mtoon.from_vrm1(mtoon.v0_to_v1(e, None, entries)) for e in entries
                ]
                self.assertEqual([b.render_queue_offset for b in back], list(offsets))
        materials = [
            ToonMaterial("a", alpha_mode="BLEND", render_queue_offset=-5),
            ToonMaterial("b", alpha_mode="BLEND", render_queue_offset=-1),
            ToonMaterial("c", alpha_mode="BLEND", render_queue_offset=-7),
            ToonMaterial("d"),
        ]
        entries = [mtoon.to_vrm0(m)[1] for m in materials]
        back = [mtoon.from_vrm1(mtoon.v0_to_v1(e, None, entries)) for e in entries]
        self.assertEqual([b.render_queue_offset for b in back], [-1, 0, -2, 0])

    def test_outlines(self):
        world = ToonMaterial(
            "anime", outline_width_mode="worldCoordinates", outline_width=0.0008
        )
        _, props = mtoon.to_vrm0(world)
        floats = props["floatProperties"]
        self.assertAlmostEqual(floats["_OutlineWidth"], 0.08)  # centimetres
        self.assertEqual(floats["_OutlineWidthMode"], 1.0)
        self.assertEqual(floats["_OutlineColorMode"], 1.0)
        self.assertEqual(
            props["keywordMap"],
            {"MTOON_OUTLINE_WIDTH_WORLD": True, "MTOON_OUTLINE_COLOR_MIXED": True},
        )
        self.assertMaterialsClose(world, through_v0(world))

        fixed = dataclasses.replace(world, outline_lighting_mix=0.0)
        _, props = mtoon.to_vrm0(fixed)
        self.assertEqual(props["floatProperties"]["_OutlineColorMode"], 0.0)
        self.assertIn("MTOON_OUTLINE_COLOR_FIXED", props["keywordMap"])
        self.assertMaterialsClose(fixed, through_v0(fixed))

        none = ToonMaterial("plain")
        _, props = mtoon.to_vrm0(none)
        self.assertEqual(props["floatProperties"]["_OutlineWidthMode"], 0.0)
        self.assertEqual(props["keywordMap"], {})

    def test_screen_outlines(self):
        # The share of the screen height f: MToon 0.x moves the vertex by 0.01 x _OutlineWidth in
        # normalised device coordinates, which span 2 screen heights, and UniVRM reads
        # 0.005 x _OutlineWidth; three-vrm reads 0.01 x _OutlineWidth and draws it as a share of
        # half the screen height.
        share = 0.004
        material = ToonMaterial(
            "screen", outline_width_mode="screenCoordinates", outline_width=share
        )
        _, props = mtoon.to_vrm0(material)
        floats = props["floatProperties"]
        width = floats["_OutlineWidth"]
        self.assertAlmostEqual(0.01 * width, 2.0 * share)  # the MToon 0.x offset in NDC
        self.assertAlmostEqual(
            width * 0.01 * 0.5, share
        )  # MigrationMToonMaterial.cs line 307
        self.assertEqual(floats["_OutlineWidthMode"], 2.0)
        self.assertEqual(floats["_OutlineScaledMaxDistance"], 10.0)
        self.assertIn("MTOON_OUTLINE_WIDTH_SCREEN", props["keywordMap"])
        back = through_v0(material)
        self.assertAlmostEqual(back.outline_width, 2.0 * share)
        self.assertMaterialsClose(material, back, skip=["outline_width"])

    def test_matcap_factor(self):
        material = ToonMaterial(
            "hair", matcap_texture=3, matcap_factor=(0.5, 0.25, 1.0)
        )
        projected, messages = mtoon.representable(material)
        self.assertEqual(projected.matcap_factor, (1.0, 1.0, 1.0))
        self.assertTrue(any("bake" in m for m in messages), messages)
        _, props = mtoon.to_vrm0(material)
        self.assertEqual(props["textureProperties"]["_SphereAdd"], 3)
        ext = mtoon.v0_to_v1(props)["extensions"]["VRMC_materials_mtoon"]
        self.assertEqual(ext["matcapFactor"], [1.0, 1.0, 1.0])
        self.assertEqual(ext["matcapTexture"], {"index": 3})
        # Without a matcap texture, the factor has no effect and nothing changes.
        plain = ToonMaterial("skin", matcap_factor=(0.5, 0.5, 0.5))
        self.assertEqual(mtoon.representable(plain), (plain, []))

    def test_notes(self):
        notes = mtoon.representable(ToonMaterial("glow", emissive=(0.2, 0.0, 0.0)))[1]
        self.assertTrue(any("emissive" in n for n in notes), notes)
        notes = mtoon.representable(ToonMaterial("lines", outline_width_texture=2))[1]
        self.assertTrue(any("R channel" in n for n in notes), notes)
        notes = mtoon.representable(ToonMaterial("skin", base_texture=0))[1]
        self.assertTrue(any("shade texture" in n for n in notes), notes)
        material = ToonMaterial("skin", base_texture=0, shade_texture=0)
        self.assertEqual(mtoon.representable(material), (material, []))

    def test_matcap_note(self):
        # MToon 0.x adds the matcap unlit and unmasked (MToonCore.cginc lines 232-242), so a
        # matcap that MToon 1.0 mixes with the lighting or masks looks different in 0.x apps.
        cases = {
            "lit": (dict(matcap_texture=3), True),
            "masked": (
                dict(matcap_texture=3, rim_lighting_mix=0.0, rim_multiply_texture=1),
                True,
            ),
            "unlit": (dict(matcap_texture=3, rim_lighting_mix=0.0), False),
            "no matcap": (dict(rim_multiply_texture=1), False),
        }
        for case, (fields, noted) in cases.items():
            with self.subTest(case=case):
                material = ToonMaterial("hair", **fields)
                projected, notes = mtoon.representable(material)
                self.assertEqual(projected, material)
                self.assertEqual(
                    any("adds the matcap unlit and unmasked" in n for n in notes),
                    noted,
                    notes,
                )


class TestColours(unittest.TestCase):
    def test_round_trip(self):
        linear = np.linspace(0.0, 1.0, 100001)
        np.testing.assert_allclose(
            mtoon.srgb_decode(mtoon.srgb_encode(linear)), linear, rtol=0, atol=1e-8
        )
        # Unity's Mathf.GammaToLinearSpace is the exact sRGB curve of the reference function.
        encoded = np.linspace(0.0, 1.0, 1001)
        np.testing.assert_allclose(
            mtoon.srgb_decode(encoded), srgb_eotf(encoded), rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            mtoon.gamma_decode(mtoon.gamma_encode(linear)), linear, rtol=0, atol=1e-12
        )
        # three-vrm's gammaEOTF is Math.pow(e, 2.2).
        np.testing.assert_allclose(mtoon.gamma_decode([0.5]), [0.5**2.2])

    def test_material_colours(self):
        material = ToonMaterial(
            "c",
            base_color=(0.25, 0.5, 0.75, 0.5),
            shade_color=(0.1, 0.2, 0.3),
            rim_color=(0.4, 0.5, 0.6),
            emissive=(0.05, 0.0, 0.9),
            outline_color=(0.002, 0.01, 0.9),
        )
        _, props = mtoon.to_vrm0(material)
        vectors = props["vectorProperties"]
        # The alpha of _Color stays linear (VRMMaterialsV0CompatPlugin.ts lines 98-100).
        self.assertEqual(vectors["_Color"][3], 0.5)
        pairs = {
            "_Color": material.base_color[:3],
            "_ShadeColor": material.shade_color,
            "_RimColor": material.rim_color,
            "_EmissionColor": material.emissive,
            "_OutlineColor": material.outline_color,
        }
        for key, linear in pairs.items():
            with self.subTest(key=key):
                # MToon 0.x in Unity decodes the values with the exact sRGB curve.
                np.testing.assert_allclose(
                    srgb_eotf(vectors[key][:3]), linear, rtol=0, atol=1e-12
                )
                # three-vrm decodes them with a 2.2 power, within 0.009.
                np.testing.assert_allclose(
                    np.array(vectors[key][:3]) ** 2.2, linear, rtol=0, atol=0.009
                )
        back = mtoon.from_vrm1(mtoon.v0_to_v1(props))
        for field, key in zip(COLOUR_FIELDS, pairs):
            with self.subTest(field=field):
                np.testing.assert_allclose(
                    np.array(getattr(back, field))[:3],
                    np.array(vectors[key][:3]) ** 2.2,
                    rtol=0,
                    atol=1e-12,
                )

    def test_three_vrm_difference_bound(self):
        # three-vrm's reading of an sRGB-encoded colour lies at most 0.0085 off, at a linear
        # 0.52 (the README and the module docstring quote this bound).
        linear = np.linspace(0.0, 1.0, 100001)
        read = np.array(mtoon.gamma_decode(mtoon.srgb_encode(linear)))
        gap = np.abs(read - linear)
        self.assertLess(gap.max(), 0.009)
        self.assertGreater(gap.max(), 0.008)


if __name__ == "__main__":
    unittest.main()
