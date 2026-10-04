# OpenSculptBoy
# Apache License, Version 2.0
"""
The VRM extensions of the shared exports (test/vrm_fixtures.py) against the VRM schemas, and the
structural checks that the schemas leave out.

The schemas of vrm-c/vrm-specification carry no licence, so the repository never holds a copy:
:func:`load_schemas` downloads them, with the glTF base schemas that they refer to, at pinned
commits into ``ANNY_CACHE_DIR/vrm_schema/<commit>/``, and the schema tests skip when the download
fails (offline). The structural checks need no download, and they are the CI guard of both
versions:

- VRM 1.0: the ``VRMC_vrm`` extension (the humanoid bone names and their hierarchy, the expression
  presets and custom names, the indices of nodes, meshes, morph targets, images and textures), the
  ``VRMC_materials_mtoon`` and ``VRMC_node_constraint`` extensions, and ``extensionsUsed``;
- VRM 0.x, whose schemas are permissive (no required keys, open objects): the key spellings of
  UniVRM (``violentUssageName``, ``stiffiness``), the enumerations (the bone names, the expression
  presets, the licence names), the bind weights from 0 to 100, one ``materialProperties`` entry per
  glTF material in the same order, and the same indices.
"""

import concurrent.futures
import copy
import functools
import json
import pathlib
import unittest

import anny.paths
from opensculptboy.export import vrm_tables
from test import vrm_fixtures

# vrm-c/vrm-specification (master) and KhronosGroup/glTF (main), pinned.
VRM_SPEC_COMMIT = "94e82dd346fa6cf0337c4421728640e5252dd38e"
GLTF_COMMIT = "8e691206fa2e981fc3b8c8bc57e671595576c971"
VRM_SPEC_URL = (
    "https://raw.githubusercontent.com/vrm-c/vrm-specification/{commit}/{path}"
)
GLTF_URL = "https://raw.githubusercontent.com/KhronosGroup/glTF/{commit}/{path}"

# The schema files of each specification folder (specification/<folder>/schema/<file>).
VRM_SCHEMA_FILES = {
    "VRMC_vrm-1.0": (
        "VRMC_vrm.schema.json",
        "VRMC_vrm.meta.schema.json",
        "VRMC_vrm.humanoid.schema.json",
        "VRMC_vrm.humanoid.humanBones.schema.json",
        "VRMC_vrm.humanoid.humanBones.humanBone.schema.json",
        "VRMC_vrm.firstPerson.schema.json",
        "VRMC_vrm.firstPerson.meshAnnotation.schema.json",
        "VRMC_vrm.lookAt.schema.json",
        "VRMC_vrm.lookAt.rangeMap.schema.json",
        "VRMC_vrm.expressions.schema.json",
        "VRMC_vrm.expressions.expression.schema.json",
        "VRMC_vrm.expressions.expression.morphTargetBind.schema.json",
        "VRMC_vrm.expressions.expression.materialColorBind.schema.json",
        "VRMC_vrm.expressions.expression.textureTransformBind.schema.json",
    ),
    "VRMC_springBone-1.0": (
        "VRMC_springBone.schema.json",
        "VRMC_springBone.collider.schema.json",
        "VRMC_springBone.colliderGroup.schema.json",
        "VRMC_springBone.joint.schema.json",
        "VRMC_springBone.shape.schema.json",
        "VRMC_springBone.spring.schema.json",
    ),
    "VRMC_materials_mtoon-1.0": (
        "VRMC_materials_mtoon.schema.json",
        "mtoon.shadingShiftTexture.schema.json",
    ),
    "VRMC_node_constraint-1.0": (
        "VRMC_node_constraint.schema.json",
        "VRMC_node_constraint.constraint.schema.json",
        "VRMC_node_constraint.rollConstraint.schema.json",
        "VRMC_node_constraint.aimConstraint.schema.json",
        "VRMC_node_constraint.rotationConstraint.schema.json",
    ),
    "0.0": (
        "vrm.schema.json",
        "vrm.meta.schema.json",
        "vrm.humanoid.schema.json",
        "vrm.humanoid.bone.schema.json",
        "vrm.firstperson.schema.json",
        "vrm.firstperson.meshannotation.schema.json",
        "vrm.firstperson.degreemap.schema.json",
        "vrm.blendshape.schema.json",
        "vrm.blendshape.group.schema.json",
        "vrm.blendshape.bind.schema.json",
        "vrm.blendshape.materialbind.schema.json",
        "vrm.material.schema.json",
        "vrm.secondaryanimation.schema.json",
        "vrm.secondaryanimation.spring.schema.json",
        "vrm.secondaryanimation.collidergroup.schema.json",
    ),
}
# The glTF 2.0 base schemas that the VRM 1.0 schemas refer to, and those that they refer to.
GLTF_SCHEMA_FILES = (
    "glTFProperty.schema.json",
    "glTFChildOfRootProperty.schema.json",
    "glTFid.schema.json",
    "textureInfo.schema.json",
    "extensions.schema.json",
    "extras.schema.json",
)

# The VRM 1.0 humanoid bones, each with its parent in the humanoid hierarchy.
_LIMBS_1 = {
    "UpperLeg": "hips",
    "LowerLeg": "{side}UpperLeg",
    "Foot": "{side}LowerLeg",
    "Toes": "{side}Foot",
    "Shoulder": "upperChest",
    "UpperArm": "{side}Shoulder",
    "LowerArm": "{side}UpperArm",
    "Hand": "{side}LowerArm",
    "ThumbMetacarpal": "{side}Hand",
    "ThumbProximal": "{side}ThumbMetacarpal",
    "ThumbDistal": "{side}ThumbProximal",
    **{
        f"{finger}{part}": parent
        for finger in ("Index", "Middle", "Ring", "Little")
        for part, parent in (
            ("Proximal", "{side}Hand"),
            ("Intermediate", "{side}" + finger + "Proximal"),
            ("Distal", "{side}" + finger + "Intermediate"),
        )
    },
    "Eye": "head",
}
HUMANOID_PARENTS_1 = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "upperChest": "chest",
    "neck": "upperChest",
    "head": "neck",
    "jaw": "head",
    **{
        side + part: parent.format(side=side)
        for side in ("left", "right")
        for part, parent in _LIMBS_1.items()
    },
}
# VRM 0.x names the thumb bones Proximal, Intermediate and Distal.
_THUMB_0 = {
    "ThumbMetacarpal": "ThumbProximal",
    "ThumbProximal": "ThumbIntermediate",
}


def _name_0(name: str | None) -> str | None:
    for side in ("left", "right"):
        if name and name.startswith(side) and name[len(side) :] in _THUMB_0:
            return side + _THUMB_0[name[len(side) :]]
    return name


HUMANOID_PARENTS_0 = {_name_0(k): _name_0(v) for k, v in HUMANOID_PARENTS_1.items()}
VRM1_BONES = frozenset(HUMANOID_PARENTS_1)
VRM0_BONES = frozenset(HUMANOID_PARENTS_0)

# The expression presets of VRM 1.0 and of VRM 0.x ("unknown" marks a custom group in 0.x).
VRM1_PRESETS = frozenset(
    "happy angry sad relaxed surprised aa ih ou ee oh blink blinkLeft blinkRight lookUp lookDown"
    " lookLeft lookRight neutral".split()
)
VRM0_PRESETS = frozenset(
    "unknown neutral a i u e o blink joy angry sorrow fun lookup lookdown lookleft lookright"
    " blink_l blink_r".split()
)

# The keys of the VRM 0.x extension, as UniVRM spells them; every object is closed.
VRM0_KEYS = {
    "": {
        "exporterVersion",
        "specVersion",
        "meta",
        "humanoid",
        "firstPerson",
        "blendShapeMaster",
        "secondaryAnimation",
        "materialProperties",
    },
    "meta": {
        "title",
        "version",
        "author",
        "contactInformation",
        "reference",
        "texture",
        "allowedUserName",
        "violentUssageName",
        "sexualUssageName",
        "commercialUssageName",
        "otherPermissionUrl",
        "licenseName",
        "otherLicenseUrl",
    },
    "humanoid": {
        "humanBones",
        "armStretch",
        "legStretch",
        "upperArmTwist",
        "lowerArmTwist",
        "upperLegTwist",
        "lowerLegTwist",
        "feetSpacing",
        "hasTranslationDoF",
    },
    "humanBone": {
        "bone",
        "node",
        "useDefaultValues",
        "min",
        "max",
        "center",
        "axisLength",
    },
    "firstPerson": {
        "firstPersonBone",
        "firstPersonBoneOffset",
        "meshAnnotations",
        "lookAtTypeName",
        "lookAtHorizontalInner",
        "lookAtHorizontalOuter",
        "lookAtVerticalDown",
        "lookAtVerticalUp",
    },
    "meshAnnotation": {"mesh", "firstPersonFlag"},
    "degreeMap": {"curve", "xRange", "yRange"},
    "vector": {"x", "y", "z"},
    "blendShapeMaster": {"blendShapeGroups"},
    "group": {"name", "presetName", "binds", "materialValues", "isBinary"},
    "bind": {"mesh", "index", "weight"},
    "materialValue": {"materialName", "propertyName", "targetValue"},
    "secondaryAnimation": {"boneGroups", "colliderGroups"},
    "boneGroup": {
        "comment",
        "stiffiness",
        "gravityPower",
        "gravityDir",
        "dragForce",
        "center",
        "hitRadius",
        "bones",
        "colliderGroups",
    },
    "colliderGroup": {"node", "colliders"},
    "collider": {"offset", "radius"},
    "material": {
        "name",
        "shader",
        "renderQueue",
        "floatProperties",
        "vectorProperties",
        "textureProperties",
        "keywordMap",
        "tagMap",
    },
}
VRM0_REQUIRED = {
    "": VRM0_KEYS[""],
    "meta": {
        "title",
        "author",
        "allowedUserName",
        "violentUssageName",
        "sexualUssageName",
        "commercialUssageName",
        "licenseName",
    },
    "humanoid": {"humanBones"},
    "humanBone": {"bone", "node"},
    "firstPerson": {"firstPersonBone", "firstPersonBoneOffset", "lookAtTypeName"},
    "group": {"name", "presetName", "binds"},
    "bind": {"mesh", "index", "weight"},
    "material": {"name", "shader"},
}
# The spellings that a writer might use for UniVRM's keys.
VRM0_MISSPELLINGS = {
    "violentUsageName": "violentUssageName",
    "sexualUsageName": "sexualUssageName",
    "commercialUsageName": "commercialUssageName",
    "stiffness": "stiffiness",
    "preset": "presetName",
    "blendShapeGroup": "blendShapeGroups",
    "licenseUrl": "otherLicenseUrl",
}
VRM0_ENUMS = {
    "allowedUserName": {"OnlyAuthor", "ExplicitlyLicensedPerson", "Everyone"},
    "violentUssageName": {"Disallow", "Allow"},
    "sexualUssageName": {"Disallow", "Allow"},
    "commercialUssageName": {"Disallow", "Allow"},
    "licenseName": {
        "Redistribution_Prohibited",
        "CC0",
        "CC_BY",
        "CC_BY_NC",
        "CC_BY_SA",
        "CC_BY_NC_SA",
        "CC_BY_ND",
        "CC_BY_NC_ND",
        "Other",
    },
    "lookAtTypeName": {"Bone", "BlendShape"},
    "firstPersonFlag": {"Auto", "Both", "ThirdPersonOnly", "FirstPersonOnly"},
}
# The shaders of UniVRM 0.x (vrm.material.schema.json).
VRM0_SHADERS = {
    "VRM/MToon",
    "VRM/UnlitTransparentZWrite",
    "VRM_USE_GLTFSHADER",
    "Standard",
    "UniGLTF/UniUnlit",
    "VRM/UnlitTexture",
    "VRM/UnlitCutout",
    "VRM/UnlitTransparent",
}


# --- structural checks (no download) -----------------------------------------------------------


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _Problems(list):
    """The problems that a check finds, as ``"path: message"`` strings."""

    def add(self, path: str, message: str) -> None:
        self.append(f"{path}: {message}")

    def check_index(self, path: str, value, count: int, what: str) -> bool:
        """Check that ``value`` indexes a list of ``count`` items; True when it does."""
        if not _is_int(value) or not 0 <= value < count:
            self.add(path, f"{value!r} is not the index of one of the {count} {what}")
            return False
        return True

    def check_keys(self, path: str, obj, kind: str) -> bool:
        """Check the keys of a VRM 0.x object against UniVRM's spellings; True for a dict."""
        if not isinstance(obj, dict):
            self.add(path, f"expected an object, got {type(obj).__name__}")
            return False
        for key in sorted(set(obj) - VRM0_KEYS[kind]):
            hint = VRM0_MISSPELLINGS.get(key)
            self.add(
                path,
                f"unknown key {key!r}"
                + (f" (UniVRM spells it {hint!r})" if hint else ""),
            )
        for key in sorted(VRM0_REQUIRED.get(kind, set()) - set(obj)):
            self.add(path, f"missing key {key!r}")
        return True


def _target_counts(doc: dict) -> list[int]:
    """The number of morph targets of each mesh (the same in every primitive of a mesh)."""
    return [
        max((len(p.get("targets", [])) for p in m.get("primitives", [])), default=0)
        for m in doc.get("meshes", [])
    ]


def _parents(doc: dict) -> list[int]:
    parents = [-1] * len(doc.get("nodes", []))
    for i, node in enumerate(doc.get("nodes", [])):
        for child in node.get("children", []):
            if _is_int(child) and 0 <= child < len(parents):
                parents[child] = i
    return parents


def _check_primitive_targets(doc: dict, problems: _Problems) -> None:
    for m, mesh in enumerate(doc.get("meshes", [])):
        counts = {len(p.get("targets", [])) for p in mesh.get("primitives", [])}
        if len(counts) > 1:
            problems.add(
                f"meshes[{m}]",
                f"the primitives carry different numbers of targets {sorted(counts)}",
            )


def _check_humanoid(
    doc: dict, bones: dict[str, int], version: str, path: str, problems: _Problems
) -> None:
    """
    The bone names, the node indices, the required bones and the hierarchy: the node of each
    bone descends from the node of its nearest humanoid ancestor that the file maps.
    """
    parents_map = HUMANOID_PARENTS_1 if version == "1.0" else HUMANOID_PARENTS_0
    nodes = len(doc.get("nodes", []))
    for bone in sorted(set(bones) - set(parents_map)):
        problems.add(path, f"{bone!r} is not a VRM {version} humanoid bone")
    for bone in sorted(vrm_tables.required_bones(version) - set(bones)):
        problems.add(path, f"the required bone {bone!r} is missing")
    seen: dict[int, str] = {}
    for bone, node in bones.items():
        if not problems.check_index(f"{path}.{bone}", node, nodes, "nodes"):
            continue
        if node in seen:
            problems.add(f"{path}.{bone}", f"node {node} is also {seen[node]!r}")
        seen[node] = bone
    parents = _parents(doc)
    for bone, node in bones.items():
        if bone not in parents_map or not (_is_int(node) and 0 <= node < nodes):
            continue
        parent = parents_map[bone]
        while parent is not None and parent not in bones:
            parent = parents_map[parent]
        if parent is None:
            continue
        ancestor, target = parents[node], bones[parent]
        for _ in range(nodes):  # at most the depth of the tree, even in a cycle
            if ancestor < 0 or ancestor == target:
                break
            ancestor = parents[ancestor]
        if ancestor != target:
            problems.add(
                f"{path}.{bone}",
                f"node {node} does not descend from node {target} ({parent!r})",
            )


def _texture_refs(obj, path: str):
    """The ``index`` of every texture info object under ``obj``, with its path."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.endswith("Texture") and isinstance(value, dict) and "index" in value:
                yield f"{path}.{key}", value["index"]
            else:
                yield from _texture_refs(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _texture_refs(value, f"{path}[{i}]")


def check_vrm1(doc: dict) -> list[str]:
    """The problems of a VRM 1.0 glTF JSON that the schemas cannot see (an empty list: none)."""
    problems = _Problems()
    used = set(doc.get("extensionsUsed", []))
    found = set(doc.get("extensions", {}))
    for kind in ("materials", "nodes", "meshes", "textures", "images"):
        for item in doc.get(kind, []):
            found |= set(item.get("extensions", {}))
    for name in sorted(found - used):
        problems.add("extensionsUsed", f"{name!r} is used but not declared")
    if "VRM" in found:
        problems.add("extensions", "a VRM 1.0 file carries the VRM 0.x extension")
    vrm = doc.get("extensions", {}).get("VRMC_vrm")
    if not isinstance(vrm, dict):
        problems.add("extensions", "no VRMC_vrm extension")
        return problems
    if vrm.get("specVersion") != "1.0":
        problems.add("VRMC_vrm.specVersion", f"{vrm.get('specVersion')!r} is not '1.0'")
    nodes = doc.get("nodes", [])
    meshes = doc.get("meshes", [])
    targets = _target_counts(doc)
    _check_primitive_targets(doc, problems)

    human = vrm.get("humanoid", {}).get("humanBones", {})
    bones = {name: entry.get("node") for name, entry in human.items()}
    _check_humanoid(doc, bones, "1.0", "humanoid.humanBones", problems)

    meta = vrm.get("meta", {})
    if "thumbnailImage" in meta:
        problems.check_index(
            "meta.thumbnailImage",
            meta["thumbnailImage"],
            len(doc.get("images", [])),
            "images",
        )

    look_at = vrm.get("lookAt", {})
    if look_at.get("type", "bone") == "bone" and look_at:
        for eye in ("leftEye", "rightEye"):
            if eye not in bones:
                problems.add("lookAt.type", f"bone look-at without the {eye} bone")

    for i, annotation in enumerate(
        vrm.get("firstPerson", {}).get("meshAnnotations", [])
    ):
        node = annotation.get("node")
        if problems.check_index(
            f"firstPerson.meshAnnotations[{i}].node", node, len(nodes), "nodes"
        ):
            if "mesh" not in nodes[node]:
                problems.add(
                    f"firstPerson.meshAnnotations[{i}].node", f"node {node} has no mesh"
                )

    expressions = vrm.get("expressions", {})
    names: dict[str, str] = {}
    for group in ("preset", "custom"):
        for name, expression in expressions.get(group, {}).items():
            path = f"expressions.{group}.{name}"
            if group == "preset" and name not in VRM1_PRESETS:
                problems.add(path, "not a VRM 1.0 expression preset")
            if group == "custom" and name.lower() in {p.lower() for p in VRM1_PRESETS}:
                problems.add(path, "a custom expression named as a preset")
            if name.lower() in names:
                problems.add(path, f"the same name as {names[name.lower()]}")
            names[name.lower()] = path
            for b, bind in enumerate(expression.get("morphTargetBinds", [])):
                node = bind.get("node")
                where = f"{path}.morphTargetBinds[{b}]"
                if not problems.check_index(where + ".node", node, len(nodes), "nodes"):
                    continue
                mesh = nodes[node].get("mesh")
                if mesh is None:
                    problems.add(where + ".node", f"node {node} has no mesh")
                elif problems.check_index(where + ".node", mesh, len(meshes), "meshes"):
                    problems.check_index(
                        where + ".index",
                        bind.get("index"),
                        targets[mesh],
                        f"morph targets of mesh {mesh}",
                    )

    textures = len(doc.get("textures", []))
    for m, material in enumerate(doc.get("materials", [])):
        mtoon = material.get("extensions", {}).get("VRMC_materials_mtoon")
        if mtoon is None:
            continue
        for path, index in _texture_refs(mtoon, f"materials[{m}].VRMC_materials_mtoon"):
            problems.check_index(path, index, textures, "textures")
    for n, node in enumerate(nodes):
        constraint = node.get("extensions", {}).get("VRMC_node_constraint")
        if constraint is None:
            continue
        for kind, body in constraint.get("constraint", {}).items():
            path = f"nodes[{n}].VRMC_node_constraint.{kind}.source"
            if problems.check_index(path, body.get("source"), len(nodes), "nodes"):
                if body["source"] == n:
                    problems.add(path, "a node constrained by itself")
    return problems


def check_vrm0(doc: dict) -> list[str]:
    """The problems of a VRM 0.x glTF JSON that the permissive 0.x schemas cannot see."""
    problems = _Problems()
    used = set(doc.get("extensionsUsed", []))
    if "VRM" not in used:
        problems.add("extensionsUsed", "'VRM' is not declared")
    vrm1 = sorted(u for u in used if u.startswith("VRMC_"))
    for kind in ("materials", "nodes"):
        for item in doc.get(kind, []):
            vrm1 += [e for e in item.get("extensions", {}) if e.startswith("VRMC_")]
    if vrm1:
        problems.add(
            "extensions", f"VRM 1.0 extensions in a VRM 0.x file: {sorted(set(vrm1))}"
        )
    vrm = doc.get("extensions", {}).get("VRM")
    if not problems.check_keys("VRM", vrm, ""):
        return problems
    if vrm.get("specVersion") != "0.0":
        problems.add("VRM.specVersion", f"{vrm.get('specVersion')!r} is not '0.0'")
    nodes = doc.get("nodes", [])
    meshes = doc.get("meshes", [])
    materials = doc.get("materials", [])
    textures = len(doc.get("textures", []))
    targets = _target_counts(doc)
    _check_primitive_targets(doc, problems)

    def enum(path: str, key: str, value) -> None:
        if value not in VRM0_ENUMS[key]:
            problems.add(
                f"{path}.{key}", f"{value!r} is not one of {sorted(VRM0_ENUMS[key])}"
            )

    def vector(path: str, value) -> None:
        if problems.check_keys(path, value, "vector"):
            for axis in ("x", "y", "z"):
                if not _is_number(value.get(axis)):
                    problems.add(
                        f"{path}.{axis}", f"{value.get(axis)!r} is not a number"
                    )

    meta = vrm.get("meta")
    if problems.check_keys("meta", meta, "meta"):
        for key in VRM0_ENUMS:
            if key in meta:
                enum("meta", key, meta[key])
        if meta.get("licenseName") == "Other" and not meta.get("otherLicenseUrl"):
            problems.add("meta.otherLicenseUrl", "licenseName 'Other' needs a URL")
        if "texture" in meta:
            problems.check_index("meta.texture", meta["texture"], textures, "textures")
        for key in ("title", "author"):
            if not (isinstance(meta.get(key), str) and meta.get(key)):
                problems.add(f"meta.{key}", "an empty or missing string")

    humanoid = vrm.get("humanoid")
    bones: dict[str, int] = {}
    if problems.check_keys("humanoid", humanoid, "humanoid"):
        for i, entry in enumerate(humanoid.get("humanBones", [])):
            path = f"humanoid.humanBones[{i}]"
            if not problems.check_keys(path, entry, "humanBone"):
                continue
            bone = entry.get("bone")
            if bone in bones:
                problems.add(path, f"{bone!r} appears twice")
            bones[bone] = entry.get("node")
        _check_humanoid(doc, bones, "0.x", "humanoid.humanBones", problems)

    first = vrm.get("firstPerson")
    if problems.check_keys("firstPerson", first, "firstPerson"):
        if "head" in bones and first.get("firstPersonBone") != bones["head"]:
            problems.add(
                "firstPerson.firstPersonBone",
                f"{first.get('firstPersonBone')!r} is not the head node {bones['head']}",
            )
        if "firstPersonBoneOffset" in first:
            vector("firstPerson.firstPersonBoneOffset", first["firstPersonBoneOffset"])
        if "lookAtTypeName" in first:
            enum("firstPerson", "lookAtTypeName", first["lookAtTypeName"])
            if first["lookAtTypeName"] == "Bone":
                for eye in ("leftEye", "rightEye"):
                    if eye not in bones:
                        problems.add("firstPerson.lookAtTypeName", f"no {eye} bone")
        for key in (
            "lookAtHorizontalInner",
            "lookAtHorizontalOuter",
            "lookAtVerticalDown",
            "lookAtVerticalUp",
        ):
            curve = first.get(key)
            if curve is None or not problems.check_keys(
                f"firstPerson.{key}", curve, "degreeMap"
            ):
                continue
            points = curve.get("curve", [])
            if len(points) % 4 or not all(_is_number(p) for p in points):
                problems.add(
                    f"firstPerson.{key}.curve",
                    "not a list of keys of four numbers (time, value, tangents)",
                )
            for r in ("xRange", "yRange"):
                if not (_is_number(curve.get(r)) and curve.get(r) >= 0):
                    problems.add(
                        f"firstPerson.{key}.{r}", f"{curve.get(r)!r} is not >= 0"
                    )
        for i, annotation in enumerate(first.get("meshAnnotations", [])):
            path = f"firstPerson.meshAnnotations[{i}]"
            if problems.check_keys(path, annotation, "meshAnnotation"):
                problems.check_index(
                    path + ".mesh", annotation.get("mesh"), len(meshes), "meshes"
                )
                enum(path, "firstPersonFlag", annotation.get("firstPersonFlag"))

    master = vrm.get("blendShapeMaster")
    weights = []
    if problems.check_keys("blendShapeMaster", master, "blendShapeMaster"):
        names: dict[str, str] = {}
        presets: dict[str, str] = {}
        preset_names = {p.lower() for p in VRM0_PRESETS - {"unknown"}}
        for g, group in enumerate(master.get("blendShapeGroups", [])):
            path = f"blendShapeGroups[{g}]"
            if not problems.check_keys(path, group, "group"):
                continue
            name, preset = group.get("name"), group.get("presetName")
            if not (isinstance(name, str) and name):
                problems.add(path + ".name", "an empty or missing name")
                name = ""
            if preset not in VRM0_PRESETS:
                problems.add(
                    path + ".presetName", f"{preset!r} is not a VRM 0.x preset"
                )
            elif preset != "unknown":
                if preset in presets:
                    problems.add(
                        path + ".presetName", f"{preset!r} is also {presets[preset]}"
                    )
                presets[preset] = path
            elif name.lower() in preset_names:
                problems.add(
                    path + ".name", f"the custom group {name!r} is named as a preset"
                )
            if name.lower() in names:
                problems.add(path + ".name", f"{name!r} is also {names[name.lower()]}")
            names[name.lower()] = path
            if "isBinary" in group and not isinstance(group["isBinary"], bool):
                problems.add(path + ".isBinary", "not a boolean")
            for b, bind in enumerate(group.get("binds", [])):
                where = f"{path}.binds[{b}]"
                if not problems.check_keys(where, bind, "bind"):
                    continue
                if problems.check_index(
                    where + ".mesh", bind.get("mesh"), len(meshes), "meshes"
                ):
                    problems.check_index(
                        where + ".index",
                        bind.get("index"),
                        targets[bind["mesh"]],
                        f"morph targets of mesh {bind['mesh']}",
                    )
                weight = bind.get("weight")
                if not (_is_number(weight) and 0 <= weight <= 100):
                    problems.add(
                        where + ".weight", f"{weight!r} is not within 0 to 100"
                    )
                else:
                    weights.append(weight)
            for v, value in enumerate(group.get("materialValues", [])):
                where = f"{path}.materialValues[{v}]"
                if problems.check_keys(where, value, "materialValue"):
                    if value.get("materialName") not in {
                        m.get("name") for m in materials
                    }:
                        problems.add(
                            where, f"no material named {value.get('materialName')!r}"
                        )
        if weights and max(weights) <= 1:
            problems.add(
                "blendShapeGroups",
                "every bind weight is 1 or less: VRM 0.x weights run from 0 to 100",
            )

    secondary = vrm.get("secondaryAnimation")
    if problems.check_keys("secondaryAnimation", secondary, "secondaryAnimation"):
        groups = secondary.get("colliderGroups", [])
        for c, group in enumerate(groups):
            path = f"secondaryAnimation.colliderGroups[{c}]"
            if problems.check_keys(path, group, "colliderGroup"):
                problems.check_index(
                    path + ".node", group.get("node"), len(nodes), "nodes"
                )
                for k, collider in enumerate(group.get("colliders", [])):
                    where = f"{path}.colliders[{k}]"
                    if problems.check_keys(where, collider, "collider"):
                        vector(where + ".offset", collider.get("offset"))
        for s, spring in enumerate(secondary.get("boneGroups", [])):
            path = f"secondaryAnimation.boneGroups[{s}]"
            if not problems.check_keys(path, spring, "boneGroup"):
                continue
            for i, bone in enumerate(spring.get("bones", [])):
                problems.check_index(f"{path}.bones[{i}]", bone, len(nodes), "nodes")
            for i, group in enumerate(spring.get("colliderGroups", [])):
                problems.check_index(
                    f"{path}.colliderGroups[{i}]", group, len(groups), "groups"
                )
            if "center" in spring and spring["center"] != -1:
                problems.check_index(
                    path + ".center", spring["center"], len(nodes), "nodes"
                )
            if "gravityDir" in spring:
                vector(path + ".gravityDir", spring["gravityDir"])

    properties = vrm.get("materialProperties")
    if not isinstance(properties, list):
        problems.add("materialProperties", "not a list")
        return problems
    if len(properties) != len(materials):
        problems.add(
            "materialProperties",
            f"{len(properties)} entries for {len(materials)} glTF materials",
        )
    for i, entry in enumerate(properties):
        path = f"materialProperties[{i}]"
        if not problems.check_keys(path, entry, "material"):
            continue
        if i < len(materials) and entry.get("name") != materials[i].get("name"):
            problems.add(
                path + ".name",
                f"{entry.get('name')!r} is not the glTF material {materials[i].get('name')!r}",
            )
        if entry.get("shader") not in VRM0_SHADERS:
            problems.add(
                path + ".shader", f"{entry.get('shader')!r} is not a UniVRM shader"
            )
        if "renderQueue" in entry and not _is_int(entry["renderQueue"]):
            problems.add(path + ".renderQueue", "not an integer")
        for key, value in entry.get("floatProperties", {}).items():
            if not _is_number(value):
                problems.add(
                    f"{path}.floatProperties.{key}", f"{value!r} is not a number"
                )
        for key, value in entry.get("vectorProperties", {}).items():
            if not (
                isinstance(value, list)
                and len(value) == 4
                and all(map(_is_number, value))
            ):
                problems.add(
                    f"{path}.vectorProperties.{key}", "not a list of 4 numbers"
                )
        for key, value in entry.get("textureProperties", {}).items():
            problems.check_index(
                f"{path}.textureProperties.{key}", value, textures, "textures"
            )
        for key, value in entry.get("keywordMap", {}).items():
            if not isinstance(value, bool):
                problems.add(f"{path}.keywordMap.{key}", "not a boolean")
        for key, value in entry.get("tagMap", {}).items():
            if not isinstance(value, str):
                problems.add(f"{path}.tagMap.{key}", "not a string")
    for m, material in enumerate(materials):
        if "VRMC_materials_mtoon" in material.get("extensions", {}):
            problems.add(f"materials[{m}]", "a VRM 0.x material carries MToon 1.0")
    return problems


# --- schemas (downloaded) ----------------------------------------------------------------------


def schema_dir() -> pathlib.Path:
    return anny.paths.get_anny_cache_path() / "vrm_schema"


def _schema_sources() -> dict[str, tuple[str, pathlib.Path]]:
    """Schema file name -> (URL, cache path). The names are unique across the folders."""
    sources = {}
    for folder, files in VRM_SCHEMA_FILES.items():
        for name in files:
            path = f"specification/{folder}/schema/{name}"
            sources[name] = (
                VRM_SPEC_URL.format(commit=VRM_SPEC_COMMIT, path=path),
                schema_dir() / VRM_SPEC_COMMIT / path,
            )
    for name in GLTF_SCHEMA_FILES:
        path = f"specification/2.0/schema/{name}"
        sources[name] = (
            GLTF_URL.format(commit=GLTF_COMMIT, path=path),
            schema_dir() / GLTF_COMMIT / path,
        )
    return sources


def _fetch(url: str, path: pathlib.Path) -> None:
    import requests

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    json.loads(response.content)  # a schema, not an error page
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(response.content)
    part.replace(path)


def _refs(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from _refs(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _refs(value)


@functools.lru_cache(maxsize=1)
def load_schemas():
    """
    The pinned schemas: (a ``referencing`` registry of every file under its file name, the
    contents by file name). Missing files are downloaded into the cache; a failed download or a
    missing ``jsonschema`` raises :class:`unittest.SkipTest`.
    """
    try:
        import jsonschema  # noqa: F401
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT4
    except ImportError:
        raise unittest.SkipTest(
            "jsonschema 4.18 or later is not installed (the dev extra)"
        )
    sources = _schema_sources()
    missing = [(url, path) for url, path in sources.values() if not path.exists()]
    if missing:
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            try:
                list(pool.map(lambda job: _fetch(*job), missing))
            except Exception as error:  # offline, blocked or moved
                raise unittest.SkipTest(
                    f"the VRM schemas could not be downloaded: {error}"
                )
    contents = {
        name: json.loads(path.read_text()) for name, (_, path) in sources.items()
    }
    unknown = sorted(
        {
            (ref, name)
            for name, c in contents.items()
            for ref in _refs(c)
            if ref not in contents
        }
    )
    if unknown:
        raise AssertionError(
            f"schemas refer to files outside the pinned lists: {unknown}"
        )
    # The VRM schemas without $schema follow draft 4, as the glTF 2.0 schemas that they extend
    # did; the glTF base schemas declare draft 2020-12.
    registry = Registry().with_resources(
        (name, Resource.from_contents(c, default_specification=DRAFT4))
        for name, c in contents.items()
    )
    return registry, contents


def schema_errors(name: str, instance) -> list[str]:
    """The errors of ``instance`` against the schema file ``name``, as ``"path: message"``."""
    import jsonschema

    registry, contents = load_schemas()
    schema = contents[name]
    cls = jsonschema.validators.validator_for(
        schema, default=jsonschema.Draft4Validator
    )
    validator = cls(schema, registry=registry)
    errors = sorted(
        validator.iter_errors(instance), key=lambda e: list(e.absolute_path)
    )
    return [
        "/".join(map(str, e.absolute_path)) + f": {e.message} (schema {name})"
        for e in errors
    ]


# --- tests -------------------------------------------------------------------------------------

# Where the 0.x schemas list the keys of each kind of object in VRM0_KEYS.
_VRM0_KEY_SOURCES = {
    "": ("vrm.schema.json",),
    "meta": ("vrm.meta.schema.json",),
    "humanoid": ("vrm.humanoid.schema.json",),
    "humanBone": ("vrm.humanoid.bone.schema.json",),
    "firstPerson": ("vrm.firstperson.schema.json",),
    "meshAnnotation": ("vrm.firstperson.meshannotation.schema.json",),
    "degreeMap": ("vrm.firstperson.degreemap.schema.json",),
    "vector": ("vrm.firstperson.schema.json", "firstPersonBoneOffset"),
    "blendShapeMaster": ("vrm.blendshape.schema.json",),
    "group": ("vrm.blendshape.group.schema.json",),
    "bind": ("vrm.blendshape.bind.schema.json",),
    "materialValue": ("vrm.blendshape.materialbind.schema.json",),
    "secondaryAnimation": ("vrm.secondaryanimation.schema.json",),
    "boneGroup": ("vrm.secondaryanimation.spring.schema.json",),
    "colliderGroup": ("vrm.secondaryanimation.collidergroup.schema.json",),
    "collider": (
        "vrm.secondaryanimation.collidergroup.schema.json",
        "colliders",
        "items",
    ),
    "material": ("vrm.material.schema.json",),
}


def _mutated(doc: dict, change) -> dict:
    """A deep copy of ``doc`` that ``change`` has edited in place."""
    doc = copy.deepcopy(doc)
    change(doc)
    return doc


class TestVrm1Structure(unittest.TestCase):
    """VRM 1.0: the structure of the shared export (CI; no download)."""

    @classmethod
    def setUpClass(cls):
        cls.doc = vrm_fixtures.vrm_json("1.0")

    def test_export(self):
        self.assertEqual(check_vrm1(self.doc), [])
        self.assertTrue(
            any(
                "VRMC_materials_mtoon" in m.get("extensions", {})
                for m in self.doc["materials"]
            ),
            "the VRM 1.0 export has no MToon material",
        )

    def test_bone_tables(self):
        self.assertEqual(len(VRM1_BONES), 55)
        self.assertEqual(len(VRM0_BONES), 55)
        self.assertLessEqual(set(vrm_tables.humanoid_bones("1.0")), VRM1_BONES)
        self.assertLessEqual(set(vrm_tables.humanoid_bones("0.x")), VRM0_BONES)
        self.assertLessEqual(vrm_tables.required_bones("1.0"), VRM1_BONES)
        self.assertLessEqual(vrm_tables.required_bones("0.x"), VRM0_BONES)

    def test_checks_find_errors(self):
        """Each check finds the error that it is written for."""

        def vrm(d):
            return d["extensions"]["VRMC_vrm"]

        def bones(d):
            return vrm(d)["humanoid"]["humanBones"]

        def unknown_bone(d):
            bones(d)["leftThumbIntermediate"] = bones(d)["leftThumbProximal"]

        def crossed_hand(d):
            bones(d)["leftHand"]["node"] = bones(d)["rightLowerArm"]["node"]

        def target_out_of_range(d):
            for expression in vrm(d)["expressions"]["custom"].values():
                for bind in expression.get("morphTargetBinds", []):
                    bind["index"] = 10_000

        cases = {
            "is not a VRM 1.0 humanoid bone": unknown_bone,
            "the required bone 'leftHand' is missing": lambda d: bones(d).pop(
                "leftHand"
            ),
            "does not descend from": crossed_hand,
            "not a VRM 1.0 expression preset": lambda d: vrm(d)["expressions"][
                "preset"
            ].update(joy={}),
            "named as a preset": lambda d: vrm(d)["expressions"]["custom"].update(
                Blink={}
            ),
            "morph targets of mesh": target_out_of_range,
            "is used but not declared": lambda d: d["extensionsUsed"].remove(
                "VRMC_vrm"
            ),
        }
        for message, change in cases.items():
            with self.subTest(message):
                problems = check_vrm1(_mutated(self.doc, change))
                self.assertTrue(any(message in p for p in problems), problems)


class TestVrm0Structure(unittest.TestCase):
    """
    VRM 0.x: the structure of the shared export (CI; no download). The 0.x schemas require no
    key and leave every object open, so these checks carry the 0.x guard.
    """

    @classmethod
    def setUpClass(cls):
        cls.doc = vrm_fixtures.vrm_json("0.x")

    def groups(self) -> list[dict]:
        return self.doc["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]

    def test_export(self):
        self.assertEqual(check_vrm0(self.doc), [])

    def test_expression_names(self):
        """The presets once each, the perfect-sync groups in PascalCase and the VSeeFace visemes."""
        presets = [
            g["presetName"] for g in self.groups() if g["presetName"] != "unknown"
        ]
        self.assertEqual(len(presets), len(set(presets)))
        self.assertLessEqual(
            {"a", "i", "u", "e", "o", "blink", "neutral"}, set(presets)
        )
        custom = {g["name"] for g in self.groups() if g["presetName"] == "unknown"}
        missing = {
            "EyeBlinkLeft",
            "JawOpen",
            "MouthSmileLeft",
            "TongueOut",
            "Surprised",
        } - custom
        self.assertFalse(missing, "groups missing from the 0.x export")
        visemes = {"SIL", "CH", "DD", "FF", "KK", "NN", "PP", "RR", "SS", "TH"}
        self.assertLessEqual(visemes, custom, "the VSeeFace visemes")

    def test_checks_find_errors(self):
        """Each check finds the error that it is written for."""

        def vrm(d):
            return d["extensions"]["VRM"]

        def groups(d):
            return vrm(d)["blendShapeMaster"]["blendShapeGroups"]

        def misspelt(d):
            meta = vrm(d)["meta"]
            meta["violentUsageName"] = meta.pop("violentUssageName")

        def heavy_bind(d):
            next(b for g in groups(d) for b in g["binds"])["weight"] = 150.0

        def unit_weights(d):
            for group in groups(d):
                for bind in group["binds"]:
                    bind["weight"] /= 100.0

        def thumb_1(d):
            vrm(d)["humanoid"]["humanBones"][0]["bone"] = "leftThumbMetacarpal"

        def custom_joy(d):
            groups(d).append({"name": "Joy", "presetName": "unknown", "binds": []})

        cases = {
            "UniVRM spells it 'violentUssageName'": misspelt,
            "is not one of": lambda d: vrm(d)["meta"].update(licenseName="CC-BY"),
            "is not a VRM 0.x preset": lambda d: groups(d)[0].update(presetName="Joy"),
            "is not within 0 to 100": heavy_bind,
            "VRM 0.x weights run from 0 to 100": unit_weights,
            "entries for": lambda d: vrm(d)["materialProperties"].pop(),
            "is not a VRM 0.x humanoid bone": thumb_1,
            "named as a preset": custom_joy,
            "VRM 1.0 extensions in a VRM 0.x file": lambda d: d[
                "extensionsUsed"
            ].append("VRMC_vrm"),
        }
        for message, change in cases.items():
            with self.subTest(message):
                problems = check_vrm0(_mutated(self.doc, change))
                self.assertTrue(any(message in p for p in problems), problems)


class TestVrmSchemas(unittest.TestCase):
    """The extension objects against the VRM schemas, downloaded at pinned commits."""

    @classmethod
    def setUpClass(cls):
        cls.registry, cls.schemas = load_schemas()

    def assertValid(self, name: str, instance):
        errors = schema_errors(name, instance)
        self.assertEqual(errors[:20], [], f"{len(errors)} errors against {name}")

    def test_tables_match_the_schemas(self):
        """The hand-written tables of the structural checks repeat the schemas."""
        own = {"extensions", "extras"}
        human = self.schemas["VRMC_vrm.humanoid.humanBones.schema.json"]
        self.assertEqual(set(human["properties"]) - own, VRM1_BONES)
        self.assertEqual(set(human["required"]), vrm_tables.required_bones("1.0"))
        expressions = self.schemas["VRMC_vrm.expressions.schema.json"]["properties"]
        self.assertEqual(set(expressions["preset"]["properties"]) - own, VRM1_PRESETS)
        bone = self.schemas["vrm.humanoid.bone.schema.json"]["properties"]["bone"]
        self.assertEqual(set(bone["enum"]), VRM0_BONES)
        group = self.schemas["vrm.blendshape.group.schema.json"]["properties"]
        self.assertEqual(set(group["presetName"]["enum"]), VRM0_PRESETS)
        for kind, (name, *path) in _VRM0_KEY_SOURCES.items():
            with self.subTest(kind=kind):
                schema = self.schemas[name]
                for key in path:
                    schema = (
                        schema["properties"][key] if key != "items" else schema[key]
                    )
                self.assertEqual(set(schema["properties"]), VRM0_KEYS[kind])
        meta = self.schemas["vrm.meta.schema.json"]["properties"]
        for key in ("allowedUserName", "violentUssageName", "licenseName"):
            self.assertEqual(set(meta[key]["enum"]), VRM0_ENUMS[key])
        first = self.schemas["vrm.firstperson.schema.json"]["properties"]
        self.assertEqual(
            set(first["lookAtTypeName"]["enum"]), VRM0_ENUMS["lookAtTypeName"]
        )

    def test_schemas_reject_errors(self):
        """The validation is not vacuous: the references resolve and the enumerations bind."""
        vrm = copy.deepcopy(vrm_fixtures.vrm_json("1.0")["extensions"]["VRMC_vrm"])
        vrm["meta"]["avatarPermission"] = "anyone"
        vrm["humanoid"]["humanBones"]["hips"]["node"] = -1
        errors = schema_errors("VRMC_vrm.schema.json", vrm)
        self.assertTrue(any("'anyone'" in e for e in errors), errors)
        self.assertTrue(any("-1" in e for e in errors), errors)
        roll = {
            "specVersion": "1.0",
            "constraint": {"roll": {"source": 0, "rollAxis": "W"}},
        }
        self.assertTrue(schema_errors("VRMC_node_constraint.schema.json", roll))
        both = {
            "roll": {"source": 0, "rollAxis": "X"},
            "aim": {"source": 0, "aimAxis": "PositiveX"},
        }
        self.assertTrue(
            schema_errors(
                "VRMC_node_constraint.schema.json",
                {"specVersion": "1.0", "constraint": both},
            )
        )
        bind = {"blendShapeGroups": [{"presetName": "Joy", "binds": [{"weight": 101}]}]}
        self.assertEqual(len(schema_errors("vrm.blendshape.schema.json", bind)), 2)

    def test_vrm1(self):
        doc = vrm_fixtures.vrm_json("1.0")
        self.assertValid("VRMC_vrm.schema.json", doc["extensions"]["VRMC_vrm"])
        if "VRMC_springBone" in doc["extensions"]:
            self.assertValid(
                "VRMC_springBone.schema.json", doc["extensions"]["VRMC_springBone"]
            )
        for m, material in enumerate(doc["materials"]):
            mtoon = material.get("extensions", {}).get("VRMC_materials_mtoon")
            if mtoon is not None:
                with self.subTest(material=material.get("name", m)):
                    self.assertValid("VRMC_materials_mtoon.schema.json", mtoon)
        for node in doc["nodes"]:
            constraint = node.get("extensions", {}).get("VRMC_node_constraint")
            if constraint is not None:
                with self.subTest(node=node["name"]):
                    self.assertValid("VRMC_node_constraint.schema.json", constraint)

    def test_vrm0(self):
        self.assertValid(
            "vrm.schema.json", vrm_fixtures.vrm_json("0.x")["extensions"]["VRM"]
        )


if __name__ == "__main__":
    unittest.main()
