# OpenSculptBoy
# Apache License, Version 2.0
"""
VRM export of an Anny character, for VTuber apps: VRM 1.0 (``VRMC_vrm``) by default and VRM 0.x
on request.

The file holds the character in the VRM T-pose (:mod:`opensculptboy.export.tpose`) with
normalised joints (every node at the identity rotation), 4 skin weights per vertex, the
humanoid bone map, the expressions as mixes of the ARKit facial actions, bone look-at on the
eye bones, MToon materials, the licence metadata and the character card in the scene extras.

The export runs in two steps. :func:`vrm_spec` builds a :class:`VrmSpec`: the nodes, skins,
meshes with their morph targets, materials, humanoid map, expressions with their binds,
look-at, first-person settings and metadata of the file, in the frame of its VRM version.
:func:`write_vrm` serialises the spec as VRM 1.0 or VRM 0.x, checks it against its budget
(:mod:`opensculptboy.export.budget`) and writes the file. :func:`export_vrm` runs both.

The pipeline of the body:

1. Anny's rest mesh of the character in two kinds of rows: the base row, with the phenotype,
   the local changes and every face shape baked in and no facial action, and one row for each
   facial action at 1. Face shapes are never morph targets of a VRM file: they move the bone
   heads, and a VRM skeleton is fixed.
2. The skin weights of :func:`~opensculptboy.export.vrm_tables.vrm_skin_weights`, truncated to
   the strongest 4 bones of each vertex.
3. The T-pose rebind (:func:`~opensculptboy.export.tpose.rebind`): the bind mesh, its targets
   and the T-pose bone matrices T_j. The ``eyeLook*`` targets become lid-only, so that the eye
   bones alone turn the eyeballs.
4. The file frame G (VRM 1.0: (x, y, z) -> (x, z, -y), the figure facing +Z; VRM 0.x:
   (x, y, z) -> (-x, z, y), the figure facing -Z) and the centring offset o, which puts the
   hips (the head of ``root``) at X = Z = 0 and the lowest vertex at Y = 0. A vertex v' of the
   bind mesh is G v' + o in the file, and the joint of bone j is P_j = G t_j + o, with t_j the
   head of T_j.
5. Normalised joints: every node keeps the identity rotation, its translation is
   P_j - P_parent in the hierarchy of the file (:func:`~opensculptboy.export.vrm_tables.file_parents`),
   and each inverse bind matrix is translate(-P_j). Anny's world bone pose M_j then maps to
   the joint matrix W_j = translate(o) G M_j T_j^-1 G^-1 translate(P_j - o).
"""

from __future__ import annotations

import dataclasses
import io
import pathlib
import warnings
from typing import Any, Sequence

import numpy as np
import torch

from opensculptboy.character import Character
from opensculptboy.export import budget as budgets
from opensculptboy.export import mtoon, tpose, vrm_tables
from opensculptboy.export.body import (
    ANNY_TO_GLTF,
    BodyMesh,
    body_primitive,
    build_body,
    top_skin_weights,
    triangulated_faces,
)
from opensculptboy.export.document import (
    ELEMENT_ARRAY_BUFFER,
    MORPH_EPSILON,
    GltfDocument,
)
from opensculptboy.export.gltf import _check_names

VRM1_LICENSE_URL = "https://vrm.dev/licenses/1.0/"
PROJECT_URL = "https://github.com/infinia-yzl/anny"
# The licences of every part of the project (the 0.x ``otherPermissionUrl``).
CREDITS_URL = PROJECT_URL + "/blob/main/LICENSE_THINGS"
# The credits of the base model, in every VRM file (VRM 1.0 ``thirdPartyLicenses``, VRM 0.x
# ``reference``); the credits of a character's assets follow them.
BASE_CREDITS = (
    f"Made with OpenSculptBoy ({PROJECT_URL}), Apache License 2.0.",
    "Anny body model: Copyright 2025 NAVER Corp., Apache License 2.0 "
    "(https://github.com/naver/anny).",
    "MakeHuman and MPFB2 assets (base mesh, targets and rig) and the Face Units by Mika "
    "Suominen: CC0 1.0 (https://static.makehumancommunity.org/).",
    "ICT-FaceKit, for the detail face shapes: MIT License "
    "(https://github.com/USC-ICT/ICT-FaceKit).",
)

VERSIONS = {"1.0": "1.0", "1": "1.0", "0.x": "0.x", "0": "0.x", "0.0": "0.x"}
# The output frame G of each version: Anny's frame (metres, Z up, the figure facing -Y) to
# the file's (Y up). VRM 1.0 faces +Z, as glTF does; VRM 0.x faces -Z.
FRAMES = {
    "1.0": ANNY_TO_GLTF.copy(),
    "0.x": np.diag([-1.0, 1.0, -1.0]) @ ANNY_TO_GLTF,
}
# The bind that the export uses by default. On the arms-down poses (``relaxed``, ``walk``) of the
# default body, measured with the file's weights by :func:`opensculptboy.export.tpose.bind_error`,
# the inverse bind lands a third closer to Anny's own posed mesh than the forward bind: p99 8.5 mm
# against 12.5 mm, a maximum of 11.6 to 14.3 mm against 21 mm, and normals within 14 to 18
# degrees at p99 against 30 degrees.
DEFAULT_BIND = "inverse"
# The linear base colour of the skin (the colour of the GLB export), until the styles of PR 2.
SKIN_COLOUR = (0.80, 0.62, 0.52, 1.0)

VRM1_PRESETS = (
    "happy",
    "angry",
    "sad",
    "relaxed",
    "surprised",
    "aa",
    "ih",
    "ou",
    "ee",
    "oh",
    "blink",
    "blinkLeft",
    "blinkRight",
    "lookUp",
    "lookDown",
    "lookLeft",
    "lookRight",
    "neutral",
)
# The VRM 0.x presets and the group name that UniVRM gives each of them.
VRM0_PRESETS = {
    "neutral": "Neutral",
    "a": "A",
    "i": "I",
    "u": "U",
    "e": "E",
    "o": "O",
    "blink": "Blink",
    "joy": "Joy",
    "angry": "Angry",
    "sorrow": "Sorrow",
    "fun": "Fun",
    "lookup": "LookUp",
    "lookdown": "LookDown",
    "lookleft": "LookLeft",
    "lookright": "LookRight",
    "blink_l": "Blink_L",
    "blink_r": "Blink_R",
}
# UniVRM's defaults for the humanoid of a VRM 0.x file.
VRM0_HUMANOID_DEFAULTS = {
    "armStretch": 0.05,
    "legStretch": 0.05,
    "upperArmTwist": 0.5,
    "lowerArmTwist": 0.5,
    "upperLegTwist": 0.5,
    "lowerLegTwist": 0.5,
    "feetSpacing": 0.0,
    "hasTranslationDoF": False,
}
FIRST_PERSON_TYPES = {
    "auto": "Auto",
    "both": "Both",
    "thirdPersonOnly": "ThirdPersonOnly",
    "firstPersonOnly": "FirstPersonOnly",
}
_META_CHOICES = {
    "avatar_permission": ("onlyAuthor", "onlySeparatelyLicensedPerson", "everyone"),
    "commercial_usage": ("personalNonProfit", "personalProfit", "corporation"),
    "credit_notation": ("required", "unnecessary"),
    "modification": (
        "prohibited",
        "allowModification",
        "allowModificationRedistribution",
    ),
}
# The text fields of the metadata (``name`` is required, the others may be None), its lists of
# texts and its permissions, which must be real booleans: the text "false" would grant one.
_META_TEXTS = (
    "name",
    "version",
    "copyright_information",
    "contact_information",
    "third_party_licenses",
    "other_license_url",
)
_META_LISTS = ("authors", "references")
_META_FLAGS = (
    "allow_excessively_violent_usage",
    "allow_excessively_sexual_usage",
    "allow_political_or_religious_usage",
    "allow_antisocial_or_hate_usage",
    "allow_redistribution",
)
_ALLOWED_USER_0 = {
    "onlyAuthor": "OnlyAuthor",
    "onlySeparatelyLicensedPerson": "ExplicitlyLicensedPerson",
    "everyone": "Everyone",
}
# The range maps of the look-at: VRM 1.0 field, VRM 0.x field, look_at_ranges key.
LOOK_AT_MAPS = (
    ("rangeMapHorizontalInner", "lookAtHorizontalInner", "lookIn"),
    ("rangeMapHorizontalOuter", "lookAtHorizontalOuter", "lookOut"),
    ("rangeMapVerticalDown", "lookAtVerticalDown", "lookDown"),
    ("rangeMapVerticalUp", "lookAtVerticalUp", "lookUp"),
)
_PNG, _JPEG = b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"


@dataclasses.dataclass
class VrmMeta:
    """The licence metadata of a VRM file, in VRM 1.0 terms (the 0.x writer maps it)."""

    name: str
    authors: list[str]
    version: str | None = None
    copyright_information: str | None = None
    contact_information: str | None = None
    references: list[str] = dataclasses.field(default_factory=list)
    # Credits of further parts (assets, textures); the credits of the base model come first.
    third_party_licenses: str | None = None
    thumbnail_png: bytes | None = None
    avatar_permission: str = "onlyAuthor"
    allow_excessively_violent_usage: bool = False
    allow_excessively_sexual_usage: bool = False
    commercial_usage: str = "personalNonProfit"
    allow_political_or_religious_usage: bool = False
    allow_antisocial_or_hate_usage: bool = False
    credit_notation: str = "required"
    allow_redistribution: bool = False
    modification: str = "prohibited"
    other_license_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VrmMeta":
        """
        Metadata from a dict of the field names of this class (a ``--meta`` file); ``name``
        and ``authors`` may be missing, for the command line to fill them. The thumbnail
        comes from ``--thumbnail``, not from the dict. Raises ValueError for unknown fields
        and for values of the wrong type or outside their choices (:meth:`check_fields`).
        """
        if not isinstance(data, dict):
            raise ValueError("VRM metadata must be a JSON object.")
        fields = {f.name for f in dataclasses.fields(cls)} - {"thumbnail_png"}
        unknown = sorted(set(data) - fields)
        if unknown:
            raise ValueError(
                f"Unknown VRM metadata fields {unknown}; the fields are {sorted(fields)}."
            )
        data = dict(data)
        if isinstance(data.get("authors"), str):
            data["authors"] = [data["authors"]]
        if isinstance(data.get("references"), str):
            data["references"] = [data["references"]]
        meta = cls(**{"name": "", "authors": [], **data})
        meta.check_fields()
        return meta

    def check_fields(self) -> None:
        """
        Raise ValueError, naming the field, for a value of the wrong type or outside its
        choices: the texts must be strings (None for the optional ones), ``authors`` and
        ``references`` lists of non-empty strings, and the ``allow_*`` permissions booleans,
        since a string such as "false" would grant the permission.
        """
        for field in _META_TEXTS:
            value = getattr(self, field)
            if not isinstance(value, str) and (field == "name" or value is not None):
                raise ValueError(
                    f"VRM metadata {field} must be a string (got {value!r})."
                )
        for field in _META_LISTS:
            value = getattr(self, field)
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(v, str) and v.strip() for v in value
            ):
                raise ValueError(
                    f"VRM metadata {field} must be a list of non-empty strings (got "
                    f"{value!r})."
                )
        for field in _META_FLAGS:
            value = getattr(self, field)
            if not isinstance(value, bool):
                raise ValueError(
                    f"VRM metadata {field} must be a boolean, true or false (got "
                    f"{value!r})."
                )
        for field, choices in _META_CHOICES.items():
            value = getattr(self, field)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(
                    f"VRM metadata {field}={value!r}; use one of {list(choices)}."
                )

    def check(self) -> None:
        """Raise ValueError for metadata that a VRM file cannot carry."""
        self.check_fields()
        if not self.name.strip():
            raise ValueError("A VRM file needs a name.")
        if not self.authors:
            raise ValueError(
                "A VRM file needs an author: pass author= (--author) or meta= with authors."
            )


@dataclasses.dataclass
class RollConstraint:
    """A ``VRMC_node_constraint`` roll constraint: the node turns with ``source`` about ``axis``."""

    source: int  # node index
    axis: str  # "X", "Y" or "Z", in the node's rest frame (the file's axes)
    weight: float


@dataclasses.dataclass
class SpecNode:
    """A node of a VRM file; ``mesh`` and ``skin`` index the meshes and skins of the spec."""

    name: str
    translation: np.ndarray  # (3,) relative to the parent, in the file frame
    children: list[int] = dataclasses.field(default_factory=list)
    mesh: int | None = None
    skin: int | None = None
    constraint: RollConstraint | None = None


@dataclasses.dataclass
class SpecSkin:
    """A skin: joint nodes and their inverse bind matrices (J, 4, 4), in the file frame."""

    name: str
    joints: list[int]
    inverse_bind: np.ndarray
    skeleton: int


@dataclasses.dataclass
class SpecPrimitive:
    """A primitive of a mesh: a material and the triangles it draws (None: every triangle)."""

    material: int  # index into VrmSpec.materials
    triangles: np.ndarray | None = None


@dataclasses.dataclass
class SpecMesh:
    """
    A mesh of a VRM file: vertex data in the file frame (plus ``translation``), its primitives
    and the names of its morph targets. Every primitive carries every target, so that an
    expression binds a target by one index per mesh. ``first_person`` is the VRM 1.0 type of
    its first-person annotation.
    """

    name: str
    body: BodyMesh
    translation: np.ndarray  # (3,) added to the positions: the centring offset
    primitives: list[SpecPrimitive]
    target_names: list[str]
    first_person: str = "auto"

    def moving_targets(self) -> list[bool]:
        """
        Whether each morph target moves a vertex in the file: by more than
        ``MORPH_EPSILON`` in float32, as the document writes the offsets.
        """
        return [
            bool(
                np.linalg.norm(np.asarray(t, dtype=np.float32), axis=1).max(initial=0.0)
                > MORPH_EPSILON
            )
            for t in self.body.targets
        ]


@dataclasses.dataclass
class SpecExpression:
    """An expression and its binds: (mesh index, target index, weight 0 to 1)."""

    expression: vrm_tables.Expression
    binds: list[tuple[int, int, float]]


@dataclasses.dataclass
class LookAt:
    """Bone look-at: the eyes' midpoint from the head joint (file frame) and the ranges."""

    offset_from_head: np.ndarray  # (3,)
    ranges: dict[str, float]  # degrees: "lookUp", "lookDown", "lookIn", "lookOut"


@dataclasses.dataclass
class VrmSpec:
    """
    The content of a VRM file, before its serialisation as VRM 1.0 or VRM 0.x.

    The geometry is in the frame of ``version`` (``frame``, see :data:`FRAMES`), the humanoid
    map uses the bone names of that version, and the expressions are those that version
    carries. Nodes, meshes and skins keep their order in the file, and ``materials`` the glTF
    material order. The other fields record the body's rebind for reports and tests.
    """

    version: str  # "1.0" or "0.x"
    frame: np.ndarray  # (3, 3) G
    nodes: list[SpecNode]
    scene_nodes: list[int]
    skins: list[SpecSkin]
    meshes: list[SpecMesh]
    materials: list[mtoon.ToonMaterial]
    humanoid: dict[str, int]  # VRM humanoid bone -> node index
    expressions: list[SpecExpression]
    look_at: LookAt
    head: int  # the node of the head (the first-person bone)
    meta: VrmMeta
    credits: list[str]
    thumbnail: bytes | None  # a PNG or JPEG file
    extras: dict[str, Any]  # the ``opensculptboy`` entry of the scene extras
    sparse_targets: bool  # sparse morph target accessors (VRM 1.0) or dense ones (0.x)
    anny_vertex: bool  # write the _ANNY_VERTEX attribute
    # The body: its rebind (Anny's frame), the centring offset o and the joints P_j (file
    # frame), the file's 4 skin weights (float64, renormalised) and the file's hierarchy.
    rebind: tpose.Rebind
    offset: np.ndarray
    joint_positions: np.ndarray
    skin_weights: np.ndarray
    skin_indices: np.ndarray
    parents: list[int]


def vrm_version(version: str) -> str:
    """The VRM version ``"1.0"`` or ``"0.x"`` of a version name ("1", "1.0", "0", "0.x")."""
    if version not in VERSIONS:
        raise ValueError(f"Unknown VRM version {version!r}; use '1.0' or '0.x'.")
    return VERSIONS[version]


def check_topology(topology: str) -> None:
    """
    Raise ValueError unless a VRM file can carry a character of this topology: the MakeHuman
    body mesh with its eyes (``anny``, ``makehuman`` and their modifiers such as ``-quads``,
    ``-full`` or ``-notongue``). The eyes turn with look-at, and the expressions and the
    look-at ranges rest on the eyeball vertices of that mesh. The topology is only parsed,
    so that no model is built and no data is downloaded.
    """
    from anny.models.model_data import TopologyConfig

    if not isinstance(topology, str):
        raise ValueError(f"The topology must be a string (got {topology!r}).")
    try:
        config = TopologyConfig.from_string(topology)
    except ValueError as error:
        raise ValueError(f"Unknown topology {topology!r} ({error})") from None
    if config.base_mesh in ("smpl", "smplx"):
        raise ValueError(
            f"VRM export does not support the {topology!r} topology: it rests on the "
            "non-commercial SMPL data. Use the 'anny' topology."
        )
    if config.base_mesh != "makehuman" or config.submodel != "body" or not config.eyes:
        raise ValueError(
            f"The {topology!r} topology cannot be exported as VRM: VRM export needs the "
            "MakeHuman body mesh with its eyes (the 'anny' or 'makehuman' topology)."
        )


def resolve_meta(
    meta: VrmMeta | None,
    author: str | None,
    name: str | None,
    character: Character,
) -> VrmMeta:
    """
    The metadata of a file: ``meta`` (or restrictive defaults), with ``name`` and ``author``
    in place of its own when they are given; the name defaults to the character's name.
    """
    if meta is None:
        meta = VrmMeta(name="", authors=[])
    meta.check_fields()
    meta = dataclasses.replace(
        meta,
        name=name or meta.name or character.name,
        authors=[author] if author else list(meta.authors),
        references=list(meta.references),
    )
    meta.check()
    return meta


def vrm_spec(
    character: Character | None = None,
    model=None,
    version: str = "1.0",
    meta: VrmMeta | None = None,
    author: str | None = None,
    name: str | None = None,
    thumbnail: str | pathlib.Path | bytes | None = None,
    bare: bool = False,
    twist: str | None = None,
    bind: str = DEFAULT_BIND,
    keep_leg_spread: bool = False,
    keep_anny_vertex: bool = False,
) -> VrmSpec:
    """
    The content of a VRM file of a character (see :func:`export_vrm` for the arguments).
    """
    # Every check runs before the model is built.
    version = vrm_version(version)
    character = character or Character()
    check_topology(character.topology)
    if character.rig != "anny":
        raise ValueError(
            f"VRM export supports the 'anny' rig only; the character has the "
            f"{character.rig!r} rig."
        )
    twist = twist or ("constraint" if version == "1.0" else "merge")
    if twist not in ("constraint", "merge"):
        raise ValueError(f"Unknown twist mode {twist!r}; use 'constraint' or 'merge'.")
    if twist == "constraint" and version == "0.x":
        raise ValueError(
            "VRM 0.x files cannot carry roll constraints: use twist 'merge' (the VRM 0.x "
            "default) or VRM 1.0 for twist 'constraint'."
        )
    if bind not in ("forward", "inverse"):
        raise ValueError(f"Unknown bind {bind!r}; use 'forward' or 'inverse'.")
    meta = resolve_meta(meta, author, name, character)
    if thumbnail is None:
        png, thumbnail_kind = (
            meta.thumbnail_png,
            "given" if meta.thumbnail_png is not None else None,
        )
    elif isinstance(thumbnail, str) and thumbnail == "auto":
        png, thumbnail_kind = None, "auto"  # rendered below, from the rest mesh
    elif isinstance(thumbnail, (bytes, bytearray)):
        png, thumbnail_kind = bytes(thumbnail), "given"
    else:
        png, thumbnail_kind = pathlib.Path(thumbnail).read_bytes(), "given"
    if png is not None:
        _thumbnail_mime_type(png)  # a square PNG or JPEG file
    if model is None:
        model = character.build_model(face_shapes=bool(character.face_shapes))
    labels = list(model.bone_labels)
    actions = list(model.facial_action_labels)
    face_labels = list(model.face_shape_labels)
    _check_names(character.phenotype, list(model.phenotype_labels), "phenotype")
    _check_names(
        character.local_changes, list(model.local_change_labels), "local change"
    )
    _check_names(character.face_shapes, face_labels, "face shape")
    G = FRAMES[version]

    # Rows: the base mesh, then each facial action at 1; every face shape is baked in.
    output = _rest_output(model, character, actions, face_labels)
    rest = output["rest_vertices"].detach().double().cpu().numpy()
    base, offsets = rest[0], rest[1:] - rest[0]
    B = output["rest_bone_poses"][0].detach().double().cpu().numpy()

    weights, indices = vrm_tables.vrm_skin_weights(model, twist)
    w4, i4, _ = top_skin_weights(weights, indices, 4)
    w4 = w4.astype(np.float64)
    w4 /= w4.sum(axis=1, keepdims=True)
    rb = tpose.rebind(
        model,
        base,
        offsets,
        B,
        w4,
        i4,
        method=bind,
        keep_leg_spread=keep_leg_spread,
    )
    eyes = vrm_tables.eyeball_vertices(model)
    targets = [
        vrm_tables.lid_only(t, eyes) if a.startswith("eyeLook") else t
        for a, t in zip(actions, rb.targets)
    ]
    # The normals of the rebind: for the inverse bind, Anny's rest normals mapped as the targets.
    body = build_body(
        model, rb.vertices, targets, G, 4, weights, indices, normals=rb.normals
    )

    # Centring: the hips at X = Z = 0, the lowest vertex at Y = 0.
    joints = np.asarray(rb.joint_positions, dtype=np.float64) @ G.T
    root = labels.index("root")
    offset = np.array(
        [-joints[root, 0], -float(body.positions[:, 1].min()), -joints[root, 2]]
    )
    joints = joints + offset

    # Nodes: the joints in the file's hierarchy, with normalised (identity) rotations.
    parents = [int(p) for p in vrm_tables.file_parents(model, version, twist)]
    nodes = []
    for j, label in enumerate(labels):
        p = parents[j]
        nodes.append(
            SpecNode(
                name=label,
                translation=joints[j] - (joints[p] if p >= 0 else 0.0),
                children=[c for c, q in enumerate(parents) if q == j],
            )
        )
    for bone, source, axis, weight in vrm_tables.twist_constraints(version, twist):
        nodes[labels.index(bone)].constraint = RollConstraint(
            labels.index(source), axis, float(weight)
        )
    inverse_bind = np.tile(np.eye(4), (len(labels), 1, 1))
    inverse_bind[:, :3, 3] = -joints
    skins = [SpecSkin(character.rig, list(range(len(labels))), inverse_bind, root)]
    materials = [mtoon.soft_material("skin", SKIN_COLOUR)]
    meshes = [
        SpecMesh(
            name="body",
            body=body,
            translation=offset,
            primitives=[SpecPrimitive(material=0)],
            target_names=actions,
        )
    ]
    nodes.append(SpecNode("body", np.zeros(3), mesh=0, skin=0))
    scene_nodes = [j for j, p in enumerate(parents) if p < 0] + [len(nodes) - 1]

    humanoid = {
        bone: labels.index(label)
        for bone, label in vrm_tables.humanoid_bones(version).items()
    }
    missing = sorted(vrm_tables.required_bones(version) - set(humanoid))
    if missing:
        raise ValueError(f"The humanoid map lacks the required bones {missing}.")
    expressions = expression_binds(vrm_tables.expressions(version, actions), meshes)
    _check_expressions(version, expressions)
    head = labels.index("head")
    eyes_mid = (joints[labels.index("eye.L")] + joints[labels.index("eye.R")]) / 2
    look_at = LookAt(
        offset_from_head=eyes_mid - joints[head],
        ranges={
            k: float(v) for k, v in vrm_tables.look_at_ranges(model, output).items()
        },
    )

    if thumbnail_kind == "auto":
        # The rest pose, arms down, gives a portrait its natural shoulders; the head and
        # the face are the same in the T-pose.
        png = portrait_png(model, base, B[:, :3, 3])

    extras = {
        "generator": "opensculptboy",
        "anny_version": _anny_version(),
        "character": character.to_dict(),
        "format": "vrm",
        "vrm_version": version,
        "options": {
            "bare": bool(bare),
            "twist": twist,
            "bind": bind,
            "keep_leg_spread": bool(keep_leg_spread),
            "thumbnail": thumbnail_kind,
            "anny_vertex": bool(keep_anny_vertex),
        },
        "morph_targets": list(actions),
        "max_influences": 4,
        "frame": "metres, Y up, the figure faces "
        + ("+Z" if version == "1.0" else "-Z"),
    }
    return VrmSpec(
        version=version,
        frame=G.copy(),
        nodes=nodes,
        scene_nodes=scene_nodes,
        skins=skins,
        meshes=meshes,
        materials=materials,
        humanoid=humanoid,
        expressions=expressions,
        look_at=look_at,
        head=head,
        meta=meta,
        credits=list(BASE_CREDITS),
        thumbnail=png,
        extras=extras,
        sparse_targets=version == "1.0",
        anny_vertex=bool(keep_anny_vertex),
        rebind=rb,
        offset=offset,
        joint_positions=joints,
        skin_weights=w4,
        skin_indices=i4,
        parents=parents,
    )


def _rest_output(model, character: Character, actions, face_labels) -> dict:
    """
    Anny's output for the rows of a VRM file: row 0 has no facial action, row 1 + a has
    facial action a at 1; the phenotype, the local changes and the face shapes of the
    character are baked into every row.
    """
    dtype = model.template_vertices.dtype
    rows = torch.zeros(len(actions) + 1, len(actions), dtype=dtype)
    rows[1:] = torch.eye(len(actions), dtype=dtype)
    call = dict(
        phenotype_kwargs=dict(character.phenotype) or None,
        local_changes_kwargs=dict(character.local_changes) or None,
    )
    if actions:
        call["facial_actions"] = rows
    if face_labels:
        call["face_shape_kwargs"] = {
            k: torch.full((len(rows),), float(v), dtype=dtype)
            for k, v in character.face_shapes.items()
        }
    with torch.no_grad():
        return model(**call)


def expression_binds(
    expressions: Sequence[vrm_tables.Expression], meshes: Sequence[SpecMesh]
) -> list[SpecExpression]:
    """
    The binds of each expression: for each facial action of its mix, one bind for each mesh
    with a target of that action that moves (the body today; the split body, the lashes, the
    teeth and the outfits of later assets alike).
    """
    moving = [m.moving_targets() for m in meshes]
    result = []
    for e in expressions:
        binds = []
        for action, weight in e.mix.items():
            if weight == 0:
                continue
            for m, mesh in enumerate(meshes):
                if action in mesh.target_names:
                    t = mesh.target_names.index(action)
                    if moving[m][t]:
                        binds.append((m, t, float(weight)))
        result.append(SpecExpression(e, binds))
    return result


def _check_expressions(version: str, expressions: Sequence[SpecExpression]) -> None:
    """Raise ValueError for expression names that the version cannot carry."""
    if version == "1.0":
        names = [s.expression.name for s in expressions]
        for s in expressions:
            e = s.expression
            if e.preset and e.name not in VRM1_PRESETS:
                raise ValueError(f"{e.name!r} is not a VRM 1.0 preset expression.")
            if not e.preset and e.name.lower() in {p.lower() for p in VRM1_PRESETS}:
                raise ValueError(
                    f"The custom expression {e.name!r} has the name of a preset."
                )
    else:
        names = [_vrm0_group_name(s.expression) for s in expressions]
        presets = [s.expression.vrm0_preset for s in expressions]
        for p in presets:
            if p is not None and p not in VRM0_PRESETS:
                raise ValueError(f"{p!r} is not a VRM 0.x preset.")
        used = [p for p in presets if p is not None]
        if len(used) != len(set(used)):
            raise ValueError("A VRM 0.x preset is used by two expressions.")
    lower = [n.lower() for n in names]
    if len(lower) != len(set(lower)):
        raise ValueError("Two expressions share a name.")


def _vrm0_group_name(e: vrm_tables.Expression) -> str:
    """The name of the VRM 0.x blend shape group of an expression."""
    if e.vrm0_name:
        return e.vrm0_name
    if e.vrm0_preset is not None:
        return VRM0_PRESETS.get(e.vrm0_preset, e.name)
    return e.name


def _thumbnail_mime_type(data: bytes) -> str:
    """
    The MIME type of a thumbnail. Raises ValueError unless it is a square PNG or JPEG file:
    VRM 1.0 requires a square thumbnail, and VRM 0.x apps show it as a square.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("A VRM thumbnail must be the bytes of a PNG or JPEG file.")
    if data.startswith(_PNG):
        mime = "image/png"
    elif data.startswith(_JPEG):
        mime = "image/jpeg"
    else:
        raise ValueError("A VRM thumbnail must be a PNG or JPEG file.")
    size = budgets.image_size(data)
    if size is None:
        raise ValueError(
            "The size of the VRM thumbnail cannot be read from its header."
        )
    width, height = size
    if width != height or width == 0:
        raise ValueError(
            f"A VRM thumbnail must be square; this one is {width} x {height} pixels."
        )
    return mime


def _srgb_hex(colour: Sequence[float]) -> str:
    """The sRGB hex code of a linear RGB colour."""
    c = np.clip(np.asarray(colour[:3], dtype=np.float64), 0.0, 1.0)
    s = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in s)


def portrait_png(
    model,
    vertices: np.ndarray,
    joint_positions: np.ndarray,
    size: int = 1024,
    colour: Sequence[float] = SKIN_COLOUR,
    background: str = "#dfe3ea",
) -> bytes:
    """
    A square head-and-shoulders portrait (PNG) of a mesh of the model in Anny's frame, with
    the heads of its bones (J, 3), seen from the front in the flat toon shading of
    :func:`opensculptboy.render.flat.shaded_png`: the thumbnail of a VRM file.

    The frame runs from just above the top of the head down to the upper chest. The mesh is
    drawn in a frame a quarter larger and then cropped, so that the cut of the arms and the
    torso, and its outline, fall outside the picture.
    """
    from PIL import Image

    from opensculptboy.render.flat import View, shaded_png

    labels = list(model.bone_labels)
    vertices = np.asarray(vertices, dtype=np.float64)
    joints = np.asarray(joint_positions, dtype=np.float64)
    faces, _ = triangulated_faces(model)
    view = View()
    right, up, toward = view.basis()
    xy, depth = view.project(vertices)
    head, neck = (
        view.project(joints[[labels.index(b)]])[0][0] for b in ("head", "neck01")
    )
    top = float(xy[:, 1].max())
    side = 1.75 * (top - float(neck[1]))
    centre = np.array([float(head[0]), top + 0.07 * side - side / 2])
    wide = 1.25 * side
    lo, hi = centre - wide / 2, centre + wide / 2
    inside = np.all((xy >= lo) & (xy <= hi), axis=1)
    kept = faces[np.all(inside[faces], axis=1)]
    used = np.unique(kept)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    # Two points that no triangle uses pin the drawing's frame to the larger square.
    d = float(np.mean(-depth[used]))
    corners = np.stack([c[0] * right + c[1] * up + d * toward for c in (lo, hi)])
    points = np.concatenate([vertices[used], corners])
    drawn = int(round(size * 1.25))
    image = shaded_png(
        points,
        remap[kept],
        view,
        size=(drawn, drawn),
        margin=0,
        colour=_srgb_hex(colour),
        supersample=2,
    )
    cut = (drawn - size) // 2
    image = image.crop((cut, cut, cut + size, cut + size))
    canvas = Image.new("RGBA", (size, size), background)
    canvas.alpha_composite(image)
    stream = io.BytesIO()
    canvas.convert("RGB").save(stream, format="PNG")
    return stream.getvalue()


def vrm_document(spec: VrmSpec) -> GltfDocument:
    """The glTF document of a spec, with the ``VRMC_vrm`` (1.0) or ``VRM`` (0.x) extension."""
    doc = GltfDocument(f"OpenSculptBoy VRM exporter (anny {_anny_version()})")
    properties = []
    for material in spec.materials:
        if spec.version == "1.0":
            doc.add_material(mtoon.to_vrm1(material))
        else:
            gltf_material, props = mtoon.to_vrm0(material)
            doc.add_material(gltf_material)
            properties.append(props)

    for node in spec.nodes:
        entry: dict[str, Any] = {"name": node.name}
        if np.any(node.translation != 0):
            entry["translation"] = [float(x) for x in node.translation]
        if node.children:
            entry["children"] = [int(c) for c in node.children]
        if node.mesh is not None:
            entry["mesh"] = int(node.mesh)
        if node.skin is not None:
            entry["skin"] = int(node.skin)
        if node.constraint is not None and spec.version == "1.0":
            entry["extensions"] = {
                "VRMC_node_constraint": {
                    "specVersion": "1.0",
                    "constraint": {
                        "roll": {
                            "source": int(node.constraint.source),
                            "rollAxis": node.constraint.axis,
                            "weight": float(node.constraint.weight),
                        }
                    },
                }
            }
            doc.use_extension("VRMC_node_constraint")
        doc.add_node(entry)

    joint_count = max((len(s.joints) for s in spec.skins), default=0)
    joint_type = np.uint8 if joint_count <= 256 else np.uint16
    for mesh in spec.meshes:
        primitives = []
        for p, part in enumerate(mesh.primitives):
            if p == 0:
                primitive = body_primitive(
                    doc,
                    mesh.body,
                    part.material,
                    translation=mesh.translation,
                    triangles=part.triangles,
                    anny_vertex=spec.anny_vertex,
                    sparse_targets=spec.sparse_targets,
                    joint_type=joint_type,
                )
            else:  # the vertex data and the targets of the first primitive, shared
                first = primitives[0]
                triangles = (
                    mesh.body.triangles if part.triangles is None else part.triangles
                )
                index_type = np.uint16 if len(mesh.body.source) < 65536 else np.uint32
                primitive = {
                    "attributes": dict(first["attributes"]),
                    "indices": doc.accessor(
                        np.asarray(triangles).reshape(-1).astype(index_type),
                        ELEMENT_ARRAY_BUFFER,
                    ),
                    "material": part.material,
                    "mode": 4,
                }
                if "targets" in first:
                    primitive["targets"] = [dict(t) for t in first["targets"]]
            if mesh.target_names:
                primitive["extras"] = {"targetNames": list(mesh.target_names)}
            primitives.append(primitive)
        entry = {"name": mesh.name, "primitives": primitives}
        if mesh.target_names:
            entry["extras"] = {"targetNames": list(mesh.target_names)}
        doc.add_mesh(entry)

    for skin in spec.skins:
        doc.add_skin(
            {
                "name": skin.name,
                "joints": [int(j) for j in skin.joints],
                "skeleton": int(skin.skeleton),
                "inverseBindMatrices": doc.accessor(
                    np.asarray(skin.inverse_bind)
                    .transpose(0, 2, 1)
                    .reshape(-1, 16)
                    .astype(np.float32)
                ),
            }
        )
    doc.scene_nodes = [int(n) for n in spec.scene_nodes]
    doc.scene_name = spec.meta.name
    doc.scene_extras = {"opensculptboy": dict(spec.extras)}

    thumbnail = None
    if spec.thumbnail is not None:
        thumbnail = doc.add_image(
            spec.thumbnail, _thumbnail_mime_type(spec.thumbnail), "thumbnail"
        )
    if spec.version == "1.0":
        doc.extensions["VRMC_vrm"] = _vrm1_extension(spec, thumbnail)
        doc.use_extension("VRMC_vrm")
    else:
        texture = None if thumbnail is None else doc.add_texture(thumbnail)
        doc.extensions["VRM"] = _vrm0_extension(spec, texture, properties)
        doc.use_extension("VRM")
    for material in doc.materials:  # MToon, unlit and the other material extensions
        for extension in material.get("extensions", {}):
            doc.use_extension(extension)
    return doc


def _mesh_nodes(spec: VrmSpec) -> dict[int, list[int]]:
    """The nodes that draw each mesh."""
    nodes: dict[int, list[int]] = {m: [] for m in range(len(spec.meshes))}
    for i, node in enumerate(spec.nodes):
        if node.mesh is not None:
            nodes[node.mesh].append(i)
    return nodes


def credits_text(spec: VrmSpec) -> str:
    """The credits of a file: the base model's, its parts', then those of the metadata."""
    lines = list(spec.credits)
    if spec.meta.third_party_licenses:
        lines.append(spec.meta.third_party_licenses)
    return "\n".join(lines)


def _vrm1_extension(spec: VrmSpec, thumbnail: int | None) -> dict:
    """The ``VRMC_vrm`` extension of a VRM 1.0 file."""
    m = spec.meta
    meta: dict[str, Any] = {"name": m.name}
    if m.version:
        meta["version"] = m.version
    meta["authors"] = list(m.authors)
    if m.copyright_information:
        meta["copyrightInformation"] = m.copyright_information
    if m.contact_information:
        meta["contactInformation"] = m.contact_information
    if m.references:
        meta["references"] = list(m.references)
    meta["thirdPartyLicenses"] = credits_text(spec)
    if thumbnail is not None:
        meta["thumbnailImage"] = int(thumbnail)
    meta.update(
        licenseUrl=VRM1_LICENSE_URL,
        avatarPermission=m.avatar_permission,
        allowExcessivelyViolentUsage=bool(m.allow_excessively_violent_usage),
        allowExcessivelySexualUsage=bool(m.allow_excessively_sexual_usage),
        commercialUsage=m.commercial_usage,
        allowPoliticalOrReligiousUsage=bool(m.allow_political_or_religious_usage),
        allowAntisocialOrHateUsage=bool(m.allow_antisocial_or_hate_usage),
        creditNotation=m.credit_notation,
        allowRedistribution=bool(m.allow_redistribution),
        modification=m.modification,
    )
    if m.other_license_url:
        meta["otherLicenseUrl"] = m.other_license_url

    mesh_nodes = _mesh_nodes(spec)
    annotations = [
        {"node": i, "type": spec.meshes[node.mesh].first_person}
        for i, node in enumerate(spec.nodes)
        if node.mesh is not None
    ]
    look_at: dict[str, Any] = {
        "offsetFromHeadBone": [float(x) for x in spec.look_at.offset_from_head],
        "type": "bone",
    }
    for field, _, key in LOOK_AT_MAPS:
        look_at[field] = {
            "inputMaxValue": 90.0,
            "outputScale": float(spec.look_at.ranges[key]),
        }
    preset: dict[str, dict] = {}
    custom: dict[str, dict] = {}
    for s in spec.expressions:
        e = s.expression
        entry: dict[str, Any] = {}
        binds = [
            {"node": int(node), "index": int(t), "weight": float(w)}
            for m_index, t, w in s.binds
            for node in mesh_nodes[m_index]
        ]
        if binds:
            entry["morphTargetBinds"] = binds
        entry.update(
            isBinary=bool(e.is_binary),
            overrideBlink=e.override_blink,
            overrideLookAt=e.override_look_at,
            overrideMouth=e.override_mouth,
        )
        (preset if e.preset else custom)[e.name] = entry
    expressions = {}
    if preset:
        expressions["preset"] = preset
    if custom:
        expressions["custom"] = custom
    extension: dict[str, Any] = {
        "specVersion": "1.0",
        "meta": meta,
        "humanoid": {
            "humanBones": {b: {"node": int(n)} for b, n in spec.humanoid.items()}
        },
    }
    if annotations:
        extension["firstPerson"] = {"meshAnnotations": annotations}
    extension["lookAt"] = look_at
    if expressions:
        extension["expressions"] = expressions
    return extension


def _vrm0_extension(spec: VrmSpec, texture: int | None, properties: list) -> dict:
    """The ``VRM`` extension of a VRM 0.x file."""
    m = spec.meta
    meta: dict[str, Any] = {"title": m.name}
    if m.version:
        meta["version"] = m.version
    meta["author"] = ", ".join(m.authors)
    if m.contact_information:
        meta["contactInformation"] = m.contact_information
    meta["reference"] = " ".join(
        list(m.references) + [credits_text(spec).replace("\n", " ")]
    )
    if texture is not None:
        meta["texture"] = int(texture)
    meta.update(
        allowedUserName=_ALLOWED_USER_0[m.avatar_permission],
        violentUssageName="Allow" if m.allow_excessively_violent_usage else "Disallow",
        sexualUssageName="Allow" if m.allow_excessively_sexual_usage else "Disallow",
        commercialUssageName="Disallow"
        if m.commercial_usage == "personalNonProfit"
        else "Allow",
        otherPermissionUrl=CREDITS_URL,
    )
    if m.other_license_url:
        meta.update(licenseName="Other", otherLicenseUrl=m.other_license_url)
    elif m.allow_redistribution:
        # VRM 0.x has no name for the VRM Public License that VRM 1.0 files carry.
        meta.update(licenseName="Other", otherLicenseUrl=VRM1_LICENSE_URL)
    else:
        meta["licenseName"] = "Redistribution_Prohibited"

    offset = spec.look_at.offset_from_head
    first_person: dict[str, Any] = {
        "firstPersonBone": int(spec.head),
        # UniVRM stores this offset with z negated (three-vrm negates it again on import).
        "firstPersonBoneOffset": {
            "x": float(offset[0]),
            "y": float(offset[1]),
            "z": -float(offset[2]),
        },
        "meshAnnotations": [
            {
                "mesh": int(m),
                "firstPersonFlag": FIRST_PERSON_TYPES[mesh.first_person],
            }
            for m, mesh in enumerate(spec.meshes)
        ],
        "lookAtTypeName": "Bone",
    }
    for _, field, key in LOOK_AT_MAPS:
        first_person[field] = {
            "curve": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "xRange": 90.0,
            "yRange": float(spec.look_at.ranges[key]),
        }
    groups = []
    for s in spec.expressions:
        e = s.expression
        groups.append(
            {
                "name": _vrm0_group_name(e),
                "presetName": e.vrm0_preset or "unknown",
                "binds": [
                    {"mesh": int(m_index), "index": int(t), "weight": 100.0 * w}
                    for m_index, t, w in s.binds
                ],
                "materialValues": [],
                "isBinary": bool(e.is_binary),
            }
        )
    return {
        "exporterVersion": f"OpenSculptBoy-{_anny_version()}",
        "specVersion": "0.0",
        "meta": meta,
        "humanoid": {
            "humanBones": [
                {"bone": b, "node": int(n), "useDefaultValues": True}
                for b, n in spec.humanoid.items()
            ],
            **VRM0_HUMANOID_DEFAULTS,
        },
        "firstPerson": first_person,
        "blendShapeMaster": {"blendShapeGroups": groups},
        "secondaryAnimation": {"boneGroups": [], "colliderGroups": []},
        "materialProperties": properties,
    }


def write_vrm(spec: VrmSpec, path: str | pathlib.Path, budget: str = "warn") -> dict:
    """
    Serialise a spec and write the VRM file.

    ``budget`` (``"strict"``, ``"warn"`` or ``"off"``) checks the file against the budget of
    its version: ``"strict"`` raises :class:`~opensculptboy.export.budget.BudgetError` before
    anything is written, and ``"warn"`` issues a
    :class:`~opensculptboy.export.budget.BudgetWarning`. The counts and the messages go into
    the scene extras and the summary.
    """
    if budget not in budgets.MODES:
        raise ValueError(f"Unknown budget mode {budget!r}; use one of {budgets.MODES}.")
    doc = vrm_document(spec)
    target = "vrm1" if spec.version == "1.0" else "vrm0"
    found = budgets.counts(doc.to_json(), bytes(doc.buffers.data))
    messages = budgets.check(found, target, budget)
    for message in messages:
        warnings.warn(message, budgets.BudgetWarning, stacklevel=2)
    doc.scene_extras["opensculptboy"]["budget"] = {
        "target": target,
        "mode": budget,
        "counts": found,
        "messages": messages,
    }
    size = doc.write(path)
    return dict(
        path=str(path),
        bytes=size,
        version=spec.version,
        vertices=found["vertices"],
        triangles=found["triangles"],
        joints=found["joints"],
        materials=found["materials"],
        morph_targets=found["morph_targets"],
        expressions=len(spec.expressions),
        bind=spec.extras["options"]["bind"],
        twist=spec.extras["options"]["twist"],
        largest_dropped_weight=max(
            (float(m.body.dropped_weight) for m in spec.meshes), default=0.0
        ),
        counts=found,
        budget=messages,
    )


def export_vrm(
    path: str | pathlib.Path,
    character: Character | None = None,
    model=None,
    version: str = "1.0",
    meta: VrmMeta | None = None,
    author: str | None = None,
    name: str | None = None,
    thumbnail: str | pathlib.Path | bytes | None = None,
    bare: bool = False,
    twist: str | None = None,
    bind: str = DEFAULT_BIND,
    keep_leg_spread: bool = False,
    budget: str = "warn",
    keep_anny_vertex: bool = False,
) -> dict:
    """
    Write a character as a VRM file.

    Args:
        path: the ``.vrm`` file to write.
        character: the character; the default character when None. Its rig must be
            ``anny`` and its topology the MakeHuman body mesh with eyes
            (:func:`check_topology`). Its face shapes are baked into the mesh, and its facial
            actions and its pose are ignored (the file starts neutral, in the T-pose).
        model: the Anny model; ``character.build_model()`` when None. It needs
            ``facial_actions="all"`` for the expressions, and face shapes for a character with
            face shapes.
        version: ``"1.0"`` (``VRMC_vrm``) or ``"0.x"`` (the ``VRM`` extension).
        meta: the licence metadata; built from ``author`` and ``name`` when None.
        author: the author, required unless ``meta`` names authors; replaces them otherwise.
        name: the avatar's name; the metadata's name, else the character's name, when None.
        thumbnail: a square PNG or JPEG file, as a path or bytes, ``"auto"`` for a rendered
            portrait (:func:`portrait_png`), or None (``meta.thumbnail_png``, if any).
        bare: the body alone (no outfit, hair cards or face kit; PR 1 has only the body).
        twist: ``"constraint"`` (the VRM 1.0 default: roll constraints turn the twist bones)
            or ``"merge"`` (the 0.x default: the twist bones' weights move to their parents).
            VRM 0.x files cannot carry constraints, so ``"constraint"`` raises ValueError
            there. The scene extras and the summary record the mode in use.
        bind: ``"forward"`` or ``"inverse"`` (see :func:`opensculptboy.export.tpose.rebind`).
        keep_leg_spread: keep the rig's leg spread in the T-pose.
        budget: ``"strict"``, ``"warn"`` or ``"off"`` (see :mod:`opensculptboy.export.budget`).
        keep_anny_vertex: write the ``_ANNY_VERTEX`` attribute (the Anny vertex of each file
            vertex), which VRM files leave out by default.

    Returns:
        A summary: counts, the file size, the version and the budget messages.

    Raises:
        ValueError: for a character, an option, metadata or a thumbnail that a VRM file
            cannot carry, before anything is written. The topology, the rig, the options,
            the metadata and the thumbnail are checked before the model is built.
        OSError: for a thumbnail file that cannot be read.
        BudgetError: for a file over its budget with ``budget="strict"``, before anything
            is written.
    """
    if budget not in budgets.MODES:
        raise ValueError(f"Unknown budget mode {budget!r}; use one of {budgets.MODES}.")
    spec = vrm_spec(
        character,
        model,
        version=version,
        meta=meta,
        author=author,
        name=name,
        thumbnail=thumbnail,
        bare=bare,
        twist=twist,
        bind=bind,
        keep_leg_spread=keep_leg_spread,
        keep_anny_vertex=keep_anny_vertex,
    )
    return write_vrm(spec, path, budget=budget)


def _anny_version() -> str:
    import anny

    return anny.__version__
