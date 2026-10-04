# OpenSculptBoy
# Apache License, Version 2.0
"""
VRM export of an Anny character, for VTuber apps: VRM 1.0 (``VRMC_vrm``) by default and VRM 0.x
on request.

The file holds the character in the VRM T-pose (:mod:`opensculptboy.export.tpose`) with
normalised joints (every node at the identity rotation), 4 skin weights per vertex, the
humanoid bone map, the expressions as mixes of the ARKit facial actions, bone look-at on the
eye bones, MToon materials, the licence metadata and the character card in the scene extras.

CONTRACT (PR 1, stream A4 implements): ``export_vrm`` below is a minimal VRM 1.0 placeholder so
that the other streams can run end to end; A4 replaces it with the version-neutral ``VrmSpec``
and the 1.0 and 0.x serialisers.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import torch

from opensculptboy.character import Character
from opensculptboy.export import mtoon, tpose, vrm_tables
from opensculptboy.export.body import ANNY_TO_GLTF, body_primitive, build_body
from opensculptboy.export.document import GltfDocument

VRM1_LICENSE_URL = "https://vrm.dev/licenses/1.0/"


@dataclasses.dataclass
class VrmMeta:
    """The licence metadata of a VRM file, in VRM 1.0 terms (the 0.x writer maps it)."""

    name: str
    authors: list[str]
    version: str | None = None
    copyright_information: str | None = None
    contact_information: str | None = None
    references: list[str] = dataclasses.field(default_factory=list)
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


def export_vrm(
    path: str | pathlib.Path,
    character: Character | None = None,
    model=None,
    version: str = "1.0",
    meta: VrmMeta | None = None,
    author: str | None = None,
    name: str | None = None,
    thumbnail: str | bytes | None = None,
    bare: bool = False,
    twist: str | None = None,
    bind: str = "forward",
    keep_leg_spread: bool = False,
    budget: str = "warn",
) -> dict:
    """
    Write a character as a VRM file.

    Args:
        path: the ``.vrm`` file to write.
        character: the character; the default character when None. Its face shapes are baked
            into the mesh, and its facial actions are ignored (the file starts neutral).
        model: the Anny model; ``character.build_model()`` when None.
        version: ``"1.0"`` (``VRMC_vrm``) or ``"0.x"`` (the ``VRM`` extension).
        meta: the licence metadata; built from ``author`` and ``name`` when None.
        author: the author, required when ``meta`` is None.
        name: the avatar's name; the character's name when None.
        thumbnail: a PNG path or PNG bytes, ``"auto"`` for a rendered portrait, or None.
        bare: the body alone (no outfit, hair cards or face kit; PR 1 has only the body).
        twist: ``"constraint"`` (the VRM 1.0 default) or ``"merge"`` (the 0.x default).
        bind: ``"forward"`` or ``"inverse"`` (see :func:`opensculptboy.export.tpose.rebind`).
        keep_leg_spread: keep the rig's leg spread in the T-pose.
        budget: ``"strict"``, ``"warn"`` or ``"off"`` (see :mod:`opensculptboy.export.budget`).

    Returns:
        A summary: counts, the file size, the version and the budget messages.
    """
    if version != "1.0":
        raise NotImplementedError("placeholder: stream A4 writes VRM 0.x")
    character = character or Character()
    if meta is None:
        if not author:
            raise ValueError("A VRM file needs an author: pass author= or meta=.")
        meta = VrmMeta(name=name or character.name, authors=[author])
    model = model or character.build_model(face_shapes=bool(character.face_shapes))
    if character.rig != "anny":
        raise ValueError("VRM export supports the 'anny' rig only.")
    actions = list(model.facial_action_labels)
    face_labels = list(model.face_shape_labels)
    dtype = model.template_vertices.dtype
    rows = torch.zeros(len(actions) + 1, len(actions), dtype=dtype)
    rows[1:] = torch.eye(len(actions), dtype=dtype)
    call = dict(
        phenotype_kwargs=dict(character.phenotype) or None,
        local_changes_kwargs=dict(character.local_changes) or None,
        facial_actions=rows,
    )
    if face_labels:
        call["face_shape_kwargs"] = {
            k: torch.full((len(rows),), float(v), dtype=dtype)
            for k, v in character.face_shapes.items()
        }
    with torch.no_grad():
        out = model(**call)
    rest = out["rest_vertices"].double().cpu().numpy()
    base, offsets = rest[0], rest[1:] - rest[0]
    B = out["rest_bone_poses"][0].double().cpu().numpy()
    weights, indices = vrm_tables.vrm_skin_weights(model, twist or "constraint")
    rb = tpose.rebind(model, base, offsets, B, weights, indices, method=bind)
    eyes = vrm_tables.eyeball_vertices(model)
    targets = [
        vrm_tables.lid_only(t, eyes) if a.startswith("eyeLook") else t
        for a, t in zip(actions, rb.targets)
    ]
    body = build_body(model, rb.vertices, targets, ANNY_TO_GLTF, 4, weights, indices)
    joints = rb.joint_positions @ ANNY_TO_GLTF.T
    lift = np.array([0.0, -float(body.positions[:, 1].min()), 0.0])
    joints = joints + lift

    doc = GltfDocument("OpenSculptBoy VRM exporter")
    parents = [int(p) for p in model.bone_parents]
    labels = list(model.bone_labels)
    for j, label in enumerate(labels):
        local = joints[j] - (joints[parents[j]] if parents[j] >= 0 else 0.0)
        node = {"name": label, "translation": local.tolist()}
        children = [c for c, p in enumerate(parents) if p == j]
        if children:
            node["children"] = children
        doc.add_node(node)
    material = doc.add_material(
        mtoon.to_vrm1(mtoon.soft_material("skin", (0.80, 0.62, 0.52, 1.0)))
    )
    doc.use_extension("VRMC_materials_mtoon")
    primitive = body_primitive(doc, body, material, translation=lift)
    mesh = doc.add_mesh(
        {
            "name": "body",
            "primitives": [primitive],
            "extras": {"targetNames": actions},
        }
    )
    primitive["extras"] = {"targetNames": actions}
    inverse_bind = np.tile(np.eye(4), (len(labels), 1, 1))
    inverse_bind[:, :3, 3] = -joints
    skin = doc.add_skin(
        {
            "joints": list(range(len(labels))),
            "skeleton": 0,
            "inverseBindMatrices": doc.accessor(
                inverse_bind.transpose(0, 2, 1).reshape(-1, 16).astype(np.float32)
            ),
        }
    )
    mesh_node = doc.add_node({"name": "body", "mesh": mesh, "skin": skin})
    doc.scene_nodes = [0, mesh_node]
    doc.scene_name = meta.name
    doc.scene_extras = {
        "opensculptboy": {
            "generator": "opensculptboy",
            "character": character.to_dict(),
            "format": "vrm",
            "vrm_version": version,
        }
    }
    human = {
        bone: {"node": labels.index(label)}
        for bone, label in vrm_tables.humanoid_bones(version).items()
    }
    preset, custom = {}, {}
    for e in vrm_tables.expressions(version, actions):
        entry = {
            "morphTargetBinds": [
                {"node": mesh_node, "index": actions.index(a), "weight": float(w)}
                for a, w in e.mix.items()
            ],
            "isBinary": e.is_binary,
            "overrideBlink": e.override_blink,
            "overrideLookAt": e.override_look_at,
            "overrideMouth": e.override_mouth,
        }
        (preset if e.preset else custom)[e.name] = entry
    vrm_meta = {
        "name": meta.name,
        "authors": list(meta.authors),
        "licenseUrl": VRM1_LICENSE_URL,
        "avatarPermission": meta.avatar_permission,
        "commercialUsage": meta.commercial_usage,
        "creditNotation": meta.credit_notation,
        "allowRedistribution": meta.allow_redistribution,
        "modification": meta.modification,
    }
    doc.use_extension("VRMC_vrm")
    doc.extensions["VRMC_vrm"] = {
        "specVersion": "1.0",
        "meta": vrm_meta,
        "humanoid": {"humanBones": human},
        "expressions": {"preset": preset, "custom": custom},
    }
    size = doc.write(path)
    return dict(
        path=str(path),
        bytes=size,
        version=version,
        vertices=int(len(body.source)),
        triangles=int(len(body.triangles)),
        joints=len(labels),
        morph_targets=len(targets),
        budget=[],
    )
