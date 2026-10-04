# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the VRM writer (opensculptboy.export.vrm), the budgets (opensculptboy.export.budget)
and the ``.vrm`` exports of the ``opensculptboy`` command.

The files come from ``test/vrm_fixtures.py`` (one VRM 1.0 and one VRM 0.x export of a
character off the default shape, shared by every VRM test). A NumPy evaluation of the files,
built on ``test/gltf_reader.py``, checks the geometry against the T-pose rebind of
``opensculptboy.export.tpose``, whatever T-pose it builds: the file at rest is G v' + o, and
Anny's posed bones M_j drive the file through W_j = translate(o) G M_j T_j^-1 G^-1
translate(P_j - o).
"""

import base64
import contextlib
import dataclasses
import io
import json
import pathlib
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import torch

from opensculptboy import cli, read_character
from opensculptboy.export import budget, vrm, vrm_tables
from opensculptboy.export.body import triangulated_faces, vertex_normals
from test import vrm_fixtures
from test.gltf_reader import GLB, node_world_matrices

VERSIONS = ("1.0", "0.x")


def _skin_matrices(glb, skin_index, world):
    """The joint matrices of a skin: world matrix times inverse bind matrix."""
    skin = glb.json["skins"][skin_index]
    ibm = glb.accessor(skin["inverseBindMatrices"]).astype(np.float64)
    ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)
    return np.stack([world[j] for j in skin["joints"]]) @ ibm


def skinned_nodes(glb, world=None):
    """
    The skinned vertices of the first primitive of every node with a mesh and a skin, for node
    world matrices (by default the file's own), with every morph weight at 0.
    """
    g = glb.json
    world = node_world_matrices(glb) if world is None else world
    result = {}
    for i, node in enumerate(g["nodes"]):
        if "mesh" not in node or "skin" not in node:
            continue
        attrs = g["meshes"][node["mesh"]]["primitives"][0]["attributes"]
        positions = glb.accessor(attrs["POSITION"]).astype(np.float64)
        matrices = _skin_matrices(glb, node["skin"], world)
        out = np.zeros_like(positions)
        s = 0
        while f"JOINTS_{s}" in attrs:
            joints = glb.accessor(attrs[f"JOINTS_{s}"]).astype(int)
            weights = glb.accessor(attrs[f"WEIGHTS_{s}"]).astype(np.float64)
            for k in range(joints.shape[1]):
                m = matrices[joints[:, k]]
                moved = np.einsum("vij,vj->vi", m[:, :3, :3], positions) + m[:, :3, 3]
                out += weights[:, k, None] * moved
            s += 1
        result[i] = out
    return result


def body_node(glb) -> int:
    return next(i for i, n in enumerate(glb.json["nodes"]) if n.get("name") == "body")


def node_positions(glb) -> dict[str, np.ndarray]:
    """The world position of every named node in the default pose."""
    world = node_world_matrices(glb)
    return {
        n["name"]: world[i][:3, 3]
        for i, n in enumerate(glb.json["nodes"])
        if world[i] is not None
    }


def rigid_inverse(m):
    inverse = np.zeros_like(m)
    r = np.swapaxes(m[..., :3, :3], -1, -2)
    inverse[..., :3, :3] = r
    inverse[..., :3, 3] = -(r @ m[..., :3, 3, None])[..., 0]
    inverse[..., 3, 3] = 1.0
    return inverse


def target_offsets(glb, mesh_index):
    """The morph target offsets (R, N, 3) of the first primitive of a mesh."""
    prim = glb.json["meshes"][mesh_index]["primitives"][0]
    return np.stack([glb.accessor(t["POSITION"]) for t in prim.get("targets", [])])


def _no_empty_arrays(value, where="VRMC_vrm"):
    """The paths of the empty lists in a JSON value."""
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            found += _no_empty_arrays(v, f"{where}.{k}")
    elif isinstance(value, list):
        if not value:
            found.append(where)
        for k, v in enumerate(value):
            found += _no_empty_arrays(v, f"{where}[{k}]")
    return found


class TestVrmGeometry(unittest.TestCase):
    """The mesh and the skeleton of both versions against the T-pose rebind."""

    def test_rest_mesh_is_the_bind_mesh(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                spec = vrm_fixtures.vrm_build(version)
                glb = vrm_fixtures.vrm_glb(version)
                nodes = skinned_nodes(glb)
                self.assertIn(body_node(glb), nodes)
                # The rest pose is the bind pose: skinning moves no vertex of any node.
                for i, at_rest in nodes.items():
                    mesh = glb.json["meshes"][glb.json["nodes"][i]["mesh"]]
                    attrs = mesh["primitives"][0]["attributes"]
                    positions = glb.accessor(attrs["POSITION"]).astype(np.float64)
                    np.testing.assert_allclose(at_rest, positions, atol=1e-6)
                # The body at rest is G v' + o.
                body = spec.meshes[0].body
                expected = (spec.rebind.vertices @ spec.frame.T + spec.offset)[
                    body.source
                ]
                at_rest = nodes[body_node(glb)]
                np.testing.assert_allclose(at_rest, expected, atol=1e-5)
                # Centred: the hips over the origin, the lowest vertex on the floor.
                hips = node_positions(glb)["root"]
                self.assertAlmostEqual(hips[0], 0.0, places=6)
                self.assertAlmostEqual(hips[2], 0.0, places=6)
                self.assertAlmostEqual(at_rest[:, 1].min(), 0.0, places=6)

    def test_normals_of_the_welded_bind_mesh(self):
        """
        The file carries the normals of the bind mesh, rotated by G: those of the rebind
        when it computes them, else those of the welded bind vertices.
        """
        triangles, _ = triangulated_faces(vrm_fixtures.model())
        for version in VERSIONS:
            with self.subTest(version=version):
                spec = vrm_fixtures.vrm_build(version)
                glb = vrm_fixtures.vrm_glb(version)
                bind_normals = getattr(spec.rebind, "normals", None)
                if bind_normals is None:
                    bind_normals = vertex_normals(spec.rebind.vertices, triangles)
                expected = np.asarray(bind_normals, dtype=np.float64) @ spec.frame.T
                attrs = glb.json["meshes"][0]["primitives"][0]["attributes"]
                normals = glb.accessor(attrs["NORMAL"]).astype(np.float64)
                np.testing.assert_allclose(
                    normals, expected[spec.meshes[0].body.source], atol=1e-4
                )
                np.testing.assert_allclose(
                    np.linalg.norm(normals, axis=1), 1.0, atol=1e-4
                )

    def test_normalised_joints(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                spec = vrm_fixtures.vrm_build(version)
                glb = vrm_fixtures.vrm_glb(version)
                for node in glb.json["nodes"]:
                    self.assertNotIn("matrix", node)
                    self.assertNotIn("scale", node)
                    np.testing.assert_allclose(
                        node.get("rotation", [0, 0, 0, 1]), [0, 0, 0, 1], atol=1e-12
                    )
                world = node_world_matrices(glb)
                skin = glb.json["skins"][0]
                ibm = glb.accessor(skin["inverseBindMatrices"]).astype(np.float64)
                ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)
                np.testing.assert_allclose(
                    ibm[:, :3, :3], np.tile(np.eye(3), (len(ibm), 1, 1)), atol=0
                )
                np.testing.assert_allclose(ibm[:, 3], [[0, 0, 0, 1]] * len(ibm), atol=0)
                for j, joint in enumerate(skin["joints"]):
                    np.testing.assert_allclose(
                        ibm[j] @ world[joint], np.eye(4), atol=1e-6
                    )
                # The joints are the heads of the T-pose: P_j = G t_j + o.
                joints = np.stack([world[j][:3, 3] for j in skin["joints"]])
                expected = spec.rebind.joint_positions @ spec.frame.T + spec.offset
                np.testing.assert_allclose(joints, expected, atol=1e-6)
                # hips is root, at its own head.
                hips = glb.json["nodes"][spec.humanoid["hips"]]["name"]
                self.assertEqual(hips, "root")

    def test_frames_of_the_versions(self):
        expected = {
            "1.0": 1.0,
            "0.x": -1.0,
        }  # the toes and the face: +Z (1.0), -Z (0.x)
        for version in VERSIONS:
            with self.subTest(version=version):
                p = node_positions(vrm_fixtures.vrm_glb(version))
                forward = expected[version]
                for side in ("L", "R"):
                    toes = p[f"toe3-1.{side}"] - p[f"foot.{side}"]
                    self.assertGreater(forward * toes[2], 2 * abs(toes[0]))
                eyes = (p["eye.L"] + p["eye.R"]) / 2 - p["head"]
                self.assertGreater(forward * eyes[2], 0.0)
                self.assertGreater(p["head"][1], p["root"][1])  # Y up
                # The figure's left (+X in Anny's frame) is +X in VRM 1.0, -X in VRM 0.x.
                self.assertGreater(forward * (p["eye.L"][0] - p["eye.R"][0]), 0.0)

    def test_pose_mapping_reproduces_anny(self):
        """
        Anny's posed bones M_j on two frames of 'walk' drive the file through W_j: the file
        then skins its vertices like Anny's bones skin the bind mesh with the file's weights,
        and every joint keeps its default translation from its parent, as VRM apps pose a
        normalised skeleton by rotations alone.
        """
        import anny.poses

        model = vrm_fixtures.model()
        character = vrm_fixtures.CHARACTER
        entry = anny.poses.pose_parameters(
            model, "walk", phenotype_kwargs=dict(character.phenotype), grounded=False
        )
        params = entry["pose_parameters"]
        frames = [0, len(params) // 2]
        with torch.no_grad():
            out = model(
                pose_parameters=params[frames],
                pose_parameterization="local-ref",
                phenotype_kwargs=dict(character.phenotype),
                face_shape_kwargs=dict(character.face_shapes),
            )
        posed = out["bone_poses"].double().cpu().numpy()
        anny_parents = [int(p) for p in model.bone_parents]
        for version in VERSIONS:
            spec = vrm_fixtures.vrm_build(version)
            glb = vrm_fixtures.vrm_glb(version)
            G = np.eye(4)
            G[:3, :3] = spec.frame
            o = np.eye(4)
            o[:3, 3] = spec.offset
            T_inv = rigid_inverse(spec.rebind.bone_poses)
            P = spec.joint_positions
            to_joint = np.tile(np.eye(4), (len(P), 1, 1))
            to_joint[:, :3, 3] = P - spec.offset
            body = spec.meshes[0].body
            children = {
                c: i
                for i, n in enumerate(glb.json["nodes"])
                for c in n.get("children", [])
            }
            for f, M in zip(frames, posed):
                with self.subTest(version=version, frame=f):
                    W = o @ G @ M @ T_inv @ G.T @ to_joint
                    world = list(W) + [np.eye(4)] * (len(glb.json["nodes"]) - len(W))
                    moved = skinned_nodes(glb, world)[body_node(glb)]
                    # Anny's bones on the bind mesh, with the file's 4 weights.
                    X = M @ T_inv
                    reference = np.zeros_like(spec.rebind.vertices)
                    for k in range(4):
                        m = X[spec.skin_indices[:, k]]
                        reference += spec.skin_weights[:, k, None] * (
                            np.einsum("vij,vj->vi", m[:, :3, :3], spec.rebind.vertices)
                            + m[:, :3, 3]
                        )
                    reference = reference @ spec.frame.T + spec.offset
                    np.testing.assert_allclose(moved, reference[body.source], atol=1e-5)
                    for j, node in enumerate(glb.json["nodes"][: len(W)]):
                        p = children.get(j, -1)
                        if p < 0 or p != anny_parents[j]:
                            continue
                        local = np.linalg.inv(W[p]) @ W[j]
                        np.testing.assert_allclose(
                            local[:3, 3], node.get("translation", [0, 0, 0]), atol=1e-5
                        )

    def test_skin_weights(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                spec = vrm_fixtures.vrm_build(version)
                glb = vrm_fixtures.vrm_glb(version)
                attrs = glb.json["meshes"][0]["primitives"][0]["attributes"]
                self.assertNotIn("JOINTS_1", attrs)
                joints_accessor = glb.json["accessors"][attrs["JOINTS_0"]]
                self.assertEqual(joints_accessor["componentType"], 5121)  # uint8
                source = spec.meshes[0].body.source
                weights = glb.accessor(attrs["WEIGHTS_0"]).astype(np.float64)
                joints = glb.accessor(attrs["JOINTS_0"]).astype(int)
                np.testing.assert_allclose(weights.sum(1), 1.0, atol=1e-6)
                np.testing.assert_allclose(
                    weights, spec.skin_weights[source], atol=1e-6
                )
                used = weights > 0
                np.testing.assert_array_equal(
                    joints[used], spec.skin_indices[source][used]
                )

    def test_morph_targets(self):
        model = vrm_fixtures.model()
        actions = list(model.facial_action_labels)
        eyeballs = vrm_tables.eyeball_vertices(model)
        eyes = np.concatenate(list(eyeballs.values()))
        for version in VERSIONS:
            with self.subTest(version=version):
                spec = vrm_fixtures.vrm_build(version)
                glb = vrm_fixtures.vrm_glb(version)
                g = glb.json
                mesh = g["meshes"][0]
                self.assertEqual(mesh["extras"]["targetNames"], actions)
                for prim in mesh["primitives"]:
                    self.assertEqual(prim["extras"]["targetNames"], actions)
                    self.assertEqual(len(prim["targets"]), len(actions))
                sparse = [
                    "sparse" in g["accessors"][t["POSITION"]]
                    for t in mesh["primitives"][0]["targets"]
                ]
                if version == "0.x":
                    self.assertFalse(any(sparse), "VRM 0.x writes dense targets")
                offsets = target_offsets(glb, 0)
                source = spec.meshes[0].body.source
                on_eyes = np.isin(source, eyes)
                for r, action in enumerate(actions):
                    expected = np.asarray(spec.rebind.targets[r])
                    if action.startswith("eyeLook"):
                        # Lid-only: the eye bones alone turn the eyeballs.
                        self.assertEqual(np.abs(offsets[r][on_eyes]).max(), 0.0)
                        expected = vrm_tables.lid_only(expected, eyeballs)
                    np.testing.assert_allclose(
                        offsets[r], (expected @ spec.frame.T)[source], atol=1e-5
                    )
                self.assertNotIn("weights", mesh)  # the file starts neutral
                self.assertNotIn("animations", g)

    def test_anny_vertex_on_request(self):
        spec = vrm_fixtures.vrm_build("1.0")
        attrs = vrm_fixtures.vrm_glb("1.0").json["meshes"][0]["primitives"][0]
        self.assertNotIn("_ANNY_VERTEX", attrs["attributes"])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "anny_vertex.vrm"
            vrm.write_vrm(dataclasses.replace(spec, anny_vertex=True), path)
            glb = GLB(path)
        attrs = glb.json["meshes"][0]["primitives"][0]["attributes"]
        np.testing.assert_array_equal(
            glb.accessor(attrs["_ANNY_VERTEX"]).astype(int),
            spec.meshes[0].body.source,
        )


class TestVrmExtension(unittest.TestCase):
    """The VRMC_vrm (1.0) and VRM (0.x) extensions."""

    def test_humanoid(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                g = vrm_fixtures.vrm_json(version)
                nodes = g["nodes"]
                if version == "1.0":
                    human = {
                        b: v["node"]
                        for b, v in g["extensions"]["VRMC_vrm"]["humanoid"][
                            "humanBones"
                        ].items()
                    }
                else:
                    bones = g["extensions"]["VRM"]["humanoid"]["humanBones"]
                    for b in bones:
                        self.assertIs(b["useDefaultValues"], True)
                    human = {b["bone"]: b["node"] for b in bones}
                    self.assertEqual(len(human), len(bones))
                self.assertLessEqual(vrm_tables.required_bones(version), set(human))
                labels = {n["name"]: i for i, n in enumerate(nodes)}
                for bone, label in vrm_tables.humanoid_bones(version).items():
                    self.assertEqual(human[bone], labels[label], bone)
                # Every humanoid bone hangs below hips.
                parent = {
                    c: i for i, n in enumerate(nodes) for c in n.get("children", [])
                }
                for bone, node in human.items():
                    chain = [node]
                    while chain[-1] in parent:
                        chain.append(parent[chain[-1]])
                    self.assertIn(human["hips"], chain, bone)

    def test_node_names_and_hierarchy(self):
        model = vrm_fixtures.model()
        for version in VERSIONS:
            with self.subTest(version=version):
                g = vrm_fixtures.vrm_json(version)
                names = [n["name"] for n in g["nodes"]]
                self.assertEqual(len(names), len(set(names)))
                twist = g["scenes"][0]["extras"]["opensculptboy"]["options"]["twist"]
                parents = vrm_tables.file_parents(model, version, twist)
                for j, node in enumerate(g["nodes"][: len(parents)]):
                    self.assertEqual(
                        node.get("children", []),
                        [c for c, p in enumerate(parents) if p == j],
                    )
                roots = [j for j, p in enumerate(parents) if p < 0]
                self.assertEqual(g["scenes"][0]["nodes"], roots + [len(parents)])

    def test_twist_constraints(self):
        model = vrm_fixtures.model()
        labels = list(model.bone_labels)
        for version in VERSIONS:
            with self.subTest(version=version):
                g = vrm_fixtures.vrm_json(version)
                twist = g["scenes"][0]["extras"]["opensculptboy"]["options"]["twist"]
                self.assertEqual(twist, {"1.0": "constraint", "0.x": "merge"}[version])
                expected = {
                    labels.index(bone): (labels.index(source), axis, weight)
                    for bone, source, axis, weight in vrm_tables.twist_constraints(
                        version, twist
                    )
                }
                found = {}
                for i, node in enumerate(g["nodes"]):
                    ext = node.get("extensions", {}).get("VRMC_node_constraint")
                    if ext is None:
                        continue
                    self.assertEqual(ext["specVersion"], "1.0")
                    roll = ext["constraint"]["roll"]
                    found[i] = (roll["source"], roll["rollAxis"], roll["weight"])
                self.assertEqual(found, expected)
                used = "VRMC_node_constraint" in g.get("extensionsUsed", [])
                self.assertEqual(used, bool(expected))

    def test_expression_binds(self):
        model = vrm_fixtures.model()
        actions = list(model.facial_action_labels)
        for version in VERSIONS:
            with self.subTest(version=version):
                glb = vrm_fixtures.vrm_glb(version)
                g = glb.json
                # (mesh, target) pairs that move, from the file.
                moving = {}
                for m, mesh in enumerate(g["meshes"]):
                    names = mesh.get("extras", {}).get("targetNames", [])
                    offsets = target_offsets(glb, m) if names else []
                    for t, name in enumerate(names):
                        moving[m, name] = (
                            t,
                            bool(np.linalg.norm(offsets[t], axis=1).max() > 0),
                        )
                mesh_nodes = {
                    i: n["mesh"] for i, n in enumerate(g["nodes"]) if "mesh" in n
                }
                table = vrm_tables.expressions(version, actions)
                if version == "1.0":
                    ext = g["extensions"]["VRMC_vrm"]["expressions"]
                    entries = {**ext.get("preset", {}), **ext.get("custom", {})}
                    for name in ext.get("preset", {}):
                        self.assertIn(name, vrm.VRM1_PRESETS)
                    lower = {p.lower() for p in vrm.VRM1_PRESETS}
                    for name in ext.get("custom", {}):
                        self.assertNotIn(name.lower(), lower)
                    self.assertEqual(len(entries), len(table))
                    for e in table:
                        entry = entries[e.name]
                        self.assertEqual(entry["isBinary"], e.is_binary)
                        for key in ("overrideBlink", "overrideLookAt", "overrideMouth"):
                            self.assertIn(entry[key], ("none", "block", "blend"))
                        binds = {
                            (b["node"], b["index"]): b["weight"]
                            for b in entry.get("morphTargetBinds", [])
                        }
                        for (node, index), weight in binds.items():
                            self.assertIn(node, mesh_nodes)
                            self.assertLess(
                                index,
                                len(
                                    g["meshes"][mesh_nodes[node]]["primitives"][0][
                                        "targets"
                                    ]
                                ),
                            )
                            self.assertTrue(0.0 <= weight <= 1.0)
                        expected = {
                            (node, moving[m, a][0]): w
                            for a, w in e.mix.items()
                            if w > 0
                            for node, m in mesh_nodes.items()
                            if (m, a) in moving and moving[m, a][1]
                        }
                        self.assertEqual(binds, expected, e.name)
                else:
                    groups = g["extensions"]["VRM"]["blendShapeMaster"][
                        "blendShapeGroups"
                    ]
                    self.assertEqual(len(groups), len(table))
                    names = [grp["name"].lower() for grp in groups]
                    self.assertEqual(len(names), len(set(names)))
                    presets = [
                        grp["presetName"]
                        for grp in groups
                        if grp["presetName"] != "unknown"
                    ]
                    self.assertEqual(len(presets), len(set(presets)))
                    for grp, e in zip(groups, table):
                        self.assertIn(grp["presetName"], ["unknown", *vrm.VRM0_PRESETS])
                        self.assertEqual(grp["presetName"], e.vrm0_preset or "unknown")
                        self.assertEqual(grp["isBinary"], e.is_binary)
                        binds = {
                            (b["mesh"], b["index"]): b["weight"] for b in grp["binds"]
                        }
                        for (m, index), weight in binds.items():
                            self.assertLess(m, len(g["meshes"]))
                            self.assertLess(
                                index, len(g["meshes"][m]["primitives"][0]["targets"])
                            )
                            self.assertTrue(0.0 <= weight <= 100.0)
                        expected = {
                            (m, moving[m, a][0]): 100.0 * w
                            for a, w in e.mix.items()
                            if w > 0
                            for m in range(len(g["meshes"]))
                            if (m, a) in moving and moving[m, a][1]
                        }
                        self.assertEqual(binds.keys(), expected.keys(), e.name)
                        for key, weight in binds.items():
                            self.assertAlmostEqual(weight, expected[key], places=9)

    def test_look_at_and_first_person(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                spec = vrm_fixtures.vrm_build(version)
                glb = vrm_fixtures.vrm_glb(version)
                g = glb.json
                p = node_positions(glb)
                offset = (p["eye.L"] + p["eye.R"]) / 2 - p["head"]
                ranges = spec.look_at.ranges
                self.assertEqual(
                    set(ranges), {"lookUp", "lookDown", "lookIn", "lookOut"}
                )
                head = [n["name"] for n in g["nodes"]].index("head")
                mesh_nodes = [i for i, n in enumerate(g["nodes"]) if "mesh" in n]
                if version == "1.0":
                    ext = g["extensions"]["VRMC_vrm"]
                    look = ext["lookAt"]
                    self.assertEqual(look["type"], "bone")
                    np.testing.assert_allclose(
                        look["offsetFromHeadBone"], offset, atol=1e-6
                    )
                    for field, _, key in vrm.LOOK_AT_MAPS:
                        self.assertEqual(look[field]["inputMaxValue"], 90.0)
                        self.assertEqual(look[field]["outputScale"], ranges[key])
                    self.assertEqual(
                        ext["firstPerson"]["meshAnnotations"],
                        [{"node": i, "type": "auto"} for i in mesh_nodes],
                    )
                else:
                    first = g["extensions"]["VRM"]["firstPerson"]
                    self.assertEqual(first["firstPersonBone"], head)
                    o = first["firstPersonBoneOffset"]
                    np.testing.assert_allclose(
                        [o["x"], o["y"], -o["z"]], offset, atol=1e-6
                    )
                    self.assertEqual(first["lookAtTypeName"], "Bone")
                    for _, field, key in vrm.LOOK_AT_MAPS:
                        self.assertEqual(
                            first[field],
                            {
                                "curve": [0, 0, 0, 1, 1, 1, 1, 0],
                                "xRange": 90,
                                "yRange": ranges[key],
                            },
                        )
                    self.assertEqual(
                        first["meshAnnotations"],
                        [
                            {"mesh": g["nodes"][i]["mesh"], "firstPersonFlag": "Auto"}
                            for i in mesh_nodes
                        ],
                    )

    def test_meta_defaults(self):
        g1 = vrm_fixtures.vrm_json("1.0")["extensions"]["VRMC_vrm"]
        meta = g1["meta"]
        self.assertEqual(meta["name"], vrm_fixtures.CHARACTER.name)
        self.assertEqual(meta["authors"], [vrm_fixtures.AUTHOR])
        self.assertEqual(meta["licenseUrl"], "https://vrm.dev/licenses/1.0/")
        self.assertEqual(meta["avatarPermission"], "onlyAuthor")
        self.assertEqual(meta["commercialUsage"], "personalNonProfit")
        self.assertEqual(meta["modification"], "prohibited")
        self.assertIs(meta["allowRedistribution"], False)
        self.assertIs(meta["allowExcessivelySexualUsage"], False)
        self.assertNotIn("references", meta)
        self.assertNotIn("thumbnailImage", meta)
        for credit in ("OpenSculptBoy", "NAVER", "MakeHuman", "Mika Suominen", "CC0"):
            self.assertIn(credit, meta["thirdPartyLicenses"])
        self.assertIn("ICT-FaceKit", meta["thirdPartyLicenses"])
        self.assertEqual(_no_empty_arrays(g1), [])
        g0 = vrm_fixtures.vrm_json("0.x")["extensions"]["VRM"]
        self.assertEqual(g0["specVersion"], "0.0")
        self.assertTrue(g0["exporterVersion"].startswith("OpenSculptBoy-"))
        meta = g0["meta"]
        self.assertEqual(meta["title"], vrm_fixtures.CHARACTER.name)
        self.assertEqual(meta["author"], vrm_fixtures.AUTHOR)
        self.assertEqual(meta["allowedUserName"], "OnlyAuthor")
        for key in ("violentUssageName", "sexualUssageName", "commercialUssageName"):
            self.assertEqual(meta[key], "Disallow")
        self.assertEqual(meta["licenseName"], "Redistribution_Prohibited")
        self.assertIn("NAVER", meta["reference"])
        self.assertIn("otherPermissionUrl", meta)
        self.assertNotIn("texture", meta)
        self.assertEqual(
            g0["secondaryAnimation"], {"boneGroups": [], "colliderGroups": []}
        )

    def test_meta_options(self):
        meta = vrm.VrmMeta(
            name="Permissive",
            authors=["A", "B"],
            version="2",
            references=["https://example.com/original"],
            third_party_licenses="Outfit: CC-BY 4.0, by someone.",
            avatar_permission="everyone",
            allow_excessively_violent_usage=True,
            commercial_usage="corporation",
            allow_redistribution=True,
            modification="allowModification",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = {}
            for version in VERSIONS:
                spec = dataclasses.replace(vrm_fixtures.vrm_build(version), meta=meta)
                path = pathlib.Path(tmp) / f"meta_{version}.vrm"
                vrm.write_vrm(spec, path)
                out[version] = GLB(path).json["extensions"]
        m1 = out["1.0"]["VRMC_vrm"]["meta"]
        self.assertEqual(m1["authors"], ["A", "B"])
        self.assertEqual(m1["references"], ["https://example.com/original"])
        self.assertEqual(m1["avatarPermission"], "everyone")
        self.assertTrue(m1["thirdPartyLicenses"].startswith(vrm.BASE_CREDITS[0]))
        self.assertTrue(m1["thirdPartyLicenses"].endswith("by someone."))
        m0 = out["0.x"]["VRM"]["meta"]
        self.assertEqual(m0["author"], "A, B")
        self.assertEqual(m0["version"], "2")
        self.assertEqual(m0["allowedUserName"], "Everyone")
        self.assertEqual(m0["violentUssageName"], "Allow")
        self.assertEqual(m0["sexualUssageName"], "Disallow")
        self.assertEqual(m0["commercialUssageName"], "Allow")
        self.assertEqual(m0["licenseName"], "Other")
        self.assertEqual(m0["otherLicenseUrl"], vrm.VRM1_LICENSE_URL)
        self.assertTrue(m0["reference"].startswith("https://example.com/original"))
        self.assertIn("CC-BY", m0["reference"])

    def test_meta_needs_an_author(self):
        model = vrm_fixtures.model()
        character = vrm_fixtures.CHARACTER
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "x.vrm"
            with self.assertRaisesRegex(ValueError, "author"):
                vrm.export_vrm(path, character, model)
            with self.assertRaisesRegex(ValueError, "avatar_permission"):
                vrm.export_vrm(
                    path,
                    character,
                    model,
                    meta=vrm.VrmMeta("x", ["y"], avatar_permission="anyone"),
                )
            with self.assertRaisesRegex(ValueError, "Unknown VRM version"):
                vrm.export_vrm(path, character, model, version="2.0", author="a")
            self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, "Unknown VRM metadata"):
            vrm.VrmMeta.from_dict({"authors": ["a"], "licence": "x"})
        meta = vrm.VrmMeta.from_dict({"authors": "a", "name": "n"})
        self.assertEqual(meta.authors, ["a"])

    def test_meta_types(self):
        """
        Each field of a --meta file must have its type: the text "false" or "no" would
        grant a permission, and a number where the schema wants a string makes an invalid
        file. The error names the field.
        """
        good = vrm.VrmMeta.from_dict(
            {
                "authors": "a",
                "references": "https://example.com",
                "version": "1",
                "allow_redistribution": True,
                "commercial_usage": "corporation",
            }
        )
        self.assertEqual(good.references, ["https://example.com"])
        self.assertIs(good.allow_redistribution, True)
        bad = {
            "allow_redistribution": "false",
            "allow_excessively_sexual_usage": "no",
            "allow_excessively_violent_usage": 0,
            "allow_political_or_religious_usage": None,
            "allow_antisocial_or_hate_usage": "true",
            "name": 7,
            "version": 2,
            "copyright_information": ["x"],
            "contact_information": 5,
            "third_party_licenses": {"a": 1},
            "other_license_url": 3.0,
            "authors": ["a", ""],
            "references": ["https://example.com", 1],
            "avatar_permission": "anyone",
            "commercial_usage": "Corporation",
            "credit_notation": True,
            "modification": ["prohibited"],
        }
        for field, value in bad.items():
            with self.subTest(field=field):
                data = {"authors": ["a"], field: value}
                with self.assertRaisesRegex(ValueError, field):
                    vrm.VrmMeta.from_dict(data)
        with self.assertRaisesRegex(ValueError, "authors"):
            vrm.VrmMeta.from_dict({"authors": None})
        # A string boolean given in Python is refused before the model is built.
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                vrm.Character, "build_model", side_effect=AssertionError("built")
            ),
        ):
            path = pathlib.Path(tmp) / "x.vrm"
            meta = vrm.VrmMeta("x", ["y"], allow_redistribution="false")
            with self.assertRaisesRegex(ValueError, "allow_redistribution"):
                vrm.export_vrm(path, vrm_fixtures.CHARACTER, meta=meta)
            meta = vrm.VrmMeta("x", "y")
            with self.assertRaisesRegex(ValueError, "authors"):
                vrm.export_vrm(path, vrm_fixtures.CHARACTER, meta=meta)
            self.assertFalse(path.exists())

    def test_twist_constraint_needs_vrm1(self):
        with mock.patch.object(
            vrm.Character, "build_model", side_effect=AssertionError("built")
        ):
            with self.assertRaisesRegex(ValueError, "VRM 0.x files cannot carry roll"):
                vrm.vrm_spec(
                    vrm_fixtures.CHARACTER,
                    version="0.x",
                    twist="constraint",
                    author="a",
                )
            with self.assertRaisesRegex(ValueError, "Unknown twist mode"):
                vrm.vrm_spec(vrm_fixtures.CHARACTER, twist="spin", author="a")
        # The extras and the summary record the twist mode in use.
        for version in VERSIONS:
            with self.subTest(version=version):
                path, summary = vrm_fixtures.vrm_export(version)
                extras = GLB(path).json["scenes"][0]["extras"]["opensculptboy"]
                self.assertEqual(
                    summary["twist"], {"1.0": "constraint", "0.x": "merge"}[version]
                )
                self.assertEqual(extras["options"]["twist"], summary["twist"])

    def test_topology(self):
        """
        VRM export takes the MakeHuman body mesh with its eyes, and refuses every other
        topology before a model is built: the SMPL topologies without downloading their
        non-commercial data.
        """
        for topology in (
            "anny",
            "anny-quads",
            "anny-full",
            "anny-notongue",
            "makehuman",
            "makehuman-tris",
        ):
            with self.subTest(topology=topology):
                vrm.check_topology(topology)
        for topology, message in (
            ("smpl", "non-commercial"),
            ("smplx", "non-commercial"),
            ("notoes", "MakeHuman body mesh"),
            ("soma", "MakeHuman body mesh"),
            ("anny_from_soma", "MakeHuman body mesh"),
            ("head", "MakeHuman body mesh"),
            ("hand.L", "MakeHuman body mesh"),
            ("anny-noeyes", "MakeHuman body mesh"),
            ("default", "Unknown topology"),
            ("cube", "Unknown topology"),
            (3, "must be a string"),
        ):
            with self.subTest(topology=topology):
                with self.assertRaisesRegex(ValueError, message):
                    vrm.check_topology(topology)
        import anny.paths

        def never(*args, **kwargs):
            self.fail("the VRM export started a download or built a model")

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(anny.paths, "download_noncommercial_data", never),
            mock.patch.object(vrm.Character, "build_model", never),
        ):
            path = pathlib.Path(tmp) / "x.vrm"
            for topology, message in (
                ("smpl", "non-commercial"),
                ("smplx", "non-commercial"),
                ("notoes", "MakeHuman body mesh"),
            ):
                with self.subTest(export=topology):
                    character = vrm.Character(topology=topology)
                    with self.assertRaisesRegex(ValueError, message):
                        vrm.export_vrm(path, character, author="a")
            self.assertFalse(path.exists())

    def test_materials(self):
        g1 = vrm_fixtures.vrm_json("1.0")
        for material in g1["materials"]:
            self.assertIn("VRMC_materials_mtoon", material["extensions"])
        self.assertIn("VRMC_materials_mtoon", g1["extensionsUsed"])
        self.assertIn("VRMC_vrm", g1["extensionsUsed"])
        g0 = vrm_fixtures.vrm_json("0.x")
        self.assertIn("VRM", g0["extensionsUsed"])
        self.assertNotIn("VRMC_vrm", g0["extensionsUsed"])
        props = g0["extensions"]["VRM"]["materialProperties"]
        self.assertEqual(len(props), len(g0["materials"]))
        for prop, material in zip(props, g0["materials"]):
            self.assertEqual(prop["shader"], "VRM/MToon")
            if "name" in prop:
                self.assertEqual(prop["name"], material["name"])
        for g in (g1, g0):
            for prim in g["meshes"][0]["primitives"]:
                self.assertLess(prim["material"], len(g["materials"]))

    def test_extras_and_read_character(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                path, summary = vrm_fixtures.vrm_export(version)
                self.assertEqual(
                    read_character(path).to_dict(), vrm_fixtures.CHARACTER.to_dict()
                )
                extras = GLB(path).json["scenes"][0]["extras"]["opensculptboy"]
                self.assertEqual(extras["format"], "vrm")
                self.assertEqual(extras["vrm_version"], version)
                self.assertEqual(extras["options"]["bind"], vrm.DEFAULT_BIND)
                self.assertEqual(summary["twist"], extras["options"]["twist"])
                self.assertEqual(extras["budget"]["counts"], summary["counts"])
                self.assertEqual(extras["budget"]["messages"], [])


class TestBudget(unittest.TestCase):
    def test_counts_of_the_exports(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                path, summary = vrm_fixtures.vrm_export(version, bare=True)
                glb = GLB(path)
                found = budget.counts(glb.json, glb.bin)
                self.assertEqual(found, summary["counts"])
                self.assertEqual(found["triangles"], 27_420)  # the bare body
                self.assertEqual(found["joints"], vrm_fixtures.model().bone_count)
                self.assertEqual(found["materials"], 1)
                self.assertEqual(found["meshes"], 1)
                self.assertEqual(found["morph_targets"], 52)
                self.assertEqual(found["images"], 0)
                self.assertEqual(summary["budget"], [])
                self.assertEqual(budget.check(found, version, "strict"), [])
                options = glb.json["scenes"][0]["extras"]["opensculptboy"]["options"]
                self.assertIs(options["bare"], True)

    def test_counts_of_a_document(self):
        png = _png(300, 200)
        jpeg = _jpeg(64, 48)
        binary = png + b"\0" * (-len(png) % 4) + jpeg
        gltf = {
            "nodes": [{"mesh": 0}, {"mesh": 0}, {"mesh": 1}, {}],
            "meshes": [
                {
                    "primitives": [
                        {"attributes": {"POSITION": 0}, "indices": 1, "targets": [{}]},
                        {"attributes": {"POSITION": 0}, "mode": 5},
                    ]
                },
                {"primitives": [{"attributes": {"POSITION": 0}, "mode": 1}]},
            ],
            "accessors": [{"count": 10}, {"count": 12}],
            "skins": [{"joints": [0, 1, 3]}, {"joints": [1, 2]}],
            "materials": [{}, {}],
            "images": [
                {"bufferView": 0},
                {"bufferView": 1},
                {
                    "uri": "data:image/png;base64,"
                    + base64.b64encode(_png(5000, 10)).decode()
                },
                {"uri": "outside.png"},
            ],
            "bufferViews": [
                {"byteOffset": 0, "byteLength": len(png)},
                {"byteOffset": len(png) + (-len(png) % 4), "byteLength": len(jpeg)},
            ],
        }
        found = budget.counts(gltf, binary)
        # Two nodes draw mesh 0 (4 triangles and a strip of 8), mesh 1 draws lines.
        self.assertEqual(found["triangles"], 2 * (4 + 8))
        self.assertEqual(found["vertices"], 5 * 10)
        self.assertEqual(found["joints"], 4)
        self.assertEqual(found["materials"], 2)
        self.assertEqual(found["images"], 4)
        self.assertEqual(found["max_image_size"], 5000)
        self.assertEqual(found["morph_targets"], 1)
        self.assertEqual(found["meshes"], 2)
        self.assertEqual(budget.image_size(png), (300, 200))
        self.assertEqual(budget.image_size(jpeg), (64, 48))
        self.assertIsNone(budget.image_size(b"GIF89a"))
        self.assertEqual(budget.counts(gltf)["max_image_size"], 5000)

    def test_check(self):
        found = {
            "triangles": 40_000,
            "joints": 150,
            "materials": 3,
            "max_image_size": 4096,
        }
        self.assertEqual(budget.check(found, "vrm1"), [])
        messages = budget.check(found, "0.x", "warn")
        self.assertEqual(len(messages), 3)
        self.assertTrue(messages[0].startswith("triangles: 40,000 over the VRM 0.x"))
        self.assertEqual(budget.check(found, "vrm0", "off"), [])
        with self.assertRaises(budget.BudgetError):
            budget.check(found, "vrm0", "strict")
        with self.assertRaises(ValueError):
            budget.check(found, "vrm0", "loud")
        with self.assertRaises(ValueError):
            budget.check(found, "vrm2")

    def test_budget_modes_of_the_writer(self):
        spec = vrm_fixtures.vrm_build("1.0")
        small = dict(budget.BUDGETS["vrm1"], triangles=1000)
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(budget.BUDGETS, {"vrm1": small}),
        ):
            path = pathlib.Path(tmp) / "over.vrm"
            with self.assertRaisesRegex(budget.BudgetError, "triangles"):
                vrm.write_vrm(spec, path, budget="strict")
            self.assertFalse(path.exists())
            with self.assertWarns(budget.BudgetWarning):
                summary = vrm.write_vrm(spec, path, budget="warn")
            self.assertEqual(len(summary["budget"]), 1)
            extras = GLB(path).json["scenes"][0]["extras"]["opensculptboy"]
            self.assertEqual(extras["budget"]["messages"], summary["budget"])
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                self.assertEqual(vrm.write_vrm(spec, path, budget="off")["budget"], [])


def _png(width, height) -> bytes:
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 90)).save(stream, format="PNG")
    return stream.getvalue()


def _jpeg(width, height) -> bytes:
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 90)).save(stream, format="JPEG")
    return stream.getvalue()


class TestThumbnail(unittest.TestCase):
    def test_auto_portrait(self):
        from PIL import Image

        g = vrm_fixtures.vrm_json("1.0", thumbnail="auto")
        meta = g["extensions"]["VRMC_vrm"]["meta"]
        glb = vrm_fixtures.vrm_glb("1.0", thumbnail="auto")
        image = g["images"][meta["thumbnailImage"]]
        self.assertEqual(image["mimeType"], "image/png")
        view = g["bufferViews"][image["bufferView"]]
        data = glb.bin[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        picture = Image.open(io.BytesIO(data))
        self.assertEqual(picture.size, (1024, 1024))
        pixels = np.asarray(picture.convert("RGB"), dtype=np.float64)
        # The figure fills the middle of the picture, the background its top corners.
        self.assertGreater(np.abs(pixels[512, 512] - pixels[8, 8]).sum(), 30)
        self.assertEqual(
            g["scenes"][0]["extras"]["opensculptboy"]["options"]["thumbnail"], "auto"
        )

    def test_given_thumbnails(self):
        with tempfile.TemporaryDirectory() as tmp:
            for version, data, mime in (
                ("1.0", _jpeg(32, 32), "image/jpeg"),
                ("0.x", _png(16, 16), "image/png"),
            ):
                with self.subTest(version=version):
                    spec = dataclasses.replace(
                        vrm_fixtures.vrm_build(version), thumbnail=data
                    )
                    path = pathlib.Path(tmp) / f"thumb_{version}.vrm"
                    vrm.write_vrm(spec, path)
                    glb = GLB(path)
                    g = glb.json
                    if version == "1.0":
                        index = g["extensions"]["VRMC_vrm"]["meta"]["thumbnailImage"]
                    else:
                        texture = g["extensions"]["VRM"]["meta"]["texture"]
                        index = g["textures"][texture]["source"]
                    image = g["images"][index]
                    self.assertEqual(image["mimeType"], mime)
                    view = g["bufferViews"][image["bufferView"]]
                    start = view["byteOffset"]
                    self.assertEqual(glb.bin[start : start + view["byteLength"]], data)
                    self.assertEqual(budget.counts(g, glb.bin)["images"], 1)
            never = pathlib.Path(tmp) / "never.vrm"
            with self.assertRaisesRegex(ValueError, "PNG or JPEG"):
                vrm.export_vrm(
                    never,
                    vrm_fixtures.CHARACTER,
                    vrm_fixtures.model(),
                    author="a",
                    thumbnail=b"GIF89a",
                )
            self.assertFalse(never.exists())

    def test_thumbnails_must_be_square(self):
        """VRM 1.0 requires a square thumbnail; both versions refuse any other."""
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                vrm.Character, "build_model", side_effect=AssertionError("built")
            ),
        ):
            tmp = pathlib.Path(tmp)
            wide = tmp / "wide.png"
            wide.write_bytes(_png(64, 32))
            for version in VERSIONS:
                for thumbnail, message in (
                    (wide, "square; this one is 64 x 32"),
                    (str(wide), "square"),
                    (_jpeg(30, 40), "square; this one is 30 x 40"),
                    (b"\x89PNG\r\n\x1a\n", "size"),
                    (tmp / "missing.png", "No such file"),
                ):
                    with self.subTest(version=version, thumbnail=str(thumbnail)[:40]):
                        error = OSError if message == "No such file" else ValueError
                        with self.assertRaisesRegex(error, message):
                            vrm.export_vrm(
                                tmp / "x.vrm",
                                vrm_fixtures.CHARACTER,
                                version=version,
                                author="a",
                                thumbnail=thumbnail,
                            )
            self.assertFalse((tmp / "x.vrm").exists())
        # A spec given a thumbnail afterwards is checked when it is written.
        spec = dataclasses.replace(
            vrm_fixtures.vrm_build("0.x"), thumbnail=_jpeg(48, 32)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "wide.vrm"
            with self.assertRaisesRegex(ValueError, "square"):
                vrm.write_vrm(spec, path)
            self.assertFalse(path.exists())


class TestCommandLine(unittest.TestCase):
    def test_vrm_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            card = tmp / "card.json"
            card.write_text(json.dumps(vrm_fixtures.CHARACTER.to_dict()))
            meta = tmp / "meta.json"
            meta.write_text(json.dumps({"authors": ["Meta Author"], "version": "3"}))
            out = tmp / "cli.vrm"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli.main(
                    [
                        "export",
                        str(out),
                        "--character",
                        str(card),
                        "--meta",
                        str(meta),
                        "--name",
                        "Avatar",
                        "--vrm-version",
                        "0",
                        "--budget",
                        "strict",
                    ]
                )
            self.assertEqual(code, 0)
            text = stdout.getvalue()  # warp may print its start-up lines first
            summary = json.loads(text[text.index("{\n") :])
            self.assertEqual(summary["version"], "0.x")
            self.assertEqual(summary["triangles"], 27_420)
            g = GLB(out).json
            meta = g["extensions"]["VRM"]["meta"]
            self.assertEqual(meta["author"], "Meta Author")
            self.assertEqual(meta["title"], "Avatar")
            self.assertEqual(meta["version"], "3")
            self.assertEqual(
                read_character(out).to_dict(), vrm_fixtures.CHARACTER.to_dict()
            )

    def test_errors(self):
        """
        Usage errors end the command with exit status 2 and one error line, without a
        traceback, before any model is built.
        """
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                vrm.Character, "build_model", side_effect=AssertionError("built")
            ),
        ):
            tmp = pathlib.Path(tmp)
            files = {
                "meta.json": {"name": "no authors"},
                "enum.json": {"authors": ["a"], "commercial_usage": "free"},
                "flag.json": {"authors": ["a"], "allow_redistribution": "false"},
                "number.json": {"authors": ["a"], "version": 2},
                "rig.json": dict(vrm_fixtures.CHARACTER.to_dict(), rig="makehuman"),
                "smpl.json": dict(vrm_fixtures.CHARACTER.to_dict(), topology="smpl"),
            }
            for file, data in files.items():
                (tmp / file).write_text(json.dumps(data))
            (tmp / "broken.json").write_text("{")
            (tmp / "wide.png").write_bytes(_png(40, 20))
            vrm_file = ["export", str(tmp / "a.vrm")]
            glb_file = ["export", str(tmp / "a.glb")]
            by_a = vrm_file + ["--author", "a"]
            for argv, message in (
                (vrm_file, "needs an author"),
                (vrm_file + ["--meta", str(tmp / "meta.json")], "needs an author"),
                (by_a + ["--animation", "walk"], "--animation"),
                (glb_file + ["--author", "a"], "--author"),
                (by_a + ["--influences", "8"], "--influences"),
                (vrm_file + ["--meta", str(tmp / "enum.json")], "commercial_usage"),
                (vrm_file + ["--meta", str(tmp / "flag.json")], "allow_redistribution"),
                (vrm_file + ["--meta", str(tmp / "number.json")], "metadata version"),
                (by_a + ["--thumbnail", str(tmp / "no.png")], "No such file"),
                (by_a + ["--thumbnail", str(tmp / "wide.png")], "square"),
                (by_a + ["--character", str(tmp / "rig.json")], "'anny' rig"),
                (by_a + ["--character", str(tmp / "smpl.json")], "non-commercial"),
                (by_a + ["--character", str(tmp / "broken.json")], "--character"),
                (by_a + ["--vrm-version", "0", "--twist", "constraint"], "roll"),
                (glb_file + ["--character", str(tmp / "no.json")], "--character"),
            ):
                with self.subTest(argv=argv[2:]):
                    stderr = io.StringIO()
                    with (
                        contextlib.redirect_stderr(stderr),
                        self.assertRaises(SystemExit) as exit_,
                    ):
                        cli.main(argv)
                    self.assertEqual(exit_.exception.code, 2)
                    text = stderr.getvalue()
                    errors = [line for line in text.splitlines() if "error:" in line]
                    self.assertEqual(len(errors), 1, text)
                    self.assertIn(message, errors[0])
                    self.assertNotIn("Traceback", text)
            self.assertEqual(list(tmp.glob("a.*")), [])

    def test_failures_print_the_warnings(self):
        """
        A file over a strict budget ends the command with exit status 1 and one line; the
        warnings of the export are printed whether it succeeds or fails.
        """

        def over_budget(*args, **kwargs):
            warnings.warn("an early warning", budget.BudgetWarning)
            raise budget.BudgetError("triangles: 99,999 over the VRM 0.x budget")

        def invalid(*args, **kwargs):
            warnings.warn("an early warning", budget.BudgetWarning)
            raise ValueError("an invalid\noption")

        argv = ["export", "x.vrm", "--author", "a"]
        with mock.patch.object(vrm, "export_vrm", over_budget):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(cli.main(argv), 1)
            self.assertEqual(
                stderr.getvalue().splitlines(),
                [
                    "warning: an early warning",
                    "opensculptboy export: triangles: 99,999 over the VRM 0.x budget",
                ],
            )
        with mock.patch.object(vrm, "export_vrm", invalid):
            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as exit_,
            ):
                cli.main(argv)
            self.assertEqual(exit_.exception.code, 2)
            lines = stderr.getvalue().splitlines()
            self.assertEqual(lines[0], "warning: an early warning")
            self.assertTrue(lines[-1].endswith("error: an invalid option"), lines)

    def test_bare_exports(self):
        """
        --bare at the command line: a GLB file keeps the structure of a plain export, and a
        VRM file holds the body alone.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            for argv in (
                ["export", str(tmp / "plain.glb")],
                ["export", str(tmp / "bare.glb"), "--bare"],
                ["export", str(tmp / "bare.vrm"), "--bare", "--author", "a"],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(argv), 0)
            plain, bare = (GLB(tmp / f"{n}.glb").json for n in ("plain", "bare"))
            for key in ("nodes", "meshes", "accessors", "skins", "materials"):
                self.assertEqual(len(bare.get(key, [])), len(plain.get(key, [])), key)
            self.assertEqual(
                [n.get("name") for n in bare["nodes"]],
                [n.get("name") for n in plain["nodes"]],
            )
            self.assertEqual(
                [len(m["primitives"]) for m in bare["meshes"]],
                [len(m["primitives"]) for m in plain["meshes"]],
            )
            self.assertEqual(budget.counts(bare), budget.counts(plain))
            g = GLB(tmp / "bare.vrm").json
            found = budget.counts(g)
            self.assertEqual(found["meshes"], 1)
            self.assertEqual(found["triangles"], 27_420)
            options = g["scenes"][0]["extras"]["opensculptboy"]["options"]
            self.assertIs(options["bare"], True)


if __name__ == "__main__":
    unittest.main()
