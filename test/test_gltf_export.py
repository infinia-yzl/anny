# Corporis
# Apache License, Version 2.0
"""
Checks of the glTF export (corporis.export.gltf): a NumPy reader evaluates the file as a glTF
engine does (node transforms, animation keyframes, skinning and morph targets) and compares the
result with Anny's own forward pass.
"""

import json
import pathlib
import struct
import tempfile
import unittest

import numpy as np
import torch

from corporis import Character, export_glb, read_character
from corporis.export.gltf import C3, C4

_DTYPES = {5126: np.float32, 5121: np.uint8, 5123: np.uint16, 5125: np.uint32}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class GLB:
    """A minimal glTF 2.0 binary reader."""

    def __init__(self, path):
        data = pathlib.Path(path).read_bytes()
        magic, version, total = struct.unpack_from("<III", data, 0)
        assert magic == 0x46546C67 and version == 2 and total == len(data)
        length, _ = struct.unpack_from("<II", data, 12)
        self.json = json.loads(data[20 : 20 + length])
        offset = 20 + length
        bin_length, _ = struct.unpack_from("<II", data, offset)
        self.bin = data[offset + 8 : offset + 8 + bin_length]

    def _view(self, index, dtype, count, width, byte_offset=0):
        view = self.json["bufferViews"][index]
        start = view.get("byteOffset", 0) + byte_offset
        array = np.frombuffer(self.bin, dtype=dtype, count=count * width, offset=start)
        return array.reshape(count, width) if width > 1 else array

    def accessor(self, index):
        a = self.json["accessors"][index]
        dtype, width = _DTYPES[a["componentType"]], _WIDTHS[a["type"]]
        if "bufferView" in a:
            array = self._view(
                a["bufferView"], dtype, a["count"], width, a.get("byteOffset", 0)
            ).copy()
        else:
            array = np.zeros(
                (a["count"], width) if width > 1 else a["count"], dtype=dtype
            )
        if "sparse" in a:
            s = a["sparse"]
            idx = self._view(
                s["indices"]["bufferView"],
                _DTYPES[s["indices"]["componentType"]],
                s["count"],
                1,
            )
            array[idx] = self._view(s["values"]["bufferView"], dtype, s["count"], width)
        return array


def trs_matrix(translation, rotation):
    x, y, z, w = rotation
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = translation
    return m


def node_world_matrices(glb, animation=None, frame=0):
    """World matrices of the nodes in the default pose, or in one keyframe of an animation."""
    g = glb.json
    nodes = [dict(n) for n in g["nodes"]]
    if animation is not None:
        anim = next(a for a in g["animations"] if a["name"] == animation)
        for channel in anim["channels"]:
            sampler = anim["samplers"][channel["sampler"]]
            nodes[channel["target"]["node"]][channel["target"]["path"]] = glb.accessor(
                sampler["output"]
            )[frame]
    world = [None] * len(nodes)

    def visit(i, parent):
        n = nodes[i]
        world[i] = parent @ trs_matrix(
            n.get("translation", [0, 0, 0]), n.get("rotation", [0, 0, 0, 1])
        )
        for c in n.get("children", []):
            visit(c, world[i])

    for root in g["scenes"][0]["nodes"]:
        visit(root, np.eye(4))
    return world


def evaluate(glb, animation=None, frame=0, weights=None):
    """Skinned glTF vertices of the default pose, or of one keyframe of an animation."""
    g = glb.json
    world = node_world_matrices(glb, animation, frame)
    skin = g["skins"][0]
    ibm = glb.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    joint_matrices = np.stack([world[j] for j in skin["joints"]]) @ ibm
    mesh = g["meshes"][0]
    prim = mesh["primitives"][0]
    attrs = prim["attributes"]
    positions = glb.accessor(attrs["POSITION"]).astype(np.float64)
    weights = mesh.get("weights", []) if weights is None else weights
    for target, w in zip(prim.get("targets", []), weights):
        positions = positions + w * glb.accessor(target["POSITION"])
    out = np.zeros_like(positions)
    s = 0
    while f"JOINTS_{s}" in attrs:
        joints = glb.accessor(attrs[f"JOINTS_{s}"]).astype(int)
        w = glb.accessor(attrs[f"WEIGHTS_{s}"]).astype(np.float64)
        for k in range(4):
            m = joint_matrices[joints[:, k]]
            out += w[:, k, None] * (
                np.einsum("vij,vj->vi", m[:, :3, :3], positions) + m[:, :3, 3]
            )
        s += 1
    return out, glb.accessor(attrs["_ANNY_VERTEX"]).astype(int)


class TestGltfExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.character = Character(
            name="test",
            phenotype={"age": 0.3, "weight": 0.7},
            facial_actions={"jawOpen": 0.25, "mouthSmileLeft": 0.5},
        )
        cls.model = cls.character.build_model(dtype=torch.float64)
        cls.paths = {}
        cls.summaries = {}
        for k in (4, 8):
            path = pathlib.Path(cls.tmp.name) / f"test_{k}.glb"
            cls.summaries[k] = export_glb(
                path, cls.character, cls.model, animations=["walk"], max_influences=k
            )
            cls.paths[k] = path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def reference(self, pose_parameters=None, max_influences=None):
        """Anny's vertices in glTF's frame; with ``max_influences``, skinned with the file's weights."""
        with torch.no_grad():
            out = self.model(
                pose_parameters=pose_parameters,
                pose_parameterization="local-ref",
                **self.character.model_kwargs(),
            )
        if max_influences is None:
            return out["vertices"].numpy() @ C3.T
        glb = GLB(self.paths[max_influences])
        attrs = glb.json["meshes"][0]["primitives"][0]["attributes"]
        source = glb.accessor(attrs["_ANNY_VERTEX"]).astype(int)
        rest = out["rest_vertices"][0].numpy()[source]
        transforms = out["bone_poses"].numpy() @ np.linalg.inv(
            out["rest_bone_poses"].numpy()
        )
        result = []
        for f in range(transforms.shape[0]):
            v = np.zeros_like(rest)
            for s in range(max_influences // 4):
                joints = glb.accessor(attrs[f"JOINTS_{s}"]).astype(int)
                w = glb.accessor(attrs[f"WEIGHTS_{s}"]).astype(np.float64)
                for k in range(4):
                    m = transforms[f][joints[:, k]]
                    v += w[:, k, None] * (
                        np.einsum("vij,vj->vi", m[:, :3, :3], rest) + m[:, :3, 3]
                    )
            result.append(v @ C3.T)
        return np.stack(result)

    def test_rest_pose_matches_anny(self):
        glb = GLB(self.paths[4])
        vertices, source = evaluate(glb)
        with torch.no_grad():
            rest = self.model(**self.character.model_kwargs())["rest_vertices"][
                0
            ].numpy()
        expected = (rest @ C3.T)[source]
        expected[:, 1] -= expected[:, 1].min()  # the rest pose stands on the floor
        self.assertLess(np.abs(vertices - expected).max(), 1e-5)

    def test_animation_frames_match_anny(self):
        import anny.poses

        entry = anny.poses.pose_parameters(
            self.model, "walk", phenotype_kwargs=self.character.phenotype
        )
        frames = entry["pose_parameters"]
        for k in (4, 8):
            glb = GLB(self.paths[k])
            expected = self.reference(frames, max_influences=k)
            full = self.reference(frames)
            for f in (0, frames.shape[0] // 2, frames.shape[0] - 1):
                vertices, source = evaluate(glb, "walk", f)
                # The file's maths: same skinning weights, so only float32 rounding differs.
                self.assertLess(
                    np.abs(vertices - expected[f]).max(),
                    1e-4,
                    f"frame {f}, {k} influences",
                )
                # Against Anny's full skinning (up to 9 bones per vertex).
                error = np.linalg.norm(vertices - full[f][source], axis=1)
                self.assertLess(
                    error.max(), 0.05 if k == 4 else 0.005, f"frame {f}, {k} influences"
                )

    def test_morph_targets_are_exact(self):
        glb = GLB(self.paths[4])
        names = glb.json["meshes"][0]["extras"]["targetNames"]
        self.assertEqual(names, list(self.model.facial_action_labels))
        weights = np.zeros(len(names))
        for name, value in {
            "jawOpen": 1.0,
            "eyeBlinkLeft": 0.6,
            "mouthFunnel": 0.3,
        }.items():
            weights[names.index(name)] = value
        vertices, source = evaluate(glb, weights=weights)
        values = dict(zip(names, weights.tolist()))
        with torch.no_grad():
            rest = self.model(
                phenotype_kwargs=self.character.phenotype, facial_actions=values
            )["rest_vertices"][0].numpy()
        expected = (rest @ C3.T)[source]
        # The file lifts the rest pose of the character (with its default weights) onto the floor.
        with torch.no_grad():
            base = self.model(**self.character.model_kwargs())["rest_vertices"][
                0
            ].numpy()
        expected[:, 1] -= (base @ C3.T)[:, 1].min()
        self.assertLess(np.abs(vertices - expected).max(), 1e-5)

    def test_default_weights_and_round_trip(self):
        glb = GLB(self.paths[4])
        mesh = glb.json["meshes"][0]
        weights = dict(zip(mesh["extras"]["targetNames"], mesh["weights"]))
        self.assertEqual(weights["jawOpen"], 0.25)
        self.assertEqual(weights["mouthSmileLeft"], 0.5)
        self.assertEqual(read_character(self.paths[4]), self.character)

    def test_skin_weights_sum_to_one(self):
        for k in (4, 8):
            glb = GLB(self.paths[k])
            attrs = glb.json["meshes"][0]["primitives"][0]["attributes"]
            total = sum(
                glb.accessor(attrs[f"WEIGHTS_{s}"]).sum(axis=1) for s in range(k // 4)
            )
            self.assertLess(np.abs(total - 1).max(), 1e-6)

    def test_frame_is_y_up_facing_z(self):
        glb = GLB(self.paths[4])
        vertices, _ = evaluate(glb)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        self.assertEqual(int(np.argmax(extent)), 1)  # the height runs along Y
        self.assertAlmostEqual(
            float(vertices[:, 1].min()), 0.0, places=5
        )  # on the floor
        world = node_world_matrices(glb)
        names = [n["name"] for n in glb.json["nodes"]]
        for side in ("L", "R"):
            foot = world[names.index(f"foot.{side}")][:3, 3]
            toe = world[names.index(f"toe3-1.{side}")][:3, 3]
            self.assertGreater(toe[2] - foot[2], 0.05)  # the toes point toward +Z

    def test_unknown_names_suggest_close_ones(self):
        with self.assertRaisesRegex(ValueError, "jawOpen"):
            export_glb(
                pathlib.Path(self.tmp.name) / "x.glb",
                self.character,
                self.model,
                morph_targets=["jawOpn"],
            )
        with self.assertRaisesRegex(ValueError, "walk"):
            export_glb(
                pathlib.Path(self.tmp.name) / "x.glb",
                self.character,
                self.model,
                animations=["wlk"],
            )

    def test_c4_is_a_rotation(self):
        self.assertTrue(np.allclose(C4 @ C4.T, np.eye(4)))
        self.assertAlmostEqual(np.linalg.det(C3), 1.0)


class TestGltfFaceShapes(unittest.TestCase):
    def test_face_shape_targets_are_exact_and_sparse(self):
        character = Character(phenotype={"age": 0.2}, face_shapes={})
        model = character.build_model(face_shapes=True, dtype=torch.float64)
        names = [
            n for n in model.face_shape_labels if model.face_shape_ranges[n][0] < 0
        ][:3]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "face.glb"
            export_glb(path, character, model, morph_targets=names)
            glb = GLB(path)
            target_names = glb.json["meshes"][0]["extras"]["targetNames"]
            self.assertEqual(
                target_names, [f"{n}.{s}" for n in names for s in ("pos", "neg")]
            )
            values = {names[0]: 0.7, names[1]: -0.5, names[2]: 1.0}
            weights = np.zeros(len(target_names))
            for name, value in values.items():
                weights[
                    target_names.index(f"{name}.{'pos' if value > 0 else 'neg'}")
                ] = abs(value)
            vertices, source = evaluate(glb, weights=weights)
            accessors = glb.json["accessors"]
            for target in glb.json["meshes"][0]["primitives"][0]["targets"]:
                self.assertIn("sparse", accessors[target["POSITION"]])
        with torch.no_grad():
            rest = model(
                phenotype_kwargs=character.phenotype, face_shape_kwargs=values
            )["rest_vertices"][0].numpy()
            base = model(phenotype_kwargs=character.phenotype)["rest_vertices"][
                0
            ].numpy()
        expected = (rest @ C3.T)[source]
        expected[:, 1] -= (base @ C3.T)[:, 1].min()
        self.assertLess(np.abs(vertices - expected).max(), 1e-5)


class TestGltfCharacterPose(unittest.TestCase):
    def test_pose_card_round_trip(self):
        import anny.poses

        model = Character().build_model(dtype=torch.float64)
        params = anny.poses.pose_parameters(model, "mh_cheer")["pose_parameters"][:1]
        pose = Character.pose_from_parameters(model, params)
        again = Character(pose=pose).pose_parameters(model)
        self.assertLess((again - params).abs().max().item(), 1e-6)
        card = Character.from_dict(
            json.loads(json.dumps(Character(pose=pose).to_dict()))
        )
        self.assertEqual(card.pose, pose)
        self.assertIsNone(Character().pose_parameters(model))

    def test_pose_animation_matches_anny(self):
        import anny.poses

        character = Character(name="posed", phenotype={"height": 0.7})
        model = character.build_model(dtype=torch.float64)
        params = anny.poses.pose_parameters(model, "mh_thinking", grounded=False)
        character.pose = Character.pose_from_parameters(
            model, params["pose_parameters"][:1]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "posed.glb"
            summary = export_glb(path, character, model, animations=["walk"])
            glb = GLB(path)
            self.assertEqual(summary["animations"], 2)
            self.assertEqual(
                [a["name"] for a in glb.json["animations"]], ["pose", "walk"]
            )
            vertices, source = evaluate(glb, "pose", 0)
            self.assertEqual(read_character(path), character)
        grounded, _ = anny.poses.ground(
            model,
            character.pose_parameters(model),
            phenotype_kwargs=character.phenotype,
        )
        with torch.no_grad():
            out = model(
                pose_parameters=grounded,
                pose_parameterization="local-ref",
                **character.model_kwargs(),
            )
        expected = (out["vertices"][0].numpy() @ C3.T)[source]
        # four skin weights per vertex against Anny's full skinning
        error = np.linalg.norm(vertices - expected, axis=1)
        self.assertLess(error.max(), 0.05)
        self.assertLess(np.median(error), 1e-3)


if __name__ == "__main__":
    unittest.main()
