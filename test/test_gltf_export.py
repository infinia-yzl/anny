# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the glTF export (opensculptboy.export.gltf): a NumPy reader evaluates the file as a glTF
engine does (node transforms, animation keyframes, skinning and morph targets) and compares the
result with Anny's own forward pass.

``TestGltfReference`` guards the default GLB (base model guarantee 2): it exports the characters
of ``test/data/gltf_reference.json`` again and compares each file with the structure and the
accessor statistics recorded there. The reference was recorded with the exporter of #8, before
the exporter moved onto ``GltfDocument``. A deliberate change of the GLB output records it again
by hand::

    python -c "from test.test_gltf_export import write_reference; write_reference()"
"""

import copy
import functools
import hashlib
import json
import pathlib
import tempfile
import unittest

import numpy as np
import torch

from opensculptboy import Character, export_glb, read_character
from opensculptboy.export.gltf import C3, C4
from test.gltf_reader import GLB, evaluate, node_world_matrices

REFERENCE = pathlib.Path(__file__).resolve().parent / "data" / "gltf_reference.json"
# Floats of the structure and the statistics match within this relative and absolute tolerance.
REFERENCE_RTOL, REFERENCE_ATOL = 1e-6, 1e-7


@functools.lru_cache(maxsize=None)
def float64_model(face_shapes: bool = False):
    """The float64 model that the characters of this module share (no local changes)."""
    return Character().build_model(face_shapes=face_shapes, dtype=torch.float64)


def _strip(value, keys):
    """``value`` without the given keys in any of its nested dicts."""
    if isinstance(value, dict):
        return {k: _strip(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_strip(v, keys) for v in value]
    return value


def _rounded(value, digits: int = 10):
    """``value`` with every float rounded to ``digits`` significant digits, to keep the file small."""
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    return value


def gltf_fingerprint(path) -> dict:
    """
    The structure of a GLB file and the statistics of its accessors.

    The structure is the glTF JSON without the extras of the scenes (the character card), the
    buffer views and every ``bufferView`` and ``byteOffset`` (the layout of the binary chunk),
    with the anny version of the generator replaced by ``<anny>``. It keeps the count, type and
    component type of each accessor, and the ``min`` and ``max`` of the accessors that carry
    them. The statistics give, for each accessor, the sum of each component (``sum``), and
    when it holds more than one element the sum of squares of its values (``sumsq``) and, unless
    the structure has them, their minimum and maximum (``min``, ``max``); for a sparse accessor,
    a hash of its index list (``indices``).
    """
    import anny

    glb = GLB(path)
    structure = _strip(copy.deepcopy(glb.json), ("bufferView", "byteOffset"))
    for scene in structure["scenes"]:
        scene.pop("extras", None)
    structure.pop("bufferViews", None)
    asset = structure["asset"]
    asset["generator"] = asset["generator"].replace(anny.__version__, "<anny>")
    stats = []
    for i, accessor in enumerate(glb.json["accessors"]):
        values = glb.accessor(i).astype(np.float64).reshape(accessor["count"], -1)
        entry = {"sum": values.sum(axis=0).tolist()}
        if accessor["count"] > 1:
            entry["sumsq"] = float(np.square(values).sum())
            if "min" not in accessor:
                entry.update(min=float(values.min()), max=float(values.max()))
        indices = glb.sparse_indices(i)
        if indices is not None:
            raw = np.asarray(indices, dtype="<i8").tobytes()
            entry["indices"] = hashlib.sha256(raw).hexdigest()[:16]
        stats.append(entry)
    return {"structure": structure, "accessors": stats}


def reference_cases() -> dict[str, dict]:
    """
    The characters and export options of the reference files: the default character, the test
    character of ``TestGltfExport`` with the walk at 4 and 8 influences, and a character with
    face shapes, facial actions and its own pose, exported with face-shape and facial-action
    targets, unground and in another colour.
    """
    import anny.poses

    test = Character(
        name="test",
        phenotype={"age": 0.3, "weight": 0.7},
        facial_actions={"jawOpen": 0.25, "mouthSmileLeft": 0.5},
    )
    face_model = float64_model(face_shapes=True)
    params = anny.poses.pose_parameters(face_model, "mh_thinking", grounded=False)
    posed = Character(
        name="posed face",
        phenotype={"age": 0.2, "height": 0.6, "muscle": 0.7},
        face_shapes={"head-fat": 0.3, "head-oval": 0.6, "nose-scale-horiz": 0.4},
        facial_actions={"jawOpen": 0.4, "eyeBlinkLeft": 0.2, "browInnerUp": 0.5},
        pose=Character.pose_from_parameters(face_model, params["pose_parameters"][:1]),
    )
    return {
        "default": dict(character=Character(), face_shapes=False, options={}),
        "walk_4": dict(
            character=test,
            face_shapes=False,
            options=dict(animations=["walk"], max_influences=4),
        ),
        "walk_8": dict(
            character=test,
            face_shapes=False,
            options=dict(animations=["walk"], max_influences=8),
        ),
        "posed_face": dict(
            character=posed,
            face_shapes=True,
            options=dict(
                morph_targets=[
                    "head-scale-vert",
                    "head-fat",
                    "head-oval",
                    "jawOpen",
                    "eyeBlinkLeft",
                ],
                ground=False,
                base_color=[0.5, 0.6, 0.7, 1.0],
            ),
        ),
    }


def export_case(case: dict, directory, name: str) -> pathlib.Path:
    """Export one reference case with a float64 model; returns the path of the file."""
    path = pathlib.Path(directory) / f"{name}.glb"
    export_glb(
        path, case["character"], float64_model(case["face_shapes"]), **case["options"]
    )
    return path


def write_reference(path=REFERENCE) -> None:
    """
    Record the reference files with the current exporter (run by hand). Floats keep 7
    significant digits, and a top-level part of the structure that equals the same part of an
    earlier file is stored as ``{"$same_as": <that file>}``, so that the file stays small.
    """
    files, structures = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, case in reference_cases().items():
            found = _rounded(gltf_fingerprint(export_case(case, tmp, name)), 7)
            structure = {}
            for key, value in found["structure"].items():
                same = [
                    other
                    for other, earlier in structures.items()
                    if earlier.get(key) == value
                ]
                large = len(json.dumps(value)) > 100
                structure[key] = {"$same_as": same[0]} if same and large else value
            structures[name] = found["structure"]
            files[name] = dict(
                character=case["character"].to_dict(),
                face_shapes=case["face_shapes"],
                options=json_value(case["options"]),
                structure=structure,
                accessors=found["accessors"],
            )
    reference = {
        "format": "opensculptboy/gltf-reference@1",
        "description": "GLB exports of fixed characters, recorded by write_reference in "
        "test/test_gltf_export.py: the glTF JSON without the scene extras and the binary "
        "layout, and the statistics of each accessor (see gltf_fingerprint). "
        "TestGltfReference exports the characters again and compares.",
        "files": files,
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, separators=(",", ":")) + "\n")


def recorded_structure(files: dict, name: str) -> dict:
    """The structure of a recorded file, with its ``$same_as`` parts resolved."""
    structure = {}
    for key, value in files[name]["structure"].items():
        while isinstance(value, dict) and "$same_as" in value:
            value = files[value["$same_as"]]["structure"][key]
        structure[key] = value
    return structure


def assert_close(test: unittest.TestCase, found, expected, where: str = "") -> None:
    """Equal JSON values, with floats within the reference tolerance."""
    if isinstance(expected, dict):
        test.assertIsInstance(found, dict, where)
        test.assertEqual(sorted(found), sorted(expected), where)
        for key in expected:
            assert_close(test, found[key], expected[key], f"{where}/{key}")
    elif isinstance(expected, list):
        test.assertIsInstance(found, list, where)
        test.assertEqual(len(found), len(expected), where)
        for i, (f, e) in enumerate(zip(found, expected)):
            assert_close(test, f, e, f"{where}/{i}")
    elif isinstance(expected, bool) or not isinstance(expected, (int, float)):
        test.assertEqual(found, expected, where)
    elif isinstance(expected, int) and isinstance(found, int):
        test.assertEqual(found, expected, where)
    else:
        test.assertNotIsInstance(found, (bool, str, list, dict, type(None)), where)
        tolerance = REFERENCE_ATOL + REFERENCE_RTOL * abs(expected)
        test.assertLessEqual(abs(found - expected), tolerance, where)


class TestGltfExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.character = Character(
            name="test",
            phenotype={"age": 0.3, "weight": 0.7},
            facial_actions={"jawOpen": 0.25, "mouthSmileLeft": 0.5},
        )
        cls.model = float64_model()
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
        model = float64_model(face_shapes=True)
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

        model = float64_model()
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
        model = float64_model()
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


class TestGltfReference(unittest.TestCase):
    """Base model guarantee 2: the default GLB keeps the output of the exporter of #8."""

    def test_exports_match_the_reference(self):
        files = json.loads(REFERENCE.read_text())["files"]
        self.assertEqual(sorted(files), ["default", "posed_face", "walk_4", "walk_8"])
        with tempfile.TemporaryDirectory() as tmp:
            for name, recorded in files.items():
                with self.subTest(name):
                    case = dict(
                        character=Character.from_dict(recorded["character"]),
                        face_shapes=recorded["face_shapes"],
                        options=recorded["options"],
                    )
                    found = gltf_fingerprint(export_case(case, tmp, name))
                    expected = recorded_structure(files, name)
                    assert_close(self, found["structure"], expected, name)
                    assert_close(self, found["accessors"], recorded["accessors"], name)

    def test_reference_is_small(self):
        self.assertLess(REFERENCE.stat().st_size, 200_000)


def json_value(value):
    """``value`` as JSON reads it back (tuples become lists)."""
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
