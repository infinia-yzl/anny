# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the glTF export (opensculptboy.export.gltf): a NumPy reader evaluates the file as a glTF
engine does (node transforms, animation keyframes, skinning and morph targets) and compares the
result with Anny's own forward pass.

``TestGltfReference`` guards the default GLB (base model guarantee 2): it exports the characters
of ``test/data/gltf_reference.json`` again and compares each file with the structure and the
accessor fingerprints recorded there (see ``gltf_fingerprint``). The reference was recorded with
the exporter on ``GltfDocument``, which writes the same bytes as the exporter of #8 for every
case but one: the buffer views of dense morph targets now name ``ARRAY_BUFFER`` (34962), as the
Khronos validator asks, so the ``head_dense`` file differs from the output of #8 in those
``target`` fields alone. A deliberate change of the GLB output records the reference again by
hand::

    uv run python -c "from test.test_gltf_export import write_reference; write_reference()"
"""

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
from test.gltf_reader import (
    GLB,
    evaluate,
    evaluate_nodes,
    node_world_matrices,
    structure_errors,
)

REFERENCE = pathlib.Path(__file__).resolve().parent / "data" / "gltf_reference.json"
REFERENCE_FORMAT = "opensculptboy/gltf-reference@2"
# The files of the reference, in the order of reference_cases.
REFERENCE_CASES = (
    "default",
    "walk_4",
    "test_8",
    "posed_face",
    "float32",
    "all_phenotypes",
    "head_dense",
)
# Floats of the structure and the fingerprints match within this relative and absolute
# tolerance. A float64 model gives the same float32 values on every machine: the rounding noise
# of its BLAS kernels (about 1e-16) stays far below the precision of the file.
REFERENCE_RTOL, REFERENCE_ATOL = 1e-6, 1e-7
# The tolerances of a case exported with the float32 model that export_glb builds, as the
# command does. Its forward pass rounds differently with other BLAS kernels (other CPUs): by up
# to 2e-7 m on a vertex and 1e-4 on the normal of a small triangle. The floats of the structure
# match within FLOAT32_RTOL and FLOAT32_ATOL. A fingerprint matches within FLOAT32_SCALE_RTOL
# times the sum of the absolute values of its terms, the usual bound of the rounding error of a
# sum, because a morph-target offset carries the rounding noise of whole positions. Between
# exports with the AVX-512, AVX2 and SSE kernels of MKL, and against a float64 export, the
# floats of the structure move by at most a fifth of their tolerance and the fingerprints by
# at most 7e-5 of their scale.
FLOAT32_RTOL, FLOAT32_ATOL, FLOAT32_SCALE_RTOL = 1e-4, 1e-6, 1e-3


@functools.lru_cache(maxsize=None)
def float64_model(face_shapes: bool = False):
    """The float64 model of the default settings (no local changes) that most tests share."""
    return Character().build_model(face_shapes=face_shapes, dtype=torch.float64)


def case_model(case: dict):
    """
    The model of a reference case: None when the case names ``model="exporter"`` (export_glb
    builds its own float32 model), else a float64 model of the character's settings, shared with
    the other tests for the default settings.
    """
    if case.get("model", "float64") == "exporter":
        return None
    character, default = case["character"], Character()
    settings = ("rig", "topology", "phenotypes")
    if not character.local_changes and all(
        getattr(character, key) == getattr(default, key) for key in settings
    ):
        return float64_model(case["face_shapes"])
    return character.build_model(face_shapes=case["face_shapes"], dtype=torch.float64)


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


def _probe(count: int) -> np.ndarray:
    """
    A fixed pseudo-random vector of ``count`` values in [-1, 1): SplitMix64 of the row numbers,
    the same on every machine and with every NumPy version.
    """
    x = np.arange(1, count + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * 2.0**-52 - 1.0


def _sha256(array: np.ndarray) -> str:
    """The SHA-256 of the little-endian bytes of an array."""
    array = np.ascontiguousarray(array)
    return hashlib.sha256(
        array.astype(array.dtype.newbyteorder("<")).tobytes()
    ).hexdigest()


def _replace_text(value, old: str, new: str):
    """``value`` with ``old`` replaced by ``new`` in every string it holds."""
    if isinstance(value, dict):
        return {k: _replace_text(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_text(v, old, new) for v in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def exact_accessors(gltf: dict) -> set[int]:
    """
    The accessors that a fingerprint compares byte for byte: those with integer components
    (indices, ``JOINTS_n``) and the ``_ANNY_VERTEX`` attributes, whose floats hold integers.
    """
    exact = {
        i
        for i, accessor in enumerate(gltf.get("accessors", []))
        if accessor["componentType"] != 5126 and not accessor.get("normalized")
    }
    for mesh in gltf.get("meshes", []):
        for primitive in mesh["primitives"]:
            if "_ANNY_VERTEX" in primitive["attributes"]:
                exact.add(primitive["attributes"]["_ANNY_VERTEX"])
    return exact


def gltf_fingerprint(path) -> dict:
    """
    The structure of a GLB file (a path, or the bytes of the file) and the fingerprints of its
    accessors.

    The structure is the glTF JSON without the binary layout (the ``byteOffset`` of the buffer
    views and accessors, and the ``byteLength`` of the buffer views) and without the character
    card in the scene extras, with the anny version replaced by ``<anny>``. It keeps the buffer
    views with their ``target``, every ``bufferView`` index, the count, type and component type
    of each accessor, and the ``min`` and ``max`` of the accessors that carry them.

    The fingerprint of an accessor with integer values (indices, ``JOINTS_n``, ``_ANNY_VERTEX``)
    is the SHA-256 of its bytes (``sha256``). For float values it holds, per component, the dot
    product of the values with a fixed pseudo-random vector (``probe``) and their sum weighted
    by the row number (``moment``): both change when rows move, unlike plain sums. A sparse
    accessor adds the SHA-256 of its index list (``indices``).
    """
    import anny

    glb = GLB(path)
    structure = _replace_text(glb.json, anny.__version__, "<anny>")  # a copy
    structure["bufferViews"] = _strip(
        structure.get("bufferViews", []), ("byteOffset", "byteLength")
    )
    structure["accessors"] = _strip(structure.get("accessors", []), ("byteOffset",))
    for scene in structure["scenes"]:
        entry = scene.get("extras", {}).get("opensculptboy")
        if isinstance(entry, dict):
            entry.pop("character", None)
    exact = exact_accessors(glb.json)
    fingerprints = []
    for i, accessor in enumerate(glb.json["accessors"]):
        values = glb.accessor(i)
        if i in exact:
            entry = {"sha256": _sha256(values)}
        else:
            values = values.astype(np.float64).reshape(accessor["count"], -1)
            weights = _row_weights(accessor["count"])
            entry = {key: (w @ values).tolist() for key, w in weights.items()}
        indices = glb.sparse_indices(i)
        if indices is not None:
            entry["indices"] = _sha256(indices)
        fingerprints.append(entry)
    return {"structure": structure, "accessors": fingerprints}


def _row_weights(count: int) -> dict[str, np.ndarray]:
    """The weights of the rows in the float fingerprints of an accessor of ``count`` rows."""
    return {"probe": _probe(count), "moment": np.arange(count, dtype=np.float64)}


def fingerprint_scales(path) -> list[dict | None]:
    """
    For each accessor of a GLB file, the scale of each float fingerprint: the sum of the
    absolute values of its terms, which bounds its rounding error; None for the accessors that
    are compared byte for byte.
    """
    glb = GLB(path)
    exact = exact_accessors(glb.json)
    scales = []
    for i, accessor in enumerate(glb.json["accessors"]):
        if i in exact:
            scales.append(None)
            continue
        values = np.abs(glb.accessor(i).astype(np.float64)).reshape(
            accessor["count"], -1
        )
        weights = _row_weights(accessor["count"])
        scales.append({key: np.abs(w) @ values for key, w in weights.items()})
    return scales


def reference_cases() -> dict[str, dict]:
    """
    The characters and export options of the reference files.

    - ``default``: the default character.
    - ``walk_4``: the test character of ``TestGltfExport`` with the walk, at 4 influences.
    - ``test_8``: the same character at 8 influences (``TestGltfExport`` checks its walk).
    - ``posed_face``: a character with face shapes, facial actions and its own pose, exported
      with face-shape and facial-action targets, off the floor and in another colour.
    - ``float32``: a character exported with the float32 model that ``export_glb`` builds, as
      the ``opensculptboy export`` command does.
    - ``all_phenotypes``: a character with ``phenotypes="all"`` (the race, cup-size and
      firmness phenotypes) and local changes, at 8 influences.
    - ``head_dense``: the head part model with its default facial-action targets; the targets
      that move 40 % of its vertices or more are dense.

    ``model`` names the model: ``"float64"`` for a float64 model of the character's settings,
    or ``"exporter"`` for the model that ``export_glb`` builds itself.
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
    float32 = Character(
        name="float32",
        phenotype={"gender": 0.8, "muscle": 0.6, "proportions": 0.3},
        facial_actions={"mouthSmileRight": 0.3, "browDownLeft": 0.6},
    )
    all_phenotypes = Character(
        name="all phenotypes",
        phenotypes="all",
        phenotype={
            "gender": 0.3,
            "age": 0.6,
            "weight": 0.4,
            "cupsize": 0.8,
            "firmness": 0.3,
            "african": 0.6,
            "asian": 0.3,
            "caucasian": 0.1,
        },
        local_changes={
            "l-upperarm-fat-incr": 0.6,
            "r-lowerleg-muscle-incr": -0.4,
            "measure-upperarm-length-incr": 0.5,
            "nose-hump-incr": 0.7,
            "torso-vshape-incr": -0.5,
            "hip-scale-horiz-incr": 0.3,
        },
        facial_actions={"jawOpen": 0.2},
    )
    head = Character(name="head", rig="makehuman-head", topology="head")
    return {
        "default": dict(character=Character(), face_shapes=False, options={}),
        "walk_4": dict(
            character=test,
            face_shapes=False,
            options=dict(animations=["walk"], max_influences=4),
        ),
        "test_8": dict(
            character=test, face_shapes=False, options=dict(max_influences=8)
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
        "float32": dict(
            character=float32, face_shapes=False, model="exporter", options={}
        ),
        "all_phenotypes": dict(
            character=all_phenotypes,
            face_shapes=False,
            options=dict(morph_targets=["jawOpen", "eyeBlinkRight"], max_influences=8),
        ),
        "head_dense": dict(character=head, face_shapes=False, options={}),
    }


def export_case(case: dict, directory, name: str) -> pathlib.Path:
    """Export one reference case with its model (see case_model); returns the path of the file."""
    path = pathlib.Path(directory) / f"{name}.glb"
    export_glb(path, case["character"], case_model(case), **case["options"])
    return path


def write_reference(path=REFERENCE) -> None:
    """
    Record the reference files with the current exporter (run by hand). The floats of the
    structure keep 7 significant digits and those of the fingerprints 9, whose rounding then
    takes a small part of the tolerance. So that the file stays small, a top-level part of the
    structure that equals the same part of an earlier file is stored as ``{"$same_as": <that
    file>}``, and an accessor fingerprint that equals one of an earlier file as
    ``{"$same_as": [<that file>, <its accessor>]}``.
    """
    files, structures, earlier_accessors = {}, {}, {}
    cases = reference_cases()
    assert tuple(cases) == REFERENCE_CASES, tuple(cases)
    with tempfile.TemporaryDirectory() as tmp:
        for name, case in cases.items():
            found = gltf_fingerprint(export_case(case, tmp, name))
            found = dict(
                structure=_rounded(found["structure"], 7),
                accessors=_rounded(found["accessors"], 9),
            )
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
            accessors = []
            for i, entry in enumerate(found["accessors"]):
                text = json.dumps(entry, separators=(",", ":"))
                same = earlier_accessors.get(text)
                accessors.append({"$same_as": same} if same else entry)
                earlier_accessors.setdefault(text, [name, i])
            files[name] = dict(
                character=case["character"].to_dict(),
                face_shapes=case["face_shapes"],
                model=case.get("model", "float64"),
                options=json_value(case["options"]),
                structure=structure,
                accessors=accessors,
            )
    reference = {
        "format": REFERENCE_FORMAT,
        "description": "GLB exports of fixed characters, recorded by write_reference in "
        "test/test_gltf_export.py: the glTF JSON without the binary layout and the character "
        "card, and the fingerprint of each accessor (see gltf_fingerprint). "
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


def recorded_accessors(files: dict, name: str) -> list[dict]:
    """The accessor fingerprints of a recorded file, with their ``$same_as`` entries resolved."""
    accessors = []
    for entry in files[name]["accessors"]:
        while "$same_as" in entry:
            other, index = entry["$same_as"]
            entry = files[other]["accessors"][index]
        accessors.append(entry)
    return accessors


def assert_close(
    test: unittest.TestCase,
    found,
    expected,
    where: str = "",
    rtol: float = REFERENCE_RTOL,
    atol: float = REFERENCE_ATOL,
) -> None:
    """Equal JSON values, with floats within a relative and an absolute tolerance."""
    if isinstance(expected, dict):
        test.assertIsInstance(found, dict, where)
        test.assertEqual(sorted(found), sorted(expected), where)
        for key in expected:
            assert_close(test, found[key], expected[key], f"{where}/{key}", rtol, atol)
    elif isinstance(expected, list):
        test.assertIsInstance(found, list, where)
        test.assertEqual(len(found), len(expected), where)
        for i, (f, e) in enumerate(zip(found, expected)):
            assert_close(test, f, e, f"{where}/{i}", rtol, atol)
    elif isinstance(expected, bool) or not isinstance(expected, (int, float)):
        test.assertEqual(found, expected, where)
    elif isinstance(expected, int) and isinstance(found, int):
        test.assertEqual(found, expected, where)
    else:
        test.assertNotIsInstance(found, (bool, str, list, dict, type(None)), where)
        tolerance = atol + rtol * abs(expected)
        test.assertLessEqual(abs(found - expected), tolerance, where)


def corner_uv_indices(model, triangles: np.ndarray) -> np.ndarray:
    """
    The UV index of each corner of triangles (T, 3) given by Anny vertices, from the model's
    faces and UV faces, or -1 where no face holds the corner. A corner lies in the face that
    holds the directed edge from it to the next corner, or else the edge from the previous
    corner to it: a triangle cut from a quad has one edge on the diagonal, and two on the quad.
    """
    faces = model.faces.cpu().numpy()
    uv_faces = model.face_texture_coordinate_indices.cpu().numpy()
    n = faces.shape[1]
    edges = {}
    for f, face in enumerate(faces.tolist()):
        for p in range(n):
            edges[face[p], face[(p + 1) % n]] = (f, p)
    result = np.full(triangles.shape, -1)
    for t, corners in enumerate(triangles.tolist()):
        for k in range(3):
            outgoing = edges.get((corners[k], corners[(k + 1) % 3]))
            incoming = edges.get((corners[k - 1], corners[k]))
            if outgoing is not None:
                result[t, k] = uv_faces[outgoing[0], outgoing[1]]
            elif incoming is not None:
                result[t, k] = uv_faces[incoming[0], (incoming[1] + 1) % n]
    return result


def assert_fingerprints_close(
    test: unittest.TestCase,
    found: list[dict],
    expected: list[dict],
    where: str,
    rtol: float = REFERENCE_RTOL,
    atol: float = REFERENCE_ATOL,
    scales: list[dict | None] | None = None,
) -> None:
    """
    Equal accessor fingerprints: the same hashes, and floats within ``atol`` plus ``rtol`` times
    the expected value, or times its scale when ``scales`` are given (fingerprint_scales).
    """
    test.assertEqual(len(found), len(expected), where)
    for i, (f, e) in enumerate(zip(found, expected)):
        here = f"{where}/accessors/{i}"
        test.assertEqual(sorted(f), sorted(e), here)
        for key, value in e.items():
            if isinstance(value, str):
                test.assertEqual(f[key], value, f"{here}/{key}")
                continue
            bound = np.abs(value) if scales is None else scales[i][key]
            tolerance = atol + rtol * np.asarray(bound)
            difference = np.abs(np.subtract(f[key], value))
            test.assertTrue(
                np.all(difference <= tolerance),
                f"{here}/{key}: found {f[key]}, recorded {value}, "
                f"tolerance {tolerance.tolist()}",
            )


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
        # a VRM file is a GLB container too
        vrm = pathlib.Path(self.tmp.name) / "test.vrm"
        vrm.write_bytes(self.paths[4].read_bytes())
        self.assertEqual(read_character(vrm), self.character)

    def test_files_are_sound(self):
        for k in (4, 8):
            glb = GLB(self.paths[k])
            self.assertEqual(structure_errors(glb), [], f"{k} influences")
            self.assertEqual(self.summaries[k]["bytes"], self.paths[k].stat().st_size)
            # one skinned mesh node with one primitive: evaluate_nodes agrees with evaluate
            nodes = evaluate_nodes(glb, "walk", 3)
            self.assertEqual(len(nodes), 1)
            (primitive,) = next(iter(nodes.values()))
            vertices, source = evaluate(glb, "walk", 3)
            np.testing.assert_array_equal(primitive.positions, vertices)
            np.testing.assert_array_equal(primitive.anny_vertex, source)

    def test_skin_weights_sum_to_one(self):
        for k in (4, 8):
            glb = GLB(self.paths[k])
            attrs = glb.json["meshes"][0]["primitives"][0]["attributes"]
            total = sum(
                glb.accessor(attrs[f"WEIGHTS_{s}"]).sum(axis=1) for s in range(k // 4)
            )
            self.assertLess(np.abs(total - 1).max(), 1e-6)

    def test_normals_and_uvs_follow_the_triangles(self):
        glb = GLB(self.paths[4])
        primitive = glb.json["meshes"][0]["primitives"][0]
        attrs = primitive["attributes"]
        positions = glb.accessor(attrs["POSITION"]).astype(np.float64)
        normals = glb.accessor(attrs["NORMAL"]).astype(np.float64)
        uvs = glb.accessor(attrs["TEXCOORD_0"]).astype(np.float64)
        source = glb.accessor(attrs["_ANNY_VERTEX"]).astype(int)
        triangles = glb.accessor(primitive["indices"]).astype(int).reshape(-1, 3)
        self.assertEqual(len(np.unique(triangles)), len(positions))
        # glTF's front faces wind counter-clockwise: the signed volume of the body is positive
        a, b, c = (positions[triangles[:, k]] for k in range(3))
        volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
        self.assertGreater(volume, 0.02)  # cubic metres
        # the normals of the corners lie on the side of the triangle that its winding gives
        geometric = np.cross(b - a, c - a)
        corners = normals[triangles].sum(axis=1)
        agree = np.einsum("ij,ij->i", geometric, corners) > 0
        self.assertGreater(agree.mean(), 0.999)
        # TEXCOORD_0 holds the model's UV of every corner, with v = 0 at the top
        uv_index = corner_uv_indices(self.model, source[triangles])
        self.assertTrue(np.all(uv_index >= 0))
        expected = self.model.texture_coordinates.double().numpy()[uv_index]
        expected[..., 1] = 1.0 - expected[..., 1]
        self.assertLess(np.abs(uvs[triangles] - expected).max(), 1e-7)

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
        reference = json.loads(REFERENCE.read_text())
        self.assertEqual(reference["format"], REFERENCE_FORMAT)
        files = reference["files"]
        self.assertEqual(list(files), list(REFERENCE_CASES))
        with tempfile.TemporaryDirectory() as tmp:
            for name, recorded in files.items():
                with self.subTest(name):
                    case = dict(
                        character=Character.from_dict(recorded["character"]),
                        face_shapes=recorded["face_shapes"],
                        model=recorded["model"],
                        options=recorded["options"],
                    )
                    path = export_case(case, tmp, name)
                    found = gltf_fingerprint(path)
                    expected = recorded_structure(files, name)
                    if case["model"] == "exporter":
                        rtol, atol = FLOAT32_RTOL, FLOAT32_ATOL
                        tolerance = dict(
                            rtol=FLOAT32_SCALE_RTOL,
                            atol=REFERENCE_ATOL,
                            scales=fingerprint_scales(path),
                        )
                    else:
                        rtol, atol = REFERENCE_RTOL, REFERENCE_ATOL
                        tolerance = dict(rtol=rtol, atol=atol)
                    assert_close(self, found["structure"], expected, name, rtol, atol)
                    assert_fingerprints_close(
                        self,
                        found["accessors"],
                        recorded_accessors(files, name),
                        name,
                        **tolerance,
                    )

    def test_fingerprints_see_moved_rows(self):
        """Two swapped rows change the fingerprints of a float and an integer accessor."""
        from opensculptboy.export.document import GltfDocument

        values = np.random.default_rng(3).normal(size=(500, 3)).astype(np.float32)
        prints = []
        for swap in (False, True):
            order = np.arange(500)
            if swap:
                order[[10, 400]] = [400, 10]
            doc = GltfDocument("test")
            doc.accessor(values[order])
            doc.accessor(order.astype(np.uint16))
            prints.append(gltf_fingerprint(doc.to_bytes())["accessors"])
        (floats, integers), (moved_floats, moved_integers) = prints
        for key in ("probe", "moment"):
            difference = np.abs(np.subtract(floats[key], moved_floats[key]))
            tolerance = REFERENCE_ATOL + REFERENCE_RTOL * np.abs(floats[key])
            self.assertTrue(np.all(difference > 100 * tolerance), key)
        self.assertNotEqual(integers["sha256"], moved_integers["sha256"])

    def test_reference_is_small(self):
        self.assertLess(REFERENCE.stat().st_size, 300_000)


def json_value(value):
    """``value`` as JSON reads it back (tuples become lists)."""
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
