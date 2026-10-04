# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the glTF document (opensculptboy.export.document) and the body data
(opensculptboy.export.body): a hand-built document with several meshes, skins, an embedded
texture, extensions and sparse and dense morph targets is written, read back with the test
reader (test/gltf_reader.py) and evaluated as a glTF engine does.
"""

import io
import pathlib
import tempfile
import types
import unittest

import numpy as np
import roma
import torch

from opensculptboy import Character, read_character
from opensculptboy.export.body import (
    BodyMesh,
    body_attributes,
    body_primitive,
    body_targets,
    build_body,
    morph_target_rows,
)
from opensculptboy.export.document import (
    ARRAY_BUFFER,
    CLAMP_TO_EDGE,
    ELEMENT_ARRAY_BUFFER,
    LINEAR,
    LINEAR_MIPMAP_LINEAR,
    REPEAT,
    GltfDocument,
    extension_names,
    pbr_material,
    read_gltf_json,
    unlit_material,
)
from test.gltf_reader import (
    GLB,
    evaluate,
    evaluate_nodes,
    node_world_matrices,
    structure_errors,
)


def quaternion(axis, degrees):
    """A unit quaternion (x, y, z, w) for a turn about an axis."""
    axis = np.asarray(axis, dtype=np.float64)
    half = np.radians(degrees) / 2
    return [*(np.sin(half) * axis / np.linalg.norm(axis)), np.cos(half)]


def matrix(translation=(0, 0, 0), rotation=(0, 0, 0, 1), scale=(1, 1, 1)):
    """The 4x4 matrix of a TRS (roma, independent of the reader)."""
    m = np.eye(4)
    q = torch.tensor(rotation, dtype=torch.float64)
    m[:3, :3] = roma.unitquat_to_rotmat(q).numpy() * np.asarray(scale)[None, :]
    m[:3, 3] = translation
    return m


# The skeleton: a root, a spine given by a matrix, an arm, and a scaled tail.
REST = {
    "root": dict(translation=(0.0, 1.0, 0.0), rotation=quaternion((0, 1, 0), 30)),
    "spine": dict(translation=(0.0, 0.5, 0.1), rotation=quaternion((1, 0, 0), 20)),
    "arm": dict(translation=(0.3, 0.0, 0.0), rotation=quaternion((0, 0, 1), -40)),
    "tail": dict(
        translation=(0.0, -0.2, -0.1),
        rotation=quaternion((1, 1, 0), 15),
        scale=(1.0, 2.0, 1.0),
    ),
}
PARENTS = {"root": None, "spine": "root", "arm": "spine", "tail": "root"}
ORDER = list(REST)


def world_matrices(pose: dict) -> dict[str, np.ndarray]:
    """World matrices of the skeleton for the TRS of each bone."""
    world = {}
    for name in ORDER:
        local = matrix(**pose[name])
        parent = PARENTS[name]
        world[name] = local if parent is None else world[parent] @ local
    return world


def rest_world() -> dict[str, np.ndarray]:
    return world_matrices(REST)


def pixels() -> np.ndarray:
    """A 2 x 4 RGBA texture."""
    return np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4) * 7


def build_document():
    """
    The hand-built document and its inputs: meshes ``a`` and ``b`` share skin 0 (``b`` has two
    primitives that share their vertices), mesh ``c`` has skin 1 and 8 influences; ``a`` has a
    sparse and a dense morph target; a PNG texture with a sampler, an unlit material and a
    material whose texture carries an extension.
    """
    rng = np.random.default_rng(7)
    doc = GltfDocument("test")
    world = rest_world()
    nodes = {}
    for name in ORDER:
        node = {"name": name}
        if name == "spine":  # a node given by its matrix (column-major)
            node["matrix"] = matrix(**REST[name]).T.reshape(-1).tolist()
        else:
            node.update({k: [float(x) for x in v] for k, v in REST[name].items()})
        nodes[name] = doc.add_node(node)
    for name in ORDER:
        children = [nodes[c] for c in ORDER if PARENTS[c] == name]
        if children:
            doc.nodes[nodes[name]]["children"] = children

    def skin(joints):
        ibm = np.stack([np.linalg.inv(world[j]) for j in joints])
        return doc.add_skin(
            {
                "joints": [nodes[j] for j in joints],
                "inverseBindMatrices": doc.accessor(
                    ibm.transpose(0, 2, 1).reshape(-1, 16).astype(np.float32)
                ),
            }
        )

    skin_a = skin(["root", "spine", "arm"])
    skin_b = skin(["tail", "root", "arm"])

    image = doc.add_png(pixels(), name="checker")
    sampler = doc.add_sampler(
        {
            "magFilter": LINEAR,
            "minFilter": LINEAR_MIPMAP_LINEAR,
            "wrapS": REPEAT,
            "wrapT": CLAMP_TO_EDGE,
        }
    )
    texture = doc.add_texture(image, sampler)
    unlit = doc.add_material(unlit_material("unlit", (1, 1, 1, 1), texture))
    plain = pbr_material("plain", (0.2, 0.4, 0.6, 1.0))
    plain["pbrMetallicRoughness"]["baseColorTexture"] = {
        "index": texture,
        "extensions": {"KHR_texture_transform": {"scale": [2.0, 2.0]}},
    }
    plain = doc.add_material(plain)

    def vertices(count, influences, joints):
        positions = rng.normal(size=(count, 3)) * 0.3 + [0.0, 1.3, 0.0]
        weights = rng.random((count, influences))
        weights[:, influences // 2 :] *= 0.2
        weights /= weights.sum(axis=1, keepdims=True)
        indices = rng.integers(0, joints, size=(count, influences))
        attributes = {
            "POSITION": doc.accessor(
                positions.astype(np.float32), ARRAY_BUFFER, minmax=True
            )
        }
        for s in range(influences // 4):
            cols = slice(4 * s, 4 * s + 4)
            attributes[f"JOINTS_{s}"] = doc.accessor(
                indices[:, cols].astype(np.uint8), ARRAY_BUFFER
            )
            attributes[f"WEIGHTS_{s}"] = doc.accessor(
                weights[:, cols].astype(np.float32), ARRAY_BUFFER
            )
        return positions.astype(np.float32).astype(np.float64), attributes

    def triangles(count, n):
        return doc.accessor(
            rng.integers(0, count, size=3 * n).astype(np.uint16), ELEMENT_ARRAY_BUFFER
        )

    inputs = {}
    # mesh a: one primitive with a sparse and a dense target
    positions_a, attributes_a = vertices(50, 4, 3)
    sparse = np.zeros((50, 3))
    sparse[[3, 17, 41]] = rng.normal(size=(3, 3)) * 0.05
    dense = np.zeros((50, 3))
    dense[[5, 6]] = rng.normal(size=(2, 3)) * 0.05
    targets_a = [
        {"POSITION": doc.morph_accessor(sparse)},
        {"POSITION": doc.morph_accessor(dense, sparse=False)},
    ]
    mesh_a = doc.add_mesh(
        {
            "name": "a",
            "primitives": [
                {
                    "attributes": attributes_a,
                    "indices": triangles(50, 30),
                    "material": unlit,
                    "targets": targets_a,
                }
            ],
            "weights": [0.0, 0.0],
            "extras": {"targetNames": ["sparse", "dense"]},
        }
    )
    inputs["a"] = (positions_a, sparse, dense)
    # mesh b: two primitives sharing their vertices, two materials
    positions_b, attributes_b = vertices(40, 4, 3)
    mesh_b = doc.add_mesh(
        {
            "name": "b",
            "primitives": [
                {
                    "attributes": attributes_b,
                    "indices": triangles(40, 10),
                    "material": m,
                }
                for m in (unlit, plain)
            ],
        }
    )
    inputs["b"] = positions_b
    # mesh c: its own skin, 8 influences
    positions_c, attributes_c = vertices(30, 8, 3)
    mesh_c = doc.add_mesh(
        {
            "name": "c",
            "primitives": [
                {
                    "attributes": attributes_c,
                    "indices": triangles(30, 10),
                    "material": plain,
                }
            ],
        }
    )
    inputs["c"] = positions_c
    mesh_nodes = [
        doc.add_node({"name": "a", "mesh": mesh_a, "skin": skin_a}),
        doc.add_node({"name": "b", "mesh": mesh_b, "skin": skin_a}),
        doc.add_node({"name": "c", "mesh": mesh_c, "skin": skin_b}),
    ]
    # an animation that bends the arm and turns the root
    times = doc.accessor(np.array([0.0, 1.0], dtype=np.float32), minmax=True)
    bend = np.array([REST["arm"]["rotation"], quaternion((0, 0, 1), 50)])
    turn = np.array([REST["root"]["rotation"], quaternion((0, 1, 0), -10)])
    doc.add_animation(
        {
            "name": "bend",
            "samplers": [
                {"input": times, "output": doc.accessor(q.astype(np.float32))}
                for q in (bend, turn)
            ],
            "channels": [
                {"sampler": 0, "target": {"node": nodes["arm"], "path": "rotation"}},
                {"sampler": 1, "target": {"node": nodes["root"], "path": "rotation"}},
            ],
        }
    )
    doc.scene_nodes = [nodes["root"], *mesh_nodes]
    doc.scene_name = "hand-built"
    doc.scene_extras = {"opensculptboy": {"character": Character(name="doc").to_dict()}}
    doc.extensions["EXT_test_root"] = {"value": 1}
    return doc, inputs


def skinned(positions, glb, node, world):
    """Linear blend skinning of a node's first primitive with given world matrices."""
    g = glb.json
    skin = g["skins"][g["nodes"][node]["skin"]]
    ibm = glb.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    names = [g["nodes"][j]["name"] for j in skin["joints"]]
    matrices = np.stack([world[n] for n in names]) @ ibm
    attrs = g["meshes"][g["nodes"][node]["mesh"]]["primitives"][0]["attributes"]
    out = np.zeros_like(positions)
    s = 0
    while f"JOINTS_{s}" in attrs:
        joints = glb.accessor(attrs[f"JOINTS_{s}"]).astype(int)
        weights = glb.accessor(attrs[f"WEIGHTS_{s}"]).astype(np.float64)
        for k in range(4):
            m = matrices[joints[:, k]]
            out += weights[:, k, None] * (
                np.einsum("vij,vj->vi", m[:, :3, :3], positions) + m[:, :3, 3]
            )
        s += 1
    return out


class TestGltfDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.doc, cls.inputs = build_document()
        cls.path = pathlib.Path(cls.tmp.name) / "doc.glb"
        cls.size = cls.doc.write(cls.path)
        cls.glb = GLB(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_file_is_sound(self):
        self.assertEqual(structure_errors(self.glb), [])
        self.assertEqual(self.size, self.path.stat().st_size)
        self.assertEqual(self.doc.to_bytes(), self.path.read_bytes())
        g = self.glb.json
        self.assertEqual(len(g["meshes"]), 3)
        self.assertEqual(len(g["skins"]), 2)
        self.assertEqual(g["buffers"][0]["byteLength"], len(self.glb.bin))

    def test_extensions_are_declared(self):
        used = self.glb.json["extensionsUsed"]
        self.assertEqual(
            sorted(used),
            ["EXT_test_root", "KHR_materials_unlit", "KHR_texture_transform"],
        )
        self.assertNotIn("extensionsRequired", self.glb.json)
        self.assertEqual(
            self.glb.json["materials"][0]["extensions"], {"KHR_materials_unlit": {}}
        )
        # extras never declare extensions
        self.assertEqual(extension_names({"extras": {"extensions": {"EXT_x": {}}}}), [])

    def test_rest_pose_gives_the_inputs(self):
        nodes = evaluate_nodes(self.glb)
        names = {self.glb.json["nodes"][i]["name"]: i for i in nodes}
        self.assertEqual(sorted(names), ["a", "b", "c"])
        positions_a, _, _ = self.inputs["a"]
        for name, expected in (
            ("a", positions_a),
            ("b", self.inputs["b"]),
            ("c", self.inputs["c"]),
        ):
            for primitive in nodes[names[name]]:
                self.assertLess(
                    np.abs(primitive.positions - expected).max(), 1e-5, name
                )
        # the two primitives of b share their vertices, and keep their own triangles
        b0, b1 = nodes[names["b"]]
        self.assertIs(b0.positions, b1.positions)
        self.assertEqual((b0.material, b1.material), (0, 1))
        self.assertFalse(np.array_equal(b0.triangles, b1.triangles))
        # evaluate keeps the old interface: the first skinned mesh node
        positions, anny_vertex = evaluate(self.glb)
        self.assertLess(np.abs(positions - positions_a).max(), 1e-5)
        self.assertIsNone(anny_vertex)

    def test_sparse_and_dense_targets(self):
        accessors = self.glb.json["accessors"]
        targets = self.glb.json["meshes"][0]["primitives"][0]["targets"]
        self.assertIn("sparse", accessors[targets[0]["POSITION"]])
        self.assertNotIn("sparse", accessors[targets[1]["POSITION"]])
        positions, sparse, dense = self.inputs["a"]
        for weights in ([1.0, 0.0], [0.0, 1.0], [0.5, -0.7]):
            nodes = evaluate_nodes(self.glb, weights={"a": weights})
            a = next(p for prims in nodes.values() for p in prims if p.mesh == 0)
            expected = positions + weights[0] * sparse + weights[1] * dense
            self.assertLess(np.abs(a.positions - expected).max(), 1e-5, weights)

    def test_animation_skins_every_mesh(self):
        pose = dict(REST)
        pose["arm"] = dict(REST["arm"], rotation=quaternion((0, 0, 1), 50))
        pose["root"] = dict(REST["root"], rotation=quaternion((0, 1, 0), -10))
        posed = world_matrices(pose)
        reader_world = node_world_matrices(self.glb, "bend", 1)
        for name in ORDER:
            i = [n["name"] for n in self.glb.json["nodes"]].index(name)
            self.assertLess(np.abs(reader_world[i] - posed[name]).max(), 1e-6, name)
        nodes = evaluate_nodes(self.glb, "bend", 1)
        names = [n["name"] for n in self.glb.json["nodes"]]
        for name, positions in (
            ("a", self.inputs["a"][0]),
            ("b", self.inputs["b"]),
            ("c", self.inputs["c"]),
        ):
            node = names.index(name)
            expected = skinned(positions, self.glb, node, posed)
            found = nodes[node][0].positions
            self.assertLess(np.abs(found - expected).max(), 1e-5, name)
            self.assertGreater(np.abs(found - positions).max(), 0.05, name)

    def test_texture_round_trip(self):
        from PIL import Image

        g = self.glb.json
        image = g["images"][g["textures"][0]["source"]]
        self.assertEqual((image["mimeType"], image["name"]), ("image/png", "checker"))
        decoded = np.asarray(Image.open(io.BytesIO(self.glb.image_bytes(0))))
        np.testing.assert_array_equal(decoded, pixels())
        self.assertEqual(
            g["samplers"][g["textures"][0]["sampler"]],
            {
                "magFilter": LINEAR,
                "minFilter": LINEAR_MIPMAP_LINEAR,
                "wrapS": REPEAT,
                "wrapT": CLAMP_TO_EDGE,
            },
        )
        texture_info = g["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"]
        self.assertEqual(texture_info["index"], 0)

    def test_structure_errors_are_found(self):
        glb = GLB(self.path)
        glb.json["meshes"][0]["primitives"][0]["material"] = 9
        glb.json["nodes"][0]["children"].append(1)  # the spine twice
        glb.json["extensionsUsed"].remove("KHR_texture_transform")
        errors = "\n".join(structure_errors(glb))
        self.assertIn("materials index 9", errors)
        self.assertIn("two parents", errors)
        self.assertIn("KHR_texture_transform", errors)

    def test_read_character_from_vrm_and_root_extras(self):
        character = Character(name="vrm card", phenotype={"height": 0.6})
        for where in ("scene", "root", "asset"):
            doc = GltfDocument("test")
            doc.add_node({"name": "empty"})
            doc.scene_nodes = [0]
            entry = {"opensculptboy": {"character": character.to_dict()}}
            setattr(
                doc,
                {"scene": "scene_extras", "root": "extras"}.get(where, "asset_extras"),
                entry,
            )
            path = pathlib.Path(self.tmp.name) / f"card_{where}.vrm"
            doc.write(path)
            self.assertEqual(read_character(path), character, where)
            self.assertEqual(read_gltf_json(path)["nodes"], [{"name": "empty"}])
        doc = GltfDocument("test")
        doc.write(pathlib.Path(self.tmp.name) / "none.vrm")
        with self.assertRaisesRegex(ValueError, "no OpenSculptBoy character"):
            read_character(pathlib.Path(self.tmp.name) / "none.vrm")

    def test_accessor_types(self):
        doc = GltfDocument("test")
        with self.assertRaisesRegex(TypeError, "float64"):
            doc.accessor(np.zeros((3, 3)))
        index = doc.accessor(np.array([[-1, 2]], dtype=np.int16), normalized=True)
        self.assertEqual(doc.buffers.accessors[index]["componentType"], 5122)
        deltas = np.zeros((10, 3), dtype=np.float32)
        deltas[2] = [1e-7, 0, 0]  # below MORPH_EPSILON: no movement
        deltas[4] = [0, 0.1, 0]
        doc.morph_accessor(deltas)
        self.assertEqual(
            deltas[2, 0], np.float32(1e-7)
        )  # the caller's array is untouched
        self.assertEqual(doc.buffers.accessors[-1]["sparse"]["count"], 1)


def fake_model():
    """
    Two quads sharing an edge, with a UV seam along that edge, and two bones: the shape of the
    model attributes that build_body reads.
    """
    vertices = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [2, 0, 0], [2, 0, 1]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2, 3], [1, 4, 5, 2]])
    uvs = torch.tensor(
        [[0, 0], [0.5, 0], [0.5, 1], [0, 1], [0.6, 0], [1, 0], [1, 1], [0.6, 1]],
        dtype=torch.float64,
    )
    uv_faces = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    weights = torch.tensor(
        [
            [1, 0, 0],
            [0.6, 0.3, 0.1],
            [0.5, 0.5, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.2, 0.8, 0],
        ],
        dtype=torch.float64,
    )
    indices = torch.tensor(
        [[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
    )
    model = types.SimpleNamespace(
        faces=faces,
        face_texture_coordinate_indices=uv_faces,
        texture_coordinates=uvs,
        vertex_bone_weights=weights,
        vertex_bone_indices=indices,
    )
    return model, vertices.numpy()


class TestBody(unittest.TestCase):
    def test_seams_split_and_map_back(self):
        model, vertices = fake_model()
        offsets = np.zeros((1, 6, 3))
        offsets[0, 4] = [0, 0, 0.1]
        body = build_body(model, vertices, offsets, max_influences=4)
        self.assertIsInstance(body, BodyMesh)
        self.assertEqual(len(body.source), 8)  # the seam splits vertices 1 and 2
        self.assertEqual(sorted(np.bincount(body.source).tolist()), [1, 1, 1, 1, 2, 2])
        frame = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
        np.testing.assert_allclose(body.positions, vertices[body.source] @ frame.T)
        np.testing.assert_allclose(body.targets[0], offsets[0][body.source] @ frame.T)
        # every glTF triangle is the Anny triangle on the split vertices
        self.assertEqual(body.triangles.shape, (4, 3))
        np.testing.assert_array_equal(
            body.source[body.triangles],
            np.array([[0, 1, 2], [1, 4, 5], [0, 2, 3], [1, 5, 2]]),
        )
        # the quads lie in the plane y = 0 of Anny: the normals point along glTF's z
        np.testing.assert_allclose(np.abs(body.normals[:, 2]), 1.0)
        self.assertAlmostEqual(body.dropped_weight, 0.0)
        np.testing.assert_allclose(body.weights.sum(axis=1), 1.0, atol=1e-6)
        # given normals take the place of the computed ones
        normals = np.tile([0.0, 0.0, 1.0], (6, 1))
        body = build_body(model, vertices, normals=normals)
        np.testing.assert_allclose(body.normals, np.tile([0.0, 1.0, 0.0], (8, 1)))

    def test_primitives_share_vertex_data(self):
        model, vertices = fake_model()
        offsets = np.zeros((2, 6, 3))
        offsets[0, 4] = [0, 0, 0.1]
        offsets[1] = 0.01
        body = build_body(model, vertices, offsets, max_influences=8)
        doc = GltfDocument("test")
        for name in ("root", "child"):
            doc.add_node({"name": name})
        doc.nodes[0]["children"] = [1]
        skin = doc.add_skin(
            {
                "joints": [0, 1],
                "inverseBindMatrices": doc.accessor(
                    np.tile(np.eye(4).reshape(16), (2, 1)).astype(np.float32)
                ),
            }
        )
        attributes = body_attributes(doc, body, translation=np.array([0, 1.0, 0]))
        targets = body_targets(doc, body)
        count = len(doc.buffers.accessors)
        first = body_primitive(
            doc,
            body,
            0,
            triangles=body.triangles[:2],
            attributes=attributes,
            targets=targets,
        )
        second = body_primitive(
            doc,
            body,
            1,
            triangles=body.triangles[2:],
            attributes=attributes,
            targets=targets,
        )
        self.assertEqual(len(doc.buffers.accessors), count + 2)  # the two index lists
        self.assertEqual(first["attributes"], second["attributes"])
        self.assertEqual(first["targets"], second["targets"])
        self.assertEqual(sorted(attributes)[:2], ["JOINTS_0", "JOINTS_1"])
        for name in ("a", "b"):
            doc.add_material(pbr_material(name))
        mesh = doc.add_mesh({"primitives": [first, second], "weights": [0.0, 0.0]})
        doc.add_node({"name": "body", "mesh": mesh, "skin": skin})
        doc.scene_nodes = [0, 2]
        glb = GLB(doc.to_bytes())
        self.assertEqual(structure_errors(glb), [])
        a, b = evaluate_nodes(glb, weights=[1.0, 1.0])[2]
        expected = body.positions + [0, 1.0, 0] + body.targets[0] + body.targets[1]
        self.assertLess(np.abs(a.positions - expected).max(), 1e-6)
        self.assertIs(a.positions, b.positions)
        np.testing.assert_array_equal(a.anny_vertex, body.source)

    def test_morph_target_rows(self):
        model = types.SimpleNamespace(
            facial_action_labels=["jawOpen", "eyeBlinkLeft"],
            face_shape_labels=["head-fat", "head-oval"],
            face_shape_ranges={"head-fat": (-1.0, 1.0), "head-oval": (0.0, 1.0)},
        )
        character = Character(
            facial_actions={"jawOpen": 0.3, "eyeBlinkLeft": 0.2},
            face_shapes={"head-fat": -0.4, "head-oval": 0.5},
        )
        targets, facial, face = morph_target_rows(
            model, character, ["jawOpen", "head-fat", "head-oval"]
        )
        self.assertEqual(
            [(t.name, t.value, t.weight) for t in targets],
            [
                ("jawOpen", 1.0, 0.3),
                ("head-fat.pos", 1.0, 0.0),
                ("head-fat.neg", -1.0, 0.4),
                ("head-oval.pos", 1.0, 0.5),
            ],
        )
        # row 0: the targets at 0, the rest of the character baked in
        np.testing.assert_array_equal(facial[0], [0.0, 0.2])
        np.testing.assert_array_equal(face[0], [0.0, 0.0])
        np.testing.assert_array_equal(facial[1], [1.0, 0.2])
        np.testing.assert_array_equal(face[3], [-1.0, 0.0])
        np.testing.assert_array_equal(face[4], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
