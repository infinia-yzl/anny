# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the VRM tables (opensculptboy.export.vrm_tables): the humanoid bone map against the
parent rules of the VRM 1.0 and 0.x specifications, the twist constraints, their roll axes
against the limbs of the T-pose and the node hierarchy of the file, the eyeballs and the VRM
skin weights, the expressions (names, mixes and override flags, and the mouth shapes of the
visemes measured on the mesh), the look-at ranges and the lid-only look targets.
"""

import types
import unittest

import numpy as np
import torch

from anny.keypoints import KeypointsRegressor
from opensculptboy.export import vrm_tables as vt
from opensculptboy.export.tpose import vrm_t_pose
from opensculptboy.export.vrm import FRAMES
from test import vrm_fixtures

SIDES = ("left", "right")
FINGERS = ("Index", "Middle", "Ring", "Little")


def _parent_rules(version: str) -> dict[str, str | None]:
    """
    The parent of each humanoid bone in the VRM specification of ``version``: the humanoid bone
    that must be its nearest humanoid ancestor, or, when that optional bone is absent, the
    parent of that bone, and so on up the table.
    """
    rules = {
        "hips": None,
        "spine": "hips",
        "chest": "spine",
        "upperChest": "chest",
        "neck": "upperChest",
        "head": "neck",
        "leftEye": "head",
        "rightEye": "head",
        "jaw": "head",
    }
    if version == "1.0":
        thumb = ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal")
    else:
        thumb = ("ThumbProximal", "ThumbIntermediate", "ThumbDistal")
    for side in SIDES:
        chains = [
            ("hips", "UpperLeg", "LowerLeg", "Foot", "Toes"),
            ("upperChest", "Shoulder", "UpperArm", "LowerArm", "Hand"),
            ("Hand",) + thumb,
        ] + [
            ("Hand", f"{f}Proximal", f"{f}Intermediate", f"{f}Distal") for f in FINGERS
        ]
        for chain in chains:
            names = [chain[0] if chain[0] in rules else side + chain[0]]
            names += [side + part for part in chain[1:]]
            for parent, child in zip(names, names[1:]):
                rules[child] = parent
    return rules


def _nearest_mapped_ancestor(bone: int, parents: list[int], mapped: dict[int, str]):
    p = parents[bone]
    while p >= 0 and p not in mapped:
        p = parents[p]
    return mapped.get(p)


class TestHumanoidMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = vrm_fixtures.model()
        cls.labels = list(cls.model.bone_labels)
        cls.parents = [int(p) for p in cls.model.bone_parents]

    def test_bones_exist_once_and_are_named_per_version(self):
        for version in vt.VERSIONS:
            with self.subTest(version=version):
                bones = vt.humanoid_bones(version)
                rules = _parent_rules(version)
                self.assertLessEqual(set(bones), set(rules), "names outside the spec")
                self.assertLessEqual(set(bones.values()), set(self.labels))
                self.assertEqual(
                    len(set(bones.values())), len(bones), "a bone mapped twice"
                )
                self.assertLessEqual(vt.required_bones(version), set(bones))
        self.assertLessEqual({"chest", "neck"}, vt.required_bones("0.x"))
        one, zero = vt.humanoid_bones("1.0"), vt.humanoid_bones("0.x")
        for side in SIDES:
            self.assertEqual(
                one[side + "ThumbMetacarpal"], zero[side + "ThumbProximal"]
            )
            self.assertEqual(
                one[side + "ThumbProximal"], zero[side + "ThumbIntermediate"]
            )
            self.assertEqual(one[side + "ThumbDistal"], zero[side + "ThumbDistal"])
            self.assertNotIn(side + "ThumbMetacarpal", zero)
            self.assertNotIn(side + "ThumbIntermediate", one)
        self.assertEqual(set(one.values()), set(zero.values()))
        self.assertRaises(ValueError, vt.humanoid_bones, "2.0")
        self.assertRaises(ValueError, vt.required_bones, "2.0")

    def test_parents_follow_the_spec(self):
        # The nearest mapped ancestor of every mapped bone is the parent that the spec asks
        # for, both in the rig and in the hierarchy of every kind of file.
        for version in vt.VERSIONS:
            bones = vt.humanoid_bones(version)
            rules = _parent_rules(version)
            mapped = {self.labels.index(label): name for name, label in bones.items()}
            for twist in vt.TWIST_MODES:
                hierarchies = {
                    "rig": self.parents,
                    "file": vt.file_parents(self.model, version, twist),
                }
                for kind, parents in hierarchies.items():
                    for name, label in bones.items():
                        with self.subTest(
                            version=version, twist=twist, kind=kind, bone=name
                        ):
                            expected = rules[name]
                            while expected is not None and expected not in bones:
                                expected = rules[expected]
                            found = _nearest_mapped_ancestor(
                                self.labels.index(label), parents, mapped
                            )
                            self.assertEqual(found, expected)


class TestTwist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = vrm_fixtures.model()
        cls.labels = list(cls.model.bone_labels)
        cls.parents = [int(p) for p in cls.model.bone_parents]

    def test_constraints(self):
        constraints = vt.twist_constraints("1.0", "constraint")
        humanoid = set(vt.humanoid_bones("1.0").values())
        # The forearm twist bones alone: the T-posed shin runs off every axis of the file
        # (test_roll_axes_follow_the_limbs), so its twist bone merges into the shin.
        self.assertEqual({c[0] for c in constraints}, {f"lowerarm02.{s}" for s in "LR"})
        for s in "LR":
            self.assertEqual(vt.TWIST_MERGE[f"lowerleg02.{s}"], f"lowerleg01.{s}")
        for bone, source, axis, weight in constraints:
            with self.subTest(bone=bone):
                self.assertNotIn(
                    bone, humanoid, "VRM apps drive humanoid bones themselves"
                )
                self.assertIn(source, humanoid)
                self.assertIn(bone, vt.TWIST_MERGE)
                # The roll axis is the axis of the forearm in the T-pose of the file.
                self.assertEqual(axis, "X")
                self.assertGreater(weight, 0.0)
                self.assertLessEqual(weight, 1.0)
        for version, twist in (
            ("1.0", "merge"),
            ("0.x", "constraint"),
            ("0.x", "merge"),
        ):
            self.assertEqual(vt.twist_constraints(version, twist), [])
        self.assertRaises(ValueError, vt.twist_constraints, "1.0", "spin")
        self.assertRaises(ValueError, vt.vrm_skin_weights, self.model, "spin")

    def test_file_parents(self):
        for version in vt.VERSIONS:
            for twist in vt.TWIST_MODES:
                with self.subTest(version=version, twist=twist):
                    parents = vt.file_parents(self.model, version, twist)
                    # A tree: one root, every parent before its children.
                    self.assertEqual(len(parents), len(self.labels))
                    self.assertEqual([j for j, p in enumerate(parents) if p < 0], [0])
                    self.assertTrue(all(p < j for j, p in enumerate(parents)))
                    changed = {
                        self.labels[j]: self.labels[p]
                        for j, p in enumerate(parents)
                        if p != self.parents[j]
                    }
                    constraints = vt.twist_constraints(version, twist)
                    if not constraints:
                        self.assertEqual(changed, {})
                        continue
                    # Only the sources move, from the twist bone to the twist bone's parent:
                    # the wrist goes under lowerarm01, and the foot stays under lowerleg02.
                    self.assertEqual(
                        changed, {f"wrist.{s}": f"lowerarm01.{s}" for s in "LR"}
                    )
                    for s in "LR":
                        foot = self.labels.index(f"foot.{s}")
                        self.assertEqual(self.labels[parents[foot]], f"lowerleg02.{s}")
                    for bone, source, _, _ in constraints:
                        b, s = self.labels.index(bone), self.labels.index(source)
                        self.assertNotIn(b, parents, f"{bone} must be a leaf")
                        self.assertEqual(parents[s], parents[b])

    def test_roll_axes_follow_the_limbs(self):
        # Every node of the file rests at the identity rotation, so a roll axis is an axis of
        # the file. It must run along the limb in the T-pose: from the joint of the twist
        # bone's file parent to the joint of its source (lowerarm01 to the wrist), within 1
        # degree, on the default body and on the body of the shared exports, with and without
        # the leg spread.
        G = FRAMES["1.0"]
        axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
        cos_1 = np.cos(np.radians(1.0))
        parents = vt.file_parents(self.model, "1.0", "constraint")
        kwargs = vrm_fixtures.CHARACTER.model_kwargs()
        bodies = {
            "default": {},
            "fixture": {
                k: kwargs[k] for k in ("phenotype_kwargs", "face_shape_kwargs")
            },
        }
        for body, body_kwargs in bodies.items():
            with torch.no_grad():
                rest = self.model(**body_kwargs)["rest_bone_poses"][0]
            for spread in (False, True):
                T = vrm_t_pose(self.model, rest, keep_leg_spread=spread)
                joints = T[:, :3, 3].double().numpy() @ G.T

                def direction(start: int, end: int) -> np.ndarray:
                    d = joints[end] - joints[start]
                    return d / np.linalg.norm(d)

                for bone, source, axis, _ in vt.twist_constraints("1.0", "constraint"):
                    with self.subTest(body=body, keep_leg_spread=spread, bone=bone):
                        b = self.labels.index(bone)
                        d = direction(parents[b], self.labels.index(source))
                        self.assertGreaterEqual(abs(d @ axes[axis]), cos_1)
                # Why the shins carry none: they run more than 1 degree off every axis.
                for s in "LR":
                    with self.subTest(body=body, keep_leg_spread=spread, shin=s):
                        d = direction(
                            self.labels.index(f"lowerleg01.{s}"),
                            self.labels.index(f"foot.{s}"),
                        )
                        self.assertLess(np.abs(d).max(), cos_1)


class TestSkinWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = vrm_fixtures.model()
        cls.labels = list(cls.model.bone_labels)
        cls.eyes = vt.eyeball_vertices(cls.model)
        w = cls.model.vertex_bone_weights.detach().double().numpy()
        i = cls.model.vertex_bone_indices.numpy()
        cls.before = cls.dense(w, i)

    @classmethod
    def dense(cls, weights, indices):
        out = np.zeros((len(weights), len(cls.labels)))
        np.add.at(out, (np.arange(len(weights))[:, None], indices), weights)
        return out

    def test_eyeballs(self):
        self.assertEqual({k: len(v) for k, v in self.eyes.items()}, {"L": 72, "R": 72})
        with torch.no_grad():
            out = self.model()
        bones = out["rest_bone_poses"][0].double().numpy()
        rest = out["rest_vertices"][0].double().numpy()
        for side, rows in self.eyes.items():
            # The eye bone sits at the centre of its eyeball, the pivot of look-at.
            head = bones[self.labels.index(f"eye.{side}"), :3, 3]
            self.assertLess(np.linalg.norm(rest[rows].mean(0) - head), 1e-3)

    def test_eyeballs_need_the_makehuman_mesh(self):
        # A retopologised mesh numbers its vertices its own way (its base-mesh indices run
        # from 0), and a mesh without eyes has no eye vertices: eyeball_vertices refuses both,
        # as it refuses a mesh whose MakeHuman indices name other vertices.
        model = self.model
        base = model.base_mesh_vertex_indices.clone()
        on_eyes = (base >= 14598) & (base <= 14741)

        def stub(indices):
            return types.SimpleNamespace(
                bone_labels=model.bone_labels,
                base_mesh_vertex_indices=indices,
                template_vertices=model.template_vertices,
                template_bone_heads=model.template_bone_heads,
            )

        self.assertEqual(
            {k: len(v) for k, v in vt.eyeball_vertices(stub(base)).items()},
            {"L": 72, "R": 72},
        )
        cases = {
            "retopologised": torch.arange(len(base)),
            "no eyes": torch.where(on_eyes, -1, base),
            "other vertices": torch.roll(base, 500),
            "no indices": None,
        }
        for case, indices in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(ValueError) as raised:
                    vt.eyeball_vertices(stub(indices))
                self.assertIn(
                    "topologies anny, anny-quads, anny-full, makehuman",
                    str(raised.exception),
                )
        # The writer's skin weights refuse such a mesh as well.
        with self.assertRaises(ValueError):
            vt.vrm_skin_weights(
                types.SimpleNamespace(
                    **vars(stub(torch.arange(len(base)))),
                    vertex_bone_weights=model.vertex_bone_weights,
                    vertex_bone_indices=model.vertex_bone_indices,
                )
            )

    def test_weights(self):
        head = self.labels.index("head")
        on_eyeball = np.zeros(len(self.before), dtype=bool)
        for rows in self.eyes.values():
            on_eyeball[rows] = True
        for twist in vt.TWIST_MODES:
            with self.subTest(twist=twist):
                weights, indices = vt.vrm_skin_weights(self.model, twist)
                self.assertEqual(
                    weights.shape, tuple(self.model.vertex_bone_weights.shape)
                )
                self.assertEqual(weights.dtype, np.float64)
                self.assertTrue(np.issubdtype(indices.dtype, np.integer))
                np.testing.assert_allclose(weights.sum(1), 1.0, atol=1e-12)
                self.assertTrue((weights >= 0).all())
                self.assertTrue(
                    (np.diff(weights, axis=1) <= 0).all(), "strongest first"
                )
                self.assertTrue((indices[weights == 0] == 0).all())
                after = self.dense(weights, indices)
                # No bone takes two slots of a row.
                np.testing.assert_array_equal((weights > 0).sum(1), (after > 0).sum(1))
                # The eye bones weigh their eyeballs alone, and each eyeball follows its
                # eye bone alone.
                for side, rows in self.eyes.items():
                    eye = self.labels.index(f"eye.{side}")
                    self.assertEqual(set(np.flatnonzero(after[:, eye])), set(rows))
                    np.testing.assert_allclose(after[rows, eye], 1.0, atol=1e-12)
                # The eyelids hand their eye weight to the head.
                eyes = [self.labels.index(f"eye.{s}") for s in "LR"]
                lids = ~on_eyeball & (self.before[:, eyes].sum(1) > 0)
                self.assertGreater(lids.sum(), 0)
                np.testing.assert_allclose(
                    after[lids, head],
                    self.before[lids, head] + self.before[lids][:, eyes].sum(1),
                    atol=1e-12,
                )
                # The twist bones hand their weights to their mapped parents.
                kept = {
                    self.labels.index(b)
                    for b in vt.TWIST_MERGE
                    if twist == "constraint" and b in vt.TWIST_BONES
                }
                expected = self.before.copy()
                for bone, parent in vt.TWIST_MERGE.items():
                    j, p = self.labels.index(bone), self.labels.index(parent)
                    if j not in kept:
                        expected[:, p] += expected[:, j]
                        expected[:, j] = 0.0
                    self.assertEqual(bool(after[:, j].any()), j in kept)
                others = [j for j in range(len(self.labels)) if j not in (head, *eyes)]
                np.testing.assert_allclose(
                    after[:, others], expected[:, others], atol=1e-12
                )


def _batched(kwargs: dict, n: int, dtype) -> dict:
    return {k: torch.full((n,), float(v), dtype=dtype) for k, v in kwargs.items()}


class TestExpressions(unittest.TestCase):
    """The expression table and its mixes, measured on the default body and on the body of the
    shared exports."""

    EMOTIONS = ("happy", "angry", "sad", "relaxed", "surprised")
    VISEMES = ("aa", "ih", "ou", "ee", "oh")
    VSEEFACE = ("SIL", "CH", "DD", "FF", "KK", "NN", "PP", "RR", "SS", "TH")
    VRM0 = {
        "aa": "a",
        "ih": "i",
        "ou": "u",
        "ee": "e",
        "oh": "o",
        "blink": "blink",
        "blinkLeft": "blink_l",
        "blinkRight": "blink_r",
        "happy": "joy",
        "angry": "angry",
        "sad": "sorrow",
        "relaxed": "fun",
        "neutral": "neutral",
        "surprised": None,
    }

    @classmethod
    def setUpClass(cls):
        cls.model = vrm_fixtures.model()
        cls.actions = list(cls.model.facial_action_labels)
        A = len(cls.actions)
        dtype = cls.model.dtype
        rows = torch.zeros(A + 1, A, dtype=dtype)
        rows[1:] = torch.eye(A, dtype=dtype)
        kwargs = vrm_fixtures.CHARACTER.model_kwargs()
        bodies = {
            "default": {},
            "fixture": dict(
                phenotype_kwargs=_batched(kwargs["phenotype_kwargs"], A + 1, dtype),
                face_shape_kwargs=_batched(kwargs["face_shape_kwargs"], A + 1, dtype),
            ),
        }
        cls.rest = {}
        with torch.no_grad():
            for name, body in bodies.items():
                out = cls.model(facial_actions=rows, **body)
                cls.rest[name] = out["rest_vertices"].double().numpy()
        regressor = KeypointsRegressor.craniofacial(cls.model, ["sto", "ch.L", "ch.R"])
        cls.landmarks = {
            k: int(np.argmax(w))
            for k, w in zip(regressor.labels, regressor.regression_weights.numpy())
        }
        cls.table = {
            v: {e.name: e for e in vt.expressions(v, cls.actions)} for v in vt.VERSIONS
        }

    def test_presets(self):
        for version in vt.VERSIONS:
            presets = {n: e for n, e in self.table[version].items() if e.preset}
            self.assertEqual(set(presets), set(self.VRM0))
            self.assertLessEqual(set(presets), set(vt.PRESETS_1))
            for name, e in presets.items():
                with self.subTest(version=version, expression=name):
                    self.assertEqual(e.vrm0_preset, self.VRM0[name])
                    self.assertIn(e.vrm0_preset, (None, *vt.PRESETS_0))
                    emotion = name in self.EMOTIONS
                    self.assertEqual(e.override_blink, "blend" if emotion else "none")
                    self.assertEqual(e.override_mouth, "blend" if emotion else "none")
                    self.assertEqual(e.override_look_at, "none")
                    self.assertFalse(e.is_binary)
        surprised = self.table["0.x"]["surprised"]
        self.assertIsNone(surprised.vrm0_preset)
        self.assertEqual(surprised.vrm0_name, "Surprised")
        self.assertEqual(self.table["1.0"]["neutral"].mix, {})
        self.assertEqual(self.table["1.0"]["blinkLeft"].mix, {"eyeBlinkLeft": 1.0})
        self.assertEqual(self.table["1.0"]["blinkRight"].mix, {"eyeBlinkRight": 1.0})

    def test_customs(self):
        perfect_sync = {a[0].upper() + a[1:]: a for a in self.actions}
        self.assertEqual(len(perfect_sync), 52)
        for version in vt.VERSIONS:
            with self.subTest(version=version):
                customs = {n: e for n, e in self.table[version].items() if not e.preset}
                vseeface = set(self.VSEEFACE) if version == "0.x" else set()
                self.assertEqual(set(customs), set(perfect_sync) | vseeface)
                for name, action in perfect_sync.items():
                    self.assertTrue(name[0].isupper())
                    self.assertEqual(customs[name].mix, {action: 1.0})
        self.assertRaises(ValueError, vt.expressions, "0.x", self.actions[:-1])
        self.assertRaises(ValueError, vt.expressions, "2.0", self.actions)

    def test_names_never_clash(self):
        # VRM 1.0: no custom expression takes the name of a preset, compared without case.
        names_1 = [n.lower() for n in self.table["1.0"]]
        self.assertEqual(len(set(names_1)), len(names_1))
        customs_1 = {n.lower() for n, e in self.table["1.0"].items() if not e.preset}
        self.assertEqual(customs_1 & {p.lower() for p in vt.PRESETS_1}, set())
        # VRM 0.x: the group names are unique, and no custom group takes a preset name.
        groups = [
            (e.vrm0_name or e.name, e.vrm0_preset) for e in self.table["0.x"].values()
        ]
        names_0 = [g.lower() for g, _ in groups]
        self.assertEqual(len(set(names_0)), len(names_0))
        reserved = set(vt.PRESETS_0) | {g.lower() for g, p in groups if p is not None}
        customs_0 = {g.lower() for g, p in groups if p is None}
        self.assertEqual(customs_0 & reserved, set())

    def test_mixes(self):
        known = set(self.actions)
        for version in vt.VERSIONS:
            for name, e in self.table[version].items():
                with self.subTest(version=version, expression=name):
                    self.assertLessEqual(set(e.mix), known)
                    for weight in e.mix.values():
                        self.assertGreater(weight, 0.0)
                        self.assertLessEqual(weight, 1.0)
                    if name in ("blinkLeft", "blinkRight") or (
                        not e.preset and name not in self.VSEEFACE
                    ):
                        continue
                    # The other presets and the VSeeFace visemes are symmetric.
                    for action, weight in e.mix.items():
                        for a, b in (("Left", "Right"), ("Right", "Left")):
                            if action.endswith(a):
                                self.assertEqual(
                                    e.mix.get(action[: -len(a)] + b), weight
                                )

    def mouth(self, body: str, mix: dict) -> tuple[float, float]:
        """The lip opening and the mouth width (metres) of a mix on a body."""
        rest = self.rest[body]
        vertices = rest[0] + sum(
            w * (rest[1 + self.actions.index(a)] - rest[0]) for a, w in mix.items()
        )
        upper, lower = self.lips(body)
        opening = np.linalg.norm(
            vertices[upper][:, None] - vertices[lower][None], axis=2
        )
        width = np.linalg.norm(
            vertices[self.landmarks["ch.L"]] - vertices[self.landmarks["ch.R"]]
        )
        return float(opening.min()), float(width)

    def lips(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """The vertices of the upper and the lower lip on the midline of the face, told apart
        by the jaw: the lower lip follows jawOpen and the upper lip stays."""
        rest = self.rest[body]
        neutral = rest[0]
        sto = neutral[self.landmarks["sto"]]
        jaw = np.linalg.norm(rest[1 + self.actions.index("jawOpen")] - neutral, axis=1)
        lip = np.flatnonzero(
            (np.abs(neutral[:, 0] - sto[0]) < 1e-5)
            & (np.linalg.norm(neutral - sto, axis=1) < 0.025)
            & (neutral[:, 1] < sto[1] + 0.008)  # in front of the teeth and the tongue
        )
        upper, lower = lip[jaw[lip] < 0.005], lip[jaw[lip] > 0.02]
        self.assertGreaterEqual(len(upper), 5)
        self.assertGreaterEqual(len(lower), 5)
        return upper, lower

    def test_visemes(self):
        e = self.table["1.0"]
        for body in self.rest:
            with self.subTest(body=body):
                closed, width = self.mouth(body, {})
                shapes = {name: self.mouth(body, e[name].mix) for name in self.VISEMES}
                self.assertLess(closed, 0.002)
                opening, _ = shapes["aa"]
                self.assertGreaterEqual(opening, 0.012)
                self.assertLessEqual(opening, 0.025)
                for name in ("ih", "ee"):
                    opening, wide = shapes[name]
                    self.assertGreaterEqual(wide / width, 1.05, name)
                    self.assertLessEqual(opening, 0.008, name)
                for name in ("ou", "oh"):
                    self.assertLessEqual(shapes[name][1] / width, 0.90, name)
                self.assertGreater(shapes["oh"][0], shapes["ou"][0])

    def test_look_at(self):
        eyes = vt.eyeball_vertices(self.model)
        ranges = vt.look_at_ranges(self.model)
        self.assertEqual(set(ranges), {"lookUp", "lookDown", "lookIn", "lookOut"})
        for body, rest in self.rest.items():
            given = vt.look_at_ranges(
                self.model, {"rest_vertices": torch.from_numpy(rest)}
            )
            for direction, angle in given.items():
                with self.subTest(body=body, direction=direction):
                    self.assertGreaterEqual(angle, 5.0)
                    self.assertLessEqual(angle, 60.0)
                    if body == "default":
                        self.assertAlmostEqual(angle, ranges[direction], places=6)
        # Each action turns the gaze (-Y, the figure faces -Y) its own way: up, down, toward the
        # nose or away from it. The left eye lies at +X.
        rest = self.rest["default"]
        for side, eye, s in (("Left", "L", 1.0), ("Right", "R", -1.0)):
            for prefix, axis, sign in (
                ("eyeLookUp", 2, 1.0),
                ("eyeLookDown", 2, -1.0),
                ("eyeLookIn", 0, -s),
                ("eyeLookOut", 0, s),
            ):
                with self.subTest(action=prefix + side):
                    rows = eyes[eye]
                    turn = vt.eyeball_rotation(
                        rest[0, rows], rest[1 + self.actions.index(prefix + side), rows]
                    )
                    gaze = turn @ np.array([0.0, -1.0, 0.0])
                    self.assertGreater(sign * gaze[axis], np.sin(np.radians(4.0)))

    def test_lid_only(self):
        eyes = vt.eyeball_vertices(self.model)
        rest = self.rest["default"]
        offsets = rest[1 + self.actions.index("eyeLookUpLeft")] - rest[0]
        lids = vt.lid_only(offsets, eyes)
        rows = np.concatenate(list(eyes.values()))
        others = np.setdiff1d(np.arange(len(offsets)), rows)
        self.assertTrue((lids[rows] == 0).all())
        np.testing.assert_array_equal(lids[others], offsets[others])
        self.assertTrue(np.abs(offsets[rows]).max() > 0, "the input stays as it was")
        self.assertTrue(np.abs(lids[others]).max() > 0, "the lids still move")


if __name__ == "__main__":
    unittest.main()
