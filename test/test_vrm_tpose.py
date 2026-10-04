# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the VRM T-pose and its rebind (opensculptboy.export.tpose): the directions of the
arms, hands, fingers, thumbs, legs and feet, the palms and the nails, the symmetry, the
agreement of the forward bind with Anny's own forward pass, the exactness of the inverse bind at
Anny's rest pose, and the pose error of both binds.
"""

import itertools
import time
import unittest

import numpy as np
import torch

from opensculptboy.export import tpose
from opensculptboy.export.body import top_skin_weights
from test import vrm_fixtures

SIDES = {"L": 1.0, "R": -1.0}
UP = np.array([0.0, 0.0, 1.0])


def angle(a, b) -> float:
    """The angle between two directions, in degrees."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def unit(v):
    return v / np.linalg.norm(v)


class TestVrmTPose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = (
            vrm_fixtures.model()
        )  # float64, with every facial action and face shape
        cls.labels = list(cls.model.bone_labels)
        cls.parents = [int(p) for p in cls.model.bone_parents]
        kwargs = vrm_fixtures.CHARACTER.model_kwargs()
        # The default body, and the body of the shared exports (a phenotype off the default
        # and a face shape).
        cls.bodies = {
            "default": {},
            "fixture": dict(
                phenotype_kwargs=kwargs["phenotype_kwargs"],
                face_shape_kwargs=kwargs["face_shape_kwargs"],
            ),
        }
        actions = list(cls.model.facial_action_labels)
        dtype = cls.model.template_vertices.dtype
        cls.rows = torch.zeros(len(actions) + 1, len(actions), dtype=dtype)
        cls.rows[1:] = torch.eye(len(actions), dtype=dtype)
        cls.rest, cls.poses = {}, {}
        for name, body in cls.bodies.items():
            with torch.no_grad():
                out = cls.model(facial_actions=cls.rows, **cls.batched(body))
            vertices = out["rest_vertices"].numpy()
            B = out["rest_bone_poses"][0].numpy()
            cls.rest[name] = (vertices[0], vertices[1:] - vertices[0], B)
            cls.poses[name] = tpose.vrm_t_pose(cls.model, torch.from_numpy(B)).numpy()
        cls.file_weights, cls.file_indices, _ = top_skin_weights(
            cls.model.vertex_bone_weights.numpy(),
            cls.model.vertex_bone_indices.numpy(),
            4,
        )
        cls.binds = {}
        for method in ("forward", "inverse"):
            base, offsets, B = cls.rest["fixture"]
            start = time.perf_counter()
            cls.binds[method] = tpose.rebind(
                cls.model,
                base,
                offsets,
                B,
                cls.file_weights,
                cls.file_indices,
                method=method,
            )
            print(
                f"\nrebind {method}: {time.perf_counter() - start:.3f} s "
                f"({len(base)} vertices, {len(offsets)} targets)"
            )

    @classmethod
    def batched(cls, body):
        """The model kwargs of a body, with face-shape values for every facial-action row."""
        body = dict(body)
        if body.get("face_shape_kwargs"):
            body["face_shape_kwargs"] = {
                k: torch.full((len(cls.rows),), float(v), dtype=cls.rows.dtype)
                for k, v in body["face_shape_kwargs"].items()
            }
        return body

    def bone(self, name):
        return self.labels.index(name)

    def heads(self, body):
        return self.poses[body][:, :3, 3]

    def turns(self, body):
        """U_j = Q_j R_j^T, the world turn of each bone from its rest pose."""
        B = self.rest[body][2]
        return self.poses[body][:, :3, :3] @ np.swapaxes(B[:, :3, :3], 1, 2)

    def direction(self, body, start, end):
        h = self.heads(body)
        return h[self.bone(end)] - h[self.bone(start)]

    def test_shared_root_heads(self):
        # In the cached anny rig, the hips of a VRM file (root) share their head with spine05
        # and both pelvis bones.
        for body in self.bodies:
            B = self.rest[body][2]
            for name in ("spine05", "pelvis.L", "pelvis.R"):
                with self.subTest(body=body, bone=name):
                    np.testing.assert_allclose(
                        B[self.bone(name), :3, 3],
                        B[self.bone("root"), :3, 3],
                        atol=1e-9,
                    )

    def test_kept_bones(self):
        for body in self.bodies:
            B = self.rest[body][2]
            for name in tpose.KEPT_BONES:
                with self.subTest(body=body, bone=name):
                    np.testing.assert_allclose(
                        self.poses[body][self.bone(name)],
                        B[self.bone(name)],
                        atol=1e-12,
                    )

    def test_forward_kinematics(self):
        # Every child head is T_p B_p^-1 h, and Anny's own world-orient pass gives the same
        # bone poses.
        for name, body in self.bodies.items():
            B = self.rest[name][2]
            T = self.poses[name]
            X = T @ tpose.rigid_inverse(B)
            for j, p in enumerate(self.parents):
                if p >= 0:
                    h = X[p] @ np.append(B[j, :3, 3], 1.0)
                    np.testing.assert_allclose(T[j, :3, 3], h[:3], atol=1e-12)
            with torch.no_grad():
                out = self.model(
                    pose_parameters=torch.from_numpy(T)[None],
                    pose_parameterization="world-orient",
                    **body,
                )
            np.testing.assert_allclose(out["bone_poses"][0].numpy(), T, atol=1e-12)

    def test_arms_and_fingers_along_x(self):
        for body in self.bodies:
            for side, s in SIDES.items():
                along = np.array([s, 0.0, 0.0])
                chains = [
                    ("upperarm01", "lowerarm01"),
                    ("lowerarm01", "wrist"),
                ] + [
                    (f"finger{k}-{i}", f"finger{k}-{i + 1}")
                    for k in tpose.FINGERS
                    for i in (1, 2)
                ]
                for start, end in chains:
                    with self.subTest(body=body, bone=f"{start}.{side}"):
                        d = self.direction(body, f"{start}.{side}", f"{end}.{side}")
                        self.assertLess(angle(d, along), 0.1)
                with self.subTest(body=body, bone=f"hand.{side}"):
                    hand, _ = tpose.palm(self.heads(body), self.labels, side)
                    self.assertLess(angle(hand, along), 0.1)

    def test_twist_and_distal_bones_follow(self):
        # The twist bones turn with the bone they twist, the metacarpals with the hand, and the
        # last bone of each finger with the bone before it.
        for body in self.bodies:
            U = self.turns(body)
            for side in SIDES:
                pairs = [
                    ("upperarm02", "upperarm01"),
                    ("lowerarm02", "lowerarm01"),
                    ("finger1-3", "finger1-2"),
                ]
                pairs += [(f"metacarpal{k - 1}", "wrist") for k in tpose.FINGERS]
                pairs += [(f"finger{k}-3", f"finger{k}-2") for k in tpose.FINGERS]
                for bone, leader in pairs:
                    with self.subTest(body=body, bone=f"{bone}.{side}"):
                        np.testing.assert_allclose(
                            U[self.bone(f"{bone}.{side}")],
                            U[self.bone(f"{leader}.{side}")],
                            atol=1e-12,
                        )

    def test_palms_down_nails_up(self):
        for body in self.bodies:
            rest_heads = self.rest[body][2][:, :3, 3]
            U = self.turns(body)
            for side in SIDES:
                with self.subTest(body=body, side=side, part="palm"):
                    _, back = tpose.palm(self.heads(body), self.labels, side)
                    self.assertLess(angle(-back, -UP), 1.0)
                # The nail of each phalanx: the nail of the bone before it (the back of the
                # hand for the first), tilted to stay perpendicular to the phalanx at rest.
                _, rest_back = tpose.palm(rest_heads, self.labels, side)
                for k in tpose.FINGERS:
                    nail = rest_back
                    for i in (1, 2):
                        j = self.bone(f"finger{k}-{i}.{side}")
                        d = unit(
                            rest_heads[self.bone(f"finger{k}-{i + 1}.{side}")]
                            - rest_heads[j]
                        )
                        nail = unit(nail - (nail @ d) * d)
                        with self.subTest(body=body, bone=self.labels[j], part="nail"):
                            self.assertLess(angle(U[j] @ nail, UP), 1.0)

    def test_thumbs(self):
        for body in self.bodies:
            for side, s in SIDES.items():
                for i in (1, 2):
                    with self.subTest(body=body, bone=f"finger1-{i}.{side}"):
                        d = self.direction(
                            body, f"finger1-{i}.{side}", f"finger1-{i + 1}.{side}"
                        )
                        level = np.degrees(np.arcsin(d[2] / np.linalg.norm(d)))
                        self.assertLess(abs(level), 0.1)
                        toward_front = np.degrees(np.arctan2(-d[1], s * d[0]))
                        self.assertLess(abs(toward_front - 45.0), 0.5)

    def test_feet(self):
        for body in self.bodies:
            rest_heads = self.rest[body][2][:, :3, 3]
            U = self.turns(body)
            for side in SIDES:
                with self.subTest(body=body, side=side):
                    d = self.direction(body, f"foot.{side}", f"toe3-1.{side}")
                    self.assertLess(angle(d * [1.0, 1.0, 0.0], [0.0, -1.0, 0.0]), 0.1)
                    r = (
                        rest_heads[self.bone(f"toe3-1.{side}")]
                        - rest_heads[self.bone(f"foot.{side}")]
                    )
                    self.assertAlmostEqual(
                        d[2] / np.linalg.norm(d), r[2] / np.linalg.norm(r), places=12
                    )
                    foot = U[self.bone(f"foot.{side}")]
                    np.testing.assert_allclose(
                        foot @ UP, UP, atol=1e-12
                    )  # a turn about Z
                    for j, label in enumerate(self.labels):
                        if label.startswith("toe") and label.endswith(side):
                            np.testing.assert_allclose(U[j], foot, atol=1e-12)

    def test_legs(self):
        for body in self.bodies:
            B = self.rest[body][2]
            spread = tpose.vrm_t_pose(
                self.model, torch.from_numpy(B), keep_leg_spread=True
            ).numpy()
            for side in SIDES:
                hip, ankle = f"upperleg01.{side}", f"foot.{side}"
                with self.subTest(body=body, side=side):
                    d = self.direction(body, hip, ankle)
                    self.assertLess(angle(d, -UP), 0.1)
                with self.subTest(body=body, side=side, keep_leg_spread=True):
                    rest = B[self.bone(ankle), :3, 3] - B[self.bone(hip), :3, 3]
                    kept = (
                        spread[self.bone(ankle), :3, 3] - spread[self.bone(hip), :3, 3]
                    )
                    np.testing.assert_allclose(kept, rest, atol=1e-12)
                    self.assertGreater(angle(kept, -UP), 3.0)
                    for name in (
                        "upperleg01",
                        "upperleg02",
                        "lowerleg01",
                        "lowerleg02",
                    ):
                        j = self.bone(f"{name}.{side}")
                        np.testing.assert_allclose(spread[j], B[j], atol=1e-12)

    def test_symmetry(self):
        mirror = np.array([-1.0, 1.0, 1.0])
        for body in self.bodies:
            h = self.heads(body)
            for j, label in enumerate(self.labels):
                if label.endswith(".L"):
                    with self.subTest(body=body, bone=label):
                        np.testing.assert_allclose(
                            h[self.bone(label[:-2] + ".R")], h[j] * mirror, atol=1e-6
                        )

    def test_every_phenotype_corner(self):
        # The directions hold at every corner of the phenotype space. (Some corners are not
        # symmetric at rest, so the symmetry check stays with the two bodies above.)
        names = list(self.model.phenotype_labels)
        corners = np.array(list(itertools.product((0.0, 1.0), repeat=len(names))))
        with torch.no_grad():
            out = self.model(
                phenotype_kwargs={
                    n: torch.from_numpy(corners[:, i]) for i, n in enumerate(names)
                }
            )
        for c, B in enumerate(out["rest_bone_poses"].numpy()):
            h = tpose.vrm_t_pose(self.model, torch.from_numpy(B)).numpy()[:, :3, 3]

            def d(start, end):
                return h[self.bone(end)] - h[self.bone(start)]

            for side, s in SIDES.items():
                hand, back = tpose.palm(h, self.labels, side)
                angles = {
                    "upper arm": angle(
                        d(f"upperarm01.{side}", f"lowerarm01.{side}"), [s, 0, 0]
                    ),
                    "forearm": angle(
                        d(f"lowerarm01.{side}", f"wrist.{side}"), [s, 0, 0]
                    ),
                    "hand": angle(hand, [s, 0, 0]),
                    "index": angle(
                        d(f"finger2-1.{side}", f"finger2-2.{side}"), [s, 0, 0]
                    ),
                    "little": angle(
                        d(f"finger5-2.{side}", f"finger5-3.{side}"), [s, 0, 0]
                    ),
                    "thumb": angle(
                        d(f"finger1-1.{side}", f"finger1-2.{side}"), [s, -1, 0]
                    ),
                    "leg": angle(d(f"upperleg01.{side}", f"foot.{side}"), -UP),
                    "foot": angle(
                        d(f"foot.{side}", f"toe3-1.{side}") * [1, 1, 0], [0, -1, 0]
                    ),
                }
                with self.subTest(body=dict(zip(names, corners[c])), side=side):
                    self.assertLess(angle(back, UP), 1.0)
                    for part, value in angles.items():
                        self.assertLess(value, 0.1, part)

    def test_forward_bind_is_anny_forward_pass(self):
        rb = self.binds["forward"]
        np.testing.assert_allclose(rb.bone_poses, self.poses["fixture"], atol=1e-12)
        with torch.no_grad():
            out = self.model(
                pose_parameters=torch.from_numpy(rb.bone_poses)[None],
                pose_parameterization="world",
                facial_actions=self.rows,
                **self.batched(self.bodies["fixture"]),
            )
        v = out["vertices"].numpy()
        np.testing.assert_allclose(rb.vertices, v[0], atol=1e-8)
        np.testing.assert_allclose(rb.vertices + rb.targets, v[1:], atol=1e-5)

    def test_face_targets_barely_turn(self):
        # |(A_i - I) delta| on the vertices that the facial actions move: the jaw and mouth
        # reach neck vertices with a little upper-arm weight.
        offsets = self.rest["fixture"][1]
        for method, rb in self.binds.items():
            with self.subTest(method=method):
                change = np.linalg.norm(rb.targets - offsets, axis=-1).max()
                print(
                    f"\n{method} bind: largest change of a face target {change * 1e3:.6f} mm"
                )
                self.assertLess(change, 5e-5)

    def test_inverse_bind_exact_at_rest(self):
        base, offsets, _ = self.rest["fixture"]
        rb = self.binds["inverse"]
        back = tpose.rigid_inverse(rb.transforms)
        np.testing.assert_allclose(
            tpose.skin(rb.vertices, self.file_weights, self.file_indices, back),
            base,
            atol=1e-10,
        )
        linear = tpose.blend(self.file_weights, self.file_indices, back)[:, :3, :3]
        np.testing.assert_allclose(
            np.einsum("vab,rvb->rva", linear, rb.targets), offsets, atol=1e-12
        )
        # The untruncated weights give the same bind: rebind keeps the file's 4 (up to the
        # float32 rounding of the renormalised weights).
        same = tpose.rebind(
            self.model,
            base,
            offsets[:0],
            self.rest["fixture"][2],
            self.model.vertex_bone_weights.numpy(),
            self.model.vertex_bone_indices.numpy(),
            method="inverse",
        )
        np.testing.assert_allclose(same.vertices, rb.vertices, atol=1e-7)

    def test_joints_and_centring(self):
        for method, rb in self.binds.items():
            with self.subTest(method=method):
                np.testing.assert_allclose(
                    rb.joint_positions, rb.bone_poses[:, :3, 3], atol=0
                )
                hips = rb.joint_positions[self.bone("root")] + rb.offset
                np.testing.assert_allclose(hips[:2], 0.0, atol=1e-12)
                self.assertAlmostEqual(
                    float((rb.vertices + rb.offset)[:, 2].min()), 0.0, places=12
                )

    def test_file_pose(self):
        # The joint matrices of the file in glTF's frame: identity rotations at the bind, and
        # the file's skin of a posed body equals G times Anny's skin of the bind mesh, plus o.
        from opensculptboy.export.body import ANNY_TO_GLTF as G

        rb = self.binds["inverse"]
        P = rb.joint_positions @ G.T + G @ rb.offset
        bind = tpose.file_pose(rb.bone_poses, rb)
        np.testing.assert_allclose(
            bind[:, :3, :3], np.tile(np.eye(3), (len(P), 1, 1)), atol=1e-12
        )
        np.testing.assert_allclose(bind[:, :3, 3], P, atol=1e-12)
        M = rb.rest_bone_poses
        W = tpose.file_pose(M, rb)
        inverse_bind = np.tile(np.eye(4), (len(P), 1, 1))
        inverse_bind[:, :3, 3] = -P
        file_vertices = rb.vertices @ G.T + G @ rb.offset
        # Weights that sum to 1 in float64, so that the offset o passes through exactly.
        weights = self.file_weights.astype(np.float64)
        weights /= weights.sum(axis=1, keepdims=True)
        skinned = tpose.skin(
            file_vertices, weights, self.file_indices, W @ inverse_bind
        )
        expected = tpose.skin(
            rb.vertices,
            weights,
            self.file_indices,
            M @ tpose.rigid_inverse(rb.bone_poses),
        )
        np.testing.assert_allclose(skinned, expected @ G.T + G @ rb.offset, atol=1e-12)

    def test_unknown_rig_and_method(self):
        with self.assertRaises(ValueError):
            tpose.t_pose_turns(["root"], [-1], np.eye(4)[None])
        base, offsets, B = self.rest["default"]
        with self.assertRaises(ValueError):
            tpose.rebind(
                self.model,
                base,
                offsets,
                B,
                self.file_weights,
                self.file_indices,
                method="dqs",
            )

    def test_bind_error(self):
        # The error report of each bind on the arms-down poses: the file's 4 weights skin the
        # bind mesh through W_j T_j^-1, against Anny's own posed vertices.
        report = {}
        for method in ("forward", "inverse"):
            start = time.perf_counter()
            report[method] = tpose.bind_error(self.model, method)
            seconds = time.perf_counter() - start
            for name, e in report[method].items():
                print(
                    f"\n{method} bind, {name} ({e['frames']} frames): max {e['max_mm']:.2f} mm, "
                    f"99th percentile {e['p99_mm']:.2f} mm, mean {e['mean_mm']:.3f} mm"
                )
            print(f"{method} bind: error report in {seconds:.2f} s")
            for name, e in report[method].items():
                with self.subTest(method=method, pose=name):
                    self.assertLess(e["p99_mm"], 30.0)
                    self.assertLess(e["max_mm"], 60.0)
        # The inverse bind stays the closer one on these poses, which makes it the default
        # that the VRM export should keep.
        for name in report["inverse"]:
            with self.subTest(pose=name):
                self.assertLess(
                    report["inverse"][name]["p99_mm"], report["forward"][name]["p99_mm"]
                )


if __name__ == "__main__":
    unittest.main()
