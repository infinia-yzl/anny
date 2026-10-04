# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the VRM T-pose and its rebind (opensculptboy.export.tpose): the directions of the
arms, hands, fingers, thumbs, legs and feet, the palms and the nails (against the mesh: the
palm side of the hand and MakeHuman's fingernail mask, in Anny's frame and in the frame of
each VRM version), the hand around the thumb without folds, the symmetry, the agreement of the
forward bind with Anny's own forward pass, the exactness of the inverse bind at Anny's rest
pose (positions, targets and normals), the pose error of both binds with the file's weights,
and the error of a file whose humanoid bones turn alone, with and without its roll
constraints.
"""

import itertools
import pathlib
import time
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

import anny
import anny.poses
from opensculptboy.export import tpose, vrm_tables
from opensculptboy.export.body import (
    top_skin_weights,
    triangulated_faces,
    vertex_normals,
)
from opensculptboy.export.vrm import FRAMES
from test import vrm_fixtures

SIDES = {"L": 1.0, "R": -1.0}
UP = np.array([0.0, 0.0, 1.0])
FINGERNAILS = (
    pathlib.Path(anny.__file__).parent
    / "data"
    / "mpfb2"
    / "textures"
    / "mpfb_fingernails.jpg"
)
# The phenotype corners of the error reports, besides the default body.
CORNERS = {
    "heavy tall": dict(gender=1.0, weight=1.0, muscle=1.0, height=1.0),
    "heavy": dict(gender=0.0, weight=1.0),
}
# The measured bind errors with the file's weights (the same for VRM 1.0 with twist
# "constraint" and VRM 0.x with "merge", within 0.02 mm): (body, method, pose) -> (max mm,
# 99th percentile mm, 99th percentile of the normal angle in degrees). The test holds each
# value between 0.9 and 1.1 times these.
MEASURED_BIND_ERROR = {
    ("default", "forward", "relaxed"): (21.08, 12.49, 28.21),
    ("default", "forward", "walk"): (21.45, 12.53, 29.33),
    ("default", "inverse", "relaxed"): (11.62, 8.45, 14.32),
    ("default", "inverse", "walk"): (14.28, 7.86, 18.13),
    ("heavy tall", "forward", "relaxed"): (26.05, 15.24, 27.99),
    ("heavy tall", "forward", "walk"): (25.94, 15.60, 31.27),
    ("heavy tall", "inverse", "relaxed"): (15.68, 11.11, 14.68),
    ("heavy tall", "inverse", "walk"): (19.27, 10.22, 18.80),
    ("heavy", "forward", "relaxed"): (24.42, 14.44, 30.35),
    ("heavy", "forward", "walk"): (24.24, 14.14, 32.69),
    ("heavy", "inverse", "relaxed"): (12.01, 8.71, 15.45),
    ("heavy", "inverse", "walk"): (13.70, 8.30, 19.94),
}
VERSIONS = {"1.0": "constraint", "0.x": "merge"}
# VRM T-pose definition 1.8 in Anny's frame (the figure faces -Y): the thumb nail faces
# (-s, -1, 0) / sqrt(2), toward the front and away from the hand; here times sqrt(2).
THUMB_NAIL_TARGET = {"L": (-1.0, -1.0, 0.0), "R": (1.0, -1.0, 0.0)}
# The same nail in the file, as the signs of its X and Z: for the left thumb between -X and +Z
# in VRM 1.0, and the mirrored signs in VRM 0.x, whose figure faces -Z.
FILE_NAIL_SIGNS = {
    "L": {"1.0": (-1.0, 1.0), "0.x": (1.0, -1.0)},
    "R": {"1.0": (1.0, 1.0), "0.x": (-1.0, -1.0)},
}
# The largest roll of the T-posed thumb nail from its target over the 64 corners of the six
# default phenotypes, measured on the mesh in degrees (on the corners at age 0; 1.24 at age 1,
# 0.72 at the median). tpose.THUMB_NAIL holds the nail of the default body, and the nails of
# other bodies sit a little off it. The test allows half a degree more.
CORNER_NAIL_ROLL = 4.05


def angle(a, b) -> float:
    """The angle between two directions, in degrees."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def unit(v):
    return v / np.linalg.norm(v)


def across(v, axis):
    """The part of ``v`` perpendicular to the unit ``axis``."""
    return v - (v @ axis) * axis


def adjacent_triangles(triangles):
    """The pairs of triangles (E, 2) that share an edge."""
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    faces = np.repeat(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, faces = edges[order], faces[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    return np.stack([faces[:-1][same], faces[1:][same]], axis=1)


def bends(vertices, triangles, pairs):
    """
    The dihedral angle of each pair of adjacent triangles, in degrees: the angle between their
    normals, 0 on a flat surface and 180 where the surface folds back onto itself.
    """
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    return tpose.angles(normals[pairs[:, 0]], normals[pairs[:, 1]])


def areas(vertices, triangles):
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    return np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0


class TestVrmTPose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = (
            vrm_fixtures.model()
        )  # float64, with every facial action and face shape
        cls.labels = list(cls.model.bone_labels)
        cls.parents = [int(p) for p in cls.model.bone_parents]
        cls.triangles, uv_triangles = triangulated_faces(cls.model)
        # The faces that MakeHuman's fingernail mask covers (the mean of its corners above
        # 0.5; the mask's rows run from v = 1 at the top).
        mask = np.asarray(Image.open(FINGERNAILS).convert("L"), dtype=np.float64) / 255
        uv = cls.model.texture_coordinates.numpy()[uv_triangles]
        rows, cols = mask.shape
        x = np.clip((uv[..., 0] * cols).astype(int), 0, cols - 1)
        y = np.clip(((1.0 - uv[..., 1]) * rows).astype(int), 0, rows - 1)
        cls.nail_faces = mask[y, x].mean(axis=1) > 0.5
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
        # The weights of a VRM 1.0 file (the eyelids moved to the head, the upper twist bones
        # merged), as the file stores them.
        cls.vrm_weights = vrm_tables.vrm_skin_weights(cls.model, "constraint")
        cls.file_weights, cls.file_indices, _ = top_skin_weights(*cls.vrm_weights, 4)
        # Anny's skin into the T-pose (the forward bind) and the inverse bind of each body,
        # without targets.
        cls.tposed, cls.inverse = (
            {
                name: tpose.rebind(
                    cls.model,
                    base,
                    offsets[:0],
                    B,
                    cls.file_weights,
                    cls.file_indices,
                    method=method,
                )
                for name, (base, offsets, B) in cls.rest.items()
            }
            for method in ("forward", "inverse")
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
        # The fixture body in the relaxed pose: Anny's vertices and world bone poses.
        params = anny.poses.pose_parameters(
            cls.model,
            "relaxed",
            phenotype_kwargs=cls.bodies["fixture"]["phenotype_kwargs"],
            grounded=False,
        )["pose_parameters"]
        with torch.no_grad():
            out = cls.model(
                pose_parameters=params,
                pose_parameterization="local-ref",
                **cls.bodies["fixture"],
            )
        cls.relaxed = (out["vertices"][0].numpy(), out["bone_poses"][0].numpy())

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

    def ancestors(self, j):
        """The bones above bone ``j`` in the rig."""
        above = []
        while self.parents[j] >= 0:
            j = self.parents[j]
            above.append(j)
        return above

    def heads(self, body):
        return self.poses[body][:, :3, 3]

    def turns(self, body):
        """U_j = Q_j R_j^T, the world turn of each bone from its rest pose."""
        B = self.rest[body][2]
        return self.poses[body][:, :3, :3] @ np.swapaxes(B[:, :3, :3], 1, 2)

    def direction(self, body, start, end):
        h = self.heads(body)
        return h[self.bone(end)] - h[self.bone(start)]

    def weight(self, *names):
        """The skin weight (Anny's weights) of each vertex on the named bones."""
        weights = self.model.vertex_bone_weights.numpy()
        indices = self.model.vertex_bone_indices.numpy()
        bones = [self.bone(name) for name in names]
        return np.where(np.isin(indices, bones), weights, 0.0).sum(axis=1)

    def nail_triangles(self, bone):
        """The triangles of the fingernail mask that ``bone`` carries with a weight above 0.9."""
        strong = self.weight(bone) > 0.9
        faces = self.triangles[self.nail_faces & strong[self.triangles].all(axis=1)]
        self.assertGreater(len(faces), 10, bone)
        return faces

    def nail(self, vertices, bone):
        """The area-weighted normal of the triangles of :meth:`nail_triangles`."""
        faces = self.nail_triangles(bone)
        a, b, c = (vertices[faces[:, k]] for k in range(3))
        return unit(np.cross(b - a, c - a).sum(axis=0))

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
        # last bone of each of the four fingers with the bone before it. The last bone of the
        # thumb turns with the bone before it and then rolls about the thumb's axis.
        for body in self.bodies:
            U = self.turns(body)
            for side, s in SIDES.items():
                pairs = [
                    ("upperarm02", "upperarm01"),
                    ("lowerarm02", "lowerarm01"),
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
                with self.subTest(body=body, bone=f"finger1-3.{side}"):
                    roll = (
                        U[self.bone(f"finger1-3.{side}")]
                        @ U[self.bone(f"finger1-2.{side}")].T
                    )
                    axis = np.array([s, -1.0, 0.0]) / np.sqrt(2.0)
                    np.testing.assert_allclose(roll @ axis, axis, atol=1e-12)
                    self.assertGreater(angle(roll[:, 2], [0.0, 0.0, 1.0]), 20.0)

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

    def test_palms_against_the_mesh(self):
        # Checks of the hands that do not use tpose.palm: in the A-pose the palms face the
        # thighs, so the palm side of the hand is the side of the metacarpals whose normals
        # point toward the body. In the T-pose that side faces down, and the thumb side of the
        # hand faces the front (the figure faces -Y): the index knuckle stands in front of the
        # little-finger knuckle, and the root of the thumb in front of the middle knuckle. (The
        # root of the thumb lies level with the index knuckle, within 2 mm.)
        hand_bones = {
            side: [f"metacarpal{k}.{side}" for k in range(1, 5)] for side in SIDES
        }
        for body in self.bodies:
            base, _, B = self.rest[body]
            rest_normals = vertex_normals(base, self.triangles)
            normals = vertex_normals(self.tposed[body].vertices, self.triangles)
            heads = self.heads(body)
            for side, s in SIDES.items():
                with self.subTest(body=body, side=side):
                    # The back of a hand that hangs at the side faces outward and up.
                    _, back = tpose.palm(B[:, :3, 3], self.labels, side)
                    self.assertGreater(s * back[0], 0.5)
                    self.assertGreater(back[2], 0.3)
                    hand = self.weight(*hand_bones[side]) > 0.9
                    palm_side = hand & (s * rest_normals[:, 0] < 0.0)
                    at_rest = rest_normals[palm_side].sum(axis=0)
                    down = normals[palm_side].sum(axis=0)
                    y = {
                        k: heads[self.bone(f"finger{k}-1.{side}"), 1] * 1000.0
                        for k in range(1, 6)
                    }
                    print(
                        f"\n{body} {side}: {palm_side.sum()} palm vertices, at rest "
                        f"{angle(at_rest, -back):.1f} degrees from the palm normal, in the "
                        f"T-pose {angle(down, -UP):.1f} degrees from down; y of the thumb "
                        "root and of the knuckles: "
                        + ", ".join(f"{v:.1f}" for v in y.values())
                        + " mm"
                    )
                    self.assertGreater(palm_side.sum(), 20)
                    self.assertLess(angle(at_rest, -back), 30.0)
                    self.assertLess(angle(down, -UP), 20.0)
                    self.assertLess(y[2], y[5] - 30.0)
                    self.assertLess(y[1], y[3] - 15.0)

    def test_thumb_nails(self):
        # VRM T-pose definition 1.8: the thumb nail faces a quarter turn from the other nails,
        # measured on the fingernail mask of the mesh against THUMB_NAIL_TARGET. The last thumb
        # bone keeps its rest bend, so the nail tilts toward the tip; its roll about the thumb
        # is what the definition fixes.
        meshes = {
            f"{body}, {method}": binds[body].vertices
            for method, binds in (("forward", self.tposed), ("inverse", self.inverse))
            for body in self.bodies
        }
        for mesh, vertices in meshes.items():
            for side, s in SIDES.items():
                with self.subTest(mesh=mesh, side=side):
                    nail = self.nail(vertices, f"finger1-3.{side}")
                    axis = np.array([s, -1.0, 0.0]) / np.sqrt(2.0)
                    target = np.array(THUMB_NAIL_TARGET[side]) / np.sqrt(2.0)
                    roll = angle(across(nail, axis), target)
                    # The nail in the file: for the left thumb between -X and +Z in VRM 1.0
                    # (the figure faces +Z), mirrored in VRM 0.x (the figure faces -Z).
                    for version, (sx, sz) in FILE_NAIL_SIGNS[side].items():
                        x, _, z = FRAMES[version] @ nail
                        self.assertGreater(sx * x, 0.3, version)
                        self.assertGreater(sz * z, 0.3, version)
                    fingers = [
                        angle(self.nail(vertices, f"finger{k}-3.{side}"), UP)
                        for k in tpose.FINGERS
                    ]
                    print(
                        f"\n{mesh} {side}: thumb nail {roll:.2f} degrees of roll from its "
                        f"target ({angle(nail, target):.1f} degrees with the tilt), "
                        f"{np.degrees(np.arcsin(nail[2])):.2f} degrees above level; the "
                        "other nails "
                        + ", ".join(f"{a:.1f}" for a in fingers)
                        + " degrees from up"
                    )
                    self.assertLess(roll, 2.0)
                    self.assertLess(abs(nail[2]), 0.05)
                    self.assertLess(max(fingers), 20.0)
        # At rest, tpose.thumb_nail is the nail of the mesh.
        for body in self.bodies:
            base, _, B = self.rest[body]
            for side in SIDES:
                with self.subTest(body=body, side=side, pose="rest"):
                    self.assertLess(
                        angle(
                            tpose.thumb_nail(B, self.labels, side),
                            self.nail(base, f"finger1-3.{side}"),
                        ),
                        5.0,
                    )

    def test_thumb_web(self):
        # The roll of the thumb leaves the hand without folds, in both binds: no edge of the
        # skin that the thumb bones, the index metacarpal or the wrist carry bends by more than
        # 120 degrees (96.6 at most on Anny's rest mesh), and the web between the thumb and
        # the index finger (the triangles that finger1-1 carries) keeps a quarter of its rest
        # area or more. A roll of the bones at the base of the thumb folds the web over (see
        # tpose.THUMB_ROLL). The phenotype corners stay out of this check: the aim of the
        # thumb alone, without a roll, folds the web of most corners at age 0.
        pairs = adjacent_triangles(self.triangles)
        meshes = {
            (body, method): binds[body].vertices
            for method, binds in (("forward", self.tposed), ("inverse", self.inverse))
            for body in self.bodies
        }
        for side in SIDES:
            hand = (
                self.weight(
                    *(f"finger1-{i}.{side}" for i in (1, 2, 3)),
                    f"metacarpal1.{side}",
                    f"wrist.{side}",
                )
                > 0.0
            )
            region = pairs[hand[self.triangles[pairs]].any(axis=(1, 2))]
            web = self.triangles[
                (self.weight(f"finger1-1.{side}") > 0.0)[self.triangles].any(axis=1)
            ]
            for (body, method), vertices in meshes.items():
                rest = self.rest[body][0]
                bend = bends(vertices, self.triangles, region)
                kept = areas(vertices, web) / areas(rest, web)
                print(
                    f"\n{body}, {method} bind, {side}: the hand bends by "
                    f"{bend.max():.1f} degrees at most over {len(region)} edges (at rest "
                    f"{bends(rest, self.triangles, region).max():.1f}); the web keeps "
                    f"{np.percentile(kept, 1):.2f} of its rest area at the 1st percentile "
                    f"and {kept.min():.2f} at least ({len(web)} triangles)"
                )
                with self.subTest(body=body, method=method, side=side):
                    self.assertLess(bend.max(), 120.0)
                    self.assertGreater(np.percentile(kept, 1), 0.35)
                    self.assertGreater(kept.min(), 0.25)

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
        # symmetric at rest, so the symmetry check stays with the two bodies above.) The thumb
        # nail is measured on the mesh, skinned into the T-pose by Anny's weights: the rest
        # frame of the last thumb bone holds the nail of the default body (tpose.THUMB_NAIL),
        # and the nails of other bodies roll a few degrees off it.
        names = list(self.model.phenotype_labels)
        corners = np.array(list(itertools.product((0.0, 1.0), repeat=len(names))))
        with torch.no_grad():
            out = self.model(
                phenotype_kwargs={
                    n: torch.from_numpy(corners[:, i]) for i, n in enumerate(names)
                }
            )
        weights = self.model.vertex_bone_weights.numpy()
        indices = self.model.vertex_bone_indices.numpy()
        nails = np.unique(
            np.concatenate([self.nail_triangles(f"finger1-3.{side}") for side in SIDES])
        )
        rolls = []
        for c, (rest, B) in enumerate(
            zip(out["rest_vertices"].numpy(), out["rest_bone_poses"].numpy())
        ):
            T = tpose.vrm_t_pose(self.model, torch.from_numpy(B)).numpy()
            h = T[:, :3, 3]
            vertices = rest.copy()
            vertices[nails] = tpose.skin(
                rest[nails],
                weights[nails],
                indices[nails],
                T @ tpose.rigid_inverse(B),
            )

            def d(start, end):
                return h[self.bone(end)] - h[self.bone(start)]

            for side, s in SIDES.items():
                hand, back = tpose.palm(h, self.labels, side)
                nail = self.nail(vertices, f"finger1-3.{side}")
                roll = angle(
                    across(nail, np.array([s, -1.0, 0.0]) / np.sqrt(2.0)),
                    np.array(THUMB_NAIL_TARGET[side]) / np.sqrt(2.0),
                )
                rolls.append(roll)
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
                    self.assertLess(roll, CORNER_NAIL_ROLL + 0.5, "thumb nail")
                    self.assertLess(abs(nail[2]), 0.1, "thumb nail")
        print(
            f"\nthumb nails of the {len(corners)} corners: {np.max(rolls):.2f} degrees of "
            f"roll from their target at most, {np.median(rolls):.2f} at the median"
        )

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
            *self.vrm_weights,
            method="inverse",
        )
        np.testing.assert_allclose(same.vertices, rb.vertices, atol=1e-7)

    def test_bind_normals(self):
        # The forward bind keeps the normals of its welded mesh. The inverse bind carries
        # Anny's rest normals by the map of its targets, so that the file, skinned as engines
        # skin normals, shows Anny's rest normals at Anny's rest pose.
        base, _, B = self.rest["fixture"]
        forward, inverse = self.binds["forward"], self.binds["inverse"]
        np.testing.assert_allclose(
            forward.normals,
            vertex_normals(forward.vertices, self.triangles),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(inverse.normals, axis=1), 1.0, atol=1e-12
        )
        rest_normals = vertex_normals(base, self.triangles)
        welded = vertex_normals(inverse.vertices, self.triangles)
        weights, indices = self.file_weights, self.file_indices
        back = tpose.rigid_inverse(inverse.transforms)
        at_rest = {
            "file": tpose.angles(
                tpose.skin_normals(inverse.normals, weights, indices, back),
                rest_normals,
            ),
            "welded bind mesh": tpose.angles(
                tpose.skin_normals(welded, weights, indices, back), rest_normals
            ),
        }
        # The relaxed pose, against the normals of Anny's posed mesh; Anny's own skin of its
        # rest normals shows how close skinned normals can come.
        vertices, W = self.relaxed
        posed = vertex_normals(vertices, self.triangles)
        transforms = W @ tpose.rigid_inverse(inverse.bone_poses)
        relaxed = {
            "file": tpose.angles(
                tpose.skin_normals(inverse.normals, weights, indices, transforms), posed
            ),
            "welded bind mesh": tpose.angles(
                tpose.skin_normals(welded, weights, indices, transforms), posed
            ),
            "Anny's skin of its rest normals": tpose.angles(
                tpose.skin_normals(
                    rest_normals,
                    self.model.vertex_bone_weights.numpy(),
                    self.model.vertex_bone_indices.numpy(),
                    W @ tpose.rigid_inverse(B),
                ),
                posed,
            ),
        }
        for name, a in at_rest.items():
            print(
                f"\ninverse bind, normals of the {name} at rest: within 1 degree for "
                f"{np.mean(a < 1.0) * 100:.2f} % of the vertices, max {a.max():.2f} degrees"
            )
        for name, a in relaxed.items():
            print(
                f"relaxed, {name}: 99th percentile {np.percentile(a, 99):.2f} degrees, "
                f"mean {a.mean():.2f}, over 45 degrees at {np.sum(a > 45)} vertices"
            )
        self.assertGreaterEqual(np.mean(at_rest["file"] < 1.0), 0.999)
        p99 = {name: np.percentile(a, 99) for name, a in relaxed.items()}
        self.assertLess(p99["file"], 18.0)
        self.assertLess(p99["file"], 0.6 * p99["welded bind mesh"])

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
        # The error report of each bind on the arms-down poses: the weights of the file (VRM
        # 1.0 with roll constraints, VRM 0.x with the twist bones merged) skin the bind mesh
        # through W_j T_j^-1, against Anny's own posed vertices and normals. The measured
        # values bound the errors from both sides, so that other weights (Anny's own strongest
        # 4, say) fail as surely as a worse bind, and the default weights of bind_error are
        # those of the file.
        bodies = {"default": None, **CORNERS}
        for version, twist in VERSIONS.items():
            for body, phenotype in bodies.items():
                report = {}
                for method in ("forward", "inverse"):
                    start = time.perf_counter()
                    with mock.patch.object(
                        tpose, "file_skin_weights", wraps=tpose.file_skin_weights
                    ) as weights:
                        report[method] = tpose.bind_error(
                            self.model,
                            method,
                            phenotype_kwargs=phenotype,
                            version=version,
                            twist=twist,
                        )
                    weights.assert_called_once_with(self.model, version, twist)
                    seconds = time.perf_counter() - start
                    for pose, e in report[method].items():
                        print(
                            f"\nVRM {version} ({twist}), {body} body, {method} bind, {pose} "
                            f"({e['frames']} frames, {seconds:.2f} s): max "
                            f"{e['max_mm']:.2f} mm, 99th percentile {e['p99_mm']:.2f} mm, "
                            f"mean {e['mean_mm']:.3f} mm; normals: 99th percentile "
                            f"{e['normal_p99_deg']:.2f} degrees, mean "
                            f"{e['normal_mean_deg']:.2f}"
                        )
                        measured = MEASURED_BIND_ERROR[(body, method, pose)]
                        keys = ("max_mm", "p99_mm", "normal_p99_deg")
                        for key, value in zip(keys, measured):
                            with self.subTest(
                                version=version,
                                body=body,
                                method=method,
                                pose=pose,
                                key=key,
                            ):
                                self.assertGreater(e[key], 0.9 * value)
                                self.assertLess(e[key], 1.1 * value)
                # The inverse bind stays the closer one on these poses, which makes it the
                # default of the VRM export.
                for pose in report["inverse"]:
                    with self.subTest(version=version, body=body, pose=pose):
                        for key in ("p99_mm", "max_mm", "normal_p99_deg"):
                            self.assertLess(
                                report["inverse"][pose][key],
                                report["forward"][pose][key],
                            )

    def test_humanoid_error(self):
        # A VRM app turns the humanoid bones alone. With their local rotations from Anny's
        # pose, the bones between them (spine05, spine03, the neck, shoulder01, the pelvis
        # bones, the metacarpals) stay at rest; with their world rotations ("folded"), only
        # the leaves (the toes) and the twist bones miss Anny's turns.
        for version, twist in VERSIONS.items():
            start = time.perf_counter()
            report = tpose.humanoid_error(self.model, version, twist)
            seconds = time.perf_counter() - start
            for pose, variants in report.items():
                for variant, e in variants.items():
                    print(
                        f"\nVRM {version} ({twist}), {pose} ({e['frames']} frames, "
                        f"{seconds:.2f} s), {variant}: max {e['max_mm']:.2f} mm, 99th "
                        f"percentile {e['p99_mm']:.2f} mm, mean {e['mean_mm']:.3f} mm"
                    )
                every, humanoid, folded = (
                    variants[v] for v in ("every_node", "humanoid", "humanoid_folded")
                )
                with self.subTest(version=version, pose=pose):
                    top, p99, _ = MEASURED_BIND_ERROR[("default", "inverse", pose)]
                    self.assertLess(every["max_mm"], 1.2 * top)
                    self.assertLess(every["p99_mm"], 1.2 * p99)
                    self.assertLess(humanoid["p99_mm"], 60.0)
                    self.assertLess(humanoid["mean_mm"], 25.0)
                    self.assertLess(folded["p99_mm"], 25.0)
                    self.assertLess(folded["mean_mm"], 6.0)
                    self.assertLess(folded["p99_mm"], humanoid["p99_mm"])
                    self.assertLess(every["p99_mm"], folded["p99_mm"])

    def test_humanoid_error_of_a_turned_forearm(self):
        # The library poses barely turn the hands about the forearms, so the roll constraints
        # hardly matter there. Here the hands of the relaxed pose turn by 60 degrees about the
        # forearms, and the forearm twist bones by 0.6 of that, as anny's pose authoring
        # shares a turn of the forearm (poses/authoring/poselib.py, arm_fk). On the skin that
        # the twist bones carry, an app that applies the roll constraints of the VRM 1.0 file
        # lands closer to Anny's posed mesh than an app that ignores them or rolls about a
        # wrong axis.
        params = anny.poses.pose_parameters(self.model, "relaxed", grounded=False)
        with torch.no_grad():
            out = self.model(
                pose_parameters=params["pose_parameters"],
                pose_parameterization="local-ref",
            )
        W = out["bone_poses"].numpy().copy()
        for side, s in SIDES.items():
            wrist = self.bone(f"wrist.{side}")
            start = W[0, self.bone(f"lowerarm01.{side}"), :3, 3]
            axis = unit(W[0, wrist, :3, 3] - start)
            hand = [wrist] + [
                j for j in range(len(self.labels)) if wrist in self.ancestors(j)
            ]
            for bones, share in ((hand, 1.0), ([self.bone(f"lowerarm02.{side}")], 0.6)):
                turn = np.eye(4)
                turn[:3, :3] = tpose._axis_angle(axis, s * np.radians(60.0) * share)
                turn[:3, 3] = start - turn[:3, :3] @ start
                W[:, bones] = turn @ W[:, bones]
        file = vrm_tables.twist_constraints("1.0", "constraint")
        apps = {
            "constraints": None,
            "no constraints": [],
            "roll about Y": [(b, src, "Y", w) for b, src, _, w in file],
            "roll about Z": [(b, src, "Z", w) for b, src, _, w in file],
        }
        forearms = self.weight("lowerarm02.L", "lowerarm02.R") > 0.0
        errors = {}
        for app, constraints in apps.items():
            report = tpose.humanoid_error(
                self.model,
                "1.0",
                "constraint",
                pose_names=(),
                poses={"turned": W},
                constraints=constraints,
                region=forearms,
            )["turned"]
            errors[app] = report
            for variant in ("humanoid", "humanoid_folded"):
                e = report[variant]
                print(
                    f"\nturned forearms, {app}, {variant}: max {e['max_mm']:.2f} mm, 99th "
                    f"percentile {e['p99_mm']:.2f} mm, mean {e['mean_mm']:.3f} mm over "
                    f"{forearms.sum()} vertices"
                )
        best = errors.pop("constraints")
        for app, report in errors.items():
            with self.subTest(app=app):
                for key in ("p99_mm", "mean_mm"):
                    self.assertLess(
                        best["humanoid_folded"][key],
                        0.5 * report["humanoid_folded"][key],
                    )
                self.assertLess(
                    best["humanoid"]["p99_mm"], report["humanoid"]["p99_mm"]
                )

    def test_roll_constraint(self):
        # The roll of a source about the axis, after the smallest turn that brings the axis
        # back, at the constraint's weight.
        axis = np.array([1.0, 0.0, 0.0])
        roll = tpose._axis_angle(axis, 0.8)
        swing = tpose._axis_angle(unit(np.array([0.0, 0.6, 0.8])), 0.5)
        half = tpose._axis_angle(axis, 0.4)
        np.testing.assert_allclose(
            tpose.roll_constraint(roll, axis, 0.5), half, atol=1e-12
        )
        np.testing.assert_allclose(
            tpose.roll_constraint(swing @ roll, axis, 0.5), half, atol=1e-12
        )
        np.testing.assert_allclose(
            tpose.roll_constraint(swing, axis, 1.0), np.eye(3), atol=1e-12
        )
        # A roll about the opposite axis is the same roll.
        np.testing.assert_allclose(
            tpose.roll_constraint(swing @ roll, -axis, 0.5), half, atol=1e-12
        )

    def test_humanoid_drive_of_every_node(self):
        # Driving every node of a file with the rig's hierarchy (VRM 0.x) by its local
        # rotation in Anny's pose gives the skin matrices W_j T_j^-1 back, up to the
        # orthonormality of Anny's posed rotations (about 1e-5: the pose library stores its
        # rotations in single precision).
        parents = [int(p) for p in vrm_tables.file_parents(self.model, "0.x", "merge")]
        self.assertEqual(parents, self.parents)
        rb = self.binds["inverse"]
        _, W = self.relaxed
        transforms = W @ tpose.rigid_inverse(rb.bone_poses)
        drive = tpose._humanoid_drive(
            transforms[:, :3, :3],
            W[:, :3, 3],
            rb.joint_positions,
            parents,
            tpose._parents_first(parents),
            dict(enumerate(parents)),
            [],
        )
        np.testing.assert_allclose(drive, transforms, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
