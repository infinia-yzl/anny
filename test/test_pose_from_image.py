# OpenSculptBoy
# Apache License, Version 2.0
"""The pose from a picture: the character card, the views and MediaPipe on a drawn pose."""

import importlib.util
import math
import unittest

import numpy as np
import torch

import anny.poses
from opensculptboy import Character
from opensculptboy.posing.picture import flat_view, pose_from_landmarks, posed_mesh
from opensculptboy.posing.retarget import AnnyLandmarks
from opensculptboy.posing.skeleton import Skeleton, rot
from opensculptboy.render.flat import View, shaded_png
from test.markers import local_only

LIMBS = [
    (a + s, b + s)
    for a, b in (
        ("upperarm01", "lowerarm01"),
        ("lowerarm01", "wrist"),
        ("upperleg01", "lowerleg01"),
        ("lowerleg01", "foot"),
    )
    for s in (".L", ".R")
]


def limb_directions(model, pose_parameters):
    with torch.no_grad():
        bp = model(pose_parameters=pose_parameters)["bone_poses"][0]
    labels = list(model.bone_labels)
    out = {}
    for a, b in LIMBS:
        d = bp[labels.index(b), :3, 3] - bp[labels.index(a), :3, 3]
        out[a] = (d / d.norm()).double().numpy()
    return out


def angles(model, first, second, flat=False):
    """the angle (degrees) between each limb's direction in two poses; ``flat`` compares the
    directions in the front camera's picture (x, z), which one picture fixes"""
    a, b = limb_directions(model, first), limb_directions(model, second)
    if flat:
        a = {k: v[[0, 2]] / np.linalg.norm(v[[0, 2]]) for k, v in a.items()}
        b = {k: v[[0, 2]] / np.linalg.norm(v[[0, 2]]) for k, v in b.items()}
    return {k: math.degrees(math.acos(np.clip(a[k] @ b[k], -1, 1))) for k in a}


class TestPoseFromLandmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.character = Character(name="test")
        cls.model = cls.character.build_model()
        cls.anny = AnnyLandmarks(Skeleton(cls.model))

    def library(self, name):
        return anny.poses.pose_parameters(self.model, name, grounded=False)[
            "pose_parameters"
        ][:1]

    def test_card_holds_the_pose_and_the_face(self):
        params = self.library("mh_hero")
        with torch.no_grad():
            out = self.model(pose_parameters=params)
        landmarks = self.anny.landmarks(
            out, face={"jawOpen": 0.4, "browInnerUp": 0.001}
        )
        posed = pose_from_landmarks(landmarks, self.character, self.model, refine=False)
        self.assertEqual(posed.facial_actions, {"jawOpen": 0.4})
        err = angles(self.model, posed.pose_parameters(self.model), params)
        self.assertLess(max(err.values()), 10.0, err)
        vertices, faces = posed_mesh(posed, self.model)
        self.assertEqual(vertices.shape, (self.model.template_vertices.shape[0], 3))
        self.assertEqual(faces.shape[1], 3)

    def test_front_view_faces_the_turned_figure(self):
        pose = Character.pose_from_parameters(self.model, _turned(self.model, 70))
        character = Character(pose=pose)
        self.assertEqual(flat_view(character, "image"), View())
        front = flat_view(character, "front")
        # the figure faces -y turned 70 degrees about z; the front view turns with it
        forward = rot("z", 70).double().numpy() @ np.array([0.0, -1.0, 0.0])
        np.testing.assert_allclose(front.basis()[2], forward, atol=1e-6)
        quarter = flat_view(character, "three-quarter")
        self.assertAlmostEqual(quarter.yaw, front.yaw - 40.0)
        with self.assertRaises(ValueError):
            flat_view(character, "side")


def _turned(model, degrees):
    params = torch.eye(4).repeat(1, model.bone_count, 1, 1)
    params[0, 0, :3, :3] = rot("z", degrees)
    return params


@local_only("MediaPipe and its downloaded models")
@unittest.skipUnless(
    importlib.util.find_spec("mediapipe"), "needs the pose extra (MediaPipe)"
)
class TestPoseFromImage(unittest.TestCase):
    def test_drawn_pose_comes_back(self):
        from opensculptboy import pose_from_image

        character = Character(name="drawn")
        model = character.build_model()
        for name in ("mh_star", "mh_hero"):
            with self.subTest(pose=name):
                params = anny.poses.pose_parameters(model, name, grounded=False)[
                    "pose_parameters"
                ][:1]
                drawn = Character(pose=Character.pose_from_parameters(model, params))
                vertices, faces = posed_mesh(drawn, model)
                picture = shaded_png(vertices, faces, View(), size=(640, 800))
                background = picture.copy()
                background.paste((236, 238, 241, 255), (0, 0, *picture.size))
                background.alpha_composite(picture)
                posed = pose_from_image(background.convert("RGB"), character, model)
                err = angles(model, posed.pose_parameters(model), params, flat=True)
                self.assertLess(np.mean(list(err.values())), 8.0, err)
                self.assertLess(max(err.values()), 15.0, err)


if __name__ == "__main__":
    unittest.main()
