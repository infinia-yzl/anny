# Corporis
# Apache License, Version 2.0
"""The pose from a picture on bad landmarks: misplaced and noisy head points, a head larger
than Anny's (as drawn by AI models), a hand with impossible proportions and a far-off point."""

import math
import os
import pathlib
import unittest

import numpy as np
import torch

import anny.poses
from corporis import Character
from corporis.posing.head import face_mesh_rest
from corporis.posing.landmarks import BODY, Landmarks
from corporis.posing.refine import refine
from corporis.posing.retarget import Retargeter
from corporis.posing.skeleton import Skeleton, bend, rot
from test.markers import local_only

PIXELS = 500.0  # pixels per metre of the synthetic pictures
N_HEAD = 11


def angle(a, b) -> float:
    """degrees between two rotations"""
    R = np.asarray(a, np.float64).T @ np.asarray(b, np.float64)
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def picture(points: np.ndarray) -> np.ndarray:
    """pixel positions (x right, y down) of points in the model's frame, seen along +y"""
    return np.stack([PIXELS * points[:, 0] + 400, 600 - PIXELS * points[:, 2]], axis=1)


class TestRobustPose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Character().build_model()
        cls.skeleton = Skeleton(cls.model)
        cls.retarget = Retargeter(cls.skeleton)
        labels = cls.skeleton.labels
        # a figure that tips its head back and turns it, as in a look up at a raised hand
        params = anny.poses.pose_parameters(cls.model, "relaxed", grounded=False)[
            "pose_parameters"
        ][:1].clone()
        h = labels.index("head")
        turn = (rot("z", -30) @ bend(35)).to(params.dtype)
        params[0, h, :3, :3] = params[0, h, :3, :3] @ turn
        with torch.no_grad():
            cls.out = cls.model(pose_parameters=params)
        # the head's turn from the rest pose, in the world
        posed = cls.out["bone_poses"][0, h, :3, :3].double().numpy()
        cls.truth = posed @ cls.skeleton.rest[h, :3, :3].double().numpy().T
        cls.clean = cls.retarget.anny.landmarks(cls.out)
        cls.clean.image = picture(cls.clean.body)
        cls.clean.visibility = np.ones(len(BODY))

    def corrupted(self) -> Landmarks:
        """the head as a drawing and a noisy picture give it: 1.3 times Anny's size, the hidden
        ear on the cheek and the hidden eye on the brow in 3D, and noise everywhere"""
        rng = np.random.default_rng(7)
        body = self.clean.body.copy()
        head = body[:N_HEAD]
        centre = head.mean(0)
        head[:] = centre + 1.3 * (head - centre)
        image = picture(body)
        p = lambda n: body[BODY.index(n)]  # noqa: E731
        body[BODY.index("right_ear")] = 0.5 * (p("nose") + p("right_eye_outer"))
        for n in ("right_eye_inner", "right_eye", "right_eye_outer", "mouth_right"):
            body[BODY.index(n)] += np.array([0.0, -0.03, 0.035])
        body += rng.normal(0, 0.004, body.shape)
        image += rng.normal(0, 1.5, image.shape)
        return Landmarks(
            body=body,
            visibility=np.ones(len(BODY)),
            image=image,
            hands={s: h.copy() for s, h in self.clean.hands.items()},
        )

    def test_clean_head_comes_from_the_points(self):
        choice = self.retarget.head_fit(self.clean)
        self.assertEqual(choice.source, "points")
        self.assertLess(angle(choice.rotation, self.truth), 0.5)

    def test_misplaced_head_points_leave_the_head_right(self):
        L = self.corrupted()
        choice = self.retarget.head_fit(L)
        self.assertEqual(choice.source, "picture")
        self.assertLess(angle(choice.rotation, self.truth), 10.0)
        W, _ = self.retarget(L)
        self.assertLess(angle(W["head"].double().numpy(), self.truth), 10.0)

    def test_face_mesh_sets_the_head(self):
        rng = np.random.default_rng(3)
        vertices = self.out["vertices"][0].double().numpy()
        mesh = face_mesh_rest(self.model, vertices) * PIXELS
        mesh += rng.normal(0, 0.5, mesh.shape)
        wrong = rng.choice(len(mesh), 40, replace=False)
        mesh[wrong] += rng.normal(0, 20.0, (len(wrong), 3))
        L = self.corrupted()
        L.face_points = mesh
        choice = self.retarget.head_fit(L)
        self.assertEqual(choice.source, "face")
        self.assertLess(angle(choice.rotation, self.truth), 3.0)

    def test_implausible_hand_keeps_the_fingers_at_rest(self):
        L = self.corrupted()
        W_good, _ = self.retarget(L)
        self.assertIn("finger2-2.L", W_good)
        hand = L.hands[".L"]
        hand[8] = hand[5] + 10 * (
            hand[8] - hand[5]
        )  # an index finger ten times too long
        W, _ = self.retarget(L)
        self.assertNotIn("finger2-2.L", W)
        self.assertIn("finger2-2.R", W)

    def test_refinement_resists_a_far_off_point(self):
        params = anny.poses.pose_parameters(self.model, "mh_hero", grounded=False)[
            "pose_parameters"
        ][:1]
        with torch.no_grad():
            out = self.model(pose_parameters=params)
        bp = out["bone_poses"][0]
        labels = self.skeleton.labels
        joints = ["lowerarm01.L", "wrist.L", "lowerarm01.R", "wrist.R", "head"]

        def centred(get):
            hips = (get("upperleg01.L") + get("upperleg01.R")) / 2
            return np.stack([get(j) - hips for j in joints])

        truth = centred(lambda j: bp[labels.index(j), :3, 3].numpy())

        def error(L):
            W, _ = self.retarget(L)
            posed = self.skeleton.posed_joints(refine(self.retarget, W, L))
            got = centred(lambda j: posed[j].numpy())
            return np.linalg.norm(got - truth, axis=1).mean()

        clean = error(self.retarget.anny.landmarks(out))
        L = self.retarget.anny.landmarks(out)
        L.body[BODY.index("left_knee")] += np.array([0.4, -0.3, 0.2])
        # the arms and the head stay where the clean landmarks put them
        self.assertLess(error(L), clean + 0.01)


@local_only("a picture outside the repository, named by CORPORIS_POSE_PICTURE")
@unittest.skipUnless(
    os.environ.get("CORPORIS_POSE_PICTURE"),
    "set CORPORIS_POSE_PICTURE to the floating figure with the raised right hand",
)
class TestReferencePicture(unittest.TestCase):
    """The logo's reference picture, an AI drawing on which MediaPipe misplaces the head
    points: the head must look up toward the raised right hand (the picture's left)."""

    def test_head_looks_up_at_the_raised_hand(self):
        from corporis.posing.landmarks import detect

        L = detect(pathlib.Path(os.environ["CORPORIS_POSE_PICTURE"]))
        retarget = Retargeter(Skeleton(Character().build_model()))
        choice = retarget.head_fit(L)
        self.assertIn(choice.source, ("face", "picture"))
        W, _ = retarget(L)
        forward = W["head"].double().numpy() @ np.array([0.0, -1.0, 0.0])
        self.assertGreater(forward[2], 0.5)  # the face points up
        self.assertLess(forward[0], -0.2)  # toward the picture's left


if __name__ == "__main__":
    unittest.main()
