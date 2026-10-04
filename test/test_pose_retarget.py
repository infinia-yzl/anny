# OpenSculptBoy
# Apache License, Version 2.0
"""The retarget from landmarks: poses of the library come back from their own landmarks."""

import unittest

import numpy as np
import torch

import anny
import anny.poses
from opensculptboy.posing.refine import refine
from opensculptboy.posing.retarget import Retargeter
from opensculptboy.posing.skeleton import Skeleton

JOINTS = [
    j + s
    for j in (
        "upperarm01",
        "lowerarm01",
        "wrist",
        "upperleg01",
        "lowerleg01",
        "foot",
        "finger1-3",
        "finger2-3",
        "finger5-3",
    )
    for s in (".L", ".R")
] + ["head"]


def centred(joints):
    c = (joints["upperleg01.L"] + joints["upperleg01.R"]) / 2
    return {k: joints[k] - c for k in JOINTS}


class TestPoseRetarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = anny.Anny(facial_actions="all").to(torch.float32)
        cls.skeleton = Skeleton(model)
        cls.retarget = Retargeter(cls.skeleton)

    def pose(self, name):
        params = anny.poses.pose_parameters(self.skeleton.model, name, grounded=False)[
            "pose_parameters"
        ][:1]
        with torch.no_grad():
            out = self.skeleton.model(
                pose_parameters=params, phenotype_kwargs=self.skeleton.phenotype
            )
        bp = out["bone_poses"][0]
        joints = {n: bp[i, :3, 3] for i, n in enumerate(self.skeleton.labels)}
        return out, centred(joints)

    def error(self, W, reference):
        got = centred(self.skeleton.posed_joints(W))
        return np.array([float((got[k] - reference[k]).norm()) for k in JOINTS])

    def test_rest_pose_is_the_identity(self):
        W, _ = self.retarget(self.retarget.rest)
        for bone in ("root", "spine01", "head", "upperarm01.L", "lowerleg01.R"):
            self.assertLess((W[bone] - torch.eye(3)).abs().max().item(), 1e-4, bone)

    def test_library_poses_come_back(self):
        for name in (
            "mh_hero",
            "arms_crossed",
            "mh_thinking",
            "mh_star",
            "seated",
            "mh_cheer",
        ):
            with self.subTest(pose=name):
                out, reference = self.pose(name)
                W, _ = self.retarget(self.retarget.anny.landmarks(out))
                err = self.error(W, reference)
                self.assertLess(err.mean(), 0.015, name)  # metres
                self.assertLess(err.max(), 0.03, name)

    def test_refinement_bends_the_spine(self):
        out, reference = self.pose("mh_toe_touch")
        landmarks = self.retarget.anny.landmarks(out)
        W, _ = self.retarget(landmarks)
        before = self.error(W, reference).mean()
        after = self.error(refine(self.retarget, W, landmarks), reference).mean()
        self.assertLess(after, 0.8 * before)
        self.assertLess(after, 0.035)

    def test_face_scores_become_facial_actions(self):
        out, _ = self.pose("relaxed")
        scores = {"jawOpen": 0.6, "eyeBlinkLeft": 1.0, "_neutral": 0.2}
        _, face = self.retarget(self.retarget.anny.landmarks(out, face=scores))
        self.assertEqual(face, {"jawOpen": 0.6, "eyeBlinkLeft": 1.0})


if __name__ == "__main__":
    unittest.main()
