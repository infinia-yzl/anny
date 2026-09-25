# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest

import roma
import torch

import anny
import anny.poses


class TestPoseLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny().to(dtype=torch.float64)
        cls.library = anny.poses.library()

    def test_contents(self):
        self.assertEqual(len(self.library.names("pose")), 50)
        self.assertEqual(
            self.library.names("loop"),
            ["idle", "walk", "run", "wave", "nod", "shrug", "jump"],
        )
        rotations, root = self.library.frames("walk")
        self.assertEqual(rotations.shape[1:], (len(self.library.bones), 4))
        self.assertEqual(root.shape, (rotations.shape[0], 3))
        norms = self.library.rotations.norm(dim=-1)
        self.assertLess((norms - 1).abs().max().item(), 1e-5)
        # the library uses the bone names of anny's rig
        self.assertTrue(set(self.model.bone_labels) <= set(self.library.bones))
        credited = [e for e in self.library.meta["entries"] if "credit" in e]
        self.assertEqual(len(credited), 40)

    def test_a_pose_is_the_rest_pose(self):
        result = anny.poses.pose_parameters(self.model, "a_pose")
        rest = self.model(phenotype_kwargs={})["rest_vertices"][0]
        rest = rest - torch.tensor(
            [0.0, 0.0, rest[:, 2].min().item()], dtype=rest.dtype
        )
        # the library keeps the root where it rests; only the grounding moves it
        offset = (result["vertices"][0] - rest).mean(0)
        self.assertLess(
            (result["vertices"][0] - rest - offset).abs().max().item(), 1e-5
        )

    def test_ground_and_stool(self):
        for age in (0.0, 0.5, 1.0):
            phenotype = {"age": torch.tensor([age], dtype=torch.float64)}
            pose = anny.poses.pose_parameters(
                self.model, "seated", phenotype_kwargs=phenotype
            )
            self.assertAlmostEqual(pose["vertices"][0, :, 2].min().item(), 0.0, 9)
            stool = pose["stool"]
            self.assertGreater(stool["top"], 0.05)
            self.assertLess(stool["top"], 0.8)
            output = self.model(
                pose_parameters=pose["pose_parameters"],
                phenotype_kwargs=phenotype,
                pose_parameterization="local-ref",
            )
            self.assertLess(
                (output["vertices"] - pose["vertices"]).abs().max().item(), 1e-9
            )

    def test_clip_keeps_one_floor(self):
        jump = anny.poses.pose_parameters(self.model, "jump")
        lowest = jump["vertices"][..., 2].min(dim=-1).values
        self.assertAlmostEqual(lowest.min().item(), 0.0, 9)
        # the figure leaves the floor in the air
        self.assertGreater(lowest.max().item(), 0.1)

    def test_rotations_follow_the_library(self):
        # an elbow bend of the library appears as the same bend in the posed skeleton
        pose = anny.poses.pose_parameters(self.model, "ready", grounded=False)
        output = self.model(
            pose_parameters=pose["pose_parameters"],
            phenotype_kwargs={},
            pose_parameterization="local-ref",
        )
        labels = list(self.model.bone_labels)
        rotations, _ = self.library.frames("ready")
        i = labels.index("lowerarm01.L")
        j = self.library.bones.index("lowerarm01.L")
        rest = self.model(phenotype_kwargs={})
        # world rotation applied to the geometry: bone pose times inverse rest pose
        world = output["bone_poses"][0, :, :3, :3] @ rest["rest_bone_poses"][
            0, :, :3, :3
        ].transpose(-1, -2)
        parent = labels.index("upperarm02.L")
        local = world[parent].T @ world[i]
        expected = roma.unitquat_to_rotmat(rotations[0, j].to(torch.float64))
        self.assertLess((local - expected).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
