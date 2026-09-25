# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest

import roma
import torch

import anny
import anny.poses
from anny.correctives import (
    SoftTissueCorrectives,
    cone_key_weights,
    hinge_key_weights,
)


class TestDriverWeights(unittest.TestCase):
    def test_hinge_weights(self):
        keys = torch.tensor([[0.0, 45.0, 90.0, 130.0]]).repeat(6, 1)
        angles = torch.tensor([-10.0, 0.0, 30.0, 90.0, 110.0, 170.0])
        w = hinge_key_weights(angles, keys)
        self.assertTrue(torch.allclose(w.sum(1), torch.ones(6)))
        self.assertTrue((w >= 0).all())
        self.assertAlmostEqual(w[2, 0].item(), 1 / 3, 6)
        self.assertAlmostEqual(w[2, 1].item(), 2 / 3, 6)
        self.assertEqual(w[0, 0].item(), 1.0)
        self.assertEqual(w[5, 3].item(), 1.0)

    def test_cone_weights(self):
        torch.manual_seed(0)
        targets = torch.tensor(
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        triangles = torch.tensor([[0, 1, 2], [0, 2, 3]])
        # directions inside the cone of the keys (y >= 0 and z <= 0)
        d = torch.randn(50, 3)
        d[:, 1] = d[:, 1].abs()
        d[:, 2] = -d[:, 2].abs()
        d = torch.nn.functional.normalize(d, dim=-1)
        w = cone_key_weights(d, targets.expand(50, -1, -1), triangles)
        self.assertTrue(torch.allclose(w.sum(1), torch.ones(50, dtype=w.dtype)))
        self.assertTrue((w >= 0).all())
        # outside every triangle of keys the correction fades out
        w = cone_key_weights(torch.tensor([[0.0, -1.0, 0.0]]), targets[None], triangles)
        self.assertEqual(w.abs().max().item(), 0.0)
        # a key direction takes the full weight of its key
        w = cone_key_weights(targets[1:2], targets[None], triangles)
        self.assertAlmostEqual(w[0, 1].item(), 1.0, 6)


class TestSoftTissueCorrectives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny().to(dtype=torch.float64)
        cls.correctives = SoftTissueCorrectives(cls.model)

    def posed(self, bends, phenotype=None):
        """the rest pose of the library with some bones turned (degrees about world axes)"""
        library = anny.poses.library()
        rotations, root = library.frames("a_pose")
        rotations = rotations.clone()
        for bone, axis, degrees in bends:
            j = library.bones.index(bone)
            rotvec = torch.zeros(3)
            rotvec["xyz".index(axis)] = torch.deg2rad(torch.tensor(degrees))
            rotations[0, j] = roma.rotvec_to_unitquat(rotvec)
        params = anny.poses.to_local_ref(
            self.model, rotations, root, library.bones, phenotype_kwargs=phenotype
        )
        return self.model(
            pose_parameters=params,
            phenotype_kwargs=phenotype,
            pose_parameterization="local-ref",
        )

    def test_nothing_at_rest(self):
        output = self.posed([])
        corrected = self.correctives(output)
        self.assertLess(corrected["corrective_weights"].abs().max().item(), 1e-6)
        self.assertLess(
            (corrected["vertices"] - output["vertices"]).abs().max().item(), 1e-7
        )

    def test_left_and_right_mirror(self):
        # the same bend of both elbows (mirrored axes) gives the same weights on both sides
        output = self.posed([("lowerarm01.L", "z", 60.0), ("lowerarm01.R", "z", -60.0)])
        weights = self.correctives.shape_weights(output)[0]
        names = self.correctives.shape_names
        for i, name in enumerate(names):
            if name.startswith("elbow") and name.endswith(".L"):
                j = names.index(name[:-2] + ".R")
                self.assertAlmostEqual(weights[i].item(), weights[j].item(), 5)
        self.assertGreater(
            sum(
                weights[i].item() for i, n in enumerate(names) if n.startswith("elbow")
            ),
            0.5,
        )

    def test_correction_stays_on_its_side(self):
        output = self.posed([("upperleg01.L", "x", -80.0)])
        corrected = self.correctives(output)
        moved = (corrected["vertices"] - output["vertices"]).norm(dim=-1)[0]
        self.assertGreater(moved.max().item(), 0.005)
        rest = output["rest_vertices"][0]
        # anny's left side is +x; the correction of the left hip stays on that side (the fit
        # keeps to x > -3 cm on the authoring body, which is 3.4 cm on anny's default body)
        self.assertTrue((rest[moved > 1e-4, 0] > -0.035).all())

    def test_follows_the_body(self):
        # the shapes scale with the size of the body around them
        scales_child = self.correctives.shape_scales(
            self.model(phenotype_kwargs={"age": 0.0})["rest_vertices"]
        )
        scales_default = self.correctives.shape_scales(
            self.model(phenotype_kwargs={})["rest_vertices"]
        )
        self.assertLess((scales_default - 1).abs().max().item(), 1e-6)
        self.assertLess(scales_child.max().item(), 0.8)


if __name__ == "__main__":
    unittest.main()
