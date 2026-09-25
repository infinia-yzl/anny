# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest

import torch

import anny
from anny.faces.distribution import DEFAULT_PATH, FaceShapeDistribution
from anny.faces.measurements import (
    ANSUR_MEASUREMENTS,
    TDFN_MEASUREMENTS,
    CraniofacialMeasurements,
    landmark_measurements,
)


class TestCraniofacialMeasurements(unittest.TestCase):
    def test_measurements_on_the_makehuman_mesh(self):
        for kwargs in (dict(), dict(rig="makehuman", topology="makehuman")):
            model = anny.Anny(face_shapes="all", **kwargs).to(dtype=torch.float64)
            measure = CraniofacialMeasurements(model)
            out = model(
                phenotype_kwargs={
                    "age": torch.tensor([0.8, 0.8]),
                    "gender": torch.tensor([0.0, 1.0]),
                }
            )
            values = measure(out)
            for name in list(TDFN_MEASUREMENTS) + list(ANSUR_MEASUREMENTS):
                self.assertEqual(values[name].shape, (2,), name)
                self.assertTrue(torch.all(values[name] > 0), name)
            # adult sizes within broad human ranges (mm)
            self.assertTrue(
                torch.all((values["headbreadth"] > 130) & (values["headbreadth"] < 175))
            )
            self.assertTrue(
                torch.all(
                    (values["headcircumference"] > 500)
                    & (values["headcircumference"] < 650)
                )
            )
            self.assertTrue(
                torch.all(
                    (values["interpupillarybreadth"] > 50)
                    & (values["interpupillarybreadth"] < 75)
                )
            )
            # the landmarks of the mesh equal those of the landmark blend shapes
            from_data = model.rest_craniofacial_landmarks(
                {"age": torch.tensor([0.8, 0.8]), "gender": torch.tensor([0.0, 1.0])}
            )
            self.assertLess(
                (measure.landmarks(out) - from_data).abs().max().item(), 1e-7
            )

    def test_landmark_measurements_on_other_topologies(self):
        model = anny.Anny(topology="smplx", face_shapes="all").to(dtype=torch.float64)
        L = model.rest_craniofacial_landmarks(
            {"age": torch.tensor([0.8])}, {"nose-scale-horiz": torch.tensor([1.0])}
        )
        values = landmark_measurements(L, model.craniofacial_landmark_labels)
        base = landmark_measurements(
            model.rest_craniofacial_landmarks({"age": torch.tensor([0.8])}),
            model.craniofacial_landmark_labels,
        )
        self.assertGreater(values["nasalwidth"].item(), base["nasalwidth"].item() + 1.0)


@unittest.skipUnless(DEFAULT_PATH.exists(), "the face prior is not built")
class TestFaceShapeDistribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny(face_shapes="all").to(dtype=torch.float64)
        cls.dist = FaceShapeDistribution(cls.model)

    def test_samples(self):
        phen = {"age": torch.rand(64) * 0.3 + 0.55, "gender": torch.rand(64)}
        a = self.dist.sample(phen, generator=torch.Generator().manual_seed(3))
        b = self.dist.sample(phen, generator=torch.Generator().manual_seed(3))
        self.assertEqual(a.shape, (64, len(self.model.face_shape_labels)))
        self.assertTrue(torch.equal(a, b))
        for i, name in enumerate(self.model.face_shape_labels):
            lo, hi = self.model.face_shape_ranges[name]
            self.assertTrue(torch.all((a[:, i] >= lo) & (a[:, i] <= hi)), name)
        out = self.model(phenotype_kwargs=phen, face_shape_kwargs=a)
        self.assertTrue(torch.isfinite(out["vertices"]).all())

    def test_conditioning(self):
        male = self.dist.mean({"age": 0.8, "gender": 0.0})
        female = self.dist.mean({"age": 0.8, "gender": 1.0})
        child = self.dist.mean({"age": 0.3, "gender": 0.0})
        self.assertGreater((male - female).abs().max().item(), 1e-3)
        self.assertGreater((male - child).abs().max().item(), 1e-3)
        # halfway in gender is halfway between the two means (weight and muscle equal)
        mid = self.dist.mean({"age": 0.8, "gender": 0.5})
        self.assertLess((mid - 0.5 * (male + female)).abs().max().item(), 1e-5)

    def test_subset_model(self):
        names = ["nose-hump", "head-round", "eye-scale"]
        model = anny.Anny(face_shapes=names).to(dtype=torch.float64)
        dist = FaceShapeDistribution(model)
        full = self.dist.mean({"age": 0.8})
        idx = [self.model.face_shape_labels.index(k) for k in model.face_shape_labels]
        self.assertLess(
            (dist.mean({"age": 0.8}) - full[:, idx]).abs().max().item(), 1e-6
        )
        self.assertEqual(dist.sample(batch_size=5).shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
