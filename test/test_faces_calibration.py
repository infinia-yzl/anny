# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import importlib.util
import inspect
import unittest

import torch

import anny
from anny.faces.distribution import (
    DEFAULT_PATH,
    DEFAULT_SPREAD,
    FaceShapeDistribution,
    load_prior,
)
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


@unittest.skipUnless(DEFAULT_PATH.exists(), "the face prior is not built")
@unittest.skipUnless(
    importlib.util.find_spec("requests"),
    "the calibration needs the examples extra (requests)",
)
class TestPlausibleFaces(unittest.TestCase):
    """
    guards of the choices that keep random faces plausible (anny.faces.authoring.calibrate):
    renders showed faces that looked old and harsh while any of them was undone. A change that
    breaks one of these tests needs the review grid (python -m anny.faces.authoring.review) and
    the judgement of a person, not only new numbers.
    """

    @classmethod
    def setUpClass(cls):
        from anny.faces.authoring import calibrate

        cls.calibrate = calibrate
        cls.tensors, meta = load_prior()
        cls.labels = meta["face_labels"]
        cls.years = meta["anchor_years"]
        tril = cls.tensors["scale_tril"].double()
        cls.sd = (tril**2).sum(-1).sqrt()  # (A, 2, F)

    def test_means_move_the_skull_only(self):
        # means that moved the facial features gave big noses, forward chins and thin lips
        other = torch.tensor([k not in self.calibrate.MEAN_SHAPES for k in self.labels])
        self.assertEqual(self.tensors["mean"][..., other].abs().max().item(), 0.0)
        self.assertEqual(
            self.tensors["race_offsets"][..., other].abs().max().item(), 0.0
        )

    def test_fixed_shapes(self):
        for name in self.calibrate.FIXED_SHAPES:
            self.assertLess(self.sd[..., self.labels.index(name)].max().item(), 1e-4)

    def test_no_adult_detail_for_children(self):
        from anny.models.face_shapes import face_shape_parameter

        detail = [
            i
            for i, k in enumerate(self.labels)
            if face_shape_parameter(k).source == "ict"
        ]
        for a, years in enumerate(self.years):
            sd = self.sd[a][..., detail].max().item()
            if years < self.calibrate.DETAIL_YEARS:
                self.assertLess(sd, 1e-4, years)
            else:
                self.assertGreater(sd, 0.05, years)

    def test_spread_settings(self):
        # the spread of the ICT faces is an upper bound, and random faces draw at most 0.6 of it
        self.assertLessEqual(self.calibrate.VARIANCE_BOUNDS[1], 1.0)
        self.assertLessEqual(DEFAULT_SPREAD, 0.6)
        default = inspect.signature(FaceShapeDistribution.sample).parameters["spread"]
        self.assertEqual(default.default, DEFAULT_SPREAD)

    # the facial spread of the approved prior (commit d888932), in mm: the mean of the 1,000 largest
    # SDs of the vertices of 48 faces drawn at spread 1, with the skull-size shapes at their mean;
    # (anny age, gender) -> SD
    APPROVED_FACIAL_SPREAD = {
        (0.05, 0.0): 5.94,  # 1 year
        (0.05, 1.0): 5.82,
        (0.329, 0.0): 5.71,  # 8 years
        (0.329, 1.0): 4.04,
        (0.466, 0.0): 5.21,  # 12 years
        (0.466, 1.0): 6.44,
        (0.783, 0.0): 5.16,  # 28 years
        (0.783, 1.0): 4.91,
        (0.87, 0.0): 5.01,  # 75 years
        (0.87, 1.0): 4.64,
    }

    def test_facial_spread_does_not_grow(self):
        # wider spreads gave harsh faces: 12-year-olds of the first calibration varied by half as
        # much again as adults
        model = anny.Anny(face_shapes="all").to(dtype=torch.float64)
        dist = FaceShapeDistribution(model)
        skull = [model.face_shape_labels.index(k) for k in self.calibrate.MEAN_SHAPES]
        for (age, gender), approved in self.APPROVED_FACIAL_SPREAD.items():
            n = 48
            phen = {"age": torch.full((n,), age), "gender": torch.full((n,), gender)}
            g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                faces = dist.sample(phen, generator=g, spread=1.0)
                faces[:, skull] = dist.mean(phen)[:, skull]
                v = model(phenotype_kwargs=phen, face_shape_kwargs=faces)["vertices"]
            spread = 1000 * v.std(0).norm(dim=-1).topk(1000).values.mean().item()
            self.assertLessEqual(spread, 1.05 * approved, (age, gender))


if __name__ == "__main__":
    unittest.main()
