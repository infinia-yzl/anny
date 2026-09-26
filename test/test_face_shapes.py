# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

import unittest

import torch

import anny
from anny.keypoints import KeypointsRegressor
from anny.models.face_shapes import SCALE_GROUPS, face_shape_spec


def _head_subtree_mask(model) -> torch.Tensor:
    """vertices with any skinning weight on the neck, the head or their descendants"""
    subtree = {model.bone_labels.index("neck01")}
    for i, parent in enumerate(model.bone_parents):
        if parent in subtree:
            subtree.add(i)
    in_subtree = torch.zeros(len(model.bone_labels), dtype=torch.bool)
    in_subtree[list(subtree)] = True
    weights = model.vertex_bone_weights * in_subtree[model.vertex_bone_indices]
    return weights.sum(-1) > 0


class TestFaceShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dtype = torch.float64
        cls.model = anny.Anny(face_shapes="all").to(dtype=dtype)
        cls.plain = anny.Anny().to(dtype=dtype)

    def test_spec(self):
        spec = face_shape_spec()
        names = [p.name for p in spec]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.model.face_shape_labels, names)
        self.assertEqual(sum(p.source == "makehuman" for p in spec), 103)
        self.assertEqual(
            [p.name for p in spec if p.source == "ict"][:2], ["detail-1", "detail-2"]
        )
        for p in spec:
            self.assertEqual(p.range[0] < 0, bool(p.negative) or p.source == "ict")

    def test_zero_values_leave_the_body(self):
        batch = 4
        phenotypes = {
            k: torch.rand(batch, dtype=torch.float64)
            for k in self.plain.phenotype_labels
        }
        for model, plain in [
            (self.model, self.plain),
            (
                anny.Anny(rig="makehuman", face_shapes="all").to(dtype=torch.float64),
                anny.Anny(rig="makehuman").to(dtype=torch.float64),
            ),
        ]:
            a = model(phenotype_kwargs=phenotypes)
            b = plain(phenotype_kwargs=phenotypes)
            for key in ("rest_vertices", "vertices", "bone_poses"):
                self.assertLess((a[key] - b[key]).abs().max().item(), 1e-10, key)

    def test_subset_matches_all(self):
        names = ["nose-hump", "head-round", "eye-scale"]
        subset = anny.Anny(face_shapes=names).to(dtype=torch.float64)
        self.assertEqual(
            subset.face_shape_labels,
            [n for n in self.model.face_shape_labels if n in names],
        )
        values = {"nose-hump": -0.7, "head-round": 0.6, "eye-scale": 0.4}
        a = subset(face_shape_kwargs=values)["rest_vertices"]
        b = self.model(face_shape_kwargs=values)["rest_vertices"]
        self.assertLess((a - b).abs().max().item(), 1e-10)
        with self.assertRaises(ValueError):
            anny.Anny(face_shapes=["no-such-shape"])

    def test_each_shape_moves_only_the_head(self):
        model = self.model
        outside = ~_head_subtree_mask(model)
        rest0 = model()["rest_vertices"][0]
        n = len(model.face_shape_labels)
        values = torch.eye(n, dtype=torch.float64)
        rest = model(face_shape_kwargs=values)["rest_vertices"]
        moved = (rest - rest0[None]).norm(dim=-1)
        self.assertTrue(torch.all(moved.max(dim=1).values > 1e-4))
        self.assertLess(moved[:, outside].max().item(), 1e-9)
        negative = model(face_shape_kwargs=-values)["rest_vertices"]
        moved = (negative - rest0[None]).norm(dim=-1)
        for i, name in enumerate(model.face_shape_labels):
            if model.face_shape_ranges[name][0] < 0:
                self.assertGreater(moved[i].max().item(), 1e-4, name)
            else:
                self.assertEqual(moved[i].max().item(), 0.0, name)

    def test_symmetry(self):
        model = self.model
        rest0 = model()["rest_vertices"][0]
        head = torch.nonzero(_head_subtree_mask(model)).squeeze(1)
        P = rest0[head]
        mirrored = P * torch.tensor([-1.0, 1.0, 1.0], dtype=P.dtype)
        dist, mirror = torch.cdist(mirrored, P).min(dim=1)
        pairs = dist < 1e-6
        self.assertGreater(pairs.float().mean().item(), 0.95)
        for name in (
            "eye-scale",
            "cheek-bones",
            "ear-lobe",
            "nose-scale-horiz",
            "detail-1",
        ):
            offsets = (
                model(face_shape_kwargs={name: 1.0})["rest_vertices"][0] - rest0
            )[head]
            flipped = offsets[mirror] * torch.tensor([-1.0, 1.0, 1.0], dtype=P.dtype)
            err = (offsets - flipped)[pairs].norm(dim=-1).max().item()
            self.assertLess(err, 2e-4, name)

    def test_scales(self):
        model = self.model
        default = model.face_shape_scales()
        self.assertEqual(default.shape, (1, len(SCALE_GROUPS)))
        self.assertLess((default - 1).abs().max().item(), 1e-12)
        ages = torch.tensor([0.0, 0.2, 0.4, 0.5], dtype=torch.float64)
        scales = model.face_shape_scales({"age": ages})
        self.assertTrue(torch.all(scales[1:] > scales[:-1]))
        self.assertTrue(torch.all(scales[0] < 0.8))

        values = {"nose-scale-vert": 1.0}
        young = {"age": 0.1}
        offsets = (
            model(phenotype_kwargs=young, face_shape_kwargs=values)["rest_vertices"]
            - model(phenotype_kwargs=young)["rest_vertices"]
        )
        unscaled = anny.Anny(face_shapes="all", scale_face_shapes=False).to(
            dtype=torch.float64
        )
        raw = (
            unscaled(phenotype_kwargs=young, face_shape_kwargs=values)["rest_vertices"]
            - unscaled(phenotype_kwargs=young)["rest_vertices"]
        )
        nose = SCALE_GROUPS.index("nose")
        ratio = offsets.norm(dim=-1).max() / raw.norm(dim=-1).max()
        expected = model.face_shape_scales(young)[0, nose]
        self.assertAlmostEqual(ratio.item(), expected.item(), places=6)
        self.assertLess(
            unscaled.face_shape_scales(young).sub(1).abs().max().item(), 1e-12
        )

    def test_gradient_at_zero(self):
        value = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        out = self.model(face_shape_kwargs={"nose-hump": value})
        out["rest_vertices"].sum().backward()
        self.assertNotEqual(value.grad.item(), 0.0)

    def test_landmarks_match_the_mesh(self):
        """build-time landmarks equal the regression of the rest mesh, for any phenotype"""
        model = self.model
        regressor = KeypointsRegressor.craniofacial(
            model, labels=model.craniofacial_landmark_labels
        ).to(dtype=torch.float64)
        phenotypes = {
            "age": torch.tensor([0.1, 0.8]),
            "gender": torch.tensor([0.9, 0.2]),
        }
        out = model(phenotype_kwargs=phenotypes)
        from_mesh = regressor({"vertices": out["rest_vertices"]})
        params = model._parse_parameter_kwargs(
            phenotypes, model.phenotype_labels, 0.5, "phenotype_kwargs"
        )
        from_data = model.phenotype_craniofacial_landmarks(params)
        self.assertLess((from_mesh - from_data).abs().max().item(), 1e-7)

    def test_other_topologies(self):
        for kwargs in (dict(topology="smplx"), dict(rig="soma", topology="soma")):
            model = anny.Anny(face_shapes=["nose-scale-horiz"], **kwargs).to(
                dtype=torch.float64
            )
            a = model()["rest_vertices"]
            b = model(face_shape_kwargs={"nose-scale-horiz": 1.0})["rest_vertices"]
            self.assertGreater((a - b).norm(dim=-1).max().item(), 1e-3, kwargs)


if __name__ == "__main__":
    unittest.main()
