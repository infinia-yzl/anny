# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

"""Test that retopologized Anny models remain left/right symmetric."""

import unittest

import torch

import anny
from anny.models.model_transforms import (
    _get_symmetric_bone_name,
    _project_onto_symmetry_plane,
)
from anny.utils.mesh_utils import get_symmetric_vertex_indices


class TestProjectOntoSymmetryPlane(unittest.TestCase):
    def _project(self, vertices, faces, points):
        vertices = torch.tensor(vertices, dtype=torch.float64)
        points = torch.tensor(points, dtype=torch.float64)
        indices, weights, distances = _project_onto_symmetry_plane(
            points, vertices, torch.tensor(faces), axis=0
        )
        # Weights are barycentric coordinates, and distances match the returned points.
        self.assertTrue(bool((weights >= 0).all()))
        torch.testing.assert_close(
            weights.sum(dim=1), torch.ones(len(points), dtype=torch.float64)
        )
        projections = (weights[..., None] * vertices[indices]).sum(dim=1)
        torch.testing.assert_close(distances, (points - projections).norm(dim=1))
        return projections, weights

    def test_edge_in_plane(self):
        # Two mirrored triangles sharing an edge along the plane, and a triangle that does not meet
        # the plane although it is closer to the query point.
        projections, weights = self._project(
            vertices=[
                [0, 0, 0],
                [0, 1, 0],
                [-1, 0.5, -1],
                [1, 0.5, -1],
                [0.5, 0.25, 1],
                [1, 0.25, 1],
                [0.5, 1, 1],
            ],
            faces=[[0, 1, 2], [1, 0, 3], [4, 5, 6]],
            points=[[0, 0.25, 1], [0, 3, 0]],
        )
        torch.testing.assert_close(
            projections, torch.tensor([[0, 0.25, 0], [0, 1, 0]], dtype=torch.float64)
        )
        # No weight on the vertices off the plane.
        self.assertEqual(float(weights[:, 2].abs().max()), 0.0)

    def test_triangle_crossing_plane(self):
        # Self-mirrored triangle: one vertex on the plane, and a pair of mirrored vertices.
        projections, weights = self._project(
            vertices=[[0, 0, 0], [-1, 1, 0.5], [1, 1, 0.5]],
            faces=[[0, 1, 2]],
            points=[[0, 0.5, 1], [0, 5, 0]],
        )
        torch.testing.assert_close(
            projections, torch.tensor([[0, 0.8, 0.4], [0, 1, 0.5]], dtype=torch.float64)
        )
        torch.testing.assert_close(weights[:, 1], weights[:, 2])

        # No vertex on the plane: the intersection joins two edge crossings.
        projections, _ = self._project(
            vertices=[[-1, 0, 0], [2, 0, 0], [-1, 2, 0]],
            faces=[[0, 1, 2]],
            points=[[0, 3, 1]],
        )
        torch.testing.assert_close(
            projections, torch.tensor([[0, 4 / 3, 0]], dtype=torch.float64)
        )

    def test_triangle_touching_plane_at_a_vertex(self):
        projections, weights = self._project(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            faces=[[0, 1, 2]],
            points=[[0.0, 2.0, 1.0]],
        )
        torch.testing.assert_close(
            projections, torch.zeros((1, 3), dtype=torch.float64)
        )
        torch.testing.assert_close(
            weights, torch.tensor([[1, 0, 0]], dtype=torch.float64)
        )

    def test_vertices_close_to_plane_are_on_it(self):
        projections, _ = self._project(
            vertices=[[1e-8, 0, 0], [-1e-8, 1, 0], [-1, 0.5, 0]],
            faces=[[0, 1, 2]],
            points=[[0.0, 0.5, 1.0]],
        )
        torch.testing.assert_close(
            projections, torch.tensor([[0, 0.5, 0]], dtype=torch.float64)
        )

    def test_unsupported_meshes(self):
        point = torch.zeros((1, 3), dtype=torch.float64)
        for name, vertices in [
            ("triangle within the plane", [[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ("mesh not meeting the plane", [[1, 0, 0], [2, 0, 0], [1, 1, 0]]),
        ]:
            with self.subTest(name), self.assertRaises(ValueError):
                _project_onto_symmetry_plane(
                    point,
                    torch.tensor(vertices, dtype=torch.float64),
                    torch.tensor([[0, 1, 2]]),
                    axis=0,
                )


class TestRetopologySymmetry(unittest.TestCase):
    def test_alternative_topologies_are_symmetric(self):
        for topology in [
            "notoes",
            "notoes_collapse3pc",
            "notoes_collapse5pc",
            "notoes_collapse10pc",
        ]:
            with self.subTest(topology=topology):
                model = anny.Anny(topology=topology)
                sym = get_symmetric_vertex_indices(
                    model.template_vertices, axis=0, threshold=1e-5
                )

                faces = {frozenset(face) for face in model.faces.tolist()}
                mirrored_faces = {
                    frozenset(sym[face].tolist()) for face in model.faces.tolist()
                }
                self.assertEqual(faces, mirrored_faces)

                N = model.template_vertices.shape[0]
                B = len(model.bone_labels)
                dense = torch.zeros(N, B, dtype=model.vertex_bone_weights.dtype)
                dense.scatter_add_(
                    1, model.vertex_bone_indices, model.vertex_bone_weights
                )
                name_to_id = {n: i for i, n in enumerate(model.bone_labels)}
                bone_mirror = [
                    name_to_id[_get_symmetric_bone_name(name)]
                    for name in model.bone_labels
                ]
                torch.testing.assert_close(
                    dense, dense[sym][:, bone_mirror], atol=1e-6, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
