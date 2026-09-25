# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest

import numpy as np
import torch
import trimesh

import anny
from anny.utils.subdivision import (
    MixedSubdivision,
    SparseOperator,
    build_edges,
    catmull_clark,
)


def reference_catmull_clark(V, quads):
    """Value-based Catmull-Clark rules of the legacy 3D Model build (build/subdiv.py)."""
    V = np.asarray(V, np.float64)
    nV = len(V)
    E, FE = build_edges(quads)
    nE = len(E)
    boundary_edge = np.bincount(FE.ravel(), minlength=nE) == 1
    fp = V[quads].mean(1)
    mid = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    fsum = np.zeros((nE, 3))
    np.add.at(fsum, FE.ravel(), np.repeat(fp, 4, axis=0))
    ep = mid.copy()
    ep[~boundary_edge] = 0.5 * mid[~boundary_edge] + 0.25 * fsum[~boundary_edge]
    valence = np.bincount(E.ravel(), minlength=nV).astype(np.float64)
    vf_sum = np.zeros((nV, 3))
    vf_cnt = np.zeros(nV)
    np.add.at(vf_sum, quads.ravel(), np.repeat(fp, 4, axis=0))
    np.add.at(vf_cnt, quads.ravel(), 1)
    r_sum = np.zeros((nV, 3))
    np.add.at(r_sum, E[:, 0], mid)
    np.add.at(r_sum, E[:, 1], mid)
    n = valence[:, None]
    vp = (vf_sum / vf_cnt[:, None] + 2 * r_sum / n + (n - 3) * V) / n
    bE = E[boundary_edge]
    bcount = np.bincount(bE.ravel(), minlength=nV)
    nb_sum = np.zeros((nV, 3))
    np.add.at(nb_sum, bE[:, 0], V[bE[:, 1]])
    np.add.at(nb_sum, bE[:, 1], V[bE[:, 0]])
    smooth = bcount == 2
    vp[smooth] = (6 * V[smooth] + nb_sum[smooth]) / 8.0
    corner = (bcount > 0) & ~smooth
    vp[corner] = V[corner]
    return np.concatenate([vp, ep, fp], 0)


def cube():
    V = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64
    )
    quads = np.array(
        [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        ]
    )
    return V, quads


def grid(nx=4, ny=3):
    V = np.array(
        [[i, j, np.sin(i + 2 * j)] for j in range(ny + 1) for i in range(nx + 1)],
        dtype=np.float64,
    )
    quads = np.array(
        [
            [
                j * (nx + 1) + i,
                j * (nx + 1) + i + 1,
                (j + 1) * (nx + 1) + i + 1,
                (j + 1) * (nx + 1) + i,
            ]
            for j in range(ny)
            for i in range(nx)
        ]
    )
    return V, quads


class TestSubdivision(unittest.TestCase):
    def test_matches_reference_rules(self):
        for V, quads in (cube(), grid()):
            level = catmull_clark(len(V), quads)
            expected = reference_catmull_clark(V, quads)
            self.assertLess(np.abs(level.operator.apply(V) - expected).max(), 1e-12)
            # a second level runs on the new quads
            V1 = level.operator.apply(V)
            level2 = catmull_clark(len(V1), level.quads)
            expected2 = reference_catmull_clark(V1, level.quads)
            self.assertLess(np.abs(level2.operator.apply(V1) - expected2).max(), 1e-12)

    def test_rows_are_affine(self):
        # every new point is an affine combination of the old ones
        V, quads = cube()
        op = catmull_clark(len(V), quads).operator
        sums = np.bincount(op.rows, weights=op.values, minlength=op.shape[0])
        self.assertLess(np.abs(sums - 1).max(), 1e-12)

    def test_operator_algebra(self):
        V, quads = grid()
        level = catmull_clark(len(V), quads)
        level2 = catmull_clark(level.operator.shape[0], level.quads)
        composed = level2.operator @ level.operator
        self.assertLess(
            np.abs(
                composed.apply(V) - level2.operator.apply(level.operator.apply(V))
            ).max(),
            1e-12,
        )
        rows = np.array([5, 0, 17])
        selected = composed.select_rows(rows)
        self.assertLess(
            np.abs(selected.apply(V) - composed.apply(V)[rows]).max(), 1e-12
        )
        dense = SparseOperator([0, 0, 1], [1, 1, 0], [1.0, 2.0, 4.0], (2, 2)).to_dense()
        np.testing.assert_allclose(dense, [[0.0, 3.0], [4.0, 0.0]])

    def test_linearity_and_torch(self):
        V, quads = cube()
        op = catmull_clark(len(V), quads).operator
        A = np.random.default_rng(0).normal(size=(len(V), 5))
        B = np.random.default_rng(1).normal(size=(len(V), 5))
        self.assertLess(
            np.abs(op.apply(2 * A - B) - (2 * op.apply(A) - op.apply(B))).max(), 1e-12
        )
        At = torch.tensor(A, requires_grad=True)
        out = op.apply(At)
        self.assertLess(np.abs(out.detach().numpy() - op.apply(A)).max(), 1e-12)
        out.sum().backward()
        column_sums = np.bincount(op.cols, weights=op.values, minlength=op.shape[1])
        np.testing.assert_allclose(At.grad.numpy()[:, 0], column_sums)


class TestMixedSubdivision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny(topology="anny-quads").to(dtype=torch.float64)
        base_index = cls.model.base_mesh_vertex_indices.numpy()
        faces = cls.model.faces.numpy()
        # the body surface (eyes and tongue are separate meshes)
        cls.body_faces = (base_index[faces] < 13380).all(1)
        cls.subdivision = MixedSubdivision.for_model(
            cls.model, faces_mask=cls.body_faces
        )

    def test_closed_surface_without_cracks(self):
        rest = self.model(phenotype_kwargs={"age": 0.1})["rest_vertices"][0]
        mesh = trimesh.Trimesh(
            self.subdivision(rest.numpy()), self.subdivision.triangles, process=False
        )
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertEqual(mesh.euler_number, 2)

    def test_head_uses_the_finer_level(self):
        fine = self.subdivision.fine_level_vertices
        self.assertGreater(fine.sum(), 0)
        self.assertGreater((~fine).sum(), 0)
        rest = self.subdivision(self.model.template_vertices.numpy())
        # the finer region is the head: it sits above the rest of the body
        self.assertGreater(np.median(rest[fine, 2]), np.max(rest[~fine, 2]) - 0.25)

    def test_follows_every_phenotype(self):
        # subdivision is linear, so subdividing blend shapes equals subdividing the result
        a = self.model(phenotype_kwargs={"age": 0.0})["rest_vertices"][0].numpy()
        b = self.model(phenotype_kwargs={"age": 1.0})["rest_vertices"][0].numpy()
        mixed = self.subdivision(0.3 * a + 0.7 * b)
        self.assertLess(
            np.abs(
                mixed - (0.3 * self.subdivision(a) + 0.7 * self.subdivision(b))
            ).max(),
            1e-12,
        )

    def test_renumbered_rows(self):
        # the body vertices alone, in a compact numbering: the same fine surface
        quads = self.subdivision.input_quads
        used = np.unique(quads)
        compact = -np.ones(self.subdivision.input_vertex_count, np.int64)
        compact[used] = np.arange(len(used))
        rows = self.subdivision.renumbered_rows(compact, len(used))
        n, q = len(used), compact[quads]
        rest = self.model(phenotype_kwargs={"age": 0.8})["rest_vertices"][0].numpy()
        x = rest[used]
        for _ in range(self.subdivision.base_level):
            level = catmull_clark(n, q)
            x = level.operator.apply(x)
            n, q = level.operator.shape[0], level.quads
        x = catmull_clark(n, q).operator.select_rows(rows).apply(x)
        self.assertLess(np.abs(x - self.subdivision(rest)).max(), 1e-12)
        with self.assertRaises(ValueError):
            self.subdivision.renumbered_rows(compact[::-1], len(used))


if __name__ == "__main__":
    unittest.main()
