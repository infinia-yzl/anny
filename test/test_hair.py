# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import unittest

import numpy as np
import trimesh

from anny.hair import StrandBinding, closest_triangles


def sphere_with_strands(seed=0):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    V, T = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    rng = np.random.default_rng(seed)
    face = rng.integers(0, len(T), 200)
    bary = rng.dirichlet(np.ones(3), 200)
    roots = np.einsum("nk,nkd->nd", bary, V[T[face]])
    normal = roots / np.linalg.norm(roots, axis=1, keepdims=True)
    s = np.linspace(0, 1, 8)[None, :, None]
    strands = roots[:, None] + 0.05 * s * normal[:, None] + 0.01 * s**2
    return V, T, strands


class TestStrandBinding(unittest.TestCase):
    def test_reproduces_the_groom(self):
        V, T, strands = sphere_with_strands()
        binding = StrandBinding(strands, V, T)
        self.assertLess(binding.root_distance.max(), 1e-12)
        self.assertLess(np.abs(binding.follow(V) - strands).max(), 1e-12)

    def test_follows_a_similarity(self):
        # a rotated, scaled and moved head carries its hair the same way
        V, T, strands = sphere_with_strands()
        binding = StrandBinding(strands, V, T)
        a = np.radians(30)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        V2 = 0.6 * V @ R.T + np.array([0.1, -0.2, 0.3])
        expected = 0.6 * strands @ R.T + np.array([0.1, -0.2, 0.3])
        self.assertLess(np.abs(binding.follow(V2) - expected).max(), 1e-12)

    def test_roots_stay_on_the_surface(self):
        V, T, strands = sphere_with_strands()
        binding = StrandBinding(strands, V, T)
        # a non-uniform change: the roots stay on the new surface
        V2 = V * np.array([1.0, 0.7, 1.3])
        roots = binding.follow(V2)[:, 0]
        _, _, distance = closest_triangles(roots, V2, T)
        self.assertLess(distance.max(), 1e-12)

    def test_tips_reproduce_the_groom(self):
        V, T, strands = sphere_with_strands()
        binding = StrandBinding(strands, V, T, tips=True)
        self.assertLess(np.abs(binding.follow(V) - strands).max(), 1e-12)
        a = np.radians(-40)
        R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        V2 = 1.4 * V @ R.T + np.array([0.0, 0.1, -0.2])
        expected = 1.4 * strands @ R.T + np.array([0.0, 0.1, -0.2])
        self.assertLess(np.abs(binding.follow(V2) - expected).max(), 1e-12)

    def test_tips_stay_over_the_skin(self):
        # strands that lie along the surface, 2 mm above it; the head grows longer along z
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.1)
        V, T = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        rng = np.random.default_rng(3)
        u = rng.normal(size=(200, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        w = np.cross(u, rng.normal(size=(200, 3)))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        s = np.linspace(0, 1, 12)[None, :, None]
        angle = np.radians(60) * s
        direction = np.cos(angle) * u[:, None] + np.sin(angle) * w[:, None]
        radius = 0.1 * (1 - 1e-9) + 0.002 * np.minimum(1, 5 * s)
        strands = radius * direction
        stretch = np.array([1.0, 1.0, 1.35])
        V2 = V * stretch

        def inside(P):
            return ((P / stretch) ** 2).sum(-1) < 0.1**2

        rigid = StrandBinding(strands, V, T).follow(V2)
        bound = StrandBinding(strands, V, T, tips=True).follow(V2)
        self.assertGreater(inside(rigid[:, -1]).sum(), 0)
        self.assertEqual(inside(bound[:, -1]).sum(), 0)


if __name__ == "__main__":
    unittest.main()
