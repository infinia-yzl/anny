# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The scalp layout, the style format and the reference of the page's hair passes (anny.hair.chart,
anny.hair.layout, anny.hair.styles and the authoring helpers).
"""

import unittest

import numpy as np
import trimesh

from anny.hair import chart
from anny.hair.layout import DEFAULT_PATH, Layout, load_layout


def smooth_curves(n=300, P=24, seg=0.02, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    S = np.zeros((n, P, 3))
    S[:, 0] = rng.normal(size=(n, 3)) * 0.1
    for j in range(1, P):
        d = d + rng.normal(size=(n, 3)) * 0.2
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        S[:, j] = S[:, j - 1] + d * seg
    return S


class TestChart(unittest.TestCase):
    def test_codec_error_stays_within_a_quantum(self):
        for seg in (0.001, 0.02):
            S = smooth_curves(seg=seg)
            roots, s, codes = chart.encode_curves(S)
            D = chart.decode_curves(roots, s, codes)
            # one 8-bit octahedral step is about 1/127 of a radian at most
            self.assertLess(np.abs(D - S).max(), 2.0 * seg / 127)
            self.assertEqual(codes.dtype, np.uint8)
            self.assertLessEqual(int(codes.max()), 254)

    def test_octahedral_round_trip(self):
        rng = np.random.default_rng(1)
        d = rng.normal(size=(1000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.assertLess(np.abs(chart.oct_decode(chart.oct_encode(d)) - d).max(), 1e-12)

    def test_resample_uniform(self):
        S = smooth_curves(seg=0.01)
        R, L = chart.resample_uniform(S, 9)
        seg = np.linalg.norm(np.diff(R, axis=1), axis=2)
        # a chord is at most as long as its arc, and nearly so for smooth curves
        self.assertLess(np.abs(seg / seg.mean(1, keepdims=True) - 1).max(), 0.2)
        self.assertLess(np.abs(R[:, 0] - S[:, 0]).max(), 1e-12)
        self.assertLess(np.abs(R[:, -1] - S[:, -1]).max(), 1e-9)

    def test_fade_grows_upward(self):
        fade = dict(
            start=[0.0] * len(chart.CURVE_PHI), width=10.0, clipper=0.0005, top=0.02
        )
        phi = np.full(50, 120.0)
        el = chart.hairline(phi) + np.linspace(-5, 30, 50)
        L = chart.fade_length(phi, el, fade)
        self.assertTrue(np.all(np.diff(L) >= 0))
        self.assertAlmostEqual(L[0], 0.0005)
        self.assertTrue(np.isinf(chart.fade_length(phi, el, None)).all())


class TestSampling(unittest.TestCase):
    def test_maximal_independent_set(self):
        from scipy.spatial import cKDTree

        from anny.hair.authoring.sampling import maximal_independent_set

        rng = np.random.default_rng(2)
        P = rng.random((5000, 3))
        r = 0.05
        sel = maximal_independent_set(P, r, rng)
        pairs = cKDTree(P[sel]).query_pairs(r)
        self.assertEqual(len(pairs), 0)
        # maximal: every other point lies within r of a chosen one
        d, _ = cKDTree(P[sel]).query(P[~sel])
        self.assertTrue(np.all(d <= r))

    def test_progressive_levels_are_poisson(self):
        from scipy.spatial import cKDTree

        from anny.hair.authoring.sampling import poisson_subset, progressive_order

        rng = np.random.default_rng(3)
        P = rng.random((20000, 2))
        P = np.concatenate([P, np.zeros((len(P), 1))], 1)
        r = 0.01
        P = P[poisson_subset(P, r, rng)]
        order, bounds = progressive_order(P, r, rng)
        self.assertEqual(sorted(order.tolist()), list(range(len(P))))
        # the prefix at level i (coarsest first) is a Poisson set at r * sqrt(2)^(levels-1-i)
        for i, b in enumerate(bounds):
            radius = r * np.sqrt(2.0) ** (len(bounds) - 1 - i)
            d, _ = cKDTree(P[order[:b]]).query(P[order[:b]], k=2)
            self.assertGreaterEqual(d[:, 1].min(), radius * (1 - 1e-9))


class TestLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = load_layout()

    def test_file(self):
        self.assertTrue(DEFAULT_PATH.exists())
        self.assertIsInstance(self.layout, Layout)

    def test_weights(self):
        L = self.layout
        self.assertTrue(np.all(L.root_weights.astype(int).sum(1) == 255))
        self.assertTrue(np.all(L.guide_sim_weights.astype(int).sum(1) == 255))
        self.assertTrue(np.all((L.root_guides >= 0) & (L.root_guides < L.guides)))
        self.assertTrue(np.all((L.guide_sim >= 0) & (L.guide_sim < L.simulated)))
        # a simulated guide follows itself
        own = np.arange(L.simulated)
        self.assertTrue(np.all(L.guide_sim[own, 0] == own))
        self.assertTrue(np.all(L.guide_sim_weights[own, 0] == 255))

    def test_mirror(self):
        L = self.layout
        m = L.guide_mirror
        self.assertTrue(np.all(m[m] == np.arange(L.guides)))
        mirrored = L.guide_position[m] * np.array([-1.0, 1.0, 1.0])
        self.assertLess(np.abs(mirrored - L.guide_position).max(), 1e-4)

    def test_prefixes_cover_the_scalp(self):
        from anny.hair.authoring.sampling import coverage_radius

        L = self.layout
        R = L.root_position
        spacing = L.meta["root_spacing"]
        for n in (2000, 8000, 32000):
            radius = coverage_radius(R, np.arange(n), R)
            self.assertLess(radius, 2.5 * spacing * np.sqrt(len(R) / n))


class TestLayers(unittest.TestCase):
    def test_buckets_of_one_reproduce_the_sequential_layers(self):
        from anny.hair.authoring import groom
        from anny.hair.chart import chart as chart_of

        rng = np.random.default_rng(4)
        G, M = 60, 8
        roots = (
            chart.CRANIUM_CENTRE
            + 0.09 * np.array([0.0, 0.8, -0.6])
            + rng.normal(size=(G, 3)) * 0.02
        )
        X = (
            roots[:, None]
            + np.linspace(0, 1, M)[None, :, None] * rng.normal(size=(G, 1, 3)) * 0.05
        )
        whorl = chart.CRANIUM_CENTRE + np.array([0, 0.09, -0.03])
        info = dict(roots=roots)
        fast = groom.layer_offsets(X, info, whorl, buckets=G)

        # the legacy loop (build/hair2.py): one guide at a time, point by point
        base, per_guide, cell = 0.6e-3, 1.25e-3, 2.0
        order = np.argsort(-np.linalg.norm(roots - whorl, axis=1))
        nphi, nel = int(360 / cell), int(180 / cell)
        Tk = np.zeros((nphi, nel))
        phi, el, _ = chart_of(X.reshape(-1, 3))
        ci = np.clip(((phi + 180) / cell).astype(int), 0, nphi - 1).reshape(G, M)
        cj = np.clip(((el + 90) / cell).astype(int), 0, nel - 1).reshape(G, M)
        offs = np.zeros((G, M))
        ker = groom.LAYER_KERNEL / groom.LAYER_KERNEL.sum()
        for g in order:
            offs[g] = base + Tk[ci[g], cj[g]]
            for j in range(1, M):
                a, b = ci[g, j], cj[g, j]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        Tk[(a + di) % nphi, min(max(b + dj, 0), nel - 1)] += (
                            per_guide * ker[di + 1, dj + 1]
                        )
        s = np.linspace(0, 1, M)
        legacy = offs * chart.smoothstep(0.0, 0.25, s)[None, :] + base
        self.assertLess(np.abs(fast - legacy).max(), 1e-12)


class TestStrands(unittest.TestCase):
    """the reference of pass B on a sphere with a toy layout"""

    @classmethod
    def setUpClass(cls):
        from anny.hair import styles as H

        mesh = trimesh.creation.icosphere(subdivisions=5, radius=0.09)
        V = np.asarray(mesh.vertices) + chart.CRANIUM_CENTRE
        T = np.asarray(mesh.faces)
        rng = np.random.default_rng(5)
        # guides and roots on the upper half
        up = V[V[:, 1] > chart.CRANIUM_CENTRE[1] + 0.03]
        G = up[rng.choice(len(up), 80, replace=False)]
        R = up[rng.choice(len(up), 400, replace=False)]
        from scipy.spatial import cKDTree

        from anny.hair.authoring.layout import inverse_distance_weights

        d, idx = cKDTree(G).query(R, k=4)
        cls.layout = Layout(
            guide_position=G,
            guide_chart=np.stack(chart.chart(G)[:2], 1),
            guide_mirror=np.arange(len(G)),
            guide_sim=np.zeros((len(G), 3), np.int32),
            guide_sim_weights=np.tile(np.array([255, 0, 0], np.uint8), (len(G), 1)),
            root_position=R,
            root_chart=np.stack(chart.chart(R)[:2], 1),
            root_guides=idx.astype(np.int32),
            root_weights=inverse_distance_weights(d, 0.0012),
            meta=dict(simulated=1),
        )
        # straight guides along the normal, 4 cm
        n = (G - chart.CRANIUM_CENTRE) / 0.09
        P = 12
        pts = G[:, None] + n[:, None] * np.linspace(0, 0.04, P)[None, :, None]
        spec = dict(
            name="test",
            render=dict(
                strand_jitter=[1.0, 1.0],
                thinning=None,
                clump=dict(
                    coarse=[0.0, 0.0],
                    coarse_tip=0.0,
                    fine=[0.0, 0.0],
                    fine_power=[1.0, 1.0],
                    sectors=6,
                ),
                curl=None,
                frizz=None,
                flyaways=None,
                hairline=[-90.0] * len(chart.CURVE_PHI),
                fade=None,
            ),
        )
        cls.style = H.HairStyle(
            spec,
            pts,
            np.full(len(G), 0.03),
            np.zeros(len(G), np.uint8),
            np.zeros(len(G)),
            np.full(len(G), 1000.0),
        )
        cls.V, cls.T = V, T

    def run_strands(self, **params):
        from anny.hair import styles as H

        bind = H.Binding.build(self.layout, self.V, self.T)
        rest = H.follow_guides(self.style, bind, self.V, self.V, self.T)
        roots = H.Roots.on(bind, self.V, self.T)
        return H.strands(
            self.style,
            self.layout,
            rest,
            H.guide_normals(bind, self.V, self.T),
            roots,
            params,
        )

    def test_radial_guides_give_radial_strands(self):
        S, ell = self.run_strands()
        self.assertTrue(np.allclose(ell, 0.03))
        # the offsets turn with the root normal: each strand stands along its own normal
        root = S[:, 0]
        n = (root - chart.CRANIUM_CENTRE) / np.linalg.norm(
            root - chart.CRANIUM_CENTRE, axis=1, keepdims=True
        )
        tip = S[:, -1] - root
        cos = (tip * n).sum(1) / np.linalg.norm(tip, axis=1)
        self.assertGreater(
            cos.min(), 0.995
        )  # the facets of the sphere tilt the normals

    def test_length_parameter(self):
        _, ell = self.run_strands(length=0.5)
        self.assertTrue(np.allclose(ell, 0.015))
        # the strands cannot grow past their guides (4 cm)
        _, ell = self.run_strands(length=2.0)
        self.assertLessEqual(ell.max(), 0.04 + 1e-9)

    def test_volume_moves_points_outward(self):
        S1, _ = self.run_strands()
        S2, _ = self.run_strands(volume=1.3)
        r1 = np.linalg.norm(S1[:, -1] - chart.CRANIUM_CENTRE, axis=1)
        r2 = np.linalg.norm(S2[:, -1] - chart.CRANIUM_CENTRE, axis=1)
        self.assertTrue(np.all(r2 > r1))
        self.assertLess(np.abs(S1[:, 0] - S2[:, 0]).max(), 1e-12)

    def test_random_values(self):
        from anny.hair.styles import rnd

        u = rnd(np.arange(100000), 3)
        self.assertGreaterEqual(u.min(), 0.0)
        self.assertLess(u.max(), 1.0)
        self.assertAlmostEqual(u.mean(), 0.5, delta=0.01)
        self.assertTrue(np.array_equal(u, rnd(np.arange(100000), 3)))
        self.assertFalse(np.array_equal(u, rnd(np.arange(100000), 4)))

    def test_rotate_between(self):
        from anny.hair.styles import rotate_between

        rng = np.random.default_rng(6)
        u = rng.normal(size=(100, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v = u + rng.normal(size=(100, 3)) * 0.3
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.assertLess(np.abs(rotate_between(u, v, u) - v).max(), 1e-12)
        x = rng.normal(size=(100, 3))
        self.assertLess(
            np.abs(
                np.linalg.norm(rotate_between(u, v, x), axis=1)
                - np.linalg.norm(x, axis=1)
            ).max(),
            1e-12,
        )


class TestBodyClearance(unittest.TestCase):
    def test_hair_on_the_body_keeps_its_gap(self):
        from anny.hair.authoring.groom import BODY_CLEARANCE, clear_body

        # a sphere for a shoulder, and guides draped over it at 1 mm from its surface
        centre, radius = np.array([0.15, 0.3, 0.0]), 0.05

        def sdf(P):
            d = P - centre
            n = np.linalg.norm(d, axis=1, keepdims=True)
            return n[:, 0] - radius, d / n

        a = np.linspace(-1.2, 1.2, 25)
        X = np.stack(
            [
                centre
                + (radius + 0.001)
                * np.stack(
                    [
                        np.sin(a) * np.cos(b),
                        np.cos(a) * np.cos(b),
                        np.full_like(a, np.sin(b)),
                    ],
                    1,
                )
                for b in np.linspace(-0.6, 0.6, 7)
            ]
        )
        seg = np.linalg.norm(np.diff(X, axis=1), axis=2)
        Y = clear_body(sdf, X)
        self.assertLess(
            np.abs(np.linalg.norm(np.diff(Y, axis=1), axis=2) - seg).max(), 1e-9
        )
        self.assertTrue(np.array_equal(Y[:, 0], X[:, 0]))
        # the chain from the fixed root keeps the first points a little closer
        gap = sdf(Y[:, 3:].reshape(-1, 3))[0]
        self.assertGreater(gap.min(), 0.8 * BODY_CLEARANCE)


class TestDensityVolume(unittest.TestCase):
    def test_cover_follows_the_fade(self):
        from anny.hair import styles as H

        layout = load_layout()
        style = H.load_style("low_taper_fade", layout)
        vol, lo, h = H.density_volume(style, layout)
        self.assertEqual(vol.shape[3], 4)
        cover = H.sample_volume(vol, lo, h, layout.guide_position, channel=2)
        rs = style.spec["render"]
        phi, el = layout.guide_chart[:, 0], layout.guide_chart[:, 1]
        D = np.minimum(style.length, style.available)
        D = np.minimum(
            D, chart.fade_length(phi, el, rs["fade"], 0.0, rs.get("hairline"))
        )
        on = chart.coverage(phi, el, rs.get("hairline")) > 0.99
        # the top shades the scalp fully; the bottom of the fade (the clipper's 0.8 mm up to 2 mm) shades it lightly
        self.assertGreater(cover[on & (D > 0.01)].mean(), 0.9)
        self.assertLess(cover[on & (D < 0.002)].mean(), 0.5)

    def test_bald_has_no_cover(self):
        from anny.hair import styles as H

        layout = load_layout()
        vol, _, _ = H.density_volume(H.load_style("bald", layout), layout)
        self.assertTrue(np.all(vol[..., 0] == 255))
        self.assertTrue(np.all(vol[..., 1:] == 0))


if __name__ == "__main__":
    unittest.main()
