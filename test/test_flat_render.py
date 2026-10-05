# OpenSculptBoy
# Apache License, Version 2.0
"""The flat pictures of a posed character: silhouette, outline and shaded picture."""

import re
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

import anny
import anny.poses
from opensculptboy.render.flat import (
    SKIN,
    View,
    _rgb,
    outline_svg,
    shaded_png,
    silhouette_svg,
)

SIZE = (300, 420)


def pose(model, name):
    params = anny.poses.pose_parameters(model, name, grounded=False)["pose_parameters"][
        :1
    ]
    with torch.no_grad():
        return model(pose_parameters=params)["vertices"][0].numpy()


def paths(svg):
    root = ET.fromstring(svg)
    return [e for e in root.iter() if e.tag.endswith("path")]


def sphere(rings=8, segments=16):
    """a coarse unit sphere (vertices, triangles wound outward)"""
    th = np.linspace(0.0, np.pi, rings + 1)[1:-1]
    ph = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    middle = [
        [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
        for t in th
        for p in ph
    ]
    v = np.array([[0.0, 0.0, 1.0]] + middle + [[0.0, 0.0, -1.0]])
    last = len(v) - 1

    def ring(i, j):
        return 1 + i * segments + j % segments

    f = [[0, ring(0, j), ring(0, j + 1)] for j in range(segments)]
    for i in range(rings - 2):
        for j in range(segments):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            f += [[a, c, b], [b, c, d]]
    f += [[last, ring(rings - 2, j + 1), ring(rings - 2, j)] for j in range(segments)]
    f = np.array(f)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    inward = (n * v[f].mean(axis=1)).sum(axis=1) < 0
    f[inward] = f[inward][:, ::-1]
    return v, f


def area(d):
    """the area of an SVG path of straight rings (even-odd), by the shoelace formula"""
    total = 0.0
    for ring in re.findall(r"M([^Z]*)Z", d):
        pts = np.array([[float(x) for x in p.split()] for p in ring.split("L")])
        x, y = pts[:, 0], pts[:, 1]
        total += 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return abs(total)


class TestFlatRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny().to(torch.float32)
        cls.faces = cls.model.get_triangular_faces().numpy()
        cls.crossed = pose(cls.model, "arms_crossed")
        cls.star = pose(cls.model, "mh_star")

    def test_silhouette_matches_the_shaded_picture(self):
        for view in (View(), View(yaw=-40, roll=-5)):
            with self.subTest(view=view):
                svg = silhouette_svg(self.star, self.faces, view, size=SIZE)
                (shape,) = paths(svg)
                picture = np.asarray(shaded_png(self.star, self.faces, view, size=SIZE))
                self.assertEqual(picture.shape, (SIZE[1], SIZE[0], 4))
                opaque = (picture[..., 3] > 127).sum()
                self.assertAlmostEqual(area(shape.get("d")) / opaque, 1.0, delta=0.04)
                self.assertEqual(picture[2, 2, 3], 0)  # the background stays clear

    def test_smooth_shading_follows_the_surface(self):
        # On a coarse sphere, smooth shading gives each pixel the band of its interpolated
        # normal, so the picture follows the toon shading of the true sphere, whose normal at a
        # pixel is known; the shading of whole triangles departs from it along the steps of
        # the triangles at the edges of the bands.
        v, f = sphere()
        size, view = (400, 400), View()
        right, up, toward = view.basis()
        L = -0.45 * right + 0.7 * up + 0.55 * toward
        L /= np.linalg.norm(L)
        bands = np.array([0.62, 0.84, 1.0])
        thresholds = np.linspace(0.0, 1.0, len(bands) + 1)[1:-1] * 0.8 + 0.1
        off = {}
        for smooth in (False, True):
            picture = np.asarray(
                shaded_png(v, f, view, size=size, margin=0, lines=False, smooth=smooth)
            ).astype(np.float64)
            ys, xs = np.nonzero(picture[..., 3] > 254)
            x = (xs + 0.5) / size[0] * 2.0 - 1.0
            y = 1.0 - (ys + 0.5) / size[1] * 2.0
            z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
            n = np.outer(x, right) + np.outer(y, up) + np.outer(z, toward)
            band = bands[np.searchsorted(thresholds, np.clip(n @ L, 0.0, 1.0))]
            rim = np.clip(1.0 - np.abs(n @ toward), 0.0, 1.0) ** 3 * 0.12
            expected = np.clip(_rgb(SKIN)[None, :] * (band + rim)[:, None], 0, 255)
            error = np.abs(picture[ys, xs, :3] - expected).mean(axis=1)
            off[smooth] = float((error > 20).mean())
        self.assertLess(off[True], 0.02, off)
        self.assertLess(off[True], 0.3 * off[False], off)

    def test_outline_draws_the_arms_across_the_chest(self):
        crossed = outline_svg(self.crossed, self.faces, size=SIZE)
        star = outline_svg(self.star, self.faces, size=SIZE)
        self.assertGreaterEqual(len(paths(crossed)), 2)  # the contour and inner lines
        inner = [p for p in paths(crossed) if p.get("fill") not in (None, "none")]
        self.assertTrue(inner, "no inner lines where the arms cross the chest")
        star_inner = [p for p in paths(star) if p.get("fill") not in (None, "none")]
        star_ink = sum(area(p.get("d")) for p in star_inner)
        self.assertGreater(area(inner[0].get("d")), 1.3 * star_ink)

    def test_view_turns_about_the_vertical(self):
        right, up, toward = View(yaw=90).basis()
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(
            toward, [-1, 0, 0], atol=1e-12
        )  # the camera on the right side
        right, up, toward = View().basis()
        np.testing.assert_allclose(
            toward, [0, -1, 0], atol=1e-12
        )  # the figure faces -y


if __name__ == "__main__":
    unittest.main()
