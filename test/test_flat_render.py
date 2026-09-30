# Corporis
# Apache License, Version 2.0
"""The flat pictures of a posed character: silhouette, outline and shaded picture."""

import re
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

import anny
import anny.poses
from corporis.render.flat import View, outline_svg, shaded_png, silhouette_svg

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
