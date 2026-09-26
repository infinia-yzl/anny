# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Parity of the page's hair data (viewer/src/hair/data.ts) with anny.hair.styles: the decoded guides
of every style and of its mirror, the density volume, and the random values. Node runs the
TypeScript on the data of ``python -m anny.viewer build`` (viewer/build); the test skips when node
or the data is missing.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

from anny.hair import styles as H
from anny.hair.layout import load_layout
from test.test_viewer_parity import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
BUILD = REPO / "viewer" / "build"
SCRIPT = REPO / "viewer" / "test" / "hair.mjs"


def data_available():
    man = BUILD / "manifest.json"
    return man.exists() and "hair" in json.loads(man.read_text())


@unittest.skipUnless(node_available() and data_available(), "node 22 and the viewer data are needed")
class TestHairParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        subprocess.run(["node", str(SCRIPT), str(BUILD), str(cls.dir)], check=True, cwd=REPO)
        cls.result = json.loads((cls.dir / "result.json").read_text())
        cls.layout = load_layout()

    def test_decoded_guides(self):
        for name, info in self.result["styles"].items():
            style = H.load_style(name, self.layout)
            pts = np.fromfile(self.dir / f"{name}_points.bin", np.float32).reshape(style.points.shape)
            self.assertLess(np.abs(pts - style.points).max(), 1e-6, name)
            mir = np.fromfile(self.dir / f"{name}_mirror.bin", np.float32).reshape(style.points.shape)
            self.assertLess(np.abs(mir - style.mirrored(self.layout).points).max(), 1e-6, name)

    def test_density_volume(self):
        for name, info in self.result["styles"].items():
            style = H.load_style(name, self.layout)
            vol, lo, h = H.density_volume(style, self.layout, info["params"], info["count"])
            nx, ny, nz = info["dims"]
            self.assertEqual(list(vol.shape[:3]), [nx, ny, nz], name)
            self.assertLess(np.abs(np.array(info["lo"]) - lo).max(), 1e-9)
            page = np.fromfile(self.dir / f"{name}_volume.bin", np.uint8).reshape(nz, ny, nx, 2)
            page = page.transpose(2, 1, 0, 3).astype(int)
            diff = np.abs(page - vol.astype(int))
            # the nearest voxel of a step can differ at a tie; the values match within one step of 8 bits almost everywhere
            self.assertGreater((diff <= 1).mean(), 0.999, name)

    def test_random_values(self):
        keys = np.arange(1000) * 7919 + 13
        streams = np.arange(1000) % 17
        ref = np.array([H.rnd(k, s) for k, s in zip(keys, streams)])
        self.assertTrue(np.array_equal(np.array(self.result["rnd"]), ref))


if __name__ == "__main__":
    unittest.main()
