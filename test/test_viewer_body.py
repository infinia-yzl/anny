# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The page's body (viewer/src/body.ts) on the data of anny.viewer: anny's default body and its hair
come out as anny.viewer built them, and the ends of the sliders give finite bodies.

The test needs node 22 or later and the data of ``python -m anny.viewer build --data`` in
viewer/build; it skips otherwise.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

from test.test_viewer_parity import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
BUILD = REPO / "viewer" / "build"
SCRIPT = REPO / "viewer" / "test" / "body.mjs"


@unittest.skipUnless(
    node_available() and (BUILD / "manifest.json").exists(),
    "node 22 or later and the viewer data (python -m anny.viewer build --data) are needed",
)
class TestViewerBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = pathlib.Path(tempfile.mkdtemp()) / "body.json"
        subprocess.run(
            ["node", str(SCRIPT), str(BUILD), str(out)], check=True, cwd=REPO
        )
        cls.result = json.loads(out.read_text())

    def test_default_body(self):
        # 16-bit positions and 16-bit shape components: well under a millimetre
        self.assertLess(self.result["rest_max"], 5e-4)

    def test_default_hair(self):
        # the strands bind to the default body as the updates rebuild it: the groom comes back exactly
        self.assertLess(self.result["hair_max"], 1e-5)

    def test_ends_of_the_sliders(self):
        for key in ("age0", "age1", "all0", "all1"):
            with self.subTest(key):
                self.assertTrue(self.result[key]["finite"])
                self.assertGreater(self.result[key]["head_scale"], 0.5)
                self.assertLess(self.result[key]["head_scale"], 1.5)


if __name__ == "__main__":
    unittest.main()
