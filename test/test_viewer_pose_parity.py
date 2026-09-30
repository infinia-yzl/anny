# Corporis
# Apache License, Version 2.0
"""
Parity of the page's retarget (viewer/src/pose_from_image.ts) with corporis.posing.retarget.
Node runs the TypeScript module directly; the test skips when node is missing.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np
import roma
import torch

import anny
import anny.poses
from corporis.posing.retarget import Retargeter
from corporis.posing.skeleton import Skeleton
from test.test_pose_robust import corrupted, long_finger, tipped_head
from test.test_viewer_parity import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "pose_parity.mjs"
POSES = ("mh_hero", "arms_crossed", "mh_thinking", "mh_star", "seated", "mh_cheer")
# bad landmarks (test_pose_robust): misplaced and noisy head points, and a finger ten times
# too long
BAD = ("misplaced head", "bad hand")


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestViewerPoseParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = anny.Anny().to(torch.float64)
        cls.skeleton = Skeleton(model)
        cls.retarget = Retargeter(cls.skeleton)
        rest = cls.skeleton.output({})
        top = model.vertex_bone_indices[
            torch.arange(model.vertex_bone_indices.shape[0]),
            model.vertex_bone_weights.argmax(dim=1),
        ]
        cases = []
        for name in POSES:
            params = anny.poses.pose_parameters(model, name, grounded=False)[
                "pose_parameters"
            ][:1]
            with torch.no_grad():
                out = model(pose_parameters=params)
            cases.append(cls.retarget.anny.landmarks(out))
        _, _, clean = tipped_head(model, cls.skeleton, cls.retarget)
        cases += [corrupted(clean), long_finger(corrupted(clean))]
        landmarks, cls.expected, cls.expected_heads = [], [], []
        for L in cases:
            entry = {
                "body": L.body.tolist(),
                "hands": {s: h.tolist() for s, h in L.hands.items()},
            }
            if L.image is not None:
                entry["image"] = L.image.tolist()
                entry["visibility"] = L.visibility.tolist()
            landmarks.append(entry)
            W, _ = cls.retarget(L)
            P = cls.skeleton.params(W)[0, :, :3, :3].double()
            cls.expected.append(roma.rotmat_to_unitquat(P).numpy())
            cls.expected_heads.append(cls.retarget.head_fit(L).source)
        inp = {
            "names": cls.skeleton.labels,
            "parents": [int(p) for p in cls.skeleton.parents],
            "heads": rest["bone_poses"][0, :, :3, 3].tolist(),
            "vertices": rest["vertices"][0].reshape(-1).tolist(),
            "top": top.tolist(),
            "landmarks": landmarks,
        }
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "input.json").write_text(json.dumps(inp))
            subprocess.run(["node", str(SCRIPT), tmp], check=True)
            cls.output = json.loads((pathlib.Path(tmp) / "output.json").read_text())

    def test_landmarks_sit_on_the_same_joints_and_vertices(self):
        anny_lm = self.retarget.anny
        labels = self.skeleton.labels

        def python(sources):
            return [
                [kind, labels.index(ref) if kind == "joint" else ref]
                for kind, ref in sources
            ]

        self.assertEqual(self.output["sources"]["body"], python(anny_lm.body))
        for s in (".L", ".R"):
            self.assertEqual(
                self.output["sources"]["hands"][s], python(anny_lm.hands[s])
            )

    def test_head_sources_match(self):
        self.assertEqual(self.output["heads"], self.expected_heads)
        self.assertEqual(self.expected_heads[-2:], ["picture", "picture"])

    def test_rotations_match(self):
        names = POSES + BAD
        for name, expected, got in zip(names, self.expected, self.output["poses"]):
            with self.subTest(pose=name):
                got = np.array(got).reshape(-1, 4)
                # q and -q are the same rotation
                gap = np.minimum(
                    np.abs(got - expected).max(1), np.abs(got + expected).max(1)
                )
                self.assertLess(gap.max(), 2e-4, self.skeleton.labels[gap.argmax()])


if __name__ == "__main__":
    unittest.main()
