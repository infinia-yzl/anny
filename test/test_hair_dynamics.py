# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The physics of anny's hair: anny.hair.dynamics on synthetic strands, and the page's solver
(viewer/src/hair/sim.ts) against it on the styles of the viewer data. Node runs the TypeScript on
the data of ``python -m anny.viewer build`` (viewer/build); the parity test skips when node or the
data is missing.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

from anny.hair import dynamics as D
from anny.hair.styles import sim_offsets
from test.test_viewer_parity import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
BUILD = REPO / "viewer" / "build"
SCRIPT = REPO / "viewer" / "test" / "sim.mjs"


def strands(S=24, P=12, seed=0):
    """wavy strands hanging from roots on a circle of 8 cm, 1 cm per segment"""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, S, endpoint=False)
    root = np.stack([0.08 * np.sin(a), np.zeros(S), 0.08 * np.cos(a)], 1)
    out = np.zeros((S, P, 3))
    out[:, 0] = root
    for j in range(1, P):
        d = np.stack([np.sin(a) * 0.3, -np.ones(S), np.cos(a) * 0.3], 1)
        d += 0.2 * rng.standard_normal((S, 3))
        out[:, j] = out[:, j - 1] + 0.01 * d / np.linalg.norm(d, axis=1, keepdims=True)
    return out


def turn(X, angle, axis=2, centre=(0.0, 0.0, 0.0)):
    c, s = np.cos(angle), np.sin(angle)
    i, k = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, k], R[k, i], R[k, k] = c, -s, s, c
    return (X - centre) @ R.T + centre, R


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.rest = strands()
        S = len(self.rest)
        # a third of the guides lie on the head for 3 cm, the others keep their root only
        self.pivot = np.where(np.arange(S) % 3 == 0, 0.03, 1000.0)
        self.caps = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06]])

    def advance(self, sim, frames, g=(0.0, 0.0, 0.0)):
        for T in frames:
            sim.step(T, g, self.caps)
        return sim

    def test_gravity_change(self):
        self.assertTrue(np.allclose(D.gravity_change(np.eye(3)), 0))
        _, R = turn(np.zeros((1, 3)), np.pi / 2, axis=2)
        # the head on its side: the groom holds gravity along the head's rest y axis, which now points along -x
        self.assertTrue(
            np.allclose(
                D.gravity_change(R), np.array([0, -9.81, 0]) - R @ [0, -9.81, 0]
            )
        )
        self.assertTrue(np.allclose(D.gravity_change(R), [-9.81, -9.81, 0]))

    def test_still_head_keeps_the_groom(self):
        sim = D.HairSim(self.rest, self.pivot, capsules=self.caps)
        self.advance(sim, [self.rest] * 60)
        self.assertLess(np.abs(sim.x - self.rest).max(), 1e-12)

    def test_lengths_and_pins_hold_in_motion(self):
        sim = D.HairSim(
            self.rest, self.pivot, dict(global_stiffness=[0.3, 0.02]), self.caps
        )
        for n in range(90):
            T, R = turn(self.rest, 0.6 * np.sin(n * 0.15), axis=0, centre=(0, 0.05, 0))
            sim.step(T, D.gravity_change(R), self.caps)
            L = np.linalg.norm(np.diff(sim.x, axis=1), axis=2)
            free = np.arange(1, sim.P)[None] >= sim.free[:, None]
            self.assertLess(np.abs(L - sim.seg[:, None])[free].max(), 1e-9)
            pinned = np.arange(sim.P)[None] < sim.free[:, None]
            self.assertLess(np.abs(sim.x - T)[pinned].max(), 1e-15)
        # the hair moved away from the groom
        self.assertGreater(np.abs(sim.x - T).max(), 0.005)

    def test_settles_back_to_the_groom(self):
        # long strands (31 cm), soft toward the tip: the motion dies down within three seconds of rest
        rest = strands(P=32)
        params = dict(global_stiffness=[0.2, 0.01], local_stiffness=0.4, damping=0.05)
        sim = D.HairSim(rest, self.pivot, params, self.caps)
        for n in range(48):
            T, R = turn(
                rest, 0.3 * np.sin(2 * np.pi * n / 48), axis=0, centre=(0, 0.05, 0)
            )
            sim.step(T, D.gravity_change(R), self.caps)
        speed = [sim.step(rest, (0, 0, 0), self.caps) for _ in range(180)]
        self.assertGreater(speed[0], 0.1)
        self.assertLess(speed[-1], 0.004)
        self.assertLess(np.abs(sim.x - rest).max(), 1e-3)

    def test_colliders(self):
        # a horizontal strand, pulled down through a sphere under its middle
        P = 16
        rest = np.zeros((1, P, 3))
        rest[0, :, 0] = np.arange(P) * 0.01
        cap = np.array([[0.08, -0.035, 0.0, 0.08, -0.035, 0.0, 0.02]])
        down = rest - [0, 0.03, 0]
        depth = []
        for caps in (cap, np.zeros((0, 7))):
            sim = D.HairSim(rest, [1000.0], dict(global_stiffness=[0.5, 0.5]), caps)
            for _ in range(120):
                sim.step(down, (0, 0, 0), caps)
            d = np.linalg.norm(sim.x[0] - cap[0, :3], axis=1)
            depth.append(cap[0, 6] + sim.margin - d.min())
        # with the sphere the strand bends over it (the length step after the colliders pulls a point back in by
        # about 2 mm); without it the strand runs through the sphere
        self.assertLess(depth[0], 0.003)
        self.assertGreater(depth[1], 0.01)

    def test_offsets_start_at_the_pivot(self):
        from anny.hair.layout import load_layout
        from anny.hair.styles import load_style

        layout = load_layout()
        style = load_style("long_straight", layout)
        S, P = layout.simulated, style.points.shape[1]
        motion = np.ones((S, P, 3))
        off = sim_offsets(layout, style, motion)
        seg = np.linalg.norm(np.diff(style.points, axis=1), axis=2).sum(1) / (P - 1)
        arc = np.arange(P)[None] * seg[:, None]
        before = arc <= style.pivot[:, None]
        self.assertTrue(np.all(off[before] == 0))
        self.assertTrue(np.allclose(off[arc >= style.pivot[:, None] + 0.01], 1))


def data_available():
    man = BUILD / "manifest.json"
    return man.exists() and "hair" in json.loads(man.read_text())


@unittest.skipUnless(
    node_available() and data_available(), "node 22 and the viewer data are needed"
)
class TestPageSolver(unittest.TestCase):
    STYLES = ["long_straight", "medium_tousled", "high_ponytail"]
    STEPS = 90

    @classmethod
    def setUpClass(cls):
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        for name in cls.STYLES:
            subprocess.run(
                ["node", str(SCRIPT), str(BUILD), str(cls.dir), name, str(cls.STEPS)],
                check=True,
                cwd=REPO,
            )

    def test_matches_the_reference(self):
        for name in self.STYLES:
            info = json.loads((self.dir / f"{name}.json").read_text())
            S, P, N = info["S"], info["P"], info["N"]
            rest = np.fromfile(self.dir / f"{name}_rest.bin").reshape(S, P, 3)
            page = np.fromfile(self.dir / f"{name}_x.bin").reshape(N, S, P, 3)
            caps0 = np.array(info["fitted"]).reshape(-1, 7)
            sim = D.HairSim(rest, info["pivot"], info["params"], caps0)
            err = 0.0
            for n in range(N):
                M = np.array(info["moves"][n]).reshape(3, 4)
                # the page's order: M[r, 0] x + M[r, 1] y + M[r, 2] z + M[r, 3]
                T = np.stack(
                    [
                        M[r, 0] * rest[..., 0]
                        + M[r, 1] * rest[..., 1]
                        + M[r, 2] * rest[..., 2]
                        + M[r, 3]
                        for r in range(3)
                    ],
                    -1,
                )
                sim.step(
                    T, info["gravity"][n], np.array(info["capsules"][n]).reshape(-1, 7)
                )
                err = max(err, np.abs(sim.x - page[n]).max())
            self.assertLess(err, 1e-9, name)
            # the hair moved, and the lengths held
            self.assertGreater(np.abs(page[-1] - T).max(), 1e-3, name)
            L = np.linalg.norm(np.diff(page[-1], axis=1), axis=2)
            free = np.arange(1, P)[None] >= sim.free[:, None]
            self.assertLess(np.abs(L - sim.seg[:, None])[free].max(), 1e-9, name)


if __name__ == "__main__":
    unittest.main()
