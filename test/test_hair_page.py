# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The hair passes of the viewer page (viewer/src/hair/glsl.ts) against their NumPy reference
(anny.hair.styles). The page runs in Chromium with SwiftShader through Playwright; the test skips
when Playwright or the built page is missing.

- pass A: the guides of a style on anny's default body match the decoded guides of the data;
- pass B: from the page's own guides and render roots, the reference builds the same strands;
- the physics: while the head nods, pass A adds the motion of the simulated guides as
  anny.hair.styles.sim_offsets blends it, and the solver sleeps once the head rests.
"""

import pathlib
import unittest

import numpy as np

from anny.hair import styles as H
from anny.hair.layout import load_layout

REPO = pathlib.Path(__file__).resolve().parents[1]
PAGE = REPO / "viewer" / "dist" / "anny_viewer.html"
CHROMIUM = pathlib.Path("/opt/pw-browsers/chromium")
N = 3000


def playwright_available():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return PAGE.exists()


def unpack_normals(w):
    """the normals packed by viewer/src/hair/data.ts packNormal (12-bit octahedral)"""
    p = np.round(w).astype(np.int64)
    x = (p & 4095) / 4095.0 * 2 - 1
    y = (p >> 12) / 4095.0 * 2 - 1
    z = 1 - np.abs(x) - np.abs(y)
    t = np.maximum(-z, 0)
    x = x + np.where(x >= 0, -t, t)
    y = y + np.where(y >= 0, -t, t)
    n = np.stack([x, y, z], 1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestHairPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright

        args = [
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
        ]
        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=args, **kw)
            tab = browser.new_page(viewport=dict(width=320, height=320))
            cls.errors = []
            tab.on(
                "console",
                lambda m: cls.errors.append(m.text)
                if "Shader Error" in m.text
                else None,
            )
            tab.goto(PAGE.resolve().as_uri() + "?shot=1&acc=1")
            tab.wait_for_function(
                "window.__READY && window.__BODY.ready", timeout=900000
            )
            if not tab.evaluate("() => typeof window.hairRest === 'function'"):
                browser.close()
                raise unittest.SkipTest("the page has no hair passes")
            cls.style_name = tab.evaluate("() => window.hairStats().style")
            cls.rest = tab.evaluate(f"() => window.hairRest({N})")
            tab.evaluate(
                "() => window.setHairParams({length: 0.8, curl: 1.3, volume: 1.2})"
            )
            cls.rest2 = tab.evaluate(f"() => window.hairRest({N})")
            # the physics on long hair: the head nods for 0.8 s, then rests
            tab.evaluate(
                "() => { window.setHairStyle('long_straight'); window.setHairPhysics(true); }"
            )
            for k in range(48):
                tab.evaluate(
                    f"() => {{ window.setMotion('nod', {k / 60}, true); window.stepHair(1 / 60); }}"
                )
            cls.moving = tab.evaluate("() => window.hairGuides()")
            cls.after = [
                tab.evaluate("() => window.stepHair(1 / 60)") for _ in range(240)
            ]
            browser.close()
        cls.layout = load_layout()
        cls.style = H.load_style(cls.style_name, cls.layout)

    def guides(self, rest):
        G, P = rest["G"], rest["P"]
        A = np.array(rest["guides"], np.float64).reshape(G, P + 2, 4)
        return A[:, :P, :3], A[:, P, :3]

    def test_shaders_compile(self):
        self.assertEqual(self.errors, [])

    def test_pass_a_follows_the_default_body(self):
        guides, normals = self.guides(self.rest)
        # the page rebuilds anny's default body by subdivision: a few micrometres from the data's body
        self.assertLess(np.abs(guides - self.style.points).max(), 5e-5)
        self.assertLess(np.abs(np.linalg.norm(normals, axis=1) - 1).max(), 1e-5)

    def check_pass_b(self, rest):
        guides, normals = self.guides(rest)
        P = rest["P"]
        R = np.array(rest["roots"], np.float64).reshape(N, 4)
        roots = H.Roots(R[:, :3], unpack_normals(R[:, 3]))
        prm = rest["params"]
        ref, ell = H.strands(
            self.style,
            self.layout,
            guides,
            normals,
            roots,
            prm,
            count=N,
            scale=rest["scale"],
        )
        page = np.array(rest["points"], np.float64).reshape(N, P, 4)
        alive_page = page[:, 0, 3] >= 0
        alive_ref = ell > 1e-4 * rest["scale"]
        self.assertGreater((alive_page == alive_ref).mean(), 0.999)
        both = alive_page & alive_ref
        err = np.abs(page[both, :, :3] - ref[both]).max(axis=(1, 2))
        # float32 against float64; a strand next to the edge of a clump sector may fall on the other side
        self.assertGreater((err < 5e-5).mean(), 0.99)
        self.assertLess(err.max(), 5e-3)

    def test_pass_b_matches_the_reference(self):
        self.check_pass_b(self.rest)

    def test_pass_b_with_parameters(self):
        self.check_pass_b(self.rest2)

    def test_physics_motion_in_pass_a(self):
        m = self.moving
        G, P, S = m["G"], m["P"], m["S"]
        on = np.array(m["on"]).reshape(G, P + 2, 4)[:, :P, :3]
        off = np.array(m["off"]).reshape(G, P + 2, 4)[:, :P, :3]
        motion = np.array(m["motion"]).reshape(S, P, 4)[..., :3]
        style = H.load_style(m["style"], self.layout)
        ref = H.sim_offsets(self.layout, style, motion)
        self.assertGreater(np.abs(motion).max(), 0.005)
        self.assertLess(np.abs((on - off) - ref).max(), 2e-5)

    def test_physics_sleeps(self):
        asleep = [s["asleep"] for s in self.after]
        self.assertFalse(asleep[0])
        self.assertTrue(asleep[-1])
        # once asleep, the solver stays asleep while the head rests
        k = asleep.index(True)
        self.assertTrue(all(asleep[k:]))


if __name__ == "__main__":
    unittest.main()
