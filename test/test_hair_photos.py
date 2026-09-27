# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The measurements of the hairline benchmark (anny.hair.authoring.photos) on drawn faces: no
MediaPipe is needed.
"""

import unittest

import numpy as np

from anny.hair.authoring import photos as H


def face(size=160, n=(80.0, 70.0), D=50.0, half=40.0):
    """landmarks of an upright face: nasion ``n``, chin D pixels below it, sides ``half`` beside it"""
    P = np.zeros((478, 2))
    P[H.N] = n
    P[H.GN] = (n[0], n[1] + D)
    P[H.STO] = (n[0], n[1] + 0.62 * D)
    P[H.PUPILS[0]] = (n[0] - 0.3 * D, n[1])
    P[H.PUPILS[1]] = (n[0] + 0.3 * D, n[1])
    P[H.SIDES[0]] = (n[0] - half, n[1] + 0.1 * D)
    P[H.SIDES[1]] = (n[0] + half, n[1] + 0.1 * D)
    return P


class TestHairMetrics(unittest.TestCase):
    def test_a_level_hairline(self):
        # hair above the row 30 pixels above the nasion, over the whole image
        P = face()
        mask = np.zeros((H.MASK, H.MASK))
        mask[: 70 - 30] = 1.0
        m = H.hair_metrics(P, mask, H.MASK)
        for k in ("forehead", "pupil", "temple"):
            self.assertAlmostEqual(m[k], 30 / 50, delta=0.02)
        self.assertAlmostEqual(m["sideburn"], 30 / 50, delta=0.02)
        self.assertTrue(m["short"])

    def test_sideburns_and_long_hair(self):
        P = face()
        mask = np.zeros((H.MASK, H.MASK))
        mask[: 70 - 30] = 1.0
        # sideburns down to 10 pixels below the nasion, beside the face
        for x in (80 - 40, 80 + 40):
            mask[: 70 + 10, x - 4 : x + 5] = 1.0
        m = H.hair_metrics(P, mask, H.MASK)
        self.assertAlmostEqual(m["sideburn"], -10 / 50, delta=0.03)
        self.assertAlmostEqual(m["forehead"], 30 / 50, delta=0.02)
        self.assertTrue(m["short"])
        # hair that falls past the mouth beside the face is long
        for x in (80 - 48, 80 + 48):
            mask[: 70 + 45, x - 3 : x + 4] = 1.0
        self.assertFalse(H.hair_metrics(P, mask, H.MASK)["short"])

    def test_a_fringe_over_the_eyes(self):
        # the hair covers the midline down to 5 pixels below the nasion: the edge is negative
        P = face()
        mask = np.zeros((H.MASK, H.MASK))
        mask[: 70 + 5] = 1.0
        self.assertAlmostEqual(
            H.hair_metrics(P, mask, H.MASK)["forehead"], -5 / 50, delta=0.03
        )


if __name__ == "__main__":
    unittest.main()
