# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The anatomy that places anny's hairline, on anny's default body in the legacy frame of the
authoring rig (``anny.poses.authoring.rig``):

- :func:`landmarks`: the craniofacial landmarks (``data/keypoints/craniofacial.json``), such as
  the nasion ``n``, the chin ``gn``, the tragion ``t.L``, the top of the ear ``sa.L`` and the back
  of the ear ``pa.L``;
- :func:`pinna`: the outer ear, the vertices that MakeHuman's ear-translation targets move in full
  (the targets fade out over the skin around the ear).

The hairline of ``anny.hair.chart`` rests on them: the sideburn stops in front of the ear, and the
hairline clears the top of the ear. ``test.test_hair_styles.TestHairlineAnatomy`` checks both.
"""

from __future__ import annotations

import functools

import numpy as np

# the share of the ear target's largest offset that marks the rigid ear
PINNA_WEIGHT = 0.99


def _coarse():
    from anny.poses.authoring.rig import authoring_rig

    return authoring_rig().preview["coarse"]


def _base_positions() -> np.ndarray:
    """the positions of the MakeHuman base-mesh vertices on anny's default body (NaN where unused)"""
    c = _coarse()
    base = c["base_index"]
    out = np.full((int(base.max()) + 1, 3), np.nan)
    out[base] = c["V"]
    return out


@functools.lru_cache(maxsize=1)
def landmarks() -> dict[str, np.ndarray]:
    """the craniofacial landmarks of anny's default body (metres, legacy frame)"""
    from anny.faces.measurements import craniofacial_landmark_vertices

    at = _base_positions()
    return {
        k: at[np.asarray(v)].mean(0)
        for k, v in craniofacial_landmark_vertices().items()
    }


@functools.lru_cache(maxsize=1)
def pinna_vertices() -> np.ndarray:
    """the indices of the vertices of both outer ears in anny's coarse mesh"""
    from anny.viewer.landmarks import read_target

    base = _coarse()["base_index"]
    keep = []
    for side in "lr":
        idx, d = read_target(f"ears/{side}-ear-trans-up")
        m = np.linalg.norm(d, axis=1)
        keep.append(idx[m >= PINNA_WEIGHT * m.max()])
    return np.flatnonzero(np.isin(base, np.concatenate(keep)))


def pinna() -> np.ndarray:
    """the points (n, 3) of both outer ears of anny's default body"""
    return _coarse()["V"][pinna_vertices()]


def pinna_mask(subdivision) -> np.ndarray:
    """the vertices of a subdivided body (``anny.viewer.geometry.fine_body``) that lie on the outer ears"""
    ind = np.zeros((len(_coarse()["V"]), 1))
    ind[pinna_vertices()] = 1.0
    return subdivision(ind)[:, 0] > 0.5
