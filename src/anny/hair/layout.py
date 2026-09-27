# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The scalp layout that every hairstyle shares, built by ``python -m anny.hair.authoring.layout``
and stored in ``data/hair/scalp_layout.safetensors``.

- ``guide_position`` (G, 3): the guide roots on anny's default body (legacy frame of the
  authoring rig), in progressive order, each root next to its mirror (``guide_mirror``);
- ``guide_sim``, ``guide_sim_weights`` (G, 3): the three simulated guides (the first
  ``simulated`` guides) that carry the motion of each guide, with weights that sum to 255;
- ``root_position`` (R, 3): the render roots in progressive order: a prefix of any length covers
  the scalp evenly, so the density and the level of detail are a strand count;
- ``root_guides``, ``root_weights`` (R, 4): the four nearest guides of each render root, with
  weights that sum to 255;
- ``guide_chart``, ``root_chart``: azimuth and elevation (``anny.hair.chart``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "hair"
DEFAULT_PATH = DATA_DIR / "scalp_layout.safetensors"


@dataclasses.dataclass
class Layout:
    guide_position: np.ndarray
    guide_chart: np.ndarray
    guide_mirror: np.ndarray
    guide_sim: np.ndarray
    guide_sim_weights: np.ndarray
    root_position: np.ndarray
    root_chart: np.ndarray
    root_guides: np.ndarray
    root_weights: np.ndarray
    meta: dict

    @property
    def guides(self) -> int:
        return len(self.guide_position)

    @property
    def roots(self) -> int:
        return len(self.root_position)

    @property
    def simulated(self) -> int:
        return int(self.meta["simulated"])

    @classmethod
    def load(cls, path=DEFAULT_PATH) -> "Layout":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            arrays = {
                k.replace("guides.", "guide_").replace("roots.", "root_"): f.get_tensor(
                    k
                )
                for k in f.keys()
            }
            meta = json.loads(f.metadata()["layout"])
        arrays = {
            k: v.astype(np.float64) if v.dtype == np.float32 else v
            for k, v in arrays.items()
        }
        return cls(**arrays, meta=meta)


_CACHE: dict = {}


def load_layout(path=DEFAULT_PATH) -> Layout:
    """the layout, loaded once per path"""
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = Layout.load(path)
    return _CACHE[key]
