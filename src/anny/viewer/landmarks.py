# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Landmarks of anny's default body in the legacy frame of the authoring rig: the centres of the
MakeHuman joint helpers (``joint-*`` groups of the base mesh) and the nipples.
"""

from __future__ import annotations

import functools
import gzip

import numpy as np
import torch

from anny.paths import get_anny_root_dir
from anny.poses.authoring import posing as P


def base_mesh_groups() -> dict:
    """vertex indices of each group of the MakeHuman base mesh"""
    groups, current = {}, None
    path = get_anny_root_dir() / "data" / "mpfb2" / "3dobjs" / "base.obj"
    with open(path) as f:
        for line in f:
            if line.startswith("g "):
                current = line.split()[1]
            elif line.startswith("f ") and current is not None:
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                groups.setdefault(current, set()).update(idx)
    return {g: np.array(sorted(v)) for g, v in groups.items()}


def read_target(name: str) -> tuple[np.ndarray, np.ndarray]:
    """indices and offsets (decimetres, MakeHuman axes) of a MakeHuman target"""
    path = get_anny_root_dir() / "data" / "mpfb2" / "targets" / f"{name}.target.gz"
    idx, d = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and not line.startswith("#"):
                idx.append(int(p[0]))
                d.append([float(x) for x in p[1:]])
    return np.array(idx), np.array(d)


@functools.lru_cache(maxsize=1)
def joint_helpers() -> dict:
    import anny

    model = anny.Anny(topology="makehuman").to(dtype=torch.float64)
    with torch.no_grad():
        rest = model(phenotype_kwargs={})["rest_vertices"][0].numpy()
    base_index = model.base_mesh_vertex_indices.numpy()
    position = np.zeros((int(base_index.max()) + 1, 3))
    position[base_index] = P.RIG.to_legacy(rest)
    out = {}
    for group, idx in base_mesh_groups().items():
        if group.startswith("joint-"):
            out[group[6:]] = position[idx].mean(0)
    # nipples: the vertex that the nipple-point target moves the most on each side
    idx, d = read_target("breast/nipple-point-incr")
    magnitude = np.linalg.norm(d, axis=1)
    for s, key in ((1, "l-nipple"), (-1, "r-nipple")):
        side = position[idx, 0] * s > 0
        out[key] = position[idx[np.argmax(magnitude * side)]]
    return out
