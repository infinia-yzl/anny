# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
MediaPipe's face landmarks on anny: ``data/keypoints/mediapipe.json``.

Anny's adult head is registered to MediaPipe's canonical face model (Apache 2.0) with the tools
of :mod:`anny.faces.authoring.fit_3d`: alignment on 14 landmark pairs, a fit of anny's face
shapes and a non-rigid refinement. Each of the 468 canonical vertices then takes the closest
point of anny's refined head: three base-mesh vertices and their barycentric weights. The iris
centres (468 and 473) take anny's pupils.

Usage::

    python -m anny.faces.authoring.mediapipe_map
"""

from __future__ import annotations

import json

import numpy as np
import torch

from anny.faces.authoring import fit_3d, ict, nicp
from anny.faces.authoring.sources import fetch
from anny.faces.measurements import craniofacial_landmark_vertices
from anny.paths import get_anny_root_dir

OUTPUT = get_anny_root_dir() / "data" / "keypoints" / "mediapipe.json"
# anny craniofacial landmark -> canonical vertices whose mean stands for it
LANDMARK_PAIRS = {
    "n": [168],
    "prn": [4],
    "sn": [2],
    "ls": [0],
    "sto": [13, 14],
    "li": [17],
    "gn": [152],
    "en.L": [362],
    "en.R": [133],
    "ex.L": [263],
    "ex.R": [33],
    "ch.L": [291],
    "ch.R": [61],
    "al.L": [331],
    "al.R": [102],
}


def canonical_mesh():
    """MediaPipe's canonical face model in anny's frame (metres)"""
    V, F = [], []
    with open(fetch("mediapipe_canonical_face")) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    T = []
    for f in F:
        for k in range(1, len(f) - 1):
            T.append([f[0], f[k], f[k + 1]])
    return np.array(V) @ ict.TO_ANNY.T, np.array(T)


def run():
    model = fit_3d.fitting_model()
    V, T = canonical_mesh()
    landmarks = {k: V[idx].mean(0) for k, idx in LANDMARK_PAIRS.items()}
    reg = fit_3d.register(model, V, T, landmarks, max_distance=0.01)
    ids = reg["vertex_ids"]
    X, tri_local = reg["refined"], reg["local_triangles"]
    tri, bary, d = nicp.closest_triangles(reg["target_V"], X, tri_local)
    base = model.base_mesh_vertex_indices.numpy()
    corners = base[ids[tri_local[tri]]]
    print(
        f"canonical vertices on anny: median {1000 * np.median(d):.2f} mm, "
        f"max {1000 * d.max():.2f} mm"
    )
    out = {}
    for i in range(len(V)):
        out[str(i)] = [corners[i].tolist(), np.round(bary[i], 6).tolist()]
    pupils = craniofacial_landmark_vertices()
    for i, name in ((468, "pu.R"), (473, "pu.L")):
        idx = pupils[name]
        out[str(i)] = [idx, [1.0 / len(idx)] * len(idx)]
    with open(OUTPUT, "w") as f:
        json.dump(out, f)
    print(f"{len(out)} MediaPipe landmarks -> {OUTPUT}")


def mediapipe_regressor(model, indices=None):
    """KeypointsRegressor of MediaPipe's landmarks (468 face points and the iris centres)"""
    from anny.keypoints import KeypointsRegressor

    with open(OUTPUT) as f:
        table = json.load(f)
    indices = sorted(int(k) for k in table) if indices is None else list(indices)
    base = model.base_mesh_vertex_indices.cpu().numpy()
    lookup = -np.ones(int(base.max()) + 1, np.int64)
    lookup[base] = np.arange(len(base))
    width = max(len(table[str(i)][0]) for i in indices)
    idx = np.zeros((len(indices), width), np.int64)
    w = np.zeros((len(indices), width))
    for r, i in enumerate(indices):
        v, b = table[str(i)]
        idx[r, : len(v)] = lookup[np.array(v)]
        w[r, : len(v)] = b
    assert (idx >= 0).all(), "the topology lacks vertices of the MediaPipe landmarks"
    return KeypointsRegressor(
        torch.tensor(w, dtype=model.dtype), [str(i) for i in indices], torch.tensor(idx)
    ), indices


if __name__ == "__main__":
    run()
