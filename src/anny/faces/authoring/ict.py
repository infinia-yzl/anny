# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The ICT Face Model Light (ICT-FaceKit, MIT licence; Li et al., CVPR 2020) in anny's frame.

Its identity space holds 100 linear modes from light-stage scans of about 178 adults aged 18 to
67; a random identity draws each weight from a standard normal (``ict_face_model.py``). The model
is in centimetres, Y up, the face toward +Z; anny is in metres, Z up, the face toward -Y, with X
toward the figure's left in both.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np

from anny.faces.authoring.sources import cache_dir, fetch_ict_facekit

# vertex ranges of the geometries of the model (README of ICT-FaceKit)
FACE = (0, 9409)
HEAD_AND_NECK = (9409, 11248)
# Multi-PIE 68 landmarks (README of ICT-FaceKit), in the usual order
MULTI_PIE_68 = [
    1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272,
    4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942,
    4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756,
    3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841,
    5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966,
]  # fmt: skip
# anny craniofacial landmark -> Multi-PIE landmarks whose mean stands for it
LANDMARK_PAIRS = {
    "n": [27],
    "prn": [30],
    "sn": [33],
    "ls": [51],
    "sto": [62, 66],
    "li": [57],
    "gn": [8],
    "en.L": [42],
    "en.R": [39],
    "ex.L": [45],
    "ex.R": [36],
    "ch.L": [54],
    "ch.R": [48],
    "al.L": [35],
    "al.R": [31],
}
TO_ANNY = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]) / 100.0


@dataclasses.dataclass
class IctModel:
    V: np.ndarray  # (N, 3) neutral mesh in anny's frame (metres)
    triangles: np.ndarray  # (T, 3) triangles of the face and the head and neck
    modes: np.ndarray  # (100, N, 3) identity modes for a weight of 1

    def landmarks(self, V: np.ndarray | None = None) -> dict[str, np.ndarray]:
        V = self.V if V is None else V
        return {
            name: V[[MULTI_PIE_68[i] for i in idx]].mean(0)
            for name, idx in LANDMARK_PAIRS.items()
        }

    def sample(self, weights: np.ndarray) -> np.ndarray:
        """meshes (B, N, 3) for identity weights (B, 100)"""
        return self.V[None] + np.einsum("bk, knd -> bnd", weights, self.modes)


def _read_obj(path):
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(V), F


@functools.lru_cache(maxsize=1)
def load() -> IctModel:
    cached = cache_dir() / "ict_model.npz"
    if cached.exists():
        with np.load(cached) as z:
            return IctModel(V=z["V"], triangles=z["triangles"], modes=z["modes"])
    root = fetch_ict_facekit() / "FaceXModel"
    V, F = _read_obj(root / "generic_neutral_mesh.obj")
    n = HEAD_AND_NECK[1]
    triangles = []
    for f in F:
        if max(f) < n:
            for k in range(1, len(f) - 1):
                triangles.append([f[0], f[k], f[k + 1]])
    modes = []
    for i in range(100):
        Vi, _ = _read_obj(root / f"identity{i:03d}.obj")
        modes.append((Vi[:n] - V[:n]) @ TO_ANNY.T)
    model = IctModel(
        V=V[:n] @ TO_ANNY.T,
        triangles=np.array(triangles),
        modes=np.stack(modes),
    )
    np.savez(cached, V=model.V, triangles=model.triangles, modes=model.modes)
    return model
