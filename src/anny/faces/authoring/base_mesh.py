# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
The MakeHuman base mesh in anny's frame (metres, Z up, the face toward -Y, X toward the
figure's left), with the labels of its body faces and the vertices that MakeHuman targets move.
"""

from __future__ import annotations

import dataclasses
import functools
import gzip

import numpy as np
import roma
import torch
import yaml
from PIL import Image

import anny.utils.obj_utils
from anny.paths import get_anny_root_dir


@dataclasses.dataclass
class BaseMesh:
    V: np.ndarray  # (19158, 3) every vertex of base.obj, helpers included
    quads: np.ndarray  # (Q, 4) body faces
    face_labels: np.ndarray  # (Q,) segmentation label of each body face
    groups: dict[str, np.ndarray]  # vertex indices of each group of base.obj
    vertex_uv: (
        np.ndarray
    )  # (19158, 2) texture coordinates of the body vertices (NaN elsewhere)

    @functools.cached_property
    def triangles(self) -> np.ndarray:
        q = self.quads
        return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])

    @functools.cached_property
    def body_vertices(self) -> np.ndarray:
        return np.unique(self.quads)


def world_transformation(dtype=torch.float64):
    """MakeHuman decimetres, Y up -> anny metres, Z up (as anny.models.full_model.load_data)"""
    return roma.Linear(
        0.1 * roma.euler_to_rotmat("X", [90], degrees=True, dtype=dtype)
    )[None]


@functools.lru_cache(maxsize=1)
def base_mesh() -> BaseMesh:
    root = get_anny_root_dir()
    V, UV, groups = anny.utils.obj_utils.load_obj_file(
        str(root / "data/mpfb2/3dobjs/base.obj"), dtype=torch.float64
    )
    V = world_transformation().apply(V).numpy()
    body = groups["body"]
    quads = body["face_vertex_indices"].numpy()
    face_uv = body["face_texture_coordinate_indices"].numpy()
    # segmentation label of each body face from the centre of its texture coordinates
    image = np.asarray(
        Image.open(root / "data/segmentation/body_parts_segmentation.png").convert(
            "RGB"
        )
    )
    with open(root / "data/segmentation/body_parts_segmentation.yaml") as f:
        colors = yaml.safe_load(f)["colors"]
    centre = UV.numpy()[face_uv].mean(1)
    u = np.clip(
        np.round(centre[:, 0] * image.shape[1]).astype(int), 0, image.shape[1] - 1
    )
    v = np.clip(
        np.round((1 - centre[:, 1]) * image.shape[0]).astype(int), 0, image.shape[0] - 1
    )
    rgb = image[v, u]
    labels = np.full(len(quads), "", dtype=object)
    for name, color in colors.items():
        labels[np.all(rgb == np.asarray(color), axis=-1)] = name
    vertex_groups = {
        name: np.unique(g["face_vertex_indices"].numpy().ravel())
        for name, g in groups.items()
        if isinstance(g["face_vertex_indices"], torch.Tensor)
    }
    vertex_uv = np.full((len(V), 2), np.nan)
    vertex_uv[quads.ravel()] = UV.numpy()[face_uv.ravel()]
    return BaseMesh(
        V=V, quads=quads, face_labels=labels, groups=vertex_groups, vertex_uv=vertex_uv
    )


def texture_mask(name: str, mesh: BaseMesh | None = None) -> np.ndarray:
    """value (0 to 1) of a MakeHuman mask texture such as ``mpfb_lips`` at each vertex"""
    mesh = mesh or base_mesh()
    image = (
        np.asarray(
            Image.open(
                get_anny_root_dir() / "data/mpfb2/textures" / f"{name}.jpg"
            ).convert("L"),
            dtype=np.float64,
        )
        / 255.0
    )
    uv = np.nan_to_num(mesh.vertex_uv, nan=-1.0)
    u = np.clip(np.round(uv[:, 0] * image.shape[1]).astype(int), 0, image.shape[1] - 1)
    v = np.clip(
        np.round((1 - uv[:, 1]) * image.shape[0]).astype(int), 0, image.shape[0] - 1
    )
    out = image[v, u]
    out[mesh.vertex_uv[:, 0] != mesh.vertex_uv[:, 0]] = 0.0
    return out


def read_target(name: str) -> tuple[np.ndarray, np.ndarray]:
    """indices and offsets (anny metres) of a MakeHuman target such as ``ears/l-ear-lobe-incr``"""
    path = get_anny_root_dir() / "data/mpfb2/targets" / f"{name}.target.gz"
    idx, d = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and not line.startswith("#"):
                idx.append(int(p[0]))
                d.append([float(x) for x in p[1:]])
    d = world_transformation().apply(torch.tensor(d, dtype=torch.float64)).numpy()
    return np.array(idx), d


def moved_vertices(
    folder: str, prefix: str = "", threshold: float = 1e-5
) -> np.ndarray:
    """vertices that any target of a MakeHuman folder (and name prefix) moves by more than threshold"""
    root = get_anny_root_dir() / "data/mpfb2/targets" / folder
    out = set()
    for path in sorted(root.glob(f"{prefix}*.target.gz")):
        idx, d = read_target(f"{folder}/{path.name[: -len('.target.gz')]}")
        out.update(idx[np.linalg.norm(d, axis=1) > threshold].tolist())
    return np.array(sorted(out))


def margin_vertices(mesh: BaseMesh, label: str) -> np.ndarray:
    """vertices on the edge between the faces of a segmentation label and the other body faces"""
    inside = mesh.face_labels == label
    edges_in, edges_out = set(), set()
    for q, is_in in zip(mesh.quads, inside):
        for k in range(4):
            e = (min(q[k], q[(k + 1) % 4]), max(q[k], q[(k + 1) % 4]))
            (edges_in if is_in else edges_out).add(e)
    shared = edges_in & edges_out
    return np.array(sorted({v for e in shared for v in e}))
