# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Face-shape parameters: named, symmetric shapes of the head and the face.

Each parameter sums the MakeHuman targets of both sides of the face
(``data/faces/face_shapes.json``, written by ``scripts/make_face_shape_spec.py``). A value of +1
applies the positive targets and -1 the negative targets; the head archetypes and
``chin-triangle`` have a positive direction only. Each direction is one blend-shape row, labelled
``face_shape:{name}.pos`` or ``face_shape:{name}.neg``.

The ``detail`` parameters come from the ICT-FaceKit identity space instead: they are the
symmetric principal components of what the MakeHuman parameters miss of the ICT identities
(``data/faces/detail_shapes.safetensors``, written by ``python -m anny.faces.authoring.detail``).

The rows of a group scale with the size of the matching part of the head (see
``Anny.face_shape_scales``), so that the offsets, which MakeHuman authored on an adult, stay in
proportion on a child.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json

import torch

from anny.paths import get_anny_root_dir

# Groups whose rows share one scale: the size measurement of each is in anny.faces.measurements
SCALE_GROUPS = ("head", "forehead", "eyes", "nose", "mouth", "chin", "ears")
GROUP_TO_SCALE = {
    "head": "head",
    "forehead": "forehead",
    "brows": "forehead",
    "eyes": "eyes",
    "nose": "nose",
    "cheeks": "chin",
    "mouth": "mouth",
    "chin": "chin",
    "ears": "ears",
    "detail": "head",
}

FACES_DIR = get_anny_root_dir() / "data" / "faces"
DETAIL_PATH = FACES_DIR / "detail_shapes.safetensors"


@dataclasses.dataclass(frozen=True)
class FaceShapeParameter:
    name: str
    group: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    range: tuple[float, float]
    # "makehuman": sums of the MakeHuman targets named in positive and negative;
    # "ict": a component of DETAIL_PATH, whose negative direction is minus the positive one
    source: str = "makehuman"

    @property
    def row_labels(self) -> list[str]:
        labels = [f"face_shape:{self.name}.pos"]
        if self.negative or self.source == "ict":
            labels.append(f"face_shape:{self.name}.neg")
        return labels


@functools.lru_cache(maxsize=1)
def face_shape_spec() -> tuple[FaceShapeParameter, ...]:
    """the face-shape parameters, in the order of their blend-shape rows"""
    with open(FACES_DIR / "face_shapes.json") as f:
        spec = json.load(f)
    return tuple(
        FaceShapeParameter(
            name=p["name"],
            group=p["group"],
            positive=tuple(p["positive"]),
            negative=tuple(p["negative"]),
            range=(float(p["range"][0]), float(p["range"][1])),
            source=p.get("source", "makehuman"),
        )
        for p in spec["parameters"]
    )


def face_shape_data_digest() -> str:
    """
    a digest of the data that the face-shape rows and the craniofacial landmarks of the model data
    come from, for the model cache. The slider ranges stay out: the calibration rewrites them, and
    they do not change the rows, so a new calibration keeps the cached models.
    """
    h = hashlib.sha256()
    with open(FACES_DIR / "face_shapes.json") as f:
        spec = json.load(f)
    rows = [{k: v for k, v in p.items() if k != "range"} for p in spec["parameters"]]
    h.update(json.dumps(rows, sort_keys=True).encode())
    landmarks = get_anny_root_dir() / "data" / "keypoints" / "craniofacial.json"
    for path in (DETAIL_PATH, landmarks):
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def load_detail_shapes(
    dtype: torch.dtype = torch.float64,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """base-mesh vertex indices (N,) and offsets (N, 3), in anny's frame, of each detail component"""
    from safetensors import safe_open

    with safe_open(str(DETAIL_PATH), "pt") as f:
        names = json.loads(f.metadata()["detail"])["names"]
        data = {k: f.get_tensor(k) for k in f.keys()}
    return {
        name: (data["vertex_indices"].long(), data["offsets"][k].to(dtype))
        for k, name in enumerate(names)
    }


def face_shape_parameter(name: str) -> FaceShapeParameter:
    for p in face_shape_spec():
        if p.name == name:
            return p
    raise KeyError(name)


def parse_row_label(label: str) -> tuple[str, int]:
    """``face_shape:{name}.pos`` -> (name, +1); ``face_shape:{name}.neg`` -> (name, -1)"""
    block, rest = label.split(":")
    assert block == "face_shape", label
    name, direction = rest.rsplit(".", 1)
    return name, 1 if direction == "pos" else -1


# A few MakeHuman face targets carry stray entries far from the face (such as one vertex of the
# pelvis in mouth-upperlip-middle); entries farther than this from the median of a target's moved
# vertices are dropped
STRAY_DISTANCE = 0.3


def load_face_shape_blendshapes(
    template_vertices: torch.Tensor,
    world_transformation,
    dtype: torch.dtype = torch.float64,
) -> tuple[list[str], torch.Tensor]:
    """Row labels and blend shapes (R, V, 3) of every face-shape direction."""
    from anny.models.full_model import load_blend_shape

    targets_dir = get_anny_root_dir() / "data" / "mpfb2" / "targets"
    vertices_count = len(template_vertices)

    def summed(names):
        total = torch.zeros((vertices_count, 3), dtype=dtype)
        for name in names:
            offsets = load_blend_shape(
                targets_dir / f"{name}.target.gz",
                vertices_count=vertices_count,
                world_transformation=world_transformation,
                dtype=dtype,
            )
            moved = offsets.norm(dim=-1) > 0
            centre = template_vertices[moved].median(dim=0).values
            far = (template_vertices - centre).norm(dim=-1) > STRAY_DISTANCE
            offsets[far] = 0.0
            total += offsets
        return total

    def detail(name):
        idx, offsets = details[name]
        total = torch.zeros((vertices_count, 3), dtype=dtype)
        total[idx] = offsets
        return total

    spec = face_shape_spec()
    details = load_detail_shapes(dtype) if any(p.source == "ict" for p in spec) else {}
    labels, rows = [], []
    for p in spec:
        labels.append(f"face_shape:{p.name}.pos")
        if p.source == "ict":
            rows.append(detail(p.name))
            labels.append(f"face_shape:{p.name}.neg")
            rows.append(-rows[-1])
            continue
        rows.append(summed(p.positive))
        if p.negative:
            labels.append(f"face_shape:{p.name}.neg")
            rows.append(summed(p.negative))
    return labels, torch.stack(rows)
