# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
A review grid of random faces, rendered by the viewer page (``viewer/dist/anny_viewer.html``).

Each row holds anny's default face, the mean face and random faces of
:class:`anny.faces.distribution.FaceShapeDistribution` at its default spread, for girls and boys,
women and men at several ages, and for Asian and Eurasian adults. Check every change to the face
calibration, the face shapes or the viewer's random faces with this grid as well as with the tests:
each face must pass as a normal person of that age. The mean face stands next to the default face,
since a shift of the mean hides behind the default face otherwise.

Usage::

    python -m anny.faces.authoring.review [--prior PATH] [--out PATH] [--seed N]

The grid goes to ``ANNY_CACHE_DIR/faces/review.png`` by default.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

import anny
from anny.faces.authoring.photos import render_anny
from anny.faces.authoring.sources import cache_dir
from anny.faces.distribution import DEFAULT_PATH, FaceShapeDistribution
from anny.shape_distribution import SimpleShapeDistribution

ASIAN = dict(african=0.0, asian=1.0, caucasian=0.0)
EURASIAN = dict(african=0.0, asian=1.0, caucasian=1.0)
# (label, years, gender, race values)
ROWS = [
    ("girl, 8 years", 8.0, 1.0, {}),
    ("boy, 8 years", 8.0, 0.0, {}),
    ("girl, 12 years", 12.0, 1.0, {}),
    ("boy, 12 years", 12.0, 0.0, {}),
    ("woman, 30 years", 30.0, 1.0, {}),
    ("man, 30 years", 30.0, 0.0, {}),
    ("woman, 75 years", 75.0, 1.0, {}),
    ("man, 75 years", 75.0, 0.0, {}),
    ("Asian woman, 30 years", 30.0, 1.0, ASIAN),
    ("Asian man, 30 years", 30.0, 0.0, ASIAN),
    ("Eurasian woman, 30 years", 30.0, 1.0, EURASIAN),
    ("Eurasian man, 30 years", 30.0, 0.0, EURASIAN),
]


def review(
    prior: pathlib.Path = DEFAULT_PATH,
    out: pathlib.Path | None = None,
    random_faces: int = 5,
    seed: int = 0,
    size: int = 224,
) -> pathlib.Path:
    """render the review grid and return its path"""
    from PIL import Image, ImageDraw

    model = anny.Anny(face_shapes="all", phenotypes="all").to(dtype=torch.float64)
    dist = FaceShapeDistribution(model, prior)
    mapping = SimpleShapeDistribution(model).morphological_age_mapping
    labels = model.face_shape_labels
    generator = torch.Generator().manual_seed(seed)
    looks = []
    for _, years, gender, race in ROWS:
        age = float(mapping.morphological_to_anny_age(torch.tensor([years])).item())
        phenotype = dict(age=age, gender=gender, **race)
        batch = {k: torch.full((random_faces,), v) for k, v in phenotype.items()}
        with torch.no_grad():
            mean = dist.mean(phenotype)[0].numpy()
            faces = dist.sample(batch, generator=generator).numpy()
        for face in [np.zeros(len(labels)), mean, *faces]:
            looks.append(
                dict(
                    format="anny-viewer/look@2",
                    name="review",
                    phenotype=phenotype,
                    face={k: float(v) for k, v in zip(labels, face) if v != 0},
                )
            )
    images = render_anny(looks, size=size, margin=1.2)
    columns = 2 + random_faces
    grid = Image.new("RGB", (size * columns, size * len(ROWS)), "white")
    draw = ImageDraw.Draw(grid)
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        grid.paste(Image.fromarray(image), (c * size, r * size))
    for r, (label, *_) in enumerate(ROWS):
        draw.text(
            (4, r * size + 4),
            f"{label}: default face, mean face, {random_faces} random faces",
            fill=(220, 40, 40),
        )
    # a white line between the default face and the faces of the distribution
    draw.line([(size, 0), (size, size * len(ROWS))], fill=(255, 255, 255), width=3)
    out = out or cache_dir() / "review.png"
    grid.save(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="render a review grid of random faces")
    parser.add_argument("--prior", type=pathlib.Path, default=DEFAULT_PATH)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(review(args.prior, args.out, seed=args.seed))


if __name__ == "__main__":
    main()
