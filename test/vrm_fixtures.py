# OpenSculptBoy
# Apache License, Version 2.0
"""
Shared VRM exports for the tests: each file is written once per test process and cached, so
that the VRM test modules stay within their time budget.

CONTRACT (PR 1): stream A4 may add fixtures; every VRM test module uses these.
"""

import functools
import pathlib
import tempfile

import torch

from opensculptboy import Character
from opensculptboy.export.document import read_gltf_json

_TMP = tempfile.TemporaryDirectory()

# The character of the shared exports: off the default shape, with a face shape and a facial
# action (the VRM export bakes the face shape and ignores the action).
CHARACTER = Character(
    name="fixture",
    phenotype={"age": 0.3, "weight": 0.7, "muscle": 0.6},
    face_shapes={"nose-scale-horiz": 0.4},
    facial_actions={"jawOpen": 0.25},
)


@functools.lru_cache(maxsize=None)
def model(face_shapes: bool = True):
    """The Anny model of the shared exports (float64)."""
    return CHARACTER.build_model(face_shapes=face_shapes, dtype=torch.float64)


@functools.lru_cache(maxsize=None)
def vrm_export(version: str = "1.0", **options) -> tuple[pathlib.Path, dict]:
    """The path and summary of a VRM export of ``CHARACTER``; ``options`` go to export_vrm."""
    from opensculptboy.export.vrm import export_vrm

    path = (
        pathlib.Path(_TMP.name)
        / f"fixture_{version}_{len(options)}_{hash(tuple(sorted(options.items())))}.vrm"
    )
    summary = export_vrm(
        path,
        CHARACTER,
        model(),
        version=version,
        author="OpenSculptBoy tests",
        **options,
    )
    return path, summary


def vrm_json(version: str = "1.0", **options) -> dict:
    """The JSON chunk of the shared VRM export."""
    return read_gltf_json(vrm_export(version, **options)[0])
