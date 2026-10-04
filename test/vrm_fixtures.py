# OpenSculptBoy
# Apache License, Version 2.0
"""
Shared VRM exports for the tests: each file is written once per test process and cached, so
that the VRM test modules stay within their time budget.

- :func:`vrm_export` gives the path and the summary of an export of :data:`CHARACTER`
  (``version`` "1.0" or "0.x", other options as for ``export_vrm``), and :func:`vrm_json` its
  JSON chunk.
- :func:`vrm_build` gives the :class:`~opensculptboy.export.vrm.VrmSpec` behind an export: the
  rebind, the centring offset, the file's skin weights and the source vertex of each file
  vertex, for tests that compare the file with Anny.
- :func:`vrm_glb` gives the export read by ``test.gltf_reader.GLB``.

Exports that differ only in ``budget`` share one spec.
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
AUTHOR = "OpenSculptBoy tests"


@functools.lru_cache(maxsize=None)
def model(face_shapes: bool = True):
    """The Anny model of the shared exports (float64)."""
    return CHARACTER.build_model(face_shapes=face_shapes, dtype=torch.float64)


@functools.lru_cache(maxsize=None)
def vrm_build(version: str = "1.0", **options):
    """The VrmSpec of a VRM export of ``CHARACTER``; ``options`` go to ``vrm_spec``."""
    from opensculptboy.export.vrm import vrm_spec

    return vrm_spec(CHARACTER, model(), version=version, author=AUTHOR, **options)


@functools.lru_cache(maxsize=None)
def vrm_export(version: str = "1.0", **options) -> tuple[pathlib.Path, dict]:
    """The path and summary of a VRM export of ``CHARACTER``; ``options`` go to export_vrm."""
    from opensculptboy.export.vrm import write_vrm

    budget = options.pop("budget", "warn")
    path = (
        pathlib.Path(_TMP.name)
        / f"fixture_{version}_{len(options)}_{budget}_{abs(hash(tuple(sorted(options.items()))))}.vrm"
    )
    summary = write_vrm(vrm_build(version, **options), path, budget=budget)
    return path, summary


def vrm_json(version: str = "1.0", **options) -> dict:
    """The JSON chunk of the shared VRM export."""
    return read_gltf_json(vrm_export(version, **options)[0])


@functools.lru_cache(maxsize=None)
def vrm_glb(version: str = "1.0", **options):
    """The shared VRM export, read by ``test.gltf_reader.GLB``."""
    from test.gltf_reader import GLB

    return GLB(vrm_export(version, **options)[0])
