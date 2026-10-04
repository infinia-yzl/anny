# OpenSculptBoy
# Apache License, Version 2.0
"""
Budgets of VRM files: the triangles, joints, materials and texture sizes that VTuber apps and
platforms handle well, and the counts of a glTF document.

CONTRACT (PR 1, stream A4 implements): ``counts`` and ``check`` are placeholders.
"""

from __future__ import annotations

BUDGETS = {
    "vrm1": {"triangles": 70_000, "joints": 200, "materials": 16, "texture_size": 4096},
    "vrm0": {"triangles": 32_000, "joints": 128, "materials": 8, "texture_size": 2048},
}


class BudgetError(ValueError):
    """A file over its budget, with ``--budget strict``."""


def counts(gltf: dict) -> dict:
    """Triangles, joints, materials, images, morph targets and meshes of a glTF JSON dict."""
    # placeholder: stream A4
    return {
        "materials": len(gltf.get("materials", [])),
        "images": len(gltf.get("images", [])),
    }


def check(found: dict, target: str, mode: str = "warn") -> list[str]:
    """
    Compare counts with ``BUDGETS[target]``. Returns the messages for each count over budget;
    ``mode="strict"`` raises :class:`BudgetError` instead, and ``mode="off"`` checks nothing.
    """
    # placeholder: stream A4
    return []
