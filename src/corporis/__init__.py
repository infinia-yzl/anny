# Corporis
# Apache License, Version 2.0
"""
Corporis: tools built on the Anny body model (``import anny``).

The ``anny`` package holds the body model and stays close to the upstream Anny project, so
that its updates merge cleanly. The ``corporis`` package holds what this project adds on
top of it: the character description (``Character``), the glTF export (``corporis.export``),
the pose from a picture (``pose_from_image``, ``corporis.posing``) and the flat drawings of a
posed character (``corporis.render``).
"""

from corporis.character import Character
from corporis.export.gltf import export_glb, read_character

__all__ = ["Character", "export_glb", "pose_from_image", "read_character"]


def __getattr__(name):
    # the posing modules load on first use
    if name == "pose_from_image":
        from corporis.posing.picture import pose_from_image

        return pose_from_image
    raise AttributeError(f"module 'corporis' has no attribute {name!r}")
