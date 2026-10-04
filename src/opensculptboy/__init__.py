# OpenSculptBoy
# Apache License, Version 2.0
"""
OpenSculptBoy: tools built on the Anny body model (``import anny``).

The ``anny`` package holds the body model and stays close to the upstream Anny project, so
that its updates merge cleanly. The ``opensculptboy`` package holds what this project adds on
top of it, starting with the character description (``Character``) and the glTF export
(``opensculptboy.export``).
"""

from opensculptboy.character import Character
from opensculptboy.export.gltf import export_glb, read_character

__all__ = ["Character", "export_glb", "read_character"]
