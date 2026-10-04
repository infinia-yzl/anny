# OpenSculptBoy
# Apache License, Version 2.0
"""
Export of Anny characters to standard 3D formats: glTF 2.0 binary files (``export_glb``) and
VRM avatars for VTuber apps (``export_vrm``). ``read_character`` reads the character back from
either file.
"""

from opensculptboy.export.gltf import export_glb, read_character

__all__ = ["export_glb", "export_vrm", "read_character"]


def __getattr__(name):
    # the VRM exporter loads on first use
    if name == "export_vrm":
        from opensculptboy.export.vrm import export_vrm

        return export_vrm
    raise AttributeError(f"module 'opensculptboy.export' has no attribute {name!r}")
