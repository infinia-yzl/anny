# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import importlib.metadata


def distribution_version() -> str:
    """
    The version of the installed distribution: "opensculptboy" in the OpenSculptBoy fork, which installs
    this package under that name, or "anny" upstream; "0.0.0+unknown" for a source directory
    imported without installing it.
    """
    for name in ("opensculptboy", "anny"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "0.0.0+unknown"
