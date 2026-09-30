# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

from os.path import dirname, basename, isfile, join
import glob

from test.markers import install_noncommercial_guard

install_noncommercial_guard()

modules = glob.glob(join(dirname(__file__), "test_*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
