# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Build the viewer page of anny.

    python -m anny.viewer build            # data (viewer/build) and page (viewer/dist)
    python -m anny.viewer build --data     # the data only
"""

import argparse
import pathlib
import subprocess
import sys

from .build import DEFAULT_OUT, REPO


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m anny.viewer")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the viewer data and page")
    b.add_argument(
        "--out", type=pathlib.Path, default=DEFAULT_OUT, help="folder of the data"
    )
    b.add_argument("--data", action="store_true", help="build the data only")
    args = parser.parse_args(argv)
    if args.command == "build":
        from .build import build

        build(args.out)
        if not args.data:
            web = REPO / "viewer"
            if not (web / "node_modules").exists():
                subprocess.run(["npm", "install"], cwd=web, check=True)
            subprocess.run(["npm", "run", "build"], cwd=web, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
