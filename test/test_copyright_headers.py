# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Every ``.py`` file in the repository, and every ``.ts`` and ``.mjs`` file of the viewer, must carry
the project copyright header (as ``//`` comments in the viewer's files).
"""

import os
import pathlib
import unittest

HEADER_LINES = [
    "# Anny",
    "# Copyright (C) 2025 NAVER Corp.",
    "# Apache License, Version 2.0",
]
# Files that the OpenSculptBoy fork adds carry its own header, since NAVER did not write them.
OPENSCULPTBOY_HEADER_LINES = [
    "# OpenSculptBoy",
    "# Apache License, Version 2.0",
]

# Repository root (this file lives in <root>/test/).
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Directories that never need the header: virtualenvs, VCS, build artifacts,
# packaging metadata, auto-generated Jupyter checkpoints, and the viewer's node
# packages and built page.
EXCLUDED_DIRS = {
    ".venv",
    ".git",
    ".ipynb_checkpoints",
    "build",
    "docs",
    "node_modules",
    "dist",
}
# The viewer's modules, whose header lines start with // in place of #.
VIEWER = ROOT / "viewer"
VIEWER_SUFFIXES = (".ts", ".mjs")


def _is_excluded(rel_path: pathlib.Path) -> bool:
    if set(rel_path.parts) & EXCLUDED_DIRS:
        return True
    return any(part.endswith(".egg-info") for part in rel_path.parts)


def _files(top: pathlib.Path, suffixes: tuple[str, ...]) -> list[pathlib.Path]:
    """The files under ``top`` with one of ``suffixes``, outside the excluded directories."""
    found = []
    for folder, dirs, names in os.walk(top):
        dirs[:] = sorted(
            d for d in dirs if d not in EXCLUDED_DIRS and not d.endswith(".egg-info")
        )
        found += [pathlib.Path(folder) / n for n in names if n.endswith(suffixes)]
    return sorted(found)


def _has_header(text: str, comment: str = "#") -> bool:
    """True if the three header lines appear consecutively within the file's
    leading run of blank/comment lines. Searching that run (and not only
    lines 1-3) lets the header follow a jupytext ``# ---`` frontmatter block in
    the tutorials, while still requiring it above any real code. ``comment`` is
    the comment marker of the language: ``#`` or ``//``."""
    header = [comment + line[1:] for line in HEADER_LINES]
    fork_header = [comment + line[1:] for line in OPENSCULPTBOY_HEADER_LINES]
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(comment):
            break
        if lines[i : i + 3] == header:
            return True
        if lines[i : i + 2] == fork_header:
            return True
    return False


class TestCopyrightHeaders(unittest.TestCase):
    def test_all_python_files_have_copyright_header(self):
        offenders = []
        for path in _files(ROOT, (".py",)):
            rel = path.relative_to(ROOT)
            if _is_excluded(rel):
                continue
            if not _has_header(path.read_text(encoding="utf-8")):
                offenders.append(str(rel))

        self.assertEqual(
            offenders,
            [],
            "The following .py files are missing the copyright header "
            f"(expected the block: {HEADER_LINES}):\n  " + "\n  ".join(offenders),
        )

    def test_all_viewer_modules_have_copyright_header(self):
        paths = _files(VIEWER, VIEWER_SUFFIXES)
        self.assertGreater(len(paths), 0)
        offenders = [
            str(path.relative_to(ROOT))
            for path in paths
            if not _has_header(path.read_text(encoding="utf-8"), comment="//")
        ]
        self.assertEqual(
            offenders,
            [],
            "The following viewer files are missing the copyright header "
            "(expected the block as // comments):\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
