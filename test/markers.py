# OpenSculptBoy
# Apache License, Version 2.0
"""
The tiers of the test suite (see "Testing" in AGENTS.md).

- ``local_only(reason)`` marks a test that needs what CI never has: Playwright with the built
  page, the viewer data build, or licensed model files. It runs locally (``scripts/check.sh
  full``) and skips when ``OPENSCULPTBOY_CI=1``, which ``scripts/check.sh ci`` and GitHub Actions set.
- With ``OPENSCULPTBOY_SKIP_NONCOMMERCIAL=1``, the non-commercial SMPL and SMPL-X data is never
  downloaded: the download raises ``unittest.SkipTest``, so the smpl and smplx cases skip
  (``install_noncommercial_guard``, called by ``test/__init__.py``).
"""

import os
import unittest

IN_CI = os.environ.get("OPENSCULPTBOY_CI") == "1"
SKIP_NONCOMMERCIAL = os.environ.get("OPENSCULPTBOY_SKIP_NONCOMMERCIAL") == "1"


def local_only(reason: str):
    """Skip a test class or method in CI (``OPENSCULPTBOY_CI=1``); ``reason`` names what CI lacks."""
    return unittest.skipIf(IN_CI, f"local only: {reason}")


def install_noncommercial_guard() -> None:
    """Make the download of the non-commercial data skip the test that asks for it."""
    if not SKIP_NONCOMMERCIAL:
        return
    import anny.paths

    def refuse():
        raise unittest.SkipTest(
            "non-commercial SMPL/SMPL-X data disabled (OPENSCULPTBOY_SKIP_NONCOMMERCIAL=1)"
        )

    anny.paths.download_noncommercial_data = refuse
