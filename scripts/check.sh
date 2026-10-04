#!/usr/bin/env bash
# OpenSculptBoy
# Apache License, Version 2.0
#
# The checks of the project, in the tiers of AGENTS.md ("Testing"). GitHub Actions runs the
# same functions (.github/workflows/lint.yml, viewer.yml and tests.yml), so a local run and CI
# stay the same.
#
#   scripts/check.sh quick [test.module ...]   while working: lint, then the named test modules
#   scripts/check.sh ci                        before a push: every job of CI, in turn
#   scripts/check.sh full                      locally only: ci, plus the local-only tests
#
# The jobs of CI, one by one: lint, viewer, tests, package. The workflows in .github/workflows
# run each job only when its files change, and the tests skip draft pull requests.
set -euo pipefail
cd "$(dirname "$0")/.."

RUFF_VERSION="$(sed -n 's/.*"ruff==\([0-9.]*\)".*/\1/p' pyproject.toml)"
EXTRAS=(--extra dev --extra examples)

step() { printf '\n== %s\n' "$*"; }

lint() {
  step "lock file matches pyproject.toml"
  uv lock --check
  step "ruff ${RUFF_VERSION}: format and lint"
  uvx "ruff@${RUFF_VERSION}" format --check --no-cache .
  uvx "ruff@${RUFF_VERSION}" check --no-cache .
  step "copyright headers"
  uv run --no-project --python 3.11 python -m unittest test.test_copyright_headers
}

viewer() {
  step "viewer: type-check and rebuild the page from the committed model data"
  (cd viewer && npm ci --no-audit --no-fund && npx tsc --noEmit && npx tsc --noEmit -p test/vrm_page && node build.mjs --reuse-data)
  step "viewer: the committed page matches viewer/src and shell.html"
  if ! git diff --quiet -- viewer/dist/anny_viewer.html; then
    echo "viewer/dist/anny_viewer.html is stale: commit the rebuilt page (cd viewer && node build.mjs --reuse-data)." >&2
    git diff --stat -- viewer/dist/anny_viewer.html >&2
    return 1
  fi
}

suite() {
  # $1: "ci" skips the local-only tests and the non-commercial data; "full" runs everything available.
  local mode="$1"
  shift
  step "test suite (${mode})"
  if [[ "${mode}" == "ci" ]]; then
    OPENSCULPTBOY_CI=1 OPENSCULPTBOY_SKIP_NONCOMMERCIAL=1 uv run "${EXTRAS[@]}" python -m unittest "$@"
  else
    uv run "${EXTRAS[@]}" python -m unittest "$@"
  fi
}

package() {
  # Build the wheel, install it into a fresh environment with CPU torch, and use it.
  local python="${1:-3.11}" work
  work="$(mktemp -d)"
  trap 'rm -rf "${work}"' RETURN
  step "package: build the wheel and install it with Python ${python}"
  uv build --wheel --out-dir "${work}/dist"
  uv venv --python "${python}" "${work}/venv"
  # --python names the new environment; UV_PYTHON (set in CI) would otherwise pick another one.
  uv pip install --python "${work}/venv/bin/python" --torch-backend cpu "${work}"/dist/*.whl
  step "package: import, version and the opensculptboy command (GLB and VRM exports)"
  (
    cd "${work}"
    ./venv/bin/python - <<'EOF'
import importlib.metadata
import anny, opensculptboy
wheel = importlib.metadata.version("opensculptboy")
assert anny.__version__ == wheel, (anny.__version__, wheel)
print("anny", anny.__version__, "| opensculptboy", wheel)
EOF
    ./venv/bin/opensculptboy names poses | head -3
    ./venv/bin/opensculptboy export smoke.glb --animation walk
    test -s smoke.glb
    ./venv/bin/opensculptboy export smoke.vrm --author smoke
    test -s smoke.vrm
    ./venv/bin/opensculptboy export smoke0.vrm --author smoke --vrm-version 0
    test -s smoke0.vrm
  )
}

mode="${1:-quick}"
shift || true
case "${mode}" in
  lint) lint ;;
  viewer) viewer ;;
  tests) suite ci discover "$@" ;;
  package) package "$@" ;;
  quick)
    lint
    if [[ $# -gt 0 ]]; then suite ci "$@"; fi
    ;;
  ci)
    lint
    viewer
    suite ci discover
    package
    ;;
  full)
    lint
    viewer
    suite full discover
    package
    ;;
  *)
    echo "usage: scripts/check.sh {quick [test.module ...]|ci|full|lint|viewer|tests|package [python]}" >&2
    exit 2
    ;;
esac
printf '\n== %s: all checks passed\n' "${mode}"
