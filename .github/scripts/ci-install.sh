#!/usr/bin/env bash
# Prepare/activate shared Python runtime for anny CI jobs on Slurm nodes.
# This script is sourced by /opt/ci-runner/submit-slurm.sh when --install-python is passed.
#
# Behavior:
#   - Downloads uv static binary to shared BeeGFS location if not present.
#   - Uses a marker file keyed on dependency file hashes to skip expensive
#     rebuilds when nothing has changed.
#   - On cache miss or marker mismatch, rebuilds the shared project venv at
#     ANNY_ENV_PATH using uv sync from uv.lock. uv manages Python too
#     (no micromamba needed).
#   - Activates the venv by prepending its bin/ to PATH.
#
# Layout on BeeGFS (all under dirname(ANNY_ENV_PATH)):
#   bin/uv                 uv static binary
#   python/                uv-managed Python installs (UV_PYTHON_INSTALL_DIR)
#   anny/                  project venv (UV_PROJECT_ENVIRONMENT = ANNY_ENV_PATH)
#   .anny-install-marker   cache marker
set -euo pipefail

: "${ANNY_ENV_PATH:?ANNY_ENV_PATH is required - set it in runner .env and forward via --env ANNY_ENV_PATH}"

echo "[ci-install] ANNY_ENV_PATH=${ANNY_ENV_PATH}"

_ENVS_ROOT="$(dirname "${ANNY_ENV_PATH}")"
_ENVS_BIN="${_ENVS_ROOT}/bin"
_UV_BIN="${_ENVS_BIN}/uv"
_INSTALL_MARKER="${_ENVS_ROOT}/.anny-install-marker"
_DESIRED_PYTHON="3.11"

# Python installs and the project venv both live on BeeGFS so they are
# visible inside the bwrap sandbox and persist across runs.
export UV_PYTHON_INSTALL_DIR="${_ENVS_ROOT}/python"
export UV_PROJECT_ENVIRONMENT="${ANNY_ENV_PATH}"

echo "[ci-install] UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}"
echo "[ci-install] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"

# Download uv static binary to shared BeeGFS location if not present.
if [[ ! -x "${_UV_BIN}" ]]; then
  echo "[ci-install] uv not found at ${_UV_BIN}; downloading static binary"
  mkdir -p "${_ENVS_BIN}"
  case "$(uname -m)" in
    x86_64)  _UV_ARCH="x86_64-unknown-linux-musl" ;;
    aarch64) _UV_ARCH="aarch64-unknown-linux-musl" ;;
    *)       _UV_ARCH="$(uname -m)-unknown-linux-musl" ;;
  esac
  _UV_TMP="$(mktemp -d)"
  curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-${_UV_ARCH}.tar.gz" \
    | tar -xz -C "${_UV_TMP}"
  mv "${_UV_TMP}/uv-${_UV_ARCH}/uv" "${_UV_BIN}"
  chmod +x "${_UV_BIN}"
  rm -rf "${_UV_TMP}"
  echo "[ci-install] uv installed at ${_UV_BIN}: $(${_UV_BIN} --version)"
fi

# Ensure uv is on PATH for this job.
export PATH="${_ENVS_BIN}:${PATH}"

# ------------------------------------------------------------------
# Marker-based cache: skip the full uv sync when nothing has changed.
#
# The marker is keyed on the SHA256 of the three files that control
# what ends up in the venv: ci-install.sh (bootstrap logic), pyproject.toml
# (extras / metadata), and uv.lock (pinned dependency graph). The repo HEAD
# commit is intentionally excluded -- unrelated changes to tests, docs, or
# workflow files must not trigger an expensive rebuild.
#
# On a cache hit the shared venv at ANNY_ENV_PATH already has all packages
# installed; ci-install.sh just activates it. On a miss the venv is removed
# and recreated from scratch via uv sync --locked.
# ------------------------------------------------------------------
_BOOTSTRAP_HASH="$(
  {
    sha256sum ".github/scripts/ci-install.sh" "pyproject.toml" "uv.lock" 2>/dev/null || true
  } | sha256sum | awk '{print $1}'
)"
_MARKER_CONTENT="bootstrap_hash=${_BOOTSTRAP_HASH};python=${_DESIRED_PYTHON}"

# Fast-path: if marker matches, skip rebuild entirely.
if [[ -f "${_INSTALL_MARKER}" && "$(cat "${_INSTALL_MARKER}" 2>/dev/null)" == "${_MARKER_CONTENT}" ]]; then
  echo "[ci-install] cache hit (py=${_DESIRED_PYTHON}); skipping install"
else
  # Log the reason so it is visible in Slurm output.
  if [[ ! -d "${ANNY_ENV_PATH}" ]]; then
    echo "[ci-install] REASON: no existing env -- fresh install"
  elif [[ ! -f "${_INSTALL_MARKER}" ]]; then
    echo "[ci-install] REASON: marker missing -- recreating env"
  else
    _EXISTING_MARKER="$(cat "${_INSTALL_MARKER}" 2>/dev/null || echo "(unreadable)")"
    echo "[ci-install] REASON: dependencies changed -- recreating env"
    echo "[ci-install]   existing marker: ${_EXISTING_MARKER}"
    echo "[ci-install]   new     marker: ${_MARKER_CONTENT}"
  fi

  # Remove stale venv so uv sync starts from a clean state.
  if [[ -d "${ANNY_ENV_PATH}" ]]; then
    echo "[ci-install] removing stale env at ${ANNY_ENV_PATH}"
    rm -rf "${ANNY_ENV_PATH}"
  fi

  # uv sync resolves UV_PROJECT_ENVIRONMENT (ANNY_ENV_PATH) and
  # UV_PYTHON_INSTALL_DIR, so the venv and its Python interpreter both
  # land on BeeGFS and persist for the next run.
  echo "[ci-install] running uv sync (py=${_DESIRED_PYTHON})"
  uv sync --link-mode=copy --all-extras --python "${_DESIRED_PYTHON}" --locked

  echo "${_MARKER_CONTENT}" > "${_INSTALL_MARKER}"
  echo "[ci-install] marker written to ${_INSTALL_MARKER}"
fi

# Activate the shared venv so python / uv resolve correctly for the
# suite command that follows this sourced script.
export PATH="${ANNY_ENV_PATH}/bin:${PATH}"

echo "[ci-install] python: $(python -V 2>&1)"
echo "[ci-install] uv: $(uv --version 2>&1)"


# Anny cache dir. Kept next to the shared env so it survives across jobs.
# The workflow forwards this via --env; the derivation below is the fallback
# for jobs (or manual runs) that do not set it.
: "${ANNY_CACHE_DIR:=$(realpath -m "${ANNY_ENV_PATH}/../../anny_cache")}"
export ANNY_CACHE_DIR
mkdir -p "${ANNY_CACHE_DIR}"
echo "[ci-install] ANNY_CACHE_DIR=${ANNY_CACHE_DIR}"
