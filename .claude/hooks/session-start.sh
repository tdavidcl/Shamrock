#!/bin/bash
set -euo pipefail

# Only run this setup in Claude Code on the web / remote sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# CLAUDE_PROJECT_DIR is unset when the environment is started from a
# manually configured directory (e.g. pasted into the environment's setup
# script box) rather than the normal session bootstrap; in that case we're
# already in the right directory, so just skip the cd.
if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  cd "$CLAUDE_PROJECT_DIR"
fi

# --- System dependencies -----------------------------------------------
# AdaptiveCpp (SYCL) needs Boost.context/fiber and an LLVM install; Shamrock
# needs an MPI implementation; pre-commit is used for linting.
NEEDED_PKGS="libboost-context-dev libboost-fiber-dev llvm-18-dev libclang-18-dev libomp-18-dev libopenmpi-dev openmpi-bin pre-commit"
MISSING_PKGS=""
for pkg in $NEEDED_PKGS; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    MISSING_PKGS="$MISSING_PKGS $pkg"
  fi
done
if [ -n "$MISSING_PKGS" ]; then
  # Some base images ship extra apt sources (e.g. deadsnakes/ondrej PPAs)
  # that this environment's network policy blocks; that makes `apt-get
  # update` exit non-zero even though the archives we actually need
  # (Ubuntu main/universe/security) refresh fine. Don't let that abort us.
  apt-get update || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y $MISSING_PKGS
fi

# pre-commit's isolated venvs pick up Debian's patched sysconfig scheme,
# which expects a distutils "install_layout" attribute that setuptools'
# vendored (local) distutils no longer provides. Forcing stdlib distutils
# avoids the AttributeError when hook environments are built.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export SETUPTOOLS_USE_DISTUTILS=stdlib' >> "$CLAUDE_ENV_FILE"
fi
export SETUPTOOLS_USE_DISTUTILS=stdlib

# --- Submodules ----------------------------------------------------------
git submodule update --init --recursive

# --- Build environment -----------------------------------------------
# CPU-only container: use AdaptiveCpp's OpenMP backend (no GPU present).
if [ ! -f build/shamenv_do ]; then
  ./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp
fi

cd build

# shamconfigure builds AdaptiveCpp from source on first run (a few minutes,
# cached after that) and then configures Shamrock with CMake. It does NOT
# compile Shamrock itself: a full `shammake` of every target takes ~40min,
# so it is intentionally left for later, on demand. Prefer building only
# the target(s) touched by a change, e.g.:
#   ./shamenv_do shammake shammodels_sph
# and reserve a full `./shamenv_do shammake` (or `shammake shamrock_test`)
# for when tests actually need to run or the binary needs to execute.
./shamenv_do shamconfigure

# --- Reference files (needed for unit tests) ----------------------------
if [ ! -d reference-files ]; then
  ./shamenv_do pull_reffiles
fi
