#!/usr/bin/env bash
# ~~~
# SHAMROCK Cloud Agent install script.
#
# Reproduces the AdaptiveCpp/OMP (CPU) build used by the phys-test CI image
# (.github/dockerfiles/phys_test_image_acpp_omp/Dockerfile) on a bare Ubuntu
# 24.04 Cloud Agent VM. It is idempotent: re-running it is safe and, when the
# base snapshot already contains the toolchain, apt and the compile steps are
# fast (ccache-backed) no-ops.
# ~~~
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CLANG_VERSION=18

# ----------------------------------------------------------------------------
# 1. System dependencies
#    The upstream CI starts from ghcr.io/shamrock-code/shamrock-ci which already
#    ships cmake, git, MPI, Boost and a C/C++ toolchain. A Cloud Agent starts
#    from a plain Ubuntu image, so we install the equivalent set here. Ubuntu
#    24.04 ships LLVM/Clang 18 natively, so apt.llvm.org is not required.
# ----------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    "llvm-${CLANG_VERSION}-dev" \
    "libclang-${CLANG_VERSION}-dev" \
    "clang-${CLANG_VERSION}" \
    "clang-tools-${CLANG_VERSION}" \
    "libomp-${CLANG_VERSION}-dev" \
    "lld-${CLANG_VERSION}" \
    libstdc++-14-dev \
    libopenmpi-dev openmpi-bin \
    libboost-dev libboost-context-dev libboost-fiber-dev \
    cmake ninja-build ccache \
    gfortran ffmpeg \
    python3-venv python3-dev python3-pip \
    git curl wget

# ----------------------------------------------------------------------------
# 2. Submodules (fmt, pybind11, mdspan, cpptrace, NVTX, nlohmann_json, ...)
# ----------------------------------------------------------------------------
git submodule update --init --recursive

# ----------------------------------------------------------------------------
# 3. AdaptiveCpp's CMake build must locate LLVM 18's CMake package. The distro
#    installs it under /usr/lib/llvm-18 which is not on the default search path.
# ----------------------------------------------------------------------------
export CMAKE_PREFIX_PATH="/usr/lib/llvm-${CLANG_VERSION}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# ----------------------------------------------------------------------------
# 4. Generate the build environment (AdaptiveCpp/OMP CPU backend). This writes
#    build/activate and build/shamenv_do. Regenerating is cheap, but we only do
#    it when the build dir has not been configured yet to preserve local edits.
# ----------------------------------------------------------------------------
if [ ! -f build/activate ]; then
    ./env/new-env --machine debian-generic.acpp --builddir build -- \
        --backend omp --gen ninja
fi

# ----------------------------------------------------------------------------
# 5. Build AdaptiveCpp (first run only) then configure + build Shamrock.
#    shamconfigure/shammake source build/activate, which compiles AdaptiveCpp
#    v25.02.0 the first time and is a no-op afterwards. ccache keeps rebuilds
#    fast. -DCXX_FLAG_ARCH_NATIVE=off is already set by the debian-generic.acpp
#    machine profile.
# ----------------------------------------------------------------------------
cd build
./shamenv_do shamconfigure
./shamenv_do shammake

# ----------------------------------------------------------------------------
# 6. Reference files consumed by the unit tests and phys-test rscripts.
# ----------------------------------------------------------------------------
test -d reference-files || ./shamenv_do pull_reffiles

# ----------------------------------------------------------------------------
# 7. Python analysis packages used by examples/ and tests_ci/ rscripts.
#    The `shamrock` binary embeds the *system* interpreter (/usr/bin/python3),
#    so `shamrock --rscript ...` resolves these from the system dist-packages
#    (matching the phys-test CI image, which pip-installs them system-wide).
#    Ubuntu 24.04 marks the base env externally managed (PEP 668), hence
#    --break-system-packages.
# ----------------------------------------------------------------------------
sudo pip3 install --break-system-packages numpy scipy matplotlib h5py

echo "SHAMROCK Cloud Agent install complete."
