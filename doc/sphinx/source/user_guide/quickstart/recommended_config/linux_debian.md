# Linux (Debian/Ubuntu) Configuration

## Recommended Setup

**From source**: Using AdaptiveCpp OpenMP backend

```bash
# Clone the repo
git clone --recurse-submodules git@github.com:Shamrock-code/Shamrock.git
# cd into it
cd Shamrock

# Required packages
wget --progress=bar:force https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18
sudo apt install -y libclang-18-dev clang-tools-18 libomp-18-dev
sudo apt install cmake libboost-all-dev python3-ipython

# Select the env to build from source
./env/new-env --builddir build --machine debian-generic.acpp -- --backend omp

# Now move in the build directory
cd build
# Activate the workspace, which will define some utility functions
source ./activate
# Configure Shamrock
shamconfigure
# Build Shamrock
shammake
```

## Pip only (self-contained)

Shamrock can also be built and installed with nothing but `pip`, without any system LLVM, Boost,
MPI or CMake. Only a C++ compiler, `git` and a Python with its development headers are needed:

```bash
# Clone the repo
git clone --recurse-submodules git@github.com:Shamrock-code/Shamrock.git

# Install in a virtual env (cmake, ninja & Open MPI are pulled from PyPI)
python3 -m venv .venv
.venv/bin/pip install ./Shamrock/env/machine/debian-generic/acpp
```

This builds Boost from source and AdaptiveCpp without its LLVM based compiler, so only the
`omp.library-only` backend (CPU, OpenMP) is available, which is slower than the `omp` backend of
the setup above. The AdaptiveCpp runtime is bundled in the wheel and MPI is provided by the
`openmpi` package from PyPI (`.venv/bin/mpirun`).

The build takes a while (AdaptiveCpp + Shamrock from scratch). To keep the build directory and
rebuild incrementally, provide it explicitly and disable pip's build isolation:

```bash
.venv/bin/pip install cmake ninja "openmpi>=5,<6" patchelf
.venv/bin/pip install --no-build-isolation ./Shamrock/env/machine/debian-generic/acpp \
    -C builddir=$PWD/build-pip
```

The python package (`import shamrock`) and the `shamrock` executable are both installed.
