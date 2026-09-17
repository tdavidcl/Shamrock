# Everything before this line will be provided by the new-env script

# ---- Modules ----
# Unlike acpp-rocm (which builds AdaptiveCpp against ROCm's bundled
# clang/llvm) this env is given a standalone, newer LLVM via modules, so
# no llvm_setup / build-from-source step is needed (unlike
# lumi/g/acpp-custom-llvm, which this env is otherwise based on).
module purge
module load cpe/25.09
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-cray
module load llvm/20.1.6
module load rocm/6.4.3
module load cray-python
module load cmake
module load ninja
module load CCE-GPU-5.0.0
module load boost/1.88.0-mpi

# ---- AdaptiveCpp config ----
export ACPP_VERSION=v25.02.0
export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir

case "$ACPP_MODE" in
    "SSCP")
        export ACPP_TARGETS=generic
        ;;
    "SMCP")
        export ACPP_TARGETS=hip:gfx90a
        ;;
    *)
        echo "Unknown ACPP_MODE: $ACPP_MODE"
        return
        ;;
esac

# Resolve the LLVM install prefix from the `llvm/20.1.6` module loaded
# above instead of pointing at $ROCM_PATH/llvm or a from-source build.
export LLVM_INSTALL_DIR=$(dirname $(dirname $(command -v clang++)))
if [ ! -x "$LLVM_INSTALL_DIR/bin/clang++" ]; then
    echo "Could not locate clang++ from the llvm module, is it loaded ?"
    return
fi

export C_INCLUDE_PATH=$ROCM_PATH/llvm/include
export CPLUS_INCLUDE_PATH=$ROCM_PATH/llvm/include

export MPICH_GPU_SUPPORT_ENABLED=1

export BOOST_ROOT_PATH="${BOOST_ROOT:-/opt/software/gaia/prod/5.0.0/boost-1.88.0-cce-18.0.0-ml3z}"
export BOOST_SYMLINK_DIR=$BUILD_DIR/.env/boost-symlinks

function setupboost {
    # here I lost 2hrs of my life
    mkdir -p "${BOOST_SYMLINK_DIR}"
    for tagged in "${BOOST_ROOT_PATH}/lib"/libboost_*-mt-x64.so; do
        # e.g. libboost_context-mt-x64.so -> libboost_context.so
        base=$(basename "$tagged")
        untagged="${base/-mt-x64/}"
        if [ ! -e "${BOOST_SYMLINK_DIR}/${untagged}" ]; then
            ln -s "$tagged" "${BOOST_SYMLINK_DIR}/${untagged}"
        fi
    done
}

setupboost

export LD_LIBRARY_PATH="${BOOST_SYMLINK_DIR}:${BOOST_ROOT_PATH}/lib:${LD_LIBRARY_PATH}"

# ---- Compiler setup ----
function setupcompiler {
    echo " ---- Running AdaptiveCpp compiler setup ----"
    echo " -- Module list"
    module list

    clone_acpp || return

    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} \
        -DROCM_PATH=$ROCM_PATH \
        -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
        -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
        -DWITH_ACCELERATED_CPU=ON \
        -DWITH_CPU_BACKEND=ON \
        -DWITH_CUDA_BACKEND=OFF \
        -DWITH_ROCM_BACKEND=ON \
        -DWITH_OPENCL_BACKEND=OFF \
        -DWITH_LEVEL_ZERO_BACKEND=OFF \
        -DBOOST_ROOT="${BOOST_ROOT_PATH}" \
        -DBoost_DIR="${BOOST_ROOT_PATH}/lib/cmake" \
        -DBoost_NO_BOOST_CMAKE=FALSE \
        -DBoost_NO_SYSTEM_PATHS=TRUE \
        -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm/ || return

    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install) || return
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ----- acpp is not configured, compiling it ... -----"
    setupcompiler || return
    echo " ----- acpp configured ! -----"
fi

# ---- Shamrock configure ----
function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DCMAKE_CXX_FLAGS="-march=znver3 -isystem ${CRAY_MPICH_PREFIX}/include" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L\"${BOOST_SYMLINK_DIR}\" -Wl,-rpath,${BOOST_SYMLINK_DIR}" \
        -DCMAKE_MODULE_LINKER_FLAGS="-L\"${BOOST_SYMLINK_DIR}\" -Wl,-rpath,${BOOST_SYMLINK_DIR}" \
        -DCMAKE_EXE_LINKER_FLAGS="-lpthread -L\"${CRAY_MPICH_PREFIX}/lib\" -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} -L\"${BOOST_SYMLINK_DIR}\" -Wl,-rpath,${BOOST_SYMLINK_DIR}" \
        -DBUILD_TEST=Yes \
        -DCXX_FLAG_ARCH_NATIVE=off \
        -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") \
        "${CMAKE_OPT[@]}" || return
}

# ---- Shamrock build ----
function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
