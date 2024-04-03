# Exports will be provided by the new env script above this line
# will be exported : ACPP_GIT_DIR, ACPP_BUILD_DIR, ACPP_INSTALL_DIR

export LD_LIBRARY_PATH=$INTELLLVM_INSTALL_DIR/lib:$LD_LIBRARY_PATH

export CUDA_PATH=/usr/local/cuda-12.2
export CUDA_NVCC_EXECUTABLE=${CUDA_PATH}/bin/nvcc

function setupcompiler {

    python3 ${INTELLLVM_GIT_DIR}/buildbot/configure.py \
        --cuda \
        --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH}" \
        --cmake-gen "${CMAKE_GENERATOR}" \
        --cmake-opt="-DCMAKE_INSTALL_PREFIX=${INTELLLVM_INSTALL_DIR}" \
        --host-target AArch64

    (cd ${INTELLLVM_GIT_DIR}/build && $MAKE_EXEC "${MAKE_OPT[@]}" all libsycldevice)
    (cd ${INTELLLVM_GIT_DIR}/build && $MAKE_EXEC install)

}

function updatecompiler {
    (cd ${ACPP_GIT_DIR} && git pull)
    setupcompiler
}

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=IntelLLVM \
        -DINTEL_LLVM_PATH="${INTELLLVM_INSTALL_DIR}" \
        -DCMAKE_CXX_COMPILER="${INTELLLVM_INSTALL_DIR}/bin/clang++" \
        -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvidia_gpu_sm_90" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}