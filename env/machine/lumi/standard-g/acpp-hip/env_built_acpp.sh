module purge

module load LUMI/24.03 
module load partition/G
module load cray-python
module load rocm/6.0.3
module load Boost/1.83.0-cpeAMD-24.03

export PATH=$HOME/.local/bin:$PATH
pip3 install -U ninja cmake

export ACPP_TARGETS="hip:gfx90a"

export C_INCLUDE_PATH=$ROCM_PATH/llvm/include
export CPLUS_INCLUDE_PATH=$ROCM_PATH/llvm/include

function setupcompiler {
    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} \
        -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} \
        -DROCM_PATH=$ROCM_PATH \
        -DCMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
        -DWITH_ACCELERATED_CPU=ON \
        -DWITH_CPU_BACKEND=ON \
        -DWITH_CUDA_BACKEND=OFF \
        -DWITH_ROCM_BACKEND=ON \
        -DWITH_OPENCL_BACKEND=OFF \
        -DWITH_LEVEL_ZERO_BACKEND=OFF \
        -DACPP_TARGETS="gfx90a" \
        -DBoost_NO_BOOST_CMAKE=TRUE \
        -DBoost_NO_SYSTEM_PATHS=TRUE \
        -DWITH_SSCP_COMPILER=OFF \
        -DLLVM_DIR=${ROCM_PATH}/llvm/lib/cmake/llvm/

    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install)
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ----- acpp is not configured, compiling it ... -----"
    setupcompiler
    echo " ----- acpp configured ! -----"
fi


function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DCMAKE_CXX_FLAGS="-isystem ${CRAY_MPICH_PREFIX}/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-L"${CRAY_MPICH_PREFIX}/lib" -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
        -DBUILD_TEST=Yes \
        -DCXX_FLAG_ARCH_NATIVE=off \
        "${CMAKE_OPT[@]}"
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}")
}
