# Everything before this line will be provided by the new-env script

export ACPP_VERSION=v24.10.0
export ACPP_APPDB_DIR=/tmp/acpp-appdb # otherwise it would we in the $HOME/.acpp
export ACPP_GIT_DIR=$BUILD_DIR/.env/acpp-git
export ACPP_BUILD_DIR=$BUILD_DIR/.env/acpp-builddir
export ACPP_INSTALL_DIR=$BUILD_DIR/.env/acpp-installdir

function setupcompiler {
    clone_acpp || return
    cmake -S ${ACPP_GIT_DIR} -B ${ACPP_BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${ACPP_INSTALL_DIR} || return
    (cd ${ACPP_BUILD_DIR} && $MAKE_EXEC "${MAKE_OPT[@]}" && $MAKE_EXEC install) || return
}

if [ ! -f "$ACPP_INSTALL_DIR/bin/acpp" ]; then
    echo " ----- acpp is not configured, compiling it ... -----"
    setupcompiler || return
    echo " ----- acpp configured ! -----"
fi

function shamconfigure {
    cmake \
        -S $SHAMROCK_DIR \
        -B $BUILD_DIR \
        -DSHAMROCK_ENABLE_BACKEND=SYCL \
        -DSYCL_IMPLEMENTATION=ACPPDirect \
        -DCMAKE_CXX_COMPILER="${ACPP_INSTALL_DIR}/bin/acpp" \
        -DCMAKE_CXX_FLAGS="${SHAMROCK_CXX_FLAGS}" \
        -DACPP_PATH="${ACPP_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE="${SHAMROCK_BUILD_TYPE}" \
        -DBUILD_TEST=Yes \
        "${CMAKE_OPT[@]}" || return
}

function shammake {
    (cd $BUILD_DIR && $MAKE_EXEC "${MAKE_OPT[@]}" "${@}") || return
}
