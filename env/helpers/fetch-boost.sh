#!/bin/bash

# Download and build the Boost libraries needed by AdaptiveCpp (context & fiber) from source,
# so that no system Boost install is required. Only python3, git, cmake and a C++ compiler
# are needed (no curl / xz: the download & extraction are done in python).

function setup_boost {

    if [ -z ${BOOST_VERSION+x} ]; then echo "BOOST_VERSION is unset"; return 1; fi
    if [ -z ${BOOST_SRC_DIR+x} ]; then echo "BOOST_SRC_DIR is unset"; return 1; fi
    if [ -z ${BOOST_BUILD_DIR+x} ]; then echo "BOOST_BUILD_DIR is unset"; return 1; fi
    if [ -z ${BOOST_INSTALL_DIR+x} ]; then echo "BOOST_INSTALL_DIR is unset"; return 1; fi

    BOOST_URL="https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}-cmake.tar.xz"

    if [ ! -f "$BOOST_SRC_DIR/CMakeLists.txt" ]; then
        echo " ------ Downloading Boost ${BOOST_VERSION} ------ "
        echo "-> $BOOST_URL"
        python3 - "$BOOST_URL" "$BOOST_SRC_DIR" <<'EOF' || return
import io, os, sys, tarfile, urllib.request

url, dest = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as r:
    data = r.read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tar:
    # strip the leading "boost-x.y.z/" component
    members = []
    for m in tar.getmembers():
        parts = m.name.split("/", 1)
        if len(parts) == 2 and parts[1]:
            m.name = parts[1]
            members.append(m)
    os.makedirs(dest, exist_ok=True)
    tar.extractall(dest, members=members)
EOF
        echo " ------  Boost downloaded  ------ "
    fi

    if [ ! -f "$BOOST_INSTALL_DIR/include/boost/version.hpp" ]; then
        echo " ------ Building Boost (context, fiber) ------ "
        cmake -S "$BOOST_SRC_DIR" -B "$BOOST_BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$BOOST_INSTALL_DIR" \
            -DBOOST_INCLUDE_LIBRARIES="context;fiber" \
            -DBUILD_SHARED_LIBS=Off \
            -DCMAKE_POSITION_INDEPENDENT_CODE=On \
            -DBUILD_TESTING=Off || return
        cmake --build "$BOOST_BUILD_DIR" || return
        cmake --install "$BOOST_BUILD_DIR" || return
        echo " ------  Boost built  ------ "
    fi
}
