# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

######################
# precompiled headers
######################

# <sycl/sycl.hpp> is by far the most expensive header of the build (~7s per translation unit with
# AdaptiveCpp), precompiling it cuts the build time by ~30%. With AdaptiveCpp (omp & generic
# targets) the resulting objects were checked to have the same host code and the same embedded
# device IR as without the precompiled header. It is only enabled by default for AdaptiveCpp in
# direct mode, and only if a small test project using the precompiled header builds with the
# current compiler & flags (e.g. clang refuses to use a host PCH in a CUDA/HIP device compilation).
if(("${SYCL_IMPLEMENTATION}" STREQUAL "ACPPDirect") AND (NOT CMAKE_VERSION VERSION_LESS 3.16))
    set(SHAMROCK_USE_PCH_DEFAULT On)
else()
    set(SHAMROCK_USE_PCH_DEFAULT Off)
endif()

option(SHAMROCK_USE_PCH "precompile the SYCL header" ${SHAMROCK_USE_PCH_DEFAULT})

if(SHAMROCK_USE_PCH AND CMAKE_VERSION VERSION_LESS 3.16)
    message(WARNING "SHAMROCK_USE_PCH requires CMake >= 3.16, disabling it")
    set(SHAMROCK_USE_PCH Off)
endif()

if(SHAMROCK_USE_PCH AND (NOT DEFINED SHAMROCK_PCH_SYCL_WORKS))
    message(STATUS "Performing Test SHAMROCK_PCH_SYCL_WORKS")
    try_compile(
        SHAMROCK_PCH_SYCL_WORKS ${CMAKE_BINARY_DIR}/compile_tests/pch_sycl
        ${CMAKE_SOURCE_DIR}/cmake/feature_test/pch_sycl shamrock_pch_sycl_test
        CMAKE_FLAGS
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
            "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            "-DCMAKE_CXX_FLAGS_${_SHAMROCK_BUILD_TYPE_UC}=${CMAKE_CXX_FLAGS_${_SHAMROCK_BUILD_TYPE_UC}}"
        OUTPUT_VARIABLE SHAMROCK_PCH_SYCL_TEST_OUTPUT
    )
    set(SHAMROCK_PCH_SYCL_WORKS ${SHAMROCK_PCH_SYCL_WORKS} CACHE INTERNAL "" FORCE)
    if(SHAMROCK_PCH_SYCL_WORKS)
        message(STATUS "Performing Test SHAMROCK_PCH_SYCL_WORKS - Success")
    else()
        message(STATUS "Performing Test SHAMROCK_PCH_SYCL_WORKS - Failed")
    endif()
endif()

if(SHAMROCK_USE_PCH AND (NOT SHAMROCK_PCH_SYCL_WORKS))
    message(WARNING "SHAMROCK_USE_PCH is enabled but a precompiled <sycl/sycl.hpp> "
                    "does not work with the current configuration, disabling it"
    )
    set(SHAMROCK_USE_PCH Off)
endif()

######################
# Shared/Object libs
######################

option(SHAMROCK_USE_SHARED_LIB "use shared libraries" On)

if(DEFINED SHAMROCK_FORCE_SHARED_LIB)
    set(SHAMROCK_USE_SHARED_LIB ${SHAMROCK_FORCE_SHARED_LIB})
    message(WARNING "SHAMROCK_USE_SHARED_LIB was forced to ${SHAMROCK_USE_SHARED_LIB}")
endif()

# Force -fPIC for object lib mode as the python lib require it
if(NOT SHAMROCK_USE_SHARED_LIB)
    message(WARNING "using shamrock in object lib mode force the use of -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

######################
# profiling control
######################

option(SHAMROCK_USE_PROFILING "use custom profiling tooling" On)
if(SHAMROCK_USE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_USE_PROFILING")
endif()

######################
# Summary
######################

message("   ---- SUMARRY ----")

message(STATUS "CMAKE_C_COMPILER        : ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER      : ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS         : ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS  : ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "SHAMROCK_USE_PROFILING  : ${SHAMROCK_USE_PROFILING}")
message(STATUS "SHAMROCK_USE_NVTX       : ${SHAMROCK_USE_NVTX}")
message(STATUS "SHAMROCK_USE_PCH        : ${SHAMROCK_USE_PCH}")
message(STATUS "SHAMROCK_USE_SHARED_LIB : ${SHAMROCK_USE_SHARED_LIB}")
message(STATUS "CMAKE_BUILD_TYPE        : ${CMAKE_BUILD_TYPE}")
message(STATUS "BUILD_TEST              : ${BUILD_TEST}")
message(STATUS "PYTHON_EXECUTABLE       : ${PYTHON_EXECUTABLE}")
