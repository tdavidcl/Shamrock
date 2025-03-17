## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

################################
# Cmake config
################################
cmake_minimum_required(VERSION 3.9...3.10)

enable_testing()

# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    )

option(EXPORT_COMPILE_COMMANDS "export the compile_commands.json" ON)

# Necessary for clangd to work properly
if(EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

include(AreSubmoduleClones)

################################
# Shamrock versioning
################################

set(SHAMROCK_VERSION_MAJOR 2025)
set(SHAMROCK_VERSION_MINOR 03)
set(SHAMROCK_VERSION_PATCH 0)

project(Shamrock VERSION ${SHAMROCK_VERSION_MAJOR}.${SHAMROCK_VERSION_MINOR}.${SHAMROCK_VERSION_PATCH})
include(ShamVersioning)

################################
# Shamrock Configuration
################################

option(SHAMROCK_USE_PCH "use precompiled headers" Off)

option(SHAMROCK_USE_SHARED_LIB "use shared libraries" On)

if(APPLE)
    if(SHAMROCK_USE_SHARED_LIB)
        message(STATUS "forcing SHAMROCK_USE_SHARED_LIB=Off on Macos because of known issues")
    endif()
    set(SHAMROCK_USE_SHARED_LIB Off)
endif()

######################
#checking compiler flag support
######################

include(ShamConfigureCXX)

include(ShamBuildAsan)
include(ShamBuildUBsan)
include(ShamBuildCoverage)

set(ValidShamBuildType "Debug" "Release" "ASAN" "UBSAN" "COVERAGE")
if(NOT CMAKE_BUILD_TYPE)
    #set default build to release
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Cmake build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${ValidShamBuildType})
endif()
if(NOT "${CMAKE_BUILD_TYPE}" IN_LIST ValidShamBuildType)
    message(FATAL_ERROR
        "The required build type in unknown -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
        "please use a build type in the following list (case-sensitive) "
        "${ValidShamBuildType}")
endif()
message(STATUS "current build type : CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

######################
# Compiler selection
######################

if("${SHAMROCK_ENABLE_BACKEND}" STREQUAL "SYCL")
    include(ShamConfigureSYCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHAM_CXX_SYCL_FLAGS}")
else()
    message(FATAL_ERROR
        "You must select a Shamrock Backend "
        "-DSHAMROCK_ENABLE_BACKEND=[SYCL]")
endif()
set(SHAMROCK_ENABLE_BACKEND "${SHAMROCK_ENABLE_BACKEND}" CACHE STRING "Shamrock backend used")

enable_language(C)
enable_language(CXX)

######################
# add libs
######################

include(ShamConfigureMPI)
include(ShamConfigurePybind11)
include(ShamConfigureNVTX)
include(ShamConfigureFmtlib)
include(ShamConfigureNlohmannJson)
include(ShamConfigurePlfNanotimer)

######################
# Force -fPIC for object lib mode as the python lib require it
######################
if(NOT SHAMROCK_USE_SHARED_LIB)
    message(WARNING "using shamrock in object lib mode force the use of -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

######################
#shamrock impl control
######################

option(SHAMROCK_USE_PROFILING "use custom profiling tooling" On)
if(SHAMROCK_USE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_USE_PROFILING")
endif()

message("   ---- SUMARRY ----")

message("-- Shamrock config : ")
message("     SHAMROCK_USE_PROFILING : ${SHAMROCK_USE_PROFILING}")
message("     SHAMROCK_USE_NVTX : ${SHAMROCK_USE_NVTX}")
message("     SHAMROCK_USE_PCH : ${SHAMROCK_USE_PCH}")
message("     SHAMROCK_USE_SHARED_LIB : ${SHAMROCK_USE_SHARED_LIB}")
message("     CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message("     CMAKE_CXX_COMPILER : ${CMAKE_CXX_COMPILER}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

########################################################################################
# Targets ##############################################################################
########################################################################################

message("   ---- targets section ----")

add_subdirectory(src)
