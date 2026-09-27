# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

message("   ---- SYCL config section ----")

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# The LSPs have no clue of what a SYCL header is,
# but since it is standard C++ simply adding them to the path works.
option(SHAMROCK_ADD_SYCL_INCLUDES "Add SYCL includes to CXX_FLAGS to make LSP happy" On)

message(STATUS "Shamrock configure SYCL backend")
###############################################################################
### Implementation choice
###############################################################################

# check that the wanted sycl backend is in the list
set(KNOWN_SYCL_IMPLEMENTATIONS "IntelLLVM" "ACPPDirect" "ACPPCmake")
if((NOT ${SYCL_IMPLEMENTATION} IN_LIST KNOWN_SYCL_IMPLEMENTATIONS) OR (NOT (DEFINED
                                                                            SYCL_IMPLEMENTATION))
)
    message(FATAL_ERROR "The Shamrock SYCL backend requires specifying a SYCL implementation with "
                        "-DSYCL_IMPLEMENTATION=[IntelLLVM;ACPPDirect,ACPPCmake]"
    )
endif()
set(SYCL_IMPLEMENTATION "${SYCL_IMPLEMENTATION}" CACHE STRING "Sycl implementation used")
set_property(CACHE SYCL_IMPLEMENTATION PROPERTY STRINGS ${KNOWN_SYCL_IMPLEMENTATIONS})

message(STATUS "Chosen SYCL implementation : ${SYCL_IMPLEMENTATION}")

set(SHAM_CXX_SYCL_FLAGS "")

# use the correct script depending on the implementation
if(${SYCL_IMPLEMENTATION} STREQUAL "IntelLLVM")
    include(SYCLAdaptIntelLLVM)
elseif(${SYCL_IMPLEMENTATION} STREQUAL "ACPPDirect")
    include(SYCLAdaptACppDirect)
elseif(${SYCL_IMPLEMENTATION} STREQUAL "ACPPCmake")
    include(SYCLAdaptACppCmake)
endif()

set(SHAMROCK_LOOP_DEFAULT "PARALLEL_FOR_ROUND" CACHE STRING "Default loop mode in shamrock")
set_property(CACHE SHAMROCK_LOOP_DEFAULT PROPERTY STRINGS PARALLEL_FOR PARALLEL_FOR_ROUND ND_RANGE)

set(SHAMROCK_LOOP_GSIZE 256 CACHE STRING "Default group size in shamrock")

######################
# Make CXX flags related to sycl
######################

message(" ---- Shamrock SYCL backend config ---- ")

message("  SYCL_COMPILER : ${SYCL_COMPILER}")

message("  sycl 2020 reduction : ${SYCL2020_FEATURE_REDUCTION}")
if(SYCL2020_FEATURE_REDUCTION)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL2020_FEATURE_REDUCTION")
endif()

message("  sycl 2020 group reduction : ${SYCL2020_FEATURE_GROUP_REDUCTION}")
if(SYCL2020_FEATURE_GROUP_REDUCTION)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL2020_FEATURE_GROUP_REDUCTION")
endif()

message("  sycl 2020 isinf : ${SYCL2020_FEATURE_ISINF}")
if(SYCL2020_FEATURE_ISINF)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL2020_FEATURE_ISINF")
endif()

message("  sycl 2020 clz : ${SYCL2020_FEATURE_CLZ}")
if(SYCL2020_FEATURE_CLZ)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL2020_FEATURE_CLZ")
endif()

message("  SHAMROCK_LOOP_DEFAULT : ${SHAMROCK_LOOP_DEFAULT}")
if(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARALLEL_FOR")
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_PARALLEL_FOR")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARALLEL_FOR_ROUND")
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_PARALLEL_FOR_ROUND")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "ND_RANGE")
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_ND_RANGE")
endif()

message("  SHAMROCK_LOOP_GSIZE : ${SHAMROCK_LOOP_GSIZE}")
set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_GSIZE=${SHAMROCK_LOOP_GSIZE}")

# AdaptiveCpp compiles every translation unit with -fopenmp (OpenMP host backend), which enables
# LLVM's OpenMPOptCGSCCPass. Its cost grows with the number of call graph SCCs times the size of
# the module, so it dominated the compile time of the largest translation units (~60% of
# sph/pySPHModel.cpp, ~50% of sph/Solver.cpp) while leaving the generated code unchanged (the
# objects are byte-identical, except for a few instructions in some omp nd_range launchers), as
# Shamrock itself has no OpenMP constructs for it to optimize.
# This only disables that LLVM pass, it does not disable OpenMP.
option(SHAMROCK_ACPP_DISABLE_OPENMP_OPT
       "Disable LLVM's OpenMPOpt pass when compiling with AdaptiveCpp" On
)
message("  SHAMROCK_ACPP_DISABLE_OPENMP_OPT : ${SHAMROCK_ACPP_DISABLE_OPENMP_OPT}")
if(SHAMROCK_ACPP_DISABLE_OPENMP_OPT AND ("${SYCL_COMPILER}" MATCHES "^ACPP"))
    set(CMAKE_REQUIRED_FLAGS "-mllvm -openmp-opt-disable")
    check_cxx_source_compiles("int main(){return 0;}" COMPILER_SUPPORT_OPENMP_OPT_DISABLE)
    unset(CMAKE_REQUIRED_FLAGS)
    if(COMPILER_SUPPORT_OPENMP_OPT_DISABLE)
        # compile only flag (added to CMAKE_CXX_FLAGS it would also be passed to the link step,
        # where clang warns about it being unused)
        add_compile_options("SHELL:-mllvm -openmp-opt-disable")
    endif()
endif()

message(" -------------------------------------- ")

message(STATUS "Shamrock configure SYCL backend - done")
