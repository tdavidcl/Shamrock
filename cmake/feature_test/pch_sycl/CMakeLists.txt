# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

# Test project checking that a target using <sycl/sycl.hpp> as a precompiled header builds with the
# current compiler and flags (see SHAMROCK_USE_PCH in cmake/ShamrockBuildOptions.cmake)

cmake_minimum_required(VERSION 3.16)

project(shamrock_pch_sycl_test CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(pch_sycl_test main.cpp)
target_precompile_headers(pch_sycl_test PRIVATE <sycl/sycl.hpp>)
