# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# ~~~

cmake_minimum_required(VERSION 3.12)

project(pybind11_extra VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(PYBIND11_EXTRA_FIND_PYBIND11 "Should i locate pybind11 ?" On)

if(PYBIND11_EXTRA_FIND_PYBIND11)
    find_package(pybind11 2.10 REQUIRED)
endif()

add_library(pybind11_extra
    src/pybind11_extra.cpp
)

add_library(pybind11_extra::pybind11_extra ALIAS pybind11_extra)

target_include_directories(
    pybind11_extra
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(pybind11_extra PUBLIC pybind11::module)

target_compile_features(pybind11_extra PUBLIC cxx_std_17)


