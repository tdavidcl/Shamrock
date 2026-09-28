# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

if("${BUILD_TEST}")

    # Whenever this glob's value changes, cmake will rerun and update the build with the
    # new/removed files.
    file(GLOB_RECURSE test_list CONFIGURE_DEPENDS "*.cpp")

    # The tests are compiled in an object library of this directory, linked into shamrock_test
    # (see ../CMakeLists.txt), rather than in shamrock_test itself, so that they come last in the
    # generated build files (this directory is the last one added). The build tools that start the
    # ready jobs in the order of the build files (e.g. ninja 1.11) then compile the libraries
    # first, and the many short test files fill the end of a full build instead of a few long
    # library files that leave most of the cores idle.
    add_library(shamrock_test_objs OBJECT ${test_list})

    target_link_libraries(shamrock_test_objs PUBLIC pybind11::embed)
    target_link_libraries(shamrock_test_objs PUBLIC nlohmann_json::nlohmann_json)

    target_link_libraries(shamrock_test_objs PUBLIC ${as_subproject})

    target_include_directories(shamrock_test_objs PUBLIC "${CMAKE_SOURCE_DIR}/src")

endif()
