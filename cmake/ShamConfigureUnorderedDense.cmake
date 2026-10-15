# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

message("   ---- unordered_dense section ----")

###############################################################################
### unordered_dense
###############################################################################

option(SHAMROCK_EXTERNAL_UNORDERED_DENSE "use unordered_dense lib from the host system" Off)

message(STATUS "SHAMROCK_EXTERNAL_UNORDERED_DENSE : ${SHAMROCK_EXTERNAL_UNORDERED_DENSE}")

if(SHAMROCK_EXTERNAL_UNORDERED_DENSE)
    find_package(unordered_dense 5.0 REQUIRED)
else()
    _check_git_submodule_cloned(
        ${CMAKE_CURRENT_SOURCE_DIR}/external/unordered_dense
        ddd49116e63be7345b448351cad153653a712fdb
    )

    add_subdirectory(external/unordered_dense)
endif()
