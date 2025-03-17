## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

###############################################################################
### test git submodules
###############################################################################

function(_git_submod_is_empty directory)
    file(GLOB files RELATIVE "${directory}" "${directory}/*")
    if(NOT files)

        message(FATAL_ERROR
            "The git submodule '${directory}' is empty\n"
            "please do : git submodule update --init --recursive\n"
        )

    endif()
    #message(STATUS "The subdirectory '${directory}' contains ${files}")
endfunction()

_git_submod_is_empty(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)
