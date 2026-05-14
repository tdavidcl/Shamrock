# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# ~~~

message("   ---- pybind11-extra section ----")

set(PYBIND11_EXTRA_FIND_PYBIND11 Off)
add_subdirectory(external/pybind11-extra)
