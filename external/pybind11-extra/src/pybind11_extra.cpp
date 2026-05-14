// ~~~
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// ~~~

#include "pybind11_extra.hpp"

namespace pybind11_extra {

std::string test_function() {
    return "pybind11-extra says: hello from the extra library!";
}

} // namespace pybind11_extra
