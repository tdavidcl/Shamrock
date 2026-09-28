// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel_M8.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Explicit instantiation of the SPH model python bindings for the M8 kernel
 */

#include "pySPHModel_add_instance.hpp"
#include "shammath/sphkernels.hpp"

template void add_instance<f64_3, shammath::M8>(
    py::module &m, std::string name_config, std::string name_model);
