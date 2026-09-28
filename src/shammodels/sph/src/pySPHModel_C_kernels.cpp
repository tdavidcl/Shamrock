// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel_C_kernels.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Instantiation of the SPH model python bindings for the C2, C4 & C6 kernels
 *
 * Split from pySPHModel.cpp (M4, M6 & M8 kernels) so that both halves compile in parallel.
 */

#include "pySPHModel_add_instance.hpp"

template void add_instance<f64_3, shammath::C2>(
    py::module &m, std::string name_config, std::string name_model);
template void add_instance<f64_3, shammath::C4>(
    py::module &m, std::string name_config, std::string name_model);
template void add_instance<f64_3, shammath::C6>(
    py::module &m, std::string name_config, std::string name_model);
