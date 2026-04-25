// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file patchdata_field.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

//%Impl status : Good

#include "shamrock/legacy/patch/base/patchdata_field.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
// #include "shamrock/legacy/patch/base/pdat_comm_impl/pdat_comm_cp_to_host.hpp"
// #include "shamrock/legacy/patch/base/pdat_comm_impl/pdat_comm_directgpu.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <cstdio>
#include <memory>

// TODO use hash for name + nvar to check if the field match before doing operation on them

namespace patchdata_field {} // namespace patchdata_field
