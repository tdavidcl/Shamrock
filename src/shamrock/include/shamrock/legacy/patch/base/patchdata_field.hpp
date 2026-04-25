// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file patchdata_field.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamcomm/wrapper.hpp"
#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamrock/legacy/utils/sycl_vector_utils.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace patchdata_field {} // namespace patchdata_field
