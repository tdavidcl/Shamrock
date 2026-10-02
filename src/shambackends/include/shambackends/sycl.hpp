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
 * @file sycl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "typeAliasBase.hpp" // IWYU pragma: export
#include "typeAliasFp16.hpp" // IWYU pragma: export
#include "typeAliasVec.hpp"  // IWYU pragma: export
#include <sycl/sycl.hpp>     /// IWYU pragma: export

/**
 * @brief Defined when work-groups of a kernel are not guaranteed to make progress concurrently
 *
 * With AdaptiveCpp omp.library-only (kernels compiled by the host compiler, no AdaptiveCpp
 * compiler plugin), nd_range work-groups are run as fibers by a few OpenMP threads. Algorithms
 * where a work-group waits on another one (e.g. decoupled look-back scans) can deadlock there and
 * must not be used.
 */
#if defined(__ACPP_ENABLE_OMPHOST_TARGET__) && !defined(__ACPP_USE_ACCELERATED_CPU__)              \
    && !defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__) && !defined(__ACPP_ENABLE_CUDA_TARGET__)         \
    && !defined(__ACPP_ENABLE_HIP_TARGET__)
    #define SHAMROCK_NO_INTERGROUP_FORWARD_PROGRESS
#endif

enum SYCLImplementation { ACPP, DPCPP, UNKNOWN };

#ifdef SYCL_COMP_ACPP
constexpr SYCLImplementation sycl_implementation = ACPP;
#else
    #ifdef SYCL_COMP_INTEL_LLVM
constexpr SYCLImplementation sycl_implementation = DPCPP;
    #else
constexpr SYCLImplementation sycl_implementation = UNKNOWN;
    #endif
#endif
