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
 * @file memory_pool.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Control of the pool of freed USM blocks used by the allocation routines
 * (implemented in internal_alloc.cpp).
 */

#include <cstddef>

namespace sham::details {

    /**
     * @brief Enable or disable the pool of freed USM blocks (enabled by default).
     *
     * When enabled, freed blocks are kept for reuse by later allocations of the same size class
     * instead of being returned to the SYCL runtime. Disabling it releases the cached blocks.
     */
    void set_memory_pool_enabled(bool enable);

    /// @brief Whether the pool of freed USM blocks is enabled
    bool is_memory_pool_enabled();

    /// @brief Release every block cached in the pool of freed USM blocks
    void release_memory_pool();

    /// @brief Number of bytes currently cached (free) in the pool of freed USM blocks
    size_t get_memory_pool_cached_bytes();

} // namespace sham::details
