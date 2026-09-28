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
 * @file PinBitonicSort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Pin the sort by key implementation to the bitonic sort for the duration of a test.
 *
 * The relative order of the objects sharing the same Morton code depends on the sort by key
 * implementation (the bitonic sort is not stable, the radix sort is). The tests whose expected
 * values encode that order pin the bitonic sort, then restore the previous selection.
 */

#include "shamalgs/primitives/sort_by_key_pow2_len.hpp"
#include "shamsys/NodeInstance.hpp"
#include <string>

/// RAII guard selecting the bitonic sort by key implementation
struct PinBitonicSort {
    bool was_set;
    std::string previous;

    PinBitonicSort() {
        namespace impl = shamalgs::primitives::impl;
        was_set        = impl::is_impl_set_sort_by_key_pow2_len();
        if (was_set) {
            previous = impl::get_current_impl_sort_by_key_pow2_len();
        }
        impl::set_impl_sort_by_key_pow2_len(R"({"implementation":"bitonic_sort"})");
    }

    ~PinBitonicSort() {
        namespace impl = shamalgs::primitives::impl;
        if (was_set) {
            impl::set_impl_sort_by_key_pow2_len(previous);
        } else {
            impl::autoselect_impl_sort_by_key_pow2_len(
                shamsys::instance::get_compute_scheduler_ptr());
        }
    }

    PinBitonicSort(const PinBitonicSort &)            = delete;
    PinBitonicSort &operator=(const PinBitonicSort &) = delete;
};
