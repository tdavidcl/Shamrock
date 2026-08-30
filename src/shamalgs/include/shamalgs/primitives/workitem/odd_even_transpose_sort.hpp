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
 * @file odd_even_transpose_sort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Work-item local odd-even transposition sort
 */

#include <utility>

namespace shamalgs::primitives::workitem {

    /**
     * @brief Sorts keys (and their associated values) held in per-work-item registers using an
     * odd-even transposition sort.
     *
     * Adapted from moderngpu's OddEvenTransposeSort, see
     * https://moderngpu.github.io/mergesort.html
     *
     * @tparam VT Number of elements per work-item.
     * @tparam T Type of the keys.
     * @tparam V Type of the values.
     * @tparam Comp Type of the comparator.
     * @param keys Pointer to the local array of VT keys to sort.
     * @param values Pointer to the local array of VT values, permuted alongside the keys.
     * @param comp Comparator used to order the keys.
     */
    template<int VT, typename T, typename V, typename Comp>
    inline void odd_even_transpose_sort(T keys[VT], V values[VT], Comp comp) {
#pragma unroll
        for (int level = 0; level < VT; ++level) {

#pragma unroll
            for (int i = 1 & level; i < VT - 1; i += 2) {
                if (comp(keys[i + 1], keys[i])) {
                    std::swap(keys[i], keys[i + 1]);
                    std::swap(values[i], values[i + 1]);
                }
            }
        }
    }

} // namespace shamalgs::primitives::workitem
