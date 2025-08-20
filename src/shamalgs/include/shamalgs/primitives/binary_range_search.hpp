// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file binary_range_search.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shamalgs/primitives/lower_bound.hpp"
#include "shamalgs/primitives/upper_bound.hpp"

namespace shamalgs::primitives {

    /// return the smallest index range for which keys[inf] <= value_min <= value_max <= keys[sup]
    template<class Tkey>
    constexpr void binary_range_search(
        const Tkey *__restrict__ key,
        u32 first,
        u32 last,
        const Tkey &value_min,
        const Tkey &value_max,
        u32 &inf,
        u32 &sup) {

        SHAM_ASSERT(value_min <= value_max);
        SHAM_ASSERT((first < last) ? key[first] <= value_min : true);
        SHAM_ASSERT((first < last) ? key[last - 1] >= value_max : true);

        inf = binary_search_lower_bound(key, first, last, value_min);
        sup = binary_search_upper_bound(key, first, last, value_max);

        // std::lower_bound and std::upper_bound are quirky:
        //  - std::lower_bound returns the first such that >= value_min (searching 8 in [7,9] return
        //      1 where we want 0)
        //  - std::upper_bound returns the first such that > value_max (searching 8 in
        //      [7,8,9] return 2 where we want 1)
        // The following ajust that to the range with expect in this function
        if (inf > first && inf < last) {
            inf -= (key[inf] > value_min);
        }
        if (sup > first && sup <= last) {
            sup -= (key[sup - 1] == value_max);
        }
    }

} // namespace shamalgs::primitives
