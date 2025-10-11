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
 * @file segmented_sort_in_place.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::primitives {

    template<class T, class Comp>
    inline void segmented_sort_in_place(
        sham::DeviceBuffer<T> &buf, const sham::DeviceBuffer<u32> &offsets, Comp &&comp) {

        auto &q = buf.get_dev_scheduler().get_queue();

        size_t interact_count = buf.get_size();
        size_t offsets_count  = offsets.get_size();
        size_t N              = offsets_count - 1;

        sham::kernel_call(
            q,
            sham::MultiRef{offsets},
            sham::MultiRef{buf},
            N,
            [interact_count,
             comp](u32 gid, const u32 *__restrict__ offsets, T *__restrict__ in_out_sorted) {
                u32 start_index = offsets[gid];
                u32 end_index   = offsets[gid + 1];

                // can be equal if there is no interaction for this sender
                SHAM_ASSERT(start_index <= end_index);

                // skip empty ranges to avoid unnecessary work
                if (start_index == end_index) {
                    return;
                }

                // if there is no interactions at the end of the offset list
                // offsets[gid] can be equal to interact_count
                // but we check that start_index != end_index, so here the correct assertions
                // is indeed start_index < interact_count
                SHAM_ASSERT(start_index < interact_count);
                SHAM_ASSERT(end_index <= interact_count); // see the for loop for this one

                // simple insertion sort between those indexes
                for (u32 i = start_index + 1; i < end_index; ++i) {
                    auto key = in_out_sorted[i];
                    u32 j    = i;
                    while (j > start_index && comp(key, in_out_sorted[j - 1])) {
                        in_out_sorted[j] = in_out_sorted[j - 1];
                        --j;
                    }
                    in_out_sorted[j] = key;
                }
            });
    }

} // namespace shamalgs::primitives
