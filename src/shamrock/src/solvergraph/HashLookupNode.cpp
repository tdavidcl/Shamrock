// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file HashLookupNode.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/HashLookupNode.hpp"
#include "shambackends/kernel_call.hpp"



namespace shamrock::solvergraph {

    template<class Tkey>
    void HashLookupNode<Tkey>::_impl_evaluate_internal() {

        auto edges = get_edges();

        auto &keys                  = edges.keys;
        auto &key_counts            = edges.key_counts;
        auto &keys_to_lookup_min    = edges.keys_to_lookup_min;
        auto &keys_to_lookup_max    = edges.keys_to_lookup_max;
        auto &lookup_request_counts = edges.lookup_request_counts;
        auto &lookup_results        = edges.lookup_results;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        // ensure allocation of lookup_results
        lookup_results.ensure_allocated(lookup_request_counts.indexes.get_ids());

        lookup_request_counts.indexes.for_each([&](u64 pid, u64 rq_count) {
            const sham::DeviceBuffer<Tkey> &buf_keys = keys.buffers.get(pid);
            u32 key_count                            = key_counts.indexes.get(pid);

            const sham::DeviceBuffer<Tkey> &buf_keys_to_lookup_min
                = keys_to_lookup_min.buffers.get(pid);
            const sham::DeviceBuffer<Tkey> &buf_keys_to_lookup_max
                = keys_to_lookup_max.buffers.get(pid);
            u32 lookup_request_count = lookup_request_counts.indexes.get(pid);

            sham::DeviceBuffer<u32> &buf_lookup_results = lookup_results.buffers.get(pid);

            // hash lookup
            sham::kernel_call(
                q,
                sham::MultiRef{buf_keys, buf_keys_to_lookup_min, buf_keys_to_lookup_max},
                sham::MultiRef{buf_lookup_results},
                lookup_request_count,
                [key_count](
                    u32 lid,
                    const Tkey *__restrict__ keys,
                    const Tkey *__restrict__ keys_to_lookup_min,
                    const Tkey *__restrict__ keys_to_lookup_max,
                    u32 *__restrict__ lookup_results) {
                    Tkey key_min = keys_to_lookup_min[lid];
                    Tkey key_max = keys_to_lookup_max[lid];

                    u32 cursor_min = 0;
                    u32 cursor_max = key_count - 1;

                    while (cursor_min <= cursor_max) {
                        u32 cursor_mid = (cursor_min + cursor_max) / 2;
                        Tkey key_mid   = keys[cursor_mid];
                    }
                });
        });
    }

    template<class Tkey>
    std::string HashLookupNode<Tkey>::_impl_get_tex() {
        return "TODO";
    }

    template class HashLookupNode<u32>;
    template class HashLookupNode<u64>;

} // namespace shamrock::solvergraph
