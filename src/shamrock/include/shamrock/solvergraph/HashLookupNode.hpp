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
 * @file HashLookupNode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief 
 *
 */

#include "shamrock/solvergraph/DistributedBuffers.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shamrock::solvergraph {

    template<class Tkey>
    class HashLookupNode : public INode {
        public:
        HashLookupNode() {}

        struct Edges {
            const DistributedBuffers<Tkey> &keys;
            const Indexes<u32> &key_counts;
            const DistributedBuffers<Tkey> &keys_to_lookup_min;
            const DistributedBuffers<Tkey> &keys_to_lookup_max;
            const Indexes<u32> &lookup_request_counts;

            DistributedBuffers<u32> &lookup_results;
        };

        inline void set_edges(
            std::shared_ptr<DistributedBuffers<Tkey>> keys,
            std::shared_ptr<Indexes<u32>> key_counts,
            std::shared_ptr<DistributedBuffers<Tkey>> keys_to_lookup_min,
            std::shared_ptr<DistributedBuffers<Tkey>> keys_to_lookup_max,
            std::shared_ptr<Indexes<u32>> lookup_request_counts,
            std::shared_ptr<DistributedBuffers<u32>> lookup_results) {
            __internal_set_ro_edges(
                {keys, key_counts, keys_to_lookup_min, keys_to_lookup_max, lookup_request_counts});
            __internal_set_rw_edges({lookup_results});
        }

        Edges get_edges() {
            return Edges{
                get_ro_edge<DistributedBuffers<Tkey>>(0),
                get_ro_edge<Indexes<u32>>(1),
                get_ro_edge<DistributedBuffers<Tkey>>(2),
                get_ro_edge<DistributedBuffers<Tkey>>(3),
                get_ro_edge<Indexes<u32>>(4),
                get_rw_edge<DistributedBuffers<u32>>(5)};
        }

        void _impl_evaluate_internal();

        std::string _impl_get_label() { return "HashLookup"; }
        
        std::string _impl_get_tex();

    };
} // namespace shamrock::solvergraph
