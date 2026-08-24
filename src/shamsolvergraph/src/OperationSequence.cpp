// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file OperationSequence.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamsolvergraph/node/OperationSequence.hpp"
#include <sstream>

namespace shamrock::solvergraph {

    void OperationSequence::_impl_evaluate_internal() {
        for (auto &node : nodes) {
            node->evaluate();
        }
    }

    NodeSubgraph OperationSequence::_impl_get_subgraph() const {
        NodeSubgraph sg;
        sg.uuid             = get_uuid();
        sg.label            = _impl_get_label();
        sg.metadata         = _impl_get_metadata();
        sg.is_meta          = true;
        sg.draws_own_anchor = false;

        auto mi = std::make_shared<SubgraphMetaInfo>();
        for (auto &node : nodes) {
            mi->children.push_back(node->get_subgraph());
        }

        for (size_t i = 0; i + 1 < mi->children.size(); i++) {
            mi->connections.push_back(
                SubgraphConnection{
                    .from_id = mi->children[i].dot_end_id,
                    .to_id   = mi->children[i + 1].dot_start_id,
                    .label   = "",
                    .dashed  = false});
        }

        sg.dot_start_id = mi->children.front().dot_start_id;
        sg.dot_end_id   = mi->children.back().dot_end_id;
        sg.meta_info    = mi;

        return sg;
    }

    std::string OperationSequence::_impl_get_tex() const {
        std::stringstream ss;
        ss << "Start : " << _impl_get_label() << "\n";
        for (auto &node : nodes) {
            ss << node->get_tex_partial() << "\n";
        }
        ss << "End : " << _impl_get_label() << "\n";
        return ss.str();
    }

} // namespace shamrock::solvergraph
