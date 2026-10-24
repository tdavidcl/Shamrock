// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file OperationIf.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamsolvergraph/node/OperationIf.hpp"
#include <sstream>

namespace shamrock::solvergraph {

    void OperationIf::_impl_evaluate_internal() {
        if (get_edges().condition.data) {
            if (then_node) {
                then_node->evaluate();
            }
        } else if (else_node) {
            else_node->evaluate();
        }
    }

    NodeSubgraph OperationIf::_impl_get_subgraph() const {
        NodeSubgraph sg;
        sg.uuid             = get_uuid();
        sg.label            = _impl_get_label();
        sg.metadata         = _impl_get_metadata();
        sg.is_meta          = true;
        sg.dot_shape        = "diamond";
        sg.dot_start_id     = sham::format("n_{}", get_uuid());
        sg.dot_end_id       = sham::format("n_{}_end", get_uuid());
        sg.draws_own_anchor = true;
        sg.ro_edges         = get_ro_edges();
        sg.rw_edges         = get_rw_edges();

        auto mi = std::make_shared<SubgraphMetaInfo>();

        if (then_node) {
            NodeSubgraph then_sg = then_node->get_subgraph();
            mi->connections.push_back(
                SubgraphConnection{
                    .from_id = sg.dot_start_id, .to_id = then_sg.dot_start_id, .label = "true"});
            mi->connections.push_back(
                SubgraphConnection{.from_id = then_sg.dot_end_id, .to_id = sg.dot_end_id});
            mi->children.push_back(std::move(then_sg));
        } else {
            mi->connections.push_back(
                SubgraphConnection{
                    .from_id = sg.dot_start_id,
                    .to_id   = sg.dot_end_id,
                    .label   = "true",
                    .dashed  = true});
        }

        if (else_node) {
            NodeSubgraph else_sg = else_node->get_subgraph();
            mi->connections.push_back(
                SubgraphConnection{
                    .from_id = sg.dot_start_id, .to_id = else_sg.dot_start_id, .label = "false"});
            mi->connections.push_back(
                SubgraphConnection{.from_id = else_sg.dot_end_id, .to_id = sg.dot_end_id});
            mi->children.push_back(std::move(else_sg));
        } else {
            mi->connections.push_back(
                SubgraphConnection{
                    .from_id = sg.dot_start_id,
                    .to_id   = sg.dot_end_id,
                    .label   = "false",
                    .dashed  = true});
        }

        sg.meta_info = mi;

        return sg;
    }

    std::string OperationIf::_impl_get_tex() const {
        std::stringstream ss;
        ss << "Start : " << _impl_get_label() << "\n";
        if (then_node) {
            ss << "Then :\n" << then_node->get_tex_partial() << "\n";
        }
        if (else_node) {
            ss << "Else :\n" << else_node->get_tex_partial() << "\n";
        }
        ss << "End : " << _impl_get_label() << "\n";
        return ss.str();
    }

} // namespace shamrock::solvergraph
