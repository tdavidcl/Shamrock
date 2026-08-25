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
 * @file OperationSequence.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamsolvergraph/node/INode.hpp"

namespace shamrock::solvergraph {

    class OperationSequence : public INode {
        std::string name;
        // Stored through INode::ChildNodes instead of a plain vector: constructing it fires
        // the self state_update automatically, putting the sequence on record as up to date
        // before it can be evaluated. A sequence owns no ro/rw edges of its own, so it never
        // goes through __internal_set_ro_edges/__internal_set_rw_edges, which is how regular
        // nodes get theirs fired. Declared after `name` so the node is fully readable
        // (e.g. get_label()) by the time observers are notified.
        ChildNodes nodes;

        /// Validated before ChildNodes stores the children (and fires the self state_update):
        /// an empty sequence must fail construction without notifying any state.
        static std::vector<std::shared_ptr<INode>> check_not_empty(
            std::vector<std::shared_ptr<INode>> &&nodes) {
            if (nodes.size() == 0) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "OperationSequence must have at least one node");
            }
            return std::move(nodes);
        }

        public:
        OperationSequence(std::string name, std::vector<std::shared_ptr<INode>> &&_nodes)
            : name(name), nodes(*this, check_not_empty(std::move(_nodes))) {}
        void _impl_evaluate_internal();

        inline std::string _impl_get_label() const { return name; }

        std::string _impl_get_dot_graph_partial() const;

        inline virtual std::string _impl_get_dot_graph_node_start() const {
            return nodes[0]->get_dot_graph_node_start();
        }
        inline virtual std::string _impl_get_dot_graph_node_end() const {
            return nodes[nodes.size() - 1]->get_dot_graph_node_end();
        }

        std::string _impl_get_tex() const;
    };

} // namespace shamrock::solvergraph
