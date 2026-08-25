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
        std::vector<std::shared_ptr<INode>> nodes;
        std::string name;

        // A sequence owns no ro/rw edges of its own, so it never goes through
        // __internal_set_ro_edges/__internal_set_rw_edges to get a state_update automatically
        // like other nodes do. Declared last so it constructs once `nodes`/`name` above are set,
        // and firing it from a member (rather than a manual notify_self_state_update() call in
        // the constructor body) keeps this correct-by-construction for any future meta node that
        // copies the pattern -- see NodeSelfStateUpdate in INode.hpp.
        NodeSelfStateUpdate<OperationSequence> self_update{*this};

        /// Validates before `nodes` is initialized, so that on failure no member of this class
        /// (in particular `self_update` above) ever gets constructed at all.
        static std::vector<std::shared_ptr<INode>> check_nonempty(
            std::vector<std::shared_ptr<INode>> nodes) {
            if (nodes.empty()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "OperationSequence must have at least one node");
            }
            return nodes;
        }

        public:
        OperationSequence(std::string name, std::vector<std::shared_ptr<INode>> &&_nodes)
            : nodes(check_nonempty(std::move(_nodes))), name(std::move(name)) {}
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
