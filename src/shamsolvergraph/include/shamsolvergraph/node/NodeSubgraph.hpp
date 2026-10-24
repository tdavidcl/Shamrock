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
 * @file NodeSubgraph.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Structured (non-string) description of an INode's structure, for introspection
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamsolvergraph/edge/IEdge.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace shamrock::solvergraph {

    class NodeSubgraph;

    /// A structural connection between two dot-graph anchor ids (not a data edge) - e.g.
    /// OperationSequence's evaluation order, or OperationIf's branch links.
    struct SubgraphConnection {
        /// dot node id this connection starts from
        std::string from_id;
        /// dot node id this connection points to
        std::string to_id;
        /// optional label, e.g. "true" / "false" for OperationIf branches
        std::string label;
        /// rendered as a dashed/placeholder edge (e.g. an absent optional branch)
        bool dashed = false;
    };

    /// Extra structure carried only by "meta" (composite/group) nodes.
    struct SubgraphMetaInfo {
        /// recursively-built subgraphs of the nested nodes
        std::vector<NodeSubgraph> children;
        /// structural connections between the children (and/or the meta node's own anchor)
        std::vector<SubgraphConnection> connections;
    };

    /// Structured description of one node and, if it is a meta node, everything nested under it.
    class NodeSubgraph {
        public:
        /// uuid of the described node
        u64 uuid;
        /// label of the described node
        std::string label;
        /// read only edges of the described node
        std::vector<std::shared_ptr<IEdge>> ro_edges;
        /// read write edges of the described node
        std::vector<std::shared_ptr<IEdge>> rw_edges;
        /// free-form key/value metadata, formatted by the node itself
        std::vector<std::pair<std::string, std::string>> metadata;
        /// true if this node is a group of nested nodes (composite/meta node)
        bool is_meta = false;
        /// non-null iff is_meta is true
        std::shared_ptr<SubgraphMetaInfo> meta_info;

        // Rendering-hint fields, needed so `get_dot_graph()` can be losslessly rebuilt from this
        // struct (see dot_graph_from_subgraph below) - not part of the introspection contract.

        /// graphviz "shape" attribute for the node's own drawn box, "" = default
        std::string dot_shape;
        /// dot node id used to connect INTO this (sub)graph from a predecessor
        std::string dot_start_id;
        /// dot node id used to connect OUT of this (sub)graph to a successor
        std::string dot_end_id;
        /// true if dot_start_id/dot_end_id refer to a node this subgraph draws itself (false when
        /// they are only an alias into a child's ids, e.g. OperationSequence)
        bool draws_own_anchor = true;
    };

    /// Render a NodeSubgraph tree to a Graphviz DOT fragment.
    std::string dot_graph_from_subgraph(const NodeSubgraph &sg);

} // namespace shamrock::solvergraph
