// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeSubgraph.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamsolvergraph/node/NodeSubgraph.hpp"
#include "sham/format/format.hpp"
#include <sstream>

namespace {

    using namespace shamrock::solvergraph;

    void render_ro_rw_edges(
        std::stringstream &ss,
        const std::string &attach_id,
        const std::vector<std::shared_ptr<IEdge>> &ro_edges,
        const std::vector<std::shared_ptr<IEdge>> &rw_edges) {
        for (auto &in : ro_edges) {
            ss << sham::format(
                "e_{} -> {} [style=\"dashed\", color=green];\n", in->get_uuid(), attach_id);
            ss << sham::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n", in->get_uuid(), in->get_label());
        }
        for (auto &out : rw_edges) {
            ss << sham::format(
                "{} -> e_{} [style=\"dashed\", color=red];\n", attach_id, out->get_uuid());
            ss << sham::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n",
                out->get_uuid(),
                out->get_label());
        }
    }

    void render_node(std::stringstream &ss, const NodeSubgraph &sg) {
        if (!sg.is_meta) {
            std::string shape_attr
                = sg.dot_shape.empty() ? "" : sham::format(", shape={}", sg.dot_shape);
            ss << sham::format("{} [label=\"{}\"{}];\n", sg.dot_start_id, sg.label, shape_attr);
            render_ro_rw_edges(ss, sg.dot_start_id, sg.ro_edges, sg.rw_edges);
            return;
        }

        const SubgraphMetaInfo &mi = *sg.meta_info;

        ss << "subgraph cluster_" << sg.uuid << " {\n";

        if (sg.draws_own_anchor) {
            std::string shape_attr
                = sg.dot_shape.empty() ? "" : sham::format(", shape={}", sg.dot_shape);
            ss << sham::format("{} [label=\"{}\"{}];\n", sg.dot_start_id, sg.label, shape_attr);
            if (sg.dot_end_id != sg.dot_start_id) {
                ss << sham::format("{} [label=\"\", shape=point, width=0.15];\n", sg.dot_end_id);
            }
            render_ro_rw_edges(ss, sg.dot_start_id, sg.ro_edges, sg.rw_edges);
        }

        for (auto &child : mi.children) {
            render_node(ss, child);
        }

        for (auto &conn : mi.connections) {
            std::string label_attr
                = conn.label.empty() ? "" : sham::format("label=\"{}\"", conn.label);
            std::string style_attr = conn.dashed ? "style=dashed" : "";
            std::string attrs;
            if (!label_attr.empty() && !style_attr.empty()) {
                attrs = sham::format(" [{}, {}]", label_attr, style_attr);
            } else if (!label_attr.empty()) {
                attrs = sham::format(" [{}]", label_attr);
            } else if (!style_attr.empty()) {
                attrs = sham::format(" [{}]", style_attr);
            }
            ss << sham::format("{} -> {}{};\n", conn.from_id, conn.to_id, attrs);
        }

        ss << sham::format("label = \"{}\";\n", sg.label);
        ss << "}\n";
    }

} // namespace

namespace shamrock::solvergraph {

    std::string dot_graph_from_subgraph(const NodeSubgraph &sg) {
        std::stringstream ss;
        render_node(ss, sg);
        return ss.str();
    }

} // namespace shamrock::solvergraph
