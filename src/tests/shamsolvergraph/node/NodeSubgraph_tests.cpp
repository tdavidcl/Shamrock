// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsolvergraph/node/OperationIf.hpp"
#include "shamsolvergraph/node/OperationSequence.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <string>

namespace {

#define DUMMY_NODE_EDGES(X_RO, X_RW)                                                               \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, in)                                                \
    X_RW(shamrock::solvergraph::IDataEdge<u32>, out)

    class DummyNode : public shamrock::solvergraph::INode {
        std::string label;

        public:
        explicit DummyNode(std::string label) : label(std::move(label)) {}

        EXPAND_NODE_EDGES(DUMMY_NODE_EDGES)

        void _impl_evaluate_internal() override {}

        std::string _impl_get_label() const override { return label; }
        std::string _impl_get_tex() const override { return ""; }
    };

#undef DUMMY_NODE_EDGES

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/node/NodeSubgraph:leaf", 1) {
    using namespace shamrock::solvergraph;

    auto in_edge  = IDataEdge<u32>::make_shared("in", "i");
    auto out_edge = IDataEdge<u32>::make_shared("out", "o");

    auto node = std::make_shared<DummyNode>("Dummy");
    node->set_edges(in_edge, out_edge);

    NodeSubgraph sg = node->get_subgraph();

    REQUIRE_EQUAL(sg.is_meta, false);
    REQUIRE_EQUAL(bool(sg.meta_info), false);
    REQUIRE_EQUAL(sg.label, std::string("Dummy"));
    REQUIRE_EQUAL(sg.ro_edges.size(), size_t(1));
    REQUIRE_EQUAL(sg.rw_edges.size(), size_t(1));
    REQUIRE_EQUAL(sg.ro_edges[0]->get_uuid(), in_edge->get_uuid());
    REQUIRE_EQUAL(sg.rw_edges[0]->get_uuid(), out_edge->get_uuid());
}

NEW_TEST(Unittest, "shamsolvergraph/node/NodeSubgraph:operation_sequence", 1) {
    using namespace shamrock::solvergraph;

    auto n1 = std::make_shared<DummyNode>("n1");
    auto n2 = std::make_shared<DummyNode>("n2");
    auto n3 = std::make_shared<DummyNode>("n3");

    std::vector<std::shared_ptr<INode>> nodes{n1, n2, n3};
    OperationSequence seq("seq", std::move(nodes));

    NodeSubgraph sg = seq.get_subgraph();

    REQUIRE_EQUAL(sg.is_meta, true);
    REQUIRE_EQUAL(bool(sg.meta_info), true);
    REQUIRE_EQUAL(sg.meta_info->children.size(), size_t(3));
    REQUIRE_EQUAL(sg.meta_info->connections.size(), size_t(2));

    REQUIRE_EQUAL(sg.meta_info->connections[0].from_id, sg.meta_info->children[0].dot_end_id);
    REQUIRE_EQUAL(sg.meta_info->connections[0].to_id, sg.meta_info->children[1].dot_start_id);
    REQUIRE_EQUAL(sg.meta_info->connections[1].from_id, sg.meta_info->children[1].dot_end_id);
    REQUIRE_EQUAL(sg.meta_info->connections[1].to_id, sg.meta_info->children[2].dot_start_id);
}

NEW_TEST(Unittest, "shamsolvergraph/node/NodeSubgraph:operation_if", 1) {
    using namespace shamrock::solvergraph;

    auto make_condition = []() {
        return IDataEdge<bool>::make_shared("cond", "cond");
    };

    {
        // both branches present
        auto then_node = std::make_shared<DummyNode>("then");
        auto else_node = std::make_shared<DummyNode>("else");

        OperationIf node("if", then_node, else_node);
        node.set_edges(make_condition());

        NodeSubgraph sg = node.get_subgraph();

        REQUIRE_EQUAL(sg.is_meta, true);
        REQUIRE_EQUAL(bool(sg.meta_info), true);
        REQUIRE_EQUAL(sg.ro_edges.size(), size_t(1));
        REQUIRE_EQUAL(sg.meta_info->children.size(), size_t(2));
        REQUIRE_EQUAL(sg.meta_info->connections.size(), size_t(4));

        bool found_true  = false;
        bool found_false = false;
        for (auto &conn : sg.meta_info->connections) {
            if (conn.label == "true") {
                found_true = true;
                REQUIRE_EQUAL(conn.dashed, false);
            }
            if (conn.label == "false") {
                found_false = true;
                REQUIRE_EQUAL(conn.dashed, false);
            }
        }
        REQUIRE_EQUAL(found_true, true);
        REQUIRE_EQUAL(found_false, true);
    }

    {
        // both branches absent
        OperationIf node("if_empty");
        node.set_edges(make_condition());

        NodeSubgraph sg = node.get_subgraph();

        REQUIRE_EQUAL(sg.is_meta, true);
        REQUIRE_EQUAL(bool(sg.meta_info), true);
        REQUIRE_EQUAL(sg.meta_info->children.size(), size_t(0));
        REQUIRE_EQUAL(sg.meta_info->connections.size(), size_t(2));

        for (auto &conn : sg.meta_info->connections) {
            REQUIRE_EQUAL(conn.dashed, true);
            REQUIRE_EQUAL(conn.from_id, sg.dot_start_id);
            REQUIRE_EQUAL(conn.to_id, sg.dot_end_id);
        }
    }
}
