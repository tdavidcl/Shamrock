#include "shamsolvergraph/LifecycleTracer.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/NodeSetEdge.hpp"
#include "shamtest/shamtest.hpp"


namespace {
    void on_create(u64 uuid) {

    }

    void on_destroy(u64 uuid) {
    }
    
    void on_state_update(shamrock::solvergraph::IEdge &edge) {
    }

    void on_op(u64 uuid, u64 op_id) {
    }
}

NEW_TEST(Unittest, "shamsolvergraph/LifetimeTracker", 1) {
    using namespace shamrock::solvergraph;

    auto edge = IDataEdge<f64>::make_shared("a", "a");

    using NodeT = NodeSetEdge<IDataEdge<f64>>;
    NodeT set_edge([](IDataEdge<f64> &edge) {
        edge.data = 1;
    });

    set_edge.set_edges(edge);

    std::shared_ptr<NodeT> ptr = std::make_shared<NodeT>(std::move(set_edge));

    ptr.reset();
    edge.reset();


}
