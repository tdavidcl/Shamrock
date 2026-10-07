// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/numeric_limits.hpp"
#include "shambase/profiling/chrome.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsolvergraph/node/OperationSequence.hpp"
#include "shamsolvergraph/tracing.hpp"
#include "shamtest/shamtest.hpp"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

    class TracingProbeNode : public shamrock::solvergraph::INode {
        public:
        std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> counter;

        explicit TracingProbeNode(std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> counter)
            : counter(std::move(counter)) {}

        TracingProbeNode() = default;

        void _impl_evaluate_internal() override {
            if (counter) {
                counter->data += 1;
            }
        }

        std::string _impl_get_label() const override { return "TracingProbe"; }
        std::string _impl_get_tex() const override { return ""; }
    };

    /// Read back and parse a JSON lines trace file
    std::vector<nlohmann::json> read_trace_file(const std::string &filename) {
        std::vector<nlohmann::json> records;
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                records.push_back(nlohmann::json::parse(line));
            }
        }
        return records;
    }

    /// Get the sequence number of the first record matching kind & uuid, or u64_max
    u64 find_seq(const std::vector<nlohmann::json> &records, const std::string &kind, u64 uuid) {
        for (const auto &r : records) {
            if (r.value("kind", "") == kind && r.value("uuid", u64_max) == uuid) {
                return r.at("seq").get<u64>();
            }
        }
        return u64_max;
    }

    /// Count the records matching kind & uuid
    u64 count_records(
        const std::vector<nlohmann::json> &records, const std::string &kind, u64 uuid) {
        u64 count = 0;
        for (const auto &r : records) {
            if (r.value("kind", "") == kind && r.value("uuid", u64_max) == uuid) {
                count++;
            }
        }
        return count;
    }

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/tracing", 1) {
    using namespace shamrock::solvergraph;
    namespace tracing = shamrock::solvergraph::tracing;

    // make sure tracing is off regardless of the environment
    tracing::disable();
    REQUIRE_NAMED("tracing disabled", !tracing::is_enabled());

    // with tracing disabled, hooks are null pointer checks and no state is recorded
    {
        TracingProbeNode node{};
        node.evaluate();
    }

    const std::string prefix   = "test_sgtrace_out";
    const u32 chrome_pid       = shambase::profiling::chrome::get_chrome_pid();
    const u32 rank             = (chrome_pid == u32_max) ? 0 : chrome_pid;
    const std::string filename = prefix + "_" + std::to_string(rank) + ".jsonl";

    std::remove(filename.c_str());

    tracing::set_outfile_prefix(prefix);
    tracing::enable();
    REQUIRE_NAMED("tracing enabled", tracing::is_enabled());

    u64 uuid_seq   = 0;
    u64 uuid_n1    = 0;
    u64 uuid_n2    = 0;
    u64 uuid_edge  = 0;
    u64 uuid_moved = 0;

    {
        auto counter = IDataEdge<u32>::make_shared("counter", "c");
        uuid_edge    = counter->get_uuid();

        auto n1 = std::make_shared<TracingProbeNode>(counter);
        auto n2 = std::make_shared<TracingProbeNode>(counter);
        uuid_n1 = n1->get_uuid();
        uuid_n2 = n2->get_uuid();

        // bind the counter edge to record topology updates
        n1->__internal_set_ro_edges({});
        n1->__internal_set_rw_edges({counter});

        std::vector<std::shared_ptr<INode>> nodes = {n1, n2};
        OperationSequence seq("test sequence", std::move(nodes));
        uuid_seq = seq.get_uuid();

        seq.evaluate();
        seq.evaluate();

        REQUIRE_EQUAL(counter->data, u32{4});

        // moved-from nodes must not emit a duplicate destroy event
        {
            TracingProbeNode a{};
            uuid_moved = a.get_uuid();

            TracingProbeNode b = std::move(a);
            REQUIRE_EQUAL(b.get_uuid(), uuid_moved);
        }
    }

    tracing::disable();
    REQUIRE_NAMED("tracing disabled again", !tracing::is_enabled());

    auto records = read_trace_file(filename);

    REQUIRE_NAMED("trace file is not empty", !records.empty());
    REQUIRE_EQUAL_NAMED("first record is the header", records.at(0).value("kind", ""), "header");
    REQUIRE_EQUAL_NAMED("header version", records.at(0).at("version").get<u32>(), u32{1});
    REQUIRE_EQUAL_NAMED("header rank", records.at(0).at("rank").get<u32>(), rank);
    REQUIRE_NAMED("header has clock_origin", records.at(0).contains("clock_origin"));

    // lifetime events
    REQUIRE_EQUAL_NAMED("n1 created once", count_records(records, "node_create", uuid_n1), 1);
    REQUIRE_EQUAL_NAMED("n1 destroyed once", count_records(records, "node_destroy", uuid_n1), 1);
    REQUIRE_EQUAL_NAMED("edge created once", count_records(records, "edge_create", uuid_edge), 1);
    REQUIRE_EQUAL_NAMED(
        "edge destroyed once", count_records(records, "edge_destroy", uuid_edge), 1);

    // move safety: exactly one create and one destroy despite the move
    REQUIRE_EQUAL_NAMED(
        "moved node created once", count_records(records, "node_create", uuid_moved), 1);
    REQUIRE_EQUAL_NAMED(
        "moved node destroyed once", count_records(records, "node_destroy", uuid_moved), 1);

    // evaluation events (two sequence evaluations, each evaluating both children)
    REQUIRE_EQUAL_NAMED(
        "seq evaluated twice", count_records(records, "node_evaluate_begin", uuid_seq), 2);
    REQUIRE_EQUAL_NAMED(
        "n1 evaluated twice", count_records(records, "node_evaluate_begin", uuid_n1), 2);
    REQUIRE_EQUAL_NAMED(
        "n2 evaluated twice", count_records(records, "node_evaluate_end", uuid_n2), 2);

    // ordering: create < evaluate_begin < evaluate_end < destroy
    u64 seq_create = find_seq(records, "node_create", uuid_n1);
    u64 seq_begin  = find_seq(records, "node_evaluate_begin", uuid_n1);
    u64 seq_end    = find_seq(records, "node_evaluate_end", uuid_n1);
    u64 seq_destr  = find_seq(records, "node_destroy", uuid_n1);
    REQUIRE_NAMED("create before evaluate_begin", seq_create < seq_begin);
    REQUIRE_NAMED("evaluate_begin before evaluate_end", seq_begin < seq_end);
    REQUIRE_NAMED("evaluate_end before destroy", seq_end < seq_destr);

    // nesting: children evaluate inside the sequence evaluation
    u64 seq_seq_begin = find_seq(records, "node_evaluate_begin", uuid_seq);
    u64 seq_seq_end   = find_seq(records, "node_evaluate_end", uuid_seq);
    u64 seq_n2_begin  = find_seq(records, "node_evaluate_begin", uuid_n2);
    REQUIRE_NAMED("sequence begin before child begin", seq_seq_begin < seq_begin);
    REQUIRE_NAMED("child n1 begin before child n2 begin", seq_begin < seq_n2_begin);
    REQUIRE_NAMED("child n2 begin before sequence end", seq_n2_begin < seq_seq_end);

    // topology: the last node_update of n1 lists the counter as a rw edge
    const nlohmann::json *last_update = nullptr;
    for (const auto &r : records) {
        if (r.value("kind", "") == "node_update" && r.value("uuid", u64_max) == uuid_n1) {
            last_update = &r;
        }
    }
    REQUIRE_NAMED("n1 has a node_update record", last_update != nullptr);
    if (last_update != nullptr) {
        REQUIRE_EQUAL_NAMED(
            "node_update label", last_update->at("label").get<std::string>(), "TracingProbe");
        REQUIRE_EQUAL_NAMED("node_update ro count", last_update->at("ro_edges").size(), 0);
        REQUIRE_EQUAL_NAMED("node_update rw count", last_update->at("rw_edges").size(), 1);
        REQUIRE_EQUAL_NAMED(
            "node_update rw edge uuid",
            last_update->at("rw_edges").at(0).at("uuid").get<u64>(),
            uuid_edge);
        REQUIRE_EQUAL_NAMED(
            "node_update rw edge label",
            last_update->at("rw_edges").at(0).at("label").get<std::string>(),
            "counter");
    }

    // all records carry a monotonically increasing seq within the file
    u64 last_seq   = 0;
    bool monotonic = true;
    for (const auto &r : records) {
        if (r.contains("seq")) {
            u64 s = r.at("seq").get<u64>();
            if (s < last_seq) {
                monotonic = false;
            }
            last_seq = s;
        }
    }
    REQUIRE_NAMED("seq is monotonically increasing", monotonic);

    std::remove(filename.c_str());
}
