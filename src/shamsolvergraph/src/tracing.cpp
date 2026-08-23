// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file tracing.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the solvergraph execution tracer
 *
 */

#include "shambase/numeric_limits.hpp"
#include "shambase/profiling/chrome.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamsolvergraph/LifetimeTracker.hpp"
#include "shamsolvergraph/edge/IEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsolvergraph/tracing.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

    using namespace shamrock::solvergraph;

    /// Version of the trace file format, bump when the schema changes
    constexpr u32 trace_format_version = 1;

    /// Kinds of the hot path POD events
    enum class EventKind : u32 {
        node_create         = 0,
        node_destroy        = 1,
        edge_create         = 2,
        edge_destroy        = 3,
        node_evaluate_begin = 4,
        node_evaluate_end   = 5,
    };

    /// Get the trace file record name of a POD event kind
    const char *kind_name(EventKind kind) {
        switch (kind) {
        case EventKind::node_create        : return "node_create";
        case EventKind::node_destroy       : return "node_destroy";
        case EventKind::edge_create        : return "edge_create";
        case EventKind::edge_destroy       : return "edge_destroy";
        case EventKind::node_evaluate_begin: return "node_evaluate_begin";
        case EventKind::node_evaluate_end  : return "node_evaluate_end";
        }
        return "unknown";
    }

    /// Hot path event, fixed size POD, no strings attached
    struct PodEvent {
        u64 seq;        ///< global sequence number, gives a total order on all records
        f64 wtime;      ///< wall clock time (shambase::details::get_wtime())
        EventKind kind; ///< kind of the event
        u64 uuid;       ///< UUID of the node or edge
    };

    /// Cold path record (topology updates), pre-serialized at record time
    struct MetaRecord {
        u64 seq;               ///< global sequence number
        std::string json_line; ///< pre-serialized JSON line
    };

    /// State of the solvergraph tracer
    struct TracerState {

        /// Is tracing enabled (callbacks installed) ?
        bool enabled = false;

        /// Global sequence number counter
        u64 seq_counter = 0;

        /// Current node evaluation depth (0 = not inside any evaluate())
        u64 eval_depth = 0;

        /// Buffered hot path events
        std::vector<PodEvent> pod_events;

        /// Buffered cold path records
        std::vector<MetaRecord> meta_records;

        /// Output file prefix
        std::string outfile_prefix;

        /// Output stream, opened at first flush once the MPI world rank is known
        std::unique_ptr<std::ofstream> stream;

        /// Force a flush (bounds buffered memory) when this many events are pending
        static constexpr size_t max_buffered_events = 1 << 20;

        TracerState() {
            pod_events.reserve(1 << 14);

            const char *prefix = std::getenv("SHAM_SOLVERGRAPH_TRACE_PREFIX");
            outfile_prefix     = (prefix != nullptr) ? prefix : "shamrock_sgtrace";
        }

        /// Final flush at program exit
        ~TracerState() { flush_events(true); }

        /// Get the next sequence number
        u64 next_seq() { return seq_counter++; }

        /**
         * @brief Flush the buffered events to the trace file
         *
         * Does nothing if the MPI world rank is not known yet (events keep buffering), unless
         * force is set in which case rank 0 is assumed.
         *
         * @param force Flush even if the rank is unknown
         */
        void flush_events(bool force) {

            if (pod_events.empty() && meta_records.empty()) {
                return;
            }

            u32 rank = shambase::profiling::chrome::get_chrome_pid();
            if (rank == shambase::get_max<u32>()) {
                if (!force) {
                    return; // world info not fetched yet, keep buffering
                }
                rank = 0;
            }

            if (!stream) {
                std::string filename = sham::format("{}_{}.jsonl", outfile_prefix, rank);
                stream               = std::make_unique<std::ofstream>(filename);

                // header record: gives the format version, the rank and the clock origin
                // (viewers should display t = wtime - clock_origin, the clock origin is
                // captured at an MPI barrier so it is aligned across ranks)
                *stream << sham::format(
                    "{{\"kind\":\"header\",\"version\":{},\"rank\":{},\"clock_origin\":{}}}\n",
                    trace_format_version,
                    rank,
                    shambase::profiling::chrome::get_time_offset());
            }

            // merge the two buffers by sequence number (both are sorted since the sequence
            // counter is monotonic)
            size_t i_pod  = 0;
            size_t i_meta = 0;
            while (i_pod < pod_events.size() || i_meta < meta_records.size()) {

                bool take_pod = i_pod < pod_events.size()
                                && (i_meta >= meta_records.size()
                                    || pod_events[i_pod].seq < meta_records[i_meta].seq);

                if (take_pod) {
                    const PodEvent &e = pod_events[i_pod++];
                    *stream << sham::format(
                        "{{\"kind\":\"{}\",\"seq\":{},\"wtime\":{},\"uuid\":{}}}\n",
                        kind_name(e.kind),
                        e.seq,
                        e.wtime,
                        e.uuid);
                } else {
                    *stream << meta_records[i_meta++].json_line << "\n";
                }
            }

            pod_events.clear();
            meta_records.clear();

            // keep the file content complete after each flush so it can be tailed live
            stream->flush();
        }
    };

    /// Get the tracer state singleton
    TracerState &state() {
        static TracerState instance{};
        return instance;
    }

    /// Record a hot path POD event
    inline void record_pod(EventKind kind, u64 uuid) {
        TracerState &s = state();
        s.pod_events.push_back(PodEvent{s.next_seq(), shambase::details::get_wtime(), kind, uuid});
        if (s.pod_events.size() + s.meta_records.size() >= TracerState::max_buffered_events) {
            s.flush_events(false);
        }
    }

    void cb_node_create(u64 uuid) { record_pod(EventKind::node_create, uuid); }
    void cb_node_destroy(u64 uuid) { record_pod(EventKind::node_destroy, uuid); }
    void cb_edge_create(u64 uuid) { record_pod(EventKind::edge_create, uuid); }
    void cb_edge_destroy(u64 uuid) { record_pod(EventKind::edge_destroy, uuid); }

    /// Record a topology update of a node (cold path, only fires at graph build/rewire time)
    void cb_node_update(INode &node) {

        nlohmann::json ro_list = nlohmann::json::array();
        node.on_edge_ro_edges([&](IEdge &e) {
            ro_list.push_back(nlohmann::json{{"uuid", e.get_uuid()}, {"label", e.get_label()}});
        });

        nlohmann::json rw_list = nlohmann::json::array();
        node.on_edge_rw_edges([&](IEdge &e) {
            rw_list.push_back(nlohmann::json{{"uuid", e.get_uuid()}, {"label", e.get_label()}});
        });

        auto &node_ref = node; // avoid -Wpotentially-evaluated-expression on typeid

        TracerState &s = state();
        u64 seq        = s.next_seq();
        s.meta_records.push_back(
            MetaRecord{
                seq,
                nlohmann::json{
                    {"kind", "node_update"},
                    {"seq", seq},
                    {"wtime", shambase::details::get_wtime()},
                    {"uuid", node.get_uuid()},
                    {"label", node.get_label()},
                    {"type", typeid(node_ref).name()},
                    {"ro_edges", ro_list},
                    {"rw_edges", rw_list}}
                    .dump()});
    }

    /// Record a node operation (evaluate begin/end) and flush at depth-0 evaluate end
    void cb_node_op(u64 uuid, u64 op_id) {

        TracerState &s = state();

        if (op_id == static_cast<u64>(NodeTraceOp::evaluate_begin)) {
            s.eval_depth++;
            record_pod(EventKind::node_evaluate_begin, uuid);
        } else if (op_id == static_cast<u64>(NodeTraceOp::evaluate_end)) {
            record_pod(EventKind::node_evaluate_end, uuid);
            if (s.eval_depth > 0) {
                s.eval_depth--;
            }
            if (s.eval_depth == 0) {
                // safe point: not inside any node evaluation
                s.flush_events(false);
            }
        }
    }

    /// Enable tracing from the environment (SHAM_SOLVERGRAPH_TRACE=1) at static init time
    const bool init_tracing_from_env = []() {
        const char *val = std::getenv("SHAM_SOLVERGRAPH_TRACE");
        if (val != nullptr && std::string(val) == "1") {
            shamrock::solvergraph::tracing::enable();
            return true;
        }
        return false;
    }();

} // namespace

bool shamrock::solvergraph::tracing::is_enabled() { return state().enabled; }

void shamrock::solvergraph::tracing::enable() {

    TracerState &s = state();
    if (s.enabled) {
        return;
    }
    s.enabled = true;

    LifetimeTracker<INode>::on_create       = cb_node_create;
    LifetimeTracker<INode>::on_destroy      = cb_node_destroy;
    LifetimeTracker<INode>::on_state_update = cb_node_update;
    LifetimeTracker<INode>::on_op           = cb_node_op;

    LifetimeTracker<IEdge>::on_create  = cb_edge_create;
    LifetimeTracker<IEdge>::on_destroy = cb_edge_destroy;
}

void shamrock::solvergraph::tracing::disable() {

    TracerState &s = state();
    if (!s.enabled) {
        return;
    }

    LifetimeTracker<INode>::on_create       = nullptr;
    LifetimeTracker<INode>::on_destroy      = nullptr;
    LifetimeTracker<INode>::on_state_update = nullptr;
    LifetimeTracker<INode>::on_op           = nullptr;

    LifetimeTracker<IEdge>::on_create  = nullptr;
    LifetimeTracker<IEdge>::on_destroy = nullptr;

    s.flush_events(true);
    s.enabled = false;
}

void shamrock::solvergraph::tracing::set_outfile_prefix(const std::string &prefix) {

    TracerState &s   = state();
    s.outfile_prefix = prefix;

    // reopen (with a new header line) on the next flush
    s.stream.reset();
}

void shamrock::solvergraph::tracing::flush() { state().flush_events(true); }
