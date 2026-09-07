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
 * @file tracing.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solvergraph execution tracing to per-rank JSON lines trace files
 *
 * When enabled, the tracer installs LifetimeTracker callbacks for INode and IEdge and records
 * node/edge creations & destructions, node topology updates (edge bindings) and node evaluations
 * to an append-only JSON lines file (one file per MPI rank).
 *
 * Hot path events (create/destroy/evaluate) are recorded as fixed-size POD structs pushed to a
 * preallocated buffer, with no allocation, formatting or I/O. Cold path records (topology
 * updates, which only fire at graph build/rewire time) capture labels and edge lists as
 * pre-serialized JSON lines. Everything is formatted and written at flush safe points (end of a
 * depth-0 evaluation, buffer overflow, disable, program exit) on the calling thread.
 *
 * When disabled (the default), the only cost at each hook site is a null pointer check.
 *
 * Tracing can be enabled with the environment variable SHAM_SOLVERGRAPH_TRACE=1, the output file
 * prefix can be set with SHAM_SOLVERGRAPH_TRACE_PREFIX (default "shamrock_sgtrace"), or
 * programmatically through the functions below.
 *
 * The resulting trace files can be visualized with the viewer app in tools/solvergraph_viewer.
 */

#include <string>

namespace shamrock::solvergraph::tracing {

    /// Is solvergraph tracing currently enabled ?
    bool is_enabled();

    /**
     * @brief Enable solvergraph tracing
     *
     * Installs the LifetimeTracker callbacks for INode and IEdge. Objects created after this
     * point will be tracked.
     */
    void enable();

    /**
     * @brief Disable solvergraph tracing
     *
     * Flushes pending events to the trace file and uninstalls the LifetimeTracker callbacks.
     */
    void disable();

    /**
     * @brief Set the output file prefix
     *
     * The trace of rank r is written to "<prefix>_<r>.jsonl". If a trace file was already
     * opened it is closed, and the next flush reopens a new file (with a new header line) using
     * the new prefix.
     *
     * @param prefix The output file prefix
     */
    void set_outfile_prefix(const std::string &prefix);

    /**
     * @brief Force a flush of the buffered events to the trace file
     *
     * If the MPI world rank is not known yet (world info not fetched), rank 0 is assumed for the
     * trace file name.
     */
    void flush();

} // namespace shamrock::solvergraph::tracing
