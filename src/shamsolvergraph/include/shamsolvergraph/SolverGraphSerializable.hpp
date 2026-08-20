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
 * @file SolverGraphSerializable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declare a class to register and retrieve nodes and edges from a unique container.
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json_fwd.hpp"
#include "shamsolvergraph/JsonSerializable.hpp"
#include "shamsolvergraph/SolverGraph.hpp"

namespace shamrock::solvergraph {

    class SolverGraphSerializable : public SolverGraph {
        public:
        SolverGraphSerializable()
            : SolverGraph(
                  SolverGraphConstraint{
                      .name       = "SolverGraphSerializable",
                      .node_check = [](const std::shared_ptr<INode> &) -> bool {
                          return false; // there is no clean mechanism to serialize a node + its
                                        // connexions
                      },
                      .edge_check = [](const std::shared_ptr<IEdge> &edge) -> bool {
                          return dynamic_cast<const JsonSerializable *>(edge.get()) != nullptr;
                      }}) {}

        ~SolverGraphSerializable() = default;
    };

    /**
     * @brief Serialize a SolverGraphSerializable to JSON.
     *
     * Writes an `"edges"` object keyed by edge name. Each edge value includes the
     * polymorphic `"type"` discriminator and fields from
     * @ref JsonSerializable::to_json.
     */
    void to_json(nlohmann::json &j, const SolverGraphSerializable &p);

    /**
     * @brief Deserialize a SolverGraphSerializable from JSON.
     *
     * Expects an `"edges"` object. Each entry is reconstructed via
     * @ref JsonSerializable::from_json and registered under its key. Edge types must
     * already be registered in @ref JsonSerializable_registry.
     *
     * Population is all-or-nothing: edges are first registered into a temporary
     * graph, and `p` is only replaced after every edge has been deserialized
     * successfully. On any exception, `p` is left unchanged.
     */
    void from_json(const nlohmann::json &j, SolverGraphSerializable &p);
} // namespace shamrock::solvergraph
