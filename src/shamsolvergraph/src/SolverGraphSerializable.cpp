// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SolverGraphSerializable.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief JSON (de)serialization of JsonSerializable edges and of SolverGraphSerializable
 *
 * Defined out of line as those functions instantiate a sizeable part of nlohmann_json which is
 * costly to compile in every translation unit including the headers.
 */

#include "shamsolvergraph/SolverGraphSerializable.hpp"
#include "shamsolvergraph/JsonSerializable.hpp"

namespace shamrock::solvergraph {

    std::unique_ptr<JsonSerializable> JsonSerializable::from_json(const nlohmann::json &j) {
        if (!j.is_object() || !j.contains("type") || !j["type"].is_string()) {
            throw std::runtime_error(
                "Invalid JSON for deserialization: expected an object with a string 'type' field.");
        }
        const std::string type = j.at("type").get<std::string>();
        return JsonSerializable_registry::instance().create(type, j);
    }

    void to_json(nlohmann::json &j, const SolverGraphSerializable &p) {
        nlohmann::json edges = nlohmann::json::object();

        for (const std::string &name : p.get_edge_names()) {
            const auto &edge_ptr = p.get_edge_ptr_base(name);

            // we use raw pointer to avoir the cost of creating a new shared pointer
            auto *serializable = dynamic_cast<const JsonSerializable *>(edge_ptr.get());
            if (serializable == nullptr) {
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "Edge '{}' is registered in SolverGraphSerializable but is not "
                    "JsonSerializable",
                    name));
            }

            nlohmann::json edge_j;
            serializable->to_json(edge_j);
            edges[name] = std::move(edge_j);
        }

        j = nlohmann::json{{"edges", std::move(edges)}};
    }

    void from_json(const nlohmann::json &j, SolverGraphSerializable &p) {
        if (!j.is_object() || !j.contains("edges") || !j.at("edges").is_object()) {
            shambase::throw_with_loc<std::invalid_argument>(
                "Invalid JSON for SolverGraphSerializable: expected an object with an 'edges' "
                "object");
        }

        SolverGraphSerializable tmp{};
        const auto &edges = j.at("edges");

        for (const auto &[name, edge_json] : edges.items()) {
            std::shared_ptr<JsonSerializable> serializable = JsonSerializable::from_json(edge_json);
            auto edge = std::dynamic_pointer_cast<IEdge>(serializable);
            if (!bool(edge)) {
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "Deserialized type for edge '{}' does not inherit from IEdge", name));
            }

            tmp.register_edge_ptr_base(name, std::move(edge));
        }

        p = std::move(tmp);
    }

} // namespace shamrock::solvergraph
