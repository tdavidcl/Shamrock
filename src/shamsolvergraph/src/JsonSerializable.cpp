// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file JsonSerializable.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Out-of-line JSON (de)serialization for JsonSerializable.
 */

#include "shamsolvergraph/JsonSerializable.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace shamrock::solvergraph {

    void JsonSerializable::to_json(nlohmann::json &j) const {
        _impl_to_json(j);
        j["type"] = type_name();
    }

    std::unique_ptr<JsonSerializable> JsonSerializable::from_json(const nlohmann::json &j) {
        if (!j.is_object() || !j.contains("type") || !j["type"].is_string()) {
            throw std::runtime_error(
                "Invalid JSON for deserialization: expected an object with a string 'type' field.");
        }
        const std::string type = j.at("type").get<std::string>();
        return JsonSerializable_registry::instance().create(type, j);
    }

} // namespace shamrock::solvergraph
