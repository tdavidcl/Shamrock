// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file PatchDataLayout.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include <unordered_map>
#include <memory>

namespace shamrock::patch {

    /**
     * @brief PatchDataLayerLayout container class
     * This class store the layout of the layers
     */
    class PatchDataLayout {
        public:
        std::unordered_map<std::string, std::shared_ptr<shamrock::patch::PatchDataLayerLayout>>
            layer_layouts;

        PatchDataLayout() = default;

        size_t get_layer_count() const { return layer_layouts.size(); }

        inline std::shared_ptr<PatchDataLayerLayout> &get_layer_ptr(const std::string &name) {
            return layer_layouts.at(name);
        }

        inline const std::shared_ptr<PatchDataLayerLayout> &get_layer_ptr(
            const std::string &name) const {
            return layer_layouts.at(name);
        }

        inline PatchDataLayerLayout &get_layer_ref(const std::string &name) {
            return shambase::get_check_ref(layer_layouts.at(name));
        }
    };

    /**
     * @brief Serialize a PatchDataLayout object to a JSON object
     *
     * This function takes a PatchDataLayout object and serializes it to a JSON object.
     * It is used to convert the PatchDataLayout object to a JSON string.
     *
     * @param j The JSON object to serialize the PatchDataLayout object to
     * @param p The PatchDataLayout object to serialize
     */
    inline void to_json(nlohmann::json &j, const PatchDataLayout &p) {
        using json = nlohmann::json;
        json layer_entries;

        for (const auto &[name, layer_ptr] : p.layer_layouts) {
            if (layer_ptr) {
                json layer_json;
                to_json(layer_json, *layer_ptr);
                layer_entries[name] = layer_json;
            }
        }

        j = layer_entries;
    }

    /**
     * @brief Deserialize a PatchDataLayout object from a JSON object
     *
     * This function takes a JSON object and deserializes it to a PatchDataLayout object.
     * It is used to convert a JSON string to a PatchDataLayout object.
     *
     * @param j The JSON object to deserialize the PatchDataLayout object from
     * @param p The PatchDataLayout object to deserialize
     */
    inline void from_json(const nlohmann::json &j, PatchDataLayout &p) {
        p.layer_layouts.clear();

        for (const auto &[name, layer_json] : j.items()) {
            auto layer_ptr = std::make_shared<PatchDataLayerLayout>();
            from_json(layer_json, *layer_ptr);
            p.layer_layouts.emplace(name, std::move(layer_ptr));
        }
    }

} // namespace shamrock::patch
