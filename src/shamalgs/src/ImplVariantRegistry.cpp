// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ImplVariantRegistry.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements the process-wide implementation control variable registry
 */

#include "shamalgs/ImplVariantRegistry.hpp"
#include "shambase/exception.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "sham/format/format.hpp"
#include <fmt/ranges.h>
#include <algorithm>
#include <stdexcept>

namespace shamalgs {

    void ImplVariantRegistry::register_control(const std::string &key, IImplVariant &control) {
        if (key.empty()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "an implementation control variable can not be registered under an empty name");
        }

        auto [it, inserted] = controls.try_emplace(key, &control);
        if (!inserted) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "an implementation control variable is already registered under the name : {}",
                key));
        }
    }

    void ImplVariantRegistry::unregister_control(const std::string &key) { controls.erase(key); }

    bool ImplVariantRegistry::has(const std::string &key) const {
        return controls.find(key) != controls.end();
    }

    IImplVariant &ImplVariantRegistry::get(const std::string &key) const {
        auto it = controls.find(key);
        if (it == controls.end()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "no implementation control variable is registered under the name : {}, registered "
                "names : {}",
                key,
                get_key_list()));
        }
        return *it->second;
    }

    std::vector<std::string> ImplVariantRegistry::get_key_list() const {
        std::vector<std::string> keys;
        keys.reserve(controls.size());
        for (const auto &[key, control] : controls) {
            keys.push_back(key);
        }
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    ImplVariantRegistry &get_impl_variant_registry() {
        // Deliberately leaked: control variables with static storage duration unregister
        // themselves from their destructor, which may run after any static registry would have
        // been destroyed.
        static ImplVariantRegistry *registry = new ImplVariantRegistry{};
        return *registry;
    }

} // namespace shamalgs
