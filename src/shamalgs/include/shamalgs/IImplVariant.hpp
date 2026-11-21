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
 * @file IImplVariant.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Type-erased interface of the implementation selectors (see ImplVariant.hpp)
 *
 * Kept separate from ImplVariant.hpp so that code only handling selectors through this
 * interface (e.g. impl_registry.hpp) does not pull <nlohmann/json.hpp> in.
 */

#include "shambackends/DeviceScheduler.hpp"
#include <string_view>
#include <string>
#include <vector>

namespace shamalgs {

    /**
     * @brief Non-template virtual interface exposed by ImplVariantGlobal, for code that needs
     * to hold or pass around an implementation selector without knowing its alternative types.
     */
    class IImplVariant {
        public:
        virtual ~IImplVariant() = default;

        /// Get the currently selected implementation as a single config json string, or a json
        /// null if no implementation has been selected yet
        virtual std::string get_current_config() const = 0;

        /// List the available implementations as config json strings, one per alternative
        virtual std::vector<std::string> get_default_config_list() const = 0;

        /// Select an implementation from a {"implementation": ..., "parameters": ...} json string
        virtual void set(std::string_view config_json) = 0;

        /// Whether an implementation has been selected yet
        virtual bool is_set() const = 0;

        /// Select the algorithm's default implementation for the device behind `sched`
        virtual void autoselect(const sham::DeviceScheduler_ptr &sched) = 0;
    };

} // namespace shamalgs
