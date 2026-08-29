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
 * @file ImplVariantRegistry.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Process-wide registry of the implementation control variables of every algorithm.
 *
 * Each algorithm supporting implementation selection owns a
 * `shamalgs::ImplVariantGlobal<Alts...>` holding its currently selected implementation. That
 * object registers itself here under the algorithm's name at construction, so that generic code
 * (the Python bindings, a benchmark driver, a config dump) can enumerate every such algorithm and
 * query or change its state through the `shamalgs::IImplVariant` base interface, without knowing
 * any of the alternative types at compile time.
 *
 * The registry holds non-owning references: the algorithms keep dispatching on their own concrete
 * selector object directly, and nothing on their hot path goes through this registry.
 *
 * Only `IImplVariant` is forward declared here, so that ImplVariant.hpp can include this header
 * (the reverse direction would be a cycle). Include shamalgs/ImplVariant.hpp to actually use the
 * references handed out by ImplVariantRegistry::get.
 */

#include <string>
#include <unordered_map>
#include <vector>

namespace shamalgs {

    class IImplVariant;

    /**
     * @brief Maps an algorithm name to that algorithm's implementation control variable.
     *
     * Entries are non-owning: `ImplVariantGlobal` adds itself on construction and removes itself
     * on destruction, so a registered reference is always valid while it is reachable here.
     */
    class ImplVariantRegistry {
        public:
        /**
         * @brief Register an algorithm's implementation control variable
         *
         * @param key the algorithm name (e.g. "is_all_true")
         * @param control the control variable, which must outlive its registration
         * @throws std::invalid_argument if the key is empty or already registered
         */
        void register_control(const std::string &key, IImplVariant &control);

        /**
         * @brief Drop an algorithm's registration, no-op if it is not registered
         *
         * @param key the algorithm name
         */
        void unregister_control(const std::string &key);

        /**
         * @brief Whether an algorithm is registered under this name
         *
         * @param key the algorithm name
         */
        bool has(const std::string &key) const;

        /**
         * @brief Get an algorithm's implementation control variable
         *
         * @param key the algorithm name
         * @return a reference to the registered control variable
         * @throws std::invalid_argument if no algorithm is registered under this name
         */
        IImplVariant &get(const std::string &key) const;

        /// List the names of every registered algorithm, sorted
        std::vector<std::string> get_key_list() const;

        private:
        /// The registered control variables, keyed by algorithm name
        std::unordered_map<std::string, IImplVariant *> controls;
    };

    /**
     * @brief Access the process-wide implementation control variable registry
     *
     * The registry is created on first use and never destroyed, so that control variables with
     * static storage duration can safely unregister themselves however late they are destroyed.
     */
    ImplVariantRegistry &get_impl_variant_registry();

} // namespace shamalgs
