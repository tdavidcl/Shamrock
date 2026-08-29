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
 * @file ImplVariant.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Generic std::variant-based implementation selector.
 *
 * Lets an algorithm expose its set of possible implementations as a plain
 * `std::variant<Alt1, Alt2, ...>` of small structs (mirroring a Rust enum,
 * variants with fields included) instead of hand-writing an enum plus the
 * name <-> enum switch/if-else chains that come with it. Each alternative
 * only needs a `static constexpr std::string_view variant_type_name`
 * (the same convention already used by the std::variant-based configs in
 * shammodels, e.g. AVConfig.hpp).
 *
 * The free functions variant_to_config_string / variant_from_config_string /
 * variant_default_type_names turn any such variant into a single config
 * string ABI: `{"implementation": "<name>", "parameters": <alternative's own json>}`.
 * ImplVariantGlobal builds on top of them to replace the hand-rolled
 * "global variable + enum + name mapping + 3 free functions" pattern used to
 * hold an algorithm's currently selected implementation; see its own doc
 * comment for the two ABI flavors it exposes and how the unset state works.
 *
 * Each ImplVariantGlobal registers itself under its algorithm's name in the
 * process-wide ImplVariantRegistry (ImplVariantRegistry.hpp), which hands it
 * back as the non-templated IImplVariant declared below. That is what lets
 * generic code query or configure any algorithm by name; algorithms
 * themselves keep dispatching on their own concrete selector, so the registry
 * never sits on their hot path.
 *
 * Dispatch on the selected implementation is then a plain
 * `std::visit(shambase::overloaded{...}, variant)` instead of a switch.
 */

#include "shambase/exception.hpp"
#include "sham/format/format.hpp"
#include "shamalgs/ImplVariantRegistry.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shamcomm/logs.hpp"
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <string_view>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace shamalgs {

    /**
     * @brief Customization point controlling how an alternative's fields (if any) are
     * serialized to / parsed from the "parameters" json value carried alongside its name.
     *
     * The default (used by empty/tag alternatives, i.e. most of them) serializes to an
     * empty json object. Alternatives with tunable fields (e.g. a group size) specialize
     * this trait, for example:
     *
     * @code{.cpp}
     * struct GpuTeamFetching {
     *     static constexpr std::string_view variant_type_name = "gpu_team_fetching";
     *     u32 group_size = 128;
     * };
     *
     * template<>
     * struct shamalgs::ImplVariantParams<GpuTeamFetching> {
     *     static nlohmann::json to_json(const GpuTeamFetching &p) {
     *         return {{"group_size", p.group_size}};
     *     }
     *     static GpuTeamFetching from_json(const nlohmann::json &j) {
     *         GpuTeamFetching p{};
     *         if (j.contains("group_size")) {
     *             p.group_size = j.at("group_size").get<u32>();
     *         }
     *         return p;
     *     }
     * };
     * @endcode
     */
    template<class Alt>
    struct ImplVariantParams {
        /// Serialize the alternative's fields (default: no fields, empty object)
        static inline nlohmann::json to_json(const Alt &) { return nlohmann::json::object(); }
        /// Parse the alternative's fields back (default: no fields, ignored)
        static inline Alt from_json(const nlohmann::json &) { return Alt{}; }
    };

    namespace details {
        /// Partial specialization target: extracts the Alts... pack out of std::variant<Alts...>
        template<class Variant>
        struct impl_variant_alts;

        template<class... Alts>
        struct impl_variant_alts<std::variant<Alts...>> {
            static inline std::vector<std::string> default_type_names() {
                return {std::string(Alts::variant_type_name)...};
            }

            static inline std::variant<Alts...> from_config_string(std::string_view s) {
                nlohmann::json j      = nlohmann::json::parse(s);
                std::string name      = j.at("implementation").get<std::string>();
                nlohmann::json params = j.value("parameters", nlohmann::json::object());

                std::optional<std::variant<Alts...>> result;
                (void) ((name == Alts::variant_type_name
                             ? (result
                                = std::variant<Alts...>{ImplVariantParams<Alts>::from_json(params)},
                                true)
                             : false)
                        || ...);

                if (!result) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                        "invalid implementation : {}, possible implementations : {}",
                        name,
                        default_type_names()));
                }
                return *result;
            }
        };
    } // namespace details

    /// Serialize the currently active alternative of a variant to a single config json string
    template<class Variant>
    inline std::string variant_to_config_string(const Variant &v) {
        return std::visit(
            [](const auto &alt) {
                using Alt = std::decay_t<decltype(alt)>;
                nlohmann::json j;
                j["implementation"] = std::string(Alt::variant_type_name);
                j["parameters"]     = ImplVariantParams<Alt>::to_json(alt);
                return j.dump();
            },
            v);
    }

    /// Parse a variant back from a config string produced by variant_to_config_string
    template<class Variant>
    inline Variant variant_from_config_string(std::string_view s) {
        return details::impl_variant_alts<Variant>::from_config_string(s);
    }

    /// List the variant_type_name of every alternative of a variant, with no params
    template<class Variant>
    inline std::vector<std::string> variant_default_type_names() {
        return details::impl_variant_alts<Variant>::default_type_names();
    }

    /**
     * @brief Non-template virtual interface exposed by ImplVariantGlobal, for code that needs
     * to hold or pass around an implementation selector without knowing its alternative types.
     *
     * This is what ImplVariantRegistry hands out, so that generic code can enumerate every
     * implementation-selectable algorithm and query or change its state by name. Algorithms
     * themselves keep dispatching on their own concrete ImplVariantGlobal object, so none of
     * these virtual calls sit on an algorithm's hot path.
     */
    class IImplVariant {
        public:
        virtual ~IImplVariant() = default;

        /// The name this selector is registered under (the algorithm's name)
        virtual const std::string &get_key() const = 0;

        /// Whether an implementation has been selected yet
        virtual bool is_set() const = 0;

        /// Get the currently selected implementation as a single config json string, or a json
        /// null if no implementation has been selected yet
        virtual std::string get_current_config() const = 0;

        /// List the available implementations as config json strings, one per alternative
        virtual std::vector<std::string> get_default_config_list() const = 0;

        /// Select an implementation from a {"implementation": ..., "parameters": ...} json string
        virtual void set(std::string_view config_json) = 0;

        /**
         * @brief Select the algorithm's own default implementation
         *
         * @param sched the device scheduler the algorithm will run on, for the algorithms whose
         * default depends on the device (e.g. compute_histogram). Algorithms whose default does
         * not depend on it simply ignore it.
         */
        virtual void autoselect(const sham::DeviceScheduler_ptr &sched) = 0;
    };

    /**
     * @brief Drop-in replacement for the hand-rolled "global variable + enum + name mapping
     * + 3 free functions" implementation-selection pattern.
     *
     * Holds the currently selected implementation as a std::variant<Alts...> and exposes it
     * through a single config json string ABI, so that an algorithm's
     * get_default_impl_list_X / get_current_impl_X / set_impl_X free functions become
     * one-liners:
     *   - get_current_config() / get_default_config_list() / set(string_view) : a single
     *     `{"implementation": ..., "parameters": ...}` json string ABI.
     *
     * Construction takes the algorithm's name and a provider returning the algorithm's default
     * implementation. The name registers the selector in the process-wide
     * ImplVariantRegistry (see ImplVariantRegistry.hpp), so that generic code can reach it
     * through IImplVariant without knowing Alts...; the provider backs autoselect(), so that
     * "reset this algorithm to its default" also works generically. It takes the device
     * scheduler the algorithm runs on, for the algorithms whose default depends on the device
     * (compute_histogram); the others ignore it.
     *
     * No implementation is selected at construction: this class has no notion of a default
     * until autoselect() or set() picks one. is_set() reports whether one has been picked yet.
     * It is up to each call site to decide what to do when unset - typically checking is_set()
     * and autoselecting right before dispatching (see e.g. segmented_sort_in_place.cpp). get()
     * assumes is_set(); get_current_config() is the one exception and safely returns a json null
     * instead of dereferencing an unset selection.
     *
     * @tparam Alts the alternative types, each requiring a
     * `static constexpr std::string_view variant_type_name`
     */
    template<class... Alts>
    class ImplVariantGlobal : public IImplVariant {
        public:
        using Variant = std::variant<Alts...>;

        /// Returns the algorithm's default implementation, given the device it will run on
        using DefaultProvider = std::function<Variant(const sham::DeviceScheduler_ptr &)>;

        /**
         * @brief Build a selector and register it under the algorithm's name
         *
         * @param key the algorithm's name, e.g. "is_all_true"
         * @param default_provider returns the algorithm's default implementation
         * @throws std::invalid_argument if default_provider is empty, or if another selector is
         * already registered under this name
         */
        inline ImplVariantGlobal(std::string key, DefaultProvider default_provider)
            : key(std::move(key)), default_provider(std::move(default_provider)) {

            if (!this->default_provider) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "the implementation selector of {} needs a default provider", this->key));
            }

            get_impl_variant_registry().register_control(this->key, *this);
        }

        /// Unregister the selector
        inline ~ImplVariantGlobal() override {
            get_impl_variant_registry().unregister_control(key);
        }

        // The registry holds a reference to this object, so it can not be relocated
        ImplVariantGlobal(const ImplVariantGlobal &)            = delete;
        ImplVariantGlobal &operator=(const ImplVariantGlobal &) = delete;

        /// The algorithm's name, which this selector is registered under
        inline const std::string &get_key() const override { return key; }

        /// Whether an implementation has been selected yet
        inline bool is_set() const override { return current.has_value(); }

        /// Get the currently selected implementation. Requires is_set()
        inline const Variant &get() const { return *current; }

        /// Get the currently selected implementation as a single config json string, or a json
        /// null if no implementation has been selected yet (see is_set())
        inline std::string get_current_config() const override {
            if (!is_set()) {
                return nlohmann::json(nullptr).dump();
            }
            return variant_to_config_string(get());
        }

        /// List the available implementations as config json strings, one per alternative
        inline std::vector<std::string> get_default_config_list() const override {
            return {variant_to_config_string<Variant>(Variant{Alts{}})...};
        }

        /// Directly select an alternative (e.g. to seed a default at the call site)
        inline void set(Variant v) {
            current = std::move(v);
            shamlog_info_ln("impl", "setting", key, "implementation to :", get_current_config());
        }

        /// Select an implementation from a {"implementation": ..., "parameters": ...} json string
        inline void set(std::string_view config_json) override {
            set(variant_from_config_string<Variant>(config_json));
        }

        /// Select the algorithm's own default implementation, on the given device scheduler
        inline void autoselect(const sham::DeviceScheduler_ptr &sched) override {
            set(default_provider(sched));
        }

        private:
        /// The algorithm's name, which this selector is registered under
        std::string key;
        /// Returns the algorithm's default implementation, backs autoselect()
        DefaultProvider default_provider;
        /// The currently selected implementation, empty until one is picked
        std::optional<Variant> current;
    };

} // namespace shamalgs
