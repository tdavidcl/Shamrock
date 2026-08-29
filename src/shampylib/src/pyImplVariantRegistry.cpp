// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyImplVariantRegistry.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Python bindings of the implementation control variable registry
 *
 * Exposes shamrock.impl, the by-name counterpart of the per-algorithm
 * get_current_impl_<algo> / set_impl_<algo> / ... functions: same state, but reachable without
 * knowing the algorithm at compile time, so a script can enumerate every implementation
 * selectable algorithm of the process at once. Algorithms of every library register here, hence
 * a top level submodule rather than one under shamrock.algs.
 */

#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/ImplVariantRegistry.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shamsys/NodeInstance.hpp"

ON_PYTHON_INIT {
    auto &m = root_module;

    py::module impl_module = m.def_submodule(
        "impl", "implementation selection of every algorithm supporting it, by algorithm name");

    impl_module.def(
        "list_keys",
        []() {
            return shamalgs::get_impl_variant_registry().get_key_list();
        },
        R"(Name of every algorithm supporting implementation selection, sorted.

Only the algorithms whose translation unit is linked into the running binary are listed.)");

    impl_module.def(
        "is_set",
        [](const std::string &key) {
            return shamalgs::get_impl_variant_registry().get(key).is_set();
        },
        py::arg("key"),
        "Whether an implementation has been selected yet for this algorithm");

    impl_module.def(
        "get_current",
        [](const std::string &key) {
            return shamalgs::get_impl_variant_registry().get(key).get_current_config();
        },
        py::arg("key"),
        R"(Implementation currently selected for this algorithm, as a config json string.

Returns "null" if none has been selected yet, see is_set and autoselect.)");

    impl_module.def(
        "get_default_list",
        [](const std::string &key) {
            return shamalgs::get_impl_variant_registry().get(key).get_default_config_list();
        },
        py::arg("key"),
        "Implementations available for this algorithm, as config json strings");

    impl_module.def(
        "set",
        [](const std::string &key, const std::string &config_json) {
            shamalgs::get_impl_variant_registry().get(key).set(config_json);
        },
        py::arg("key"),
        py::arg("config_json"),
        R"(Select an implementation for this algorithm.

config_json is one of the strings returned by get_default_list, of the form
{"implementation": "<name>", "parameters": {...}}.)");

    impl_module.def(
        "autoselect",
        [](const std::string &key) {
            shamalgs::get_impl_variant_registry().get(key).autoselect(
                shamsys::instance::get_compute_scheduler_ptr());
        },
        py::arg("key"),
        "Select this algorithm's own default implementation, on the current compute scheduler");
}
