// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/ImplVariantRegistry.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    /// Whether the registry lists this algorithm name
    bool key_is_listed(const std::string &key) {
        auto keys = shamalgs::get_impl_variant_registry().get_key_list();
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    }

    /// An alternative of the throwaway selectors built by the tests below
    struct DummyAlt {
        static constexpr std::string_view variant_type_name = "dummy_alt";
    };

} // namespace

NEW_TEST(Unittest, "shamalgs/ImplVariantRegistry:registered_algorithms", 1) {

    // Every algorithm linked into shamrock_test registers itself at static init
    for (const std::string &key :
         {"clbvh_dual_tree_traversal",
          "compute_histogram",
          "is_all_true",
          "reduction",
          "scan_exclusive_sum_in_place",
          "segmented_sort_in_place",
          "sort_by_key_pow2_len",
          "sort_by_keys"}) {

        REQUIRE_NAMED(key, shamalgs::get_impl_variant_registry().has(key));
        REQUIRE_NAMED(key, key_is_listed(key));
        REQUIRE_EQUAL_NAMED(key, shamalgs::get_impl_variant_registry().get(key).get_key(), key);
    }

    // get_key_list is sorted
    auto keys = shamalgs::get_impl_variant_registry().get_key_list();
    REQUIRE(std::is_sorted(keys.begin(), keys.end()));
}

NEW_TEST(Unittest, "shamalgs/ImplVariantRegistry:unknown_key", 1) {

    REQUIRE(!shamalgs::get_impl_variant_registry().has("this_algorithm_does_not_exist"));

    REQUIRE_EXCEPTION_THROW(
        shamalgs::get_impl_variant_registry().get("this_algorithm_does_not_exist"),
        std::invalid_argument);
}

NEW_TEST(Unittest, "shamalgs/ImplVariantRegistry:same_state_as_the_algorithm", 1) {

    using namespace shamalgs::primitives;

    // The registry hands out the very object is_all_true dispatches on, so the two views of the
    // state must agree in both directions
    shamalgs::IImplVariant &control = shamalgs::get_impl_variant_registry().get("is_all_true");

    std::string restore = impl::get_current_impl_is_all_true();
    bool was_set        = impl::is_impl_set_is_all_true();

    REQUIRE_EQUAL(control.get_default_config_list(), impl::get_default_impl_list_is_all_true());
    REQUIRE_EQUAL(control.get_current_config(), restore);
    REQUIRE_EQUAL(control.is_set(), was_set);

    auto impl_list = control.get_default_config_list();
    REQUIRE(!impl_list.empty());

    for (const std::string &config : impl_list) {
        // set through the registry, observe through the algorithm's own accessor
        control.set(config);
        REQUIRE(impl::is_impl_set_is_all_true());
        REQUIRE_EQUAL(impl::get_current_impl_is_all_true(), config);

        // and the other way around
        impl::set_impl_is_all_true(config);
        REQUIRE_EQUAL(control.get_current_config(), config);
    }

    // autoselect through the registry yields the algorithm's own default
    control.autoselect(shamsys::instance::get_compute_scheduler_ptr());
    REQUIRE(control.is_set());

    std::string from_registry = control.get_current_config();
    impl::autoselect_impl_is_all_true(shamsys::instance::get_compute_scheduler_ptr());
    REQUIRE_EQUAL(impl::get_current_impl_is_all_true(), from_registry);

    if (was_set) {
        impl::set_impl_is_all_true(restore);
    }
}

NEW_TEST(Unittest, "shamalgs/ImplVariantRegistry:registration_lifetime", 1) {

    const std::string key = "impl_variant_registry_test_dummy";

    REQUIRE(!shamalgs::get_impl_variant_registry().has(key));

    {
        shamalgs::ImplVariantGlobal<DummyAlt> dummy{key, [](const sham::DeviceScheduler_ptr &) {
                                                       return DummyAlt{};
                                                   }};

        // registered on construction, and reachable through the base interface
        REQUIRE(shamalgs::get_impl_variant_registry().has(key));
        REQUIRE(key_is_listed(key));

        shamalgs::IImplVariant &control = shamalgs::get_impl_variant_registry().get(key);
        REQUIRE(&control == &dummy);
        REQUIRE(!control.is_set());

        // autoselecting through the base interface runs the default provider given at
        // construction, DummyAlt being the only alternative here
        control.autoselect(shamsys::instance::get_compute_scheduler_ptr());
        REQUIRE(dummy.is_set());
        REQUIRE_EQUAL(dummy.get_current_config(), dummy.get_default_config_list().at(0));

        // a second selector can not steal an already registered name
        REQUIRE_EXCEPTION_THROW(
            (shamalgs::ImplVariantGlobal<DummyAlt>{key,
                                                   [](const sham::DeviceScheduler_ptr &) {
                                                       return DummyAlt{};
                                                   }}),
            std::invalid_argument);
    }

    // unregistered on destruction
    REQUIRE(!shamalgs::get_impl_variant_registry().has(key));
    REQUIRE(!key_is_listed(key));
}

NEW_TEST(Unittest, "shamalgs/ImplVariantRegistry:invalid_registration", 1) {

    // an empty name is refused
    REQUIRE_EXCEPTION_THROW(
        (shamalgs::ImplVariantGlobal<DummyAlt>{"",
                                               [](const sham::DeviceScheduler_ptr &) {
                                                   return DummyAlt{};
                                               }}),
        std::invalid_argument);

    // so is a missing default provider
    REQUIRE_EXCEPTION_THROW(
        (shamalgs::ImplVariantGlobal<DummyAlt>{
            "impl_variant_registry_test_no_provider",
            shamalgs::ImplVariantGlobal<DummyAlt>::DefaultProvider{}}),
        std::invalid_argument);
}
