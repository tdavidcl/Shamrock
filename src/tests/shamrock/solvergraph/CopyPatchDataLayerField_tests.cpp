// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/CopyPatchDataLayerFields.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

NEW_TEST(Unittest, "shamrock/solvergraph/CopyPatchDataLayerFields", 1) {
    using namespace shamrock::solvergraph;
    using namespace shamrock::patch;

    // Create layouts with complex field types
    auto layout_source = std::make_shared<PatchDataLayerLayout>();
    auto layout_target = std::make_shared<PatchDataLayerLayout>();

    // Add various field types to source
    layout_source->add_field<f32>("field_1", 1);
    layout_source->add_field<f64>("field_2", 1);
    layout_source->add_field<u32>("field_3", 1);
    layout_source->add_field<u64>("field_4", 1);

    // Add same fields to target (buf one field is not in the target)
    layout_target->add_field<f32>("field_1", 1);
    layout_target->add_field<u32>("field_3", 1);
    layout_target->add_field<u64>("field_4", 1);

    // Create the copy node
    auto copy_node = std::make_shared<CopyPatchDataLayerFields>(layout_source, layout_target);

    // Create source data edge
    auto source_refs = std::make_shared<PatchDataLayerRefs>("source", "source_refs");

    // Create target data edge
    auto target_edge = std::make_shared<PatchDataLayerEdge>("target", "target_edge", layout_target);

    // Set up edges
    copy_node->set_edges(source_refs, target_edge);

    // Create mock patch data with multiple objects
    u64 seed      = 0xABCD;
    u32 obj_count = 250;

    auto source_patchdata = PatchDataLayer::mock_patchdata(seed, obj_count, layout_source);
    source_refs->get_refs().add_obj(1, std::ref(source_patchdata));

    // Execute the copy operation
    copy_node->evaluate();

    // Verify target data was copied correctly
    REQUIRE_EQUAL(target_edge->get_refs().get_element_count(), 1);
    REQUIRE_EQUAL(target_edge->get(1).get_obj_cnt(), obj_count);

    // Verify all field types were copied correctly
    auto &source_pdat = source_refs->get_const_refs().get(1).get();
    auto &target_pdat = target_edge->get(1);

    source_pdat.for_each_field_any([&](auto &source_field) {
        using T = typename std::remove_reference<decltype(source_field)>::type::Field_type;

        if (source_field.get_name() == "field_2") {
            REQUIRE_EXCEPTION_THROW(
                target_pdat.get_field<T>(source_field.get_name()), std::invalid_argument);
            return;
        }

        auto &target_field = target_pdat.get_field<T>(source_field.get_name());

        REQUIRE_EQUAL(target_field.get_obj_cnt(), source_field.get_obj_cnt());
        REQUIRE_EQUAL(target_field.get_nvar(), source_field.get_nvar());

        // Verify actual data values match
        auto source_data = source_field.get_buf().copy_to_stdvec();
        auto target_data = target_field.get_buf().copy_to_stdvec();
        REQUIRE_EQUAL_CUSTOM_COMP(source_data, target_data, sham::equals);
    });
}

NEW_TEST(Unittest, "shamrock/solvergraph/CopyPatchDataLayerFields(reused target)", 1) {
    using namespace shamrock::solvergraph;
    using namespace shamrock::patch;

    auto layout_source = std::make_shared<PatchDataLayerLayout>();
    layout_source->add_field<f32>("field_1", 1);
    layout_source->add_field<f64>("field_2", 1);
    layout_source->add_field<u32>("field_3", 3);

    auto layout_target = std::make_shared<PatchDataLayerLayout>();
    layout_target->add_field<f32>("field_1", 1);
    layout_target->add_field<u32>("field_3", 3);

    auto copy_node   = std::make_shared<CopyPatchDataLayerFields>(layout_source, layout_target);
    auto source_refs = std::make_shared<PatchDataLayerRefs>("source", "source_refs");
    auto target_edge = std::make_shared<PatchDataLayerEdge>("target", "target_edge", layout_target);
    copy_node->set_edges(source_refs, target_edge);

    auto check_copy = [&](const std::vector<u64> &ids) {
        REQUIRE_EQUAL(target_edge->get_refs().get_element_count(), ids.size());
        for (u64 id : ids) {
            auto &source_pdat = source_refs->get_const_refs().get(id).get();
            auto &target_pdat = target_edge->get(id);
            REQUIRE_EQUAL(target_pdat.get_obj_cnt(), source_pdat.get_obj_cnt());

            target_pdat.for_each_field_any([&](auto &target_field) {
                using T = typename std::remove_reference<decltype(target_field)>::type::Field_type;
                auto &source_field = source_pdat.get_field<T>(target_field.get_name());
                REQUIRE_EQUAL(target_field.get_nvar(), source_field.get_nvar());
                REQUIRE_EQUAL_CUSTOM_COMP(
                    source_field.get_buf().copy_to_stdvec(),
                    target_field.get_buf().copy_to_stdvec(),
                    sham::equals);
            });
        }
    };

    // first evaluation
    auto pdat_1 = PatchDataLayer::mock_patchdata(0x1, 250, layout_source);
    auto pdat_2 = PatchDataLayer::mock_patchdata(0x2, 30, layout_source);
    source_refs->get_refs().add_obj(1, std::ref(pdat_1));
    source_refs->get_refs().add_obj(2, std::ref(pdat_2));

    copy_node->evaluate();
    check_copy({1, 2});

    // append some objects to the target, as ghost zones would be
    target_edge->get(1).insert_elements(PatchDataLayer::mock_patchdata(0x3, 500, layout_target));
    REQUIRE_EQUAL(target_edge->get(1).get_obj_cnt(), 750);

    // second evaluation with other patches and sizes, the target is resized in place
    auto pdat_1b            = PatchDataLayer::mock_patchdata(0x4, 100, layout_source);
    auto pdat_3             = PatchDataLayer::mock_patchdata(0x5, 40, layout_source);
    source_refs->get_refs() = {};
    source_refs->get_refs().add_obj(1, std::ref(pdat_1b));
    source_refs->get_refs().add_obj(3, std::ref(pdat_3));

    copy_node->evaluate();
    check_copy({1, 3});
}
