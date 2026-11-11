// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shammodels/ramses/modules/AddExtForceVerticalDiscPotential.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

using Tvec  = f64_3;
using Tscal = f64;

NEW_TEST(Unittest, "shammodels/ramses/modules/AddExtForceVerticalDiscPotential", 1) {

    const Tscal G = 2, M = 3, R0 = 1.5, z0 = 0.25;

    // 2 cells per block, 2 patches
    auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
    sizes->indexes.add_obj(0, 2);
    sizes->indexes.add_obj(1, 1);

    std::vector<std::vector<Tvec>> coords_v
        = {{{0, 0, 0.25}, {1, 2, 1.0}, {0, 0, -3}, {5, 5, 0.3}}, {{0, 1, 10}, {2, 0, -0.5}}};
    std::vector<std::vector<Tvec>> a_ext_v
        = {{{1, 2, 3}, {0, 0, 0}, {-1, 0, 1}, {0, 0, 0}}, {{0, 0, 0}, {4, 5, 6}}};

    auto coords = std::make_shared<shamrock::solvergraph::Field<Tvec>>(2, "", "");
    auto a_ext  = std::make_shared<shamrock::solvergraph::Field<Tvec>>(2, "", "");
    coords->ensure_sizes(sizes->indexes);
    a_ext->ensure_sizes(sizes->indexes);
    for (u64 id : {0, 1}) {
        coords->get_buf(id).copy_from_stdvec(coords_v[id]);
        a_ext->get_buf(id).copy_from_stdvec(a_ext_v[id]);
    }

    auto constant_G   = std::make_shared<shamrock::solvergraph::ScalarEdge<Tscal>>("G", "G");
    constant_G->value = G;

    shammodels::basegodunov::modules::AddExtForceVerticalDiscPotential<Tvec> node{2, M, R0, z0};
    node.set_edges(sizes, coords, constant_G, a_ext);
    node.evaluate();

    for (u64 id : {0, 1}) {
        std::vector<Tvec> res = a_ext->get_buf(id).copy_to_stdvec();
        REQUIRE_EQUAL(res.size(), a_ext_v[id].size());
        for (u32 i = 0; i < res.size(); i++) {
            Tscal z      = coords_v[id][i][2] - z0;
            Tscal expect = a_ext_v[id][i][2] - G * M * z / (R0 * R0 * std::sqrt(R0 * R0 + z * z));
            REQUIRE_EQUAL(res[i][0], a_ext_v[id][i][0]);
            REQUIRE_EQUAL(res[i][1], a_ext_v[id][i][1]);
            REQUIRE_FLOAT_EQUAL(res[i][2], expect, 1e-14);
        }
    }
}
