// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shammodels/ramses/modules/NodeComputeDragAlphas.hpp"
#include "shamphys/Dust.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

using Tvec  = f64_3;
using Tscal = f64;
using Node  = shammodels::basegodunov::modules::NodeComputeDragAlphas<Tvec>;

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeComputeDragAlphas", 1) {

    const u32 block_size = 2;
    const u32 ndust      = 2;

    // 2 patches of 2 and 1 blocks, the gas fields have an extra (ghost) block that is ignored
    auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
    sizes->indexes.add_obj(0, 2);
    sizes->indexes.add_obj(1, 1);

    shambase::DistributedData<u32> sizes_with_ghost;
    sizes_with_ghost.add_obj(0, 3);
    sizes_with_ghost.add_obj(1, 2);

    std::vector<std::vector<Tscal>> rho_v   = {{1, 2, 0.5, 4, 9, 9}, {3, 0.25, 9, 9}};
    std::vector<std::vector<Tscal>> press_v = {{1, 0.5, 2, 1, 9, 9}, {0.1, 3, 9, 9}};

    auto rho   = std::make_shared<shamrock::solvergraph::Field<Tscal>>(block_size, "", "");
    auto press = std::make_shared<shamrock::solvergraph::Field<Tscal>>(block_size, "", "");
    rho->ensure_sizes(sizes_with_ghost);
    press->ensure_sizes(sizes_with_ghost);
    for (u64 id : {0, 1}) {
        rho->get_buf(id).copy_from_stdvec(rho_v[id]);
        press->get_buf(id).copy_from_stdvec(press_v[id]);
    }

    auto alphas = std::make_shared<shamrock::solvergraph::Field<Tscal>>(ndust * block_size, "", "");

    { // Epstein: alpha_j = 1 / t_j with the gas density and adiabatic sound speed of the cell
        const Tscal gamma                  = 1.4;
        const std::vector<Tscal> sizes_g   = {1e-3, 5e-2};
        const std::vector<Tscal> densities = {2.3, 1.1};

        Node node{block_size, ndust, Node::Epstein{gamma, sizes_g, densities}};
        node.set_edges(sizes, rho, press, alphas);
        node.evaluate();

        for (u64 id : {0, 1}) {
            u32 ncell              = sizes->indexes.get(id) * block_size;
            std::vector<Tscal> res = alphas->get_buf(id).copy_to_stdvec();
            REQUIRE_EQUAL(res.size(), u64(ncell * ndust));
            for (u32 i = 0; i < ncell; i++) {
                Tscal cs = sycl::sqrt(gamma * press_v[id][i] / rho_v[id][i]);
                for (u32 j = 0; j < ndust; j++) {
                    Tscal expect = 1
                                   / shamphys::epstein_stopping_time(
                                       densities[j], sizes_g[j], rho_v[id][i], cs, gamma);
                    REQUIRE_FLOAT_EQUAL(res[i * ndust + j], expect, 1e-12 * expect);
                }
            }
        }
    }

    { // static: the same alphas in every cell
        Node node{block_size, ndust, Node::Static{{3., 7.}}};
        node.set_edges(sizes, rho, press, alphas);
        node.evaluate();

        for (u64 id : {0, 1}) {
            std::vector<Tscal> res = alphas->get_buf(id).copy_to_stdvec();
            for (u32 k = 0; k < res.size(); k++) {
                REQUIRE_EQUAL(res[k], (k % ndust == 0) ? 3. : 7.);
            }
        }
    }

    { // one value per dust species is required
        Node node{block_size, ndust, Node::Static{{3.}}};
        node.set_edges(sizes, rho, press, alphas);
        REQUIRE_EXCEPTION_THROW(node.evaluate(), std::invalid_argument);
    }
}
