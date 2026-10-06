// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/vec.hpp"
#include "shammodels/sph/modules/NodeEvolveDustCOALASourceTerm.hpp"
#include "shamphys/coala_interface.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// Run the node (SYCL kernel) and compare with the host reference implementation.
NEW_TEST(Unittest, "shammodels/sph/modules/NodeEvolveDustCOALASourceTerm", 1) {
    using namespace shamrock::solvergraph;
    using Tscal = f64;
    using Tvec  = f64_3;

    std::mt19937 eng(0xC0A1A);
    std::uniform_real_distribution<Tscal> dist(0., 1.);

    // 40 bins with a dense tensor -> several pair chunks; counts not multiples of the group size
    for (u32 nbins : {5u, 40u}) {
        for (u32 count : {1u, 13u, 100u}) {

            const Tscal rho_eps = 0.2, dv_max = 1.2;

            std::vector<Tscal> massgrid(nbins + 1), tensor(nbins * nbins * nbins);
            for (u32 i = 0; i <= nbins; ++i) {
                massgrid[i] = std::pow(2., i);
            }
            for (auto &x : tensor) {
                x = (dist(eng) < 0.7) ? dist(eng) - 0.5 : 0; // partly sparse
            }

            std::vector<Tscal> s_data(count * nbins);
            std::vector<Tvec> v_data(count * nbins);
            for (auto &x : s_data) {
                x = dist(eng); // rho = s^2, some below rho_eps
            }
            for (auto &v : v_data) {
                v = Tvec{dist(eng), dist(eng), dist(eng)}; // some |dv| above dv_max
            }

            auto s_j     = std::make_shared<Field<Tscal>>(nbins, "s_j", "s_j");
            auto delta_v = std::make_shared<Field<Tvec>>(nbins, "delta_v", "delta_v");
            auto S_coag  = std::make_shared<Field<Tscal>>(nbins, "S_coag", "S_coag");

            auto counts = std::make_shared<Indexes<u32>>("", "");
            counts->indexes.add_obj(0, u32(count));

            s_j->ensure_sizes(counts->indexes);
            delta_v->ensure_sizes(counts->indexes);
            s_j->get(0).get_buf().copy_from_stdvec(s_data);
            delta_v->get(0).get_buf().copy_from_stdvec(v_data);

            auto e_rho_eps   = IDataEdge<Tscal>::make_shared("", "");
            auto e_dv_max    = IDataEdge<Tscal>::make_shared("", "");
            auto e_massgrid  = IDataEdge<std::vector<Tscal>>::make_shared("", "");
            auto e_tensor    = IDataEdge<std::vector<Tscal>>::make_shared("", "");
            e_rho_eps->data  = rho_eps;
            e_dv_max->data   = dv_max;
            e_massgrid->data = massgrid;
            e_tensor->data   = tensor;

            shammodels::sph::modules::NodeEvolveDustCOALASourceTerm<Tvec> node(nbins);
            node.set_edges(e_rho_eps, e_dv_max, e_massgrid, e_tensor, counts, s_j, delta_v, S_coag);
            node.evaluate();

            auto res = S_coag->get(0).get_buf().copy_to_stdvec();

            // reference
            using M1      = std::mdspan<Tscal, std::dextents<u32, 1>>;
            using M3      = std::mdspan<const Tscal, std::dextents<u32, 3>>;
            Tscal max_err = 0, scale = 0;
            for (u32 a = 0; a < count; ++a) {
                std::vector<Tscal> gij(nbins), flux(nbins), ref(nbins);
                auto rho_dust = [&](int j) {
                    return s_data[a * nbins + j] * s_data[a * nbins + j];
                };
                auto dv = [&](int i, int j) {
                    Tvec d  = v_data[a * nbins + j] - v_data[a * nbins + i];
                    Tscal t = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    return (t > dv_max) ? Tscal(0) : t;
                };
                shamphys::coala_k0_source_term(
                    int(nbins),
                    dv,
                    rho_dust,
                    rho_eps,
                    std::mdspan<const Tscal, std::dextents<u32, 1>>(massgrid.data(), nbins + 1),
                    M3(tensor.data(), nbins, nbins, nbins),
                    M1(gij.data(), nbins),
                    M1(flux.data(), nbins),
                    M1(ref.data(), nbins));
                for (u32 j = 0; j < nbins; ++j) {
                    max_err = std::max(max_err, std::abs(ref[j] - res[a * nbins + j]));
                    scale   = std::max(scale, std::abs(ref[j]));
                }
            }
            REQUIRE(scale > 0);
            REQUIRE(max_err <= 1e-11 * scale);
        }
    }
}
