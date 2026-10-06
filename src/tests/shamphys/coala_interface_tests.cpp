// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shamphys/coala_interface.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>
#include <random>
#include <vector>

// The table based source term must match the reference one for ANY dv (no symmetry nor zero
// diagonal assumed), on a triangular (T[j,l,m] = 0 for l > j) and on a dense tensor.
NEW_TEST(Unittest, "shamphys/coala_interface::flux_diff_table", 1) {

    using T = f64;

    std::mt19937 eng(0x1234);
    std::uniform_real_distribution<T> dist(-1., 1.);

    for (bool triangular : {true, false}) {
        for (u32 nbins : {1u, 2u, 7u, 20u}) {

            std::vector<T> tensor(nbins * nbins * nbins, 0);
            std::mdspan<T, std::dextents<u32, 3>> T3(tensor.data(), nbins, nbins, nbins);
            for (u32 j = 0; j < nbins; ++j) {
                for (u32 l = 0; l < nbins; ++l) {
                    for (u32 m = 0; m < nbins; ++m) {
                        if ((!triangular || (l <= j && l + m >= nbins / 2))) {
                            T3(j, l, m) = dist(eng);
                        }
                    }
                }
            }

            // non symmetric dv, non zero diagonal on purpose
            std::vector<T> dvm(nbins * nbins);
            for (auto &x : dvm) {
                x = std::abs(dist(eng));
            }
            auto dv = [&](int l, int m) {
                return dvm[l * nbins + m];
            };

            std::vector<T> massgrid(nbins + 1);
            for (u32 i = 0; i <= nbins; ++i) {
                massgrid[i] = std::pow(10., i);
            }
            std::vector<T> rho(nbins);
            for (auto &x : rho) {
                x = std::abs(dist(eng));
            }
            auto rho_dust = [&](int j) {
                return rho[j];
            };

            std::vector<T> gij(nbins), flux(nbins), ref(nbins), res(nbins);
            std::mdspan<const T, std::dextents<u32, 3>> cT3(tensor.data(), nbins, nbins, nbins);
            std::mdspan<T, std::dextents<u32, 1>> gij_s(gij.data(), nbins);

            shamphys::coala_k0_source_term(
                int(nbins),
                dv,
                rho_dust,
                T(0.1),
                std::mdspan<const T, std::dextents<u32, 1>>(massgrid.data(), nbins + 1),
                cT3,
                gij_s,
                std::mdspan<T, std::dextents<u32, 1>>(flux.data(), nbins),
                std::mdspan<T, std::dextents<u32, 1>>(ref.data(), nbins));

            auto table = shamphys::build_coala_flux_diff_table<T>(nbins, cT3);
            shamphys::coala_k0_source_term_table(
                table, dv, gij_s, std::mdspan<T, std::dextents<u32, 1>>(res.data(), nbins));

            T scale = 0;
            for (u32 j = 0; j < nbins; ++j) {
                scale = std::max(scale, std::abs(ref[j]));
            }
            T err = 0;
            for (u32 j = 0; j < nbins; ++j) {
                err = std::max(err, std::abs(ref[j] - res[j]));
            }
            REQUIRE(err <= 1e-12 * std::max(scale, T(1e-300)) + 1e-300);

            if (triangular && nbins > 1) {
                REQUIRE(table.npairs < nbins * nbins);
            }
        }
    }
}
