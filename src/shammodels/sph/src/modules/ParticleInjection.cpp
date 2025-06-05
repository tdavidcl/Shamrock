// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ParticleInjection.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/ParticleInjection.hpp"
#include "shamalgs/random.hpp"
#include "shammath/sphkernels.hpp"
#include <vector>

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ParticleInjection<Tvec, SPHKernel>::inject_particles() {

    using namespace shamrock::patch;
    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    auto has_pdat = [&]() {
        bool ret = false;
        sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
            ret = true;
        });
        return ret;
    };

    std::vector<Tvec> pos_gen;
    std::vector<Tvec> vel_gen;
    std::vector<Tscal> h_gen;

    static std::mt19937 eng(0);

    Tvec pos_inject = {1.0, 0.0, 0.0};
    Tscal r_inject  = 0.1;

    for (u32 i = 0; i < 5; i++) {
        auto r = shamalgs::random::mock_value<Tvec>(
            eng, {-r_inject, -r_inject, -r_inject}, {r_inject, r_inject, r_inject});
        auto v = shamalgs::random::mock_value<Tvec>(eng, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});

        pos_gen.push_back(r);
        vel_gen.push_back(v);
        h_gen.push_back(1e-2);
    }

    PatchCoordTransform<Tvec> ptransf = sched.get_sim_box().get_patch_transform<Tvec>();

    sched.for_each_local_patchdata([&](const Patch p, PatchData &pdat) {
        shammath::CoordRange<Tvec> patch_coord = ptransf.to_obj_coord(p);

        std::vector<Tvec> pos;
        std::vector<Tvec> vel;
        std::vector<Tscal> h;

        for (u64 i = 0; i < pos_gen.size(); i++) {
            Tvec r = pos_gen[i];
            if (patch_coord.contain_pos(r)) {
                pos.push_back(pos_gen[i]);
                vel.push_back(vel_gen[i]);
                h.push_back(h_gen[i]);
            }
        }

        PatchData tmp(sched.pdl);

        if (!pos.empty()) {
            tmp.resize(pos.size());
            tmp.fields_raz();

            {
                u32 len                 = pos.size();
                PatchDataField<Tvec> &f = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
                sycl::buffer<Tvec> buf(pos.data(), len);
                f.override(buf, len);
            }

            {
                u32 len = pos.size();
                PatchDataField<Tvec> &f
                    = tmp.get_field<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));
                sycl::buffer<Tvec> buf(vel.data(), len);
                f.override(buf, len);
            }
            {
                u32 len = pos.size();
                PatchDataField<Tscal> &f
                    = tmp.get_field<Tscal>(sched.pdl.get_field_idx<Tscal>("hpart"));
                sycl::buffer<Tscal> buf(h.data(), len);
                f.override(buf, len);
            }

            pdat.insert_elements(tmp);
        }
    });

    sched.check_patchdata_locality_corectness();
}

using namespace shammath;
template class shammodels::sph::modules::ParticleInjection<f64_3, M4>;
template class shammodels::sph::modules::ParticleInjection<f64_3, M6>;
template class shammodels::sph::modules::ParticleInjection<f64_3, M8>;
