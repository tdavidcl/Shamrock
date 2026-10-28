// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Penalization.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/Penalization.hpp"
#include "shambackends/kernel_call.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::Penalization<Tvec, TgridVec>::apply(Tscal dt) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;

    using Zone = typename PenalizationConfig<Tvec>::Zone;

    const PenalizationConfig<Tvec> &cfg = solver_config.penalization_config;

    // with dt == 0 nothing should change, and eta == 0 would give 0/0 in the weights
    if (!cfg.is_enabled() || dt == 0) {
        return;
    }

    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");

    const Tscal dxfact         = solver_config.grid_coord_to_pos_fact;
    const Tscal one_over_Nside = Tscal(1) / AMRBlock::Nside;

    // zones are applied one after the other in config order, so overlapping zones give the same
    // result on every run
    for (const Zone &zone : cfg.zones) {

        // implicit Euler weights of the current and target states
        const Tscal w_old    = zone.eta / (zone.eta + dt);
        const Tscal w_target = dt / (zone.eta + dt);

        const Tvec target_vel  = zone.target_vel;
        const bool has_rho     = zone.target_rho.has_value();
        const Tscal target_rho = zone.target_rho.value_or(0);
        const bool has_eint    = zone.target_press.has_value();
        const Tscal target_eint
            = (has_eint) ? *zone.target_press / (solver_config.get_eos_gamma() - 1) : 0;

        std::visit(
            [&](const auto shape) {
                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln("[Ramses]", "penalization patch", p.id_patch);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                    u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

                    sham::DeviceBuffer<TgridVec> &buf_cell_min
                        = pdat.get_field_buf_ref<TgridVec>(icell_min);
                    sham::DeviceBuffer<TgridVec> &buf_cell_max
                        = pdat.get_field_buf_ref<TgridVec>(icell_max);
                    sham::DeviceBuffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
                    sham::DeviceBuffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
                    sham::DeviceBuffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

                    sham::kernel_call(
                        q,
                        sham::MultiRef{buf_cell_min, buf_cell_max},
                        sham::MultiRef{buf_rho, buf_rhov, buf_rhoe},
                        cell_count,
                        [=](u32 id_a,
                            const TgridVec *__restrict cell_min,
                            const TgridVec *__restrict cell_max,
                            Tscal *__restrict rho,
                            Tvec *__restrict rhov,
                            Tscal *__restrict rhoe) {
                            const u32 block_id = id_a / AMRBlock::block_size;
                            const u32 lid      = id_a % AMRBlock::block_size;

                            Tvec block_min = cell_min[block_id].template convert<Tscal>() * dxfact;
                            Tvec block_max = cell_max[block_id].template convert<Tscal>() * dxfact;
                            Tvec cell_size = (block_max - block_min) * one_over_Nside;

                            auto coord = AMRBlock::get_coord(lid);
                            Tvec cell_center
                                = block_min
                                  + Tvec{coord[0] + 0.5, coord[1] + 0.5, coord[2] + 0.5}
                                        * cell_size;

                            if (!shape.contains(cell_center)) {
                                return;
                            }

                            const Tscal rho_old = rho[id_a];
                            const Tvec vel_old  = rhov[id_a] / rho_old;
                            const Tscal eint_old
                                = rhoe[id_a] - 0.5 * rho_old * sycl::dot(vel_old, vel_old);

                            const Tscal rho_new
                                = (has_rho) ? w_old * rho_old + w_target * target_rho : rho_old;
                            const Tvec vel_new   = w_old * vel_old + w_target * target_vel;
                            const Tscal eint_new = (has_eint)
                                                       ? w_old * eint_old + w_target * target_eint
                                                       : eint_old * (rho_new / rho_old);

                            rho[id_a]  = rho_new;
                            rhov[id_a] = rho_new * vel_new;
                            rhoe[id_a] = eint_new + 0.5 * rho_new * sycl::dot(vel_new, vel_new);
                        });
                });
            },
            zone.shape);
    }
}

template class shammodels::basegodunov::modules::Penalization<f64_3, i64_3>;
