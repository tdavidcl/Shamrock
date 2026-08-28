// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeDragIntegratorIRK1.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shammodels/ramses/modules/NodeDragIntegratorIRK1.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeDragIntegratorIRK1<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.rho_next.check_sizes(edges.sizes.indexes);
        edges.alphas.check_sizes(edges.sizes.indexes);

        const Tscal dt             = edges.dt.value;
        const u32 ndust            = this->ndust;
        const u32 friction_control = (enable_frictional_heating == false) ? 1 : 0;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.sizes.indexes.for_each([&](u64 id, u32 block_count) {
            u32 cell_count = block_count * block_size;

            if (cell_count == 0) {
                return;
            }

            auto &rho_new_patch    = edges.rho_next.get_field(id).get_buf();
            auto &rhov_new_patch   = edges.rhov_next.get_field(id).get_buf();
            auto &rhoe_new_patch   = edges.rhoe_next.get_field(id).get_buf();
            auto &rho_d_new_patch  = edges.rho_dust_next.get_field(id).get_buf();
            auto &rhov_d_new_patch = edges.rhov_dust_next.get_field(id).get_buf();

            auto &rho_old    = edges.spans_rho.get_field(id).get_buf();
            auto &rhov_old   = edges.spans_rhov.get_field(id).get_buf();
            auto &rhoe_old   = edges.spans_rhoe.get_field(id).get_buf();
            auto &rho_d_old  = edges.spans_rho_dust.get_field(id).get_buf();
            auto &rhov_d_old = edges.spans_rhov_dust.get_field(id).get_buf();

            auto &alphas_buf = edges.alphas.get_field(id).get_buf();

            sham::EventList depend_list;
            auto acc_rho_new_patch    = rho_new_patch.get_read_access(depend_list);
            auto acc_rhov_new_patch   = rhov_new_patch.get_read_access(depend_list);
            auto acc_rhoe_new_patch   = rhoe_new_patch.get_read_access(depend_list);
            auto acc_rho_d_new_patch  = rho_d_new_patch.get_read_access(depend_list);
            auto acc_rhov_d_new_patch = rhov_d_new_patch.get_read_access(depend_list);

            auto acc_rho_old    = rho_old.get_write_access(depend_list);
            auto acc_rhov_old   = rhov_old.get_write_access(depend_list);
            auto acc_rhoe_old   = rhoe_old.get_write_access(depend_list);
            auto acc_rho_d_old  = rho_d_old.get_write_access(depend_list);
            auto acc_rhov_d_old = rhov_d_old.get_write_access(depend_list);

            auto acc_alphas = alphas_buf.get_read_access(depend_list);

            auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
                shambase::parallel_for(cgh, cell_count, "add_drag [irk1]", [=](u32 id_a) {
                    Tvec tmp_mom_1 = acc_rhov_new_patch[id_a];
                    Tscal tmp_rho  = acc_rho_old[id_a];

                    for (u32 i = 0; i < ndust; i++) {
                        const Tscal alpha_i       = acc_alphas[id_a * ndust + i];
                        const Tscal inv_dt_alphas = 1.0 / (1.0 + alpha_i * dt);
                        const Tscal dt_alphas     = dt * alpha_i;

                        tmp_mom_1
                            = tmp_mom_1
                              + dt_alphas * inv_dt_alphas * acc_rhov_d_new_patch[id_a * ndust + i];
                        tmp_rho
                            = tmp_rho + dt_alphas * inv_dt_alphas * acc_rho_d_old[id_a * ndust + i];
                    }

                    Tscal tmp_inv_rho = 1.0 / tmp_rho;
                    Tvec tmp_vel      = tmp_inv_rho * tmp_mom_1;
                    Tscal Eg          = 0.0;

                    Tscal inv_rho_g = 1.0 / acc_rho_new_patch[id_a];
                    Tvec vg_bf      = inv_rho_g * acc_rhov_new_patch[id_a];
                    Tvec vg_af      = inv_rho_g * acc_rho_old[id_a] * tmp_vel;

                    Tscal work_drag
                        = 0.5
                          * ((acc_rho_old[id_a] * tmp_vel[0] - acc_rhov_new_patch[id_a][0])
                                 * (vg_bf[0] + vg_af[0])
                             + (acc_rho_old[id_a] * tmp_vel[1] - acc_rhov_new_patch[id_a][1])
                                   * (vg_bf[1] + vg_af[1])
                             + (acc_rho_old[id_a] * tmp_vel[2] - acc_rhov_new_patch[id_a][2])
                                   * (vg_bf[2] + vg_af[2]));

                    Tscal dissipation = 0.0;
                    for (u32 i = 0; i < ndust; i++) {
                        const Tscal alpha_i       = acc_alphas[id_a * ndust + i];
                        const Tscal inv_dt_alphas = 1.0 / (1.0 + alpha_i * dt);
                        const Tscal dt_alphas     = dt * alpha_i;

                        Tscal inv_rho_d = 1.0 / acc_rho_d_new_patch[id_a * ndust + i];
                        Tvec vd_bf      = inv_rho_d * acc_rhov_d_new_patch[id_a * ndust + i];
                        Tvec vd_af = inv_rho_d * inv_dt_alphas
                                     * (acc_rhov_d_new_patch[id_a * ndust + i]
                                        + dt_alphas * acc_rho_d_old[id_a * ndust + i] * tmp_vel);
                        dissipation += 0.5 * dt_alphas * inv_dt_alphas
                                       * ((acc_rho_d_old[id_a * ndust + i] * tmp_vel[0]
                                           - acc_rhov_d_new_patch[id_a * ndust + i][0])
                                              * (vd_af[0] + vd_bf[0])
                                          + (acc_rho_d_old[id_a * ndust + i] * tmp_vel[1]
                                             - acc_rhov_d_new_patch[id_a * ndust + i][1])
                                                * (vd_af[1] + vd_bf[1])
                                          + (acc_rho_d_old[id_a * ndust + i] * tmp_vel[2]
                                             - acc_rhov_d_new_patch[id_a * ndust + i][2])
                                                * (vd_af[2] + vd_bf[2]));
                    }

                    Eg += acc_rhoe_new_patch[id_a] + (1 - friction_control) * work_drag
                          - friction_control * dissipation;
                    acc_rhov_old[id_a] = tmp_vel * acc_rho_old[id_a];
                    acc_rhoe_old[id_a] = Eg;
                    acc_rho_old[id_a]  = acc_rho_new_patch[id_a];
                    for (u32 i = 0; i < ndust; i++) {
                        const Tscal alpha_i       = acc_alphas[id_a * ndust + i];
                        const Tscal inv_dt_alphas = 1.0 / (1.0 + alpha_i * dt);
                        const Tscal dt_alphas     = dt * alpha_i;

                        acc_rhov_d_old[id_a * ndust + i]
                            = inv_dt_alphas
                              * (acc_rhov_d_new_patch[id_a * ndust + i]
                                 + dt_alphas * acc_rho_d_old[id_a * ndust + i] * tmp_vel);
                        acc_rho_d_old[id_a * ndust + i] = acc_rho_d_new_patch[id_a * ndust + i];
                    }
                });
            });

            rho_new_patch.complete_event_state(e);
            rhov_new_patch.complete_event_state(e);
            rhoe_new_patch.complete_event_state(e);
            rho_d_new_patch.complete_event_state(e);
            rhov_d_new_patch.complete_event_state(e);

            rho_old.complete_event_state(e);
            rhov_old.complete_event_state(e);
            rhoe_old.complete_event_state(e);
            rho_d_old.complete_event_state(e);
            rhov_d_old.complete_event_state(e);

            alphas_buf.complete_event_state(e);
        });
    }

    template<class Tvec>
    std::string NodeDragIntegratorIRK1<Tvec>::_impl_get_tex() const {
        auto alphas = get_ro_edge_base(2).get_tex_symbol();
        auto dt     = get_ro_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            First order implicit Runge-Kutta drag integrator, solving
            \begin{equation}
            \partial_t v_{{\rm d},j} = {alphas}_j (v_{\rm g} - v_{{\rm d},j})
            \end{equation}
            together with the back reaction on the gas, implicitly over ${dt}$, keeping the total
            momentum of the mixture constant.
        )tex";

        shambase::replace_all(tex, "{alphas}", alphas);
        shambase::replace_all(tex, "{dt}", dt);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeDragIntegratorIRK1<f64_3>;
