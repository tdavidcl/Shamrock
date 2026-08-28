// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeDragUpdateNoSrc.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shammodels/ramses/modules/NodeDragUpdateNoSrc.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeDragUpdateNoSrc<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);

        edges.rho_next.ensure_sizes(edges.sizes.indexes);
        edges.rhov_next.ensure_sizes(edges.sizes.indexes);
        edges.rhoe_next.ensure_sizes(edges.sizes.indexes);
        edges.rho_dust_next.ensure_sizes(edges.sizes.indexes);
        edges.rhov_dust_next.ensure_sizes(edges.sizes.indexes);

        const Tscal dt  = edges.dt.value;
        const u32 ndust = this->ndust;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.sizes.indexes.for_each([&](u64 id, u32 block_count) {
            u32 cell_count = block_count * block_size;

            if (cell_count == 0) {
                return;
            }

            auto &buf_rho    = edges.spans_rho.get_field(id).get_buf();
            auto &buf_rhov   = edges.spans_rhov.get_field(id).get_buf();
            auto &buf_rhoe   = edges.spans_rhoe.get_field(id).get_buf();
            auto &buf_rho_d  = edges.spans_rho_dust.get_field(id).get_buf();
            auto &buf_rhov_d = edges.spans_rhov_dust.get_field(id).get_buf();

            auto &buf_dtrho    = edges.spans_dtrho.get_field(id).get_buf();
            auto &buf_dtrhov   = edges.spans_dtrhov.get_field(id).get_buf();
            auto &buf_dtrhoe   = edges.spans_dtrhoe.get_field(id).get_buf();
            auto &buf_dtrho_d  = edges.spans_dtrho_dust.get_field(id).get_buf();
            auto &buf_dtrhov_d = edges.spans_dtrhov_dust.get_field(id).get_buf();

            auto &out_rho    = edges.rho_next.get_field(id).get_buf();
            auto &out_rhov   = edges.rhov_next.get_field(id).get_buf();
            auto &out_rhoe   = edges.rhoe_next.get_field(id).get_buf();
            auto &out_rho_d  = edges.rho_dust_next.get_field(id).get_buf();
            auto &out_rhov_d = edges.rhov_dust_next.get_field(id).get_buf();

            { // gas
                sham::EventList depend_list;

                auto rho     = buf_rho.get_read_access(depend_list);
                auto rhov    = buf_rhov.get_read_access(depend_list);
                auto rhoe    = buf_rhoe.get_read_access(depend_list);
                auto dt_rho  = buf_dtrho.get_read_access(depend_list);
                auto dt_rhov = buf_dtrhov.get_read_access(depend_list);
                auto dt_rhoe = buf_dtrhoe.get_read_access(depend_list);

                auto acc_rho  = out_rho.get_write_access(depend_list);
                auto acc_rhov = out_rhov.get_write_access(depend_list);
                auto acc_rhoe = out_rhoe.get_write_access(depend_list);

                auto e = q.submit(depend_list, [&, dt](sycl::handler &cgh) {
                    shambase::parallel_for(
                        cgh, cell_count, "evolve field with no drag", [=](u32 id_a) {
                            acc_rho[id_a]  = rho[id_a] + dt * dt_rho[id_a];
                            acc_rhov[id_a] = rhov[id_a] + dt * dt_rhov[id_a];
                            acc_rhoe[id_a] = rhoe[id_a] + dt * dt_rhoe[id_a];
                        });
                });

                buf_rho.complete_event_state(e);
                buf_rhov.complete_event_state(e);
                buf_rhoe.complete_event_state(e);
                buf_dtrho.complete_event_state(e);
                buf_dtrhov.complete_event_state(e);
                buf_dtrhoe.complete_event_state(e);
                out_rho.complete_event_state(e);
                out_rhov.complete_event_state(e);
                out_rhoe.complete_event_state(e);
            }

            { // dust
                sham::EventList depend_list;

                auto rho_d     = buf_rho_d.get_read_access(depend_list);
                auto rhov_d    = buf_rhov_d.get_read_access(depend_list);
                auto dt_rho_d  = buf_dtrho_d.get_read_access(depend_list);
                auto dt_rhov_d = buf_dtrhov_d.get_read_access(depend_list);

                auto acc_rho_d  = out_rho_d.get_write_access(depend_list);
                auto acc_rhov_d = out_rhov_d.get_write_access(depend_list);

                auto e = q.submit(depend_list, [&, dt, ndust](sycl::handler &cgh) {
                    shambase::parallel_for(
                        cgh, ndust * cell_count, "dust evolve field no drag", [=](u32 id_a) {
                            acc_rho_d[id_a]  = rho_d[id_a] + dt * dt_rho_d[id_a];
                            acc_rhov_d[id_a] = rhov_d[id_a] + dt * dt_rhov_d[id_a];
                        });
                });

                buf_rho_d.complete_event_state(e);
                buf_rhov_d.complete_event_state(e);
                buf_dtrho_d.complete_event_state(e);
                buf_dtrhov_d.complete_event_state(e);
                out_rho_d.complete_event_state(e);
                out_rhov_d.complete_event_state(e);
            }
        });
    }

    template<class Tvec>
    std::string NodeDragUpdateNoSrc<Tvec>::_impl_get_tex() const {
        auto rho      = get_ro_edge_base(2).get_tex_symbol();
        auto rho_next = get_rw_edge_base(0).get_tex_symbol();
        auto dt       = get_ro_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Flux only update of the conservative state (drag source applied separately)
            \begin{equation}
            {rho_next} = {rho} + {dt} \, \partial_t {rho}
            \end{equation}
            and likewise for the gas momentum and energy, and for the dust density and momentum.
        )tex";

        shambase::replace_all(tex, "{rho_next}", rho_next);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{dt}", dt);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeDragUpdateNoSrc<f64_3>;
