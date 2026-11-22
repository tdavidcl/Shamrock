// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyShearingBoxForceDust.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node applying the shearing box forces to the dust conservative state
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ApplyShearingBoxForceDust.hpp"
#include "shammodels/ramses/modules/ShearingBoxForceUtils.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void ApplyShearingBoxForceDust<Tvec, TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_coordinates.check_sizes(edges.sizes.indexes);
        edges.spans_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rhov_dust.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_dust.check_sizes(edges.sizes.indexes);

        Tscal dt = edges.dt.value;

        // center of the box along the radial axis
        const auto &box = edges.sim_box.value;
        Tscal x_c
            = (Tscal(box.lower.x()) + Tscal(box.upper.x())) * grid_coord_to_pos_fact / Tscal{2};

        shambase::DistributedData<u32> dust_cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size * ndust;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.spans_coordinates.get_spans(),
                edges.spans_rho_dust.get_spans(),
                edges.spans_dt_rho_dust.get_spans(),
                edges.spans_dt_rhov_dust.get_spans()},
            sham::DDMultiRef{edges.spans_rhov_dust.get_spans()},
            dust_cell_counts,
            [dt, x_c, ndust = ndust, Omega_0 = Omega_0, tidal_coef = 2 * q * Omega_0 * Omega_0](
                u32 i,
                const Tvec *__restrict coords,
                const Tscal *__restrict rho_dust,
                const Tscal *__restrict dt_rho_dust,
                const Tvec *__restrict dt_rhov_dust,
                Tvec *__restrict rhov_dust) {
                // dust fields are laid out as [cell * ndust + idust]
                u32 cell_id = i / ndust;

                // state at t^{n+1} without the shearing box forces
                Tscal rho_np1 = rho_dust[i] + dt * dt_rho_dust[i];
                Tvec p_star   = rhov_dust[i] + dt * dt_rhov_dust[i];

                Tscal g_x = tidal_coef * (coords[cell_id].x() - x_c);

                Tvec p_np1 = shearing_box_momentum_update<Tvec>(p_star, rho_np1, g_x, Omega_0, dt);

                rhov_dust[i] += p_np1 - p_star;
            });
    }

    template<class Tvec, class TgridVec>
    std::string ApplyShearingBoxForceDust<Tvec, TgridVec>::_impl_get_tex() const {
        auto dt           = get_ro_edge_base(0).get_tex_symbol();
        auto coords       = get_ro_edge_base(3).get_tex_symbol();
        auto rho_dust     = get_ro_edge_base(4).get_tex_symbol();
        auto dt_rho_dust  = get_ro_edge_base(5).get_tex_symbol();
        auto dt_rhov_dust = get_ro_edge_base(6).get_tex_symbol();
        auto rhov_dust    = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Apply the shearing box forces to the dust (trapezoidal rule, implicit in the momentum)

            \begin{align}
            \rho^{n+1}_{d,i} &= {rho_dust}_i + {dt} \, {dt_rho_dust}_i \\
            \mathbf{p}^*_i &= {rhov_dust}_i + {dt} \, {dt_rhov_dust}_i \\
            g_{x,i} &= 2 q \Omega_0^2 ({coords}_{x,i} - x_c) \\
            \mathbf{p}^{n+1}_i &= \mathbf{p}^*_i + \frac{{dt}}{2} \left(
                \mathbf{S}(\rho^{n+1}_{d,i}, \mathbf{p}^*_i)
                + \mathbf{S}(\rho^{n+1}_{d,i}, \mathbf{p}^{n+1}_i) \right), \quad
                \mathbf{S}(\rho, \mathbf{p}) = \rho g_{x,i} \mathbf{e}_x
                + 2 \Omega_0 (p_y, -p_x, 0) \\
            {rhov_dust}_i &= {rhov_dust}_i + \mathbf{p}^{n+1}_i - \mathbf{p}^*_i
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt}", dt);
        shambase::replace_all(tex, "{coords}", coords);
        shambase::replace_all(tex, "{rho_dust}", rho_dust);
        shambase::replace_all(tex, "{dt_rho_dust}", dt_rho_dust);
        shambase::replace_all(tex, "{dt_rhov_dust}", dt_rhov_dust);
        shambase::replace_all(tex, "{rhov_dust}", rhov_dust);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyShearingBoxForceDust<f64_3, i64_3>;
