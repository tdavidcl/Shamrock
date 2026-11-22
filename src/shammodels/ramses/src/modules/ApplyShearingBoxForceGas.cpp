// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyShearingBoxForceGas.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node applying the shearing box forces to the gas conservative state
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ApplyShearingBoxForceGas.hpp"
#include "shammodels/ramses/modules/ShearingBoxForceUtils.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void ApplyShearingBoxForceGas<Tvec, TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_coordinates.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rho.check_sizes(edges.sizes.indexes);
        edges.spans_dt_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe.check_sizes(edges.sizes.indexes);

        Tscal dt = edges.dt.value;

        // center of the box along the radial axis
        const auto &box = edges.sim_box.value;
        Tscal x_c
            = (Tscal(box.lower.x()) + Tscal(box.upper.x())) * grid_coord_to_pos_fact / Tscal{2};

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.spans_coordinates.get_spans(),
                edges.spans_rho.get_spans(),
                edges.spans_dt_rho.get_spans(),
                edges.spans_dt_rhov.get_spans()},
            sham::DDMultiRef{edges.spans_rhov.get_spans(), edges.spans_rhoe.get_spans()},
            cell_counts,
            [dt,
             x_c,
             Omega_0     = Omega_0,
             tidal_coef  = 2 * q * Omega_0 * Omega_0,
             eta_forcing = 2 * Omega_0 * eta](
                u32 i,
                const Tvec *__restrict coords,
                const Tscal *__restrict rho,
                const Tscal *__restrict dt_rho,
                const Tvec *__restrict dt_rhov,
                Tvec *__restrict rhov,
                Tscal *__restrict rhoe) {
                // state at t^{n+1} without the shearing box forces
                Tscal rho_np1 = rho[i] + dt * dt_rho[i];
                Tvec p_star   = rhov[i] + dt * dt_rhov[i];

                Tscal g_x = tidal_coef * (coords[i].x() - x_c) + eta_forcing;

                Tvec p_np1 = shearing_box_momentum_update<Tvec>(p_star, rho_np1, g_x, Omega_0, dt);

                rhov[i] += p_np1 - p_star;
                rhoe[i] += shearing_box_energy_update<Tvec>(p_star, p_np1, rho_np1);
            });
    }

    template<class Tvec, class TgridVec>
    std::string ApplyShearingBoxForceGas<Tvec, TgridVec>::_impl_get_tex() const {
        auto dt      = get_ro_edge_base(0).get_tex_symbol();
        auto coords  = get_ro_edge_base(3).get_tex_symbol();
        auto rho     = get_ro_edge_base(4).get_tex_symbol();
        auto dt_rho  = get_ro_edge_base(5).get_tex_symbol();
        auto dt_rhov = get_ro_edge_base(6).get_tex_symbol();
        auto rhov    = get_rw_edge_base(0).get_tex_symbol();
        auto rhoe    = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Apply the shearing box forces to the gas (trapezoidal rule, implicit in the momentum)

            \begin{align}
            \rho^{n+1}_i &= {rho}_i + {dt} \, {dt_rho}_i \\
            \mathbf{p}^*_i &= {rhov}_i + {dt} \, {dt_rhov}_i \\
            g_{x,i} &= 2 q \Omega_0^2 ({coords}_{x,i} - x_c) + 2 \Omega_0 \eta \\
            \mathbf{p}^{n+1}_i &= \mathbf{p}^*_i + \frac{{dt}}{2} \left(
                \mathbf{S}(\rho^{n+1}_i, \mathbf{p}^*_i)
                + \mathbf{S}(\rho^{n+1}_i, \mathbf{p}^{n+1}_i) \right), \quad
                \mathbf{S}(\rho, \mathbf{p}) = \rho g_{x,i} \mathbf{e}_x
                + 2 \Omega_0 (p_y, -p_x, 0) \\
            {rhoe}_i &= {rhoe}_i + \frac{|\mathbf{p}^{n+1}_i|^2 - |\mathbf{p}^*_i|^2}{2 \rho^{n+1}_i} \\
            {rhov}_i &= {rhov}_i + \mathbf{p}^{n+1}_i - \mathbf{p}^*_i
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt}", dt);
        shambase::replace_all(tex, "{coords}", coords);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{dt_rho}", dt_rho);
        shambase::replace_all(tex, "{dt_rhov}", dt_rhov);
        shambase::replace_all(tex, "{rhov}", rhov);
        shambase::replace_all(tex, "{rhoe}", rhoe);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyShearingBoxForceGas<f64_3, i64_3>;
