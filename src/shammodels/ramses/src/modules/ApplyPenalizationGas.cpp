// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ApplyPenalizationGas.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ApplyPenalizationGas.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ApplyPenalizationGas<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_sdf.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_rhov.check_sizes(edges.sizes.indexes);
        edges.spans_rhoe.check_sizes(edges.sizes.indexes);

        Tscal dt = edges.dt.value;

        // with dt == 0 nothing should change, and eta == 0 would give 0/0 in the weights
        if (dt == 0) {
            return;
        }

        // implicit Euler weights of the current and target states
        Tscal w_old    = eta / (eta + dt);
        Tscal w_target = dt / (eta + dt);

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{edges.spans_sdf.get_spans()},
            sham::DDMultiRef{
                edges.spans_rho.get_spans(),
                edges.spans_rhov.get_spans(),
                edges.spans_rhoe.get_spans()},
            cell_counts,
            [w_old,
             w_target,
             target_vel  = target_vel,
             has_rho     = has_rho,
             target_rho  = target_rho,
             has_eint    = has_eint,
             target_eint = target_eint](
                u32 i,
                const Tscal *__restrict sdf,
                Tscal *__restrict rho,
                Tvec *__restrict rhov,
                Tscal *__restrict rhoe) {
                if (sdf[i] > 0) {
                    return;
                }

                const Tscal rho_old  = rho[i];
                const Tvec vel_old   = rhov[i] / rho_old;
                const Tscal eint_old = rhoe[i] - 0.5 * rho_old * sycl::dot(vel_old, vel_old);

                const Tscal rho_new = (has_rho) ? w_old * rho_old + w_target * target_rho : rho_old;
                const Tvec vel_new  = w_old * vel_old + w_target * target_vel;
                const Tscal eint_new = (has_eint) ? w_old * eint_old + w_target * target_eint
                                                  : eint_old * (rho_new / rho_old);

                rho[i]  = rho_new;
                rhov[i] = rho_new * vel_new;
                rhoe[i] = eint_new + 0.5 * rho_new * sycl::dot(vel_new, vel_new);
            });
    }

    template<class Tvec>
    std::string ApplyPenalizationGas<Tvec>::_impl_get_tex() const {
        auto dt   = get_ro_edge_base(0).get_tex_symbol();
        auto sdf  = get_ro_edge_base(2).get_tex_symbol();
        auto rho  = get_rw_edge_base(0).get_tex_symbol();
        auto rhov = get_rw_edge_base(1).get_tex_symbol();
        auto rhoe = get_rw_edge_base(2).get_tex_symbol();

        std::string tex = R"tex(
            Apply volume penalization to the gas (implicit Euler), for the cells with ${sdf}_i \leq 0$

            \begin{align}
            w &= \frac{\eta}{\eta + {dt}} \\
            \rho_i^{\rm new} &= w {rho}_i + (1 - w) \rho_{\rm target} \quad \text{(if a target density is set)} \\
            \mathbf{v}_i^{\rm new} &= w \frac{{rhov}_i}{{rho}_i} + (1 - w) \mathbf{v}_{\rm target} \\
            e_{{\rm int},i}^{\rm new} &= w e_{{\rm int},i} + (1 - w) \frac{P_{\rm target}}{\gamma - 1} \quad \text{(if a target pressure is set, else } e_{{\rm int},i} \rho_i^{\rm new} / {rho}_i \text{)} \\
            {rho}_i &= \rho_i^{\rm new} \\
            {rhov}_i &= \rho_i^{\rm new} \mathbf{v}_i^{\rm new} \\
            {rhoe}_i &= e_{{\rm int},i}^{\rm new} + \frac{1}{2} \rho_i^{\rm new} (\mathbf{v}_i^{\rm new})^2
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt}", dt);
        shambase::replace_all(tex, "{sdf}", sdf);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{rhov}", rhov);
        shambase::replace_all(tex, "{rhoe}", rhoe);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ApplyPenalizationGas<f64_3>;
