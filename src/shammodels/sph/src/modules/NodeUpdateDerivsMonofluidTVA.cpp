// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeUpdateDerivsMonofluidTVA.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/dust_variables.hpp"
#include "shammodels/sph/modules/NodeUpdateDerivsMonofluidTVA.hpp"
#include "shamrock/patch/PatchDataField.hpp" // IWYU pragma: keep

template<class Tvec, template<class> class SPHKernel, shammodels::sph::DustVariable dust_var>
struct KernelUpdateDerivsMonofluidTVA {
    using Tscal                   = shambase::VecComponent<Tvec>;
    using Kernel                  = SPHKernel<Tscal>;
    using DustVariable            = shammodels::sph::DustVariable;
    static constexpr Tscal hfactd = Kernel::hfactd;
    static constexpr Tscal Rkern  = Kernel::Rkern;
    static constexpr Tscal Rker2  = Rkern * Rkern;

    Tscal pmass;
    u32 ndust;

    inline void operator()(
        u32 thread_id,
        // input
        const Tvec *__restrict xyz,
        const Tscal *__restrict hpart,
        const Tvec *__restrict vxyz,
        const Tscal *__restrict omega,
        const Tscal *__restrict pressure,
        const Tscal *__restrict s_j,
        const Tscal *__restrict Ttilde_sj,
        shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
        // output
        Tscal *__restrict ds_j_dt) const {

        u32 id_a  = thread_id / ndust;
        u32 jdust = thread_id % ndust;

        Tscal h_a         = hpart[id_a];
        Tvec xyz_a        = xyz[id_a];
        Tvec vxyz_a       = vxyz[id_a];
        Tscal P_a         = pressure[id_a];
        Tscal omega_a     = omega[id_a];
        Tscal s_j_a       = s_j[thread_id];
        Tscal Ttilde_sj_a = Ttilde_sj[thread_id];

        using namespace shamrock::sph;
        Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);

        Tscal term1 = 0;
        Tscal term2 = 0;

        shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);
        particle_looper.for_each_object(id_a, [&](u32 id_b) {
            Tvec dr    = xyz_a - xyz[id_b];
            Tscal rab2 = sycl::dot(dr, dr);
            Tscal h_b  = hpart[id_b];

            if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                return;
            }

            Tscal P_b         = pressure[id_b];
            Tscal s_j_b       = s_j[id_b * ndust + jdust];
            Tscal Ttilde_sj_b = Ttilde_sj[id_b * ndust + jdust];

            Tscal rab         = sycl::sqrt(rab2);
            Tscal rab_inv_sat = sham::inv_sat_positive(rab);

            Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);

            Tscal Fab_a = Kernel::dW_3d(rab, h_a);
            Tscal Fab_b = Kernel::dW_3d(rab, h_b);

            Tscal F_ab_bar = (Fab_a + Fab_b) / 2;
            Tscal delta_P  = P_a - P_b;

            if constexpr (dust_var == DustVariable::SqrtRhoEps) {
                Tvec vxyz_b    = vxyz[id_b];
                Tvec v_ab      = vxyz_a - vxyz_b;
                Tvec r_ab_unit = dr * rab_inv_sat;

                Tscal Ts_weighted = (Ttilde_sj_a / rho_a + Ttilde_sj_b / rho_b);

                term1 += (pmass * s_j_b / rho_b) * Ts_weighted * delta_P * F_ab_bar * rab_inv_sat;
                term2 += pmass * sham::dot(v_ab, r_ab_unit * Fab_a);
            } else if constexpr (dust_var == DustVariable::Eps) {
                // arithmetic pair coefficient eps_a Ttilde_a + eps_b Ttilde_b
                Tscal kappa_ab = s_j_a * Ttilde_sj_a + s_j_b * Ttilde_sj_b;

                term1 += (pmass / rho_b) * kappa_ab * delta_P * F_ab_bar * rab_inv_sat;
            } else if constexpr (dust_var == DustVariable::SqrtEpsOverOneMinusEps) {
                // reduced coefficient D_j = (1 - eps_j) Ttilde_sj, with 1 - eps_j = 1/(1 + s_j^2)
                Tscal D_a = Ttilde_sj_a / (1 + s_j_a * s_j_a);
                Tscal D_b = Ttilde_sj_b / (1 + s_j_b * s_j_b);

                term1 += (pmass * s_j_b / rho_b) * (D_a + D_b) * delta_P * F_ab_bar * rab_inv_sat;
            }
        });

        Tscal ds_j_dt_a = 0;
        if constexpr (dust_var == DustVariable::SqrtRhoEps) {
            // eq 51, Hutchison 2018
            ds_j_dt_a = Tscal{-0.5} * term1 + (s_j_a / (2 * rho_a * omega_a)) * term2;
        } else if constexpr (dust_var == DustVariable::Eps) {
            // Price & Laibe 2015 direct second derivative, no div(v) term (eps is advected)
            ds_j_dt_a = -term1 / rho_a;
        } else if constexpr (dust_var == DustVariable::SqrtEpsOverOneMinusEps) {
            // Ballabio et al. 2018 eq 29 generalised per species, 1 - eps_j = 1 / (1 + s_j^2)
            Tscal one_minus_eps_a = 1 / (1 + s_j_a * s_j_a);
            ds_j_dt_a             = -term1 / (2 * rho_a * one_minus_eps_a * one_minus_eps_a);
        }

        ds_j_dt[thread_id] = ds_j_dt_a;
    }
};

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<Tvec, SPHKernel>::
    _impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts_with_ghost = edges.part_counts_with_ghost.indexes;
    auto &part_counts            = edges.part_counts.indexes;

    // check that all input edges have the particles with ghosts zones
    edges.xyz.check_sizes(part_counts_with_ghost);
    edges.hpart.check_sizes(part_counts_with_ghost);
    edges.vxyz.check_sizes(part_counts_with_ghost);
    edges.omega.check_sizes(part_counts_with_ghost);
    edges.pressure.check_sizes(part_counts_with_ghost);
    edges.s_j.check_sizes(part_counts_with_ghost);
    edges.Ttilde_sj.check_sizes(part_counts_with_ghost);

    // ensure that the output edges are of size part_counts (output without ghosts zones)
    edges.ds_j_dt.ensure_sizes(part_counts);

    const Tscal pmass = edges.gpart_mass.data;

    auto total_specie_count = part_counts.template map<u32>([&](u64 id, u32 count) {
        return count * ndust;
    });

    auto launch = [&](auto compute_kernel) {
        // call the kernel for each patches with part_counts.get(id_patch) * ndust threads of
        // patch id_patch
        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{
                edges.xyz.get_spans(),
                edges.hpart.get_spans(),
                edges.vxyz.get_spans(),
                edges.omega.get_spans(),
                edges.pressure.get_spans(),
                edges.s_j.get_spans(),
                edges.Ttilde_sj.get_spans(),
                edges.neigh_cache},
            sham::DDMultiRef{edges.ds_j_dt.get_spans()},
            total_specie_count,
            compute_kernel);
    };

    switch (dust_var) {
    case DustVariable::SqrtRhoEps:
        launch(
            KernelUpdateDerivsMonofluidTVA<Tvec, SPHKernel, DustVariable::SqrtRhoEps>{
                pmass, ndust});
        break;
    case DustVariable::Eps:
        launch(KernelUpdateDerivsMonofluidTVA<Tvec, SPHKernel, DustVariable::Eps>{pmass, ndust});
        break;
    case DustVariable::SqrtEpsOverOneMinusEps:
        launch(
            KernelUpdateDerivsMonofluidTVA<Tvec, SPHKernel, DustVariable::SqrtEpsOverOneMinusEps>{
                pmass, ndust});
        break;
    }
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<Tvec, SPHKernel>::_impl_get_tex()
    const {
    const std::string ndust_str         = std::to_string(ndust);
    const std::string kernel_radius_str = std::to_string(kernel_radius);

    std::string tex_common = R"tex(
        For gas particle $a$ and dust bin $j$:

        \begin{align}
        \rho_a &= \rho_h({gpart_mass}, {hpart}_a) \\
        \rho_b &= \rho_h({gpart_mass}, {hpart}_b) \\
        \mathbf{v}_{a,b} &= {vxyz}_a - {vxyz}_b \\
        \hat{\mathbf{r}}_{a,b} &= \frac{{xyz}_a - {xyz}_b}{\lvert {xyz}_a - {xyz}_b\rvert} \\
        F_{a,b}^a &= \mathrm{d}W_{3d}(\lvert \mathbf{r}_{a,b}\rvert, {hpart}_a) \\
        F_{a,b}^b &= \mathrm{d}W_{3d}(\lvert \mathbf{r}_{a,b}\rvert, {hpart}_b) \\
        \bar{F}_{a,b} &= \frac{F_{a,b}^a + F_{a,b}^b}{2}
        \end{align}
    )tex";

    std::string tex_var;
    switch (dust_var) {
    case DustVariable::SqrtRhoEps:
        tex_var = R"tex(
        NodeUpdateDerivsMonofluidTVA, evolved variable ${s_j} = \sqrt{\rho \epsilon_j}$
        (eq. 51, Hutchison 2018)
        )tex" + tex_common
                  + R"tex(
        \begin{align}
        T^{\rm w}_{j,a,b} &= \frac{{Ttilde_sj}_{j,a}}{\rho_a} + \frac{{Ttilde_sj}_{j,b}}{\rho_b}
        \\
        {ds_j_dt}_{j,a} &=
            -\frac{1}{2}\sum_{b \in \mathcal{N}(a)}
            {gpart_mass}\;
            \frac{{s_j}_{j,b}}{\rho_b}\;
            T^{\rm w}_{j,a,b}\;
            ({pressure}_a - {pressure}_b)\;
            \frac{\bar{F}_{a,b}}{\lvert \mathbf{r}_{a,b}\rvert}
        \\
        &\quad +
            \frac{{s_j}_{j,a}}{2\rho_a\,{omega}_a}
            \sum_{b \in \mathcal{N}(a)}
            {gpart_mass}\;
            \mathbf{v}_{a,b}\cdot\left(\hat{\mathbf{r}}_{a,b} F_{a,b}^a\right)
        \end{align}
        )tex";
        break;
    case DustVariable::Eps:
        tex_var = R"tex(
        NodeUpdateDerivsMonofluidTVA, evolved variable ${s_j} = \epsilon_j$
        (arithmetic pair coefficient, Price & Laibe 2015 direct form)
        )tex" + tex_common
                  + R"tex(
        \begin{align}
        {ds_j_dt}_{j,a} &=
            -\frac{1}{\rho_a}\sum_{b \in \mathcal{N}(a)}
            \frac{{gpart_mass}}{\rho_b}\;
            \left({s_j}_{j,a} {Ttilde_sj}_{j,a} + {s_j}_{j,b} {Ttilde_sj}_{j,b}\right)\;
            ({pressure}_a - {pressure}_b)\;
            \frac{\bar{F}_{a,b}}{\lvert \mathbf{r}_{a,b}\rvert}
        \end{align}
        )tex";
        break;
    case DustVariable::SqrtEpsOverOneMinusEps:
        tex_var = R"tex(
        NodeUpdateDerivsMonofluidTVA, evolved variable
        ${s_j} = \sqrt{\epsilon_j / (1 - \epsilon_j)}$ (per species Ballabio et al. 2018)
        )tex" + tex_common
                  + R"tex(
        \begin{align}
        1 - \epsilon_{j,a} &= \frac{1}{1 + {s_j}_{j,a}^2} \\
        D_{j,a} &= (1 - \epsilon_{j,a}) {Ttilde_sj}_{j,a} \\
        {ds_j_dt}_{j,a} &=
            -\frac{1}{2 \rho_a (1 - \epsilon_{j,a})^2}\sum_{b \in \mathcal{N}(a)}
            {gpart_mass}\;
            \frac{{s_j}_{j,b}}{\rho_b}\;
            \left(D_{j,a} + D_{j,b}\right)\;
            ({pressure}_a - {pressure}_b)\;
            \frac{\bar{F}_{a,b}}{\lvert \mathbf{r}_{a,b}\rvert}
        \end{align}
        )tex";
        break;
    }

    std::string tex = tex_var + R"tex(
        with the neighbor set $\mathcal{N}(a)$ defined by the kernel support:
        $\lvert \mathbf{r}_{a,b}\rvert \le \max({hpart}_a,{hpart}_b)\,{Rkern}$.

        $a \in [0,{part_counts})$, $j \in [0,{ndust})$.
    )tex";

    replace_edges_tex_symbols(tex);

    shambase::replace_all(tex, "{ndust}", ndust_str);
    shambase::replace_all(tex, "{Rkern}", kernel_radius_str);

    return tex;
}

using namespace shammath;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, M4>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, M6>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, M8>;

template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, C2>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, C4>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, C6>;
