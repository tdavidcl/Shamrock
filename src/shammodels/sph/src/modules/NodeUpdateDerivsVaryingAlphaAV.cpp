// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeUpdateDerivsVaryingAlphaAV.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/flatten.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/impl_variants.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shammodels/sph/math/kernel_inv_h.hpp"
#include "shammodels/sph/math/q_ab.hpp"
#include "shammodels/sph/math/sqrt_noerrno.hpp"
#include "shammodels/sph/modules/NodeUpdateDerivsVaryingAlphaAV.hpp"
#include "shamrock/patch/PatchDataField.hpp"

template<
    class Tvec,
    template<class> class SPHKernel,
    bool compute_vsig_cfl,
    bool reciprocal,
    bool blocked>
struct KernelUpdateDerivsVaryingAlphaAV {
    using Tscal                   = shambase::VecComponent<Tvec>;
    using Kernel                  = SPHKernel<Tscal>;
    static constexpr Tscal hfactd = Kernel::hfactd;
    static constexpr Tscal Rkern  = Kernel::Rkern;
    static constexpr Tscal Rker2  = Rkern * Rkern;

    Tscal pmass;
    Tscal alpha_u;
    Tscal beta_AV;

    inline void operator()(
        unsigned int id_a,
        const Tvec *__restrict xyz,
        const Tscal *__restrict hpart,
        const Tvec *__restrict vxyz,
        const Tscal *__restrict uint,
        const Tscal *__restrict omega,
        const Tscal *__restrict pressure,
        const Tscal *__restrict cs,
        const Tscal *__restrict alpha_AV,
        shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
        Tvec *__restrict axyz,
        Tscal *__restrict duint) const
        requires(!compute_vsig_cfl)
    {
        compute(
            id_a,
            xyz,
            hpart,
            vxyz,
            uint,
            omega,
            pressure,
            cs,
            alpha_AV,
            ploop_ptrs,
            axyz,
            duint,
            nullptr);
    }

    inline void operator()(
        unsigned int id_a,
        const Tvec *__restrict xyz,
        const Tscal *__restrict hpart,
        const Tvec *__restrict vxyz,
        const Tscal *__restrict uint,
        const Tscal *__restrict omega,
        const Tscal *__restrict pressure,
        const Tscal *__restrict cs,
        const Tscal *__restrict alpha_AV,
        shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
        Tvec *__restrict axyz,
        Tscal *__restrict duint,
        Tscal *__restrict vsig_cfl) const
        requires(compute_vsig_cfl)
    {
        compute(
            id_a,
            xyz,
            hpart,
            vxyz,
            uint,
            omega,
            pressure,
            cs,
            alpha_AV,
            ploop_ptrs,
            axyz,
            duint,
            vsig_cfl);
    }

    inline void compute(
        unsigned int id_a,
        const Tvec *__restrict xyz,
        const Tscal *__restrict hpart,
        const Tvec *__restrict vxyz,
        const Tscal *__restrict uint,
        const Tscal *__restrict omega,
        const Tscal *__restrict pressure,
        const Tscal *__restrict cs,
        const Tscal *__restrict alpha_AV,
        shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
        Tvec *__restrict axyz,
        Tscal *__restrict duint,
        Tscal *__restrict vsig_cfl) const {

        using namespace shamrock::sph;

        shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

        Tvec xyz_a    = xyz[id_a];
        Tscal h_a     = hpart[id_a];
        Tvec vxyz_a   = vxyz[id_a];
        Tscal u_a     = uint[id_a];
        Tscal omega_a = omega[id_a];
        Tscal P_a     = pressure[id_a];
        Tscal cs_a    = cs[id_a];
        Tscal alpha_a = alpha_AV[id_a];

        Tscal rho_a     = rho_h(pmass, h_a, hfactd);
        Tscal rho_a_sq  = rho_a * rho_a;
        Tscal rho_a_inv = 1. / rho_a;

        Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

        // reciprocal arithmetic : quantities of a hoisted out of the neighbour loop
        Tscal hinv_a               = Tscal{1} / h_a;
        Tscal inv_rho_a_sq_omega_a = sham::inv_sat_zero(rho_a_sq * omega_a);
        // 1/rho = h^3 / (m hfact^3)
        const Tscal inv_m_hfact3 = Tscal{1} / (pmass * hfactd * hfactd * hfactd);

        Tvec force_pressure  = Tvec{0, 0, 0};
        Tscal tmpdU_pressure = Tscal{0};

        // signal velocity of the courant CFL, same expression as the "compute vsig" loop of the
        // solver (fixed alpha = 1, beta = 2, and r_ab_unit = dr / rab) to get identical results
        Tscal vsig_cfl_max = 0;

        // contribution of the pair (a, b) with the reciprocals arithmetic, from the quantities
        // depending only on the separation and h_b : 1/h_b, the saturated inverse of r_ab, and
        // the kernel derivatives (shared by the scalar and the blocked loops)
        // quantities of b depending only on h_b, omega_b and P_b (same expressions as
        // shamrock::sph::vsig_u, with a square root that does not prevent the vectorization)
        auto b_quantities = [&](Tscal h_b,
                                Tscal hinv_b,
                                Tscal omega_b,
                                Tscal P_b,
                                Tscal &rho_b,
                                Tscal &rho_b_inv,
                                Tscal &omega_b_inv,
                                Tscal &vsig_u) SHAM_FLATTEN {
            Tscal hfact_hinv_b = hfactd * hinv_b;
            rho_b              = pmass * (hfact_hinv_b * hfact_hinv_b * hfact_hinv_b);
            rho_b_inv          = (h_b * h_b * h_b) * inv_m_hfact3;
            omega_b_inv        = Tscal{1} / omega_b;

            Tscal rho_avg = (rho_a + rho_b) * 0.5;
            Tscal abs_dp  = sham::abs(P_a - P_b);
            vsig_u        = shamrock::sph::sqrt_noerrno(abs_dp / rho_avg);
        };

        auto add_pair_reciprocal = [&](u32 id_b,
                                       const Tvec &dr,
                                       Tscal inv_rab,
                                       Tscal Fab_a,
                                       Tscal Fab_b,
                                       Tscal omega_b,
                                       Tscal P_b,
                                       Tscal rho_b,
                                       Tscal rho_b_inv,
                                       Tscal omega_b_inv,
                                       Tscal vsig_u) SHAM_FLATTEN {
            Tvec vxyz_b         = vxyz[id_b];
            const Tscal u_b     = uint[id_b];
            const Tscal alpha_b = alpha_AV[id_b];
            Tscal cs_b          = cs[id_b];

            Tvec v_ab = vxyz_a - vxyz_b;

            Tvec r_ab_unit = dr * inv_rab;

            Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
            Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

            Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;
            Tscal vsig_b = alpha_b * cs_b + beta_AV * abs_v_ab_r_ab;

            Tscal qa_ab = shamrock::sph::q_av(rho_a, vsig_a, v_ab_r_ab);
            Tscal qb_ab = shamrock::sph::q_av(rho_b, vsig_b, v_ab_r_ab);

            // same as add_to_derivs_sph_artif_visco_cond, with the divisions by rho_b and
            // omega_b replaced by multiplications by their inverses
            Tscal AV_P_a = P_a + qa_ab;
            Tscal AV_P_b = P_b + qb_ab;

            // same semantic as sham::inv_sat_zero(rho_b^2 omega_b) (rho_b > 0)
            Tscal inv_rho_b_sq_omega_b = (omega_b != Tscal{0} && omega_b == omega_b)
                                             ? rho_b_inv * rho_b_inv * omega_b_inv
                                             : Tscal{0};

            Tvec nabla_Wab_ha = r_ab_unit * Fab_a;
            Tvec nabla_Wab_hb = r_ab_unit * Fab_b;

            force_pressure += -pmass
                              * ((AV_P_a * inv_rho_a_sq_omega_a) * nabla_Wab_ha
                                 + (AV_P_b * inv_rho_b_sq_omega_b) * nabla_Wab_hb);

            tmpdU_pressure += duint_dt_pressure(
                pmass, AV_P_a, omega_a_rho_a_inv * rho_a_inv, v_ab, nabla_Wab_ha);

            tmpdU_pressure += lambda_shock_conductivity(
                pmass,
                alpha_u,
                vsig_u,
                u_a - u_b,
                Fab_a * omega_a_rho_a_inv,
                Fab_b * (rho_b_inv * omega_b_inv));

            if constexpr (compute_vsig_cfl) {
                // same signal velocity as the dedicated CFL loop (alpha = 1, beta = 2), up to
                // the rounding of r_ab_unit
                Tscal vsig_cfl_a = cs_a + Tscal{2} * abs_v_ab_r_ab;
                vsig_cfl_max     = sycl::fmax(vsig_cfl_max, vsig_cfl_a);
            }
        };

        if constexpr (blocked && reciprocal) {
            using KInv = shamrock::sph::KernelInvH<Kernel>;

            // neighbours processed by blocks : a gather loop computes the separations, a branch
            // free (vectorizable) loop evaluates the square roots, inverses and kernel
            // derivatives, then the interacting pairs are accumulated in the neighbour order with
            // the same expressions as the scalar loop (identical sums)
            constexpr u32 block = 16;

            const u32 cnt      = ploop_ptrs.cnt_neigh[id_a];
            const u32 *neigh_b = ploop_ptrs.index_neigh_map + ploop_ptrs.scanned_cnt[id_a];

            const Tscal h_a_sq_rker2 = h_a * h_a * Rker2;

            Tscal dx_b[block], dy_b[block], dz_b[block], rab2_b[block], h_b_b[block];
            Tscal omega_b_b[block], P_b_b[block];
            Tscal inv_rab_b[block], fab_a_b[block], fab_b_b[block];
            Tscal rho_b_b[block], rho_b_inv_b[block], omega_b_inv_b[block], vsig_u_b[block];
            bool inside_b[block];

            for (u32 b0 = 0; b0 < cnt; b0 += block) {
                u32 n = sham::min(block, cnt - b0);

                for (u32 t = 0; t < n; t++) {
                    Tvec dr      = xyz_a - xyz[neigh_b[b0 + t]];
                    dx_b[t]      = dr.x();
                    dy_b[t]      = dr.y();
                    dz_b[t]      = dr.z();
                    rab2_b[t]    = sycl::dot(dr, dr);
                    h_b_b[t]     = hpart[neigh_b[b0 + t]];
                    omega_b_b[t] = omega[neigh_b[b0 + t]];
                    P_b_b[t]     = pressure[neigh_b[b0 + t]];
                }

                for (u32 t = 0; t < n; t++) {
                    Tscal rab2   = rab2_b[t];
                    Tscal h_b    = h_b_b[t];
                    inside_b[t]  = !(rab2 > h_a_sq_rker2 && rab2 > h_b * h_b * Rker2);
                    Tscal rab    = shamrock::sph::sqrt_noerrno(rab2);
                    Tscal hinv_b = Tscal{1} / h_b;
                    inv_rab_b[t] = sham::inv_sat_positive(rab);
                    fab_a_b[t]   = KInv::dW_3d(rab, hinv_a);
                    fab_b_b[t]   = KInv::dW_3d(rab, hinv_b);
                    b_quantities(
                        h_b,
                        hinv_b,
                        omega_b_b[t],
                        P_b_b[t],
                        rho_b_b[t],
                        rho_b_inv_b[t],
                        omega_b_inv_b[t],
                        vsig_u_b[t]);
                }

                for (u32 t = 0; t < n; t++) {
                    if (!inside_b[t]) {
                        continue;
                    }
                    add_pair_reciprocal(
                        neigh_b[b0 + t],
                        Tvec{dx_b[t], dy_b[t], dz_b[t]},
                        inv_rab_b[t],
                        fab_a_b[t],
                        fab_b_b[t],
                        omega_b_b[t],
                        P_b_b[t],
                        rho_b_b[t],
                        rho_b_inv_b[t],
                        omega_b_inv_b[t],
                        vsig_u_b[t]);
                }
            }
        } else
            particle_looper.for_each_object(id_a, [&](u32 id_b) {
                Tvec dr    = xyz_a - xyz[id_b];
                Tscal rab2 = sycl::dot(dr, dr);
                Tscal h_b  = hpart[id_b];

                if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                    return;
                }

                Tscal rab = sycl::sqrt(rab2);

                if constexpr (reciprocal) {
                    using KInv = shamrock::sph::KernelInvH<Kernel>;

                    Tscal hinv_b  = Tscal{1} / h_b;
                    Tscal omega_b = omega[id_b];
                    Tscal P_b     = pressure[id_b];
                    Tscal rho_b, rho_b_inv, omega_b_inv, vsig_u;
                    b_quantities(h_b, hinv_b, omega_b, P_b, rho_b, rho_b_inv, omega_b_inv, vsig_u);
                    add_pair_reciprocal(
                        id_b,
                        dr,
                        sham::inv_sat_positive(rab),
                        KInv::dW_3d(rab, hinv_a),
                        KInv::dW_3d(rab, hinv_b),
                        omega_b,
                        P_b,
                        rho_b,
                        rho_b_inv,
                        omega_b_inv,
                        vsig_u);
                    return;
                }

                Tvec vxyz_b         = vxyz[id_b];
                const Tscal u_b     = uint[id_b];
                Tscal P_b           = pressure[id_b];
                Tscal omega_b       = omega[id_b];
                const Tscal alpha_b = alpha_AV[id_b];
                Tscal cs_b          = cs[id_b];

                Tscal rho_b = rho_h(pmass, h_b, hfactd);

                Tscal Fab_a = Kernel::dW_3d(rab, h_a);
                Tscal Fab_b = Kernel::dW_3d(rab, h_b);

                Tvec v_ab = vxyz_a - vxyz_b;

                Tvec r_ab_unit = dr * sham::inv_sat_positive(rab);

                Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
                Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);

                Tscal vsig_a = alpha_a * cs_a + beta_AV * abs_v_ab_r_ab;
                Tscal vsig_b = alpha_b * cs_b + beta_AV * abs_v_ab_r_ab;

                Tscal vsig_u = shamrock::sph::vsig_u(P_a, P_b, rho_a, rho_b);

                Tscal qa_ab = shamrock::sph::q_av(rho_a, vsig_a, v_ab_r_ab);
                Tscal qb_ab = shamrock::sph::q_av(rho_b, vsig_b, v_ab_r_ab);

                add_to_derivs_sph_artif_visco_cond(
                    pmass,
                    rho_a_sq,
                    omega_a_rho_a_inv,
                    rho_a_inv,
                    rho_b,
                    omega_a,
                    omega_b,
                    Fab_a,
                    Fab_b,
                    u_a,
                    u_b,
                    P_a,
                    P_b,
                    alpha_u,
                    v_ab,
                    r_ab_unit,
                    vsig_u,
                    qa_ab,
                    qb_ab,

                    force_pressure,
                    tmpdU_pressure);

                if constexpr (compute_vsig_cfl) {
                    Tvec r_ab_unit_cfl = dr / rab;
                    if (rab < 1e-9) {
                        r_ab_unit_cfl = {0, 0, 0};
                    }
                    Tscal abs_v_ab_r_ab_cfl = sycl::fabs(sycl::dot(v_ab, r_ab_unit_cfl));
                    Tscal vsig_cfl_a        = Tscal{1} * cs_a + Tscal{2} * abs_v_ab_r_ab_cfl;
                    vsig_cfl_max            = sycl::fmax(vsig_cfl_max, vsig_cfl_a);
                }
            });

        axyz[id_a]  = force_pressure;
        duint[id_a] = tmpdU_pressure;

        if constexpr (compute_vsig_cfl) {
            vsig_cfl[id_a] = vsig_cfl_max;
        }
    }
};

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV<Tvec, SPHKernel>::
    _impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts_with_ghost = edges.part_counts_with_ghost.indexes;
    auto &part_counts            = edges.part_counts.indexes;

    // check that all input edges have the particles with ghosts zones
    edges.xyz.check_sizes(part_counts_with_ghost);
    edges.hpart.check_sizes(part_counts_with_ghost);
    edges.vxyz.check_sizes(part_counts_with_ghost);
    edges.uint.check_sizes(part_counts_with_ghost);
    edges.omega.check_sizes(part_counts_with_ghost);
    edges.pressure.check_sizes(part_counts_with_ghost);
    edges.cs.check_sizes(part_counts_with_ghost);
    edges.alpha_AV.check_sizes(part_counts_with_ghost);

    // ensure that the output edges are of size part_counts (output without ghosts zones)
    edges.axyz.ensure_sizes(part_counts);
    edges.duint.ensure_sizes(part_counts);

    const Tscal pmass   = edges.gpart_mass.data;
    const Tscal alpha_u = edges.alpha_u.data;
    const Tscal beta_AV = edges.beta_AV.data;

    auto inputs = sham::DDMultiRef{
        edges.xyz.get_spans(),
        edges.hpart.get_spans(),
        edges.vxyz.get_spans(),
        edges.uint.get_spans(),
        edges.omega.get_spans(),
        edges.pressure.get_spans(),
        edges.cs.get_spans(),
        edges.alpha_AV.get_spans(),
        edges.neigh_cache};

    // call the kernel for each patches with part_counts.get(id_patch) threads of patch id_patch
    auto run = [&](auto reciprocal_tag, auto blocked_tag) {
        constexpr bool reciprocal = decltype(reciprocal_tag)::value;
        constexpr bool blocked    = decltype(blocked_tag)::value;

        if (edges.vsig_cfl.has_value()) {
            auto &vsig_cfl = edges.vsig_cfl.value().get();
            vsig_cfl.ensure_sizes(part_counts);

            using ComputeKernel
                = KernelUpdateDerivsVaryingAlphaAV<Tvec, SPHKernel, true, reciprocal, blocked>;
            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                inputs,
                sham::DDMultiRef{
                    edges.axyz.get_spans(), edges.duint.get_spans(), vsig_cfl.get_spans()},
                part_counts,
                ComputeKernel{pmass, alpha_u, beta_AV});
        } else {
            using ComputeKernel
                = KernelUpdateDerivsVaryingAlphaAV<Tvec, SPHKernel, false, reciprocal, blocked>;
            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                inputs,
                sham::DDMultiRef{edges.axyz.get_spans(), edges.duint.get_spans()},
                part_counts,
                ComputeKernel{pmass, alpha_u, beta_AV});
        }
    };

    if (shammodels::sph::impl::use_reciprocal_arithmetic()) {
        // the blocked evaluation is only implemented for the reciprocals arithmetic
        if (std::holds_alternative<shammodels::sph::impl::derivs_evaluation::Blocked>(
                shammodels::sph::impl::get_impl_derivs_evaluation())) {
            run(std::true_type{}, std::true_type{});
        } else {
            run(std::true_type{}, std::false_type{});
        }
    } else {
        run(std::false_type{}, std::false_type{});
    }
}

using namespace shammath;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV<f64_3, M4>;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV<f64_3, M6>;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV<f64_3, M8>;

template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV<f64_3, C2>;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV<f64_3, C4>;
template class shammodels::sph::modules::NodeUpdateDerivsVaryingAlphaAV<f64_3, C6>;
