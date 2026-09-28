// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeOmega.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/impl_variants.hpp"
#include "shammodels/sph/math/kernel_inv_h.hpp"
#include "shammodels/sph/math/sqrt_noerrno.hpp"
#include "shammodels/sph/modules/ComputeOmega.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeComputeOmega<Tvec, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    edges.omega.ensure_sizes(edges.part_counts.indexes);

    auto run = [&](auto reciprocal_tag, auto blocked_tag) {
        constexpr bool reciprocal = decltype(reciprocal_tag)::value;
        constexpr bool blocked    = decltype(blocked_tag)::value;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{
                edges.xyz.get_spans(), edges.hpart.get_spans(), edges.neigh_cache.neigh_cache},
            sham::DDMultiRef{edges.omega.get_spans()},
            edges.part_counts.indexes,
            [part_mass = this->part_mass, Rkern = kernel_radius](
                u32 id_a, const Tvec *r, const Tscal *hpart, const auto ploop_ptrs, Tscal *omega) {
                shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

                Tvec xyz_a = r[id_a]; // could be recovered from lambda

                Tscal h_a  = hpart[id_a];
                Tscal dint = h_a * h_a * Rkern * Rkern;

                Tscal rho_sum        = 0;
                Tscal part_omega_sum = 0;

                Tscal hinv_a = Tscal{1} / h_a;

                if constexpr (blocked) {
                    // same blocked evaluation as the smoothing length iteration (identical sums)
                    constexpr u32 block = 16;

                    const u32 cnt      = ploop_ptrs.cnt_neigh[id_a];
                    const u32 *neigh_b = ploop_ptrs.index_neigh_map + ploop_ptrs.scanned_cnt[id_a];

                    Tscal rab2_b[block];
                    Tscal dhw_b[block];

                    for (u32 b0 = 0; b0 < cnt; b0 += block) {
                        u32 n = sham::min(block, cnt - b0);

                        for (u32 t = 0; t < n; t++) {
                            Tvec dr   = xyz_a - r[neigh_b[b0 + t]];
                            rab2_b[t] = sycl::dot(dr, dr);
                        }

                        for (u32 t = 0; t < n; t++) {
                            Tscal rab2  = rab2_b[t];
                            bool inside = !(rab2 > dint);
                            Tscal rab   = shamrock::sph::sqrt_noerrno(rab2);

                            Tscal dhw;
                            if constexpr (reciprocal) {
                                dhw = shamrock::sph::KernelInvH<SPHKernel<Tscal>>::dhW_3d(
                                    rab, hinv_a);
                            } else {
                                dhw = SPHKernel<Tscal>::dhW_3d(rab, h_a);
                            }
                            dhw_b[t] = (inside) ? dhw : Tscal{0};
                        }

                        for (u32 t = 0; t < n; t++) {
                            part_omega_sum += part_mass * dhw_b[t];
                        }
                    }
                } else
                    particle_looper.for_each_object(id_a, [&](u32 id_b) {
                        Tvec dr    = xyz_a - r[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);

                        if (rab2 > dint) {
                            return;
                        }

                        Tscal rab = sycl::sqrt(rab2);

                        if constexpr (reciprocal) {
                            using KInv = shamrock::sph::KernelInvH<SPHKernel<Tscal>>;
                            rho_sum += part_mass * KInv::W_3d(rab, hinv_a);
                            part_omega_sum += part_mass * KInv::dhW_3d(rab, hinv_a);
                        } else {
                            rho_sum += part_mass * SPHKernel<Tscal>::W_3d(rab, h_a);
                            part_omega_sum += part_mass * SPHKernel<Tscal>::dhW_3d(rab, h_a);
                        }
                    });

                using namespace shamrock::sph;

                Tscal rho_ha  = rho_h(part_mass, h_a, SPHKernel<Tscal>::hfactd);
                Tscal omega_a = 1 + (h_a / (3 * rho_ha)) * part_omega_sum;
                omega[id_a]   = omega_a;
            });
    };

    // blocked evaluation that also tightens the neighbour lists : the lists are compacted in place
    // to the pairs within the kernel support of either particle (with the final smoothing
    // lengths), which are the only pairs used by the following neighbour loops (same criterion,
    // same order, hence identical results)
    auto run_tighten = [&](auto reciprocal_tag) {
        constexpr bool reciprocal = decltype(reciprocal_tag)::value;

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{edges.xyz.get_spans(), edges.hpart.get_spans()},
            sham::DDMultiRef{edges.omega.get_spans(), edges.neigh_cache.neigh_cache},
            edges.part_counts.indexes,
            [part_mass = this->part_mass, Rkern = kernel_radius](
                u32 id_a, const Tvec *r, const Tscal *hpart, Tscal *omega, auto ploop_ptrs) {
                constexpr Tscal Rker2 = SPHKernel<Tscal>::Rkern * SPHKernel<Tscal>::Rkern;

                Tvec xyz_a = r[id_a];

                Tscal h_a  = hpart[id_a];
                Tscal dint = h_a * h_a * Rkern * Rkern;

                // same expression as the neighbour loops using the lists
                const Tscal h_a_sq_rker2 = h_a * h_a * Rker2;

                Tscal part_omega_sum = 0;

                Tscal hinv_a = Tscal{1} / h_a;

                constexpr u32 block = 16;

                const u32 cnt = ploop_ptrs.cnt_neigh[id_a];
                u32 *neigh_b  = ploop_ptrs.index_neigh_map + ploop_ptrs.scanned_cnt[id_a];

                Tscal rab2_b[block];
                Tscal dhw_b[block];
                bool keep_b[block];

                u32 new_cnt = 0;

                for (u32 b0 = 0; b0 < cnt; b0 += block) {
                    u32 n = sham::min(block, cnt - b0);

                    for (u32 t = 0; t < n; t++) {
                        u32 id_b   = neigh_b[b0 + t];
                        Tvec dr    = xyz_a - r[id_b];
                        Tscal rab2 = sycl::dot(dr, dr);
                        Tscal h_b  = hpart[id_b];
                        rab2_b[t]  = rab2;
                        keep_b[t]  = !(rab2 > h_a_sq_rker2 && rab2 > h_b * h_b * Rker2);
                    }

                    for (u32 t = 0; t < n; t++) {
                        Tscal rab2  = rab2_b[t];
                        bool inside = !(rab2 > dint);
                        Tscal rab   = shamrock::sph::sqrt_noerrno(rab2);

                        Tscal dhw;
                        if constexpr (reciprocal) {
                            dhw = shamrock::sph::KernelInvH<SPHKernel<Tscal>>::dhW_3d(rab, hinv_a);
                        } else {
                            dhw = SPHKernel<Tscal>::dhW_3d(rab, h_a);
                        }
                        dhw_b[t] = (inside) ? dhw : Tscal{0};
                    }

                    for (u32 t = 0; t < n; t++) {
                        part_omega_sum += part_mass * dhw_b[t];
                    }

                    // in place, order preserving and branch free compaction (the write position
                    // never exceeds the read position)
                    for (u32 t = 0; t < n; t++) {
                        u32 id_b         = neigh_b[b0 + t];
                        neigh_b[new_cnt] = id_b;
                        new_cnt += u32(keep_b[t]);
                    }
                }

                ploop_ptrs.cnt_neigh[id_a] = new_cnt;

                using namespace shamrock::sph;

                Tscal rho_ha  = rho_h(part_mass, h_a, SPHKernel<Tscal>::hfactd);
                Tscal omega_a = 1 + (h_a / (3 * rho_ha)) * part_omega_sum;
                omega[id_a]   = omega_a;
            });
    };

    bool blocked = shammodels::sph::impl::use_blocked_neigh_evaluation();
    bool tighten
        = blocked
          && std::holds_alternative<shammodels::sph::impl::neigh_cache_tighten::AfterOmega>(
              shammodels::sph::impl::get_impl_neigh_cache_tighten());
    if (tighten) {
        if (shammodels::sph::impl::use_reciprocal_arithmetic()) {
            run_tighten(std::true_type{});
        } else {
            run_tighten(std::false_type{});
        }
    } else if (shammodels::sph::impl::use_reciprocal_arithmetic()) {
        if (blocked) {
            run(std::true_type{}, std::true_type{});
        } else {
            run(std::true_type{}, std::false_type{});
        }
    } else {
        if (blocked) {
            run(std::false_type{}, std::true_type{});
        } else {
            run(std::false_type{}, std::false_type{});
        }
    }
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeComputeOmega<Tvec, SPHKernel>::_impl_get_tex() const {
    return "TODO";
}

template<class T>
void shammodels::sph::modules::SetWhenMask<T>::_impl_evaluate_internal() {
    __shamrock_stack_entry();

    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    edges.mask.check_sizes(edges.part_counts.indexes);
    edges.field_to_set.ensure_sizes(edges.part_counts.indexes);

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{edges.mask.get_spans()},
        sham::DDMultiRef{edges.field_to_set.get_spans()},
        edges.part_counts.indexes,
        [val_to_set = this->val_to_set](u32 id, const u32 *mask, T *field_to_set) {
            if (mask[id] == 1) {
                field_to_set[id] = val_to_set;
            }
        });
}

template<class T>
std::string shammodels::sph::modules::SetWhenMask<T>::_impl_get_tex() const {
    return "TODO";
}

template class shammodels::sph::modules::SetWhenMask<f64>;

using namespace shammath;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, M4>;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, M6>;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, M8>;

template class shammodels::sph::modules::NodeComputeOmega<f64_3, C2>;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, C4>;
template class shammodels::sph::modules::NodeComputeOmega<f64_3, C6>;
