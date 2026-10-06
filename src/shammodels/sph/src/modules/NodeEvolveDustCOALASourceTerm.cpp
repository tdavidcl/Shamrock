// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeEvolveDustCOALASourceTerm.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/modules/NodeEvolveDustCOALASourceTerm.hpp"
#include "shamphys/coala_interface.hpp"
#include "shamrock/patch/PatchDataField.hpp" // IWYU pragma: keep
#include "shamsys/NodeInstance.hpp"
#include <experimental/mdspan>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace shammodels::sph::modules {

    /**
     * @brief Differential velocity between dust bins, evaluated on staged (local memory) data.
     *
     * This is the only place that defines the form of dv; the kernel below only requires
     * `dv(particle, l, m)`, nothing is assumed on its symmetry or on its diagonal.
     */
    template<class Tvec>
    struct CoalaDvMonofluid {
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal dv_max;

        // staged delta_v of the particles of the group, SoA: [particle * nbins + bin]
        const Tscal *vx;
        const Tscal *vy;
        const Tscal *vz;

        inline Tscal operator()(u32 off, u32 l, u32 m) const {
            // dv_lm = v_dust_m - v_dust_l = delta_v[m] - delta_v[l]
            Tscal dx  = vx[off + m] - vx[off + l];
            Tscal dy  = vy[off + m] - vy[off + l];
            Tscal dz  = vz[off + m] - vz[off + l];
            Tscal tmp = sycl::sqrt(dx * dx + dy * dy + dz * dz);
            return (tmp > dv_max) ? 0 : tmp;
        }
    };

    /**
     * @brief COALA k=0 source term.
     *
     * One work-group handles `ppg` particles. Per particle, g_j and delta_v are staged once in
     * local memory (coalesced loads, O(nbins) local memory per particle instead of
     * O(nbins * group_size)). Then, chunk by chunk, the group cooperatively evaluates
     * term(l,m) = dv(l,m) g_l g_m in local memory while each thread owns one output bin j and
     * accumulates S[j] = sum_p C[j,p] term[p] (coefficients coalesced along j, flux difference
     * already folded in, see shamphys::CoalaFluxDiffTable).
     */
    template<class Tvec, u32 ppg>
    struct KernelGenCoala_k0 {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 nbins;
        u32 npairs;
        u32 chunk;
        Tscal rho_eps;
        Tscal dv_max;
        u32 group_size;
        u32 n_groups;
        u32 true_size;

        auto operator()(
            u32 /**/,
            // common to all kernel calls
            const Tscal *__restrict massgrid_ptr,
            const Tscal *__restrict coeff,
            const u32 *__restrict pairs,
            // field specific data
            const Tscal *__restrict s_j,
            const Tvec *__restrict delta_v_j,
            Tscal *__restrict S_coag) const {

            auto range = sycl::nd_range<1>{n_groups * group_size, group_size};

            auto nbins      = this->nbins;
            auto npairs     = this->npairs;
            auto chunk      = this->chunk;
            auto rho_eps    = this->rho_eps;
            auto dv_max     = this->dv_max;
            auto group_size = this->group_size;
            auto true_size  = this->true_size;

            return [=](sycl::handler &cgh) {
                sycl::local_accessor<Tscal> g_acc{sycl::range<1>{ppg * nbins}, cgh};
                sycl::local_accessor<Tscal> vx_acc{sycl::range<1>{ppg * nbins}, cgh};
                sycl::local_accessor<Tscal> vy_acc{sycl::range<1>{ppg * nbins}, cgh};
                sycl::local_accessor<Tscal> vz_acc{sycl::range<1>{ppg * nbins}, cgh};
                sycl::local_accessor<Tscal> term_acc{sycl::range<1>{ppg * chunk}, cgh};

                cgh.parallel_for(range, [=](sycl::nd_item<1> tid) {
                    const u32 lid    = tid.get_local_linear_id();
                    const u64 first  = u64(tid.get_group_linear_id()) * ppg;
                    const u32 n_part = (first + ppg <= true_size) ? ppg : u32(true_size - first);

                    Tscal *g    = &g_acc[0];
                    Tscal *vx   = &vx_acc[0];
                    Tscal *vy   = &vy_acc[0];
                    Tscal *vz   = &vz_acc[0];
                    Tscal *term = &term_acc[0];

                    // stage g_j and delta_v of the group's particles (coalesced)
                    for (u32 idx = lid; idx < ppg * nbins; idx += group_size) {
                        u32 pp = idx / nbins;
                        u32 b  = idx - pp * nbins;

                        Tscal gj = 0;
                        Tvec v{0, 0, 0};
                        if (pp < n_part) {
                            u64 gid   = (first + pp) * nbins + b;
                            Tscal s   = s_j[gid];
                            Tscal rho = s * s;
                            gj = (rho > rho_eps) ? rho / (massgrid_ptr[b + 1] - massgrid_ptr[b])
                                                 : 0;
                            v  = delta_v_j[gid];
                        }
                        g[idx]  = gj;
                        vx[idx] = v[0];
                        vy[idx] = v[1];
                        vz[idx] = v[2];
                    }
                    tid.barrier(sycl::access::fence_space::local_space);

                    CoalaDvMonofluid<Tvec> dv{dv_max, vx, vy, vz};

                    // output bins are spread over threads; nbins > group_size loops again
                    for (u32 j0 = 0; j0 < nbins; j0 += group_size) {
                        // recomputed at each use rather than kept alive across the barriers below:
                        // the AdaptiveCpp OpenMP work-item splitting collapsed it to lane 0
                        auto out_bin = [&]() -> u32 {
                            return j0 + u32(tid.get_local_linear_id());
                        };

                        Tscal acc[ppg];
                        for (u32 pp = 0; pp < ppg; ++pp) {
                            acc[pp] = 0;
                        }

                        for (u32 p0 = 0; p0 < npairs; p0 += chunk) {
                            const u32 cnt = (npairs - p0 < chunk) ? (npairs - p0) : chunk;

                            // cooperative evaluation of term(l,m) for the chunk
                            for (u32 idx = lid; idx < ppg * cnt; idx += group_size) {
                                u32 pp   = idx / cnt;
                                u32 c    = idx - pp * cnt;
                                u32 pair = pairs[p0 + c];
                                u32 l    = pair & 0xffff;
                                u32 m    = pair >> 16;
                                u32 off  = pp * nbins;

                                Tscal gg = g[off + l] * g[off + m];
                                // dv is only evaluated when it can matter (finite dv assumed)
                                term[pp * chunk + c] = (gg != 0) ? dv(off, l, m) * gg : Tscal(0);
                            }
                            tid.barrier(sycl::access::fence_space::local_space);

                            if (out_bin() < nbins) {
                                for (u32 c = 0; c < cnt; ++c) {
                                    Tscal t[ppg];
                                    bool any = false;
                                    for (u32 pp = 0; pp < ppg; ++pp) {
                                        t[pp] = term[pp * chunk + c];
                                        any   = any || (t[pp] != 0);
                                    }
                                    // uniform over the group: skips the coefficient load
                                    if (!any) {
                                        continue;
                                    }
                                    Tscal C = coeff[u64(p0 + c) * nbins + out_bin()];
                                    for (u32 pp = 0; pp < ppg; ++pp) {
                                        acc[pp] += C * t[pp];
                                    }
                                }
                            }
                            tid.barrier(sycl::access::fence_space::local_space);
                        }

                        if (out_bin() < nbins) {
                            for (u32 pp = 0; pp < ppg; ++pp) {
                                if (pp < n_part) {
                                    S_coag[(first + pp) * nbins + out_bin()] = acc[pp];
                                }
                            }
                        }
                    }
                });
            };
        }
    };

    template<class Tvec>
    inline void NodeEvolveDustCOALASourceTerm<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        auto s_j_spans       = edges.s_j.get_spans();
        auto delta_v_j_spans = edges.delta_v_j.get_spans();

        auto counts = edges.part_counts.indexes;

        edges.S_coag.ensure_sizes(counts);
        auto S_coag_spans = edges.S_coag.get_spans();

        Tscal rho_eps                                 = edges.rhodust_eps.data;
        Tscal dv_max                                  = edges.dv_max.data;
        const std::vector<Tscal> &massgrid            = edges.massgrid.data;
        const std::vector<Tscal> &tensor_tabflux_coag = edges.tensor_tabflux_coag.data;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = shambase::get_check_ref(dev_sched).get_queue();

        // fold the flux difference into the tensor and drop the all-zero (l,m) pairs
        auto table = shamphys::build_coala_flux_diff_table<Tscal>(
            nbins,
            std::mdspan<const Tscal, std::dextents<u32, 3>>(
                tensor_tabflux_coag.data(), nbins, nbins, nbins));

        sham::DeviceBuffer<Tscal> massgrid_buf(nbins + 1, dev_sched);
        massgrid_buf.copy_from_stdvec(massgrid);

        // keep buffers non-empty even for a fully null tensor
        sham::DeviceBuffer<Tscal> coeff_buf(std::max<u32>(table.coeff.size(), 1), dev_sched);
        sham::DeviceBuffer<u32> pairs_buf(std::max<u32>(table.pairs.size(), 1), dev_sched);
        if (table.npairs > 0) {
            coeff_buf.copy_from_stdvec(table.coeff);
            pairs_buf.copy_from_stdvec(table.pairs);
        }

        // threads own output bins: one thread per bin, multiple of 32, capped
        const u32 group_size = std::min<u32>(256, ((nbins + 31) / 32) * 32);

        // local memory budget: stay well below the device limit to keep occupancy
        const u64 lmem_budget
            = std::min<u64>(q.get_device_prop().local_mem_size, 48 * 1024) * 2 / 3;

        // particles per group (as many as fit), then shrink the pair chunk if needed
        auto lmem_bytes = [&](u32 ppg, u32 chunk) {
            return u64(ppg) * (4 * u64(nbins) + chunk) * sizeof(Tscal);
        };
        u32 chunk = std::max<u32>(1, std::min<u32>(256, table.npairs));
        u32 ppg   = 8;
        while (ppg > 1 && lmem_bytes(ppg, chunk) > lmem_budget) {
            ppg /= 2;
        }
        while (chunk > 1 && lmem_bytes(ppg, chunk) > lmem_budget) {
            chunk /= 2;
        }
        if (lmem_bytes(ppg, chunk) > q.get_device_prop().local_mem_size) {
            shambase::throw_with_loc<std::runtime_error>(sham::format(
                "COALA source term: nbins={} needs {} of local memory per group",
                nbins,
                shambase::readable_sizeof(lmem_bytes(ppg, chunk))));
        }

        counts.for_each([&](u64 id_patch, u64 count) {
            if (count == 0) {
                return;
            }

            auto launch = [&]<u32 PPG>() {
                sham::kernel_call_hndl(
                    q,
                    sham::MultiRef{
                        massgrid_buf,
                        coeff_buf,
                        pairs_buf,
                        s_j_spans.get(id_patch),
                        delta_v_j_spans.get(id_patch)},
                    sham::MultiRef{S_coag_spans.get(id_patch)},
                    count,
                    KernelGenCoala_k0<Tvec, PPG>{
                        .nbins      = nbins,
                        .npairs     = table.npairs,
                        .chunk      = chunk,
                        .rho_eps    = rho_eps,
                        .dv_max     = dv_max,
                        .group_size = group_size,
                        .n_groups   = shambase::group_count(u32(count), PPG),
                        .true_size  = u32(count)});
            };

            switch (ppg) {
            case 8 : launch.template operator()<8>(); break;
            case 4 : launch.template operator()<4>(); break;
            case 2 : launch.template operator()<2>(); break;
            default: launch.template operator()<1>(); break;
            }
        });
    }

    template<class Tvec>
    std::string NodeEvolveDustCOALASourceTerm<Tvec>::_impl_get_tex() const {
        std::string tex = R"tex(
            COALA dust coagulation source term, DG $k=0$ (Lombart et al., 2021)

            Per gas particle $a$ and mass bin $j$ (monofluid: $\rho_{{\rm d},j,a} = {s_j}_{j,a}^2$):

            \begin{align}
            \rho_{{\rm d},j,a} &= {s_j}_{j,a}^2 \\
            \Delta m_j &= {massgrid}_{j+1} - {massgrid}_j \\
            g_{j,a} &= \begin{cases}
                \rho_{{\rm d},j,a} / \Delta m_j & \rho_{{\rm d},j,a} > \rho_{\rm eps} \\
                0 & \text{otherwise}
            \end{cases} \\
            \mathrm{dv}_{l,m,a} &= \left| {delta_v_j}_{m,a} - {delta_v_j}_{l,a} \right| \\
            \mathrm{flux}_{j,a} &= \sum_{l,m}
                {tensor_tabflux_coag}_{j,l,m}\,
                \mathrm{dv}_{l,m,a}\, g_{l,a}\, g_{m,a} \\
            {S_coag}_{0,a} &= -\mathrm{flux}_{0,a}, \quad
            {S_coag}_{j,a} = \mathrm{flux}_{j-1,a} - \mathrm{flux}_{j,a}
            \quad (j \ge 1) \\
            a &\in [0, {part_counts}), \quad j,l,m \in [0, N_{\rm bins}) \\
            \rho_{\rm eps} &= {rhodust_eps}, \quad N_{\rm bins} = {nbins}
            \end{align}
        )tex";

        replace_edges_tex_symbols(tex);

        shambase::replace_all(tex, "{nbins}", sham::format("{}", nbins));

        return tex;
    }
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::NodeEvolveDustCOALASourceTerm<f64_3>;
