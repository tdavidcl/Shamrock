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
#include "shambase/mdspan_func_accessor.hpp"
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
#include <vector>

namespace shammodels::sph::modules {

    template<class Tvec>
    struct KernelGenCoala_k0 {
        using Tscal = shambase::VecComponent<Tvec>;

        using mdspan_rank_1       = std::mdspan<Tscal, std::dextents<u32, 1>>;
        using const_mdspan_rank_1 = std::mdspan<const Tscal, std::dextents<u32, 1>>;

        /// number of m per block of the sparse tabflux (see shamphys::TabfluxCoagK0Sparse)
        static constexpr u32 tabflux_block_size = 4;

        u32 nbins;
        Tscal rho_eps;
        Tscal dv_max;
        u32 corrected_len;
        u32 group_size;
        u32 true_size;

        auto operator()(
            u32 /**/,
            // common to all kernel calls
            const Tscal *__restrict inv_dm,
            const u32 *__restrict tabflux_block_offset,
            const u32 *__restrict tabflux_block_jmin,
            const Tscal *__restrict tabflux_values,
            // field specific data
            const Tscal *__restrict s_j,
            const Tvec *__restrict delta_v_j,
            Tscal *__restrict S_coag) const {

            auto range = sycl::nd_range<1>{corrected_len, group_size};

            auto local_acc_sz_nbins = sycl::range<1>{group_size * nbins};

            auto true_size = this->true_size;
            auto rho_eps   = this->rho_eps;
            auto dv_max    = this->dv_max;

            return [=, nbins = this->nbins](sycl::handler &cgh) {
                auto flux_acc = sycl::local_accessor<Tscal>{local_acc_sz_nbins, cgh};

                cgh.parallel_for(range, [=](sycl::nd_item<1> tid) {
                    const u64 id_a = tid.get_global_linear_id();
                    const u64 lid  = tid.get_local_linear_id();

                    if (id_a >= true_size) {
                        return;
                    }

                    u32 id_a_d = id_a * nbins;

                    /* inputs */
                    shamphys::TabfluxCoagK0SparseView<Tscal, tabflux_block_size> tabflux_coag{
                        tabflux_block_offset, tabflux_block_jmin, tabflux_values};

                    /* internal */
                    auto flux_loc = &(flux_acc[nbins * lid]);

                    mdspan_rank_1 flux(flux_loc, nbins);

                    /* output */
                    mdspan_rank_1 S_coag_span(S_coag + id_a_d, nbins);

                    /* lambda getters */
                    auto rho_dust = [&](int j) {
                        auto tmp = s_j[id_a_d + j];
                        return tmp * tmp;
                    };

                    auto dv = [&, delta_v = delta_v_j + id_a_d](int i, int j) {
                        // dv_ij = v_dust_j - v_dust_i = delta_v_j[j] - delta_v_j[i]
                        auto tmp = sycl::length(delta_v[j] - delta_v[i]);
                        return (tmp > dv_max) ? 0 : tmp;
                    };

                    // gij is not stored but recomputed from s_j on access (see
                    // shamphys::compute_gij_k0), the flux helper reading it O(nbins^2 / B) times
                    auto gij = shambase::make_func_mdspan_rank_1(nbins, [&](std::size_t j) {
                        return shamphys::gij_k0_inv_dm<Tscal>(rho_dust(j), rho_eps, inv_dm[j]);
                    });

                    // should implement the same content as
                    // src/pylib/shamrock/external/coala/interface_coala_shamrock.py

                    shamphys::compute_flux_coag_k0_kdv(nbins, gij, tabflux_coag, dv, flux);
                    shamphys::coala_flux_diff(flux, S_coag_span);
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

        // inverse of the bin widths, to compute gij without divisions in the kernel
        std::vector<Tscal> inv_dm(nbins);
        for (u32 j = 0; j < nbins; j++) {
            inv_dm[j] = 1 / (massgrid[j + 1] - massgrid[j]);
        }
        sham::DeviceBuffer<Tscal> inv_dm_buf(nbins, dev_sched);
        inv_dm_buf.copy_from_stdvec(inv_dm);

        // only the non-zero part of the tensor is used on device
        auto tabflux_sparse = shamphys::
            make_tabflux_coag_k0_sparse<Tscal, KernelGenCoala_k0<Tvec>::tabflux_block_size>(
                nbins,
                std::mdspan<const Tscal, std::dextents<u32, 3>>(
                    tensor_tabflux_coag.data(), nbins, nbins, nbins));

        sham::DeviceBuffer<u32> tabflux_block_offset_buf(
            tabflux_sparse.block_offset.size(), dev_sched);
        tabflux_block_offset_buf.copy_from_stdvec(tabflux_sparse.block_offset);

        sham::DeviceBuffer<u32> tabflux_block_jmin_buf(tabflux_sparse.block_jmin.size(), dev_sched);
        tabflux_block_jmin_buf.copy_from_stdvec(tabflux_sparse.block_jmin);

        sham::DeviceBuffer<Tscal> tabflux_values_buf(tabflux_sparse.values.size(), dev_sched);
        tabflux_values_buf.copy_from_stdvec(tabflux_sparse.values);

        // per thread local memory: flux, one per bin
        usize local_mem_per_thread = nbins * sizeof(Tscal);
        usize local_mem_size       = q.get_device_prop().local_mem_size;

        u32 group_size = 64;
        while (group_size > 1 && group_size * local_mem_per_thread > local_mem_size) {
            group_size /= 2;
        }
        if (group_size * local_mem_per_thread > local_mem_size) {
            shambase::throw_with_loc<std::runtime_error>(shambase::format(
                "COALA kernel: not enough local memory for nbins = {} ({} B per thread, {} B "
                "available)",
                nbins,
                local_mem_per_thread,
                local_mem_size));
        }

        counts.for_each([&](u64 id_patch, u64 count) {
            u32 group_cnt     = shambase::group_count(count, group_size);
            u32 corrected_len = group_cnt * group_size;

            sham::kernel_call_hndl(
                q,
                sham::MultiRef{
                    inv_dm_buf,
                    tabflux_block_offset_buf,
                    tabflux_block_jmin_buf,
                    tabflux_values_buf,
                    s_j_spans.get(id_patch),
                    delta_v_j_spans.get(id_patch)},
                sham::MultiRef{S_coag_spans.get(id_patch)},
                count,
                KernelGenCoala_k0<Tvec>{
                    .nbins         = nbins,
                    .rho_eps       = rho_eps,
                    .dv_max        = dv_max,
                    .corrected_len = corrected_len,
                    .group_size    = group_size,
                    .true_size     = u32(count)});
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
