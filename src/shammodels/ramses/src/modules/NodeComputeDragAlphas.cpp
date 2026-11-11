// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeComputeDragAlphas.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Compute the dust collision rates (inverse stopping times) of each cell, read by the drag
 * integrators.
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/ramses/modules/NodeComputeDragAlphas.hpp"
#include "shamphys/Dust.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeComputeDragAlphas<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_press.check_sizes(edges.sizes.indexes);
        // (only reallocated if the block counts have changed)
        edges.spans_alphas.ensure_sizes(edges.sizes.indexes);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        u32 ndust      = this->ndust;

        auto check_size = [&](const std::vector<Tscal> &v, const char *name) {
            if (v.size() != ndust) {
                shambase::throw_with_loc<std::invalid_argument>(sham::format(
                    "NodeComputeDragAlphas: {} has {} values, expected ndust = {}",
                    name,
                    v.size(),
                    ndust));
            }
        };

        std::visit(
            shambase::overloaded{
                [&](const Static &cfg) {
                    check_size(cfg.alphas, "alphas");

                    sham::DeviceBuffer<Tscal> alphas_buf(ndust, dev_sched);
                    alphas_buf.copy_from_stdvec(cfg.alphas);

                    edges.sizes.indexes.for_each([&](u64 id_patch, u32 block_count) {
                        sham::kernel_call(
                            dev_sched->get_queue(),
                            sham::MultiRef{alphas_buf},
                            sham::MultiRef{edges.spans_alphas.get_spans().get(id_patch)},
                            block_count * block_size * ndust,
                            [ndust](
                                u32 thread_id,
                                const Tscal *__restrict alphas_static,
                                Tscal *__restrict alphas) {
                                alphas[thread_id] = alphas_static[thread_id % ndust];
                            });
                    });
                },
                [&](const Epstein &cfg) {
                    check_size(cfg.grain_sizes, "grain_sizes");
                    check_size(cfg.grain_densities, "grain_densities");

                    sham::DeviceBuffer<Tscal> grain_sizes_buf(ndust, dev_sched);
                    sham::DeviceBuffer<Tscal> grain_densities_buf(ndust, dev_sched);
                    grain_sizes_buf.copy_from_stdvec(cfg.grain_sizes);
                    grain_densities_buf.copy_from_stdvec(cfg.grain_densities);

                    edges.sizes.indexes.for_each([&](u64 id_patch, u32 block_count) {
                        sham::kernel_call(
                            dev_sched->get_queue(),
                            sham::MultiRef{
                                edges.spans_rho.get_spans().get(id_patch),
                                edges.spans_press.get_spans().get(id_patch),
                                grain_sizes_buf,
                                grain_densities_buf},
                            sham::MultiRef{edges.spans_alphas.get_spans().get(id_patch)},
                            block_count * block_size * ndust,
                            [ndust, gamma = cfg.gamma](
                                u32 thread_id,
                                const Tscal *__restrict rho,
                                const Tscal *__restrict press,
                                const Tscal *__restrict grain_sizes,
                                const Tscal *__restrict grain_densities,
                                Tscal *__restrict alphas) {
                                u32 id_a  = thread_id / ndust;
                                u32 jdust = thread_id % ndust;

                                Tscal rho_g = rho[id_a];
                                Tscal cs    = sycl::sqrt(gamma * press[id_a] / rho_g);

                                Tscal ts = shamphys::epstein_stopping_time(
                                    grain_densities[jdust], grain_sizes[jdust], rho_g, cs, gamma);

                                alphas[thread_id] = Tscal{1} / ts;
                            });
                    });
                },
            },
            mode);
    }

    template<class Tvec>
    std::string NodeComputeDragAlphas<Tvec>::_impl_get_tex() const {
        std::string tex = std::visit(
            shambase::overloaded{
                [](const Static &) -> std::string {
                    return R"tex(
            Set the static dust collision rates in every cell

            \begin{align}
            {spans_alphas}_{i,j} &= \alpha_j
            \end{align}
        )tex";
                },
                [](const Epstein &) -> std::string {
                    return R"tex(
            Epstein dust collision rates of each cell

            \begin{align}
            c_{s,i} &= \sqrt{\gamma {spans_press}_i / {spans_rho}_i} \\
            {spans_alphas}_{i,j} &= \frac{{spans_rho}_i c_{s,i}}
                {\rho_{{\rm grain},j} s_j \sqrt{\pi \gamma / 8}}
            \end{align}
        )tex";
                },
            },
            mode);

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeDragAlphas<f64_3>;
