// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeJacobiPreconditioner.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Diagonal preconditioner of the PCG Poisson solver.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shammodels/ramses/modules/NodeJacobiPreconditioner.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeJacobiPreconditioner<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_x.check_sizes(edges.sizes.indexes);
        edges.spans_y.ensure_sizes(edges.sizes.indexes);

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{edges.spans_block_cell_sizes.get_spans(), edges.spans_x.get_spans()},
            sham::DDMultiRef{edges.spans_y.get_spans()},
            cell_counts,
            [block_size = block_size](
                u32 i, const T *__restrict cell_sizes, const T *__restrict x, T *__restrict y) {
                T dx = cell_sizes[i / block_size];
                y[i] = (dx * dx) * x[i] / 6;
            });
    }

    template<class T>
    std::string NodeJacobiPreconditioner<T>::_impl_get_tex() const {
        std::string tex = R"tex(
            Diagonal preconditioner
            \begin{equation}
            {spans_y}_i = \frac{{spans_block_cell_sizes}_i^2}{6} {spans_x}_i
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeJacobiPreconditioner<f64>;
