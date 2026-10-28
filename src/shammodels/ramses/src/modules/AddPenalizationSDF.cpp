// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AddPenalizationSDF.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/AddPenalizationSDF.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include <type_traits>

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec, class Shape>
    void AddPenalizationSDF<Tvec, TgridVec, Shape>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * AMRBlock::block_size;
              });

        edges.spans_sdf.check_sizes(edges.sizes.indexes);

        const Tscal one_over_Nside = Tscal(1) / AMRBlock::Nside;

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{edges.spans_block_min.get_spans(), edges.spans_block_max.get_spans()},
            sham::DDMultiRef{edges.spans_sdf.get_spans()},
            cell_counts,
            [shape = shape, dxfact = grid_coord_to_pos_fact, one_over_Nside](
                u32 i,
                const TgridVec *__restrict block_min,
                const TgridVec *__restrict block_max,
                Tscal *__restrict sdf) {
                const u32 block_id = i / AMRBlock::block_size;
                const u32 lid      = i % AMRBlock::block_size;

                Tvec bmin      = block_min[block_id].template convert<Tscal>() * dxfact;
                Tvec bmax      = block_max[block_id].template convert<Tscal>() * dxfact;
                Tvec cell_size = (bmax - bmin) * one_over_Nside;

                auto coord = AMRBlock::get_coord(lid);
                Tvec cell_center
                    = bmin + Tvec{coord[0] + 0.5, coord[1] + 0.5, coord[2] + 0.5} * cell_size;

                sdf[i] = sycl::fmin(sdf[i], shape.sdf(cell_center));
            });
    }

    template<class Tvec, class TgridVec, class Shape>
    std::string AddPenalizationSDF<Tvec, TgridVec, Shape>::_impl_get_label() const {
        using Cfg = PenalizationConfig<Tvec>;
        if constexpr (std::is_same_v<Shape, typename Cfg::Cylinder>) {
            return "AddPenalizationSDFCylinder";
        } else if constexpr (std::is_same_v<Shape, typename Cfg::Sphere>) {
            return "AddPenalizationSDFSphere";
        } else if constexpr (std::is_same_v<Shape, typename Cfg::Box>) {
            return "AddPenalizationSDFBox";
        }
    }

    template<class Tvec, class TgridVec, class Shape>
    std::string AddPenalizationSDF<Tvec, TgridVec, Shape>::_impl_get_tex() const {
        using Cfg = PenalizationConfig<Tvec>;

        auto sdf = get_rw_edge_base(0).get_tex_symbol();

        std::string shape_tex;
        if constexpr (std::is_same_v<Shape, typename Cfg::Cylinder>) {
            shape_tex = R"tex(
            \mathbf{d} &= \mathbf{x}_i - \mathbf{c} \\
            \phi_{\rm shape}(\mathbf{x}_i) &= \lVert \mathbf{d} - (\mathbf{d}\cdot\hat{\mathbf{a}})\hat{\mathbf{a}} \rVert - R
            )tex";
        } else if constexpr (std::is_same_v<Shape, typename Cfg::Sphere>) {
            shape_tex = R"tex(
            \phi_{\rm shape}(\mathbf{x}_i) &= \lVert \mathbf{x}_i - \mathbf{c} \rVert - R
            )tex";
        } else if constexpr (std::is_same_v<Shape, typename Cfg::Box>) {
            shape_tex = R"tex(
            \mathbf{q} &= |\mathbf{x}_i - \mathbf{c}| - \mathbf{h} \\
            \phi_{\rm shape}(\mathbf{x}_i) &= \lVert \max(\mathbf{q}, 0) \rVert + \min(\max(q_x, q_y, q_z), 0)
            )tex";
        }

        std::string tex = R"tex(
            Add a shape to the penalization signed distance field (cell centers $\mathbf{x}_i$)

            \begin{align}
            {shape_tex} \\
            {sdf}_i &= \min({sdf}_i, \phi_{\rm shape}(\mathbf{x}_i))
            \end{align}
        )tex";

        shambase::replace_all(tex, "{shape_tex}", shape_tex);
        shambase::replace_all(tex, "{sdf}", sdf);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

using PenalCfg = shammodels::basegodunov::PenalizationConfig<f64_3>;

template class shammodels::basegodunov::modules::
    AddPenalizationSDF<f64_3, i64_3, PenalCfg::Cylinder>;
template class shammodels::basegodunov::modules::AddPenalizationSDF<f64_3, i64_3, PenalCfg::Sphere>;
template class shammodels::basegodunov::modules::AddPenalizationSDF<f64_3, i64_3, PenalCfg::Box>;
