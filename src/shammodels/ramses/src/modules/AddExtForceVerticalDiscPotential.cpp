// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AddExtForceVerticalDiscPotential.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Add the thin disc vertical acceleration of a central mass at the center of each cell.
 */

#include "shammodels/ramses/modules/AddExtForceVerticalDiscPotential.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void AddExtForceVerticalDiscPotential<Tvec>::_impl_evaluate_internal() {
        using Tscal = shambase::VecComponent<Tvec>;

        auto edges = get_edges();

        edges.spans_coordinates.check_sizes(edges.sizes.indexes);
        edges.spans_a_ext.check_sizes(edges.sizes.indexes);

        auto &coordinates_spans = edges.spans_coordinates.get_spans();
        auto &a_ext_spans       = edges.spans_a_ext.get_spans();

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  u32 cell_count = block_count * block_size;
                  return cell_count;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{coordinates_spans},
            sham::DDMultiRef{a_ext_spans},
            cell_counts,
            [mGM_R02 = -central_mass * edges.constant_G.value / (R0 * R0), R02 = R0 * R0, z0 = z0](
                u32 i, const Tvec *__restrict cell_coor, Tvec *__restrict a_ext) {
                Tscal z = cell_coor[i].z() - z0;
                a_ext[i].z() += mGM_R02 * z / sycl::sqrt(R02 + z * z);
            });
    }

    template<class Tvec>
    std::string AddExtForceVerticalDiscPotential<Tvec>::_impl_get_tex() const {
        auto coords = get_ro_edge_base(1).get_tex_symbol();
        auto a_ext  = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Add the thin disc vertical acceleration of a central mass to the external forces
        acceleration

        \begin{align}
        z_i &= ({{coords}}_i)_z - z_0 \\
        ({{a_ext}}_i)_z &= ({{a_ext}}_i)_z - \frac{G M z_i}{R_0^2 \sqrt{R_0^2 + z_i^2}}
        \end{align}
    )tex";

        shambase::replace_all(tex, "{a_ext}", a_ext);
        shambase::replace_all(tex, "{coords}", coords);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::AddExtForceVerticalDiscPotential<f64_3>;
