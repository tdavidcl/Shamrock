// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeDragIntegratorEXPO.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shammath/matrix_exponential.hpp"
#include "shammodels/ramses/modules/NodeDragIntegratorEXPO.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeDragIntegratorEXPO<Tvec>::_impl_evaluate_internal() {
        using namespace shammath;

        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.rho_next.check_sizes(edges.sizes.indexes);
        edges.alphas.check_sizes(edges.sizes.indexes);

        const Tscal dt             = edges.dt.value;
        const u32 ndust            = this->ndust;
        const u32 friction_control = (enable_frictional_heating == false) ? 1 : 0;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        edges.sizes.indexes.for_each([&](u64 id, u32 block_count) {
            u32 cell_count = block_count * block_size;

            if (cell_count == 0) {
                return;
            }

            auto &rho_new_patch    = edges.rho_next.get_field(id).get_buf();
            auto &rhov_new_patch   = edges.rhov_next.get_field(id).get_buf();
            auto &rhoe_new_patch   = edges.rhoe_next.get_field(id).get_buf();
            auto &rho_d_new_patch  = edges.rho_dust_next.get_field(id).get_buf();
            auto &rhov_d_new_patch = edges.rhov_dust_next.get_field(id).get_buf();

            auto &rho_old    = edges.spans_rho.get_field(id).get_buf();
            auto &rhov_old   = edges.spans_rhov.get_field(id).get_buf();
            auto &rhoe_old   = edges.spans_rhoe.get_field(id).get_buf();
            auto &rho_d_old  = edges.spans_rho_dust.get_field(id).get_buf();
            auto &rhov_d_old = edges.spans_rhov_dust.get_field(id).get_buf();

            auto &alphas_buf = edges.alphas.get_field(id).get_buf();

            sham::EventList depend_list;
            auto acc_rho_new_patch    = rho_new_patch.get_read_access(depend_list);
            auto acc_rhov_new_patch   = rhov_new_patch.get_read_access(depend_list);
            auto acc_rhoe_new_patch   = rhoe_new_patch.get_read_access(depend_list);
            auto acc_rho_d_new_patch  = rho_d_new_patch.get_read_access(depend_list);
            auto acc_rhov_d_new_patch = rhov_d_new_patch.get_read_access(depend_list);

            auto acc_rho_old    = rho_old.get_write_access(depend_list);
            auto acc_rhov_old   = rhov_old.get_write_access(depend_list);
            auto acc_rhoe_old   = rhoe_old.get_write_access(depend_list);
            auto acc_rho_d_old  = rho_d_old.get_write_access(depend_list);
            auto acc_rhov_d_old = rhov_d_old.get_write_access(depend_list);

            auto acc_alphas = alphas_buf.get_read_access(depend_list);

            size_t mat_size         = ndust + 1;
            size_t mat_size_squared = mat_size * mat_size;
            size_t group_size
                = (q.get_device_prop().local_mem_size) / (5 * mat_size_squared * sizeof(Tscal));
            size_t loc_acc_size = mat_size_squared * group_size;

            size_t loc_mem_size = 5 * sizeof(Tscal) * loc_acc_size;

            using MdSpan = std::
                mdspan<Tscal, std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>;

            // Per cell drag update, shared by the global memory and shared memory variants: the
            // only difference between the two is where the five scratch matrices live.
            auto cell_update = [=](u32 id_a,
                                   Tscal *ptr_A,
                                   Tscal *ptr_B,
                                   Tscal *ptr_F,
                                   Tscal *ptr_I,
                                   Tscal *ptr_Id) {
                // sparse jacobian matrix
                auto get_jacobian = [=](u32 id, MdSpan &jacobian) {
                    mat_set_nul<Tscal>(jacobian);
                    // fill first row
                    for (auto j = 1; j < jacobian.extent(1); j++)
                        jacobian(0, j) = acc_alphas[id * ndust + (j - 1)];
                    // fil first column
                    for (auto i = 1; i < jacobian.extent(0); i++) {
                        jacobian(i, 0)
                            = acc_alphas[id * ndust + (i - 1)]
                              * (acc_rho_d_new_patch[id * ndust + (i - 1)] / acc_rho_new_patch[id]);
                        jacobian(0, 0) -= jacobian(i, 0);
                    }
                    // fill diagonal from (i,j)=(1,1)
                    for (auto i = 1; i < jacobian.extent(0); i++)
                        jacobian(i, i) = -acc_alphas[id * ndust + (i - 1)];
                    // the rest of the buffer is set to zero
                };

                Tscal mu = 0;
                for (auto i = 0; i < ndust; i++) {
                    mu += (1 + (acc_rho_d_new_patch[id_a * ndust + i] / acc_rho_new_patch[id_a]))
                          * acc_alphas[id_a * ndust + i];
                }
                mu *= (-dt / (ndust + 1));

                MdSpan mdspan_A(ptr_A, mat_size, mat_size);
                MdSpan mdspan_B(ptr_B, mat_size, mat_size);
                MdSpan mdspan_F(ptr_F, mat_size, mat_size);
                MdSpan mdspan_I(ptr_I, mat_size, mat_size);
                MdSpan mdspan_Id(ptr_Id, mat_size, mat_size);

                get_jacobian(id_a, mdspan_A);

                // pre-processing step
                shammath::mat_set_identity<Tscal>(mdspan_Id);
                shammath::mat_axpy_beta<Tscal, Tscal>(-mu, mdspan_Id, dt, mdspan_A);

                // compute matrix exponential
                const i32 K_exp = 9;
                shammath::mat_exp<Tscal, Tscal>(
                    K_exp, mdspan_A, mdspan_F, mdspan_B, mdspan_I, mdspan_Id, ndust + 1);

                // post-processing step
                shammath::mat_mul_scalar<Tscal>(mdspan_A, sycl::exp(mu));

                // use the matrix exponential to for to updates momemtum
                Tvec r = {0., 0., 0.}, dd = {0., 0., 0.};
                r += mdspan_A(0, 0) * acc_rhov_new_patch[id_a];

                for (auto j = 1; j < ndust + 1; j++) {
                    r += mdspan_A(0, j) * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
                }

                dd = r - acc_rhov_new_patch[id_a];

                Tscal dissipation = 0, drag_work = 0;

                // compute work of drag terms
                Tscal inv_rho = 1.0 / (acc_rho_new_patch[id_a]);

                Tvec v_bf = inv_rho * acc_rhov_new_patch[id_a];
                Tvec v_af = inv_rho * r;

                drag_work = 0.5
                            * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                               + dd[2] * (v_bf[2] + v_af[2]));

                // save gas momentum back
                acc_rhov_old[id_a] = r;
                acc_rho_old[id_a]  = acc_rho_new_patch[id_a];

                for (auto d_id = 1; d_id <= ndust; d_id++) {
                    r *= 0;
                    r += mdspan_A(d_id, 0) * acc_rhov_new_patch[id_a];

                    for (auto j = 1; j <= ndust; j++) {

                        r += mdspan_A(d_id, j) * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
                    }

                    dd = r - acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

                    inv_rho = 1.0 / (acc_rho_d_new_patch[id_a * ndust + (d_id - 1)]);

                    v_bf = inv_rho * acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

                    v_af = inv_rho * r;

                    // compute dissipaation by id-th dust
                    dissipation += 0.5
                                   * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                                      + dd[2] * (v_bf[2] + v_af[2]));

                    // save dust momentum back
                    acc_rhov_d_old[id_a * ndust + (d_id - 1)] = r;
                    acc_rho_d_old[id_a * ndust + (d_id - 1)]
                        = acc_rho_d_new_patch[id_a * ndust + (d_id - 1)];
                }

                // updates energy
                acc_rhoe_old[id_a] = acc_rhoe_new_patch[id_a] + (1 - friction_control) * drag_work
                                     - friction_control * dissipation;
            };

            if (group_size < 8) {
                sham::DeviceBuffer<Tscal> scratch_expo(
                    5 * mat_size_squared * cell_count,
                    shamsys::instance::get_compute_scheduler_ptr());
                Tscal *exp_scratch_ptr_base = scratch_expo.get_write_access(depend_list);

                auto e = q.submit(depend_list, [&](sycl::handler &cgh) {
                    shambase::parallel_for(
                        cgh, cell_count, "add_drag [expo-global-mem]", [=](u32 id_a) {
                            Tscal *base = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared);
                            cell_update(
                                id_a,
                                base,
                                base + mat_size_squared,
                                base + 2 * mat_size_squared,
                                base + 3 * mat_size_squared,
                                base + 4 * mat_size_squared);
                        });
                });

                scratch_expo.complete_event_state(e);

                rho_new_patch.complete_event_state(e);
                rhov_new_patch.complete_event_state(e);
                rhoe_new_patch.complete_event_state(e);
                rho_d_new_patch.complete_event_state(e);
                rhov_d_new_patch.complete_event_state(e);

                rho_old.complete_event_state(e);
                rhov_old.complete_event_state(e);
                rhoe_old.complete_event_state(e);
                rho_d_old.complete_event_state(e);
                rhov_d_old.complete_event_state(e);

                alphas_buf.complete_event_state(e);

            } else {

                if (loc_mem_size > q.get_device_prop().local_mem_size) {
                    shambase::throw_with_loc<std::runtime_error>(sham::format(
                        "not enough local memory for expo drag integrator:\n"
                        "loc_mem_size: {} > max_local_mem: {}\n"
                        "loc_acc_size: {}\n"
                        "group_size: {}\n"
                        "ndust: {}\n",
                        loc_mem_size,
                        q.get_device_prop().local_mem_size,
                        loc_acc_size,
                        group_size,
                        ndust));
                }

                auto e = q.submit(depend_list, [&](sycl::handler &cgh) {
                    // local/shared memory alloc for each work-item
                    sycl::local_accessor<Tscal> local_A(loc_acc_size, cgh);
                    sycl::local_accessor<Tscal> local_B(loc_acc_size, cgh);
                    sycl::local_accessor<Tscal> local_F(loc_acc_size, cgh);
                    sycl::local_accessor<Tscal> local_I(loc_acc_size, cgh);
                    sycl::local_accessor<Tscal> local_Id(loc_acc_size, cgh);

                    logger::debug_sycl_ln(
                        "SYCL", sham::format("parallel_for add_drag [expo-shared-mem]"));
                    cgh.parallel_for(
                        shambase::make_range(cell_count, group_size), [=](sycl::nd_item<1> id) {
                            u32 loc_id = id.get_local_id();
                            u32 id_a   = id.get_global_id();
                            if (id_a >= cell_count)
                                return;

                            cell_update(
                                id_a,
                                &(local_A[0]) + mat_size_squared * loc_id,
                                &(local_B[0]) + mat_size_squared * loc_id,
                                &(local_F[0]) + mat_size_squared * loc_id,
                                &(local_I[0]) + mat_size_squared * loc_id,
                                &(local_Id[0]) + mat_size_squared * loc_id);
                        });
                });

                rho_new_patch.complete_event_state(e);
                rhov_new_patch.complete_event_state(e);
                rhoe_new_patch.complete_event_state(e);
                rho_d_new_patch.complete_event_state(e);
                rhov_d_new_patch.complete_event_state(e);

                rho_old.complete_event_state(e);
                rhov_old.complete_event_state(e);
                rhoe_old.complete_event_state(e);
                rho_d_old.complete_event_state(e);
                rhov_d_old.complete_event_state(e);

                alphas_buf.complete_event_state(e);
            }
        });
    }

    template<class Tvec>
    std::string NodeDragIntegratorEXPO<Tvec>::_impl_get_tex() const {
        auto alphas = get_ro_edge_base(2).get_tex_symbol();
        auto dt     = get_ro_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Exponential drag integrator: the momenta of the gas and of the $N_{\rm dust}$ dust
            species are advanced over ${dt}$ by the exponential of the drag Jacobian built from
            ${alphas}$
            \begin{equation}
            \left( \rho v \right)^{n+1} = \exp \left( {dt} J \right) \left( \rho v \right)^{*}
            \end{equation}
            the matrix exponential being computed per cell by a scaled Taylor expansion.
        )tex";

        shambase::replace_all(tex, "{alphas}", alphas);
        shambase::replace_all(tex, "{dt}", dt);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeDragIntegratorEXPO<f64_3>;
