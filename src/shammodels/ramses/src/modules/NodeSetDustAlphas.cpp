// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeSetDustAlphas.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shammath/riemann.hpp"
#include "shammodels/ramses/modules/NodeSetDustAlphas.hpp"
#include "shamphys/Dust.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>

namespace shammodels::basegodunov::modules {

    template<class Tscal>
    void NodeSetDustAlphasConstant<Tscal>::_impl_evaluate_internal() {
        auto edges = get_edges();

        if (alphas.size() != ndust) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "the size of alphas ({}) must match ndust = {}", alphas.size(), ndust));
        }

        edges.alphas_field.ensure_sizes(edges.sizes.indexes);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<Tscal> alphas_buf(ndust, shamsys::instance::get_compute_scheduler_ptr());
        alphas_buf.copy_from_stdvec(alphas);

        const u32 ndust_ = ndust;

        edges.sizes.indexes.for_each([&](u64 id, u32 block_count) {
            u32 cell_count = block_count * block_size;

            if (cell_count == 0) {
                return;
            }

            auto &out = edges.alphas_field.get_field(id).get_buf();

            sham::EventList depend_list;
            auto acc_alphas_in = alphas_buf.get_read_access(depend_list);
            auto acc_alphas    = out.get_write_access(depend_list);

            auto e = q.submit(depend_list, [&, ndust_](sycl::handler &cgh) {
                shambase::parallel_for(
                    cgh, ndust_ * cell_count, "set dust alphas [constant]", [=](u32 thread_id) {
                        acc_alphas[thread_id] = acc_alphas_in[thread_id % ndust_];
                    });
            });

            alphas_buf.complete_event_state(e);
            out.complete_event_state(e);
        });
    }

    template<class Tscal>
    std::string NodeSetDustAlphasConstant<Tscal>::_impl_get_tex() const {
        auto alphas_field = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Constant dust drag rates (inverse stopping times)
            \begin{equation}
            {alphas}_{i,j} = \alpha_j
            \end{equation}
            with $i$ the cell index and $j$ the dust species index.
        )tex";

        shambase::replace_all(tex, "{alphas}", alphas_field);

        return tex;
    }

    template<class Tvec>
    void NodeSetDustAlphasEpstein<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        if (grains_sizes.size() != ndust || grains_densities.size() != ndust) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "the sizes of grains_sizes ({}) and grains_densities ({}) must match ndust = "
                "{}",
                grains_sizes.size(),
                grains_densities.size(),
                ndust));
        }

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.alphas_field.ensure_sizes(edges.sizes.indexes);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<Tscal> sgrain_buf(ndust, shamsys::instance::get_compute_scheduler_ptr());
        sham::DeviceBuffer<Tscal> rho_grain_buf(
            ndust, shamsys::instance::get_compute_scheduler_ptr());
        sgrain_buf.copy_from_stdvec(grains_sizes);
        rho_grain_buf.copy_from_stdvec(grains_densities);

        const u32 ndust_       = ndust;
        const Tscal gamma_     = gamma;
        const bool supersonic_ = supersonic_correction;

        edges.sizes.indexes.for_each([&](u64 id, u32 block_count) {
            u32 cell_count = block_count * block_size;

            if (cell_count == 0) {
                return;
            }

            auto &buf_rho    = edges.spans_rho.get_field(id).get_buf();
            auto &buf_rhov   = edges.spans_rhov.get_field(id).get_buf();
            auto &buf_rhoe   = edges.spans_rhoe.get_field(id).get_buf();
            auto &buf_rho_d  = edges.spans_rho_dust.get_field(id).get_buf();
            auto &buf_rhov_d = edges.spans_rhov_dust.get_field(id).get_buf();

            auto &out = edges.alphas_field.get_field(id).get_buf();

            sham::EventList depend_list;
            auto rho     = buf_rho.get_read_access(depend_list);
            auto rhov    = buf_rhov.get_read_access(depend_list);
            auto rhoe    = buf_rhoe.get_read_access(depend_list);
            auto rho_d   = buf_rho_d.get_read_access(depend_list);
            auto rhov_d  = buf_rhov_d.get_read_access(depend_list);
            auto sgrain  = sgrain_buf.get_read_access(depend_list);
            auto rgrain  = rho_grain_buf.get_read_access(depend_list);
            auto acc_out = out.get_write_access(depend_list);

            auto e = q.submit(depend_list, [&, ndust_, gamma_, supersonic_](sycl::handler &cgh) {
                shambase::parallel_for(
                    cgh, ndust_ * cell_count, "set dust alphas [epstein]", [=](u32 thread_id) {
                        const u32 jdust = thread_id % ndust_;
                        const u32 id_a  = thread_id / ndust_;

                        auto cons = shammath::ConsState<Tvec>{rho[id_a], rhoe[id_a], rhov[id_a]};
                        auto prim = shammath::cons_to_prim(cons, gamma_);

                        // The stopping time diverges for a pressureless or empty cell, which
                        // means no drag. Guarding here also keeps the asserts of
                        // epstein_stopping_time satisfied.
                        if (!(prim.rho > 0) || !(prim.press > 0)) {
                            acc_out[thread_id] = Tscal(0);
                            return;
                        }

                        Tscal cs = shammath::sound_speed(prim, gamma_);

                        Tscal f = Tscal(1);
                        if (supersonic_) {
                            const u32 id_d = id_a * ndust_ + jdust;
                            Tvec dv        = prim.vel;
                            if (rho_d[id_d] > 0) {
                                dv = rhov_d[id_d] / rho_d[id_d] - prim.vel;
                            } else {
                                dv = Tvec{0, 0, 0};
                            }
                            f = shamphys::epstein_supersonic_correction(sycl::length(dv), cs);
                        }

                        // Note: the two fluid drag rate uses the *gas* density, see the note on
                        // shamphys::epstein_stopping_time.
                        Tscal ts = shamphys::epstein_stopping_time(
                            rgrain[jdust], sgrain[jdust], prim.rho, cs, gamma_, f);

                        acc_out[thread_id] = Tscal(1) / ts;
                    });
            });

            buf_rho.complete_event_state(e);
            buf_rhov.complete_event_state(e);
            buf_rhoe.complete_event_state(e);
            buf_rho_d.complete_event_state(e);
            buf_rhov_d.complete_event_state(e);
            sgrain_buf.complete_event_state(e);
            rho_grain_buf.complete_event_state(e);
            out.complete_event_state(e);
        });
    }

    template<class Tvec>
    std::string NodeSetDustAlphasEpstein<Tvec>::_impl_get_tex() const {
        auto rho          = get_ro_edge_base(1).get_tex_symbol();
        auto alphas_field = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Epstein drag rates (inverse stopping times) from the local gas state
            \begin{equation}
            {alphas}_{i,j} = \frac{ {rho}_i c_{s,i} }
                                  { \rho_{{\rm grain},j} s_{{\rm grain},j} }
                             \sqrt{ \frac{8}{\pi \gamma} }
            \end{equation}
            with $i$ the cell index and $j$ the dust species index, and $c_s$ the gas sound
            speed. Note that ${rho}$ is the gas density, as required by the two fluid drag
            equation $\partial_t v_{\rm d} = \alpha (v_{\rm g} - v_{\rm d})$.
        )tex";

        shambase::replace_all(tex, "{alphas}", alphas_field);
        shambase::replace_all(tex, "{rho}", rho);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeSetDustAlphasConstant<f64>;
template class shammodels::basegodunov::modules::NodeSetDustAlphasEpstein<f64_3>;
