// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SinkParticlesUpdate.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/sink_edges_helper.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/edge/IDataEdgeSerializable.hpp"
#include "shamsolvergraph/node/ForwardEulerHost.hpp"
#include "shamsolvergraph/node/ForwardEulerHost2Deriv.hpp"
#include "shamsolvergraph/node/OperationIf.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW) X_RW(shamrock::solvergraph::IDataEdge<std::vector<T>>, field)

namespace shammodels::sph::modules {

    /// Reset a host-side (std::vector) field to its default value: field[i] = T{}
    template<class T>
    class ResetFieldHost : public shamrock::solvergraph::INode {

        public:
        ResetFieldHost() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {
            StackEntry stack_loc{};

            auto edges = get_edges();

            std::vector<T> &field = edges.field.data;

            for (size_t i = 0; i < field.size(); i++) {
                field[i] = T{};
            }
        }

        inline virtual std::string _impl_get_label() const { return "ResetFieldHost"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, G)                                               \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, epsilon)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, positions)                           \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, masses)                             \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, acc_ext)

namespace shammodels::sph::modules {

    /// Host-side pairwise (N^2) gravitational self-interaction between sink particles:
    /// acc_ext[i] = -sum_j G*mass[j]*rij / (|rij|^3 + epsilon)
    template<class Tvec>
    class SinkSelfGravityHost : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        SinkSelfGravityHost() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {
            StackEntry stack_loc{};

            auto edges = get_edges();

            Tscal G                        = edges.G.data;
            Tscal epsilon                  = edges.epsilon.data;
            const std::vector<Tvec> &pos   = edges.positions.data;
            const std::vector<Tscal> &mass = edges.masses.data;
            std::vector<Tvec> &acc_ext     = edges.acc_ext.data;

            for (size_t i = 0; i < pos.size(); i++) {
                Tvec sum{};
                for (size_t j = 0; j < pos.size(); j++) {
                    Tvec rij       = pos[i] - pos[j];
                    Tscal rij_scal = sycl::length(rij);
                    sum -= G * mass[j] * rij / (rij_scal * rij_scal * rij_scal + epsilon);
                }
                acc_ext[i] = sum;
            }
        }

        inline virtual std::string _impl_get_label() const { return "SinkSelfGravityHost"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::predictor_step(Tscal dt) {

    StackEntry stack_loc{};

    auto &sync = scheduler().synchronized_data;
    auto &pos  = get_sink_pos<Tvec>(sync);
    if (pos.empty()) {
        return;
    }

    compute_ext_forces();

    using VecEdge = shamrock::solvergraph::IDataEdgeSerializable<std::vector<Tvec>>;

    auto dt_half_edge  = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("dt_half", "dt/2");
    dt_half_edge->data = dt / 2;

    shamrock::solvergraph::ForwardEulerHost2Deriv<Tvec, Tscal> vel_update{};
    vel_update.set_edges(
        dt_half_edge,
        sync.template get_edge_ptr<VecEdge>("sink_acc_sph"),
        sync.template get_edge_ptr<VecEdge>("sink_acc_ext"),
        sync.template get_edge_ptr<VecEdge>("sink_vel"));
    vel_update.evaluate();

    auto dt_edge  = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("dt", "dt");
    dt_edge->data = dt;

    shamrock::solvergraph::ForwardEulerHost<Tvec, Tscal> pos_update{};
    pos_update.set_edges(
        dt_edge,
        sync.template get_edge_ptr<VecEdge>("sink_vel"),
        sync.template get_edge_ptr<VecEdge>("sink_pos"));
    pos_update.evaluate();
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::corrector_step(Tscal dt) {

    StackEntry stack_loc{};

    auto &sync = scheduler().synchronized_data;
    auto &vel  = get_sink_vel<Tvec>(sync);
    if (vel.empty()) {
        return;
    }

    auto &acc_sph = get_sink_acc_sph<Tvec>(sync);
    auto &acc_ext = get_sink_acc_ext<Tvec>(sync);

    for (size_t i = 0; i < vel.size(); i++) {
        vel[i] += (dt / 2) * (acc_sph[i] + acc_ext[i]);
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::compute_sph_forces() {

    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;

    auto &sync = scheduler().synchronized_data;
    auto &pos  = get_sink_pos<Tvec>(sync);
    if (pos.empty()) {
        return;
    }

    auto &mass             = get_sink_mass<Tvec>(sync);
    auto &accretion_radius = get_sink_accretion_radius<Tvec>(sync);
    auto &acc_sph          = get_sink_acc_sph<Tvec>(sync);

    Tscal G            = solver_config.get_constant_G();
    Tscal epsilon_grav = 1e-9;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 iaxyz_ext       = pdl.get_field_idx<Tvec>("axyz_ext");

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    std::vector<Tvec> result_acc_sinks{};

    for (size_t sink_id = 0; sink_id < pos.size(); sink_id++) {

        Tvec sph_acc_sink = {};

        scheduler().for_each_patchdata_nonempty(
            [&, G, epsilon_grav, gpart_mass](Patch cur_p, PatchDataLayer &pdat) {
                sham::DeviceBuffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(ixyz);
                sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

                sham::DeviceBuffer<Tvec> buf_sync_axyz(pdat.get_obj_cnt(), dev_sched);

                Tscal sink_mass = mass[sink_id];
                Tscal sink_racc = accretion_radius[sink_id];
                Tvec sink_pos   = pos[sink_id];

                sham::EventList depends_list;
                auto xyz       = buf_xyz.get_read_access(depends_list);
                auto axyz_ext  = buf_axyz_ext.get_write_access(depends_list);
                auto axyz_sync = buf_sync_axyz.get_write_access(depends_list);

                auto e = q.submit(
                    depends_list,
                    [&, G, epsilon_grav, sink_mass, sink_pos, sink_racc](sycl::handler &cgh) {
                        shambase::parallel_for(
                            cgh, pdat.get_obj_cnt(), "sink-sph forces", [=](i32 id_a) {
                                Tvec r_a = xyz[id_a];

                                Tvec delta = r_a - sink_pos;
                                Tscal d    = sycl::length(delta);

                                Tvec force = G * delta / (d * d * d);

                                // This is a hack to avoid the sink kaboom effect
                                // when the particle is being advected close to the sink before
                                // being accreted
                                if (d < sink_racc) {
                                    force = {0, 0, 0};
                                }

                                axyz_sync[id_a] = force * gpart_mass;
                                axyz_ext[id_a] += -force * sink_mass;
                            });
                    });

                buf_xyz.complete_event_state(e);
                buf_axyz_ext.complete_event_state(e);
                buf_sync_axyz.complete_event_state(e);

                sph_acc_sink
                    += shamalgs::primitives::sum(dev_sched, buf_sync_axyz, 0, pdat.get_obj_cnt());
            });

        result_acc_sinks.push_back(sph_acc_sink);
    }

    std::vector<Tvec> gathered_result_acc_sinks{};
    shamalgs::collective::vector_allgatherv(
        result_acc_sinks, gathered_result_acc_sinks, MPI_COMM_WORLD);

    for (size_t id_s = 0; id_s < pos.size(); id_s++) {

        acc_sph[id_s] = {};

        for (u32 rid = 0; rid < shamcomm::world_size(); rid++) {
            acc_sph[id_s] += gathered_result_acc_sinks[rid * pos.size() + id_s];
        }
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::compute_ext_forces() {

    StackEntry stack_loc{};

    auto &sync = scheduler().synchronized_data;
    auto &pos  = get_sink_pos<Tvec>(sync);
    if (pos.empty()) {
        return;
    }

    using VecEdge  = shamrock::solvergraph::IDataEdgeSerializable<std::vector<Tvec>>;
    using ScalEdge = shamrock::solvergraph::IDataEdgeSerializable<std::vector<Tscal>>;

    ResetFieldHost<Tvec> reset_acc_ext{};
    reset_acc_ext.set_edges(sync.template get_edge_ptr<VecEdge>("sink_acc_ext"));
    reset_acc_ext.evaluate();

    auto g_edge  = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("G", "G");
    g_edge->data = solver_config.get_constant_G();

    auto epsilon_edge
        = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("epsilon_grav_sink", "\\epsilon");
    epsilon_edge->data = 1e-9;

    auto self_gravity = std::make_shared<SinkSelfGravityHost<Tvec>>();
    self_gravity->set_edges(
        g_edge,
        epsilon_edge,
        sync.template get_edge_ptr<VecEdge>("sink_pos"),
        sync.template get_edge_ptr<ScalEdge>("sink_mass"),
        sync.template get_edge_ptr<VecEdge>("sink_acc_ext"));

    auto has_sinks_edge
        = shamrock::solvergraph::IDataEdge<bool>::make_shared("has_sinks", "\\rm has\\_sinks");
    has_sinks_edge->data = !pos.empty();

    shamrock::solvergraph::OperationIf if_has_sinks("if_has_sinks", self_gravity);
    if_has_sinks.set_edges(has_sinks_edge);
    if_has_sinks.evaluate();
}

using namespace shammath;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M6>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M8>;

template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C2>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C6>;
