// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodePenalization.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solver graph node applying the SPH volume penalization zones over a time interval
 *
 */

#include "shambase/memory.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/config/PenalizationConfig.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"
#include <memory>
#include <variant>
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, dt)                                              \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, xyz)                                             \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, vxyz)                                            \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, uint)

namespace shammodels::sph::modules {

    /**
     * @brief Relax the particles inside the penalization zones toward the zone target state over
     * the time interval dt
     *
     * Solves exactly \f$ dq/dt = -(q - q_t)/\eta \f$ over dt for the velocity (and the specific
     * internal energy if the zone has a target for it), i.e.
     * \f$ q \leftarrow q_t + (q - q_t) e^{-dt/\eta} \f$. Zones are applied in config order, each
     * particle independently, using the current particle positions for the mask.
     *
     * The velocity removed inside each zone, \f$ \sum_a (\mathbf{v}_a^{\rm before} -
     * \mathbf{v}_a^{\rm after}) \f$, is added to `velocity_removed` (one entry per zone, summed
     * over all ranks in a fixed order); multiplied by the particle mass it is the momentum the
     * zone took from the gas.
     */
    template<class Tvec>
    class NodePenalization : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;
        using Cfg   = PenalizationConfig<Tvec>;

        Cfg config;
        std::shared_ptr<std::vector<Tvec>> velocity_removed;

        public:
        NodePenalization(Cfg config, std::shared_ptr<std::vector<Tvec>> velocity_removed)
            : config(std::move(config)), velocity_removed(std::move(velocity_removed)) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            std::vector<Tvec> &removed = shambase::get_check_ref(velocity_removed);
            removed.resize(config.zones.size(), Tvec{});

            Tscal dt = edges.dt.data;
            if (dt == 0) {
                return;
            }

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            // per particle velocity removed by the current zone (0 outside of it)
            shambase::DistributedData<sham::DeviceBuffer<Tvec>> dv
                = edges.part_counts.indexes.template map<sham::DeviceBuffer<Tvec>>(
                    [&](u64 id, u32 cnt) {
                        return sham::DeviceBuffer<Tvec>(cnt, dev_sched);
                    });

            for (u32 izone = 0; izone < config.zones.size(); izone++) {
                const typename Cfg::Zone &zone = config.zones[izone];

                // eta == 0 gives exp(-inf) = 0, i.e. the target is set directly
                Tscal keep   = (zone.eta > 0) ? sycl::exp(-dt / zone.eta) : Tscal{0};
                Tvec v_t     = zone.target_vel;
                bool relax_u = zone.target_uint.has_value();
                Tscal u_t    = zone.target_uint.value_or(Tscal{0});

                std::visit(
                    [&](const auto &shape) {
                        sham::distributed_data_kernel_call(
                            dev_sched,
                            sham::DDMultiRef{edges.xyz.get_spans()},
                            sham::DDMultiRef{edges.vxyz.get_spans(), edges.uint.get_spans(), dv},
                            edges.part_counts.indexes,
                            [shape, keep, v_t, relax_u, u_t](
                                u32 id_a,
                                const Tvec *__restrict xyz,
                                Tvec *__restrict vxyz,
                                Tscal *__restrict uint,
                                Tvec *__restrict dv) {
                                if (!shape.contains(xyz[id_a])) {
                                    dv[id_a] = Tvec{0, 0, 0};
                                    return;
                                }

                                Tvec v_old = vxyz[id_a];
                                Tvec v_new = v_t + (v_old - v_t) * keep;

                                vxyz[id_a] = v_new;
                                dv[id_a]   = v_old - v_new;

                                if (relax_u) {
                                    uint[id_a] = u_t + (uint[id_a] - u_t) * keep;
                                }
                            });
                    },
                    zone.shape);

                // sum over patches in increasing id order then over ranks (deterministic)
                Tvec rank_sum = Tvec{0, 0, 0};
                edges.part_counts.indexes.for_each_ordered([&](u64 id, const u32 &cnt) {
                    if (cnt > 0) {
                        rank_sum += shamalgs::primitives::sum(dev_sched, dv.get(id), 0, cnt);
                    }
                });

                removed[izone] += shamalgs::collective::allreduce_sum(rank_sum);
            }
        }

        inline virtual std::string _impl_get_label() const { return "Penalization"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
