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
 * @file AddForceVerticalDiscPotential.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Adds the vertical acceleration of a central mass seen at cylindrical radius R0,
 * a_z = -GMz / (R0^2 + z^2)^(3/2).
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"

/// declare the list of edges for this node
#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* inputs */                                                                                   \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, constant_G)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, central_mass)                                    \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, R0)                                              \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_positions)                                 \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)

namespace shammodels::common::modules {
    template<class Tvec>
    class AddForceVerticalDiscPotential : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceVerticalDiscPotential() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.spans_positions.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal cmass = edges.central_mass.data;
            Tscal G     = edges.constant_G.data;
            Tscal R0    = edges.R0.data;

            // call the GPU kernel for each patches
            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.spans_positions.get_spans()},
                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
                edges.sizes.indexes,
                [mGM = -cmass * G, R02 = R0 * R0](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                    // Phi = -GM / sqrt(R0^2 + z^2) => a_z = -GM z / (R0^2 + z^2)^(3/2)
                    Tscal z_a = xyz[gid].z();
                    Tscal r2  = R02 + z_a * z_a;
                    axyz_ext[gid].z() += mGM * z_a / (r2 * sycl::sqrt(r2));
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "AddForceVerticalDiscPotential";
        };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::common::modules

// remove the macro before exiting
#undef NODE_EDGES
