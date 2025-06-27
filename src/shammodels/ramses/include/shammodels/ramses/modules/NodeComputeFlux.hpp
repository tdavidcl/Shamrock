// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodeComputeFlux.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/NeighGrapkLinkFieldEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/OperationSequence.hpp"

namespace shammodels::basegodunov::modules {

    using RiemannSolverMode     = shammodels::basegodunov::RiemmanSolverMode;
    using DustRiemannSolverMode = shammodels::basegodunov::DustRiemannSolverMode;
    using Direction             = shammodels::basegodunov::modules::Direction;

    template<class Tvec, class TgridVec, RiemannSolverMode mode, Direction dir>
    class NodeComputeFluxGasDirMode : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal gamma;

        public:
        NodeComputeFluxGasDirMode(Tscal gamma) : gamma(gamma) {}

        struct Edges {
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &press_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tscal> &flux_rho_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tvec> &flux_rhov_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tscal> &flux_rhoe_face;
        };

        inline void set_edges(
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> press_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rhoe_face) {
            __internal_set_ro_edges({cell_neigh_graph, rho_face, vel_face, press_face});
            __internal_set_rw_edges({flux_rho_face, flux_rhov_face, flux_rhoe_face});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(0),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(1),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(2),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(3),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(0),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(1),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(2),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeComputeFluxGasDirMode"; };

        virtual std::string _impl_get_tex();
    };

    template<class Tvec, class TgridVec, DustRiemannSolverMode mode, Direction dir>
    class NodeComputeFluxDustDirMode : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 ndust;

        public:
        NodeComputeFluxDustDirMode(u32 ndust) : ndust(ndust) {}

        struct Edges {
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face;
            const solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tscal> &flux_rho_face;
            solvergraph::NeighGrapkLinkFieldEdge<Tvec> &flux_rhov_face;
        };

        inline void set_edges(
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>> rho_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>> vel_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tscal>> flux_rho_face,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<Tvec>> flux_rhov_face) {
            __internal_set_ro_edges({cell_neigh_graph, rho_face, vel_face});
            __internal_set_rw_edges({flux_rho_face, flux_rhov_face});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(0),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(1),
                get_ro_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(2),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(0),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(1)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeComputeFluxDustDirMode"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
