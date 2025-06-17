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
 * @file InterpolateToFace.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/NeighGrapkLinkFieldEdge.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class InterpolateToFaceRho : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        using SlopeMode = shammodels::basegodunov::SlopeMode;

        u32 block_size;

        public:
        InterpolateToFaceRho(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::ScalarEdge<Tscal> &dt_interp;

            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;

            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_cell0block_aabb_lower;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rhos;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_grad_rho;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_vel;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_dx_vel;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_dy_vel;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_dz_vel;

            solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_xp;
            solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_xm;
            solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_yp;
            solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_ym;
            solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_zp;
            solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_zm;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> dt_interp,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_cell0block_aabb_lower,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rhos,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_grad_rho,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_vel,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_dx_vel,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_dy_vel,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_dz_vel,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                &rho_face_xp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                &rho_face_xm,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                &rho_face_yp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                &rho_face_ym,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                &rho_face_zp,
            std::shared_ptr<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>
                &rho_face_zm) {
            __internal_set_ro_edges({
                dt_interp,
                cell_neigh_graph,
                spans_block_cell_sizes,
                spans_cell0block_aabb_lower,
                spans_rhos,
                spans_grad_rho,
                spans_vel,
                spans_dx_vel,
                spans_dy_vel,
                spans_dz_vel,
            });
            __internal_set_rw_edges({
                rho_face_xp,
                rho_face_xm,
                rho_face_yp,
                rho_face_ym,
                rho_face_zp,
                rho_face_zm,
            });
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(5),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(6),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(7),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(8),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(9),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(0),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(1),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(2),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(3),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(4),
                get_rw_edge<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(5),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "InterpolateRhoToFaceRho"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
