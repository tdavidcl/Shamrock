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
 * @file NodePCGLoop.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Preconditioned conjugate gradient solver of the Poisson equation for the self-gravity.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/modules/CGInit.hpp"
#include "shammodels/ramses/modules/NodeJacobiPreconditioner.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shammodels/ramses/modules/ResidualDot.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/ExchangeGhostField.hpp"
#include "shamrock/solvergraph/ExtractGhostField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/NodeAXPY.hpp"
#include "shamrock/solvergraph/NodeAYPX.hpp"
#include "shamrock/solvergraph/NodeHadamardProd.hpp"
#include "shamrock/solvergraph/NodeSumReduction.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"
#include "shamrock/solvergraph/RankGetter.hpp"
#include "shamrock/solvergraph/ReplaceGhostField.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <memory>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes_no_gz)                                         \
    X_RO(CellGraphEdge, cell_neigh_graph)                                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, mean_rho)                                       \
    X_RO(shamrock::solvergraph::DDSharedBuffers<u32>, idx_in_ghost)                                \
    X_RO(shamrock::solvergraph::RankGetter, rank_owner)                                            \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, spans_phi)                                      \
                                                                                                   \
    /* ------------------- work fields ------------------- */                                      \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_res)                                       \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_p)                                         \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_Ap)                                        \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_hadamard_prod)                             \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_rhs)                                       \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_z)                                         \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, old_values)                                      \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, new_values)                                      \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, e_norm)                                          \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, alpha)                                           \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, beta)                                            \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, rhs_norm_values)                                 \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, rz_old_values)                                   \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, rz_new_values)                                   \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_rz_hadamard_prod)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Solve \f$ \Delta \phi = 4 \pi G (\rho - \bar{\rho}) \f$ with the preconditioned
     * conjugate gradient method, starting from the current content of `spans_phi`.
     *
     * Same as NodeCGLoop (see it for the matrix, the right-hand side, the ghost blocks and the
     * stopping criterion), with the diagonal preconditioner of NodeJacobiPreconditioner
     * (`spans_phi_z` holds \f$ z = M^{-1} r \f$).
     *
     * Port of the (not compiled) `NodePCGLoop.cpp` of #1942, with its numerics unchanged:
     * \f$ z_0 \f$ and \f$ p_0 = r_0 \f$ come from CGInit (\f$ z_0 = r_0 / A_{ii} \f$), then
     * \f$ z_{k+1} = M^{-1} r_{k+1} \f$ with the preconditioner of NodeJacobiPreconditioner,
     * \f$ \alpha_k = \langle r_k, z_k \rangle / \langle p_k, A p_k \rangle \f$,
     * \f$ \beta_k = \langle r_{k+1}, z_{k+1} \rangle / \langle r_k, z_k \rangle \f$ and
     * \f$ p_{k+1} = r_{k+1} + \beta_k p_k \f$.
     *
     * TODO: (kept from #1942) this is not the textbook PCG, which uses \f$ p_0 = z_0 \f$,
     * \f$ p_{k+1} = z_{k+1} + \beta_k p_k \f$ and the same preconditioner for \f$ z_0 \f$ as in
     * the loop (here \f$ 1 / A_{ii} = 1 / (6 \Delta x) \f$ for \f$ z_0 \f$ but
     * \f$ \Delta x^2 / 6 \f$ afterwards). As is, the step lengths are scaled by the
     * preconditioner, so the iteration is not a conjugate gradient one: it does not converge (the
     * residual grows), whereas the textbook PCG converges like CG (see PoissonSolversTests.cpp).
     *
     * The work fields are resized to match `sizes` if needed.
     */
    template<class Tvec, class TgridVec>
    class NodePCGLoop : public shamrock::solvergraph::INode {
        using Tscal         = shambase::VecComponent<Tvec>;
        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        u32 block_size;
        Tscal fourPiG;
        u32 Niter_max;
        Tscal tol;

        u32 last_iteration_count = 0;

        /// r0 = p0 = b - A phi0, z0 = r0 / A_ii
        CGInit<Tvec, TgridVec> cg_init_node{block_size, fourPiG};
        /// <r0, r0>
        ResidualDot<Tscal> res_ddot_node{block_size};
        /// <b, b>
        ResidualDot<Tscal> rhs_node{block_size};
        /// A p_k
        NodeSpMVPoisson3D<Tvec, TgridVec> spmv_node{block_size};
        /// p_k x A p_k (elementwise)
        shamrock::solvergraph::NodeHadamardProd<Tscal> hadamard_prod_node{block_size};
        /// <p_k, A p_k>
        shamrock::solvergraph::NodeSumReduction<Tscal> a_norm_node{block_size};
        /// phi_{k+1} = phi_k + alpha_k p_k
        shamrock::solvergraph::NodeAXPY<Tscal> new_potential_node{block_size};
        /// r_{k+1} = r_k - alpha_k A p_k
        shamrock::solvergraph::NodeAXPY<Tscal> new_residual_node{block_size};
        /// z_{k+1} = M^-1 r_{k+1}
        NodeJacobiPreconditioner<Tscal> res_precond_node{block_size};
        /// <r_{k+1}, r_{k+1}>
        ResidualDot<Tscal> res_ddot_new_node{block_size};
        /// r_k x z_k (elementwise)
        shamrock::solvergraph::NodeHadamardProd<Tscal> rz_hadamard_prod_node{block_size};
        /// <r_0, z_0>
        shamrock::solvergraph::NodeSumReduction<Tscal> rz_reduction_node{block_size};
        /// <r_{k+1}, z_{k+1}>
        shamrock::solvergraph::NodeSumReduction<Tscal> rz_new_reduction_node{block_size};
        /// p_{k+1} = r_{k+1} + beta_k p_k
        shamrock::solvergraph::NodeAYPX<Tscal> new_p_node{block_size};

        std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>> p_ghosts
            = std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>>(
                "p_ghosts", "p_{\\rm ghosts}");
        shamrock::solvergraph::ExtractGhostField<Tscal> node_gz_p{};
        shamrock::solvergraph::ExchangeGhostField<Tscal> node_exch_gz_p{};
        shamrock::solvergraph::ReplaceGhostField<Tscal> node_replace_gz_p{};

        public:
        NodePCGLoop(u32 block_size, Tscal fourPiG, u32 Niter_max, Tscal tol)
            : block_size(block_size), fourPiG(fourPiG), Niter_max(Niter_max), tol(tol) {}

        // same as EXPAND_NODE_EDGES(NODE_EDGES), with set_edges also wiring the inner nodes
        struct Edges {
            NODE_EDGES(INODE_DECL_RO, INODE_DECL_RW)
        };

        inline void set_edges(
            NODE_EDGES(INODE_PARAM_RO, INODE_PARAM_RW) SourceLocation loc = SourceLocation{}) {
            __shamrock_log_callsite(loc);

            __internal_set_ro_edges({NODE_EDGES(INODE_PUSH_RO1, INODE_PUSH_RW1)});
            __internal_set_rw_edges({NODE_EDGES(INODE_PUSH_RO2, INODE_PUSH_RW2)});

            cg_init_node.set_edges(
                sizes,
                cell_neigh_graph,
                spans_block_cell_sizes,
                spans_phi,
                spans_rho,
                mean_rho,
                spans_phi_res,
                spans_phi_p,
                spans_phi_rhs,
                spans_phi_z);
            res_ddot_node.set_edges(sizes_no_gz, spans_phi_res, old_values);
            rhs_node.set_edges(sizes_no_gz, spans_phi_rhs, rhs_norm_values);
            spmv_node.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_p, spans_phi_Ap);
            hadamard_prod_node.set_edges(sizes, spans_phi_p, spans_phi_Ap, spans_phi_hadamard_prod);
            a_norm_node.set_edges(sizes_no_gz, spans_phi_hadamard_prod, e_norm);
            new_potential_node.set_edges(sizes, spans_phi_p, alpha, spans_phi);
            new_residual_node.set_edges(sizes, spans_phi_Ap, alpha, spans_phi_res);
            res_precond_node.set_edges(sizes, spans_block_cell_sizes, spans_phi_res, spans_phi_z);
            res_ddot_new_node.set_edges(sizes_no_gz, spans_phi_res, new_values);
            rz_hadamard_prod_node.set_edges(
                sizes, spans_phi_res, spans_phi_z, spans_rz_hadamard_prod);
            rz_reduction_node.set_edges(sizes_no_gz, spans_rz_hadamard_prod, rz_old_values);
            rz_new_reduction_node.set_edges(sizes_no_gz, spans_rz_hadamard_prod, rz_new_values);
            new_p_node.set_edges(sizes, spans_phi_res, beta, spans_phi_p);

            node_gz_p.set_edges(spans_phi_p, idx_in_ghost, p_ghosts);
            node_exch_gz_p.set_edges(rank_owner, p_ghosts);
            node_replace_gz_p.set_edges(p_ghosts, spans_phi_p);
        }

        inline Edges get_edges() {
            int ro = 0;
            int rw = 0;
            return Edges{NODE_EDGES(INODE_GET_RO, INODE_GET_RW)};
        }

        INODE_EXPAND_TEX_SYMBOLS(
            NODE_EDGES(INODE_TEX_DECL, INODE_TEX_DECL),
            NODE_EDGES(INODE_TEX_GET_RO, INODE_TEX_GET_RW),
            NODE_EDGES(INODE_TEX_REPLACE, INODE_TEX_REPLACE))

        /// Number of iterations done by the last evaluation
        inline u32 get_last_iteration_count() const { return last_iteration_count; }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "PCGMainLoop"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
