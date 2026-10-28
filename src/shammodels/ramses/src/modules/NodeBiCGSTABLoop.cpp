// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeBiCGSTABLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief BiCGSTAB solver of the Poisson equation for the self-gravity.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/ramses/modules/NodeBiCGSTABLoop.hpp"
#include <cmath>

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeBiCGSTABLoop<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_res_bis.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_Ap.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_s.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_As.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_hadamard_prod.ensure_sizes(edges.sizes.indexes);

        // r_0 = b - A phi_0, r'_0 = r_0 / 2, p_0 = r_0
        node_init.evaluate();

        u32 k                     = 0;
        bool converged            = false;
        Tscal last_residual_norm2 = 0;
        while (k < Niter_max) {

            // <r_k, r'_0>
            node_had_prod_rj_rp0.evaluate();
            node_ddot_rj_rp0.evaluate();

            k = k + 1;

            // exchange the ghost blocks of p_k
            node_gz_p.evaluate();
            node_exch_gz_p.evaluate();
            node_replace_gz_p.evaluate();

            // A p_k
            node_Apj.evaluate();

            // <A p_k, r'_0>
            node_had_Apj_rp0.evaluate();
            node_ddot_Apj_rp0.evaluate();

            // alpha_k = <r_k, r'_0> / <A p_k, r'_0>
            edges.alpha.data = edges.old_values.data / edges.e_norm.data;

            Tscal alp_saved = edges.alpha.data;

            // s_k = r_k - alpha_k A p_k
            edges.e_norm.data = 0;
            edges.beta.data   = 1;
            edges.alpha.data  = -alp_saved;
            node_sj_vec.evaluate();

            // <s_k, s_k>
            node_ddot_sj_sj.evaluate();
            last_residual_norm2 = edges.e_norm.data;

            if (edges.e_norm.data < tol_cvg * tol_cvg) {
                // phi_{k+1} = phi_k + alpha_k p_k
                edges.alpha.data = alp_saved;
                node_new_phi_happy_break.evaluate();
                shamlog_debug_ln(
                    "Ramses::BiCGSTAB",
                    sham::format("k = {}, converged on <s,s> = {}", k, edges.e_norm.data));
                converged = true;
                break;
            }

            // exchange the ghost blocks of s_k
            node_gz_s.evaluate();
            node_exch_gz_s.evaluate();
            node_replace_gz_s.evaluate();

            // A s_k
            node_Asj.evaluate();

            // <A s_k, s_k>
            node_had_Asj_sj.evaluate();
            node_ddot_Asj_sj.evaluate();

            // <A s_k, A s_k>
            node_ddot_Asj_Asj.evaluate();

            // w_k = <A s_k, s_k> / <A s_k, A s_k>
            edges.w_stab.data = edges.e_norm.data / edges.new_values.data;

            Tscal w_saved = edges.w_stab.data;

            // phi_{k+1} = phi_k + alpha p_k + w_k s_k
            // TODO: (kept from #1942) alpha holds -alpha_k here (set for s_k above), whereas
            // BiCGSTAB uses phi_{k+1} = phi_k + alpha_k p_k + w_k s_k. As is, the recursively
            // updated residual converges but phi does not solve the Poisson equation; setting
            // alpha back to alp_saved before this update makes it converge at second order (see
            // PoissonSolversTests.cpp).
            node_new_phi.evaluate();

            // r_{k+1} = s_k - w_k A s_k
            edges.w_stab.data     = -w_saved;
            edges.e_norm.data     = 0;
            edges.new_values.data = 1;
            node_new_res.evaluate();

            // <r_{k+1}, r_{k+1}>
            node_ddot_rj_rj.evaluate();
            last_residual_norm2 = edges.e_norm.data;

            shamlog_debug_ln(
                "Ramses::BiCGSTAB", sham::format("k = {}, <r,r> = {}", k, edges.e_norm.data));

            if (edges.e_norm.data < tol_cvg * tol_cvg) {
                converged = true;
                break;
            }

            // <r_{k+1}, r'_0>
            node_had_rjnew_rp0.evaluate();
            node_ddot_rjnew_rp0.evaluate();

            // beta_k = (alpha_k / w_k) (<r_{k+1}, r'_0> / <r_k, r'_0>)
            edges.beta.data
                = (alp_saved / w_saved) * (edges.new_values.data / edges.old_values.data);

            // p_{k+1} = r_{k+1} + beta_k (p_k - w_k A p_k)
            edges.e_norm.data = 1;
            edges.alpha.data  = (-w_saved) * (edges.beta.data);
            node_new_p_vec.evaluate();

            // lucky breakdown: restart with r'_0 = p_{k+1} = r_{k+1}
            if ((edges.new_values.data * edges.new_values.data) < (tol_happy_bk * tol_happy_bk)) {
                shamlog_debug_ln("Ramses::BiCGSTAB", sham::format("k = {}, lucky breakdown", k));
                node_overwrite_rp0.evaluate();
                node_overwrite_p.evaluate();
            }
        }

        last_iteration_count = k;

        if (shamcomm::world_rank() == 0) {
            Tscal res = std::sqrt(last_residual_norm2);
            if (converged) {
                logger::info_ln(
                    "Ramses::BiCGSTAB",
                    sham::format("converged after {} iterations, |r| = {}", k, res));
            } else {
                logger::warn_ln(
                    "Ramses::BiCGSTAB",
                    sham::format(
                        "not converged after {} iterations, |r| = {} (tol = {})", k, res, tol_cvg));
            }
        }
    }

    template<class Tvec, class TgridVec>
    std::string NodeBiCGSTABLoop<Tvec, TgridVec>::_impl_get_tex() const {
        std::string tex = R"tex(
            BiCGSTAB solve of the Poisson equation
            \begin{equation}
            -\Delta x^3 \Delta {spans_phi} = -4 \pi G ({spans_rho} - {mean_rho}) \Delta x^3
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeBiCGSTABLoop<f64_3, i64_3>;
