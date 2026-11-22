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
 * @file InterpolateShearGhosts.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Fill the ghost cells beyond the shearing-periodic faces
 *
 * The ghost blocks beyond the shearing faces were built with a reflective geometry, so they are
 * aligned on the integer grid. This node overwrites their values with a conservative remap of the
 * cells of the opposite face (received through FindShearSourceGhosts / TransformShearSource),
 * then applies the velocity jump across the face.
 *
 * Remap of a target ghost cell \f$ [a, b) \f$ (integer bounds, along the shear direction): the
 * source cells were translated by the integer part of the shift, so the footprint of the target is
 * \f$ [a + \varphi, b + \varphi) \f$ with \f$ 0 \le \varphi < 1 \f$. For a source cell
 * \f$ [c, d) \f$, all bounds being integers,
 * \f[
 *   \max(a + \varphi, c) = (a \ge c)\ ?\ a + \varphi : c, \qquad
 *   \min(b + \varphi, d) = (b < d)\ ?\ b + \varphi : d,
 * \f]
 * so which source cells overlap the footprint is decided with integer comparisons only, and
 * \f$ \varphi \f$ only enters the overlap length and the centroid of the overlap. The value of the
 * target is the overlap-weighted average of the slope-limited linear reconstruction of the source
 * cells along the shear direction. Targets whose footprint is not fully covered by source cells
 * (e.g. corners beyond a non-periodic face) keep their reflective values.
 *
 * Velocity jump: with \f$ w = s\, v_{\rm shear} \f$ for the ghosts beyond the face on side
 * \f$ s = \pm 1 \f$: \f$ \rho E \leftarrow \rho E + w\, \rho v_{\rm dir} + \rho w^2 / 2 \f$,
 * then \f$ \rho v_{\rm dir} \leftarrow \rho v_{\rm dir} + \rho w \f$ (also for the dust momenta).
 */

#include "shammath/AABB.hpp"
#include "shammodels/ramses/config/enum_SlopeMode.hpp"
#include "shammodels/ramses/modules/ShearingPeriodicUtils.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>, sim_box)                     \
    X_RO(shamrock::solvergraph::ScalarEdge<f64>, shear_offset)                                     \
    X_RO(shamrock::solvergraph::PatchDataLayerDDShared, shear_src)                                 \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IPatchDataLayerRefs, patch_data_layers)

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class InterpolateShearGhosts : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        ShearingPeriodicGeom geom;
        Tscal shear_speed;
        SlopeMode slope_mode;
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout;

        public:
        InterpolateShearGhosts(
            ShearingPeriodicGeom geom,
            Tscal shear_speed,
            SlopeMode slope_mode,
            std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout)
            : geom(geom), shear_speed(shear_speed), slope_mode(slope_mode),
              ghost_layer_layout(std::move(ghost_layer_layout)) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "InterpolateShearGhosts"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
