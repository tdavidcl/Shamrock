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
 * @file ShearingPeriodicUtils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Shared definitions of the shearing-periodic boundary of the Ramses solver
 *
 * The ghost blocks beyond the two shearing faces get the geometry of a reflective boundary
 * (aligned on the integer grid). Their values are then interpolated from the cells of the opposite
 * face, shifted along the shear direction by the shear offset (see InterpolateShearGhosts).
 *
 * Shift convention (same as the SPH solver): the periodic image of the box at \f$ +L \f$ along
 * the base axis is displaced by \f$ +o \f$ along the shear direction, where \f$ o \f$ is the shear
 * offset in grid units. The source cells of the ghosts beyond the face on side \f$ s = \pm 1 \f$
 * are moved by the integer \f$ T_s = \lceil s\,o \rceil \f$ along the shear direction, so that
 * they stay on the integer grid. The footprint of a target ghost cell \f$ [a, b) \f$ among those
 * moved cells is then \f$ [a + \varphi_s, b + \varphi_s) \f$ with
 * \f$ \varphi_s = T_s - s\,o \in [0, 1) \f$.
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/sycl.hpp"
#include "shammath/AABB.hpp"
#include <cmath>

namespace shammodels::basegodunov::modules {

    /// Geometry of the shearing-periodic boundary
    struct ShearingPeriodicGeom {
        u32 base_axis;       ///< axis whose two faces are shearing periodic
        u32 dir_axis;        ///< axis along which the periodic images slide
        bool third_periodic; ///< whether the remaining axis is periodic
    };

    /// Index of the axis that is neither the base nor the shear direction axis
    inline u32 get_third_axis(const ShearingPeriodicGeom &geom) {
        return 3 - geom.base_axis - geom.dir_axis;
    }

    /// Split of the shear shift for one side of the box
    struct ShearSideShift {
        i64 int_shift; ///< integer translation applied to the source cells (grid units)
        f64 phi;       ///< residual offset of the target footprint, in [0, 1) (grid units)
    };

    /**
     * @brief Split the shear offset for the ghosts beyond the face on side `side`
     *
     * @param offset_grid shear offset \f$ o \f$ in grid units, wrapped in \f$ [-L/2, L/2] \f$
     * @param side +1 for the ghosts beyond the upper face, -1 for the lower face
     */
    inline ShearSideShift get_shear_side_shift(f64 offset_grid, i32 side) {
        f64 o = f64(side) * offset_grid;
        f64 t = std::ceil(o);
        f64 p = t - o;
        // guard against rounding, the residual must be in [0, 1)
        if (!(p >= 0) || !(p < 1)) {
            p = 0;
        }
        return {i64(t), p};
    }

    /**
     * @brief Shear offset in grid units at time `t`, wrapped in \f$ [-L/2, L/2] \f$
     *
     * Recomputed from the time at every step (not accumulated) so that long runs keep full
     * precision. Centring it on zero keeps the periodic tiles needed along the shear direction in
     * \f$ \{-1, 0, 1\} \f$.
     */
    inline f64 get_shear_offset_grid(f64 shear_speed, f64 t, f64 dxfact, i64 box_size_dir) {
        return std::remainder(shear_speed * t / dxfact, f64(box_size_dir));
    }

    /// Overlap of a target cell footprint with a source cell
    template<class Tscal>
    struct ShearOverlap {
        bool positive; ///< the overlap has a non zero volume
        i64 vol_int;   ///< integer part of the overlap volume
        i64 vol_coef;  ///< coefficient of phi in the overlap volume
        Tscal dr;      ///< centroid of the overlap minus the source cell centre, along dir
    };

    /**
     * @brief Overlap of the footprint of the target cell `t` (shifted by `phi` along `idir`) with
     * the source cell `s`
     *
     * All bounds are integers and 0 <= phi < 1, so the overlap test only uses integer
     * comparisons, and the overlap volume is vol_int + phi * vol_coef exactly.
     */
    template<class Tscal, class TgridVec>
    inline ShearOverlap<Tscal> compute_shear_overlap(
        const shammath::AABB<TgridVec> &t, const shammath::AABB<TgridVec> &s, u32 idir, Tscal phi) {

        ShearOverlap<Tscal> ret{false, 0, 0, 0};

        i64 lperp = 1;
        for (u32 ax = 0; ax < 3; ax++) {
            if (ax == idir) {
                continue;
            }
            i64 l = sycl::min(t.upper[ax], s.upper[ax]) - sycl::max(t.lower[ax], s.lower[ax]);
            if (l <= 0) {
                return ret;
            }
            lperp *= l;
        }

        const i64 a = t.lower[idir];
        const i64 b = t.upper[idir];
        const i64 c = s.lower[idir];
        const i64 d = s.upper[idir];

        // max(a + phi, c) and min(b + phi, d) as (integer part, carries phi)
        const bool lo_phi = (a >= c);
        const i64 lo_int  = lo_phi ? a : c;
        const bool hi_phi = (b < d);
        const i64 hi_int  = hi_phi ? b : d;

        const bool has_phi = phi > 0;
        const i64 len_int  = hi_int - lo_int;
        const i64 len_coef = i64(hi_phi) - i64(lo_phi);

        // len = len_int + phi * len_coef with len_coef in {-1, 0, 1} and phi < 1
        const bool positive = (len_int > 0) || (len_int == 0 && len_coef == 1 && has_phi);
        if (!positive) {
            return ret;
        }

        ret.positive = true;
        ret.vol_int  = lperp * len_int;
        ret.vol_coef = has_phi ? lperp * len_coef : 0;

        // ((lo + hi) - (c + d)) / 2, formed from small integers and one phi term
        Tscal centroid_phi = has_phi ? phi * Tscal(i64(lo_phi) + i64(hi_phi)) : Tscal(0);
        ret.dr             = (Tscal((lo_int + hi_int) - (c + d)) + centroid_phi) * Tscal(0.5);

        return ret;
    }

    /// Information on a shear source interface (sender patch, receiver patch)
    template<class TgridVec>
    struct ShearSourceInfos {
        i32 side;        ///< side of the receiver ghosts the sender cells feed (+1 or -1)
        TgridVec offset; ///< integer translation of the sender blocks to the receiver frame
    };

} // namespace shammodels::basegodunov::modules
