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
 * @file ShearingBoxForceUtils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Time integration of the shearing box forces (tidal, Coriolis and radial pressure
 * gradient forcing) of the Ramses solver
 */

#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shammodels::basegodunov::modules {

    /**
     * @brief Momentum at t^{n+1} including the kick of the shearing box forces
     *
     * The source term of the momentum is
     * \f[
     *  \mathbf{S}(\rho, \mathbf{p}) = \rho g_x \mathbf{e}_x + \mathbf{C} \mathbf{p}, \qquad
     *  \mathbf{C} \mathbf{p} = 2 \Omega_0 (p_y, - p_x, 0)
     * \f]
     * with \f$ g_x \f$ the velocity independent radial acceleration (tidal term
     * \f$ 2 q \Omega_0^2 x' \f$, plus the pressure-gradient forcing \f$ 2 \Omega_0 \eta \f$ for the
     * gas) and \f$ \mathbf{C} \f$ the Coriolis term. It is applied after the flux update
     * \f$ \mathbf{p}^* = \mathbf{p}^n + \Delta t \, \partial_t \mathbf{p}_{\rm flux} \f$,
     * \f$ \rho^{n+1} = \rho^n + \Delta t \, \partial_t \rho \f$ (operator splitting), with the
     * trapezoidal rule, implicit in the momentum (Crank-Nicolson):
     * \f[
     *  \mathbf{p}^{n+1} = \mathbf{p}^* + \frac{\Delta t}{2} \left(
     *      \mathbf{S}(\rho^{n+1}, \mathbf{p}^*) + \mathbf{S}(\rho^{n+1}, \mathbf{p}^{n+1}) \right)
     * \f]
     * whose 2x2 linear system has the closed form solution (\f$ \theta = \Omega_0 \Delta t \f$)
     * \f[
     *  \mathbf{r} = \mathbf{p}^* + \frac{\Delta t}{2} \mathbf{C} \mathbf{p}^*
     *      + \Delta t \rho^{n+1} g_x \mathbf{e}_x , \quad
     *  p^{n+1}_x = \frac{r_x + \theta r_y}{1 + \theta^2}, \quad
     *  p^{n+1}_y = \frac{r_y - \theta r_x}{1 + \theta^2}
     * \f]
     *
     * Properties:
     *  - unconditionally stable, the Coriolis rotation keeps the norm of the momentum (Cayley
     *    transform of the rotation, phase error O((Omega dt)^3) per step);
     *  - the kinetic energy change is exactly the trapezoidal work of g_x (the Coriolis term does
     *    no work);
     *  - the equilibrium shear flow v_y = -q Omega_0 x' - eta (without fluxes) is kept to
     *    round-off;
     *  - epicycles: the background shear advection term (+q Omega v_x on v_y) comes from the
     *    explicit fluxes. Evaluating the explicit side at p* rather than at p^n makes the step
     *    map of (v_x, v_y) the product of a rotation and a shear, both of determinant one: the
     *    epicycles are not damped (using p^n gives a determinant 1 - q theta^2 / (1 + theta^2),
     *    i.e. a damping rate q Omega^2 dt / 2), and their frequency is kappa^2 = 2 (2 - q)
     *    Omega^2 to leading order.
     *
     * @param p_star momentum at t^{n+1} without the shearing box forces
     * @param rho_np1 density at t^{n+1}
     * @param g_x velocity independent radial acceleration
     * @param Omega_0 angular velocity of the frame
     * @param dt time step
     * @return the momentum at t^{n+1}
     */
    template<class Tvec, class Tscal = shambase::VecComponent<Tvec>>
    inline Tvec shearing_box_momentum_update(
        Tvec p_star, Tscal rho_np1, Tscal g_x, Tscal Omega_0, Tscal dt) {

        Tscal theta = Omega_0 * dt;

        Tscal r_x = p_star.x() + theta * p_star.y() + dt * rho_np1 * g_x;
        Tscal r_y = p_star.y() - theta * p_star.x();

        Tscal inv_det = Tscal{1} / (Tscal{1} + theta * theta);

        return Tvec{(r_x + theta * r_y) * inv_det, (r_y - theta * r_x) * inv_det, p_star.z()};
    }

    /**
     * @brief Change of the total energy of the gas due to the kick of the shearing box forces
     *
     * The kick changes the kinetic energy only, so the internal energy at t^{n+1} is the one
     * without the shearing box forces:
     * \f[
     *  \Delta (\rho E) = \frac{|\mathbf{p}^{n+1}|^2 - |\mathbf{p}^*|^2}{2 \rho^{n+1}}
     *      = \frac{(\mathbf{p}^{n+1} - \mathbf{p}^*) \cdot (\mathbf{p}^{n+1} + \mathbf{p}^*)}
     *      {2 \rho^{n+1}}
     * \f]
     * which is the work of the forces over the step (the Coriolis term does no work, its kinetic
     * energy change being the one of the tidal and pressure-gradient terms).
     *
     * @param p_star momentum at t^{n+1} without the shearing box forces
     * @param p_np1 momentum at t^{n+1} with the shearing box forces
     * @param rho_np1 density at t^{n+1}
     * @return the total energy change
     */
    template<class Tvec, class Tscal = shambase::VecComponent<Tvec>>
    inline Tscal shearing_box_energy_update(Tvec p_star, Tvec p_np1, Tscal rho_np1) {
        return sycl::dot(p_np1 - p_star, p_np1 + p_star) / (2 * rho_np1);
    }

} // namespace shammodels::basegodunov::modules
