"""
Generating gravitational choreography initial conditions
========================================================

This is the tool that produced the ``CHOREOGRAPHIES`` table in
``examples/sph/run_sink_integ_test_problems.py``. It is kept so the table can
be regenerated, extended with more orbits, or checked, rather than being a set
of magic numbers nobody can reproduce.

A gravitational choreography is a periodic solution of the N-body problem in
which every body travels along the same closed curve, equally spaced in time:
body ``i`` sits at ``q(t + i T / N)``. Such a loop is a critical point of the
action

.. math::
    A[q] = \\int_0^T \\left[ \\sum_i \\frac{1}{2} |\\dot{q}_i|^2
           + \\sum_{i<j} \\frac{G m^2}{|q_i - q_j|} \\right] dt

and the search below works in three stages.

**Finding the loops.** The loop is expanded in a Fourier series and the action
is made stationary. The equations are solved as a root-finding problem on
``grad A = 0`` rather than by minimising ``A``: the figure eight is the only
orbit here that minimises the action, and every other one is a saddle point, so
a minimiser walks away from them and collapses onto a collision. Modes with
``k = 0 mod N`` are dropped, being exactly the ones that move the centre of
mass, since ``sum_i exp(i k (t + 2 pi i / N)) = 0`` unless ``N`` divides ``k``.
Starting points are random loops with a ``1/k^2`` spectrum, and solutions are
told apart by their action.

**Refining them.** A truncated Fourier series solves a projected problem, not
the real one. The defining property of a choreography gives an exact condition
instead: with ``tau = T / N``, integrating every body for one ``tau`` has to
cyclically permute the state,

.. math::
    \\Phi_\\tau(z) = P z, \\qquad (P z)_i = z_{i+1 \\bmod N}

Solving that for ``z`` with ``tau`` held at ``2 pi / N`` pins the period and
leaves initial conditions limited only by the reference integrator. The Newton
Jacobian comes from the variational equations integrated alongside the
trajectory, so no finite differences are involved and it converges to round-off
in two or three iterations. This matters: the Fourier values were up to 1e-6
off, and on an orbit whose perturbations grow by seven e-folds per period that
alone costs several periods of usable lifetime.

**Grading them.** The same variational equations give the monodromy matrix over
a full period, whose largest multiplier is the Floquet rate quoted in the table.
Two checks come for free: the multipliers of a Hamiltonian system have to come
in reciprocal pairs, and the ring solutions have to match the analytic
``omega^2 = (G m / 4 R^3) sum_k 1 / sin(pi k / N)``.

Everything here is in the natural normalisation of the problem, ``G = 1``, unit
masses and period ``2 pi``. The example rescales to code units at run time from
whatever ``G`` the unit system reports, which is why the numbers stay an exact
solution rather than assuming ``G = 4 pi^2``.

One caveat on regenerating. A choreography is only defined up to a rigid
rotation and a shift of the time origin, and nothing here picks a preferred one,
so a fresh run finds the same orbits at a different place along the loop and at
a different orientation. The coordinates will not match the committed table term
by term. Everything that does not depend on that choice will: re-running the
``N = 3`` search reproduces actions of 19.604328 and 24.371926, rates of 4.4429
and 0 e-folds per period, and loop radii of 0.8327 and 1.0761, all matching the
table exactly. Replacing an entry means taking the whole entry, not splicing
new positions into an old one.

.. note::
    This file is deliberately not named ``run_*``, so the documentation build
    renders it without executing it: a full search over ``N = 3`` to ``6`` takes
    a good fraction of an hour.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

TWO_PI = 2.0 * np.pi
DIM = 2

# tolerances for the reference integrator used by the refinement and the
# monodromy; near the smallest scipy will accept
RTOL = 3e-14
ATOL = 1e-15


# %%
# The Fourier representation of a loop, and the action on it
class ChoreoBasis:
    """Fourier basis for a choreography loop of period ``2 pi``.

    ``x(t) = sum ax_k cos(k t) + sum bx_k sin(k t)`` and likewise for ``y``,
    over the modes ``k`` that are not multiples of ``nbody``.
    """

    def __init__(self, nbody, kmax, ngrid_per_body=250):
        modes = np.array([k for k in range(1, kmax + 1) if k % nbody != 0], dtype=int)
        self.nbody = nbody
        self.groups = [("x", "cos", modes), ("x", "sin", modes)]
        self.groups += [("y", "cos", modes), ("y", "sin", modes)]
        self.offsets = np.cumsum([0] + [group[2].size for group in self.groups])
        self.nparam = int(self.offsets[-1])

        self.ngrid = nbody * ngrid_per_body
        theta = TWO_PI * np.arange(self.ngrid) / self.ngrid
        self.design = [
            (np.cos if kind == "cos" else np.sin)(np.outer(theta, ks))
            for _comp, kind, ks in self.groups
        ]

    def split(self, params):
        return [params[self.offsets[i] : self.offsets[i + 1]] for i in range(len(self.groups))]

    def curve_on_grid(self, params):
        coeffs = self.split(params)
        qx = self.design[0] @ coeffs[0] + self.design[1] @ coeffs[1]
        qy = self.design[2] @ coeffs[2] + self.design[3] @ coeffs[3]
        return qx, qy

    def eval(self, params, times):
        """Position and velocity of the generating body at the given times."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        pos = np.zeros((times.size, DIM))
        vel = np.zeros((times.size, DIM))
        for (comp, kind, ks), coeff in zip(self.groups, self.split(params)):
            if ks.size == 0:
                continue
            axis = 0 if comp == "x" else 1
            arg = np.outer(times, ks)
            if kind == "cos":
                pos[:, axis] += np.cos(arg) @ coeff
                vel[:, axis] -= np.sin(arg) @ (ks * coeff)
            else:
                pos[:, axis] += np.sin(arg) @ coeff
                vel[:, axis] += np.cos(arg) @ (ks * coeff)
        return pos, vel

    def action(self, params, grad=False):
        """The action, and optionally its analytic gradient.

        The kinetic term is exact in the coefficients. The potential is summed
        on a uniform grid, which is spectrally accurate for a smooth periodic
        integrand. Taking the grid size to be a multiple of ``nbody`` makes the
        shifted positions of the other bodies land on grid points too, so a pair
        separation is just the curve against a rolled copy of itself.
        """
        nbody, ngrid = self.nbody, self.ngrid
        coeffs = self.split(params)

        kinetic = 0.0
        for (_comp, _kind, ks), coeff in zip(self.groups, coeffs):
            kinetic += np.sum((ks * coeff) ** 2)
        kinetic *= 0.5 * nbody * np.pi

        qx, qy = self.curve_on_grid(params)
        gx = np.zeros(ngrid)
        gy = np.zeros(ngrid)
        potential = 0.0
        for shift in range(1, nbody):
            offset = shift * ngrid // nbody
            dx = qx - np.roll(qx, -offset)
            dy = qy - np.roll(qy, -offset)
            radius = np.sqrt(dx * dx + dy * dy)
            weight = (TWO_PI / ngrid) * (nbody - shift)
            potential += weight * np.sum(1.0 / radius)
            if grad:
                inv_cube = 1.0 / radius**3
                tx = weight * dx * inv_cube
                ty = weight * dy * inv_cube
                gx += -tx + np.roll(tx, offset)
                gy += -ty + np.roll(ty, offset)

        if not grad:
            return kinetic + potential

        gradient = np.zeros(self.nparam)
        for i, ((comp, _kind, ks), coeff) in enumerate(zip(self.groups, coeffs)):
            if ks.size == 0:
                continue
            gvec = gx if comp == "x" else gy
            gradient[self.offsets[i] : self.offsets[i + 1]] = (
                self.design[i].T @ gvec + nbody * np.pi * (ks**2) * coeff
            )
        return kinetic + potential, gradient

    def initial_conditions(self, params):
        """Positions and velocities of all N bodies at ``t = 0``."""
        return self.eval(params, (TWO_PI / self.nbody) * np.arange(self.nbody))

    def min_separation(self, params):
        """Smallest pairwise distance along the loop, on the quadrature grid."""
        qx, qy = self.curve_on_grid(params)
        smallest = np.inf
        for shift in range(1, self.nbody):
            offset = shift * self.ngrid // self.nbody
            dx = qx - np.roll(qx, -offset)
            dy = qy - np.roll(qy, -offset)
            smallest = min(smallest, float(np.min(np.hypot(dx, dy))))
        return smallest


# %%
# Searching for critical points of the action
def solve_critical_point(basis, guess):
    """Drive ``grad A`` to zero. Converges onto saddles as well as minima."""
    result = least_squares(
        lambda params: basis.action(params, grad=True)[1],
        guess,
        method="lm",
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=20000,
    )
    return result.x, float(np.max(np.abs(result.fun)))


def search_choreographies(nbody, ntry=200, kmax=20, seed=1, min_separation=5e-2, verbose=True):
    """Multistart search, returning one parameter vector per distinct action."""
    basis = ChoreoBasis(nbody, kmax)
    rng = np.random.default_rng(seed)
    modes = np.concatenate([group[2] for group in basis.groups])
    found = {}

    for attempt in range(ntry):
        # a random loop whose spectrum falls off like 1/k^2, so it is dominated
        # by the low modes and looks like something a choreography could relax to
        guess = rng.normal(size=basis.nparam) / modes**2
        guess *= rng.uniform(0.6, 1.6) / np.linalg.norm(guess)
        try:
            params, residual = solve_critical_point(basis, guess)
        except (ValueError, np.linalg.LinAlgError) as error:
            # a start that runs into a collision makes the action non-finite and
            # the solver gives up; that is expected, most random loops do it
            if verbose:
                print(f"    start {attempt} abandoned: {error}")
            continue
        if residual > 1e-9:
            continue
        if basis.min_separation(params) < min_separation:
            continue
        action = float(basis.action(params))
        if not np.isfinite(action) or action > 500.0:
            continue
        key = round(action, 3)
        if key not in found or residual < found[key][1]:
            found[key] = (params, residual)

    if verbose:
        print(f"N={nbody}: {len(found)} distinct critical points")
        for key in sorted(found):
            print(f"    action = {key:10.3f}")
    return basis, [found[key][0] for key in sorted(found)]


# %%
# The N-body equations with their variational equations
def accelerations_and_jacobian(pos, masses, want_jacobian=True):
    """Accelerations, and ``d(acc)/d(pos)`` for the softening-free problem."""
    nbody = masses.size
    delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r_squared = np.sum(delta**2, axis=-1)
    np.fill_diagonal(r_squared, np.inf)
    inv_cube = r_squared**-1.5
    acc = np.einsum("ij,ijk,j->ik", inv_cube, delta, masses)
    if not want_jacobian:
        return acc, None

    inv_fifth = r_squared**-2.5
    jac = np.zeros((nbody, DIM, nbody, DIM))
    eye = np.eye(DIM)
    for i in range(nbody):
        for j in range(nbody):
            if i == j:
                continue
            block = masses[j] * (
                inv_cube[i, j] * eye - 3.0 * inv_fifth[i, j] * np.outer(delta[i, j], delta[i, j])
            )
            jac[i, :, j, :] = block
            jac[i, :, i, :] -= block
    return acc, jac.reshape(nbody * DIM, nbody * DIM)


def rhs_with_variational(_t, state, masses, nvar):
    nbody = masses.size
    ndof = nbody * DIM
    pos = state[:ndof].reshape(nbody, DIM)
    acc, jac = accelerations_and_jacobian(pos, masses, want_jacobian=nvar > 0)

    out = np.empty_like(state)
    out[:ndof] = state[ndof : 2 * ndof]
    out[ndof : 2 * ndof] = acc.ravel()
    if nvar:
        transition = state[2 * ndof :].reshape(2 * ndof, nvar)
        derivative = np.empty_like(transition)
        derivative[:ndof] = transition[ndof:]
        derivative[ndof:] = jac @ transition[:ndof]
        out[2 * ndof :] = derivative.ravel()
    return out


def flow(state, duration, masses, variational=False):
    """Integrate the state, and optionally the state transition matrix."""
    ndof = masses.size * DIM
    nvar = 2 * ndof if variational else 0
    initial = np.concatenate([state, np.eye(2 * ndof).ravel()]) if variational else state.copy()
    solution = solve_ivp(
        rhs_with_variational,
        (0.0, duration),
        initial,
        args=(masses, nvar),
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
    )
    final = solution.y[:, -1]
    if not variational:
        return final[: 2 * ndof], None
    return final[: 2 * ndof], final[2 * ndof :].reshape(2 * ndof, 2 * ndof)


# %%
# Refining on the exact equations of motion, and grading the result
def cyclic_permutation_matrix(nbody):
    """``(P z)_i = z_{i+1 mod N}`` acting on the stacked positions and velocities."""
    ndof = nbody * DIM
    permutation = np.zeros((2 * ndof, 2 * ndof))
    for i in range(nbody):
        j = (i + 1) % nbody
        for axis in range(DIM):
            permutation[i * DIM + axis, j * DIM + axis] = 1.0
            permutation[ndof + i * DIM + axis, ndof + j * DIM + axis] = 1.0
    return permutation


def shoot_refine(pos, vel, nbody, maxit=30):
    """Gauss-Newton on ``Phi_tau(z) - P z = 0`` with ``tau = 2 pi / N`` fixed.

    The residual has a two dimensional null space, time translation along the
    orbit and rigid rotation, so the step uses a pseudo-inverse.
    """
    masses = np.ones(nbody)
    tau = TWO_PI / nbody
    permutation = cyclic_permutation_matrix(nbody)
    state = np.concatenate([np.asarray(pos).ravel(), np.asarray(vel).ravel()])

    residual = np.inf
    for _iteration in range(maxit):
        shifted, transition = flow(state, tau, masses, variational=True)
        defect = shifted - permutation @ state
        residual = float(np.max(np.abs(defect)))
        if residual < 1e-14:
            break
        step, *_ = np.linalg.lstsq(transition - permutation, -defect, rcond=1e-10)
        candidate = state + step
        # near round-off Gauss-Newton can overshoot; keep the step only if it helps
        moved, _ = flow(candidate, tau, masses, variational=False)
        if np.max(np.abs(moved - permutation @ candidate)) > residual:
            break
        state = candidate
    return state, residual


def floquet_rate(state, nbody):
    """Largest Floquet exponent, in e-folds per period, and the reciprocal check.

    A Hamiltonian system's multipliers come in pairs ``mu`` and ``1/mu``, so the
    product of the largest and smallest is a free check on the monodromy.
    """
    _final, monodromy = flow(state, TWO_PI, np.ones(nbody), variational=True)
    magnitudes = np.sort(np.abs(np.linalg.eigvals(monodromy)))[::-1]
    reciprocal_error = float(abs(magnitudes[0] * magnitudes[-1] - 1.0))
    return float(np.log(magnitudes[0])), reciprocal_error


def equation_of_motion_residual(basis, params, nsample=2000):
    """``max |qddot - a_grav| / max |a_grav|`` along the loop, without integrating."""
    nbody = basis.nbody
    times = np.linspace(0.0, TWO_PI, nsample, endpoint=False)
    tau = TWO_PI / nbody
    bodies = np.stack([basis.eval(params, times + j * tau)[0] for j in range(nbody)])

    acc = np.zeros((nsample, DIM))
    for (comp, kind, ks), coeff in zip(basis.groups, basis.split(params)):
        if ks.size == 0:
            continue
        axis = 0 if comp == "x" else 1
        arg = np.outer(times, ks)
        trig = np.cos(arg) if kind == "cos" else np.sin(arg)
        acc[:, axis] -= trig @ (ks**2 * coeff)

    gravity = np.zeros((nsample, DIM))
    for j in range(1, nbody):
        delta = bodies[j] - bodies[0]
        gravity += delta / np.linalg.norm(delta, axis=1)[:, None] ** 3

    return float(
        np.max(np.linalg.norm(acc - gravity, axis=1)) / np.max(np.linalg.norm(gravity, axis=1))
    )


def loop_radius(state, nbody, nsample=2000):
    """Largest distance from the centre of mass reached over one period."""
    masses = np.ones(nbody)
    ndof = nbody * DIM
    solution = solve_ivp(
        rhs_with_variational,
        (0.0, TWO_PI),
        state,
        args=(masses, 0),
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
        dense_output=True,
    )
    times = np.linspace(0.0, TWO_PI, nsample)
    positions = solution.sol(times)[:ndof].reshape(nbody, DIM, nsample)
    return float(np.max(np.linalg.norm(positions, axis=1)))


# %%
# Emitting a table entry in the form the SPH example expects
def format_entry(key, name, state, nbody, action, rate, radius):
    """One ``CHOREOGRAPHIES`` entry, ready to paste into the example."""
    ndof = nbody * DIM
    pos = state[:ndof].reshape(nbody, DIM)
    vel = state[ndof:].reshape(nbody, DIM)

    # These three are starting points only. They are tuning knobs, not physics,
    # and were settled in the example by measuring how far the sinks drift: a
    # faster Floquet rate needs a smaller timestep and a shorter window.
    n_periods = 10 if rate < 1e-3 else 1
    deviation_periods = 10 if rate < 1e-3 else int(np.clip(round(28.0 / rate), 2, 5))
    eta_sink = 0.2 if rate < 5.0 else 0.05

    lines = [f'    "{key}": {{']
    lines.append(f'        "name": "{name}",')
    lines.append(f'        "n_periods": {n_periods},  # tune by running the example')
    lines.append(
        f'        "deviation_periods": {deviation_periods},  # tune by running the example'
    )
    lines.append(f'        "eta_sink": {eta_sink},  # tune by running the example')
    lines.append(f'        "lyapunov_rate": {rate:.4f},')
    lines.append(f'        "loop_radius": {radius:.4f},')
    lines.append(f"        # action {action:.6f}")
    lines.append('        "positions": [')
    for row in pos:
        lines.append(f"            ({row[0]: .16f}, {row[1]: .16f}, 0.0),")
    lines.append("        ],")
    lines.append('        "velocities": [')
    for row in vel:
        lines.append(f"            ({row[0]: .16f}, {row[1]: .16f}, 0.0),")
    lines.append("        ],")
    lines.append("    },")
    return "\n".join(lines)


# %%
# Driver
def generate(nbodies=(3, 4, 5, 6), ntry=200, search_kmax=20, refine_kmax=48, seed=1):
    """Search, refine and grade, printing a table entry for each orbit found."""
    entries = []

    for nbody in nbodies:
        basis, solutions = search_choreographies(nbody, ntry=ntry, kmax=search_kmax, seed=seed)

        for params in solutions:
            # re-solve in a larger basis before shooting, so the starting point
            # for Newton is already close
            wide = ChoreoBasis(nbody, refine_kmax)
            padded = np.zeros(wide.nparam)
            for group in range(len(basis.groups)):
                small = list(basis.groups[group][2])
                large = list(wide.groups[group][2])
                block = params[basis.offsets[group] : basis.offsets[group + 1]]
                for mode, value in zip(small, block):
                    padded[wide.offsets[group] + large.index(mode)] = value
            params_wide, _residual = solve_critical_point(wide, padded)

            action = float(wide.action(params_wide))
            eom = equation_of_motion_residual(wide, params_wide)

            pos, vel = wide.initial_conditions(params_wide)
            state, defect = shoot_refine(pos, vel, nbody)
            rate, reciprocal_error = floquet_rate(state, nbody)
            radius = loop_radius(state, nbody)

            key = f"n{nbody}_a{action:.0f}".replace(".", "")
            print(
                f"\nN={nbody} action={action:.6f} EoM residual={eom:.1e} "
                f"|Phi_tau(z) - P z|={defect:.1e} lambda={rate:.4f}/period "
                f"(reciprocal check {reciprocal_error:.0e}) loop radius={radius:.4f}"
            )
            print(
                format_entry(
                    key, f"N={nbody} action {action:.3f}", state, nbody, action, rate, radius
                )
            )
            entries.append((nbody, action, state, rate, radius))

    return entries


if __name__ == "__main__":
    # A full sweep over N = 3 to 6 takes a good fraction of an hour, most of it
    # in the multistart search. Narrow ``nbodies`` or drop ``ntry`` to try it out.
    generate()
