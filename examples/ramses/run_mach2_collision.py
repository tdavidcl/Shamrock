"""
Mach 2 colliding flows in RAMSES solver
=======================================

Two uniform flows collide head-on at Mach 2: :math:`\\rho = P = 1` on both sides,
:math:`v_x = +2 c_s` for :math:`x < 0.5` and :math:`v_x = -2 c_s` for :math:`x > 0.5`.
The exact solution is two shocks receding from the collision point, leaving gas at rest
in between.

This is a demanding case for HLLC with pressure based (PVRS) wave speed estimates: at the
collision face the PVRS underestimates the star pressure so much that the estimated wave
speeds cross (:math:`s_L > 0 > s_R`). Since the RAMSES solver computes each face flux from
both of its sides, each cell then sees itself as upwind and only lets mass out: no shock
forms and mass vanishes at the collision face. HLLC therefore averages the branch selected
by both sides of a face, which makes the two face fluxes exactly opposite.

We compare the Riemann solvers against the exact solution and check that the mass is
conserved.
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Setup parameters
# ^^^^^^^^^^^^^^^^
# A quasi 1D box: 128 cells along x over :math:`[0, 1]`, 8 cells along y and z.

gamma = 1.4
mach = 2.0
rho_0 = 1.0
P_0 = 1.0
cs_0 = np.sqrt(gamma * P_0 / rho_0)
v_0 = mach * cs_0
x_0 = 0.5
tmax = 0.25
n_outputs = 25

sz = 1 << 1  # cells per block along each axis
blocks = (64, 4, 4)
scale_fact = 1 / (sz * blocks[0])
box = [sz * b * scale_fact for b in blocks]
cross_section = box[1] * box[2]

positions_plot = [(x, box[1] / 2, box[2] / 2) for x in np.linspace(0, 1, 512)[:-1]]


# %%
# Exact solution
# ^^^^^^^^^^^^^^
# For a symmetric collision the gas between the two shocks is at rest. The star pressure
# :math:`P_*` solves the Rankine-Hugoniot relation :math:`v_0 = (P_* - P_0)
# \sqrt{A / (P_* + B)}` with :math:`A = 2 / ((\gamma + 1) \rho_0)` and
# :math:`B = P_0 (\gamma - 1) / (\gamma + 1)` (Toro, chapter 4).


def exact_solution(x, t):
    A = 2 / ((gamma + 1) * rho_0)
    B = (gamma - 1) / (gamma + 1) * P_0
    lo, hi = P_0, P_0 * (1 + 10 * mach) ** 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (mid - P_0) * np.sqrt(A / (mid + B)) > v_0:
            hi = mid
        else:
            lo = mid
    P_star = 0.5 * (lo + hi)
    G = (gamma - 1) / (gamma + 1)
    rho_star = rho_0 * (P_star / P_0 + G) / (G * P_star / P_0 + 1)
    s_shock = cs_0 * np.sqrt((gamma + 1) / (2 * gamma) * P_star / P_0 + (gamma - 1) / (2 * gamma))
    s_shock -= v_0  # speed of the right-going shock (the left one is symmetric)

    x = np.asarray(x)
    inside = np.abs(x - x_0) < s_shock * t
    rho = np.where(inside, rho_star, rho_0)
    vx = np.where(inside, 0.0, np.where(x < x_0, v_0, -v_0))
    P = np.where(inside, P_star, P_0)
    return rho, vx, P


# %%
# Run a case
# ^^^^^^^^^^
# The total mass is tracked through ``ctx.collect_data()``. Uniform flows enter through
# both outflow boundaries, so the mass should grow as :math:`M_0 + 2 \rho_0 v_0 S t`
# (with :math:`S` the box cross section) until the shocks reach the boundaries.


def total_mass(ctx):
    dic = ctx.collect_data()
    block_size = np.array(dic["cell_max"], dtype=float) - np.array(dic["cell_min"], dtype=float)
    vol = np.repeat(np.prod(block_size / sz * scale_fact, axis=1), sz**3)
    return float(np.sum(np.array(dic["rho"]) * vol))


def run_case(slope_limiter: str, riemann_solver: str):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(gamma)

    if slope_limiter == "none":
        cfg.set_slope_lim_none()
    elif slope_limiter == "minmod":
        cfg.set_slope_lim_minmod()
    else:
        raise ValueError(f"Invalid slope limiter: {slope_limiter}")

    if riemann_solver == "rusanov":
        cfg.set_riemann_solver_rusanov()
    elif riemann_solver == "hll":
        cfg.set_riemann_solver_hll()
    elif riemann_solver == "hllc":
        cfg.set_riemann_solver_hllc()
    else:
        raise ValueError(f"Invalid Riemann solver: {riemann_solver}")

    cfg.set_face_time_interpolation(True)
    cfg.set_boundary_condition("x", "outflow")
    cfg.set_boundary_condition("y", "outflow")
    cfg.set_boundary_condition("z", "outflow")
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), blocks)

    def vx_init(x):
        return v_0 if x < x_0 else -v_0

    def rho_map(rmin, rmax):
        return rho_0

    def rhoetot_map(rmin, rmax):
        x = 0.5 * (rmin[0] + rmax[0])
        return P_0 / (gamma - 1) + 0.5 * rho_0 * vx_init(x) ** 2

    def rhovel_map(rmin, rmax):
        x = 0.5 * (rmin[0] + rmax[0])
        return (rho_0 * vx_init(x), 0.0, 0.0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    print(f"running {slope_limiter} {riemann_solver}")

    mass_0 = total_mass(ctx)
    times, mass_err = [0.0], [0.0]
    for i in range(1, n_outputs + 1):
        t = tmax * i / n_outputs
        model.evolve_until(t)
        expected = mass_0 + 2 * rho_0 * v_0 * cross_section * t
        times.append(t)
        mass_err.append((total_mass(ctx) - expected) / mass_0)

    rho = np.array(model.render_slice("rho", "f64", positions_plot))
    rhovel = np.array(model.render_slice("rhovel", "f64_3", positions_plot))
    rhoetot = np.array(model.render_slice("rhoetot", "f64", positions_plot))
    vx = rhovel[:, 0] / rho
    P = (rhoetot - 0.5 * rho * vx**2) * (gamma - 1)

    return {"rho": rho, "vx": vx, "P": P, "t": np.array(times), "mass_err": np.array(mass_err)}


cases = [
    ("minmod", "rusanov"),
    ("minmod", "hll"),
    ("minmod", "hllc"),
    ("none", "hllc"),
]
results = {f"{lim} {solver}": run_case(lim, solver) for lim, solver in cases}

# %%
# Profiles against the exact solution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

arr_x = np.array([p[0] for p in positions_plot])
rho_ex, vx_ex, P_ex = exact_solution(arr_x, tmax)

fig, axes = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
fig.suptitle(f"Mach {mach:g} collision, t = {tmax}")
for ax, key, exact, label in zip(
    axes, ["rho", "vx", "P"], [rho_ex, vx_ex, P_ex], ["$\\rho$", "$v_x$", "$P$"]
):
    ax.plot(arr_x, exact, "k-", lw=1.5, label="exact")
    for name, res in results.items():
        ax.plot(arr_x, res[key], lw=1, label=name)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[0].legend()
axes[-1].set_xlabel("$x$")
plt.tight_layout()

for name, res in results.items():
    l1 = np.mean(np.abs(res["rho"] - rho_ex))
    print(
        f"{name:15}: L1(rho) = {l1:.4e}, max |mass error| = {np.max(np.abs(res['mass_err'])):.2e}"
    )

# %%
# Mass conservation
# ^^^^^^^^^^^^^^^^^
# Relative difference between the total mass and the mass expected from the boundary inflow.
# It should stay at round-off level for every solver.

fig, ax = plt.subplots(figsize=(7, 4))
for name, res in results.items():
    ax.plot(res["t"], np.abs(res["mass_err"]) + 1e-17, label=name)
ax.set_yscale("log")
ax.set_xlabel("$t$")
ax.set_ylabel("relative mass error")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()

plt.show()
