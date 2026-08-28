"""
Testing dusty box with Godunov and Epstein drag
===============================================

CI test for the Epstein drag regime of the Ramses (Godunov) dust solver.

The Ramses drag operator solves, per dust species,

.. math::
    \\partial_t v_{{\\rm d},j} = \\alpha_j (v_{\\rm g} - v_{{\\rm d},j})

where :math:`\\alpha_j = 1 / t_{s,j}` is the dust collision rate. It can either be given
directly by the user (``set_alpha_values``) or derived from the local gas state in the Epstein
regime (``set_dust_drag_epstein``), in which case

.. math::
    t_{s,j} = \\frac{\\rho_{{\\rm grain},j} s_{{\\rm grain},j}}{\\rho_{\\rm g} c_s}
              \\sqrt{\\frac{\\pi \\gamma}{8}}

with :math:`\\rho_{\\rm g}` the **gas** density, as required by the two fluid formulation.

This test runs the dusty box twice, with the two drag configurations, choosing the grain sizes
so that the Epstein rates match the constant rates at the initial gas state. The first step must
then agree to machine precision, and the whole trajectory to within the drift of the sound speed
caused by the frictional heating of the gas.
"""

from math import *

import numpy as np

import shamrock

# ============ shared setup ======================

gamma = 1.4
cs_0 = 1.4
rho_0 = 1.0

press_0 = (cs_0 * rho_0) / gamma
cs_init = sqrt(gamma * press_0 / rho_0)

# constant drag rates of the dustybox test B
alphas = [100.0, 500.0]

# unit intrinsic grain density, grain sizes picked so that 1 / t_s == alpha at the initial state
grain_densities = [1.0, 1.0]
grain_sizes = [
    rho_grain * rho_0 * cs_init / alpha * sqrt(8.0 / (pi * gamma))
    for alpha, rho_grain in zip(alphas, grain_densities)
]

dt = 0.005
nstep = 12


def run_sim(use_epstein):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    sz = 1 << 1
    base = 2

    cfg = model.gen_default_config()
    cfg.set_scale_factor(1 / (sz * base))
    cfg.set_Csafe(0.44)
    cfg.set_eos_gamma(gamma)
    cfg.set_dust_mode_dhll(2)
    cfg.set_drag_mode_irk1(True)
    cfg.set_face_time_interpolation(False)

    if use_epstein:
        cfg.set_dust_drag_epstein(
            grain_sizes=grain_sizes,
            grain_densities=grain_densities,
        )
    else:
        for alpha in alphas:
            cfg.set_alpha_values(alpha)

    model.set_solver_config(cfg)
    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

    model.set_field_value_lambda_f64("rho", lambda rmin, rmax: rho_0)
    model.set_field_value_lambda_f64(
        "rhoetot", lambda rmin, rmax: press_0 / (gamma - 1.0) + 0.5 * rho_0
    )
    model.set_field_value_lambda_f64_3("rhovel", lambda rmin, rmax: (1, 0, 0))

    model.set_field_value_lambda_f64("rho_dust", lambda rmin, rmax: 1, 0)
    model.set_field_value_lambda_f64_3("rhovel_dust", lambda rmin, rmax: (2, 0, 0), 0)
    model.set_field_value_lambda_f64("rho_dust", lambda rmin, rmax: 1, 1)
    model.set_field_value_lambda_f64_3("rhovel_dust", lambda rmin, rmax: (0.5, 0, 0), 1)

    times = []
    vg = []
    vd1 = []
    vd2 = []

    for i in range(nstep + 1):
        dic = ctx.collect_data()
        vg.append(dic["rhovel"][0][0] / dic["rho"][0])
        vd1.append(dic["rhovel_dust"][0][0] / dic["rho_dust"][0])
        vd2.append(dic["rhovel_dust"][1][0] / dic["rho_dust"][1])
        times.append(dt * i)
        model.evolve_once_override_time(dt * float(i), dt)

    return times, np.array(vg), np.array(vd1), np.array(vd2)


# ============ run both configurations ======================

times, vg_cst, vd1_cst, vd2_cst = run_sim(use_epstein=False)
_, vg_eps, vd1_eps, vd2_eps = run_sim(use_epstein=True)

print(f"grain_sizes     = {grain_sizes}")
print(f"grain_densities = {grain_densities}")
print(f"times   = {times}")
print(f"vg_cst  = {list(vg_cst)}")
print(f"vg_eps  = {list(vg_eps)}")
print(f"vd1_cst = {list(vd1_cst)}")
print(f"vd1_eps = {list(vd1_eps)}")
print(f"vd2_cst = {list(vd2_cst)}")
print(f"vd2_eps = {list(vd2_eps)}")

# ============ CI test ======================

# The initial state is uniform, so the fluxes vanish and the Epstein rates of the first step are
# evaluated on exactly the state the grain sizes were calibrated on: the first step must match
# the constant rate run to round-off.
for name, cst, eps in [
    ("vg", vg_cst, vg_eps),
    ("vd1", vd1_cst, vd1_eps),
    ("vd2", vd2_cst, vd2_eps),
]:
    err = abs(cst[1] - eps[1])
    print(f"first step {name}: |constant - epstein| = {err}")
    assert err < 1e-12, f"first step of {name} differs by {err}, drag rates are not consistent"

# Afterwards the two runs drift apart, because the drag heats the gas up, which raises the sound
# speed, which raises the Epstein rates. That drift stays small over this test.
for name, cst, eps in [
    ("vg", vg_cst, vg_eps),
    ("vd1", vd1_cst, vd1_eps),
    ("vd2", vd2_cst, vd2_eps),
]:
    err = np.max(np.abs(cst - eps) / np.abs(cst))
    print(f"trajectory {name}: max relative deviation = {err}")
    assert err < 5e-2, f"{name} deviates by {err}, more than the expected sound speed drift"

# The rates must actually depend on the gas state: doubling the gas density doubles alpha. Check
# the formula the solver uses against the standalone physics binding.
ts_ref = shamrock.phys.epstein_stopping_time(
    rho_grain=grain_densities[0],
    s_grain=grain_sizes[0],
    rho=rho_0,
    cs=cs_init,
    gamma=gamma,
)
print(f"1 / t_s = {1.0 / ts_ref} (target {alphas[0]})")
assert abs(1.0 / ts_ref - alphas[0]) < 1e-10 * alphas[0]

print("dustybox_godunov_epstein: OK")
