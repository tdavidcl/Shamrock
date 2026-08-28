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

The dusty box keeps a uniform state, so the Epstein rates stay constant in time (the drag work
term leaves the gas internal energy, hence the sound speed, unchanged). The test exploits that
to compare an Epstein run against a constant rate run that must be exactly equivalent:

1. at a reference gas density, with the grain sizes calibrated so that :math:`1/t_s` equals the
   constant rates,
2. at twice that gas density, where the Epstein rates must double on their own while the
   constant rates would not move -- this is what checks that the drag rate really is derived
   from the local gas state.
"""

from math import *

import numpy as np

import shamrock

# ============ shared setup ======================

gamma = 1.4
cs_0 = 1.4
rho_ref = 1.0

# reference thermodynamic state, of sound speed cs_init
press_ref = (cs_0 * rho_ref) / gamma
cs_init = sqrt(gamma * press_ref / rho_ref)

# constant drag rates of the dustybox test B, at the reference gas density
alphas_ref = [100.0, 500.0]

# unit intrinsic grain density, grain sizes picked so that 1 / t_s == alpha at (rho_ref, cs_init)
grain_densities = [1.0, 1.0]
grain_sizes = [
    rho_grain * rho_ref * cs_init / alpha * sqrt(8.0 / (pi * gamma))
    for alpha, rho_grain in zip(alphas_ref, grain_densities)
]

dt = 0.005
nstep = 12


def run_sim(rho_gas, alphas, drag_mode="irk1"):
    """Run the dusty box at a given gas density.

    ``alphas`` is either a list of constant drag rates, or None to use the Epstein regime.
    The pressure is scaled with the density so that the sound speed stays ``cs_init``.
    """

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
    if drag_mode == "irk1":
        cfg.set_drag_mode_irk1(True)
    elif drag_mode == "expo":
        cfg.set_drag_mode_expo(True)
    else:
        raise ValueError(f"unknown drag mode {drag_mode}")
    cfg.set_face_time_interpolation(False)

    if alphas is None:
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

    # constant sound speed across the density variants
    press = cs_init * cs_init * rho_gas / gamma

    model.set_field_value_lambda_f64("rho", lambda rmin, rmax: rho_gas)
    model.set_field_value_lambda_f64(
        "rhoetot", lambda rmin, rmax: press / (gamma - 1.0) + 0.5 * rho_gas
    )
    model.set_field_value_lambda_f64_3("rhovel", lambda rmin, rmax: (rho_gas, 0, 0))

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


def compare(label, run_cst, run_eps, tol=1e-13):
    _, vg_c, vd1_c, vd2_c = run_cst
    _, vg_e, vd1_e, vd2_e = run_eps

    ok = True
    for name, cst, eps in [("vg", vg_c, vg_e), ("vd1", vd1_c, vd1_e), ("vd2", vd2_c, vd2_e)]:
        err = float(np.max(np.abs(cst - eps)))
        print(f"{label} {name}: max |constant - epstein| = {err}")
        if not (err < tol):
            print(f"  -> constant = {list(cst)}")
            print(f"  -> epstein  = {list(eps)}")
            ok = False
    return ok


# ============ run the configurations ======================

print(f"grain_sizes     = {grain_sizes}")
print(f"grain_densities = {grain_densities}")

# 1. reference density: the calibrated Epstein rates must reproduce the constant rates
run_cst_ref = run_sim(rho_ref, alphas_ref)
run_eps_ref = run_sim(rho_ref, None)

# 2. twice the density at the same sound speed: the Epstein rates must double on their own
run_cst_2x = run_sim(2.0 * rho_ref, [2.0 * a for a in alphas_ref])
run_eps_2x = run_sim(2.0 * rho_ref, None)

# The exponential integrator (drag_mode="expo") would be the natural third case here, but it
# cannot run on a CPU backend: it sizes its shared memory work group as
# local_mem_size / (5 (ndust+1)^2 sizeof(Tscal)), which asks for a local_accessor as large as the
# whole reported local memory and throws std::bad_alloc. That is independent of the drag rates
# and reproduces on the constant rate path as well.

# ============ CI test ======================

test_pass = True
test_pass &= compare("[rho = rho_ref]", run_cst_ref, run_eps_ref)
test_pass &= compare("[rho = 2 rho_ref]", run_cst_2x, run_eps_2x)

# the two density variants must actually differ, otherwise the comparisons above are vacuous
spread = float(np.max(np.abs(run_eps_ref[1] - run_eps_2x[1])))
print(f"gas velocity spread between the two densities = {spread}")
if not (spread > 1e-3):
    print("the two density variants gave the same result, the test is vacuous")
    test_pass = False

# cross check of the formula itself against the standalone physics binding
ts_ref = shamrock.phys.epstein_stopping_time(
    rho_grain=grain_densities[0],
    s_grain=grain_sizes[0],
    rho=rho_ref,
    cs=cs_init,
    gamma=gamma,
)
print(f"1 / t_s = {1.0 / ts_ref} (target {alphas_ref[0]})")
if not (abs(1.0 / ts_ref - alphas_ref[0]) < 1e-10 * alphas_ref[0]):
    print("the stopping time formula does not match the calibrated rate")
    test_pass = False

if not test_pass:
    exit("Test did not pass")

print("dustybox_godunov_epstein: OK")
