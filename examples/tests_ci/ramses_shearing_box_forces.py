"""
Testing the shearing box forces of the RAMSES solver
====================================================

CI test of ``cfg.add_ext_force_shearing_box`` (tidal, Coriolis and radial pressure-gradient
forcing) together with the shearing-periodic boundary (``cfg.set_boundary_shearing_periodic``):

* equilibrium: the shear flow :math:`v_y = -q \\Omega (x - x_c) - \\eta` (gas, and dust without
  pressure forcing) must stay steady to round-off, the internal energy unchanged.
* epicycles: a small uniform velocity perturbation of the shear flow must oscillate at the
  epicyclic frequency :math:`\\kappa^2 = 2 (2 - q) \\Omega^2`, without damping, for the gas and
  for the dust (no drag).
"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

gamma = 5.0 / 3.0
Omega = 1.0
q = 1.5
kappa = np.sqrt(2 * (2 - q)) * Omega

# 16^3 cells in a single patch, box [0, 1]^3
sz = 1 << 1
base = 8
L = 1.0
x_c = 0.5

rho_g = 1.0
rho_d = 0.5
P0 = 1.0


def cell_center(rmin, rmax):
    return (np.array(rmin) + np.array(rmax)) / 2


def run(eta, dvel, dt, nsteps, with_dust, sample_every=None):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(L / (sz * base))
    cfg.set_eos_gamma(gamma)
    if with_dust:
        cfg.set_dust_mode_dhll(1)
    cfg.set_boundary_shearing_periodic((1, 0, 0), (0, 1, 0), -q * Omega * L)
    cfg.add_ext_force_shearing_box(Omega_0=Omega, eta=eta, q=q)
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

    def vel_gas(rmin, rmax):
        x = cell_center(rmin, rmax)[0]
        return np.array([dvel[0], -q * Omega * (x - x_c) - eta + dvel[1], 0.0])

    def vel_dust(rmin, rmax):
        x = cell_center(rmin, rmax)[0]
        return np.array([dvel[0], -q * Omega * (x - x_c) + dvel[1], 0.0])

    model.set_field_value_lambda_f64("rho", lambda rmin, rmax: rho_g)
    model.set_field_value_lambda_f64_3(
        "rhovel", lambda rmin, rmax: tuple(rho_g * vel_gas(rmin, rmax))
    )

    def rhoetot(rmin, rmax):
        v = vel_gas(rmin, rmax)
        return 0.5 * rho_g * float(np.dot(v, v)) + P0 / (gamma - 1)

    model.set_field_value_lambda_f64("rhoetot", rhoetot)

    if with_dust:
        model.set_field_value_lambda_f64("rho_dust", lambda rmin, rmax: rho_d, 0)
        model.set_field_value_lambda_f64_3(
            "rhovel_dust", lambda rmin, rmax: tuple(rho_d * vel_dust(rmin, rmax)), 0
        )

    samples = []

    def sample(t):
        data = ctx.collect_data()
        if shamrock.sys.world_rank() == 0:
            samples.append((t, data))

    sample(0.0)
    for i in range(nsteps):
        model.evolve_once_override_time(dt * float(i), dt)
        if sample_every is not None and (i + 1) % sample_every == 0 and i + 1 != nsteps:
            sample(dt * float(i + 1))
    sample(dt * float(nsteps))

    return samples


def cell_positions(data):
    # cell_min / cell_max are per block of 2^3 cells (x fastest within a block)
    block_min = np.array(data["cell_min"], dtype=float)
    block_max = np.array(data["cell_max"], dtype=float)
    lid = np.arange(sz**3)
    local_coord = np.stack([lid % sz, (lid // sz) % sz, lid // (sz * sz)], axis=1)
    cell_size = (block_max - block_min) / sz
    pos = block_min[:, None, :] + (local_coord[None, :, :] + 0.5) * cell_size[:, None, :]
    return pos.reshape(-1, 3) * (L / (sz * base))


def internal_energy(data):
    rho = np.array(data["rho"])
    rhov = np.array(data["rhovel"])
    return np.array(data["rhoetot"]) - 0.5 * np.sum(rhov * rhov, axis=1) / rho


def test_equilibrium():
    for eta in [0.0, 0.05]:
        samples = run(eta, (0.0, 0.0), dt=0.01, nsteps=100, with_dust=True)
        if shamrock.sys.world_rank() != 0:
            continue
        (_, d0), (t1, d1) = samples[0], samples[-1]

        err_rho = np.max(np.abs(np.array(d1["rho"]) - np.array(d0["rho"])))
        err_rhov = np.max(np.abs(np.array(d1["rhovel"]) - np.array(d0["rhovel"])))
        err_eint = np.max(np.abs(internal_energy(d1) - internal_energy(d0)))
        err_rhov_d = np.max(np.abs(np.array(d1["rhovel_dust"]) - np.array(d0["rhovel_dust"])))
        err_rho_d = np.max(np.abs(np.array(d1["rho_dust"]) - np.array(d0["rho_dust"])))

        print(
            f"equilibrium eta = {eta}, t = {t1}: err rho = {err_rho:.3e}, err rhovel = {err_rhov:.3e}, "
            f"err e_int = {err_eint:.3e}, err rho_dust = {err_rho_d:.3e}, "
            f"err rhovel_dust = {err_rhov_d:.3e}"
        )

        tol = 1e-12
        assert err_rho < tol, f"eta = {eta}: gas density changed ({err_rho})"
        assert err_rhov < tol, f"eta = {eta}: gas momentum changed ({err_rhov})"
        assert err_eint < tol, f"eta = {eta}: internal energy changed ({err_eint})"
        assert err_rho_d < tol, f"eta = {eta}: dust density changed ({err_rho_d})"
        assert err_rhov_d < tol, f"eta = {eta}: dust momentum changed ({err_rhov_d})"


def test_epicycles():
    du = 1e-2
    period = 2 * np.pi / kappa
    dt = period / 400
    nsteps = 2 * 400  # two epicycles

    samples = run(0.0, (du, 0.0), dt=dt, nsteps=nsteps, with_dust=True, sample_every=20)
    if shamrock.sys.world_rank() != 0:
        return

    # w = v_y + q Omega x' is the deviation from the shear flow
    ratio = kappa / (2 * Omega)
    max_err = {"gas": 0.0, "dust": 0.0}
    max_spread = 0.0
    for t, data in samples:
        u_exp = du * np.cos(kappa * t)
        w_exp = -ratio * du * np.sin(kappa * t)

        for name, rho_key, rhov_key in [
            ("gas", "rho", "rhovel"),
            ("dust", "rho_dust", "rhovel_dust"),
        ]:
            rho = np.array(data[rho_key])
            v = np.array(data[rhov_key]) / rho[:, None]
            x = cell_positions(data)[:, 0]
            u = v[:, 0]
            w = v[:, 1] + q * Omega * (x - x_c)

            err = max(np.max(np.abs(u - u_exp)), np.max(np.abs(w - w_exp))) / du
            max_err[name] = max(max_err[name], err)
            max_spread = max(max_spread, (np.max(u) - np.min(u)) / du)

    print(
        f"epicycles over 2 periods (dt = T/400): max rel err gas = {max_err['gas']:.3e}, "
        f"dust = {max_err['dust']:.3e}, max spread of v_x = {max_spread:.3e}"
    )

    assert max_err["gas"] < 1e-2, f"gas epicycles do not match kappa ({max_err['gas']})"
    assert max_err["dust"] < 1e-2, f"dust epicycles do not match kappa ({max_err['dust']})"


test_equilibrium()
test_epicycles()

if shamrock.sys.world_rank() == 0:
    print("ramses_shearing_box_forces: OK")
