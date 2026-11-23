"""
Streaming instability in RAMSES solver
======================================

Unstratified streaming instability in a shearing box: shearing-periodic boundary
(``cfg.set_boundary_shearing_periodic``), shearing box forces with a radial pressure-gradient
forcing (``cfg.add_ext_force_shearing_box``), one dust species with drag.

Units: Omega = 1, c_s = 1 (H = 1), Pi = eta v_K / c_s (``SI_PI``, default 0.05), so that
eta r = Pi H. The gas and dust start at the Nakagawa-Sekiya-Hayashi (NSH) equilibrium. Setups
(``SI_SETUP``):

* ``AB`` (default): tau_s = 0.1, eps = rho_d / rho_g = 1 (Johansen & Youdin 2007, Bai & Stone
  2010), 2% white noise on the dust density. Box of ``SI_L`` H along x (default 0.2), on
  ``SI_NX`` x ``SI_NY`` x ``SI_NZ`` cells (default 128 x 64 x 128).
* ``linA``: tau_s = 0.1, eps = 3, K_x = K_z = k eta r = 30, growth rate 0.4190 Omega.
* ``linB``: tau_s = 0.1, eps = 0.2, K_x = K_z = 6, growth rate 0.0154 Omega.

For linA and linB, the eigenmode is computed from the linearized equations (compressible gas,
same parameters as the run) and seeded with an amplitude ``SI_AMP`` (dust density, relative,
default 1e-4). The box is one wavelength along x and z by default, ``SI_L`` is rounded to a whole
number of wavelengths; the grid defaults to 32^3 cells (single patch). The amplitude of the mode
is measured from the dust density and its growth rate compared to the linear one.

The cells are cubic (dx = L / SI_NX), the box is SI_L x (SI_L * SI_NY / SI_NX) x
(SI_L * SI_NZ / SI_NX). The patches are cubes of gcd(block counts) blocks of 2^3 cells, so keep
the cell counts with a large common power of two (a single patch if they are equal).

The run stops at ``SI_TEND`` (simulation time) or ``SI_WALL`` (wall time, in seconds). Every
``SI_DTFRAME`` it writes an x-z slice of the dust density and statistics in ``SI_OUTDIR``, and
updates the GIF of the slices with the growth of the dust density (``streaming_instability.gif``),
so that it can be followed during the run. The time step is set by the sound speed,
dt ~ 0.24 dx / c_s: about 20 min of wall time per 1/Omega for 64^3 cells of 0.1 H / 64 on 4 CPU
cores.
"""

import json
import os
import time

import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Parameters

setup = os.environ.get("SI_SETUP", "AB")
setups = {
    # eps, tau_s, K (k eta r) of the seeded mode, growth rate from the literature
    "AB": {"eps": 1.0, "tau_s": 0.1, "K": None, "s_ref": None},
    "linA": {"eps": 3.0, "tau_s": 0.1, "K": 30.0, "s_ref": 0.4190},
    "linB": {"eps": 0.2, "tau_s": 0.1, "K": 6.0, "s_ref": 0.0154},
}
if setup not in setups:
    raise ValueError(f"SI_SETUP must be one of {list(setups)}, got {setup}")
is_linear = setups[setup]["K"] is not None

outdir = os.environ.get("SI_OUTDIR", "si_out")
t_end = float(os.environ.get("SI_TEND", "60"))
wall_max = float(os.environ.get("SI_WALL", "6000"))  # seconds
dt_frame = float(os.environ.get("SI_DTFRAME", "0.5"))
os.makedirs(outdir, exist_ok=True)

Omega = 1.0
q = 1.5
cs = 1.0
H = cs / Omega
gamma = 5.0 / 3.0
Pi = float(os.environ.get("SI_PI", "0.05"))
eta_vK = Pi * cs  # headwind velocity eta v_K
eta_r = eta_vK / Omega
tau_s = setups[setup]["tau_s"]
t_stop = tau_s / Omega
eps = setups[setup]["eps"]

rho_g0 = 1.0
rho_d0 = eps * rho_g0

# cells per axis (multiples of 2, blocks of 2^3 cells)
default_n = ("32", "32", "32") if is_linear else ("128", "64", "128")
Nx = int(os.environ.get("SI_NX", default_n[0]))
Ny = int(os.environ.get("SI_NY", default_n[1]))
Nz = int(os.environ.get("SI_NZ", default_n[2]))

if is_linear:
    # box: a whole number of wavelengths of the mode along x and z
    K = setups[setup]["K"]
    wavelength = 2 * np.pi * eta_r / K
    n_wave = max(1, round(float(os.environ.get("SI_L", wavelength)) / wavelength))
    if Nz != Nx:
        raise ValueError("the linear setups need SI_NZ == SI_NX (same box size along x and z)")
    Lx = n_wave * wavelength
else:
    Lx = float(os.environ.get("SI_L", "0.2")) * H

dxg = Lx / Nx  # cell size
Ly, Lz = Ny * dxg, Nz * dxg
x_c = Lx / 2

# Nakagawa-Sekiya-Hayashi equilibrium (velocities relative to the Keplerian shear)
Delta = (1 + eps) ** 2 + tau_s**2
ug = 2 * eps * tau_s * eta_vK / Delta
vg = -eta_vK * (1 + eps + tau_s**2) / Delta
ud = -2 * tau_s * eta_vK / Delta
vd = -eta_vK * (1 + eps) / Delta


# %%
# Linear eigenmode
#
# Perturbations proportional to exp(i (kx x + kz z) + s t) of
# (rho_g, u_x, u_y, u_z, rho_d, w_x, w_y, w_z) around the NSH equilibrium, with the
# y velocities relative to the shear flow -q Omega x' (which gives the -(2 - q) Omega
# epicyclic term), isothermal response of the gas pressure (c_s is the adiabatic sound speed,
# no entropy perturbation) and the drag of a constant stopping time.


def linear_mode(kx, kz):
    M = np.zeros((8, 8), dtype=complex)
    ag = -1j * kx * ug  # advection by the background
    ad = -1j * kx * ud
    sx = (ud - ug) / t_stop  # background slip
    sy = (vd - vg) / t_stop

    # gas
    M[0, 0] = ag
    M[0, 1] = -1j * kx * rho_g0
    M[0, 3] = -1j * kz * rho_g0
    M[1, 0] = -1j * kx * cs**2 / rho_g0 - sx * eps / rho_g0
    M[1, 1] = ag - eps / t_stop
    M[1, 2] = 2 * Omega
    M[1, 4] = sx / rho_g0
    M[1, 5] = eps / t_stop
    M[2, 0] = -sy * eps / rho_g0
    M[2, 1] = -(2 - q) * Omega
    M[2, 2] = ag - eps / t_stop
    M[2, 4] = sy / rho_g0
    M[2, 6] = eps / t_stop
    M[3, 0] = -1j * kz * cs**2 / rho_g0
    M[3, 3] = ag - eps / t_stop
    M[3, 7] = eps / t_stop

    # dust
    M[4, 4] = ad
    M[4, 5] = -1j * kx * rho_d0
    M[4, 7] = -1j * kz * rho_d0
    M[5, 1] = 1 / t_stop
    M[5, 5] = ad - 1 / t_stop
    M[5, 6] = 2 * Omega
    M[6, 2] = 1 / t_stop
    M[6, 5] = -(2 - q) * Omega
    M[6, 6] = ad - 1 / t_stop
    M[7, 3] = 1 / t_stop
    M[7, 7] = ad - 1 / t_stop

    eigvals, eigvecs = np.linalg.eig(M)
    i = int(np.argmax(eigvals.real))
    # normalized to a relative dust density perturbation of 1
    return eigvals[i], eigvecs[:, i] / eigvecs[4, i] * rho_d0


if is_linear:
    kx = kz = K / eta_r
    s_lin, mode = linear_mode(kx, kz)
    amp = float(os.environ.get("SI_AMP", "1e-4"))
    if shamrock.sys.world_rank() == 0:
        print(
            f"SI {setup}: K = {K}, box = {n_wave} wavelength(s) = {Lx:.5g} H, "
            f"linear growth rate s = {s_lin.real:.5f} Omega "
            f"(literature {setups[setup]['s_ref']}), oscillation {s_lin.imag:.5f} Omega",
            flush=True,
        )
        print(
            "SI eigenvector (u, w in units of eta v_K): u =",
            np.round(mode[1:4] / eta_vK, 4),
            "w =",
            np.round(mode[5:8] / eta_vK, 4),
            "drho_g / rho_g =",
            mode[0] / rho_g0,
            flush=True,
        )

# %%
# Setup

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
cfg.set_scale_factor(dxg)
cfg.set_eos_gamma(gamma)
cfg.set_slope_lim_vanleer_sym()
cfg.set_face_time_interpolation(True)
cfg.set_Csafe(0.8)
cfg.set_dust_mode_hb(1)
cfg.set_drag_mode_irk1(True)
cfg.set_alpha_values(1.0 / t_stop)
cfg.set_boundary_shearing_periodic((1, 0, 0), (0, 1, 0), -q * Omega * Lx)
cfg.add_ext_force_shearing_box(Omega_0=Omega, eta=eta_vK, q=q)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (2, 2, 2), (Nx // 2, Ny // 2, Nz // 2))


# the lambdas get the cell bounds in physical units
def center(rmin, rmax):
    return (np.array(rmin) + np.array(rmax)) * 0.5


if is_linear:

    def perturbation(rmin, rmax):
        x, y, z = center(rmin, rmax)
        return (amp * mode * np.exp(1j * (kx * x + kz * z))).real

else:
    # deterministic white noise on the dust density, indexed by the integer cell coordinates
    rng = np.random.default_rng(1234)
    noise = rng.uniform(-1.0, 1.0, size=(Nx, Ny, Nz))
    noise_amp = 0.02

    def perturbation(rmin, rmax):
        i, j, k = (round(c / dxg) for c in rmin)
        dq = np.zeros(8)
        dq[4] = rho_d0 * noise_amp * noise[i, j, k]
        return dq


def gas_state(rmin, rmax):
    x = center(rmin, rmax)[0]
    dq = perturbation(rmin, rmax)
    rho = rho_g0 + dq[0]
    vel = np.array([ug, vg - q * Omega * (x - x_c), 0.0]) + dq[1:4]
    return rho, vel


def dust_state(rmin, rmax):
    x = center(rmin, rmax)[0]
    dq = perturbation(rmin, rmax)
    rho = rho_d0 + dq[4]
    vel = np.array([ud, vd - q * Omega * (x - x_c), 0.0]) + dq[5:8]
    return rho, vel


def rhovel_gas(rmin, rmax):
    rho, vel = gas_state(rmin, rmax)
    return tuple(rho * vel)


def rhoetot(rmin, rmax):
    # isothermal-like perturbation: P = c_s^2 rho / gamma
    rho, vel = gas_state(rmin, rmax)
    return 0.5 * rho * float(np.dot(vel, vel)) + rho * cs**2 / gamma / (gamma - 1)


def rhovel_dust(rmin, rmax):
    rho, vel = dust_state(rmin, rmax)
    return tuple(rho * vel)


model.set_field_value_lambda_f64("rho", lambda rmin, rmax: gas_state(rmin, rmax)[0])
model.set_field_value_lambda_f64_3("rhovel", rhovel_gas)
model.set_field_value_lambda_f64("rhoetot", rhoetot)
model.set_field_value_lambda_f64("rho_dust", lambda rmin, rmax: dust_state(rmin, rmax)[0], 0)
model.set_field_value_lambda_f64_3("rhovel_dust", rhovel_dust, 0)

# x-z slice at the mid plane in y, sampled at the cell centers
xs = (np.arange(Nx) + 0.5) * dxg
zs = (np.arange(Nz) + 0.5) * dxg
X, Z = np.meshgrid(xs, zs, indexing="xy")
positions = [(float(x), Ly / 2 + 0.5 * dxg, float(z)) for x, z in zip(X.ravel(), Z.ravel())]


def cell_positions(data):
    # cell_min / cell_max are per block of 2^3 cells (x fastest within a block), in grid units
    block_min = np.array(data["cell_min"], dtype=float)
    block_max = np.array(data["cell_max"], dtype=float)
    lid = np.arange(8)
    local_coord = np.stack([lid % 2, (lid // 2) % 2, lid // 4], axis=1)
    cell_size = (block_max - block_min) / 2
    pos = block_min[:, None, :] + (local_coord[None, :, :] + 0.5) * cell_size[:, None, :]
    return pos.reshape(-1, 3) * dxg


# %%
# GIF of the dust density slices and growth of the dust density


def make_gif(outdir, gif):
    import glob

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    files = sorted(glob.glob(os.path.join(outdir, "slice_*.npy")))
    with open(os.path.join(outdir, "stats.jsonl")) as f:
        stats = [json.loads(line) for line in f if line.strip()]
    n = min(len(files), len(stats))
    stats = stats[:n]
    slices = [np.load(f) for f in files[:n]]

    t = np.array([s["t"] for s in stats])
    rms = np.array([s["rho_d_rms"] for s in stats])
    dmax = np.array([s["rho_d_max"] for s in stats])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    if is_linear:
        # relative perturbation, color scale of each frame
        def frame_data(i):
            return slices[i] / rho_d0 - 1

        im = ax.imshow(frame_data(0), origin="lower", extent=(0, Lx, 0, Lz), cmap="RdBu_r")
        cb_label = r"$\delta \rho_d / \rho_{d,0}$ (scale of each frame)"
    else:

        def frame_data(i):
            return slices[i]

        vmin = min(s.min() for s in slices)
        vmax = max(s.max() for s in slices)
        im = ax.imshow(
            frame_data(0),
            origin="lower",
            extent=(0, Lx, 0, Lz),
            cmap="magma",
            norm=matplotlib.colors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax),
        )
        cb_label = r"$\rho_d / \rho_{g,0}$"
    fig.colorbar(im, ax=ax, label=cb_label, fraction=0.046)
    ax.set_xlabel("x / H")
    ax.set_ylabel("z / H")
    title = ax.set_title("")

    ax2.semilogy(t, rms / rho_d0, color="C0", label=r"rms($\rho_d$) / $\rho_{d,0}$")
    ax2.semilogy(
        t, np.abs(dmax / rho_d0 - 1), color="C1", label=r"max($\rho_d$) / $\rho_{d,0}$ - 1"
    )
    if is_linear:
        mode_amp = np.array([s["mode_amp"] for s in stats])
        ax2.semilogy(t, mode_amp, "C2o", ms=3, label="mode amplitude")
        ax2.semilogy(
            t,
            mode_amp[0] * np.exp(s_lin.real * t),
            "k--",
            lw=0.8,
            label=rf"linear, s = {s_lin.real:.4f} $\Omega$",
        )
    marker = ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel(r"$\Omega t$")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_title(f"{setup}: dust density growth")
    fig.tight_layout()

    def update(i):
        d = frame_data(i)
        im.set_data(d)
        if is_linear:
            lim = max(float(np.max(np.abs(d))), 1e-30)
            im.set_clim(-lim, lim)
        title.set_text(rf"$\rho_d$, x-z slice, $\Omega t$ = {t[i]:.2f}")
        marker.set_xdata([t[i], t[i]])
        return im, title, marker

    # written to temporary files then renamed, so that the GIF can be opened during the run
    anim = animation.FuncAnimation(fig, update, frames=len(slices), blit=False)
    tmp_gif = gif + ".tmp.gif"
    anim.save(tmp_gif, writer=animation.PillowWriter(fps=8), dpi=90)
    os.replace(tmp_gif, gif)
    update(len(slices) - 1)
    tmp_png = gif.replace(".gif", "_last.tmp.png")
    fig.savefig(tmp_png, dpi=110)
    os.replace(tmp_png, gif.replace(".gif", "_last.png"))
    plt.close(fig)
    print("wrote", gif, len(slices), "frames", flush=True)


gif_path = os.path.join(outdir, "streaming_instability.gif")


def dump_frame(iframe, t):
    sl = np.array(model.render_slice("rho_dust", "f64", positions)).reshape(Nz, Nx)
    data = ctx.collect_data()
    if shamrock.sys.world_rank() == 0:
        rd = np.array(data["rho_dust"])
        rg = np.array(data["rho"])
        stats = {
            "t": t,
            "rho_d_max": float(rd.max()),
            "rho_d_rms": float(np.sqrt(np.mean((rd - rd.mean()) ** 2))),
            "rho_d_mean": float(rd.mean()),
            "rho_g_min": float(rg.min()),
            "rho_g_max": float(rg.max()),
            "dust_mass": float(rd.sum()),
            "gas_mass": float(rg.sum()),
        }
        if is_linear:
            # amplitude of the seeded Fourier mode of the relative dust density
            pos = cell_positions(data)
            phase = np.exp(-1j * (kx * pos[:, 0] + kz * pos[:, 2]))
            stats["mode_amp"] = float(2 * np.abs(np.mean((rd / rho_d0 - 1) * phase)))
        np.save(os.path.join(outdir, f"slice_{iframe:04d}.npy"), sl)
        with open(os.path.join(outdir, "stats.jsonl"), "a") as f:
            f.write(json.dumps(stats) + "\n")
        print("SI frame", iframe, json.dumps(stats), flush=True)
        make_gif(outdir, gif_path)


# %%
# Run

open(os.path.join(outdir, "stats.jsonl"), "w").close()

wall0 = time.time()
t = 0.0
iframe = 0
dump_frame(iframe, t)
while t < t_end and time.time() - wall0 < wall_max:
    t_next = min(t_end, (iframe + 1) * dt_frame)
    model.evolve_until(t_next)
    t = model.get_time()
    iframe += 1
    dump_frame(iframe, t)
    if shamrock.sys.world_rank() == 0:
        print(
            f"SI t = {t:.3f}, dt = {model.get_dt():.3e}, wall = {time.time() - wall0:.0f} s",
            flush=True,
        )

if is_linear and shamrock.sys.world_rank() == 0:
    with open(os.path.join(outdir, "stats.jsonl")) as f:
        stats = [json.loads(line) for line in f if line.strip()]
    ts_ = np.array([s["t"] for s in stats])
    amps = np.array([s["mode_amp"] for s in stats])
    if len(ts_) >= 3:
        # growth rate fitted on the second half of the run (after the transients)
        half = len(ts_) // 2
        s_fit = np.polyfit(ts_[half:], np.log(amps[half:]), 1)[0]
        print(
            f"SI {setup}: measured growth rate {s_fit:.5f} Omega, linear {s_lin.real:.5f} Omega "
            f"(ratio {s_fit / s_lin.real:.3f})",
            flush=True,
        )

print("SI run done")
