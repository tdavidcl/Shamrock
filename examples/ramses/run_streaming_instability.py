"""
Streaming instability in RAMSES solver
======================================

Unstratified streaming instability (Johansen & Youdin 2007 / Bai & Stone 2010 "AB"-like run):
shearing-periodic box (``cfg.set_boundary_shearing_periodic``), shearing box forces with a radial
pressure-gradient forcing (``cfg.add_ext_force_shearing_box``), one dust species with drag.

Units: Omega = 1, c_s = 1 (H = 1). tau_s = Omega t_stop = 0.1, eps = rho_d / rho_g = 1,
Pi = eta v_K / c_s = 0.05. Cubic cells of size dx = 0.1 H / 64 (= 2 eta r / 64), on a grid of
``SI_NX`` x ``SI_NY`` x ``SI_NZ`` cells (default 128 x 128 x 64, i.e. a 0.2 H x 0.2 H x 0.1 H
box). The gas and dust start at the Nakagawa-Sekiya-Hayashi equilibrium, with a 2% white noise on
the dust density.

The run is long (about 20 min of wall time per 1/Omega for 64^3 cells on 4 CPU cores, roughly
four times more at the default resolution), it stops at ``SI_TEND`` (simulation time) or
``SI_WALL`` (wall time, in seconds). Every ``SI_DTFRAME`` it writes an x-z slice of the dust
density and statistics in ``SI_OUTDIR``, and updates the GIF of the slices with the growth of the
dust density (``streaming_instability.gif``), so that it can be followed during the run.
"""

import json
import os
import time

import numpy as np

import shamrock

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

outdir = os.environ.get("SI_OUTDIR", "si_out")
t_end = float(os.environ.get("SI_TEND", "60"))
wall_max = float(os.environ.get("SI_WALL", "6000"))  # seconds
dt_frame = float(os.environ.get("SI_DTFRAME", "0.5"))
os.makedirs(outdir, exist_ok=True)

Omega = 1.0
q = 1.5
cs = 1.0
gamma = 5.0 / 3.0
Pi = 0.05
eta_vK = Pi * cs  # headwind velocity eta v_K
tau_s = 0.1
t_stop = tau_s / Omega
eps = 1.0

rho_g0 = 1.0
rho_d0 = eps * rho_g0
P0 = rho_g0 * cs**2 / gamma

# cells per axis (multiples of 2, blocks of 2^3 cells)
Nx = int(os.environ.get("SI_NX", "128"))
Ny = int(os.environ.get("SI_NY", "128"))
Nz = int(os.environ.get("SI_NZ", "64"))
dxg = 0.1 / 64  # cell size
Lx, Ly, Lz = Nx * dxg, Ny * dxg, Nz * dxg
x_c = Lx / 2

# Nakagawa-Sekiya-Hayashi equilibrium (velocities relative to the Keplerian shear)
Delta = (1 + eps) ** 2 + tau_s**2
ug = 2 * eps * tau_s * eta_vK / Delta
vg = -eta_vK * (1 + eps + tau_s**2) / Delta
ud = -2 * tau_s * eta_vK / Delta
vd = -eta_vK * (1 + eps) / Delta

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
# blocks of 2^3 cells, the patches are cubes of gcd(block counts) blocks (2 x 2 x 1 patches
# for 128 x 128 x 64 cells)
model.make_base_grid((0, 0, 0), (2, 2, 2), (Nx // 2, Ny // 2, Nz // 2))


# the lambdas get the cell bounds in physical units
def center(rmin, rmax):
    return (np.array(rmin) + np.array(rmax)) * 0.5


# deterministic white noise on the dust density, indexed by the integer cell coordinates
rng = np.random.default_rng(1234)
noise = rng.uniform(-1.0, 1.0, size=(Nx, Ny, Nz))
noise_amp = 0.02


def cell_index(rmin):
    return tuple(round(c / dxg) for c in rmin)


def rho_dust(rmin, rmax):
    i, j, k = cell_index(rmin)
    return rho_d0 * (1 + noise_amp * noise[i, j, k])


def vel_gas(rmin, rmax):
    x = center(rmin, rmax)[0]
    return np.array([ug, vg - q * Omega * (x - x_c), 0.0])


def vel_dust(rmin, rmax):
    x = center(rmin, rmax)[0]
    return np.array([ud, vd - q * Omega * (x - x_c), 0.0])


model.set_field_value_lambda_f64("rho", lambda rmin, rmax: rho_g0)
model.set_field_value_lambda_f64_3("rhovel", lambda rmin, rmax: tuple(rho_g0 * vel_gas(rmin, rmax)))


def rhoetot(rmin, rmax):
    v = vel_gas(rmin, rmax)
    return 0.5 * rho_g0 * float(np.dot(v, v)) + P0 / (gamma - 1)


model.set_field_value_lambda_f64("rhoetot", rhoetot)
model.set_field_value_lambda_f64("rho_dust", rho_dust, 0)
model.set_field_value_lambda_f64_3(
    "rhovel_dust",
    lambda rmin, rmax: tuple(rho_dust(rmin, rmax) * vel_dust(rmin, rmax)),
    0,
)

# x-z slice at the mid plane in y, sampled at the cell centers
xs = (np.arange(Nx) + 0.5) * dxg
zs = (np.arange(Nz) + 0.5) * dxg
X, Z = np.meshgrid(xs, zs, indexing="xy")
positions = [(float(x), Ly / 2 + 0.5 * dxg, float(z)) for x, z in zip(X.ravel(), Z.ravel())]


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
    stats = stats[: len(files)]
    slices = [np.load(f) for f in files]

    vmin = min(s.min() for s in slices)
    vmax = max(s.max() for s in slices)

    t = np.array([s["t"] for s in stats])
    rms = np.array([s["rho_d_rms"] for s in stats])
    dmax = np.array([s["rho_d_max"] for s in stats])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    im = ax.imshow(
        slices[0],
        origin="lower",
        extent=(0, Lx, 0, Lz),
        cmap="magma",
        norm=matplotlib.colors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax),
    )
    fig.colorbar(im, ax=ax, label=r"$\rho_d / \rho_{g,0}$", fraction=0.046)
    ax.set_xlabel("x / H")
    ax.set_ylabel("z / H")
    title = ax.set_title("")

    ax2.semilogy(t, rms, color="C0", label=r"rms($\rho_d$)")
    ax2.semilogy(t, dmax - 1, color="C1", label=r"max($\rho_d$) - 1")
    marker = ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel(r"$\Omega t$")
    ax2.legend(loc="lower right")
    ax2.set_title("dust density growth")
    fig.tight_layout()

    def update(i):
        im.set_data(slices[i])
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
        np.save(os.path.join(outdir, f"slice_{iframe:04d}.npy"), sl)
        with open(os.path.join(outdir, "stats.jsonl"), "a") as f:
            f.write(json.dumps(stats) + "\n")
        print("SI frame", iframe, json.dumps(stats), flush=True)
        make_gif(outdir, gif_path)


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


print("SI run done")
