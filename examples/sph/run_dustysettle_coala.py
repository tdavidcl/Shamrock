"""
Dusty settling SPH test with COALA coagulation
===============================================

Perform a dust settling test in a local stratified box, with the dust size
distribution evolved by the COALA coagulation source term.

If you want to change the resolution, you can set the LZ environment variable.

Example (from build directory):
```bash
LZ=96 ./shamrock --sycl-cfg 0:0 --smi --loglevel 1 --rscript ../examples/sph/run_dustysettle_coala.py
```
"""

# sphinx_gallery_multi_image = "single"

# %%
# Imports
# ------------------------------------------

import json
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import PillowWriter
from matplotlib.lines import Line2D
from scipy.special import erfinv
from shamrock.external import coala
from shamrock.utils.DustMRNDistribution import DustMRNDistribution
from shamrock.utils.plot import show_image_sequence

import shamrock
from shamrock import NeighCacheStrategy

# %%
# Shamrock initialization
# ------------------------------------------

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
# ------------------------------------------

shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Define simulation parameters and unit system
# ------------------------------------------

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

# Disc / box
rho_i = 1e-6  # Initial density
central_mass = 1  # Central mass
R0 = 1  # Where are we in radius
H_r_0 = 0.05  # Disc aspect ratio
gamma = 1.4  # Adiabatic index
box_H_count = 8  # box size in number of H
random_vel_pert = True  # add a small vecolity pertubation to fasten the setup

# Dust
ndust = 20
mrn_pow = 3.5
mrn_cutoff_si = 250e-9  # like standard MRN
epsilon_base = 0.01
rho_grains_si_edges = np.array([2.3 * 1000 for i in range(ndust + 1)])
grain_size_si_edges = np.logspace(-9, -2, ndust + 1)

# COALA coagulation
use_coala = True
dv_max = 1000000 * codeu.get("m") / codeu.get("s")
Q = 5  # number of points for the Gauss-Legendre quadrature
rhodust_eps = 1e-17  # dust density floor
tau_frac = 10.0  # the source term step moves at least tau_frac * rhodust_eps
K0_multiplier = 1

# Resolution (lattice size)
lx = 12
ly = 12
lz = int(os.environ.get("LZ", "64"))

# Time
tinject = 2.0  # time at which the dust is injected
t_end = tinject + 3.0
dt_analysis = 0.1  # time between two analysis plots

# Scheduler
scheduler_split_val = int(2e7)
scheduler_merge_val = 1

# Artificial viscosity
av_alpha_min = 0.0
av_alpha_max = 1
av_sigma_decay = 0.1
av_alpha_u = 1
av_beta_AV = 2
vel_dissipation_eta = 5

# CFL (before dust injection)
cfl_cour_setup = 0.25
cfl_force_setup = 0.25
# CFL (after dust injection)
cfl_cour_inject = 0.1
cfl_force_inject = 0.1

# Injection check
max_v_inject_threshold = 1.0

# Paths
sim_folder = f"_to_trash/dusty_settle_{lz}_coala_{use_coala}/"
dump_folder = sim_folder + "dump/"

# Plotting
cmap = "plasma"
dpi = 250

# %%
# Derived physical scales and lattice geometry
# ------------------------------------------

print("codeu.get('m') / codeu.get('s') =", codeu.get("m") / codeu.get("s"))
print("codeu.to('m') / codeu.to('s') =", codeu.to("m") / codeu.to("s"))

G = ucte.G()


def kep_profile(r):
    return (G * central_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


H = H_r_0 * R0
cs = H * omega_k(R0)
box = box_H_count * H

print(f"cs = {cs}")
print(f"H = {H}")

mrn_distribution = DustMRNDistribution(
    codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges
)

massgrid_edges = mrn_distribution.massgrid_edges

if use_coala:
    print(f"massgrid = {mrn_distribution.massgrid} [code units]")
    print(f"massgrid = {mrn_distribution.massgrid * codeu.to('kg')} [kg]")

    K0 = np.pi * ((4.0 / 3.0) * np.pi * mrn_distribution.rho_grains[0]) ** (-2.0 / 3.0)
    K0 *= K0_multiplier
    print(f"K0 = {K0}")

    tabflux_coag = coala.coala_precalc_tabflux_coag(K0, ndust, Q, massgrid_edges)

lmin = (-(lx // 2), -(ly // 2), -(lz // 2))
lmax = (lx // 2, ly // 2, lz // 2)

# Call with dr = 1 as we will rescale on next call
(xm, ym, zm), (xM, yM, zM) = shamrock.math.get_periodic_hcp_box(1.0, lmin, lmax)
print(f"base lattice : xM = {xM}, yM = {yM}, zM = {zM}")
dr = 2 * box / (zM - zm)
print(f"dr = {dr}")
bmin, bmax = shamrock.math.get_periodic_hcp_box(dr, lmin, lmax)
print(f"new lattice : bmin = {bmin}, bmax = {bmax}")
xm, ym, zm = bmin
xM, yM, zM = bmax

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho_i * vol_b
print("Total mass :", totmass)

pmass = -1

# %%
# Initial conditions and dust injection functions
# ------------------------------------------

cs_g = cs


def scaling_rho(r):
    x, y, z = r

    loc_h = H / (2**0.5)
    gaussian = np.exp(-(y**2) / (2 * loc_h * loc_h)) / (loc_h * np.sqrt(2 * np.pi))
    return gaussian


def func_rho_t(r):
    return rho_i * scaling_rho(r)


def func_rho_d_j(r, idust):
    return (0.1 / ndust) * rho_i * scaling_rho(r)


def func_rho_g(r):
    return rho_i * scaling_rho(r) - sum([func_rho_d_j(r, i) for i in range(ndust)])


def uint_g(r):
    rho_g = func_rho_g(r)
    P = rho_g * cs_g * cs_g / gamma
    return P / ((gamma - 1) * rho_g)


def remap_positions_z(r):
    x, y, z = r

    rn = max(abs(zM), abs(zm))
    z = H * erfinv(z / rn)

    z = min(z, zM)
    z = max(z, zm)
    return (x, y, z)


def compute_sj_new_j(patchdata, j):
    pmass = model.get_particle_mass()

    hpart = patchdata["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    z = patchdata["xyz"][:, 2]
    # mask to only modify particles with |z| < H
    mask = 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))

    epsilon_target = epsilon_base * mrn_distribution.mrn_weight[j] * mask
    print(f"epsilon_target = {epsilon_target} {j}")
    s = np.sqrt(rho * epsilon_target)

    print(
        f"s = {s} {np.isnan(s).any()} epsilon_target = {epsilon_target} mrn_weight = {mrn_distribution.mrn_weight[j]} mask = {mask}, rho = {rho}"
    )

    return s


# %%
# Analysis helpers
# ------------------------------------------


def save_analysis_data(filename, key, value, ianalysis):
    """Helper to save analysis data to a JSON file."""
    if shamrock.sys.world_rank() == 0:
        filepath = os.path.join(dump_folder, filename)
        try:
            with open(filepath, "r") as fp:
                data = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {key: []}
        data[key] = data[key][:ianalysis]
        data[key].append({"t": model.get_time(), key: value})
        with open(filepath, "w") as fp:
            json.dump(data, fp, indent=4)


def load_data_from_json(filename, key):
    """Helper to load analysis data from a JSON file."""
    filepath = os.path.join(dump_folder, filename)
    with open(filepath, "r") as fp:
        data = json.load(fp)[key]
    t = [d["t"] for d in data]
    values = [d[key] for d in data]
    return t, values


def get_max_v():
    dic = ctx.collect_data()
    v = dic["vxyz"]
    vnorms = np.linalg.norm(v, axis=1)
    return vnorms.max()


def compute_delta_v_eff(s_j, delta_v):
    """Effective collision speed between the dust bins (same as in run_dustydisc.py)

    delta_v_eff = sum_i sum_j rho_i rho_j |dv_i - dv_j| / (sum_i rho_i)^2
    """
    rho_d = s_j**2

    # sum_i sum_j rho_i * rho_j == (sum_i rho_i)**2
    denom = np.sum(rho_d, axis=1) ** 2

    # dv_ii == 0 and dv_ij == dv_ji, so only the i<j pairs are needed, each counted twice
    numer = np.zeros(rho_d.shape[0])
    for i in range(ndust):
        for j in range(i + 1, ndust):
            dv_ij = np.linalg.norm(delta_v[:, i, :] - delta_v[:, j, :], axis=-1)
            numer += 2 * rho_d[:, i] * rho_d[:, j] * dv_ij

    # nan before the dust injection (no dust at all)
    with np.errstate(invalid="ignore", divide="ignore"):
        return numer / denom


# %%
# Plotting
# ------------------------------------------


def analyse_and_plot(j):

    pmass = model.get_particle_mass()
    hfact = model.get_hfact()

    dic = ctx.collect_data()
    print(dic["s_j"])

    print(dic["xyz"].shape)

    x = dic["xyz"][:, 0]
    y = dic["xyz"][:, 1]
    z = dic["xyz"][:, 2]
    s_j = dic["s_j"].reshape(-1, ndust)
    ds_j_dt = dic["ds_j_dt"].reshape(-1, ndust)
    cs = dic["soundspeed"]
    delta_v = dic["delta_v"].reshape(-1, ndust, 3)

    print(s_j)

    hpart = dic["hpart"]
    rho = pmass * (hfact / np.array(hpart)) ** 3

    print("compute original rho")
    estimated_rho = [func_rho_t(dic["xyz"][kk]) for kk in range(len(dic["xyz"]))]

    sz = 1

    fig = plt.figure(figsize=(12, 7), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], wspace=0.2)

    ax_rho = fig.add_subplot(gs[0, 0])  # Top left
    ax_epsilon = fig.add_subplot(gs[0, 1])  # Top right
    ax_delta_v = fig.add_subplot(gs[1, :])  # Bottom spans both columns

    time = model.get_time() - tinject
    fig.suptitle(f"t = {time:.2f} [yr]")

    fig.subplots_adjust(left=0.08, right=1.05, wspace=0.35)

    to_dens = codeu.to("kg") * codeu.to("m") ** -3
    to_speed = codeu.to("m") / codeu.to("s")

    dust_cmap = plt.colormaps[cmap]
    dust_norm = mcolors.LogNorm(
        vmin=mrn_distribution.grain_size_si.min(), vmax=mrn_distribution.grain_size_si.max() * 10
    )
    dust_colors = dust_cmap(dust_norm(mrn_distribution.grain_size_si))

    rho_dust_all = np.zeros(len(z))
    epsilon_dust_all = np.zeros(len(z))

    for i in range(ndust):
        c = dust_colors[i]
        ax_rho.scatter(z, s_j[:, i] ** 2 * to_dens, s=sz, color=c, edgecolors="none")
        ax_epsilon.scatter(z, s_j[:, i] ** 2 / rho, s=sz, color=c, edgecolors="none")

        rho_dust_all += s_j[:, i] ** 2 * to_dens
        epsilon_dust_all += s_j[:, i] ** 2 / rho

    dust_mass = analysis_dust_mass.get_dust_mass()

    # if all dust mass is zero replace by nans
    if np.max(dust_mass) == 0:
        print("all dust mass is zero, replacing by nans")
        dust_mass = [np.nan for _ in range(ndust)]

    save_analysis_data("dust_mass.json", "dust_mass", dust_mass, j)

    ax_rho.scatter(z, rho * to_dens, s=sz, color="0.0", edgecolors="none")
    ax_rho.scatter(z, rho_dust_all, s=sz, color="0.5", edgecolors="none")
    ax_epsilon.scatter(z, 1 - epsilon_dust_all, s=sz, color="0.0", edgecolors="none")
    ax_epsilon.scatter(z, epsilon_dust_all, s=sz, color="0.5", edgecolors="none")

    range_plot = 2.5 * H

    ax_rho.set_ylabel(r"$\rho$ [kg/m$^3$]")
    ax_rho.set_xlabel(r"$z$")
    ax_rho.set_yscale("log")
    ax_rho.set_ylim(1e-20, 1e-8)
    ax_rho.set_xlim(-range_plot, range_plot)

    ax_epsilon.set_ylabel(r"$\epsilon_j$")
    ax_epsilon.set_xlabel(r"$z$")
    ax_epsilon.set_yscale("log")
    ax_epsilon.set_ylim(1e-4, 1e-1)  # if you want to see the dust only
    ax_epsilon.set_xlim(-range_plot, range_plot)

    ax_delta_v.set_ylabel(
        r"$\Delta v_{z,j} \, \mathbb{1}_{\rho_{{\rm d},j} > \rho_\epsilon}$ [m/s]"
    )
    ax_delta_v.set_xlabel(r"$z$")
    ax_delta_v.set_xlim(-range_plot, range_plot)
    ax_delta_v.set_yscale("symlog", linthresh=1e-1)
    ax_delta_v.set_ylim(-1e2, 1e2)

    for i in range(ndust):
        c = dust_colors[i]
        # null where the bin is below the dust density floor
        resolved = s_j[:, i] ** 2 > rhodust_eps
        ax_delta_v.scatter(
            z, delta_v[:, i, 2] * resolved * to_speed, s=sz, color=c, edgecolors="none"
        )

    delta_v_eff = compute_delta_v_eff(s_j, delta_v)
    isort = np.argsort(z)
    ax_delta_v.plot(
        z[isort],
        delta_v_eff[isort] * to_speed,
        color="0.0",
        linewidth=1,
        label=r"$\Delta v_{\rm eff}$",
    )
    ax_delta_v.legend(loc="upper right", fontsize=8)

    gas_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="0.",
        markeredgecolor="none",
        label="gas",
    )

    dust_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="0.5",
        markeredgecolor="none",
        label="dust",
    )

    shamrock.matplotlib.add_cmap_legend_entry(
        ax_rho,
        dust_cmap,
        label="dust species",
        extra_handles=[gas_handle, dust_handle],
        extra_labels=["gas", "dust sum"],
        loc="upper right",
        fontsize=8,
    )

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(
        dust_sm, ax=[ax_rho, ax_epsilon, ax_delta_v], pad=0.03, shrink=0.95, aspect=40
    )
    cbar.set_label(r"grain size $s$ [m]")

    os.makedirs(f"{dump_folder}/plots", exist_ok=True)
    plt.savefig(f"{dump_folder}/plots/vert_slice_dens_{j:04d}.png")
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=dpi)
    time = model.get_time() - tinject
    fig.suptitle(f"t = {time:.2f} [yr]")

    fig.subplots_adjust(left=0.07, right=1.05, wspace=0.35)

    for i in range(ndust):
        c = dust_colors[i]
        axs[0].scatter(z, s_j[:, i], s=sz, color=c, edgecolors="none")
        axs[1].scatter(z, ds_j_dt[:, i], s=sz, color=c, edgecolors="none")

    axs[0].set_ylabel(r"$s_j$")
    axs[0].set_xlabel(r"$z$")
    axs[0].set_xlim(-4 * H, 4 * H)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-20, 1e-1)

    axs[1].set_ylabel(r"$\dot{s}_j$")
    axs[1].set_xlabel(r"$z$")
    axs[1].set_xlim(-4 * H, 4 * H)
    axs[1].set_yscale("symlog", linthresh=1e-10)

    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
    dust_sm.set_array([])
    cbar = fig.colorbar(dust_sm, ax=axs, pad=0.02, shrink=0.85)
    cbar.set_label(r"grain size $s$ [m]")

    plt.savefig(f"{dump_folder}/plots/vert_slice_s_{j:04d}.png")
    plt.close()


# %%
# Simulation setup and restore
# ------------------------------------------

if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)
    os.makedirs(dump_folder, exist_ok=True)

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

dump_helper = shamrock.utils.dump.ShamrockDumpHandleHelper(model, dump_folder)


def setup_model():
    global bmin, bmax

    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=av_alpha_min,
        alpha_max=av_alpha_max,
        sigma_decay=av_sigma_decay,
        alpha_u=av_alpha_u,
        beta_AV=av_beta_AV,
    )

    cfg.set_dust_mode_monofluid_tva(
        nvar=ndust, C_1_fluid=0.1, C_drift=1.0, cfl_density_threshold=1e-50
    )
    cfg.set_dust_drag_epstein(gamma, mrn_distribution.grain_size, mrn_distribution.rho_grains)
    if use_coala:
        cfg.set_dust_evol_coala_coag(
            rhodust_eps, dv_max, massgrid_edges, tabflux_coag, tau_frac=tau_frac
        )
    cfg.add_ext_force_vertical_disc_potential(central_mass=1, R0=1)
    cfg.add_ext_force_velocity_dissipation(eta=vel_dissipation_eta)
    cfg.set_neigh_cache_strategy(NeighCacheStrategy.SingleStage)
    cfg.set_show_cfl_detail(True)
    cfg.set_boundary_periodic()
    cfg.set_units(codeu)
    cfg.set_eos_isothermal(cs)
    cfg.print_status()
    model.set_solver_config(cfg)

    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen, insert_step=scheduler_split_val)

    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)
    print("Current part mass :", pmass)

    # Correct the barycenter
    analysis_barycenter = shamrock.model_sph.analysisBarycenter(model=model)
    barycenter, disc_mass = analysis_barycenter.get_barycenter()

    if shamrock.sys.world_rank() == 0:
        print(f"disc barycenter = {barycenter}")

    model.apply_position_offset((-barycenter[0], -barycenter[1], -barycenter[2]))

    model.remap_positions(remap_positions_z)
    model.set_field_value_lambda_f64("uint", uint_g)

    if random_vel_pert:
        rng = np.random.default_rng(42)

        ampl = dr / 1.0
        print(f"random velocity perturbation amplitude = {ampl}")

        def pert_func(r):
            return tuple(rng.uniform(-ampl, ampl, size=3))

        model.set_field_value_lambda_f64_3("vxyz", pert_func)

    model.set_cfl_cour(cfl_cour_setup)
    model.set_cfl_force(cfl_force_setup)

    model.timestep()


analysis_dust_mass = shamrock.model_sph.analysisDustMass(model=model)
pmass = model.get_particle_mass()


# %%
# Run simulation
# ------------------------------------------

from shamrock.utils.SimulationRunner import SimulationRunner, callback, simulation_setup


class Simulation(SimulationRunner):
    # Use the global vars defined at the top of the file
    t_end = t_end
    dump_prefix = dump_folder + "dump_"

    @callback(at_tsim=[tinject])
    def inject_dust(self, _):
        max_v = get_max_v()
        print(f"max_v = {max_v}")

        if max_v > max_v_inject_threshold:
            raise ValueError("max_v is too high, please increase the injection time")

        for k in range(ndust):

            def compute_sj_new(patchdata):
                return compute_sj_new_j(patchdata, k)

            self.model.overwrite_field_value_f64("s_j", compute_sj_new, k)

            self.model.set_cfl_cour(cfl_cour_inject)
            self.model.set_cfl_force(cfl_force_inject)

        self.model.set_dt(0.0)  # to help the corrector on next step after adding dust

    @callback(tsim_interval=dt_analysis)
    def analysis_plots(self, j):
        analyse_and_plot(j)

    @callback(walltime_interval=30.0)  # Checkpoint the simulation every 30 seconds
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)

    @simulation_setup
    def setup(self):
        setup_model()


Simulation(model).run()

# %%
# Build animations from plot sequences
# ------------------------------------------

glob_str = f"{dump_folder}/plots/vert_slice_dens_*.png"
ani = show_image_sequence(glob_str)

writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save("_to_trash/dustysettle_vert_slice_coala.gif", writer=writer)

if shamrock.sys.world_rank() == 0:
    plt.show()

# %%

glob_str = f"{dump_folder}/plots/vert_slice_s_*.png"
ani = show_image_sequence(glob_str)

writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save("_to_trash/dustysettle_vert_slice_s_coala.gif", writer=writer)

if shamrock.sys.world_rank() == 0:
    plt.show()

# %%
# Plot dust mass conservation history
# ------------------------------------------

t, dust_mass = load_data_from_json("dust_mass.json", "dust_mass")
dust_mass = np.array(dust_mass)

# first analysis with dust (the dust mass is nan before the injection)
iinject = np.argmax(~np.isnan(dust_mass)[:, 0])

t = np.array(t) - tinject

St = np.zeros(ndust)

for k in range(ndust):
    t_dyn = 1
    ts = shamrock.phys.epstein_stopping_time(
        rho_grain=mrn_distribution.rho_grains[k],
        s_grain=mrn_distribution.grain_size[k],
        rho=rho_i,
        cs=cs,
        gamma=gamma,
    )
    St[k] = ts / t_dyn

plt.figure()
for k in range(ndust):
    mh = dust_mass[:, k]
    deviation = (mh / mh[iinject]) - 1

    plt.plot(
        t,
        deviation,
        label=f"dust {k}, s = {mrn_distribution.grain_size_si[k]:.1e} [m], St = {St[k]:.1e}",
    )

total_dust_mass = np.sum(dust_mass, axis=1)
plt.plot(
    t,
    (total_dust_mass / total_dust_mass[iinject]) - 1,
    color="grey",
    label="total dust mass",
    linestyle="--",
)

plt.xlabel("t")
plt.ylabel("$\\delta M_{dust} / M_{dust,0}$")
plt.yscale("symlog", linthresh=1e-8)
# coagulation moves mass between bins, so only the total dust mass is conserved
plt.title("Dust mass evolution")
plt.legend()
plt.tight_layout()
plt.savefig(f"{dump_folder}/plots/dust_mass_history.png")
plt.show()

# %%

print("##### Test result #####")
result = {
    "t": float(t[-1]),
    "dust_mass": (dust_mass[-1] / dust_mass[iinject] - 1).tolist(),
    "St": St.tolist(),
    "lz": lz,
}
print(result)

json.dump(result, open(f"{dump_folder}/test_result.json", "w"), indent=4)
