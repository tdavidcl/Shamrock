"""
Dusty diffusion SPH test
========================

Test that the diffusion of epsilon is correct when the
momentum & energy equation are disabled.
"""


# %%
# Here are the initial condition for the dustydiffuse test
#
# .. math::
#    \rho(\mathbf{r}, 0) = \rho_0
#
# .. math::
#    \epsilon(\mathbf{r}, 0) =
#    \epsilon_0 \max \left(0, 1 - \left(\frac{r}{r_c}\right)^2\right),\quad
#    r = \sqrt{x^2 + y^2 + z^2}
#
# with :math:`\rho_0 = 1`, :math:`\epsilon_0 = 0.1`, :math:`r_c = 0.25`.
#
# Then we use the dust TVA solver but force :math:`d \mathbf{v} / dt = 0` and :math:`d u / dt = 0`.
# In that context the epsilon equation becomes:
#
# .. math::
#    \frac{d \epsilon}{dt} = \nabla \cdot \left( \epsilon \eta \nabla \epsilon \right)
#
# With the initial condition above, the analytical solution is:
#
# .. math::
#    \epsilon(r, t) =
#    A\,|10\eta t + B|^{-3/5} - \frac{r^2}{10\eta t + B},
#
# where
#
# .. math::
#    B = \frac{r_c^2}{\epsilon_0},\qquad
#    A = \epsilon_0 B^{3/5}.
#

# sphinx_gallery_multi_image = "single"
# sphinx_gallery_thumbnail_number = 2

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shamrock.utils import dust_variables as dvar

import shamrock

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Sim parameters
rho = 1
epsilon_0 = 0.1
cs_g = 1
ts = 0.1
rc = 0.25


bmin = (-0.5, -0.5, -0.5)
bmax = (0.5, 0.5, 0.5)

N_target = 3e4

# %%
# The TVA solver can evolve different dust variables (see the note of
# :py:mod:`shamrock.utils.dust_variables`). The test is first run with the default
# :math:`S_j = \sqrt{\rho \epsilon_j}` and then repeated with the other variables for comparison.
# :math:`\epsilon_j` is not positivity preserving, so it is only run with the hard limiter
# :math:`\epsilon_j = \max(\epsilon_j, 0)`.
hard_limiter = True
dust_variables = dvar.selected_dust_variables(hard_limiter=hard_limiter)


def func_rho_t(r):
    return rho


def func_eps(pos):
    r = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    return epsilon_0 * max(0, 1 - (r / rc) ** 2)


# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()


# %%
# Setup
def setup_model(dust_variable):
    xm, ym, zm = bmin
    xM, yM, zM = bmax
    vol_b = (xM - xm) * (yM - ym) * (zM - zm)

    part_vol = vol_b / N_target

    # lattice volume
    HCP_PACKING_DENSITY = 0.74
    part_vol_lattice = HCP_PACKING_DENSITY * part_vol

    dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

    bmin_lat, bmax_lat = shamrock.math.get_ideal_hcp_box(dr, bmin, bmax)
    xm, ym, zm = bmin_lat
    xM, yM, zM = bmax_lat

    vol_b = (xM - xm) * (yM - ym) * (zM - zm)
    totmass = rho * vol_b

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
    # cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    cfg.set_dust_mode_monofluid_tva(
        nvar=1,
        pure_diffusion_mode=True,
        ensure_s_j_positivity=hard_limiter,
        dust_variable=dust_variable,
    )
    cfg.set_dust_drag_constant([ts])
    cfg.set_boundary_periodic()
    cfg.set_eos_isothermal(cs_g)
    cfg.print_status()
    model.set_solver_config(cfg)

    scheduler_split_val = int(2e7)
    scheduler_merge_val = 1

    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    model.resize_simulation_box(bmin_lat, bmax_lat)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin_lat, bmax_lat)
    setup.apply_setup(gen, insert_step=scheduler_split_val)

    def func_X(r):
        return dvar.eps_to_var(dust_variable, func_eps(r), func_rho_t(r))

    model.set_field_value_lambda_f64(dvar.field_name(dust_variable), func_X, 0)

    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)

    model.set_cfl_cour(0.3)
    model.set_cfl_force(0.3)

    # the context must outlive the model, return both
    return ctx, model


t_snapshot = [0.0, 0.1, 0.3, 1, 3, 10]


# %%
# Field recovery for plots
def get_field_results(model, dust_variable):
    def custom_getter_r(size: int, dic_out: dict) -> np.array:
        return np.sqrt(
            dic_out["xyz"][:, 0] ** 2 + dic_out["xyz"][:, 1] ** 2 + dic_out["xyz"][:, 2] ** 2
        )

    r_field = model.compute_field("custom", "f64", custom_getter_r)
    rho_field = model.compute_field("rho", "f64")
    X_field = model.compute_field(dvar.field_name(dust_variable), "f64")
    dXdt_field = model.compute_field(dvar.deriv_field_name(dust_variable), "f64")

    def internal_eps(size: int, s: np.array, rho: np.array) -> np.array:
        return dvar.var_to_eps(dust_variable, s, rho)

    eps_field = shamrock.map_fields_f64(internal_eps, s=X_field, rho=rho_field)

    def internal_rho_g(size: int, rho: np.array, eps: np.array) -> np.array:
        return rho * (1 - eps)

    def internal_rho_d(size: int, rho: np.array, eps: np.array) -> np.array:
        return rho * eps

    rho_g_field = shamrock.map_fields_f64(internal_rho_g, rho=rho_field, eps=eps_field)
    rho_d_field = shamrock.map_fields_f64(internal_rho_d, rho=rho_field, eps=eps_field)

    r_data = np.asarray(r_field.collect_data())
    rho_data = np.asarray(rho_field.collect_data())
    rho_g_data = np.asarray(rho_g_field.collect_data())
    rho_d_data = np.asarray(rho_d_field.collect_data())
    dXdt_data = np.asarray(dXdt_field.collect_data())
    return r_data, rho_data, rho_g_data, rho_d_data, dXdt_data


# %%
# Analytical solutions
r_ana = np.linspace(0, 0.5, 100)


def analytic_eps(r, t, eta=0.1):

    B = (rc**2) / epsilon_0  # that frac is in the wrong way in PL15
    A = epsilon_0 * (B ** (3.0 / 5.0))

    return A * np.abs(10 * eta * t + B) ** (-3.0 / 5.0) - (r**2 / (10 * eta * t + B))


def analytic_eps_curve(t):
    return np.array([analytic_eps(r, t) for r in r_ana])


def analytic_dsdt(t):
    dt = 1e-4
    deps_dt = (analytic_eps_curve(t + dt) - analytic_eps_curve(t - dt)) / (2 * dt)
    s = np.sqrt(rho * analytic_eps_curve(t))
    return deps_dt / (2 * s + 1e-9)


# %%
# Perform the simulation
os.makedirs("_to_trash", exist_ok=True)


def run_case(dust_variable, make_frames):
    ctx, model = setup_model(dust_variable)
    model.timestep()

    analysis_dust_mass = shamrock.model_sph.analysisDustMass(model=model)

    snapshots = []
    t_hist = []
    dust_mass_hist = []

    for t in [0.1 * i for i in range(20)]:
        model.evolve_until(t)
        r_data, rho_data, rho_g_data, rho_d_data, dXdt_data = get_field_results(
            model, dust_variable
        )
        eps = rho_d_data / rho_data

        t_hist.append(t)
        dust_mass_hist.append(analysis_dust_mass.get_dust_mass()[0])

        if any(np.isclose(t, ts, atol=1e-6) for ts in t_snapshot):
            snapshots.append((t, r_data, eps))

        if not make_frames:
            continue

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(r_data, eps, ".", label="eps")
        axs[0].plot(r_ana, analytic_eps_curve(t), "--", color="black", label="analytic")
        axs[0].set_xlabel(r"$r$")
        axs[0].set_ylabel(r"$\epsilon$")
        axs[0].set_xlim(0, 0.5)
        axs[0].set_ylim(0, 0.11)
        axs[0].text(
            0.02,
            0.98,
            f"t = {t:.2f}",
            transform=axs[0].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axs[1].plot(r_data, dXdt_data, ".", label="ds/dt")
        axs[1].plot(r_ana, analytic_dsdt(t), "--", color="black", label="analytic")
        axs[1].set_xlabel(r"$r$")
        axs[1].set_ylabel(r"$\frac{d s}{d t}$")
        axs[1].set_xlim(0, 0.5)
        axs[1].set_ylim(-0.16, 0.4)
        plt.tight_layout()
        plt.savefig(f"_to_trash/dump_dustydiffuse_tva_{t:.2f}.png")
        plt.close()

    return {
        "snapshots": snapshots,
        "t": np.array(t_hist),
        "dust_mass": np.array(dust_mass_hist),
    }


results = {}
for dust_variable in dust_variables:
    results[dust_variable] = run_case(dust_variable, make_frames=(dust_variable == "sqrt_rho_eps"))

####################################################
# Plot making
####################################################

# %%
# You may notice the precense of a small kink at the edge of the diffusion or a spike in the ds/dt
# This is due to the low resolution of the test. If you push it is will soften.
#
# Also remember that :math:`s = \sqrt{\rho \epsilon}` raise sharply from 0 which does not help.
# In that context using the :math:`\epsilon` behaves better.

####################################################
# Convert PNG sequence to Image sequence in mpl
####################################################

from shamrock.utils.plot import show_image_sequence

# If the animation is not returned only a static image will be shown in the doc

if "sqrt_rho_eps" in results:
    glob_str = os.path.join("_to_trash", "dump_dustydiffuse_tva_*.png")
    ani = show_image_sequence(glob_str)

    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save("_to_trash/dump_dustydiffuse_tva.gif", writer=writer)

    if shamrock.sys.world_rank() == 0:
        # Show the animation
        plt.show()

####################################################
# PL15 like figure
####################################################

if "sqrt_rho_eps" in results:
    plt.figure()
    for i, (t, r_data, eps) in enumerate(results["sqrt_rho_eps"]["snapshots"]):
        plt.plot(
            r_ana,
            analytic_eps_curve(t),
            "--",
            color="black",
            label="analytic" if i == 0 else "_nolegend_",
        )
        plt.plot(r_data, eps, ".", label=f"t = {t:.2f}")

    plt.xlabel(r"$r$")
    plt.ylabel(r"$\epsilon$")
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.11)
    plt.legend()
    plt.show()


####################################################
# Comparison of the dust variables
####################################################


def binned_profile(r, y, bins):
    """Mean of y in radial bins (particles are noisy, the mean makes the comparison readable)"""
    idx = np.digitize(r, bins) - 1
    centers = 0.5 * (bins[1:] + bins[:-1])
    mean = np.array(
        [np.mean(y[idx == i]) if np.any(idx == i) else np.nan for i in range(len(centers))]
    )
    return centers, mean


r_bins = np.linspace(0, 0.5, 41)

# %%
# Comparison of the dust variables: :math:`\epsilon(r)` profiles
#
# Top: radially binned :math:`\epsilon(r)` for each evolved dust variable, with the analytic
# solution in black. Bottom: :math:`\epsilon - \epsilon_{\rm analytic}`.

snapshot_times = [t for (t, _, _) in results[dust_variables[0]]["snapshots"] if t > 0]

fig, axs = plt.subplots(
    2,
    len(snapshot_times),
    figsize=(4 * len(snapshot_times), 6),
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [2, 1]},
    squeeze=False,
)

for k, t_snap in enumerate(snapshot_times):
    eps_ana_bins = np.array([analytic_eps(r, t_snap) for r in 0.5 * (r_bins[1:] + r_bins[:-1])])
    axs[0, k].plot(r_ana, analytic_eps_curve(t_snap), "-", color="black", label="analytic")

    for dust_variable in dust_variables:
        for t, r_data, eps in results[dust_variable]["snapshots"]:
            if not np.isclose(t, t_snap, atol=1e-6):
                continue
            centers, eps_mean = binned_profile(r_data, eps, r_bins)
            axs[0, k].plot(
                centers,
                eps_mean,
                marker=dvar.MARKERS[dust_variable],
                markersize=3,
                linestyle="none",
                color=dvar.COLORS[dust_variable],
                label=dvar.LABELS[dust_variable],
            )
            axs[1, k].plot(
                centers,
                eps_mean - eps_ana_bins,
                marker=dvar.MARKERS[dust_variable],
                markersize=3,
                linestyle="-",
                linewidth=0.8,
                color=dvar.COLORS[dust_variable],
            )

    axs[0, k].set_title(f"t = {t_snap:.2f}")
    axs[1, k].axhline(0, color="black", linewidth=0.8)
    axs[1, k].set_xlabel(r"$r$")
    axs[0, k].set_xlim(0, 0.5)

axs[0, 0].set_ylabel(r"$\epsilon$")
axs[1, 0].set_ylabel(r"$\epsilon - \epsilon_{\rm analytic}$")
axs[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("_to_trash/dustydiffuse_tva_dust_variables_profiles.png")
plt.show()

# %%
# Comparison of the dust variables: dust mass conservation
#
# Relative drift of the total dust mass. :math:`\epsilon_j` is linear in the evolved variable so
# it conserves the dust mass to round-off, except where the hard limiter clips negative values.
# The square-root variables have an :math:`\mathcal{O}(\Delta t^2)` time-integration error.

plt.figure()
for dust_variable in dust_variables:
    res = results[dust_variable]
    # floor at 1e-17 so that an exactly conserved mass still shows on the log scale
    drift = np.maximum(np.abs(res["dust_mass"] / res["dust_mass"][0] - 1), 1e-17)
    plt.plot(
        res["t"][1:],
        drift[1:],
        marker=dvar.MARKERS[dust_variable],
        markersize=3,
        color=dvar.COLORS[dust_variable],
        label=dvar.LABELS[dust_variable],
    )
plt.yscale("log")
plt.xlabel("t")
plt.ylabel(r"$|M_{\rm dust}(t) / M_{\rm dust}(0) - 1|$")
plt.title("Dust mass conservation")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("_to_trash/dustydiffuse_tva_dust_variables_mass.png")
plt.show()
