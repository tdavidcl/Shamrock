"""
Dusty wave SPH test
========================

Test that the dust/gas wave evolution match the eigen mode analysis.
"""

# sphinx_gallery_multi_image = "single"

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
epsilon_0 = 0.5
cs_g_list = np.logspace(-4, -1, 3).tolist()
ts = 1

ampl_perturbation = 0.001
plot_scaling = 1e3
label_scaling = "10^3 \\cdot"
delta_v_0_list = [cs * ampl_perturbation for cs in cs_g_list]

lx = int(os.environ.get("LZ", "18"))
ly = 12
lz = 12

# %%
# The TVA solver can evolve different dust variables (see :py:mod:`shamrock.utils.dust_variables`).
# The test is first run with the default :math:`S_j = \sqrt{\rho \epsilon_j}` and then repeated
# with the other variables for comparison. :math:`\epsilon_j` is not positivity preserving, so it
# is only run with the hard limiter :math:`\epsilon_j = \max(\epsilon_j, 0)`.
hard_limiter = True
dust_variables = dvar.selected_dust_variables(hard_limiter=hard_limiter)

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()


# %%
# Do setup

lmin = (-(lx // 2), -(ly // 2), -(lz // 2))
lmax = (lx // 2, ly // 2, lz // 2)

# Call with dr = 1 as we will rescale on next call
(xm, ym, zm), (xM, yM, zM) = shamrock.math.get_periodic_hcp_box(1.0, lmin, lmax)
print(f"base lattice : xM = {xM}, yM = {yM}, zM = {zM}")
dr = 1.0 / (xM - xm)
print(f"dr = {dr}")
bmin, bmax = shamrock.math.get_periodic_hcp_box(dr, lmin, lmax)
print(f"new lattice : bmin = {bmin}, bmax = {bmax}")
xm, ym, zm = bmin
xM, yM, zM = bmax

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho * vol_b


def do_setup(model, cs, delta_v_0, dust_variable="sqrt_rho_eps"):

    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
    # cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    cfg.set_dust_mode_monofluid_tva(
        nvar=1, ensure_s_j_positivity=hard_limiter, dust_variable=dust_variable
    )
    cfg.set_dust_drag_constant([ts])
    cfg.set_boundary_periodic()
    cfg.set_eos_isothermal(cs)
    cfg.print_status()
    model.set_solver_config(cfg)

    scheduler_split_val = int(2e7)
    scheduler_merge_val = 1

    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen, insert_step=scheduler_split_val)

    def func_s(r):
        return dvar.eps_to_var(dust_variable, epsilon_0, rho)

    model.set_field_value_lambda_f64(dvar.field_name(dust_variable), func_s, 0)

    print(delta_v_0)

    def vel_func(r):
        global mm, MM
        x, y, z = r

        f = 2 * np.pi / (xM - xm)

        vel = delta_v_0 * np.sin(x * f)

        return (vel, 0.0, 0.0)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)

    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    model.timestep()


# %%
# Field recovery for plots
def get_field_results(model, dust_variable="sqrt_rho_eps"):
    def custom_getter_x(size: int, dic_out: dict) -> np.array:
        return dic_out["xyz"][:, 0]

    def custom_getter_vx(size: int, dic_out: dict) -> np.array:
        return dic_out["vxyz"][:, 0]

    x_field = model.compute_field("custom", "f64", custom_getter_x)
    vx_field = model.compute_field("custom", "f64", custom_getter_vx)
    rho_field = model.compute_field("rho", "f64")
    s_j_field = model.compute_field(dvar.field_name(dust_variable), "f64")

    def internal_eps(size: int, s: np.array, rho: np.array) -> np.array:
        return dvar.var_to_eps(dust_variable, s, rho)

    eps_field = shamrock.map_fields_f64(internal_eps, s=s_j_field, rho=rho_field)

    def internal_rho_g(size: int, rho: np.array, eps: np.array) -> np.array:
        return rho * (1 - eps)

    def internal_rho_d(size: int, rho: np.array, eps: np.array) -> np.array:
        return rho * eps

    rho_g_field = shamrock.map_fields_f64(internal_rho_g, rho=rho_field, eps=eps_field)
    rho_d_field = shamrock.map_fields_f64(internal_rho_d, rho=rho_field, eps=eps_field)

    x_data = np.asarray(x_field.collect_data())
    vx_data = np.asarray(vx_field.collect_data())
    eps_data = np.asarray(eps_field.collect_data())
    rho_data = np.asarray(rho_field.collect_data())
    rho_g_data = np.asarray(rho_g_field.collect_data())
    rho_d_data = np.asarray(rho_d_field.collect_data())
    return x_data, rho_data, rho_g_data, rho_d_data, vx_data, eps_data


# %%
# Analytics


def dustywave_tva_matrix(k, cs, ts, eps):
    a = k * cs
    b = k * k * ts * cs * cs * eps

    return np.array(
        [
            [0, 0, -1j * a],
            [b * (1 - eps), -b, 0],
            [-1j * a * (1 - eps), 1j * a, 0],
        ],
        dtype=complex,
    )


def eigensystem_dustywave_tva(k, cs, ts, eps):
    M = dustywave_tva_matrix(k, cs, ts, eps)
    vals, vecs = np.linalg.eig(M)
    return 1j * vals, vecs


def dustywave_dispersion_relation(omega_k: float, k: float, cs: float, ts: float, eps: float):
    # w^4 + i w^3 / ts - cs^2 k^2 w^2 - i cs^2 k^2 (1-eps) w / ts = 0
    return (
        omega_k**4
        + 1j * (omega_k**3 / ts)
        - (cs**2 * k**2 * omega_k**2)
        - 1j * (cs**2 * k**2 * (1 - eps) * omega_k / ts)
    )


def get_dustywave_omega_k(k: float, cs: float, ts: float, eps: float) -> np.ndarray:
    # w^4 + i w^3/ts - cs^2 k^2 w^2 - i cs^2 k^2 (1-eps) w/ts = 0
    coeffs = [
        1.0,
        1j / ts,
        -(cs**2 * k**2),
        -1j * (cs**2 * k**2 * (1.0 - eps) / ts),
        0.0,
    ]
    return np.roots(coeffs)


def eigen_model(x, t, offset, ampl, omega, k):
    return offset + np.real(ampl * np.exp(1j * (k * x - omega * t)))


def project_eigenmode(
    eigenvec: complex, eigenval: complex, rho_on_rho_0: complex, eps: complex, v_on_cs: complex
):
    v = np.array([rho_on_rho_0, eps, v_on_cs], dtype=complex)
    print(f"eigenvec={eigenvec}")
    print(f"v={v}")

    c = np.linalg.solve(eigenvec, v)

    print(f"c={c}")
    return c


def find_eigen_decomp(x_data, rho_data, eps_data, vx_data, eigval, eigvec, cs):

    offset_rho, ampl_rho, phi_rho = fit_sine_wave(x_data, rho_data)
    offset_eps, ampl_eps, phi_eps = fit_sine_wave(x_data, eps_data)
    offset_vx, ampl_vx, phi_vx = fit_sine_wave(x_data, vx_data)

    print(f"offset_rho={offset_rho:.6g}, ampl_rho={ampl_rho:.6g}, phi_rho={phi_rho:.6g} rad")
    print(f"offset_eps={offset_eps:.6g}, ampl_eps={ampl_eps:.6g}, phi_eps={phi_eps:.6g} rad")
    print(f"offset_vx={offset_vx:.6g}, ampl_vx={ampl_vx:.6g}, phi_vx={phi_vx:.6g} rad")

    print(f"eigenval = {eigval}")
    print(f"eigenvec = {eigvec}")

    coefs = project_eigenmode(eigvec, eigval, ampl_rho / rho, ampl_eps, -1j * ampl_vx / cs)
    print(f"coefs={coefs}")

    return coefs


# %%
# Curve fitting
from scipy.linalg import lstsq
from scipy.optimize import curve_fit


def fit_sine_wave(x, y, prev_phi=None):
    offset = np.mean(y)
    y0 = y - offset

    z = np.exp(1j * k * x)

    lam = np.vdot(z, y0) / np.vdot(z, z)

    ampl = 2 * np.abs(lam)
    phi = np.angle(lam)

    # enforce continuity with previous phase if available
    if prev_phi is not None:
        # candidate 1
        phi1 = phi
        a1 = ampl

        # candidate 2 (flip sign via phase shift)
        phi2 = phi + np.pi
        a2 = -ampl

        # choose whichever is closer to previous phase
        if abs(np.angle(np.exp(1j * (phi1 - prev_phi)))) > abs(
            np.angle(np.exp(1j * (phi2 - prev_phi)))
        ):
            phi, ampl = phi2, a2

    # wrap phase
    phi = np.mod(phi, 2 * np.pi)

    return offset, ampl, phi


# %%
# Perform the simulation

k = 2 * np.pi / (xM - xm)


def run_case(ics, cs, dust_variable, make_frames):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")
    do_setup(model, cs, delta_v_0_list[ics], dust_variable)

    # Compute Omega
    omega_k = get_dustywave_omega_k(k, cs, ts, epsilon_0)
    print(omega_k)

    print(f"k={k} cs={cs} ts={ts} epsilon_0={epsilon_0}")
    eigval, eigvec = eigensystem_dustywave_tva(k, cs, ts, epsilon_0)

    print(f"eigenval = {eigval}")
    print(f"eigenvec = {eigvec}")

    Twave = 2 * np.pi / np.max(np.abs(np.real(eigval)))
    print(Twave)

    Twave_cnt = 40
    nwave = 2.0

    t_list = []
    rho_t_list = []
    eps_t_list = []
    vx_t_list = []
    rho_t_list_analytic = []
    eps_t_list_analytic = []
    vx_t_list_analytic = []

    rho_last_phi = np.pi
    eps_last_phi = 0
    vx_last_phi = 3 * np.pi / 2

    os.makedirs("_to_trash", exist_ok=True)
    for i in range(int(Twave_cnt * nwave)):
        t = Twave * i / (Twave_cnt)
        model.evolve_until(t)

        x_data, rho_data, rho_g_data, rho_d_data, vx_data, eps_data = get_field_results(
            model, dust_variable
        )

        x_ana = np.linspace(xm, xM, 256)

        if i == 0:
            coefs = find_eigen_decomp(x_data, rho_data, eps_data, vx_data, eigval, eigvec, cs)
            print(f"coefs={coefs}")

            decomp = np.array(eigvec[0] * 0)
            for ieig in range(len(eigval)):
                decomp += coefs[ieig] * eigvec[:, ieig]
            print(f"decomp={decomp}")

        model_rho_on_rho_0 = np.zeros_like(x_ana, dtype=complex)
        model_eps = np.zeros_like(x_ana, dtype=complex)
        model_vx_on_cs = np.zeros_like(x_ana, dtype=complex)

        for ieig in range(len(eigval)):
            model_rho_on_rho_0 += eigen_model(
                x_ana, model.get_time(), 0.0, coefs[ieig] * eigvec[0, ieig], eigval[ieig], k
            )
            model_eps += eigen_model(
                x_ana, model.get_time(), 0.0, coefs[ieig] * eigvec[1, ieig], eigval[ieig], k
            )
            model_vx_on_cs += eigen_model(
                x_ana, model.get_time(), 0.0, coefs[ieig] * eigvec[2, ieig], eigval[ieig], k
            )

        model_rho = rho * np.real(model_rho_on_rho_0)
        model_eps = np.real(model_eps)
        model_vx = cs * np.real(model_vx_on_cs)

        model_rho += 1
        model_eps += 0.5
        model_vx += 0

        _, rho_t_ampl, rho_t_phi = fit_sine_wave(x_data, rho_data, rho_last_phi)
        _, eps_t_ampl, eps_t_phi = fit_sine_wave(x_data, eps_data, eps_last_phi)
        _, vx_t_ampl, vx_t_phi = fit_sine_wave(x_data, vx_data, vx_last_phi)
        _, rho_ana_ampl, _ = fit_sine_wave(x_ana, model_rho, rho_last_phi)
        _, eps_ana_ampl, _ = fit_sine_wave(x_ana, model_eps, eps_last_phi)
        _, vx_ana_ampl, _ = fit_sine_wave(x_ana, model_vx, vx_last_phi)

        print(f"rho_t_ampl={rho_t_ampl:.6g}, rho_t_phi={rho_t_phi:.6g} rad")
        print(f"eps_t_ampl={eps_t_ampl:.6g}, eps_t_phi={eps_t_phi:.6g} rad")
        print(f"vx_t_ampl={vx_t_ampl:.6g}, vx_t_phi={vx_t_phi:.6g} rad")

        t_list.append(model.get_time())
        rho_t_list.append(rho_t_ampl)
        eps_t_list.append(eps_t_ampl)
        vx_t_list.append(vx_t_ampl)
        rho_t_list_analytic.append(rho_ana_ampl)
        eps_t_list_analytic.append(eps_ana_ampl)
        vx_t_list_analytic.append(vx_ana_ampl)

        rho_last_phi = rho_t_phi
        eps_last_phi = eps_t_phi
        vx_last_phi = vx_t_phi

        if not make_frames:
            continue

        fig, axs = plt.subplots(1, 1, figsize=(10, 5))

        axs.plot(x_data, rho_data - 1, ".", label=r"$\delta \rho$")
        axs.plot(x_data, eps_data - 0.5, ".", label=r"$\delta \epsilon$")
        axs.plot(x_data, vx_data / cs, ".", label=r"$\delta v_x / c_s$")

        axs.plot(x_ana, model_rho - 1, "-", label=r"$\delta \rho$ analytic")
        axs.plot(x_ana, model_eps - 0.5, "-", label=r"$\delta \epsilon$ analytic")
        axs.plot(x_ana, model_vx / cs, "-", label=r"$\delta v_x / c_s$ analytic")

        axs.set_xlabel(r"$x$ [code unit]")
        axs.set_ylabel(r"$\delta$ fields [code unit]")
        axs.set_xlim(xm, xM)
        # axs.set_ylim(rho / 2 - 1e-3, rho / 2 + 1e-3)
        axs.set_ylim(-ampl_perturbation * 1.1, +ampl_perturbation * 1.1)
        axs.text(
            0.02,
            0.98,
            f"t = {t:.2f} | cs = {cs:e}",
            transform=axs.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"_to_trash/dump_dustywave_tva_{ics:02d}_{i:02d}.png")
        plt.close()

        # if i == 1:
        #    break

    t_arr = np.asarray(t_list)

    amplitudes = {
        "t": t_arr,
        "rho": np.array(rho_t_list),
        "eps": np.array(eps_t_list),
        "vx_on_cs": np.array(vx_t_list) / cs,
        "rho_ana": np.array(rho_t_list_analytic),
        "eps_ana": np.array(eps_t_list_analytic),
        "vx_on_cs_ana": np.array(vx_t_list_analytic) / cs,
    }

    if not make_frames:
        return None, amplitudes

    rho_t_list = np.array(rho_t_list)
    eps_t_list = np.array(eps_t_list)
    vx_t_list = np.array(vx_t_list)
    rho_t_list_analytic = np.array(rho_t_list_analytic)
    eps_t_list_analytic = np.array(eps_t_list_analytic)
    vx_t_list_analytic = np.array(vx_t_list_analytic)

    curves = []

    factor = plot_scaling

    def add_curve(x, y, symbol, label):
        curves.append(
            {
                "x": x,
                "y": factor * y,
                "symbol": symbol,
                "label": label,
            }
        )

    add_curve(t_arr, rho_t_list, ".", r"$\delta \rho (t)$")
    add_curve(t_arr, eps_t_list, ".", r"$\delta \epsilon (t)$")
    add_curve(t_arr, vx_t_list / cs, ".", r"$\delta v_x / c_s (t)$")
    add_curve(t_arr, rho_t_list_analytic, "--", r"$\delta \rho (t)$ analytic")
    add_curve(t_arr, eps_t_list_analytic, "--", r"$\delta \epsilon (t)$ analytic")
    add_curve(t_arr, vx_t_list_analytic / cs, "--", r"$\delta v_x / c_s(t)$ analytic")

    return_dict = {
        "curves": curves,
        "cs": cs,
        "ics": ics,
        "xlabel": "$t$ [code unit]",
        "ylabel": f"${label_scaling} \\delta$ fields [code unit]",
        "title": f"cs={cs:.2e} [code unit]",
    }

    plt.figure(dpi=150)
    for curve in curves:
        plt.plot(curve["x"], curve["y"], curve["symbol"], label=curve["label"])
    plt.xlabel(return_dict["xlabel"])
    plt.ylabel(return_dict["ylabel"])
    plt.title(return_dict["title"])
    plt.legend(fontsize=12, loc="upper right")
    plt.savefig(f"_to_trash/dustywave_tva_scan_{return_dict['ics']:04}.png")

    return return_dict, amplitudes


all_case_plot = []
amplitudes = {v: [] for v in dust_variables}

for dust_variable in dust_variables:
    for ics, cs in enumerate(cs_g_list):
        return_dict, ampl = run_case(
            ics, cs, dust_variable, make_frames=(dust_variable == "sqrt_rho_eps")
        )
        amplitudes[dust_variable].append(ampl)
        if return_dict is not None:
            all_case_plot.append(return_dict)

# %%
# make gifs
from matplotlib.animation import PillowWriter
from shamrock.utils.plot import show_image_sequence

keep_list = []

# %%
# show them the gifs (i have to unroll the loop otherwise the doc does not capture the gifs ...)
ani0 = show_image_sequence(f"_to_trash/dump_dustywave_tva_{0:02d}_*.png")
writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani0.save(f"_to_trash/dustywave_tva_scan_{0:04}.gif", writer=writer)
plt.show()

# %%
ani1 = show_image_sequence(f"_to_trash/dump_dustywave_tva_{1:02d}_*.png")
writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani1.save(f"_to_trash/dustywave_tva_scan_{1:04}.gif", writer=writer)
plt.show()

# %%
ani2 = show_image_sequence(f"_to_trash/dump_dustywave_tva_{2:02d}_*.png")
writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani2.save(f"_to_trash/dustywave_tva_scan_{2:04}.gif", writer=writer)
plt.show()


# %%
fig, axs = plt.subplots(1, len(all_case_plot), figsize=(12, 5), sharey=True)
for i, case in enumerate(all_case_plot):
    for curve in case["curves"]:
        axs[i].plot(curve["x"], curve["y"], curve["symbol"], label=curve["label"])
    axs[i].set_xlabel(case["xlabel"])
    if i == 0:
        axs[i].set_ylabel(case["ylabel"])
    axs[i].set_title(case["title"])

axs[0].legend(fontsize=11, loc="upper left")
plt.tight_layout()
plt.savefig("_to_trash/dustywave_tva_scan_all.png")
plt.savefig("_to_trash/dustywave_tva_scan_all.pdf")
plt.show()


####################################################
# Comparison of the dust variables
####################################################

fields_cmp = [
    ("rho", r"$\delta \rho$"),
    ("eps", r"$\delta \epsilon$"),
    ("vx_on_cs", r"$\delta v_x / c_s$"),
]

# %%
# Comparison of the dust variables: fitted amplitudes
#
# Fitted amplitude of each field for every evolved dust variable (markers) against the eigenmode
# solution (black). Columns are the sound speeds of the scan.

fig, axs = plt.subplots(
    len(fields_cmp),
    len(cs_g_list),
    figsize=(4.5 * len(cs_g_list), 3 * len(fields_cmp)),
    sharex="col",
    squeeze=False,
)
for ics, cs in enumerate(cs_g_list):
    ref = amplitudes[dust_variables[0]][ics]
    for irow, (key, label) in enumerate(fields_cmp):
        ax = axs[irow, ics]
        ax.plot(ref["t"], plot_scaling * ref[key + "_ana"], "-", color="black", label="analytic")
        for dust_variable in dust_variables:
            a = amplitudes[dust_variable][ics]
            ax.plot(
                a["t"],
                plot_scaling * a[key],
                marker=dvar.MARKERS[dust_variable],
                fillstyle=dvar.FILLSTYLES[dust_variable],
                markersize=3,
                linestyle="none",
                color=dvar.COLORS[dust_variable],
                label=dvar.LABELS[dust_variable],
            )
        if ics == 0:
            ax.set_ylabel(f"${label_scaling}$ " + label)
        if irow == 0:
            ax.set_title(f"cs={cs:.2e} [code unit]")
        if irow == len(fields_cmp) - 1:
            ax.set_xlabel("$t$ [code unit]")
axs[0, 0].legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig("_to_trash/dustywave_tva_dust_variables_amplitudes.png")
plt.show()

# %%
# Comparison of the dust variables: amplitude error
#
# Error of the fitted amplitude with respect to the eigenmode solution, normalised by the
# initial perturbation amplitude of that field.

fig, axs = plt.subplots(
    len(fields_cmp),
    len(cs_g_list),
    figsize=(4.5 * len(cs_g_list), 3 * len(fields_cmp)),
    sharex="col",
    squeeze=False,
)
for ics, cs in enumerate(cs_g_list):
    for irow, (key, label) in enumerate(fields_cmp):
        ax = axs[irow, ics]
        for dust_variable in dust_variables:
            a = amplitudes[dust_variable][ics]
            norm = np.max(np.abs(a[key + "_ana"]))
            ax.plot(
                a["t"],
                (a[key] - a[key + "_ana"]) / norm,
                linestyle=dvar.LINESTYLES[dust_variable],
                color=dvar.COLORS[dust_variable],
                label=dvar.LABELS[dust_variable],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        if ics == 0:
            ax.set_ylabel("error " + label + r" / max$|$analytic$|$")
        if irow == 0:
            ax.set_title(f"cs={cs:.2e} [code unit]")
        if irow == len(fields_cmp) - 1:
            ax.set_xlabel("$t$ [code unit]")
axs[0, 0].legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("_to_trash/dustywave_tva_dust_variables_error.png")
plt.show()

# %%
# Comparison of the dust variables: RMS error summary
#
# Time averaged RMS of the normalised amplitude error, as a function of the sound speed of the
# scan (i.e. of the stopping time relative to the wave period).

fig, axs = plt.subplots(1, len(fields_cmp), figsize=(4.5 * len(fields_cmp), 4), squeeze=False)
for irow, (key, label) in enumerate(fields_cmp):
    ax = axs[0, irow]
    for dust_variable in dust_variables:
        rms = []
        for ics, cs in enumerate(cs_g_list):
            a = amplitudes[dust_variable][ics]
            norm = np.max(np.abs(a[key + "_ana"]))
            rms.append(np.sqrt(np.mean(((a[key] - a[key + "_ana"]) / norm) ** 2)))
        ax.plot(
            cs_g_list,
            rms,
            marker=dvar.MARKERS[dust_variable],
            fillstyle=dvar.FILLSTYLES[dust_variable],
            color=dvar.COLORS[dust_variable],
            label=dvar.LABELS[dust_variable],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$c_s$ [code unit]")
    ax.set_title("RMS error " + label)
axs[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("_to_trash/dustywave_tva_dust_variables_rms.png")
plt.show()
