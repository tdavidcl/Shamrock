"""
Testing Sod tube with SPH
=========================

CI test for Sod tube with SPH
"""

import os

import matplotlib.pyplot as plt

import shamrock

shamrock.enable_experimental_features()
import numpy as np
import shamrock.external.coala as coala

print(f"coala path : {coala.__file__}")


gamma = 1.4
rho_i = 1e-7
central_mass = 1
R0 = 1
H_r_0 = 0.05

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()


def kep_profile(r):
    return (G * central_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


H = H_r_0 * R0
cs = H * omega_k(R0)
box = 8 * H

print(f"cs = {cs}")
print(f"H = {H}")


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


cs_g = 1


def uint_g(r):
    rho_g = func_rho_g(r)
    P = rho_g * cs_g * cs_g / gamma
    return P / ((gamma - 1) * rho_g)


ndust = 20

rho_grains_si_edges = np.array([2.3 * 1000 for i in range(ndust + 1)])
grain_size_si_edges = np.logspace(-9, -2, ndust + 1)

print(f"grains sizes = {grain_size_si_edges} [m]")
print(f"grains dens  = {rho_grains_si_edges} [kg.m^-3]")

grain_size_edges = grain_size_si_edges * codeu.get("m")
rho_grains_edges = codeu.get("kg") * codeu.get("m", power=-3) * np.array(rho_grains_si_edges)

print(f"grains sizes = {grain_size_edges} [code u]")
print(f"grains dens  = {rho_grains_edges} [code u]")

grain_size = np.sqrt(grain_size_edges[:-1] * grain_size_edges[1:])
rho_grains = np.sqrt(rho_grains_edges[:-1] * rho_grains_edges[1:])

grain_size_si = np.sqrt(grain_size_si_edges[:-1] * grain_size_si_edges[1:])
rho_grains_si = np.sqrt(rho_grains_si_edges[:-1] * rho_grains_si_edges[1:])

dustlabels = [f"dust {i} s = {grain_size_si[i]:.2e} [m]" for i in range(ndust)]

print(f"grains sizes = {grain_size_si} [m]")
print(f"grains dens  = {rho_grains_si} [kg.m^-3]")

print(f"grains sizes = {grain_size} [code units]")
print(f"grains dens  = {rho_grains} [code units]")

massgrid_edges = (4 * np.pi / 3) * rho_grains_edges * grain_size_edges**3
massgrid = np.sqrt(massgrid_edges[:-1] * massgrid_edges[1:])

print(f"massgrid = {massgrid} [code units]")
print(f"massgrid = {massgrid * codeu.to('kg')} [kg]")

do_coag = True


K0 = np.pi * ((4.0 / 3.0) * np.pi * rho_grains[0]) ** (-2.0 / 3.0)
Q = 5
rhodust_eps = 1e-15

K0 *= 100
print(f"K0 = {K0}")

mrn_pow = 3.5

tabflux_coag = coala.coala_precalc_tabflux_coag(K0, ndust, Q, massgrid_edges)


from scipy.special import erfinv

bmin = (-box / 4, -box / 4, -box)
bmax = (box / 4, box / 4, box)

N_target = 1e4
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

part_vol = vol_b / N_target

# lattice volume
HCP_PACKING_DENSITY = 0.74
part_vol_lattice = HCP_PACKING_DENSITY * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

pmass = -1

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_dust_mode_monofluid_tvi(ndust)

cfg.set_dust_drag_epstein(grain_size, rho_grains)

if do_coag:
    cfg.set_dust_evol_coala_coag(rhodust_eps, massgrid_edges, tabflux_coag)
cfg.add_ext_force_vertical_disc_potential(central_mass=1, R0=1)
cfg.add_ext_force_velocity_dissipation(eta=10)
cfg.set_boundary_periodic()
cfg.set_units(codeu)
cfg.set_eos_isothermal(cs)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8), 1)


bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
setup.apply_setup(gen, insert_step=scheduler_split_val)

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho_i * vol_b
print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)

# Correct the barycenter
analysis_barycenter = shamrock.model_sph.analysisBarycenter(model=model)
barycenter, disc_mass = analysis_barycenter.get_barycenter()

if shamrock.sys.world_rank() == 0:
    print(f"disc barycenter = {barycenter}")

model.apply_position_offset((-barycenter[0], -barycenter[1], -barycenter[2]))


def f_remap(r):
    x, y, z = r

    rn = max(abs(zM), abs(zm))
    # print(y, H, H * erfinv(y / rn))
    z = H * erfinv(z / rn)

    z = min(z, zM)
    z = max(z, zm)
    return (x, y, z)


model.remap_positions(f_remap)
model.set_field_value_lambda_f64("uint", uint_g)


model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.timestep()


mrn_weight = grain_size ** (4 - mrn_pow)
mrn_weight = mrn_weight / np.sum(mrn_weight)
mrn_weight *= grain_size_si < 250e-9

print(f"mrn_weight = {mrn_weight}")


def compute_sj_new_j(patchdata, j):
    hpart = patchdata["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    z = patchdata["xyz"][:, 2]
    # mask to only modify particles with |z| < H
    mask = 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))

    epsilon_target = 0.1 * mrn_weight[j] * mask
    print(f"epsilon_target = {epsilon_target} {j}")
    s = np.sqrt(rho * epsilon_target)

    return s


# TODO: add function to modify fields e.g. get rho and do stuff according to it

tnext = 0
for j in range(1000):
    if j == 30:
        for k in range(ndust):

            def compute_sj_new(patchdata):
                return compute_sj_new_j(patchdata, k)

            model.overwrite_field_value_f64("s_j", compute_sj_new, k)

    if j > 0:
        tnext += 0.1
        model.evolve_until(tnext)
        # model.timestep()

    dic = ctx.collect_data()
    print(dic["s_j"])

    print(dic["xyz"].shape)

    x = dic["xyz"][:, 0]
    y = dic["xyz"][:, 1]
    z = dic["xyz"][:, 2]
    s_j = dic["s_j"].reshape(-1, ndust)
    ds_j_dt = dic["ds_j_dt"].reshape(-1, ndust)
    cs = dic["soundspeed"]

    print(s_j)

    hpart = dic["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    print("compute original rho")
    estimated_rho = [func_rho_t(dic["xyz"][kk]) for kk in range(len(dic["xyz"]))]

    sz = 1

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), dpi=150)
    time = model.get_time()
    fig.suptitle(f"t = {time:.2f}")

    fig.subplots_adjust(left=0.07, right=0.82, wspace=0.35)

    to_dens = codeu.to("kg") * codeu.to("m") ** -3

    # tab20: 20 distinct qualitative colors (no repetition for ndust <= 20)
    dust_colors = plt.colormaps["tab20"](np.linspace(0, 1, ndust, endpoint=False))
    legend_handles = []
    legend_labels = []

    h_gas = axs[0].scatter(z, rho * to_dens, label="gas", s=sz, color="0.3", edgecolors="none")
    legend_handles.append(h_gas)
    legend_labels.append("gas")

    for i in range(ndust):
        c = dust_colors[i]
        h = axs[0].scatter(
            z, s_j[:, i] ** 2 * to_dens, label=dustlabels[i], s=sz, color=c, edgecolors="none"
        )
        legend_handles.append(h)
        legend_labels.append(dustlabels[i])
        axs[1].scatter(z, s_j[:, i] ** 2 / rho, s=sz, color=c, edgecolors="none")
        axs[2].scatter(z, ds_j_dt[:, i], s=sz, color=c, edgecolors="none")

    # axs[0].scatter(y,estimated_rho)
    axs[0].set_ylabel(r"$\rho$")
    axs[0].set_xlabel(r"$z$")
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e-20, 1e-9)
    # axs[0].set_ylim(1e-12, 10**2)

    axs[1].set_ylabel(r"$\epsilon_j$")
    axs[1].set_xlabel(r"$z$")
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-12, 1)

    axs[2].set_ylabel(r"$\frac{d s_j}{dt}$")
    axs[2].set_xlabel(r"$z$")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.83, 0.5),
        fontsize=8,
    )

    os.makedirs(f"mono_{'coag' if do_coag else 'mono'}", exist_ok=True)
    plt.savefig(f"mono_{'coag' if do_coag else 'mono'}/{j}.png")
    # model.do_vtk_dump(f"dump_stratif_{j}.vtk", True)
    plt.close()
