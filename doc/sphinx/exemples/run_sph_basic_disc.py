"""
Basic disc simulation
========================

This simple example shows how to run a basic disc simulation in SPH
"""

# sphinx_gallery_multi_image = "single"

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup units

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()


# %%
# List parameters

# Resolution
Npart = 1000000

# Sink parameters
center_mass = 1.0
center_racc = 0.1

# Disc parameter
disc_mass = 0.01  # sol mass
rout = 10.0  # au
rin = 1.0  # au
H_r_0 = 0.05
q = 0.5
p = 3.0 / 2.0
r0 = 1.0

# Viscosity parameter
alpha_AV = 1.0e-3 / 0.08
alpha_u = 1.0
beta_AV = 2.0

# Integrator parameters
C_cour = 0.3
C_force = 0.25


# Disc profiles
def sigma_profile(r):
    sigma_0 = 1.0  # We do not care as it will be renormalized
    return sigma_0 * (r / r0) ** (-p)


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_r_0 * r0) * omega_k(r0)
    return ((r / r0) ** (-q)) * cs_in


# %%
# Utility functions and quantities deduced from the base one

# Deduced quantities
pmass = disc_mass / Npart
bmin = (-rout * 2, -rout * 2, -rout * 2)
bmax = (rout * 2, rout * 2, rout * 2)

cs0 = cs_profile(r0)


def rot_profile(r):
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3. # factor taken from phantom, to fasten thermalizing
    fact = 1.0
    return fact * H


# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the data and configure it

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

# Generate the default config
cfg = model.gen_default_config()
# Use disc alpha model viscosity
cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
# use the Lodato Price 2007 equation of state
cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)
# Use the given code units
cfg.set_units(codeu)
# Change particle mass
cfg.set_particle_mass(pmass)
# Set the CFL
cfg.set_cfl_cour(C_cour)
cfg.set_cfl_force(C_force)

# Set the solver config to be the one stored in cfg
model.set_solver_config(cfg)

# Print the solver config
model.get_current_config().print_status()

# We want the patches to split above 10^8 part and merge if smaller than 1 part (e.g. disable patch)
model.init_scheduler(int(1e8), 1)

# Set the simulation box size
model.resize_simulation_box(bmin, bmax)

# %%
# Add the sink particle

# null position and velocity
model.add_sink(center_mass, (0, 0, 0), (0, 0, 0), center_racc)

# %%
# Create the setup

setup = model.get_setup()
gen_disc = setup.make_generator_disc_mc(
    part_mass=pmass,
    disc_mass=disc_mass,
    r_in=rin,
    r_out=rout,
    sigma_profile=sigma_profile,
    H_profile=H_profile,
    rot_profile=rot_profile,
    cs_profile=cs_profile,
    random_seed=666,
)

# Print the dot graph of the setup
print(gen_disc.get_dot())

# %%
# Apply the setup
setup.apply_setup(gen_disc)

# %%
# Run a single step to init the integrator and smoothing lenght of the particles
# Here the htolerance is the maximum factor of evolution of the smoothing lenght in each
# Smoothing lenght iterations, increasing it affect the performance negatively but increse the
# convergence rate of the smoothing lenght
# this is why we increase it temporely to 1.3 before lowering it back to 1.1 (default value)
# Note that both ``change_htolerance`` can be removed and it will work the same but would converge
# more slowly at the first timestep

model.change_htolerance(1.3)
model.timestep()
model.change_htolerance(1.1)

# %%
# Manipulating the simulation
# ------------------------
# Dump files (path relative to where you have started shamrock)

dump_folder = "_to_trash"
import os

os.system("mkdir -p " + dump_folder)

# VTK dump
model.do_vtk_dump(dump_folder + "/init_disc.vtk", True)

# Shamrock restart dump files
model.dump(dump_folder + "/init_disc.sham")

# Phantom dump
dump = model.make_phantom_dump()
dump.save_dump(dump_folder + "/init_disc.phdump")

# %%
# Single timestep
model.evolve_once()

# %%
# Evolve until a given time (code units)
model.evolve_until(0.001)

# %%
# Get the sinks positions
print(model.get_sinks())

# %%
# Get the fields as python dictionary of numpy arrays
#

# %%
# .. warning::
#     Do not do this on a large distributed simulation as this gather all data on MPI rank 0
#     and will use a lot of memory (and crash if the simulation is too large)
print(ctx.collect_data())

# %%
# Performing a timestep loop
dt_stop = 0.001
for i in range(10):
    t_target = i * dt_stop
    # skip if the model is already past the target
    if model.get_time() > t_target:
        continue

    model.evolve_until(i * dt_stop)

    # Dump name is "dump_xxxx.sham" where xxxx is the timestep
    model.dump(dump_folder + f"/dump_{i:04}.sham")

# %%
# Plot column integrated density
import matplotlib.pyplot as plt

pixel_x = 1200
pixel_y = 1080
radius = 5
center = (0.0, 0.0, 0.0)

aspect = pixel_x / pixel_y
pic_range = [-radius * aspect, radius * aspect, -radius, radius]
delta_x = (radius * 2 * aspect, 0.0, 0.0)
delta_y = (0.0, radius * 2, 0.0)

arr_rho = model.render_cartesian_column_integ(
    "rho", "f64", center=(0.0, 0.0, 0.0), delta_x=delta_x, delta_y=delta_y, nx=pixel_x, ny=pixel_y
)

import copy

import matplotlib

my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
my_cmap.set_bad(color="black")

fig_width = 6
fig_height = fig_width / aspect
plt.figure(figsize=(fig_width, fig_height))
res = plt.imshow(arr_rho, cmap=my_cmap, origin="lower", extent=pic_range, norm="log", vmin=1e-9)

cbar = plt.colorbar(res, extend="both")
cbar.set_label(r"$\int \rho \, \mathrm{d} z$ [code unit]")
# or r"$\rho$ [code unit]" for slices

plt.title("t = {:0.3f} [code unit]".format(model.get_time()))
plt.xlabel("x")
plt.ylabel("z")
plt.show()


# %%
# Plot vertical profiles at r=1
import numpy as np

dat = ctx.collect_data()

for rcenter in [1.0, 2.0, 3.0]:

    z = []
    h = []
    vz = []
    az = []

    delta_r = 0.01

    for i in range(len(dat["xyz"])):
        r = (dat["xyz"][i][0] ** 2 + dat["xyz"][i][1] ** 2) ** 0.5
        if r < rcenter + delta_r and r > rcenter - delta_r:
            z.append(dat["xyz"][i][2])
            h.append(dat["hpart"][i])
            vz.append(dat["vxyz"][i][2])
            az.append(dat["axyz"][i][2])

    rho = pmass * (model.get_hfact() / np.array(h)) ** 3

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)

    from scipy.optimize import curve_fit

    def func(x, a, c):
        return a * np.exp(-((x / c) ** 2) / 2)

    rho_0 = 0.001
    p0 = [rho_0, H_profile(rcenter)]  # a, b, c
    popt, pcov = curve_fit(func, z, rho, p0=p0)

    z_ana = np.linspace(-5.0 * H_profile(rcenter), 5.0 * H_profile(rcenter), 100)
    rho_fit = func(z_ana, *popt)

    axs[0].scatter(z, rho, label="rho")

    axs[0].plot(z_ana, rho_fit, c="black", label="gaussian fit")
    stddev = abs(popt[1])
    axs[0].annotate(
        f"Stddev: {stddev:.5f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="w"),
    )

    axs[0].set_ylabel("rho")
    axs[0].legend()

    axs[1].scatter(z, vz, label="vz")

    vz_fit = np.polyfit(z, vz, 1)
    vz_fit_fn = np.poly1d(vz_fit)
    axs[1].plot(z_ana, vz_fit_fn(z_ana), c="red", label="linear fit")

    axs[1].set_ylabel("vz")
    axs[1].legend()

    axs[2].scatter(z, az, label="az")

    az_fit = np.polyfit(z, az, 1)
    az_fit_fn = np.poly1d(az_fit)
    print(f"r={rcenter} az_fit={az_fit}")
    axs[2].plot(z_ana, az_fit_fn(z_ana), c="red", label="linear fit")

    axs[2].set_ylabel("az")
    axs[2].set_xlabel("z")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
