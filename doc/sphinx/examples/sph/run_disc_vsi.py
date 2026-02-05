"""
Production run: Circular disc & central potential
=================================================

This example demonstrates how to run a smoothed particle hydrodynamics (SPH)
simulation of a circular disc orbiting around a central point mass potential.

The simulation models:

- A central star with a given mass and accretion radius
- A gaseous disc with specified mass, inner/outer radii, and vertical structure
- Artificial viscosity for angular momentum transport
- Locally isothermal equation of state

Also this simulation feature rolling dumps (see `purge_old_dumps` function) to save disk space.

This example is the accumulation of 3 files in a single one to showcase the complete workflow.

- The actual run script (runscript.py)
- Plot generation (make_plots.py)
- Animation from the plots (plot_to_gif.py)

On a cluster or laptop, one can run the code as follows:

.. code-block:: bash

    mpirun <your parameters> ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript runscript.py


then after the run is done (or while it is running), one can run the following to generate the plots:

.. code-block:: bash

    python make_plots.py


"""

# %%
# Runscript (runscript.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The runscript is the actual simulation with on the fly analysis & rolling dumps

import glob
import json
import os  # for makedirs

import numpy as np

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
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()

# %%
# List parameters

# Resolution
Npart = int(float(os.environ.get("NPART")))
Kernel = os.environ.get("KERNEL")
UpscaleFrom = os.environ.get("UPSCALE_FROM", None)
UpscaleFactor = int(os.environ.get("UPSCALE_FACTOR", 1))

if shamrock.sys.world_rank() == 0:
    print(f"Npart = {Npart}")
    print(f"Kernel = {Kernel}")
    print(f"UpscaleFrom = {UpscaleFrom}")
    print(f"UpscaleFactor = {UpscaleFactor}")


# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Dump and plot frequency and duration of the simulation
dump_freq_stop = 4
plot_freq_stop = 1

dt_stop = 0.25
nstop = 4*30

# The list of times at which the simulation will pause for analysis / dumping
t_stop = [i * dt_stop for i in range(nstop + 1)]


# Sink parameters
center_mass = 1.0
center_racc = 0.7

# Disc parameter
disc_mass = 0.001  # sol mass
rout = 10.0  # au
rin = 1.0  # au
H_r_0 = 0.1
q = 0.5
p = 3.0 / 2.0
r0 = 1.0

# Integrator parameters
C_cour = 0.3
C_force = 0.25

sim_folder = f"_to_trash/disc_vsi_{Npart}_{Kernel}/"

dump_folder = sim_folder + "dump/"
analysis_folder = sim_folder + "analysis/"
plot_folder = analysis_folder + "plots/"

dump_prefix = dump_folder + "dump_"


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
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)
    os.makedirs(dump_folder, exist_ok=True)
    os.makedirs(analysis_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

# %%
# Utility functions and quantities deduced from the base one

# Deduced quantities
pmass = disc_mass / Npart

bsize = rout * 1.3
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)

cs0 = cs_profile(r0)


def rot_profile(r,z):
    term_vk = r**3 / (r**2 + z**2)**(3./2.)
    return ((kep_profile(r) ** 2)*term_vk - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


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
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=Kernel)


# %%
# Dump handling


def get_vtk_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".vtk"


def get_ph_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".phdump"


dump_helper = shamrock.utils.dump.ShamrockDumpHandleHelper(model, dump_prefix)

# %%
# Load the last dump if it exists, setup otherwise


def setup_model():
    global disc_mass

    # Generate the default config
    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)

    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box

    cfg.set_units(codeu)
    cfg.set_particle_mass(pmass)
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)

    cfg.set_tree_reduction_level(6)
    cfg.set_two_stage_search(False)
    cfg.set_smoothing_length_density_based_neigh_lim(500)

    cfg.set_save_dt_to_fields(True)

    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)

    # Print the solver config
    model.get_current_config().print_status()

    # Init the scheduler & fields
    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    # Set the simulation box size
    model.resize_simulation_box(bmin, bmax)

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

    # Apply the setup
    setup.apply_setup(gen_disc)

    # correct the momentum and barycenter of the disc to 0
    analysis_momentum = shamrock.model_sph.analysisTotalMomentum(model=model)
    total_momentum = analysis_momentum.get_total_momentum()

    if shamrock.sys.world_rank() == 0:
        print(f"disc momentum = {total_momentum}")

    model.apply_momentum_offset((-total_momentum[0], -total_momentum[1], -total_momentum[2]))

    # Correct the barycenter
    analysis_barycenter = shamrock.model_sph.analysisBarycenter(model=model)
    barycenter, disc_mass = analysis_barycenter.get_barycenter()

    if shamrock.sys.world_rank() == 0:
        print(f"disc barycenter = {barycenter}")

    model.apply_position_offset((-barycenter[0], -barycenter[1], -barycenter[2]))

    total_momentum = shamrock.model_sph.analysisTotalMomentum(model=model).get_total_momentum()

    if shamrock.sys.world_rank() == 0:
        print(f"disc momentum after correction = {total_momentum}")

    barycenter, disc_mass = shamrock.model_sph.analysisBarycenter(model=model).get_barycenter()

    if shamrock.sys.world_rank() == 0:
        print(f"disc barycenter after correction = {barycenter}")

    if not np.allclose(total_momentum, 0.0):
        raise RuntimeError("disc momentum is not 0")
    if not np.allclose(barycenter, 0.0):
        raise RuntimeError("disc barycenter is not 0")

    # now that the barycenter & momentum are 0, we can add the sink
    model.add_sink(center_mass, (0, 0, 0), (0, 0, 0), center_racc)


    # Run a single step to init the integrator and smoothing length of the particles
    # Here the htolerance is the maximum factor of evolution of the smoothing length in each
    # Smoothing length iterations, increasing it affect the performance negatively but increse the
    # convergence rate of the smoothing length
    # this is why we increase it temporely to 1.3 before lowering it back to 1.1 (default value)
    # Note that both ``change_htolerances`` can be removed and it will work the same but would converge
    # more slowly at the first timestep

    model.change_htolerances(coarse=1.3, fine=1.1)
    model.timestep()
    model.change_htolerances(coarse=1.1, fine=1.1)


def setup_upscale():
    source_path = UpscaleFrom

    # here we can dump and load it into another context i we want like so
    ctx_data_source = shamrock.Context()
    ctx_data_source.pdata_layout_new()
    model_data_source = shamrock.get_Model_SPH(
        context=ctx_data_source, vector_type="f64_3", sph_kernel=Kernel
    )
    model_data_source.load_from_dump(source_path)

    # trigger rebalancing
    model_data_source.set_dt(0.0)
    model_data_source.timestep()

    # reset dt to 0 for the init of the next simulation
    model_data_source.set_dt(0.0)

    cfg = model_data_source.get_current_config()
    cfg.print_status()

    model.set_solver_config(cfg)
    model.init_scheduler(scheduler_split_val, scheduler_merge_val)
    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_from_context(ctx_data_source)
    split_part = setup.make_modifier_split_part(parent=gen, n_split=UpscaleFactor, seed=42,h_scaling=0.5)
    setup.apply_setup(split_part, insert_step=scheduler_split_val)

    print(model_data_source.get_sinks())
    for s in model_data_source.get_sinks():
        model.add_sink(s["mass"], s["pos"], s["velocity"], s["accretion_radius"])

    model.change_htolerances(coarse=1.3, fine=1.1)
    model.timestep()
    model.change_htolerances(coarse=1.1, fine=1.1)


if UpscaleFrom is not None:
    dump_helper.load_last_dump_or(setup_upscale)
else:
    dump_helper.load_last_dump_or(setup_model)


# %%
# On the fly analysis
from shamrock.utils.analysis import (
    ColumnAverageAngularMomentumTransportCoefficientPlot,
    ColumnAverageVzPlot,
    ColumnDensityPlot,
    ColumnParticleCount,
    PerfHistory,
    SliceAlphaAV,
    SliceAngularMomentumTransportCoefficientPlot,
    SliceDensityPlot,
    SliceDiffVthetaProfile,
    SliceDtPart,
    SliceVzPlot,
    VerticalShearGradient,
)

param_slice_kwargs = {
    "ext_r": rout * 0.6 / (16.0 / 9.0),  # aspect ratio of 16:9
    "nx": 1920,
    "ny": 1080,
    "ex": (1, 0, 0),
    "ey": (0, 0, 1),
    "center": ((rin + rout) / 2, 0, 0),
}

param_column_kwargs = {
    "ext_r": rout * 1.05,
    "nx": 1024,
    "ny": 1024,
    "ex": (1, 0, 0),
    "ey": (0, 1, 0),
    "center": (0, 0, 0),
}

perf_analysis = PerfHistory(model, analysis_folder, "perf_history")

column_density_plot = ColumnDensityPlot(
    model,
    **param_column_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="rho_integ_normal",
)

vertical_density_plot = SliceDensityPlot(
    model,
    **param_slice_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="rho_slice",
)

v_z_slice_plot = SliceVzPlot(
    model,
    **param_slice_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="v_z_slice",
    do_normalization=True,
    div_by_cs=True,
)

v_z_column_plot = ColumnAverageVzPlot(
    model,
    **param_column_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="v_z_column",
    div_by_cs=True,
)

relative_azy_velocity_slice_plot = SliceDiffVthetaProfile(
    model,
    **param_slice_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="relative_azy_velocity_slice",
    velocity_profile=kep_profile,
    div_by_cs=True,
    do_normalization=True,
    min_normalization=1e-9,
)

vertical_shear_gradient_slice_plot = VerticalShearGradient(
    model,
    **param_slice_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="vertical_shear_gradient_slice",
    do_normalization=True,
    min_normalization=1e-9,
)

dt_part_slice_plot = SliceDtPart(
    model,
    **param_slice_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="dt_part_slice",
)

alpha_av_slice_plot = SliceAlphaAV(
    model,
    **param_slice_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="alpha_av_slice",
)

column_particle_count_plot = ColumnParticleCount(
    model,
    **param_column_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="particle_count",
)

angular_momentum_transport_coefficient_slice_plot = SliceAngularMomentumTransportCoefficientPlot(
    model,
    **param_slice_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="angular_momentum_transport_coefficient_slice",
    velocity_profile=kep_profile,
)

angular_momentum_transport_coefficient_column_plot = ColumnAverageAngularMomentumTransportCoefficientPlot(
    model,
    **param_column_kwargs,
    analysis_folder=analysis_folder,
    analysis_prefix="angular_momentum_transport_coefficient_column",
    velocity_profile=kep_profile,
)



def make_plots(ianalysis):
    face_on_render_kwargs = {
        "x_unit": "au",
        "y_unit": "au",
        "time_unit": "year",
        "x_label": "x",
        "y_label": "y",
    }

    vz_delta = 0.2

    column_density_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="kg.m^-2",
        field_label="$\\int \\rho \\, \\mathrm{{d}} z$",
        vmin=1,
        vmax=1e4,
        norm="log",
    )

    vertical_density_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="kg.m^-3",
        field_label="$\\rho$",
        vmin=1e-12,
        vmax=1e-6,
        norm="log",
    )

    v_z_slice_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="unitless",
        field_label="$\\mathrm{v}_z / c_s$",
        cmap="seismic",
        cmap_bad_color="white",
        vmin=-vz_delta,
        vmax=vz_delta,
    )

    relative_azy_velocity_slice_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="unitless",
        field_label="$(\\mathrm{v}_{\\theta} - v_k) / c_s$",
        cmap="seismic",
        cmap_bad_color="white",
        vmin=-0.5,
        vmax=0.5,
    )

    vertical_shear_gradient_slice_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="yr^-1",
        field_label="${{\\partial R \\Omega}}/{{\\partial z}}$",
        cmap="seismic",
        cmap_bad_color="white",
        vmin=-1,
        vmax=1,
    )

    dt_part_slice_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="year",
        field_label="$\\Delta t$",
        vmin=1e-5,
        vmax=1,
        norm="log",
        contour_list=[1e-4, 1e-3, 1e-2, 1e-1, 1],
    )

    alpha_av_slice_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="unitless",
        field_label="$\\alpha_\\mathrm{AV}$",
        vmin=1e-6,
        vmax=1,
        norm="log",
        contour_list=[1e-4, 1e-3, 1e-2, 1e-1, 1],
    )

    column_particle_count_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit=None,
        field_label="$\\int \\frac{1}{h_\\mathrm{part}} \\, \\mathrm{{d}} z$",
        vmin=1,
        vmax=1e3,
        norm="log",
        contour_list=[1, 10, 100, 1000],
    )

    angular_momentum_transport_coefficient_slice_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="unitless",
        field_label="$\\alpha$",
        vmin=1e-6,
        vmax=1,
        norm="log",
        contour_list=[
            1e-4,
        ],
    )

    angular_momentum_transport_coefficient_column_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="unitless",
        field_label="$\\alpha$",
        vmin=1e-3,
        vmax=1,
        norm="log",
        contour_list=[1e-4, 1e-3, 1e-2, 1e-1, 1],
    )

    v_z_column_plot.make_plot(
        ianalysis,
        **face_on_render_kwargs,
        field_unit="unitless",
        field_label="$\\mathrm{v}_z / c_s$",
        cmap="seismic",
        cmap_bad_color="white",
        vmin=-vz_delta,
        vmax=vz_delta,
    )

    if perf_analysis.has_analysis():
        perf_analysis.plot_perf_history(close_plots=True)


def analysis(ianalysis):
    column_density_plot.analysis_save(ianalysis)
    vertical_density_plot.analysis_save(ianalysis)
    v_z_slice_plot.analysis_save(ianalysis)
    v_z_column_plot.analysis_save(ianalysis)
    relative_azy_velocity_slice_plot.analysis_save(ianalysis)
    vertical_shear_gradient_slice_plot.analysis_save(ianalysis)
    dt_part_slice_plot.analysis_save(ianalysis)
    alpha_av_slice_plot.analysis_save(ianalysis)
    column_particle_count_plot.analysis_save(ianalysis)
    angular_momentum_transport_coefficient_slice_plot.analysis_save(ianalysis)
    angular_momentum_transport_coefficient_column_plot.analysis_save(ianalysis)

    perf_analysis.analysis_save(ianalysis)

    make_plots(ianalysis)


# %%
# Evolve the simulation
model.solver_logs_reset_cumulated_step_time()
model.solver_logs_reset_step_count()

t_start = model.get_time()

iplot = 0
istop = 0
for ttarg in t_stop:
    if ttarg >= t_start:
        model.evolve_until(ttarg)

        if istop % dump_freq_stop == 0:
            model.do_vtk_dump(get_vtk_dump_name(istop), True)

        dump_helper.write_dump(istop, purge_old_dumps=False)

        if istop % plot_freq_stop == 0:
            analysis(iplot)

    if istop % plot_freq_stop == 0:
        iplot += 1

    istop += 1
