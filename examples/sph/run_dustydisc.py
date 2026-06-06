"""
Dusty SPH disc
========================

Perform a dust settling test in a local stratified box.
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import erfinv

import shamrock

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Sim parameters
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),  # year
    unit_length=sicte.au(),  # astro unit
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)


# Resolution
Npart = 100000

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Dump and plot frequency and duration of the simulation
dump_freq_stop = 2
plot_freq_stop = 1

dt_stop = 0.1
nstop = 30

# The list of times at which the simulation will pause for analysis / dumping
t_stop = [i * dt_stop for i in range(nstop + 1)]


# Sink parameters
center_mass = 1.0
center_racc = 0.8

# Disc parameter
disc_mass = 0.01  # sol mass
rout = 10.0  # au
rin = 1.0  # au
H_r_0 = 0.1
q = 0.5
p = 3.0 / 2.0
r0 = 1.0

# Viscosity parameter
alpha_AV = 1.0e-3 / 0.08
alpha_u = 1.0
beta_AV = 2.0

# Dust parameters
ndust = 5
mrn_pow = 3.5
mrn_cutoff_si = np.inf  # would be 250e-9 normally

epsilon_base = 0.01

rho_grains_si_edges = np.array([2.3 * 1000 for _ in range(ndust + 1)])  # 2.3 g.cm^-3
grain_size_si_edges = np.logspace(-6, -2, ndust + 1)  # 10um -> 1mm

# Integrator parameters
C_cour = 0.1
C_force = 0.1

sim_folder = f"_to_trash/circular_dustydisc_{Npart}/"

dump_folder = sim_folder + "dump/"
analysis_folder = sim_folder + "analysis/"
plot_folder = analysis_folder + "plots/"

dump_prefix = dump_folder + "dump_"

# Physical constants
G = ucte.G()


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

bsize = rout * 2
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)

cs0 = cs_profile(r0)


def rot_profile(r):
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3. # factor taken from phantom, to fasten thermalizing
    fact = 1.0
    return fact * H


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

print(f"grains sizes = {grain_size_si} [m]")
print(f"grains dens  = {rho_grains_si} [kg.m^-3]")

print(f"grains sizes = {grain_size} [code units]")
print(f"grains dens  = {rho_grains} [code units]")

# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")


dump_helper = shamrock.utils.dump.ShamrockDumpHandleHelper(model, dump_prefix)

# %%
# Load the last dump if it exists, setup otherwise

mrn_weight = grain_size ** (4 - mrn_pow)
mrn_weight *= grain_size_si < mrn_cutoff_si
mrn_weight = mrn_weight / np.sum(mrn_weight)

print(f"mrn_weight = {mrn_weight}")


def compute_sj_new_j(patchdata, j):
    global pmass

    hpart = patchdata["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    epsilon_target = epsilon_base * mrn_weight[j]
    print(f"epsilon_target = {epsilon_target} {j}")
    s = np.sqrt(rho * epsilon_target)

    print(
        f"s = {s} {np.isnan(s).any()} epsilon_target = {epsilon_target} mrn_weight = {mrn_weight[j]}, rho = {rho}"
    )

    return s


def setup_model():
    global disc_mass

    # Generate the default config
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)

    # cfg.set_artif_viscosity_VaryingCD10(
    #    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    # )

    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)

    cfg.set_dust_mode_monofluid_tvi(nvar=ndust)
    cfg.set_dust_drag_epstein(grain_size, rho_grains)

    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box

    cfg.set_units(codeu)
    cfg.set_particle_mass(pmass)
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)

    # Enable this to debug the neighbor counts
    # cfg.set_show_neigh_stats(True)

    # Standard way to set the smoothing length (e.g. Price et al. 2018)
    cfg.set_smoothing_length_density_based()

    cfg.set_save_dt_to_fields(True)

    # Standard density based smoothing length but with a neighbor count limit
    # Use it if you have large slowdowns due to giant particles
    # I recommend to use it if you have a circumbinary discs as the issue is very likely to happen
    # cfg.set_smoothing_length_density_based_neigh_lim(500)

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

    # Correct the barycenter before adding the sink
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
    # Smoothing length iterations, increasing it affect the performance negatively but increase the
    # convergence rate of the smoothing length
    # this is why we increase it temporely to 1.3 before lowering it back to 1.1 (default value)
    # Note that both ``change_htolerances`` can be removed and it will work the same but would converge
    # more slowly at the first timestep

    model.change_htolerances(coarse=1.3, fine=1.1)
    model.timestep()
    model.change_htolerances(coarse=1.1, fine=1.1)

    # Add the dust
    for k in range(ndust):

        def compute_sj_new(patchdata):
            return compute_sj_new_j(patchdata, k)

        model.overwrite_field_value_f64("s_j", compute_sj_new, k)

    model.set_dt(0.0)  # to help the corrector on next step after adding dust


dump_helper.load_last_dump_or(setup_model)

from shamrock.utils.analysis import (
    AnalysisHelper,
    ColumnDensityPlot,
    ColumnDensityPlotDust,
    ColumnParticleCount,
    PerfHistory,
    SliceDensityPlot,
    SliceDensityPlotDust,
    SliceDiffVthetaProfile,
    SliceDtPart,
    SliceVzPlot,
    StandardPlotHelper,
    VerticalShearGradient,
    get_epsilon_getter,
)

face_on_render_kwargs = {
    "x_unit": "au",
    "y_unit": "au",
    "time_unit": "year",
    "x_label": "x",
    "y_label": "y",
}

sink_params = {
    "sink_scale_factor": 1,
    "sink_color": "green",
    "sink_linewidth": 1,
    "sink_fill": False,
}

perf_analysis = PerfHistory(model, analysis_folder, "perf_history")

column_density_plot = ColumnDensityPlot(
    model,
    ext_r=rout * 1.5,
    nx=1024,
    ny=1024,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="rho_integ_gas",
)

column_density_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": "kg.m^-2",
    "field_label": "$\\int \\rho \\, \\mathrm{{d}} z$",
    "vmin": 1e-2,
    "vmax": 1e4,
    "norm": "log",
    **sink_params,
    "extra_title": "[gas + dust]",
}

dust_column_density_plot = []

for jdust in range(ndust):
    dust_column_density_plot.append(
        ColumnDensityPlotDust(
            model,
            ext_r=rout * 1.5,
            nx=1024,
            ny=1024,
            ex=(1, 0, 0),
            ey=(0, 1, 0),
            center=(0, 0, 0),
            ndust=ndust,
            jdust=jdust,
            analysis_folder=analysis_folder,
            analysis_prefix=f"rho_integ_dust_{jdust}",
        )
    )

    fact = epsilon_base * mrn_weight[jdust]
    dust_column_density_plot[jdust].render_args = {
        **column_density_plot.render_args,
        "field_unit": "kg.m^-2",
        "field_label": f"$\\int \\rho_{{d, {jdust} }} \\, \\mathrm{{d}} z$",
        "vmin": fact * 1e-2,
        "vmax": fact * 1e4,
        "norm": "log",
        **sink_params,
        "extra_title": f"[$s_{{grain}}$ = {grain_size_si[jdust]:.2e} m]",
    }

vertical_density_plot = SliceDensityPlot(
    model,
    ext_r=rout * 1.1 / (16.0 / 9.0),  # aspect ratio of 16:9
    nx=1920,
    ny=1080,
    ex=(1, 0, 0),
    ey=(0, 0, 1),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="rho_slice_gas",
)

vertical_density_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": "kg.m^-3",
    "field_label": "$\\rho$",
    "vmin": 1e-12,
    "vmax": 1e-6,
    "norm": "log",
    **sink_params,
    "extra_title": "[gas + dust]",
}

dust_slice_density_plot = []

for jdust in range(ndust):
    dust_slice_density_plot.append(
        SliceDensityPlotDust(
            model,
            ext_r=rout * 1.1 / (16.0 / 9.0),  # aspect ratio of 16:9
            nx=1920,
            ny=1080,
            ex=(1, 0, 0),
            ey=(0, 0, 1),
            center=(0, 0, 0),
            ndust=ndust,
            jdust=jdust,
            analysis_folder=analysis_folder,
            analysis_prefix=f"rho_slice_dust_{jdust}",
        )
    )
    fact = epsilon_base * mrn_weight[jdust]

    dust_slice_density_plot[jdust].render_args = {
        **vertical_density_plot.render_args,
        "field_unit": "kg.m^-3",
        "field_label": f"$\\rho_{{d, {jdust} }}$",
        "vmin": fact * 1e-12,
        "vmax": fact * 1e-6,
        "norm": "log",
        **sink_params,
        "extra_title": f"[$s_{{grain}}$ = {grain_size_si[jdust]:.2e} m]",
    }


dust_slice_epsilon_plot = []

for jdust in range(ndust):

    def compute_epsilon_integ(helper, internal_jdust=jdust):
        return helper.slice_render(
            "custom",
            "f64",
            do_normalization=True,
            min_normalization=1e-9,
            custom_getter=get_epsilon_getter(model, internal_jdust, ndust),
        )

    dust_slice_epsilon_plot.append(
        StandardPlotHelper(
            model,
            ext_r=rout * 1.1 / (16.0 / 9.0),  # aspect ratio of 16:9
            nx=1920,
            ny=1080,
            ex=(1, 0, 0),
            ey=(0, 0, 1),
            center=(0, 0, 0),
            analysis_folder=analysis_folder,
            analysis_prefix=f"epsilon_slice_dust_{jdust}",
            compute_function=compute_epsilon_integ,
        )
    )

    dust_slice_epsilon_plot[jdust].render_args = {
        **vertical_density_plot.render_args,
        "field_unit": None,
        "field_label": f"$\\epsilon_{{d, {jdust} }}$",
        "vmin": 1e-6,
        "vmax": 1e-1,
        "norm": "log",
        **sink_params,
        "extra_title": f"[$s_{{grain}}$ = {grain_size_si[jdust]:.2e} m]",
    }


v_z_slice_plot = SliceVzPlot(
    model,
    ext_r=rout * 1.1 / (16.0 / 9.0),  # aspect ratio of 16:9
    nx=1920,
    ny=1080,
    ex=(1, 0, 0),
    ey=(0, 0, 1),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="v_z_slice",
    do_normalization=True,
)

v_z_slice_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": "m.s^-1",
    "field_label": "$\\mathrm{v}_z$",
    "cmap": "seismic",
    "cmap_bad_color": "white",
    "vmin": -300,
    "vmax": 300,
    **sink_params,
}

dt_part_slice_plot = SliceDtPart(
    model,
    ext_r=rout * 0.5 / (16.0 / 9.0),  # aspect ratio of 16:9
    nx=1920,
    ny=1080,
    ex=(1, 0, 0),
    ey=(0, 0, 1),
    center=((rin + rout) / 2, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="dt_part_slice",
)

dt_part_slice_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": "year",
    "field_label": "$\\Delta t$",
    "vmin": 1e-4,
    "vmax": 1,
    "norm": "log",
    "contour_list": [1e-4, 1e-3, 1e-2, 1e-1, 1],
    **sink_params,
}

column_particle_count_plot = ColumnParticleCount(
    model,
    ext_r=rout * 1.5,
    nx=1024,
    ny=1024,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="particle_count",
)

column_particle_count_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": None,
    "field_label": "$\\int \\frac{1}{h_\\mathrm{part}} \\, \\mathrm{{d}} z$",
    "vmin": 1,
    "vmax": 1e2,
    "norm": "log",
    "contour_list": [1, 10, 100, 1000],
    **sink_params,
}


def analysis(ianalysis):

    column_density_plot.analysis_save(ianalysis)
    vertical_density_plot.analysis_save(ianalysis)
    v_z_slice_plot.analysis_save(ianalysis)
    dt_part_slice_plot.analysis_save(ianalysis)
    column_particle_count_plot.analysis_save(ianalysis)

    for p in dust_column_density_plot:
        p.analysis_save(ianalysis)

    for p in dust_slice_density_plot:
        p.analysis_save(ianalysis)

    for p in dust_slice_epsilon_plot:
        p.analysis_save(ianalysis)

    perf_analysis.analysis_save(ianalysis)


def render_analysis(iplot):

    column_density_plot.make_plot(
        iplot,
        **column_density_plot.render_args,
    )

    for jdust, p in enumerate(dust_column_density_plot):
        p.make_plot(
            iplot,
            **p.render_args,
        )

    vertical_density_plot.make_plot(
        iplot,
        **vertical_density_plot.render_args,
    )

    for jdust, p in enumerate(dust_slice_density_plot):
        p.make_plot(
            iplot,
            **p.render_args,
        )

    for jdust, p in enumerate(dust_slice_epsilon_plot):
        p.make_plot(
            iplot,
            **p.render_args,
        )

    v_z_slice_plot.make_plot(
        iplot,
        **v_z_slice_plot.render_args,
    )

    dt_part_slice_plot.make_plot(
        iplot,
        **dt_part_slice_plot.render_args,
    )

    column_particle_count_plot.make_plot(
        iplot,
        **column_particle_count_plot.render_args,
    )


# %%
# Evolve the simulation
model.solver_logs_reset_cumulated_step_time()
model.solver_logs_reset_step_count()

t_start = model.get_time()

idump = 0
iplot = 0
istop = 0
for ttarg in t_stop:
    if ttarg >= t_start:
        model.evolve_until(ttarg)

        if istop % dump_freq_stop == 0:
            dump_helper.write_dump(idump, purge_old_dumps=True, keep_first=1, keep_last=3)

        if istop % plot_freq_stop == 0:
            analysis(iplot)
            render_analysis(iplot)

    if istop % dump_freq_stop == 0:
        idump += 1

    if istop % plot_freq_stop == 0:
        iplot += 1

    istop += 1

# %%
# Plot generation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Load the on-the-fly analysis after the run to make the plots
# (everything in this section can be in another file)
import matplotlib

column_density_plot.render_all(
    **column_density_plot.render_args,
)

for jdust, p in enumerate(dust_column_density_plot):
    p.render_all(
        **p.render_args,
    )


vertical_density_plot.render_all(
    **vertical_density_plot.render_args,
)

for jdust, p in enumerate(dust_slice_density_plot):
    p.render_all(
        **p.render_args,
    )

for jdust, p in enumerate(dust_slice_epsilon_plot):
    p.render_all(
        **p.render_args,
    )

v_z_slice_plot.render_all(
    **v_z_slice_plot.render_args,
)


dt_part_slice_plot.render_all(
    **dt_part_slice_plot.render_args,
)

column_particle_count_plot.render_all(
    **column_particle_count_plot.render_args,
)


# %%
# Plot the performance history (Switch close_plots to True if doing a long run)
perf_analysis.plot_perf_history(close_plots=False)
plt.show()
