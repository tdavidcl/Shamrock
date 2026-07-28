"""
Dusty SPH disc
========================

A disc but with dust
"""

import os

import numpy as np
from shamrock.utils.analysis import StandardPlotHelper
from shamrock.utils.DustMRNDistribution import DustMRNDistribution
from shamrock.utils.SimulationRunner import SimulationRunner, callback, simulation_setup

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
analysis_frequency = 1.0  # years
t_end = 1000.0  # years

# Sink parameters
center_mass = 1.0  # sol mass
center_racc = 8.0  # au

# Disc parameters
disc = shamrock.utils.disc_setup.StandardDisc(
    units=codeu,
    center_mass=center_mass,
    disc_mass=0.05,  # sol mass
    rin=10.0,  # au
    rout=150.0,  # au
    H_r_0=0.1,
    q=0.5,
    p=3.0 / 2.0,
    r0=10.0,
    rotation="subkeplerian_3d",
    inner_tapering=True,
)

# Solver parameters
kernel = "M6"

# Hydro parameters
alpha_min = 0.0
alpha_max = 1
sigma_decay = 0.1
alpha_u = 1
beta_AV = 2

# Dust parameters
ndust = int(os.environ.get("NDUST", 5))
gamma = 1.4

print(f"ndust = {ndust}")

mrn_pow = 3.5
mrn_cutoff_si = np.inf  # would be 250e-9 normally

epsilon_base = 0.01

rho_grains_si_edges = np.array([2.3 * 1000 for _ in range(ndust + 1)])  # 2.3 g.cm^-3
grain_size_si_edges = np.logspace(-9, -2, ndust + 1)  # 10um -> 1mm

mrn_distribution = DustMRNDistribution(
    codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges
)


# Integrator parameters
C_cour = 0.1
C_force = 0.1

sim_folder = f"_to_trash/circular_dustydisc_{ndust}_{Npart}_{kernel}/"

dump_folder = sim_folder + "dump/"
analysis_folder = sim_folder + "analysis/"
plot_folder = analysis_folder + "plots/"

dump_prefix = dump_folder + "dump_"


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

bsize = disc.rout * 2
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)
profiles = disc.get_profiles()

grain_size = mrn_distribution.grain_size
grain_size_si = mrn_distribution.grain_size_si
rho_grains = mrn_distribution.rho_grains
massgrid_edges = mrn_distribution.massgrid_edges
mrn_weight = mrn_distribution.mrn_weight


# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)


class Simulation(SimulationRunner):
    # Use the global vars defined at the top of the file
    t_end = t_end
    dump_prefix = dump_prefix

    def compute_sj_new_j(self, patchdata, j):
        pmass = model.get_particle_mass()

        hpart = patchdata["hpart"]
        rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

        epsilon_target = epsilon_base * mrn_weight[j]
        print(f"epsilon_target = {epsilon_target} {j}")
        s = np.sqrt(rho * epsilon_target)

        print(
            f"s = {s} {np.isnan(s).any()} epsilon_target = {epsilon_target} mrn_weight = {mrn_weight[j]}, rho = {rho}"
        )

        return s

    @callback(at_tsim=0.0)
    def inject_dust(self, _):
        # Add the dust
        for k in range(ndust):

            def compute_sj_new(patchdata):
                return self.compute_sj_new_j(patchdata, k)

            model.overwrite_field_value_f64("s_j", compute_sj_new, k)

        model.set_dt(0.0)  # to help the corrector on next step after adding dust

    @callback(tsim_interval=analysis_frequency)  # Do the analysis every analysis_frequency
    def analysis(self, ianalysis):
        # Run all the analysis modules (declared below)
        for a in self.analysis_modules:
            a.analysis_save(ianalysis)

            if hasattr(a, "make_plot"):
                a.make_plot(
                    ianalysis,
                    **a.render_args,
                )
            elif hasattr(a, "plot_perf_history"):
                a.plot_perf_history(close_plots=True)

    @callback(walltime_interval=10 * 60)  # Checkpoint the simulation every 10 minutes
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)

    @simulation_setup
    def setup(self):
        cfg = model.gen_default_config()

        cfg.set_artif_viscosity_VaryingCD10(
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_decay=sigma_decay,
            alpha_u=alpha_u,
            beta_AV=beta_AV,
        )

        cfg.set_eos_locally_isothermalLP07(cs0=disc.cs0(), q=disc.q, r0=disc.r0)

        if ndust > 0:
            cfg.set_dust_mode_monofluid_tva(nvar=ndust)
            cfg.set_dust_drag_epstein(gamma, grain_size, rho_grains)

        cfg.add_kill_sphere(
            center=(0, 0, 0), radius=bsize
        )  # kill particles outside the simulation box

        cfg.set_units(codeu)
        cfg.set_particle_mass(disc.part_mass(Npart))

        print(C_cour, C_force)

        # Set the CFL
        cfg.set_cfl_cour(C_cour)
        cfg.set_cfl_force(C_force)
        cfg.set_show_cfl_detail(True)

        cfg.set_smoothing_length_density_based_neigh_lim(500)

        # On a chaotic disc, we disable to two stage search to avoid giant leaves
        cfg.set_tree_reduction_level(6)
        cfg.set_two_stage_search(False)

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
        gen_disc = disc.make_generator(setup, Npart, random_seed=666)

        # Print the dot graph of the setup
        if shamrock.sys.world_rank() == 0:
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


# %%
# Custom Analysis modules


def get_rhod_j_getter(model, jdust, ndust):

    def int_getter(size: int, dic_out: dict) -> np.array:
        s_j = dic_out["s_j"].reshape(-1, ndust)
        return s_j[:, jdust] ** 2

    return int_getter


def ColumnDensityPlotDust(
    model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix, jdust, ndust
):
    def compute_rhod_integ(helper):
        return helper.column_integ_render(
            "custom", "f64", custom_getter=get_rhod_j_getter(model, jdust, ndust)
        )

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rhod_integ,
    )


def SliceDensityPlotDust(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    jdust,
    ndust,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_rho_slice(helper):
        return helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=get_rhod_j_getter(model, jdust, ndust),
        )

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rho_slice,
    )


# %%
# Define sim analysis

max_rho_plot = 1e-9
min_rho_plot = 1e-18
max_rho_integ_plot = 1e3
min_rho_integ_plot = 1e-5


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


slice_params = {
    "ext_r": disc.rout * 0.6 / (16.0 / 9.0),  # aspect ratio of 16:9
    "nx": 1920,
    "ny": 1080,
    "ex": (1, 0, 0),
    "ey": (0, 0, 1),
    "center": ((disc.rin + disc.rout) / 2, 0, 0),
}


from shamrock.utils.analysis import (
    ColumnDensityPlot,
    PerfHistory,
    SliceDensityPlot,
)

perf_analysis = PerfHistory(model, analysis_folder, "perf_history")


column_density_plot = ColumnDensityPlot(
    model,
    ext_r=disc.rout * 1.5,
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
    "vmin": min_rho_integ_plot,
    "vmax": max_rho_integ_plot,
    "norm": "log",
    **sink_params,
    "extra_title": "[gas + dust]",
}
if ndust > 0:
    dust_column_density_plot = []

    for jdust in range(ndust):
        dust_column_density_plot.append(
            ColumnDensityPlotDust(
                model,
                ext_r=disc.rout * 1.5,
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

        dust_column_density_plot[-1].render_args = {
            **column_density_plot.render_args,
            "field_unit": "kg.m^-2",
            "field_label": f"$\\int \\rho_{{d, {jdust} }} \\, \\mathrm{{d}} z$",
            "vmin": min_rho_integ_plot,
            "vmax": max_rho_integ_plot,
            "norm": "log",
            **sink_params,
            "extra_title": f"[$s_{{grain}}$ = {grain_size_si[jdust]:.2e} m]",
        }


vertical_density_plot = SliceDensityPlot(
    model,
    **slice_params,
    analysis_folder=analysis_folder,
    analysis_prefix="rho_slice_gas",
)

vertical_density_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": "kg.m^-3",
    "field_label": "$\\rho$",
    "vmin": min_rho_plot,
    "vmax": max_rho_plot,
    "norm": "log",
    **sink_params,
    "extra_title": "[gas + dust]",
}

if ndust > 0:
    dust_slice_density_plot = []

    for jdust in range(ndust):
        dust_slice_density_plot.append(
            SliceDensityPlotDust(
                model,
                **slice_params,
                ndust=ndust,
                jdust=jdust,
                analysis_folder=analysis_folder,
                analysis_prefix=f"rho_slice_dust_{jdust}",
            )
        )

        dust_slice_density_plot[-1].render_args = {
            **vertical_density_plot.render_args,
            "field_unit": "kg.m^-3",
            "field_label": f"$\\rho_{{d, {jdust} }}$",
            "vmin": min_rho_plot,
            "vmax": max_rho_plot,
            "norm": "log",
            **sink_params,
            "extra_title": f"[$s_{{grain}}$ = {grain_size_si[jdust]:.2e} m]",
        }

# %%
# Run the simulation

sim = Simulation(model)

sim.analysis_modules = [
    perf_analysis,
    column_density_plot,
    vertical_density_plot,
]

if ndust > 0:
    sim.analysis_modules.extend(dust_column_density_plot)
    sim.analysis_modules.extend(dust_slice_density_plot)

sim.run()
