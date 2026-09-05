"""
Dusty SPH disc
==============

A disc with dust
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import erfinv
from shamrock.external import coala
from shamrock.utils.DustMRNDistribution import DustMRNDistribution
from shamrock.utils.numba_helper import maybe_njit
from shamrock.utils.SimulationRunner import SimulationRunner, callback, simulation_setup

try:
    import matplotlib
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

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

codeu_kg_m3 = codeu.get("kg") * codeu.get("m", power=-3)

# CLI Parameters
Npart = int(float(os.environ.get("NPART", "1e5")))
ndust = int(os.environ.get("NDUST", 0))
use_coala = os.environ.get("COALA", "False") == "True"

if shamrock.sys.world_rank() == 0:
    print("-" * 60)
    print("Simulation paramters:")
    params = [("Npart", Npart), ("ndust", ndust), ("use_coala", use_coala)]
    name_w = max(len(name) for name, _ in params)
    val_w = max(len(str(val)) for _, val in params)
    sep = "+-" + "-" * name_w + "-+-" + "-" * val_w + "-+"
    print(sep)
    print(f"| {'param':<{name_w}} | {'value':<{val_w}} |")
    print(sep)
    for name, val in params:
        print(f"| {name:<{name_w}} | {val!s:<{val_w}} |")
    print(sep)
    print("-" * 60)

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Dump and plot frequency and duration of the simulation
dump_freq_stop = 2
plot_freq_stop = 1

dt_stop = 10
dt_stop_fast = 1

# Sink parameters
center_mass = 1.0
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

# Viscosity parameter
alpha_AV = 1.0e-3 / 0.08
alpha_u = 1.0
beta_AV = 2.0

# Dust parameters
kernel = "M6"
gamma = 1.4
t_inject = 5.0

if ndust > 0:
    mrn_pow = 3.5
    mrn_cutoff_si = 250e-9  # would be 250e-9 normally

    epsilon_base = 0.01

    rho_grains_si_edges = np.array([2.3 * 1000 for _ in range(ndust + 1)])  # 2.3 g.cm^-3
    grain_size_si_edges = np.logspace(-9, -2, ndust + 1)  # 10um -> 1mm

    mrn_distribution = DustMRNDistribution(
        codeu, mrn_pow, mrn_cutoff_si, grain_size_si_edges, rho_grains_si_edges
    )

if ndust > 0 and use_coala is True:
    dv_max = 1000000 * codeu.get("m") / codeu.get("s")
    Q = 5
    rhodust_eps = 1e-17
    K0_multiplier = 1

# Integrator parameters
C_cour = 0.1
C_force = 0.1

sim_folder = f"_to_trash/circular_dustydisc_{ndust}_{Npart}_{kernel}_coala_{use_coala}/"

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

if ndust > 0:
    grain_size = mrn_distribution.grain_size
    grain_size_si = mrn_distribution.grain_size_si
    rho_grains = mrn_distribution.rho_grains
    massgrid_edges = mrn_distribution.massgrid_edges
    mrn_weight = mrn_distribution.mrn_weight


if ndust > 0 and use_coala is True:
    K0 = np.pi * ((4.0 / 3.0) * np.pi * mrn_distribution.rho_grains[0]) ** (-2.0 / 3.0)
    K0 *= K0_multiplier
    print(f"K0 = {K0}")

    tabflux_coag = coala.coala_precalc_tabflux_coag(K0, ndust, Q, mrn_distribution.massgrid_edges)

# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)


def compute_sj_new_j(patchdata, j):
    pmass = model.get_particle_mass()

    hpart = patchdata["hpart"]
    rho = pmass * (model.get_hfact() / np.array(hpart)) ** 3

    epsilon_target = epsilon_base * mrn_weight[j]
    s = np.sqrt(rho * epsilon_target)

    return s


def setup_model():
    global disc_mass

    # Generate the default config
    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)

    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )

    cfg.set_eos_locally_isothermalLP07(cs0=disc.cs0(), q=disc.q, r0=disc.r0)

    if ndust > 0:
        cfg.set_dust_mode_monofluid_tva(
            nvar=ndust, cfl_density_threshold=1e-22 * codeu_kg_m3, clamp_dust_frac=0.95
        )
        cfg.set_dust_drag_epstein(gamma, grain_size, rho_grains)
        if use_coala:
            cfg.set_dust_evol_coala_coag(rhodust_eps, dv_max, massgrid_edges, tabflux_coag)

    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box

    cfg.set_units(codeu)
    cfg.set_particle_mass(disc.part_mass(Npart))
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)
    cfg.set_show_cfl_detail(True)

    # On a chaotic disc, we disable to two stage search to avoid giant leaves
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
# Prepare the simulation class


class Simulation(SimulationRunner):
    # Use the global vars defined at the top of the file
    t_end = np.inf
    dump_prefix = dump_prefix

    analysis_modules = []
    analysis_modules_fast = []

    @callback(at_tsim=[t_inject])
    def inject_dust(self, _):
        for k in range(ndust):

            def compute_sj_new(patchdata):
                return compute_sj_new_j(patchdata, k)

            self.model.overwrite_field_value_f64("s_j", compute_sj_new, k)

        self.model.set_dt(0.0)  # to help the corrector on next step after adding dust

    def ana_module_run(self, a, ianalysis):
        a.analysis_save(ianalysis)

        if hasattr(a, "make_plot"):
            a.make_plot(
                ianalysis,
                **a.render_args,
            )

        if hasattr(a, "plot_perf_history"):
            a.plot_perf_history(close_plots=True)

        if hasattr(a, "plot_history"):
            a.plot_history(close_plots=True)

    @callback(tsim_interval=dt_stop)  # Do the analysis every dt_stop
    def analysis(self, ianalysis):
        for a in self.analysis_modules:
            self.ana_module_run(a, ianalysis)

    @callback(tsim_interval=dt_stop_fast)  # Do the analysis every dt_stop
    def analysis_fast(self, ianalysis):
        for a in self.analysis_modules_fast:
            self.ana_module_run(a, ianalysis)

        self.model.do_vtk_dump(self.dump_prefix + f"{ianalysis:07}" + ".vtk", True)

    @callback(walltime_interval=30)  # Checkpoint the simulation every 10 minutes
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)

    @simulation_setup
    def setup(self):
        setup_model()


sim = Simulation(model)


from shamrock.utils.analysis import (
    MassAnalysis,
    PerfHistory,
)
from shamrock.utils.analysis.compute_field_dust import compute_s_mean_field

perf_analysis = PerfHistory(model, analysis_folder, "perf_history")
sim.analysis_modules_fast.append(perf_analysis)

mass_analysis = MassAnalysis(model, analysis_folder, "mass_history")
sim.analysis_modules_fast.append(mass_analysis)

sim.run()
