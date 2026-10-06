"""
Dusty SPH circumbinary disc
===========================

Same dusty disc as ``run_dustydisc.py``, but orbiting a central binary instead
of a single star.

The binary is set only by its mass ratio ``q = M2/M1`` and its eccentricity.
Its semi-major axis comes from the empirical cavity-size relation of
Ragusa et al. (2026): we ask for the binary that would carve a cavity whose
periastron is the inner edge of the disc.

The run is configured through environment variables: ``BINARY_Q`` (default 1),
``BINARY_E`` (default 0), ``NPART`` (default 1e5, scientific notation accepted),
``NDUST`` (default 0) and ``COALA`` (default False). ``SINGLE_STAR=True`` replaces
the binary by a single star of the same total mass, for comparison with the
binary runs (``BINARY_Q`` and ``BINARY_E`` are then ignored).
"""

# sphinx_gallery_multi_image = "single"

import json
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
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
ndust = int(os.environ.get("NDUST", "0"))
use_coala = os.environ.get("COALA", "False") == "True"
single_star = os.environ.get("SINGLE_STAR", "False") == "True"
binary_q = float(os.environ.get("BINARY_Q", "1.0"))  # mass ratio M2/M1
binary_e = float(os.environ.get("BINARY_E", "0.0"))  # eccentricity

if shamrock.sys.world_rank() == 0:
    print("-" * 60)
    print("Simulation paramters:")
    params = [
        ("Npart", Npart),
        ("ndust", ndust),
        ("use_coala", use_coala),
        ("single_star", single_star),
    ]
    if not single_star:
        params += [("binary_q", binary_q), ("binary_e", binary_e)]
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
dt_stop = 25
dt_stop_fast = 5

# Central binary (q and e are read from the environment above)
center_mass = 1.0  # total mass of the binary (sol mass)
# longitude of periastron (rad). With pi the periastron points to -x, so the secondary reaches
# its apoastron, its closest approach to the cavity edge, on the +x side
binary_varpi = np.pi
binary_nu = 0.0  # initial true anomaly (rad), 0 = start at periastron
# Largest accretion radii that stay inside the cavity: each sink is sized so that, at the
# farthest point of its star's orbit, it reaches this fraction of the cavity radius
binary_racc_cavity_frac = 0.8

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
alpha_ss = 1.0e-3  # Shakura-Sunyaev alpha of the disc, also used in the cavity relation
alpha_AV = alpha_ss / 0.08
alpha_u = 1.0
beta_AV = 2.0


# %%
# Binary separation from the cavity size (Ragusa et al. 2026)
#
# .. math::
#
#    a_{\rm bin} = R_p / [ c_1 + c_2 f(q) + c_3 e_{\rm bin} + c_4 e_{\rm bin}^2
#                  + c_5 e_{\rm cav} + c_6 e_{\rm cav}^2
#                  + c_7 e_{\rm bin} e_{\rm cav} \cos(\varpi_{\rm cav} - \varpi_{\rm bin})
#                  + c_8 \sqrt{\alpha / \alpha_0} + c_9 h / h_0 ]
#
# with :math:`f(q) = (\min(q, 0.4) / 3)^{1/3}` and :math:`\alpha_0 = h_0 = 0.1`.
#
# Here the cavity is the inner edge of the disc, which is set up circular:
# :math:`R_p = r_{\rm in}` and :math:`e_{\rm cav} = 0`. :math:`h` is the disc
# aspect ratio at :math:`R_p` and :math:`\alpha` is ``alpha_ss``.

ragusa2026_coeffs = (1.06, 2.24, 2.78, -1.97, 0.07, 1.19, -1.53, -0.18, 0.04)


def calculate_abin(R_p, q, e_bin, e_cav, varpi_bin, varpi_cav, alpha, h, coeffs=ragusa2026_coeffs):
    """
    Binary semi-major axis that carves a cavity of periastron ``R_p``,
    from the empirical relation of Ragusa et al. (2026).
    """
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeffs

    q_cut = 0.4
    alpha_0 = 0.1
    h_0 = 0.1

    f_q = (min(q, q_cut) / 3.0) ** (1.0 / 3.0)

    denom = (
        c1
        + c2 * f_q
        + c3 * e_bin
        + c4 * e_bin**2
        + c5 * e_cav
        + c6 * e_cav**2
        + c7 * e_bin * e_cav * np.cos(varpi_cav - varpi_bin)
        + c8 * np.sqrt(alpha / alpha_0)
        + c9 * (h / h_0)
    )

    if denom <= 0:
        raise ValueError(
            f"Ragusa et al. (2026) relation gives a non-positive denominator ({denom})"
        )

    return R_p / denom


cavity_Rp = disc.rin
cavity_e = 0.0
cavity_varpi = 0.0
cavity_h = disc.get_profiles().H(cavity_Rp) / cavity_Rp

# Central sinks as (mass, position, velocity, accretion radius), added in setup_model
# once the disc momentum and barycenter are zeroed.
if single_star:
    # Same rule as for the binary below: the sink reaches binary_racc_cavity_frac * R_p
    central_sinks = [
        (center_mass, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), binary_racc_cavity_frac * cavity_Rp)
    ]

    if shamrock.sys.world_rank() == 0:
        print(f"cavity: R_p = {cavity_Rp} au")
        print(f"single star: m = {center_mass}, racc = {central_sinks[0][3]} au")

else:
    binary_a = calculate_abin(
        R_p=cavity_Rp,
        q=binary_q,
        e_bin=binary_e,
        e_cav=cavity_e,
        varpi_bin=binary_varpi,
        varpi_cav=cavity_varpi,
        alpha=alpha_ss,
        h=cavity_h,
    )

    binary_m1 = center_mass / (1.0 + binary_q)
    binary_m2 = center_mass * binary_q / (1.0 + binary_q)

    # Large sinks keep the timestep from being set by gas falling close to the stars.
    # Star i orbits the center of mass on an ellipse scaled by m_other/M, so its largest
    # distance from the origin is (m_other/M) * a * (1 + e). Its sink then reaches at most
    # racc_i + (m_other/M) * a * (1 + e) = binary_racc_cavity_frac * R_p.
    # The sinks may overlap or contain the other star: sinks do not accrete each other.
    binary_apoastron = binary_a * (1.0 + binary_e)
    binary_rmax1 = binary_m2 / center_mass * binary_apoastron
    binary_rmax2 = binary_m1 / center_mass * binary_apoastron
    binary_racc1 = binary_racc_cavity_frac * cavity_Rp - binary_rmax1
    binary_racc2 = binary_racc_cavity_frac * cavity_Rp - binary_rmax2

    if min(binary_racc1, binary_racc2) <= 0:
        raise ValueError(
            f"a star goes beyond {binary_racc_cavity_frac} R_p on its orbit, no room left for its sink"
        )

    if binary_apoastron >= cavity_Rp:
        raise ValueError("binary apoastron is outside the cavity, the setup makes no sense")

    # get_binary_rotated puts the center of mass of the binary at rest at the origin
    x1, x2, v1, v2 = shamrock.phys.get_binary_rotated(
        m1=binary_m1,
        m2=binary_m2,
        a=binary_a,
        e=binary_e,
        nu=binary_nu,
        G=ucte.G(),
        roll=0.0,
        pitch=0.0,
        yaw=binary_varpi,
    )

    central_sinks = [
        (binary_m1, tuple(x1), tuple(v1), binary_racc1),
        (binary_m2, tuple(x2), tuple(v2), binary_racc2),
    ]

    if shamrock.sys.world_rank() == 0:
        print(f"cavity: R_p = {cavity_Rp} au, e = {cavity_e}, h = {cavity_h}")
        print(f"binary: a = {binary_a} au, R_p / a = {cavity_Rp / binary_a}")
        print(f"binary: m1 = {binary_m1}, m2 = {binary_m2}")
        print(f"binary: racc1 = {binary_racc1} au, racc2 = {binary_racc2} au")
        print(f"star 1: x = {x1}, v = {v1}")
        print(f"star 2: x = {x2}, v = {v2}")

# Dust parameters
kernel = "M6"
gamma = 1.4
t_inject = 0.0

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
    rhodust_eps = 1e-23 * codeu_kg_m3
    K0_multiplier = 1

# Integrator parameters
C_cour = 0.1
C_force = 0.1

central_tag = "single_dustydisc" if single_star else f"binary_dustydisc_q{binary_q}_e{binary_e}"
sim_folder = f"_to_trash/{central_tag}_{ndust}_{Npart}_{kernel}_coala_{use_coala}/"

dump_folder = sim_folder + "dump/"
analysis_folder = sim_folder + "analysis/"
plot_folder = analysis_folder + "plots/"

dump_prefix = dump_folder + "dump_"
# Permanent snapshots, kept apart from the checkpoints (dump_prefix), which are purged and
# used to restart the run
snapshot_prefix = dump_folder + "snapshot_"
snapshot_every = 50  # write a snapshot every 50 analyses

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

    # Farris et al. (2014) locally isothermal EOS: the sound speed follows the potential of
    # the central star(s). Same as LP07 for a single star, or far from the binary.
    cfg.set_eos_locally_isothermalFA2014_extended(
        cs0=disc.cs0(), q=disc.q, r0=disc.r0, n_sinks=len(central_sinks)
    )

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
    cfg.set_eta_sink(0.1)
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

    # now that the barycenter & momentum are 0, we can add the central star(s)
    for mass, pos, vel, racc in central_sinks:
        model.add_sink(mass, pos, vel, racc)

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

        if ianalysis % snapshot_every == 0:
            self.model.dump(snapshot_prefix + f"{ianalysis:07}.sham")

    @callback(tsim_interval=dt_stop_fast)  # Do the analysis every dt_stop
    def analysis_fast(self, ianalysis):
        for a in self.analysis_modules_fast:
            self.ana_module_run(a, ianalysis)

        self.model.do_vtk_dump(self.dump_prefix + f"{ianalysis:07}" + ".vtk", True)

    @callback(walltime_interval=60 * 10)  # Checkpoint the simulation every 10 minutes
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)

    @simulation_setup
    def setup(self):
        setup_model()


sim = Simulation(model)


from shamrock.utils.analysis import (
    AnalysisHelper,
    ColumnParticleCount,
    MassAnalysis,
    PerfHistory,
    SliceDiffVthetaProfile,
    SliceDtPart,
    SliceVzPlot,
    StandardPlotHelper,
)
from shamrock.utils.analysis.compute_field_dust import (
    compute_dlog_s_mean_dt_field,
    compute_effective_dust_col_speed_field,
    compute_rho_d,
    compute_rho_dj,
    compute_rho_g,
    compute_s_mean_field,
)

perf_analysis = PerfHistory(model, analysis_folder, "perf_history")
sim.analysis_modules_fast.append(perf_analysis)

mass_analysis = MassAnalysis(model, analysis_folder, "mass_history")
sim.analysis_modules_fast.append(mass_analysis)


def ColumnAverageDustSizePlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
):
    def compute_s_mean_integ(helper):
        return helper.column_average_render(compute_s_mean_field(model), "f64")

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
        compute_function=compute_s_mean_integ,
    )


def SliceDustSizePlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_s_mean_slice(helper):
        return helper.slice_render(
            compute_s_mean_field(model),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_s_mean_slice,
    )


def SliceDVeffPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_s_mean_slice(helper):
        return helper.slice_render(
            compute_effective_dust_col_speed_field(model),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_s_mean_slice,
    )


def SliceDustEvolSizePlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_dlog_s_dt_mean_slice(helper):
        return helper.slice_render(
            compute_dlog_s_mean_dt_field(model),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_dlog_s_dt_mean_slice,
    )


def ColumnAverageDustEvolSizePlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
):
    def compute_dlog_s_dt_mean_integ(helper):
        return helper.column_average_render(compute_dlog_s_mean_dt_field(model), "f64")

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
        compute_function=compute_dlog_s_dt_mean_integ,
    )


def ColumnAverageDVeffPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
):
    def compute_dveff_integ(helper):
        return helper.column_average_render(compute_effective_dust_col_speed_field(model), "f64")

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
        compute_function=compute_dveff_integ,
    )


def ColumnIntegFieldPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    compute_field,
):
    """Column integral of the field returned by ``compute_field(model)``"""

    def compute_integ(helper):
        return helper.column_integ_render(compute_field(model), "f64")

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
        compute_function=compute_integ,
    )


def SliceRhoGasPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_rho_g_slice(helper):
        return helper.slice_render(
            compute_rho_g(model),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_rho_g_slice,
    )


def SliceRhoDustPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_rho_d_slice(helper):
        return helper.slice_render(
            compute_rho_d(model),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_rho_d_slice,
    )


def SliceRhoDustSpeciePlot(
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
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_rho_dj_slice(helper):
        return helper.slice_render(
            compute_rho_dj(model, helper.jdust),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
        )

    tmp = StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rho_dj_slice,
    )
    tmp.jdust = jdust
    return tmp


def SliceAlphaAVPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_alpha_av_slice(helper):
        return helper.slice_render(
            "alpha_AV",
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_alpha_av_slice,
    )


def compute_vz_cs(model):

    def int_getter(size: int, dic_out: dict) -> np.array:
        return dic_out["vxyz"][:, 2] / dic_out["soundspeed"]

    return model.compute_field("custom", "f64", maybe_njit(int_getter))


def compute_delta_vtheta_cs(model):

    vel_profile_jit = maybe_njit(profiles.vtheta_kepler)

    def internal(
        size: int, x: np.array, y: np.array, vx: np.array, vy: np.array, vz: np.array
    ) -> np.array:
        r = np.sqrt(x**2 + y**2)
        r_safe = r + 1e-9
        v_theta = (-y * vx + x * vy) / r_safe
        v_relative = v_theta - vel_profile_jit(r)
        return v_relative

    def int_getter(size: int, dic_out: dict) -> np.array:
        return (
            internal(
                size,
                dic_out["xyz"][:, 0],
                dic_out["xyz"][:, 1],
                dic_out["vxyz"][:, 0],
                dic_out["vxyz"][:, 1],
                dic_out["vxyz"][:, 2],
            )
            / dic_out["soundspeed"]
        )

    return model.compute_field("custom", "f64", maybe_njit(int_getter))


def compute_angular_momt(model, Lprojection=(0.0, 0.0, 1.0)):

    pmass = model.get_particle_mass()
    hfact = model.get_hfact()

    vel_profile_jit = maybe_njit(profiles.vtheta_kepler)

    def internal(
        x: np.array,
        y: np.array,
        z: np.array,
        vx: np.array,
        vy: np.array,
        vz: np.array,
        hpart: np.array,
        cs: np.array,
    ) -> np.array:
        rho = pmass * (hfact / hpart) ** 3
        P = cs**2 * rho  # TODO: use true pressure

        r = np.sqrt(x**2 + y**2)
        r_safe = r + 1e-9
        v_r = (x * vx + y * vy) / r_safe
        v_theta = (-y * vx + x * vy) / r_safe

        delta_vtheta = v_theta - vel_profile_jit(r)
        alpha = rho * v_r * delta_vtheta / P

        return alpha

    internal = maybe_njit(internal)

    def int_getter(size: int, dic_out: dict) -> np.array:
        return internal(
            dic_out["xyz"][:, 0],
            dic_out["xyz"][:, 1],
            dic_out["xyz"][:, 2],
            dic_out["vxyz"][:, 0],
            dic_out["vxyz"][:, 1],
            dic_out["vxyz"][:, 2],
            dic_out["hpart"],
            dic_out["soundspeed"],
        )

    return model.compute_field("custom", "f64", maybe_njit(int_getter))


def SliceVzCs(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_alpha_av_slice(helper):
        return helper.slice_render(
            compute_vz_cs(helper.model),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_alpha_av_slice,
    )


def SliceDVthetaCs(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_dvtheta_cs_slice(helper):
        return helper.slice_render(
            compute_delta_vtheta_cs(helper.model),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_dvtheta_cs_slice,
    )


def SliceAlphaTransport(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_alpha_momt(helper):
        return helper.slice_render(
            compute_angular_momt(helper.model),
            "f64",
            do_normalization=do_normalization,
            min_normalization=min_normalization,
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
        compute_function=compute_alpha_momt,
    )


face_on_render_kwargs = {
    "x_unit": "au",
    "y_unit": "au",
    "time_unit": "year",
    "x_label": "x",
    "y_label": "y",
}

slice_params = {
    "ext_r": disc.rout * 0.6 / (16.0 / 9.0),  # aspect ratio of 16:9
    "nx": 1920,
    "ny": 1080,
    "ex": (1, 0, 0),
    "ey": (0, 0, 1),
    "center": ((disc.rin + disc.rout) / 2, 0, 0),
}

sink_params = {
    "sink_scale_factor": 1,
    "sink_color": "green",
    "sink_linewidth": 1,
    "sink_fill": False,
}

max_rho_plot = 1e-9
min_rho_plot = 1e-16
max_rho_integ_plot = 1e4
min_rho_integ_plot = 1e-3

face_on_params = {
    "ext_r": disc.rout,  # face-on view from -rout to rout
    "nx": 1024,
    "ny": 1024,
    "ex": (1, 0, 0),
    "ey": (0, 1, 0),
    "center": (0, 0, 0),
}

if ndust > 0:
    col_smean_plot = ColumnAverageDustSizePlot(
        model,
        ext_r=disc.rout,  # face-on view from -rout to rout
        nx=1024,
        ny=1024,
        ex=(1, 0, 0),
        ey=(0, 1, 0),
        center=(0, 0, 0),
        analysis_folder=analysis_folder,
        analysis_prefix="s_mean_column/plot",
    )
    col_smean_plot.render_args = {
        **face_on_render_kwargs,
        "field_unit": "m",
        "field_label": "$\\langle s \\rangle$",
        "vmin": mrn_distribution.grain_size_si.min(),
        "vmax": mrn_distribution.grain_size_si.max(),
        "contour_list": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "norm": "log",
    }

    sim.analysis_modules.append(col_smean_plot)

    slice_smean_plot = SliceDustSizePlot(
        model,
        **slice_params,
        analysis_folder=analysis_folder,
        analysis_prefix="s_mean_slice/plot",
    )

    slice_smean_plot.render_args = {
        **face_on_render_kwargs,
        "field_unit": "m",
        "field_label": "$\\langle s \\rangle$",
        "vmin": mrn_distribution.grain_size_si.min(),
        "vmax": mrn_distribution.grain_size_si.max(),
        "contour_list": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "norm": "log",
    }

    sim.analysis_modules.append(slice_smean_plot)

    slice_dveff = SliceDVeffPlot(
        model,
        **slice_params,
        analysis_folder=analysis_folder,
        analysis_prefix="delta_v_eff/plot",
    )

    slice_dveff.render_args = {
        **face_on_render_kwargs,
        "field_unit": "m.s^-1",
        "field_label": "$v_{\\rm eff}$",
        "vmin": 1e-4,
        "vmax": 1000,
        "contour_list": [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 10, 100],
        "norm": "log",
    }

    sim.analysis_modules.append(slice_dveff)

    slice_smean_evol_plot = SliceDustEvolSizePlot(
        model,
        **slice_params,
        analysis_folder=analysis_folder,
        analysis_prefix="s_mean_evol_slice/plot",
    )

    rnorm = mcolors.SymLogNorm(
        vmin=-1e-2,
        vmax=1e-2,
        linthresh=1e-5,
    )

    slice_smean_evol_plot.render_args = {
        **face_on_render_kwargs,
        "field_unit": "yr^-1",
        "field_label": r"$\partial_t{ \langle s \rangle}  / \langle s \rangle$",
        "contour_list": [-1, -1e-1, -1e-2, -1e-3, -1e-4, -1e-5, 0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        "cmap": "seismic",
        "cmap_bad_color": "white",
        "norm": rnorm,
    }

    sim.analysis_modules.append(slice_smean_evol_plot)

    col_smean_evol_plot = ColumnAverageDustEvolSizePlot(
        model,
        ext_r=disc.rout,  # face-on view from -rout to rout
        nx=1024,
        ny=1024,
        ex=(1, 0, 0),
        ey=(0, 1, 0),
        center=(0, 0, 0),
        analysis_folder=analysis_folder,
        analysis_prefix="s_mean_evol_column/plot",
    )

    col_smean_evol_plot.render_args = {
        **slice_smean_evol_plot.render_args,
        **sink_params,
    }

    sim.analysis_modules.append(col_smean_evol_plot)

    col_dveff = ColumnAverageDVeffPlot(
        model,
        **face_on_params,
        analysis_folder=analysis_folder,
        analysis_prefix="delta_v_eff_column/plot",
    )

    col_dveff.render_args = {
        **slice_dveff.render_args,
        **sink_params,
    }

    sim.analysis_modules.append(col_dveff)

    # Column densities, as in the basic disc example of the documentation
    col_rhog = ColumnIntegFieldPlot(
        model,
        **face_on_params,
        analysis_folder=analysis_folder,
        analysis_prefix="rho_gas_column/plot",
        compute_field=compute_rho_g,
    )

    col_rhog.render_args = {
        **face_on_render_kwargs,
        "field_unit": "kg.m^-2",
        "field_label": "$\\int \\rho_{{\\rm g}} \\, \\mathrm{{d}} z$",
        "vmin": min_rho_integ_plot,
        "vmax": max_rho_integ_plot,
        "norm": "log",
        **sink_params,
    }

    sim.analysis_modules.append(col_rhog)

    col_rhod = ColumnIntegFieldPlot(
        model,
        **face_on_params,
        analysis_folder=analysis_folder,
        analysis_prefix="rho_dust_column_all/plot",
        compute_field=compute_rho_d,
    )

    col_rhod.render_args = {
        **col_rhog.render_args,
        "field_label": "$\\int \\rho_{{\\rm d}} \\, \\mathrm{{d}} z$",
        "vmin": 0.02 * min_rho_integ_plot,
        "vmax": 0.02 * max_rho_integ_plot,
    }

    sim.analysis_modules.append(col_rhod)

    for j in range(ndust):
        col_rhodj = ColumnIntegFieldPlot(
            model,
            **face_on_params,
            analysis_folder=analysis_folder,
            analysis_prefix=f"rho_dust_column_{j}/plot",
            compute_field=lambda model, j=j: compute_rho_dj(model, j),
        )

        col_rhodj.render_args = {
            **col_rhog.render_args,
            "field_label": f"$\\int \\rho_{{\\rm d , {j} }} \\, \\mathrm{{d}} z$",
            "vmin": 0.01 * min_rho_integ_plot,
            "vmax": 0.01 * max_rho_integ_plot,
            "extra_title": f"[$s_{{grain}}$ = {mrn_distribution.grain_size_si[j]:.2e} m]",
        }

        sim.analysis_modules.append(col_rhodj)

    slice_rhog = SliceRhoGasPlot(
        model,
        **slice_params,
        analysis_folder=analysis_folder,
        analysis_prefix="rho_gas_slice/plot",
    )

    slice_rhog.render_args = {
        **face_on_render_kwargs,
        "field_unit": "kg.m^-3",
        "field_label": "$\\rho_{{\\rm g}}$",
        "vmin": min_rho_plot,
        "vmax": max_rho_plot,
        "norm": "log",
        **sink_params,
    }

    sim.analysis_modules.append(slice_rhog)

    slice_rhod = SliceRhoDustPlot(
        model,
        **slice_params,
        analysis_folder=analysis_folder,
        analysis_prefix="rho_dust_slice_all/plot",
    )

    slice_rhod.render_args = {
        **face_on_render_kwargs,
        "field_unit": "kg.m^-3",
        "field_label": "$\\rho_{{\\rm d}}$",
        "vmin": 0.02 * min_rho_plot,
        "vmax": 0.02 * max_rho_plot,
        "norm": "log",
        **sink_params,
    }

    sim.analysis_modules.append(slice_rhod)

    for j in range(ndust):
        slice_rhodj = SliceRhoDustSpeciePlot(
            model,
            **slice_params,
            analysis_folder=analysis_folder,
            analysis_prefix=f"rho_dust_slice_{j}/plot",
            jdust=j,
        )

        slice_rhodj.render_args = {
            **face_on_render_kwargs,
            "field_unit": "kg.m^-3",
            "field_label": f"$\\rho_{{\\rm d , {j} }}$",
            "vmin": 0.01 * min_rho_plot,
            "vmax": 0.01 * max_rho_plot,
            "norm": "log",
            **sink_params,
            "extra_title": f"[$s_{{grain}}$ = {mrn_distribution.grain_size_si[j]:.2e} m]",
        }

        sim.analysis_modules.append(slice_rhodj)

v_z_slice_plot = SliceVzCs(
    model,
    **slice_params,
    analysis_folder=analysis_folder,
    analysis_prefix="v_z_slice/plot",
    do_normalization=True,
)

v_z_slice_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": None,
    "field_label": "$\\mathrm{v}_z / c_s$",
    "cmap": "seismic",
    "cmap_bad_color": "white",
    "vmin": -0.2,
    "vmax": 0.2,
    **sink_params,
}

sim.analysis_modules.append(v_z_slice_plot)


dvtheta_cs_slice_plot = SliceDVthetaCs(
    model,
    **slice_params,
    analysis_folder=analysis_folder,
    analysis_prefix="dvtheta_cs_slice/plot",
    do_normalization=True,
)

dvtheta_cs_slice_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": None,
    "field_label": "$(\\mathrm{v}_{\\theta} - v_k) / c_s$",
    "cmap": "seismic",
    "cmap_bad_color": "white",
    "vmin": -0.4,
    "vmax": 0.4,
    **sink_params,
}

sim.analysis_modules.append(dvtheta_cs_slice_plot)


dt_part_slice_plot = SliceDtPart(
    model,
    **slice_params,
    analysis_folder=analysis_folder,
    analysis_prefix="dt_part_slice/plot",
)

dt_part_slice_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": "year",
    "field_label": "$\\Delta t$",
    "vmin": 1e-4,
    "vmax": 100,
    "norm": "log",
    "contour_list": [1e-2, 1e-1, 1, 10, 100],
    **sink_params,
}

sim.analysis_modules.append(dt_part_slice_plot)

column_particle_count_plot = ColumnParticleCount(
    model,
    ext_r=disc.rout,  # face-on view from -rout to rout
    nx=1024,
    ny=1024,
    ex=(1, 0, 0),
    ey=(0, 1, 0),
    center=(0, 0, 0),
    analysis_folder=analysis_folder,
    analysis_prefix="particle_count/plot",
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

sim.analysis_modules.append(column_particle_count_plot)

alpha_av_plot = SliceAlphaAVPlot(
    model,
    **slice_params,
    analysis_folder=analysis_folder,
    analysis_prefix="alpha_av_slice/plot",
    do_normalization=True,
)

alpha_av_plot.render_args = {
    **face_on_render_kwargs,
    "field_unit": None,
    "field_label": "$\\alpha_{\\rm AV}$",
    "vmin": 1e-6,
    "vmax": 1,
    "norm": "log",
    "contour_list": [1e-4, 1e-3, 1e-2, 1e-1],
    **sink_params,
}

sim.analysis_modules.append(alpha_av_plot)


alpha_av_momt = SliceAlphaTransport(
    model,
    **slice_params,
    analysis_folder=analysis_folder,
    analysis_prefix="alpha_momt_slice/plot",
    do_normalization=True,
)

alpha_av_momt.render_args = {
    **face_on_render_kwargs,
    "field_unit": None,
    "field_label": "$\\alpha_{\\rm L}$",
    "vmin": 1e-6,
    "vmax": 1,
    "norm": "log",
    "contour_list": [1e-4, 1e-3, 1e-2, 1e-1],
    **sink_params,
}

sim.analysis_modules.append(alpha_av_momt)


class radial_profile_plot:
    def __init__(self):
        self.profile_plot = AnalysisHelper(
            analysis_folder=os.path.join(analysis_folder, "plots"),
            analysis_prefix="density_profile/plot",
        )
        self.render_args = {}

    def analysis_save(self, ianalysis):
        def internal(size: int, x: np.array, y: np.array) -> np.array:
            r = np.sqrt(x**2 + y**2)
            return r

        def custom_getter_r(size: int, dic_out: dict) -> np.array:
            return internal(
                size,
                dic_out["xyz"][:, 0],
                dic_out["xyz"][:, 1],
            )

        x_min = min(racc for *_, racc in central_sinks) / 1.1
        x_max = disc.rout * 2
        x_min_log = np.log10(x_min)
        x_max_log = np.log10(x_max)

        bin_edges_x1d = np.logspace(x_min_log, x_max_log, 1025)

        dens_fact = codeu.to("kg") * codeu.to("m", power=-3)

        rho_t_field = model.compute_field("rho", "f64")
        hpart_field = model.compute_field("hpart", "f64")
        r_field = model.compute_field("custom", "f64", custom_getter_r)

        rho_g_field = compute_rho_g(model)
        rho_d_field = compute_rho_d(model)

        dic_ret = {
            "time": model.get_time(),
            "bin_edges_x1d": bin_edges_x1d,
        }

        histo_rho_t = shamrock.compute_histogram(
            bin_edges=bin_edges_x1d,
            x_field=r_field,
            y_field=rho_t_field,
            do_average=True,
        )

        histo_rho_g = shamrock.compute_histogram(
            bin_edges=bin_edges_x1d,
            x_field=r_field,
            y_field=rho_g_field,
            do_average=True,
        )

        histo_rho_d = shamrock.compute_histogram(
            bin_edges=bin_edges_x1d,
            x_field=r_field,
            y_field=rho_d_field,
            do_average=True,
        )

        dic_ret["histo_rho_t"] = np.array(histo_rho_t) * dens_fact
        dic_ret["histo_rho_g"] = np.array(histo_rho_g) * dens_fact
        dic_ret["histo_rho_d"] = np.array(histo_rho_d) * dens_fact

        dic_ret["histo_rho_d_j"] = []

        for jdust in range(ndust):
            rhod_j_field = compute_rho_dj(model, jdust)

            histo_rho_d_j = shamrock.compute_histogram(
                bin_edges=bin_edges_x1d,
                x_field=r_field,
                y_field=rhod_j_field,
                do_average=True,
            )

            dic_ret["histo_rho_d_j"].append(np.array(histo_rho_d_j) * dens_fact)

        self.profile_plot.analysis_save(ianalysis, dic_ret)

    def plot_func(self, iplot, data):

        data = data.item()
        print(data.keys())

        time = data["time"]

        bin_edges_x1d = data["bin_edges_x1d"]

        bin_center = (bin_edges_x1d[:-1] + bin_edges_x1d[1:]) / 2

        fig = plt.figure(dpi=250, figsize=(8, 5))

        dust_cmap = plt.colormaps["plasma"]
        dust_norm = mcolors.LogNorm(vmin=grain_size_si.min(), vmax=grain_size_si.max() * 10)
        dust_colors = dust_cmap(dust_norm(grain_size_si))

        plt.plot(bin_center, data["histo_rho_t"], "--")
        plt.plot(bin_center, data["histo_rho_g"], color="0.0")
        plt.plot(bin_center, data["histo_rho_d"], color="0.5")

        for jdust in range(ndust):
            c = dust_colors[jdust]
            plt.plot(bin_center, data["histo_rho_d_j"][jdust], color=c)

        plt.xlabel("r [au]")
        plt.ylabel("$\\langle \\rho \\rangle_z$ [kg.m^-3]")

        plt.xscale("log")
        plt.yscale("log")

        plt.xlim(np.min(bin_edges_x1d), np.max(bin_edges_x1d))
        plt.ylim(min_rho_plot, max_rho_plot)

        text = f"t = {time:0.3f} [yr]"
        from matplotlib.offsetbox import AnchoredText

        anchored_text = AnchoredText(text, loc=2)
        plt.gca().add_artist(anchored_text)

        dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
        dust_sm.set_array([])
        cbar = fig.colorbar(dust_sm, ax=plt.gca(), pad=0.02, shrink=0.85)
        cbar.set_label(r"grain size $s$ [m]")

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
        plt.gca().legend(handles=[gas_handle, dust_handle], loc="upper right", fontsize=8)

        plt.savefig(self.profile_plot.analysis_prefix + f"_curves_{iplot:07}.png")
        plt.savefig(self.profile_plot.analysis_prefix + f"_curves_{iplot:07}.pdf")
        plt.close()

        fig, axs = plt.subplots(
            2, 1, dpi=250, figsize=(10, 7), sharex=True, gridspec_kw={"wspace": 0.0, "hspace": 0}
        )
        axs[0].plot(bin_center, data["histo_rho_t"], "--", label="total")
        axs[0].plot(bin_center, data["histo_rho_g"], color="0.0", label="gas")
        axs[0].plot(bin_center, data["histo_rho_d"], color="0.5", label="dust")

        for jdust in range(ndust):
            axs[0].plot(bin_center, data["histo_rho_d_j"][jdust], color=dust_colors[jdust])

        axs[1].set_xlabel("r [au]")
        axs[0].set_ylabel("$\\langle \\rho \\rangle_z$ [kg.m^-3]")

        axs[0].set_xscale("log")
        axs[0].set_yscale("log")

        axs[0].set_xlim(np.min(bin_edges_x1d), np.max(bin_edges_x1d))
        axs[0].set_ylim(min_rho_plot * 1.1, max_rho_plot)  # 1.1 to avoid the tick

        axs[0].legend(loc="upper right", fontsize=8)

        im = np.zeros((ndust, len(bin_center)))

        for jdust in range(ndust):
            im[jdust, :] = data["histo_rho_d_j"][jdust]

        rho_norm = mcolors.LogNorm(
            vmin=min_rho_plot * epsilon_base, vmax=max_rho_plot * epsilon_base
        )
        im = np.where(im <= 0, 1e-30, im)

        axs[1].pcolormesh(
            bin_edges_x1d,
            grain_size_si_edges,
            im,
            cmap=dust_cmap,
            norm=rho_norm,
            shading="auto",
            rasterized=True,
        )
        axs[1].set_ylabel("grain size [m]")
        axs[1].set_yscale("log")
        axs[1].set_ylim(grain_size_si_edges[0], grain_size_si_edges[-1])

        rho_sm = cm.ScalarMappable(cmap=dust_cmap, norm=rho_norm)
        rho_sm.set_array([])
        fig.subplots_adjust(right=0.88)
        cbar_dust = fig.colorbar(
            dust_sm, ax=axs[0], pad=0.03, fraction=0.025, aspect=20, shrink=0.85
        )
        cbar_dust.set_label(r"grain size $s$ [m]")
        cbar_rho = fig.colorbar(rho_sm, ax=axs[1], pad=0.03, fraction=0.025, aspect=20, shrink=0.85)
        cbar_rho.set_label(r"$\langle \rho(r,s_{{grain}}) \rangle_z$ [kg.m^-3]")

        text = f"t = {time:0.3f} [yr]"
        from matplotlib.offsetbox import AnchoredText

        anchored_text = AnchoredText(text, loc=2)
        axs[0].add_artist(anchored_text)

        plt.savefig(self.profile_plot.analysis_prefix + f"_image_{iplot:07}.png")
        plt.savefig(self.profile_plot.analysis_prefix + f"_image_{iplot:07}.pdf")
        plt.close()

    def make_plot(self, iplot):
        self.profile_plot.make_plot(iplot, self.plot_func)

    def render_all(self):
        self.profile_plot.render_all(self.plot_func)


class vert_slices_plots:
    def __init__(self):
        self.profile_plot = AnalysisHelper(
            analysis_folder=os.path.join(analysis_folder, "plots"),
            analysis_prefix="vert_slices/plot",
        )
        self.rcenters = [20, 50, 100]
        self.rextents_fact = 0.25
        self.render_args = {}

    def analysis_save(self, ianalysis):

        z_r_extent = 4 * disc.H_r_0
        z_r_edges = np.linspace(-z_r_extent, z_r_extent, 1025)

        dens_fact = codeu.to("kg") * codeu.to("m", power=-3)

        dic_ret = {
            "time": model.get_time(),
            "z_r_edges": z_r_edges,
            "rcases": [{} for _ in self.rcenters],
        }

        rho_t_field = model.compute_field("rho", "f64")
        hpart_field = model.compute_field("hpart", "f64")

        rho_g_field = compute_rho_g(model)
        rho_d_field = compute_rho_d(model)

        rho_d_j_fields = []
        for jdust in range(ndust):
            rho_d_j_fields.append(compute_rho_dj(model, jdust))

        for ir, rcenter in enumerate(self.rcenters):
            r_extent = rcenter * self.rextents_fact

            r_min = rcenter - r_extent
            r_max = rcenter + r_extent

            def internal(size: int, x: np.array, y: np.array, z: np.array) -> np.array:
                r = np.sqrt(x**2 + y**2)
                # make a mask for the particles inside the ring
                valid = (r >= r_min) & (r <= r_max)
                # fill a array with nans
                out = np.full_like(z, np.nan, dtype=np.float64)
                # replace nans by values in the ring
                out[valid] = z[valid] / r[valid]

                # show average of the array
                # print(f"Average of the array: {np.nanmean(r[valid])} for r = {r_extent} rmin = {r_min} rmax = {r_max}")

                return out

            def custom_getter(size: int, dic_out: dict) -> np.array:
                return internal(
                    size,
                    dic_out["xyz"][:, 0],
                    dic_out["xyz"][:, 1],
                    dic_out["xyz"][:, 2],
                )

            z_r_field = model.compute_field("custom", "f64", custom_getter)

            histo_rho_t = shamrock.compute_histogram(
                bin_edges=z_r_edges,
                x_field=z_r_field,
                y_field=rho_t_field,
                do_average=True,
            )

            histo_rho_g = shamrock.compute_histogram(
                bin_edges=z_r_edges,
                x_field=z_r_field,
                y_field=rho_g_field,
                do_average=True,
            )

            histo_rho_d = shamrock.compute_histogram(
                bin_edges=z_r_edges,
                x_field=z_r_field,
                y_field=rho_d_field,
                do_average=True,
            )

            dic_ret["rcases"][ir]["histo_rho_t"] = np.array(histo_rho_t) * dens_fact
            dic_ret["rcases"][ir]["histo_rho_g"] = np.array(histo_rho_g) * dens_fact
            dic_ret["rcases"][ir]["histo_rho_d"] = np.array(histo_rho_d) * dens_fact

            dic_ret["rcases"][ir]["histo_rho_d_j"] = []

            for jdust in range(ndust):
                histo_rho_d_j = shamrock.compute_histogram(
                    bin_edges=z_r_edges,
                    x_field=z_r_field,
                    y_field=rho_d_j_fields[jdust],
                    do_average=True,
                )

                dic_ret["rcases"][ir]["histo_rho_d_j"].append(np.array(histo_rho_d_j) * dens_fact)

        self.profile_plot.analysis_save(ianalysis, dic_ret)

    def plot_func(self, iplot, data):

        data = data.item()

        time = data["time"]
        z_r_edges = data["z_r_edges"]

        bin_center = (z_r_edges[:-1] + z_r_edges[1:]) / 2

        dust_cmap = plt.colormaps["plasma"]
        dust_norm = mcolors.LogNorm(vmin=grain_size_si.min(), vmax=grain_size_si.max() * 10)
        dust_colors = dust_cmap(dust_norm(grain_size_si))

        fig, axs = plt.subplots(
            2,
            len(self.rcenters),
            figsize=(15, 7),
            dpi=250,
            sharex=True,
            sharey="row",
            gridspec_kw={"wspace": 0.0, "hspace": 0},
        )
        rho_norm = mcolors.LogNorm(
            vmin=min_rho_plot * epsilon_base, vmax=max_rho_plot * epsilon_base
        )
        for ir, rcenter in enumerate(self.rcenters):
            axs[0, ir].plot(bin_center, data["rcases"][ir]["histo_rho_t"], "--", label="total")
            axs[0, ir].plot(bin_center, data["rcases"][ir]["histo_rho_g"], color="0.0", label="gas")
            axs[0, ir].plot(
                bin_center, data["rcases"][ir]["histo_rho_d"], color="0.5", label="dust"
            )

            for jdust in range(ndust):
                axs[0, ir].plot(
                    bin_center, data["rcases"][ir]["histo_rho_d_j"][jdust], color=dust_colors[jdust]
                )

            axs[0, ir].set_yscale("log")
            axs[0, ir].set_xlim(np.min(z_r_edges), np.max(z_r_edges))
            axs[0, ir].set_ylim(1.1 * min_rho_plot, max_rho_plot)  # 1.1 to avoid the tick
            axs[0, ir].legend(loc="upper right", fontsize=8)

            axs[0, ir].set_title(rf"$r \in {rcenter} \pm {self.rextents_fact * rcenter}$")
            if ir == 0:
                axs[0, ir].set_ylabel(r"$\langle \rho \rangle_r$ [kg.m^-3]")
            else:
                axs[0, ir].tick_params(labelleft=False, left=False)
            axs[0, ir].tick_params(labelbottom=False)

            im = np.zeros((ndust, len(bin_center)))

            for jdust in range(ndust):
                im[jdust, :] = data["rcases"][ir]["histo_rho_d_j"][jdust]

            im = np.where(im <= 0, 1e-30, im)

            axs[1, ir].pcolormesh(
                z_r_edges,
                grain_size_si_edges,
                im,
                cmap=dust_cmap,
                norm=rho_norm,
                shading="auto",
                rasterized=True,
            )

            if ir == 0:
                axs[1, ir].set_ylabel(r"grain size $s$ [m]")
            else:
                axs[1, ir].tick_params(labelleft=False, left=False)

            axs[1, ir].set_xlim(np.min(z_r_edges), np.max(z_r_edges))
            axs[1, ir].set_ylim(grain_size_si_edges[0], grain_size_si_edges[-1])

            axs[1, ir].set_yscale("log")

            labels = axs[1, ir].get_xticklabels()
            if ir > 0 and labels:
                labels[0].set_visible(False)
            if ir < len(self.rcenters) - 1 and labels:
                labels[-1].set_visible(False)

            axs[1, ir].set_xlabel(r"z/r")

        dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
        dust_sm.set_array([])
        rho_sm = cm.ScalarMappable(cmap=dust_cmap, norm=rho_norm)
        rho_sm.set_array([])
        fig.tight_layout(rect=[0, 0, 0.94, 1])
        cbar_dust = fig.colorbar(
            dust_sm, ax=axs[0, :], pad=0.03, fraction=0.025, aspect=20, shrink=0.85
        )
        cbar_dust.set_label(r"grain size $s$ [m]")
        cbar_rho = fig.colorbar(
            rho_sm, ax=axs[1, :], pad=0.03, fraction=0.025, aspect=20, shrink=0.85
        )
        cbar_rho.set_label(r"$\langle \rho(z/r,s_{\mathrm{grain}}) \rangle_r$ [kg.m^-3]")

        text = f"t = {time:0.3f} [yr]"
        from matplotlib.offsetbox import AnchoredText

        anchored_text = AnchoredText(text, loc=2)
        axs[0, 0].add_artist(anchored_text)

        plt.savefig(self.profile_plot.analysis_prefix + f"_vert_slices_{iplot:07}.png")
        plt.savefig(self.profile_plot.analysis_prefix + f"_vert_slices_{iplot:07}.pdf")
        plt.close()

        for ir, rcenter in enumerate(self.rcenters):
            plt.figure()

            im = np.zeros((ndust, len(bin_center)))

            for jdust in range(ndust):
                im[jdust, :] = data["rcases"][ir]["histo_rho_d_j"][jdust]

            z_r_bins = np.linspace(0, 0.25, 10)

            zr_cmap = plt.colormaps["plasma"]
            zr_norm = mcolors.Normalize(vmin=z_r_bins[0], vmax=z_r_bins[-1])

            second_hand_plot_dat = {"grain_size_si": grain_size_si.tolist()}
            z_r_bin_dat = []

            for ibin in range(len(z_r_bins) - 1):
                z_r_min = z_r_bins[ibin]
                z_r_max = z_r_bins[ibin + 1]
                z_r_bin_center = 0.5 * (z_r_min + z_r_max)

                # compute the average between z_r_min and z_r_max of im[:, :] to get a <rho_d> (jdust) curve
                mask = (bin_center >= z_r_min) & (bin_center < z_r_max)
                rho_avg_z_r = np.nanmean(im[:, mask], axis=1)

                plt.plot(grain_size_si, rho_avg_z_r, color=zr_cmap(zr_norm(z_r_bin_center)))

                z_r_bin_dat.append(
                    {
                        "rho_avg_z_r": rho_avg_z_r.tolist(),
                        "z_r_min": z_r_min.tolist(),
                        "z_r_max": z_r_max.tolist(),
                    }
                )

            second_hand_plot_dat["z_r_bin_dat"] = z_r_bin_dat

            mean_dist = np.nanmean(im, axis=1)
            plt.plot(grain_size_si, mean_dist, linestyle="dashed", c="grey", label="mean")
            second_hand_plot_dat["mean_dist"] = mean_dist.tolist()

            with open(
                self.profile_plot.analysis_prefix + f"_second_hand_plot_{rcenter}_{iplot:07}.json",
                "w",
            ) as f:
                json.dump(second_hand_plot_dat, f)

            handles, labels = plt.gca().get_legend_handles_labels()
            shamrock.matplotlib.add_cmap_legend_entry(
                plt.gca(),
                dust_cmap,
                label=r"$\rho_{z/r}(s_{\rm grain})$",
                extra_handles=handles,
                extra_labels=labels,
                loc="best",
                fontsize=12,
            )

            zr_sm = cm.ScalarMappable(cmap=zr_cmap, norm=zr_norm)
            zr_sm.set_array([])
            cbar_zr = plt.colorbar(zr_sm, ax=plt.gca())
            cbar_zr.set_label("z/r")

            plt.xlabel("grain size [m]")
            plt.ylabel(r"$\langle \rho_d \rangle_{z/r}$ [$kg.m^{-3}$]")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim(np.min(grain_size_si), np.max(grain_size_si))
            plt.ylim(1e-16, 2e-12)
            plt.title(rf"r = {rcenter} [au] {text}")
            plt.tight_layout(pad=0.2)

            plt.savefig(self.profile_plot.analysis_prefix + f"_distrib_{rcenter}_{iplot:07}.png")
            plt.savefig(self.profile_plot.analysis_prefix + f"_distrib_{rcenter}_{iplot:07}.pdf")

            plt.close()

    def make_plot(self, iplot):
        self.profile_plot.make_plot(iplot, self.plot_func)

    def render_all(self):
        self.profile_plot.render_all(self.plot_func)


if ndust > 0:
    rad_plot = radial_profile_plot()
    vert_plot = vert_slices_plots()

    sim.analysis_modules.append(rad_plot)
    sim.analysis_modules.append(vert_plot)

sim.run()
