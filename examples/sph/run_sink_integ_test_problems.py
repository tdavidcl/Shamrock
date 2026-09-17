"""
Sink integration test problems
=======================================

This example shows how to use the sink integration test problems.
"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    # loglevel 0 (the shamrock executable's own default) rather than 1: the
    # choreographies below need thousands of timesteps per period, and the
    # per-step performance report is 35 lines each.
    shamrock.change_loglevel(0)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()


# %%
# Define the unit system
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
# Build the SPH model with the sink particles
def build_sink_sph_model(
    positions,
    velocities,
    masses,
    accretion_radii,
    box_extent,
    eta_sink=1,
    cfl_force=0.1,
    cfl_cour=0.1,
    show_cfl_detail=True,
):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    # Allow experimental features (required for self-gravity)
    shamrock.enable_experimental_features()

    cfg = model.gen_default_config()
    # Disable direct self-gravity in this simple example; direct mode requires
    # a single-patch setup which is not prepared here.
    cfg.set_self_gravity_none()
    cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
    cfg.set_particle_mass(1e-6)
    cfg.set_eos_isothermal(1.0)
    cfg.set_show_cfl_detail(show_cfl_detail)
    cfg.set_eta_sink(eta_sink)
    cfg.set_cfl_force(cfl_force)
    cfg.set_cfl_cour(cfl_cour)
    # Set code units so warnings about unit system disappear
    cfg.set_units(codeu)

    model.set_solver_config(cfg)

    # Initialise the scheduler before adding sinks
    model.init_scheduler(int(1e7), 1)

    for position, velocity, mass, accretion_radius in zip(
        positions, velocities, masses, accretion_radii
    ):
        model.add_sink(mass, tuple(position), tuple(velocity), accretion_radius)

    ext = box_extent
    bmin = (-ext, -ext, -ext)
    bmax = (ext, ext, ext)
    model.resize_simulation_box(bmin, bmax)

    return ctx, model


# %%
# Extract sink positions from the model
def get_sink_positions(model):
    sinks = model.get_sinks()
    positions = [tuple(sink["pos"]) for sink in sinks]
    velocities = [tuple(sink["velocity"]) for sink in sinks]
    return positions, velocities


# %%
def correct_sink_velocities_zero_momentum(velocities, masses):
    """Subtract center-of-mass velocity so total momentum vanishes."""
    masses = np.asarray(masses)
    vels = np.asarray(velocities)
    v_com = np.sum(masses[:, np.newaxis] * vels, axis=0) / np.sum(masses)
    return [tuple(v - v_com) for v in vels]


# %%
# Run a simple orbit evolution and collect sink snapshots
def run_sim(model, max_time, use_dt=None):
    """Evolve binary orbit until max_time"""
    snapshots = []
    current_time = 0.0

    # Print initial conditions
    initial_sinks = model.get_sinks()
    print("\n=== INITIAL CONDITIONS ===")
    for i, sink in enumerate(initial_sinks):
        print(f"Sink {i + 1}: pos={sink['pos']}, vel={sink['velocity']}, mass={sink['mass']}")
    print()

    while current_time < max_time:
        if use_dt is None:
            model.timestep()
        else:
            model.evolve_once_override_time(current_time + use_dt, use_dt)
        current_time = model.get_time()

        positions, velocities = get_sink_positions(model)

        snapshots.append(
            {
                "time": current_time,
                "positions": positions,
                "velocities": velocities,
            }
        )

    return snapshots


# %%
# Plot complete orbital trajectories
def plot_orbit_trajectory(snapshots, suptitle):
    import matplotlib.pyplot as plt

    sinks_positions = np.array([snap["positions"] for snap in snapshots])

    nstep, nsink, ndim = sinks_positions.shape

    print(sinks_positions.shape)

    # Extract trajectories for both sinks
    sink2_positions = np.array([snap["positions"][1] for snap in snapshots])

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(suptitle)

    # 3D plot
    ax3d = fig.add_subplot(121, projection="3d")

    for isink in range(nsink):
        ax3d.plot(
            sinks_positions[:, isink, 0],
            sinks_positions[:, isink, 1],
            sinks_positions[:, isink, 2],
            "o-",
            label=f"Sink {isink + 1}",
            markersize=3,
            linewidth=1,
        )
    ax3d.set_xlabel("x (AU)")
    ax3d.set_ylabel("y (AU)")
    ax3d.set_zlabel("z (AU)")
    ax3d.set_title("3D Orbit")
    ax3d.legend()
    ax3d.set_aspect("equal")

    # 2D plot (xy plane)
    ax2d = fig.add_subplot(122)

    for isink in range(nsink):
        ax2d.plot(
            sinks_positions[:, isink, 0],
            sinks_positions[:, isink, 1],
            "o-",
            label=f"Sink {isink + 1}",
            markersize=1,
            linewidth=1,
        )

    ax2d.set_xlabel("x (AU)")
    ax2d.set_ylabel("y (AU)")
    ax2d.set_title("Orbit (xy plane)")
    ax2d.legend()
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# %%
# Circular orbit
m1 = 1.0
m2 = 1e-4
a = 1.0
_x1, _x2, _v1, _v2 = shamrock.phys.get_binary_rotated(
    m1=1.0, m2=m2, a=a, e=0.0, nu=0.0, G=G, roll=0.0, pitch=0.0, yaw=0.0
)
ctx, model = build_sink_sph_model(
    positions=[_x1, _x2],
    velocities=[_v1, _v2],
    masses=[m1, m2],
    accretion_radii=[1, 1],
    box_extent=3,
    eta_sink=0.5,
)
snapshots = run_sim(model, 10, use_dt=None)
plot_orbit_trajectory(snapshots, "Circular orbit")

# %%
# Elliptical orbit
m1 = 1.0
m2 = 1e-4
a = 1.0
_x1, _x2, _v1, _v2 = shamrock.phys.get_binary_rotated(
    m1=1.0, m2=m2, a=a, e=0.9, nu=0.0, G=G, roll=0.0, pitch=0.0, yaw=np.pi
)
ctx, model = build_sink_sph_model(
    positions=[_x1, _x2],
    velocities=[_v1, _v2],
    masses=[m1, m2],
    accretion_radii=[1, 1],
    box_extent=3,
    eta_sink=0.5,
)
snapshots = run_sim(model, 10, use_dt=None)
plot_orbit_trajectory(snapshots, "Elliptical orbit")

# %%
# Elliptical orbit (similar mass)
m1 = 1.0
m2 = 1.0 / 3.0
a = 1.0
_x1, _x2, _v1, _v2 = shamrock.phys.get_binary_rotated(
    m1=1.0, m2=m2, a=a, e=0.9, nu=0.0, G=G, roll=0.0, pitch=0.0, yaw=np.pi
)
ctx, model = build_sink_sph_model(
    positions=[_x1, _x2],
    velocities=[_v1, _v2],
    masses=[m1, m2],
    accretion_radii=[1, 1],
    box_extent=3,
    eta_sink=0.5,
)
snapshots = run_sim(model, 10, use_dt=None)
plot_orbit_trajectory(snapshots, "Elliptical orbit (similar mass)")

# %%
# 1 star multiple planets (resonance 3:2)
m1 = 1.0
m2 = 1e-2
a = 1.0
_x1, _x2, _v1, _v2 = shamrock.phys.get_binary_rotated(
    m1=1.0, m2=m2, a=a, e=0.0, nu=0.0, G=G, roll=0.0, pitch=0.0, yaw=0.0
)
a = a * (3.0 / 2.0) ** (2.0 / 3.0)
_x1, _x3, _v1, _v3 = shamrock.phys.get_binary_rotated(
    m1=1.0, m2=m2, a=a, e=0.0, nu=0.0, G=G, roll=0.0, pitch=0.0, yaw=np.pi
)

_v1, _v2, _v3 = correct_sink_velocities_zero_momentum([_v1, _v2, _v3], [m1, m2, m2])

ctx, model = build_sink_sph_model(
    positions=[_x1, _x2, _x3],
    velocities=[_v1, _v2, _v3],
    masses=[m1, m2, m2],
    accretion_radii=[1, 1, 1],
    box_extent=3,
    eta_sink=2.0,
)
snapshots = run_sim(model, 10, use_dt=None)
plot_orbit_trajectory(snapshots, "1 star multiple planets (resonance 3:2)")


# %%
# Gravitational choreographies
# ============================
#
# A gravitational choreography is a periodic solution of the N-body problem in
# which every body travels along the *same* closed curve, equally spaced in
# time: body ``i`` sits at ``q(t + i T / N)``. All the systems below start in
# the ``z = 0`` plane with no velocity component normal to it, so they stay
# planar forever and the 3D panel of the plots doubles as a check that the
# integrator does not push the sinks out of the plane.
#
# The loops were obtained as critical points of the action
#
# .. math::
#     A[q] = \int_0^T \left[ \sum_i \frac{1}{2} |\dot{q}_i|^2
#            + \sum_{i<j} \frac{G m^2}{|q_i - q_j|} \right] dt
#
# expanded in a Fourier series, and the initial conditions were then refined on
# the equations of motion themselves: integrating every body for one ``T / N``
# must cyclically permute the state, so solving that condition by a Newton
# shooting method pins the orbit down to a residual of about ``1e-15``.
#
# Apart from the figure eight, all of these orbits are linearly unstable.
# ``lyapunov_rate`` below is the largest Floquet exponent, in e-folds per
# period: an integrator accurate to ``eps`` per period follows the loop for
# roughly ``ln(tol / eps) / lyapunov_rate`` periods before the bodies visibly
# leave it. That exponential sensitivity is what makes these orbits a sharp
# test of the sink integrator, and it is also why ``n_periods`` is small for
# the fast ones.
#
# The table is written in the natural normalisation of the problem (``G = 1``,
# all masses equal to 1, period ``2 pi``) and rescaled to the code units at run
# time, so the numbers stay an exact solution whatever ``G`` is worth in the
# unit system chosen above.

CHOREOGRAPHIES = {
    "figure_eight": {
        "name": "Figure eight (Chenciner-Montgomery)",
        "n_periods": 10,
        "eta_sink": 0.2,
        "lyapunov_rate": 0.0000,
        "loop_radius": 1.0761,
        # action 24.371926, marginally stable (all Floquet multipliers on the unit circle)
        # with the settings above the sinks stay within 2.0e-04 of the loop
        "positions": [
            (-0.4641985800155460, 0.6138076008958897, 0.0),
            (0.2234329805895614, 0.4300831954951171, 0.0),
            (0.2407655994259845, -1.0438907963910067, 0.0),
        ],
        "velocities": [
            (-0.1160752246112903, 0.9899453509452854, 0.0),
            (-0.3292092683482385, -1.1661479996669719, 0.0),
            (0.4452844929595292, 0.1762026487216866, 0.0),
        ],
    },
    "super_eight": {
        "name": "Super eight",
        "n_periods": 1,
        "eta_sink": 0.05,
        "lyapunov_rate": 5.1177,
        "loop_radius": 1.3085,
        # action 44.437886, Floquet rate 5.12 e-folds/period
        # with the settings above the sinks stay within 3.3e-04 of the loop
        "positions": [
            (-0.4038762220290363, 0.1985692946429415, 0.0),
            (0.7203687816847304, -0.8624679668854116, 0.0),
            (0.7932678669074159, 0.2521451538161689, 0.0),
            (-1.1097604265631100, 0.4117535184263010, 0.0),
        ],
        "velocities": [
            (0.3911911293788881, -1.4805596570053108, 0.0),
            (0.8359875347674515, 0.2867539941403371, 0.0),
            (-0.8285678288275163, 0.5243804169641790, 0.0),
            (-0.3986108353188244, 0.6694252459007946, 0.0),
        ],
    },
    "eight_5body": {
        "name": "5-body eight",
        "n_periods": 1,
        "eta_sink": 0.05,
        "lyapunov_rate": 7.6970,
        "loop_radius": 1.4487,
        # action 71.331244, Floquet rate 7.70 e-folds/period
        # with the settings above the sinks stay within 2.3e-03 of the loop
        "positions": [
            (0.2215908046378909, -0.6350176261811384, 0.0),
            (1.2576902492880151, -0.7065908483760491, 0.0),
            (0.3649016836300293, 0.0666152724828391, 0.0),
            (-1.2375252914976229, 0.3160127872388074, 0.0),
            (-0.6066574460583124, 0.9589804148355410, 0.0),
        ],
        "velocities": [
            (0.7949249141897033, -1.1864526129990527, 0.0),
            (0.3395311300681252, 0.8572854748586699, 0.0),
            (-1.6567226783459534, -0.1584554434231686, 0.0),
            (-0.4266036259046186, 0.8584685817681540, 0.0),
            (0.9488702599927430, -0.3708460002046013, 0.0),
        ],
    },
    "eight_6body": {
        "name": "6-body eight",
        "n_periods": 1,
        "eta_sink": 0.05,
        "lyapunov_rate": 8.3237,
        "loop_radius": 1.6305,
        # action 102.122413, Floquet rate 8.32 e-folds/period
        # with the settings above the sinks stay within 3.0e-03 of the loop
        "positions": [
            (0.2946906470198442, 0.2519448235925313, 0.0),
            (-1.1298455676762773, -0.2504790196314109, 0.0),
            (-0.8814355932018649, -1.2375037521685581, 0.0),
            (0.1128667303274771, -0.7549733756783465, 0.0),
            (0.3778852015783642, 0.9719036206631373, 0.0),
            (1.2258385819524564, 1.0191077032226468, 0.0),
        ],
        "velocities": [
            (-1.9270663260494580, -0.0320072451442911, 0.0),
            (-0.6502687820738448, -0.9795639420092696, 0.0),
            (1.0024889379300952, -0.4312586095875412, 0.0),
            (0.5746722079213887, 1.2056613152748474, 0.0),
            (0.7006883972450532, 1.1593017698296983, 0.0),
            (0.2994855650267662, -0.9221332883634445, 0.0),
        ],
    },
    "chain_5body": {
        "name": "5-body chain (3 lobes)",
        "n_periods": 1,
        "eta_sink": 0.05,
        "lyapunov_rate": 8.3182,
        "loop_radius": 1.5921,
        # action 77.158798, Floquet rate 8.32 e-folds/period
        # with the settings above the sinks stay within 1.2e-03 of the loop
        "positions": [
            (1.5501268965712847, 0.3165780279345012, 0.0),
            (0.1316949649277918, 0.4614685043380578, 0.0),
            (-1.2608614614839651, -0.5300925068401681, 0.0),
            (-1.0101422454340585, -0.1114718253911439, 0.0),
            (0.5891818454188571, -0.1364822000412938, 0.0),
        ],
        "velocities": [
            (-0.0676444505851886, -0.5004410380847638, 0.0),
            (-1.2955086697715756, 0.0345789158403192, 0.0),
            (-1.1825482247170074, -0.0817012261699769, 0.0),
            (1.5620316235951079, -0.5781134391746547, 0.0),
            (0.9836697214786655, 1.1256767875890763, 0.0),
        ],
    },
    "chain_6body": {
        "name": "6-body chain (3 lobes)",
        "n_periods": 1,
        "eta_sink": 0.05,
        "lyapunov_rate": 7.3180,
        "loop_radius": 1.7161,
        # action 108.992084, Floquet rate 7.32 e-folds/period
        # with the settings above the sinks stay within 1.2e-03 of the loop
        "positions": [
            (1.6980754242677578, 0.2477812743068945, 0.0),
            (0.7062301828117237, -0.3261918346504651, 0.0),
            (-0.5993567401178459, -0.5053188731756981, 0.0),
            (-1.6980754242677616, -0.2477812743068684, 0.0),
            (-0.7062301828117281, 0.3261918346504578, 0.0),
            (0.5993567401178520, 0.5053188731756920, 0.0),
        ],
        "velocities": [
            (-0.0707970207882384, 0.4205534048866357, 0.0),
            (-1.1040590678585802, -1.0540224000301939, 0.0),
            (-1.3564358415027034, 0.7193408344334531, 0.0),
            (0.0707970207882118, -0.4205534048866399, 0.0),
            (1.1040590678585944, 1.0540224000301963, 0.0),
            (1.3564358415027178, -0.7193408344334518, 0.0),
        ],
    },
    "chain_6body_4": {
        "name": "6-body chain (4 lobes)",
        "n_periods": 1,
        "eta_sink": 0.05,
        "lyapunov_rate": 12.0551,
        "loop_radius": 1.7963,
        # action 110.912199, Floquet rate 12.06 e-folds/period
        # with the settings above the sinks stay within 1.1e-02 of the loop
        "positions": [
            (0.3324057567640413, -0.4221818055287314, 0.0),
            (-0.9787889547282376, 0.6129622645144733, 0.0),
            (-1.6525101512791298, 0.0877542342593324, 0.0),
            (-0.4564897675956794, 0.0117810250116186, 0.0),
            (1.0188486460036430, -0.0906968480775526, 0.0),
            (1.7365344708353800, -0.1996188701791442, 0.0),
        ],
        "velocities": [
            (-1.3230262316539578, 0.2325527103673017, 0.0),
            (-1.2181955452272395, 0.4270313905229277, 0.0),
            (0.3491818664830734, -0.9243766727519219, 0.0),
            (1.5465732707638118, 1.0035236963101388, 0.0),
            (1.0410592730630861, -1.3230871450665727, 0.0),
            (-0.3955926334287724, 0.5843560206181266, 0.0),
        ],
    },
    "triangle_ring": {
        "name": "Lagrange equilateral triangle",
        "n_periods": 3,
        "eta_sink": 0.2,
        "lyapunov_rate": 4.4429,
        "loop_radius": 0.8327,
        # action 19.604328, Floquet rate 4.44 e-folds/period
        # with the settings above the sinks stay within 1.7e-04 of the loop
        "positions": [
            (0.1139333665692508, -0.8248517820389498, 0.0),
            (-0.7713092808872208, 0.3137567012318189, 0.0),
            (0.6573759143179698, 0.5110950808071312, 0.0),
        ],
        "velocities": [
            (-0.8248517820389499, -0.1139333665692510, 0.0),
            (0.3137567012318188, 0.7713092808872208, 0.0),
            (0.5110950808071313, -0.6573759143179697, 0.0),
        ],
    },
    "square_ring": {
        "name": "Square ring",
        "n_periods": 3,
        "eta_sink": 0.2,
        "lyapunov_rate": 5.4006,
        "loop_radius": 0.9855,
        # action 36.613230, Floquet rate 5.40 e-folds/period
        # with the settings above the sinks stay within 1.0e-04 of the loop
        "positions": [
            (-0.7892313641489256, 0.5901778984304857, 0.0),
            (0.5901778984304857, 0.7892313641489251, 0.0),
            (0.7892313641489257, -0.5901778984304856, 0.0),
            (-0.5901778984304857, -0.7892313641489251, 0.0),
        ],
        "velocities": [
            (0.5901778984304860, 0.7892313641489250, 0.0),
            (0.7892313641489256, -0.5901778984304856, 0.0),
            (-0.5901778984304857, -0.7892313641489251, 0.0),
            (-0.7892313641489258, 0.5901778984304856, 0.0),
        ],
    },
    "pentagon_ring": {
        "name": "Pentagon ring",
        "n_periods": 3,
        "eta_sink": 0.2,
        "lyapunov_rate": 5.9007,
        "loop_radius": 1.1124,
        # action 58.308755, Floquet rate 5.90 e-folds/period
        # with the settings above the sinks stay within 8.6e-05 of the loop
        "positions": [
            (-0.6325360556022823, -0.9150127634996795, 0.0),
            (-1.0656932419556084, 0.3188230434807982, 0.0),
            (-0.0260985895073492, 1.1120562407674983, 0.0),
            (1.0495634265816352, 0.3684655107149524, 0.0),
            (0.6747644604836046, -0.8843320314635693, 0.0),
        ],
        "velocities": [
            (-0.9150127634996793, 0.6325360556022818, 0.0),
            (0.3188230434807986, 1.0656932419556084, 0.0),
            (1.1120562407674985, 0.0260985895073496, 0.0),
            (0.3684655107149521, -1.0495634265816349, 0.0),
            (-0.8843320314635698, -0.6747644604836049, 0.0),
        ],
    },
    "hexagon_ring": {
        "name": "Hexagon ring",
        "n_periods": 3,
        "eta_sink": 0.2,
        "lyapunov_rate": 6.2163,
        "loop_radius": 1.2226,
        # action 84.522094, Floquet rate 6.22 e-folds/period
        # with the settings above the sinks stay within 6.9e-05 of the loop
        "positions": [
            (-0.0460441294024495, -1.2217032021252754, 0.0),
            (-1.0810480736265080, -0.5709762153049786, 0.0),
            (-1.0350039442240588, 0.6507269868202968, 0.0),
            (0.0460441294024494, 1.2217032021252754, 0.0),
            (1.0810480736265078, 0.5709762153049789, 0.0),
            (1.0350039442240593, -0.6507269868202962, 0.0),
        ],
        "velocities": [
            (-1.2217032021252756, 0.0460441294024493, 0.0),
            (-0.5709762153049789, 1.0810480736265080, 0.0),
            (0.6507269868202967, 1.0350039442240586, 0.0),
            (1.2217032021252758, -0.0460441294024495, 0.0),
            (0.5709762153049792, -1.0810480736265076, 0.0),
            (-0.6507269868202964, -1.0350039442240591, 0.0),
        ],
    },
}


# %%
# Rescale a normalised choreography to the code units
def scale_choreography(choreography, length_scale=1.0, sink_mass=1.0):
    """Convert a normalised choreography to code units.

    The tabulated loops use ``G = 1``, unit masses and a period of ``2 pi``.
    Scaling lengths by ``length_scale`` and masses by ``sink_mass`` forces the
    time unit to ``sqrt(length_scale**3 / (G * sink_mass))``; deriving it from
    ``G`` rather than hardcoding it is what keeps the tabulated numbers an
    exact solution of the equations the code actually integrates.
    """
    time_scale = np.sqrt(length_scale**3 / (G * sink_mass))
    velocity_scale = length_scale / time_scale

    positions = [tuple(length_scale * np.array(p)) for p in choreography["positions"]]
    velocities = [tuple(velocity_scale * np.array(v)) for v in choreography["velocities"]]

    return positions, velocities, 2 * np.pi * time_scale


# %%
# Run a choreography and plot the resulting trajectories
def run_choreography(key, length_scale=1.0, sink_mass=1.0, eta_sink=None, max_plot_points=2000):
    choreography = CHOREOGRAPHIES[key]
    # Each entry carries its own eta_sink: the sink timestep has to shrink as
    # the Floquet rate grows, otherwise the orbit leaves the loop within the
    # plotted window. n_periods is chosen the same way.
    if eta_sink is None:
        eta_sink = choreography["eta_sink"]
    positions, velocities, period = scale_choreography(choreography, length_scale, sink_mass)
    nsink = len(positions)
    loop_radius = choreography["loop_radius"] * length_scale

    # The sinks pass within a fraction of the loop radius of each other. The
    # accretion radius only gates the sink-gas force and there is no gas here,
    # but keeping it well below the closest approach avoids any confusion.
    accretion_radius = 0.01 * loop_radius

    ctx, model = build_sink_sph_model(
        positions=positions,
        velocities=velocities,
        masses=[sink_mass] * nsink,
        accretion_radii=[accretion_radius] * nsink,
        # generous box: once an unstable choreography breaks up the sinks
        # wander well outside the loop
        box_extent=3.0 * loop_radius,
        eta_sink=eta_sink,
        show_cfl_detail=False,
    )

    snapshots = run_sim(model, choreography["n_periods"] * period, use_dt=None)

    # these runs take many steps per period, so thin the trajectory before plotting
    stride = max(1, len(snapshots) // max_plot_points)
    nper = choreography["n_periods"]
    plot_orbit_trajectory(
        snapshots[::stride],
        "{} ({} sinks, {} period{})".format(
            choreography["name"], nsink, nper, "s" if nper > 1 else ""
        ),
    )

    return snapshots


# %%
# Figure eight
# ------------
# The one choreography here that is marginally stable (every Floquet multiplier
# sits on the unit circle), so it can be run for many periods without drifting
# off the loop.
snapshots = run_choreography("figure_eight")

# %%
# Super eight (4 bodies)
snapshots = run_choreography("super_eight")

# %%
# 5-body eight
snapshots = run_choreography("eight_5body")

# %%
# 6-body eight
snapshots = run_choreography("eight_6body")

# %%
# 5-body chain
snapshots = run_choreography("chain_5body")

# %%
# 6-body chain (3 lobes)
snapshots = run_choreography("chain_6body")

# %%
# 6-body chain (4 lobes)
# The most unstable orbit of the set, at 12 e-folds per period, so it is only
# run for a single period.
snapshots = run_choreography("chain_6body_4")

# %%
# Lagrange equilateral triangle
# -----------------------------
# The simplest choreography of all: N equal masses on a circle, each one
# following the same orbit a fraction of a period behind the previous one.
snapshots = run_choreography("triangle_ring")

# %%
# Square ring (4 bodies)
snapshots = run_choreography("square_ring")

# %%
# Pentagon ring (5 bodies)
snapshots = run_choreography("pentagon_ring")

# %%
# Hexagon ring (6 bodies)
snapshots = run_choreography("hexagon_ring")


# %%
# Comparing the drift with the growth predicted by the Floquet rate
# =================================================================
#
# ``lyapunov_rate`` in the table above is the largest Floquet exponent of the
# orbit, in e-folds per period, taken from the monodromy matrix of the refined
# solution. It is a property of the orbit and not of the solver, so it predicts
# how fast *any* integration error gets amplified,
#
# .. math::
#     E(t) \simeq E_0 \, e^{\lambda t}
#
# and the measured drift can be checked against it without fitting the rate.
# Running the same initial conditions through a high order reference integrator
# isolates the integration error from any error in the initial conditions, and
# in the plots below only the offset of the dashed line is fitted: its slope is
# fixed by ``lyapunov_rate``.
#
# The three cases are deliberately different:
#
# - the super eight follows the predicted growth over about six decades, until
#   the orbit breaks up and the comparison saturates;
# - the figure eight has ``lyapunov_rate = 0``, so nothing grows exponentially
#   and its error just wanders around a small value for ten periods;
# - the hexagon ring drifts far more slowly than its rate allows. Its error
#   does eventually grow at exactly that rate, but the leapfrog error happens
#   to overlap the unstable eigenvector only weakly, which delays the onset by
#   a couple of periods.


def reference_solution(positions, velocities, sink_mass, tmax, stop_separation):
    """Integrate the same initial conditions to near machine precision.

    Once an unstable choreography has broken up the sinks start having close
    encounters, which costs the reference integrator an enormous number of
    steps for a stretch that is past the point of being interesting anyway.
    ``stop_separation`` ends the integration there; it is set below the closest
    approach of every intact orbit, so it only ever triggers after break-up.
    """
    from scipy.integrate import solve_ivp

    nsink = len(positions)
    masses = np.full(nsink, sink_mass)

    def separations(state):
        pos = state[: 3 * nsink].reshape(nsink, 3)
        sep = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r_squared = np.sum(sep**2, axis=-1)
        np.fill_diagonal(r_squared, np.inf)
        return sep, r_squared

    def rhs(_t, state):
        sep, r_squared = separations(state)
        acc = G * np.einsum("ij,ijk,j->ik", r_squared ** (-1.5), sep, masses)
        return np.concatenate([state[3 * nsink :], acc.ravel()])

    def close_approach(_t, state):
        _, r_squared = separations(state)
        return np.sqrt(np.min(r_squared)) - stop_separation

    close_approach.terminal = True
    close_approach.direction = -1

    state0 = np.concatenate([np.asarray(positions).ravel(), np.asarray(velocities).ravel()])
    return solve_ivp(
        rhs,
        (0.0, tmax),
        state0,
        method="DOP853",
        rtol=3e-14,
        atol=1e-16,
        dense_output=True,
        events=close_approach,
    )


# %%
# Run a choreography and record how far it drifts from the reference
def measure_choreography_deviation(key, n_periods, length_scale=1.0, sink_mass=1.0):
    choreography = CHOREOGRAPHIES[key]
    positions, velocities, period = scale_choreography(choreography, length_scale, sink_mass)
    nsink = len(positions)
    loop_radius = choreography["loop_radius"] * length_scale

    ctx, model = build_sink_sph_model(
        positions=positions,
        velocities=velocities,
        masses=[sink_mass] * nsink,
        accretion_radii=[0.01 * loop_radius] * nsink,
        box_extent=3.0 * loop_radius,
        eta_sink=choreography["eta_sink"],
        show_cfl_detail=False,
    )

    tmax = n_periods * period
    reference = reference_solution(
        positions, velocities, sink_mass, 1.02 * tmax, 0.05 * loop_radius
    )

    times = []
    deviations = []
    current_time = 0.0
    while current_time < tmax:
        model.timestep()
        current_time = model.get_time()
        if current_time > reference.t[-1]:
            break
        pos = np.array([sink["pos"] for sink in model.get_sinks()])
        exact = reference.sol(current_time)[: 3 * nsink].reshape(nsink, 3)
        deviation = float(np.max(np.linalg.norm(pos - exact, axis=1)))
        times.append(current_time / period)
        deviations.append(deviation)
        if deviation > 0.5 * loop_radius:
            # the orbit has left the choreography for good; past this point the
            # comparison says nothing and the timestep collapses into the close
            # encounters of the break-up, which is expensive for no benefit
            break

    return np.array(times), np.array(deviations), choreography, loop_radius


# %%
# Plot the measured drift next to the predicted growth
def plot_choreography_deviation(runs, ncols=4):
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.0 * nrows), squeeze=False)
    for unused in axes.ravel():
        unused.set_visible(False)

    for index, (times, deviations, choreography, loop_radius) in enumerate(runs):
        ax = axes[index // ncols][index % ncols]
        ax.set_visible(True)
        rate = choreography["lyapunov_rate"]
        ax.semilogy(times, np.maximum(deviations, 1e-18), lw=1.4, label="Shamrock")

        # the slope is fixed by the Floquet rate, only the offset is fitted,
        # over the stretch that is above round-off and below saturation
        window = (deviations > 1e-6) & (deviations < 0.05 * loop_radius)
        note = ""
        if rate > 0 and np.count_nonzero(window) > 10:
            offset = np.exp(np.median(np.log(deviations[window]) - rate * times[window]))
            predicted = offset * np.exp(rate * times)
            # only claim agreement when the fixed slope really does describe the
            # measurement, otherwise the unstable mode has not taken over inside
            # the window and drawing the line would be misleading
            spread = np.max(np.abs(np.log(deviations[window] / predicted[window])))
            if spread < np.log(30.0):
                ax.semilogy(times, predicted, "k--", lw=1.2, label=r"$\propto e^{\lambda t}$")
            else:
                note = "\nunstable mode not yet dominant"

        ax.set_xlabel("time / period")
        ax.set_ylim(1e-10, 10.0 * loop_radius)
        ax.set_xlim(0.0, times[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(
            "{}\n$\\lambda$ = {:.2f} e-folds/period{}".format(choreography["name"], rate, note),
            fontsize=9,
        )
        if index % ncols == 0:
            ax.set_ylabel("deviation from the reference (AU)")

    fig.tight_layout()
    plt.show()


# %%
# The number of periods is chosen per orbit: just past the point where the
# unstable ones break up and the comparison saturates, and long enough for the
# marginally stable figure eight to show that it does not.
plot_choreography_deviation(
    [
        measure_choreography_deviation("figure_eight", 10),
        measure_choreography_deviation("super_eight", 3.5),
        measure_choreography_deviation("eight_5body", 2.5),
        measure_choreography_deviation("eight_6body", 2.5),
        measure_choreography_deviation("chain_5body", 2.5),
        measure_choreography_deviation("chain_6body", 2.5),
        measure_choreography_deviation("chain_6body_4", 2.0),
        measure_choreography_deviation("triangle_ring", 5),
        measure_choreography_deviation("square_ring", 5),
        measure_choreography_deviation("pentagon_ring", 5),
        measure_choreography_deviation("hexagon_ring", 5),
    ]
)
