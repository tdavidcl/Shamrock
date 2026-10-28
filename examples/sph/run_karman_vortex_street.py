"""
Karman vortex street behind a cylinder in SPH
=============================================

A uniform subsonic flow past a cylinder sheds alternating vortices (Karman vortex street).

The cylinder is an immersed obstacle made with volume penalization: the velocity of the particles
inside it is relaxed toward zero (here set to zero directly, ``eta = 0``), so these particles are
frozen and act as wall particles.

SPH only has free, periodic and shearing periodic boundaries, so the box is periodic in every
direction and the inflow is made by a fringe region: penalization zones just before the right
edge of the box relax the gas toward the upstream state, which removes the wake before it
re-enters the box through the periodic boundary.

The force on the cylinder is the momentum its penalization zone removes from the gas, which gives
the drag and lift coefficients.
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Setup parameters

kernel = "M6"  # SPH kernel to use

# upstream state, the sound speed is 1
gamma = 5.0 / 3.0
rho_inf = 1.0
P_inf = 1.0 / gamma
mach = 0.3
U_inf = mach * np.sqrt(gamma * P_inf / rho_inf)
u_inf = P_inf / ((gamma - 1) * rho_inf)

# cylinder (axis along z)
D = 0.25
x_cyl = 1.0
eta_cyl = 0.0

# resolution: D / dr particle spacings across the cylinder
# (D / dr = 8 keeps the documentation build short, D / dr = 12 resolves the wake better)
D_over_dr = int(os.environ.get("KARMAN_D_OVER_DR", "8"))
dr = D / D_over_dr

# periodic box of about 4 x 2 in the xy plane, 6 particle layers in z
nx = round(4.0 / dr)
ny = 2 * round(2.0 / (dr * np.sqrt(3) / 2) / 2)
nz = 6

# fringe region at the right of the box, relaxing toward the upstream state with a strength
# that ramps up toward the box edge (zones are applied one after the other, so overlapping zones
# with the same target add their rates)
fringe_width = 0.5
fringe_etas = [1.0, 0.3, 0.1]

tmax = float(os.environ.get("KARMAN_TMAX", "60"))
n_frames = int(os.environ.get("KARMAN_NFRAMES", "60"))

sim_folder = os.environ.get("KARMAN_FOLDER", "_to_trash/sph_karman/")

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)

# %%
# Setup the solver
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

# the lattice functions take half the particle spacing
(Lx, Ly, Lz) = model.get_box_dim_fcc_3d(dr / 2, nx, ny, nz)
y_cyl = Ly / 2
z_mid = Lz / 2
x_fringe = Lx - fringe_width

# vertical velocity probe in the wake
probe = (x_cyl + 4 * D, y_cyl, z_mid)

print(f"box = {Lx:.4f} x {Ly:.4f} x {Lz:.4f}, N = {nx * ny * nz}, D/dr = {D / dr}")
print(f"U_inf = {U_inf}, Mach = {mach}")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)

# immersed cylinder: the particles inside are frozen
cfg.add_penalization_cylinder(
    center=(x_cyl, y_cyl, 0.0),
    axis=(0.0, 0.0, 1.0),
    radius=D / 2,
    eta=eta_cyl,
    target_vel=(0.0, 0.0, 0.0),
)

# fringe region: velocity and specific internal energy relaxed toward the upstream state
for i, eta in enumerate(fringe_etas):
    x_start = x_fringe + i * fringe_width / len(fringe_etas)
    cfg.add_penalization_box(
        lower=(x_start, -1.0, -1.0),
        upper=(Lx + 1.0, Ly + 1.0, Lz + 1.0),
        eta=eta,
        target_vel=(U_inf, 0.0, 0.0),
        target_uint=u_inf,
    )

cfg.print_status()
model.set_solver_config(cfg)

# Set scheduler criteria to effectively disable patch splitting and merging.
model.init_scheduler(int(1e9), 1)

model.resize_simulation_box((0, 0, 0), (Lx, Ly, Lz))
model.add_cube_fcc_3d(dr / 2, (0, 0, 0), (Lx, Ly, Lz))

model.set_value_in_a_box("uint", "f64", u_inf, (0, 0, 0), (Lx, Ly, Lz))


# %%
# Initial conditions: uniform flow, at rest in the cylinder, with a small transverse kick in the
# near wake to break the up-down symmetry and start the shedding earlier
def vel_func(r):
    x, y, z = r
    if (x - x_cyl) ** 2 + (y - y_cyl) ** 2 <= (D / 2) ** 2:
        return (0.0, 0.0, 0.0)
    vy = 0.0
    if x > x_cyl and x < x_cyl + 4 * D and y > y_cyl:
        vy = 0.1 * U_inf
    return (U_inf, vy, 0.0)


model.set_field_value_lambda_f64_3("vxyz", vel_func)

pmass = model.total_mass_to_part_mass(rho_inf * Lx * Ly * Lz)
model.set_particle_mass(pmass)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

# %%
# Plotting utilities: vorticity from the SPH interpolated velocity in the midplane
nx_render, ny_render = 2 * nx, round(2 * nx * Ly / Lx)
dx_render = Lx / nx_render
dy_render = Ly / ny_render
x_render = (np.arange(nx_render) + 0.5) * dx_render
y_render = (np.arange(ny_render) + 0.5) * dy_render


def render_vorticity():
    vel = model.render_cartesian_slice(
        "vxyz",
        "f64_3",
        center=(Lx / 2, Ly / 2, z_mid),
        delta_x=(Lx, 0.0, 0.0),
        delta_y=(0.0, Ly, 0.0),
        nx=nx_render,
        ny=ny_render,
    )
    vel = np.array(vel)
    dvy_dx = np.gradient(vel[:, :, 1], dx_render, axis=1)
    dvx_dy = np.gradient(vel[:, :, 0], dy_render, axis=0)
    return dvy_dx - dvx_dy


# note: render_cartesian_slice does not wrap around the periodic boundaries, so the velocity
# within a smoothing length of y = 0 and y = Ly is underestimated, which shows up as thin
# spurious vorticity bands along these edges
def plot_vorticity(t, iplot):
    omega = render_vorticity()
    omega_max = 2 * U_inf / D

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    res = ax.imshow(
        omega,
        cmap="RdBu_r",
        origin="lower",
        extent=[0, Lx, 0, Ly],
        vmin=-omega_max,
        vmax=omega_max,
    )
    ax.add_patch(plt.Circle((x_cyl, y_cyl), D / 2, fill=False, color="black"))
    ax.axvspan(x_fringe, Lx, color="grey", alpha=0.3)
    ax.plot(probe[0], probe[1], "x", color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t = {t:0.2f}")
    cbar = fig.colorbar(res, ax=ax, extend="both", shrink=0.7)
    cbar.set_label(r"$\omega_z$")
    fig.savefig(os.path.join(sim_folder, f"vorticity_{iplot:04d}.png"))
    plt.close(fig)


# %%
# Run the simulation, recording every timestep the force on the cylinder and the transverse
# velocity at the probe
hist_t = []
hist_force = []
hist_probe_vy = []


def record():
    hist_t.append(model.get_time() + model.get_dt())
    hist_force.append(model.get_penalization_forces()[0])
    hist_probe_vy.append(model.render_slice("vxyz", "f64_3", [probe])[0][1])


model.add_timestep_callback(step_end=record)

for i, t in enumerate(np.linspace(0, tmax, n_frames + 1)):
    model.evolve_until(t)
    plot_vorticity(t, i)

hist_t = np.array(hist_t)
hist_force = np.array(hist_force)
hist_probe_vy = np.array(hist_probe_vy)

np.savez(
    os.path.join(sim_folder, "history.npz"),
    t=hist_t,
    force=hist_force,
    probe_vy=hist_probe_vy,
)

# %%
# Drag and lift coefficients, normalised by the dynamic pressure on the cylinder section of the
# slab (the force is noisy step to step, it is shown averaged over windows of 0.5)
q_dyn = 0.5 * rho_inf * U_inf**2 * D * Lz
c_d = hist_force[:, 0] / q_dyn
c_l = hist_force[:, 1] / q_dyn


def window_average(t, f, width):
    edges = np.arange(t[0], t[-1] + width, width)
    idx = np.digitize(t, edges)
    tc = np.array([t[idx == k].mean() for k in np.unique(idx)])
    fc = np.array([f[idx == k].mean() for k in np.unique(idx)])
    return tc, fc


t_avg, c_d_avg = window_average(hist_t, c_d, 0.5)
_, c_l_avg = window_average(hist_t, c_l, 0.5)

# %%
# Shedding frequency: the Strouhal number St = f D / U_inf is close to 0.2 for a cylinder.
#
# The period is the mean time between upward zero crossings of a signal, which is much more
# precise than the peak of a spectrum over a few periods. It is measured both on the probe
# velocity and on the (window averaged) lift. The lift is noisy, so a crossing only counts once
# the signal went below -0.3 standard deviation since the previous one (hysteresis), otherwise
# the noise around zero gives spurious double crossings.
t_start = tmax / 2


def strouhal_from_crossings(t, f):
    mask = t >= t_start
    t_fit = t[mask]
    f_fit = f[mask] - f[mask].mean()
    threshold = 0.3 * f_fit.std()

    t_cross = []
    armed = False
    for i in range(len(f_fit) - 1):
        if f_fit[i] < -threshold:
            armed = True
        if armed and f_fit[i] < 0 <= f_fit[i + 1]:
            dt = t_fit[i + 1] - t_fit[i]
            t_cross.append(t_fit[i] - f_fit[i] * dt / (f_fit[i + 1] - f_fit[i]))
            armed = False
    t_cross = np.array(t_cross)

    if len(t_cross) < 2:
        return np.nan, t_cross
    period = np.mean(np.diff(t_cross))
    return D / (period * U_inf), t_cross


st_probe, t_cross_probe = strouhal_from_crossings(hist_t, hist_probe_vy)
st_lift, t_cross_lift = strouhal_from_crossings(t_avg, c_l_avg)

mean_cd = c_d[hist_t >= t_start].mean()
print(f"Strouhal number (probe v_y) = {st_probe:.3f}, from {len(t_cross_probe)} crossings")
print(f"Strouhal number (lift)      = {st_lift:.3f}, from {len(t_cross_lift)} crossings")
print(f"mean drag coefficient C_D = {mean_cd:.3f}")

fig, axs = plt.subplots(3, 1, figsize=(7, 7), dpi=125, sharex=True)
axs[0].plot(hist_t, hist_probe_vy / U_inf)
axs[0].plot(
    t_cross_probe, np.zeros_like(t_cross_probe), "o", color="black", label=f"St = {st_probe:.3f}"
)
axs[0].set_ylabel(r"$v_y / U_\infty$ at probe")
axs[0].legend()
axs[1].plot(t_avg, c_l_avg)
axs[1].plot(
    t_cross_lift, np.zeros_like(t_cross_lift), "o", color="black", label=f"St = {st_lift:.3f}"
)
axs[1].set_ylabel(r"$C_L$")
axs[1].legend()
axs[2].plot(t_avg, c_d_avg)
axs[2].set_ylabel(r"$C_D$")
axs[2].set_xlabel("t")
for ax in axs:
    ax.axvline(t_start, color="grey", ls="--")
fig.tight_layout()
fig.savefig(os.path.join(sim_folder, "history.png"))
plt.show()

# %%
# Vorticity animation
from shamrock.utils.plot import show_image_sequence

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(os.path.join(sim_folder, "vorticity_*.png"), render_gif=True)

if shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(os.path.join(sim_folder, "vorticity.gif"), writer=writer)

plt.show()
