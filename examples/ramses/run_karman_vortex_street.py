"""
Karman vortex street behind a cylinder in RAMSES solver
======================================================

A uniform subsonic flow past a cylinder sheds alternating vortices (Karman vortex street).

The cylinder is an immersed obstacle made with volume penalization: inside the cylinder the gas
velocity is relaxed toward zero on a short timescale ``eta``. The inflow is a penalization band at
the left of the box that relaxes the gas toward the upstream state, next to an outflow boundary.
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

# upstream state, the sound speed is 1
gamma = 5.0 / 3.0
rho_inf = 1.0
P_inf = 1.0 / gamma
mach = 0.3
U_inf = mach * np.sqrt(gamma * P_inf / rho_inf)

# box [0, Lx] x [0, Ly] x [0, Lz], periodic in z
#
# The base grid is made of cubic patches whose side is the gcd of the block counts, so the block
# counts are (2k, k, k) to get a 2 x 1 x 1 patch grid (a thin box in z would give one patch per
# block in the xy plane and make the solver very slow)
Lx = 4.0
Ly = 2.0
Lz = 2.0

# cylinder (axis along z)
D = 0.25
x_cyl = 1.0
y_cyl = Ly / 2
eta_cyl = 1e-3

# inflow forcing band at the left of the box
x_inflow = 0.125
eta_inflow = 1e-2

# resolution: blocks of 2^3 cells
sz = 2
k = 32
nblock_x = 2 * k
nblock_y = k
nblock_z = k

scale_fact = Lx / (sz * nblock_x)
dx = scale_fact

tmax = 40.0
n_frames = 80

# vertical velocity probe in the wake
probe = (x_cyl + 4 * D, y_cyl, Lz / 2)

sim_folder = "_to_trash/ramses_karman/"

print(f"U_inf = {U_inf}, Mach = {mach}, D/dx = {D / dx}")

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)

# %%
# Setup the solver
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(gamma)
cfg.set_riemann_solver_hllc()
cfg.set_slope_lim_vanleer_sym()
cfg.set_face_time_interpolation(True)

cfg.set_boundary_condition("x", "outflow")
cfg.set_boundary_condition("y", "periodic")
cfg.set_boundary_condition("z", "periodic")

# immersed cylinder: the velocity is relaxed toward 0 inside it
cfg.add_penalization_cylinder(
    center=(x_cyl, y_cyl, 0.0),
    axis=(0.0, 0.0, 1.0),
    radius=D / 2,
    eta=eta_cyl,
    target_vel=(0.0, 0.0, 0.0),
)

# inflow: the whole state is relaxed toward the upstream one, the outflow boundary next to it
# then copies this state in the ghost cells
cfg.add_penalization_box(
    lower=(-1.0, -1.0, -1.0),
    upper=(x_inflow, Ly + 1.0, Lz + 1.0),
    eta=eta_inflow,
    target_vel=(U_inf, 0.0, 0.0),
    target_rho=rho_inf,
    target_press=P_inf,
)

model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (nblock_x, nblock_y, nblock_z))


# %%
# Initial conditions: uniform flow, with a small transverse kick in the wake to break the up-down
# symmetry and start the shedding earlier
def vel_map(rmin, rmax):
    x, y, z = rmin
    vy = 0.0
    if x > x_cyl and x < x_cyl + 4 * D and y > y_cyl:
        vy = 0.1 * U_inf
    return (U_inf, vy, 0.0)


def rho_map(rmin, rmax):
    return rho_inf


def rhovel_map(rmin, rmax):
    vx, vy, vz = vel_map(rmin, rmax)
    return (rho_inf * vx, rho_inf * vy, rho_inf * vz)


def rhoetot_map(rmin, rmax):
    vx, vy, vz = vel_map(rmin, rmax)
    return 0.5 * rho_inf * (vx * vx + vy * vy + vz * vz) + P_inf / (gamma - 1)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)

# %%
# Plotting utilities: the vorticity is computed from a velocity slice in the midplane
nx_render, ny_render = 512, 256
eps = 1e-6
x_render = np.linspace(eps, Lx - eps, nx_render)
y_render = np.linspace(eps, Ly - eps, ny_render)
xx, yy = np.meshgrid(x_render, y_render)
positions = [(x, y, probe[2]) for x, y in zip(xx.ravel(), yy.ravel())]


def render_vorticity():
    rho = np.array(model.render_slice("rho", "f64", positions)).reshape(ny_render, nx_render)
    rhovel = np.array(model.render_slice("rhovel", "f64_3", positions)).reshape(
        ny_render, nx_render, 3
    )
    vx = rhovel[:, :, 0] / rho
    vy = rhovel[:, :, 1] / rho
    dvy_dx = np.gradient(vy, x_render, axis=1)
    dvx_dy = np.gradient(vx, y_render, axis=0)
    return dvy_dx - dvx_dy


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
    ax.add_patch(plt.Circle((x_cyl, y_cyl), D / 2, color="black"))
    ax.axvspan(0, x_inflow, color="grey", alpha=0.3)
    ax.plot(probe[0], probe[1], "x", color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t = {t:0.2f}")
    cbar = fig.colorbar(res, ax=ax, extend="both", shrink=0.7)
    cbar.set_label(r"$\omega_z$")
    fig.savefig(os.path.join(sim_folder, f"vorticity_{iplot:04d}.png"))
    plt.close(fig)


# %%
# Run the simulation, recording the transverse velocity at the probe every timestep
probe_t = []
probe_vy = []


def record_probe():
    rho = model.render_slice("rho", "f64", [probe])[0]
    rhovel = model.render_slice("rhovel", "f64_3", [probe])[0]
    probe_t.append(model.get_time())
    probe_vy.append(rhovel[1] / rho)


model.add_timestep_callback(step_end=record_probe)

for i, t in enumerate(np.linspace(0, tmax, n_frames + 1)):
    model.evolve_until(t)
    plot_vorticity(t, i)

# %%
# Shedding frequency: the Strouhal number St = f D / U_inf is close to 0.2 for a cylinder
probe_t = np.array(probe_t)
probe_vy = np.array(probe_vy)

# only keep the established shedding regime and resample uniformly for the FFT
t_start = tmax / 2
t_uniform = np.linspace(t_start, probe_t[-1], 4096)
vy_uniform = np.interp(t_uniform, probe_t, probe_vy)
vy_uniform -= vy_uniform.mean()

freqs = np.fft.rfftfreq(len(t_uniform), t_uniform[1] - t_uniform[0])
spectrum = np.abs(np.fft.rfft(vy_uniform))
f_shed = freqs[np.argmax(spectrum[1:]) + 1]
strouhal = f_shed * D / U_inf
print(f"shedding frequency = {f_shed:.4f}, Strouhal number = {strouhal:.3f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), dpi=125)
ax1.plot(probe_t, probe_vy / U_inf)
ax1.axvline(t_start, color="grey", ls="--")
ax1.set_xlabel("t")
ax1.set_ylabel(r"$v_y / U_\infty$ at probe")
ax2.plot(freqs * D / U_inf, spectrum)
ax2.axvline(strouhal, color="grey", ls="--", label=f"St = {strouhal:.3f}")
ax2.set_xlim(0, 1)
ax2.set_xlabel(r"$f D / U_\infty$")
ax2.set_ylabel("spectrum")
ax2.legend()
fig.tight_layout()
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
