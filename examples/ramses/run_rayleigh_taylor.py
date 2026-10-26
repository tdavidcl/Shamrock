"""
Rayleigh-Taylor instability in RAMSES solver
============================================

A heavy fluid (:math:`\\rho = 2`) sits on top of a light one (:math:`\\rho = 1`) in a constant
gravitational field :math:`\\mathbf{g} = -g \\, \\mathbf{e}_y`, applied as an external force
(``cfg.add_ext_force_constant_gravity``). The initial state is in hydrostatic equilibrium, and a
small velocity perturbation seeds the instability at the interface.
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

# Physical parameters
g = 0.1  # gravitational acceleration (along -y)
gamma = 1.4

rho_light = 1.0  # bottom fluid
rho_heavy = 2.0  # top fluid

Lx, Ly = 0.5, 1.5  # box size
y_interface = Ly / 2
P0 = 2.5  # pressure at the interface

v_pert = 1e-2  # amplitude of the velocity perturbation

t_end = 5.0
n_frames = 26

# resolution (the run is quasi 2D: a single block of cells along z)
sz = 1 << 1
base = 32
multx = 1
multy = 3

sim_folder = "_to_trash/ramses_rayleigh_taylor/"

# %%
# Deduced quantities

# grid int coords to [0, Lx] x [0, Ly] x [0, Lz]
scale_fact = Lx / (sz * base * multx)

nx = sz * base * multx
ny = sz * base * multy
Lz = sz * scale_fact

# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)

# %%
# Setup the solver
# ^^^^^^^^^^^^^^^^

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(gamma)
cfg.set_riemann_solver_hllc()
cfg.set_Csafe(0.8)
cfg.set_slope_lim_vanleer_sym()
cfg.set_face_time_interpolation(True)
cfg.add_ext_force_constant_gravity(g=(0.0, -g, 0.0))
cfg.set_boundary_condition("x", "periodic")
cfg.set_boundary_condition("y", "reflective")
cfg.set_boundary_condition("z", "reflective")
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, 1))

# %%
# Initial conditions
# ^^^^^^^^^^^^^^^^^^
# Hydrostatic equilibrium :math:`P = P_0 - \rho g (y - y_{\rm interface})` (continuous at the
# interface), with a velocity perturbation along y localized around the interface.


def rho_map(rmin, rmax):
    x, y, z = rmin
    if y <= y_interface:
        return rho_light
    else:
        return rho_heavy


def vy_map(rmin, rmax):
    x, y, z = rmin
    xloc = x - Lx / 2
    yloc = y - y_interface
    return (
        v_pert
        * (1.0 / 4.0)
        * (1.0 + np.cos(4.0 * np.pi * xloc))
        * (1.0 + np.cos(3.0 * np.pi * yloc))
    )


def rhovel_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    return (0.0, rho * vy_map(rmin, rmax), 0.0)


def rhoetot_map(rmin, rmax):
    rho = rho_map(rmin, rmax)
    x, y, z = rmin
    P = P0 - g * rho * (y - y_interface)
    vy = vy_map(rmin, rmax)
    return 0.5 * rho * vy**2 + P / (gamma - 1.0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

# %%
# Plotting utilities
# ^^^^^^^^^^^^^^^^^^


def make_cartesian_coords(nx, ny, z_val, min_x, max_x, min_y, max_y):
    x_vals = np.linspace(min_x, max_x, nx)
    y_vals = np.linspace(min_y, max_y, ny)

    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
    z_grid = z_val * np.ones_like(x_grid)

    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    return [tuple(pos) for pos in positions]


positions = make_cartesian_coords(nx, ny, Lz / 2, 0, Lx - 1e-6, 0, Ly - 1e-6)


def plot_rho(t, iplot, dpi=150):
    arr_rho = np.array(model.render_slice("rho", "f64", positions)).reshape(nx, ny)

    plt.figure(figsize=(4, 9), dpi=dpi)
    res = plt.imshow(
        arr_rho.T,
        cmap=matplotlib.colormaps["rainbow"],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        vmin=rho_light,
        vmax=rho_heavy,
        interpolation="bicubic",
        aspect="equal",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {t:0.3f}")
    cbar = plt.colorbar(res)
    cbar.set_label(r"$\rho$ [code unit]")
    plt.tight_layout()
    plt.savefig(os.path.join(sim_folder, f"rho_{iplot:04d}.png"))
    plt.close()


# %%
# Run the simulation
# ^^^^^^^^^^^^^^^^^^

for i, t in enumerate(np.linspace(0, t_end, n_frames)):
    model.evolve_until(t)
    plot_rho(t, i)

# %%
# Density evolution
# ^^^^^^^^^^^^^^^^^

from shamrock.utils.plot import show_image_sequence

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(os.path.join(sim_folder, "rho_*.png"), render_gif=True)

if shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(os.path.join(sim_folder, "rho_rayleigh_taylor.gif"), writer=writer)

plt.show()
