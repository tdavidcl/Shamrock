"""
Production run: Circular disc & central sink particle
=====================================================

This example demonstrates how to run a smoothed particle hydrodynamics (SPH)
simulation of a circular disc orbiting around a central sink.

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
import scipy.interpolate
import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# Get the skip sim ENV variable
run_simulation = os.getenv("RUN_SIMULATION", "0")

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
Npart = int(os.getenv("NPART"))

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Dump and plot frequency and duration of the simulation
dump_freq_stop = 2
plot_freq_stop = 1

dt_stop = 0.1
nstop = 2000

# The list of times at which the simulation will pause for analysis / dumping
t_stop = [i * dt_stop for i in range(nstop + 1)]


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
alpha_AV = 1.0e-5 / 0.08
alpha_u = 1.0
beta_AV = 2.0

# Integrator parameters
C_cour = 0.3
C_force = 0.25

sim_folder = f"_to_trash/reflex_instab_{Npart}/"

dump_folder = sim_folder + "dump/"
analysis_folder = sim_folder + "analysis/"
plot_folder = analysis_folder + "plots/"

dump_prefix = dump_folder + "dump_"



class Vz_z_plot:
    def __init__(self, model,ext_z, ext_r, nr,nz, analysis_folder, analysis_prefix):
        self.model = model
        self.ext_z = ext_z
        self.ext_r = ext_r
        self.nr = nr
        self.nz = nz

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix) + "_"

        self.npy_data_filename = self.analysis_prefix + "vz_z_{:07}.npy"
        self.json_data_filename = self.analysis_prefix + "vz_z_{:07}.json"
        self.plot_filename = self.analysis_prefix + "vz_z_{:07}.png"
        self.glob_str = self.analysis_prefix + "vz_z_*.png"

    def compute_vz_z(self):
        
        arr_v_vslice = model.render_cartesian_slice(
            "vxyz",
            "f64_3",
            center=(0.0, 0.0, 0.0),
            delta_x=(self.ext_r * 2, 0, 0.0),
            delta_y=(0.0, 0.0, self.ext_z * 2),
            nx=self.nr,
            ny=self.nz,
        )
        
        return arr_v_vslice[:,:,2]

    def analysis_save(self, iplot):
        arr_vz_z = self.compute_vz_z()
        if shamrock.sys.world_rank() == 0:
            metadata = {"extent": [-self.ext_r, self.ext_r, -self.ext_z, self.ext_z], "time": self.model.get_time()}
            np.save(self.npy_data_filename.format(iplot), arr_vz_z)

            with open(self.json_data_filename.format(iplot), "w") as fp:
                json.dump(metadata, fp)

    def load_analysis(self, iplot):
        with open(self.json_data_filename.format(iplot), "r") as fp:
            metadata = json.load(fp)
        return np.load(self.npy_data_filename.format(iplot)), metadata

    def plot_vz_z(self, iplot):
        arr_vz_z, metadata = self.load_analysis(iplot)
        if shamrock.sys.world_rank() == 0:

            # Reset the figure using the same memory as the last one
            plt.figure(num=1, clear=True, dpi=200)

            v_ext = np.max(arr_vz_z)
            v_ext = max(v_ext, np.abs(np.min(arr_vz_z)))
            res = plt.imshow(
                arr_vz_z,
                cmap="seismic",
                origin="lower",
                extent=metadata["extent"],
                vmin=-v_ext,
                vmax=v_ext,
            )

            plt.xlabel("x")
            plt.ylabel("z")
            plt.title("t = {:0.3f} [Year]".format(metadata["time"]))

            cbar = plt.colorbar(res, extend="both")
            cbar.set_label(r"$v_z$ [code unit]")

            plt.savefig(self.plot_filename.format(iplot))
            plt.close()



# Disc profiles
def sigma_profile(r):
    sigma_0 = 1.0  # We do not care as it will be renormalized
    return sigma_0 * (r / r0) ** (-p)

def get_sigma_norm():
    x_list = np.linspace(rin, rout, 2048)
    term = [sigma_profile(x)*2*np.pi*x for x in x_list]
    return disc_mass/(np.sum(term)*(x_list[1] - x_list[0]))

sigma_norm = get_sigma_norm()

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


# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")


# %%
# Dump handling
def get_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".sham"


def get_vtk_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".vtk"


def get_ph_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".phdump"


def get_last_dump():
    res = glob.glob(dump_prefix + "*.sham")

    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len(dump_prefix) : -5])
            if dump_num > num_max:
                num_max = dump_num
        except ValueError:
            pass

    if num_max == -1:
        return None
    else:
        return num_max


def purge_old_dumps():
    if shamrock.sys.world_rank() == 0:
        res = glob.glob(dump_prefix + "*.sham")
        res.sort()

        # The list of dumps to remove (keep the first and last 3 dumps)
        to_remove = res[1:-3]

        for f in to_remove:
            os.remove(f)


idump_last_dump = get_last_dump()

if shamrock.sys.world_rank() == 0:
    print("Last dump:", idump_last_dump)

# %%
# Load the last dump if it exists, setup otherwise

if run_simulation == "1" and idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
elif run_simulation == "1":
    # Generate the default config
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)

    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box

    cfg.set_units(codeu)
    cfg.set_particle_mass(pmass)
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)

    # On a chaotic disc, we disable to two stage search to avoid giant leaves
    cfg.set_tree_reduction_level(6)
    cfg.set_two_stage_search(False)

    # Enable this to debug the neighbor counts
    # cfg.set_show_neigh_stats(True)

    # Standard way to set the smoothing length (e.g. Price et al. 2018)
    # cfg.set_smoothing_length_density_based()

    # Standard density based smoothing lenght but with a neighbor count limit
    # Use it if you have large slowdowns due to giant particles
    # I recommend to use it if you have a circumbinary discs as the issue is very likely to happen
    cfg.set_smoothing_length_density_based_neigh_lim(500)
    
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
    # Smoothing length iterations, increasing it affect the performance negatively but increse the
    # convergence rate of the smoothing length
    # this is why we increase it temporely to 1.3 before lowering it back to 1.1 (default value)
    # Note that both ``change_htolerances`` can be removed and it will work the same but would converge
    # more slowly at the first timestep

    model.change_htolerances(coarse=1.3, fine=1.1)
    model.timestep()
    model.change_htolerances(coarse=1.1, fine=1.1)

vz_analysis = Vz_z_plot(model, rout, rout*1.5, 1024, 1024, plot_folder, "vz_z")

# %%
# On the fly analysis
def save_rho_integ(ext, arr_rho, iplot):
    if shamrock.sys.world_rank() == 0:
        metadata = {"extent": [-ext, ext, -ext, ext], "time": model.get_time()}
        np.save(plot_folder + f"rho_integ_{iplot:07}.npy", arr_rho)

        with open(plot_folder + f"rho_integ_{iplot:07}.json", "w") as fp:
            json.dump(metadata, fp)


def save_analysis_data(filename, key, value, ianalysis):
    """Helper to save analysis data to a JSON file."""
    if shamrock.sys.world_rank() == 0:
        filepath = os.path.join(analysis_folder, filename)
        try:
            with open(filepath, "r") as fp:
                data = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {key: []}
        data[key] = data[key][:ianalysis]
        data[key].append({"t": model.get_time(), key: value})
        with open(filepath, "w") as fp:
            json.dump(data, fp, indent=4)


def make_cartesian_coords(nx, ny, z_val, min_x, max_x, min_y, max_y):

    # Create the cylindrical coordinate grid
    x_vals = np.linspace(min_x, max_x, nx)
    y_vals = np.linspace(min_y, max_y, ny)

    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Convert to Cartesian coordinates (z = 0 for a disc in the xy-plane)
    z_grid = z_val * np.ones_like(x_grid)

    # Flatten and stack to create list of positions
    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    return [tuple(pos) for pos in positions]

def positions_to_rays(positions):
    return [shamrock.math.Ray_f64_3(tuple(position), (0.0, 0.0, 1.0)) for position in positions]

def compute_avg_sigma_profile(ntheta, r):

    theta = np.linspace(0, 2*np.pi, ntheta)
    
    r_grid, theta_grid = np.meshgrid(r, theta)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(r_grid)

    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    rays = positions_to_rays(positions)
    arr_sigma = model.render_column_integ("rho", "f64", rays)

    arr_sigma = np.array(arr_sigma).reshape(ntheta,len(r))
    
    #average over the theta direction
    arr_sigma = np.mean(arr_sigma, axis=0)
    return arr_sigma

def rho_perturb_render(ext, nx, ny, ianalysis):

    positions = make_cartesian_coords(nx, ny, 0.0, -ext, ext, -ext, ext)

    rays = positions_to_rays(positions)

    arr_sigma = model.render_column_integ("rho", "f64", rays)

    r = np.linspace(1e-9, rout*1.5, 2048)

    arr_sigma_avg = compute_avg_sigma_profile(32, r)

    # create a scipy inter
    interp_sigma_avg = scipy.interpolate.interp1d(r, arr_sigma_avg, kind='cubic', fill_value=0.0, bounds_error=False)

    perturb_rho = np.zeros_like(arr_sigma)
    for i in range(len(arr_sigma)):
        pos = positions[i]
        rho = arr_sigma[i]
        x,y,z = pos
        r_c = np.sqrt(x**2 + y**2)
        
        sigma = interp_sigma_avg(r_c)

        perturb_rho[i] = (rho - sigma)

    perturb_rho = np.array(perturb_rho).reshape(nx, ny)

    if shamrock.sys.world_rank() == 0:
        metadata = {"extent": [-ext, ext, -ext, ext], "time": model.get_time()}
        np.save(plot_folder + f"sigma_delta_{iplot:07}.npy", perturb_rho)

        with open(plot_folder + f"sigma_delta_{iplot:07}.json", "w") as fp:
            json.dump(metadata, fp)

def plot_profiles(ext, ianalysis):
    x_list = np.linspace(0, rout*1.5, 2049)[1:]
    positions = [(x,0.,0.) for x in x_list.tolist()]

    rays = positions_to_rays(positions)

    arr_rho = model.render_slice("rho", "f64", positions)
    arr_rho_integ = model.render_column_integ("rho", "f64", rays)
    arr_sigma_avg = compute_avg_sigma_profile(32, x_list)

    arr_sigma = np.zeros_like(arr_rho_integ)
    for i in range(len(arr_rho_integ)):
        x,_,_ = positions[i]
        sigma = sigma_profile(x)*sigma_norm

        if x < rin:
            sigma = 0.0
        elif x > rout:
            sigma = 0.0
        arr_sigma[i] = sigma

    if shamrock.sys.world_rank() == 0:
        metadata = {"extent": [-ext, ext, -ext, ext], "time": model.get_time()}
        np.save(plot_folder + f"profile_xlist_{iplot:07}.npy", x_list)
        np.save(plot_folder + f"profile_arr_rho_{iplot:07}.npy", arr_rho)
        np.save(plot_folder + f"profile_arr_rho_integ_{iplot:07}.npy", arr_rho_integ)
        np.save(plot_folder + f"profile_arr_sigma_{iplot:07}.npy", arr_sigma)
        np.save(plot_folder + f"profile_arr_sigma_avg_{iplot:07}.npy", arr_sigma_avg)

        with open(plot_folder + f"profiles_{iplot:07}.json", "w") as fp:
            json.dump(metadata, fp)

def compute_anulus_sigma( ianalysis):
    theta = np.linspace(0, 2*np.pi, 2048)

    avg_count = 5
    arr_sigma_avg = np.zeros_like(theta)

    rs = rout - rin
    r_list = np.linspace(rin + rs*0.3, rout - rs*0.3, avg_count)
    for r in r_list:
        positions = [(r*np.cos(theta), r*np.sin(theta), 0.0) for theta in theta]
        rays = positions_to_rays(positions)
        arr_sigma = model.render_column_integ("rho", "f64", rays)
        arr_sigma_avg += arr_sigma
    arr_sigma_avg /= avg_count
    
    if shamrock.sys.world_rank() == 0:
        metadata = { "time": model.get_time()}
        np.save(plot_folder + f"anulus_sigma_{iplot:07}.npy", arr_sigma_avg)
        np.save(plot_folder + f"anulus_theta_{iplot:07}.npy", theta)
        with open(plot_folder + f"anulus_sigma_{iplot:07}.json", "w") as fp:
            json.dump(metadata, fp)

def analysis(ianalysis):

    ext = rout * 1.5
    nx = 1024
    ny = 1024

    arr_rho2 = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=nx,
        ny=ny,
    )

    save_rho_integ(ext, arr_rho2, ianalysis)

    rho_perturb_render(ext, nx, ny, ianalysis)
    compute_anulus_sigma(ianalysis)

    plot_profiles(ext, ianalysis)

    vz_analysis.analysis_save(ianalysis)

    barycenter, disc_mass = shamrock.model_sph.analysisBarycenter(model=model).get_barycenter()

    total_momentum = shamrock.model_sph.analysisTotalMomentum(model=model).get_total_momentum()

    potential_energy = shamrock.model_sph.analysisEnergyPotential(
        model=model
    ).get_potential_energy()

    kinetic_energy = shamrock.model_sph.analysisEnergyKinetic(model=model).get_kinetic_energy()

    save_analysis_data("barycenter.json", "barycenter", barycenter, ianalysis)
    save_analysis_data("disc_mass.json", "disc_mass", disc_mass, ianalysis)
    save_analysis_data("total_momentum.json", "total_momentum", total_momentum, ianalysis)
    save_analysis_data("potential_energy.json", "potential_energy", potential_energy, ianalysis)
    save_analysis_data("kinetic_energy.json", "kinetic_energy", kinetic_energy, ianalysis)

    sinks = model.get_sinks()
    save_analysis_data("sinks.json", "sinks", sinks, ianalysis)

    sim_time_delta = model.solver_logs_cumulated_step_time()
    scount = model.solver_logs_step_count()

    save_analysis_data("sim_time_delta.json", "sim_time_delta", sim_time_delta, ianalysis)
    save_analysis_data("sim_step_count_delta.json", "sim_step_count_delta", scount, ianalysis)

    model.solver_logs_reset_cumulated_step_time()
    model.solver_logs_reset_step_count()

    part_count = model.get_total_part_count()
    save_analysis_data("part_count.json", "part_count", part_count, ianalysis)


# %%
# Evolve the simulation
if run_simulation == "1":
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
                model.do_vtk_dump(get_vtk_dump_name(idump), True)
                model.dump(get_dump_name(idump))

                # dump = model.make_phantom_dump()
                # dump.save_dump(get_ph_dump_name(idump))

                purge_old_dumps()

            if istop % plot_freq_stop == 0:
                analysis(iplot)

        if istop % dump_freq_stop == 0:
            idump += 1

        if istop % plot_freq_stop == 0:
            iplot += 1

        istop += 1

# %%
# Plot generation (make_plots.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Load the on-the-fly analysis after the run to make the plots
# (everything in this section can be in another file)

import matplotlib
import matplotlib.pyplot as plt

# Uncomment this and replace by you dump folder, here since it is just above i comment it out
# dump_folder = "my_masterpiece"
# dump_folder += "/"


def plot_rho_integ(metadata, arr_rho, iplot):

    ext = metadata["extent"]

    dpi = 200

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho, cmap=my_cmap, origin="lower", extent=ext, norm="log", vmin=1e-8, vmax=1e-4
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [years]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int \rho \, \mathrm{d}z$ [code unit]")

    plt.savefig(plot_folder + "rho_integ_{:04}.png".format(iplot))
    plt.close()



def plot_sigma_delta(metadata, arr_sigma, iplot):

    ext = metadata["extent"]

    dpi = 200

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = matplotlib.colormaps["seismic"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    cen = 0.0
    ampl = 1e-5
    res = plt.imshow(
        arr_sigma, cmap=my_cmap, origin="lower", extent=ext,vmin=cen-ampl, vmax=cen+ampl#, norm="log"#, vmin=1e-8, vmax=1e-4
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [years]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\sigma - \sigma_0$ [code unit]")

    plt.savefig(plot_folder + "sigma_delta_{:04}.png".format(iplot))
    plt.close()


def get_list_dumps_id():
    import glob

    list_files = glob.glob(plot_folder + "rho_integ_*.npy")
    list_files.sort()
    list_dumps_id = []
    for f in list_files:
        list_dumps_id.append(int(f.split("_")[-1].split(".")[0]))
    return list_dumps_id


def load_rho_integ(iplot):
    with open(plot_folder + f"rho_integ_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(plot_folder + f"rho_integ_{iplot:07}.npy"), metadata


if shamrock.sys.world_rank() == 0:
    for iplot in get_list_dumps_id():
        print("Rendering rho integ plot for dump", iplot)
        arr_rho, metadata = load_rho_integ(iplot)
        plot_rho_integ(metadata, arr_rho, iplot)


def load_sigma_delta(iplot):
    with open(plot_folder + f"sigma_delta_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(plot_folder + f"sigma_delta_{iplot:07}.npy"), metadata


if shamrock.sys.world_rank() == 0:
    for iplot in get_list_dumps_id():
        print("Rendering sigma delta plot for dump", iplot)
        arr_sigma, metadata = load_sigma_delta(iplot)
        plot_sigma_delta(metadata, arr_sigma, iplot)


def load_profiles(iplot):
    with open(plot_folder + f"profiles_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(plot_folder + f"profile_xlist_{iplot:07}.npy"), np.load(plot_folder + f"profile_arr_rho_{iplot:07}.npy"), np.load(plot_folder + f"profile_arr_rho_integ_{iplot:07}.npy"), np.load(plot_folder + f"profile_arr_sigma_{iplot:07}.npy"), np.load(plot_folder + f"profile_arr_sigma_avg_{iplot:07}.npy"), metadata

def plot_profiles(metadata, x_list, arr_rho, arr_rho_integ, arr_sigma, arr_sigma_avg, iplot):

    dpi = 200
    plt.figure(dpi=dpi)
    plt.plot(x_list, arr_rho, label="rho")
    plt.plot(x_list, arr_rho_integ, label="rho integ")
    plt.plot(x_list, arr_sigma, label="sigma")
    plt.plot(x_list, arr_sigma_avg, label="sigma avg")
    plt.plot(x_list, arr_sigma*x_list/(2*np.pi), label="sigma*x/(2*pi)")
    plt.xlabel("x")
    plt.ylabel("rho")
    plt.title(f"t = {metadata['time']:0.3f} [years]")
    plt.legend()
    plt.ylim(1e-6,1e-3)
    plt.yscale("log")
    plt.savefig(plot_folder + "profiles_{:04}.png".format(iplot))
    plt.close()

if shamrock.sys.world_rank() == 0:
    for iplot in get_list_dumps_id():
        print("Rendering profiles for dump", iplot)
        x_list, arr_rho, arr_rho_integ, arr_sigma, arr_sigma_avg, metadata = load_profiles(iplot)
        plot_profiles(metadata, x_list, arr_rho, arr_rho_integ, arr_sigma, arr_sigma_avg, iplot)


def load_anulus_sigma(iplot):
    with open(plot_folder + f"anulus_sigma_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(plot_folder + f"anulus_sigma_{iplot:07}.npy"), np.load(plot_folder + f"anulus_theta_{iplot:07}.npy"), metadata

def plot_anulus_sigma(metadata, arr_sigma, theta, iplot):
    dpi = 200
    plt.figure(dpi=dpi)
    print(arr_sigma)
    plt.plot(theta, arr_sigma)
    plt.xlabel("theta")
    plt.ylabel("sigma")
    plt.ylim(2e-5,5e-5)
    plt.title(f"t = {metadata['time']:0.3f} [years]")
    plt.savefig(plot_folder + "anulus_sigma_{:04}.png".format(iplot))
    plt.close()

    # Plot FFT of sigma(theta) (mean-subtracted to remove the DC component)
    arr_sigma = np.asarray(arr_sigma)
    sigma_ms = arr_sigma - np.mean(arr_sigma)
    # Apply a Hann (Hanning) window to reduce spectral leakage
    if arr_sigma.size > 1:
        window = np.hanning(arr_sigma.size)
        sigma_ms = sigma_ms * window
    fft_sigma = np.fft.rfft(sigma_ms)
    amp = np.abs(fft_sigma) / max(arr_sigma.size, 1)
    m = np.arange(amp.size)

    plt.figure(dpi=dpi)
    if amp.size > 1:
        plt.plot(m[1:], amp[1:])
    else:
        plt.plot(m, amp)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("mode m")
    plt.ylim(1e-12,1e-6)
    plt.ylabel(r"|FFT($\sigma-\langle\sigma\rangle$)|")
    plt.title(f"t = {metadata['time']:0.3f} [years]")
    plt.savefig(plot_folder + "anulus_sigmafft_{:04}.png".format(iplot))
    plt.close()

if shamrock.sys.world_rank() == 0:
    for iplot in get_list_dumps_id():
        print("Rendering anulus sigma for dump", iplot)
        arr_sigma, theta, metadata = load_anulus_sigma(iplot)
        plot_anulus_sigma(metadata, arr_sigma, theta, iplot)

if shamrock.sys.world_rank() == 0:
    for iplot in get_list_dumps_id():
        print("Rendering vz z plot for dump", iplot)
        vz_analysis.plot_vz_z(iplot)

# %%
# Make gif for the doc (plot_to_gif.py)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Convert PNG sequence to Image sequence in mpl

# sphinx_gallery_multi_image = "single"

import matplotlib.animation as animation


def show_image_sequence(glob_str, render_gif):

    if render_gif and shamrock.sys.world_rank() == 0:

        import glob

        files = sorted(glob.glob(glob_str))

        from PIL import Image

        image_array = []
        for my_file in files:
            image = Image.open(my_file)
            image_array.append(image)

        if not image_array:
            raise RuntimeError(f"Warning: No images found for glob pattern: {glob_str}")

        pixel_x, pixel_y = image_array[0].size

        # Create the figure and axes objects
        # Remove axes, ticks, and frame & set aspect ratio
        dpi = 200
        fig = plt.figure(dpi=dpi)
        plt.gca().set_position((0, 0, 1, 1))
        plt.gcf().set_size_inches(pixel_x / dpi, pixel_y / dpi)
        plt.axis("off")

        # Set the initial image with correct aspect ratio
        im = plt.imshow(image_array[0], animated=True, aspect="auto")

        def update(i):
            im.set_array(image_array[i])
            return (im,)

        # Create the animation object
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(image_array),
            interval=50,
            blit=True,
            repeat_delay=10,
        )

        return ani



# %%
# Do it for rho integ
render_gif = True
glob_str = os.path.join(plot_folder, "rho_integ_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(analysis_folder + "rho_integ.gif", writer=writer)

    # Show the animation
    plt.show()

# %%
# Do it for sigma delta
render_gif = True
glob_str = os.path.join(plot_folder, "sigma_delta_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(analysis_folder + "sigma_delta.gif", writer=writer)

    # Show the animation
    plt.show()

# %%
# Do it for anulus sigma
render_gif = True
glob_str = os.path.join(plot_folder, "anulus_sigma_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)
if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(analysis_folder + "anulus_sigma.gif", writer=writer)
    plt.show()

render_gif = True
glob_str = os.path.join(plot_folder, "anulus_sigmafft_*.png")
ani = show_image_sequence(glob_str, render_gif)
if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(analysis_folder + "anulus_sigmafft.gif", writer=writer)
    plt.show()

# %%
# Do it for profiles
render_gif = True
glob_str = os.path.join(plot_folder, "profiles_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str, render_gif)
if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(analysis_folder + "profiles.gif", writer=writer)
    plt.show()


# %%
# Do it for vz z
render_gif = True
glob_str = os.path.join(plot_folder, "vz_z_*.png")
ani = show_image_sequence(glob_str, render_gif)
if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(analysis_folder + "vz_z.gif", writer=writer)
    plt.show()

# %%
# helper function to load data from JSON files
def load_data_from_json(filename, key):
    filepath = os.path.join(analysis_folder, filename)
    with open(filepath, "r") as fp:
        data = json.load(fp)[key]
    t = [d["t"] for d in data]
    values = [d[key] for d in data]
    return t, values


# %%
# load the json file for barycenter
t, barycenter = load_data_from_json("barycenter.json", "barycenter")
barycenter_x = [d[0] for d in barycenter]
barycenter_y = [d[1] for d in barycenter]
barycenter_z = [d[2] for d in barycenter]

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, barycenter_x)
plt.plot(t, barycenter_y)
plt.plot(t, barycenter_z)
plt.xlabel("t")
plt.ylabel("barycenter")
plt.legend(["x", "y", "z"])
plt.savefig(analysis_folder + "barycenter.png")
plt.show()

# %%
# load the json file for disc_mass
t, disc_mass = load_data_from_json("disc_mass.json", "disc_mass")

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, disc_mass)
plt.xlabel("t")
plt.ylabel("disc_mass")
plt.savefig(analysis_folder + "disc_mass.png")
plt.show()

# %%
# load the json file for total_momentum
t, total_momentum = load_data_from_json("total_momentum.json", "total_momentum")
total_momentum_x = [d[0] for d in total_momentum]
total_momentum_y = [d[1] for d in total_momentum]
total_momentum_z = [d[2] for d in total_momentum]

plt.figure(figsize=(8, 5), dpi=200)

plt.plot(t, total_momentum_x)
plt.plot(t, total_momentum_y)
plt.plot(t, total_momentum_z)
plt.xlabel("t")
plt.ylabel("total_momentum")
plt.legend(["x", "y", "z"])
plt.savefig(analysis_folder + "total_momentum.png")
plt.show()

# %%
# load the json file for energies
t, potential_energy = load_data_from_json("potential_energy.json", "potential_energy")
_, kinetic_energy = load_data_from_json("kinetic_energy.json", "kinetic_energy")

total_energy = [p + k for p, k in zip(potential_energy, kinetic_energy)]

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, potential_energy)
plt.plot(t, kinetic_energy)
plt.plot(t, total_energy)
plt.xlabel("t")
plt.ylabel("energy")
plt.legend(["potential_energy", "kinetic_energy", "total_energy"])
plt.savefig(analysis_folder + "energies.png")
plt.show()

# %%
# load the json file for sinks
t, sinks = load_data_from_json("sinks.json", "sinks")

sinks_x = [d[0]["pos"][0] for d in sinks]
sinks_y = [d[0]["pos"][1] for d in sinks]
sinks_z = [d[0]["pos"][2] for d in sinks]

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, sinks_x, label="sink 0 (x)")
plt.plot(t, sinks_y, label="sink 0 (y)")
plt.plot(t, sinks_z, label="sink 0 (z)")
plt.xlabel("t")
plt.ylabel("sink position")
plt.legend()
plt.savefig(analysis_folder + "sinks.png")
plt.show()

t,sinks = load_data_from_json("sinks.json", "sinks")
sinks_x = [d[0]["pos"][0] for d in sinks]
sinks_y = [d[0]["pos"][1] for d in sinks]
sinks_z = [d[0]["pos"][2] for d in sinks]

plt.figure(figsize=(8, 5), dpi=200)
# Use time as colormap
sinks_x = np.asarray(sinks_x)
sinks_y = np.asarray(sinks_y)
t = np.asarray(t)
sc = plt.scatter(sinks_x, sinks_y, c=t, cmap="viridis", s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(sc, label="t")
plt.axis("equal")
plt.savefig(analysis_folder + "sinks_xy.png")
plt.show()

# %%
# Sink to barycenter distance
t, sinks = load_data_from_json("sinks.json", "sinks")
_, barycenter = load_data_from_json("barycenter.json", "barycenter")

barycenter_x = np.array([d[0] for d in barycenter])
barycenter_y = np.array([d[1] for d in barycenter])
barycenter_z = np.array([d[2] for d in barycenter])

sinks_x = np.array([d[0]["pos"][0] for d in sinks])
sinks_y = np.array([d[0]["pos"][1] for d in sinks])
sinks_z = np.array([d[0]["pos"][2] for d in sinks])


plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, sinks_x - barycenter_x, label="sink 0 (x)")
plt.plot(t, sinks_y - barycenter_y, label="sink 0 (y)")
plt.plot(t, sinks_z - barycenter_z, label="sink 0 (z)")
plt.xlabel("t")
plt.ylabel("sink pos - barycenter pos")
plt.legend()
plt.savefig(analysis_folder + "sink_to_barycenter_distance.png")
plt.show()

# %%
# load the json file for sim_time_delta
t, sim_time_delta = load_data_from_json("sim_time_delta.json", "sim_time_delta")

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, sim_time_delta)
plt.xlabel("t")
plt.ylabel("sim_time_delta")
plt.savefig(analysis_folder + "sim_time_delta.png")
plt.show()

# %%
# load the json file for sim_step_count_delta
t, sim_step_count_delta = load_data_from_json("sim_step_count_delta.json", "sim_step_count_delta")

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, sim_step_count_delta)
plt.xlabel("t")
plt.ylabel("sim_step_count_delta")
plt.savefig(analysis_folder + "sim_step_count_delta.png")
plt.show()

# %%
# Time per step
t, sim_time_delta = load_data_from_json("sim_time_delta.json", "sim_time_delta")
_, sim_step_count_delta = load_data_from_json("sim_step_count_delta.json", "sim_step_count_delta")
_, part_count = load_data_from_json("part_count.json", "part_count")

time_per_step = []

for td, sc, pc in zip(sim_time_delta, sim_step_count_delta, part_count):
    if sc > 0:
        time_per_step.append(td / sc)
    else:
        # NAN here because the step count is 0
        time_per_step.append(np.nan)

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, time_per_step, "+-")
plt.xlabel("t")
plt.ylabel("time_per_step")
plt.savefig(analysis_folder + "time_per_step.png")
plt.show()

rate = []

for td, sc, pc in zip(sim_time_delta, sim_step_count_delta, part_count):
    if sc > 0:
        rate.append(pc / (td / sc))
    else:
        # NAN here because the step count is 0
        rate.append(np.nan)

plt.figure(figsize=(8, 5), dpi=200)
plt.plot(t, rate, "+-")
plt.xlabel("t")
plt.ylabel("Particles / second")
plt.yscale("log")
plt.savefig(analysis_folder + "rate.png")
plt.show()