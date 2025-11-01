from statistics import mean, stdev

import shamrock
# Particle tracking is an experimental feature

shamrock.enable_experimental_features()

gamma = 5.0 / 3.0
rho_g = 100
initial_u = 0.05

sphere_radius = 0.1
sim_radius = 0.5

Npart = 1e4

bmin = (-sim_radius, -sim_radius, -sim_radius)
bmax = (sim_radius, sim_radius, sim_radius)

init_part_bmin = (-sphere_radius, -sphere_radius, -sphere_radius)
init_part_bmax = (sphere_radius, sphere_radius, sphere_radius)


scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)


N_target = Npart
xm, ym, zm = init_part_bmin
xM, yM, zM = init_part_bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

if shamrock.sys.world_rank() == 0:
    print("Npart", Npart)
    print("scheduler_split_val", scheduler_split_val)
    print("scheduler_merge_val", scheduler_merge_val)
    print("N_target", N_target)
    print("vol_b", vol_b)

part_vol = vol_b / N_target

# lattice volume
part_vol_lattice = 0.74 * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * 3.1416)) ** (1.0 / 3.0)

pmass = -1

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.set_max_neigh_cache_size(int(100e9))
cfg.set_self_gravity_direct()
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, init_part_bmin, init_part_bmax)

# On aurora /2 was correct to avoid out of memory
setup.apply_setup(gen, insert_step=int(scheduler_split_val / 2))


vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho_g * vol_b
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

model.set_value_in_a_box("uint", "f64", initial_u, bmin, bmax)


model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

tot_u = pmass * model.get_sum("uint", "f64")
if shamrock.sys.world_rank() == 0:
    print("total u :", tot_u)

for i in range(10):
    model.do_vtk_dump(f"step_{i}.vtk", True)
    for i in range(10):
        model.timestep()