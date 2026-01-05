import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import shamrock

# Particle tracking is an experimental feature

shamrock.enable_experimental_features()

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

m_s_codeu = codeu.get("m") * codeu.get("s", power=-1)
kg_m3_codeu = codeu.get("kg") * codeu.get("m", power=-3)

print(f"m_s_codeu: {kg_m3_codeu}")

cs = 190.0  # m/s
rho_g = kg_m3_codeu*1e-15
initial_u = 1.0

sim_radius = 0.5


kb = ucte.kb()
print(f"kb: {kb}")
mu = 2.375
mh = 1.00784 * ucte.dalton()
print(f"mu * mh * kb: {mu * mh * kb}")

cs = cs * m_s_codeu  # m/s
rho_c1 = 1.92e-13 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3
rho_c2 = 3.84e-8 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3
rho_c3 = 1.92e-3 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3

Npart = 5e5

bmin = (-sim_radius, -sim_radius, -sim_radius)
bmax = (sim_radius, sim_radius, sim_radius)


scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)


N_target = Npart
xm, ym, zm = bmin
xM, yM, zM = bmax
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
cfg.set_eos_machida06(rho_c1=rho_c1, rho_c2=rho_c2, rho_c3=rho_c3, cs=cs, mu=mu, mh=mh, kb=kb)

cfg.set_self_gravity_fmm(order=5, opening_angle=0.5)

cfg.set_units(codeu)
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)

# On aurora /2 was correct to avoid out of memory
setup.apply_setup(gen, insert_step=int(scheduler_split_val / 2))

vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho_g * vol_b
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

model.set_value_in_a_box("uint", "f64", initial_u, bmin, bmax)

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)


ampl = 0.5
eng = shamrock.algs.gen_seed(111)
def vel_func(r):
    global eng
    x, y, z = shamrock.algs.mock_gaussian_f64_3(eng)
    return (x * ampl, y * ampl, z * ampl)

model.set_field_value_lambda_f64_3("vxyz", vel_func)

model.timestep()


def analysis(i):
    ext = sim_radius/1.5
    arr_rho2 = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )

    dpi = 200
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho2,
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        #norm="log",
        #vmin=1e0,
        #vmax=1e3,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t = {:0.3f} [year]".format(model.get_time()))


    plt.colorbar()
    plt.savefig(f"collapse/rho_{i:04}.png")
    plt.close()


i = 0
t = 0
while True:
    analysis(i)
    
    model.evolve_until(t, niter_max=50)

    i += 1
    t = model.get_time() + 1.e-1
    
    if i> 200:
        analysis(i)
        break
