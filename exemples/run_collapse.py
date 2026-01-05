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
    unit_length=sicte.parsec(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

m_s_codeu = codeu.get("m") * codeu.get("s", power=-1)
kg_m3_codeu = codeu.get("kg") * codeu.get("m", power=-3)

print(f"m_s_codeu: {kg_m3_codeu}")

cs = 190.0  # m/s
rho_g = kg_m3_codeu * 2e-17
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

sim_directory = "collapse"
import os

os.makedirs(sim_directory, exist_ok=True)


tsound = sim_radius / cs
tff = shamrock.phys.free_fall_time(rho_g, codeu)

P_init, _cs_init, T_init = shamrock.phys.eos.eos_Machida06(
    cs=cs, rho=rho_g, rho_c1=rho_c1, rho_c2=rho_c2, rho_c3=rho_c3, mu=mu, mh=mh, kb=kb
)

print("---------------------------------------------------")
print(f"rho                           : {rho_g / kg_m3_codeu:.3e} [kg/m^3]")
print(f"P                             : {P_init / codeu.get("Pa"):.3e} [Pa]")
print(f"T                             : {T_init:.3e} [K]")
print(f"cs                            : {cs / m_s_codeu:.3e} [m/s]")
print(f"tsound (= R/cs)               : {tsound:9.1f} [years]")
print(f"tff (= sqrt(3*pi/(32*G*rho))) : {tff:9.1f} [years]")
print(f"tff/tsound                    : {tff/tsound:.4f} (<1 = collapse)")
print("---------------------------------------------------")

Npart = 1e5

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


def setup_particles():
    eng = shamrock.algs.gen_seed(11)

    o_xx, o_xy, o_xz = shamrock.algs.mock_gaussian_f64_3(eng)
    o_yx, o_yy, o_yz = shamrock.algs.mock_gaussian_f64_3(eng)
    o_zx, o_zy, o_zz = shamrock.algs.mock_gaussian_f64_3(eng)

    o_xx *= 1000
    o_xy *= 1000
    o_xz *= 1000
    o_yx *= 1000
    o_yy *= 1000
    o_yz *= 1000
    o_zx *= 1000
    o_zy *= 1000
    o_zz *= 1000

    print(f"o_xx: {o_xx}, o_xy: {o_xy}, o_xz: {o_xz}")
    print(f"o_yx: {o_yx}, o_yy: {o_yy}, o_yz: {o_yz}")
    print(f"o_zx: {o_zx}, o_zy: {o_zy}, o_zz: {o_zz}")

    perlin = shamrock.math.PerlinNoise()

    def noise_func(x, y, z):

        x *= 3
        y *= 3
        z *= 3

        x = perlin.noise_3d(x + o_xx, y + o_xy, z + o_xz)
        y = perlin.noise_3d(x + o_yx, y + o_yy, z + o_yz)
        z = perlin.noise_3d(x + o_zx, y + o_zy, z + o_zz)
        return (x, y, z)

    ampl = 30.0 * cs

    def vel_func(r):
        global eng, perlin
        x, y, z = r
        vx, vy, vz = 0.0, 0.0, 0.0

        for i in range(15):
            tx, ty, tz = noise_func(
                x * (i + 1) + i * o_xx * 10,
                y * (i + 1) + i * o_xy * 10,
                z * (i + 1) + i * o_xz * 10,
            )
            vx += tx / (i + 1.0)
            vy += ty / (i + 1.0)
            vz += tz / (i + 1.0)
        return (vx * ampl, vy * ampl, vz * ampl)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)

    model.timestep()


setup_particles()


def analysis(i):
    ext = sim_radius / 1.5
    arr_rho2 = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )

    arr_rho2 /= kg_m3_codeu

    dpi = 200
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("inferno"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho2,
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        norm="log",
        # vmin=1e0,
        # vmax=1e3,
    )

    plt.xlabel("x [pc]")
    plt.ylabel("y [pc]")
    plt.title("t = {:0.3f} [year]".format(model.get_time()))

    cbar = plt.colorbar()
    cbar.set_label("rho [kg/m^3]")
    plt.savefig(f"collapse/rho_{i:04}.png")
    plt.close()

    dat = ctx.collect_data()

    # get index with max rho
    max_rho_index = np.argmin(dat["hpart"])
    hpart_min = dat["hpart"][max_rho_index]
    max_rho = pmass * (model.get_hfact() / hpart_min) ** 3

    max_rho_x = dat["xyz"][max_rho_index, 0]
    max_rho_y = dat["xyz"][max_rho_index, 1]
    max_rho_z = dat["xyz"][max_rho_index, 2]
    print(f"max rho: {max_rho:.3e} at ({max_rho_x:.3e}, {max_rho_y:.3e}, {max_rho_z:.3e})")

    # render around max rho
    ext_loc = 0.1
    arr_rho = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(max_rho_x, max_rho_y, max_rho_z),
        delta_x=(ext_loc * 2, 0, 0.0),
        delta_y=(0.0, ext_loc * 2, 0.0),
        nx=1000,
        ny=1000,
    )
    arr_rho /= kg_m3_codeu
    plt.figure(num=3, clear=True, dpi=dpi)
    plt.imshow(
        arr_rho,
        cmap=my_cmap,
        origin="lower",
        extent=[max_rho_x - ext_loc, max_rho_x + ext_loc, max_rho_y - ext_loc, max_rho_y + ext_loc],
        norm="log",
    )
    plt.xlabel("x [pc]")
    plt.ylabel("y [pc]")
    plt.title("t = {:0.3f} [year]".format(model.get_time()))
    cbar = plt.colorbar(extend="both")
    cbar.set_label("rho [kg/m^3]")
    plt.savefig(f"collapse/centered_rho_{i:04}.png")
    plt.close()

    arr_vxyz = model.render_cartesian_column_integ(
        "vxyz",
        "f64_3",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )

    arr_vxyz /= cs

    v_plot = arr_vxyz[:, :, 0]
    max_v = np.max(np.abs(v_plot))

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("seismic"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    plt.figure(num=2, clear=True, dpi=dpi)
    plt.imshow(
        v_plot, cmap=my_cmap, origin="lower", vmin=-max_v, vmax=max_v, extent=[-ext, ext, -ext, ext]
    )
    plt.xlabel("x [pc]")
    plt.ylabel("y [pc]")
    plt.title("t = {:0.3f} [year]".format(model.get_time()))
    cbar = plt.colorbar(extend="both")
    cbar.set_label("v/cs")
    plt.savefig(f"collapse/vxyz_{i:04}.png")
    plt.close()


i = 0
next_t = 0
while True:

    model.evolve_until(next_t, niter_max=50)

    next_t = model.get_time() + tff * 0.1

    analysis(i)

    i += 1

    if i > 200:
        analysis(i)
        break
