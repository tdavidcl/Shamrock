import shamrock


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time = 3600*24*365,unit_length = sicte.au(), unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_eos_locally_isothermal()
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)


bmin = (-10,-10,-10)
bmax = (10,10,10)
model.resize_simulation_box(bmin,bmax)

disc_mass = 0.001

pmass = model.add_disc_3d(
    (0,0,0),
    1,
    100000,
    0.2,3,
    disc_mass,
    1.,
    0.05,
    1./4.)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

print("Current part mass :", pmass)

model.set_particle_mass(pmass)


model.add_sink(1,(0,0,0),(0,0,0),0.1)

vk_p = (ucte.G() * 1 / 1)**0.5
model.add_sink(3*ucte.jupiter_mass(),(1,0,0),(0,0,vk_p),0.01)
#model.add_sink(100,(0,2,0),(0,0,1))

print("Run")


print("Current part mass :", pmass)


t_sum = 0
t_target = 1

i_dump = 0
dt_dump = 1e-2
next_dt_target = t_sum + dt_dump

while next_dt_target <= t_target:

    fname = "dump_{:04}.phfile".format(i_dump)

    model.evolve_until(next_dt_target)
    dump = model.make_phantom_dump()
    dump.save_dump(fname)

    i_dump += 1

    next_dt_target += dt_dump
