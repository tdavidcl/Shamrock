import shamrock
import matplotlib.pyplot as plt
import numpy as np


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

model.init_scheduler(int(1e4),1)


bmin = (-10,-10,-10)
bmax = (10,10,10)
model.resize_simulation_box(bmin,bmax)

disc_mass = 0.001

pmass = model.add_big_disc_3d(
    (0,0,0),
    1,
    100000,
    0.2,3,
    disc_mass,
    1.,
    0.05,
    1./4., 11)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

print("Current part mass :", pmass)

model.set_particle_mass(pmass)


model.add_sink(1,(0,0,0),(0,0,0),0.1)

vk_p = (ucte.G() * 1 / 1)**0.5
#model.add_sink(3*ucte.jupiter_mass(),(1,0,0),(0,0,vk_p),0.01)
#model.add_sink(100,(0,2,0),(0,0,1))

def compute_rho(h):
    return np.array([ model.rho_h(h[i]) for i in range(len(h))])


def plot_vertical_profile(r, rrange, label = ""):

    data = ctx.collect_data()

    rhosel = []
    ysel = []

    for i in range(len(data["hpart"][:])):
        rcy = data["xyz"][i,0]**2 + data["xyz"][i,2]**2

        if rcy > r - rrange and rcy < r + rrange:
            rhosel.append(model.rho_h(data["hpart"][i]))
            ysel.append(data["xyz"][i,1])

    rhosel = np.array(rhosel)
    ysel = np.array(ysel)

    rhobar = np.mean(rhosel)
    
    plt.scatter(ysel, rhosel/rhobar, s=1, label = label)



print("Run")
model.change_htolerance(1.3)
model.evolve_once_override_time(0,0)
model.change_htolerance(1.1)

for i in range(4):
    model.timestep()

model.do_vtk_dump("end.vtk", True)
dump = model.make_phantom_dump()
dump.save_dump("end.phdump")

