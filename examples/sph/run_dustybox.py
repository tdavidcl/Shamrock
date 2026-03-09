"""
Testing Sod tube with SPH
=========================

CI test for Sod tube with SPH
"""

import matplotlib.pyplot as plt

import shamrock
import numpy as np

gamma = 1.4
rho_g = 1
P_g = 1

u_g = P_g / ((gamma - 1) * rho_g)


bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)

N_target = 1e4
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

part_vol = vol_b / N_target

# lattice volume
HCP_PACKING_DENSITY = 0.74
part_vol_lattice = HCP_PACKING_DENSITY * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

pmass = -1

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8), 1)


bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
setup.apply_setup(gen, insert_step=scheduler_split_val)



vol_b = (xM - xm) * (yM - ym) * (zM - zm)
totmass = rho_g * vol_b
print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)
print("Current part mass :", pmass)


model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

dic = ctx.collect_data()

print(dic["xyz"].shape)

x = dic["xyz"][:,0]
y = dic["xyz"][:,1]
z = dic["xyz"][:,2]

r = np.sqrt(x*x + y*y + z*z)

hpart = dic["hpart"]

plt.scatter(r,hpart)
plt.show()
