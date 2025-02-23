"""
A very basic disc in SPH
============================

This simple example shows how to run a basic disc simulation in SPH
"""

import shamrock
shamrock.change_loglevel(1)
shamrock.sys.init(0,0)


# %%
# Setup units

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time = 3600*24*365,
    unit_length = sicte.au(),
    unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)
G = ucte.G()


# %%
# List parameters

# Resolution
Npart = 1000000

# Sink parameters
center_mass = 1
center_racc = 0.1

# Disc parameter
disc_mass = 0.01 #sol mass
rout = 10 # au
rin = 1 # au
H_r_in = 0.05
q = 0.5
p = 3./2.
r0 = 1

# Viscosity parameter
alpha_AV = 1e-3 / 0.08
alpha_u = 1
beta_AV = 2

# Integrator parameters
C_cour = 0.3
C_force = 0.25

# Disc profiles
def sigma_profile(r):
    sigma_0 = 1
    return sigma_0 * (r / rin)**(-p)

def kep_profile(r):
    return (G * center_mass / r)**0.5

def omega_k(r):
    return kep_profile(r) / r

def cs_profile(r):
    cs_in = (H_r_in * rin) * omega_k(rin)
    return ((r / rin)**(-q))*cs_in


# %%
# Utility functions and quantities deduced from the base one

# Deduced quantities
pmass = disc_mass/Npart
bmin = (-rout*2,-rout*2,-rout*2)
bmax = (rout*2,rout*2,rout*2)

cs0 = cs_profile(rin)

def rot_profile(r):
    return ((kep_profile(r)**2) - (2*p+q)*cs_profile(r)**2)**0.5

def H_profile(r):
    H = (cs_profile(r) / omega_k(r))
    #fact = (2.**0.5) * 3. # factor taken from phantom, to fasten thermalizing
    fact = 1
    return fact * H

# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the data and configure it

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")
cfg = model.gen_default_config()
cfg.set_artif_viscosity_ConstantDisc(alpha_u = alpha_u, alpha_AV = alpha_AV, beta_AV = beta_AV)
cfg.set_eos_locally_isothermalLP07(cs0 = cs0, q = q, r0 = r0)
cfg.set_units(codeu)

# Print the config in the terminal
cfg.print_status()

# Set the solver config to be the one stored in cfg
model.set_solver_config(cfg)

# We want the patches to split above 10^8 part and merge if smaller than 1 part (e.g. disable patch)
model.init_scheduler(int(1e8),1)

# Set the simulation box size
model.resize_simulation_box(bmin,bmax)

# %%
