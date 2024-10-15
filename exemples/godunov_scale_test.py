import shamrock
from math import log2

world_size = shamrock.sys.world_size()
world_rank = shamrock.sys.world_rank()

def is_pow8(a):
    return 8**int(log2(a)/3) == a

if not (is_pow8(world_size)):
    exit("This scalling test only works on multiple of 8 ranks")

fact_res = 2**int(log2(world_size)/3)
print(fact_res)

###########################################
#### parameters
gamma = 1.4
multx = 1
multy = 1
multz = 1

cell_int_size = 1 << 1
Nside_base = 128
Nside_block = Nside_base*fact_res
scale_fact = 2/(cell_int_size*Nside_block*multx)

NBlock_GPU = Nside_base**3
NBlock_all = Nside_block**3

scheduler_split_val = int(NBlock_GPU)
scheduler_merge_val = int(1)
###########################################

if world_rank == 0:
    print("Nside_block :",Nside_block)
    print("NBlock_GPU :",NBlock_GPU)
    print("NBlock_all :",NBlock_all)
    print("Grid size multiplier :",fact_res)


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_AMRGodunov(
    context = ctx,
    vector_type = "f64_3",
    grid_repr = "i64_3")



cfg = model.gen_default_config()
cfg.set_scale_factor(scale_fact)

cfg.set_eos_gamma(gamma)
#cfg.set_riemann_solver_rusanov()
cfg.set_riemann_solver_hll()

#cfg.set_slope_lim_none()
#cfg.set_slope_lim_vanleer_f()
#cfg.set_slope_lim_vanleer_std()
#cfg.set_slope_lim_vanleer_sym()
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)
model.set_config(cfg)

model.init_scheduler(scheduler_split_val,scheduler_merge_val)

model.make_base_grid(
    (0,0,0),
    (cell_int_size,cell_int_size,cell_int_size),
    (Nside_block*multx,Nside_block*multy,Nside_block*multz))



def rho_map(rmin,rmax):

    x,y,z = rmin
    if x < 1:
        return 1
    else:
        return 0.125


etot_L = 1./(gamma-1)
etot_R = 0.1/(gamma-1)

def rhoetot_map(rmin,rmax):

    rho = rho_map(rmin,rmax)

    x,y,z = rmin
    if x < 1:
        return etot_L
    else:
        return etot_R

def rhovel_map(rmin,rmax):
    rho = rho_map(rmin,rmax)

    return (0,0,0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

for i in range(5):
    model.timestep()

