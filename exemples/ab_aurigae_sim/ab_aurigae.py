import shamrock
import os
import matplotlib.pyplot as plt
import numpy as np

####################################################
# Setup parameters
####################################################
Npart = 100000
disc_mass = 0.001 #sol mass

rout = 350
rin = 90

# alpha_ss ~ alpha_AV * 0.08
alpha_AV = 0.3
alpha_u = 1
beta_AV = 2

q = 0.15
p = 1.
r0 = 1
Rcav = 2.5
delta_0 = 1e-5

C_cour = 0.3
C_force = 0.25

H_r_in = 0.05

dump_folder = "AB_Aurigae_lim500_"+str(Npart)
os.makedirs(dump_folder, exist_ok=True)
dump_folder += "/"

dump_prefix = dump_folder + "AB_Aurigae_"
init_dump = dump_folder + "AB_Aurigae_init.sham"

#central star params
center_mass = 2.5
center_racc = 1

# hierarichle split
split_list = [
    {"index" : 0, "mass_ratio" : 0.5/2, "a": 40, "e":0.5, "euler_angle" : (np.radians(90),0.,0.)},
    #{"index" : 0, "mass_ratio" : 0.5, "a": 0.33333333, "e":0., "euler_angle" :(0,0,0)}
]

do_plots = True

####################################################
####################################################
####################################################

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time = sicte.year(),
    unit_length = sicte.au() ,
    unit_mass = sicte.sol_mass(), )
ucte = shamrock.Constants(codeu)

# Deduced quantities
pmass = disc_mass/Npart
bmin = (-rout*2,-rout*2,-rout*2)
bmax = (rout*2,rout*2,rout*2)
G = ucte.G()

print("GM =",G * center_mass)

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

cs0 = cs_profile(rin)

def rot_profile(r):
    # return kep_profile(r)

    # subkeplerian correction
    return ((kep_profile(r)**2) - (2*p+q)*cs_profile(r)**2)**0.5

def H_profile(r):
    H = (cs_profile(r) / omega_k(r))
    #fact = (2.**0.5) * 3.
    fact = 1
    return fact * H # factor taken from phantom, to fasten thermalizing

def plot_curve_in():
    x = np.linspace(rin,rout)
    sigma = []
    kep = []
    cs = []
    rot = []
    H = []
    H_r = []
    for i in range(len(x)):
        _x = x[i]

        sigma.append(sigma_profile(_x))
        kep.append(kep_profile(_x))
        cs.append(cs_profile(_x))
        rot.append(rot_profile(_x))
        H.append(H_profile(_x))
        H_r.append(H_profile(_x)/_x)

    plt.plot(x,sigma, label = "sigma")
    plt.plot(x,kep, label = "keplerian speed")
    plt.plot(x,cs, label = "cs")
    plt.plot(x,rot, label = "rot speed")
    plt.plot(x,H, label = "H")
    plt.plot(x,H_r, label = "H_r")

if do_plots and False:
    plot_curve_in()
    plt.legend()
    plt.yscale("log")
    #plt.xscale("log")
    plt.show()


####################################################
# Split as binary
####################################################
def split_as_binary(sink,m1, m2, a, e, euler_angles = (0.,0.,0.)):
    roll, pitch, yaw = euler_angles
    
    m1 = float(m1)
    m2 = float(m2)
    a = float(a)
    e = float(e)
    roll = float(roll)
    pitch = float(pitch)
    yaw = float(yaw)

    r1, r2, v1, v2 =shamrock.phys.get_binary_rotated(
            m1=m1, m2=m2, a=a, e=e, nu=float(np.radians(0.)), G=G, roll=roll, pitch=pitch, yaw=yaw
        )

    s1 = {
        "mass": m1,
        "racc": sink["racc"],
        "pos" : (sink["pos"][0] + r1[0],sink["pos"][1] + r1[1],sink["pos"][2] + r1[2]),
        "vel" : (sink["vel"][0] + v1[0],sink["vel"][1] + v1[1],sink["vel"][2] + v1[2])}

    s2 = {
        "mass": m2,
        "racc": sink["racc"],
        "pos" : (sink["pos"][0] + r2[0],sink["pos"][1] + r2[1],sink["pos"][2] + r2[2]),
        "vel" : (sink["vel"][0] + v2[0],sink["vel"][1] + v2[1],sink["vel"][2] + v2[2])}

    print("-------------")
    print(sink)
    print(s1)
    print(s2)
    print(r1,r2,v1,v2)
    print("-------------")

    return s1, s2


####################################################
# Dump handling
####################################################
def get_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".sham"

def get_vtk_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".vtk"

def get_last_dump():
    import glob

    res = glob.glob(dump_prefix + "*.sham")

    f_max = ""
    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len(dump_prefix):-5])
            if dump_num > num_max:
                f_max = f
                num_max = dump_num
        except:
            pass

    if num_max == -1:
        return None
    else:
        return num_max

def purge_old_dumps():
    import glob

    res = glob.glob(dump_prefix + "*.sham")
    res.sort()
    
    torem = res[1:-3]
    #print(res,torem)

    for f in torem:
        os.remove(f)

idump_last_dump = get_last_dump()


####################################################
####################################################
####################################################

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

if idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
else:
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_VaryingCD10(alpha_min = 1e-3,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    #cfg.set_artif_viscosity_ConstantDisc(alpha_u = alpha_u, alpha_AV = alpha_AV, beta_AV = beta_AV)
    #cfg.set_eos_locally_isothermalLP07(cs0 = cs0, q = q, r0 = r0)
    cfg.set_eos_locally_isothermalFA2014(h_over_r = H_r_in)
    cfg.print_status()
    cfg.set_units(codeu)
    model.set_solver_config(cfg)

    model.init_scheduler(int(8e5),1)

    model.resize_simulation_box(bmin,bmax)

    sink_list = [
            {"mass": center_mass, "racc": center_racc, "pos" : (0,0,0), "vel" : (0,0,0)},
        ]

    print(f"sink_list = {sink_list}")

    for split in split_list:
        index_split = split["index"]
        mass_ratio = split["mass_ratio"]
        asplit = split["a"]
        esplit = split["e"]
        euler_angle_split = split["euler_angle"]

        print(f"splitting sink {split}")

        new_sink_list = []

        for i in range(len(sink_list)):
            if i == index_split:
                smass = sink_list[i]["mass"]

                s1,s2 = split_as_binary(
                    sink_list[i],
                    smass*mass_ratio,
                    smass*(1-mass_ratio),
                    asplit,
                    esplit,
                    euler_angle_split)

                new_sink_list.append(s1)
                new_sink_list.append(s2)
            else:
                new_sink_list.append(sink_list[i])

        sink_list = new_sink_list
        print(f"sink_list = {sink_list}")


    sum_mass = sum(s["mass"] for s in sink_list)
    vel_bary = (
        sum(s["mass"]*s["vel"][0] for s in sink_list) / sum_mass,
        sum(s["mass"]*s["vel"][1] for s in sink_list) / sum_mass,
        sum(s["mass"]*s["vel"][2] for s in sink_list) / sum_mass
    )
    pos_bary = (
        sum(s["mass"]*s["pos"][0] for s in sink_list) / sum_mass,
        sum(s["mass"]*s["pos"][1] for s in sink_list) / sum_mass,
        sum(s["mass"]*s["pos"][2] for s in sink_list) / sum_mass
    )
    print("sinks baryenceter : velocity {} position {}".format(vel_bary,pos_bary))

    #plot_sim_orbit(sink_list)
    #plot_sim_orbit2(sink_list)

    model.set_particle_mass(pmass)
    for s in sink_list:
        mass = s["mass"]
        x,y,z = s["pos"]
        vx,vy,vz = s["vel"]
        racc = s["racc"]

        x -= pos_bary[0]
        y -= pos_bary[1]
        z -= pos_bary[2]

        vx -= vel_bary[0]
        vy -= vel_bary[1]
        vz -= vel_bary[2]

        print("add sink : mass {} pos {} vel {} racc {}".format(mass,(x,y,z),(vx,vy,vz),racc))
        model.add_sink(mass,(x,y,z),(vx,vy,vz),racc)


    setup = model.get_setup()
    gen_disc = setup.make_generator_disc_mc(
            part_mass = pmass,
            disc_mass = disc_mass,
            r_in = rin,
            r_out = rout,
            sigma_profile = sigma_profile,
            H_profile = H_profile,
            rot_profile = rot_profile,
            cs_profile = cs_profile,
            random_seed = 666
        )
        
    setup.apply_setup(gen_disc)

    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)

    model.dump(init_dump)

    model.change_htolerance(1.3)
    model.timestep()
    model.change_htolerance(1.1)

def save_rho_integ(ext,sinks,arr_rho, iplot):
    metadata = {
        "extent": [-ext, ext, -ext, ext],
        "sinks": sinks,
        "time": model.get_time()
    }
    np.save(dump_folder + f"rho_integ_{iplot:07}.npy",arr_rho)
    import json
    with open(dump_folder + f"rho_integ_{iplot:07}.json", 'w') as fp:
        json.dump(metadata, fp)


def save_vxyz_xz(ext,sinks,arr_v, iplot):
    metadata = {
        "extent": [-ext, ext, -ext, ext],
        "sinks": sinks,
        "time": model.get_time()
    }
    np.save(dump_folder + f"vxyz_xz_{iplot:07}.npy",arr_v)
    import json
    with open(dump_folder + f"vxyz_xz_{iplot:07}.json", 'w') as fp:
        json.dump(metadata, fp)


def save_vxyz_xy(ext,sinks,arr_v, iplot):
    metadata = {
        "extent": [-ext, ext, -ext, ext],
        "sinks": sinks,
        "time": model.get_time()
    }
    np.save(dump_folder + f"vxyz_xy_{iplot:07}.npy",arr_v)
    import json
    with open(dump_folder + f"vxyz_xy_{iplot:07}.json", 'w') as fp:
        json.dump(metadata, fp)




def analysis_plot(iplot):
    sinks = model.get_sinks()

    x1,y1,z1 = sinks[0]["pos"]
    x2,y2,z2 = sinks[1]["pos"]

    ext = 500
    nx = 1024
    ny = 1024

    d_x = x2-x1
    d_y = y2-y1
    d_z = z2-z1
    d = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    d_x = d_x/d
    d_y = d_y/d
    d_z = d_z/d

    f = 2*ext
    e_r = (f*d_x,f*d_y,0)
    e_theta = (-f*d_y,f*d_x,0)
    e_z = (0,0,f*1)

    arr_rho2 = model.render_cartesian_column_integ("rho","f64",center = (0.,0.,0.),delta_x = (ext*2,0,0.),delta_y = (0.,ext*2,0.), nx = nx, ny = ny)
    arr_v_vslice_xz = model.render_cartesian_slice("vxyz","f64_3",center = (0.,0.,0.),delta_x = (ext*2,0,0.),delta_y = (0.,0.,ext*2), nx = nx, ny = ny)
    arr_v_vslice_xy = model.render_cartesian_slice("vxyz","f64_3",center = (0.,0.,0.),delta_x = (ext*2,0,0.),delta_y = (0.,ext*2,0.), nx = nx, ny = ny)

    if shamrock.sys.world_rank() == 0:
        save_rho_integ(ext,sinks,arr_rho2, iplot)
        save_vxyz_xz(ext,sinks,arr_v_vslice_xz, iplot)
        save_vxyz_xy(ext,sinks,arr_v_vslice_xy, iplot)



sink_history = []

t_start = model.get_time()

freq_stop = 1000
norbit = 5
dump_freq_stop = 50
plot_freq_stop = 50

dt_stop = 0.1
nstop = 5000

t_stop = [i*dt_stop for i in range(nstop+1)]

idump = 0
iplot = 0
istop = 0
for ttarg in t_stop:

    if ttarg >= t_start:
        model.evolve_until(ttarg)

        if istop % dump_freq_stop == 0:
            #model.do_vtk_dump(get_vtk_dump_name(idump), True)
            model.dump(get_dump_name(idump))
            purge_old_dumps()

        if istop % plot_freq_stop == 0:
            analysis_plot(iplot)

    if istop % dump_freq_stop == 0:
        idump += 1

    if istop % plot_freq_stop == 0:
        iplot += 1

    istop += 1