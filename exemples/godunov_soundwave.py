import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

model.init_scheduler(int(1e7), 1)

multx = 1
multy = 1
multz = 1

sz = 1 << 1
base = 32
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

cfg = model.gen_default_config()
scale_fact = 1 / (sz * base * multx)
cfg.set_scale_factor(scale_fact)

cfg.set_eos_gamma(5.0 / 3.0)
model.set_config(cfg)


gamma = 5.0 / 3.0

u_cs1 = 1 / (gamma * (gamma - 1))

kx, ky, kz = 4 * np.pi, 0, 0
delta_rho = 0
delta_v = 1e-4


def rho_map(rmin, rmax):

    x, y, z = rmin

    return 1.0 + delta_rho * np.cos(kx * x + ky * y + kz * z)


def rhoetot_map(rmin, rmax):

    rho = rho_map(rmin, rmax)

    x, y, z = rmin
    # return x
    return (u_cs1 + u_cs1 * delta_rho * np.cos(kx * x + ky * y + kz * z)) * rho


def rhovel_map(rmin, rmax):

    rho = rho_map(rmin, rmax)

    x, y, z = rmin
    return (0 + delta_v * np.cos(kx * x + ky * y + kz * z) * rho, 0, 0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

# model.evolve_once(0,0.1)
tmax = 0.127
dt = 1 / 1024
t = 0

freq = 16


for i in range(1000):

    if i % freq == 0:
        model.dump_vtk("test" + str(i // freq) + ".vtk")

    model.evolve_once_override_time(t, dt)
    t += dt

    if t >= tmax:
        break


def convert_to_cell_coords(dic):

    cmin = dic["cell_min"]
    cmax = dic["cell_max"]

    xmin = []
    ymin = []
    zmin = []
    xmax = []
    ymax = []
    zmax = []

    for i in range(len(cmin)):

        m, M = cmin[i], cmax[i]

        mx, my, mz = m
        Mx, My, Mz = M

        for j in range(8):
            a, b = model.get_cell_coords(((mx, my, mz), (Mx, My, Mz)), j)

            x, y, z = a
            xmin.append(x)
            ymin.append(y)
            zmin.append(z)

            x, y, z = b
            xmax.append(x)
            ymax.append(y)
            zmax.append(z)

    dic["xmin"] = np.array(xmin)
    dic["ymin"] = np.array(ymin)
    dic["zmin"] = np.array(zmin)
    dic["xmax"] = np.array(xmax)
    dic["ymax"] = np.array(ymax)
    dic["zmax"] = np.array(zmax)

    return dic


dic = convert_to_cell_coords(ctx.collect_data())


X = []
rho = []
velx = []
rhoe = []

for i in range(len(dic["xmin"])):

    X.append(dic["xmin"][i])
    rho.append(dic["rho"][i])
    velx.append(dic["rhovel"][i][0])
    rhoe.append(dic["rhoetot"][i])


fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(X, rho, ".", label="rho")
axs[1].plot(X, velx, ".", label="vx")
axs[2].plot(X, rhoe, ".", label="rhoe")


plt.show()
