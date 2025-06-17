
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib

dump_folder = "AB_Aurigae_lim500_100000"
dump_folder += "/"

def plot_rho_integ(metadata, arr_rho, iplot):

    ext = metadata["extent"]
    sinks = metadata["sinks"]

    dpi = 200
    
    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True,dpi=dpi)
    import copy
    my_cmap = copy.copy(matplotlib.colormaps.get_cmap('gist_heat')) # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(arr_rho, cmap=my_cmap,origin='lower', extent=ext, norm="log", vmin=1e-12, vmax=1e-8)

    ax = plt.gca()

    output_list = []
    for s in sinks:
        print(s)
        x,y,z = s["pos"]
        output_list.append(
            plt.Circle((x, y), s["accretion_radius"], color="blue", fill=False))
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [years]")

    cbar = plt.colorbar(res, extend='both')
    cbar.set_label(r"$\int \rho \, \mathrm{d}z$ [code unit]")

    plt.savefig(dump_folder + "plot_rho_integ_{:04}.png".format(iplot))

def plot_vz_z(metadata, arr_vz, iplot):

    ext = metadata["extent"]
    sinks = metadata["sinks"]

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True,dpi=200)

    v_ext = np.max(arr_vz)
    v_ext = max(v_ext,np.abs(np.min(arr_vz)))
    res = plt.imshow(arr_vz, cmap="seismic",origin='lower', extent=ext, vmin=-v_ext, vmax=v_ext)

    ax = plt.gca()

    output_list = []
    for s in sinks:
        print(s)
        x,y,z = s["pos"]
        output_list.append(
            plt.Circle((x, z), s["accretion_radius"], color="blue", fill=False))
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"t = {metadata['time']:0.3f} [years]")

    cbar = plt.colorbar(res, extend='both')
    cbar.set_label(r"$v_z$ [code unit]")

    plt.savefig(dump_folder + "plot_vz_z_{:04}.png".format(iplot))

def get_list_dumps_id():
    import glob
    list_files = glob.glob(dump_folder+"rho_integ_*.npy")
    list_files.sort()
    list_dumps_id = []
    for f in list_files:
        list_dumps_id.append(int(f.split('_')[-1].split('.')[0]))
    return list_dumps_id

analysis_files = get_list_dumps_id()

def load_rho_integ(iplot):
    with open(dump_folder + f"rho_integ_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(dump_folder + f"rho_integ_{iplot:07}.npy"), metadata


def load_vxyz_xz(iplot):
    with open(dump_folder + f"vxyz_xz_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(dump_folder + f"vxyz_xz_{iplot:07}.npy"), metadata


def load_vxyz_xy(iplot):
    with open(dump_folder + f"vxyz_xy_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(dump_folder + f"vxyz_xy_{iplot:07}.npy"), metadata

for iplot in analysis_files:
    arr_rho, metadata = load_rho_integ(iplot)
    plot_rho_integ(metadata,arr_rho, iplot)

    arr_vz, metadata = load_vxyz_xz(iplot)
    plot_vz_z(metadata,arr_vz[:,:,2], iplot)
