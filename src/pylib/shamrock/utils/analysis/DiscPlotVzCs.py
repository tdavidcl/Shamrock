import json
import os

import numpy as np

import shamrock.sys

try:
    import matplotlib
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from .StandardPlotHelper import DiscSlicePlotHelper
from .UnitHelper import plot_codeu_to_unit


class DiscPlotVzCs:
    def init_custom_getter(self):
        if _HAS_NUMBA:
            if shamrock.sys.world_rank() == 0:
                print("Using numba for velocity profile in DiscPlotVzCs")

        def internal(size: int, vz: np.array, cs: np.array) -> np.array:
            v_z_cs = np.zeros(size)
            for i in range(size):
                v_z_cs[i] = vz[i] / cs[i]
            return v_z_cs

        if _HAS_NUMBA:
            internal = njit(internal)

        def custom_getter(size: int, dic_out: dict) -> np.array:
            return internal(
                size,
                dic_out["vxyz"][:, 2],
                dic_out["soundspeed"],
            )

        self.custom_getter = custom_getter

    def __init__(
        self,
        model,
        ext_r,
        z_r_max,
        nr,
        nz,
        center,
        analysis_folder,
        analysis_prefix,
        do_normalization=True,
        min_normalization=1e-9,
        deproject=False,
    ):
        self.model = model
        self.helper = DiscSlicePlotHelper(
            model, ext_r, z_r_max, nr, nz, center, analysis_folder, analysis_prefix, deproject
        )
        self.do_normalization = do_normalization
        self.min_normalization = min_normalization
        self.init_custom_getter()

    def compute(self):
        arr_v = self.helper.azymuthal_average_render_explicit(
            "custom",
            "f64",
            do_normalization=self.do_normalization,
            min_normalization=self.min_normalization,
            custom_getter=self.custom_getter,
        )

        return arr_v

    def analysis_save(self, iplot):
        arr = self.compute()
        self.helper.analysis_save(iplot, arr)

    def load_analysis(self, iplot):
        return self.helper.load_analysis(iplot)

    def get_list_analysis_id(self):
        return self.helper.get_list_analysis_id()

    def plot(
        self,
        iplot,
        holywood_mode=False,
        dist_unit="au",
        time_unit="year",
        **kwargs,
    ):
        if shamrock.sys.world_rank() == 0:
            arr_v_z_cs, metadata = self.load_analysis(iplot)

            dist_label, dist_conv = plot_codeu_to_unit(self.model.get_units(), dist_unit)
            metadata["extent"] = [metadata["extent"][i] * dist_conv for i in range(4)]

            time_label, time_conv = plot_codeu_to_unit(self.model.get_units(), time_unit)
            metadata["time"] *= time_conv

            self.helper.figure_init(holywood_mode)

            import copy

            my_cmap = matplotlib.colormaps["seismic"].copy()  # copy the default cmap
            my_cmap.set_bad(color="white")

            res = plt.imshow(
                arr_v_z_cs,
                cmap=my_cmap,
                origin="lower",
                extent=metadata["extent"],
                aspect="auto",
                **kwargs,
            )

            ax = plt.gca()

            plt.xlabel(f"r {dist_label}")
            plt.ylabel("z/r [unitless]")

            text = f"t = {metadata['time']:0.3f} {time_label}"
            self.helper.figure_add_time_info(text, holywood_mode)

            cmap_label = "$\\mathrm{v}_z / c_s$ [unitless]"
            self.helper.figure_add_colorbar(res, cmap_label, holywood_mode)

            print(f"Saving plot to {self.helper.plot_filename.format(iplot)}")
            plt.savefig(self.helper.plot_filename.format(iplot))
            plt.close()

    def render_all(self, holywood_mode=False, **kwargs):
        for iplot in self.get_list_analysis_id():
            self.plot(iplot, holywood_mode, **kwargs)

    def render_gif(
        self, save_animation=False, fps=15, bitrate=1800, gif_filename="disc_plot_vz_cs.gif"
    ):
        if shamrock.sys.world_rank() == 0:
            ani = shamrock.utils.plot.show_image_sequence(
                self.helper.glob_str_plot, render_gif=True
            )
            if save_animation:
                # To save the animation using Pillow as a gif
                writer = animation.PillowWriter(
                    fps=fps, metadata=dict(artist="Me"), bitrate=bitrate
                )
                ani.save(self.helper.analysis_prefix + gif_filename, writer=writer)
            return ani
        return None
