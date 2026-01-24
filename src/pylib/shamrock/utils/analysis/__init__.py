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

from .PerfHistory import PerfHistory
from .StandardPlotHelper import StandardPlotHelper


class column_density_plot:
    def __init__(self, model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix):
        self.model = model
        self.helper = StandardPlotHelper(
            model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix
        )

    def compute_rho_xy(self):
        arr_rho_xy = self.helper.column_integ_render("rho", "f64")

        # Convert to kg/m^2
        codeu = self.model.get_units()
        kg_m2_codeu = codeu.get("kg") * codeu.get("m", power=-2)
        arr_rho_xy /= kg_m2_codeu

        return arr_rho_xy

    def analysis_save(self, iplot):
        arr_rho_xy = self.compute_rho_xy()
        self.helper.analysis_save(iplot, arr_rho_xy)

    def load_analysis(self, iplot):
        return self.helper.load_analysis(iplot)

    def get_list_analysis_id(self):
        return self.helper.get_list_analysis_id()

    def plot_rho_xy(self, iplot, holywood_mode=False, **kwargs):
        if shamrock.sys.world_rank() == 0:
            arr_rho_xy, metadata = self.load_analysis(iplot)

            self.helper.figure_init(holywood_mode)

            import copy

            my_cmap = matplotlib.colormaps["magma"].copy()  # copy the default cmap
            my_cmap.set_bad(color="black")

            res = plt.imshow(
                arr_rho_xy, cmap=my_cmap, origin="lower", extent=metadata["extent"], **kwargs
            )

            ax = plt.gca()

            self.helper.figure_render_sinks(metadata, ax)

            plt.xlabel("x [au]")
            plt.ylabel("y [au]")

            text = "t = {:0.3f} [Year]".format(metadata["time"])
            self.helper.figure_add_time_info(text, holywood_mode)

            cmap_label = r"$\int \rho \, \mathrm{d} z$ [code unit]"
            self.helper.figure_add_colorbar(res, cmap_label, holywood_mode)

            print(f"Saving plot to {self.helper.plot_filename.format(iplot)}")
            plt.savefig(self.helper.plot_filename.format(iplot))
            plt.close()

    def render_all(self, holywood_mode=False, **kwargs):
        for iplot in self.get_list_analysis_id():
            self.plot_rho_xy(iplot, holywood_mode, **kwargs)

    def render_gif(self, save_animation=False, show_animation=False):
        if shamrock.sys.world_rank() == 0:
            ani = shamrock.utils.plot.show_image_sequence(
                self.helper.glob_str_plot, render_gif=True
            )
            if save_animation:
                # To save the animation using Pillow as a gif
                writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
                ani.save(self.helper.analysis_prefix + "rho_integ.gif", writer=writer)
            if show_animation:
                plt.show()


class slice_density_plot:
    def __init__(
        self,
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        do_normalization=True,
        min_normalization=1e-9,
    ):
        self.model = model
        self.helper = StandardPlotHelper(
            model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix
        )
        self.do_normalization = do_normalization
        self.min_normalization = min_normalization

    def compute_rho_xy(self):
        arr_rho_xy = self.helper.slice_render(
            "rho", "f64", self.do_normalization, self.min_normalization
        )

        # Convert to kg/m^2
        codeu = self.model.get_units()
        kg_m2_codeu = codeu.get("kg") * codeu.get("m", power=-3)
        arr_rho_xy /= kg_m2_codeu

        return arr_rho_xy

    def analysis_save(self, iplot):
        arr_rho_xy = self.compute_rho_xy()
        self.helper.analysis_save(iplot, arr_rho_xy)

    def load_analysis(self, iplot):
        return self.helper.load_analysis(iplot)

    def get_list_analysis_id(self):
        return self.helper.get_list_analysis_id()

    def plot_rho_xy(self, iplot, holywood_mode=False, **kwargs):
        if shamrock.sys.world_rank() == 0:
            arr_rho_xy, metadata = self.load_analysis(iplot)

            self.helper.figure_init(holywood_mode)

            import copy

            my_cmap = matplotlib.colormaps["magma"].copy()  # copy the default cmap
            my_cmap.set_bad(color="black")

            res = plt.imshow(
                arr_rho_xy, cmap=my_cmap, origin="lower", extent=metadata["extent"], **kwargs
            )

            ax = plt.gca()

            self.helper.figure_render_sinks(metadata, ax)

            plt.xlabel("x [au]")
            plt.ylabel("y [au]")

            text = "t = {:0.3f} [Year]".format(metadata["time"])
            self.helper.figure_add_time_info(text, holywood_mode)

            cmap_label = r"$\rho$ [code unit]"
            self.helper.figure_add_colorbar(res, cmap_label, holywood_mode)

            print(f"Saving plot to {self.helper.plot_filename.format(iplot)}")
            plt.savefig(self.helper.plot_filename.format(iplot))
            plt.close()

    def render_all(self, holywood_mode=False, **kwargs):
        for iplot in self.get_list_analysis_id():
            self.plot_rho_xy(iplot, holywood_mode, **kwargs)

    def render_gif(self, save_animation=False, show_animation=False):
        if shamrock.sys.world_rank() == 0:
            ani = shamrock.utils.plot.show_image_sequence(
                self.helper.glob_str_plot, render_gif=True
            )
            if save_animation:
                # To save the animation using Pillow as a gif
                writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
                ani.save(self.helper.analysis_prefix + "rho_slice.gif", writer=writer)
            if show_animation:
                plt.show()


class v_z_slice_plot:
    def __init__(
        self,
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        do_normalization=True,
        min_normalization=1e-9,
    ):
        self.model = model
        self.helper = StandardPlotHelper(
            model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix
        )
        self.do_normalization = do_normalization
        self.min_normalization = min_normalization

    def compute_v_z(self):
        def keep_only_v_z(arr_v):
            return arr_v[:, :, 2]

        arr_v = self.helper.slice_render(
            "vxyz", "f64_3", self.do_normalization, self.min_normalization, keep_only_v_z
        )

        # Convert to kg/m^2
        codeu = self.model.get_units()
        m_s_codeu = codeu.get("m") * codeu.get("s", power=-1)
        arr_v /= m_s_codeu

        return arr_v

    def analysis_save(self, iplot):
        arr_v_z = self.compute_v_z()
        self.helper.analysis_save(iplot, arr_v_z)

    def load_analysis(self, iplot):
        return self.helper.load_analysis(iplot)

    def get_list_analysis_id(self):
        return self.helper.get_list_analysis_id()

    def plot_v_z(self, iplot, holywood_mode=False, **kwargs):
        if shamrock.sys.world_rank() == 0:
            arr_v_z, metadata = self.load_analysis(iplot)

            self.helper.figure_init(holywood_mode)

            import copy

            my_cmap = matplotlib.colormaps["seismic"].copy()  # copy the default cmap
            my_cmap.set_bad(color="white")

            res = plt.imshow(
                arr_v_z, cmap=my_cmap, origin="lower", extent=metadata["extent"], **kwargs
            )

            ax = plt.gca()

            self.helper.figure_render_sinks(metadata, ax)

            plt.xlabel("x [au]")
            plt.ylabel("y [au]")

            text = "t = {:0.3f} [Year]".format(metadata["time"])
            self.helper.figure_add_time_info(text, holywood_mode)

            cmap_label = r"$v_z$ [code unit]"
            self.helper.figure_add_colorbar(res, cmap_label, holywood_mode)

            print(f"Saving plot to {self.helper.plot_filename.format(iplot)}")
            plt.savefig(self.helper.plot_filename.format(iplot))
            plt.close()

    def render_all(self, holywood_mode=False, **kwargs):
        for iplot in self.get_list_analysis_id():
            self.plot_v_z(iplot, holywood_mode, **kwargs)

    def render_gif(self, save_animation=False, show_animation=False):
        if shamrock.sys.world_rank() == 0:
            ani = shamrock.utils.plot.show_image_sequence(
                self.helper.glob_str_plot, render_gif=True
            )
            if save_animation:
                # To save the animation using Pillow as a gif
                writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
                ani.save(self.helper.analysis_prefix + "v_z_slice_plot.gif", writer=writer)
            if show_animation:
                plt.show()
