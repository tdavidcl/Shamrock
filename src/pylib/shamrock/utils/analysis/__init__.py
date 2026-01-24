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


class standard_plot_analysis_helper:
    def __init__(self, model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix):
        self.model = model
        self.ext_r = ext_r
        self.nx = nx
        self.ny = ny
        self.ex = ex
        self.ey = ey
        self.center = center
        self.aspect = float(self.nx) / float(self.ny)

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix) + "_"
        self.plot_prefix = os.path.join(analysis_folder, "plot_" + analysis_prefix) + "_"

        self.npy_data_filename = self.analysis_prefix + "{:07}.npy"
        self.json_data_filename = self.analysis_prefix + "{:07}.json"
        self.plot_filename = self.plot_prefix + "{:07}.png"
        self.glob_str_plot = self.plot_prefix + "*.png"
        self.glob_str_data = self.analysis_prefix + "*.json"  # json is writen in last

    def get_dx_dy(self):
        ext_x = 2 * self.ext_r * self.aspect
        ext_y = 2 * self.ext_r

        dx = (self.ex[0] * ext_x, self.ex[1] * ext_x, self.ex[2] * ext_x)
        dy = (self.ey[0] * ext_y, self.ey[1] * ext_y, self.ey[2] * ext_y)

        return dx, dy

    def column_integ_render(self, field_name, field_type):
        dx, dy = self.get_dx_dy()
        arr_field = self.model.render_cartesian_column_integ(
            field_name,
            field_type,
            center=(self.center[0], self.center[1], self.center[2]),
            delta_x=dx,
            delta_y=dy,
            nx=self.nx,
            ny=self.ny,
        )

        return arr_field

    def slice_render(
        self,
        field_name,
        field_type,
        do_normalization=True,
        min_normalization=1e-9,
        field_transform=None,
    ):
        dx, dy = self.get_dx_dy()
        arr_field_data = self.model.render_cartesian_slice(
            field_name,
            field_type,
            center=(self.center[0], self.center[1], self.center[2]),
            delta_x=dx,
            delta_y=dy,
            nx=self.nx,
            ny=self.ny,
        )

        if field_transform is not None:
            arr_field_data = field_transform(arr_field_data)

        if not do_normalization:
            return arr_field_data

        arr_field_normalization = self.model.render_cartesian_slice(
            "unity",
            "f64",
            center=(self.center[0], self.center[1], self.center[2]),
            delta_x=dx,
            delta_y=dy,
            nx=self.nx,
            ny=self.ny,
        )
        ret = arr_field_data / arr_field_normalization

        # set to nan below min_normalization
        ret[arr_field_normalization < min_normalization] = np.nan

        return ret

    def analysis_save(self, iplot, data):
        if shamrock.sys.world_rank() == 0:
            x_e_x = (
                self.ex[0] * self.center[0]
                + self.ex[1] * self.center[1]
                + self.ex[2] * self.center[2]
            )
            y_e_y = (
                self.ey[0] * self.center[0]
                + self.ey[1] * self.center[1]
                + self.ey[2] * self.center[2]
            )

            metadata = {
                "extent": [
                    -self.ext_r * self.aspect + x_e_x,
                    self.ext_r * self.aspect + x_e_x,
                    -self.ext_r + y_e_y,
                    self.ext_r + y_e_y,
                ],
                "time": self.model.get_time(),
                "sinks": self.model.get_sinks(),
            }

            print(f"Saving data to {self.npy_data_filename.format(iplot)}")
            np.save(self.npy_data_filename.format(iplot), data)

            with open(self.json_data_filename.format(iplot), "w") as fp:
                print(f"Saving metadata to {self.json_data_filename.format(iplot)}")
                json.dump(metadata, fp)

    def load_analysis(self, iplot):
        with open(self.json_data_filename.format(iplot), "r") as fp:
            metadata = json.load(fp)
        return np.load(self.npy_data_filename.format(iplot)), metadata

    def get_list_analysis_id(self):
        import glob

        list_files = glob.glob(self.glob_str_data)
        list_files.sort()
        list_analysis_id = []
        for f in list_files:
            list_analysis_id.append(int(f.split("_")[-1].split(".")[0]))
        return list_analysis_id

    def metadata_to_screen_sink_pos(self, metadata):
        output_list = []
        for s in metadata["sinks"]:
            # print(s)
            x, y, z = s["pos"]

            x_e_x = self.ex[0] * x + self.ex[1] * y + self.ex[2] * z
            y_e_y = self.ey[0] * x + self.ey[1] * y + self.ey[2] * z

            output_list.append((x_e_x, y_e_y, s))
        return output_list

    def figure_init(self, holywood_mode=False, dpi=200):
        figsize = (self.aspect * 6, 1.0 * 6)

        if not holywood_mode:
            fx, fy = figsize
            figsize = (fx + 1, fy)

        dpi = 200

        # Reset the figure using the same memory as the last one
        plt.figure(figsize=figsize, num=1, clear=True, dpi=dpi)

        if holywood_mode:
            plt.gca().set_position((0, 0, 1, 1))
            plt.gcf().set_size_inches(self.nx / dpi, self.ny / dpi)
            plt.axis("off")

    def figure_render_sinks(
        self, metadata, ax, scale_factor=5, color="green", linewidth=1, fill=False
    ):
        sink_list_plot = self.metadata_to_screen_sink_pos(metadata)
        output_list = []
        for x, y, s in sink_list_plot:
            output_list.append(
                plt.Circle(
                    (x, y),
                    s["accretion_radius"] * scale_factor,
                    linewidth=linewidth,
                    color=color,
                    fill=fill,
                )
            )
        for circle in output_list:
            ax.add_artist(circle)

    def figure_add_time_info(self, text, holywood_mode=False):
        if holywood_mode:
            from matplotlib.offsetbox import AnchoredText

            anchored_text = AnchoredText(text, loc=2)
            plt.gca().add_artist(anchored_text)
        else:
            plt.title(text)

    def figure_add_colorbar(self, imshow_result, label, holywood_mode=False):
        if holywood_mode:
            axins = plt.gca().inset_axes([0.73, 0.1, 0.25, 0.025])
            cbar = plt.colorbar(imshow_result, cax=axins, orientation="horizontal", extend="both")
            cbar.set_label(label, color="white")

            # Set colorbar elements to white
            cbar.outline.set_edgecolor("white")
            # cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.get_yticklabels(), color="white")
            plt.setp(cbar.ax.get_xticklabels(), color="white")
            cbar.ax.tick_params(color="white", labelcolor="white", length=6, width=1)

        else:
            cbar = plt.colorbar(imshow_result, extend="both")
            cbar.set_label(label)


class column_density_plot:
    def __init__(self, model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix):
        self.model = model
        self.helper = standard_plot_analysis_helper(
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
        self.helper = standard_plot_analysis_helper(
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
        self.helper = standard_plot_analysis_helper(
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
