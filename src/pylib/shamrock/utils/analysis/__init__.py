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


class perf_history:
    def __init__(self, model, analysis_folder, analysis_prefix):
        self.model = model

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix)
        self.plot_prefix = os.path.join(analysis_folder, "plot_" + analysis_prefix)

        self.json_data_filename = self.analysis_prefix + ".json"
        self.plot_filename = self.plot_prefix

    def analysis_save(self, iplot):
        sim_time_delta = self.model.solver_logs_cumulated_step_time()
        scount = self.model.solver_logs_step_count()
        part_count = self.model.get_total_part_count()

        self.model.solver_logs_reset_cumulated_step_time()
        self.model.solver_logs_reset_step_count()

        if shamrock.sys.world_rank() == 0:
            perf_hist_new = {
                "time": self.model.get_time(),
                "sim_time_delta": sim_time_delta,
                "world_size": shamrock.sys.world_size(),
                "sim_step_count_delta": scount,
                "part_count": part_count,
            }

            try:
                with open(self.json_data_filename, "r") as fp:
                    perf_hist = json.load(fp)
            except (FileNotFoundError, json.JSONDecodeError):
                perf_hist = {"history": []}

            perf_hist["history"] = perf_hist["history"][:iplot] + [perf_hist_new]

            if scount == 0:
                print("Warning: step count is 0, skipping save of perf history")
                return

            with open(self.json_data_filename, "w") as fp:
                print(f"Saving perf history to {self.json_data_filename}")
                json.dump(perf_hist, fp, indent=4)

    def load_analysis(self):
        with open(self.json_data_filename, "r") as fp:
            perf_hist = json.load(fp)
        return perf_hist

    def digest_perf_history(self):
        perf_hist = self.load_analysis()

        t = [h["time"] for h in perf_hist["history"]]
        sim_time_delta = [h["sim_time_delta"] for h in perf_hist["history"]]
        world_size = [h["world_size"] for h in perf_hist["history"]]
        sim_step_count_delta = [h["sim_step_count_delta"] for h in perf_hist["history"]]
        part_count = [h["part_count"] for h in perf_hist["history"]]

        t = np.array(t)
        dt_code = np.diff(t)

        sim_time_delta = np.array(sim_time_delta)
        world_size = np.array(world_size)
        sim_time_delta_all_proc = sim_time_delta * world_size
        sim_step_count_delta = np.array(sim_step_count_delta)
        part_count = np.array(part_count)

        # cumulative sim_time & step_count
        cum_sim_time_delta = np.cumsum(sim_time_delta)
        cum_sim_time_delta_all_proc = np.cumsum(sim_time_delta_all_proc)
        cum_sim_step_count_delta = np.cumsum(sim_step_count_delta)

        tsim_per_hour = dt_code / (sim_time_delta[1:] / 3600)

        time_per_step = []

        for td, sc, pc in zip(sim_time_delta, sim_step_count_delta, part_count):
            if sc > 0:
                time_per_step.append(td / sc)
            else:
                # NAN here because the step count is 0
                time_per_step.append(np.nan)

        rate = []

        for td, sc, pc in zip(sim_time_delta, sim_step_count_delta, part_count):
            if sc > 0:
                rate.append(pc / (td / sc))
            else:
                # NAN here because the step count is 0
                rate.append(np.nan)

        return {
            "t": t,
            "dt_code": dt_code,
            "part_count": part_count,
            "world_size": world_size,
            "sim_time_delta": sim_time_delta,
            "sim_step_count_delta": sim_step_count_delta,
            "cum_sim_time_delta": cum_sim_time_delta,
            "cum_sim_time_delta_all_proc": cum_sim_time_delta_all_proc,
            "cum_sim_step_count_delta": cum_sim_step_count_delta,
            "time_per_step": time_per_step,
            "rate": rate,
            "tsim_per_hour": tsim_per_hour,
        }

    def plot_perf_history(self, close_plots=True, figsize=(8, 5), dpi=200):
        if not _HAS_MATPLOTLIB:
            print("Warning: matplotlib is not installed, plot_perf_history is a no-op")
            return

        if shamrock.sys.world_rank() == 0:
            perf_hist = self.digest_perf_history()

            print(f"Plotting perf history from {self.json_data_filename}")

            t = perf_hist["t"]

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["cum_sim_time_delta"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("t [s] (real time)")
            plt.savefig(self.plot_filename + "_sim_time.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(
                t,
                perf_hist["cum_sim_time_delta_all_proc"] / 3600.0,
                "+-",
                label="Used compute time",
            )
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel(r"$\sum_{processes} t$ [h] (real time)")

            ax1 = plt.gca()

            # Right y-axis
            ax2 = ax1.twinx()
            ax2.plot(t, perf_hist["world_size"], "+-", color="tab:orange", label="World size")
            ax2.set_ylabel("World size")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

            plt.savefig(self.plot_filename + "_sim_time_all_proc.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["cum_sim_step_count_delta"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$N_\\mathrm{step}$")
            plt.savefig(self.plot_filename + "_step_count.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["sim_time_delta"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$d t_\\mathrm{real} / d i_\\mathrm{analysis}$ [s]")
            plt.savefig(self.plot_filename + "_sim_time_delta.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["sim_step_count_delta"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$d N_\\mathrm{step} / d i_\\mathrm{analysis}$")
            plt.savefig(self.plot_filename + "_step_count_delta.png")
            if close_plots:
                plt.close()

            # tsim per hour
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t[1:], perf_hist["tsim_per_hour"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("$d t_\\mathrm{sim} / d t_\\mathrm{realtime}$ [code unit (time) / hour]")
            plt.savefig(self.plot_filename + "_tsim_per_hour.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["time_per_step"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("time per step [s]")
            plt.savefig(self.plot_filename + "_time_per_step.png")
            if close_plots:
                plt.close()

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, perf_hist["rate"], "+-")
            plt.xlabel("t [code unit] (simulation)")
            plt.ylabel("Particles / second")
            plt.yscale("log")
            plt.savefig(self.plot_filename + "_rate.png")
            if close_plots:
                plt.close()


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

    def slice_render(self, field_name, field_type, do_normalization=True, min_normalization=1e-9):
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
        ret[ret < min_normalization] = np.nan

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

            # Reset the figure using the same memory as the last one
            figsize = (self.helper.aspect * 6, 1.0 * 6)

            if not holywood_mode:
                fx, fy = figsize
                figsize = (fx + 1, fy)

            dpi = 200
            plt.figure(figsize=figsize, num=1, clear=True, dpi=dpi)

            if holywood_mode:
                plt.gca().set_position((0, 0, 1, 1))
                plt.gcf().set_size_inches(self.helper.nx / dpi, self.helper.ny / dpi)
                plt.axis("off")

            import copy

            my_cmap = matplotlib.colormaps["magma"].copy()  # copy the default cmap
            my_cmap.set_bad(color="black")

            res = plt.imshow(
                arr_rho_xy, cmap=my_cmap, origin="lower", extent=metadata["extent"], **kwargs
            )

            ax = plt.gca()

            sink_list_plot = self.helper.metadata_to_screen_sink_pos(metadata)
            output_list = []
            for x, y, s in sink_list_plot:
                output_list.append(
                    plt.Circle(
                        (x, y),
                        s["accretion_radius"] * 5,
                        linewidth=0.5,
                        color="blue",
                        fill=False,
                    )
                )
            for circle in output_list:
                ax.add_artist(circle)

            plt.xlabel("x [au]")
            plt.ylabel("y [au]")
            text = "t = {:0.3f} [Year]".format(metadata["time"])

            if holywood_mode:
                from matplotlib.offsetbox import AnchoredText

                anchored_text = AnchoredText(text, loc=2)
                plt.gca().add_artist(anchored_text)
            else:
                plt.title(text)

            if holywood_mode:
                axins = plt.gca().inset_axes([0.73, 0.1, 0.25, 0.025])
                cbar = plt.colorbar(res, cax=axins, orientation="horizontal", extend="both")
                cbar.set_label(r"$\int \rho \, \mathrm{d} z$ [code unit]", color="white")

                # Set colorbar elements to white
                cbar.outline.set_edgecolor("white")
                # cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(cbar.ax.get_yticklabels(), color="white")
                plt.setp(cbar.ax.get_xticklabels(), color="white")
                cbar.ax.tick_params(color="white", labelcolor="white", length=6, width=1)

            else:
                cbar = plt.colorbar(res, extend="both")
                cbar.set_label(r"$\int \rho \, \mathrm{d} z$ [code unit]")

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
