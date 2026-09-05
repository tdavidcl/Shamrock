import json
import os

import numpy as np

import shamrock

try:
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import cm

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class MassAnalysis:
    """
    Analysis utility to report the disc (gas + dust) mass history during the simulation.
    """

    def __init__(
        self, model, analysis_folder, analysis_prefix, time_unit="yr", mass_unit="sol_mass"
    ):
        self.model = model
        self.time_unit = time_unit
        self.mass_unit = mass_unit

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix)
        self.plot_prefix = os.path.join(analysis_folder, "plot_" + analysis_prefix)

        self.json_data_filename = self.analysis_prefix + ".json"
        self.plot_filename = self.plot_prefix

    def analysis_save(self, iplot):
        model = self.model

        solver_config = model.get_current_config().to_json()
        has_dust = not (solver_config["dust_config"]["mode"]["type"] == "none")

        model_units = model.get_current_config().get_units()

        barycenter, disc_mass = shamrock.model_sph.analysisBarycenter(model=model).get_barycenter()

        mass_hist_new = {
            "time": self.model.get_time(),
            "disc_mass": disc_mass,
            "barycenter": barycenter,
            "unit_system": model_units.to_json(),
        }

        if has_dust:
            dust_mass = shamrock.model_sph.analysisDustMass(model=model).get_dust_mass()
            mass_hist_new["dust_mass"] = dust_mass

            drag_mode = solver_config["dust_config"]["drag_mode"]

            if "grains_sizes" in drag_mode:
                mass_hist_new["grains_sizes"] = drag_mode["grains_sizes"]

        if shamrock.sys.world_rank() == 0:
            try:
                with open(self.json_data_filename, "r") as fp:
                    mass_hist = json.load(fp)
            except (FileNotFoundError, json.JSONDecodeError):
                mass_hist = {"history": []}

            mass_hist["history"] = mass_hist["history"][:iplot] + [mass_hist_new]

            with open(self.json_data_filename, "w") as fp:
                print(f"Saving mass history to {self.json_data_filename}")
                json.dump(mass_hist, fp, indent=4)

    def load_analysis(self):
        with open(self.json_data_filename, "r") as fp:
            mass_hist = json.load(fp)
        return mass_hist

    def digest_perf_history(self, remove_null, smooth_window):
        mass_hist = self.load_analysis()

        has_dust = any("dust_mass" in h for h in mass_hist["history"])

        unit_system_json = next(
            (h["unit_system"] for h in mass_hist["history"] if "unit_system" in h), None
        )
        if unit_system_json is not None:
            model_units = shamrock.UnitSystem(**unit_system_json)
            time_conv = model_units.to(self.time_unit)
            mass_conv = model_units.to(self.mass_unit)
            length_conv = model_units.to("m")
        else:
            time_conv = 1.0
            mass_conv = 1.0
            length_conv = 1.0

        def rolling_smooth(arr, axis=0):
            # Average each sample with its neighbors within smooth_window
            # (a duration, in time_unit) of it in time.
            if smooth_window is None or arr.shape[axis] < 2:
                return arr
            half_window = smooth_window / 2.0
            smoothed = np.empty_like(arr, dtype=float)
            for i in range(t.shape[0]):
                mask = np.abs(t - t[i]) <= half_window
                smoothed[i] = np.mean(np.take(arr, np.nonzero(mask)[0], axis=axis), axis=axis)
            return smoothed

        def time_gradient(arr, axis=0):
            # np.gradient needs at least 2 samples along the gradient axis
            if t.shape[0] < 2:
                return np.array([])
            return rolling_smooth(np.gradient(arr, t, axis=axis), axis=axis)

        t = [h["time"] for h in mass_hist["history"]]
        disc_mass = [h["disc_mass"] for h in mass_hist["history"]]

        t = np.array(t) * time_conv
        disc_mass = np.array(disc_mass) * mass_conv

        result = {
            "t": t,
            "disc_mass": disc_mass,
            "d_disc_mass_dt": time_gradient(disc_mass),
        }

        if has_dust:
            dust_mass = [h["dust_mass"] for h in mass_hist["history"]]

            if remove_null:
                dust_mass = [[np.nan for v in dm] if np.max(dm) == 0 else dm for dm in dust_mass]

            dust_mass = np.array(dust_mass) * mass_conv
            dust_mass_all = np.sum(dust_mass, axis=-1)
            gas_mass = disc_mass - dust_mass_all

            result["dust_mass"] = dust_mass
            result["dust_mass_all"] = dust_mass_all
            result["gas_mass"] = gas_mass

            grains_sizes = next(
                h["grains_sizes"] for h in mass_hist["history"] if "grains_sizes" in h
            )
            result["grains_sizes"] = np.array(grains_sizes) * length_conv

            result["d_dust_mass_dt"] = time_gradient(dust_mass)
            result["d_dust_mass_all_dt"] = time_gradient(dust_mass_all)
            result["d_gas_mass_dt"] = time_gradient(gas_mass)

        return result

    @staticmethod
    def _symlog_linthresh(*arrays):
        """Pick a symlog linear threshold from the smallest data scale present."""
        abs_vals = np.concatenate([np.abs(a).ravel() for a in arrays if a.size > 0])
        abs_vals = abs_vals[np.isfinite(abs_vals) & (abs_vals > 0)]
        if abs_vals.size == 0:
            return 1e-10
        return np.nanmax(abs_vals) * 1e-6

    def plot_history(
        self, close_plots=True, figsize=(8, 5), dpi=200, remove_null=True, smooth_window=None
    ):
        if not _HAS_MATPLOTLIB:
            print("Warning: matplotlib is not installed, plot_perf_history is a no-op")
            return

        if shamrock.sys.world_rank() == 0:
            mass_hist = self.digest_perf_history(
                remove_null=remove_null, smooth_window=smooth_window
            )

            print(f"Plotting mass history from {self.json_data_filename}")

            t = mass_hist["t"]

            mass_unit_text = self.mass_unit
            if mass_unit_text == "sol_mass":
                mass_unit_text = "sol mass"

            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(t, mass_hist["disc_mass"], "+-")
            plt.xlabel(f"t [{self.time_unit}]")
            plt.ylabel(f"total mass [{mass_unit_text}]")
            plt.savefig(self.plot_filename + "_total_mass.png")
            if close_plots:
                plt.close()

            if mass_hist["d_disc_mass_dt"].size > 0:
                plt.figure(figsize=figsize, dpi=dpi)
                plt.plot(t, mass_hist["d_disc_mass_dt"], "+-")
                plt.xlabel(f"t [{self.time_unit}]")
                plt.ylabel(rf"$dM/dt$ [{mass_unit_text} / {self.time_unit}]")
                plt.yscale("symlog", linthresh=self._symlog_linthresh(mass_hist["d_disc_mass_dt"]))
                plt.savefig(self.plot_filename + "_total_mass_dot.png")
                if close_plots:
                    plt.close()

            if "dust_mass" in mass_hist:
                ndust = mass_hist["dust_mass"].shape[-1]

                grains_sizes = mass_hist["grains_sizes"]

                dust_cmap = plt.colormaps["plasma"]
                dust_vmin = 10 ** np.floor(np.log10(grains_sizes.min()))
                dust_vmax = 10 ** np.ceil(np.log10(grains_sizes.max()))
                dust_norm = mcolors.LogNorm(vmin=dust_vmin, vmax=dust_vmax)
                dust_colors = dust_cmap(dust_norm(grains_sizes))

                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax = fig.gca()
                ax.plot(t, mass_hist["disc_mass"], "+-", color="0.0", label="$M$")
                ax.plot(
                    t, mass_hist["gas_mass"], "+-", color="cornflowerblue", label=r"$M_{\rm gas}$"
                )
                ax.plot(t, mass_hist["dust_mass_all"], "+-", color="0.5", label=r"$M_{\rm dust}$")
                for i in range(ndust):
                    ax.plot(t, mass_hist["dust_mass"][:, i], "+-", color=dust_colors[i])

                ax.set_xlabel(f"t [{self.time_unit}]")
                ax.set_ylabel(f"mass [{mass_unit_text}]")
                ax.set_yscale("log")
                handles, labels = ax.get_legend_handles_labels()
                shamrock.matplotlib.add_cmap_legend_entry(
                    ax,
                    dust_cmap,
                    label=r"$M_{\rm dust}(s_{\rm grain})$",
                    extra_handles=handles,
                    extra_labels=labels,
                    loc="best",
                )

                dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
                dust_sm.set_array([])
                cbar = fig.colorbar(dust_sm, ax=ax)
                cbar.set_label("grain size [m]")

                fig.savefig(self.plot_filename + "_masses.png")
                if close_plots:
                    plt.close(fig)

                if mass_hist["d_dust_mass_dt"].size > 0:
                    fig = plt.figure(figsize=figsize, dpi=dpi)
                    ax = fig.gca()
                    ax.plot(t, mass_hist["d_disc_mass_dt"], "+-", color="0.0", label=r"$\dot{M}$")
                    ax.plot(
                        t,
                        mass_hist["d_gas_mass_dt"],
                        "+-",
                        color="cornflowerblue",
                        label=r"$\dot{M}_{\rm gas}$",
                    )
                    ax.plot(
                        t,
                        mass_hist["d_dust_mass_all_dt"],
                        "+-",
                        color="0.5",
                        label=r"$\dot{M}_{\rm dust}$",
                    )
                    for i in range(ndust):
                        ax.plot(t, mass_hist["d_dust_mass_dt"][:, i], "+-", color=dust_colors[i])

                    ax.set_xlabel(f"t [{self.time_unit}]")
                    ax.set_ylabel(rf"$dM/dt$ [{mass_unit_text} / {self.time_unit}]")
                    ax.set_yscale(
                        "symlog",
                        linthresh=self._symlog_linthresh(
                            mass_hist["d_disc_mass_dt"],
                            mass_hist["d_gas_mass_dt"],
                            mass_hist["d_dust_mass_all_dt"],
                            mass_hist["d_dust_mass_dt"],
                        ),
                    )
                    handles, labels = ax.get_legend_handles_labels()
                    shamrock.matplotlib.add_cmap_legend_entry(
                        ax,
                        dust_cmap,
                        label=r"$\dot{M}_{\rm dust}(s_{\rm grain})$",
                        extra_handles=handles,
                        extra_labels=labels,
                        loc="best",
                    )

                    dust_sm = cm.ScalarMappable(cmap=dust_cmap, norm=dust_norm)
                    dust_sm.set_array([])
                    cbar = fig.colorbar(dust_sm, ax=ax)
                    cbar.set_label("grain size [m]")

                    fig.savefig(self.plot_filename + "_masses_dot.png")
                    if close_plots:
                        plt.close(fig)
