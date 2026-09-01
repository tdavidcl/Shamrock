"""
reduction performance benchmarks
================================

This example benchmarks the reduction performance for the different algorithms available in Shamrock
"""

# sphinx_gallery_multi_image = "single"

import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Recover microbenchmark results
microbench_results = shamrock.sys.get_microbench_results(allow_run=True)
if len(microbench_results) == 0:
    print("no microbench results, please run with --benchmark-mpi")
    raise ValueError("no microbench results")


# %%
# Main benchmark functions
def benchmark_f32(N, nb_repeat=10):
    times = []
    for i in range(nb_repeat):
        buf = shamrock.backends.DeviceBuffer_f32()
        buf.resize(N)
        buf.fill(0)
        times.append(shamrock.algs.benchmark_reduction_sum(buf, N))
    return min(times), max(times), sum(times) / nb_repeat


def benchmark_f64(N, nb_repeat=10):
    times = []
    for i in range(nb_repeat):
        buf = shamrock.backends.DeviceBuffer_f64()
        buf.resize(N)
        buf.fill(0)
        times.append(shamrock.algs.benchmark_reduction_sum(buf, N))
    return min(times), max(times), sum(times) / nb_repeat


# %%
# Run the performance test for all parameters
def run_performance_sweep():
    # Define parameter ranges
    # logspace as array
    particle_counts = np.logspace(2, 7, 20).astype(int).tolist()

    # Initialize results matrix
    results_f32 = []
    results_f64 = []

    print(f"Particle counts: {particle_counts}")

    total_runs = len(particle_counts)
    current_run = 0

    for i, N in enumerate(particle_counts):
        current_run += 1

        print(
            f"[{current_run:2d}/{total_runs}] Running N={N:5d}...",
            end=" ",
        )

        start_time = time.time()
        min_time, max_time, mean_time = benchmark_f32(N)
        results_f32.append(min_time)
        min_time, max_time, mean_time = benchmark_f64(N)
        results_f64.append(min_time)
        elapsed = time.time() - start_time

        print(f"mean={mean_time:.3f}s (took {elapsed:.1f}s)")

    return particle_counts, results_f32, results_f64


# %%
# List current implementation
if not shamrock.algs.is_impl_set_reduction():
    shamrock.algs.autoselect_impl_reduction()
current_impl = shamrock.algs.get_current_impl_reduction()

print(current_impl)

# %%
# List all implementations available
all_default_impls = shamrock.algs.get_default_impl_list_reduction()

print(all_default_impls)

# %%
# Run the performance benchmarks for all implementations

dic_bench = {}
for impl in all_default_impls:
    shamrock.algs.set_impl_reduction(impl)

    impl_name = json.loads(impl)["implementation"]

    print(f"Running reduction performance benchmarks for {impl}...")

    # Run the performance sweep
    particle_counts, results_f32, results_f64 = run_performance_sweep()

    dic_bench[impl_name] = {
        "particle_counts": particle_counts,
        "results_f32": results_f32,
        "results_f64": results_f64,
    }


# %%
# Plot results (time)

print_ref = True
for label, item in dic_bench.items():
    if print_ref:
        Nobj = np.array(item["particle_counts"])
        Time100M = Nobj / 1e8
        plt.plot(
            item["particle_counts"],
            Time100M,
            color="grey",
            linestyle="-",
            alpha=0.7,
            label="100M obj/sec",
        )
        print_ref = False

    (line,) = plt.plot(item["particle_counts"], item["results_f64"], "--.", label=label + " (f64)")
    plt.plot(
        item["particle_counts"],
        item["results_f32"],
        ":",
        color=line.get_color(),
        label=label + " (f32)",
    )


plt.xlabel("Number of elements")
plt.ylabel("Time (s)")
plt.title("reduction performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

plt.legend(fontsize=10)
plt.show()


# %%
# Helper to place non-overlapping value callouts outside the right edge of
# the axes, each linked back to its line's last data point with a leader line
def add_end_labels(ax, entries, x_frac=1.1, min_gap_px=25, fontsize=9):
    if not entries:
        return

    # Sort by data y-value and convert to display (pixel) coordinates so
    # spacing can be reasoned about independently of the (log) data scale
    order = sorted(range(len(entries)), key=lambda i: entries[i][1])
    disp_y = [ax.transData.transform((0, entries[i][1]))[1] for i in order]

    # Group overlapping labels into clusters and spread each cluster
    # symmetrically around the mean of its members' true positions, rather
    # than cascading everything upward when things get crammed
    clusters = []  # each: {"center": mean y, "count": n}
    for y in disp_y:
        clusters.append({"center": y, "count": 1})
        while len(clusters) >= 2:
            a, b = clusters[-2], clusters[-1]
            span_a = (a["count"] - 1) * min_gap_px
            span_b = (b["count"] - 1) * min_gap_px
            top_a = a["center"] + span_a / 2
            bot_b = b["center"] - span_b / 2
            if bot_b - top_a < min_gap_px:
                count = a["count"] + b["count"]
                center = (a["center"] * a["count"] + b["center"] * b["count"]) / count
                clusters[-2:] = [{"center": center, "count": count}]
            else:
                break

    disp_y = []
    for c in clusters:
        span = (c["count"] - 1) * min_gap_px
        start = c["center"] - span / 2
        disp_y.extend(start + k * min_gap_px for k in range(c["count"]))

    inv = ax.transData.inverted()
    for idx, y_disp in zip(order, disp_y):
        x_data, y_data, text, color = entries[idx]
        label_y_data = inv.transform((0, y_disp))[1]
        # mirror the bend when the label lands below its point, otherwise the
        # corner ends up on the wrong side and the leader line doubles back
        angle_b = 60 if label_y_data >= y_data else -60
        ax.annotate(
            text,
            xy=(x_data, y_data),
            xycoords="data",
            xytext=(x_frac, label_y_data),
            textcoords=("axes fraction", "data"),
            color=color,
            fontsize=fontsize,
            va="center",
            ha="left",
            annotation_clip=False,
            bbox=dict(boxstyle="round", fc="0.8"),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.8,
                shrinkA=0,
                shrinkB=2,
                connectionstyle=f"angle,angleA=0,angleB={angle_b},rad=10",
            ),
        )


# %%
# Plot results (bandwidth)

peak_bw_f32 = microbench_results["saxpy_f32"]
peak_bw_f64 = microbench_results["saxpy_f64"]


color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plot_data = {}
for i, (label, item) in enumerate(dic_bench.items()):
    color = color_cycle[i % len(color_cycle)]
    Nobj = np.array(item["particle_counts"])

    Bytes_f64 = 8 * Nobj  # 1 read f64 (sizeof = 8)
    BW_f64 = Bytes_f64 / np.array(item["results_f64"])
    plot_data[label + " (f64)"] = {
        "x": item["particle_counts"],
        "y": BW_f64,
        "color": color,
        "label": label + " (f64)",
        "linestyle": "-",
        "marker" : "x"
    }

    Bytes_f32 = 4 * Nobj  # 1 read f32 (sizeof = 4)
    BW_f32 = Bytes_f32 / np.array(item["results_f32"])
    plot_data[label + " (f32)"] = {
        "x": item["particle_counts"],
        "y": BW_f32,
        "color": color,
        "label": label + " (f32)",
        "linestyle": ":",
        "marker" : "x"
    }

plt.figure(dpi=250,figsize=(10,6))

plt.axhline(
    y=peak_bw_f64,
    color="black",
    linestyle=":",
    label="microbenchmark peak BW f64",
)
plt.axhline(
    y=peak_bw_f32,
    color="black",
    linestyle="--",
    label="microbenchmark peak BW f32",
)

end_labels = []
for d in plot_data.values():
    plt.plot(d["x"], d["y"], d["linestyle"], color=d["color"], label=d["label"], marker=d["marker"])
    end_labels.append((d["x"][-1], d["y"][-1], f"{d['y'][-1] / 1e9:.2f} GB.s^-1", d["color"]))



plt.xlabel("Number of elements")
plt.ylabel("Bandwidth (B.s^-1)")
plt.title("reduction performance benchmarks")

plt.xscale("log")
plt.yscale("log")

plt.grid(True)

add_end_labels(plt.gca(), end_labels, min_gap_px=75)

plt.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
plt.gcf().subplots_adjust(right=0.72, bottom=0.32)

#plt.tight_layout()
plt.show()
