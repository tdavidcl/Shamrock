import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

results = [
    {
        "device_properties": {
            "vendor": "Unknown",
            "backend": "CUDA",
            "type": "GPU",
            "name": "NVIDIA GeForce RTX 3070",
            "platform": "CUDA (platform 0)",
            "global_mem_size": 8187674624,
            "global_mem_cache_line_size": 128,
            "global_mem_cache_size": 4194304,
            "local_mem_size": 49152,
            "max_compute_units": 46,
            "max_mem_alloc_size_dev": 8187674624,
            "max_mem_alloc_size_host": 66755719168,
            "mem_base_addr_align": 8,
        },
        "microbench_results": {
            "vector_allgather_u64_128": 4.06469812624154e-06,
            "vector_allgather_u64_8": 4.06125435568661e-06,
            "vector_allgather_u64_1": 4.098756701369335e-06,
            "fma_chains_f64_4": 367062525512.4953,
            "fma_chains_f32_4": 21667275636977.293,
            "fma_chains_f64_3": 365705334110.48315,
            "fma_chains_f32_3": 20855735136596.113,
            "p2p_bandwidth": 24635728496.92073,
            "saxpy_f32": 405789122337.4236,
            "saxpy_f64_2": 389104939124.6199,
            "vector_allgather_u64_1024": 4.142013585722499e-06,
            "saxpy_f64": 400829410183.6643,
            "saxpy_f32_3": 393060157789.0016,
            "saxpy_f64_3": 395892916556.8748,
            "saxpy_f32_4": 404586907122.59247,
            "fma_chains_f32": 20646258572772.945,
            "saxpy_f64_4": 403977421989.2809,
            "vector_allgather_u64_150": 4.072438915129118e-06,
            "saxpy_f32_2": 401750450737.1935,
            "fma_chains_f64": 367518191833.0948,
            "vector_allgather_u64_64": 4.0703468739887745e-06,
            "fma_chains_f32_2": 21114969964124.04,
            "fma_chains_f64_2": 360396157668.36255,
        },
        "world_size": 1,
        "rate": 1930700.5696579285,
        "cnt": 4254912,
        "step_time": 2.2038176540000003,
        "system_metric_duration": 2.2038306930000005,
    },
    {
        "device_properties": {
            "vendor": "Unknown",
            "backend": "OpenMP",
            "type": "CPU",
            "name": "Intel(R) Core(TM) Ultra 9 285K",
            "platform": "OpenMP (platform 0)",
            "global_mem_size": 66755719168,
            "global_mem_cache_line_size": 64,
            "global_mem_cache_size": 1,
            "local_mem_size": 18446744073709551615,
            "max_compute_units": 24,
            "max_mem_alloc_size_dev": 18446744073709551615,
            "max_mem_alloc_size_host": 66755719168,
            "mem_base_addr_align": 8,
        },
        "microbench_results": {
            "vector_allgather_u64_128": 3.495953085128192e-06,
            "vector_allgather_u64_8": 3.4738327369989553e-06,
            "vector_allgather_u64_1": 3.4773515195766557e-06,
            "fma_chains_f64_4": 127745011520.24513,
            "fma_chains_f32_4": 154561171416.3423,
            "fma_chains_f64_3": 95522814116.85965,
            "fma_chains_f32_3": 106967373888.96805,
            "p2p_bandwidth": 16351389339.70154,
            "saxpy_f32": 117552814121.75764,
            "saxpy_f64_2": 96968505052.00351,
            "vector_allgather_u64_1024": 3.532044679116098e-06,
            "saxpy_f64": 105297160804.0858,
            "saxpy_f32_3": 81551795765.8618,
            "saxpy_f64_3": 106605485777.22237,
            "saxpy_f32_4": 92552886551.49269,
            "fma_chains_f32": 41444843536.23771,
            "saxpy_f64_4": 99491091954.59099,
            "vector_allgather_u64_150": 3.4815538418659823e-06,
            "saxpy_f32_2": 96483856544.14882,
            "fma_chains_f64": 39580948524.24351,
            "vector_allgather_u64_64": 3.472840493140639e-06,
            "fma_chains_f32_2": 96304427002.1922,
            "fma_chains_f64_2": 64945203386.71476,
        },
        "world_size": 1,
        "rate": 1338883.387241708,
        "cnt": 33848064,
        "step_time": 25.280815583000003,
        "system_metric_duration": 25.28083724000001,
    },
]


results_per_model = {}

for result in results:
    name = result["device_properties"]["name"]
    if name not in results_per_model:
        results_per_model[name] = result
    else:
        if result["rate"] > results_per_model[name]["rate"]:
            results_per_model[name] = result

for name, result in results_per_model.items():
    print(f"{name}:")
    print(
        f"  - {result['world_size']} ranks, {result['rate']} rate, {result['cnt']} cnt, {result['step_time']} step time"
    )


def _micro_bw_and_fma(result):
    """saxpy f64 -> GB/s; fma_chains f32/f64 -> Gflops (MicroBenchmark raw flop/s, /1e9)."""
    m = result.get("microbench_results") or {}
    bw_bs = m.get("saxpy_f64")
    f64 = m.get("fma_chains_f64")
    f32 = m.get("fma_chains_f32")
    bw_gbps = (bw_bs / 1e9) if bw_bs is not None else float("nan")
    gflops_f64 = (f64 / 1e9) if f64 is not None else float("nan")
    gflops_f32 = (f32 / 1e9) if f32 is not None else float("nan")
    return bw_gbps, gflops_f64, gflops_f32


# Stable sort by rate descending for a readable chart
items = sorted(results_per_model.items(), key=lambda kv: kv[1]["rate"], reverse=True)
names = [kv[0] for kv in items]
rates = [kv[1]["rate"] for kv in items]
bw_gbps = []
gflops_f64 = []
gflops_f32 = []
for _, r in items:
    bw, gf64, gf32 = _micro_bw_and_fma(r)
    bw_gbps.append(bw)
    gflops_f64.append(gf64)
    gflops_f32.append(gf32)

h_in = max(3.0, 0.45 * len(names) + 5)
y = np.arange(len(names))

fig, (ax_rate, ax_micro) = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(15, h_in),
    gridspec_kw={"width_ratios": [75, 25], "wspace": 0.025},
)

# Wrap long device names so they stay inside the figure margin
_name_labels = ["\n".join(textwrap.wrap(n, 34)) for n in names]

bars = ax_rate.barh(y, rates, color="steelblue", edgecolor="white", linewidth=0.5)
ax_rate.set_yticks(y)
ax_rate.set_yticklabels(_name_labels)
ax_rate.set_xlabel("rate (solver objects / s)")
ax_rate.set_title("SPH homogeneous - rate by device")
ax_rate.bar_label(bars, fmt="%.3g", padding=3)
ax_rate.grid(axis="x", linestyle=":", alpha=0.6)
ax_rate.invert_yaxis()

# Extra room for bar-end labels; drop rightmost x tick (avoids clash with right panel)
_xmin, _xmax = ax_rate.get_xlim()
ax_rate.set_xlim(_xmin, _xmax + 0.1 * (_xmax - _xmin))
ax_rate.xaxis.set_major_locator(MaxNLocator(prune="upper"))

# Three equal-height rows per device, evenly spaced around the tick (name at y)
_bar_h = 0.22
_spacing = 0.26  # distance between bar centers; middle bar (f32) on the tick
_y_saxpy = y - _spacing
_y_f32 = y
_y_f64 = y + _spacing

ax_micro.barh(
    _y_saxpy,
    bw_gbps,
    height=_bar_h,
    color="coral",
    label="saxpy f64 (GB/s)",
    edgecolor="white",
    linewidth=0.5,
)
ax_micro.set_xlabel("Memory bandwidth saxpy f64 (GB/s)")
ax_micro.grid(axis="x", linestyle=":", alpha=0.6)
ax_micro.tick_params(axis="y", labelleft=False)

# f32 / f64 FMA can differ a lot in scale -> log-scaled Gflops axis (same y layout as saxpy)
ax_micro_top = ax_micro.twiny()
ax_micro_top.barh(
    _y_f32,
    gflops_f32,
    height=_bar_h,
    color="mediumpurple",
    label="fma_chains f32 (Gflops)",
    edgecolor="white",
    linewidth=0.5,
)
ax_micro_top.barh(
    _y_f64,
    gflops_f64,
    height=_bar_h,
    color="seagreen",
    label="fma_chains f64 (Gflops)",
    edgecolor="white",
    linewidth=0.5,
)
ax_micro_top.set_xlabel("Peak FMA f32 / f64 (Gflops, log scale)")
ax_micro_top.set_xscale("log")
ax_micro.set_xscale("log")

h0, l0 = ax_micro.get_legend_handles_labels()
h1, l1 = ax_micro_top.get_legend_handles_labels()
ax_micro.legend(h0 + h1, l0 + l1, loc="lower right", fontsize=8)

# Flush panels: constrained_layout always leaves a gap; manual wspace=0 truly abuts axes
ax_rate.spines["right"].set_visible(True)
ax_micro.spines["left"].set_visible(False)
fig.subplots_adjust(left=0.22, right=0.99, top=0.90, bottom=0.12, wspace=0)

plt.show()
