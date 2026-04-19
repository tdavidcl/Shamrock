import matplotlib.pyplot as plt

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
        "world_size": 1,
        "rate": 1928113.4065535532,
        "cnt": 4254912,
        "step_time": 2.20677476,
        "system_metric_duration": 2.2067901639999974,
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
        "world_size": 1,
        "rate": 1233112.1594398895,
        "cnt": 33848064,
        "step_time": 27.449298704,
        "system_metric_duration": 27.449321989000026,
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


# Stable sort by rate descending for a readable chart
items = sorted(results_per_model.items(), key=lambda kv: kv[1]["rate"], reverse=True)
names = [kv[0] for kv in items]
rates = [kv[1]["rate"] for kv in items]

fig, ax = plt.subplots(figsize=(10, max(3.0, 0.45 * len(names))))
bars = ax.barh(names, rates, color="steelblue", edgecolor="white", linewidth=0.5)
ax.set_xlabel("rate (solver objects / s)")
ax.set_title("SPH homogeneous benchmark - rate by device")
ax.bar_label(bars, fmt="%.3g", padding=3)
ax.grid(axis="x", linestyle=":", alpha=0.6)
fig.tight_layout()
plt.show()
