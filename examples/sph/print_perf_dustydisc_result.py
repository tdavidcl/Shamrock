import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.facecolor": "#f2f2f2",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
    }
)

RESULTS_H100_BASELINE = [
    {"kernel": "M4", "ndust": 0, "coala": False, "rate": np.float64(23256119.36582804)},
    {"kernel": "M6", "ndust": 0, "coala": False, "rate": np.float64(11771177.467998186)},
    {"kernel": "M6", "ndust": 1, "coala": False, "rate": np.float64(11219877.199072368)},
    {"kernel": "M6", "ndust": 2, "coala": False, "rate": np.float64(10830262.946567805)},
    {"kernel": "M6", "ndust": 3, "coala": False, "rate": np.float64(10716284.43528645)},
    {"kernel": "M6", "ndust": 4, "coala": False, "rate": np.float64(10251402.850916568)},
    {"kernel": "M6", "ndust": 5, "coala": False, "rate": np.float64(10030935.676461471)},
    {"kernel": "M6", "ndust": 5, "coala": True, "rate": np.float64(9805456.180198805)},
    {"kernel": "M6", "ndust": 6, "coala": False, "rate": np.float64(9870201.403196786)},
    {"kernel": "M6", "ndust": 6, "coala": True, "rate": np.float64(9383995.885545714)},
    {"kernel": "M6", "ndust": 7, "coala": False, "rate": np.float64(9703813.52729616)},
    {"kernel": "M6", "ndust": 7, "coala": True, "rate": np.float64(9281608.27193562)},
    {"kernel": "M6", "ndust": 8, "coala": False, "rate": np.float64(9232083.680640474)},
    {"kernel": "M6", "ndust": 8, "coala": True, "rate": np.float64(8556604.837840885)},
    {"kernel": "M6", "ndust": 9, "coala": False, "rate": np.float64(9355411.31818971)},
    {"kernel": "M6", "ndust": 9, "coala": True, "rate": np.float64(8342567.464030105)},
    {"kernel": "M6", "ndust": 10, "coala": False, "rate": np.float64(9008671.654545346)},
    {"kernel": "M6", "ndust": 10, "coala": True, "rate": np.float64(7562416.727866154)},
    {"kernel": "M6", "ndust": 11, "coala": False, "rate": np.float64(8964359.561111016)},
    {"kernel": "M6", "ndust": 11, "coala": True, "rate": np.float64(6659322.016063737)},
    {"kernel": "M6", "ndust": 12, "coala": False, "rate": np.float64(8514316.388351161)},
    {"kernel": "M6", "ndust": 12, "coala": True, "rate": np.float64(6499190.12389516)},
    {"kernel": "M6", "ndust": 13, "coala": False, "rate": np.float64(8852563.931176195)},
    {"kernel": "M6", "ndust": 13, "coala": True, "rate": np.float64(5620356.801944364)},
    {"kernel": "M6", "ndust": 14, "coala": False, "rate": np.float64(8446476.647205016)},
    {"kernel": "M6", "ndust": 14, "coala": True, "rate": np.float64(5522798.30469974)},
    {"kernel": "M6", "ndust": 15, "coala": False, "rate": np.float64(8534674.592722233)},
    {"kernel": "M6", "ndust": 15, "coala": True, "rate": np.float64(4803424.264321001)},
    {"kernel": "M6", "ndust": 16, "coala": False, "rate": np.float64(8197170.576893955)},
    {"kernel": "M6", "ndust": 16, "coala": True, "rate": np.float64(4955859.848753292)},
    {"kernel": "M6", "ndust": 17, "coala": False, "rate": np.float64(8282156.089487441)},
    {"kernel": "M6", "ndust": 17, "coala": True, "rate": np.float64(4174063.305111331)},
    {"kernel": "M6", "ndust": 18, "coala": False, "rate": np.float64(7964434.143946191)},
    {"kernel": "M6", "ndust": 18, "coala": True, "rate": np.float64(3885100.3876372897)},
    {"kernel": "M6", "ndust": 19, "coala": False, "rate": np.float64(8063438.808088085)},
    {"kernel": "M6", "ndust": 19, "coala": True, "rate": np.float64(3189109.057859443)},
    {"kernel": "M6", "ndust": 20, "coala": False, "rate": np.float64(7578210.108450332)},
    {"kernel": "M6", "ndust": 20, "coala": True, "rate": np.float64(3520835.713902391)},
    {"kernel": "M6", "ndust": 21, "coala": False, "rate": np.float64(7840191.728689926)},
    {"kernel": "M6", "ndust": 21, "coala": True, "rate": np.float64(2710915.465064908)},
    {"kernel": "M6", "ndust": 22, "coala": False, "rate": np.float64(7495687.028531827)},
    {"kernel": "M6", "ndust": 22, "coala": True, "rate": np.float64(2953736.771429571)},
    {"kernel": "M6", "ndust": 23, "coala": False, "rate": np.float64(7546844.476822889)},
    {"kernel": "M6", "ndust": 23, "coala": True, "rate": np.float64(2260519.205183544)},
    {"kernel": "M6", "ndust": 24, "coala": False, "rate": np.float64(7180471.978641445)},
    {"kernel": "M6", "ndust": 24, "coala": True, "rate": np.float64(2091087.1371507098)},
    {"kernel": "M6", "ndust": 25, "coala": False, "rate": np.float64(7427410.706355693)},
    {"kernel": "M6", "ndust": 25, "coala": True, "rate": np.float64(1938057.1319513076)},
    {"kernel": "M6", "ndust": 26, "coala": False, "rate": np.float64(7044847.523285718)},
    {"kernel": "M6", "ndust": 26, "coala": True, "rate": np.float64(1850641.6275012386)},
    {"kernel": "M6", "ndust": 27, "coala": False, "rate": np.float64(7240169.745866573)},
    {"kernel": "M6", "ndust": 27, "coala": True, "rate": np.float64(1471081.5624961006)},
    {"kernel": "M6", "ndust": 28, "coala": False, "rate": np.float64(6738982.576377396)},
    {"kernel": "M6", "ndust": 28, "coala": True, "rate": np.float64(1740315.4879349351)},
    {"kernel": "M6", "ndust": 29, "coala": False, "rate": np.float64(6999043.036243548)},
    {"kernel": "M6", "ndust": 29, "coala": True, "rate": np.float64(1340813.050173749)},
    {"kernel": "M6", "ndust": 30, "coala": False, "rate": np.float64(6638762.333148276)},
    {"kernel": "M6", "ndust": 30, "coala": True, "rate": np.float64(1314997.2961275445)},
    {"kernel": "M6", "ndust": 31, "coala": False, "rate": np.float64(6896835.869019296)},
    {"kernel": "M6", "ndust": 31, "coala": True, "rate": np.float64(994982.0606032925)},
]

RESULTS_H100_OPTI_BOUCLE = [
    {"kernel": "M4", "ndust": 0, "coala": False, "rate": np.float64(23237309.489499085)},
    {"kernel": "M6", "ndust": 0, "coala": False, "rate": np.float64(11760984.706791712)},
    {"kernel": "M6", "ndust": 1, "coala": False, "rate": np.float64(11196300.414515367)},
    {"kernel": "M6", "ndust": 2, "coala": False, "rate": np.float64(10823428.08461366)},
    {"kernel": "M6", "ndust": 3, "coala": False, "rate": np.float64(10699302.632507563)},
    {"kernel": "M6", "ndust": 4, "coala": False, "rate": np.float64(10209931.037582353)},
    {"kernel": "M6", "ndust": 5, "coala": False, "rate": np.float64(10098872.149374316)},
    {"kernel": "M6", "ndust": 5, "coala": True, "rate": np.float64(9786171.59288371)},
    {"kernel": "M6", "ndust": 6, "coala": False, "rate": np.float64(9839204.105677899)},
    {"kernel": "M6", "ndust": 6, "coala": True, "rate": np.float64(9478879.517538834)},
    {"kernel": "M6", "ndust": 7, "coala": False, "rate": np.float64(9812639.165363131)},
    {"kernel": "M6", "ndust": 7, "coala": True, "rate": np.float64(9429833.847308313)},
    {"kernel": "M6", "ndust": 8, "coala": False, "rate": np.float64(9206450.25079406)},
    {"kernel": "M6", "ndust": 8, "coala": True, "rate": np.float64(8869108.002996856)},
    {"kernel": "M6", "ndust": 9, "coala": False, "rate": np.float64(9484759.009512117)},
    {"kernel": "M6", "ndust": 9, "coala": True, "rate": np.float64(8946543.772909492)},
    {"kernel": "M6", "ndust": 10, "coala": False, "rate": np.float64(9111509.684180824)},
    {"kernel": "M6", "ndust": 10, "coala": True, "rate": np.float64(8608875.58388698)},
    {"kernel": "M6", "ndust": 11, "coala": False, "rate": np.float64(9142486.19405502)},
    {"kernel": "M6", "ndust": 11, "coala": True, "rate": np.float64(8587107.019376751)},
    {"kernel": "M6", "ndust": 12, "coala": False, "rate": np.float64(8674866.146099688)},
    {"kernel": "M6", "ndust": 12, "coala": True, "rate": np.float64(8038996.257615718)},
    {"kernel": "M6", "ndust": 13, "coala": False, "rate": np.float64(8822329.408897718)},
    {"kernel": "M6", "ndust": 13, "coala": True, "rate": np.float64(8126469.223987541)},
    {"kernel": "M6", "ndust": 14, "coala": False, "rate": np.float64(8364869.273885723)},
    {"kernel": "M6", "ndust": 14, "coala": True, "rate": np.float64(7809752.339984354)},
    {"kernel": "M6", "ndust": 15, "coala": False, "rate": np.float64(8558106.179764397)},
    {"kernel": "M6", "ndust": 15, "coala": True, "rate": np.float64(7765256.109033365)},
    {"kernel": "M6", "ndust": 16, "coala": False, "rate": np.float64(8196014.338058306)},
    {"kernel": "M6", "ndust": 16, "coala": True, "rate": np.float64(6081248.978201941)},
    {"kernel": "M6", "ndust": 17, "coala": False, "rate": np.float64(8287609.569201535)},
    {"kernel": "M6", "ndust": 17, "coala": True, "rate": np.float64(7397216.224259309)},
    {"kernel": "M6", "ndust": 18, "coala": False, "rate": np.float64(7917646.989138921)},
    {"kernel": "M6", "ndust": 18, "coala": True, "rate": np.float64(7026835.881138243)},
    {"kernel": "M6", "ndust": 19, "coala": False, "rate": np.float64(8034989.312303158)},
    {"kernel": "M6", "ndust": 19, "coala": True, "rate": np.float64(6973853.562668619)},
    {"kernel": "M6", "ndust": 20, "coala": False, "rate": np.float64(7556999.414336324)},
    {"kernel": "M6", "ndust": 20, "coala": True, "rate": np.float64(6277248.334496303)},
    {"kernel": "M6", "ndust": 21, "coala": False, "rate": np.float64(7851785.741328105)},
    {"kernel": "M6", "ndust": 21, "coala": True, "rate": np.float64(6571722.144714564)},
    {"kernel": "M6", "ndust": 22, "coala": False, "rate": np.float64(7443752.094941676)},
    {"kernel": "M6", "ndust": 22, "coala": True, "rate": np.float64(6167850.491817243)},
    {"kernel": "M6", "ndust": 23, "coala": False, "rate": np.float64(7520528.376440372)},
    {"kernel": "M6", "ndust": 23, "coala": True, "rate": np.float64(6126948.345372382)},
    {"kernel": "M6", "ndust": 24, "coala": False, "rate": np.float64(7174247.586597281)},
    {"kernel": "M6", "ndust": 24, "coala": True, "rate": np.float64(4799399.80549002)},
    {"kernel": "M6", "ndust": 25, "coala": False, "rate": np.float64(7431682.541518425)},
    {"kernel": "M6", "ndust": 25, "coala": True, "rate": np.float64(5663016.125265128)},
    {"kernel": "M6", "ndust": 26, "coala": False, "rate": np.float64(7051248.093194439)},
    {"kernel": "M6", "ndust": 26, "coala": True, "rate": np.float64(5329061.381515614)},
    {"kernel": "M6", "ndust": 27, "coala": False, "rate": np.float64(7203068.772946451)},
    {"kernel": "M6", "ndust": 27, "coala": True, "rate": np.float64(5284194.422916842)},
    {"kernel": "M6", "ndust": 28, "coala": False, "rate": np.float64(6764176.03409898)},
    {"kernel": "M6", "ndust": 28, "coala": True, "rate": np.float64(4667656.530363518)},
    {"kernel": "M6", "ndust": 29, "coala": False, "rate": np.float64(6976366.276285325)},
    {"kernel": "M6", "ndust": 29, "coala": True, "rate": np.float64(4852656.888776229)},
    {"kernel": "M6", "ndust": 30, "coala": False, "rate": np.float64(6687429.320183346)},
    {"kernel": "M6", "ndust": 30, "coala": True, "rate": np.float64(4520038.057111307)},
    {"kernel": "M6", "ndust": 31, "coala": False, "rate": np.float64(6872555.178697465)},
    {"kernel": "M6", "ndust": 31, "coala": True, "rate": np.float64(4454400.792412956)},
    {"kernel": "M6", "ndust": 32, "coala": False, "rate": np.float64(6888003.308252884)},
    {"kernel": "M6", "ndust": 32, "coala": True, "rate": np.float64(2336048.803572924)},
    {"kernel": "M6", "ndust": 33, "coala": False, "rate": np.float64(6721434.6548454445)},
    {"kernel": "M6", "ndust": 33, "coala": True, "rate": np.float64(4063198.5910667097)},
    {"kernel": "M6", "ndust": 34, "coala": False, "rate": np.float64(6386881.578231751)},
    {"kernel": "M6", "ndust": 34, "coala": True, "rate": np.float64(3657677.7465728605)},
    {"kernel": "M6", "ndust": 35, "coala": False, "rate": np.float64(6567850.945360274)},
    {"kernel": "M6", "ndust": 35, "coala": True, "rate": np.float64(3672903.716657806)},
    {"kernel": "M6", "ndust": 36, "coala": False, "rate": np.float64(6111388.263587429)},
    {"kernel": "M6", "ndust": 36, "coala": True, "rate": np.float64(3306035.5225509675)},
    {"kernel": "M6", "ndust": 37, "coala": False, "rate": np.float64(6432345.248772625)},
    {"kernel": "M6", "ndust": 37, "coala": True, "rate": np.float64(3432982.0265786573)},
    {"kernel": "M6", "ndust": 38, "coala": False, "rate": np.float64(6095527.206674034)},
    {"kernel": "M6", "ndust": 38, "coala": True, "rate": np.float64(3329423.1079015657)},
    {"kernel": "M6", "ndust": 39, "coala": False, "rate": np.float64(6284348.48527522)},
    {"kernel": "M6", "ndust": 39, "coala": True, "rate": np.float64(2927424.02232121)},
    {"kernel": "M6", "ndust": 40, "coala": False, "rate": np.float64(5883316.233847309)},
    {"kernel": "M6", "ndust": 40, "coala": True, "rate": np.float64(2160344.867914631)},
    {"kernel": "M6", "ndust": 41, "coala": False, "rate": np.float64(6173531.395176899)},
    {"kernel": "M6", "ndust": 41, "coala": True, "rate": np.float64(2676558.6736552883)},
    {"kernel": "M6", "ndust": 42, "coala": False, "rate": np.float64(5785665.040902311)},
    {"kernel": "M6", "ndust": 42, "coala": True, "rate": np.float64(2441293.334223103)},
    {"kernel": "M6", "ndust": 43, "coala": False, "rate": np.float64(6020936.7575229155)},
    {"kernel": "M6", "ndust": 43, "coala": True, "rate": np.float64(2410190.88169008)},
    {"kernel": "M6", "ndust": 44, "coala": False, "rate": np.float64(5565977.049791246)},
    {"kernel": "M6", "ndust": 44, "coala": True, "rate": np.float64(2163410.383998948)},
    {"kernel": "M6", "ndust": 45, "coala": False, "rate": np.float64(5918440.6364385905)},
    {"kernel": "M6", "ndust": 45, "coala": True, "rate": np.float64(2416654.153050213)},
    {"kernel": "M6", "ndust": 46, "coala": False, "rate": np.float64(5602515.400781197)},
    {"kernel": "M6", "ndust": 46, "coala": True, "rate": np.float64(2268536.726508454)},
    {"kernel": "M6", "ndust": 47, "coala": False, "rate": np.float64(5720473.786136813)},
    {"kernel": "M6", "ndust": 47, "coala": True, "rate": np.float64(2183617.149168822)},
    {"kernel": "M6", "ndust": 48, "coala": False, "rate": np.float64(5499804.178097287)},
    {"kernel": "M6", "ndust": 48, "coala": True, "rate": np.float64(906978.3240579169)},
    {"kernel": "M6", "ndust": 49, "coala": False, "rate": np.float64(5654996.619604188)},
    {"kernel": "M6", "ndust": 49, "coala": True, "rate": np.float64(1640180.4001815373)},
    {"kernel": "M6", "ndust": 50, "coala": False, "rate": np.float64(5388720.182938816)},
    {"kernel": "M6", "ndust": 50, "coala": True, "rate": np.float64(1554200.6131661166)},
]


def _lookup_rate(results, kernel, ndust, coala):
    for entry in results:
        if entry["kernel"] == kernel and entry["ndust"] == ndust and entry["coala"] == coala:
            return float(entry["rate"])
    return None


def plot_perf_results(results, title="Dusty disc performance"):
    visible_ndust = {1, 5, 10, 20, 30, 40, 50, 60, 70}
    labels = ["gas only M4", "gas only M6"]
    rates_off = [
        _lookup_rate(results, "M4", 0, False),
        _lookup_rate(results, "M6", 0, False),
    ]
    rates_on = [None, None]

    ndust_values = sorted({entry["ndust"] for entry in results if entry["ndust"] > 0})
    for ndust in ndust_values:
        labels.append(str(ndust))
        rates_off.append(_lookup_rate(results, "M6", ndust, False))
        rates_on.append(_lookup_rate(results, "M6", ndust, True))

    fig, ax = plt.subplots(figsize=(12, 6))
    gas_count = 2
    ndust_gap = 1.0
    x = np.arange(len(labels), dtype=float)
    x[gas_count:] += ndust_gap
    width = 0.35

    off_x = []
    off_y = []
    on_x = []
    on_y = []
    for i, (rate_off, rate_on) in enumerate(zip(rates_off, rates_on)):
        if rate_on is not None:
            off_x.append(x[i] - width / 2)
            off_y.append(rate_off)
            on_x.append(x[i] + width / 2)
            on_y.append(rate_on)
        else:
            off_x.append(x[i])
            off_y.append(rate_off)

    ax.bar(off_x, off_y, width, label="coala off", color="C0")
    if on_x:
        ax.bar(on_x, on_y, width, label="coala on", color="C1")

    separator_x = (x[gas_count - 1] + x[gas_count]) / 2
    ax.axvline(separator_x, color="0.4", linewidth=1.2, linestyle="-", zorder=0)

    # ax.set_yscale("log")
    # ax.set_ylim(1e6,3e7)
    ax.set_ylabel("rate (particles/s)")
    ax.set_xlabel("dust species")
    ax.set_title(title)
    ax.set_xticks(x)
    visible_indices = {0, 1}
    for ndust in visible_ndust:
        visible_indices.add(1 + ndust)
    tick_labels = [labels[i] if i in visible_indices else "" for i in range(len(labels))]
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=12)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if i not in visible_indices:
            tick.tick1line.set_visible(False)
    ax.legend()
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    plot_perf_results(RESULTS_H100_BASELINE, title="Dusty disc performance (H100)")
    plot_perf_results(
        RESULTS_H100_OPTI_BOUCLE, title="Dusty disc performance (H100) with optimized loop"
    )
    plt.show()
