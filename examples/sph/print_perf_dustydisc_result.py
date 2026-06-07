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
    {"kernel": "M4", "ndust": 0, "coala": False, "rate": np.float64(27102620.630986407)},
    {"kernel": "M6", "ndust": 0, "coala": False, "rate": np.float64(12835289.628708366)},
    {"kernel": "M6", "ndust": 1, "coala": False, "rate": np.float64(12198222.977514502)},
    {"kernel": "M6", "ndust": 2, "coala": False, "rate": np.float64(11748054.231748581)},
    {"kernel": "M6", "ndust": 3, "coala": False, "rate": np.float64(11539339.633387355)},
    {"kernel": "M6", "ndust": 4, "coala": False, "rate": np.float64(10871197.823157435)},
    {"kernel": "M6", "ndust": 5, "coala": False, "rate": np.float64(10988354.066564523)},
    {"kernel": "M6", "ndust": 5, "coala": True, "rate": np.float64(10664158.729535943)},
    {"kernel": "M6", "ndust": 6, "coala": False, "rate": np.float64(10602025.276050976)},
    {"kernel": "M6", "ndust": 6, "coala": True, "rate": np.float64(10223542.573661665)},
    {"kernel": "M6", "ndust": 7, "coala": False, "rate": np.float64(10594195.52121358)},
    {"kernel": "M6", "ndust": 7, "coala": True, "rate": np.float64(10076920.073380776)},
    {"kernel": "M6", "ndust": 8, "coala": False, "rate": np.float64(10049304.187635196)},
    {"kernel": "M6", "ndust": 8, "coala": True, "rate": np.float64(9477424.454139423)},
    {"kernel": "M6", "ndust": 9, "coala": False, "rate": np.float64(10066143.501540551)},
    {"kernel": "M6", "ndust": 9, "coala": True, "rate": np.float64(9657360.108232891)},
    {"kernel": "M6", "ndust": 10, "coala": False, "rate": np.float64(9788945.020178853)},
    {"kernel": "M6", "ndust": 10, "coala": True, "rate": np.float64(9232761.200418646)},
    {"kernel": "M6", "ndust": 11, "coala": False, "rate": np.float64(9884954.451859754)},
    {"kernel": "M6", "ndust": 11, "coala": True, "rate": np.float64(9100447.029519854)},
    {"kernel": "M6", "ndust": 12, "coala": False, "rate": np.float64(9254894.81368272)},
    {"kernel": "M6", "ndust": 12, "coala": True, "rate": np.float64(8555386.888574604)},
    {"kernel": "M6", "ndust": 13, "coala": False, "rate": np.float64(9477551.52642431)},
    {"kernel": "M6", "ndust": 13, "coala": True, "rate": np.float64(8699695.096218158)},
    {"kernel": "M6", "ndust": 14, "coala": False, "rate": np.float64(9049512.703830818)},
    {"kernel": "M6", "ndust": 14, "coala": True, "rate": np.float64(8251609.026215291)},
    {"kernel": "M6", "ndust": 15, "coala": False, "rate": np.float64(9212897.56166948)},
    {"kernel": "M6", "ndust": 15, "coala": True, "rate": np.float64(8379834.389098332)},
    {"kernel": "M6", "ndust": 16, "coala": False, "rate": np.float64(8752456.315597778)},
    {"kernel": "M6", "ndust": 16, "coala": True, "rate": np.float64(6463127.920491189)},
    {"kernel": "M6", "ndust": 17, "coala": False, "rate": np.float64(8905306.47443031)},
    {"kernel": "M6", "ndust": 17, "coala": True, "rate": np.float64(7943600.216045273)},
    {"kernel": "M6", "ndust": 18, "coala": False, "rate": np.float64(8397008.125851939)},
    {"kernel": "M6", "ndust": 18, "coala": True, "rate": np.float64(7442740.4116608)},
    {"kernel": "M6", "ndust": 19, "coala": False, "rate": np.float64(8529903.364110636)},
    {"kernel": "M6", "ndust": 19, "coala": True, "rate": np.float64(7490786.755520162)},
    {"kernel": "M6", "ndust": 20, "coala": False, "rate": np.float64(8090642.582126832)},
    {"kernel": "M6", "ndust": 20, "coala": True, "rate": np.float64(6664945.959795037)},
    {"kernel": "M6", "ndust": 21, "coala": False, "rate": np.float64(8396125.85960586)},
    {"kernel": "M6", "ndust": 21, "coala": True, "rate": np.float64(7114588.018497052)},
    {"kernel": "M6", "ndust": 22, "coala": False, "rate": np.float64(8030205.231528544)},
    {"kernel": "M6", "ndust": 22, "coala": True, "rate": np.float64(6607000.400893624)},
    {"kernel": "M6", "ndust": 23, "coala": False, "rate": np.float64(8154418.74031945)},
    {"kernel": "M6", "ndust": 23, "coala": True, "rate": np.float64(6546831.7388836695)},
    {"kernel": "M6", "ndust": 24, "coala": False, "rate": np.float64(7642621.364712632)},
    {"kernel": "M6", "ndust": 24, "coala": True, "rate": np.float64(5057299.020901347)},
    {"kernel": "M6", "ndust": 25, "coala": False, "rate": np.float64(7949706.645172185)},
    {"kernel": "M6", "ndust": 25, "coala": True, "rate": np.float64(6059580.004777505)},
    {"kernel": "M6", "ndust": 26, "coala": False, "rate": np.float64(7552356.444833241)},
    {"kernel": "M6", "ndust": 26, "coala": True, "rate": np.float64(5715202.328232605)},
    {"kernel": "M6", "ndust": 27, "coala": False, "rate": np.float64(7680374.0216804845)},
    {"kernel": "M6", "ndust": 27, "coala": True, "rate": np.float64(5631928.441311956)},
    {"kernel": "M6", "ndust": 28, "coala": False, "rate": np.float64(7211663.261451488)},
    {"kernel": "M6", "ndust": 28, "coala": True, "rate": np.float64(4881443.071687573)},
    {"kernel": "M6", "ndust": 29, "coala": False, "rate": np.float64(7559436.70609435)},
    {"kernel": "M6", "ndust": 29, "coala": True, "rate": np.float64(5109921.7495709825)},
    {"kernel": "M6", "ndust": 30, "coala": False, "rate": np.float64(7180007.51205414)},
    {"kernel": "M6", "ndust": 30, "coala": True, "rate": np.float64(4823488.313601385)},
    {"kernel": "M6", "ndust": 31, "coala": False, "rate": np.float64(7380437.070752764)},
    {"kernel": "M6", "ndust": 31, "coala": True, "rate": np.float64(4672124.180216651)},
    {"kernel": "M6", "ndust": 32, "coala": False, "rate": np.float64(7324479.922428249)},
    {"kernel": "M6", "ndust": 32, "coala": True, "rate": np.float64(2400721.428504786)},
    {"kernel": "M6", "ndust": 33, "coala": False, "rate": np.float64(7214719.339376894)},
    {"kernel": "M6", "ndust": 33, "coala": True, "rate": np.float64(4267351.9009655155)},
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
    ax.set_ylabel("rate")
    ax.set_xlabel("configuration")
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
