cases = {
    "racc=0.8": {
        "path": "_to_trash/circular_disc_sink_False_0.8_10000000",
        "is_torque_free": False,
        "racc": 0.8,
    },
    "racc=0.25": {
        "path": "_to_trash/circular_disc_sink_False_0.25_10000000",
        "is_torque_free": False,
        "racc": 0.25,
    },
    "racc=0.8 (torque-free)": {
        "path": "_to_trash/circular_disc_sink_True_0.8_10000000",
        "is_torque_free": True,
        "racc": 0.8,
    },
    "racc=0.25 (torque-free)": {
        "path": "_to_trash/circular_disc_sink_True_0.25_10000000",
        "is_torque_free": True,
        "racc": 0.25,
    },
}

import os

import matplotlib.pyplot as plt
from shamrock.utils.analysis import (
    AnalysisHelper,
)


def render_profiles(cases):
    helpers = {}
    for key, case in cases.items():
        helpers[case] = {}
        helpers[case]["helper"] = AnalysisHelper(
            analysis_folder=os.path.join(cases[case]["path"], "analysis", "plots"),
            analysis_prefix="density_profile",
        )

    for k in helpers:
        helpers[k]["list_analysis_id"] = helpers[k]["helper"].get_list_analysis_id()

    profile_list_analysis_id1 = helpers["racc=0.8"]["list_analysis_id"]

    for iplot in profile_list_analysis_id1:
        has_iplot_data = True
        for k in helpers:
            has_iplot_data = has_iplot_data and iplot in helpers[k]["list_analysis_id"]
        if not has_iplot_data:
            continue

        data = {}
        for k in helpers:
            data[k] = helpers[k]["helper"].load_analysis(iplot).item()

        time = data["racc=0.8"]["time"]
        bin_edges_x1d = data["racc=0.8"]["bin_edges_x1d"]
        bin_center = (bin_edges_x1d[:-1] + bin_edges_x1d[1:]) / 2

        plt.figure(dpi=150)
        for k, v in data.items():
            plt.plot(bin_center, v["histo_convolve"], label=k)

        text = f"t = {time:0.3f}"
        from matplotlib.offsetbox import AnchoredText

        anchored_text = AnchoredText(text, loc=1)
        plt.gca().add_artist(anchored_text)

        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1e-6, 1e-3)
        plt.xlim(0.1, 10)
        plt.legend(loc="upper left")
        plt.savefig(f"_to_trash/compare_torque_free_sink_{iplot:07}.png")
        plt.close()


render_profiles(cases)
