# Build JS-friendly plot datasets from metrics-history aggregated JSON files.

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

BUILD_PROFILE_SUMMARY_RE = re.compile(
    r"Compilation \((\d+) times\):\s+"
    r"Parsing \(frontend\):\s+([0-9.]+) s\s+"
    r"Codegen & opts \(backend\):\s+([0-9.]+) s"
)


def load_snapshots(root):
    aggregated = Path(root) / "aggregated"
    if not aggregated.is_dir():
        raise FileNotFoundError(f"missing aggregated directory: {aggregated}")

    snapshots = []
    for path in sorted(aggregated.glob("*.json")):
        with path.open() as handle:
            snapshots.append(json.load(handle))
    snapshots.sort(key=lambda item: item.get("datetime", ""))
    return snapshots


def to_iso8601(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_doxygen_warnings(snapshots):
    data = []
    for snapshot in snapshots:
        data.append(
            {
                "datetime": to_iso8601(snapshot["datetime"]),
                "doxygen_warning_count": snapshot["metrics"]["doxygen_warn"][
                    "doxygen_warning_count"
                ],
            }
        )
    return data


def parse_build_profile_summary(text):
    match = BUILD_PROFILE_SUMMARY_RE.search(text)
    if match is None:
        raise ValueError("could not parse build profile summary")
    parsing_time_s = float(match.group(2))
    codegen_time_s = float(match.group(3))
    return {
        "translation_unit_count": int(match.group(1)),
        "parsing_time_s": parsing_time_s,
        "codegen_time_s": codegen_time_s,
        "total_time_s": parsing_time_s + codegen_time_s,
    }


def build_build_time_total(snapshots):
    data = []
    for snapshot in snapshots:
        profile = snapshot.get("metrics", {}).get("build_profile")
        if profile is None:
            continue
        parsed = parse_build_profile_summary(profile["data"])
        data.append({"datetime": to_iso8601(snapshot["datetime"]), **parsed})
    return data


TEMPLATE_SETS_TOP_N = 10
TEMPLATE_SETS_SECTION_RE = re.compile(
    r"\*\*\*\* Template sets that took longest to instantiate:\n"
    r"(.*?)(?=\n\*\*\*\* |\Z)",
    re.DOTALL,
)
TEMPLATE_SET_LINE_RE = re.compile(
    r"^\s*(\d+)\s+ms:\s+(.+?)\s+\((\d+)\s+times,\s+avg\s+([0-9.]+)\s+ms\)\s*$"
)


def parse_template_sets(text):
    text = (text or "").replace("\r\n", "\n")
    match = TEMPLATE_SETS_SECTION_RE.search(text)
    if match is None:
        return []

    ranked = []
    for line in match.group(1).splitlines():
        line_match = TEMPLATE_SET_LINE_RE.match(line)
        if line_match is None:
            continue
        ranked.append(
            {
                "name": line_match.group(2).strip(),
                "time_ms": int(line_match.group(1)),
                "count": int(line_match.group(3)),
                "avg_ms": float(line_match.group(4)),
            }
        )
    return ranked


def top_template_sets(entries, limit=TEMPLATE_SETS_TOP_N):
    by_name = {}
    for item in entries:
        name = item.get("name")
        time_ms = item.get("time_ms")
        if not name or time_ms is None:
            continue
        time_ms = int(time_ms)
        prev = by_name.get(name)
        if prev is None or time_ms > prev["time_ms"]:
            by_name[name] = {
                "name": name,
                "time_ms": time_ms,
                "count": item.get("count"),
                "avg_ms": item.get("avg_ms"),
            }
    ranked = sorted(by_name.values(), key=lambda item: (-item["time_ms"], item["name"]))
    return ranked[:limit]


def template_sets_top10_layout():
    return {
        "title": {"text": "Top 10 template sets by instantiate time"},
        "margin": {"l": 72, "r": 24, "t": 56, "b": 180},
        "xaxis": {
            "title": {"text": "Date (UTC)"},
            "type": "date",
        },
        "yaxis": {
            "title": {"text": "Instantiation time (s)"},
            "rangemode": "tozero",
        },
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 11},
        },
        "hovermode": "x unified",
    }


def build_template_sets_top10(snapshots):
    # One trace per template set that is in any snapshot's top 10. y is null
    # when that set is outside the top 10 of a given commit, so Plotly drops
    # it there.
    dated_top = []
    for snapshot in snapshots:
        profile = snapshot.get("metrics", {}).get("build_profile") or {}
        text = profile.get("data")
        ranking = top_template_sets(parse_template_sets(text))
        if not ranking:
            continue
        dated_top.append((to_iso8601(snapshot["datetime"]), ranking))

    xs = [dt for dt, _ranking in dated_top]
    series_by_name = {}
    extras_by_name = {}
    for dt_idx, (_dt, ranking) in enumerate(dated_top):
        for item in ranking:
            name = item["name"]
            series = series_by_name.setdefault(name, [None] * len(xs))
            extras = extras_by_name.setdefault(name, [None] * len(xs))
            series[dt_idx] = item["time_ms"] / 1000.0
            extras[dt_idx] = [item.get("count"), item.get("avg_ms")]

    names = sorted(
        series_by_name,
        key=lambda name: (-max(v for v in series_by_name[name] if v is not None), name),
    )

    traces = []
    for name in names:
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": name,
                "x": xs,
                "y": series_by_name[name],
                "customdata": extras_by_name[name],
                "connectgaps": False,
                "hovertemplate": (
                    "%{fullData.name}<br>%{x|%Y-%m-%d %H:%M UTC}<br>"
                    "Time: %{y:.1f} s<br>%{customdata[0]} times, avg %{customdata[1]:.0f} ms"
                    "<extra></extra>"
                ),
            }
        )
    return traces


# Exclusive loc partitions from tools/count_loc.py. Nested shammodels/* counts
# are subsets of "code" and must not be added into an extension total.
LOC_PARTITION_KINDS = ("code", "examples", "doc")
LOC_TOTAL_KEYS = ("totals", "total")


def loc_extension_total(counts):
    return sum(counts.get(kind, 0) for kind in LOC_PARTITION_KINDS)


def flatten_loc_totals(totals):
    flattened = {}
    for key, value in totals.items():
        flattened[key if key == "all" else f"all_{key}"] = value
    return flattened


def build_loc(snapshots):
    data = []
    for snapshot in snapshots:
        loc = snapshot.get("metrics", {}).get("loc")
        if loc is None:
            continue
        totals = next((loc[key] for key in LOC_TOTAL_KEYS if key in loc), None)
        if totals is None:
            continue
        row = {"datetime": to_iso8601(snapshot["datetime"]), **flatten_loc_totals(totals)}
        for key, counts in loc.items():
            if key in LOC_TOTAL_KEYS:
                continue
            row[key] = loc_extension_total(counts)
        data.append(row)
    return data


def write_dataset(output_dir, name, data, layout=None):
    path = output_dir / f"{name}.json"
    payload = {"name": name, "data": data}
    if layout is not None:
        payload["layout"] = layout
    path.write_text(json.dumps(payload, indent=3) + "\n")
    print(path)


def build_datasets(root, output_dir):
    snapshots = load_snapshots(root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    write_dataset(output_dir, "doxygen_warnings", build_doxygen_warnings(snapshots))
    write_dataset(output_dir, "build_time_total", build_build_time_total(snapshots))
    write_dataset(output_dir, "loc", build_loc(snapshots))
    write_dataset(
        output_dir,
        "template_sets_top10",
        build_template_sets_top10(snapshots),
        layout=template_sets_top10_layout(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build JS-friendly plot datasets from metrics-history aggregated JSON."
    )
    parser.add_argument(
        "metrics_history_root",
        type=Path,
        help="Root of the metrics-history checkout (contains aggregated/)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write dataset JSON files into",
    )
    args = parser.parse_args()
    build_datasets(args.metrics_history_root, args.output_dir)


if __name__ == "__main__":
    main()
