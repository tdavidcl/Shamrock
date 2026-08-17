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


CODEGEN_TIME_TOP_N = 10
REPO_PATH_MARKER = "/Shamrock/Shamrock/"
CMAKE_OBJECT_RE = re.compile(r"^(?:./)?(.+)/CMakeFiles/[^/]+\.dir/(.+)\.o$")
CODEGEN_FILES_HEADING = "**** Files that took longest to codegen (compiler backend):"
FILE_TIME_LINE_RE = re.compile(r"^\s*([0-9.]+)\s+ms:\s+(.+?)\s*$")
SECTION_END_RE = re.compile(r"\n\*{4} |\n done in ")


def normalize_profile_path(path):
    path = (path or "").replace("\\", "/").strip()
    path = path.removeprefix("./")
    idx = path.rfind(REPO_PATH_MARKER)
    if idx != -1:
        path = path[idx + len(REPO_PATH_MARKER) :]
    match = CMAKE_OBJECT_RE.match(path)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return path.removesuffix(".o")


def parse_codegen_files(text, limit=CODEGEN_TIME_TOP_N):
    if not text:
        return []
    heading_idx = text.find(CODEGEN_FILES_HEADING)
    if heading_idx == -1:
        return []
    rest = text[heading_idx + len(CODEGEN_FILES_HEADING) :]
    end_match = SECTION_END_RE.search(rest)
    body = rest if end_match is None else rest[: end_match.start()]

    by_path = {}
    for line in body.splitlines():
        match = FILE_TIME_LINE_RE.match(line)
        if match is None:
            continue
        time_ms = float(match.group(1))
        path = normalize_profile_path(match.group(2))
        if not path:
            continue
        prev = by_path.get(path)
        if prev is None or time_ms > prev:
            by_path[path] = time_ms

    ranked = sorted(by_path.items(), key=lambda item: (-item[1], item[0]))
    return [(path, time_ms / 1000.0) for path, time_ms in ranked[:limit]]


def codegen_time_top10_layout():
    return {
        "title": {"text": "Top 10 codegen time"},
        "margin": {"l": 72, "r": 24, "t": 56, "b": 180},
        "xaxis": {
            "title": {"text": "Date (UTC)"},
            "type": "date",
        },
        "yaxis": {
            "title": {"text": "Codegen time (s)"},
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


def build_codegen_time_top10(snapshots):
    # One trace per file that is in any snapshot's top 10. y is null when that
    # file is outside the top 10 of a given commit, so Plotly drops it there.
    dated_top = []
    for snapshot in snapshots:
        profile = snapshot.get("metrics", {}).get("build_profile") or {}
        ranking = parse_codegen_files(profile.get("data"))
        if not ranking:
            continue
        dated_top.append((to_iso8601(snapshot["datetime"]), ranking))

    xs = [dt for dt, _ranking in dated_top]
    time_by_file = {}
    for dt_idx, (_dt, ranking) in enumerate(dated_top):
        for path, time_s in ranking:
            series = time_by_file.setdefault(path, [None] * len(xs))
            series[dt_idx] = time_s

    files = sorted(
        time_by_file,
        key=lambda path: (-max(v for v in time_by_file[path] if v is not None), path),
    )

    traces = []
    for path in files:
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": path,
                "x": xs,
                "y": time_by_file[path],
                "connectgaps": False,
                "hovertemplate": (
                    "%{fullData.name}<br>%{x|%Y-%m-%d %H:%M UTC}"
                    "<br>Codegen: %{y:.1f} s<extra></extra>"
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
        row = {
            "datetime": to_iso8601(snapshot["datetime"]),
            **flatten_loc_totals(totals),
        }
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
        "codegen_time_top10",
        build_codegen_time_top10(snapshots),
        layout=codegen_time_top10_layout(),
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
