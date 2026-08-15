# Build JS-friendly plot datasets from metrics-history aggregated JSON files.

import json
import re
import sys
from pathlib import Path

TOP_N_FILES = 10

TIME_PARSE_RE = re.compile(r"Parsing \(frontend\):\s+([\d.]+)\s+s")
TIME_CODEGEN_RE = re.compile(r"Codegen & opts \(backend\):\s+([\d.]+)\s+s")
FILE_TIME_RE = re.compile(r"^(\d+)\s+ms:\s+(.+)$")
FILE_RSS_RE = re.compile(r"^([\d.]+)\s+MB:\s+(.+)$")
CMAKE_DIR_RE = re.compile(r"CMakeFiles/[^/]+\.dir/")
WORKSPACE_SPLIT = "/Shamrock/Shamrock/"

PARSE_FILES_HEADER = "Files that took longest to parse (compiler frontend)"
CODEGEN_FILES_HEADER = "Files that took longest to codegen (compiler backend)"
RSS_FILES_HEADER = "Files with highest peak RSS (compiler process)"

DATASET_FILES = (
    "doxygen_warnings.json",
    "compile_times.json",
    "top_parse_files.json",
    "top_codegen_files.json",
    "top_rss_files.json",
)


def to_iso8601(datetime_str):
    return datetime_str.replace(" ", "T")


def object_file_label(path):
    label = path.strip().removeprefix("./")
    if WORKSPACE_SPLIT in label:
        label = label.split(WORKSPACE_SPLIT, 1)[1]
    label = CMAKE_DIR_RE.sub("", label)
    return label.removesuffix(".o")


def iter_section_lines(text, header):
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("****"):
            if in_section:
                return
            if header in stripped:
                in_section = True
            continue
        if in_section and stripped:
            yield stripped


def parse_time_summary(text):
    parse_match = TIME_PARSE_RE.search(text)
    codegen_match = TIME_CODEGEN_RE.search(text)
    if parse_match is None or codegen_match is None:
        return None
    parsing_s = float(parse_match.group(1))
    codegen_s = float(codegen_match.group(1))
    return {
        "parsing_s": parsing_s,
        "codegen_s": codegen_s,
        "total_s": parsing_s + codegen_s,
    }


def parse_file_times(text, header, limit=TOP_N_FILES):
    rows = []
    for line in iter_section_lines(text, header):
        match = FILE_TIME_RE.match(line)
        if match is None:
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "file": match.group(2).strip(),
                "label": object_file_label(match.group(2)),
                "time_ms": int(match.group(1)),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def parse_file_rss(text, header, limit=TOP_N_FILES):
    rows = []
    for line in iter_section_lines(text, header):
        match = FILE_RSS_RE.match(line)
        if match is None:
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "file": match.group(2).strip(),
                "label": object_file_label(match.group(2)),
                "rss_mb": float(match.group(1)),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def parse_rss_usage(usage, limit=TOP_N_FILES):
    records = []
    for item in usage:
        if not isinstance(item, dict) or "rss_mb" not in item or "path" not in item:
            continue
        records.append(item)
    records.sort(key=lambda item: -float(item["rss_mb"]))
    rows = []
    for item in records[:limit]:
        path = item["path"]
        rows.append(
            {
                "rank": len(rows) + 1,
                "file": path,
                "label": object_file_label(path),
                "rss_mb": float(item["rss_mb"]),
            }
        )
    return rows


def peak_rss_mb(profile):
    usage = profile.get("compile_memory_usage")
    if not isinstance(usage, list):
        return None
    values = [
        float(item["rss_mb"]) for item in usage if isinstance(item, dict) and "rss_mb" in item
    ]
    if not values:
        return None
    return max(values)


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


def snapshot_metrics(snapshot):
    metrics = snapshot.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return {}


def build_doxygen_warnings(snapshots):
    data = []
    for snapshot in snapshots:
        warn = snapshot_metrics(snapshot).get("doxygen_warn")
        if not isinstance(warn, dict) or "doxygen_warning_count" not in warn:
            continue
        data.append(
            {
                "datetime": to_iso8601(snapshot["datetime"]),
                "sha": snapshot.get("sha"),
                "doxygen_warning_count": warn["doxygen_warning_count"],
            }
        )
    return data


def build_compile_times(snapshots):
    data = []
    for snapshot in snapshots:
        profile = snapshot_metrics(snapshot).get("build_profile")
        if not isinstance(profile, dict):
            continue
        summary = parse_time_summary(profile.get("data", ""))
        if summary is None:
            continue
        row = {
            "datetime": to_iso8601(snapshot["datetime"]),
            "sha": snapshot.get("sha"),
        }
        row.update(summary)
        rss = peak_rss_mb(profile)
        if rss is not None:
            row["peak_rss_mb"] = rss
        data.append(row)
    return data


def build_top_files(snapshots, header):
    data = []
    for snapshot in snapshots:
        profile = snapshot_metrics(snapshot).get("build_profile")
        if not isinstance(profile, dict):
            continue
        datetime_iso = to_iso8601(snapshot["datetime"])
        sha = snapshot.get("sha")
        for row in parse_file_times(profile.get("data", ""), header):
            item = {
                "datetime": datetime_iso,
                "sha": sha,
            }
            item.update(row)
            data.append(item)
    return data


def build_top_rss_files(snapshots):
    data = []
    for snapshot in snapshots:
        profile = snapshot_metrics(snapshot).get("build_profile")
        if not isinstance(profile, dict):
            continue
        usage = profile.get("compile_memory_usage")
        if isinstance(usage, list) and usage:
            rows = parse_rss_usage(usage)
        else:
            rows = parse_file_rss(profile.get("data", ""), RSS_FILES_HEADER)
        datetime_iso = to_iso8601(snapshot["datetime"])
        sha = snapshot.get("sha")
        for row in rows:
            item = {
                "datetime": datetime_iso,
                "sha": sha,
            }
            item.update(row)
            data.append(item)
    return data


def write_dataset(output_dir, name, data):
    path = output_dir / f"{name}.json"
    payload = {"name": name, "data": data}
    path.write_text(json.dumps(payload, indent=3) + "\n")
    return path


def prune_output_dir(output_dir, keep_paths):
    keep = {path.resolve() for path in keep_paths}
    if not output_dir.is_dir():
        return
    for path in output_dir.iterdir():
        if path.is_file() and path.resolve() not in keep:
            path.unlink()


def build_datasets(root, output_dir):
    snapshots = load_snapshots(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced = [
        write_dataset(output_dir, "doxygen_warnings", build_doxygen_warnings(snapshots)),
        write_dataset(output_dir, "compile_times", build_compile_times(snapshots)),
        write_dataset(
            output_dir,
            "top_parse_files",
            build_top_files(snapshots, PARSE_FILES_HEADER),
        ),
        write_dataset(
            output_dir,
            "top_codegen_files",
            build_top_files(snapshots, CODEGEN_FILES_HEADER),
        ),
        write_dataset(output_dir, "top_rss_files", build_top_rss_files(snapshots)),
    ]
    prune_output_dir(output_dir, produced)
    return produced


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: make_metrics_datasets.py METRICS_HISTORY_ROOT OUTPUT_DIR\n")
        sys.exit(2)

    produced = build_datasets(sys.argv[1], sys.argv[2])
    for path in produced:
        sys.stdout.write(f"{path}\n")


if __name__ == "__main__":
    main()
