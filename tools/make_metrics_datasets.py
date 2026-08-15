# Build JS-friendly plot datasets from metrics-history aggregated JSON files.

import argparse
import json
from pathlib import Path

DATASET_FILES = ("doxygen_warnings.json",)


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
                "datetime": snapshot["datetime"],
                "sha": snapshot.get("sha"),
                "doxygen_warning_count": warn["doxygen_warning_count"],
            }
        )
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
    ]
    prune_output_dir(output_dir, produced)
    return produced


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

    produced = build_datasets(args.metrics_history_root, args.output_dir)
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
