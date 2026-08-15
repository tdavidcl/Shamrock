#!/usr/bin/env python3
"""Aggregate memlog RSS records into metric JSON and a ClangBuildAnalyzer section."""

import argparse
import json
import os
from pathlib import Path


def prefix_candidates():
    prefixes = []
    for key in ("SHAMROCK_DIR", "GITHUB_WORKSPACE"):
        value = os.environ.get(key)
        if value:
            prefixes.append(Path(value).resolve())
    prefixes.append(Path.cwd().resolve())
    return prefixes


def strip_prefix(path_str, prefixes):
    if not path_str:
        return path_str
    try:
        path = Path(path_str)
        if not path.is_absolute():
            return path.as_posix()
        resolved = path.resolve()
    except OSError:
        return path_str.replace("\\", "/")

    for prefix in prefixes:
        try:
            return resolved.relative_to(prefix).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def load_records(memlog_dir):
    records = []
    directory = Path(memlog_dir)
    if not directory.is_dir():
        return records
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        records.append(data)
    return records


def aggregate(records, prefixes):
    by_key = {}
    for record in records:
        try:
            peak_rss_kb = float(record.get("peak_rss_kb") or 0)
        except (TypeError, ValueError):
            peak_rss_kb = 0.0
        file_path = strip_prefix(str(record.get("file") or ""), prefixes)
        object_path = strip_prefix(str(record.get("object") or ""), prefixes)
        key = file_path or object_path
        if not key:
            continue
        peak_rss_mb = round(peak_rss_kb / 1024.0, 1)
        previous = by_key.get(key)
        if previous is None or peak_rss_mb > previous["peak_rss_mb"]:
            by_key[key] = {
                "file": file_path,
                "object": object_path,
                "peak_rss_mb": peak_rss_mb,
            }
    files = sorted(by_key.values(), key=lambda item: -item["peak_rss_mb"])
    max_peak = files[0]["peak_rss_mb"] if files else 0.0
    return {
        "rss_unit": "MB",
        "file_count": len(files),
        "max_peak_rss_mb": max_peak,
        "files": files,
    }


def format_top_section(compile_memory, count=10):
    lines = ["**** Files with highest peak RSS (compiler process):"]
    for item in compile_memory["files"][:count]:
        path = item["file"] or item["object"]
        lines.append(f"{item['peak_rss_mb']:8.1f} MB: {path}")
    if not compile_memory["files"]:
        lines.append("  (no memlog records)")
    lines.append("")
    return "\n".join(lines) + "\n"


def append_report(report_path, section):
    path = Path(report_path)
    existing = path.read_text() if path.is_file() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + section)


def main():
    parser = argparse.ArgumentParser(description="Parse memlog RSS records")
    parser.add_argument("--memlog-dir", required=True)
    parser.add_argument("--append-report")
    parser.add_argument("--metric-out")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    compile_memory = aggregate(load_records(args.memlog_dir), prefix_candidates())
    section = format_top_section(compile_memory, args.top)

    if args.append_report:
        append_report(args.append_report, section)

    if args.metric_out:
        report_text = ""
        if args.append_report and Path(args.append_report).is_file():
            report_text = Path(args.append_report).read_text()
        Path(args.metric_out).write_text(
            json.dumps(
                {"data": report_text, "compile_memory": compile_memory},
                indent=3,
            )
            + "\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
