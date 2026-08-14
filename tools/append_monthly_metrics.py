# Append a metric JSON document into metrics/<metric_id>/YYYY-MM.json as an array.

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

METRIC_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def monthly_path(metrics_root, metric_id, now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    return Path(metrics_root) / metric_id / f"{now.strftime('%Y-%m')}.json"


def validate_metric_id(metric_id):
    if not METRIC_ID_RE.fullmatch(metric_id):
        raise ValueError(
            f"invalid metric_id {metric_id!r}; expected characters in [A-Za-z0-9._-]"
        )


def load_entries(monthly_file):
    if not monthly_file.exists():
        return []
    entries = json.loads(monthly_file.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{monthly_file} is not a JSON array")
    return entries


def already_present(entries, incoming):
    if not isinstance(incoming, dict):
        return False
    run_id = incoming.get("run_id")
    if run_id is None:
        return False
    return any(isinstance(entry, dict) and entry.get("run_id") == run_id for entry in entries)


def write_entries(monthly_file, entries):
    monthly_file.parent.mkdir(parents=True, exist_ok=True)
    monthly_file.write_text(json.dumps(entries, indent=3) + "\n")


def append_monthly_metrics(metrics_root, metric_id, incoming, now=None):
    validate_metric_id(metric_id)
    monthly_file = monthly_path(metrics_root, metric_id, now=now)
    entries = load_entries(monthly_file)
    if already_present(entries, incoming):
        return monthly_file, False
    entries.append(incoming)
    write_entries(monthly_file, entries)
    return monthly_file, True


def relative_branch_path(metrics_root, monthly_file):
    return monthly_file.resolve().relative_to(Path(metrics_root).resolve().parent)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 3:
        sys.stderr.write("usage: append_monthly_metrics.py METRICS_ROOT METRIC_ID INCOMING.json\n")
        return 2

    metrics_root, metric_id, incoming_path = argv
    with Path(incoming_path).open() as handle:
        incoming = json.load(handle)

    monthly_file, appended = append_monthly_metrics(metrics_root, metric_id, incoming)
    status = "appended" if appended else "skipped"
    rel = relative_branch_path(metrics_root, monthly_file)
    sys.stdout.write(f"{status} {rel.as_posix()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
