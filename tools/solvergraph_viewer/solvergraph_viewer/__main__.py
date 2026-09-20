"""Command line entry point of the solvergraph trace viewer."""

from __future__ import annotations

import argparse
import glob
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="solvergraph-viewer",
        description=(
            "Visualize Shamrock solvergraph execution traces "
            "(produced with SHAM_SOLVERGRAPH_TRACE=1 or "
            "shamrock.enable_solvergraph_tracing())."
        ),
    )
    parser.add_argument(
        "traces",
        nargs="+",
        help="trace files (one per rank), e.g. shamrock_sgtrace_*.jsonl",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="live mode: tail the trace files while the run is in progress",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="initial replay speed factor (e.g. 0.1 for 1/10 real time)",
    )
    args = parser.parse_args(argv)

    # expand globs not expanded by the shell (windows)
    paths = []
    for pattern in args.traces:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])

    from .app import run_app

    run_app(paths, live=args.live, speed=args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
