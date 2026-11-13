#!/usr/bin/env python3
"""
Render every example of a tree, in parallel, then build the index pages (prototype).

Each example runs in its own process (``render_example.py``), so a crash or a
Shamrock/MPI initialisation in one example cannot leak into another. Results are
cached: an example is skipped when its source, the runner and ``--cache-key``
(e.g. the shamrock version or commit) are unchanged and its last run succeeded.

Examples:

    # plain python (shamrock importable from PYTHONPATH)
    python render_all_examples.py --src examples --out _examples_gen -j 4

    # through the shamrock executable, 2 MPI ranks per example
    python render_all_examples.py --src examples --out _examples_gen -j 2 \\
        --launcher "mpirun -n 2 ./build/shamrock --sycl-cfg 0:0 --loglevel 1 --rscript"

Files whose name does not start with ``run_`` are rendered without execution,
always with the plain python interpreter (no launcher needed).
"""

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "render_example.py"


def find_examples(src, pattern, excludes):
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src)
        if any(part.startswith(("_", ".")) for part in rel.parts):
            continue
        if any(fnmatch.fnmatch(str(rel), ex) for ex in excludes):
            continue
        if pattern and not re.search(pattern, str(rel)):
            continue
        yield path


def cache_key(path, extra):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    h.update(RUNNER.read_bytes())
    h.update((extra or "").encode())
    return h.hexdigest()


def out_folder(out, src, path):
    return out / path.relative_to(src).parent / path.stem


def render_one(path, opts, executes):
    key = cache_key(path, opts.cache_key)
    folder = out_folder(opts.out, opts.src, path)
    meta_file = folder / "meta.json"
    if opts.missing_only and meta_file.exists():
        return path, "present", 0.0, None
    if not opts.force and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("cache_key") == key and meta.get("status") in ("ok", "not-executed"):
            return path, "cached", 0.0, None

    env = dict(os.environ)
    env.update(
        SHAMDOC_EXAMPLE=str(path),
        SHAMDOC_ROOT=str(opts.src),
        SHAMDOC_OUT=str(opts.out),
        SHAMDOC_EXEC=opts.exec,
        SHAMDOC_CACHE_KEY=key,
    )
    launcher = opts.launcher if executes else sys.executable
    cmd = [*shlex.split(launcher), str(RUNNER)]
    log = opts.out / "_logs" / (str(path.relative_to(opts.src)).replace(os.sep, "__") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log.open("w") as f:
        f.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    if proc.returncode == 0 and not executes:
        status = "not-executed"
    return path, status, elapsed, log


def prune(out, src):
    """Remove rendered examples whose source file no longer exists."""
    for meta_file in out.rglob("meta.json"):
        meta = json.loads(meta_file.read_text())
        if not (src / meta["source"]).exists():
            print(f"[shamdoc] pruning stale {meta_file.parent}")
            shutil.rmtree(meta_file.parent)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render all examples of a tree")
    parser.add_argument("--src", required=True, type=Path, help="examples root")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--pattern", help="regex on the path relative to --src")
    parser.add_argument(
        "--exclude", action="append", default=[], help="glob on the relative path (repeatable)"
    )
    parser.add_argument(
        "--launcher",
        default=sys.executable,
        help="command used to run the runner script for executed examples",
    )
    parser.add_argument("--exec", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--cache-key", default="", help="extra cache key, e.g. shamrock commit")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="only render examples without an output folder (e.g. not provided as a CI artifact)",
    )
    parser.add_argument("--no-index", action="store_true", help="do not build index pages")
    parser.add_argument("--title", help="title of the root index page")
    opts = parser.parse_args(argv)
    opts.src = opts.src.resolve()
    opts.out = opts.out.resolve()
    opts.out.mkdir(parents=True, exist_ok=True)

    paths = list(find_examples(opts.src, opts.pattern, opts.exclude))
    print(f"[shamdoc] {len(paths)} example(s) under {opts.src}")

    def executes(p):
        return opts.exec == "always" or (opts.exec == "auto" and p.stem.startswith("run_"))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, opts.jobs)) as pool:
        futures = [pool.submit(render_one, p, opts, executes(p)) for p in paths]
        for fut in concurrent.futures.as_completed(futures):
            path, status, elapsed, log = fut.result()
            rel = path.relative_to(opts.src)
            print(f"  {status:<22} {elapsed:8.1f} s  {rel}" + (f"  (log: {log})" if log else ""))
            results.append((rel, status))

    prune(opts.out, opts.src)

    if not opts.no_index:
        sys.path.insert(0, str(HERE))
        from build_example_index import build_index

        build_index(opts.out, opts.src, title_override=opts.title)
        print(f"[shamdoc] index pages written under {opts.out}")

    failed = [rel for rel, status in results if status.startswith("FAILED")]
    if failed:
        print(f"[shamdoc] {len(failed)} example(s) failed: " + ", ".join(map(str, failed)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
