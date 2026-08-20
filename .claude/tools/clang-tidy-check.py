#!/usr/bin/env python3
"""Run clang-tidy on a single file using build/compile_commands.json.

clang-tidy can't invoke the AdaptiveCpp `acpp` compiler wrapper directly, so
this strips the SYCL/acpp-only flags (the same ones .clangd removes for
clangd) and swaps the compiler for plain clang++ before calling clang-tidy.

Picks the newest clang-tidy/clang++ pair found on PATH rather than a
hardcoded version, since the host running this (e.g. outside Claude Code on
the web) may not have the same LLVM version installed as this container.

Usage: .claude/tools/clang-tidy-check.py <path/to/file.cpp>
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

BAD_PREFIXES = (
    "-fsycl",
    "-fsycl-targets=",
    "--hipsycl-targets=",
    "--hipsycl-platform=",
    "--hipsycl-config-file=",
    "--hipsycl-cpu-cxx=",
    "--acpp-targets=",
    "--driver-mode=",
)


def versions_on_path(tool_name):
    """Major versions N for which <tool_name>-N is an executable on PATH."""
    versions = set()
    pattern = re.compile(rf"^{re.escape(tool_name)}-(\d+)$")
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        try:
            entries = os.listdir(directory or ".")
        except OSError:
            continue
        for entry in entries:
            m = pattern.match(entry)
            if m and os.access(os.path.join(directory, entry), os.X_OK):
                versions.add(int(m.group(1)))
    return versions


def pick_toolchain():
    """Newest clang-tidy-N/clang++-N pair on PATH, falling back to unversioned names."""
    common = versions_on_path("clang-tidy") & versions_on_path("clang++")
    if common:
        version = max(common)
        return f"clang-tidy-{version}", f"clang++-{version}"
    if shutil.which("clang-tidy") and shutil.which("clang++"):
        return "clang-tidy", "clang++"
    print("error: no clang-tidy/clang++ pair found on PATH", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/file.cpp>", file=sys.stderr)
        return 1

    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
    target = os.path.realpath(sys.argv[1])

    cdb_path = os.path.join(repo_root, "build", "compile_commands.json")
    if not os.path.exists(cdb_path):
        print(f"error: {cdb_path} not found — run shamconfigure first", file=sys.stderr)
        return 1

    with open(cdb_path) as f:
        cdb = json.load(f)

    entry = next((e for e in cdb if os.path.realpath(e["file"]) == target), None)
    if entry is None:
        print(f"error: {target} has no entry in {cdb_path}", file=sys.stderr)
        return 1

    clang_tidy, clang_cxx = pick_toolchain()

    parts = shlex.split(entry["command"])
    parts = [p for p in parts if not any(p.startswith(b) for b in BAD_PREFIXES)]
    parts[0] = clang_cxx
    new_entry = dict(entry)
    new_entry["command"] = " ".join(shlex.quote(p) for p in parts)

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "compile_commands.json"), "w") as f:
            json.dump([new_entry], f)
        return subprocess.run([clang_tidy, "-p", tmpdir, target]).returncode


if __name__ == "__main__":
    sys.exit(main())
