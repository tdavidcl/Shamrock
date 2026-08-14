# Count lines of tracked source files, excluding git submodules.

import json
import os
import subprocess
import sys
from pathlib import Path


def submodule_prefixes():
    try:
        out = subprocess.check_output(
            [
                "git",
                "config",
                "--file",
                ".gitmodules",
                "--get-regexp",
                r"^submodule\..*\.path$",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [line.split(maxsplit=1)[1].strip() for line in out.splitlines() if line.strip()]


def in_submodule(path, prefixes):
    s = str(path)
    return any(s == prefix or s.startswith(prefix + "/") for prefix in prefixes)


def list_files(patterns, prefixes):
    out = subprocess.check_output(["git", "ls-files", "-z", "--"] + patterns)
    files = []
    seen = set()
    for raw in out.split(b"\0"):
        if not raw:
            continue
        path = Path(raw.decode())
        if path in seen or in_submodule(path, prefixes) or not path.is_file():
            continue
        seen.add(path)
        files.append(path)
    return files


def count_lines(files):
    total = 0
    for path in files:
        with path.open("rb") as handle:
            total += sum(1 for _ in handle)
    return total


def count_loc():
    prefixes = submodule_prefixes()
    categories = {
        "*.cpp": ["*.cpp"],
        "*.hpp": ["*.hpp"],
        "*.py": ["*.py"],
        "CMakeLists.txt + *.cmake": [
            "CMakeLists.txt",
            "**/CMakeLists.txt",
            "*.cmake",
        ],
    }

    result = {}
    for name, patterns in categories.items():
        result[name] = count_lines(list_files(patterns, prefixes))
    result["total"] = sum(result.values())
    return result


def main():
    os.chdir(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())

    result = count_loc()
    text = json.dumps(result, indent=3) + "\n"

    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
