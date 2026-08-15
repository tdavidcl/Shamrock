#!/usr/bin/env bash
# Compiler launcher: run a command and record its peak RSS.
# Linux: GNU /usr/bin/time -f '%M' (kilobytes)
# macOS: BSD /usr/bin/time -l (bytes)
#
# When MEMLOG_DIR is set, update MEMLOG_DIR/compile_memory.json (max RSS per file).

src=""
obj=""
prev=""
for a in "$@"; do
    if [ "$prev" = "-o" ]; then
        obj=$a
    fi
    case "$a" in
        -o) ;;
        -o*) obj=${a#-o} ;;
        *.cpp | *.cc | *.cxx | *.c) src=$a ;;
    esac
    prev=$a
done

if [ ! -x /usr/bin/time ]; then
    echo "memlog.sh: /usr/bin/time not found; running without RSS measurement" >&2
    exec "$@"
fi

tmp=$(mktemp) || exit 1
trap 'rm -f "$tmp"' EXIT

if [ "$(uname -s)" = Darwin ]; then
    /usr/bin/time -l -o "$tmp" "$@"
else
    /usr/bin/time -f '%M' -o "$tmp" "$@"
fi
status=$?

python3 - "$tmp" "$src" "$obj" "$(uname -s)" "${MEMLOG_DIR:-}" <<'PY'
import fcntl
import json
import os
import sys
import tempfile

time_file, src, obj, uname, memlog_dir = sys.argv[1:6]
text = open(time_file).read()
rss_kb = 0.0
if uname == "Darwin":
    for line in text.splitlines():
        if "maximum resident set size" in line:
            rss_kb = int(line.split()[0]) / 1024.0
            break
else:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.isdigit():
            rss_kb = float(stripped)

peak_rss_mb = round(rss_kb / 1024.0, 1)
print(f"MEM_PEAK_MB={peak_rss_mb:.1f} FILE={src or obj}", file=sys.stderr)

if not memlog_dir:
    raise SystemExit(0)

def strip_prefix(path):
    if not path:
        return path
    for key in ("SHAMROCK_DIR", "GITHUB_WORKSPACE"):
        root = os.environ.get(key)
        if not root:
            continue
        root = os.path.abspath(root)
        prefix = root if root.endswith(os.sep) else root + os.sep
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path

file_path = strip_prefix(src)
object_path = strip_prefix(obj)
key = file_path or object_path
if not key:
    raise SystemExit(0)

os.makedirs(memlog_dir, exist_ok=True)
out_path = os.path.join(memlog_dir, "compile_memory.json")
lock_path = out_path + ".lock"
with open(lock_path, "a+") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    data = {"rss_unit": "MB", "file_count": 0, "max_peak_rss_mb": 0.0, "files": []}
    if os.path.isfile(out_path):
        with open(out_path) as handle:
            data = json.load(handle)
    by_key = {(item.get("file") or item.get("object")): item for item in data.get("files", [])}
    previous = by_key.get(key)
    if previous is None or peak_rss_mb > previous.get("peak_rss_mb", 0):
        by_key[key] = {
            "file": file_path,
            "object": object_path,
            "peak_rss_mb": peak_rss_mb,
        }
    files = sorted(by_key.values(), key=lambda item: -item["peak_rss_mb"])
    data = {
        "rss_unit": "MB",
        "file_count": len(files),
        "max_peak_rss_mb": files[0]["peak_rss_mb"] if files else 0.0,
        "files": files,
    }
    fd, tmp_path = tempfile.mkstemp(dir=memlog_dir, suffix=".json")
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=3)
        handle.write("\n")
    os.replace(tmp_path, out_path)
PY

exit "$status"
