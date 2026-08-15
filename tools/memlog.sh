#!/usr/bin/env bash
# Compiler launcher: run a command and record its peak RSS.
# Linux: GNU /usr/bin/time -f '%M' (kilobytes)
# macOS: BSD /usr/bin/time -l (bytes)
#
# When MEMLOG_DIR is set, write one JSON record per invocation:
#   {"file": "...", "object": "...", "peak_rss_kb": 123456}

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

print(f"MEM_PEAK_MB={rss_kb / 1024.0:.1f} FILE={src or obj}", file=sys.stderr)

if memlog_dir:
    os.makedirs(memlog_dir, exist_ok=True)
    fd, _path = tempfile.mkstemp(dir=memlog_dir)
    with os.fdopen(fd, "w") as handle:
        json.dump(
            {"file": src, "object": obj, "peak_rss_kb": int(round(rss_kb))},
            handle,
        )
        handle.write("\n")
PY

exit "$status"
