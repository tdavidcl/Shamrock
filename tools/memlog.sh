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
        -o)
            ;;
        -o*)
            obj=${a#-o}
            ;;
        *.cpp | *.cc | *.cxx | *.c)
            src=$a
            ;;
    esac
    prev=$a
done

if [ ! -x /usr/bin/time ]; then
    echo "memlog.sh: /usr/bin/time not found; running without RSS measurement" >&2
    exec "$@"
fi

tmp=$(mktemp) || exit 1
cleanup() {
    rm -f "$tmp"
}
trap cleanup EXIT

uname_s=$(uname -s)
if [ "$uname_s" = Darwin ]; then
    /usr/bin/time -l -o "$tmp" "$@"
    status=$?
    rss_bytes=$(awk '/maximum resident set size/ { print $1; exit }' "$tmp")
    rss_kb=$(awk -v b="${rss_bytes:-0}" 'BEGIN { printf "%.0f", b / 1024 }')
else
    /usr/bin/time -f '%M' -o "$tmp" "$@"
    status=$?
    rss_kb=$(awk '/^[0-9]+$/ { v=$1 } END { print v+0 }' "$tmp")
fi

if [ -z "$rss_kb" ]; then
    rss_kb=0
fi

mb=$(awk -v kb="$rss_kb" 'BEGIN { printf "%.1f", kb / 1024 }')
label=$src
if [ -z "$label" ]; then
    label=$obj
fi
printf 'MEM_PEAK_MB=%s FILE=%s\n' "$mb" "$label" >&2

if [ -n "${MEMLOG_DIR:-}" ]; then
    mkdir -p "$MEMLOG_DIR" || exit "$status"
    rec=$(mktemp "$MEMLOG_DIR/XXXXXX") || exit "$status"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import json, sys; print(json.dumps({"file": sys.argv[1], "object": sys.argv[2], "peak_rss_kb": int(float(sys.argv[3] or 0))}))' \
            "$src" "$obj" "$rss_kb" >"$rec"
    else
        printf '{"file":"%s","object":"%s","peak_rss_kb":%s}\n' "$src" "$obj" "$rss_kb" >"$rec"
    fi
fi

exit "$status"
