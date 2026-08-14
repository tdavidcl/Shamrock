# Convert a ClangBuildAnalyzer --analyze text report into JSON.

import json
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SECTION_RE = re.compile(r"^\*{4} (.+):$")
COMPILATION_RE = re.compile(r"^Compilation \((\d+) times\):$")
PARSE_RE = re.compile(r"^Parsing \(frontend\):\s+([0-9.]+) s$")
CODEGEN_RE = re.compile(r"^Codegen & opts \(backend\):\s+([0-9.]+) s$")
MS_LINE_RE = re.compile(r"^(\d+) ms: (.+)$")
TIMES_AVG_RE = re.compile(r"^(.*) \((\d+) times, avg (\d+) ms\)$")
FUNC_FILE_RE = re.compile(r"^(.*) \((.+)\)$")
HEADER_RE = re.compile(r"^(\d+) ms: (.+) \(included (\d+) times, avg (\d+) ms\), included via:$")
CHAIN_RE = re.compile(r"^(\d+)x: (.*)$")

SECTION_KEYS = {
    "Files that took longest to parse (compiler frontend)": "files_parse",
    "Files that took longest to codegen (compiler backend)": "files_codegen",
    "Templates that took longest to instantiate": "templates",
    "Template sets that took longest to instantiate": "template_sets",
    "Functions that took longest to compile": "functions",
    "Function sets that took longest to compile / optimize": "function_sets",
    "Expensive headers": "expensive_headers",
}


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def empty_result():
    return {
        "compilation_count": 0,
        "parse_frontend_s": 0.0,
        "codegen_backend_s": 0.0,
        "files_parse": [],
        "files_codegen": [],
        "templates": [],
        "template_sets": [],
        "functions": [],
        "function_sets": [],
        "expensive_headers": [],
    }


def parse_ms_entry(section_key, rest):
    times_avg = TIMES_AVG_RE.fullmatch(rest)
    if times_avg:
        return {
            "ms": None,
            "name": times_avg.group(1),
            "count": int(times_avg.group(2)),
            "avg_ms": int(times_avg.group(3)),
        }

    if section_key == "functions":
        func_file = FUNC_FILE_RE.fullmatch(rest)
        if func_file:
            return {
                "ms": None,
                "name": func_file.group(1),
                "file": func_file.group(2),
            }

    return {"ms": None, "file": rest}


def parse_report(text):
    result = empty_result()
    section = None
    current_header = None

    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("Analyzing build trace"):
            continue
        if stripped.startswith("done in "):
            continue

        section_match = SECTION_RE.match(stripped)
        if section_match:
            title = section_match.group(1)
            if title == "Time summary":
                section = "time_summary"
            else:
                section = SECTION_KEYS.get(title)
            current_header = None
            continue

        if section == "time_summary":
            compilation = COMPILATION_RE.match(stripped)
            if compilation:
                result["compilation_count"] = int(compilation.group(1))
                continue
            parse = PARSE_RE.match(stripped)
            if parse:
                result["parse_frontend_s"] = float(parse.group(1))
                continue
            codegen = CODEGEN_RE.match(stripped)
            if codegen:
                result["codegen_backend_s"] = float(codegen.group(1))
            continue

        if section == "expensive_headers":
            header = HEADER_RE.match(stripped)
            if header:
                current_header = {
                    "ms": int(header.group(1)),
                    "file": header.group(2),
                    "count": int(header.group(3)),
                    "avg_ms": int(header.group(4)),
                    "included_via": [],
                }
                result["expensive_headers"].append(current_header)
                continue
            if stripped == "...":
                continue
            chain = CHAIN_RE.match(stripped)
            if chain and current_header is not None:
                chain_text = chain.group(2).strip()
                if chain_text == "<direct include>":
                    files = ["<direct include>"]
                else:
                    files = chain_text.split() if chain_text else []
                current_header["included_via"].append(
                    {
                        "count": int(chain.group(1)),
                        "chain": files,
                    }
                )
            continue

        if section in {
            "files_parse",
            "files_codegen",
            "templates",
            "template_sets",
            "functions",
            "function_sets",
        }:
            ms_line = MS_LINE_RE.match(stripped)
            if not ms_line:
                continue
            entry = parse_ms_entry(section, ms_line.group(2))
            entry["ms"] = int(ms_line.group(1))
            result[section].append(entry)

    return result


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: clang_build_analyzer_to_json.py <report.txt> <out.json>\n")
        sys.exit(2)

    report_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    result = parse_report(report_path.read_text())
    text = json.dumps(result, indent=3) + "\n"
    out_path.write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
