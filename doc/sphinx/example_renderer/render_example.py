#!/usr/bin/env python3
"""
Render one Shamrock example into a MyST page + assets (prototype).

The example is a plain Python script written in the jupytext "percent" format:

    # %% [markdown]
    # # Title
    # Some *Markdown* prose, {math}`\\rho` roles and ```{directives}``` work.

    # %%
    import shamrock
    ...

    # %% [rst]
    # A cell kept as reStructuredText (rendered through ``{eval-rst}``).

    # %% tags=["hide-input"]
    plt.show()

Each code cell is executed in a shared namespace (like a notebook) and the runner
records, per cell: stdout at the file-descriptor level (so Shamrock's C++ logs are
captured too, while still being echoed to the terminal), the repr of the last
expression, matplotlib figures and matplotlib animations.

The output is a self-contained folder that can be uploaded as a CI artifact and
dropped anywhere inside the Sphinx source tree:

    <out>/<path relative to root>/<stem>/
        <stem>.md        MyST page, starts with the label (ex-<path>-<stem>)=
        <stem>.py        copy of the source, for the download link
        images/*         png figures, mp4 animations
        meta.json        title, summary, thumbnail, status, timings, ...

Usage with the python interpreter:

    python render_example.py --example examples/sph/run_x.py --root examples --out _gen

Usage through the shamrock executable (e.g. under MPI): shamrock owns the command
line, so the options are passed through environment variables instead:

    SHAMDOC_EXAMPLE=examples/sph/run_x.py SHAMDOC_ROOT=examples SHAMDOC_OUT=_gen \\
        mpirun -n 2 ./shamrock --sycl-cfg 0:0 --rscript render_example.py

Only rank 0 writes the output folder.

Supported cell tags (jupytext syntax ``# %% tags=["a", "b"]``):
    hide-input, hide-output     put the code / outputs in a collapsed dropdown
    remove-input, remove-output do not show the code / outputs
    remove-cell                 execute but show nothing
    skip-execution              show but do not execute
    thumbnail                   use the first figure of this cell as thumbnail
    full-output                 never truncate the text output of this cell

Supported config comments (at the top of a code cell, hidden from the page):
    # shamdoc_thumbnail_number = 2    (1-based, negative counts from the end)
"""

import argparse
import ast
import builtins
import contextlib
import ctypes
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

RUNNER_VERSION = "0.1"

CELL_RE = re.compile(r"^# %%(?P<rest>.*)$")
KIND_RE = re.compile(r"\[(?P<kind>\w+)\]")
TAGS_RE = re.compile(r"tags\s*=\s*(?P<tags>\[[^\]]*\])")
CONFIG_RE = re.compile(r"^#\s*shamdoc_(?P<key>\w+)\s*=\s*(?P<value>.+?)\s*$")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

CELL_KINDS = {"markdown": "markdown", "md": "markdown", "rst": "rst"}

KNOWN_TAGS = {
    "hide-input",
    "hide-output",
    "remove-input",
    "remove-output",
    "remove-cell",
    "skip-execution",
    "thumbnail",
    "full-output",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    kind: str  # "code", "markdown" or "rst"
    source: str
    lineno: int  # 1-based line number of the first source line in the file
    tags: list = field(default_factory=list)
    outputs: list = field(default_factory=list)  # list of (type, payload)
    elapsed: float = 0.0
    executed: bool = False


def _uncomment(line):
    if line.startswith("# "):
        return line[2:]
    if line.startswith("#"):
        return line[1:]
    return line


def _parse_header(rest):
    kind = "code"
    tags = []
    m = KIND_RE.search(rest)
    if m:
        kind = CELL_KINDS.get(m.group("kind").lower(), "code")
    m = TAGS_RE.search(rest)
    if m:
        try:
            tags = [str(t) for t in ast.literal_eval(m.group("tags"))]
        except (ValueError, SyntaxError):
            print(f"[shamdoc] warning: could not parse cell tags in '# %%{rest}'")
    for t in tags:
        if t not in KNOWN_TAGS:
            print(f"[shamdoc] warning: unknown cell tag '{t}'")
    return kind, tags


def parse_cells(text):
    """Split a percent-format script into cells."""
    cells = []

    def flush(lines, kind, tags, start):
        i, j = 0, len(lines)
        while i < j and not lines[i].strip():
            i += 1
        while j > i and not lines[j - 1].strip():
            j -= 1
        body = lines[i:j]
        if not body:
            return
        if kind != "code":
            body = [_uncomment(line) for line in body]
        cells.append(Cell(kind, "\n".join(body), start + i, list(tags)))

    cur, kind, tags, start = [], "code", [], 1
    for n, line in enumerate(text.splitlines(), 1):
        m = CELL_RE.match(line)
        if m:
            flush(cur, kind, tags, start)
            cur, start = [], n + 1
            kind, tags = _parse_header(m.group("rest"))
            continue
        cur.append(line)
    flush(cur, kind, tags, start)
    return cells


def extract_config(cells):
    """Pop ``# shamdoc_key = value`` lines at the top of code cells."""
    config = {}
    for cell in cells:
        if cell.kind != "code":
            continue
        lines = cell.source.split("\n")
        dropped = 0
        while dropped < len(lines):
            line = lines[dropped]
            m = CONFIG_RE.match(line)
            if m:
                try:
                    config[m.group("key")] = ast.literal_eval(m.group("value"))
                except (ValueError, SyntaxError):
                    config[m.group("key")] = m.group("value")
            elif line.strip():
                break
            dropped += 1
        if dropped:
            cell.source = "\n".join(lines[dropped:])
            cell.lineno += dropped
    return config


def shamrock_api_names(text):
    """Dotted ``shamrock.*`` names referenced by the script (for backreferences)."""
    names = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id == "shamrock":
                names.add(".".join(["shamrock", *reversed(parts)]))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "shamrock":
                for alias in node.names:
                    names.add(f"{node.module}.{alias.name}")
    return sorted(names)


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _env_int(*names):
    for name in names:
        if name in os.environ:
            try:
                return int(os.environ[name])
            except ValueError:
                pass
    return None


def _shamrock_initialized():
    sham = sys.modules.get("shamrock")
    if sham is None:
        return None
    try:
        if sham.sys.is_initialized():
            return sham
    except (AttributeError, RuntimeError):
        pass
    return None


def current_rank():
    sham = _shamrock_initialized()
    if sham is not None:
        return sham.sys.world_rank()
    rank = _env_int(
        "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID"
    )
    return rank or 0


def current_world_size():
    sham = _shamrock_initialized()
    if sham is not None:
        return sham.sys.world_size()
    size = _env_int("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS")
    return size or 1


def _libc_fflush():
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError):
        pass


def _write_all(fd, data):
    while data:
        n = os.write(fd, data)
        data = data[n:]


class StdoutCapture:
    """Capture everything written to fd 1 (Python, C, C++) and still echo it live.

    Nothing on the capture path may need the GIL: pybind11 calls (all of Shamrock)
    hold it for their whole duration, e.g. a full ``model.evolve_until(...)``.

    * With ``tee`` available (Linux, macOS), fd 1 is a pipe into a ``tee`` *process*
      that writes a temporary file and the real stdout. A separate process drains
      the pipe whatever the GIL does, so output stays live and the writer never
      blocks for good.
    * Otherwise fd 1 goes straight to the temporary file (writing a regular file
      never blocks) and a Python thread echoes it; the echo then only appears when
      the GIL is released.

    A pipe drained by a Python thread must not be used: once it holds 64 kB the
    C++ writer blocks while the reader waits for the GIL, a deadlock.
    """

    def __enter__(self):
        sys.stdout.flush()
        _libc_fflush()
        self._saved = os.dup(1)
        fd, self._path = tempfile.mkstemp(prefix="shamdoc-", suffix=".out")
        os.close(fd)
        self._proc = None
        self._thread = None
        tee = shutil.which("tee")
        if tee:
            self._proc = subprocess.Popen(
                [tee, self._path], stdin=subprocess.PIPE, stdout=self._saved, close_fds=True
            )
            os.dup2(self._proc.stdin.fileno(), 1)
        else:
            self._file = open(self._path, "wb")  # closed in __exit__
            os.dup2(self._file.fileno(), 1)
            self._echoed = 0
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._tail, daemon=True)
            self._thread.start()
        return self

    def _echo_new(self):
        with open(self._path, "rb") as f:
            f.seek(self._echoed)
            while data := f.read(1 << 20):
                self._echoed += len(data)
                with contextlib.suppress(OSError):
                    _write_all(self._saved, data)

    def _tail(self):
        while not self._stop.wait(0.2):
            self._echo_new()

    def __exit__(self, *exc):
        sys.stdout.flush()
        _libc_fflush()
        os.dup2(self._saved, 1)  # fd 1 no longer refers to the pipe / file
        if self._proc is not None:
            self._proc.stdin.close()  # last write end: tee gets EOF and exits
            self._proc.wait()
        else:
            self._stop.set()
            self._thread.join()
            self._file.close()
            self._echo_new()
        os.close(self._saved)
        with open(self._path, "rb") as f:
            self.text = f.read().decode("utf-8", "replace")
        os.unlink(self._path)
        return False


def run_code(source, filename, namespace):
    """Exec ``source``; return (has_value, value) for a trailing expression."""
    tree = ast.parse(source, filename)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
    exec(compile(tree, filename, "exec"), namespace)  # noqa: S102 (running examples is the point)
    if last is None:
        return False, None
    expr = ast.Expression(last.value)
    return True, eval(compile(expr, filename, "eval"), namespace)


def format_example_exception(exc):
    """Traceback without the runner's own frames."""
    runner_file = run_code.__code__.co_filename
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename == runner_file:
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(exc), exc, tb))


def format_repr(value):
    if value is None:
        return None
    if (type(value).__module__ or "").startswith("matplotlib"):
        return None
    if isinstance(value, (list, tuple)) and value:
        if all((type(v).__module__ or "").startswith("matplotlib") for v in value):
            return None
    html_repr = getattr(value, "_repr_html_", None)
    if callable(html_repr):
        try:
            text = html_repr()
        except Exception:  # noqa: BLE001 (arbitrary user object)
            text = None
        if text:
            return ("html", text)
    return ("text", repr(value))


def clean_text_output(text, max_lines, full):
    text = ANSI_RE.sub("", text)
    lines = [line.rstrip("\r").rsplit("\r", 1)[-1].rstrip() for line in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    if not full and max_lines and len(lines) > max_lines:
        head = max_lines // 2
        tail = max_lines - head
        skipped = len(lines) - max_lines
        lines = [*lines[:head], f"... [{skipped} lines truncated] ...", *lines[-tail:]]
    return "\n".join(lines)


@contextlib.contextmanager
def example_context(example):
    """cwd, sys.argv, sys.path and matplotlib set up as if the script ran alone."""
    old_cwd = os.getcwd()
    old_argv = sys.argv
    old_path = list(sys.path)
    os.environ["MPLBACKEND"] = "Agg"
    plt = None
    old_show = None
    with contextlib.suppress(ImportError):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        old_show = plt.show
        plt.show = lambda *args, **kwargs: None
    try:
        os.chdir(example.parent)
        sys.argv = [str(example)]
        sys.path.insert(0, str(example.parent))
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path[:] = old_path
        if plt is not None:
            plt.show = old_show


def peak_rss_mib():
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 2**20
    return rss / 2**10


# ---------------------------------------------------------------------------
# MyST emission helpers
# ---------------------------------------------------------------------------


def _longest_backtick_run(text):
    runs = re.findall(r"`+", text)
    return max((len(r) for r in runs), default=0)


def fence(directive, body="", options=None, arg=""):
    ticks = "`" * max(3, _longest_backtick_run(body) + 1)
    head = f"{ticks}{{{directive}}}" + (f" {arg}" if arg else "")
    lines = [head]
    for key, value in (options or {}).items():
        lines.append(f":{key}:" + (f" {value}" if value != "" else ""))
    if body:
        if options:
            lines.append("")
        lines.append(body)
    lines.append(ticks)
    return "\n".join(lines) + "\n"


def slugify(text):
    return re.sub(r"[^a-z0-9_]+", "-", text.lower()).strip("-")


def first_paragraph(markdown):
    para = []
    for line in markdown.split("\n"):
        if line.startswith("#") or line.startswith("```") or line.startswith(":::"):
            if para:
                break
            continue
        if not line.strip():
            if para:
                break
            continue
        para.append(line.strip())
    text = " ".join(para)
    return text if len(text) <= 300 else text[:297] + "..."


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    def __init__(self, example, root, out_root, opts):
        self.example = example
        self.rel_dir = example.parent.relative_to(root)
        self.out_dir = out_root / self.rel_dir / example.stem
        self.opts = opts
        self.stem = example.stem
        self.label = "ex-" + slugify("-".join([*self.rel_dir.parts, self.stem]))
        self.assets = {}  # name -> bytes, written at the end by rank 0
        self.images = []  # (cell index, asset name) of displayed png figures
        self.anim_thumbs = []
        self.seen_anims = []
        self.failed = False
        self.stopped = False

    # -- execution ----------------------------------------------------------

    def should_execute(self):
        if self.opts.exec == "always":
            return True
        if self.opts.exec == "never":
            return False
        return self.stem.startswith("run_")

    def execute(self):
        namespace = {
            "__name__": "__main__",
            "__file__": str(self.example),
            "__builtins__": builtins,
        }
        with example_context(self.example):
            for index, cell in enumerate(self.cells):
                if cell.kind != "code" or "skip-execution" in cell.tags:
                    continue
                if self.stopped:
                    break
                self.run_cell(index, cell, namespace)

    def run_cell(self, index, cell, namespace):
        padded = "\n" * (cell.lineno - 1) + cell.source
        suppress = cell.source.rstrip().endswith(";")
        has_value, value, error = False, None, None
        t0 = time.perf_counter()
        with StdoutCapture() as cap:
            try:
                has_value, value = run_code(padded, str(self.example), namespace)
            except SystemExit as exc:
                self.stopped = True
                if exc.code not in (0, None):
                    self.failed = True
                    error = format_example_exception(exc)
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # noqa: BLE001 (report any error of the example)
                self.stopped = True
                self.failed = True
                error = format_example_exception(exc)
        cell.elapsed = time.perf_counter() - t0
        cell.executed = True

        full = "full-output" in cell.tags
        text = clean_text_output(cap.text, self.opts.max_output_lines, full)
        if text:
            cell.outputs.append(("stdout", text))
        if error:
            sys.stderr.write(error)
            cell.outputs.append(("error", error))
        if has_value and not suppress:
            rep = format_repr(value)
            if rep is not None:
                cell.outputs.append(rep)

        anim_figs = self.collect_animations(cell, namespace)
        self.collect_figures(index, cell, anim_figs)

    def collect_animations(self, cell, namespace):
        anim_mod = sys.modules.get("matplotlib.animation")
        figs = []
        if anim_mod is None:
            return figs
        for name, value in list(namespace.items()):
            if not isinstance(value, anim_mod.Animation):
                continue
            if any(value is seen for seen in self.seen_anims):
                continue
            self.seen_anims.append(value)
            fig = getattr(value, "_fig", None)
            if fig is not None:
                figs.append(fig)
            if current_rank() != 0:
                continue
            base = f"{self.stem}_{name}"
            if anim_mod.writers.is_available("ffmpeg"):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "anim.mp4"
                    value.save(str(path), writer="ffmpeg")
                    self.assets[base + ".mp4"] = path.read_bytes()
                cell.outputs.append(("video", base + ".mp4"))
            else:
                print("[shamdoc] ffmpeg not available, embedding the animation as HTML")
                cell.outputs.append(("html", value.to_jshtml()))
            if fig is not None:
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                self.assets[base + "_thumb.png"] = buf.getvalue()
                self.anim_thumbs.append(base + "_thumb.png")
        return figs

    def collect_figures(self, index, cell, exclude):
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is None:
            return
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if not any(fig is f for f in exclude) and current_rank() == 0:
                name = f"{self.stem}_{len(self.images) + 1:03d}.png"
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                self.assets[name] = buf.getvalue()
                self.images.append((index, name))
                cell.outputs.append(("image", name))
            plt.close(fig)

    # -- output -------------------------------------------------------------

    def pick_thumbnail(self):
        for index, name in self.images:
            if "thumbnail" in self.cells[index].tags:
                return name
        number = self.config.get("thumbnail_number")
        if isinstance(number, int) and self.images:
            pos = number - 1 if number > 0 else len(self.images) + number
            if 0 <= pos < len(self.images):
                return self.images[pos][1]
        if self.images:
            return self.images[0][1]
        if self.anim_thumbs:
            return self.anim_thumbs[0]
        return None

    def render_outputs(self, cell):
        parts = []
        for kind, payload in cell.outputs:
            if kind == "stdout":
                parts.append(fence("code-block", payload, {"class": "sham-output"}, "text"))
            elif kind == "text":
                parts.append(fence("code-block", payload, {"class": "sham-output"}, "python"))
            elif kind == "error":
                parts.append(fence("code-block", payload, {"class": "sham-error"}, "pytb"))
            elif kind == "html":
                parts.append(fence("raw", payload, arg="html"))
            elif kind == "image":
                parts.append(fence("image", "", {"class": "sham-figure"}, f"images/{payload}"))
            elif kind == "video":
                parts.append(fence("video", "", {"width": "100%"}, f"images/{payload}"))
        return "\n".join(parts)

    def render_code_cell(self, cell):
        tags = cell.tags
        if "remove-cell" in tags:
            return ""
        parts = []
        if "remove-input" not in tags:
            code = fence(
                "code-block",
                cell.source,
                {"linenos": "", "lineno-start": cell.lineno},
                "python",
            )
            if "hide-input" in tags:
                code = fence("dropdown", code, {"class-container": "sham-hidden-code"}, "Code")
            parts.append(code)
        if "remove-output" not in tags and cell.outputs:
            out = self.render_outputs(cell)
            if "hide-output" in tags:
                out = fence("dropdown", out, arg="Output")
            parts.append(out)
        return "\n".join(parts)

    def build_page(self, status, stats):
        blocks = [f"({self.label})="]
        cells = list(self.cells)
        title = None
        if cells and cells[0].kind == "markdown" and cells[0].source.startswith("# "):
            first = cells.pop(0)
            title_line, _, rest = first.source.partition("\n")
            title = title_line[2:].strip()
            blocks.append(title_line + "\n")
            intro = rest.strip("\n")
        else:
            title = self.stem
            blocks.append(f"# {title}\n")
            intro = ""
        # The intro is the first cell up to its first sub-heading; the download note and
        # the failure warning go right after it, before any sub-section starts.
        intro_lines = intro.split("\n") if intro else []
        split = next((i for i, ln in enumerate(intro_lines) if ln.startswith("#")), None)
        if split is None:
            split = len(intro_lines)
        intro = "\n".join(intro_lines[:split]).strip("\n")
        intro_rest = "\n".join(intro_lines[split:]).strip("\n")
        if intro:
            blocks.append(intro + "\n")

        info = f"{{download}}`Download this example <{self.stem}.py>`"
        blocks.append(fence("note", info))
        if status == "failed":
            blocks.append(
                fence(
                    "warning",
                    "This example failed while the documentation was rendered. "
                    "Outputs after the failing cell are missing.",
                )
            )
        if intro_rest:
            blocks.append(intro_rest + "\n")

        for cell in cells:
            if cell.kind == "markdown":
                blocks.append(cell.source + "\n")
            elif cell.kind == "rst":
                blocks.append(fence("eval-rst", cell.source))
            else:
                rendered = self.render_code_cell(cell)
                if rendered:
                    blocks.append(rendered)

        if stats["executed"]:
            lines = [
                f"- Status: {status}",
                f"- Wall time: {stats['wall_time_s']:.2f} s",
                f"- MPI ranks: {stats['world_size']}",
            ]
            if stats["peak_rss_mib"] is not None:
                lines.append(f"- Peak memory (RSS, rank 0): {stats['peak_rss_mib']:.0f} MiB")
            if stats["shamrock_version"]:
                lines.append(f"- Shamrock version: {stats['shamrock_version']}")
            blocks.append(fence("dropdown", "\n".join(lines), arg="Run information"))

        return title, intro, "\n".join(blocks)

    def write(self, status, stats):
        title, intro, page = self.build_page(status, stats)
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        (self.out_dir / "images").mkdir(parents=True)
        for name, data in self.assets.items():
            (self.out_dir / "images" / name).write_bytes(data)
        (self.out_dir / f"{self.stem}.md").write_text(page)
        shutil.copy2(self.example, self.out_dir / f"{self.stem}.py")
        thumb = self.pick_thumbnail()
        meta = {
            "runner_version": RUNNER_VERSION,
            "source": str(self.rel_dir / self.example.name),
            "label": self.label,
            "page": f"{self.stem}.md",
            "title": title,
            "summary": first_paragraph(intro),
            "thumbnail": f"images/{thumb}" if thumb else None,
            "status": status,
            "cache_key": self.opts.cache_key,
            "shamrock_api": shamrock_api_names(self.source),
            "cells": [
                {"lineno": c.lineno, "elapsed_s": round(c.elapsed, 4)}
                for c in self.cells
                if c.executed
            ],
            **stats,
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[shamdoc] wrote {self.out_dir} ({status})")

    def run(self):
        self.source = self.example.read_text()
        self.cells = parse_cells(self.source)
        self.config = extract_config(self.cells)
        executed = self.should_execute()
        t0 = time.perf_counter()
        if executed:
            self.execute()
        wall = time.perf_counter() - t0
        status = "failed" if self.failed else ("ok" if executed else "not-executed")
        sham = _shamrock_initialized()
        stats = {
            "executed": executed,
            "wall_time_s": round(wall, 3),
            "world_size": current_world_size(),
            "peak_rss_mib": peak_rss_mib() if executed else None,
            "shamrock_version": sham.version_string() if sham is not None else None,
        }
        if current_rank() == 0:
            self.write(status, stats)
        return status


def parse_options(argv=None):
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--example", default=env("SHAMDOC_EXAMPLE"))
    parser.add_argument(
        "--root",
        default=env("SHAMDOC_ROOT"),
        help="examples root; the output mirrors the path relative to it (default: example dir)",
    )
    parser.add_argument("--out", default=env("SHAMDOC_OUT", "_examples_gen"))
    parser.add_argument(
        "--exec",
        choices=["auto", "always", "never"],
        default=env("SHAMDOC_EXEC", "auto"),
        help="auto: execute only files named run_*.py",
    )
    parser.add_argument(
        "--max-output-lines", type=int, default=int(env("SHAMDOC_MAX_OUTPUT_LINES", "400"))
    )
    parser.add_argument("--cache-key", default=env("SHAMDOC_CACHE_KEY"))
    # parse_known_args: under `shamrock --rscript`, sys.argv holds shamrock's own options
    opts, _ = parser.parse_known_args(argv)
    if not opts.example:
        parser.error("no example given (--example or SHAMDOC_EXAMPLE)")
    return opts


def main(argv=None):
    opts = parse_options(argv)
    example = Path(opts.example).resolve()
    root = Path(opts.root).resolve() if opts.root else example.parent
    renderer = Renderer(example, root, Path(opts.out).resolve(), opts)
    status = renderer.run()
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    code = main()
    if code:
        raise SystemExit(code)
