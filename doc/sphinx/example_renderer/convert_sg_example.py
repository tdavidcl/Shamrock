#!/usr/bin/env python3
"""
Convert sphinx-gallery examples to the percent format used by render_example.py (prototype).

Input (sphinx-gallery):          Output (jupytext percent, MyST Markdown prose):

    '''                              # %% [markdown]
    Title                            # # Title
    =====                            #
                                     # Intro with {math}`x` roles
    Intro with :math:`x` roles
    '''                              # %%
    import shamrock                  import shamrock

    # %%                             # %% [markdown]
    # Section                        # ## Section
    # -------                        #
    # Text, ``code``                 # Text, `code`
    x = 1                            # %%
                                     x = 1

* The docstring becomes the first Markdown cell (its title becomes the H1).
* Text blocks (``# %%`` or a line of 20+ ``#`` followed by comment lines) become
  Markdown cells; rST is converted to MyST: headings, ``code``, :role:`x`,
  `text <url>`_, ``.. directive::`` blocks, ``.. _label:``, literal ``::`` blocks.
* A text chunk using rST the converter does not handle (substitutions, tables,
  footnotes, named references, ...) is kept as a ``# %% [rst]`` cell; headings
  are always converted to Markdown so that sections nest properly.
* ``# sphinx_gallery_thumbnail_number = N`` becomes ``# shamdoc_thumbnail_number = N``;
  ``# sphinx_gallery_multi_image`` lines are dropped.
* The code is checked to be unchanged (same AST, docstring aside).

Usage:
    python convert_sg_example.py examples/sph/run_x.py -o /tmp/new/run_x.py
    python convert_sg_example.py examples -o /tmp/examples_new     # whole tree
    python convert_sg_example.py examples --in-place
    add --keep-rst to keep all prose as rST cells (only headings converted)
"""

import argparse
import ast
import inspect
import re
import shutil
import sys
from pathlib import Path

SEPARATOR_RE = re.compile(r"^(# %%.*|#{20,}.*)$")
ADORNMENT_RE = re.compile(r"^([=\-`:'\"~^_*+#<>])\1{2,}\s*$")
DIRECTIVE_RE = re.compile(r"^(?P<indent>\s*)\.\.\s+(?P<name>[\w:.-]+)::\s*(?P<arg>.*)$")
LABEL_RE = re.compile(r"^\s*\.\.\s+_(?P<label>[^:`]+):\s*$")
COMMENT_RE = re.compile(r"^\s*\.\.(\s+(?P<text>.*))?$")
OPTION_RE = re.compile(r"^:(?P<key>[\w-]+):\s*(?P<value>.*)$")
LINK_RE = re.compile(r"`(?P<text>[^`<]+?)\s*<(?P<url>[^<>`]+)>`__?")
AUTOLINK_RE = re.compile(r"`<(?P<url>[^<>`]+)>`__?")
ROLE_RE = re.compile(
    r"(?<![\w`]):(?P<role>[A-Za-z][\w+.-]*(?::[A-Za-z][\w+.-]*)?):`(?P<content>[^`]+)`"
)
DOUBLE_RE = re.compile(r"``(?P<content>.+?)``")
UNSUPPORTED_INLINE = [
    re.compile(r"`[^`]+`_"),  # named reference
    re.compile(r"\|[\w -]+\|"),  # substitution
    re.compile(r"\[(#\w*|\d+|\*)\]_"),  # footnote
]
UNSUPPORTED_LINE = [
    re.compile(r"^\s*\+[-=+]+\+\s*$"),  # grid table
    re.compile(r"^\s*=+(\s+=+)+\s*$"),  # simple table
    re.compile(r"^\s*\|\s"),  # line block
]
LITERAL_DIRECTIVES = {
    "math",
    "code-block",
    "code",
    "sourcecode",
    "raw",
    "literalinclude",
    "csv-table",
    "graphviz",
    "highlight",
}
DIRECTIVE_ALIASES = {"sourcecode": "code-block"}


class Unsupported(Exception):
    pass


def _indent(line):
    return len(line) - len(line.lstrip())


def _uncomment(line):
    if line.startswith("# "):
        return line[2:]
    if line.startswith("#"):
        return line[1:]
    return line


def _fence(name, arg, options, body):
    text = "\n".join(body)
    ticks = "`" * max(3, max((len(r) for r in re.findall(r"`+", text)), default=0) + 1)
    out = [f"{ticks}{{{name}}}" + (f" {arg}" if arg else "")]
    out += options
    if body:
        if options:
            out.append("")
        out += body
    out.append(ticks)
    return out


# ---------------------------------------------------------------------------
# sphinx-gallery splitting
# ---------------------------------------------------------------------------


def split_sphinx_gallery(src):
    """Return (blocks, original_code_ast); blocks are ("text"|"code", [lines])."""
    tree = ast.parse(src)
    body = tree.body
    lines = src.splitlines()
    blocks = []
    start = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        blocks.append(("text", inspect.cleandoc(body[0].value.value).splitlines()))
        start = body[0].end_lineno
        body = body[1:]
    code_ast = ast.dump(ast.Module(body=body, type_ignores=[]))

    code = []
    i = start
    while i < len(lines):
        if SEPARATOR_RE.match(lines[i]):
            blocks.append(("code", code))
            code = []
            i += 1
            text = []
            while i < len(lines) and lines[i].startswith("#"):
                text.append(_uncomment(lines[i]))
                i += 1
            blocks.append(("text", text))
            continue
        code.append(lines[i])
        i += 1
    blocks.append(("code", code))
    return blocks, code_ast


# ---------------------------------------------------------------------------
# rST -> MyST
# ---------------------------------------------------------------------------


def convert_inline(line):
    line = AUTOLINK_RE.sub(lambda m: f"<{m['url']}>", line)
    line = LINK_RE.sub(lambda m: f"[{m['text']}]({m['url']})", line)
    line = ROLE_RE.sub(lambda m: f"{{{m['role']}}}`{m['content']}`", line)
    line = DOUBLE_RE.sub(lambda m: f"`{m['content']}`", line)
    outside_code = re.sub(r"`[^`]*`", "", line)
    for pattern in UNSUPPORTED_INLINE:
        if pattern.search(outside_code):
            raise Unsupported(f"inline construct {pattern.pattern!r} in: {line.strip()}")
    for pattern in UNSUPPORTED_LINE:
        if pattern.match(line):
            raise Unsupported(f"block construct {pattern.pattern!r} in: {line.strip()}")
    line = re.sub(r"^(\s*)#\.(\s)", r"\g<1>1.\2", line)  # auto-enumerated list
    return line


def _take_block(lines, i, base_indent):
    """Lines after ``i`` that are blank or indented more than ``base_indent``."""
    j = i
    while j < len(lines) and (not lines[j].strip() or _indent(lines[j]) > base_indent):
        j += 1
    block = lines[i:j]
    while block and not block[-1].strip():
        block.pop()
    non_blank = [_indent(line) for line in block if line.strip()]
    shift = min(non_blank) if non_blank else 0
    return [line[shift:] if line.strip() else "" for line in block], i + len(block)


def convert_rst_body(lines):
    """Convert rST lines (no headings) to MyST; raise Unsupported on unknown constructs."""
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = DIRECTIVE_RE.match(line)
        if m:
            if m["indent"]:
                raise Unsupported(f"indented directive: {line.strip()}")
            block, i = _take_block(lines, i + 1, 0)
            options = []
            while block and OPTION_RE.match(block[0]):
                options.append(block.pop(0))
            while block and not block[0].strip():
                block.pop(0)
            name = DIRECTIVE_ALIASES.get(m["name"], m["name"])
            if m["name"] not in LITERAL_DIRECTIVES:
                block = convert_rst_body(block)
            out += _fence(name, m["arg"].strip(), options, block)
            continue
        m = LABEL_RE.match(line)
        if m:
            out.append(f"({m['label'].strip()})=")
            i += 1
            continue
        m = COMMENT_RE.match(line)
        if m:
            block, i = _take_block(lines, i + 1, _indent(line))
            out += [f"% {m['text'] or ''}".rstrip()] + [f"% {b}".rstrip() for b in block]
            continue
        stripped = line.rstrip()
        if stripped.endswith("::"):
            nxt = i + 1
            while nxt < len(lines) and not lines[nxt].strip():
                nxt += 1
            if nxt < len(lines) and _indent(lines[nxt]) > _indent(line):
                head = stripped[:-2].rstrip()
                if head and not stripped.endswith(" ::"):
                    head += ":"
                if head.strip():
                    out += [convert_inline(head), ""]
                block, i = _take_block(lines, i + 1, _indent(line))
                while block and not block[0].strip():
                    block.pop(0)
                out += _fence("code-block", "text", [], block)
                continue
        out.append(convert_inline(line).rstrip())
        i += 1
    return out


def split_headings(lines, levels):
    """Yield ("heading", "## Title") and ("text", [lines]) segments."""
    segments = []
    text = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        style = title = None
        if (
            ADORNMENT_RE.match(line)
            and i + 2 < n
            and lines[i + 1].strip()
            and lines[i + 2].rstrip() == line.rstrip()
        ):
            style, title, step = (line[0], True), lines[i + 1].strip(), 3
        elif (
            line.strip()
            and not line[0].isspace()
            and not ADORNMENT_RE.match(line)
            and i + 1 < n
            and ADORNMENT_RE.match(lines[i + 1])
            and len(lines[i + 1].rstrip()) >= len(line.rstrip())
        ):
            style, title, step = (lines[i + 1][0], False), line.strip(), 2
        if style is None:
            text.append(line)
            i += 1
            continue
        if text:
            segments.append(("text", text))
            text = []
        level = levels.setdefault(style, len(levels) + 1)
        segments.append(("heading", "#" * min(level, 6) + " " + convert_inline(title)))
        i += step
    if text:
        segments.append(("text", text))
    return segments


def convert_text_block(lines, levels, keep_rst, where, warnings):
    """Return a list of ("markdown"|"rst", [lines]) cells for one text block."""
    cells = []

    def add(kind, body):
        while body and not body[0].strip():
            body = body[1:]
        while body and not body[-1].strip():
            body = body[:-1]
        if not body:
            return
        if cells and cells[-1][0] == kind:
            cells[-1][1].extend(["", *body])
        else:
            cells.append((kind, list(body)))

    for kind, payload in split_headings(lines, levels):
        if kind == "heading":
            add("markdown", [payload])
            continue
        if keep_rst:
            add("rst", payload)
            continue
        try:
            add("markdown", convert_rst_body(payload))
        except Unsupported as exc:
            warnings.append(f"{where}: kept as rST cell ({exc})")
            add("rst", payload)
    return cells


# ---------------------------------------------------------------------------
# Whole file
# ---------------------------------------------------------------------------


def convert_code_lines(lines, where, warnings):
    out = []
    for line in lines:
        m = re.match(r"^(\s*)#\s*sphinx_gallery_(\w+)\s*=\s*(.*)$", line)
        if m:
            key = m.group(2)
            if key == "multi_image":
                continue
            if key != "thumbnail_number":
                warnings.append(f"{where}: unsupported config sphinx_gallery_{key}, renamed")
            line = f"{m.group(1)}# shamdoc_{key} = {m.group(3)}"
        out.append(line.rstrip())
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def convert_source(src, name="<example>", keep_rst=False):
    blocks, code_ast = split_sphinx_gallery(src)
    warnings = []
    levels = {}
    cells = []  # (kind, lines)
    for n, (kind, lines) in enumerate(blocks):
        where = f"{name} block {n}"
        if kind == "code":
            code = convert_code_lines(lines, where, warnings)
            if code:
                cells.append(("code", code))
        else:
            for cell in convert_text_block(lines, levels, keep_rst, where, warnings):
                if cells and cells[-1][0] == cell[0] == "markdown":
                    cells[-1][1].extend(["", *cell[1]])
                else:
                    cells.append(cell)

    out = []
    header = {"code": "# %%", "markdown": "# %% [markdown]", "rst": "# %% [rst]"}
    for kind, lines in cells:
        if out:
            out.append("")
        out.append(header[kind])
        if kind == "code":
            out += lines
        else:
            out += ["#" if not line.strip() else f"# {line}".rstrip() for line in lines]
    new_src = "\n".join(out) + "\n"

    new_ast = ast.dump(ast.parse(new_src))
    if new_ast != code_ast:
        raise RuntimeError(f"{name}: converted code differs from the original")
    return new_src, warnings


def convert_file(src_path, dst_path, keep_rst):
    text = src_path.read_text()
    if re.search(r"^# %% \[(markdown|md|rst)\]", text, re.MULTILINE):
        print(f"skip (already converted): {src_path}")
        if dst_path != src_path:
            shutil.copy2(src_path, dst_path)
        return []
    new_text, warnings = convert_source(text, str(src_path), keep_rst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(new_text)
    return warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert sphinx-gallery examples")
    parser.add_argument("input", type=Path, help="example file or directory")
    parser.add_argument("-o", "--output", type=Path, help="output file or directory")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--keep-rst", action="store_true", help="keep prose as rST cells")
    opts = parser.parse_args(argv)
    if not opts.in_place and opts.output is None:
        parser.error("give -o/--output or --in-place")

    if opts.input.is_file():
        pairs = [(opts.input, opts.input if opts.in_place else opts.output)]
    else:
        out_root = opts.input if opts.in_place else opts.output
        pairs = []
        for path in sorted(opts.input.rglob("*")):
            rel = path.relative_to(opts.input)
            if any(p.startswith(("_to_trash", "__pycache__", ".")) for p in rel.parts):
                continue
            if path.is_dir():
                continue
            if path.suffix == ".py":
                pairs.append((path, out_root / rel))
            elif not opts.in_place:  # copy data files, GALLERY_HEADER.rst, ...
                (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out_root / rel)

    failures = 0
    for src, dst in pairs:
        try:
            for warning in convert_file(src, dst, opts.keep_rst):
                print(f"warning: {warning}")
        except (SyntaxError, RuntimeError) as exc:
            failures += 1
            print(f"error: {src}: {exc}")
    print(f"converted {len(pairs) - failures}/{len(pairs)} file(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
