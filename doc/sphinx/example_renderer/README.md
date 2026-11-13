# Example renderer (prototype)

A replacement for sphinx-gallery: each example is executed on its own and
rendered to a **self-contained folder** (MyST page + images + `meta.json`) that
Sphinx then ingests like any other page. Running the examples and building the
docs become two separate steps, so a CI job can run one example and upload its
folder as an artifact, and the main job runs `sphinx-build` once without
executing anything.

| File | Role |
| --- | --- |
| `render_example.py` | run one example, write `<out>/<rel dir>/<stem>/{<stem>.md, <stem>.py, images/, meta.json}` |
| `render_all_examples.py` | render a whole tree in parallel (one process per example), with a cache, then build the indexes |
| `build_example_index.py` | write an `index.md` (cards + hidden toctree) in every folder, at any depth |
| `convert_sg_example.py` | convert sphinx-gallery scripts to the new format |
| `demo/` | small examples in the new format (3 nesting levels, figures, animation, tags) |
| `proto_site/` | minimal Sphinx project to preview the result |

## Example format

Plain Python scripts in the [jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format),
so they still run with `python` or `./shamrock --rscript`, and editors (VS Code,
PyCharm, Spyder) can run them cell by cell:

```python
# %% [markdown]
# # Title (becomes the page title)
#
# MyST Markdown: {math}`\rho`, {py:class}`shamrock.Context`, directives, `$$ ... $$`.
#
# ## Any heading depth

# %%
import shamrock

# %% [rst]
# A cell kept in reStructuredText (rendered through ``{eval-rst}``).

# %% tags=["hide-input"]
plt.show()
```

- Only files named `run_*.py` are executed (same rule as the current
  `filename_pattern`); others are rendered without outputs. Override with
  `--exec always|never`.
- Captured per cell: stdout at the file descriptor level (Python *and* C++
  output, still echoed live), the repr of the last expression (`_repr_html_`
  or `repr`, suppressed by a trailing `;`), matplotlib figures, matplotlib
  animations stored in a variable (mp4 with ffmpeg, HTML widget otherwise).
- Cell tags: `hide-input`, `hide-output` (collapsed dropdown), `remove-input`,
  `remove-output`, `remove-cell`, `skip-execution`, `thumbnail`, `full-output`.
- Config comment at the top of a code cell: `# shamdoc_thumbnail_number = 2`.
- The script runs with its own folder as working directory (like sphinx-gallery).
- A failing example still produces its page (traceback shown, warning at the
  top, `status: failed` in `meta.json`) and the runner exits with code 1.

## Testing locally

Dependencies of the doc environment (`pyyaml` comes with `myst-parser`):

```bash
pip install sphinx myst-parser sphinx-design sphinxcontrib-video sphinx-copybutton pydata-sphinx-theme matplotlib numpy
```

### 1. The demo (no shamrock needed)

```bash
cd doc/sphinx/example_renderer
python render_all_examples.py --src demo --out proto_site/examples_gen -j 4
sphinx-build -b html proto_site proto_site/_build/html
```

Open `proto_site/_build/html/index.html`. Run the first command again: every
example is reported as `cached`.

### 2. The real examples

Convert a copy of the examples tree (the originals are left untouched, data
files and `GALLERY_HEADER.rst` are copied too):

```bash
python convert_sg_example.py ../../../examples -o /tmp/examples_new
```

Render them with shamrock importable from Python (e.g. the `PYTHONPATH` /
`LD_LIBRARY_PATH` set by `env/helpers/sphinx.sh`)...

```bash
python render_all_examples.py --src /tmp/examples_new --out proto_site/examples_gen \
    --exclude 'TO_MIGRATE/*' --pattern '^sph/' -j 2 --title Examples
```

... or through the shamrock executable, e.g. with 2 MPI ranks per example
(options go through `SHAMDOC_*` environment variables since shamrock owns the
command line; only rank 0 writes):

```bash
python render_all_examples.py --src /tmp/examples_new --out proto_site/examples_gen \
    --exclude 'TO_MIGRATE/*' --pattern '^sph/' -j 1 --title Examples \
    --launcher "mpirun -n 2 /path/to/build/shamrock --sycl-cfg 0:0 --loglevel 1 --rscript"
```

Then `sphinx-build -b html proto_site proto_site/_build/html`. Per-example logs
are in `proto_site/examples_gen/_logs/`. Use `--cache-key "$(git rev-parse HEAD)"`
to invalidate the cache when shamrock changes, `--force` to rerun everything.

A single example can also be rendered directly:

```bash
python render_example.py --example /tmp/examples_new/sph/run_kill_particle_sphere.py \
    --root /tmp/examples_new --out proto_site/examples_gen
python build_example_index.py --gen proto_site/examples_gen --src /tmp/examples_new
```

## Section titles, intro and ordering

Taken from the source tree, per folder, by `build_example_index.py`:

- `_section.md` (preferred): optional front matter (`order: [...]`,
  `description: ...` shown on the parent's card, `title: ...`), then an H1 and
  Markdown intro.
- `GALLERY_HEADER.rst` (legacy fallback): its title is used and the rest is
  wrapped in `{eval-rst}`.

Sub-folders become sub-sections at any depth. Every page has a stable label,
`ex-<path>-<stem>` for examples and `ex-section-<path>` for sections, so
`` {ref}`ex-sph-run_kill_particle_sphere` `` keeps working wherever the page ends up.

## CI sketch (not wired yet)

```yaml
# matrix job: one example, no Sphinx needed
- run: |
    SHAMDOC_EXAMPLE=examples/${{ matrix.testfile }} SHAMDOC_ROOT=examples SHAMDOC_OUT=out \
      mpirun -n ${{ matrix.worldsize }} ./shamrock --sycl-cfg 0:0 --rscript \
      doc/sphinx/example_renderer/render_example.py
- uses: actions/upload-artifact@v7
  with: { name: "example-${{ matrix.artifact_name }}", path: out }

# main job: merge, index, build once
- uses: actions/download-artifact@v8
  with: { pattern: "example-*", merge-multiple: true, path: doc/sphinx/source/examples_gen }
- run: |
    # examples without an artifact (not run_*, or not in the matrix) rendered without execution
    python doc/sphinx/example_renderer/render_all_examples.py --src examples \
      --out doc/sphinx/source/examples_gen --exec never --missing-only --no-index
    python doc/sphinx/example_renderer/build_example_index.py \
      --gen doc/sphinx/source/examples_gen --src examples
    sphinx-build ...
```

## Integrating into the main docs (next step)

In `doc/sphinx/source/conf.py`: add `sphinxcontrib.video`, the CSS from
`proto_site/_static/`, and `examples_gen/index` in a toctree of `index.rst`;
then drop `sphinx_gallery.gen_gallery` once all examples are converted.
The `sphx_glr_*` references (API templates, `user_guide/quickstart.md`) need
updating; `meta.json` lists the `shamrock.*` names each example uses, which can
rebuild the "examples using X" mini-galleries.

## Known limitations of the prototype

- stderr is not captured (only echoed).
- Animations are only detected when assigned to a variable of the script.
- No `.ipynb` download (jupytext could generate one from the source).
