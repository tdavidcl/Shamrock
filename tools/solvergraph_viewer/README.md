# Shamrock solvergraph viewer

GUI viewer for Shamrock solvergraph execution traces: visualize node
evaluation and edge dataflow, live during a run or replayed afterwards with a
speed factor, in a node editor where boxes can be dragged around freely
(Blender node editor style).

This app is standalone on purpose: its dependencies (Dear PyGui) are **not**
dependencies of the Shamrock build.

## Install

```bash
pip install ./tools/solvergraph_viewer
```

or run it in place (only needs `pip install dearpygui`):

```bash
python -m solvergraph_viewer <trace files>
```

## Producing traces

Run Shamrock with `SHAM_SOLVERGRAPH_TRACE=1` (optionally
`SHAM_SOLVERGRAPH_TRACE_PREFIX=<prefix>`), or call
`shamrock.enable_solvergraph_tracing()` from a runscript. Each MPI rank
writes `<prefix>_<rank>.jsonl` (default prefix `shamrock_sgtrace`). See
`doc/mkdocs/docs/devdoc/solvergraph_tracing.md` for the trace format.

## Usage

```bash
# replay a finished run (all ranks; pick the rank in the GUI)
solvergraph-viewer shamrock_sgtrace_*.jsonl

# replay at 1/10 of real time
solvergraph-viewer shamrock_sgtrace_*.jsonl --speed 0.1

# live view of a running simulation (tails the growing files)
solvergraph-viewer shamrock_sgtrace_*.jsonl --live
```

In the GUI:

- **drag boxes** to rearrange the graph (initial positions come from an
  automatic layered layout, `Re-layout` recomputes them);
- solver nodes are shown as blue-titled boxes, data edges as green-titled
  boxes; arcs are green for read-only bindings, red for read-write (matching
  the DOT export colors);
- during playback the node currently being evaluated lights up orange, and
  the edges it reads/writes pulse green/red (usage is inferred from the
  topology and the evaluation intervals);
- each node overlays its evaluation count and cumulated evaluation time at
  the current playback time;
- the **time slider** scrubs through the trace, **Play/Pause** + the speed
  slider replay it (0.1x to 100x real time), **Live (follow)** sticks to the
  end of a growing trace;
- clicking a box shows its details (type, bindings, stats) in the side panel;
- **zoom** the graph with the mouse wheel over the canvas (or `+` / `-` /
  `0` while hovered, or the side-panel buttons). This spreads or packs node
  positions around the cursor — stock Dear PyGui cannot scale the node
  editor itself, so box sizes stay constant. `Re-layout` resets zoom.

## Edge payload previews (future)

The trace format reserves an `edge_data` record kind for edge payload
snapshots (e.g. rendering a field as a texture inside the graph). The viewer
already parses those records; to display them, register a renderer for the
format tag in `solvergraph_viewer/preview.py` (`default_registry.register`).
No Shamrock producer emits them yet.

## Development

Logic (trace reader, graph model, layout, playback) is GUI-free and tested
headlessly:

```bash
python -m unittest discover tools/solvergraph_viewer/tests
```
