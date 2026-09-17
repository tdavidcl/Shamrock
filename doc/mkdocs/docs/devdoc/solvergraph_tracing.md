# SolverGraph execution tracing

The solvergraph can trace its own execution: node & edge lifetimes, node
topology (edge bindings) and node evaluations are recorded to an append-only
JSON lines file (one file per MPI rank). The trace can then be visualized
live, or replayed with a speed factor, in the viewer app located in
`tools/solvergraph_viewer` (a standalone Python app whose dependencies are
**not** part of the Shamrock build).

## Enabling tracing

From the environment:

```bash
export SHAM_SOLVERGRAPH_TRACE=1
# optional, default "shamrock_sgtrace"
export SHAM_SOLVERGRAPH_TRACE_PREFIX=mytrace
./shamrock --sycl-cfg 0:0 --rscript run.py
```

Or programmatically from a runscript:

```python
import shamrock
shamrock.enable_solvergraph_tracing()          # env prefix or default
shamrock.enable_solvergraph_tracing("mytrace") # explicit prefix
...
shamrock.disable_solvergraph_tracing()         # flush & stop
```

The rank `r` trace is written to `<prefix>_<r>.jsonl`.

Nodes and edges created *before* tracing was enabled emit no creation record,
but their evaluations are still traced; the viewer tolerates events referring
to unknown UUIDs.

## Design

The hooks are the `LifetimeTracker` callbacks
(`src/shamsolvergraph/include/shamsolvergraph/LifetimeTracker.hpp`): `INode`
and `IEdge` hold a `std::shared_ptr<LifetimeTracker<T>>` member which notifies
creation (constructor), destruction (destructor), state updates (edge
rebinding) and operations (evaluate begin/end) through static function
pointers. Holding the tracker as a shared pointer member (instead of
inheriting it) makes the notifications move-safe: a moved-from object carries
a null tracker and cannot emit a duplicate destroy notification.

When tracing is disabled (default) the callbacks are null and each hook site
costs a single null pointer check — no allocation, no branch misprediction in
practice, nothing else.

When tracing is enabled (`shamrock::solvergraph::tracing::enable()`), the
callbacks record:

- **hot path** (create/destroy/evaluate): fixed size POD events
  `{seq, wtime, kind, uuid}` appended to a preallocated buffer;
- **cold path** (`node_update`, which only fires when edges are bound, i.e. at
  graph build/rewire time): labels, type names and edge lists, pre-serialized
  to JSON.

Buffers are formatted (with `fmt`) and written at *flush safe points*: at the
end of a depth-0 evaluation (never in the middle of a node evaluation), on
buffer overflow, on `disable()`, and at program exit. The file is flushed
after each write so it can be tailed by a live viewer. Flush cost for a ~50
node / ~300 edge graph is tens of microseconds per step.

Timestamps use `shambase::details::get_wtime()`; the header record carries a
`clock_origin` captured at the MPI barrier of world info fetching (the same
offset used by the chrome profiler), so `wtime - clock_origin` is aligned
across ranks.

## Trace format (version 1)

One JSON object per line. Every record except the header carries a `seq`
(monotonic counter giving a total order within the file) and a raw `wtime`.

| kind | fields | meaning |
| ---- | ------ | ------- |
| `header` | `version`, `rank`, `clock_origin` | first line of the file |
| `node_create` / `node_destroy` | `seq`, `wtime`, `uuid` | node lifetime |
| `edge_create` / `edge_destroy` | `seq`, `wtime`, `uuid` | edge lifetime |
| `node_update` | `seq`, `wtime`, `uuid`, `label`, `type`, `ro_edges`, `rw_edges` | topology: the node's read-only / read-write edge bindings (`{uuid, label}` lists) |
| `node_evaluate_begin` / `node_evaluate_end` | `seq`, `wtime`, `uuid` | node evaluation span |

Note: node UUIDs and edge UUIDs are **separate namespaces** (separate
counters), so a node and an edge may share the same numeric UUID.

Edge *usage* is not recorded per-access: consumers infer that an edge is read
(resp. written) at time `t` when a node whose `ro_edges` (resp. `rw_edges`)
list contains it is being evaluated at `t`.

### Reserved kinds (not emitted yet)

Consumers must ignore record kinds they do not know, so the format can grow:

- `edge_access` — reserved for explicit per-access edge events, if inferred
  usage ever turns out to be too coarse.
- `edge_data` — reserved for edge payload previews (e.g. a field snapshot to
  be displayed as a texture in the viewer, Blender-style): `uuid`, `wtime`, a
  `format` tag and a `blob` reference `{"file", "offset", "len"}` into a
  per-rank binary sidecar file (`<prefix>_<r>_blobs.bin`), keeping bulky
  payloads out of the JSON stream. The viewer already parses these records
  and exposes a preview renderer registry (`solvergraph_viewer/preview.py`).

## Visualizing traces

See `tools/solvergraph_viewer/README.md`. Quick start:

```bash
pip install ./tools/solvergraph_viewer
solvergraph-viewer shamrock_sgtrace_*.jsonl            # replay
solvergraph-viewer shamrock_sgtrace_*.jsonl --speed 0.1 # 1/10 real time
solvergraph-viewer shamrock_sgtrace_*.jsonl --live      # tail a running sim
```
