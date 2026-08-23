"""Graph model rebuilt from a solvergraph trace.

The model ingests trace records (from :mod:`.trace_reader`) and keeps the full
history: node/edge lifetimes, node topology (ro/rw edge bindings) and node
evaluation intervals. Queries by trace time then answer "what does the graph
look like at time t" for playback and scrubbing.

Edge usage is *inferred*: an edge is considered read (resp. written) at time t
when some node whose ro (resp. rw) list contains it is being evaluated at t.

Node UUIDs and edge UUIDs are separate namespaces (they come from separate
counters on the producer side), hence the separate dicts.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .trace_reader import TraceRecord

# intervals of one node only overlap through recursion (e.g. loop nodes); a
# self-recursion deeper than this is not expected
_MAX_NESTING = 64


@dataclass
class EvalIntervals:
    """Evaluation intervals of one node.

    ``starts``/``ends`` are aligned and sorted by start time (events arrive in
    order); an open interval (evaluation still running, live mode) has an end
    of +inf. Closed intervals close in increasing time order, so the
    ``closed_at``/``closed_cum`` prefix sums stay append-only and sorted.
    """

    starts: list[float] = field(default_factory=list)
    ends: list[float] = field(default_factory=list)
    _open_stack: list[int] = field(default_factory=list)
    closed_at: list[float] = field(default_factory=list)
    closed_cum: list[float] = field(default_factory=list)

    def begin(self, t: float) -> None:
        self.starts.append(t)
        self.ends.append(float("inf"))
        self._open_stack.append(len(self.starts) - 1)

    def end(self, t: float) -> None:
        if not self._open_stack:
            return  # tolerate an end without begin (trace started mid-run)
        i = self._open_stack.pop()
        self.ends[i] = t
        prev = self.closed_cum[-1] if self.closed_cum else 0.0
        self.closed_at.append(t)
        self.closed_cum.append(prev + (t - self.starts[i]))

    def is_active(self, t: float, slack: float = 0.0) -> bool:
        """Is some evaluation interval containing time t ?

        With a nonzero ``slack``, intervals that ended less than ``slack`` ago
        also count: evaluations are typically much shorter than a display
        frame, so viewers use the slack as an afterglow to keep recently
        evaluated nodes visibly highlighted.
        """
        i = bisect.bisect_right(self.starts, t)
        lo = max(0, i - _MAX_NESTING)
        if any(self.ends[k] >= t for k in range(lo, i)):
            return True
        if slack > 0.0:
            # some interval ended within [t - slack, t] ?
            j = bisect.bisect_right(self.closed_at, t)
            if j > 0 and self.closed_at[j - 1] >= t - slack:
                return True
        return False

    def count_before(self, t: float) -> int:
        """Number of evaluations started at or before time t."""
        return bisect.bisect_right(self.starts, t)

    def cumulative_time(self, t: float) -> float:
        """Total evaluation time of intervals fully closed at time t."""
        j = bisect.bisect_right(self.closed_at, t)
        return self.closed_cum[j - 1] if j > 0 else 0.0


@dataclass
class NodeState:
    """History of one solver node."""

    uuid: int
    label: str = ""
    type_name: str = ""
    ro_edges: list[tuple[int, str]] = field(default_factory=list)
    rw_edges: list[tuple[int, str]] = field(default_factory=list)
    created_t: float | None = None
    destroyed_t: float | None = None
    evals: EvalIntervals = field(default_factory=EvalIntervals)

    def exists_at(self, t: float) -> bool:
        if self.created_t is not None and t < self.created_t:
            return False
        return not (self.destroyed_t is not None and t >= self.destroyed_t)

    def is_active(self, t: float, slack: float = 0.0) -> bool:
        return self.evals.is_active(t, slack)


@dataclass
class EdgeState:
    """History of one data edge."""

    uuid: int
    label: str = ""
    created_t: float | None = None
    destroyed_t: float | None = None
    # reserved for the future edge_data records: list of (t, format, blob ref)
    previews: list[tuple[float, str, dict]] = field(default_factory=list)

    def exists_at(self, t: float) -> bool:
        if self.created_t is not None and t < self.created_t:
            return False
        return not (self.destroyed_t is not None and t >= self.destroyed_t)


class GraphModel:
    """Full-history graph model of one rank's trace."""

    def __init__(self, clock_origin: float = 0.0):
        self.clock_origin = clock_origin
        self.nodes: dict[int, NodeState] = {}
        self.edges: dict[int, EdgeState] = {}
        self.t_min: float = float("inf")
        self.t_max: float = float("-inf")
        self.unknown_kinds: set[str] = set()

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #

    def _t(self, record: TraceRecord) -> float:
        t = record.wtime - self.clock_origin
        self.t_min = min(self.t_min, t)
        self.t_max = max(self.t_max, t)
        return t

    def _node(self, uuid: int) -> NodeState:
        if uuid not in self.nodes:
            self.nodes[uuid] = NodeState(uuid=uuid)
        return self.nodes[uuid]

    def _edge(self, uuid: int) -> EdgeState:
        if uuid not in self.edges:
            self.edges[uuid] = EdgeState(uuid=uuid)
        return self.edges[uuid]

    def apply(self, record: TraceRecord) -> None:
        """Ingest one trace record."""
        kind = record.kind
        t = self._t(record)

        if kind == "node_create":
            self._node(record.uuid).created_t = t
        elif kind == "node_destroy":
            self._node(record.uuid).destroyed_t = t
        elif kind == "edge_create":
            self._edge(record.uuid).created_t = t
        elif kind == "edge_destroy":
            self._edge(record.uuid).destroyed_t = t
        elif kind == "node_evaluate_begin":
            self._node(record.uuid).evals.begin(t)
        elif kind == "node_evaluate_end":
            self._node(record.uuid).evals.end(t)
        elif kind == "node_update":
            node = self._node(record.uuid)
            node.label = record.raw.get("label", node.label)
            node.type_name = record.raw.get("type", node.type_name)
            node.ro_edges = [
                (e["uuid"], e.get("label", "")) for e in record.raw.get("ro_edges", [])
            ]
            node.rw_edges = [
                (e["uuid"], e.get("label", "")) for e in record.raw.get("rw_edges", [])
            ]
            for uuid, label in node.ro_edges + node.rw_edges:
                edge = self._edge(uuid)
                if label:
                    edge.label = label
        elif kind == "edge_data":
            # reserved kind: payload previews (e.g. field snapshots rendered
            # as textures). Recorded so preview plugins can render them.
            edge = self._edge(record.uuid)
            edge.previews.append((t, record.raw.get("format", ""), record.raw.get("blob", {})))
        else:
            self.unknown_kinds.add(kind)

    def apply_all(self, records) -> None:
        for record in records:
            self.apply(record)

    # ------------------------------------------------------------------ #
    # queries at a given playback time
    # ------------------------------------------------------------------ #

    def active_nodes(self, t: float, slack: float = 0.0) -> set[int]:
        """UUIDs of the nodes being evaluated at time t (or within the
        ``slack`` afterglow window before t)."""
        return {u for u, n in self.nodes.items() if n.exists_at(t) and n.is_active(t, slack)}

    def edge_activity(self, t: float, slack: float = 0.0) -> tuple[set[int], set[int]]:
        """Inferred (read edge uuids, written edge uuids) at time t."""
        reads: set[int] = set()
        writes: set[int] = set()
        for uuid in self.active_nodes(t, slack):
            node = self.nodes[uuid]
            reads.update(u for u, _ in node.ro_edges)
            writes.update(u for u, _ in node.rw_edges)
        return reads, writes

    def time_span(self) -> tuple[float, float]:
        if self.t_min > self.t_max:
            return (0.0, 0.0)
        return (self.t_min, self.t_max)
