"""Headless tests of the viewer logic (reader, model, layout, playback).

Run with: python -m unittest discover tools/solvergraph_viewer/tests
No GUI (dearpygui) required.
"""

import inspect
import itertools
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solvergraph_viewer.layout import (
    build_display_graph,
    force_directed_layout,
    layered_layout,
)
from solvergraph_viewer.model import GraphModel
from solvergraph_viewer.playback import PlaybackClock
from solvergraph_viewer.trace_reader import TraceReader


def write_lines(path, records, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def synthetic_trace():
    """A small synthetic trace: a sequence evaluating two nodes sharing an edge."""
    seq = iter(range(1000))
    t = iter([x * 0.01 for x in range(1000)])
    records = [
        {"kind": "header", "version": 1, "rank": 0, "clock_origin": 100.0},
        {"kind": "edge_create", "seq": next(seq), "wtime": 100.0 + next(t), "uuid": 7},
        {"kind": "node_create", "seq": next(seq), "wtime": 100.0 + next(t), "uuid": 1},
        {"kind": "node_create", "seq": next(seq), "wtime": 100.0 + next(t), "uuid": 2},
        {"kind": "node_create", "seq": next(seq), "wtime": 100.0 + next(t), "uuid": 3},
        {
            "kind": "node_update",
            "seq": next(seq),
            "wtime": 100.0 + next(t),
            "uuid": 1,
            "label": "reader",
            "type": "T1",
            "ro_edges": [{"uuid": 7, "label": "field"}],
            "rw_edges": [],
        },
        {
            "kind": "node_update",
            "seq": next(seq),
            "wtime": 100.0 + next(t),
            "uuid": 2,
            "label": "writer",
            "type": "T2",
            "ro_edges": [],
            "rw_edges": [{"uuid": 7, "label": "field"}],
        },
        # sequence node 3 evaluates node 1 then node 2
        {"kind": "node_evaluate_begin", "seq": next(seq), "wtime": 100.06, "uuid": 3},
        {"kind": "node_evaluate_begin", "seq": next(seq), "wtime": 100.07, "uuid": 1},
        {"kind": "node_evaluate_end", "seq": next(seq), "wtime": 100.08, "uuid": 1},
        {"kind": "node_evaluate_begin", "seq": next(seq), "wtime": 100.09, "uuid": 2},
        {"kind": "node_evaluate_end", "seq": next(seq), "wtime": 100.10, "uuid": 2},
        {"kind": "node_evaluate_end", "seq": next(seq), "wtime": 100.11, "uuid": 3},
        {"kind": "node_destroy", "seq": next(seq), "wtime": 100.20, "uuid": 1},
        # unknown kinds must be tolerated
        {"kind": "made_up_kind", "seq": next(seq), "wtime": 100.21, "uuid": 1},
    ]
    return records


class TestTraceReader(unittest.TestCase):
    def test_incremental_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace_0.jsonl")
            records = synthetic_trace()
            write_lines(path, records[:5])

            reader = TraceReader(path)
            got = reader.poll()
            self.assertEqual(reader.version, 1)
            self.assertEqual(reader.rank, 0)
            self.assertEqual(reader.clock_origin, 100.0)
            self.assertEqual(len(got), 4)  # header consumed separately

            # append the rest, including a torn partial line
            write_lines(path, records[5:-1], mode="a")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(records[-1])[:10])  # partial line, no newline

            got2 = reader.poll()
            self.assertEqual(len(got2), len(records) - 5 - 1)

            # complete the partial line
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(records[-1])[10:] + "\n")
            got3 = reader.poll()
            self.assertEqual(len(got3), 1)
            self.assertEqual(got3[0].kind, "made_up_kind")


class TestGraphModel(unittest.TestCase):
    def build_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace_0.jsonl")
            write_lines(path, synthetic_trace())
            reader = TraceReader(path)
            records = reader.poll()
            model = GraphModel(clock_origin=reader.clock_origin)
            model.apply_all(records)
            return model

    def test_lifetimes_and_topology(self):
        model = self.build_model()
        self.assertIn(1, model.nodes)
        self.assertIn(7, model.edges)
        self.assertEqual(model.nodes[1].label, "reader")
        self.assertEqual(model.nodes[1].ro_edges, [(7, "field")])
        self.assertEqual(model.nodes[2].rw_edges, [(7, "field")])
        self.assertEqual(model.edges[7].label, "field")
        # node 1 destroyed at t=0.20
        self.assertTrue(model.nodes[1].exists_at(0.1))
        self.assertFalse(model.nodes[1].exists_at(0.3))
        # unknown kinds are recorded, not fatal
        self.assertIn("made_up_kind", model.unknown_kinds)

    def test_activity_and_inferred_edge_use(self):
        model = self.build_model()
        # at t=0.075 node 1 (and enclosing sequence 3) are evaluating
        active = model.active_nodes(0.075)
        self.assertIn(1, active)
        self.assertIn(3, active)
        self.assertNotIn(2, active)
        reads, writes = model.edge_activity(0.075)
        self.assertIn(7, reads)
        self.assertNotIn(7, writes)
        # at t=0.095 node 2 is evaluating: edge 7 is written
        reads, writes = model.edge_activity(0.095)
        self.assertIn(7, writes)
        # outside evaluations nothing is active
        self.assertEqual(model.active_nodes(0.15), set())

    def test_stats(self):
        model = self.build_model()
        node = model.nodes[1]
        self.assertEqual(node.evals.count_before(0.5), 1)
        self.assertEqual(node.evals.count_before(0.05), 0)
        self.assertAlmostEqual(node.evals.cumulative_time(0.5), 0.01, places=6)
        self.assertAlmostEqual(node.evals.duration_at(0.075), 0.01, places=6)
        self.assertAlmostEqual(node.evals.duration_at(0.05), 0.0, places=6)
        self.assertAlmostEqual(node.evals.duration_at(0.5), 0.01, places=6)

    def test_min_duration_highlight(self):
        model = self.build_model()
        # node 1 evaluates on [0.07, 0.08)
        self.assertIn(1, model.active_nodes(0.075))
        # must not light before start, even with a large min duration
        self.assertNotIn(1, model.active_nodes(0.069, min_duration=1.0))
        # exact interval: off once the eval has ended
        self.assertNotIn(1, model.active_nodes(0.081, min_duration=0.0))
        # min_duration=0.02 holds from 0.07 to 0.09
        self.assertIn(1, model.active_nodes(0.085, min_duration=0.02))
        self.assertNotIn(1, model.active_nodes(0.10, min_duration=0.02))
        reads, _writes = model.edge_activity(0.085, min_duration=0.02)
        self.assertIn(7, reads)

    def test_nested_intervals(self):
        model = GraphModel()
        from solvergraph_viewer.trace_reader import TraceRecord

        # recursion of the same node: [0, 10] containing [2, 4]
        for kind, wtime in [
            ("node_evaluate_begin", 0.0),
            ("node_evaluate_begin", 2.0),
            ("node_evaluate_end", 4.0),
            ("node_evaluate_end", 10.0),
        ]:
            model.apply(TraceRecord(kind=kind, raw={"kind": kind, "wtime": wtime, "uuid": 5}))
        node = model.nodes[5]
        self.assertTrue(node.is_active(1.0))
        self.assertTrue(node.is_active(3.0))
        self.assertTrue(node.is_active(5.0))
        self.assertFalse(node.is_active(11.0))


class TestLayout(unittest.TestCase):
    def test_layers_follow_dataflow(self):
        model = TestGraphModel().build_model()
        items, arcs = build_display_graph(model)
        positions = layered_layout(items, arcs)
        # every displayed item has a position
        for it in items:
            self.assertIn(it, positions)
        # edge 7 is read by node 1: the edge box is on an earlier layer
        self.assertLess(positions[("edge", 7)][0], positions[("node", 1)][0])
        # node 2 writes edge 7... which node 1 reads; cycle-free here:
        # node2 -> edge7 -> node1 means node 2 is before edge 7
        self.assertLess(positions[("node", 2)][0], positions[("edge", 7)][0])

    def test_cycle_tolerance(self):
        items = [("node", 1), ("edge", 1)]
        arcs = [(("node", 1), ("edge", 1)), (("edge", 1), ("node", 1))]
        positions = layered_layout(items, arcs)
        self.assertEqual(len(positions), 2)


class TestForceDirectedLayout(unittest.TestCase):
    def test_every_item_placed(self):
        model = TestGraphModel().build_model()
        items, arcs = build_display_graph(model)
        positions = force_directed_layout(items, arcs)
        for it in items:
            self.assertIn(it, positions)

    def test_deterministic(self):
        model = TestGraphModel().build_model()
        items, arcs = build_display_graph(model)
        first = force_directed_layout(items, arcs)
        second = force_directed_layout(items, arcs)
        self.assertEqual(first, second)

    def test_empty_graph(self):
        self.assertEqual(force_directed_layout([], []), {})

    def test_isolated_item(self):
        items = [("node", 1)]
        positions = force_directed_layout(items, [])
        self.assertIn(("node", 1), positions)

    def test_links_flow_left_to_right(self):
        # node 2 --rw--> edge 7 --ro--> node 1: each arc's source (drawn from
        # an output/right pin) should end up left of its destination
        # (drawn from an input/left pin), so links don't loop backward.
        model = TestGraphModel().build_model()
        items, arcs = build_display_graph(model)
        positions = force_directed_layout(items, arcs)
        self.assertLess(positions[("edge", 7)][0], positions[("node", 1)][0])
        self.assertLess(positions[("node", 2)][0], positions[("edge", 7)][0])

    def test_no_overlap_on_hub_and_disconnected_items(self):
        # a hub with many leaves plus a separate disconnected chain and some
        # fully isolated items: nothing here pulls the two components or the
        # isolated items toward each other, so only the overlap-removal pass
        # keeps every pair at least `spacing` apart.
        spacing = inspect.signature(force_directed_layout).parameters["spacing"].default
        items = [("node", i) for i in range(30)]
        arcs = [(("node", 0), ("node", i)) for i in range(1, 12)]
        arcs += [(("node", i - 1), ("node", i)) for i in range(13, 25)]
        # nodes 25..29 stay fully isolated
        positions = force_directed_layout(items, arcs)
        min_dist = min(
            math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1])
            for a, b in itertools.combinations(items, 2)
        )
        self.assertGreaterEqual(min_dist, spacing - 1.0)


class TestPlayback(unittest.TestCase):
    def test_speed_factor(self):
        clock = PlaybackClock(speed=10.0)
        clock.set_span(0.0, 100.0)
        clock.playing = True
        clock.advance(1.0)
        self.assertAlmostEqual(clock.t, 10.0)
        clock.speed = 0.1
        clock.advance(1.0)
        self.assertAlmostEqual(clock.t, 10.1)

    def test_clamp_and_stop(self):
        clock = PlaybackClock(speed=1.0)
        clock.set_span(0.0, 1.0)
        clock.playing = True
        clock.advance(5.0)
        self.assertEqual(clock.t, 1.0)
        self.assertFalse(clock.playing)

    def test_follow(self):
        clock = PlaybackClock(follow=True)
        clock.set_span(0.0, 42.0)
        clock.advance(0.1)
        self.assertEqual(clock.t, 42.0)


if __name__ == "__main__":
    unittest.main()
