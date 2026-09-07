"""Automatic initial layout of the solvergraph.

Nodes and data edges are both displayed as boxes in the node editor (matching
the DOT export of the solvergraph, where data edges are rectangles). The
displayed graph has an arc edge->node for each ro binding and node->edge for
each rw binding.

The layout is a simple layered (Sugiyama-like) placement: longest-path
layering followed by a few barycenter ordering sweeps to reduce crossings.
Users can then drag boxes around freely; this only provides the initial
positions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import networkx as nx

from .model import GraphModel

# displayed graph item ids: ("node", uuid) or ("edge", uuid) since node and
# edge uuids are separate namespaces
Item = tuple[str, int]


def build_display_graph(
    model: GraphModel,
) -> tuple[list[Item], list[tuple[Item, Item]]]:
    """Build the displayed items and arcs from the model topology."""
    items: list[Item] = []
    arcs: list[tuple[Item, Item]] = []

    for uuid in model.nodes:
        items.append(("node", uuid))
    for uuid in model.edges:
        items.append(("edge", uuid))

    for uuid, node in model.nodes.items():
        for e_uuid, _ in node.ro_edges:
            arcs.append((("edge", e_uuid), ("node", uuid)))
        for e_uuid, _ in node.rw_edges:
            arcs.append((("node", uuid), ("edge", e_uuid)))

    return items, arcs


def layered_layout(
    items: Iterable[Item],
    arcs: Iterable[tuple[Item, Item]],
    dx: float = 260.0,
    dy: float = 120.0,
) -> dict[Item, tuple[float, float]]:
    """Compute initial positions with a layered layout.

    Handles cycles gracefully (relaxation is capped), and places disconnected
    items on layer 0.
    """
    items = list(items)
    arcs = list(arcs)
    layer: dict[Item, int] = {it: 0 for it in items}

    # longest-path layering by capped relaxation (tolerates cycles)
    for _ in range(min(len(items), 100)):
        changed = False
        for src, dst in arcs:
            if src in layer and dst in layer and layer[dst] < layer[src] + 1:
                layer[dst] = layer[src] + 1
                changed = True
                layer[dst] = min(layer[dst], len(items))  # cycle guard
        if not changed:
            break

    # group by layer
    layers: dict[int, list[Item]] = {}
    for it in items:
        layers.setdefault(layer[it], []).append(it)
    ordered_layers = [layers[k] for k in sorted(layers)]

    # neighbor map for barycenter sweeps
    neighbors: dict[Item, list[Item]] = {it: [] for it in items}
    for src, dst in arcs:
        if src in neighbors and dst in neighbors:
            neighbors[src].append(dst)
            neighbors[dst].append(src)

    index: dict[Item, int] = {}
    for row in ordered_layers:
        for i, it in enumerate(row):
            index[it] = i

    for _ in range(4):  # a few barycenter ordering sweeps
        for row in ordered_layers:

            def barycenter(it: Item) -> float:
                ns = neighbors[it]
                if not ns:
                    return float(index[it])
                return sum(index[n] for n in ns) / len(ns)

            row.sort(key=barycenter)
            for i, it in enumerate(row):
                index[it] = i

    positions: dict[Item, tuple[float, float]] = {}
    for li, row in enumerate(ordered_layers):
        for i, it in enumerate(row):
            positions[it] = (li * dx, i * dy)
    return positions


def _relax_positions(
    positions: dict[Item, tuple[float, float]],
    arcs: list[tuple[Item, Item]],
    min_gap: float,
    min_dist: float,
    iterations: int = 300,
) -> None:
    """Jointly enforce left-to-right arc ordering and pairwise no-overlap.

    Two corrections run every iteration, in the same pass, so each reacts to
    what the other just did instead of undoing it:

    - direction: every displayed link is drawn from its source's output pin
      (right edge of the box) to its destination's input pin (left edge of
      the box): ro bindings as edge->node, rw as node->edge (see
      build_display_graph). Undirected spring forces don't preserve that
      left/right order, so this nudges src.x + min_gap <= dst.x wherever the
      arc graph isn't outright cyclic (cycles settle on a compromise gap).
    - overlap: the spring relaxation only pulls *connected* items toward an
      equilibrium distance, so two unconnected items (or whole disconnected
      components) can still end up overlapping. This treats every pair as a
      hard-disk collision and separates any pair closer than min_dist.

    Running direction correction first and overlap correction after in the
    same loop, rather than as two separate passes, means an overlap fix that
    nudges an arc endpoint gets immediately re-checked against its arc
    partner on the next iteration instead of only after the fact.
    """
    items = list(positions)
    for _ in range(iterations):
        moved = False

        for src, dst in arcs:
            sx, sy = positions[src]
            dx, dy = positions[dst]
            gap = dx - sx
            if gap < min_gap:
                shortfall = (min_gap - gap) * 0.25
                positions[src] = (sx - shortfall, sy)
                positions[dst] = (dx + shortfall, dy)
                moved = True

        for i, a in enumerate(items):
            ax, ay = positions[a]
            for b in items[i + 1 :]:
                bx, by = positions[b]
                dx, dy = bx - ax, by - ay
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    if dist < 1e-6:
                        dx, dy, dist = 1.0, 0.0, 1.0
                    shortfall = (min_dist - dist) * 0.5
                    ux, uy = dx / dist, dy / dist
                    ax, ay = ax - ux * shortfall, ay - uy * shortfall
                    bx, by = bx + ux * shortfall, by + uy * shortfall
                    positions[b] = (bx, by)
                    moved = True
            positions[a] = (ax, ay)

        if not moved:
            break


def force_directed_layout(
    items: Iterable[Item],
    arcs: Iterable[tuple[Item, Item]],
    spacing: float = 100.0,
    iterations: int = 100,
    seed: int = 0,
) -> dict[Item, tuple[float, float]]:
    """Force-directed placement: connected items attract, all items repel.

    Edge attraction is weighted down by 1/sqrt(deg(u)*deg(v)) so items with
    many connections don't drag all their neighbors into one overlapping
    clump: a hub and a leaf still keep roughly `spacing` apart, but two hubs
    pull on each other much more weakly than two leaves would.

    Seeded from layered_layout() positions for fast, stable convergence
    instead of networkx's default random initialization. A final relaxation
    pass restores left-to-right arc ordering and guarantees every pair of
    items ends up at least `spacing` apart, since the springs alone only
    pull *connected* items toward that distance (see _relax_positions).
    """
    items = list(items)
    arcs = list(arcs)
    if not items:
        return {}

    graph = nx.Graph()
    graph.add_nodes_from(items)
    graph.add_edges_from(arcs)

    degree = dict(graph.degree())
    for u, v in graph.edges():
        graph.edges[u, v]["weight"] = 1.0 / (max(degree[u], 1) * max(degree[v], 1)) ** 0.5

    seed_pos = layered_layout(items, arcs, dx=spacing, dy=spacing * 0.6)

    box = spacing * max(1.0, len(items) ** 0.6)
    raw = nx.spring_layout(
        graph,
        pos=seed_pos,
        k=spacing * 1.5,
        weight="weight",
        iterations=iterations,
        scale=box,
        center=(box, box),
        seed=seed,
    )
    positions = {item: (float(x), float(y)) for item, (x, y) in raw.items()}
    _relax_positions(positions, arcs, min_gap=spacing, min_dist=spacing)
    return positions
