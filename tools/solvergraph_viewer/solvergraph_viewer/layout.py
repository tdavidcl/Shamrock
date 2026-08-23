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

from collections.abc import Iterable

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
