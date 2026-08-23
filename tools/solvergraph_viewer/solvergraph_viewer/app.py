"""Dear PyGui application: solvergraph trace viewer.

Displays the solver graph in a node editor (nodes draggable like in Blender's
node editor), and animates the trace either live (tailing growing files) or
replayed with a speed factor:

- the node currently being evaluated is highlighted,
- edges bound as read-only pulse green, read-write pulse red (inferred from
  the topology and the evaluation intervals),
- per-node cumulative evaluation time and call count are overlaid,
- multi-rank traces are browsed with a rank selector.
"""

from __future__ import annotations

import os
import time

import dearpygui.dearpygui as dpg

from .layout import Item, build_display_graph, layered_layout
from .model import GraphModel
from .playback import PlaybackClock
from .preview import BlobStore, default_registry
from .trace_reader import TraceReader

# ---------------------------------------------------------------------- #
# colors
# ---------------------------------------------------------------------- #

COL_NODE_TITLE = (60, 60, 90, 255)
COL_NODE_ACTIVE = (230, 140, 20, 255)
COL_NODE_DEAD = (40, 40, 40, 200)
COL_EDGE_TITLE = (45, 75, 45, 255)
COL_EDGE_READ = (40, 180, 60, 255)
COL_EDGE_WRITE = (200, 60, 50, 255)
COL_LINK_RO = (70, 140, 80, 160)
COL_LINK_RO_ACTIVE = (60, 250, 90, 255)
COL_LINK_RW = (150, 80, 70, 160)
COL_LINK_RW_ACTIVE = (255, 80, 60, 255)


def _make_node_theme(title_color) -> int:
    with dpg.theme() as theme, dpg.theme_component(dpg.mvNode):
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, title_color, category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(
            dpg.mvNodeCol_TitleBarHovered, title_color, category=dpg.mvThemeCat_Nodes
        )
        dpg.add_theme_color(
            dpg.mvNodeCol_TitleBarSelected, title_color, category=dpg.mvThemeCat_Nodes
        )
    return theme


def _make_link_theme(color) -> int:
    with dpg.theme() as theme, dpg.theme_component(dpg.mvNodeLink):
        dpg.add_theme_color(dpg.mvNodeCol_Link, color, category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, color, category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, color, category=dpg.mvThemeCat_Nodes)
    return theme


class RankView:
    """Reader + model of one rank's trace file."""

    def __init__(self, path: str):
        self.path = path
        self.reader = TraceReader(path)
        self.model: GraphModel | None = None

    def poll(self) -> bool:
        """Ingest newly available records; returns True if anything changed."""
        new_records = self.reader.poll()
        if self.model is None and self.reader.version is not None:
            self.model = GraphModel(clock_origin=self.reader.clock_origin)
        if self.model is not None and new_records:
            self.model.apply_all(new_records)
        return bool(new_records)

    @property
    def label(self) -> str:
        rank = self.reader.rank if self.reader.rank is not None else "?"
        return f"rank {rank} ({os.path.basename(self.path)})"


class ViewerApp:
    """The viewer application."""

    def __init__(self, paths: list[str], live: bool = False, speed: float = 1.0):
        self.views = [RankView(p) for p in paths]
        self.current_view = 0
        self.clock = PlaybackClock(speed=speed, follow=live)
        self.blob_store = BlobStore(os.path.dirname(os.path.abspath(paths[0])))
        self.preview_registry = default_registry

        # gui item registries
        self.gui_items: dict[Item, int] = {}  # display item -> dpg node id
        self.gui_stats: dict[Item, int] = {}  # display item -> dpg text id
        self.gui_links: dict[tuple[Item, Item], int] = {}
        self.gui_in_attr: dict[Item, int] = {}
        self.gui_out_attr: dict[Item, int] = {}
        self.bound_theme: dict[int, int] = {}  # dpg id -> last bound theme
        self.selected_item: Item | None = None
        self._known_arcs: set = set()
        self._last_frame_time = time.monotonic()
        self._poll_countdown = 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @property
    def view(self) -> RankView:
        return self.views[self.current_view]

    def _bind_theme(self, dpg_id: int, theme: int) -> None:
        if self.bound_theme.get(dpg_id) != theme:
            dpg.bind_item_theme(dpg_id, theme)
            self.bound_theme[dpg_id] = theme

    # ------------------------------------------------------------------ #
    # gui construction
    # ------------------------------------------------------------------ #

    def build(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Shamrock solvergraph viewer", width=1600, height=900)

        self.theme_node_idle = _make_node_theme(COL_NODE_TITLE)
        self.theme_node_active = _make_node_theme(COL_NODE_ACTIVE)
        self.theme_node_dead = _make_node_theme(COL_NODE_DEAD)
        self.theme_edge_idle = _make_node_theme(COL_EDGE_TITLE)
        self.theme_edge_read = _make_node_theme(COL_EDGE_READ)
        self.theme_edge_write = _make_node_theme(COL_EDGE_WRITE)
        self.theme_link_ro = _make_link_theme(COL_LINK_RO)
        self.theme_link_ro_active = _make_link_theme(COL_LINK_RO_ACTIVE)
        self.theme_link_rw = _make_link_theme(COL_LINK_RW)
        self.theme_link_rw_active = _make_link_theme(COL_LINK_RW_ACTIVE)

        with dpg.window(tag="main_window"), dpg.group(horizontal=True):
            with dpg.child_window(width=330, tag="side_panel"):
                dpg.add_text("Trace")
                dpg.add_combo(
                    [v.label for v in self.views],
                    default_value=self.views[0].label,
                    callback=self._on_rank_selected,
                    tag="rank_combo",
                )
                dpg.add_separator()
                dpg.add_text("Playback")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Play/Pause", callback=self._on_play_pause)
                    dpg.add_checkbox(
                        label="Live (follow)",
                        default_value=self.clock.follow,
                        callback=self._on_follow_toggled,
                        tag="follow_checkbox",
                    )
                dpg.add_slider_float(
                    label="speed",
                    default_value=self.clock.speed,
                    min_value=0.1,
                    max_value=100.0,
                    format="%.1fx",
                    callback=self._on_speed_changed,
                    tag="speed_slider",
                )
                dpg.add_slider_float(
                    label="time",
                    min_value=0.0,
                    max_value=1.0,
                    callback=self._on_seek,
                    tag="time_slider",
                )
                dpg.add_text("t = 0.0 s", tag="time_label")
                dpg.add_separator()
                dpg.add_button(label="Re-layout", callback=self._on_relayout)
                dpg.add_separator()
                dpg.add_text("Selection", tag="selection_title")
                dpg.add_text("(click a box)", tag="selection_info", wrap=310)
                dpg.add_group(tag="preview_slot")
            with dpg.child_window(tag="editor_panel"):
                dpg.add_node_editor(
                    tag="node_editor",
                    minimap=True,
                    minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
                )

        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #

    def _on_rank_selected(self, sender, app_data) -> None:
        for i, view in enumerate(self.views):
            if view.label == app_data:
                self.current_view = i
                break
        self._rebuild_editor()

    def _on_play_pause(self) -> None:
        self.clock.follow = False
        dpg.set_value("follow_checkbox", False)
        self.clock.toggle_play()

    def _on_follow_toggled(self, sender, app_data) -> None:
        self.clock.follow = bool(app_data)

    def _on_speed_changed(self, sender, app_data) -> None:
        self.clock.speed = float(app_data)

    def _on_seek(self, sender, app_data) -> None:
        self.clock.follow = False
        dpg.set_value("follow_checkbox", False)
        self.clock.playing = False
        self.clock.seek(float(app_data))

    def _on_relayout(self) -> None:
        model = self.view.model
        if model is None:
            return
        items, arcs = build_display_graph(model)
        positions = layered_layout(items, arcs)
        for item, pos in positions.items():
            if item in self.gui_items:
                dpg.set_item_pos(self.gui_items[item], [pos[0], pos[1]])

    def _on_item_clicked(self, sender, app_data, user_data: Item) -> None:
        self.selected_item = user_data
        self._update_selection_panel()

    # ------------------------------------------------------------------ #
    # editor sync
    # ------------------------------------------------------------------ #

    def _rebuild_editor(self) -> None:
        for dpg_id in list(self.gui_links.values()):
            dpg.delete_item(dpg_id)
        for dpg_id in list(self.gui_items.values()):
            dpg.delete_item(dpg_id)
        self.gui_items.clear()
        self.gui_stats.clear()
        self.gui_links.clear()
        self.gui_in_attr.clear()
        self.gui_out_attr.clear()
        self.bound_theme.clear()
        self._known_arcs.clear()
        self._sync_editor(full_layout=True)

    def _sync_editor(self, full_layout: bool = False) -> None:
        """Add gui items for any model item not displayed yet."""
        model = self.view.model
        if model is None:
            return

        items, arcs = build_display_graph(model)
        new_items = [it for it in items if it not in self.gui_items]
        if not new_items and not full_layout:
            new_arcs = [a for a in arcs if a not in self._known_arcs]
            if not new_arcs:
                return

        positions = layered_layout(items, arcs)

        for item in items:
            if item in self.gui_items:
                continue
            kind, uuid = item
            if kind == "node":
                state = model.nodes[uuid]
                label = state.label or f"node {uuid}"
                theme = self.theme_node_idle
            else:
                state = model.edges[uuid]
                label = state.label or f"edge {uuid}"
                theme = self.theme_edge_idle
            pos = positions.get(item, (0.0, 0.0))
            node_id = dpg.add_node(parent="node_editor", label=label, pos=[pos[0], pos[1]])
            with dpg.item_handler_registry() as handlers:
                dpg.add_item_clicked_handler(callback=self._on_item_clicked, user_data=item)
            dpg.bind_item_handler_registry(node_id, handlers)
            in_attr = dpg.add_node_attribute(parent=node_id, attribute_type=dpg.mvNode_Attr_Input)
            static_attr = dpg.add_node_attribute(
                parent=node_id, attribute_type=dpg.mvNode_Attr_Static
            )
            stats_id = dpg.add_text("", parent=static_attr)
            out_attr = dpg.add_node_attribute(parent=node_id, attribute_type=dpg.mvNode_Attr_Output)
            self.gui_items[item] = node_id
            self.gui_stats[item] = stats_id
            self.gui_in_attr[item] = in_attr
            self.gui_out_attr[item] = out_attr
            self._bind_theme(node_id, theme)

        for arc in arcs:
            if arc in self._known_arcs:
                continue
            src, dst = arc
            if src not in self.gui_out_attr or dst not in self.gui_in_attr:
                continue
            link_id = dpg.add_node_link(
                self.gui_out_attr[src], self.gui_in_attr[dst], parent="node_editor"
            )
            # arcs from an edge box to a node box are read-only bindings
            is_ro = src[0] == "edge"
            self._bind_theme(link_id, self.theme_link_ro if is_ro else self.theme_link_rw)
            self.gui_links[arc] = link_id
            self._known_arcs.add(arc)

    # ------------------------------------------------------------------ #
    # per-frame update
    # ------------------------------------------------------------------ #

    def _update_selection_panel(self) -> None:
        model = self.view.model
        if model is None or self.selected_item is None:
            return
        kind, uuid = self.selected_item
        lines: list[str] = []
        if kind == "node" and uuid in model.nodes:
            node = model.nodes[uuid]
            lines.append(f"node: {node.label} (uuid {uuid})")
            lines.append(f"type: {node.type_name}")
            lines.append(f"ro edges: {[label or u for u, label in node.ro_edges]}")
            lines.append(f"rw edges: {[label or u for u, label in node.rw_edges]}")
            lines.append(f"evaluations: {node.evals.count_before(self.clock.t)}")
            lines.append(f"cumulated time: {node.evals.cumulative_time(self.clock.t) * 1e3:.3f} ms")
        elif kind == "edge" and uuid in model.edges:
            edge = model.edges[uuid]
            lines.append(f"edge: {edge.label} (uuid {uuid})")
            self._update_preview(edge)
        dpg.set_value("selection_info", "\n".join(lines))

    def _update_preview(self, edge) -> None:
        """Render the latest edge payload preview if a renderer is available.

        No producer emits edge_data records yet: this is the reserved preview
        slot of the viewer (see preview.py).
        """
        if not edge.previews:
            return
        _t, format_tag, blob_ref = edge.previews[-1]
        if not self.preview_registry.can_render(format_tag):
            dpg.set_value(
                "selection_info",
                dpg.get_value("selection_info")
                + f"\npayload preview: no renderer for '{format_tag}'",
            )
            return
        blob = self.blob_store.read(blob_ref)
        if blob is None:
            return
        rendered = self.preview_registry.render(format_tag, blob)
        if rendered is None:
            return
        width, height, rgba = rendered
        dpg.delete_item("preview_slot", children_only=True)
        with dpg.texture_registry():
            tex = dpg.add_dynamic_texture(width, height, rgba)
        dpg.add_image(tex, parent="preview_slot")

    def _frame(self) -> None:
        now = time.monotonic()
        real_dt = now - self._last_frame_time
        self._last_frame_time = now

        # poll the trace files (throttled)
        self._poll_countdown -= 1
        if self._poll_countdown <= 0:
            self._poll_countdown = 10
            changed = False
            for view in self.views:
                changed |= view.poll()
            if changed:
                self._sync_editor()

        model = self.view.model
        if model is None:
            return

        t_min, t_max = model.time_span()
        self.clock.set_span(t_min, t_max)
        self.clock.advance(real_dt)
        t = self.clock.t

        dpg.configure_item("time_slider", min_value=t_min, max_value=t_max)
        dpg.set_value("time_slider", t)
        dpg.set_value("time_label", f"t = {t:.6f} s")

        active = model.active_nodes(t)
        reads, writes = model.edge_activity(t)

        for item, dpg_id in self.gui_items.items():
            kind, uuid = item
            if kind == "node":
                state = model.nodes[uuid]
                if not state.exists_at(t):
                    theme = self.theme_node_dead
                elif uuid in active:
                    theme = self.theme_node_active
                else:
                    theme = self.theme_node_idle
                self._bind_theme(dpg_id, theme)
                count = state.evals.count_before(t)
                cum_ms = state.evals.cumulative_time(t) * 1e3
                dpg.set_value(self.gui_stats[item], f"n={count}  t={cum_ms:.2f} ms")
            else:
                state = model.edges[uuid]
                if not state.exists_at(t):
                    theme = self.theme_node_dead
                elif uuid in writes:
                    theme = self.theme_edge_write
                elif uuid in reads:
                    theme = self.theme_edge_read
                else:
                    theme = self.theme_edge_idle
                self._bind_theme(dpg_id, theme)

        for arc, link_id in self.gui_links.items():
            src, dst = arc
            if src[0] == "edge":  # ro binding: edge -> node
                is_active = dst[1] in active and src[1] in reads
                theme = self.theme_link_ro_active if is_active else self.theme_link_ro
            else:  # rw binding: node -> edge
                is_active = src[1] in active and dst[1] in writes
                theme = self.theme_link_rw_active if is_active else self.theme_link_rw
            self._bind_theme(link_id, theme)

        if self.selected_item is not None:
            self._update_selection_panel()

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.build()
        for view in self.views:
            view.poll()
        self._sync_editor(full_layout=True)
        while dpg.is_dearpygui_running():
            self._frame()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def run_app(paths: list[str], live: bool = False, speed: float = 1.0) -> None:
    ViewerApp(paths, live=live, speed=speed).run()
