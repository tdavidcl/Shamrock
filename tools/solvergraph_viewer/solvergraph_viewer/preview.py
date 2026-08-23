"""Edge payload preview plugins (reserved ``edge_data`` trace records).

The trace format reserves an ``edge_data`` record kind carrying a format tag
and a blob reference ``{"file": ..., "offset": ..., "len": ...}`` into a
per-rank binary sidecar file. The producer does not emit those yet; this
module provides the plumbing so that when it does, previews (e.g. field
snapshots rendered as textures, Blender-style) can be displayed inside the
node editor by registering a renderer for the format tag.
"""

from __future__ import annotations

import os
from collections.abc import Callable


class BlobStore:
    """Reads blob references out of trace sidecar files."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def read(self, blob_ref: dict) -> bytes | None:
        path = blob_ref.get("file")
        if not path:
            return None
        if not os.path.isabs(path):
            path = os.path.join(self.base_dir, path)
        try:
            with open(path, "rb") as f:
                f.seek(blob_ref.get("offset", 0))
                return f.read(blob_ref.get("len", 0))
        except OSError:
            return None


# a renderer takes (format tag, raw blob bytes) and returns an RGBA float
# texture as (width, height, [r, g, b, a, ...]) or None if it cannot render
Renderer = Callable[[str, bytes], tuple | None]


class PreviewRegistry:
    """Registry of preview renderers keyed by format tag."""

    def __init__(self):
        self._renderers: dict[str, Renderer] = {}

    def register(self, format_tag: str, renderer: Renderer) -> None:
        self._renderers[format_tag] = renderer

    def can_render(self, format_tag: str) -> bool:
        return format_tag in self._renderers

    def render(self, format_tag: str, blob: bytes) -> tuple | None:
        renderer = self._renderers.get(format_tag)
        if renderer is None:
            return None
        return renderer(format_tag, blob)


#: default registry; no renderers are shipped yet (no producer emits
#: edge_data records yet) - register yours here
default_registry = PreviewRegistry()
