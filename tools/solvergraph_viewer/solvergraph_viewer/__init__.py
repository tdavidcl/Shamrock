"""Viewer for Shamrock solvergraph execution traces.

Reads the JSON lines trace files produced by the solvergraph tracer
(``SHAM_SOLVERGRAPH_TRACE=1`` or ``shamrock.enable_solvergraph_tracing()``)
and visualizes node evaluation and edge dataflow in a movable node editor,
either live (tailing a growing trace) or replayed with a speed factor.
"""

__version__ = "0.1.0"
