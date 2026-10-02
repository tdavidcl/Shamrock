"""Incremental reader for solvergraph JSON lines trace files.

Each MPI rank writes one append-only file ``<prefix>_<rank>.jsonl``. The first
line is a header record carrying the format version, the rank and the clock
origin. The reader is incremental so it can tail a file that is still being
written (live mode): call :meth:`TraceReader.poll` repeatedly, it returns the
newly available complete records.

Unknown record kinds are kept (not dropped) so newer producers with extra
event kinds (e.g. the reserved ``edge_data`` kind) stay compatible with this
reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SUPPORTED_VERSIONS = (1,)


@dataclass
class TraceRecord:
    """One record (line) of a trace file."""

    kind: str
    raw: dict

    @property
    def seq(self) -> int:
        return self.raw.get("seq", -1)

    @property
    def wtime(self) -> float:
        return self.raw.get("wtime", 0.0)

    @property
    def uuid(self) -> int | None:
        return self.raw.get("uuid")


@dataclass
class TraceReader:
    """Incremental reader of one per-rank trace file."""

    path: str

    version: int | None = None
    rank: int | None = None
    clock_origin: float = 0.0

    _offset: int = 0
    _partial: str = ""
    _header_seen: bool = False
    records: list[TraceRecord] = field(default_factory=list)

    def poll(self) -> list[TraceRecord]:
        """Read the newly available complete records of the file.

        Returns the list of new records (also appended to :attr:`records`).
        Safe to call while the producer is still appending: an incomplete
        trailing line is buffered until its newline arrives.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except FileNotFoundError:
            return []

        if not chunk:
            return []

        data = self._partial + chunk
        lines = data.split("\n")
        # the last element is either "" (data ended with \n) or a partial line
        self._partial = lines.pop()

        new_records: list[TraceRecord] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate torn/corrupted lines
            kind = raw.get("kind", "unknown")
            if kind == "header" and not self._header_seen:
                self._header_seen = True
                self.version = raw.get("version")
                self.rank = raw.get("rank")
                self.clock_origin = raw.get("clock_origin", 0.0)
                continue
            new_records.append(TraceRecord(kind=kind, raw=raw))

        self.records.extend(new_records)
        return new_records

    def to_time(self, wtime: float) -> float:
        """Convert a raw wall clock time to a trace-aligned time."""
        return wtime - self.clock_origin
