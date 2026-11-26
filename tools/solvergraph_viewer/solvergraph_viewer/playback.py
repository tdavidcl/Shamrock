"""Playback clock: replay a trace in real time scaled by a speed factor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaybackClock:
    """Current playback position within [t_min, t_max] trace time."""

    t: float = 0.0
    t_min: float = 0.0
    t_max: float = 0.0
    speed: float = 1.0
    playing: bool = False
    follow: bool = False  # live mode: stick to the end of the trace

    def set_span(self, t_min: float, t_max: float) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.t = min(max(self.t, t_min), t_max)
        if self.follow:
            self.t = t_max

    def advance(self, real_dt: float) -> None:
        """Advance the playback position by a real time delta (seconds)."""
        if self.follow:
            self.t = self.t_max
            return
        if not self.playing:
            return
        self.t += real_dt * self.speed
        if self.t >= self.t_max:
            self.t = self.t_max
            self.playing = False

    def seek(self, t: float) -> None:
        self.t = min(max(t, self.t_min), self.t_max)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing and self.t >= self.t_max:
            self.t = self.t_min  # restart from the beginning
