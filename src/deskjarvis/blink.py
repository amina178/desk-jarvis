"""Streaming blink detector on the Eye Aspect Ratio (baseline model).

Why an adaptive threshold: a fixed EAR threshold (the classic 0.2)
fails across people - eye shape, glasses and camera angle move the
"open eye" level (hypothesis H1). We track each user's own open-eye
level with a rolling median and call a blink when EAR drops below a
fraction of it. The median is robust because blinks occupy only a few
percent of frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class BlinkConfig:
    ratio: float = 0.72  # blink if EAR < ratio * open-eye baseline
    min_frames: int = 2  # shorter dips are landmark jitter
    max_duration_ms: int = 500  # longer closures are not blinks
    baseline_window: int = 90  # frames (~3 s at 30 FPS)
    rate_window_s: float = 60.0


class BlinkDetector:
    """Feed one EAR value per frame; get blink events and blinks/min."""

    def __init__(self, config: BlinkConfig | None = None):
        self.cfg = config or BlinkConfig()
        self._history = deque(maxlen=self.cfg.baseline_window)
        self._closed_frames = 0
        self._closed_since_ms: int | None = None
        self._blink_times: deque[int] = deque()
        self.total_blinks = 0

    @property
    def baseline(self) -> float | None:
        # Wait for ~1 s of data before trusting the baseline.
        if len(self._history) < self.cfg.baseline_window // 3:
            return None
        return float(np.median(self._history))

    def update(self, ear: float | None, timestamp_ms: int) -> bool:
        """Returns True on the frame where a blink is confirmed.

        A blink is counted when the eye RE-OPENS (falling then rising
        edge); only then do we know the closure was short enough.
        """
        if ear is None:  # face lost: reset the closure state
            self._closed_frames = 0
            self._closed_since_ms = None
            return False

        baseline = self.baseline
        self._history.append(ear)
        if baseline is None:
            return False

        closed = ear < self.cfg.ratio * baseline
        if closed:
            if self._closed_frames == 0:
                self._closed_since_ms = timestamp_ms
            self._closed_frames += 1
            return False

        blinked = False
        if self._closed_frames >= self.cfg.min_frames:
            duration = timestamp_ms - (self._closed_since_ms or 0)
            if duration <= self.cfg.max_duration_ms:
                blinked = True
                self.total_blinks += 1
                self._blink_times.append(timestamp_ms)
        self._closed_frames = 0
        self._closed_since_ms = None
        return blinked

    def blinks_per_minute(self, now_ms: int) -> float:
        window_ms = self.cfg.rate_window_s * 1000
        while self._blink_times and now_ms - self._blink_times[0] > window_ms:
            self._blink_times.popleft()
        return len(self._blink_times) * 60.0 / self.cfg.rate_window_s
