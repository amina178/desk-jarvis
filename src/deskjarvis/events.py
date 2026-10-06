"""Turning noisy per-frame predictions into reliable events.

A classifier running at 30 FPS flickers: one frame "palm", the next
"no_gesture". Firing an OS action on every frame would be unusable, so
the product metric (false triggers per hour, hypothesis H4) depends as
much on this layer as on model accuracy.
"""
from __future__ import annotations

from collections import Counter, deque


class GestureDebouncer:
    """Majority vote over a sliding window + cooldown + re-arm.

    - window/min_votes: a gesture must dominate recent frames.
    - cooldown_s: minimum time between two fired events.
    - re-arm: holding a gesture fires once; it must disappear before
      the same gesture can fire again.
    """

    def __init__(self, window=8, min_votes=6, cooldown_s=1.0,
                 idle_label="no_gesture"):
        self.window = deque(maxlen=window)
        self.min_votes = min_votes
        self.cooldown_ms = int(cooldown_s * 1000)
        self.idle = idle_label
        self._last_fire_ms = -10**12
        self._armed_for: str | None = None  # label currently held

    def update(self, label: str | None, timestamp_ms: int) -> str | None:
        self.window.append(label or self.idle)
        top, votes = Counter(self.window).most_common(1)[0]

        if top == self.idle or votes < self.min_votes:
            if top == self.idle:
                self._armed_for = None  # gesture released
            return None
        if top == self._armed_for:
            return None  # still holding the same gesture
        if timestamp_ms - self._last_fire_ms < self.cooldown_ms:
            return None

        self._last_fire_ms = timestamp_ms
        self._armed_for = top
        return top


class PersistentState:
    """Reports a state only after it has lasted `hold_s` seconds.

    Used for posture: a two-second stretch is not "slouching"; only a
    bad posture held for e.g. 30 s should trigger a reminder.
    """

    def __init__(self, hold_s: float):
        self.hold_ms = int(hold_s * 1000)
        self._state: str | None = None
        self._since_ms = 0

    def update(self, state: str, timestamp_ms: int) -> tuple[str, bool]:
        """Returns (state, is_persistent)."""
        if state != self._state:
            self._state, self._since_ms = state, timestamp_ms
        return state, timestamp_ms - self._since_ms >= self.hold_ms


class BreakTimer:
    """20-20-20 style reminder driven by actual presence at the desk.

    Continuous presence accumulates; an absence longer than
    `reset_after_s` counts as a real break and resets the timer.
    """

    def __init__(self, work_minutes=20.0, reset_after_s=120.0):
        self.work_ms = int(work_minutes * 60_000)
        self.reset_ms = int(reset_after_s * 1000)
        self._present_since: int | None = None
        self._absent_since: int | None = None
        self._reminded = False

    def update(self, present: bool, timestamp_ms: int) -> bool:
        """Returns True once when a break reminder should be shown."""
        if present:
            self._absent_since = None
            if self._present_since is None:
                self._present_since = timestamp_ms
            worked = timestamp_ms - self._present_since
            if worked >= self.work_ms and not self._reminded:
                self._reminded = True
                return True
            return False

        if self._absent_since is None:
            self._absent_since = timestamp_ms
        if timestamp_ms - self._absent_since >= self.reset_ms:
            self._present_since = None
            self._reminded = False
        return False
