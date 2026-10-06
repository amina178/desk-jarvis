"""Turning recognised gestures into computer actions.

Dry-run is the default: the action is only shown on screen. Real key
presses require --actions, because a vision model WILL misfire
sometimes, and an unexpected keystroke in the wrong window can do
damage. Real mode needs `pip install pyautogui`; on macOS also allow
Terminal in System Settings -> Privacy & Security -> Accessibility.
"""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path

# Mute hotkeys of the most common call apps (Zoom by default).
MUTE_HOTKEY = {
    "Darwin": ("command", "shift", "a"),
    "Windows": ("alt", "a"),
    "Linux": ("alt", "a"),
}
SCREENSHOT_DIR = Path.home() / "Pictures" / "DeskJarvis"

DESCRIPTIONS = {
    "next_slide": "Next slide (Right arrow)",
    "prev_slide": "Previous slide (Left arrow)",
    "toggle_mute": "Toggle mute (Zoom hotkey)",
    "screenshot": "Screenshot saved",
}


class ActionRunner:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._gui = None
        if not dry_run:
            import pyautogui  # optional dependency, only for real mode

            pyautogui.FAILSAFE = True  # mouse to a corner aborts
            self._gui = pyautogui

    def run(self, action: str) -> str:
        """Performs (or simulates) the action; returns a status line."""
        if action not in DESCRIPTIONS:
            raise ValueError(f"Unknown action: {action}")
        text = DESCRIPTIONS[action]
        if self.dry_run:
            return f"[dry-run] {text}"

        if action == "next_slide":
            self._gui.press("right")
        elif action == "prev_slide":
            self._gui.press("left")
        elif action == "toggle_mute":
            self._gui.hotkey(*MUTE_HOTKEY.get(platform.system(),
                                              MUTE_HOTKEY["Linux"]))
        elif action == "screenshot":
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            name = datetime.now().strftime("shot_%Y%m%d_%H%M%S.png")
            self._gui.screenshot().save(SCREENSHOT_DIR / name)
            text = f"{text}: {SCREENSHOT_DIR / name}"
        return text
