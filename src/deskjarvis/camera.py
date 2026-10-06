"""Opening the webcam with a clear error instead of a silent exit."""
from __future__ import annotations

import platform

import cv2

_MAC_HINT = (
    "macOS blocks camera access for this app.\n"
    "  1. System Settings -> Privacy & Security -> Camera\n"
    "  2. Enable the app you run Python from (Terminal, iTerm, VS Code)\n"
    "  3. Quit that app completely (Cmd+Q) and open it again\n"
    "  If it is not listed: run `tccutil reset Camera`, then retry."
)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Return an opened capture that has delivered at least one frame.

    `VideoCapture` does not raise when access is denied: `isOpened()` is
    False or `read()` returns nothing. We check both and fail loudly.
    """
    cap = cv2.VideoCapture(index)
    ok = cap.isOpened() and cap.read()[0]
    if not ok:
        cap.release()
        hint = _MAC_HINT if platform.system() == "Darwin" else (
            "Check that the camera is connected and not used by another "
            "app (Zoom, Teams, browser), or try --camera 1."
        )
        raise RuntimeError(f"Cannot read from camera {index}.\n{hint}")
    return cap
