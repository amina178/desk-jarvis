"""Desk Jarvis live: gestures, posture, blinks and break reminders.

    python scripts/live_demo.py              # actions only shown (safe)
    python scripts/live_demo.py --actions    # really press keys

Models are picked up automatically when present:
    models/gestures/mlp/        (notebook 02) -> gesture control
    models/posture/model.joblib (notebook 03) -> ML posture classifier
Without them the demo falls back to the baselines (no gestures, rules).

Keys (click the video window first):
    c  (re)calibrate posture - sit upright for 5 seconds
    d  toggle debug view: keypoints and raw values
    q  quit
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from deskjarvis import config as cfg
from deskjarvis.camera import open_camera
from deskjarvis.landmarks import LandmarkExtractor
from deskjarvis.pipeline import GESTURE_DIR, POSTURE_POINTS, Analyzer

EYE_POINTS = list(cfg.RIGHT_EYE_IDX) + list(cfg.LEFT_EYE_IDX)
GREEN, RED, YELLOW, ORANGE, WHITE, CYAN = (
    (0, 200, 0), (0, 0, 255), (0, 220, 255), (0, 140, 255),
    (255, 255, 255), (255, 220, 0))


def put(frame, text, y, color=WHITE, scale=0.65):
    for c, t in (((0, 0, 0), 4), (color, 2)):  # outline, then text
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    c, t)


def draw_points(frame, pts, color):
    for x, y in np.asarray(pts)[:, :2].astype(int):
        cv2.circle(frame, (x, y), 3, color, -1)


def draw(frame, st, an, res, debug, actions_on, banner):
    if an.gestures is None:
        put(frame, "Gestures: off (no model in models/gestures/mlp)", 30,
            CYAN)
    else:
        put(frame, f"Gesture: {st.gesture} ({st.gesture_conf:.2f})", 30,
            CYAN)

    blink = f"Blinks/min {st.blink_rate:.0f}  total {st.blinks_total}"
    if debug and st.ear:
        blink += f"  EAR {st.ear:.3f}"
    put(frame, blink, 60, RED if st.low_blink_rate else GREEN)

    if st.posture_status == "not_visible":
        put(frame, "Posture: shoulders not visible - move back or lower "
            "the camera", 90, ORANGE)
    elif st.posture_status == "calibrating":
        put(frame, f"Calibrating: sit upright... {st.calib_left_s:.0f}s", 90,
            YELLOW)
    else:
        put(frame, f"Posture ({an.posture_kind}): {st.posture}", 90,
            RED if st.posture_bad else GREEN)
    if debug and st.features:
        put(frame, ("neck {neck_ratio:.2f}  head {head_size_ratio:.2f}  "
                    "tilt {shoulder_tilt:+.1f}").format(**st.features), 120,
            YELLOW)
    if banner:
        put(frame, banner, frame.shape[0] // 2, CYAN, scale=1.0)

    if debug:
        if res.face is not None:
            draw_points(frame, res.face[EYE_POINTS], YELLOW)
        if res.hand is not None:
            draw_points(frame, res.hand, CYAN)
        if res.pose is not None:
            for idx in POSTURE_POINTS:
                ok = res.pose[idx, 3] >= an.s.min_visibility
                draw_points(frame, res.pose[[idx]], GREEN if ok else RED)
    mode = "ACTIONS ON" if actions_on else "dry-run"
    put(frame, f"[{mode}]  c: calibrate  d: debug  q: quit",
        frame.shape[0] - 15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", action="store_true",
                    help="really press keys (default: only show them)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--gesture-model", type=Path, default=GESTURE_DIR)
    args = ap.parse_args()

    an = Analyzer.with_default_models(dry_run=not args.actions,
                                      gesture_dir=args.gesture_model)
    print(f"gestures: {'on' if an.gestures else 'off'} | "
          f"posture: {an.posture_kind}")
    banner, banner_until, reminder_until, debug = "", 0, 0, False

    cap = open_camera(args.camera)
    start = time.monotonic()
    with LandmarkExtractor(hand=an.gestures is not None, pose=True,
                           face=True) as extractor:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t_ms = int((time.monotonic() - start) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = extractor.process(rgb, t_ms)
            st = an.step(res, rgb)

            if st.action_text:
                banner, banner_until = st.action_text, t_ms + 2000
            if st.break_reminder:
                reminder_until = t_ms + 10_000
            draw(frame, st, an, res, debug, args.actions,
                 banner if t_ms < banner_until else "")
            if t_ms < reminder_until:
                put(frame, "Time for a break: look 20 s into the distance",
                    150, ORANGE)

            cv2.imshow("Desk Jarvis - live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                an.recalibrate(t_ms)
            elif key == ord("d"):
                debug = not debug
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
