"""Record a labelled landmark session from the webcam.

Only landmarks are saved (never images): privacy by design, and a
session of several minutes is a few MB instead of gigabytes of video.

Examples:
    python scripts/record_session.py --task posture --subject s01
    python scripts/record_session.py --task gesture --subject s01

Controls (in the video window):
    number keys  set the current label (see config.*_LABELS)
    space        start / pause recording
    q            save and quit
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from deskjarvis import config as cfg
from deskjarvis.camera import open_camera
from deskjarvis.features import face_ear
from deskjarvis.landmarks import LandmarkExtractor

TASK_LABELS = {"posture": cfg.POSTURE_LABELS, "gesture": cfg.GESTURE_LABELS,
               "blink": {"0": "unlabeled"}}


def frame_row(res, label, subject, session):
    """Flatten one frame into a dict (one parquet row)."""
    row = {"subject": subject, "session": session,
           "t_ms": res.timestamp_ms, "label": label}
    if res.pose is not None:
        for i, (x, y, z, v) in enumerate(res.pose):
            row.update({f"pose_{i}_x": x, f"pose_{i}_y": y,
                        f"pose_{i}_z": z, f"pose_{i}_v": v})
    if res.hand is not None:
        row["handedness"] = res.handedness
        for i, (x, y, z) in enumerate(res.hand):
            row.update({f"hand_{i}_x": x, f"hand_{i}_y": y,
                        f"hand_{i}_z": z})
    if res.face is not None:
        row["ear"] = face_ear(res.face)
    row.update(res.blendshapes)
    return row


def draw_hud(frame, label, recording, n_rows, labels):
    color = (0, 0, 255) if recording else (200, 200, 200)
    status = "REC" if recording else "PAUSED"
    cv2.putText(frame, f"{status}  label={label}  rows={n_rows}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    keys = "  ".join(f"{k}:{v}" for k, v in labels.items())
    cv2.putText(frame, keys, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASK_LABELS, required=True)
    ap.add_argument("--subject", required=True, help="e.g. s01")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    labels = TASK_LABELS[args.task]
    label = next(iter(labels.values()))
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, recording = [], False

    cap = open_camera(args.camera)
    start = time.monotonic()
    use = {"hand": args.task == "gesture", "pose": args.task == "posture",
           "face": args.task == "blink"}
    with LandmarkExtractor(**use) as extractor:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view feels natural
            t_ms = int((time.monotonic() - start) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = extractor.process(rgb, t_ms)
            if recording:
                rows.append(frame_row(res, label, args.subject, session))

            draw_hud(frame, label, recording, len(rows), labels)
            cv2.imshow("Desk Jarvis - recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
            elif chr(key) in labels:
                label = labels[chr(key)]
    cap.release()
    cv2.destroyAllWindows()

    if rows:
        out_dir = cfg.RAW_DIR / args.task
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{args.subject}_{session}.parquet"
        df = pd.DataFrame(rows).astype({"t_ms": np.int64})
        df.to_parquet(out, index=False)
        print(f"Saved {len(df)} frames -> {out}")
        print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
