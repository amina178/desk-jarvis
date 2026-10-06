"""Offline analysis of a recorded video (used by the Streamlit app).

Every frame goes through the same LandmarkExtractor + Analyzer as the
live demo, with timestamps taken from the video's own frame rate, so a
recorded session gives the same results as watching it live.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from deskjarvis.landmarks import LandmarkExtractor
from deskjarvis.pipeline import Analyzer

REPORT_COLUMNS = ["t_s", "gesture", "gesture_conf", "event", "ear",
                  "blinked", "blink_rate", "posture_status", "posture",
                  "posture_bad"]


def analyze_video(path: str, analyzer: Analyzer, max_seconds: float = 300,
                  max_width: int = 640,
                  progress: Callable[[float], None] | None = None
                  ) -> pd.DataFrame:
    """Returns one row per processed frame (see REPORT_COLUMNS)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(min(total or 10**9, max_seconds * fps))

    rows = []
    with LandmarkExtractor(hand=analyzer.gestures is not None) as extractor:
        for i in range(limit):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:  # landmarks are scale-free; smaller = faster
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_ms = int(i * 1000 / fps)
            st = analyzer.step(extractor.process(rgb, t_ms), rgb)
            row = {k: v for k, v in asdict(st).items() if k != "features"}
            row["t_s"] = t_ms / 1000
            rows.append(row)
            if progress and limit:
                progress(min(1.0, (i + 1) / limit))
    cap.release()
    df = pd.DataFrame(rows)
    return df[REPORT_COLUMNS] if not df.empty else pd.DataFrame(
        columns=REPORT_COLUMNS)


def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers for the report."""
    if df.empty:
        return {"duration_s": 0.0}
    duration = float(df.t_s.iloc[-1] - df.t_s.iloc[0]) or 1e-9
    judged = df[df.posture_status == "ok"]
    shares = (judged.posture.value_counts(normalize=True).to_dict()
              if len(judged) else {})
    return {
        "duration_s": duration,
        "blinks": int(df.blinked.sum()),
        "blinks_per_min": float(df.blinked.sum() / duration * 60),
        "face_found": float(df.ear.notna().mean()),
        "posture_judged": float(len(judged) / len(df)),
        "posture_shares": shares,
        "bad_posture_alert_s": float(
            df.posture_bad.sum() / max(1, len(df)) * duration),
        "gesture_events": df.event.dropna().value_counts().to_dict(),
    }


def blinks_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    minute = (df.t_s // 60).astype(int)
    out = df.groupby(minute)["blinked"].sum().rename("blinks").reset_index()
    return out.rename(columns={"t_s": "minute"})


def write_test_video(path: str, frame: np.ndarray, n_frames: int = 60,
                     fps: float = 30.0) -> str:
    """Helper for tests and demos: repeat one RGB frame as a video."""
    h, w = frame.shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for _ in range(n_frames):
        out.write(bgr)
    out.release()
    return path
