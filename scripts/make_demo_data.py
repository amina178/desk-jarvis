"""Generate SYNTHETIC sessions in the same format as record_session.py.

Purpose: run and debug the EDA notebook before real data is recorded.
Synthetic data only checks that the code works - never report metrics
on it. Files are written to data/demo/ (not data/raw/) so they cannot
be mixed up with real recordings.

    python scripts/make_demo_data.py
"""
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from deskjarvis import config as cfg

OUT = cfg.DATA_DIR / "demo"
FPS = 30
rng = np.random.default_rng(42)


def pose_frame(scale, cx, label):
    """33 pose points; only the ones used by posture_features matter."""
    pose = np.zeros((33, 4))
    pose[:, 3] = 0.99
    shoulder_w, neck, ear_w = 200 * scale, 120 * scale, 80 * scale
    tilt = 0.0
    if label == "slouch":
        neck *= 0.75
    elif label == "forward_head":
        ear_w *= 1.2
        neck *= 0.92
    elif label == "side_lean":
        tilt = np.tan(np.radians(12)) * shoulder_w
    sy = 400.0
    noise = lambda: rng.normal(0, 2.0 * scale, 2)  # noqa: E731
    pose[cfg.POSE_LEFT_SHOULDER, :2] = [cx + shoulder_w / 2, sy + tilt]
    pose[cfg.POSE_RIGHT_SHOULDER, :2] = [cx - shoulder_w / 2, sy]
    pose[cfg.POSE_LEFT_EAR, :2] = [cx + ear_w / 2, sy - neck]
    pose[cfg.POSE_RIGHT_EAR, :2] = [cx - ear_w / 2, sy - neck]
    pose[cfg.POSE_NOSE, :2] = [cx, sy - neck - 10 * scale]
    for i in (0, 7, 8, 11, 12):
        pose[i, :2] += noise()
    return pose


def posture_session(subject):
    scale = rng.uniform(0.7, 1.3)  # distance to the camera
    cx = rng.uniform(250, 390)
    rows, t = [], 0
    plan = [("upright", 10), ("slouch", 20), ("upright", 10),
            ("forward_head", 20), ("side_lean", 15), ("upright", 10)]
    for label, seconds in plan:
        for _ in range(seconds * FPS):
            row = {"subject": subject, "session": "demo", "t_ms": t,
                   "label": label}
            for i, (x, y, z, v) in enumerate(pose_frame(scale, cx, label)):
                row.update({f"pose_{i}_x": x, f"pose_{i}_y": y,
                            f"pose_{i}_z": z, f"pose_{i}_v": v})
            rows.append(row)
            t += 1000 // FPS
    return pd.DataFrame(rows)


def blink_session(subject, minutes=10):
    """EAR with blinks whose rate drops over time (hypothesis H3)."""
    n = minutes * 60 * FPS
    open_ear = rng.uniform(0.24, 0.34)
    ear = open_ear + rng.normal(0, 0.01, n)
    blink = np.zeros(n)
    t = 0
    while t < n - 10:
        minute = t / (60 * FPS)
        rate = 18 - 0.8 * minute  # blinks/min fall as focus grows
        t += int(rng.exponential(60 / rate) * FPS) + 5
        if t < n - 5:
            ear[t:t + 4] = open_ear * 0.3
            blink[t:t + 4] = 0.8
    return pd.DataFrame({
        "subject": subject, "session": "demo", "label": "unlabeled",
        "t_ms": np.arange(n) * (1000 // FPS), "ear": ear,
        "eyeBlinkLeft": blink + rng.uniform(0, 0.1, n),
        "eyeBlinkRight": blink + rng.uniform(0, 0.1, n),
    })


def main():
    for task, make in (("posture", posture_session),
                       ("blink", blink_session)):
        (OUT / task).mkdir(parents=True, exist_ok=True)
        for s in range(1, 5):
            subject = f"demo{s:02d}"
            make(subject).to_parquet(OUT / task / f"{subject}.parquet",
                                     index=False)
    print(f"Synthetic sessions written to {OUT}")


if __name__ == "__main__":
    main()
