"""Paths, landmark indices and default thresholds in one place.

Keeping magic numbers here (instead of scattered through the code) makes
experiments reproducible: every threshold used in a run can be logged.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "mediapipe"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

MODEL_URLS = {
    "hand": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "pose": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "face": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
}

# Face Mesh indices for the Eye Aspect Ratio, ordered p1..p6 as in
# Soukupova & Cech (2016): p1/p4 are eye corners, p2/p6 and p3/p5 are
# vertical pairs on the upper/lower lid.
RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)
LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)

# Pose landmark indices (BlazePose topology).
POSE_NOSE = 0
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# Hand landmark indices.
HAND_WRIST = 0
HAND_MIDDLE_MCP = 9

# Labels used when recording data (key -> label).
POSTURE_LABELS = {
    "1": "upright",
    "2": "slouch",
    "3": "forward_head",
    "4": "side_lean",
}
GESTURE_LABELS = {  # same names as the HaGRID classes we train on
    "0": "no_gesture",
    "1": "like",
    "2": "dislike",
    "3": "palm",
    "4": "peace",
    "5": "fist",
    "6": "one",
}
