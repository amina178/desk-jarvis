"""Static hand-gesture recognition: shared preprocessing + ONNX inference.

The SAME functions are used in the training notebook (Colab) and in the
live app. Training/serving skew - preprocessing that differs between the
two - is one of the most common reasons a model that scored well offline
fails live, so there is exactly one implementation of each step here.

Two model families are supported:
    "landmarks"  input = 63 numbers from normalize_hand()  (MLP)
    "crop"       input = RGB hand crop, ImageNet-normalised  (CNN)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deskjarvis.features import normalize_hand

# HaGRID class names used in this project. Sorted = index order of the
# model outputs (the notebook saves the exact list to labels.json).
GESTURES = ("dislike", "fist", "like", "no_gesture", "one", "palm", "peace")
IDLE = "no_gesture"

# What each gesture does. Gestures without an action (fist, one) are
# still trained: they teach the model what NOT to fire on.
GESTURE_ACTIONS = {
    "like": "next_slide",
    "dislike": "prev_slide",
    "palm": "toggle_mute",
    "peace": "screenshot",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------ features


def landmark_input(hand: np.ndarray, mirror: bool = False) -> np.ndarray:
    """(21, 3) pixel landmarks -> (63,) float32 model input.

    mirror=True flips x before normalising. Training uses it as an
    augmentation so the model works for both left and right hands (and
    for the mirrored webcam view).
    """
    pts = np.asarray(hand, dtype=np.float64)[:, :3].copy()
    if mirror:
        pts[:, 0] = -pts[:, 0]
    return normalize_hand(pts).astype(np.float32)


def landmark_bbox(hand: np.ndarray, width: int, height: int,
                  margin: float = 0.25) -> tuple[int, int, int, int]:
    """Square box around the landmarks, enlarged by `margin` per side.

    Landmarks lie on joints, so their tight box cuts off the palm edge
    and fingertips; the margin brings it close to a hand bounding box
    like HaGRID's annotations.
    """
    x0, y0 = hand[:, 0].min(), hand[:, 1].min()
    x1, y1 = hand[:, 0].max(), hand[:, 1].max()
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(max(0, cx - side / 2))
    top = int(max(0, cy - side / 2))
    right = int(min(width, cx + side / 2))
    bottom = int(min(height, cy + side / 2))
    return left, top, right, bottom


def crop_input(rgb: np.ndarray, box: tuple[int, int, int, int],
               size: int = 160) -> np.ndarray:
    """Crop + resize + ImageNet normalisation -> (3, size, size) float32."""
    import cv2  # local import: the landmark path needs no OpenCV

    left, top, right, bottom = box
    crop = rgb[top:bottom, left:right]
    if crop.size == 0:
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    x = crop.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)  # HWC -> CHW, as PyTorch expects


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)  # numerical safety
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ------------------------------------------------------------- model


@dataclass
class Prediction:
    label: str
    confidence: float


class GestureClassifier:
    """ONNX Runtime wrapper; the model folder holds model.onnx +
    labels.json ({"labels": [...], "input": "landmarks"|"crop",
    "size": 160})."""

    def __init__(self, model_dir: Path | str, min_confidence: float = 0.7):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "labels.json").read_text())
        self.labels: list[str] = meta["labels"]
        self.kind: str = meta["input"]
        self.size: int = meta.get("size", 160)
        self.min_confidence = min_confidence
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self.session.get_inputs()[0].name

    def predict(self, hand: np.ndarray | None, rgb: np.ndarray | None = None
                ) -> Prediction:
        """hand: (21, 3) pixel landmarks; rgb: full frame (crop models).

        Low-confidence predictions become no_gesture: when unsure, doing
        nothing is the safe default for a UI that triggers actions.
        """
        if hand is None:
            return Prediction(IDLE, 1.0)
        if self.kind == "landmarks":
            x = landmark_input(hand)[None]
        else:
            h, w = rgb.shape[:2]
            x = crop_input(rgb, landmark_bbox(hand, w, h), self.size)[None]
        probs = softmax(self.session.run(None, {self._input_name: x})[0])[0]
        idx = int(probs.argmax())
        label, conf = self.labels[idx], float(probs[idx])
        if conf < self.min_confidence:
            return Prediction(IDLE, conf)
        return Prediction(label, conf)
