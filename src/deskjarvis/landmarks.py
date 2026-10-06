"""Thin wrapper around the MediaPipe Tasks landmarkers.

MediaPipe acts as a frozen, pre-trained feature extractor (like a
backbone): it turns a frame into hand / pose / face keypoints. Our own
models are trained on top of these keypoints and on image crops.

Note: the legacy `mp.solutions` API is removed in recent MediaPipe
releases, so this module uses the Tasks API with `.task` model files
(see scripts/download_models.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from deskjarvis import config as cfg


@dataclass
class FrameLandmarks:
    """Landmarks of one frame in pixel coordinates (None if not found)."""

    timestamp_ms: int
    hand: np.ndarray | None = None  # (21, 3)
    handedness: str | None = None
    pose: np.ndarray | None = None  # (33, 4): x, y, z, visibility
    face: np.ndarray | None = None  # (478, 3)
    blendshapes: dict[str, float] = field(default_factory=dict)


def _to_pixels(landmarks, width: int, height: int, visibility=False):
    """Normalised MediaPipe landmarks -> numpy array in pixels.

    z is scaled by width, following MediaPipe's convention that z uses
    roughly the same scale as x.
    """
    rows = []
    for lm in landmarks:
        row = [lm.x * width, lm.y * height, lm.z * width]
        if visibility:
            row.append(lm.visibility or 0.0)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


class LandmarkExtractor:
    """Runs the selected landmarkers on a video stream.

    Usage:
        with LandmarkExtractor(hand=True, pose=True, face=True) as ex:
            result = ex.process(rgb_frame, timestamp_ms)
    """

    def __init__(self, hand=True, pose=True, face=True, models_dir=None):
        models_dir = models_dir or cfg.MODELS_DIR
        missing = [
            n for n, on in (("hand", hand), ("pose", pose), ("face", face))
            if on and not (models_dir / f"{n}.task").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing models {missing} in {models_dir}. "
                "Run: python scripts/download_models.py"
            )
        mode = vision.RunningMode.VIDEO  # uses tracking between frames
        self._last_ts = -1
        self._hand = self._pose = self._face = None

        if hand:
            self._hand = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(models_dir / "hand.task")
                    ),
                    running_mode=mode,
                    num_hands=1,
                )
            )
        if pose:
            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(models_dir / "pose.task")
                    ),
                    running_mode=mode,
                )
            )
        if face:
            self._face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(models_dir / "face.task")
                    ),
                    running_mode=mode,
                    output_face_blendshapes=True,
                )
            )

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> FrameLandmarks:
        """Timestamps must strictly increase within one extractor.

        VIDEO mode rejects repeated timestamps, which can happen when two
        frames arrive within the same millisecond - we nudge them.
        """
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        out = FrameLandmarks(timestamp_ms=timestamp_ms)

        if self._hand:
            res = self._hand.detect_for_video(image, timestamp_ms)
            if res.hand_landmarks:
                out.hand = _to_pixels(res.hand_landmarks[0], w, h)
                out.handedness = res.handedness[0][0].category_name
        if self._pose:
            res = self._pose.detect_for_video(image, timestamp_ms)
            if res.pose_landmarks:
                out.pose = _to_pixels(res.pose_landmarks[0], w, h, True)
        if self._face:
            res = self._face.detect_for_video(image, timestamp_ms)
            if res.face_landmarks:
                out.face = _to_pixels(res.face_landmarks[0], w, h)
            if res.face_blendshapes:
                # eyeBlinkLeft/Right: MediaPipe's own blink score, a free
                # extra baseline to compare against our EAR detector.
                out.blendshapes = {
                    c.category_name: c.score
                    for c in res.face_blendshapes[0]
                    if c.category_name.startswith("eyeBlink")
                }
        return out

    def close(self):
        for task in (self._hand, self._pose, self._face):
            if task:
                task.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
