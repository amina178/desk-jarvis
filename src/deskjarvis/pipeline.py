"""Per-frame logic of Desk Jarvis, independent of any UI.

The OpenCV live demo and the Streamlit app both feed landmarks into
`Analyzer.step()` and only differ in how they draw the result. Keeping
the logic here means it is tested once and behaves identically in both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deskjarvis import config as cfg
from deskjarvis.actions import ActionRunner
from deskjarvis.blink import BlinkDetector
from deskjarvis.events import BreakTimer, GestureDebouncer, PersistentState
from deskjarvis.features import face_ear, posture_features
from deskjarvis.gestures import GESTURE_ACTIONS, IDLE, GestureClassifier
from deskjarvis.landmarks import FrameLandmarks
from deskjarvis.posture import PostureClassifier, PostureModel

GESTURE_DIR = cfg.ROOT / "models" / "gestures" / "mlp"
POSTURE_MODEL = cfg.ROOT / "models" / "posture" / "model.joblib"
POSTURE_POINTS = [cfg.POSE_LEFT_EAR, cfg.POSE_RIGHT_EAR,
                  cfg.POSE_LEFT_SHOULDER, cfg.POSE_RIGHT_SHOULDER]


@dataclass
class Settings:
    calibration_s: float = 5.0
    bad_posture_hold_s: float = 20.0
    low_blink_rate: float = 10.0  # blinks/min
    min_visibility: float = 0.6
    break_minutes: float = 20.0


@dataclass
class FrameState:
    """Everything a UI needs to draw one frame (and a report row)."""

    t_ms: int
    gesture: str = IDLE
    gesture_conf: float = 0.0
    event: str | None = None  # a fired gesture
    action_text: str | None = None
    ear: float | None = None
    blinked: bool = False
    blinks_total: int = 0
    blink_rate: float = 0.0
    low_blink_rate: bool = False
    posture_status: str = "not_visible"  # calibrating | ok | not_visible
    posture: str | None = None
    posture_bad: bool = False
    calib_left_s: float = 0.0
    break_reminder: bool = False
    features: dict = field(default_factory=dict)


def pose_visible(pose, min_visibility: float) -> bool:
    return pose is not None and bool(
        pose[POSTURE_POINTS, 3].min() >= min_visibility)


class Analyzer:
    def __init__(self, gesture_model: GestureClassifier | None = None,
                 posture_model_path: Path | None = None,
                 runner: ActionRunner | None = None,
                 settings: Settings | None = None):
        self.s = settings or Settings()
        self.gestures = gesture_model
        self.runner = runner or ActionRunner(dry_run=True)
        self.posture_model_path = posture_model_path
        self.debouncer = GestureDebouncer(window=8, min_votes=6,
                                          cooldown_s=1.0)
        self.blink = BlinkDetector()
        self.breaks = BreakTimer(work_minutes=self.s.break_minutes)
        self.recalibrate(0)

    @classmethod
    def with_default_models(cls, dry_run: bool = True,
                            gesture_dir: Path = GESTURE_DIR, **kw):
        """Uses the trained models when present, baselines otherwise."""
        gestures = (GestureClassifier(gesture_dir)
                    if (Path(gesture_dir) / "model.onnx").exists() else None)
        posture = POSTURE_MODEL if POSTURE_MODEL.exists() else None
        return cls(gestures, posture, ActionRunner(dry_run=dry_run), **kw)

    @property
    def posture_kind(self) -> str:
        return "ML model" if isinstance(self.posture, PostureModel) \
            else "rules"

    def recalibrate(self, t_ms: int) -> None:
        self.posture = (PostureModel(self.posture_model_path)
                        if self.posture_model_path else PostureClassifier())
        self.posture_state = PersistentState(self.s.bad_posture_hold_s)
        self._calib: list[dict] = []
        self._calib_until = t_ms + int(self.s.calibration_s * 1000)

    def step(self, res: FrameLandmarks, rgb: np.ndarray | None = None
             ) -> FrameState:
        t = res.timestamp_ms
        st = FrameState(t_ms=t)

        # gestures -> debounced event -> action
        if self.gestures is not None:
            pred = self.gestures.predict(res.hand, rgb)
            st.gesture, st.gesture_conf = pred.label, pred.confidence
            st.event = self.debouncer.update(pred.label, t)
            if st.event in GESTURE_ACTIONS:
                st.action_text = self.runner.run(GESTURE_ACTIONS[st.event])

        # eyes
        st.ear = face_ear(res.face) if res.face is not None else None
        st.blinked = self.blink.update(st.ear, t)
        st.blinks_total = self.blink.total_blinks
        st.blink_rate = self.blink.blinks_per_minute(t)
        st.low_blink_rate = t > 60_000 and st.blink_rate < \
            self.s.low_blink_rate

        # posture (only on reliably visible ears and shoulders)
        if pose_visible(res.pose, self.s.min_visibility):
            feats = posture_features(res.pose)
            st.features = feats
            if self.posture.reference is None:
                if t < self._calib_until:
                    self._calib.append(feats)
                    st.posture_status = "calibrating"
                    st.calib_left_s = (self._calib_until - t) / 1000
                elif len(self._calib) >= 10:
                    self.posture.calibrate(self._calib)
                else:  # too few visible frames: start again
                    self._calib = []
                    self._calib_until = t + int(self.s.calibration_s * 1000)
                    st.posture_status = "calibrating"
            if self.posture.reference is not None:
                label, held = self.posture_state.update(
                    self.posture.predict(feats), t)
                st.posture_status, st.posture = "ok", label
                st.posture_bad = held and label != "upright"

        st.break_reminder = self.breaks.update(res.face is not None, t)
        return st
