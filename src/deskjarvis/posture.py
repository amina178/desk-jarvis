"""Rule-based posture classifier with per-user calibration (baseline).

This is the interpretable baseline that the ML models (Gradient
Boosting, MLP on landmark windows) must beat. The user sits upright for
a few seconds; the median features become their personal reference,
and deviations from it define the classes. Calibration matters because
a frontal camera cannot measure absolute neck angle - only change.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

FEATURES = ("neck_ratio", "head_size_ratio", "shoulder_tilt")


@dataclass
class PostureRules:
    slouch_drop: float = 0.15  # neck_ratio falls by >15 %
    forward_growth: float = 0.12  # head_size_ratio grows by >12 %
    lean_deg: float = 8.0  # absolute shoulder tilt change


class PostureClassifier:
    def __init__(self, rules: PostureRules | None = None):
        self.rules = rules or PostureRules()
        self.reference: dict[str, float] | None = None

    def calibrate(self, samples: list[dict[str, float]]) -> None:
        """samples: posture_features() of frames with upright posture."""
        if len(samples) < 10:
            raise ValueError("Need at least 10 upright frames")
        self.reference = {
            k: float(np.median([s[k] for s in samples])) for k in FEATURES
        }

    def predict(self, feats: dict[str, float]) -> str:
        if self.reference is None:
            raise RuntimeError("Call calibrate() first")
        ref, r = self.reference, self.rules

        # Order matters: leaning also shortens the neck on one side, so
        # the most specific pattern is checked first.
        if abs(feats["shoulder_tilt"] - ref["shoulder_tilt"]) > r.lean_deg:
            return "side_lean"
        if feats["head_size_ratio"] > ref["head_size_ratio"] * (
            1 + r.forward_growth
        ):
            return "forward_head"
        if feats["neck_ratio"] < ref["neck_ratio"] * (1 - r.slouch_drop):
            return "slouch"
        return "upright"


# ------------------------------------------------- trained ML model

ALL_FEATURES = ("neck_ratio", "head_size_ratio", "shoulder_tilt",
                "head_tilt", "lateral_offset")


def reference_from(samples: list[dict[str, float]]) -> dict[str, float]:
    """Median of upright frames = the user's personal reference."""
    return {k: float(np.median([s[k] for s in samples]))
            for k in ALL_FEATURES}


def model_features(feats: dict[str, float],
                   reference: dict[str, float]) -> np.ndarray:
    """10 inputs: 5 absolute features + 5 relative to the calibration.

    Ratios for the scale-free features and differences for the angles
    turn "how this person sits" into "how far from THEIR upright pose",
    which is what generalises to people the model has never seen.
    """
    raw = [feats[k] for k in ALL_FEATURES]
    rel = [
        feats["neck_ratio"] / (reference["neck_ratio"] + 1e-6),
        feats["head_size_ratio"] / (reference["head_size_ratio"] + 1e-6),
        feats["shoulder_tilt"] - reference["shoulder_tilt"],
        feats["head_tilt"] - reference["head_tilt"],
        feats["lateral_offset"] - reference["lateral_offset"],
    ]
    return np.asarray(raw + rel, dtype=np.float64)


MODEL_FEATURE_NAMES = list(ALL_FEATURES) + [
    "neck_rel", "head_size_rel", "shoulder_tilt_rel", "head_tilt_rel",
    "lateral_rel"]


class PostureModel:
    """A trained scikit-learn classifier (saved by notebook 03) with the
    same calibration and smoothing as used in training."""

    def __init__(self, path):
        import joblib

        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.window = int(bundle.get("window", 15))
        self.labels = list(bundle.get("labels", []))
        self.reference: dict[str, float] | None = None
        self._buffer: deque = deque(maxlen=self.window)

    def calibrate(self, samples: list[dict[str, float]]) -> None:
        if len(samples) < 10:
            raise ValueError("Need at least 10 upright frames")
        self.reference = reference_from(samples)
        self._buffer.clear()

    def predict(self, feats: dict[str, float]) -> str:
        """Rolling mean over the last `window` frames, as in training."""
        if self.reference is None:
            raise RuntimeError("Call calibrate() first")
        self._buffer.append(model_features(feats, self.reference))
        x = np.mean(self._buffer, axis=0)[None]
        pred = self.model.predict(x)[0]
        # The notebook trains on integer targets; map back to names
        if self.labels and isinstance(pred, (int, np.integer)):
            return self.labels[int(pred)]
        return str(pred)
