import numpy as np
import pytest

from deskjarvis import config as cfg
from deskjarvis.features import (eye_aspect_ratio, face_ear, normalize_hand,
                                 posture_features)


def make_eye(width=30.0, opening=9.0):
    """Six points p1..p6 of an eye centred at the origin."""
    h = opening / 2
    return np.array([
        [-width / 2, 0], [-width / 6, -h], [width / 6, -h],
        [width / 2, 0], [width / 6, h], [-width / 6, h],
    ])


def test_ear_value_and_scale_invariance():
    ear = eye_aspect_ratio(make_eye(30, 9))
    assert ear == pytest.approx(0.3)
    # Same eye twice as large (closer to the camera) -> same EAR.
    assert eye_aspect_ratio(make_eye(60, 18)) == pytest.approx(ear)


def test_ear_drops_when_closed():
    assert eye_aspect_ratio(make_eye(30, 2)) < 0.1


def test_face_ear_uses_both_eyes():
    face = np.zeros((478, 3))
    face[list(cfg.RIGHT_EYE_IDX), :2] = make_eye(30, 9) + [100, 100]
    face[list(cfg.LEFT_EYE_IDX), :2] = make_eye(30, 3) + [200, 100]
    assert face_ear(face) == pytest.approx((0.3 + 0.1) / 2)


def make_pose(shoulder_w=200.0, neck=120.0, ear_w=80.0, tilt_px=0.0):
    pose = np.zeros((33, 4))
    cx, sy = 320.0, 400.0
    pose[cfg.POSE_LEFT_SHOULDER, :2] = [cx + shoulder_w / 2, sy + tilt_px]
    pose[cfg.POSE_RIGHT_SHOULDER, :2] = [cx - shoulder_w / 2, sy]
    pose[cfg.POSE_LEFT_EAR, :2] = [cx + ear_w / 2, sy - neck]
    pose[cfg.POSE_RIGHT_EAR, :2] = [cx - ear_w / 2, sy - neck]
    pose[cfg.POSE_NOSE, :2] = [cx, sy - neck - 10]
    return pose


def test_posture_features_are_scale_invariant():
    near = posture_features(make_pose(200, 120, 80))
    far = posture_features(make_pose(100, 60, 40))
    for key in near:
        assert near[key] == pytest.approx(far[key], abs=1e-6)
    assert near["neck_ratio"] == pytest.approx(0.6)
    assert near["head_size_ratio"] == pytest.approx(0.4)


def test_shoulder_tilt_sign_and_mirroring():
    tilt = posture_features(make_pose(tilt_px=35))["shoulder_tilt"]
    assert tilt == pytest.approx(np.degrees(np.arctan2(35, 200)), abs=1e-4)
    # Mirrored frame (x flipped) gives the same magnitude.
    mirrored = make_pose(tilt_px=35)
    mirrored[:, 0] = 640 - mirrored[:, 0]
    assert abs(posture_features(mirrored)["shoulder_tilt"]) == pytest.approx(
        abs(tilt), abs=1e-4
    )


def test_normalize_hand_invariance():
    rng = np.random.default_rng(0)
    hand = rng.uniform(0, 100, size=(21, 3))
    base = normalize_hand(hand)
    shifted_scaled = hand * 2.5 + [300, 50, 0]
    assert base.shape == (63,)
    np.testing.assert_allclose(normalize_hand(shifted_scaled), base,
                               atol=1e-9)
    assert np.allclose(base[:3], 0)  # wrist at the origin
