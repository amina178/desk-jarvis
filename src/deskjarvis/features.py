"""Pure-numpy feature engineering on landmark arrays.

All functions expect landmarks in PIXEL coordinates, shape (N, 2) or
(N, 3). MediaPipe returns x normalised by image width and y by image
height, so on a 16:9 frame one unit of x is longer than one unit of y.
Distances and angles computed on raw normalised values are therefore
distorted; `landmarks.py` converts to pixels before anything reaches
this module.
"""
from __future__ import annotations

import numpy as np

from deskjarvis import config as cfg

_EPS = 1e-6


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


# ---------------------------------------------------------------- eyes


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).

    Open eye ~0.25-0.35, closed eye ~0.05-0.15. Dividing by the eye width
    makes the value independent of the distance to the camera.
    """
    p1, p2, p3, p4, p5, p6 = eye[:, :2]
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    return vertical / (horizontal + _EPS)


def face_ear(face: np.ndarray) -> float:
    """Mean EAR of both eyes from a full Face Mesh (478 points)."""
    right = eye_aspect_ratio(face[list(cfg.RIGHT_EYE_IDX)])
    left = eye_aspect_ratio(face[list(cfg.LEFT_EYE_IDX)])
    return (right + left) / 2.0


# ------------------------------------------------------------- posture


def posture_features(pose: np.ndarray) -> dict[str, float]:
    """Scale-invariant posture descriptors from a frontal camera.

    Everything is divided by shoulder width so that sitting closer to or
    further from the camera does not change the values (hypothesis H2).

    neck_ratio       vertical ear-to-shoulder gap; drops when slouching
    head_size_ratio  ear-to-ear width; grows when the head moves forward
    shoulder_tilt    degrees; non-zero when leaning to one side
    head_tilt        degrees of the ear line
    lateral_offset   nose offset from the shoulder midpoint
    """
    l_sh = pose[cfg.POSE_LEFT_SHOULDER, :2]
    r_sh = pose[cfg.POSE_RIGHT_SHOULDER, :2]
    l_ear = pose[cfg.POSE_LEFT_EAR, :2]
    r_ear = pose[cfg.POSE_RIGHT_EAR, :2]
    nose = pose[cfg.POSE_NOSE, :2]

    shoulder_w = _dist(l_sh, r_sh) + _EPS
    mid_sh = (l_sh + r_sh) / 2.0
    mid_ear = (l_ear + r_ear) / 2.0

    # Image y grows downwards, so shoulders have the larger y.
    neck_ratio = (mid_sh[1] - mid_ear[1]) / shoulder_w
    head_size_ratio = _dist(l_ear, r_ear) / shoulder_w
    return {
        "neck_ratio": float(neck_ratio),
        "head_size_ratio": float(head_size_ratio),
        "shoulder_tilt": _line_angle(r_sh, l_sh),
        "head_tilt": _line_angle(r_ear, l_ear),
        "lateral_offset": float((nose[0] - mid_sh[0]) / shoulder_w),
    }


def _line_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle of segment a->b to the horizontal, in degrees.

    The result is folded into (-90, 90] so that it does not depend on
    which side of the body the camera sees as "left" (mirrored video).
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle


# ---------------------------------------------------------------- hand


def normalize_hand(hand: np.ndarray) -> np.ndarray:
    """Translation- and scale-invariant hand vector of length 63.

    1. Subtract the wrist -> position in the frame no longer matters.
    2. Divide by wrist-to-middle-MCP distance (palm size) -> distance to
       the camera no longer matters.
    Rotation is deliberately kept: "thumb up" and "thumb down" differ
    only by rotation.
    """
    pts = hand[:, :3].astype(float).copy()
    pts -= pts[cfg.HAND_WRIST]
    palm = np.linalg.norm(pts[cfg.HAND_MIDDLE_MCP, :2]) + _EPS
    return (pts / palm).reshape(-1)
