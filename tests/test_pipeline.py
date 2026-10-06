import numpy as np

from deskjarvis import config as cfg
from deskjarvis.gestures import Prediction
from deskjarvis.landmarks import FrameLandmarks
from deskjarvis.pipeline import Analyzer, Settings

DT = 33


def pose(neck=120.0, visibility=0.99):
    p = np.zeros((33, 4))
    p[:, 3] = visibility
    p[cfg.POSE_LEFT_SHOULDER, :2] = [420, 400]
    p[cfg.POSE_RIGHT_SHOULDER, :2] = [220, 400]
    p[cfg.POSE_LEFT_EAR, :2] = [360, 400 - neck]
    p[cfg.POSE_RIGHT_EAR, :2] = [280, 400 - neck]
    p[cfg.POSE_NOSE, :2] = [320, 390 - neck]
    return p


def face(opening=9.0):
    f = np.zeros((478, 3))
    h = opening / 2
    eye = np.array([[-15, 0], [-5, -h], [5, -h], [15, 0], [5, h], [-5, h]])
    f[list(cfg.RIGHT_EYE_IDX), :2] = eye + [280, 250]
    f[list(cfg.LEFT_EYE_IDX), :2] = eye + [360, 250]
    return f


class StubGestures:
    """Replays a fixed label sequence instead of running a model."""

    def __init__(self, labels):
        self.labels = iter(labels)

    def predict(self, hand, rgb=None):
        return Prediction(next(self.labels, "no_gesture"), 0.95)


def run(an, frames):
    return [an.step(fr) for fr in frames]


def test_calibration_then_bad_posture_after_hold():
    an = Analyzer(settings=Settings(calibration_s=1, bad_posture_hold_s=2))
    frames = [FrameLandmarks(i * DT, pose=pose(120), face=face())
              for i in range(60)]  # 2 s upright
    frames += [FrameLandmarks((60 + i) * DT, pose=pose(80), face=face())
               for i in range(90)]  # 3 s slouching
    states = run(an, frames)
    assert states[0].posture_status == "calibrating"
    assert states[59].posture == "upright" and not states[59].posture_bad
    assert states[-1].posture == "slouch" and states[-1].posture_bad
    # Bad posture is reported only after it was held for 2 s
    first_bad = next(i for i, s in enumerate(states) if s.posture_bad)
    assert (first_bad - 60) * DT >= 2000


def test_low_visibility_is_not_judged():
    an = Analyzer()
    st = an.step(FrameLandmarks(0, pose=pose(visibility=0.2)))
    assert st.posture_status == "not_visible" and st.posture is None


def test_blinks_are_counted():
    an = Analyzer()
    frames = []
    for i in range(300):
        closed = i in range(150, 154) or i in range(250, 254)
        frames.append(FrameLandmarks(i * DT, face=face(2 if closed else 9)))
    states = run(an, frames)
    assert states[-1].blinks_total == 2
    assert sum(s.blinked for s in states) == 2


def test_held_gesture_fires_one_dry_run_action():
    labels = ["no_gesture"] * 10 + ["like"] * 40 + ["no_gesture"] * 10
    an = Analyzer(gesture_model=StubGestures(labels))
    hand = np.zeros((21, 3))
    states = run(an, [FrameLandmarks(i * DT, hand=hand)
                      for i in range(len(labels))])
    fired = [s for s in states if s.action_text]
    assert len(fired) == 1
    assert fired[0].event == "like"
    assert fired[0].action_text.startswith("[dry-run] Next slide")
