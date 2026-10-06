import numpy as np
import pytest

from deskjarvis.blink import BlinkDetector
from deskjarvis.events import BreakTimer, GestureDebouncer, PersistentState
from deskjarvis.posture import PostureClassifier

FPS = 30
DT = 1000 // FPS


def ear_stream(n_frames, blink_starts, open_ear=0.30, blink_len=4,
               noise=0.01, seed=0):
    """Synthetic EAR signal: open level + noise, with blink dips."""
    rng = np.random.default_rng(seed)
    ear = open_ear + rng.normal(0, noise, n_frames)
    for s in blink_starts:
        ear[s:s + blink_len] = open_ear * 0.3
    return ear


def run_detector(ears, detector=None):
    det = detector or BlinkDetector()
    fired = [det.update(float(e), i * DT) for i, e in enumerate(ears)]
    return det, fired


def test_counts_blinks():
    starts = [100, 200, 300, 400, 500]
    det, _ = run_detector(ear_stream(600, starts))
    assert det.total_blinks == len(starts)


@pytest.mark.parametrize("open_ear", [0.22, 0.30, 0.38])
def test_adaptive_threshold_works_for_different_eyes(open_ear):
    # A fixed 0.2 threshold would miss blinks when open_ear is high and
    # fire constantly when it is low; the adaptive one handles both.
    det, _ = run_detector(ear_stream(600, [150, 300, 450], open_ear))
    assert det.total_blinks == 3


def test_long_closure_is_not_a_blink():
    det, _ = run_detector(ear_stream(400, [150], blink_len=30))  # 1 s
    assert det.total_blinks == 0


def test_blinks_per_minute():
    starts = list(range(100, 1800, 150))  # every 5 s after warm-up
    det, _ = run_detector(ear_stream(1800, starts))
    rate = det.blinks_per_minute(1800 * DT)
    assert rate == pytest.approx(12, abs=1)


def upright(noise=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return {
        "neck_ratio": 0.60 + rng.normal(0, noise),
        "head_size_ratio": 0.40 + rng.normal(0, noise),
        "shoulder_tilt": 0.0 + rng.normal(0, noise * 10),
    }


def test_posture_rules():
    clf = PostureClassifier()
    rng = np.random.default_rng(1)
    clf.calibrate([upright(0.01, rng) for _ in range(30)])
    assert clf.predict(upright()) == "upright"
    assert clf.predict({**upright(), "neck_ratio": 0.45}) == "slouch"
    assert clf.predict({**upright(), "head_size_ratio": 0.48}) == (
        "forward_head"
    )
    assert clf.predict({**upright(), "shoulder_tilt": 12.0}) == "side_lean"


def test_posture_requires_calibration():
    with pytest.raises(RuntimeError):
        PostureClassifier().predict(upright())


def test_debouncer_fires_once_per_hold():
    deb = GestureDebouncer(window=8, min_votes=6, cooldown_s=0.5)
    labels = ["no_gesture"] * 10 + ["palm"] * 40 + ["no_gesture"] * 20
    events = [deb.update(lb, i * DT) for i, lb in enumerate(labels)]
    assert [e for e in events if e] == ["palm"]


def test_debouncer_ignores_flicker():
    deb = GestureDebouncer(window=8, min_votes=6)
    labels = ["palm", "no_gesture", "no_gesture"] * 30  # noisy 1-in-3
    events = [deb.update(lb, i * DT) for i, lb in enumerate(labels)]
    assert not any(events)


def test_debouncer_refires_after_release():
    deb = GestureDebouncer(window=4, min_votes=3, cooldown_s=0.2)
    one_swipe = ["swipe_right"] * 10 + ["no_gesture"] * 10
    events = [deb.update(lb, i * DT) for i, lb in enumerate(one_swipe * 2)]
    assert [e for e in events if e] == ["swipe_right", "swipe_right"]


def test_persistent_state():
    ps = PersistentState(hold_s=2)
    assert ps.update("slouch", 0) == ("slouch", False)
    assert ps.update("slouch", 2500) == ("slouch", True)
    assert ps.update("upright", 2600) == ("upright", False)


def test_break_timer():
    bt = BreakTimer(work_minutes=1, reset_after_s=10)
    fired = [bt.update(True, t) for t in range(0, 70_000, 1000)]
    assert sum(fired) == 1  # reminded once, not every frame
    # Short absence does not reset; long absence does.
    bt.update(False, 71_000)
    bt.update(False, 90_000)
    assert bt.update(True, 91_000) is False
    assert any(bt.update(True, t) for t in range(92_000, 160_000, 1000))


def test_posture_model_bundle(tmp_path):
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression

    from deskjarvis.posture import (ALL_FEATURES, PostureModel,
                                    model_features, reference_from)

    labels = ["upright", "slouch"]
    rng = np.random.default_rng(0)
    base = {k: 0.5 for k in ALL_FEATURES}
    ref = reference_from([base] * 10)
    X, y = [], []
    for _ in range(100):
        up = {k: v + rng.normal(0, 0.01) for k, v in base.items()}
        sl = {**up, "neck_ratio": up["neck_ratio"] * 0.7}
        X += [model_features(up, ref), model_features(sl, ref)]
        y += [0, 1]
    clf = LogisticRegression(max_iter=1000).fit(np.array(X), y)
    joblib.dump({"model": clf, "window": 3, "labels": labels},
                tmp_path / "model.joblib")

    model = PostureModel(tmp_path / "model.joblib")
    with pytest.raises(RuntimeError):
        model.predict(base)
    model.calibrate([base] * 10)
    assert model.predict(base) == "upright"
    slouch = {**base, "neck_ratio": 0.35}
    preds = [model.predict(slouch) for _ in range(3)]  # window fills up
    assert preds[-1] == "slouch"
