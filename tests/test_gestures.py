import json

import numpy as np
import pytest

from deskjarvis.actions import ActionRunner
from deskjarvis.gestures import (GESTURE_ACTIONS, GESTURES, crop_input,
                                 landmark_bbox, landmark_input, softmax)


def fake_hand(seed=0, offset=(300, 200), scale=80.0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (21, 3)) * scale
    pts[:, :2] += offset
    return pts


def test_landmark_input_shape_and_dtype():
    x = landmark_input(fake_hand())
    assert x.shape == (63,) and x.dtype == np.float32


def test_mirror_flips_x_only():
    hand = fake_hand()
    a, b = landmark_input(hand), landmark_input(hand, mirror=True)
    a, b = a.reshape(21, 3), b.reshape(21, 3)
    np.testing.assert_allclose(a[:, 0], -b[:, 0], atol=1e-6)
    np.testing.assert_allclose(a[:, 1:], b[:, 1:], atol=1e-6)


def test_landmark_bbox_is_clipped_square():
    hand = fake_hand(offset=(5, 5))  # near the corner
    left, top, right, bottom = landmark_bbox(hand, 640, 480, margin=0.25)
    assert left == 0 and top == 0
    assert right <= 640 and bottom <= 480
    centre = fake_hand(offset=(300, 200))
    l2, t2, r2, b2 = landmark_bbox(centre, 640, 480)
    assert abs((r2 - l2) - (b2 - t2)) <= 1


def test_crop_input_normalised_chw():
    rgb = np.full((480, 640, 3), 124, dtype=np.uint8)
    x = crop_input(rgb, (100, 100, 260, 260), size=64)
    assert x.shape == (3, 64, 64) and x.dtype == np.float32
    assert abs(float(x[0].mean())) < 0.05  # 124/255 ~ ImageNet red mean


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1000.0, 1000.0, 0.0]]))  # no overflow
    assert p.sum() == pytest.approx(1.0) and p[0, 0] == pytest.approx(0.5)


def test_actions_are_known_and_dry_run_is_safe():
    runner = ActionRunner(dry_run=True)
    for gesture, action in GESTURE_ACTIONS.items():
        assert gesture in GESTURES
        assert runner.run(action).startswith("[dry-run]")
    with pytest.raises(ValueError):
        runner.run("format_disk")


def test_classifier_end_to_end_with_onnx(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from deskjarvis.gestures import GestureClassifier
    from deskjarvis.training import LandmarkMLP, export_onnx

    torch.manual_seed(0)
    model = LandmarkMLP(len(GESTURES)).eval()
    export_onnx(model, torch.zeros(2, 63), tmp_path / "model.onnx")
    (tmp_path / "labels.json").write_text(json.dumps(
        {"labels": list(GESTURES), "input": "landmarks"}))

    clf = GestureClassifier(tmp_path, min_confidence=0.0)
    pred = clf.predict(fake_hand())
    assert pred.label in GESTURES and 0 <= pred.confidence <= 1
    # Same answer as PyTorch -> the export preserved the model.
    with torch.no_grad():
        ref = model(torch.from_numpy(landmark_input(fake_hand())[None]))
    assert GESTURES[int(ref.argmax())] == pred.label
    # Unsure -> idle.
    strict = GestureClassifier(tmp_path, min_confidence=0.99)
    assert strict.predict(fake_hand()).label == "no_gesture"
    assert clf.predict(None).label == "no_gesture"
