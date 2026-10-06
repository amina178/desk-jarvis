import numpy as np
import pytest

from deskjarvis import config as cfg
from deskjarvis.pipeline import Analyzer, Settings
from deskjarvis.video import (REPORT_COLUMNS, analyze_video,
                              blinks_per_minute, summarize, write_test_video)

needs_models = pytest.mark.skipif(
    not (cfg.MODELS_DIR / "face.task").exists(),
    reason="run scripts/download_models.py first")


@needs_models
def test_video_with_a_real_face(tmp_path):
    data = pytest.importorskip("skimage.data")
    frame = np.ascontiguousarray(data.astronaut())
    path = write_test_video(str(tmp_path / "v.mp4"), frame, n_frames=90)
    df = analyze_video(path, Analyzer(settings=Settings(calibration_s=1)))
    assert list(df.columns) == REPORT_COLUMNS and len(df) == 90
    assert df.ear.notna().all() and 0.2 < df.ear.median() < 0.4
    assert (df.posture_status == "ok").any()
    s = summarize(df)
    assert s["face_found"] == 1.0 and s["posture_shares"] == {"upright": 1.0}
    assert blinks_per_minute(df).blinks.sum() == 0


@needs_models
def test_video_without_people(tmp_path):
    frame = np.full((240, 320, 3), 127, dtype=np.uint8)
    path = write_test_video(str(tmp_path / "empty.mp4"), frame, n_frames=20)
    df = analyze_video(path, Analyzer())
    assert len(df) == 20 and df.ear.isna().all()
    assert (df.posture_status == "not_visible").all()
