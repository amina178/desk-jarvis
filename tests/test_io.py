import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deskjarvis.landmarks import FrameLandmarks, LandmarkExtractor

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_row_flattens_all_modalities():
    rec = load_script("record_session")
    res = FrameLandmarks(
        timestamp_ms=33,
        hand=np.ones((21, 3)), handedness="Right",
        pose=np.ones((33, 4)), face=np.ones((478, 3)),
        blendshapes={"eyeBlinkLeft": 0.1},
    )
    row = rec.frame_row(res, "palm", "s01", "sess")
    assert row["label"] == "palm" and row["subject"] == "s01"
    assert "pose_32_v" in row and "hand_20_z" in row
    assert "ear" in row and row["eyeBlinkLeft"] == 0.1


def test_extractor_explains_missing_models(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models"):
        LandmarkExtractor(models_dir=tmp_path)
