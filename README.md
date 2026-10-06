# Desk Jarvis

A webcam-based desk assistant: **hand-gesture shortcuts**, **posture
monitoring** and **eye-fatigue (blink-rate) tracking**. Frames are processed
locally and never stored.

> Course project, AI Engineer — Computer Vision.

[![CI](https://github.com/amina178/desk-jarvis/actions/workflows/ci.yml/badge.svg)](https://github.com/amina178/desk-jarvis/actions/workflows/ci.yml)

| Gesture | Action |
|---|---|
| 👍 `like` | next slide (→) |
| 👎 `dislike` | previous slide (←) |
| ✋ `palm` | toggle mute (Zoom hotkey) |
| ✌️ `peace` | screenshot |

`fist`, `one` and `no_gesture` trigger nothing: they teach the model what
*not* to react to.

## How it works

```
webcam ─► MediaPipe (frozen, pre-trained keypoint extractor)
            ├─ hand 21 pts ─► gesture classifier (ONNX) ─► debouncer ─► action
            ├─ pose 33 pts ─► posture features ─► rules | ML model (+ calibration)
            └─ face 478 pts ─► Eye Aspect Ratio ─► adaptive blink detector
                                                  └─► break reminders
```

MediaPipe plays the role of a pre-trained backbone. The models trained in
this project sit on top of it (classifiers on keypoints) and next to it
(CNNs on hand crops). All per-frame logic lives in
`src/deskjarvis/pipeline.py` and is shared by the live demo and the web app.

## Models compared

| Task | Baseline | From scratch | Fine-tuned | Notebook |
|---|---|---|---|---|
| Gestures (HaGRID, split by user) | — | MLP on 63 keypoint coords; SmallCNN on crops | MobileNetV3-Small (ImageNet) | `02_gesture_models.ipynb` |
| Posture (own recordings; LOSO, or blocked time CV for one person) | calibrated rules | LogReg, HistGradientBoosting, MLP | — | `03_posture_models.ipynb` |
| Blinks | fixed EAR threshold | adaptive EAR threshold | — (MediaPipe blendshape as reference) | `01_eda_posture_blinks.ipynb` |

Optimisation (notebook 02): ONNX export, dynamic INT8 quantisation,
global L1 pruning of MobileNet (30 % / 50 %) with fine-tuning.

## Results

**Gestures** — HaGRID test set: 651 people unseen in training, 1 866 hands
(`results/gesture_metrics.json`).

| Model | macro-F1, hand found | macro-F1, system | CPU latency | Size |
|---|---|---|---|---|
| MLP on keypoints (scratch) | 0.974 | 0.908 | 0.02 ms | 0.2 MB |
| SmallCNN on crops (scratch) | 0.970 | 0.957 | 20.8 ms | 4.7 MB |
| **MobileNetV3-Small (fine-tuned)** | **0.984** | **0.976** | 1.4 ms | 6.1 MB |

"System" counts hands MediaPipe missed (12.4 %) as *no gesture*. The live
app uses the MLP: its crops would also come from MediaPipe keypoints, and it
is ~70× faster than MobileNet for −0.01 F1.

**Out of domain** — own webcam, 3 min, never seen in training
(`results/gesture_own_webcam.json`): macro-F1 **0.981** on frames with a
hand; after the debouncer **6/6** gestures fired correctly with **0** false
commands. Limitation: *no_gesture* was recorded without a hand in frame.

**Optimisation** — pruning 30 % keeps F1 (0.977) and shrinks the
compressed model by 22 %; 50 % drops F1 to 0.914. Dynamic INT8 works for the
MLP (3.4× smaller, same F1) but **breaks MobileNetV3** (F1 0.055, 12×
slower): its hard-swish / squeeze-excitation layers need static, calibrated
quantisation.

**Posture** — one person, 8 k frames, blocked time CV, 5 folds
(`results/posture_metrics.json`):

| Model | macro-F1 | worst fold |
|---|---|---|
| Calibrated rules | 0.46 | 0.28 |
| Logistic Regression | 0.55 | 0.43 |
| **HistGradientBoosting** | **0.80 ± 0.10** | 0.62 |
| MLP | 0.78 ± 0.07 | 0.67 |

Hand-set thresholds miss real posture changes (slouching lowers the
neck ratio by 12 %, the rule needs 15 %). The same rules scored 0.98 on
synthetic data — a reminder not to evaluate on generated data. With one
participant this measures stability over time, not transfer to new people;
leave-one-subject-out switches on automatically when more people are
recorded.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/download_models.py        # MediaPipe .task files
python scripts/live_demo.py              # actions only shown on screen
python scripts/live_demo.py --actions    # really press keys (pip install pyautogui)
streamlit run app/streamlit_app.py       # web app: video report + live tab
```

Trained models are picked up automatically when present:
`models/gestures/mlp/` (from notebook 02) and `models/posture/model.joblib`
(from notebook 03). Without them the app runs the baselines.

macOS: allow the terminal app in *System Settings → Privacy & Security →
Camera* (and *Accessibility* for `--actions`). MediaPipe is pinned to
0.10.35 because 1.0.1 crashes on Apple Silicon
([issue #6356](https://github.com/google-ai-edge/mediapipe/issues/6356)).

## Reproducing the results

1. **Gestures** — open `notebooks/02_gesture_models.ipynb` in Google Colab
   (T4 GPU) and run all cells. It downloads the
   [HaGRID 30k sample](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p),
   runs `scripts/prepare_hagrid.py` (MediaPipe landmarks + hand crops +
   user-level split), trains and compares the three models, and downloads
   `gesture_models.zip` — unzip it into the project root.
2. **Posture** — record sessions (below), then run
   `notebooks/03_posture_models.ipynb` locally.
3. **EDA** — `notebooks/01_eda_posture_blinks.ipynb`.

Seeds are fixed (42); splits are saved in `meta.csv`; metrics are written to
`results/*.json`.

## Collecting data

```bash
python scripts/record_session.py --task posture --subject s01   # keys 1-4
python scripts/record_session.py --task gesture --subject s01   # keys 0-6
python scripts/record_session.py --task blink   --subject s01
```

Click the video window, choose a label with the number key, `space` starts or
pauses recording, `q` saves. Only landmarks are saved (`data/raw/<task>/`),
never images.

Posture protocol per person (4–5 min): ~10 s upright first (calibration),
then about a minute of each class; move naturally (type, drink, turn the
head); keep both shoulders in frame and hands away from them. Record at
least 3 people — the posture model is evaluated on people it has not seen.

To try the notebooks before recording: `python scripts/make_demo_data.py`
(synthetic, `data/demo/`, for code checks only).

## Deployment

- **Docker**: `docker build -t desk-jarvis . && docker run -p 7860:7860 desk-jarvis`,
  then open http://localhost:7860.
- **Hugging Face Spaces**: `.github/workflows/sync-to-hf.yml` pushes the app to
  the Space `<HF_USERNAME>/desk-jarvis` (Docker SDK) on every push to `main`.
  One-time setup: create the empty Space, add the repository secret
  `HF_TOKEN` (write token) and the variable `HF_USERNAME`.
- **CI**: `.github/workflows/ci.yml` runs flake8 and pytest (with the real
  MediaPipe models) on every push.

## Project layout

```
src/deskjarvis/
  config.py      paths, landmark indices, labels
  landmarks.py   MediaPipe Tasks wrapper -> pixel-space numpy arrays
  features.py    EAR, posture features, hand normalisation
  blink.py       adaptive-threshold blink detector
  posture.py     calibrated rules + trained-model wrapper
  gestures.py    shared gesture preprocessing + ONNX classifier
  events.py      gesture debouncer, persistent state, break timer
  actions.py     dry-run / real keyboard actions
  pipeline.py    per-frame logic shared by all UIs
  video.py       offline video analysis for the web app
  training.py    PyTorch models, datasets, training loop, ONNX export
app/             Streamlit app
scripts/         models, recording, data preparation, live demo, notebooks
notebooks/       01 EDA · 02 gesture models (Colab) · 03 posture models
tests/           pytest (synthetic data + real MediaPipe on a sample photo)
```

## Development

```bash
pip install -r requirements-dev.txt
pytest
flake8 src scripts tests app
```

## Data and licences

HaGRID: CC BY-SA 4.0 (SberDevices); the gesture models in `models/gestures/`
are trained on it. Own recordings contain landmarks only, are not published
(`data/raw/` is git-ignored) and were collected with consent.
