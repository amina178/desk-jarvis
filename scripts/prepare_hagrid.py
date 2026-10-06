"""Turn a HaGRID sample into the two training sets of this project.

For every annotated hand box of the selected classes:
  1. MediaPipe HandLandmarker runs on a generous crop around the box
     -> 21 landmarks (input of the MLP). Same detector as the live app.
  2. A square hand crop is saved as JPEG (input of the CNNs). When
     landmarks were found, the crop is built from them with
     `landmark_bbox` - exactly like the live app does - so that training
     and serving see the same kind of crop.
  3. Users are split into train / val / test (70 / 15 / 15 %) by
     `user_id`: no person appears in two splits.

Usage (Colab):
    python scripts/prepare_hagrid.py --src /content/hagrid --out /content/gd

Output in --out:
    meta.csv          one row per hand: id, label, user_id, split, ...
    landmarks.npy     (N, 21, 3) float32, NaN where MediaPipe missed
    crops/<label>/<id>.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision
from sklearn.model_selection import GroupShuffleSplit

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from deskjarvis import config as cfg
from deskjarvis.gestures import GESTURES, landmark_bbox

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def load_annotations(src: Path) -> dict:
    """Merge every HaGRID-style JSON ({image_id: {bboxes, labels,
    user_id, ...}}) found under src."""
    merged = {}
    for path in sorted(src.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict):
            merged.update({k: v for k, v in data.items()
                           if isinstance(v, dict) and "bboxes" in v})
    return merged


def index_images(src: Path) -> dict[str, Path]:
    return {p.stem: p for p in src.rglob("*")
            if p.suffix.lower() in IMAGE_EXT}


def build_records(ann: dict, images: dict, classes: set[str],
                  max_per_class: int, seed: int) -> pd.DataFrame:
    rows = []
    for image_id, a in ann.items():
        if image_id not in images:
            continue
        for k, (box, label) in enumerate(zip(a["bboxes"], a["labels"])):
            if label in classes:
                rows.append({"id": f"{image_id}_{k}", "image": str(
                    images[image_id]), "label": label,
                    "user_id": a.get("user_id", image_id),
                    "bx": box[0], "by": box[1], "bw": box[2], "bh": box[3]})
    df = pd.DataFrame(rows)
    # Cap the classes so that no_gesture (from every image) cannot
    # dominate; sampling is seeded for reproducibility.
    parts = [g.sample(min(len(g), max_per_class), random_state=seed)
             for _, g in df.groupby("label")]
    return pd.concat(parts).reset_index(drop=True)


def split_by_user(df: pd.DataFrame, seed: int) -> pd.Series:
    split = pd.Series("train", index=df.index)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    _, rest = next(gss.split(df, groups=df["user_id"]))
    rest_df = df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val, test = next(gss2.split(rest_df, groups=rest_df["user_id"]))
    split.iloc[rest[val]] = "val"
    split.iloc[rest[test]] = "test"
    return split


def square_box(cx, cy, side, w, h):
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
            int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", nargs="+", default=list(GESTURES))
    ap.add_argument("--max-per-class", type=int, default=2500)
    ap.add_argument("--crop-size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hand-model", type=Path,
                    default=cfg.MODELS_DIR / "hand.task")
    args = ap.parse_args()

    ann = load_annotations(args.src)
    images = index_images(args.src)
    print(f"annotations: {len(ann)}  images: {len(images)}")
    df = build_records(ann, images, set(args.classes), args.max_per_class,
                       args.seed)
    if df.empty:
        raise SystemExit("No matching boxes: check --src and --classes")
    df["split"] = split_by_user(df, args.seed)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.hand_model)),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=0.3))

    landmarks = np.full((len(df), 21, 3), np.nan, dtype=np.float32)
    has_lm, crop_paths = [], []
    for i, row in enumerate(df.itertuples()):
        bgr = cv2.imread(row.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bx, by, bw, bh = row.bx * w, row.by * h, row.bw * w, row.bh * h
        cx, cy, side = bx + bw / 2, by + bh / 2, max(bw, bh)

        # Detector needs context around the hand: 2x the box.
        dl, dt, dr, db = square_box(cx, cy, side * 2.0, w, h)
        region = np.ascontiguousarray(rgb[dt:db, dl:dr])
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=region))
        found = bool(res.hand_landmarks)
        if found:
            rh, rw = region.shape[:2]
            pts = np.array([[lm.x * rw + dl, lm.y * rh + dt, lm.z * rw]
                            for lm in res.hand_landmarks[0]])
            landmarks[i] = pts
            box = landmark_bbox(pts, w, h)
        else:
            box = square_box(cx, cy, side * 1.2, w, h)
        has_lm.append(found)

        crop = cv2.resize(rgb[box[1]:box[3], box[0]:box[2]],
                          (args.crop_size, args.crop_size),
                          interpolation=cv2.INTER_AREA)
        out = args.out / "crops" / row.label / f"{row.id}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        crop_paths.append(str(out))
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(df)} processed")
    detector.close()

    df["has_landmarks"] = has_lm
    df["crop_path"] = crop_paths
    args.out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["image"]).to_csv(args.out / "meta.csv", index=False)
    np.save(args.out / "landmarks.npy", landmarks)
    print(f"MediaPipe found a hand in {np.mean(has_lm):.1%} of boxes")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
