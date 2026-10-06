"""Builds notebooks/01_eda_posture_blinks.ipynb (kept as code for review).

Markdown is in Russian, code comments in English.
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 01 · EDA: осанка и моргания

**Цель ноутбука:** проверить качество собранных данных, посмотреть на
признаки осанки и частоту морганий и проверить первичные гипотезы:

- **H1** — фиксированный порог EAR плохо переносится между людьми;
- **H2** — признаки, нормированные на ширину плеч, не зависят от
  расстояния до камеры, а «сырые» пиксели — зависят;
- **H3** — частота морганий падает при долгой работе без перерыва.

Данные: `data/raw/` (записи `scripts/record_session.py`). Пока реальных
записей нет, можно запустить на синтетике из
`scripts/make_demo_data.py` — **только для проверки кода, не для
выводов**."""),
    code("""import sys
from pathlib import Path

# Make src/ importable even without `pip install -e .` (see scripts/_bootstrap.py)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score)
from sklearn.preprocessing import StandardScaler

from deskjarvis import config as cfg
from deskjarvis.blink import BlinkDetector, BlinkConfig
from deskjarvis.features import posture_features
from deskjarvis.posture import PostureClassifier

# "raw" = real recordings, "demo" = synthetic data for debugging only
DATA_SOURCE = "demo"
DATA = cfg.DATA_DIR / DATA_SOURCE
sns.set_theme(style="whitegrid")"""),
    md("## 1. Загрузка и качество данных"),
    code("""def load_task(task):
    files = sorted((DATA / task).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No sessions in {DATA / task}")
    return pd.concat([pd.read_parquet(f) for f in files],
                     ignore_index=True)


posture = load_task("posture")
print(posture.shape)
posture.groupby(["subject", "label"]).size().unstack(fill_value=0)"""),
    md("""Что проверяем: баланс классов по людям и долю кадров, где
MediaPipe не нашёл тело. Кадры без позы не выбрасываем молча — это
отдельная характеристика условий съёмки."""),
    code("""POSE_KEYS = [cfg.POSE_NOSE, cfg.POSE_LEFT_EAR, cfg.POSE_RIGHT_EAR,
             cfg.POSE_LEFT_SHOULDER, cfg.POSE_RIGHT_SHOULDER]

has_pose = posture[f"pose_{cfg.POSE_LEFT_SHOULDER}_x"].notna()
print(f"Frames without pose: {1 - has_pose.mean():.1%}")

vis_cols = [f"pose_{i}_v" for i in POSE_KEYS]
posture.loc[has_pose, vis_cols].describe().T[["mean", "min"]]"""),
    md("## 2. Признаки осанки"),
    code("""def row_to_pose(row):
    # Rebuild the (33, 4) array expected by posture_features
    pose = np.zeros((33, 4))
    for i in range(33):
        pose[i] = [row[f"pose_{i}_{c}"] for c in "xyzv"]
    return pose


posture = posture[has_pose].reset_index(drop=True)
feats = pd.DataFrame([posture_features(row_to_pose(r))
                      for _, r in posture.iterrows()])
# Raw pixel measures, used only to test H2
feats["shoulder_w_px"] = np.hypot(
    posture[f"pose_{cfg.POSE_LEFT_SHOULDER}_x"]
    - posture[f"pose_{cfg.POSE_RIGHT_SHOULDER}_x"],
    posture[f"pose_{cfg.POSE_LEFT_SHOULDER}_y"]
    - posture[f"pose_{cfg.POSE_RIGHT_SHOULDER}_y"])
feats["neck_px"] = feats["neck_ratio"] * feats["shoulder_w_px"]
df = pd.concat([posture[["subject", "t_ms", "label"]], feats], axis=1)
df.head()"""),
    code("""FEATURES = ["neck_ratio", "head_size_ratio", "shoulder_tilt",
            "head_tilt", "lateral_offset"]
fig, axes = plt.subplots(1, len(FEATURES), figsize=(18, 4))
for ax, f in zip(axes, FEATURES):
    sns.boxplot(data=df, x="label", y=f, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f)
plt.tight_layout()"""),
    md("""### H2: нормировка убирает зависимость от расстояния до камеры

Ширина плеч в пикселях — прокси расстояния до камеры. Если гипотеза
верна, «сырая» длина шеи в пикселях сильно коррелирует с ней, а
нормированный `neck_ratio` — нет. Смотрим только на ровную посадку,
чтобы не смешивать эффект позы и эффект расстояния."""),
    code("""up = df[df["label"] == "upright"]
h2 = pd.Series({
    "corr(shoulder_w_px, neck_px)": up["shoulder_w_px"].corr(
        up["neck_px"], method="spearman"),
    "corr(shoulder_w_px, neck_ratio)": up["shoulder_w_px"].corr(
        up["neck_ratio"], method="spearman"),
})
h2.round(3)"""),
    md("### Связь признаков с меткой и структура данных"),
    code("""mi = mutual_info_classif(df[FEATURES], df["label"], random_state=0)
pd.Series(mi, index=FEATURES).sort_values().plot.barh(
    title="Mutual information with label")
plt.show()

sns.heatmap(df[FEATURES].corr(method="spearman"), annot=True, fmt=".2f",
            cmap="vlag", center=0)
plt.title("Spearman correlation between features")
plt.show()"""),
    code("""X = StandardScaler().fit_transform(df[FEATURES])
pc = PCA(n_components=2, random_state=0).fit_transform(X)
sample = np.random.default_rng(0).choice(len(df), min(3000, len(df)),
                                         replace=False)
sns.scatterplot(x=pc[sample, 0], y=pc[sample, 1],
                hue=df["label"].iloc[sample], s=8, alpha=0.6)
plt.title("PCA of posture features")
plt.show()"""),
    md("""## 3. Baseline: правила с калибровкой

Для каждого человека калибруемся на **первых 5 секундах** ровной
посадки (как в реальном приложении), затем предсказываем остальные
кадры. Это baseline, который должны обойти ML-модели на следующем
этапе. Метрика — macro-F1: классы несбалансированы, важен каждый."""),
    code("""preds = []
for subject, g in df.groupby("subject"):
    g = g.sort_values("t_ms")
    calib = g[(g["label"] == "upright") & (g["t_ms"] < 5000)]
    clf = PostureClassifier()
    clf.calibrate(calib[FEATURES].to_dict("records"))
    rest = g.drop(calib.index)
    preds.append(rest.assign(
        pred=[clf.predict(r) for r in rest[FEATURES].to_dict("records")]))
preds = pd.concat(preds)

print(classification_report(preds["label"], preds["pred"], digits=3))
print("Macro-F1 by subject:")
print(preds.groupby("subject").apply(
    lambda g: f1_score(g["label"], g["pred"], average="macro")).round(3))
ConfusionMatrixDisplay.from_predictions(preds["label"], preds["pred"],
                                        xticks_rotation=45,
                                        normalize="true")
plt.show()"""),
    md("## 4. Моргания"),
    code("""blinks = load_task("blink").sort_values(["subject", "t_ms"])
blinks.groupby("subject")["ear"].describe()[["mean", "50%", "min"]]"""),
    md("""### H1: один порог для всех или адаптивный?

Сравниваем классический фиксированный порог EAR = 0.2 и адаптивный
детектор. Опорный сигнал — оценка моргания самого MediaPipe
(`eyeBlink*`), бинаризованная по 0.5. На реальных данных опорой будет
ручная разметка и датасеты Eyeblink8 / RT-BENE; здесь это грубая
проверка согласованности."""),
    code("""def count_events(mask):
    # Number of False->True transitions
    m = np.asarray(mask, dtype=int)
    return int(np.sum(np.diff(m) == 1))


rows = []
for subject, g in blinks.groupby("subject"):
    ear, t = g["ear"].to_numpy(), g["t_ms"].to_numpy()
    reference = (g[["eyeBlinkLeft", "eyeBlinkRight"]].mean(axis=1) > 0.5)
    det = BlinkDetector(BlinkConfig())
    for e, ts in zip(ear, t):
        det.update(float(e), int(ts))
    rows.append({"subject": subject, "open_ear": np.median(ear),
                 "reference": count_events(reference),
                 "fixed_0.2": count_events(ear < 0.2),
                 "adaptive": det.total_blinks})
pd.DataFrame(rows).round(3)"""),
    md("### H3: как меняется частота морганий за сессию"),
    code("""per_min = []
for subject, g in blinks.groupby("subject"):
    det = BlinkDetector()
    fired = [det.update(float(e), int(ts))
             for e, ts in zip(g["ear"], g["t_ms"])]
    minute = (g["t_ms"] // 60_000).to_numpy()
    per_min.append(pd.DataFrame({"subject": subject, "minute": minute,
                                 "blink": fired})
                   .groupby(["subject", "minute"])["blink"].sum()
                   .reset_index())
per_min = pd.concat(per_min)
sns.lineplot(data=per_min, x="minute", y="blink", hue="subject",
             marker="o")
plt.ylabel("blinks per minute")
plt.title("Blink rate over the session")
plt.show()"""),
    md("""## Выводы (заполнить после реальных данных)

- Баланс классов и качество детекции: …
- H1: …
- H2: …
- H3: …
- Что улучшить в сборе данных: …"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3",
                                "display_name": "Python 3",
                                "language": "python"}
nbf.write(nb, "notebooks/01_eda_posture_blinks.ipynb")
print("notebook written")
