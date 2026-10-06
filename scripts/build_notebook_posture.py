"""Builds notebooks/03_posture_models.ipynb (runs locally, CPU is enough).

Markdown in Russian, code comments in English.
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 03 · Осанка: правила vs ML, проверка на новых людях

**Задача:** по ключевым точкам тела с фронтальной веб-камеры определить
позу: `upright`, `slouch`, `forward_head`, `side_lean`.

**Главный вопрос:** работает ли модель на **человеке, которого не было в
обучении**? Поэтому валидация — *leave-one-subject-out* (LOSO): каждый
человек по очереди становится тестом, модель учится на остальных.
Обычный случайный сплит кадров дал бы завышенный результат: соседние
кадры одного человека почти одинаковы.

| Модель | Тип | Обучение |
|---|---|---|
| Правила + калибровка | baseline | без обучения |
| Logistic Regression | линейная | с нуля |
| HistGradientBoosting | деревья | с нуля |
| MLP | нейросеть | с нуля |

Почему здесь всё учится **с нуля**: вход — 10 табличных признаков, а не
изображение; предобученных весов для такой задачи нет, и они не нужны.

Данные: `data/raw/posture/` (`scripts/record_session.py --task posture`).
Протокол записи: каждая сессия начинается с ~10 с ровной посадки —
это калибровка, как в приложении."""),
    code("""import sys
from pathlib import Path

# Make src/ importable even without `pip install -e .` (see scripts/_bootstrap.py)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deskjarvis import config as cfg
from deskjarvis.features import posture_features
from deskjarvis.posture import (ALL_FEATURES, MODEL_FEATURE_NAMES,
                                PostureClassifier, model_features,
                                reference_from)

SEED = 42
# "raw" = real recordings; "demo" = synthetic, only to test the code
DATA_SOURCE = os.environ.get("DJ_DATA_SOURCE", "raw")
DATA = cfg.DATA_DIR / DATA_SOURCE / "posture"
WINDOW = 15            # frames (~0.5 s at 30 FPS) for the rolling mean
CALIB_MS = 5000        # first 5 s of upright = calibration
GRACE_MS = 1000        # frames right after a label switch are dropped
MIN_VIS = 0.6
LABELS = ["upright", "slouch", "forward_head", "side_lean"]
RESULTS = Path("../results") if Path.cwd().name == "notebooks" \\
    else Path("results")
MODELS = RESULTS.parent / "models" / "posture"
RESULTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")"""),
    md("## 1. Загрузка и очистка"),
    code("""files = sorted(DATA.glob("*.parquet"))
assert files, f"No sessions in {DATA}"
raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"{len(files)} sessions, {len(raw)} frames, "
      f"{raw.subject.nunique()} subjects")

KEYS = [cfg.POSE_LEFT_EAR, cfg.POSE_RIGHT_EAR, cfg.POSE_LEFT_SHOULDER,
        cfg.POSE_RIGHT_SHOULDER]
has_pose = raw[f"pose_{cfg.POSE_LEFT_SHOULDER}_x"].notna()
visible = has_pose & (raw[[f"pose_{k}_v" for k in KEYS]].min(axis=1)
                      >= MIN_VIS)
print(f"no pose: {1 - has_pose.mean():.1%}, "
      f"low visibility: {(has_pose & ~visible).mean():.1%}")
df = raw[visible].reset_index(drop=True)"""),
    code("""# Rebuild (N, 33, 4) pose arrays and compute the shared features
cols = [f"pose_{i}_{c}" for i in range(33) for c in "xyzv"]
poses = df[cols].to_numpy(dtype=float).reshape(-1, 33, 4)
feats = pd.DataFrame([posture_features(p) for p in poses])
df = pd.concat([df[["subject", "session", "t_ms", "label"]], feats],
               axis=1)

# Drop frames right after a label switch: the person is still moving.
# A "segment" is a run of frames with the same label within a session.
df = df.sort_values(["session", "subject", "t_ms"]).reset_index(drop=True)
by_session = df.groupby(["session", "subject"])
segment = by_session["label"].transform(lambda s: s.ne(s.shift()).cumsum())
seg_start = df.groupby([df.session, df.subject, segment])["t_ms"].transform(
    "min")
keep = (df.t_ms - seg_start >= GRACE_MS) | (segment == 1)  # 1st = calib
print(f"dropped as transitions: {1 - keep.mean():.1%}")
df = df[keep].reset_index(drop=True)
df.groupby(["subject", "label"]).size().unstack(fill_value=0)"""),
    md("""## 2. Калибровка и признаки для модели

Для каждой сессии эталон = медиана первых 5 секунд ровной посадки.
К 5 абсолютным признакам добавляем 5 **относительных** (отношение или
разница с эталоном). Затем скользящее среднее по 15 кадрам — то же самое
делает приложение (`PostureModel`), чтобы не было *training / serving
skew*."""),
    code("""def session_features(g):
    g = g.sort_values("t_ms")
    start = g["t_ms"].min()
    calib = g[(g.label == "upright") & (g.t_ms < start + CALIB_MS + 2000)]
    if len(calib) < 10:  # fallback: all upright frames of the session
        calib = g[g.label == "upright"]
    ref = reference_from(calib[list(ALL_FEATURES)].to_dict("records"))
    X = np.stack([model_features(r, ref)
                  for r in g[list(ALL_FEATURES)].to_dict("records")])
    X = pd.DataFrame(X, columns=MODEL_FEATURE_NAMES, index=g.index)
    X = X.rolling(WINDOW, min_periods=1).mean()
    out = pd.concat([g[["subject", "session", "t_ms", "label"]], X], axis=1)
    out["is_calib"] = out.index.isin(calib.index)
    out["ref"] = [ref] * len(out)
    return out


data = pd.concat([session_features(g)
                  for _, g in df.groupby(["session", "subject"])])
# Calibration frames are not test material: the reference is built on them
data = data[~data.is_calib].reset_index(drop=True)
# Integer targets: robust across scikit-learn / pandas string dtypes
X = data[MODEL_FEATURE_NAMES]
y = data["label"].map(LABELS.index).astype(int)
IDX = list(range(len(LABELS)))
N_BLOCKS = 5


def time_blocks(df, k=N_BLOCKS):
    # Blocked time split for a single person: every posture segment is
    # cut into k consecutive blocks; fold i tests block i of EVERY
    # segment, so neighbouring frames rarely end up on both sides.
    seg = (df.label != df.label.shift()).cumsum()
    pos = df.groupby([df.session, seg]).cumcount()
    size = df.groupby([df.session, seg])["label"].transform("size")
    return (pos * k // size).rename("block")


if data.subject.nunique() >= 2:
    GROUPS, CV_MODE = data["subject"], "leave-one-subject-out"
else:
    data = data.sort_values(["session", "t_ms"]).reset_index(drop=True)
    X = data[MODEL_FEATURE_NAMES]
    y = data["label"].map(LABELS.index).astype(int)
    GROUPS, CV_MODE = time_blocks(data), f"blocked time CV ({N_BLOCKS} folds)"
print("CV:", CV_MODE, "| groups:", GROUPS.nunique())
if CV_MODE.startswith("blocked"):
    print("NOTE: one person only - this measures generalisation over time, "
          "not to new people; expect lower scores on others.")"""),
    md("""**Схема валидации.** Если записей несколько человек —
*leave-one-subject-out*. Если человек один — *блочная временная
кросс-валидация*: каждая поза делится на 5 последовательных кусков, и в
каждом фолде тестом служит свой кусок каждой позы. Это честнее случайного
сплита кадров (соседние кадры почти одинаковы), но всё ещё оценивает
модель на том же человеке — перенос на других людей это не проверяет."""),
    md("## 3. EDA: относительные признаки по классам"),
    code("""fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, f in zip(axes, MODEL_FEATURE_NAMES[5:]):
    sns.boxplot(data=data, x="label", y=f, order=LABELS, ax=ax,
                showfliers=False)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

corr = data[MODEL_FEATURE_NAMES].corr(method="spearman")
sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".1f")
plt.title("Spearman correlation of model features")
plt.show()"""),
    md("## 4. Leave-one-subject-out: сравнение моделей"),
    code("""class RulesModel:
    \"\"\"Wraps the calibrated rules so they fit the CV loop.\"\"\"

    def fit(self, X, y):
        return self

    def predict(self, rows):
        out = []
        for _, r in rows.iterrows():
            clf = PostureClassifier()
            clf.reference = {k: r["ref"][k] for k in ALL_FEATURES}
            out.append(LABELS.index(
                clf.predict({k: r[k] for k in ALL_FEATURES})))
        return np.array(out)


MODELS_CV = {
    "Rules (baseline)": RulesModel(),
    "LogisticRegression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000,
                                             class_weight="balanced")),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced",
        random_state=SEED),
    "MLP": make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                                        early_stopping=True, max_iter=500,
                                        random_state=SEED)),
}


def loso(model, features=MODEL_FEATURE_NAMES):
    preds = pd.Series(0, index=data.index, dtype=int)
    fold_f1 = {}
    for g in GROUPS.unique():
        test = GROUPS == g
        if isinstance(model, RulesModel):
            p = model.predict(data[test])
        else:
            m = clone(model).fit(X.loc[~test, features], y[~test])
            p = m.predict(X.loc[test, features])
        preds[test] = p
        fold_f1[g] = f1_score(y[test], p, average="macro", labels=IDX)
    return preds, fold_f1


results, all_preds = [], {}
for name, model in MODELS_CV.items():
    t0 = time.perf_counter()
    preds, fold_f1 = loso(model)
    all_preds[name] = preds
    f = np.array(list(fold_f1.values()))
    results.append({"model": name, "macroF1_mean": f.mean(),
                    "macroF1_std": f.std(), "worst_person": f.min(),
                    "cv_seconds": time.perf_counter() - t0})
comparison = pd.DataFrame(results).set_index("model").round(3)
comparison.to_csv(RESULTS / "posture_comparison.csv")
comparison"""),
    md("""`worst_person` — F1 на самом «трудном» фолде (человеке или куске времени). Для продукта он
важнее среднего: если ассистент не работает для части пользователей,
средняя метрика это скрывает."""),
    code("""best = comparison.drop("Rules (baseline)")["macroF1_mean"].idxmax()
print("Best ML model:", best)
print(classification_report(y, all_preds[best], labels=IDX,
                            target_names=LABELS, digits=3))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, name in zip(axes, ["Rules (baseline)", best]):
    ConfusionMatrixDisplay.from_predictions(
        y, all_preds[name], labels=IDX, display_labels=LABELS,
        normalize="true",
        values_format=".2f", ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(name)
plt.tight_layout()
plt.show()"""),
    md("""## 5. Абляция: нужна ли персональная калибровка?

Та же лучшая модель, но только на 5 абсолютных признаках. Если F1
заметно падает, калибровка под пользователя — ключевая часть решения
(гипотеза: у людей разное телосложение и посадка камеры)."""),
    code("""_, f_abs = loso(MODELS_CV[best], features=list(ALL_FEATURES))
_, f_rel = loso(MODELS_CV[best])
ablation = pd.DataFrame({
    "absolute only": f_abs, "absolute + relative": f_rel}).round(3)
ablation.loc["mean"] = ablation.mean()
ablation"""),
    md("## 6. Какие признаки важны (permutation importance)"),
    code("""imp = []
for g in GROUPS.unique():
    test = GROUPS == g
    m = clone(MODELS_CV[best]).fit(X[~test], y[~test])
    r = permutation_importance(m, X[test], y[test], scoring="f1_macro",
                               n_repeats=5, random_state=SEED)
    imp.append(r.importances_mean)
importance = pd.Series(np.mean(imp, axis=0), index=MODEL_FEATURE_NAMES)
importance.sort_values().plot.barh(
    title=f"Permutation importance ({best}, LOSO folds)")
plt.xlabel("drop in macro-F1 when the feature is shuffled")
plt.show()"""),
    md("## 7. Финальная модель для приложения"),
    code("""final = clone(MODELS_CV[best]).fit(X, y)
x1 = X.iloc[:1]
t0 = time.perf_counter()
for _ in range(200):
    final.predict(x1)
latency_ms = (time.perf_counter() - t0) / 200 * 1000

MODELS.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": final, "window": WINDOW, "labels": LABELS,
             "features": MODEL_FEATURE_NAMES, "name": best},
            MODELS / "model.joblib")
summary = {"comparison": comparison.reset_index().to_dict("records"),
           "best": best, "latency_ms": latency_ms,
           "ablation_mean": ablation.loc["mean"].to_dict(),
           "subjects": int(data.subject.nunique()), "cv": CV_MODE,
           "frames": int(len(data)), "data_source": DATA_SOURCE}
(RESULTS / "posture_metrics.json").write_text(
    json.dumps(summary, indent=2, default=float))
print(f"saved {MODELS / 'model.joblib'}  ({latency_ms:.2f} ms/prediction)")"""),
    md("""Приложение (`scripts/live_demo.py`) само подхватит
`models/posture/model.joblib`; если файла нет — использует правила.

## 8. Выводы (заполнить после запуска на реальных данных)

- Лучшая модель: …, LOSO macro-F1 = … ± …; правила: …
- Худший человек: F1 = …; почему (камера, одежда, посадка): …
- Абляция: калибровка даёт +… F1
- Важнейшие признаки: …
- Чаще всего путаются …"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3",
                                "display_name": "Python 3",
                                "language": "python"}
nbf.write(nb, "notebooks/03_posture_models.ipynb")
print("notebook written")
