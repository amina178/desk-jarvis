"""Builds notebooks/02_gesture_models.ipynb (Colab, GPU T4).

Markdown in Russian, code comments in English.
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 02 · Распознавание жестов: с нуля vs fine-tuning

**Задача:** классифицировать статичный жест руки для управления
компьютером: 👍 `like` — следующий слайд, 👎 `dislike` — предыдущий,
✋ `palm` — микрофон, ✌️ `peace` — скриншот. Классы `fist`, `one` и
`no_gesture` не запускают действий: они учат модель, на что **не**
реагировать.

**Сравниваем три подхода (бонус за сравнение моделей):**

| Модель | Вход | Обучение | Гипотеза |
|---|---|---|---|
| A. MLP | 63 координаты ключевых точек | с нуля | быстрая и устойчивая к фону, но зависит от того, нашёл ли MediaPipe руку |
| B. SmallCNN | кроп руки 160×160 | с нуля | на ограниченных данных уступит предобученной сети |
| C. MobileNetV3-Small | кроп руки 160×160 | fine-tuning ImageNet | лучшая точность на пикселях |

**Данные:** HaGRID sample 30k 384p (Hugging Face, CC BY-SA 4.0).
Сплит **по `user_id`**: ни один человек не попадает в две выборки.

**Метрики:** macro-F1 (классы равноценны, `no_gesture` не должен
«перевешивать»), матрица ошибок, задержка на CPU, размер модели.

Среда: Google Colab, *Runtime → Change runtime type → T4 GPU*."""),
    md("## 0. Настройка"),
    code("""import os
import sys

IN_COLAB = "google.colab" in sys.modules
# SMOKE_TEST = tiny run on fake data to check the code, never for results
SMOKE_TEST = os.environ.get("DJ_SMOKE_TEST") == "1"
print("Colab:", IN_COLAB, "| smoke test:", SMOKE_TEST)"""),
    md("""Код проекта нужен в Colab. Варианты: `git clone` своего
репозитория (вставь ссылку) **или** загрузка `desk-jarvis.zip`."""),
    code("""REPO_URL = ""  # e.g. "https://github.com/<you>/desk-jarvis.git"

if IN_COLAB:
    if not os.path.exists("/content/desk-jarvis"):
        if REPO_URL:
            !git clone -q {REPO_URL} /content/desk-jarvis
        else:
            from google.colab import files
            uploaded = files.upload()  # choose desk-jarvis.zip
            !unzip -q -o desk-jarvis.zip -d /content
    %cd /content/desk-jarvis
    !pip install -q -e . "mediapipe==0.10.35" onnx onnxruntime onnxscript
    !python scripts/download_models.py
    # An editable install is only seen by a NEW Python process; the running
    # kernel needs the source folder on sys.path explicitly.
    sys.path.insert(0, "/content/desk-jarvis/src")"""),
    code("""import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score)
from torch.utils.data import DataLoader

from deskjarvis.gestures import GESTURE_ACTIONS, GESTURES, landmark_input
from deskjarvis.training import (CropDataset, LandmarkDataset, LandmarkMLP,
                                 SmallCNN, count_params, crop_transforms,
                                 export_onnx, fit, mobilenet_v3,
                                 onnx_latency_ms, predict,
                                 set_backbone_trainable, set_seed)

SEED = 42
set_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = sorted(GESTURES)
assert LABELS == list(GESTURES)
CROP = 160
WORKERS = 2 if IN_COLAB else 0

RAW = Path(os.environ.get("DJ_HAGRID_RAW", "/content/hagrid"))
DATA = Path(os.environ.get("DJ_GESTURE_DATA", "/content/gesture_data"))
OUT = Path("models/gestures")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
print("device:", DEVICE)"""),
    md("## 1. Загрузка HaGRID sample и подготовка данных"),
    code("""if not SMOKE_TEST and not (DATA / "meta.csv").exists():
    from huggingface_hub import snapshot_download

    snapshot_download("cj-mills/hagrid-sample-30k-384p",
                      repo_type="dataset", local_dir=RAW)
    # The dataset may ship as archives: unpack everything in place
    for z in RAW.rglob("*.zip"):
        !unzip -q -o "{z}" -d "{z.parent}"
    !python scripts/prepare_hagrid.py --src {RAW} --out {DATA} \\
        --max-per-class 2500
print(sorted(p.name for p in DATA.iterdir()))"""),
    md("""`prepare_hagrid.py` для каждой рамки руки запускает **тот же
MediaPipe HandLandmarker**, что и живое приложение, и сохраняет
ключевые точки и кроп. Кроп строится по точкам так же, как в приложении:
одинаковая предобработка при обучении и в работе (иначе — *training /
serving skew*)."""),
    md("## 2. EDA"),
    code("""meta = pd.read_csv(DATA / "meta.csv")
landmarks = np.load(DATA / "landmarks.npy")
meta["y"] = meta["label"].map(LABELS.index)
assert len(meta) == len(landmarks)

counts = meta.groupby(["label", "split"]).size().unstack(fill_value=0)
counts["total"] = counts.sum(axis=1)
counts"""),
    code("""# Leakage check: every user must live in exactly one split
users = meta.groupby("user_id")["split"].nunique()
assert (users == 1).all(), "user present in several splits!"
meta.groupby("split")["user_id"].nunique().rename("unique users")"""),
    md("""### Где MediaPipe не находит руку

Модель A (MLP) работает только если детектор нашёл руку. Доля промахов
по классам — реальное ограничение пайплайна на точках, которое CNN на
пикселях не имеет."""),
    code("""det = meta.groupby("label")["has_landmarks"].mean().sort_values()
det.plot.barh(color="#2E7D8C", xlim=(0, 1),
              title="Share of hand boxes where MediaPipe found a hand")
plt.xlabel("detection rate")
plt.show()
print(f"overall: {meta['has_landmarks'].mean():.1%}")"""),
    code("""from PIL import Image

fig, axes = plt.subplots(len(LABELS), 6, figsize=(10, 1.8 * len(LABELS)))
for r, label in enumerate(LABELS):
    sample = meta[meta.label == label].sample(
        min(6, (meta.label == label).sum()), random_state=SEED)
    for c in range(6):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(sample):
            ax.imshow(Image.open(sample.crop_path.iloc[c]))
        if c == 0:
            ax.set_title(label, loc="left", fontsize=10)
plt.tight_layout()
plt.show()"""),
    md("""### Структура данных в пространстве ключевых точек

Если классы образуют отдельные облака уже в PCA (линейная проекция в 2D),
задача на точках простая, и MLP должен справиться. Пересекающиеся облака
подсказывают, какие жесты модель будет путать."""),
    code("""lm_rows = meta[meta.has_landmarks].sample(
    min(4000, int(meta.has_landmarks.sum())), random_state=SEED)
X_vis = np.stack([landmark_input(landmarks[i]) for i in lm_rows.index])
pc = PCA(n_components=2, random_state=SEED).fit_transform(X_vis)
sns.scatterplot(x=pc[:, 0], y=pc[:, 1], hue=lm_rows["label"].values, s=8,
                alpha=0.6)
plt.title("PCA of normalised hand landmarks")
plt.legend(markerscale=2, bbox_to_anchor=(1, 1))
plt.show()"""),
    md("## 3. Модель A — MLP на ключевых точках (с нуля)"),
    code("""def lm_split(name):
    rows = meta[(meta.split == name) & meta.has_landmarks]
    return landmarks[rows.index], rows["y"].to_numpy(), rows


EPOCHS_MLP = 3 if SMOKE_TEST else 80
Xtr, ytr, _ = lm_split("train")
Xva, yva, _ = lm_split("val")
mlp_train = DataLoader(LandmarkDataset(Xtr, ytr, augment=True, seed=SEED),
                       batch_size=256, shuffle=True)
mlp_val = DataLoader(LandmarkDataset(Xva, yva), batch_size=1024)

set_seed(SEED)
mlp, hist_mlp = fit(LandmarkMLP(len(LABELS)), mlp_train, mlp_val,
                    epochs=EPOCHS_MLP, lr=3e-3, device=DEVICE, patience=12,
                    verbose=not SMOKE_TEST)
print("params:", count_params(mlp))"""),
    md("""Аугментации для точек (`LandmarkDataset`): поворот ±15° вокруг
запястья, растяжение по x (разные камеры и соотношения сторон), шум и
**зеркалирование** — чтобы модель одинаково понимала левую и правую
руку и зеркальное изображение веб-камеры."""),
    md("## 4. Модель B — SmallCNN на кропах (с нуля)"),
    code("""def crop_loader(name, train=False, batch=64):
    rows = meta[meta.split == name]
    ds = CropDataset(rows.crop_path, rows.y, crop_transforms(CROP, train))
    return DataLoader(ds, batch_size=batch, shuffle=train,
                      num_workers=WORKERS, pin_memory=DEVICE == "cuda")


cnn_train, cnn_val = crop_loader("train", train=True), crop_loader("val")
EPOCHS_CNN = 2 if SMOKE_TEST else 25

set_seed(SEED)
small_cnn, hist_cnn = fit(SmallCNN(len(LABELS)), cnn_train, cnn_val,
                          epochs=EPOCHS_CNN, lr=1e-3, device=DEVICE,
                          patience=6, verbose=not SMOKE_TEST)
print("params:", count_params(small_cnn))"""),
    md("""## 5. Модель C — MobileNetV3-Small (fine-tuning)

Fine-tuning в два этапа:
1. **Заморожен backbone, учится только новая голова** (lr 1e-3). Голова
   инициализирована случайно; если сразу размораживать всё, её большие
   случайные градиенты «испортят» предобученные признаки.
2. **Размораживаем всю сеть** с меньшим learning rate (3e-4): признаки
   ImageNet аккуратно подстраиваются под руки."""),
    code("""PRETRAINED = not SMOKE_TEST  # smoke test runs offline
set_seed(SEED)
mnet = mobilenet_v3(len(LABELS), pretrained=PRETRAINED)

set_backbone_trainable(mnet, False)
mnet, hist_head = fit(mnet, cnn_train, cnn_val,
                      epochs=1 if SMOKE_TEST else 3, lr=1e-3,
                      device=DEVICE, patience=3, verbose=not SMOKE_TEST)
set_backbone_trainable(mnet, True)
mnet, hist_ft = fit(mnet, cnn_train, cnn_val,
                    epochs=1 if SMOKE_TEST else 12, lr=3e-4,
                    device=DEVICE, patience=4, verbose=not SMOKE_TEST)
print("params:", count_params(mnet))"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
curves = {"A. MLP": hist_mlp, "B. SmallCNN": hist_cnn,
          "C. MobileNetV3 (head)": hist_head,
          "C. MobileNetV3 (full)": hist_ft}
for name, h in curves.items():
    axes[0].plot(h.val_loss, label=name)
    axes[1].plot(h.val_f1, label=name)
axes[0].set_title("validation loss")
axes[1].set_title("validation macro-F1")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend()
plt.show()"""),
    md("""## 6. Сравнение на тесте

Честное сравнение требует двух срезов:
- **Тест с найденной рукой** — все три модели на одних и тех же
  примерах (сравнение самих классификаторов).
- **Весь тест на уровне системы** — если MediaPipe не нашёл руку,
  приложение с MLP ничего не делает, то есть фактически предсказывает
  `no_gesture`. Так видно, сколько стоит зависимость от детектора."""),
    code("""test_rows = meta[meta.split == "test"]
test_lm_rows = test_rows[test_rows.has_landmarks]
idle = LABELS.index("no_gesture")


def cnn_predict(model, rows):
    loader = DataLoader(CropDataset(rows.crop_path, rows.y,
                                    crop_transforms(CROP)),
                        batch_size=128, num_workers=WORKERS)
    return predict(model, loader, DEVICE)


Xte = landmarks[test_lm_rows.index]
lm_loader = DataLoader(LandmarkDataset(Xte, test_lm_rows.y.to_numpy()),
                       batch_size=1024)
y_lm, p_mlp_lm, _, _ = predict(mlp, lm_loader, DEVICE)

preds = {"A. MLP": {}, "B. SmallCNN": {}, "C. MobileNetV3": {}}
# MLP at system level: no hand found -> no_gesture
sys_pred = pd.Series(idle, index=test_rows.index)
sys_pred.loc[test_lm_rows.index] = p_mlp_lm
preds["A. MLP"] = {"lm": p_mlp_lm, "all": sys_pred.to_numpy()}
for name, model in (("B. SmallCNN", small_cnn), ("C. MobileNetV3", mnet)):
    y_all, p_all, _, _ = cnn_predict(model, test_rows)
    mask = test_rows.has_landmarks.to_numpy()
    preds[name] = {"all": p_all, "lm": p_all[mask]}
y_all = test_rows.y.to_numpy()"""),
    code("""def onnx_info(model, example, name):
    path = export_onnx(model, example, OUT / name / "model.onnx")
    return path, path.stat().st_size / 1e6, onnx_latency_ms(
        path, example.numpy())


rows = []
for name, model, example in (
        ("A. MLP", mlp, torch.zeros(1, 63)),
        ("B. SmallCNN", small_cnn, torch.zeros(1, 3, CROP, CROP)),
        ("C. MobileNetV3", mnet, torch.zeros(1, 3, CROP, CROP))):
    folder = {"A. MLP": "mlp", "B. SmallCNN": "small_cnn",
              "C. MobileNetV3": "mobilenet"}[name]
    _, size_mb, lat = onnx_info(model, example, folder)
    rows.append({
        "model": name,
        "params": count_params(model),
        "onnx_MB": round(size_mb, 2),
        "cpu_ms": round(lat, 2),
        "macroF1_hand_found": f1_score(y_lm, preds[name]["lm"],
                                       average="macro"),
        "macroF1_system": f1_score(y_all, preds[name]["all"],
                                   average="macro"),
    })
comparison = pd.DataFrame(rows).set_index("model").round(3)
comparison.to_csv(RESULTS / "gesture_comparison.csv")
comparison"""),
    code("""best = comparison["macroF1_system"].idxmax()
print("Best at system level:", best)
print(classification_report(y_all, preds[best]["all"], target_names=LABELS,
                            digits=3))
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, name in zip(axes, preds):
    ConfusionMatrixDisplay.from_predictions(
        y_all, preds[name]["all"], display_labels=LABELS, normalize="true",
        values_format=".2f", ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(name)
plt.tight_layout()
plt.show()"""),
    md("""### 6b. Проверка на своих записях (out-of-domain)

HaGRID снят на телефоны в разных комнатах; моя веб-камера за столом —
другой домен. Если загрузить свои записи
`scripts/record_session.py --task gesture` в `data/raw/gesture/`, этот
блок покажет, насколько модель на точках переносится на реальные
условия использования."""),
    code("""OWN = Path("data/raw/gesture")
own_files = sorted(OWN.glob("*.parquet")) if OWN.exists() else []
if not own_files:
    print("No own recordings - skipped")
else:
    own = pd.concat([pd.read_parquet(f) for f in own_files])
    own = own[own["hand_0_x"].notna() & own["label"].isin(LABELS)]
    hands = own[[f"hand_{i}_{c}" for i in range(21) for c in "xyz"]]
    own_X = hands.to_numpy(float).reshape(-1, 21, 3)
    own_y = own["label"].map(LABELS.index).to_numpy()
    own_loader = DataLoader(LandmarkDataset(own_X, own_y), batch_size=1024)
    y_own, p_own, _, _ = predict(mlp, own_loader, DEVICE)
    print(f"{len(own)} frames, {own.subject.nunique()} people")
    present = sorted(set(y_own))  # only gestures that were recorded
    print(classification_report(y_own, p_own, labels=present,
                                target_names=[LABELS[i] for i in present],
                                digits=3, zero_division=0))"""),
    md("""## 7. Интерпретация: Grad-CAM

Grad-CAM показывает, на какие области кропа опирается MobileNet при
решении: градиент выбранного класса по картам признаков последнего
свёрточного слоя усредняется в веса каналов, взвешенная сумма карт даёт
«тепловую карту». Хорошая модель смотрит на пальцы, а не на фон или
рукав."""),
    code("""class GradCAM:
    def __init__(self, model, layer):
        self.model = model.eval()
        self.acts = self.grads = None
        layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        # Keep the activations and ask autograd for their gradient
        self.acts = output.detach()
        output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        idx = int(logits.argmax()) if class_idx is None else class_idx
        logits[0, idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.acts).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        return (cam - cam.min()) / (np.ptp(cam) + 1e-8), idx


cam_model = mobilenet_v3(len(LABELS), pretrained=False)
cam_model.load_state_dict(mnet.state_dict())
cam_model.to(DEVICE)
gradcam = GradCAM(cam_model, cam_model.features[-1])
tf = crop_transforms(CROP)
sample = test_rows.sample(min(8, len(test_rows)), random_state=SEED)
fig, axes = plt.subplots(2, len(sample), figsize=(2.2 * len(sample), 4.6))
for k, (_, row) in enumerate(sample.iterrows()):
    img = Image.open(row.crop_path).convert("RGB")
    x = tf(img)[None].to(DEVICE).requires_grad_(True)
    cam, idx = gradcam(x)
    axes[0, k].imshow(img)
    axes[0, k].set_title(f"true: {row.label}", fontsize=8)
    axes[1, k].imshow(img)
    axes[1, k].imshow(cam, cmap="jet", alpha=0.45)
    axes[1, k].set_title(f"pred: {LABELS[idx]}", fontsize=8)
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()"""),
    md("## 8. Анализ ошибок"),
    code("""best_pred = preds[best]["all"]
errors = test_rows.assign(pred=[LABELS[i] for i in best_pred])
errors = errors[errors.pred != errors.label]
print("Most frequent confusions (true -> predicted):")
print(errors.groupby(["label", "pred"]).size().sort_values(
    ascending=False).head(8))

show = errors.sample(min(8, len(errors)), random_state=SEED)
fig, axes = plt.subplots(1, max(1, len(show)),
                         figsize=(2.2 * max(1, len(show)), 2.6))
for ax, (_, row) in zip(np.atleast_1d(axes), show.iterrows()):
    ax.imshow(Image.open(row.crop_path))
    ax.set_title(f"{row.label} -> {row.pred}", fontsize=8)
    ax.axis("off")
plt.show()"""),
    md("""## 9. Оптимизация (бонус): квантизация и pruning

- **Динамическая INT8-квантизация** (ONNX Runtime): веса хранятся в 8
  битах вместо 32 → файл примерно в 4 раза меньше. Скорость зависит от
  архитектуры: для полносвязных слоёв (MLP) обычно выигрыш, а свёртки
  при динамической квантизации превращаются в `ConvInteger`, который на
  CPU бывает **медленнее** FP32. Для CNN правильный путь — статическая
  квантизация с калибровкой на примерах данных. Это тоже вывод, который
  стоит показать в отчёте.
- **Pruning MobileNet**: обнуляем 30 % и 50 % весов с наименьшей
  абсолютной величиной (глобальный L1), затем коротко дообучаем, чтобы
  сеть восстановилась. Важно: *неструктурированный* pruning делает
  матрицы разреженными, но обычный CPU всё равно умножает нули —
  выигрыш виден в сжатом размере файла, а не в скорости. Для ускорения
  нужен структурный pruning (удаление каналов целиком)."""),
    code("""from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(folder, example):
    fp32 = OUT / folder / "model.onnx"
    int8 = OUT / folder / "model_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return int8


opt_rows = []
for folder, model, example, lm in (
        ("mlp", mlp, torch.zeros(1, 63), True),
        ("mobilenet", mnet, torch.zeros(1, 3, CROP, CROP), False)):
    int8 = quantize(folder, example)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    if lm:
        x = np.stack([landmark_input(p) for p in Xte])
        y_ref = y_lm
    else:
        x = np.stack([crop_transforms(CROP)(Image.open(p).convert("RGB"))
                      .numpy() for p in test_rows.crop_path])
        y_ref = y_all
    logits = np.concatenate([sess.run(None, {"input": x[i:i + 256]})[0]
                             for i in range(0, len(x), 256)])
    fp32 = OUT / folder / "model.onnx"
    opt_rows.append({
        "model": folder, "variant": "int8",
        "onnx_MB": round(int8.stat().st_size / 1e6, 2),
        "fp32_MB": round(fp32.stat().st_size / 1e6, 2),
        "cpu_ms": round(onnx_latency_ms(int8, example.numpy()), 2),
        "fp32_cpu_ms": round(onnx_latency_ms(fp32, example.numpy()), 2),
        "macroF1": round(f1_score(y_ref, logits.argmax(1),
                                  average="macro"), 3)})
pd.DataFrame(opt_rows)"""),
    code("""import copy
import gzip

import torch.nn.utils.prune as prune


def prunable(model):
    return [(m, "weight") for m in model.modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]


def sparsity(model):
    w = [m.weight for m, _ in prunable(model)]
    return float(sum((p == 0).sum() for p in w) / sum(p.numel() for p in w))


prune_rows = [{"amount": 0.0, "sparsity": round(sparsity(mnet), 3),
               "macroF1": round(f1_score(y_all, preds["C. MobileNetV3"]
                                         ["all"], average="macro"), 3)}]
for amount in (0.3, 0.5):
    pruned = copy.deepcopy(mnet)
    prune.global_unstructured(prunable(pruned),
                              pruning_method=prune.L1Unstructured,
                              amount=amount)
    pruned, _ = fit(pruned, cnn_train, cnn_val,
                    epochs=1 if SMOKE_TEST else 2, lr=1e-4, device=DEVICE,
                    patience=2, verbose=False)
    for m, name in prunable(pruned):
        prune.remove(m, name)  # make the zeros permanent
    _, p, _, _ = cnn_predict(pruned, test_rows)
    path = export_onnx(pruned, torch.zeros(1, 3, CROP, CROP),
                       OUT / f"mobilenet_pruned{int(amount * 100)}"
                       / "model.onnx")
    prune_rows.append({
        "amount": amount, "sparsity": round(sparsity(pruned), 3),
        "macroF1": round(f1_score(y_all, p, average="macro"), 3),
        "gzip_MB": round(len(gzip.compress(path.read_bytes())) / 1e6, 2)})
base = OUT / "mobilenet" / "model.onnx"
prune_rows[0]["gzip_MB"] = round(len(gzip.compress(base.read_bytes()))
                                 / 1e6, 2)
pd.DataFrame(prune_rows)"""),
    md("## 10. Экспорт для приложения"),
    code("""for folder, kind in (("mlp", "landmarks"), ("mobilenet", "crop"),
                     ("small_cnn", "crop")):
    meta_json = {"labels": LABELS, "input": kind, "size": CROP,
                 "actions": GESTURE_ACTIONS}
    (OUT / folder / "labels.json").write_text(json.dumps(meta_json,
                                                         indent=2))

summary = {"comparison": comparison.reset_index().to_dict("records"),
           "quantization": opt_rows, "pruning": prune_rows,
           "test_users": int(test_rows.user_id.nunique()),
           "test_samples": int(len(test_rows)),
           "detection_rate": float(meta.has_landmarks.mean())}
(RESULTS / "gesture_metrics.json").write_text(json.dumps(summary, indent=2,
                                                         default=float))
!cd {OUT.parent} && zip -q -r ../gesture_models.zip gestures
print("models:", sorted(str(p) for p in OUT.rglob("*.onnx")))
if IN_COLAB:
    from google.colab import files
    files.download("gesture_models.zip")
    files.download(str(RESULTS / "gesture_metrics.json"))"""),
    md("""Скачанный `gesture_models.zip` распакуй в корень проекта на
компьютере (появится `models/gestures/...`), и живое демо подхватит
модель.

## 11. Выводы (заполнить после запуска)

- Лучшая модель на уровне системы: …, macro-F1 = …
- С нуля vs fine-tuning на пикселях: разница …, потому что …
- Точки vs пиксели: MLP в … раз быстрее, но теряет … % примеров, где
  MediaPipe не нашёл руку.
- Чаще всего путаются …, вероятная причина …
- Grad-CAM: модель смотрит на …
- Квантизация: размер −… %, F1 изменился на …
- Pruning 30 / 50 %: F1 …; ускорения на CPU нет, потому что …"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3",
                                "display_name": "Python 3",
                                "language": "python"}
nb["metadata"]["accelerator"] = "GPU"
nbf.write(nb, "notebooks/02_gesture_models.ipynb")
print("notebook written")
