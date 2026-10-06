"""Models, datasets and training loop for the gesture notebook.

Imported only for training (needs torch / torchvision); the live app
runs exported ONNX models and does not depend on this module.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from deskjarvis.gestures import landmark_input


def set_seed(seed: int = 42) -> None:
    """Same seed -> same splits, init and batches: reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------- models


class LandmarkMLP(nn.Module):
    """63 normalised landmark coordinates -> gesture logits.

    Trained from scratch: the input is a short vector of geometry, not
    an image, so ImageNet weights have nothing to offer here.
    """

    def __init__(self, n_classes: int, in_dim: int = 63,
                 hidden: tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Four conv blocks trained from scratch: the pixel baseline that
    shows what we lose without pre-training."""

    def __init__(self, n_classes: int, widths=(32, 64, 128, 256),
                 dropout: float = 0.3):
        super().__init__()
        blocks, prev = [], 3
        for w in widths:
            blocks += [nn.Conv2d(prev, w, 3, padding=1, bias=False),
                       nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False),
                       nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            prev = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout),
                                  nn.Linear(prev, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def mobilenet_v3(n_classes: int, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with a new last layer for our classes.

    Only the final Linear is replaced: everything before it keeps the
    ImageNet features (edges, textures, shapes) that we fine-tune.
    """
    from torchvision.models import (MobileNet_V3_Small_Weights,
                                    mobilenet_v3_small)

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, n_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze MobileNet's feature extractor (not the head)."""
    for p in model.features.parameters():
        p.requires_grad = trainable


# ------------------------------------------------------------ datasets


class LandmarkDataset(Dataset):
    """Raw (N, 21, 3) landmarks -> normalised vectors, with geometric
    augmentation in training mode."""

    def __init__(self, landmarks: np.ndarray, labels: np.ndarray,
                 augment: bool = False, seed: int = 0):
        self.landmarks = landmarks.astype(np.float64)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def _augment(self, pts: np.ndarray) -> tuple[np.ndarray, bool]:
        pts = pts.copy()
        # Small in-plane rotation around the wrist (people tilt hands).
        a = np.radians(self.rng.uniform(-15, 15))
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts[:, :2] = (pts[:, :2] - pts[0, :2]) @ rot.T + pts[0, :2]
        # Aspect jitter: training landmarks and webcam frames differ in
        # resolution and aspect ratio (domain shift).
        pts[:, 0] *= self.rng.uniform(0.85, 1.15)
        pts += self.rng.normal(0, 0.01, pts.shape) * np.ptp(pts[:, :2])
        return pts, bool(self.rng.random() < 0.5)

    def __getitem__(self, i):
        pts, mirror = self.landmarks[i], False
        if self.augment:
            pts, mirror = self._augment(pts)
        x = landmark_input(pts, mirror=mirror)
        return torch.from_numpy(x), self.labels[i]


def crop_transforms(size: int = 160, train: bool = False):
    from torchvision import transforms as T

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(),  # left/right hand symmetry
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.ToTensor(), norm,
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])


class CropDataset(Dataset):
    """Hand crops listed in a dataframe with `crop_path` and `y`."""

    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


# ------------------------------------------------------------ training


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_f1: list = field(default_factory=list)
    epoch_time: list = field(default_factory=list)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str):
    """Returns (y_true, y_pred, probs, mean loss)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    ys, preds, probs, losses = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item() * len(y))
        p = logits.softmax(-1)
        ys.append(y.cpu())
        probs.append(p.cpu())
        preds.append(p.argmax(-1).cpu())
    y = torch.cat(ys).numpy()
    return (y, torch.cat(preds).numpy(), torch.cat(probs).numpy(),
            float(np.sum(losses) / len(y)))


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        epochs: int, lr: float, device: str, weight_decay: float = 1e-4,
        patience: int = 5, class_weights: torch.Tensor | None = None,
        verbose: bool = True) -> tuple[nn.Module, History]:
    """AdamW + cosine LR, early stopping on validation macro-F1.

    Selection uses macro-F1 (not loss or accuracy) because that is the
    metric we report, and it weights every gesture class equally.
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device))
    hist, best_f1, best_state, bad = History(), -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        y_true, y_pred, _, val_loss = predict(model, val_loader, device)
        f1 = f1_score(y_true, y_pred, average="macro")
        hist.train_loss.append(total / n)
        hist.val_loss.append(val_loss)
        hist.val_f1.append(f1)
        hist.epoch_time.append(time.perf_counter() - t0)
        if verbose:
            print(f"epoch {epoch + 1:3d}  train_loss {total / n:.4f}  "
                  f"val_loss {val_loss:.4f}  val_macroF1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state, bad = f1, copy.deepcopy(
                model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)
    return model, hist


# ---------------------------------------------------- export & speed


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_onnx(model: nn.Module, example: torch.Tensor, path: Path) -> Path:
    """Export with a dynamic batch dimension; opset 17 is widely
    supported by ONNX Runtime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    torch.onnx.export(model, example.cpu(), str(path), input_names=["input"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"input": {0: "batch"},
                                    "logits": {0: "batch"}},
                      dynamo=False)
    return path


def onnx_latency_ms(path: Path, example: np.ndarray, runs: int = 200
                    ) -> float:
    """Median single-sample CPU latency of an ONNX model.

    Median, not mean: the first runs and OS hiccups create outliers.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(10):  # warm-up
        sess.run(None, {name: example})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: example})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))
