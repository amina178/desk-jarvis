import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deskjarvis.training import (LandmarkDataset, LandmarkMLP,  # noqa: E402
                                 SmallCNN, fit, mobilenet_v3, predict,
                                 set_backbone_trainable, set_seed)
from torch.utils.data import DataLoader  # noqa: E402


def toy_landmarks(n_per_class=60, n_classes=3, seed=0):
    """Each class = a different 'hand shape' + noise."""
    rng = np.random.default_rng(seed)
    protos = rng.uniform(0, 100, (n_classes, 21, 3))
    protos[:, 9, :2] = protos[:, 0, :2] + [0, -40]  # sane palm size
    X, y = [], []
    for c in range(n_classes):
        X.append(protos[c] + rng.normal(0, 2, (n_per_class, 21, 3)))
        y += [c] * n_per_class
    return np.concatenate(X), np.array(y)


def test_mlp_learns_separable_classes():
    set_seed(0)
    X, y = toy_landmarks()
    train = DataLoader(LandmarkDataset(X, y, augment=True), batch_size=32,
                       shuffle=True)
    val = DataLoader(LandmarkDataset(X, y), batch_size=64)
    model, hist = fit(LandmarkMLP(3), train, val, epochs=15, lr=3e-3,
                      device="cpu", verbose=False)
    assert max(hist.val_f1) > 0.9
    y_true, y_pred, probs, _ = predict(model, val, "cpu")
    assert probs.shape == (len(y), 3)


def test_cnn_shapes():
    x = torch.zeros(2, 3, 96, 96)
    assert SmallCNN(7)(x).shape == (2, 7)
    mnet = mobilenet_v3(7, pretrained=False)
    assert mnet(x).shape == (2, 7)
    set_backbone_trainable(mnet, False)
    trainable = [n for n, p in mnet.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("classifier") for n in trainable)
