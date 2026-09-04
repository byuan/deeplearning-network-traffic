"""Smoke tests: the pipeline runs end-to-end on synthetic data and metrics are sane."""
import numpy as np
import torch
from src.data import load_data
from src.evaluate import compute_metrics
from src.model import build_model, build_optimizer

CFG = {
    "seed": 0,
    "data": {"source": "synthetic", "n_samples": 400, "n_bytes": 64, "n_classes": 3,
             "test_size": 0.25},
    "model": {"filters": 2, "kernel_size": 5, "dropout": 0.2, "dense_sizes": [16, 8]},
    "train": {"optimizer": "Adam", "lr": 0.001},
}


def test_load_data_shapes():
    Xtr, Xte, ytr, yte, names = load_data(CFG)
    assert Xtr.shape[1:] == (1, 64) and Xte.shape[1:] == (1, 64)
    assert len(ytr) == 300 and len(yte) == 100
    assert len(names) == 3 and ytr.max() < 3
    assert Xtr.min() >= 0.0 and Xtr.max() <= 1.0  # normalised bytes


def test_model_builds_trains_and_predicts():
    torch.manual_seed(0)
    Xtr, Xte, ytr, yte, names = load_data(CFG)
    model = build_model(Xtr.shape[-1], len(names), CFG)
    opt = build_optimizer(model.parameters(), CFG["train"])
    x, y = torch.tensor(Xtr[:32]), torch.tensor(ytr[:32])

    def eval_loss():                          # measure in eval mode: no dropout noise
        model.eval()
        with torch.no_grad():
            return torch.nn.functional.cross_entropy(model(x), y).item()

    loss0 = eval_loss()
    model.train()
    for _ in range(20):                       # a few steps must reduce the loss
        opt.zero_grad(); torch.nn.functional.cross_entropy(model(x), y).backward(); opt.step()
    assert eval_loss() < loss0
    with torch.no_grad():
        prob = torch.softmax(model(torch.tensor(Xte)), dim=1).numpy()
    assert prob.shape == (len(yte), len(names))
    assert np.allclose(prob.sum(axis=1), 1.0, atol=1e-5)


def test_metrics_range():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.eye(3)[y] * 0.8 + 0.1  # confident, correct predictions
    m = compute_metrics(y, p, ["a", "b", "c"])
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= m[k] <= 1.0
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0
    assert set(m["per_class"]) == {"a", "b", "c"}
