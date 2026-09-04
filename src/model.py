"""Model definition — the 1-D CNN from the original study, in PyTorch.

Original Keras layout: Conv1D(2, k=5, relu) -> MaxPool1D(2) -> Dropout(0.2) -> Flatten
                       -> Dense(16, relu) -> Dense(8, relu) -> Dense(n_classes, softmax)
Softmax is folded into the loss (nn.CrossEntropyLoss); apply torch.softmax to the logits
to get probabilities.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Conv1DNet(nn.Module):
    def __init__(self, n_bytes: int, n_classes: int, filters: int = 2, kernel_size: int = 5,
                 pool_size: int = 2, dropout: float = 0.2, dense_sizes=(16, 8)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size),          # "valid" padding, as in Keras
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        flat = filters * ((n_bytes - kernel_size + 1) // pool_size)
        layers, prev = [], flat
        for h in dense_sizes:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        layers.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*layers)
        self.apply(self._init_keras_style)

    @staticmethod
    def _init_keras_style(m):
        """Glorot-uniform weights, zero biases — Keras' defaults, which the original
        study used. PyTorch's default (Kaiming-uniform + random bias) trains noticeably
        slower for this very narrow network."""
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):                                 # x: (batch, 1, n_bytes)
        return self.classifier(self.features(x))


def build_model(n_bytes: int, n_classes: int, cfg: dict) -> Conv1DNet:
    m = cfg["model"]
    return Conv1DNet(n_bytes, n_classes, filters=int(m["filters"]),
                     kernel_size=int(m["kernel_size"]), pool_size=int(m.get("pool_size", 2)),
                     dropout=float(m["dropout"]), dense_sizes=m["dense_sizes"])


def build_optimizer(params, train_cfg: dict) -> torch.optim.Optimizer:
    """Same five choices as the original study. Epsilons follow Keras defaults (1e-7)
    so the runs stay comparable with the TensorFlow version."""
    name = train_cfg["optimizer"]
    lr = float(train_cfg["lr"])
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, eps=1e-7)
    if name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.9, eps=1e-7)
    if name == "SGD-Momentum":
        return torch.optim.SGD(params, lr=lr, momentum=float(train_cfg.get("momentum", 0.9)))
    if name == "SGD-Nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=float(train_cfg.get("momentum", 0.8)),
                               nesterov=True)
    raise ValueError(f"Unknown optimizer: {name!r} "
                     "(choose SGD, Adam, RMSprop, SGD-Momentum, SGD-Nesterov)")
