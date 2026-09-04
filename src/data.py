"""Data loading and preparation.

The real dataset is a header-less CSV: column 0 is the nDPI protocol label, the
remaining 1024 columns are the first 1024 payload bytes (0-255) of a TCP flow.
See data/README.md for how to obtain / place the file.

A small synthetic option (`data.source: synthetic`) exists only so the smoke
tests and CI can run without the dataset.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _load_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "See data/README.md — place the flow CSV at the path given by "
            "`data.csv_path` in config.yaml."
        )
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, 0].astype(str).to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype="float32")
    return X, y


def _make_synthetic(n_samples: int, n_bytes: int, n_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n_samples)
    # each class gets its own byte-distribution so the task is learnable
    X = rng.integers(0, 256, size=(n_samples, n_bytes)).astype("float32")
    X[:, :8] = (y[:, None] * 10 + rng.integers(0, 10, size=(n_samples, 8))) % 256
    return X, np.array([f"proto_{c}" for c in y])


def load_data(cfg: dict):
    """Return (X_train, X_test, y_train, y_test, class_names).

    X arrays have shape (n, 1, n_bytes) as float32 (channels-first, for nn.Conv1d);
    y arrays are integer class indices (int64).
    """
    d = cfg["data"]
    seed = cfg["seed"]
    if d.get("source", "csv") == "synthetic":
        X, y = _make_synthetic(d.get("n_samples", 600), d.get("n_bytes", 1024),
                               d.get("n_classes", 4), seed)
    else:
        X, y = _load_csv(d["csv_path"])

    n_bytes = d.get("n_bytes", X.shape[1])
    X = X[:, :n_bytes]

    # Label encoding is fit on the FULL label set only to get a stable, complete
    # class list (labels are categorical names, so this is not leakage).
    encoder = LabelEncoder().fit(y)
    y_idx = encoder.transform(y).astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=d["test_size"], random_state=seed,
        stratify=y_idx if d.get("stratify", True) else None,
    )

    # Byte values are on a fixed 0-255 scale, so a constant rescale is
    # sufficient and cannot leak test statistics into training.
    if d.get("normalize", True):
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    X_train = X_train.astype("float32")[:, None, :]
    X_test = X_test.astype("float32")[:, None, :]
    return X_train, X_test, y_train, y_test, list(encoder.classes_)
