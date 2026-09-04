"""Evaluation metrics and figures for the multi-class protocol classifier."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: works in CI and on the cluster
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_prob, class_names) -> dict:
    """y_prob has shape (n, n_classes). Headline scores are macro-averaged."""
    y_pred = np.argmax(y_prob, axis=1)
    n_classes = len(class_names)
    labels = np.arange(n_classes)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(label_binarize(y_true, classes=labels), y_prob,
                                       average="macro", multi_class="ovr")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
        "n_classes": n_classes,
        "per_class": {},
    }
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=n_classes)
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {"precision": float(p[i]), "recall": float(r[i]),
                                      "f1": float(f[i]), "support": int(support[i])}
    return metrics


def classification_text(y_true, y_prob, class_names) -> str:
    y_pred = np.argmax(y_prob, axis=1)
    return classification_report(y_true, y_pred, labels=np.arange(len(class_names)),
                                 target_names=class_names, digits=4, zero_division=0)


def save_confusion_matrix(y_true, y_prob, class_names, path):
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * n), max(5, 0.45 * n)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    thresh = cm.max() / 2.0
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                color="white" if v > thresh else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_roc_curve(y_true, y_prob, class_names, path):
    """One-vs-rest ROC per class plus the macro-average curve."""
    n = len(class_names)
    Y = label_binarize(y_true, classes=np.arange(n))
    fig, ax = plt.subplots(figsize=(6, 5))
    grid = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(grid)
    for i in range(n):
        fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=0.8, alpha=0.5, label=f"{class_names[i]} ({auc(fpr, tpr):.2f})")
        mean_tpr += np.interp(grid, fpr, tpr)
    mean_tpr /= n
    ax.plot(grid, mean_tpr, color="black", lw=2, label=f"macro-avg ({auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC (one-vs-rest)")
    ax.legend(fontsize=5, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_per_class_metrics(metrics: dict, path):
    names = list(metrics["per_class"])
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(names)), 4.5))
    for k, key in enumerate(("precision", "recall", "f1")):
        ax.bar(x + (k - 1) * w, [metrics["per_class"][n][key] for n in names], w, label=key)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Per-class performance (test set)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_training_curves(history: dict, path):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot(epochs, history["loss"], label="train")
    if "val_loss" in history: a1.plot(epochs, history["val_loss"], label="val")
    a1.set_title("Loss"); a1.set_xlabel("Epoch"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(epochs, history["accuracy"], label="train")
    if "val_accuracy" in history: a2.plot(epochs, history["val_accuracy"], label="val")
    a2.set_title("Accuracy"); a2.set_xlabel("Epoch"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
