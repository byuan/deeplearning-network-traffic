"""Single entry point: `python -m src.train --config config.yaml`.

Reproducibility standard: this command must recreate the reported results.
It trains the 1-D CNN (PyTorch) and writes to the output directory:
  metrics.json, classification_report.txt, confusion_matrix.png, roc_curve.png,
  per_class_metrics.png, training_curves.png, history.json, model.pt
"""
from __future__ import annotations
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import load_data
from .model import build_model, build_optimizer
from .evaluate import (compute_metrics, classification_text, save_confusion_matrix,
                       save_roc_curve, save_per_class_metrics, save_training_curves)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}"
    return "cpu (torch.cuda.is_available() is False)"


def predict_proba(model: nn.Module, X: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            out.append(torch.softmax(model(X[i:i + batch_size]), dim=1).cpu())
    return torch.cat(out).numpy()


def run_epoch_eval(model, X, y, loss_fn, batch_size):
    """Mean loss and accuracy over (X, y) in eval mode."""
    model.eval()
    total, correct, n = 0.0, 0, X.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            logits = model(X[i:i + batch_size])
            total += loss_fn(logits, y[i:i + batch_size]).item() * logits.size(0)
            correct += (logits.argmax(1) == y[i:i + batch_size]).sum().item()
    return total / n, correct / n


def main(config_path: str, seed: int | None = None, out_dir: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if seed is not None:                       # CLI overrides (used by the seed sweep)
        cfg["seed"] = int(seed)
    if out_dir is not None:
        cfg["output"]["dir"] = out_dir
    set_seed(cfg["seed"])
    out = cfg["output"]["dir"]; os.makedirs(out, exist_ok=True)
    tr = cfg["train"]
    bs = int(tr["batch_size"]); epochs = int(tr["epochs"]); verbose = int(tr.get("verbose", 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, X_te, y_tr, y_te, class_names = load_data(cfg)
    print(f"Device: {describe_device(device)} | torch {torch.__version__}")
    print(f"train {X_tr.shape} | test {X_te.shape} | {len(class_names)} classes")

    # Hold out the LAST `validation_split` fraction of the training set for
    # model selection (same convention as Keras' validation_split).
    n_val = int(round(len(X_tr) * float(tr["validation_split"])))
    n_fit = len(X_tr) - n_val
    Xfit = torch.tensor(X_tr[:n_fit], device=device); yfit = torch.tensor(y_tr[:n_fit], device=device)
    Xval = torch.tensor(X_tr[n_fit:], device=device); yval = torch.tensor(y_tr[n_fit:], device=device)
    Xte = torch.tensor(X_te, device=device)

    model = build_model(X_tr.shape[-1], len(class_names), cfg).to(device)
    print(model)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = build_optimizer(model.parameters(), tr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    patience = int(tr.get("early_stopping_patience") or 0)
    ckpt = os.path.join(out, "model.pt")
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_fit, generator=gen).to(device)
        run_loss, run_correct = 0.0, 0
        for i in range(0, n_fit, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = model(Xfit[idx])
            loss = loss_fn(logits, yfit[idx])
            loss.backward(); opt.step()
            run_loss += loss.item() * idx.numel()
            run_correct += (logits.argmax(1) == yfit[idx]).sum().item()
        tr_loss, tr_acc = run_loss / n_fit, run_correct / n_fit
        va_loss, va_acc = run_epoch_eval(model, Xval, yval, loss_fn, bs) if n_val else (tr_loss, tr_acc)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(float(v))
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {tr_loss:.4f} - accuracy: {tr_acc:.4f} "
                  f"- val_loss: {va_loss:.4f} - val_accuracy: {va_acc:.4f}", flush=True)

        # Keep the checkpoint with the best validation loss (as the original study did).
        if va_loss < best_val:
            best_val, best_epoch, bad_epochs = va_loss, epoch + 1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best epoch {best_epoch})")
                break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(ckpt, map_location=device))
    y_prob = predict_proba(model, Xte, bs)

    metrics = compute_metrics(y_te, y_prob, class_names)
    # Non-score bookkeeping goes under "run" (top-level numbers are treated as scores
    # in [0, 1] by grading/grade.py).
    metrics["run"] = {"seed": cfg["seed"], "framework": f"torch {torch.__version__}",
                      "device": describe_device(device),
                      "optimizer": tr["optimizer"], "epochs_run": len(history["loss"]),
                      "best_epoch": best_epoch, "train_seconds": round(train_time, 1),
                      "n_test": metrics.pop("n_test"), "n_classes": metrics.pop("n_classes")}

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out, "classification_report.txt"), "w") as f:
        f.write(classification_text(y_te, y_prob, class_names))
    save_confusion_matrix(y_te, y_prob, class_names, os.path.join(out, "confusion_matrix.png"))
    save_roc_curve(y_te, y_prob, class_names, os.path.join(out, "roc_curve.png"))
    save_per_class_metrics(metrics, os.path.join(out, "per_class_metrics.png"))
    save_training_curves(history, os.path.join(out, "training_curves.png"))

    print("Results written to", out)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--out", default=None, help="override output directory")
    a = ap.parse_args()
    main(a.config, seed=a.seed, out_dir=a.out)
