# Reproduction note — how this repository's numbers relate to the paper

`Jain_Akshit_Independent_Study_Paper.pdf` (this folder) is the original 2018 independent-study
report. This repository is a 2026 conversion of its Keras script into the course's reproducible
layout. **The numbers a grader should hold this repository to are the ones in
`results/metrics.json`, reproduced below — not the figures in the paper's Table II.** This note
explains why they differ and what was checked.

## 1. What the paper reports vs. what `make reproduce` gives

| | paper, Table II (best: SGD-Momentum) | this repo, Adam (`make reproduce`) | this repo, SGD-Momentum |
|---|---|---|---|
| accuracy | 0.9632 | **0.8198** | 0.8173 |
| precision (weighted) | 0.9543 | 0.8229 | 0.8225 |
| recall (weighted) | 0.9632 | 0.8198 | 0.8173 |
| F1 (weighted) | 0.9574 | 0.8206 | 0.8185 |
| F1 (macro) | — | **0.7931** | 0.7915 |
| ROC-AUC (OvR, macro) | — | **0.9849** | 0.9835 |

All repo numbers: 34,929 flows, 24 classes, stratified 70/30 split with seed 42, 200 epochs,
batch 32, bytes/255, best-validation-loss checkpoint, evaluated once on the 30 % test split.
Two independent runs of `make reproduce` give byte-identical metrics on the same CPU.

## 2. Averaging is part of the gap, but not most of it

The paper's aggregated metrics are **weighted** averages (its §4 says so explicitly). The
course rubric asks for macro averages, which weight the 24 protocols equally and therefore
penalise the weak minority classes more. That explains the 0.82 → 0.79 difference between the
repo's weighted and macro F1 — but not the 0.96 → 0.82 gap between the paper and the repo.

## 3. What was tried to close the remaining gap

Each of these was run with this codebase on the same 34,929-flow CSV:

- **Paper's best optimizer (SGD-Momentum, lr 0.001, momentum 0.9), 200 epochs:** 0.817 accuracy,
  0.819 weighted F1 — indistinguishable from Adam.
- **Original preprocessing (raw 0–255 bytes, unstratified split), Adam:** the model overfits —
  training accuracy 0.83 while validation loss rises to 0.97 (vs. 0.63 with bytes/255) by epoch 44 —
  so the original input scaling does not produce a better model; the run was stopped there.
- **Keras-style initialisation** (Glorot-uniform, zero bias) is used so that the PyTorch port
  matches the original framework's starting point; PyTorch's default init gave 0.776 accuracy.
- A **Keras 3 / TensorFlow** implementation of the same pipeline (the first pass of this
  conversion) gave 0.814 accuracy / 0.796 macro-F1 / 0.983 ROC-AUC — the same band as PyTorch,
  so the gap is not a framework-port artefact.

## 4. Candidate explanations that could not be tested

The original training runs, their exact data file and Keras/TensorFlow versions are not
archived, so the following remain hypotheses:

1. **A different, easier data file.** The repository holds several CSVs (`dataset-day*.csv`,
   `big-dataset.csv`, `dataset-file*.csv`); Table II may have been computed on an earlier cut with a
   different class mix — the paper itself says the distribution was "fairly unbalanced", whereas
   the curated file is capped at 2,500 flows per class.
2. **Evaluation protocol.** The original script re-fits its label encoder on the test labels and
   re-loads the checkpoint by optimizer name; if a PRED run was made against a model checkpoint
   trained on a different split of the same file, part of the "test" set would have been seen in
   training. This cannot be confirmed or excluded from the surviving files.
3. **Keras-2-era numerics** (the script used `optimizers.Adam(lr=…)`, `np_utils.to_categorical`):
   unlikely to move accuracy by 14 points on its own.

None of these is asserted as the cause. The reproducible result of the method as described in
the paper, on the data that ships with it, is the one in `results/metrics.json`.

## 5. Results analysis (what the repo's figures show)

- **Macro vs. weighted F1 (0.793 vs 0.821).** Six large, "easy" plaintext or fixed-format
  protocols — Kerberos, NetBIOS, LDAP, SMB, SSL_No_Cert, HTTP — score F1 ≥ 0.98 and carry 4,416
  of the 10,479 test flows, pulling the weighted average up. The macro average exposes the weak
  tail below.
- **The TLS cluster is where the model fails.** The five lowest F1 scores are Amazon (0.45), SSL
  (0.56), MS_OneDrive (0.59), SSH (0.59) and Cloudflare (0.61); Google sits at 0.63. In
  `results/confusion_matrix.png` these confuse mostly with *each other* (Amazon↔SSL↔Google↔SSH).
  All are encrypted or TLS-fronted services whose first 1024 bytes are a handshake plus ciphertext;
  nDPI separates them using the SNI in the ClientHello, which a 2-filter, kernel-5 CNN over the
  raw bytes is not well placed to isolate. The paper's own discussion (§4) anticipates exactly this
  (entropy of encrypted payloads).
- **Small classes are not the problem per se.** SMTP (124 test flows) reaches F1 0.86 and
  LinkedIn (144) 0.74; class size matters less than whether the payload is encrypted.
- **Training budget.** With Adam the best validation loss occurs at epoch 110 of 200 and the curve
  is nearly flat afterwards (`results/training_curves.png`); the gap between training and
  validation accuracy (0.88 vs 0.83) indicates mild overfitting rather than under-training.
  The 200-epoch budget is kept for comparability with the paper and was checked rather than
  assumed: the same configuration with `early_stopping_patience: 30` (and a 500-epoch ceiling)
  stops at epoch 94 (best epoch 64) at 0.812 accuracy / 0.784 macro-F1 — i.e. early stopping
  halves the run time for a cost of about 0.01 in both metrics. Set that key in `config.yaml`
  if the shorter run is preferred.
- **Variance, and why the optimizer ranking is not meaningful.** `bash scripts/seed_sweep.sh`
  retrains with seeds 42, 1, 2, 3, 4 (`results/seed_summary.json`). Over those five runs:

  | metric | mean ± std | min – max |
  |---|---|---|
  | accuracy | 0.8153 ± 0.0149 | 0.7888 – 0.8241 |
  | precision (macro) | 0.8078 ± 0.0192 | — |
  | recall (macro) | 0.7829 ± 0.0195 | — |
  | F1 (macro) | 0.7925 ± 0.0172 | 0.7625 – 0.8025 |
  | ROC-AUC (OvR macro) | 0.9845 ± 0.0013 | 0.9823 – 0.9855 |
  | F1 (weighted) | 0.8169 ± 0.0160 | — |

  Seed 42 (the configured default) reproduces `results/metrics.json` exactly, so the sweep also
  serves as a determinism check. **The spread across seeds (±0.015 accuracy, ±0.017 macro-F1) is
  six times larger than the Adam-vs-SGD-Momentum difference measured in §3 (0.0025 accuracy).**
  Any ranking of the five optimizers on a single run each — including the paper's Table II, where
  SGD-Momentum is declared best — is therefore within noise and should not be read as evidence
  that one optimizer beats another on this task. Distinguishing them would need several seeds per
  optimizer and a comparison of the resulting distributions. ROC-AUC is by far the most stable
  metric here (±0.0013) and is the one to prefer when comparing configurations.
  One run (seed 4) lands about one and a half standard deviations low at 0.789 accuracy, which is
  worth remembering when a single number is quoted from a single run of anything in this repo.

## 6. What would move the numbers

Feed the model the ClientHello SNI (or hand it more filters and a longer kernel so it can learn
to read it), add an entropy feature per 64-byte window to separate ciphertext from plaintext,
or merge the TLS-fronted CDN labels (Amazon/Cloudflare/Google/SSL) into one "TLS-other" class
if the downstream use case does not need them apart.
