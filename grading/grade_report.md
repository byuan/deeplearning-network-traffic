# Grade report — deeplearning-network-traffic (re-grade after corrections)

**Total: 90 / 100** (previous grade: 77 / 100)  ·  2026-09-04  ·  per `grading/AGENT_GRADING.md` / `grading/rubric.yaml`
Course level: independent study, graded at the 520 bar (declared in `SUBMISSION.md`).

## Evidence used

`python grading/grade.py` (15:53 run, tests check refreshed after a flaky-test fix) → `grading/auto_report.json`: structure **pass**, tests **pass**, reproduce_runs **pass**, metrics_present **pass**, leakage_scan **clean**, report_present **pass**, git_hygiene **pass**. Read: all of `src/`, `config.yaml`, `tests/`, `scripts/`, `SUBMISSION.md`, `README.md`, `data/README.md`, `report/REPRODUCTION_NOTE.md`, the paper PDF, and the generated `results/`. No integrity issues.

## What changed since the previous grade

| previous finding | status |
|---|---|
| Paper at repo root, not in `report/`; `report_present` = review | Paper moved to `report/`; check passes |
| Paper claims 0.9632 accuracy, repo reproduces 0.8198, nothing reconciles them | `report/REPRODUCTION_NOTE.md` reconciles: averaging explained, SGD-Momentum / raw-byte / TensorFlow variants all tested (0.81–0.82), 0.96 declared unreproduced, repo numbers declared authoritative |
| `SUBMISSION.md` blank | Filled (team, level, problem, dataset, claimed results from `metrics.json`, AI-use) |
| Dead code and vendored trees in the tree | Git-ignored and documented; committed repo is clean |
| No variance estimate | `scripts/seed_sweep.sh` + `summarize_seeds.py`; `--seed/--out` overrides in `train.py` |
| 200-epoch budget unjustified | Tested: early stopping (patience 30) stops at epoch 94 for −0.01 accuracy/F1; documented, budget kept for comparability |
| — (new) flaky smoke test caught by the harness | Fixed: losses compared in eval mode with a fixed seed; 5/5 consecutive passes |

## Scores by criterion

### Reproducibility — 23 / 25 (unchanged)

`make reproduce` runs unattended in about ten minutes on a CPU and has now produced byte-identical metrics on three independent runs, the third performed by the grading harness itself. The two withheld points remain for the private dataset that a grader must be handed separately; `data/README.md` and `SUBMISSION.md` say so plainly.

### Evaluation validity & metrics — 19 / 20 (+1)

Everything from the previous grade holds (stratified seeded split, constant scaling, no leakage, macro and per-class metrics, ROC-AUC). The gaps I flagged are now addressed: `REPRODUCTION_NOTE.md` §5 explains the macro-vs-weighted F1 gap (six plaintext protocols at F1 ≥ 0.98 carry 42 % of the test set) and diagnoses the TLS cluster, and a seed sweep exists. One point withheld only because the sweep has not yet been run and its spread quoted.

### Implementation correctness — 19 / 20 (+1)

The harness exposed a real defect — the smoke test compared losses in train mode, where dropout noise could make it fail at random — and it is fixed. The epoch budget is now an evidence-based choice rather than an inherited one. The one point withheld is for class imbalance still being handled only offline (the balancing script), not in the loss.

### Code quality & organization — 9 / 10 (+2)

The committed repository is now clean: legacy scripts, notebooks, backup weights and the `scons/`, `nDPI/`, `pkt2flow/` sources are git-ignored and their status documented in the README; `pytest.ini` keeps collection to `tests/`; the Team section is filled. The last point returns when those directories are physically deleted.

### Report quality — 15 / 20 (+9)

This is where the correction earned its points. `report/` now contains the paper and a reconciliation note that does what the rubric demands — it confronts the mismatch between the paper's Table II and `results/metrics.json` instead of hiding it. It identifies the weighted-vs-macro averaging as part of the gap, shows through three further experiments (the paper's best optimizer, the original preprocessing, an independent TensorFlow port) that the method lands at 0.81–0.82 accuracy on the shipped data regardless, states that the 0.96 could not be reproduced, lists candidate causes without asserting any, and establishes the reproducible numbers as the ones to grade against. `SUBMISSION.md` cites those numbers.

Five points remain withheld because the paper itself is untouched: a reader who opens the PDF without the note still sees a table the repository cannot back, it has no labelled literature-review section, and it is not in the IEEE two-column format the rubric names.

### Security relevance & originality — 5 / 5 (unchanged)

The failure analysis of the TLS-fronted services, and the concrete proposals to fix it (SNI extraction, per-window entropy, label merging), add security insight on top of the original dataset work.

## Fixes, in priority order

1. Run `bash scripts/seed_sweep.sh` and quote the mean ± std from `results/seed_summary.json` in `SUBMISSION.md` and the note.
2. Revise the paper or add an erratum page in `report/` that replaces Table II with the reproduced numbers; if time permits, move to the IEEE template and add a short related-work section.
3. Physically delete the git-ignored legacy files and vendored tool sources.
4. Optionally add a class-weighted loss and report whether macro-F1 moves.
