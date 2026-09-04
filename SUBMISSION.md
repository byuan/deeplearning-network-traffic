# Submission Manifest

## Team
- Akshit Jain (RIT ID: ______) — original study: data collection at CPTC 2017, pcap → flow → 1024-byte
  CSV pipeline (`helper-code-files/`, pkt2flow, nDPI), CNN design, optimizer comparison, paper.
- Bo Yuan — advisor; conversion of the original Keras script to the CSEC 520/620 reproducible
  PyTorch layout (`src/`, `config.yaml`, tests, grading harness).

## Course level
- [ ] 520 (undergraduate)
- [ ] 620 (graduate)
- [x] Independent study (this repository is the course's worked exemplar; graded at the 520 bar)

## Problem
Identify the application-layer protocol of a TCP flow (24 classes: HTTP, SSL, SSH, LDAP, Kerberos,
SMB, cloud/web services, …) from the first 1024 payload bytes alone, without port numbers or
flow statistics — the deep-packet-inspection problem that network-monitoring systems need to
solve without hand-written signatures.

## Dataset
- Name / source / version / URL: TCP flows captured at the national Collegiate Penetration
  Testing Competition (CPTC) held at RIT, November 2017; labelled with nDPI; not public.
  34,929 flows × (label + 1024 bytes), 24 classes, class-balanced to ≤ 2,500 flows per class.
- How to obtain it: copy the curated CSV to `data/cptc2017_flows.csv` (see `data/README.md`).
- Target column: column 0 (protocol label); no header.

## How to reproduce
```bash
make setup
make reproduce          # ≈ 9 min on a laptop CPU, ≈ 2 min on a GPU
```
Non-default notes for the grader: the dataset must be supplied out of band (private capture).
`bash scripts/seed_sweep.sh` re-trains with seeds 1–3 and writes `results/seed_summary.json`
(≈ 30 min on CPU); it is a variance check, not part of `make reproduce`.

## Claimed results
Headline numbers from `results/metrics.json` (Adam, 200 epochs, seed 42, stratified 70/30 split,
macro-averaged over 24 classes; see `report/REPRODUCTION_NOTE.md` for how these relate to the
figures in the original paper):
- Accuracy: 0.8198
- Precision / Recall / F1 (macro): 0.7965 / 0.7927 / 0.7931  (weighted F1: 0.8206)
- ROC-AUC (one-vs-rest, macro): 0.9849

## AI-use acknowledgment
The original study, dataset pipeline and paper are the student's own work (2018, no AI
assistance). The 2026 conversion of the monolithic Keras script into this template layout
(`src/`, `config.yaml`, `tests/`, `scripts/`, README and this manifest) was done with Claude
under the advisor's direction; the model architecture, optimizers and evaluation protocol were
carried over from the original script unchanged except where `README.md` says otherwise.
