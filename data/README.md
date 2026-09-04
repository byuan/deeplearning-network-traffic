# Data

**Do not commit datasets to Git.** They are large and bloat history.

## This project's dataset

- **Name / source:** TCP flows captured during the national CPTC held at RIT (November 2017),
  labelled with the application protocol detected by nDPI. Payloads were extracted with
  `tcpflow`; only the first 1024 bytes of each flow are kept and flows shorter than 1024 bytes
  were discarded. See the accompanying independent-study paper for details.
- **File:** `data/cptc2017_flows.csv` — 34,929 rows, 24 protocol classes, header-less CSV.
  Column 0 is the protocol label (string); columns 1–1024 are the raw payload bytes (0–255).
  This is the class-balanced file produced by `helper-code-files/balance-dataset.py`
  (≤ 2,500 flows per class; classes with < 400 flows dropped).
- **Obtain:** the capture is not public. Copy the curated CSV into this folder:

  ```bash
  cp /path/to/dataset.csv data/cptc2017_flows.csv
  ```

  (In the original project layout the file lives at the repository root as `dataset.csv`.)
- **Config:** `config.yaml` → `data.source: csv`, `data.csv_path: data/cptc2017_flows.csv`.
- **Notes:** class frequencies range from 2,842 (LDAP) down to 412 (SMTP). The split is
  stratified 70/30 with `seed: 42`; bytes are rescaled by a constant 1/255 (no fitted
  scaler, so no train/test leakage).

The raw, un-balanced per-day CSVs (`dataset-day*.csv`, `big-dataset.csv`, …) are inputs to
the helper scripts only and are not used by `make reproduce`.
