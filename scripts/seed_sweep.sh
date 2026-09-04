#!/usr/bin/env bash
# Variance estimate: retrain with several seeds, then summarise.
#   bash scripts/seed_sweep.sh            # seeds 1 2 3
#   bash scripts/seed_sweep.sh 1 2 3 4 5
# Writes results/seeds/<seed>/ and results/seed_summary.json. Not part of `make reproduce`.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3)
for s in "${SEEDS[@]}"; do
  echo "=== seed $s ==="
  $PY -m src.train --config config.yaml --seed "$s" --out "results/seeds/$s" | tail -n 12
done
$PY scripts/summarize_seeds.py results/seeds results/seed_summary.json
