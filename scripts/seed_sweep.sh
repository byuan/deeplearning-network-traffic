#!/usr/bin/env bash
# Variance estimate: retrain with several seeds, then summarise.
#   bash scripts/seed_sweep.sh            # seeds 42 1 2 3 4 (the ones reported)
#   bash scripts/seed_sweep.sh 7 8 9      # or any seeds you choose
# Writes results/seeds/<seed>/ and results/seed_summary.json. Not part of `make reproduce`.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 1 2 3 4)   # 42 = the seed in config.yaml
for s in "${SEEDS[@]}"; do
  echo "=== seed $s ==="
  $PY -m src.train --config config.yaml --seed "$s" --out "results/seeds/$s" | tail -n 12
done
$PY scripts/summarize_seeds.py results/seeds results/seed_summary.json
