"""Summarise metrics.json files from a seed sweep: mean and std per headline metric.

usage: python scripts/summarize_seeds.py results/seeds results/seed_summary.json
"""
import json, sys
from pathlib import Path
import numpy as np

KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "f1_weighted")

root, out = Path(sys.argv[1]), Path(sys.argv[2])
runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/metrics.json"))}
if not runs:
    sys.exit(f"no metrics.json under {root}")
summary = {"n_runs": len(runs), "seeds": sorted(runs, key=lambda k: int(k) if k.isdigit() else k),
           "per_seed": {s: {k: m[k] for k in KEYS} for s, m in runs.items()}, "mean": {}, "std": {}}
for k in KEYS:
    v = np.array([m[k] for m in runs.values()])
    summary["mean"][k] = float(v.mean()); summary["std"][k] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
out.write_text(json.dumps(summary, indent=2))
print(f"{len(runs)} runs (seeds {summary['seeds']})")
for k in KEYS:
    print(f"  {k:12s} {summary['mean'][k]:.4f} ± {summary['std'][k]:.4f}")
print("wrote", out)
