"""Nested estimate of the transductive stage: choose (alpha, tau_e, new_scale) on 4 splits, score on the 5th.
Compares with the in-sample (tuned-on-all-5) CV to measure selection optimism.
Usage: py scripts/nested_check.py EXPERIMENT_ID
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = {"hyena": 1217 / 2791, "leopard": 1574 / 2791}
m = json.loads((ROOT / "experiments" / sys.argv[1] / "metrics.json").read_text())
tr = m["trans"]
nested, insample, off = np.zeros(5), np.zeros(5), np.zeros(5)
for sp, w in W.items():
    grid = tr[sp]["grid"]
    P = np.array([g[3] for g in grid])            # combos x splits
    for r in range(5):
        others = [k for k in range(5) if k != r]
        c = P[:, others].mean(1).argmax()
        nested[r] += w * P[c, r]
    insample += w * np.array(tr[sp]["per_split"])
    off += w * np.array(tr[sp]["per_split_off"])
d = nested - off
print(f"in-sample tuned : {insample.mean():.4f}")
print(f"nested (honest) : {nested.mean():.4f} ± {nested.std(ddof=1):.4f}")
print(f"no propagation  : {off.mean():.4f}")
print(f"nested gain over no-propagation: {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(5):.4f})")
