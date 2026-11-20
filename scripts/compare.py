"""Paired comparison of experiments on the same 5 CV splits.
Accept rule: mean delta > max(0.002, SE of paired per-split deltas).
Usage: py scripts/compare.py PARENT_EXP CHILD_EXP [CHILD_EXP ...]   (prefix match on experiment folder names)
"""
import json, sys
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parents[1] / "experiments"


def load(prefix):
    ds = sorted(d for d in E.iterdir() if d.is_dir() and (d.name == prefix or d.name.endswith("_" + prefix)))
    d = ds[-1]
    m = json.loads((d / "metrics.json").read_text())
    return d.name, m


def main():
    pn, pm = load(sys.argv[1])
    p = np.array(pm["per_split"])
    print(f"parent {pn}: {p.mean():.4f} ± {p.std(ddof=1):.4f}")
    for c in sys.argv[2:]:
        cn, cm = load(c)
        x = np.array(cm["per_split"]); d = x - p
        se = d.std(ddof=1) / np.sqrt(len(d))
        ok = d.mean() > max(0.002, se)
        print(f"  {cn}: {x.mean():.4f} ± {x.std(ddof=1):.4f} | delta {d.mean():+.4f} (SE {se:.4f}) -> "
              f"{'ACCEPT' if ok else 'reject'}")


if __name__ == "__main__":
    main()
