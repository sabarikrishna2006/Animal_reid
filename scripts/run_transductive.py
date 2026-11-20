"""Re-tune only the transductive stage from a cached second-level output (written by 02_run_cv.py).
Usage: py scripts/run_transductive.py --config configs/exp_g5_trans.json [--no-consensus]
"""
import argparse, json, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from reid.io import load_config, P
from reid.pipeline import prepare, W_TEST
from reid.transductive import apply_transductive
from reid import SPECIES

ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--no-consensus", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config)
if a.no_consensus:
    cfg["transductive"]["consensus"] = False
sl = pickle.loads((P(cfg, "cache") / f"second_level_{cfg.get('exp_slug')}.pkl").read_bytes())
data, ctx, splits = prepare(cfg)
_, best, _ = apply_transductive(cfg, ctx, data, splits, sl["cv"], sl["qdf"], sl["test"], sl["tqdf"], sl["T"], sl["beta"])
ps = np.array([sum(W_TEST[sp] * best[sp]["per_split"][r] for sp in SPECIES) for r in range(len(splits))])
po = np.array([sum(W_TEST[sp] * best[sp]["per_split_off"][r] for sp in SPECIES) for r in range(len(splits))])
d = ps - po
print(f"transductive CV {ps.mean():.4f} ± {ps.std(ddof=1):.4f} | off {po.mean():.4f} | delta {d.mean():+.4f} (SE {d.std(ddof=1)/np.sqrt(5):.4f})")
print(json.dumps({sp: {k: v for k, v in best[sp].items() if "per_split" not in k} for sp in SPECIES}, indent=1))
