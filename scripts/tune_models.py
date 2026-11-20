"""Second-level model variants on cached tables (pre-propagation CV MAP@5, paired vs default).
Usage: py scripts/tune_models.py --config configs/exp_v3_sift.json"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from reid.io import load_config
from reid.pipeline import prepare, build_tables, run_second_level, tune_beta, decide, evaluate
from reid import models

ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
cfg = load_config(a.config); data, ctx, splits = prepare(cfg); tabs = build_tables(cfg, data, ctx, splits)
base_params = dict(models.LGB_PARAMS)
VARIANTS = {
    "default": {},
    "leaves15": {"num_leaves": 15},
    "leaves63_minchild50": {"num_leaves": 63, "min_child_samples": 50},
    "ff0.5": {"feature_fraction": 0.5},
    "lr0.015_l2_5": {"learning_rate": 0.015, "lambda_l2": 5.0},
    "minchild100": {"min_child_samples": 100},
}
ref = None
for name, over in VARIANTS.items():
    models.LGB_PARAMS = dict(base_params, **over)
    o = run_second_level(tabs["cv"], cfg, log=lambda *x: None)
    b = {sp: v[0] for sp, v in tune_beta(o["cv"], o["qdf"], "pnew", o["T"], "softmax").items()}
    s, _ = evaluate(o["cv"], decide(o["cv"], o["qdf"], "pnew", b, o["T"], "softmax"), o["qdf"], "pnew")
    ps = np.array(s["per_split"])
    if ref is None: ref = ps
    d = ps - ref
    print(f"{name:22s} CV={ps.mean():.4f} hyena={s['map5_hyena']:.4f} leopard={s['map5_leopard']:.4f} "
          f"delta={d.mean():+.4f} (SE {d.std(ddof=1)/np.sqrt(len(d)):.4f})", flush=True)
