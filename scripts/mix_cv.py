"""Train second-level models on CV tables from TWO split designs (v1 easier, v2 harder), evaluate on one of them.
The test set's difficulty lies between the two designs; mixing makes the reranker / P(new) model robust to that shift.
OOF grouping is by true identity across BOTH designs, so no animal is ever both train and evaluation for a model.
Usage: py scripts/mix_cv.py --eval configs/exp_v7_g5wide.json --extra configs/exp_v3_sift.json
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from reid.io import load_config
from reid.pipeline import prepare, build_tables, run_second_level, tune_beta, decide, evaluate


def tables(cfg_path):
    cfg = load_config(cfg_path); data, ctx, splits = prepare(cfg, log=lambda *x: None)
    return cfg, build_tables(cfg, data, ctx, splits, log=lambda *x: None)


def score(o, mask_df, mask_q):
    cv, q = o["cv"][mask_df], o["qdf"][mask_q]
    b = {sp: v[0] for sp, v in tune_beta(cv, q, "pnew", o["T"], "softmax").items()}
    s, _ = evaluate(cv, decide(cv, q, "pnew", b, o["T"], "softmax"), q, "pnew")
    return s, b


ap = argparse.ArgumentParser(); ap.add_argument("--eval"); ap.add_argument("--extra"); a = ap.parse_args()
cfg, te = tables(a.eval); _, tx = tables(a.extra)
A = te["cv"].copy(); B = tx["cv"].copy(); B["split"] = B["split"] + 10        # keep (split, query) keys unique
common = [c for c in A.columns if c in B.columns]
A, B = A[common], B[common]

o_single = run_second_level(A, cfg, log=lambda *x: None)
s1, _ = score(o_single, np.ones(len(o_single["cv"]), bool), np.ones(len(o_single["qdf"]), bool))
o_mix = run_second_level(pd.concat([A, B], ignore_index=True), cfg, log=lambda *x: None)
mdf = (o_mix["cv"].split < 10).values; mq = (o_mix["qdf"].split < 10).values
s2, b2 = score(o_mix, mdf, mq)
d = np.array(s2["per_split"]) - np.array(s1["per_split"])
print(f"eval design only : CV={s1['cv_mean']:.4f} hyena={s1['map5_hyena']:.4f} leopard={s1['map5_leopard']:.4f} newAUC h/l={s1['new_auc_hyena']:.3f}/{s1['new_auc_leopard']:.3f}")
print(f"mixed training   : CV={s2['cv_mean']:.4f} hyena={s2['map5_hyena']:.4f} leopard={s2['map5_leopard']:.4f} newAUC h/l={s2['new_auc_hyena']:.3f}/{s2['new_auc_leopard']:.3f} beta={b2}")
print(f"paired delta {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})")
s3, _ = score(o_mix, ~mdf, ~mq)
print(f"mixed model on the extra (harder) design: CV={s3['cv_mean']:.4f}")
