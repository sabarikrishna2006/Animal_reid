"""MAP@5 and diagnostics. The only place query labels (FoldTruth) are consumed."""
import numpy as np
from sklearn.metrics import roc_auc_score

from . import NEW


def map_at_5(truth: dict, preds: dict) -> float:
    total = 0.0
    for qid, y in truth.items():
        labels = preds[qid][:5]
        assert len(labels) == len(set(labels)) and 1 <= len(labels) <= 5
        for k, lab in enumerate(labels, start=1):
            if lab == y:
                total += 1.0 / k
                break
    return total / len(truth)


def per_query_ap(truth: dict, preds: dict) -> dict:
    out = {}
    for qid, y in truth.items():
        labels = preds[qid][:5]
        out[qid] = next((1.0 / k for k, lab in enumerate(labels, 1) if lab == y), 0.0)
    return out


def diagnostics(truth: dict, preds: dict, p_new: dict = None, cand: dict = None) -> dict:
    """truth/preds keyed by query id. p_new: qid -> P(new). cand: qid -> list of candidate identities (for recall@N)."""
    ap = per_query_ap(truth, preds)
    qs = list(truth)
    is_new = np.array([truth[q] == NEW for q in qs])
    apv = np.array([ap[q] for q in qs])
    top1 = np.array([preds[q][0] == truth[q] for q in qs])
    d = {"map5": float(apv.mean()), "n": len(qs), "frac_new": float(is_new.mean()),
         "map5_known": float(apv[~is_new].mean()) if (~is_new).any() else np.nan,
         "map5_new": float(apv[is_new].mean()) if is_new.any() else np.nan,
         "top1_acc": float(top1.mean()),
         "new_recall_at1": float(np.mean([preds[q][0] == NEW for q in np.array(qs)[is_new]])) if is_new.any() else np.nan,
         "new_precision_at1": float(np.mean([truth[q] == NEW for q in qs if preds[q][0] == NEW]))
         if any(preds[q][0] == NEW for q in qs) else np.nan}
    if p_new is not None and 0 < is_new.sum() < len(qs):
        d["new_auc"] = float(roc_auc_score(is_new, [p_new[q] for q in qs]))
    if cand is not None and (~is_new).any():
        d["cand_recall"] = float(np.mean([truth[q] in cand[q] for q in np.array(qs)[~is_new]]))
    return d
