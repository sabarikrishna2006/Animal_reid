"""Two-round LightGBM model training, Round-2 neighbor features, and MAP@5 ranking for dev_pipeline."""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from core import NEW

N_THREADS = min(16, os.cpu_count() or 8)
PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=N_THREADS
)


def fit(P, Qf, pf, qf, seed=0):
    p = dict(PARAMS, seed=seed)
    return (
        lgb.train(p, lgb.Dataset(P[pf], P.y), 600),
        lgb.train(p, lgb.Dataset(Qf[qf], Qf.is_new), 400),
    )


def round2_feats(P, Qf):
    P = P.copy()
    Qf = Qf.copy()
    P["p1_rank"] = P.groupby("qid").p1.rank(ascending=False).astype(np.float32)
    P["p1_norm"] = (P.p1 / P.groupby("qid").p1.transform("sum").clip(1e-9)).astype(np.float32)
    pmap = {(q, c): float(p) for q, c, p in zip(P.qid.values, P.cand.values, P.p1.values)}
    pn = dict(zip(Qf.qid.values, Qf.pn1.values))
    P["pn1_self"] = P.qid.map(pn).astype(np.float32)
    top = P.sort_values("p1", ascending=False).drop_duplicates("qid").set_index("qid")
    topc, topp = top.cand.to_dict(), top.p1.to_dict()
    nb = {q: (ids, np.array(s, np.float32)) for q, ids, s in zip(Qf.qid.values, Qf.nbr_ids.values, Qf.nbr_sims.values)}
    a = np.zeros((len(P), 4), np.float32)
    for r, (q, c) in enumerate(zip(P.qid.values, P.cand.values)):
        ids, s = nb[q]
        v = np.array([pmap.get((n, c), 0.0) for n in ids], np.float32)
        w = np.clip(s, 0.0, None)
        a[r] = (np.max(s * v), (w * v).sum() / max(w.sum(), 1e-6), v[0] * s[0], np.sum((v > 0.5) & (s > 0.6)))
    P[["nb_p_max", "nb_p_w", "nb0_p", "nb_cnt"]] = a
    g = P.groupby("qid").p1
    Qf["p1_max"] = Qf.qid.map(g.max()).astype(np.float32)
    Qf["p1_2nd"] = Qf.qid.map(g.apply(lambda x: float(np.sort(x.values)[-2]) if len(x) > 1 else float(x.values[0]))).astype(np.float32)
    b = np.zeros((len(Qf), 4), np.float32)
    for r, (q, ids, s) in enumerate(zip(Qf.qid.values, Qf.nbr_ids.values, Qf.nbr_sims.values)):
        s = np.array(s, np.float32)
        w = np.clip(s, 0.0, None)
        pns = np.array([pn[n] for n in ids], np.float32)
        conf = np.array([topp[n] for n in ids], np.float32)
        agree = np.array([float(topc[n] == topc[q]) for n in ids], np.float32)
        b[r] = ((w * pns).sum() / max(w.sum(), 1e-6), pns[0], np.max(s * conf), np.max(s * agree))
    Qf[["nb_pn_w", "nb0_pn", "nb_known_conf", "nb_agree"]] = b
    return P, Qf


def rank(P, Qf, pcol, ncol, w=1.0):
    pn = dict(zip(Qf.qid, Qf[ncol]))
    out = {}
    for qid, g in P.groupby("qid", sort=False):
        s = g[pcol].values
        o = np.argsort(-s)
        lst = []
        placed = False
        for j in o[:5]:
            if not placed and s[j] < w * pn[qid]:
                lst.append(NEW)
                placed = True
            lst.append(g.cand.values[j])
        if not placed:
            lst.append(NEW)
        out[qid] = lst[:5]
    return out


def score(preds, Qf):
    s = 0.0
    for q, t in zip(Qf.qid.values, Qf.truth.values):
        for k, lab in enumerate(preds[q][:5]):
            if lab == t:
                s += 1.0 / (k + 1)
                break
    return s / len(Qf)
