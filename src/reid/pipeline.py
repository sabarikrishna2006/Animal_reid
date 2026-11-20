"""End-to-end pipeline shared by scripts/02_run_cv.py and scripts/03_make_submission.py."""
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from scipy.special import logit

from . import NEW, SPECIES
from .calibration import fit_temperature, softmax_groups, shift_logit
from .candidates import Context, build_table, sift_pairs_for_fold, query_neighbour_pairs
from .decision import rank_labels
from .io import P, Sift, load_data
from .metrics import diagnostics, map_at_5
from .models import feature_cols, oof_predict, fit_full
from .retrieval import embed_unsupervised
from .sift_cache import PairCache
from .splits import build_splits, test_fold

W_TEST = {"hyena": 1217 / 2791, "leopard": 1574 / 2791}


def _hash(d):
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def prepare(cfg, log=print):
    t0 = time.time()
    data = load_data(cfg)
    r = cfg["retrieval"]
    emb = {"l2": embed_unsupervised(data.X, data.meta, "l2"),
           "pcaw": embed_unsupervised(data.X, data.meta, "pcaw", d=r.get("pcaw_d", 512), eps=r.get("pcaw_eps", 1.0))}
    kp = Sift(cfg).n_kp().astype(float)
    splits = build_splits(data, cfg, emb["l2"])
    use_sift = cfg["sift"].get("enabled", True)
    pc = PairCache(cfg) if use_sift else None
    ctx = Context(data, emb, kp, cfg, pc)
    log(f"[prepare] data+embeddings+splits in {time.time() - t0:.0f}s")
    return data, ctx, splits


def build_tables(cfg, data, ctx, splits, log=print):
    tkey = {"enabled": False} if not cfg["transductive"].get("features") else {"enabled": False, "features": True}
    skey = {k: v for k, v in cfg["sift"].items() if k != "workers"}
    skey["workers"] = 10
    key = _hash({"cv": cfg["cv"], "retrieval": cfg["retrieval"], "sift": skey, "transductive": tkey} | {"v": 3})
    path = P(cfg, "cache") / "tables" / f"tables_{key}.pkl"
    if path.exists():
        log(f"[tables] cached {path.name}")
        return pickle.loads(path.read_bytes())
    folds = [(r, sp, fold, truth) for r, e in enumerate(splits) for sp, (fold, truth) in e.items()]
    tfolds = [(sp, test_fold(data, sp)) for sp in SPECIES]
    trans = cfg["transductive"].get("features", False)
    if ctx.pc is not None:
        t0 = time.time()
        pairs = []
        for _, _, fold, _ in folds:
            pairs += sift_pairs_for_fold(fold, ctx)
        for _, fold in tfolds:
            pairs += sift_pairs_for_fold(fold, ctx)
        log(f"[sift] {len(set(pairs))} pairs requested ({time.time() - t0:.0f}s to list)")
        ctx.pc.ensure(pairs)
    n_cand = cfg["retrieval"]["n_cand"]
    lda_cfg = cfg["retrieval"].get("lda", {"gamma": 0.3, "d_out": 256})
    cv, test = [], []
    t0 = time.time()
    for r, sp, fold, truth in folds:
        df = build_table(fold, ctx, n_cand, transductive=trans, lda_cfg=lda_cfg)
        td = truth.as_dict()
        df["truth"] = df.query_id.map(td)
        df["y"] = (df.cand_id == df.truth).astype(int)
        df["is_new"] = (df.truth == NEW).astype(int)
        # group = true underlying identity (for OOF grouping only, never a feature)
        df["group"] = data.meta.identity.values[fold.query_rows[df.qi.values]]
        df["split"] = r; df["species"] = sp
        cv.append(df)
        log(f"  [table] split {r} {sp}: {len(fold.query_rows)} queries, {len(df)} rows ({time.time() - t0:.0f}s)")
    for sp, fold in tfolds:
        df = build_table(fold, ctx, n_cand, transductive=trans, lda_cfg=lda_cfg)
        df["split"] = -1; df["species"] = sp
        test.append(df)
    out = {"cv": pd.concat(cv, ignore_index=True), "test": pd.concat(test, ignore_index=True)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    log(f"[tables] built in {time.time() - t0:.0f}s -> {path.name}")
    return out


# ---------------------------------------------------------------- query-level table
def query_table(df, pcol="p"):
    df = df.copy()
    df["_key"] = df.split.astype(str) + "|" + df.query_id
    g = df.groupby("_key", sort=False)
    first = g.head(1).set_index("_key")
    q = pd.DataFrame(index=first.index)
    srt = g[pcol].apply(lambda s: np.sort(s.values)[::-1])
    q["p1"] = srt.map(lambda a: a[0]); q["p2"] = srt.map(lambda a: a[1] if len(a) > 1 else 0)
    q["psum"] = g[pcol].sum()
    q["lgap"] = logit(np.clip(q.p1, 1e-6, 1 - 1e-6)) - logit(np.clip(q.p2, 1e-6, 1 - 1e-6))
    q["n_p10"] = g[pcol].apply(lambda s: (s > 0.1).sum())
    top = df.loc[g[pcol].idxmax()].set_index("_key")
    for c in ("max_same_side", "has_same_side", "compat_best", "max_minus_compat", "recip_rank", "log_nimg",
              "max_opp_side", "sift_inl_max", "z_pcaw", "z_l2", "z_lda"):
        if c in top:
            q[f"top_{c}"] = top[c]
    for c in first.columns:
        if c.startswith("q_") or c in ("species_is_leo",):
            q[c] = first[c]
    for c in ("split", "query_id", "species", "is_new", "group", "truth"):
        if c in first:
            q[c] = first[c]
    return q.reset_index(drop=True)


# ---------------------------------------------------------------- decision
def decide(df, qdf, pnew_col, beta, T, method="softmax"):
    """Return {(split, query_id): [labels]}"""
    df = df.copy()
    df["_key"] = df.split.astype(str) + "|" + df.query_id
    qkey = qdf.split.astype(str) + "|" + qdf.query_id
    pn = pd.Series(qdf[pnew_col].values, index=qkey)
    gid = pd.factorize(df["_key"])[0]
    if method == "softmax":
        Tv = df.species.map(T).values if isinstance(T, dict) else T
        post = softmax_groups(logit(np.clip(df.p.values, 1e-6, 1 - 1e-6)), gid, Tv)
    else:
        post = df.p.values
    sp_beta = df.species.map(beta) if isinstance(beta, dict) else beta
    df["post"] = post
    out = {}
    qb = qdf.species.map(beta).values if isinstance(beta, dict) else np.full(len(qdf), beta)
    pn = pd.Series(shift_logit(qdf[pnew_col].values, qb), index=qkey)
    for key, g in df.groupby("_key", sort=False):
        pi = pn[key]
        probs = g.post.values * (1 - pi) if method == "softmax" else g.post.values
        out[key] = rank_labels(g.cand_id.values, probs, pi)
    return out


def evaluate(df, preds, qdf=None, pnew_col=None):
    """MAP per split/species + test-weighted overall."""
    res = {}
    qkey = None
    for (r, sp), g in df.groupby(["split", "species"]):
        q = g.drop_duplicates("query_id")
        truth = dict(zip(q.query_id, q.truth))
        pr = {qid: preds[f"{r}|{qid}"] for qid in truth}
        pnew = None
        if qdf is not None:
            qq = qdf[(qdf.split == r) & (qdf.species == sp)]
            pnew = dict(zip(qq.query_id, qq[pnew_col]))
        cand = g.groupby("query_id").cand_id.apply(set).to_dict()
        res[(r, sp)] = diagnostics(truth, pr, pnew, cand)
    splits = sorted({r for r, _ in res})
    sps = [sp for sp in SPECIES if any(k[1] == sp for k in res)]
    wsum = sum(W_TEST[sp] for sp in sps)
    overall = np.array([sum(W_TEST[sp] / wsum * res[(r, sp)]["map5"] for sp in sps) for r in splits])
    summ = {"cv_mean": float(overall.mean()), "cv_std": float(overall.std(ddof=1)), "per_split": overall.tolist()}
    for sp in sps:
        for k in ("map5", "map5_known", "map5_new", "top1_acc", "new_recall_at1", "new_precision_at1", "new_auc",
                  "cand_recall", "frac_new"):
            vals = [res[(r, sp)].get(k, np.nan) for r in splits]
            summ[f"{k}_{sp}"] = float(np.nanmean(vals))
    return summ, res


def _query_arrays(df, qdf, pnew_col, T, method):
    """Per query: top-5 candidate posteriors (desc, padded 0), rank of truth among candidates (0-based, 99 if absent),
    is_new flag, raw P(new), species, split."""
    df = df.copy()
    df["_key"] = df.split.astype(str) + "|" + df.query_id
    gid = pd.factorize(df["_key"])[0]
    if method == "softmax":
        df["post"] = softmax_groups(logit(np.clip(df.p.values, 1e-6, 1 - 1e-6)), gid, T if not isinstance(T, dict) else
                                    df.species.map(T).values)
    else:
        df["post"] = df.p.values
    df = df.sort_values(["_key", "post"], ascending=[True, False])
    df["r"] = df.groupby("_key").cumcount()
    top = df[df.r < 5].pivot(index="_key", columns="r", values="post").reindex(columns=range(5)).fillna(0)
    tr = df[df.y == 1].set_index("_key").r
    q = qdf.assign(_key=qdf.split.astype(str) + "|" + qdf.query_id).set_index("_key")
    top = top.reindex(q.index).fillna(0)
    return dict(top=top.values, trank=tr.reindex(q.index).fillna(99).values.astype(int), is_new=q.is_new.values,
                pnew=q[pnew_col].values, species=q.species.values, split=q.split.values)


def fast_ap(A, beta, method):
    pi = shift_logit(A["pnew"], beta)
    cand = A["top"] * ((1 - pi)[:, None] if method == "softmax" else 1.0)
    knew = 1 + (cand > pi[:, None]).sum(1)                  # rank position of new_individual
    tr = A["trank"]
    pos = tr + 1 + (knew <= tr + 1)                          # position of true identity after inserting new
    ap_known = np.where(pos <= 5, 1.0 / pos, 0.0)
    ap_new = np.where(knew <= 5, 1.0 / knew, 0.0)
    return np.where(A["is_new"] == 1, ap_new, np.where(tr < 99, ap_known, 0.0))


def tune_beta(df, qdf, pnew_col, T, method, grid=np.round(np.arange(-3, 3.01, 0.05), 2)):
    A = _query_arrays(df, qdf, pnew_col, T, method)
    best = {}
    for sp in SPECIES:
        m = A["species"] == sp
        sub = {k: v[m] for k, v in A.items()}
        scores = [np.mean([fast_ap(sub, b, method)[sub["split"] == r].mean() for r in np.unique(sub["split"])])
                  for b in grid]
        best[sp] = (float(grid[int(np.argmax(scores))]), float(np.max(scores)))
    return best


def run_second_level(cv, cfg, log=print, test=None):
    """OOF reranker -> OOF P(new) -> T, beta. If test given, also fit on all CV and predict test."""
    mcfg = cfg["model"]
    drop = tuple(mcfg.get("drop_features", []))
    feats = feature_cols(cv, drop)
    groups = cv.group.values
    t0 = time.time()
    cv = cv.copy()
    kinds = mcfg.get("rerankers", [mcfg.get("reranker", "lgb")])
    logits = []
    for kd in kinds:
        pk, iters = oof_predict(cv, feats, "y", groups, kind=kd)
        cv[f"p_{kd}"] = pk
        logits.append(logit(np.clip(pk, 1e-6, 1 - 1e-6)))
        log(f"[reranker:{kd}] {len(feats)} feats, OOF done in {time.time() - t0:.0f}s, best iters {iters}")
    cv["p"] = 1 / (1 + np.exp(-np.mean(logits, axis=0)))
    qdf = query_table(cv)
    qfeats = feature_cols(qdf)
    qgroups = qdf.group.values
    qdf["pnew"], iters = oof_predict(qdf, qfeats, "is_new", qgroups, kind=mcfg.get("pnew_model", "lgb"))
    log(f"[pnew] {len(qfeats)} feats, OOF done, iters {iters}")
    T = {}
    for sp in SPECIES:
        d = cv[cv.species == sp]
        gid = pd.factorize(d.split.astype(str) + "|" + d.query_id)[0]
        T[sp] = float(fit_temperature(logit(np.clip(d.p.values, 1e-6, 1 - 1e-6)), gid, d.y.values))
    out = {"cv": cv, "qdf": qdf, "T": T, "feats": feats, "qfeats": qfeats}
    if test is not None:
        test = test.copy(); tl = []
        for kd in kinds:
            rer = fit_full(cv, feats, "y", groups, kind=kd)
            pk = rer.predict(test[feats].to_numpy(np.float32)); test[f"p_{kd}"] = pk
            tl.append(logit(np.clip(pk, 1e-6, 1 - 1e-6)))
        test["p"] = 1 / (1 + np.exp(-np.mean(tl, axis=0)))
        tq = query_table(test)
        pm = fit_full(qdf, qfeats, "is_new", qgroups, kind=mcfg.get("pnew_model", "lgb"))
        tq["pnew"] = pm.predict(tq[qfeats].to_numpy(np.float32))
        out.update(test=test, tqdf=tq, reranker=rer, pnew_model=pm)
    return out
