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
from .models import feature_cols, oof_predict, fit_full, predict_bag
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
def query_table(df, pcol="p", rich_q=False):
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
    if rich_q:
        q["p1_minus_p2"] = q["p1"] - q["p2"]
        q["p1_over_psum"] = q["p1"] / np.maximum(q["psum"], 1e-6)
        q["p_entropy"] = g[pcol].apply(lambda s: float(-np.sum((s.values / max(s.values.sum(), 1e-9)) * np.log(np.clip(s.values / max(s.values.sum(), 1e-9), 1e-9, 1.0)))))
        q["p3"] = srt.map(lambda a: a[2] if len(a) > 2 else 0)
    top = df.loc[g[pcol].idxmax()].set_index("_key")
    top_cols = ("max_same_side", "has_same_side", "compat_best", "max_minus_compat", "recip_rank", "log_nimg",
                "max_opp_side", "sift_inl_max", "z_pcaw", "z_l2", "z_lda",
                "lnb_img", "lnb_img_rel", "lnb_img_share", "lnb_k3", "lnb_log", "lnb_rel",
                "krec_l2", "krec_lda", "diff_l2", "diff_lda", "cls_lr", "cls_mlp",
                "lnb_uimg", "lnb_uimg_rel", "sift_inl_clean", "sift_best_is_bg",
                "lnb_uimg_share", "sift_inl_clean2", "sift_is_bg2",
                "hub_l2", "hub_lda", "best_g_hub_l2", "best_g_hub_lda",
                "lnb_rimg", "lnb_rimg_rel", "max_lda512", "max_slda512")
    if rich_q:
        top_cols = top_cols + ("sift_rot_best", "sift_spread_best", "sift_n6_clean")
    for c in top_cols:
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


def load_extra_cv(cfg, log=print):
    """Optional second CV design whose tables are added to the TRAINING data of the second-level models
    (config: model.extra_cv = path of another experiment config). Its splits are offset by +10."""
    path = cfg["model"].get("extra_cv")
    if not path:
        return None
    from .io import load_config, ROOT
    c2 = load_config(ROOT / path)
    d2, ctx2, s2 = prepare(c2, log=lambda *x: None)
    ex = build_tables(c2, d2, ctx2, s2, log=lambda *x: None)["cv"].copy()
    ex["split"] = ex["split"] + 10
    log(f"[extra_cv] {path}: {len(ex)} extra training rows")
    return ex


def _rg_diff(Zg, glab, K=10):
    Sgg = Zg @ Zg.T
    Sgg_diff = np.where(glab[:, None] == glab[None, :], -2.0, Sgg)
    k = min(K, max(1, len(Zg) - 1))
    return np.partition(Sgg_diff, -k, axis=1)[:, -k:].mean(1).astype(np.float32)


def _fold_hubness(fold, ctx, split_id):
    """100% gallery-only different-identity hubness features for MiewID L2 & Gallery-LDA."""
    from .retrieval import fit_lda
    Zg_l2 = ctx.emb["l2"][fold.gallery_rows]
    Zq_l2 = ctx.emb["l2"][fold.query_rows]
    glab = fold.gallery_labels
    f_lda = fit_lda(Zg_l2, glab, gamma=0.3, d_out=256)
    Zg_ld, Zq_ld = f_lda(Zg_l2), f_lda(Zq_l2)

    rg_l2 = _rg_diff(Zg_l2, glab, K=10)
    rg_ld = _rg_diff(Zg_ld, glab, K=10)
    Hl2 = (Zq_l2 @ Zg_l2.T) - 0.50 * rg_l2[None, :]
    Hld = (Zq_ld @ Zg_ld.T) - 0.75 * rg_ld[None, :]

    names, inv = np.unique(glab, return_inverse=True)
    nid = len(names)
    members = [np.where(inv == j)[0] for j in range(nid)]
    nq = len(fold.query_ids)
    Il2 = np.full((nq, nid), -9.0, np.float32); Ild = np.full((nq, nid), -9.0, np.float32)
    Bl2 = np.zeros((nq, nid), np.float32); Bld = np.zeros((nq, nid), np.float32)
    ar = np.arange(nq)
    for j, cols in enumerate(members):
        sub2 = Hl2[:, cols]; idx2 = sub2.argmax(1)
        Il2[:, j] = sub2[ar, idx2]; Bl2[:, j] = rg_l2[cols[idx2]]
        subd = Hld[:, cols]; idxd = subd.argmax(1)
        Ild[:, j] = subd[ar, idxd]; Bld[:, j] = rg_ld[cols[idxd]]
    top2_l2 = -np.sort(-Il2, axis=1)[:, :2]
    top2_ld = -np.sort(-Ild, axis=1)[:, :2]
    id_pos = {n: j for j, n in enumerate(names)}
    q_info = {}
    for i, qid in enumerate(fold.query_ids):
        m1 = float(top2_l2[i, 0]); m2 = float(top2_l2[i, 1]) if nid > 1 else m1
        l1 = float(top2_ld[i, 0]); l2 = float(top2_ld[i, 1]) if nid > 1 else l1
        q_info[(split_id, qid)] = (id_pos, Il2[i], Ild[i], Bl2[i], Bld[i], m1, m1 - m2, l1, l1 - l2)
    return q_info


def attach_hubness(cfg, data, ctx, splits, cv, test, extra=None, log=print):
    """Attach 100% gallery-only different-identity hubness features to cv, test, and extra tables."""
    t0 = time.time()
    q_info = {}
    for r, e in enumerate(splits):
        for sp in SPECIES:
            q_info.update(_fold_hubness(e[sp][0], ctx, r))
    if test is not None:
        for sp in SPECIES:
            q_info.update(_fold_hubness(test_fold(data, sp), ctx, -1))
    if extra is not None and cfg["model"].get("extra_cv"):
        from .io import load_config, ROOT
        c2 = load_config(ROOT / cfg["model"]["extra_cv"])
        d2, ctx2, s2 = prepare(c2, log=lambda *x: None)
        for r, e in enumerate(s2):
            for sp in SPECIES:
                q_info.update(_fold_hubness(e[sp][0], ctx2, r + 10))

    cols = ["hub_l2", "hub_lda", "hub_l2_gap", "hub_lda_gap", "best_g_hub_l2", "best_g_hub_lda",
            "q_hub_l2_max", "q_hub_l2_gap", "q_hub_lda_max", "q_hub_lda_gap"]

    def _apply(df):
        if df is None:
            return None
        df = df.copy()
        rows = []
        for r, q, c in zip(df["split"].values, df["query_id"].values, df["cand_id"].values):
            id_pos, il2, ild, bl2, bld, m1, mg, l1, lg = q_info[(int(r), q)]
            j = id_pos.get(c, -1)
            if j >= 0:
                v2, vd = float(il2[j]), float(ild[j])
                rows.append((v2, vd, m1 - v2, l1 - vd, float(bl2[j]), float(bld[j]), m1, mg, l1, lg))
            else:
                rows.append((-9.0, -9.0, 9.0, 9.0, 0.0, 0.0, m1, mg, l1, lg))
        df[cols] = np.array(rows, dtype=np.float32)
        return df

    cv = _apply(cv); test = _apply(test); extra = _apply(extra)
    log(f"[hubness] attached gallery-only r_g_diff features in {time.time() - t0:.1f}s")
    return cv, test, extra


def attach_rimg(cfg, data, ctx, splits, cv, test, extra=None, log=print):
    """Attach strict cross-identity Lowe ratio LNBNN (lnb_rimg) features strictly masked to each fold's gallery."""
    from .lnbnn import scorer
    t0 = time.time()
    sc_map = {sp: scorer(ctx, sp) for sp in SPECIES}
    q_info = {}

    def _run_fold(fold, split_id):
        S = sc_map[fold.species]
        gids = data.meta.id.values[fold.gallery_rows]
        gmask = np.isin(S.gal, gids)
        for qid in fold.query_ids:
            v = S.scores_rimg(qid, gmask)
            o = np.argsort(-v)
            m1 = float(v[o[0]]); m2 = float(v[o[1]]) if len(v) > 1 else 0.0
            q_info[(split_id, qid)] = (S.names, v, m1, m1 - m2)

    for r, e in enumerate(splits):
        for sp in SPECIES:
            _run_fold(e[sp][0], r)
    if test is not None:
        for sp in SPECIES:
            _run_fold(test_fold(data, sp), -1)
    if extra is not None and cfg["model"].get("extra_cv"):
        from .io import load_config, ROOT
        c2 = load_config(ROOT / cfg["model"]["extra_cv"])
        d2, _, s2 = prepare(c2, log=lambda *x: None)
        for r, e in enumerate(s2):
            for sp in SPECIES:
                _run_fold(e[sp][0], r + 10)

    cols = ["lnb_rimg", "lnb_rimg_rel", "q_lnb_rimg_max", "q_lnb_rimg_gap"]

    def _apply(df):
        if df is None:
            return None
        df = df.copy()
        rows = []
        for r, q, c in zip(df["split"].values, df["query_id"].values, df["cand_id"].values):
            names, v, m1, mg = q_info[(int(r), q)]
            j = int(np.searchsorted(names, c))
            val = float(v[j]) if (j < len(names) and names[j] == c) else 0.0
            rows.append((val, val / m1 if m1 > 0 else 0.0, m1, mg))
        df[cols] = np.array(rows, dtype=np.float32)
        return df

    cv = _apply(cv); test = _apply(test); extra = _apply(extra)
    log(f"[lnb_rimg] attached strict cross-ID ratio LNBNN features in {time.time() - t0:.1f}s")
    return cv, test, extra


def _fold_lda512(fold, ctx, split_id):
    """100% gallery-only PCA(512)->LDA(0.5) AND Side-Conditioned PCA(512)->LDA(0.3) + r_g_diff hubness."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from .retrieval import l2n
    Zg0 = ctx.emb["l2"][fold.gallery_rows]
    Zq0 = ctx.emb["l2"][fold.query_rows]
    glab = fold.gallery_labels
    gside = ctx.side[fold.gallery_rows]
    glab_side = np.array([f"{a}|{b}" for a, b in zip(glab, gside)])

    m0 = Zg0.mean(0)
    nc = min(512, len(Zg0) - 1, Zg0.shape[1])
    pca = PCA(nc, random_state=0).fit(Zg0 - m0)
    g5, q5 = pca.transform(Zg0 - m0), pca.transform(Zq0 - m0)

    l5 = LDA(solver="eigen", shrinkage=0.5).fit(g5, glab)
    Zg_l, Zq_l = l2n(l5.transform(g5)), l2n(l5.transform(q5))
    ls = LDA(solver="eigen", shrinkage=0.3).fit(g5, glab_side)
    Zg_s, Zq_s = l2n(ls.transform(g5)), l2n(ls.transform(q5))

    rg_l = _rg_diff(Zg_l, glab, K=10)
    rg_s = _rg_diff(Zg_s, glab, K=10)
    Sl = (Zq_l @ Zg_l.T) - 0.50 * rg_l[None, :]
    Ss = (Zq_s @ Zg_s.T) - 0.75 * rg_s[None, :]

    names, inv = np.unique(glab, return_inverse=True)
    nid = len(names); nq = len(fold.query_ids)
    members = [np.where(inv == j)[0] for j in range(nid)]
    Il = np.full((nq, nid), -9.0, np.float32); Is = np.full((nq, nid), -9.0, np.float32)
    for j, cols in enumerate(members):
        Il[:, j] = Sl[:, cols].max(1)
        Is[:, j] = Ss[:, cols].max(1)
    top2_l = -np.sort(-Il, axis=1)[:, :2]
    top2_s = -np.sort(-Is, axis=1)[:, :2]
    id_pos = {n: j for j, n in enumerate(names)}
    q_info = {}
    for i, qid in enumerate(fold.query_ids):
        m1 = float(top2_l[i, 0]); m2 = float(top2_l[i, 1]) if nid > 1 else m1
        s1 = float(top2_s[i, 0]); s2 = float(top2_s[i, 1]) if nid > 1 else s1
        q_info[(split_id, qid)] = (id_pos, Il[i], Is[i], m1, m1 - m2, s1, s1 - s2)
    return q_info


def attach_lda512(cfg, data, ctx, splits, cv, test, extra=None, log=print):
    """Attach 100% gallery-only PCA(512)-LDA + Side-Conditioned LDA + hubness features."""
    t0 = time.time()
    q_info = {}
    for r, e in enumerate(splits):
        for sp in SPECIES:
            q_info.update(_fold_lda512(e[sp][0], ctx, r))
    if test is not None:
        for sp in SPECIES:
            q_info.update(_fold_lda512(test_fold(data, sp), ctx, -1))
    if extra is not None and cfg["model"].get("extra_cv"):
        from .io import load_config, ROOT
        c2 = load_config(ROOT / cfg["model"]["extra_cv"])
        d2, ctx2, s2 = prepare(c2, log=lambda *x: None)
        for r, e in enumerate(s2):
            for sp in SPECIES:
                q_info.update(_fold_lda512(e[sp][0], ctx2, r + 10))

    cols = ["max_lda512", "max_slda512", "gap_lda512", "gap_slda512",
            "q_lda512_max", "q_lda512_gap", "q_slda512_max", "q_slda512_gap"]

    def _apply(df):
        if df is None:
            return None
        df = df.copy()
        rows = []
        for r, q, c in zip(df["split"].values, df["query_id"].values, df["cand_id"].values):
            id_pos, il, is_, m1, mg, s1, sg = q_info[(int(r), q)]
            j = id_pos.get(c, -1)
            if j >= 0:
                vl, vs = float(il[j]), float(is_[j])
                rows.append((vl, vs, m1 - vl, s1 - vs, m1, mg, s1, sg))
            else:
                rows.append((-9.0, -9.0, 9.0, 9.0, m1, mg, s1, sg))
        df[cols] = np.array(rows, dtype=np.float32)
        return df

    cv = _apply(cv); test = _apply(test); extra = _apply(extra)
    log(f"[lda512] attached gallery-only PCA(512)-LDA & Side-LDA features in {time.time() - t0:.1f}s")
    return cv, test, extra


def _sw_whiten_2152(Zg, Zq, m0, labels_sw, gamma=0.5):
    """100% gallery-only Full 2152-dim Within-Class Covariance Whitening (zero PCA truncation)."""
    from .retrieval import l2n
    names_sw, inv_sw = np.unique(labels_sw, return_inverse=True)
    cnt = np.bincount(inv_sw)
    valid = cnt[inv_sw] >= 2
    mu = np.zeros((len(names_sw), Zg.shape[1]), dtype=np.float32)
    for j in np.where(cnt >= 2)[0]:
        mu[j] = Zg[inv_sw == j].mean(0)
    Zw = (Zg[valid] - mu[inv_sw[valid]]).astype(np.float64)
    Sw = (Zw.T @ Zw) / max(len(Zw), 1)
    ew, vw = np.linalg.eigh(Sw)
    ew = np.maximum(ew, 0.0)
    scale = (1.0 / np.sqrt((1.0 - gamma) * ew + gamma * ew.mean())).astype(np.float32)
    vw32 = vw.astype(np.float32)
    return l2n(((Zg - m0) @ vw32) * scale[None, :]), l2n(((Zq - m0) @ vw32) * scale[None, :])


def _fold_sw2152(fold, ctx, split_id):
    """100% gallery-only Full 2152-dim Within-Class Covariance Whitening (ID + Side) + r_g_diff hubness."""
    Zg0 = ctx.emb["l2"][fold.gallery_rows]
    Zq0 = ctx.emb["l2"][fold.query_rows]
    glab = fold.gallery_labels
    gside = ctx.side[fold.gallery_rows]
    glab_side = np.array([f"{a}|{b}" if b in ("L", "R") else a for a, b in zip(glab, gside)])
    m0 = Zg0.mean(0)

    Zg_i, Zq_i = _sw_whiten_2152(Zg0, Zq0, m0, glab, gamma=0.5)
    Zg_s, Zq_s = _sw_whiten_2152(Zg0, Zq0, m0, glab_side, gamma=0.5)
    rg_i = _rg_diff(Zg_i, glab, K=10)
    rg_s = _rg_diff(Zg_s, glab, K=10)
    Si = (Zq_i @ Zg_i.T) - 0.50 * rg_i[None, :]
    Ss = (Zq_s @ Zg_s.T) - 0.50 * rg_s[None, :]

    names, inv = np.unique(glab, return_inverse=True)
    nid = len(names); nq = len(fold.query_ids)
    members = [np.where(inv == j)[0] for j in range(nid)]
    Ii = np.full((nq, nid), -9.0, np.float32); Is = np.full((nq, nid), -9.0, np.float32)
    for j, cols in enumerate(members):
        Ii[:, j] = Si[:, cols].max(1)
        Is[:, j] = Ss[:, cols].max(1)
    top2_i = -np.sort(-Ii, axis=1)[:, :2]
    top2_s = -np.sort(-Is, axis=1)[:, :2]
    id_pos = {n: j for j, n in enumerate(names)}
    q_info = {}
    for i, qid in enumerate(fold.query_ids):
        m1 = float(top2_i[i, 0]); m2 = float(top2_i[i, 1]) if nid > 1 else m1
        s1 = float(top2_s[i, 0]); s2 = float(top2_s[i, 1]) if nid > 1 else s1
        q_info[(split_id, qid)] = (id_pos, Ii[i], Is[i], m1, m1 - m2, s1, s1 - s2)
    return q_info


def attach_sw2152(cfg, data, ctx, splits, cv, test, extra=None, log=print):
    """Attach 100% gallery-only Full 2152-dim Sw Whitening (ID + Side) + r_g_diff hubness features."""
    t0 = time.time()
    q_info = {}
    for r, e in enumerate(splits):
        for sp in SPECIES:
            q_info.update(_fold_sw2152(e[sp][0], ctx, r))
    if test is not None:
        for sp in SPECIES:
            q_info.update(_fold_sw2152(test_fold(data, sp), ctx, -1))
    if extra is not None and cfg["model"].get("extra_cv"):
        from .io import load_config, ROOT
        c2 = load_config(ROOT / cfg["model"]["extra_cv"])
        d2, ctx2, s2 = prepare(c2, log=lambda *x: None)
        for r, e in enumerate(s2):
            for sp in SPECIES:
                q_info.update(_fold_sw2152(e[sp][0], ctx2, r + 10))

    cols = ["max_sw2152", "max_ssw2152", "gap_sw2152", "gap_ssw2152",
            "q_sw2152_max", "q_sw2152_gap", "q_ssw2152_max", "q_ssw2152_gap"]

    def _apply(df):
        if df is None:
            return None
        df = df.copy()
        rows = []
        for r, q, c in zip(df["split"].values, df["query_id"].values, df["cand_id"].values):
            id_pos, ii, is_, m1, mg, s1, sg = q_info[(int(r), q)]
            j = id_pos.get(c, -1)
            if j >= 0:
                vi, vs = float(ii[j]), float(is_[j])
                rows.append((vi, vs, m1 - vi, s1 - vs, m1, mg, s1, sg))
            else:
                rows.append((-9.0, -9.0, 9.0, 9.0, m1, mg, s1, sg))
        df[cols] = np.array(rows, dtype=np.float32)
        return df

    cv = _apply(cv); test = _apply(test); extra = _apply(extra)
    log(f"[sw2152] attached gallery-only Full 2152-dim Sw Whitening features in {time.time() - t0:.1f}s")
    return cv, test, extra


def _add_bg_clean_v2(df):
    """Full-frame static camera-trap killer (opt-in bg_clean_v2):
    Flags 0-deg rotation matches whose unique-ID LNBNN share (lnb_uimg / lnb_img) is < 0.35
    (92% FP rate on leopard, 100% FP rate on hyena, keeping 99.2%+ of true matches)."""
    df = df.copy()
    u_share = df["lnb_uimg"].values / np.maximum(df["lnb_img"].values, 1e-6)
    df["lnb_uimg_share"] = u_share.astype(np.float32)
    df["q_lnb_uimg_share"] = (df["q_lnb_uimg_max"].values / np.maximum(df["q_lnb_img_max"].values, 1e-6)).astype(np.float32)
    full_bg = (df["sift_rot_best"].values < 0.50) & (u_share < 0.35) & (df["sift_inl_max"].values >= 6)
    is_bg2 = (df["sift_best_is_bg"].values > 0.5) | full_bg
    df["sift_is_bg2"] = is_bg2.astype(np.float32)
    df["sift_inl_clean2"] = np.where(is_bg2, 0.0, df["sift_inl_clean"].values).astype(np.float32)
    key = df["split"].astype(str) + "|" + df["query_id"]
    qmax = df.groupby(key)["sift_inl_clean2"].transform("max").values
    df["q_sift_inl_clean2_max"] = qmax.astype(np.float32)
    df["sift_inl_clean2_minus_qmax"] = (df["sift_inl_clean2"].values - qmax).astype(np.float32)
    return df


def run_second_level(cv, cfg, log=print, test=None, extra=None):
    """OOF reranker -> OOF P(new) -> T, beta. If test given, also fit on all CV and predict test.
    extra: additional CV tables (another split design) used ONLY as training data; outputs are restricted to `cv`."""
    mcfg = cfg["model"]
    if extra is not None:
        common = [c for c in cv.columns if c in extra.columns]
        cv = pd.concat([cv[common], extra[common]], ignore_index=True)
    if mcfg.get("bg_clean_v2", False):
        cv = _add_bg_clean_v2(cv)
        if test is not None:
            test = _add_bg_clean_v2(test)
    drop = tuple(mcfg.get("drop_features", []))
    feats = feature_cols(cv, drop)
    groups = cv.group.values
    t0 = time.time()
    cv = cv.copy()
    kinds = mcfg.get("rerankers", [mcfg.get("reranker", "lgb")])
    bag_seeds = mcfg.get("bag_seeds", None)
    bag_test = bool(mcfg.get("bag_test", False))
    rich_q = bool(mcfg.get("rich_q", False))
    logits = []; rer_bags = {}
    for kd in kinds:
        if bag_test:
            pk, iters, mods = oof_predict(cv, feats, "y", groups, kind=kd, return_models=True, seeds=bag_seeds)
            rer_bags[kd] = mods
        else:
            pk, iters = oof_predict(cv, feats, "y", groups, kind=kd, seeds=bag_seeds)
        cv[f"p_{kd}"] = pk
        logits.append(logit(np.clip(pk, 1e-6, 1 - 1e-6)))
        log(f"[reranker:{kd}] {len(feats)} feats, OOF done in {time.time() - t0:.0f}s, best iters {iters[:10]}")
    cv["p"] = 1 / (1 + np.exp(-np.mean(logits, axis=0)))
    qdf = query_table(cv, rich_q=rich_q)
    qfeats = feature_cols(qdf)
    qgroups = qdf.group.values
    pnew_mods = None
    if bag_test:
        qdf["pnew"], iters, pnew_mods = oof_predict(qdf, qfeats, "is_new", qgroups, kind=mcfg.get("pnew_model", "lgb"),
                                                    return_models=True, seeds=bag_seeds)
    else:
        qdf["pnew"], iters = oof_predict(qdf, qfeats, "is_new", qgroups, kind=mcfg.get("pnew_model", "lgb"), seeds=bag_seeds)
    log(f"[pnew] {len(qfeats)} feats, OOF done, iters {iters[:10]}")
    T = {}
    for sp in SPECIES:
        d = cv[cv.species == sp]
        gid = pd.factorize(d.split.astype(str) + "|" + d.query_id)[0]
        T[sp] = float(fit_temperature(logit(np.clip(d.p.values, 1e-6, 1 - 1e-6)), gid, d.y.values))
    cv_all, qdf_all = cv, qdf
    cv = cv[cv.split < 10].reset_index(drop=True)          # report / tune only on the primary design
    qdf = qdf[qdf.split < 10].reset_index(drop=True)
    T = {}
    for sp in SPECIES:
        d = cv[cv.species == sp]
        gid = pd.factorize(d.split.astype(str) + "|" + d.query_id)[0]
        T[sp] = float(fit_temperature(logit(np.clip(d.p.values, 1e-6, 1 - 1e-6)), gid, d.y.values))
    out = {"cv": cv, "qdf": qdf, "T": T, "feats": feats, "qfeats": qfeats}
    cv, qdf = cv_all, qdf_all                                # final models are fit on everything
    if test is not None:
        test = test.copy(); tl = []; rer = None
        for kd in kinds:
            if bag_test:
                pk = predict_bag(rer_bags[kd], test[feats].to_numpy(np.float32))
                rer = rer_bags[kd]
            else:
                rer = fit_full(cv, feats, "y", groups, kind=kd)
                pk = rer.predict(test[feats].to_numpy(np.float32))
            test[f"p_{kd}"] = pk
            tl.append(logit(np.clip(pk, 1e-6, 1 - 1e-6)))
        test["p"] = 1 / (1 + np.exp(-np.mean(tl, axis=0)))
        tq = query_table(test, rich_q=rich_q)
        if bag_test:
            tq["pnew"] = predict_bag(pnew_mods, tq[qfeats].to_numpy(np.float32))
            pm = pnew_mods
        else:
            pm = fit_full(qdf, qfeats, "is_new", qgroups, kind=mcfg.get("pnew_model", "lgb"))
            tq["pnew"] = pm.predict(tq[qfeats].to_numpy(np.float32))
        out.update(test=test, tqdf=tq, reranker=rer, pnew_model=pm)
    return out
