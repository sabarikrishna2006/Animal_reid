"""Paradigm 2: Two-round neighbor-stacked diverse ensemble (Unconstrained LGB + Listwise LambdaRank + HistGBR).

Provides genuine architectural and objective diversity relative to Paradigm 1 (single-round monotone LGB):
1. Round-1 combines three distinct learners grouped by true underlying identity (GroupKFold):
   - Unconstrained LightGBM (deep non-monotone interactions for background/crop suppression)
   - Listwise LightGBM LambdaRank (optimizes within-query ranking NDCG@5, group-softmax calibrated)
   - sklearn HistGradientBoostingClassifier (independent histogram boosting implementation & L2 reg)
2. Round-2 builds strictly leak-free within-fold query-query neighbor consensus features from OOF Round-1
   posteriors + sw2152/l2 query affinity + cached SIFT verification.
3. Round-2 + P2(new) re-rank using multi-seed bagged models, followed by nested-tuned transductive
   propagation (optional) and OOF probability / test probability export for evaluation and blending.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import NEW, SPECIES
from .calibration import fit_temperature, shift_logit, softmax_groups
from .models import feature_cols
from .pipeline import evaluate, query_table, tune_beta
from .retrieval import fit_sw_and_fullda
from .sift_match import FEATS as SF
from .splits import test_fold


def _group_counts(qkeys):
    _, idx, counts = np.unique(qkeys, return_index=True, return_counts=True)
    # Preserve contiguous block order of qkeys
    order = np.argsort(idx)
    return counts[order]


class UnconstrainedLGB:
    def __init__(self, seed=2026, params=None):
        self.params = dict(
            objective="binary",
            learning_rate=0.025,
            num_leaves=47,
            min_child_samples=30,
            feature_fraction=0.65,
            bagging_fraction=0.80,
            bagging_freq=1,
            lambda_l2=3.0,
            verbose=-1,
            num_threads=10,
            seed=seed,
            bagging_seed=seed,
            feature_fraction_seed=seed,
        )
        if params:
            self.params.update(params)

    def fit(self, X, y, groups, feats, qkeys=None):
        import gc
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=self.params["seed"] % 1000).split(X, y, groups))
        Xtr = X[tr]; dtr = lgb.Dataset(Xtr, y[tr], feature_name=feats, free_raw_data=True).construct()
        del Xtr
        Xva = X[va]; dva = lgb.Dataset(Xva, y[va], reference=dtr, free_raw_data=True).construct()
        del Xva
        gc.collect()
        self.model = lgb.train(
            self.params, dtr, 3000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        self.best_iter = self.model.best_iteration
        del dtr, dva
        gc.collect()
        return self

    def predict(self, X, qkeys=None):
        return np.clip(self.model.predict(X, num_iteration=self.best_iter), 1e-6, 1 - 1e-6)


class LambdaRankLGB:
    """Listwise ranking model trained on contiguous query blocks; calibrated via per-query softmax."""

    def __init__(self, seed=2026, params=None):
        self.params = dict(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 5],
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=25,
            feature_fraction=0.75,
            bagging_fraction=0.80,
            bagging_freq=1,
            lambda_l2=2.0,
            verbose=-1,
            num_threads=10,
            seed=seed,
            bagging_seed=seed,
            feature_fraction_seed=seed,
        )
        if params:
            self.params.update(params)

    def fit(self, X, y, groups, feats, qkeys):
        import gc
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=self.params["seed"] % 1000).split(X, y, groups))
        tr_ord = tr[np.argsort(qkeys[tr], kind="stable")]
        va_ord = va[np.argsort(qkeys[va], kind="stable")]
        Xtr = X[tr_ord]; dtr = lgb.Dataset(Xtr, y[tr_ord], group=_group_counts(qkeys[tr_ord]),
                                          feature_name=feats, free_raw_data=True).construct()
        del Xtr
        Xva = X[va_ord]; dva = lgb.Dataset(Xva, y[va_ord], group=_group_counts(qkeys[va_ord]),
                                          reference=dtr, free_raw_data=True).construct()
        del Xva
        gc.collect()
        self.model = lgb.train(
            self.params, dtr, 2500, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        self.best_iter = self.model.best_iteration
        del dtr, dva
        gc.collect()
        return self

    def predict_raw(self, X):
        return self.model.predict(X, num_iteration=self.best_iter)


class ExtraTreesLGB:
    """Randomized-threshold LightGBM (ExtraTrees GBDT) to decorrelate split boundaries without OpenMP conflicts."""

    def __init__(self, seed=2026, params=None):
        self.params = dict(
            objective="binary",
            extra_trees=True,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=40,
            feature_fraction=0.50,
            bagging_fraction=0.75,
            bagging_freq=1,
            lambda_l2=5.0,
            lambda_l1=1.0,
            verbose=-1,
            num_threads=10,
            seed=seed + 17,
            bagging_seed=seed + 17,
            feature_fraction_seed=seed + 17,
        )
        if params:
            self.params.update(params)

    def fit(self, X, y, groups, feats, qkeys=None):
        import gc
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=(self.params["seed"] + 3) % 1000).split(X, y, groups))
        Xtr = X[tr]; dtr = lgb.Dataset(Xtr, y[tr], feature_name=feats, free_raw_data=True).construct()
        del Xtr
        Xva = X[va]; dva = lgb.Dataset(Xva, y[va], reference=dtr, free_raw_data=True).construct()
        del Xva
        gc.collect()
        self.model = lgb.train(
            self.params, dtr, 3000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        self.best_iter = self.model.best_iteration
        del dtr, dva
        gc.collect()
        return self

    def predict(self, X, qkeys=None):
        return np.clip(self.model.predict(X, num_iteration=self.best_iter), 1e-6, 1 - 1e-6)


class SGDLogModel:
    """Fast regularized linear log-loss model for smooth linear boundary diversity on large tables."""

    def __init__(self, seed=2026):
        self.seed = seed

    def fit(self, X, y, groups, feats, qkeys=None):
        import gc
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(copy=False),
            SGDClassifier(loss="log_loss", alpha=3e-4, max_iter=35, tol=1e-3,
                          early_stopping=True, validation_fraction=0.1, random_state=self.seed),
        )
        self.model.fit(X, y)
        gc.collect()
        return self

    def predict(self, X, qkeys=None):
        return np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


class LRCalibModel:
    def __init__(self, C=0.3):
        self.C = C

    def fit(self, X, y, groups, feats, qkeys=None):
        self.model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=3000),
        )
        self.model.fit(X, y)
        return self

    def predict(self, X, qkeys=None):
        return np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


class MLPCalibModel:
    def __init__(self, seed=2026):
        self.seed = seed

    def fit(self, X, y, groups, feats, qkeys=None):
        self.model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                          validation_fraction=0.15, max_iter=200, random_state=self.seed),
        )
        self.model.fit(X, y)
        return self

    def predict(self, X, qkeys=None):
        return np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


def _rank_to_prob(raw_scores, qkeys):
    gid = pd.factorize(qkeys)[0]
    sm = softmax_groups(raw_scores.astype(np.float64), gid, 1.0)
    return np.clip(sm, 1e-6, 1 - 1e-6)


def _oof_ensemble(df, feats, label, groups, qkeys_col=None, seeds=(2026,), use_rank=True):
    """Out-of-fold 4-family ensemble (Unconstrained LGB + ExtraTrees LGB + LambdaRank + SGD Linear) grouped by `groups`."""
    import gc
    X = df[feats].to_numpy(np.float32)
    if np.isinf(X).any():
        X = np.clip(X, -1e4, 1e4).astype(np.float32)
    y = df[label].to_numpy()
    qk = df[qkeys_col].values if qkeys_col else None
    seeds = list(seeds)

    ulgb_oof = np.zeros(len(df), np.float64)
    et_oof = np.zeros(len(df), np.float64)
    rank_oof_raw = np.zeros(len(df), np.float64)
    sgd_oof = np.zeros(len(df), np.float64)

    ulgb_mods, et_mods, rank_mods, sgd_mods = [], [], [], []
    for sd in seeds:
        for tr, va in GroupKFold(5).split(X, y, groups):
            m1 = UnconstrainedLGB(seed=sd).fit(X[tr], y[tr], groups[tr], feats)
            ulgb_oof[va] += m1.predict(X[va]) / len(seeds)
            ulgb_mods.append(m1)

            m2 = ExtraTreesLGB(seed=sd).fit(X[tr], y[tr], groups[tr], feats)
            et_oof[va] += m2.predict(X[va]) / len(seeds)
            et_mods.append(m2)

            if sd == seeds[0]:
                m4 = SGDLogModel(seed=sd).fit(X[tr], y[tr], groups[tr], feats)
                sgd_oof[va] = m4.predict(X[va])
                sgd_mods.append(m4)

            if use_rank and qk is not None:
                m3 = LambdaRankLGB(seed=sd).fit(X[tr], y[tr], groups[tr], feats, qk[tr])
                rank_oof_raw[va] += m3.predict_raw(X[va]) / len(seeds)
                rank_mods.append(m3)

    out = {
        "ulgb": np.clip(ulgb_oof, 1e-6, 1 - 1e-6),
        "etlgb": np.clip(et_oof, 1e-6, 1 - 1e-6),
        "sgd": np.clip(sgd_oof, 1e-6, 1 - 1e-6),
    }
    if use_rank and qk is not None:
        out["rank_raw"] = rank_oof_raw
        out["rank_p"] = _rank_to_prob(rank_oof_raw, qk)
        l_mix = (
            0.35 * logit(out["ulgb"])
            + 0.25 * logit(out["etlgb"])
            + 0.30 * (rank_oof_raw - pd.Series(rank_oof_raw).groupby(qk).transform("mean").values)
            + 0.10 * logit(out["sgd"])
        )
        out["mix"] = expit(l_mix)
    else:
        l_mix = 0.50 * logit(out["ulgb"]) + 0.35 * logit(out["etlgb"]) + 0.15 * logit(out["sgd"])
        out["mix"] = expit(l_mix)
    mods = {"ulgb": ulgb_mods, "etlgb": et_mods, "rank": rank_mods, "sgd": sgd_mods}
    return out, mods


def _predict_ensemble(mods, X, qk=None, use_rank=True):
    if np.isinf(X).any():
        X = np.clip(X, -1e4, 1e4).astype(np.float32)
    pu = np.mean([m.predict(X) for m in mods["ulgb"]], axis=0)
    pe = np.mean([m.predict(X) for m in mods["etlgb"]], axis=0)
    ps = np.mean([m.predict(X) for m in mods["sgd"]], axis=0)
    out = {
        "ulgb": np.clip(pu, 1e-6, 1 - 1e-6),
        "etlgb": np.clip(pe, 1e-6, 1 - 1e-6),
        "sgd": np.clip(ps, 1e-6, 1 - 1e-6),
    }
    if use_rank and mods["rank"] and qk is not None:
        rr = np.mean([m.predict_raw(X) for m in mods["rank"]], axis=0)
        out["rank_raw"] = rr
        out["rank_p"] = _rank_to_prob(rr, qk)
        l_mix = (
            0.35 * logit(out["ulgb"])
            + 0.25 * logit(out["etlgb"])
            + 0.30 * (rr - pd.Series(rr).groupby(qk).transform("mean").values)
            + 0.10 * logit(out["sgd"])
        )
        out["mix"] = expit(l_mix)
    else:
        l_mix = 0.50 * logit(out["ulgb"]) + 0.35 * logit(out["etlgb"]) + 0.15 * logit(out["sgd"])
        out["mix"] = expit(l_mix)
    return out


def _oof_pnew_per_species(qdf, qfeats, seeds=(2026, 2027)):
    """Per-species OOF P(new) ensemble: 0.65 Unconstrained/ExtraTrees LGB + 0.20 MLP + 0.15 LR."""
    oof = np.zeros(len(qdf), dtype=np.float64)
    mods = {}
    for sp in SPECIES:
        idx = np.where((qdf.species == sp).values)[0]
        sub = qdf.iloc[idx].reset_index(drop=True)
        X = sub[qfeats].to_numpy(np.float32)
        y = sub["is_new"].to_numpy()
        grp = sub["group"].values
        p_lgb = np.zeros(len(sub), np.float64)
        p_mlp = np.zeros(len(sub), np.float64)
        p_lr = np.zeros(len(sub), np.float64)
        m_lgb, m_mlp, m_lr = [], [], []
        for si, sd in enumerate(seeds):
            for tr, va in GroupKFold(5).split(X, y, grp):
                ml = UnconstrainedLGB(seed=sd, params={"num_leaves": 23, "min_child_samples": 20, "lambda_l2": 2.0}).fit(X[tr], y[tr], grp[tr], qfeats)
                p_lgb[va] += ml.predict(X[va]) / len(seeds)
                m_lgb.append(ml)
                if si == 0:
                    mm = MLPCalibModel(seed=sd).fit(X[tr], y[tr], grp[tr], qfeats)
                    mr = LRCalibModel(C=0.3).fit(X[tr], y[tr], grp[tr], qfeats)
                    p_mlp[va] = mm.predict(X[va]); p_lr[va] = mr.predict(X[va])
                    m_mlp.append(mm); m_lr.append(mr)
        oof[idx] = np.clip(0.65 * p_lgb + 0.20 * p_mlp + 0.15 * p_lr, 1e-6, 1 - 1e-6)
        mods[sp] = (m_lgb, m_mlp, m_lr)
    return oof, mods


def _predict_pnew_per_species(mods, tq, qfeats):
    out = np.zeros(len(tq), dtype=np.float64)
    for sp in SPECIES:
        idx = np.where((tq.species == sp).values)[0]
        if not len(idx):
            continue
        X = tq.iloc[idx][qfeats].to_numpy(np.float32)
        m_lgb, m_mlp, m_lr = mods[sp]
        pl = np.mean([m.predict(X) for m in m_lgb], axis=0)
        pm = np.mean([m.predict(X) for m in m_mlp], axis=0)
        pr = np.mean([m.predict(X) for m in m_lr], axis=0)
        out[idx] = np.clip(0.65 * pl + 0.20 * pm + 0.15 * pr, 1e-6, 1 - 1e-6)
    return out


def _attach_r2_neighbors(df, qdf, pcol, pnew_col, data, ctx, folds_by_split, k_nb=8):
    """Attach Round-2 strictly fold-isolated OOF neighbor-stacked consensus features."""
    df = df.reset_index(drop=True).copy()
    # Map (split, query_id) -> R1 candidate dict & P(new)
    post_map = {}
    df["_key"] = df.split.astype(str) + "|" + df.query_id
    for key, g in df.groupby("_key", sort=False):
        post_map[key] = dict(zip(g.cand_id.values, g[pcol].values))
    qkey = qdf.split.astype(str) + "|" + qdf.query_id
    pnew_map = dict(zip(qkey.values, qdf[pnew_col].values))

    # Precompute within-query R1 stats
    rk = df.groupby("_key")[pcol].rank(ascending=False, method="min").values.astype(np.float32)
    qmax = df.groupby("_key")[pcol].transform("max").values
    qtop2 = df.groupby("_key")[pcol].transform(lambda s: np.sort(s.values)[-2] if len(s) > 1 else s.values[0]).values
    df["r1_p"] = df[pcol].astype(np.float32)
    df["r1_rank"] = rk
    df["r1_gap_top"] = (df[pcol].values - np.where(df[pcol].values >= qmax - 1e-9, qtop2, qmax)).astype(np.float32)
    df["r1_pnew"] = df["_key"].map(pnew_map).astype(np.float32)

    # Fold-isolated query-query neighbor aggregation
    nb_feats = np.zeros((len(df), 6), dtype=np.float32)
    idx_inl = SF.index("inl_aff"); idx_rot = SF.index("abs_rot"); idx_spr = SF.index("spread_min")

    for (r, sp), g in df.groupby(["split", "species"], sort=False):
        fold = folds_by_split[(int(r), sp)]
        qr = fold.query_rows; qids = fold.query_ids
        Zq_l2 = ctx.emb["l2"][qr]
        # Gallery-only sw2152 projection for query affinity
        projs = fit_sw_and_fullda(ctx.emb["l2"][fold.gallery_rows], fold.gallery_labels, reg=0.5, d_lda=256)
        Zq_sw = projs["sw"](Zq_l2)
        S_sw = Zq_sw @ Zq_sw.T; np.fill_diagonal(S_sw, -np.inf)
        S_l2 = Zq_l2 @ Zq_l2.T; np.fill_diagonal(S_l2, -np.inf)
        comb = 0.65 * S_sw + 0.35 * S_l2
        nn = np.argsort(-comb, axis=1)[:, :k_nb]

        q_pos = {qid: i for i, qid in enumerate(qids)}
        # Prebuild fast arrays for fold queries
        nb_pnew_w = np.zeros(len(qr), dtype=np.float32)
        nb_cache = []
        for i in range(len(qr)):
            jlist = nn[i]
            w_raw = np.maximum(comb[i, jlist], 0.0)
            ok = (S_sw[i, jlist] > 0.25) | (S_l2[i, jlist] > 0.45)
            w_vec = np.where(ok, w_raw, 0.0)
            wsum = float(w_vec.sum())
            items = []
            pnew_acc = 0.0
            for j, w in zip(jlist, w_vec):
                if w <= 0:
                    continue
                k_j = f"{int(r)}|{qids[j]}"
                d_j = post_map.get(k_j, {})
                pn_j = float(pnew_map.get(k_j, 0.5))
                pnew_acc += w * pn_j
                top1_j = max(d_j, key=d_j.get) if d_j else None
                inl_c = 0.0
                if ctx.pc is not None:
                    f = ctx.pc.get(qr[i], qr[j])
                    if f is not None and np.isfinite(f[idx_inl]):
                        is_bg = np.isfinite(f[idx_rot]) and np.isfinite(f[idx_spr]) and f[idx_rot] < 0.50 and f[idx_spr] < 0.25
                        inl_c = 0.0 if is_bg else float(f[idx_inl])
                items.append((d_j, float(w), float(S_sw[i, j]), top1_j, inl_c))
            nb_pnew_w[i] = (pnew_acc / wsum) if wsum > 0 else float(pnew_map.get(f"{int(r)}|{qids[i]}", 0.5))
            nb_cache.append((items, wsum))

        g_idx = g.index.values
        q_arr = g.query_id.values
        c_arr = g.cand_id.values
        for row_idx, qid, cid in zip(g_idx, q_arr, c_arr):
            qi = q_pos[qid]
            items, wsum = nb_cache[qi]
            if wsum <= 0:
                nb_feats[row_idx, 5] = nb_pnew_w[qi]
                continue
            pmax, pacc, tvote, ssupp, cmax = 0.0, 0.0, 0.0, 0.0, 0.0
            for d_j, w, csw, top1_j, inl_c in items:
                pj = float(d_j.get(cid, 0.0))
                pmax = max(pmax, min(w, pj))
                pacc += w * pj
                if top1_j == cid:
                    tvote += w
                if pj > 0.05 and csw > cmax:
                    cmax = csw
                ssupp = max(ssupp, inl_c * pj)
            nb_feats[row_idx] = (pmax, pacc / wsum, tvote / wsum, ssupp, cmax, nb_pnew_w[qi])

    cols = ["nb_p1_max", "nb_p1_wmean", "nb_top1_votes", "nb_sift_support", "nb_cos_sw_best", "q_nb_pnew_wmean"]
    df[cols] = nb_feats
    df.drop(columns=["_key"], inplace=True)
    return df


def run_dual_pipeline(cfg, data, ctx, splits, cv, test=None, extra=None, log=print):
    """End-to-end Paradigm 2 (Two-round neighbor-stacked LGB + LambdaRank + HistGB)."""
    t0 = time.time()
    mcfg = cfg["model"]
    seeds = mcfg.get("bag_seeds", [2026])
    rich_q = bool(mcfg.get("rich_q", True))

    folds_by_split = {(r, sp): e[sp][0] for r, e in enumerate(splits) for sp in SPECIES}
    if test is not None:
        for sp in SPECIES:
            folds_by_split[(-1, sp)] = test_fold(data, sp)
    if extra is not None and mcfg.get("extra_cv"):
        from .io import ROOT, load_config
        from .pipeline import prepare
        c2 = load_config(ROOT / mcfg["extra_cv"])
        d2, _, s2 = prepare(c2, log=lambda *x: None, load_sift=False)
        for r, e in enumerate(s2):
            for sp in SPECIES:
                folds_by_split[(r + 10, sp)] = e[sp][0]
        del d2, s2

    import gc
    from .pipeline import _shrink_df
    if extra is not None:
        common = [c for c in cv.columns if c in extra.columns]
        cv = pd.concat([cv[common], extra[common]], ignore_index=True)
        del extra
        gc.collect()
    else:
        cv = cv.copy()
    _shrink_df(cv)
    cv["_qkey"] = cv.split.astype(str) + "|" + cv.query_id
    if test is not None:
        test = _shrink_df(test.copy())
        test["_qkey"] = test.split.astype(str) + "|" + test.query_id

    drop = tuple(mcfg.get("drop_features", []))
    feats_r1 = [c for c in feature_cols(cv, drop) if c != "_qkey"]

    # Round 1 OOF + Test
    r1_oof, r1_mods = _oof_ensemble(cv, feats_r1, "y", cv.group.values, qkeys_col="_qkey", seeds=seeds, use_rank=True)
    cv["p_r1"] = r1_oof["mix"]
    cv["p_r1_rank"] = r1_oof["rank_p"]
    cv["p_r1_ulgb"] = r1_oof["ulgb"]
    cv["p_r1_etlgb"] = r1_oof["etlgb"]
    log(f"[dual R1] {len(feats_r1)} feats, 4-family OOF done in {time.time() - t0:.1f}s")

    qdf_r1 = query_table(cv, pcol="p_r1", rich_q=rich_q)
    qfeats_r1 = [c for c in feature_cols(qdf_r1) if c != "_qkey"]
    pn1_oof, pn1_mods = _oof_pnew_per_species(qdf_r1, qfeats_r1, seeds=(2026, 2027))
    qdf_r1["pnew_r1"] = pn1_oof

    if test is not None:
        r1_te = _predict_ensemble(r1_mods, test[feats_r1].to_numpy(np.float32), qk=test["_qkey"].values, use_rank=True)
        test["p_r1"] = r1_te["mix"]
        test["p_r1_rank"] = r1_te["rank_p"]
        test["p_r1_ulgb"] = r1_te["ulgb"]
        test["p_r1_etlgb"] = r1_te["etlgb"]
        tq_r1 = query_table(test, pcol="p_r1", rich_q=rich_q)
        tq_r1["pnew_r1"] = _predict_pnew_per_species(pn1_mods, tq_r1, qfeats_r1)

    # Round 2 neighbor-stacked features (strictly OOF R1 inputs)
    t1 = time.time()
    cv = _attach_r2_neighbors(cv, qdf_r1, "p_r1", "pnew_r1", data, ctx, folds_by_split, k_nb=int(mcfg.get("r2_k", 8)))
    if test is not None:
        test = _attach_r2_neighbors(test, tq_r1, "p_r1", "pnew_r1", data, ctx, folds_by_split, k_nb=int(mcfg.get("r2_k", 8)))
    log(f"[dual R2] neighbor stacking attached in {time.time() - t1:.1f}s")

    feats_r2 = [c for c in feature_cols(cv, drop) if c != "_qkey"]
    r2_oof, r2_mods = _oof_ensemble(cv, feats_r2, "y", cv.group.values, qkeys_col="_qkey", seeds=seeds, use_rank=True)
    cv["p"] = r2_oof["mix"]
    cv["p_ulgb"] = r2_oof["ulgb"]
    cv["p_rank"] = r2_oof["rank_p"]
    cv["p_etlgb"] = r2_oof["etlgb"]
    log(f"[dual R2] {len(feats_r2)} feats, 4-family OOF done in {time.time() - t0:.1f}s")

    qdf = query_table(cv, pcol="p", rich_q=rich_q)
    # Also carry R2 neighbor-level query features into qdf
    q_extra_cols = ["q_nb_pnew_wmean", "r1_pnew", "nb_p1_max", "nb_p1_wmean", "nb_top1_votes", "nb_sift_support"]
    top_r2 = cv.loc[cv.groupby("_qkey")["p"].idxmax()].set_index("_qkey")
    q_idx = qdf.split.astype(str) + "|" + qdf.query_id
    for c in q_extra_cols:
        qdf[f"top_{c}"] = q_idx.map(top_r2[c]).values
    qfeats = [c for c in feature_cols(qdf) if c != "_qkey"]
    pn2_oof, pn2_mods = _oof_pnew_per_species(qdf, qfeats, seeds=(2026, 2027))
    qdf["pnew"] = pn2_oof

    if test is not None:
        r2_te = _predict_ensemble(r2_mods, test[feats_r2].to_numpy(np.float32), qk=test["_qkey"].values, use_rank=True)
        test["p"] = r2_te["mix"]
        test["p_ulgb"] = r2_te["ulgb"]
        test["p_rank"] = r2_te["rank_p"]
        test["p_etlgb"] = r2_te["etlgb"]
        tq = query_table(test, pcol="p", rich_q=rich_q)
        top_t2 = test.loc[test.groupby("_qkey")["p"].idxmax()].set_index("_qkey")
        tq_idx = tq.split.astype(str) + "|" + tq.query_id
        for c in q_extra_cols:
            tq[f"top_{c}"] = tq_idx.map(top_t2[c]).values
        tq["pnew"] = _predict_pnew_per_species(pn2_mods, tq, qfeats)
        test.drop(columns=["_qkey"], inplace=True)
    cv.drop(columns=["_qkey"], inplace=True)

    # Restrict reporting/evaluation to primary splits (< 10)
    cv = cv[cv.split < 10].reset_index(drop=True)
    qdf = qdf[qdf.split < 10].reset_index(drop=True)
    T = {}
    for sp in SPECIES:
        d = cv[cv.species == sp]
        gid = pd.factorize(d.split.astype(str) + "|" + d.query_id)[0]
        T[sp] = float(fit_temperature(logit(np.clip(d.p.values, 1e-6, 1 - 1e-6)), gid, d.y.values))
    bt = tune_beta(cv, qdf, "pnew", T, "softmax")
    beta = {sp: v[0] for sp, v in bt.items()}
    preds = None
    from .pipeline import decide
    preds = decide(cv, qdf, "pnew", beta, T, "softmax")
    summ, split_res = evaluate(cv, preds, qdf, "pnew")
    from sklearn.metrics import roc_auc_score
    aucs = {sp: roc_auc_score(qdf[qdf.species == sp].is_new, qdf[qdf.species == sp].pnew) for sp in SPECIES}
    log(f"[dual softmax] CV MAP@5 = {summ['cv_mean']:.4f} ± {summ['cv_std']:.4f} "
        f"| hyena {summ['map5_hyena']:.4f} leopard {summ['map5_leopard']:.4f} "
        f"| known h/l {summ['map5_known_hyena']:.3f}/{summ['map5_known_leopard']:.3f} "
        f"| newAUC h/l {aucs['hyena']:.3f}/{aucs['leopard']:.3f} | beta {beta} T {[round(T[s], 2) for s in SPECIES]}")

    out = {"cv": cv, "qdf": qdf, "T": T, "beta": beta, "feats": feats_r2, "qfeats": qfeats}
    if test is not None:
        out.update(test=test, tqdf=tq)
    return out
