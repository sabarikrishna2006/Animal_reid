"""(query, candidate identity) feature tables. Receives a FoldData (no query labels) -> features never read labels.

Embeddings used: 'l2' (raw MiewID, L2-normalised), 'pcaw' (PCA-whitened on train U test features, unsupervised),
'lda' (regularised LDA fitted on THIS fold's gallery labels only).
"""
import numpy as np
import pandas as pd

from .retrieval import IdentityIndex, fit_lda, fit_sw_and_fullda, l2n, rg_diff, vp_compat_stats
from .sift_match import FEATS as SF

SIDE = {"left": "L", "frontleft": "L", "backleft": "L", "right": "R", "frontright": "R", "backright": "R"}
EMBS = ("l2", "pcaw", "lda")


def side_of(v):
    return SIDE.get(v, "N") if isinstance(v, str) else "U"


def impute_viewpoint(U, meta, seed=2026):
    """Fill missing viewpoints with a per-species LogisticRegression on PCA-128 embeddings.
    Trained on all images WITH a recorded viewpoint (viewpoint is a given input, never an identity label)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    vp = meta.viewpoint.copy()
    for sp in meta.species.unique():
        m = (meta.species == sp).values
        miss = m & vp.isna().values
        if not miss.any():
            continue
        Z = PCA(128, random_state=seed).fit_transform(U[m])
        known = ~vp.isna().values[m]
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Z[known], vp.values[m][known])
        vp.iloc[np.where(miss)[0]] = clf.predict(Z[~known])
    return vp


class Context:
    """Global, label-free resources shared across folds."""

    def __init__(self, data, emb, kp, cfg, pair_cache=None):
        self.data, self.emb, self.kp, self.cfg, self.pc = data, emb, kp, cfg, pair_cache
        vp = data.meta.viewpoint.copy()
        if cfg["retrieval"].get("vp_impute", False):
            vp = impute_viewpoint(emb["l2"], data.meta)
        self.vp = vp.fillna("NaN").values
        self.side = np.array([side_of(v) for v in vp.values])
        self.lognorm = np.log(np.linalg.norm(data.X.astype(np.float64), axis=1))
        self._v25_cache = {}


def vp_compat(Zg, labels, vps):
    """Mean same-identity cosine per viewpoint pair, from gallery labels only. Returns dict (va, vb) -> mean."""
    out = {}
    s = pd.Series(np.arange(len(labels))).groupby(labels)
    acc = {}
    for _, idx in s:
        idx = idx.values
        if len(idx) < 2:
            continue
        S = Zg[idx] @ Zg[idx].T
        iu = np.triu_indices(len(idx), 1)
        va, vb = vps[idx][iu[0]], vps[idx][iu[1]]
        for a, b, x in zip(va, vb, S[iu]):
            k = (a, b) if a <= b else (b, a)
            t = acc.setdefault(k, [0.0, 0])
            t[0] += x; t[1] += 1
    for k, (sm, n) in acc.items():
        if n >= 10:
            out[k] = sm / n
    return out


def manifold_diffusion(Zq, Zg, k=20, alpha=0.85, gamma=3.0, iters=25):
    """Iscen et al. CVPR 2017 manifold diffusion on [Zq; Zg] cosine kNN graph (unsupervised, zero labels)."""
    import scipy.sparse as sp
    nq, ng = len(Zq), len(Zg)
    Z = np.vstack([Zq, Zg]).astype(np.float32)
    N = len(Z)
    S = np.clip(Z @ Z.T, 0.0, 1.0)
    np.fill_diagonal(S, 0.0)
    kk = min(k, N - 1)
    top = np.argpartition(-S, kk, axis=1)[:, :kk]
    rows = np.repeat(np.arange(N), kk)
    cols = top.ravel()
    vals = (S[rows, cols] ** gamma).astype(np.float32)
    W = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    W = 0.5 * (W + W.T)
    d = np.asarray(W.sum(1)).ravel()
    dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    Sn = sp.diags(dinv) @ W @ sp.diags(dinv)
    Y = np.zeros((N, nq), dtype=np.float32)
    Y[np.arange(nq), np.arange(nq)] = 1.0
    F = Y.copy()
    for _ in range(iters):
        F = alpha * (Sn @ F) + (1.0 - alpha) * Y
    return F[nq:, :].T


def k_reciprocal_rerank(Zq, Zg, k1=20, k2=6, lam=0.3):
    """Classical k-reciprocal Jaccard re-ranking (Zhong et al. CVPR 2017) on cosine distances (unsupervised)."""
    nq, ng = len(Zq), len(Zg)
    Z = np.vstack([Zq, Zg]).astype(np.float32)
    N = len(Z)
    orig_dist = np.clip(2.0 - 2.0 * (Z @ Z.T), 0.0, 4.0)
    orig_dist = orig_dist / np.maximum(orig_dist.max(axis=1, keepdims=True), 1e-6)
    initial_rank = np.argsort(orig_dist, axis=1).astype(np.int32)
    V = np.zeros((N, N), dtype=np.float32)
    hk = int(round(k1 / 2)) + 1
    for i in range(N):
        fwd = initial_rank[i, :k1 + 1]
        bwd = initial_rank[fwd, :k1 + 1]
        k_recip = fwd[np.where(bwd == i)[0]]
        k_recip_exp = k_recip
        for cand in k_recip:
            cfwd = initial_rank[cand, :hk]
            cbwd = initial_rank[cfwd, :hk]
            c_recip = cfwd[np.where(cbwd == cand)[0]]
            if len(np.intersect1d(c_recip, k_recip)) > (2.0 / 3.0) * len(c_recip):
                k_recip_exp = np.append(k_recip_exp, c_recip)
        k_recip_exp = np.unique(k_recip_exp)
        w = np.exp(-orig_dist[i, k_recip_exp])
        V[i, k_recip_exp] = w / np.maximum(w.sum(), 1e-6)
    if k2 > 1:
        V = np.array([V[initial_rank[i, :k2]].mean(axis=0) for i in range(N)], dtype=np.float32)
    Vq, Vg = V[:nq], V[nq:]
    jaccard = np.ones((nq, ng), dtype=np.float32)
    for i in range(nq):
        nz = np.nonzero(Vq[i])[0]
        if len(nz):
            mn = np.minimum(Vq[i, nz][None, :], Vg[:, nz]).sum(axis=1)
            jaccard[i] = 1.0 - mn / (2.0 - mn)
    return 1.0 - ((1.0 - lam) * jaccard + lam * orig_dist[:nq, nq:])


def _fold_v25(fold, ctx, idx, with_diff=False):
    """Cached per-fold Full-2152 Sw whitening, Full Fisher LDA(512), cl2 centering, krec_sw10, diff_sw, vp_compat."""
    key = (fold.species, len(fold.gallery_rows), len(fold.query_rows),
           int(fold.gallery_rows[0]), int(fold.query_rows[0]))
    b = ctx._v25_cache.get(key)
    if b is None:
        rcfg = ctx.cfg["retrieval"]
        qr, gr = fold.query_rows, fold.gallery_rows
        glab = fold.gallery_labels
        projs = fit_sw_and_fullda(ctx.emb["l2"][gr], glab,
                                  reg=float(rcfg.get("sw_reg", 0.5)),
                                  d_lda=int(rcfg.get("fl_dim", 512)),
                                  gside=ctx.side[gr])
        hub_w = float(rcfg.get("sw_hub", 0.5))
        Z_p, S_raw, M_v = {}, {}, {}
        for k in ("cl2", "sw", "ssw", "fl512"):
            zq, zg = projs[k](ctx.emb["l2"][qr]), projs[k](ctx.emb["l2"][gr])
            Z_p[k] = (zq, zg)
            sr = (zq @ zg.T).astype(np.float32)
            S_raw[k] = sr
            if k == "cl2":
                M_v["cl2"] = idx.max(sr)
            else:
                rg = rg_diff(zg, glab, K=10)
                M_v[f"{k}2152" if k in ("sw", "ssw") else k] = idx.max(sr - hub_w * rg[None, :])
                M_v[f"{k}_raw"] = idx.max(sr)
        Kr_sw = k_reciprocal_rerank(Z_p["sw"][0], Z_p["sw"][1], k1=10, k2=4, lam=0.3)
        Mk_sw = idx.max(Kr_sw)
        compat_sw = vp_compat_stats(Z_p["sw"][1], glab, ctx.vp[gr])
        b = {"Z": Z_p, "S_sw_raw": S_raw["sw"], "M": M_v, "Mk_sw": Mk_sw,
             "compat_sw": compat_sw, "Md_sw": None}
        ctx._v25_cache[key] = b
    if with_diff and b["Md_sw"] is None:
        Dr_sw = manifold_diffusion(b["Z"]["sw"][0], b["Z"]["sw"][1], k=20, alpha=0.85, gamma=3.0, iters=25)
        b["Md_sw"] = idx.max(Dr_sw)
    return b


def sift_pairs_for_fold(fold, ctx, n_cand_sift=None, per_id=None):
    """Pairs (query_row, gallery_row) needed for SIFT features, chosen from fold-legal info only."""
    cfg = ctx.cfg["sift"]; rcfg = ctx.cfg["retrieval"]
    n_cand_sift = n_cand_sift or cfg["top_ids"]; per_id = per_id or cfg["imgs_per_id"]
    Zq, Zg = ctx.emb["l2"][fold.query_rows], ctx.emb["l2"][fold.gallery_rows]
    S = Zq @ Zg.T
    idx = IdentityIndex(fold.gallery_labels)
    M = idx.max(S)
    M2 = (ctx.emb["pcaw"][fold.query_rows] @ ctx.emb["pcaw"][fold.gallery_rows].T)
    M2 = idx.max(M2)
    lnb_k = int(rcfg.get("lnbnn_cand", 0))
    lnb_uk = int(rcfg.get("lnbnn_ucand", 0))
    v25 = bool(rcfg.get("v25_stage1", False))
    Lim, Lu, ImgMat, UimgMat = (None, None, None, None)
    if (lnb_k > 0 or lnb_uk > 0) and rcfg.get("lnbnn", False):
        from .lnbnn import fold_uimg_scores
        Lu, Lim, ImgMat, UimgMat = fold_uimg_scores(fold, ctx, idx.ids)
    bv = _fold_v25(fold, ctx, idx, with_diff=False) if v25 else None
    sw_k = int(rcfg.get("sw_cand", 15)) if v25 else 0
    fl_k = int(rcfg.get("fl_cand", 15)) if v25 else 0
    kr_k = int(rcfg.get("krec_sw_cand", 10)) if v25 else 0
    per_id_sw = int(rcfg.get("sift_imgs_sw", 2)) if v25 else 0
    k_im = int(rcfg.get("lnbnn_sift_imgs", 2))
    k_um = int(rcfg.get("lnbnn_uimg_sift_imgs", 1)) if v25 else 0
    pairs = []
    for i in range(len(Zq)):
        r1 = np.argsort(-M[i])[:n_cand_sift]; r2 = np.argsort(-M2[i])[:n_cand_sift]
        cset = np.union1d(r1, r2)
        if Lim is not None and lnb_k > 0:
            cset = np.union1d(cset, np.argsort(-Lim[i])[:lnb_k])
        if Lu is not None and lnb_uk > 0:
            cset = np.union1d(cset, np.argsort(-Lu[i])[:lnb_uk])
        if bv is not None:
            if sw_k > 0:
                cset = np.union1d(cset, np.argsort(-bv["M"]["sw2152"][i])[:sw_k])
            if fl_k > 0:
                cset = np.union1d(cset, np.argsort(-bv["M"]["fl512"][i])[:fl_k])
            if kr_k > 0:
                cset = np.union1d(cset, np.argsort(-bv["Mk_sw"][i])[:kr_k])
        for c in cset:
            cols = idx.order[idx.starts[c]: idx.starts[c] + idx.counts[c]]
            best = cols[np.argsort(-S[i, cols])[:per_id]]
            if bv is not None and per_id_sw > 0:
                best = np.union1d(best, cols[np.argsort(-bv["S_sw_raw"][i, cols])[:per_id_sw]])
            if ImgMat is not None and k_im > 0:
                best = np.union1d(best, cols[np.argsort(-ImgMat[i, cols])[:k_im]])
            if UimgMat is not None and k_um > 0:
                best = np.union1d(best, cols[np.argsort(-UimgMat[i, cols])[:k_um]])
            pairs += [(fold.query_rows[i], fold.gallery_rows[j]) for j in best]
    return pairs


def query_neighbour_pairs(fold, ctx, k):
    """Query-query pairs (top-k by l2 cosine) for the transductive graph."""
    Z = ctx.emb["l2"][fold.query_rows]
    S = Z @ Z.T; np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    return [(fold.query_rows[i], fold.query_rows[j]) for i in range(len(Z)) for j in nn[i]]


def build_table(fold, ctx, n_cand=20, transductive=False, lda_cfg=None):
    lda_cfg = lda_cfg or {"gamma": 0.3, "d_out": 256}
    gr, qr = fold.gallery_rows, fold.query_rows
    Z = {"l2": (ctx.emb["l2"][qr], ctx.emb["l2"][gr]), "pcaw": (ctx.emb["pcaw"][qr], ctx.emb["pcaw"][gr])}
    f = fit_lda(ctx.emb["l2"][gr], fold.gallery_labels, gamma=lda_cfg["gamma"], d_out=lda_cfg["d_out"])
    Z["lda"] = (f(ctx.emb["l2"][qr]), f(ctx.emb["l2"][gr]))
    EMBS = ("l2", "pcaw", "lda")
    if ctx.cfg["retrieval"].get("xlda", False):
        fx = fit_lda(ctx.emb["l2"][gr], fold.gallery_labels, gamma=lda_cfg["gamma"], d_out=lda_cfg["d_out"],
                     sides=ctx.side[gr], w_cross=1.0)
        Z["xlda"] = (fx(ctx.emb["l2"][qr]), fx(ctx.emb["l2"][gr]))
        EMBS = EMBS + ("xlda",)
    idx = IdentityIndex(fold.gallery_labels)
    nq, nid = len(qr), len(idx.ids)
    S, M, T2, PR = {}, {}, {}, {}
    for e in EMBS:
        zq, zg = Z[e]
        S[e] = zq @ zg.T
        M[e] = idx.max(S[e])
        T2[e] = idx.sorted_desc_k(S[e], 2)
        PR[e] = zq @ idx.prototypes(zg).T
    # ---- second-order "profile" similarity + within-identity spread (gallery labels only)
    PROF, INTRA = {}, {}
    if ctx.cfg["retrieval"].get("profile", False):
        for e in ("l2", "lda"):
            zq, zg = Z[e]
            Sgg = zg @ zg.T
            np.fill_diagonal(Sgg, -np.inf)
            Mgg = idx.max(Sgg)                                   # gallery image x identity
            Mgg[~np.isfinite(Mgg)] = np.nan
            Pc = np.zeros((nid, nid)); np.add.at(Pc, idx.inv, np.nan_to_num(Mgg, nan=0.0)); Pc /= idx.counts[:, None]
            X = M[e]; Y = Pc; n1 = nid - 1
            sx = X.sum(1)[:, None] - X; sy = (Y.sum(1) - np.diag(Y))[None, :]
            sxy = X @ Y.T - X * np.diag(Y)[None, :]
            sxx = (X ** 2).sum(1)[:, None] - X ** 2; syy = ((Y ** 2).sum(1) - np.diag(Y) ** 2)[None, :]
            num = n1 * sxy - sx * sy
            den = np.sqrt(np.maximum(n1 * sxx - sx ** 2, 1e-12) * np.maximum(n1 * syy - sy ** 2, 1e-12))
            PROF[e] = num / den
            # within-identity: mean over c's images of their best same-identity cosine (NaN for 1-image ids)
            same = idx.inv[:, None] == idx.inv[None, :]
            best_same = np.where(same, Sgg, -np.inf).max(1)
            best_same[~np.isfinite(best_same)] = np.nan
            INTRA[e] = pd.Series(best_same).groupby(idx.inv).mean().reindex(range(nid)).values
    # candidate set: union of top n_cand//2+5 under each embedding, capped by combined rank
    rank = {e: np.argsort(np.argsort(-M[e], axis=1), axis=1) for e in EMBS}
    comb = sum(rank[e] for e in EMBS)
    # gallery stats
    gvp, gside = ctx.vp[gr], ctx.side[gr]
    id_of_col = idx.inv
    n_img = idx.counts
    n_vp = pd.Series(gvp).groupby(id_of_col).nunique().reindex(range(nid)).values
    has_L = pd.Series(gside == "L").groupby(id_of_col).any().reindex(range(nid)).values
    has_R = pd.Series(gside == "R").groupby(id_of_col).any().reindex(range(nid)).values
    compat = vp_compat(Z["l2"][1], fold.gallery_labels, gvp)
    # reciprocity: for each gallery image, similarities to pool (gallery U queries) sorted
    pool = np.vstack([Z["l2"][1], Z["l2"][0]]).astype(np.float32)
    # sorted similarities of every gallery image to the pool (ascending), for reciprocal-rank lookups
    GS = np.sort(Z["l2"][1].astype(np.float32) @ pool.T, axis=1)
    npool = GS.shape[1]
    # best l2 gallery image of each (q, c)
    So = S["l2"][:, idx.order]
    # query-level stats
    qf = {}
    for e in EMBS:
        srt = -np.sort(-M[e], axis=1)
        qf[f"q_top1_{e}"] = srt[:, 0]; qf[f"q_gap12_{e}"] = srt[:, 0] - srt[:, 1]
        qf[f"q_mean10_{e}"] = srt[:, :10].mean(1); qf[f"q_std10_{e}"] = srt[:, :10].std(1)
    qf["q_lognorm"] = ctx.lognorm[qr]; qf["q_kp"] = ctx.kp[qr]
    qside = ctx.side[qr]; qvp = ctx.vp[qr]
    qf["q_side"] = pd.Categorical(qside, categories=["L", "R", "N", "U"]).codes
    # transductive: query-query similarities
    if transductive:
        Sqq = Z["l2"][0] @ Z["l2"][0].T; np.fill_diagonal(Sqq, -np.inf)
        qf["q_nnq_cos"] = Sqq.max(1)
        qf["q_nnq_minus_top1"] = Sqq.max(1) - qf["q_top1_l2"]
        nnq = np.argsort(-Sqq, axis=1)[:, :5]
        top1_id = M["l2"].argmax(1)
        # the neighbour's own gallery evidence
        qf["q_nnq_top1"] = qf["q_top1_l2"][nnq[:, 0]]
    rows = []
    sift_on = ctx.pc is not None
    sift_top = ctx.cfg["sift"]["top_ids"]; per_id = ctx.cfg["sift"]["imgs_per_id"]
    ii = SF.index
    rcfg = ctx.cfg["retrieval"]
    v25 = bool(rcfg.get("v25_stage1", False))
    lnb_k = int(rcfg.get("lnbnn_cand", 0))
    lnb_uk = int(rcfg.get("lnbnn_ucand", 0)) if v25 else 0
    Lim, Lu, ImgMat, UimgMat = (None, None, None, None)
    if (lnb_k > 0 or lnb_uk > 0) and rcfg.get("lnbnn", False):
        from .lnbnn import fold_uimg_scores
        Lu, Lim, ImgMat, UimgMat = fold_uimg_scores(fold, ctx, idx.ids)
    bv = _fold_v25(fold, ctx, idx, with_diff=True) if v25 else None
    compat_l2 = vp_compat_stats(Z["l2"][1], fold.gallery_labels, gvp) if v25 else None
    V25_EMBS = ("sw2152", "ssw2152", "sw_raw", "ssw_raw", "fl512", "fl512_raw", "cl2")
    V25_R, V25_Q = {}, {}
    if bv is not None:
        Mv = bv["M"]
        for e in V25_EMBS:
            V25_R[e] = np.argsort(np.argsort(-Mv[e], axis=1), axis=1)
            srt = -np.sort(-Mv[e], axis=1)
            V25_Q[f"q_{e}_max"] = srt[:, 0]
            V25_Q[f"q_{e}_gap"] = srt[:, 0] - srt[:, 1]
            V25_Q[f"q_{e}_mean10"] = srt[:, :10].mean(1)
            V25_Q[f"q_{e}_std10"] = srt[:, :10].std(1)
        for nm, mat in (("krec_sw10", bv["Mk_sw"]), ("diff_sw", bv["Md_sw"])):
            srt = -np.sort(-mat, axis=1)
            V25_Q[f"q_{nm}_max"] = srt[:, 0]
            V25_Q[f"q_{nm}_gap"] = srt[:, 0] - srt[:, 1]
            V25_Q[f"q_{nm}_mean10"] = srt[:, :10].mean(1)
            V25_Q[f"q_{nm}_std10"] = srt[:, :10].std(1)
    sw_k = int(rcfg.get("sw_cand", 15)) if v25 else 0
    fl_k = int(rcfg.get("fl_cand", 15)) if v25 else 0
    kr_k = int(rcfg.get("krec_sw_cand", 10)) if v25 else 0
    per_id_sw = int(rcfg.get("sift_imgs_sw", 2)) if v25 else 0
    k_im = int(rcfg.get("lnbnn_sift_imgs", 2))
    k_um = int(rcfg.get("lnbnn_uimg_sift_imgs", 1)) if v25 else 0
    qf.update({k: v for k, v in V25_Q.items() if k.endswith(("_max", "_gap"))})
    KREC = {}
    if ctx.cfg["retrieval"].get("krec", False):
        for e in ("l2", "lda"):
            Kr = k_reciprocal_rerank(Z[e][0], Z[e][1], k1=20, k2=6, lam=0.3)
            Mk = idx.max(Kr)
            srtk = -np.sort(-Mk, axis=1)
            KREC[e] = (Mk, srtk[:, 0], srtk[:, 0] - srtk[:, 1])
            qf[f"q_krec_{e}_max"] = srtk[:, 0]
            qf[f"q_krec_{e}_gap"] = srtk[:, 0] - srtk[:, 1]
    DIFF, CLS = {}, {}
    if ctx.cfg["retrieval"].get("diff_cls", False):
        for e in ("l2", "lda"):
            Dr = manifold_diffusion(Z[e][0], Z[e][1], k=20, alpha=0.85, gamma=3.0)
            Md = idx.max(Dr)
            srtd = -np.sort(-Md, axis=1)
            DIFF[e] = (Md, srtd[:, 0])
            qf[f"q_diff_{e}_max"] = srtd[:, 0]
            qf[f"q_diff_{e}_gap"] = srtd[:, 0] - srtd[:, 1]
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        zq_l2, zg_l2 = Z["l2"]
        mu_g = zg_l2.mean(0)
        cg_c = l2n(zg_l2 - mu_g); cq_c = l2n(zq_l2 - mu_g)
        lr = LogisticRegression(C=30.0, max_iter=500, solver="lbfgs").fit(cg_c, idx.inv)
        Plr = lr.predict_log_proba(cq_c).astype(np.float32)
        slr = -np.sort(-Plr, axis=1)
        CLS["lr"] = (Plr, slr[:, 0])
        qf["q_lr_max"] = slr[:, 0]; qf["q_lr_gap"] = slr[:, 0] - slr[:, 1]
        mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=150,
                            early_stopping=True, validation_fraction=0.1, random_state=0).fit(cg_c, idx.inv)
        Pmlp = np.log(np.clip(mlp.predict_proba(cq_c), 1e-7, 1.0)).astype(np.float32)
        smlp = -np.sort(-Pmlp, axis=1)
        CLS["mlp"] = (Pmlp, smlp[:, 0])
        qf["q_mlp_max"] = smlp[:, 0]; qf["q_mlp_gap"] = smlp[:, 0] - smlp[:, 1]
    for i in range(nq):
        cands = list(np.argsort(comb[i])[:n_cand])
        cset_now = set(cands)
        def _add_top(scores, k):
            if scores is not None and k > 0:
                for c_extra in np.argsort(-scores)[:k]:
                    ci = int(c_extra)
                    if ci not in cset_now:
                        cands.append(ci); cset_now.add(ci)
        _add_top(Lim[i] if Lim is not None else None, lnb_k)
        _add_top(Lu[i] if Lu is not None else None, lnb_uk)
        if bv is not None:
            _add_top(bv["M"]["sw2152"][i], sw_k)
            _add_top(bv["M"]["fl512"][i], fl_k)
            _add_top(bv["Mk_sw"][i], kr_k)
        cands = np.array(cands, dtype=int)
        r1 = set(np.argsort(-M["l2"][i])[:sift_top]) | set(np.argsort(-M["pcaw"][i])[:sift_top])
        if Lim is not None and lnb_k > 0:
            r1 |= set(int(x) for x in np.argsort(-Lim[i])[:lnb_k])
        if Lu is not None and lnb_uk > 0:
            r1 |= set(int(x) for x in np.argsort(-Lu[i])[:lnb_uk])
        if bv is not None:
            if sw_k > 0:
                r1 |= set(int(x) for x in np.argsort(-bv["M"]["sw2152"][i])[:sw_k])
            if fl_k > 0:
                r1 |= set(int(x) for x in np.argsort(-bv["M"]["fl512"][i])[:fl_k])
            if kr_k > 0:
                r1 |= set(int(x) for x in np.argsort(-bv["Mk_sw"][i])[:kr_k])
        for c in cands:
            row = {"qi": i, "cand": c}
            for e in EMBS:
                row[f"max_{e}"] = M[e][i, c]; row[f"top2_{e}"] = T2[e][i, c]; row[f"proto_{e}"] = PR[e][i, c]
                row[f"rank_{e}"] = rank[e][i, c]
                row[f"gap_{e}"] = M[e][i, c] - qf[f"q_top1_{e}"][i]
                row[f"z_{e}"] = (M[e][i, c] - qf[f"q_mean10_{e}"][i]) / (qf[f"q_std10_{e}"][i] + 1e-6)
            for e, (Mk, m1, _) in KREC.items():
                row[f"krec_{e}"] = Mk[i, c]
                row[f"krec_{e}_gap"] = Mk[i, c] - m1[i]
            for e, (Md, m1) in DIFF.items():
                row[f"diff_{e}"] = Md[i, c]
                row[f"diff_{e}_gap"] = Md[i, c] - m1[i]
            for e, (Pc, m1) in CLS.items():
                row[f"cls_{e}"] = Pc[i, c]
                row[f"cls_{e}_gap"] = Pc[i, c] - m1[i]
            row["comb_rank"] = int(np.where(cands == c)[0][0])
            if bv is not None:
                Mv = bv["M"]
                for e in V25_EMBS:
                    val = float(Mv[e][i, c])
                    row[f"max_{e}"] = val
                    row[f"rank_{e}"] = float(V25_R[e][i, c])
                    row[f"gap_{e}"] = val - float(V25_Q[f"q_{e}_max"][i])
                    row[f"z_{e}"] = (val - float(V25_Q[f"q_{e}_mean10"][i])) / (float(V25_Q[f"q_{e}_std10"][i]) + 1e-6)
                for nm, mat in (("krec_sw10", bv["Mk_sw"]), ("diff_sw", bv["Md_sw"])):
                    val = float(mat[i, c])
                    row[f"max_{nm}"] = val
                    row[f"gap_{nm}"] = val - float(V25_Q[f"q_{nm}_max"][i])
                    row[f"z_{nm}"] = (val - float(V25_Q[f"q_{nm}_mean10"][i])) / (float(V25_Q[f"q_{nm}_std10"][i]) + 1e-6)
            for e in PROF:
                row[f"prof_{e}"] = PROF[e][i, c]
                row[f"intra_{e}"] = INTRA[e][c]
                row[f"max_minus_intra_{e}"] = M[e][i, c] - INTRA[e][c]
            row["log_nimg"] = np.log(n_img[c]); row["n_vp"] = n_vp[c]
            a, b = idx.starts[c], idx.counts[c]
            cols = idx.order[a:a + b]
            sims = So[i, a:a + b]
            jbest = cols[sims.argmax()]
            sd = qside[i]
            row["has_same_side"] = float(has_L[c] if sd == "L" else has_R[c] if sd == "R" else np.nan)
            same = gside[cols] == sd
            row["max_same_side"] = sims[same].max() if (sd in "LR" and same.any()) else np.nan
            opp = gside[cols] == ("R" if sd == "L" else "L")
            row["max_opp_side"] = sims[opp].max() if (sd in "LR" and opp.any()) else np.nan
            k = (qvp[i], gvp[jbest]) if qvp[i] <= gvp[jbest] else (gvp[jbest], qvp[i])
            row["compat_best"] = compat.get(k, np.nan)
            row["max_minus_compat"] = M["l2"][i, c] - row["compat_best"]
            if bv is not None:
                c_l2 = compat_l2.get(k)
                row["z_compat_l2"] = (M["l2"][i, c] - c_l2[0]) / c_l2[1] if c_l2 is not None else np.nan
                sims_sw = bv["S_sw_raw"][i, cols]
                jbest_sw = cols[sims_sw.argmax()]
                row["max_same_side_sw"] = sims_sw[same].max() if (sd in "LR" and same.any()) else np.nan
                row["max_opp_side_sw"] = sims_sw[opp].max() if (sd in "LR" and opp.any()) else np.nan
                k_sw = (qvp[i], gvp[jbest_sw]) if qvp[i] <= gvp[jbest_sw] else (gvp[jbest_sw], qvp[i])
                c_sw = bv["compat_sw"].get(k_sw)
                row["compat_best_sw"] = c_sw[0] if c_sw is not None else np.nan
                row["max_minus_compat_sw"] = bv["M"]["sw_raw"][i, c] - row["compat_best_sw"]
                row["z_compat_sw"] = (bv["M"]["sw_raw"][i, c] - c_sw[0]) / c_sw[1] if c_sw is not None else np.nan
            # reciprocity of best image: how many pool items are closer to g* than q is
            n_closer = npool - np.searchsorted(GS[jbest], np.float32(M["l2"][i, c]), side="right")
            row["recip_rank"] = np.log1p(max(n_closer - 1, 0))
            if sift_on:
                if c in r1:
                    best = cols[np.argsort(-S["l2"][i, cols])[:per_id]]
                    if bv is not None and per_id_sw > 0:
                        best = np.union1d(best, cols[np.argsort(-bv["S_sw_raw"][i, cols])[:per_id_sw]])
                    if ImgMat is not None and k_im > 0:
                        best = np.union1d(best, cols[np.argsort(-ImgMat[i, cols])[:k_im]])
                    if UimgMat is not None and k_um > 0:
                        best = np.union1d(best, cols[np.argsort(-UimgMat[i, cols])[:k_um]])
                    F = np.array([ctx.pc.get(qr[i], gr[j]) for j in best], dtype=np.float32)
                    inl = np.nan_to_num(F[:, ii("inl_aff")], nan=-1)
                    k2 = np.argsort(-inl)
                    row["sift_checked"] = len(best)
                    row["sift_inl_max"] = np.nanmax(F[:, ii("inl_aff")]) if np.isfinite(F[:, ii("inl_aff")]).any() else np.nan
                    row["sift_inl_strict_max"] = np.nanmax(F[:, ii("inl_aff_strict")]) if np.isfinite(F[:, ii("inl_aff_strict")]).any() else np.nan
                    row["sift_inlh_max"] = np.nanmax(F[:, ii("inl_h")]) if np.isfinite(F[:, ii("inl_h")]).any() else np.nan
                    row["sift_inl_top2"] = np.nansum(np.sort(np.nan_to_num(F[:, ii("inl_aff")]))[::-1][:2])
                    row["sift_n6"] = float((np.nan_to_num(F[:, ii("inl_aff")]) >= 6).sum())
                    row["sift_mutual_max"] = np.nanmax(F[:, ii("n_mutual")]) if np.isfinite(F[:, ii("n_mutual")]).any() else np.nan
                    fb = F[k2[0]]
                    row["sift_ratio_best"] = fb[ii("inl_ratio")]; row["sift_overkp_best"] = fb[ii("inl_over_kp")]
                    row["sift_sim_best"] = fb[ii("inl_sim")]; row["sift_spread_best"] = fb[ii("spread_min")]
                    row["sift_rot_best"] = fb[ii("abs_rot")]; row["sift_scale_best"] = fb[ii("log_scale")]
                    row["sift_kpmin_best"] = fb[ii("kp_min")]
                    if ctx.cfg["retrieval"].get("bg_clean", False):
                        rot_v = np.nan_to_num(F[:, ii("abs_rot")], nan=99.0)
                        spr_v = np.nan_to_num(F[:, ii("spread_min")], nan=1.0)
                        is_bg = (rot_v < 0.50) & (spr_v < 0.25)
                        inl_c = np.where(is_bg, 0.0, np.nan_to_num(F[:, ii("inl_aff")], nan=-1.0))
                        row["sift_inl_clean"] = float(inl_c.max()) if (inl_c >= 0).any() else np.nan
                        row["sift_n6_clean"] = float((inl_c >= 6).sum())
                        row["sift_best_is_bg"] = float(is_bg[k2[0]])
                    if bv is not None and UimgMat is not None:
                        rot_v = np.nan_to_num(F[:, ii("abs_rot")], nan=99.0)
                        spr_v = np.nan_to_num(F[:, ii("spread_min")], nan=1.0)
                        inl_v = np.nan_to_num(F[:, ii("inl_aff")], nan=-1.0)
                        im_g = ImgMat[i, best]; uim_g = UimgMat[i, best]
                        ush_g = uim_g / np.maximum(im_g, 1e-6)
                        row["sift_best_ushare"] = float(ush_g[k2[0]])
                        opp_g = gside[best] == ("R" if sd == "L" else "L")
                        row["sift_opp_side_inl"] = float(inl_v[opp_g].max()) if (sd in "LR" and opp_g.any() and (inl_v[opp_g] >= 0).any()) else np.nan
                        row["sift_n_vps_hi"] = float(len(set(gvp[best][inl_v >= 10])))
                        if rcfg.get("bg_clean_v2_img", False):
                            is_bg2_g = ((rot_v < 0.50) & (spr_v < 0.25)) | ((rot_v < 0.50) & (ush_g < 0.35) & (inl_v >= 6))
                            inl_c2 = np.where(is_bg2_g, 0.0, inl_v)
                            row["sift_inl_clean2"] = float(inl_c2.max()) if (inl_c2 >= 0).any() else np.nan
                            row["sift_n6_clean2"] = float((inl_c2 >= 6).sum())
                            row["sift_is_bg2"] = float(is_bg2_g[k2[0]])
                else:
                    row["sift_checked"] = 0
            rows.append(row)
    df = pd.DataFrame(rows)
    for k, v in qf.items():
        df[k] = np.asarray(v)[df.qi.values]
    if sift_on:
        g = df.groupby("qi")
        df["q_sift_inl_max"] = g.sift_inl_max.transform("max")
        df["q_sift_n6_cands"] = g.sift_n6.transform(lambda s: (s > 0).sum())
        df["sift_inl_minus_qmax"] = df.sift_inl_max - df.q_sift_inl_max
        # second-best identity's inliers (for the P(new) model / margin features)
        df["sift_inl_rank"] = g.sift_inl_max.rank(ascending=False, method="min")
        if ctx.cfg["retrieval"].get("bg_clean", False):
            df["q_sift_inl_clean_max"] = g.sift_inl_clean.transform("max")
            df["sift_inl_clean_minus_qmax"] = df.sift_inl_clean - df.q_sift_inl_clean_max
        if "sift_inl_clean2" in df.columns:
            df["q_sift_inl_clean2_max"] = g.sift_inl_clean2.transform("max")
            df["sift_inl_clean2_minus_qmax"] = df.sift_inl_clean2 - df.q_sift_inl_clean2_max
    if transductive:
        # neighbour votes: share of q's top-5 query neighbours (cos>0.5) whose top-1 identity (pcaw) is c,
        # and best (neighbour cos x neighbour's score for c)
        top1p = M["pcaw"].argmax(1)
        vote, nbbest = [], []
        for i, c in zip(df.qi.values, df.cand.values):
            nb = nnq[i]; w = Sqq[i, nb]
            ok = w > 0.5
            vote.append(float(((top1p[nb] == c) & ok).sum()))
            nbbest.append(float(np.max(np.minimum(w, M["l2"][nb, c]))) if len(nb) else np.nan)
        df["tr_vote"] = vote; df["tr_nb_score"] = nbbest
        df["tr_nb_minus_max"] = df.tr_nb_score - df.max_l2
    if ctx.cfg["retrieval"].get("lnbnn", False):
        from .lnbnn import add_features
        df = add_features(df, fold, ctx, idx.ids)
    df["query_id"] = fold.query_ids[df.qi.values]
    df["cand_id"] = idx.ids[df.cand.values]
    df["species_is_leo"] = int(fold.species == "leopard")
    return df
