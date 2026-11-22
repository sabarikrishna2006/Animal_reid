"""HotSpotter-style one-vs-many keypoint matching (LNBNN) on the provided SIFT descriptors (classical NN matching,
faiss-cpu). Neighbour lists (k=50 per query descriptor, over all training descriptors of the species) are built once
by scripts/04_build_lnbnn.py (or Dev's notebook, same format) and read from `paths.lnbnn` (default data/cache/lnbnn). Images outside the fold's gallery
and the query itself are masked at scoring time, so CV stays honest."""
import json
from pathlib import Path

import numpy as np

K = 10
_CACHE = {}


class Scorer:
    def __init__(self, d, sp, train_ids, train_labels):
        m = json.loads((d / f"{sp}_meta.json").read_text())
        self.gal = np.array(m["gallery"]); self.qpos = {q: i for i, q in enumerate(m["queries"])}
        self.qoff = np.array(m["qoff"])
        self.NB = np.load(d / f"{sp}_nb.npy", mmap_mode="r"); self.DI = np.load(d / f"{sp}_d.npy", mmap_mode="r")
        lab = dict(zip(train_ids, train_labels))
        self.names, self.lab = np.unique([lab[g] for g in self.gal], return_inverse=True)
        self.gpos = {g: i for i, g in enumerate(self.gal)}

    def scores(self, qid, gmask):
        i = self.qpos[qid]; a, b = self.qoff[i], self.qoff[i + 1]
        out = np.zeros(len(self.names), np.float32)
        if b == a:
            return out
        nb = np.asarray(self.NB[a:b]); d = np.asarray(self.DI[a:b], np.float32)
        ok = (nb >= 0) & gmask[np.maximum(nb, 0)]
        if qid in self.gpos:
            ok &= nb != self.gpos[qid]
        rank = np.cumsum(ok, 1); has_bg = rank[:, -1] >= K + 1
        bg = np.where(has_bg, d[np.arange(len(d)), np.argmax(rank >= K + 1, 1)], d.max(1))
        lab = self.lab[np.maximum(nb, 0)]; r, c = np.nonzero(ok & (rank <= K))
        _, first = np.unique(r.astype(np.int64) * len(self.names) + lab[r, c], return_index=True)
        r, c = r[first], c[first]
        np.add.at(out, lab[r, c], np.maximum(bg[r] - d[r, c], 0))
        return out

    def scores_img_k3(self, qid, gmask, want_img_sc=False):
        """Image-level LNBNN (max over gallery images per identity) and strict K=3 identity LNBNN."""
        i = self.qpos[qid]; a, b = self.qoff[i], self.qoff[i + 1]
        n_id, n_gal = len(self.names), len(self.gal)
        out_img = np.zeros(n_id, np.float32)
        out_k3 = np.zeros(n_id, np.float32)
        img_sc = np.zeros(n_gal, np.float32)
        if b == a:
            return (out_img, out_k3, img_sc) if want_img_sc else (out_img, out_k3)
        nb = np.asarray(self.NB[a:b]); d = np.asarray(self.DI[a:b], np.float32)
        ok = (nb >= 0) & gmask[np.maximum(nb, 0)]
        if qid in self.gpos:
            ok &= nb != self.gpos[qid]
        rank = np.cumsum(ok, 1)
        has_bg10 = rank[:, -1] >= K + 1
        bg10 = np.where(has_bg10, d[np.arange(len(d)), np.argmax(rank >= K + 1, 1)], d.max(1))
        r, c = np.nonzero(ok & (rank <= K))
        if len(r):
            g_idx = nb[r, c]
            _, first = np.unique(r.astype(np.int64) * n_gal + g_idx, return_index=True)
            r1, c1, g1 = r[first], c[first], g_idx[first]
            np.add.at(img_sc, g1, np.maximum(bg10[r1] - d[r1, c1], 0.0))
            np.maximum.at(out_img, self.lab, img_sc)
        has_bg3 = rank[:, -1] >= 4
        bg3 = np.where(has_bg3, d[np.arange(len(d)), np.argmax(rank >= 4, 1)], d.max(1))
        r, c = np.nonzero(ok & (rank <= 3))
        if len(r):
            lab = self.lab[nb[r, c]]
            _, first = np.unique(r.astype(np.int64) * n_id + lab, return_index=True)
            r1, c1, l1 = r[first], c[first], lab[first]
            np.add.at(out_k3, l1, np.maximum(bg3[r1] - d[r1, c1], 0.0))
        return (out_img, out_k3, img_sc) if want_img_sc else (out_img, out_k3)


def scorer(ctx, sp):
    if sp not in _CACHE:
        from .io import P
        d = Path(ctx.cfg["paths"].get("lnbnn", "data/cache/lnbnn"))
        d = d if d.is_absolute() else P(ctx.cfg, "cache").parent.parent / d
        tr = ctx.data.train
        _CACHE[sp] = Scorer(d, sp, tr.id.values, tr.identity.values)
    return _CACHE[sp]


_FOLD_CACHE = {}


def fold_img_scores(fold, ctx, cand_ids):
    """Cached per-fold (Lim [nq x nid], ImgMat [nq x ngal_fold]) for lnb_img candidate expansion & SIFT selection."""
    key = (fold.species, len(fold.query_rows), len(fold.gallery_rows),
           int(fold.query_rows[0]), int(fold.gallery_rows[0]))
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    S = scorer(ctx, fold.species)
    gids = ctx.data.meta.id.values[fold.gallery_rows]
    gmask = np.isin(S.gal, gids)
    col_idx = np.searchsorted(S.names, cand_ids)
    g_col = np.array([S.gpos[g] for g in gids], dtype=np.int32)
    res = [S.scores_img_k3(q, gmask, want_img_sc=True) for q in fold.query_ids]
    Lim = np.stack([x[0] for x in res])[:, col_idx]
    ImgMat = np.stack([x[2][g_col] for x in res])
    _FOLD_CACHE[key] = (Lim, ImgMat)
    return Lim, ImgMat


def add_features(df, fold, ctx, cand_ids):
    """Adds per-(query, candidate) LNBNN features and q_ query-level ones. cand_ids: identity names of df.cand index."""
    S = scorer(ctx, fold.species)
    gids = ctx.data.meta.id.values[fold.gallery_rows]
    gmask = np.isin(S.gal, gids)
    L = np.stack([S.scores(q, gmask) for q in fold.query_ids])[:, np.searchsorted(S.names, cand_ids)]
    mx = L.max(1); second = -np.sort(-L, 1)[:, 1] if L.shape[1] > 1 else mx
    rk = np.argsort(np.argsort(-L, 1), 1)
    qi, c = df.qi.values, df.cand.values
    v = L[qi, c]
    df["lnb"] = v; df["lnb_rank"] = rk[qi, c]; df["lnb_rel"] = np.where(mx[qi] > 0, v / np.maximum(mx[qi], 1e-9), 0.0)
    df["lnb_log"] = np.log1p(v)
    df["q_lnb_max"] = mx[qi]; df["q_lnb_gap"] = (mx - second)[qi]; df["q_lnb_log_max"] = np.log1p(mx)[qi]
    if ctx.cfg["retrieval"].get("lnbnn_img", False):
        col_idx = np.searchsorted(S.names, cand_ids)
        res = [S.scores_img_k3(q, gmask) for q in fold.query_ids]
        Lim = np.stack([x[0] for x in res])[:, col_idx]
        Lk3 = np.stack([x[1] for x in res])[:, col_idx]
        mx_i = Lim.max(1); sec_i = -np.sort(-Lim, 1)[:, 1] if Lim.shape[1] > 1 else mx_i
        rk_i = np.argsort(np.argsort(-Lim, 1), 1)
        vi = Lim[qi, c]
        df["lnb_img"] = vi; df["lnb_img_rank"] = rk_i[qi, c]
        df["lnb_img_rel"] = np.where(mx_i[qi] > 0, vi / np.maximum(mx_i[qi], 1e-9), 0.0)
        df["lnb_img_log"] = np.log1p(vi)
        df["lnb_img_share"] = np.where(v > 0, vi / np.maximum(v, 1e-6), 0.0)
        df["q_lnb_img_max"] = mx_i[qi]; df["q_lnb_img_gap"] = (mx_i - sec_i)[qi]
        df["q_lnb_img_log_max"] = np.log1p(mx_i)[qi]
        mx_k = Lk3.max(1); sec_k = -np.sort(-Lk3, 1)[:, 1] if Lk3.shape[1] > 1 else mx_k
        vk = Lk3[qi, c]
        df["lnb_k3"] = vk
        df["lnb_k3_rel"] = np.where(mx_k[qi] > 0, vk / np.maximum(mx_k[qi], 1e-9), 0.0)
        df["q_lnb_k3_max"] = mx_k[qi]; df["q_lnb_k3_gap"] = (mx_k - sec_k)[qi]
    return df
