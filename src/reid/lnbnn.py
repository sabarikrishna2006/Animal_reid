"""HotSpotter-style one-vs-many keypoint matching (LNBNN) on the provided SIFT descriptors (classical NN matching,
faiss-cpu). Neighbour lists (k=50 per query descriptor, over all training descriptors of the species) are built once
by Dev's lnbnn.py / notebook and read from `paths.lnbnn` (default data/cache/lnbnn). Images outside the fold's gallery
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

    def scores_uimg(self, qid, gmask):
        """Camera-trap background-suppressed image-level LNBNN: for each descriptor, score gallery images of the
        nearest identity (id1) by max(0, d_id2 - d) where d_id2 is the nearest DIFFERENT identity.
        Background keypoints shared by >=2 identities get margin ~0; unique coat spots get huge margin."""
        i = self.qpos[qid]; a, b = self.qoff[i], self.qoff[i + 1]
        n_id, n_gal = len(self.names), len(self.gal)
        out_uimg = np.zeros(n_id, np.float32)
        if b == a:
            return out_uimg
        nb = np.asarray(self.NB[a:b]); d = np.asarray(self.DI[a:b], np.float32)
        ok = (nb >= 0) & gmask[np.maximum(nb, 0)]
        if qid in self.gpos:
            ok &= nb != self.gpos[qid]
        has1 = ok.any(1)
        c1 = np.argmax(ok, 1)
        rows_all = np.arange(len(d))
        g1 = nb[rows_all, c1]
        id1 = np.where(has1, self.lab[np.maximum(g1, 0)], -1)
        lab_mat = self.lab[np.maximum(nb, 0)]
        ok_diff = ok & (lab_mat != id1[:, None])
        has2 = ok_diff.any(1)
        d2 = np.where(has2, d[rows_all, np.argmax(ok_diff, 1)], d.max(1))
        ok_same = ok & (lab_mat == id1[:, None]) & (d < d2[:, None])
        rs, cs = np.nonzero(ok_same)
        if len(rs):
            gs = nb[rs, cs]
            _, f_g = np.unique(rs.astype(np.int64) * n_gal + gs, return_index=True)
            rs1, cs1, gs1 = rs[f_g], cs[f_g], gs[f_g]
            img_u = np.zeros(n_gal, np.float32)
            np.add.at(img_u, gs1, np.maximum(d2[rs1] - d[rs1, cs1], 0.0))
            np.maximum.at(out_uimg, self.lab, img_u)
        return out_uimg

    def scores_rimg(self, qid, gmask, ratio=0.85):
        """Strict cross-identity Lowe ratio image-level LNBNN (d < ratio * d_id2)."""
        i = self.qpos[qid]; a, b = self.qoff[i], self.qoff[i + 1]
        n_id, n_gal = len(self.names), len(self.gal)
        out_rimg = np.zeros(n_id, np.float32)
        if b == a:
            return out_rimg
        nb = np.asarray(self.NB[a:b]); d = np.asarray(self.DI[a:b], np.float32)
        ok = (nb >= 0) & gmask[np.maximum(nb, 0)]
        if qid in self.gpos:
            ok &= nb != self.gpos[qid]
        has1 = ok.any(1)
        c1 = np.argmax(ok, 1)
        rows_all = np.arange(len(d))
        g1 = nb[rows_all, c1]
        id1 = np.where(has1, self.lab[np.maximum(g1, 0)], -1)
        lab_mat = self.lab[np.maximum(nb, 0)]
        ok_diff = ok & (lab_mat != id1[:, None])
        has2 = ok_diff.any(1)
        c2 = np.argmax(ok_diff, 1)
        d2 = np.where(has2, d[rows_all, c2], d.max(1))
        ok_same = ok & (lab_mat == id1[:, None]) & (d < ratio * d2[:, None]) & has2[:, None]
        rs, cs = np.nonzero(ok_same)
        if len(rs):
            gs = nb[rs, cs]
            _, f_g = np.unique(rs.astype(np.int64) * n_gal + gs, return_index=True)
            rs1, cs1, gs1 = rs[f_g], cs[f_g], gs[f_g]
            img_r = np.zeros(n_gal, np.float32)
            np.add.at(img_r, gs1, np.maximum(d2[rs1] - d[rs1, cs1], 0.0))
            np.maximum.at(out_rimg, self.lab, img_r)
        return out_rimg

    def scores_all(self, qid, gmask, ratio=0.85):
        """Single-pass computation of (k10, img, k3, uimg, rimg, img_sc) for a query."""
        i = self.qpos[qid]; a, b = self.qoff[i], self.qoff[i + 1]
        n_id, n_gal = len(self.names), len(self.gal)
        out_k10 = np.zeros(n_id, np.float32)
        out_img = np.zeros(n_id, np.float32)
        out_k3 = np.zeros(n_id, np.float32)
        out_uimg = np.zeros(n_id, np.float32)
        out_rimg = np.zeros(n_id, np.float32)
        img_sc = np.zeros(n_gal, np.float32)
        img_u = np.zeros(n_gal, np.float32)
        if b == a:
            return out_k10, out_img, out_k3, out_uimg, out_rimg, img_sc, img_u
        nb = np.asarray(self.NB[a:b]); d = np.asarray(self.DI[a:b], np.float32)
        ok = (nb >= 0) & gmask[np.maximum(nb, 0)]
        if qid in self.gpos:
            ok &= nb != self.gpos[qid]
        rank = np.cumsum(ok, 1)
        rows_all = np.arange(len(d))
        lab_mat = self.lab[np.maximum(nb, 0)]

        # 1. Standard K=10 identity LNBNN + image-level K=10 LNBNN
        has_bg10 = rank[:, -1] >= K + 1
        bg10 = np.where(has_bg10, d[rows_all, np.argmax(rank >= K + 1, 1)], d.max(1))
        r10, c10 = np.nonzero(ok & (rank <= K))
        if len(r10):
            l10 = lab_mat[r10, c10]
            _, f_id = np.unique(r10.astype(np.int64) * n_id + l10, return_index=True)
            np.add.at(out_k10, l10[f_id], np.maximum(bg10[r10[f_id]] - d[r10[f_id], c10[f_id]], 0.0))
            g10 = nb[r10, c10]
            _, f_g = np.unique(r10.astype(np.int64) * n_gal + g10, return_index=True)
            np.add.at(img_sc, g10[f_g], np.maximum(bg10[r10[f_g]] - d[r10[f_g], c10[f_g]], 0.0))
            np.maximum.at(out_img, self.lab, img_sc)

        # 2. Strict K=3 identity LNBNN
        has_bg3 = rank[:, -1] >= 4
        bg3 = np.where(has_bg3, d[rows_all, np.argmax(rank >= 4, 1)], d.max(1))
        r3, c3 = np.nonzero(ok & (rank <= 3))
        if len(r3):
            l3 = lab_mat[r3, c3]
            _, f3 = np.unique(r3.astype(np.int64) * n_id + l3, return_index=True)
            np.add.at(out_k3, l3[f3], np.maximum(bg3[r3[f3]] - d[r3[f3], c3[f3]], 0.0))

        # 3. Background-suppressed image-level LNBNN (uimg) and cross-ID ratio LNBNN (rimg)
        has1 = rank[:, -1] >= 1
        c1 = np.argmax(ok, 1)
        g1 = nb[rows_all, c1]
        id1 = np.where(has1, self.lab[np.maximum(g1, 0)], -1)
        ok_diff = ok & (lab_mat != id1[:, None])
        has2 = ok_diff.any(1)
        d2 = np.where(has2, d[rows_all, np.argmax(ok_diff, 1)], d.max(1))
        ok_same = ok & (lab_mat == id1[:, None]) & (d < d2[:, None])
        rs, cs = np.nonzero(ok_same)
        if len(rs):
            gs = nb[rs, cs]
            _, fg = np.unique(rs.astype(np.int64) * n_gal + gs, return_index=True)
            rs1, cs1, gs1 = rs[fg], cs[fg], gs[fg]
            vm = np.maximum(d2[rs1] - d[rs1, cs1], 0.0)
            np.add.at(img_u, gs1, vm)
            np.maximum.at(out_uimg, self.lab, img_u)
            if has2.any():
                vr = np.where(has2[rs1] & (d[rs1, cs1] < ratio * d2[rs1]), vm, 0.0)
                img_r = np.zeros(n_gal, np.float32)
                np.add.at(img_r, gs1, vr)
                np.maximum.at(out_rimg, self.lab, img_r)
        return out_k10, out_img, out_k3, out_uimg, out_rimg, img_sc, img_u


def scorer(ctx, sp):
    from .io import P
    d = Path(ctx.cfg["paths"].get("lnbnn", "data/cache/lnbnn"))
    d = d if d.is_absolute() else P(ctx.cfg, "cache").parent.parent / d
    key = (str(d), sp)
    if key not in _CACHE:
        tr = ctx.data.train
        _CACHE[key] = Scorer(d, sp, tr.id.values, tr.identity.values)
    return _CACHE[key]


_FOLD_CACHE = {}
_FOLD_UIMG_CACHE = {}


def fold_all_scores(fold, ctx, cand_ids):
    """Single-pass cached per-fold LNBNN matrices: (L, Lim, Lk3, Lu, Lr, ImgMat)."""
    lnb_dir = ctx.cfg["paths"].get("lnbnn", "data/cache/lnbnn")
    key = (lnb_dir, fold.species, len(fold.query_rows), len(fold.gallery_rows),
           int(fold.query_rows[0]), int(fold.gallery_rows[0]))
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    S = scorer(ctx, fold.species)
    gids = ctx.data.meta.id.values[fold.gallery_rows]
    gmask = np.isin(S.gal, gids)
    col_idx = np.searchsorted(S.names, cand_ids)
    g_col = np.array([S.gpos[g] for g in gids], dtype=np.int32)
    res = [S.scores_all(q, gmask) for q in fold.query_ids]
    L = np.stack([x[0] for x in res])[:, col_idx]
    Lim = np.stack([x[1] for x in res])[:, col_idx]
    Lk3 = np.stack([x[2] for x in res])[:, col_idx]
    Lu = np.stack([x[3] for x in res])[:, col_idx]
    Lr = np.stack([x[4] for x in res])[:, col_idx]
    ImgMat = np.stack([x[5][g_col] for x in res])
    UimgMat = np.stack([x[6][g_col] for x in res])
    _FOLD_CACHE[key] = (L, Lim, Lk3, Lu, Lr, ImgMat)
    _FOLD_UIMG_CACHE[key] = UimgMat
    return _FOLD_CACHE[key]


def fold_uimg_scores(fold, ctx, cand_ids):
    """Cached per-fold (Lu, Lim, ImgMat, UimgMat) using single-pass LNBNN cache."""
    _, Lim, _, Lu, _, ImgMat = fold_all_scores(fold, ctx, cand_ids)
    lnb_dir = ctx.cfg["paths"].get("lnbnn", "data/cache/lnbnn")
    key = (lnb_dir, fold.species, len(fold.query_rows), len(fold.gallery_rows),
           int(fold.query_rows[0]), int(fold.gallery_rows[0]))
    return Lu, Lim, ImgMat, _FOLD_UIMG_CACHE[key]


def fold_img_scores(fold, ctx, cand_ids):
    """Cached per-fold (Lim [nq x nid], ImgMat [nq x ngal_fold]) for lnb_img candidate expansion & SIFT selection."""
    _, Lim, _, _, _, ImgMat = fold_all_scores(fold, ctx, cand_ids)
    return Lim, ImgMat


def add_features(df, fold, ctx, cand_ids):
    """Adds per-(query, candidate) LNBNN features and q_ query-level ones. cand_ids: identity names of df.cand index."""
    L, Lim, Lk3, Lu, _, _ = fold_all_scores(fold, ctx, cand_ids)
    mx = L.max(1); second = -np.sort(-L, 1)[:, 1] if L.shape[1] > 1 else mx
    rk = np.argsort(np.argsort(-L, 1), 1)
    qi, c = df.qi.values, df.cand.values
    v = L[qi, c]
    df["lnb"] = v; df["lnb_rank"] = rk[qi, c]; df["lnb_rel"] = np.where(mx[qi] > 0, v / np.maximum(mx[qi], 1e-9), 0.0)
    df["lnb_log"] = np.log1p(v)
    df["q_lnb_max"] = mx[qi]; df["q_lnb_gap"] = (mx - second)[qi]; df["q_lnb_log_max"] = np.log1p(mx)[qi]
    if ctx.cfg["retrieval"].get("lnbnn_img", False):
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
    if ctx.cfg["retrieval"].get("bg_clean", False):
        mx_u = Lu.max(1); sec_u = -np.sort(-Lu, 1)[:, 1] if Lu.shape[1] > 1 else mx_u
        rk_u = np.argsort(np.argsort(-Lu, 1), 1)
        vu = Lu[qi, c]
        df["lnb_uimg"] = vu; df["lnb_uimg_rank"] = rk_u[qi, c]
        df["lnb_uimg_rel"] = np.where(mx_u[qi] > 0, vu / np.maximum(mx_u[qi], 1e-9), 0.0)
        df["lnb_uimg_log"] = np.log1p(vu)
        df["q_lnb_uimg_max"] = mx_u[qi]; df["q_lnb_uimg_gap"] = (mx_u - sec_u)[qi]
        df["q_lnb_uimg_log_max"] = np.log1p(mx_u)[qi]
    return df
