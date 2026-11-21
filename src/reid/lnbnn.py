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


def scorer(ctx, sp):
    if sp not in _CACHE:
        from .io import P
        d = Path(ctx.cfg["paths"].get("lnbnn", "data/cache/lnbnn"))
        d = d if d.is_absolute() else P(ctx.cfg, "cache").parent.parent / d
        tr = ctx.data.train
        _CACHE[sp] = Scorer(d, sp, tr.id.values, tr.identity.values)
    return _CACHE[sp]


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
    return df
