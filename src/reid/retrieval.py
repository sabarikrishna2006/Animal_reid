"""Stage A: embeddings (unsupervised / gallery-supervised transforms) and identity-level aggregation."""
import numpy as np

from . import NEW


def l2n(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def embed_unsupervised(X, meta, kind, **kw):
    """Label-free transforms fitted on train U test features (allowed at test time). Returns L2-normalised matrix."""
    X = X.astype(np.float64)
    if kind == "l2":
        return l2n(X)
    if kind == "std":
        return l2n((X - X.mean(0)) / X.std(0))
    if kind == "pcaw":
        d, eps = kw.get("d", 512), kw.get("eps", 1.0)
        out = np.zeros((len(X), d))
        for sp in ("hyena", "leopard"):
            m = (meta.species == sp).values
            Z = l2n(X[m]); mu = Z.mean(0)
            Zc = Z - mu
            C = (Zc.T @ Zc) / max(len(Z) - 1, 1)
            ev, V = np.linalg.eigh(C)
            idx = np.argsort(ev)[::-1]
            ev = np.maximum(ev[idx], 0.0)
            Vd = V[:, idx[:d]]
            out[m] = Zc @ (Vd / np.sqrt(ev[:d] + eps * ev.mean()))
        return l2n(out)
    if kind == "qe":
        base = kw["base"]; k, alpha = kw.get("k", 3), kw.get("alpha", 3.0)
        out = np.zeros_like(base)
        for sp in ("hyena", "leopard"):
            m = np.where((meta.species == sp).values)[0]
            Z = base[m]; S = Z @ Z.T
            nn = np.argsort(-S, axis=1)[:, :k + 1]            # includes self
            w = np.maximum(np.take_along_axis(S, nn, 1), 0) ** alpha
            out[m] = (w[:, :, None] * Z[nn]).sum(1)
        return l2n(out)
    raise ValueError(kind)


def fit_lda(Zg, labels, gamma=0.2, d_out=256, pca_d=512, sides=None, w_cross=0.0):
    """Regularised LDA / learned whitening on GALLERY labels only.
    Zg: gallery embeddings (L2-normalised). Returns projection function z -> L2-normalised projected z."""
    mu = Zg.mean(0)
    Zc = (Zg - mu).astype(np.float64)
    C = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
    _, V = np.linalg.eigh(C)
    Vp = V[:, ::-1][:, :pca_d]
    Y = Zc @ Vp
    uniq, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    means = np.zeros((len(uniq), Y.shape[1])); np.add.at(means, inv, Y); means /= cnt[:, None]
    multi = cnt[inv] >= 2
    R = Y[multi] - means[inv[multi]]
    Sw = R.T @ R / max(len(R) - 1, 1)
    if sides is not None and w_cross > 0:
        # cross-flank scatter: per-identity (left mean - right mean); suppresses flank-dependent directions
        D = []
        for k in range(len(uniq)):
            m = inv == k; L = m & (sides == "L"); Rr = m & (sides == "R")
            if L.any() and Rr.any():
                D.append(Y[L].mean(0) - Y[Rr].mean(0))
        if len(D) >= 10:
            D = np.array(D); Sx = D.T @ D / len(D)
            Sw = Sw / np.trace(Sw) + w_cross * Sx / np.trace(Sx)
    Sw = (1 - gamma) * Sw + gamma * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    ew, Ew = np.linalg.eigh(Sw)
    Wh = Ew / np.sqrt(np.maximum(ew, 1e-12))       # whitening by within-class scatter
    Mb = (means - means.mean(0)) @ Wh
    Sb = (Mb * cnt[:, None]).T @ Mb / cnt.sum()
    eb, Eb = np.linalg.eigh(Sb)
    Eb = Eb[:, ::-1][:, :d_out]
    W = Vp @ Wh @ Eb
    return lambda Z: l2n((Z - mu) @ W)


class IdentityIndex:
    """Groups gallery columns by identity to aggregate image similarities into identity scores."""

    def __init__(self, gallery_labels):
        self.labels = np.asarray(gallery_labels)
        self.ids, inv = np.unique(self.labels, return_inverse=True)
        self.order = np.argsort(inv, kind="stable")
        self.inv = inv
        self.counts = np.bincount(inv)
        self.starts = np.r_[0, np.cumsum(self.counts)[:-1]]

    def max(self, S):
        return np.maximum.reduceat(S[:, self.order], self.starts, axis=1)

    def sorted_desc_k(self, S, k):
        """mean of top-k image similarities per identity (identities with < k images use what they have)."""
        out = np.zeros((S.shape[0], len(self.ids)))
        So = S[:, self.order]
        for j, (a, c) in enumerate(zip(self.starts, self.counts)):
            blk = So[:, a:a + c]
            kk = min(k, c)
            out[:, j] = -np.sort(-blk, axis=1)[:, :kk].mean(1) if c > 1 else blk[:, 0]
        return out

    def logsumexp(self, S, tau):
        So = np.exp((S[:, self.order] - 1) / tau)
        return 1 + tau * np.log(np.add.reduceat(So, self.starts, axis=1))

    def prototypes(self, Zg):
        P = np.zeros((len(self.ids), Zg.shape[1])); np.add.at(P, self.inv, Zg)
        return l2n(P)


def baseline_predict(id_scores, ids, tau, k=5):
    """Insert new_individual after identities with score >= tau."""
    preds = []
    order = np.argsort(-id_scores, axis=1)[:, :k]
    for i in range(len(id_scores)):
        labs = [ids[j] for j in order[i]]
        sc = id_scores[i, order[i]]
        pos = int((sc >= tau).sum())
        labs = labs[:pos] + [NEW] + labs[pos:]
        preds.append(labs[:k])
    return preds


def rg_diff(Z, labels, K=10):
    """Gallery-only hubness: mean top-K cosine to DIFFERENT-identity gallery images."""
    Sgg = Z @ Z.T
    Sgg_diff = np.where(labels[:, None] == labels[None, :], -2.0, Sgg)
    k = min(K, max(1, len(Z) - 1))
    return np.partition(Sgg_diff, -k, axis=1)[:, -k:].mean(1).astype(np.float32)


def _sw_basis(Zg, m0, labels, reg=0.5):
    names, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    means = np.zeros((len(names), Zg.shape[1]), dtype=np.float64)
    np.add.at(means, inv, Zg)
    means /= cnt[:, None]
    multi = cnt[inv] >= 2
    Zw = (Zg[multi] - means[inv[multi]]).astype(np.float64)
    Sw = (Zw.T @ Zw) / max(len(Zw), 1)
    ew, vw = np.linalg.eigh(Sw)
    ew = np.maximum(ew, 0.0)
    scale = (1.0 / np.sqrt((1.0 - reg) * ew + reg * max(ew.mean(), 1e-12))).astype(np.float32)
    return vw.astype(np.float32), scale, means, cnt


def fit_sw_and_fullda(Zg, glab, reg=0.5, d_lda=512, gside=None):
    """100% gallery-only Full 2152-dim Sw whitening, Full-2152d Fisher LDA, cl2 centering, and optional Side-Sw."""
    Zg = np.asarray(Zg, dtype=np.float32)
    m0 = Zg.mean(0).astype(np.float32)
    vw, scale, means, cnt = _sw_basis(Zg, m0, glab, reg=reg)
    W_sw = vw * scale[None, :]
    Mc_w = ((means - means.mean(0)) @ vw.astype(np.float64)) * scale.astype(np.float64)[None, :]
    Sb_w = (Mc_w * cnt[:, None]).T @ Mc_w / max(cnt.sum(), 1)
    _, Eb = np.linalg.eigh(Sb_w)
    W_fl = (W_sw @ Eb[:, ::-1][:, :d_lda].astype(np.float32))
    out = {
        "cl2": lambda Z, m0=m0: l2n(Z - m0).astype(np.float32),
        "sw": lambda Z, m0=m0, W=W_sw: l2n((Z - m0) @ W).astype(np.float32),
        "fl512": lambda Z, m0=m0, W=W_fl: l2n((Z - m0) @ W).astype(np.float32),
    }
    if gside is not None:
        glab_side = np.array([f"{a}|{b}" if b in ("L", "R") else a for a, b in zip(glab, gside)])
        vws, scs, _, _ = _sw_basis(Zg, m0, glab_side, reg=reg)
        W_ssw = vws * scs[None, :]
        out["ssw"] = lambda Z, m0=m0, W=W_ssw: l2n((Z - m0) @ W).astype(np.float32)
    return out


def vp_compat_stats(Zg, labels, vps, min_n=8):
    """Mean and std of same-identity cosines per viewpoint pair from gallery labels only."""
    import pandas as pd
    acc = {}
    for _, idx in pd.Series(np.arange(len(labels))).groupby(labels):
        idx = idx.values
        if len(idx) < 2:
            continue
        S = Zg[idx] @ Zg[idx].T
        iu = np.triu_indices(len(idx), 1)
        for a, b, x in zip(vps[idx][iu[0]], vps[idx][iu[1]], S[iu]):
            k = (a, b) if a <= b else (b, a)
            acc.setdefault(k, []).append(float(x))
    return {k: (float(np.mean(v)), float(max(np.std(v), 0.05))) for k, v in acc.items() if len(v) >= min_n}

