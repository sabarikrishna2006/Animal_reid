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
            _, s, Vt = np.linalg.svd(Z - mu, full_matrices=False)
            ev = s ** 2 / (len(Z) - 1)
            out[m] = (Z - mu) @ (Vt[:d].T / np.sqrt(ev[:d] + eps * ev.mean()))
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
    _, s, Vt = np.linalg.svd(Zg - mu, full_matrices=False)
    Vp = Vt[:pca_d].T
    Y = (Zg - mu) @ Vp
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
