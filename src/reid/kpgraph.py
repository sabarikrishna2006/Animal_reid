"""Query-query keypoint graph (transductive, no labels): for every query image of a fold, find the other QUERY images that
share the most SIFT keypoints, scored LNBNN-style (classical NN matching of the provided descriptors, faiss-cpu).
Used to add candidate edges (and a pair feature) to the label-propagation graph, which otherwise only links images
that are MiewID-cosine neighbours."""
import numpy as np
import faiss

from .io import Sift

KNN, K = 30, 10
_SIFT = {}


def _rootsift(ctx, row):
    if "s" not in _SIFT:
        _SIFT["s"] = Sift(ctx.cfg)
    d = _SIFT["s"][row][1].astype(np.float32).reshape(-1, 128)
    return np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1e-7))


def query_graph(fold, ctx, m=5):
    """Returns (score dict {(i, j): s}, top dict {i: [j1..jm]}) over fold-local query indices."""
    rows = fold.query_rows
    X = [_rootsift(ctx, r) for r in rows]
    lens = np.array([len(x) for x in X]); owner = np.repeat(np.arange(len(rows)), lens).astype(np.int32)
    X = np.concatenate(X).astype(np.float32)
    nlist = int(min(1024, max(64, len(X) // 1000)))
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(128), 128, nlist)
    index.train(X[np.random.RandomState(0).choice(len(X), min(len(X), 100000), replace=False)])
    index.add(X); index.nprobe = 8
    d, nb = index.search(X, KNN)
    nb = np.where(nb >= 0, owner[np.maximum(nb, 0)], -1)
    d = np.sqrt(np.maximum(d, 0))
    ok = (nb >= 0) & (nb != owner[:, None])                       # never the query image itself
    rank = np.cumsum(ok, 1); has_bg = rank[:, -1] >= K + 1
    bg = np.where(has_bg, d[np.arange(len(d)), np.argmax(rank >= K + 1, 1)], np.max(np.where(ok, d, 0), 1))
    r, c = np.nonzero(ok & (rank <= K))
    key = owner[r].astype(np.int64) * len(rows) + nb[r, c]
    # one vote per (query descriptor, other image): nearest occurrence
    dk = r.astype(np.int64) * len(rows) + nb[r, c]
    _, first = np.unique(dk, return_index=True)
    key, val = key[first], np.maximum(bg[r[first]] - d[r[first], c[first]], 0)
    uk, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, weights=val)
    score = {(int(k // len(rows)), int(k % len(rows))): float(v) for k, v in zip(uk, tot)}
    top = {}
    for (i, j), v in score.items():
        top.setdefault(i, []).append((v, j))
    top = {i: [j for _, j in sorted(l, reverse=True)[:m]] for i, l in top.items()}
    return score, top
