"""Core data loader, L2 normalization, and 1-query-per-encounter Honest CV splits for dev_pipeline."""
import os
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

NEW = "new_individual"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "reid_student_data_main")


def l2n(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def load():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    mids = pd.read_csv(os.path.join(DATA, "miewid_ids.csv"))["id"].values
    Xall = np.load(os.path.join(DATA, "miewid_features.npy")).astype(np.float32)
    idx = {k: i for i, k in enumerate(mids)}
    Xtr = Xall[[idx[i] for i in tr.id]]
    Xte = Xall[[idx[i] for i in te.id]]
    return tr, te, Xtr, Xte


def encounters(tr, A, thr=0.8):
    enc = np.zeros(len(tr), int)
    nxt = 0
    for _, g in tr.groupby("identity"):
        idx = g.index.values
        n, lab = connected_components(A[idx] @ A[idx].T > thr, directed=False)
        enc[idx] = lab + nxt
        nxt += n
    return enc


def make_split(tr, enc, seed, new_frac=0.1, q_frac=0.4, per_enc=1):
    rng = np.random.RandomState(seed)
    is_q = np.zeros(len(tr), bool)
    for s in ["hyena", "leopard"]:
        m = (tr.species == s).values
        idents = tr.identity[m].unique()
        held = set(rng.choice(idents, int(round(len(idents) * new_frac)), replace=False))
        is_q |= m & tr.identity.isin(held).values
        for _, g in tr[m & ~tr.identity.isin(held).values].groupby("identity"):
            e = np.unique(enc[g.index.values])
            if len(e) < 2:
                continue
            pick = [x for x in e if rng.rand() < q_frac]
            if len(pick) == len(e):
                pick = pick[1:]
            is_q[g.index.values[np.isin(enc[g.index.values], pick)]] = True
    gal = ~is_q
    rng2 = np.random.RandomState(1000 + seed)
    idx = np.where(is_q)[0]
    keep = np.zeros_like(is_q)
    for e in np.unique(enc[idx]):
        m = idx[enc[idx] == e]
        keep[rng2.choice(m, min(per_enc, len(m)), replace=False)] = True
    gal_ids = set(tr.identity[gal])
    truth = np.array([t if t in gal_ids else NEW for t in tr.identity[keep]])
    return gal, keep, truth


def val_splits(tr, A, seeds=range(16), thr=0.8, new_frac=0.1, q_frac=0.4, per_enc=1):
    enc = encounters(tr, A, thr=thr)
    return [make_split(tr, enc, s, new_frac=new_frac, q_frac=q_frac, per_enc=per_enc) for s in seeds]
