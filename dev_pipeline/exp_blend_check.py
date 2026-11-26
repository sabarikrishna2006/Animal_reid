"""Probability calibration, heuristic keypoint/cosine graph propagation, and blend evaluation helpers."""
import numpy as np
import scipy.sparse as sp_sparse
from core import NEW


def to_probs_raw(P, Qf, T):
    pn = dict(zip(Qf.qid.values, Qf.pn2.values))
    out = {}
    for q, g in P.groupby("qid", sort=False):
        labs = list(g.cand.values) + [NEW]
        sc = np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)) / T
        e = np.exp(sc - sc.max())
        out[q] = dict(zip(labs, e / e.sum()))
    return out


def propagate_probs(raw_probs, Qf, G, alpha=0.30, kp_thr=2.0, cos_thr=0.72, iters=30):
    qids = list(Qf.qid.values)
    q_idx = {q: i for i, q in enumerate(qids)}
    n = len(qids)
    ii, jj, ww = [], [], []
    for q, ids, sims in zip(Qf.qid.values, Qf.nbr_ids.values, Qf.nbr_sims.values):
        i = q_idx[q]
        for nb, s in zip(ids, sims):
            if s >= cos_thr and nb in q_idx:
                ii.append(i)
                jj.append(q_idx[nb])
                ww.append(float(s))
        for nb, s1, s2 in G.get(q, []):
            if min(s1, s2) >= kp_thr and nb in q_idx:
                ii.append(i)
                jj.append(q_idx[nb])
                ww.append(float(min(1.0, min(s1, s2) / 10.0)))
    if not ii or alpha <= 0.0:
        return raw_probs
    lab_index = {}
    for q in qids:
        for l in raw_probs[q]:
            lab_index.setdefault(l, len(lab_index))
    P0 = np.zeros((n, len(lab_index)), dtype=np.float64)
    for i, q in enumerate(qids):
        for l, pr in raw_probs[q].items():
            P0[i, lab_index[l]] = pr
    ii = np.array(ii, np.int32)
    jj = np.array(jj, np.int32)
    ww = np.array(ww, np.float64)
    W = sp_sparse.coo_matrix((np.r_[ww, ww], (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n)).tocsr()
    W.data = np.minimum(W.data, 1.0)
    d = np.asarray(W.sum(1)).ravel()
    dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    Sn = sp_sparse.diags(dinv) @ W @ sp_sparse.diags(dinv)
    F = P0.copy()
    for _ in range(iters):
        F = alpha * (Sn @ F) + (1.0 - alpha) * P0
    F /= np.maximum(F.sum(1, keepdims=True), 1e-12)
    inv_labs = list(lab_index.keys())
    out = {}
    for i, q in enumerate(qids):
        nz = np.where(F[i] > 1e-9)[0]
        out[q] = {inv_labs[k]: float(F[i, k]) for k in nz}
    return out


def eval_probs(probs, truth):
    s = 0.0
    for q, t in truth.items():
        d = probs[q]
        top5 = sorted(d, key=d.get, reverse=True)[:5]
        for k, lab in enumerate(top5):
            if lab == t:
                s += 1.0 / (k + 1)
                break
    return s / len(truth)


def blend_dict(p1, p2, w=0.5):
    out = {}
    for q in p1:
        labs = set(p1[q]) | set(p2[q])
        out[q] = {l: (1.0 - w) * p1[q].get(l, 0.0) + w * p2[q].get(l, 0.0) for l in labs}
    return out
