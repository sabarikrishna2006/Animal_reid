"""Gate 5: test-to-test (query-to-query) graph + label propagation, simulated identically on CV query sets.

1. Image-pair model P(same | q, q') on query-query neighbour pairs (cosines, mutual ranks, SIFT/RANSAC, sides),
   trained out-of-fold on CV pairs grouped by identity; fit on all CV pairs for test.
2. Graph per fold: edge (q, q') if q' in q's top-k (or vice versa) and P(same) > tau_e; optional consensus filter
   (agglomerative clusters AND Louvain communities must agree).
3. Label propagation (Zhou et al. 2004): F <- alpha S F + (1 - alpha) P0, S = D^-1/2 W D^-1/2, rows renormalised.
   P0 = per-query posterior over candidate identities U {new}. alpha, tau_e and the 'new' offset beta tuned on CV.
"""
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import logit, expit

from . import NEW, SPECIES
from .calibration import softmax_groups, shift_logit
from .candidates import query_neighbour_pairs
from .models import oof_predict, fit_full
from .sift_match import FEATS as SF
from .splits import test_fold

EXPORT_PROBS = {}   # test query id -> {label: probability}, filled by apply_transductive

KP_FEATS = ["kp_qq", "kp_qq_sym", "kp_qq_rank"]   # keypoint query-graph pair features (transductive.kp_graph)
PAIR_FEATS = ["cos_l2", "cos_pcaw", "rank_ij", "rank_ji", "same_side", "opp_side", "kp_min", "dlognorm",
              "sift_inl", "sift_inl_strict", "sift_mutual", "sift_ratio", "sift_overkp", "sift_spread",
              "top1_same", "p1_min", "species_is_leo"]


def _posteriors(df, qdf, T, beta):
    """Per query: candidate labels and probabilities (incl. new) before propagation."""
    df = df.copy()
    df["_key"] = df.split.astype(str) + "|" + df.query_id
    gid = pd.factorize(df["_key"])[0]
    Tv = df.species.map(T).values
    df["post"] = softmax_groups(logit(np.clip(df.p.values, 1e-6, 1 - 1e-6)), gid, Tv)
    qkey = qdf.split.astype(str) + "|" + qdf.query_id
    pn = pd.Series(shift_logit(qdf.pnew.values, qdf.species.map(beta).values), index=qkey.values)
    out = {}
    for key, g in df.groupby("_key", sort=False):
        pi = pn[key]
        out[key] = (list(g.cand_id.values) + [NEW], np.r_[g.post.values * (1 - pi), pi])
    return out


def _pair_table(fold, ctx, k, post, split, kpg=None):
    Z = ctx.emb["l2"][fold.query_rows]; Zp = ctx.emb["pcaw"][fold.query_rows]
    S = Z @ Z.T; np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    rank = np.empty_like(order); rank[np.arange(len(S))[:, None], order] = np.arange(S.shape[1])[None, :]
    nn = order[:, :k]
    rows = []
    qr, qids = fold.query_rows, fold.query_ids
    side, kp, ln = ctx.side, ctx.kp, ctx.lognorm
    kscore, ktop = kpg if kpg is not None else ({}, {})
    for i in range(len(qr)):
        li, pi_ = post[f"{split}|{qids[i]}"]
        cand = list(nn[i]) + [j for j in ktop.get(i, []) if j not in set(nn[i])]   # + keypoint-graph neighbours
        for j in cand:
            if j == i:
                continue
            lj, pj = post[f"{split}|{qids[j]}"]
            f = ctx.pc.get(qr[i], qr[j]) if ctx.pc is not None else None
            if f is None:
                f = np.full(len(SF), np.nan, np.float32)
            si, sj = side[qr[i]], side[qr[j]]
            rows.append((i, j, S[i, j], float(Zp[i] @ Zp[j]), rank[i, j], rank[j, i],
                         float(si == sj and si in "LR"), float({si, sj} == {"L", "R"}),
                         min(kp[qr[i]], kp[qr[j]]), abs(ln[qr[i]] - ln[qr[j]]),
                         f[SF.index("inl_aff")], f[SF.index("inl_aff_strict")], f[SF.index("n_mutual")],
                         f[SF.index("inl_ratio")], f[SF.index("inl_over_kp")], f[SF.index("spread_min")],
                         float(li[int(np.argmax(pi_))] == lj[int(np.argmax(pj))]), float(min(pi_.max(), pj.max())),
                         int(fold.species == "leopard"))
                        + ((kscore.get((i, j), 0.0), kscore.get((j, i), 0.0),
                            ktop[i].index(j) if j in ktop.get(i, []) else 99) if kpg is not None else ()))
    df = pd.DataFrame(rows, columns=["i", "j"] + PAIR_FEATS + (KP_FEATS if kpg is not None else []))
    return df


def _propagate(n, labels_per_q, probs_per_q, edges, w, alpha, iters=30):
    """Returns list of (labels, probs) after propagation over a union label space."""
    if alpha == 0 or len(edges) == 0:
        return list(zip(labels_per_q, probs_per_q))
    lab_index = {}
    for labs in labels_per_q:
        for l in labs:
            lab_index.setdefault(l, len(lab_index))
    P0 = np.zeros((n, len(lab_index)))
    for q, (labs, pr) in enumerate(zip(labels_per_q, probs_per_q)):
        P0[q, [lab_index[l] for l in labs]] = pr
    i, j = edges[:, 0], edges[:, 1]
    W = sp.coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    W.data = np.minimum(W.data, 1.0)  # duplicates summed -> cap
    d = np.asarray(W.sum(1)).ravel()
    dinv = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1e-12)), 0)
    Sn = sp.diags(dinv) @ W @ sp.diags(dinv)
    F = P0.copy()
    for _ in range(iters):
        F = alpha * (Sn @ F) + (1 - alpha) * P0
    F /= F.sum(1, keepdims=True)
    inv = np.array(list(lab_index.keys()), dtype=object)
    out = []
    for q in range(n):
        nz = np.where(F[q] > 1e-9)[0]
        out.append((list(inv[nz]), F[q, nz]))
    return out


def _consensus_filter(n, edges, w, thr):
    """Keep edges whose endpoints share both an agglomerative cluster (avg linkage on 1-P) and a Louvain community."""
    if len(edges) == 0:
        return edges, w
    G = nx.Graph(); G.add_nodes_from(range(n))
    G.add_weighted_edges_from([(int(a), int(b), float(x)) for (a, b), x in zip(edges, w)])
    comm = nx.community.louvain_communities(G, weight="weight", seed=0)
    cid = np.empty(n, int)
    for k, c in enumerate(comm):
        cid[list(c)] = k
    # agglomerative (average linkage on 1 - P(same)) inside each connected component
    acl = np.arange(n)
    comp_id = np.full(n, -1)
    comps = [sorted(c) for c in nx.connected_components(G)]
    for k_, c in enumerate(comps):
        comp_id[c] = k_
    local = np.zeros(n, int)
    for c in comps:
        local[c] = np.arange(len(c))
    Ds = [np.ones((len(c), len(c))) for c in comps]
    for (a, b), x in zip(edges, w):
        D = Ds[comp_id[a]]; ia, ib = local[a], local[b]
        D[ia, ib] = D[ib, ia] = min(D[ia, ib], 1 - x)
    nxt = n
    for c, D in zip(comps, Ds):
        if len(c) < 2:
            continue
        np.fill_diagonal(D, 0)
        lab = fcluster(linkage(D[np.triu_indices(len(c), 1)], "average"), 1 - thr, "distance")
        acl[c] = lab + nxt; nxt += lab.max() + 1
    keep = (cid[edges[:, 0]] == cid[edges[:, 1]]) & (acl[edges[:, 0]] == acl[edges[:, 1]])
    return edges[keep], w[keep]


def _ap(labels, probs, truth, new_scale):
    pr = np.array(probs, float).copy()
    labs = list(labels)
    if NEW in labs:
        pr[labs.index(NEW)] *= new_scale
    order = np.argsort(-pr, kind="stable")[:5]
    for k, o in enumerate(order, 1):
        if labs[o] == truth:
            return 1.0 / k
    return 0.0


def _rank(labels, probs, new_scale):
    pr = np.array(probs, float).copy(); labs = list(labels)
    if NEW in labs:
        pr[labs.index(NEW)] *= new_scale
    else:
        labs.append(NEW); pr = np.r_[pr, 0.0]
    order = np.argsort(-pr, kind="stable")[:5]
    return [labs[o] for o in order]


def apply_transductive(cfg, ctx, data, splits, cv, qdf, test, tq, T, beta, log=print):
    tc = cfg["transductive"]
    k = cfg["sift"]["graph_k"]
    t0 = time.time()
    folds = [(r, sp_, e[sp_][0]) for r, e in enumerate(splits) for sp_ in SPECIES]
    tfolds = [(-1, sp_, test_fold(data, sp_)) for sp_ in SPECIES]
    KP = {}
    if tc.get("kp_graph", False):
        # keypoint query graph: extra candidate edges between query images that share SIFT keypoints (cached per fold)
        import hashlib, pickle as _pk
        from .io import P as _P
        from .kpgraph import query_graph
        for r, sp_, f in folds + tfolds:
            h = hashlib.md5(np.asarray(f.query_rows).tobytes()).hexdigest()[:12]
            cp = _P(cfg, "cache") / "kpgraph" / f"{sp_}_{h}.pkl"
            if cp.exists():
                KP[(r, sp_)] = _pk.loads(cp.read_bytes())
            else:
                KP[(r, sp_)] = query_graph(f, ctx); cp.parent.mkdir(parents=True, exist_ok=True); cp.write_bytes(_pk.dumps(KP[(r, sp_)]))
        log(f"[trans] keypoint query graph ready for {len(KP)} folds ({time.time() - t0:.0f}s)")
    PF = PAIR_FEATS + (KP_FEATS if KP else [])
    if ctx.pc is not None:
        pairs = []
        for r, sp_, f in folds + tfolds:
            pairs += query_neighbour_pairs(f, ctx, k)
            if KP:
                pairs += [(f.query_rows[i], f.query_rows[j]) for i, js in KP[(r, sp_)][1].items() for j in js]
        ctx.pc.ensure(pairs)
    post_cv = _posteriors(cv, qdf, T, beta)
    post_te = _posteriors(test, tq, T, beta)
    ident = data.meta.identity.values
    ptabs = []
    for r, sp_, f in folds:
        pt = _pair_table(f, ctx, k, post_cv, r, KP.get((r, sp_)))
        pt["y"] = (ident[f.query_rows[pt.i.values]] == ident[f.query_rows[pt.j.values]]).astype(int)
        pt["group"] = ident[f.query_rows[pt.i.values]]
        pt["split"] = r; pt["species"] = sp_
        ptabs.append(pt)
    pcv = pd.concat(ptabs, ignore_index=True)
    pcv["p_same"], _ = oof_predict(pcv, PF, "y", pcv.group.values, kind="lgb")
    from sklearn.metrics import roc_auc_score
    log(f"[trans] pair model OOF AUC={roc_auc_score(pcv.y, pcv.p_same):.4f}, "
        f"AUC cos only={roc_auc_score(pcv.y, pcv.cos_l2):.4f}, pairs={len(pcv)} ({time.time() - t0:.0f}s)")
    pm = fit_full(pcv, PF, "y", pcv.group.values, kind="lgb")
    tt = []
    for r, sp_, f in tfolds:
        pt = _pair_table(f, ctx, k, post_te, r, KP.get((r, sp_))); pt["split"] = r; pt["species"] = sp_
        tt.append(pt)
    pte = pd.concat(tt, ignore_index=True)
    pte["p_same"] = pm.predict(pte[PF].to_numpy(np.float32))

    def run_fold(post, ptab, f, r, alpha, tau_e, consensus):
        keys = [f"{r}|{q}" for q in f.query_ids]
        labs = [post[kk][0] for kk in keys]; prs = [post[kk][1] for kk in keys]
        m = ptab.p_same.values > tau_e
        edges = ptab[["i", "j"]].values[m]; w = ptab.p_same.values[m]
        if consensus and len(edges):
            edges, w = _consensus_filter(len(keys), edges, w, tau_e)
        return keys, _propagate(len(keys), labs, prs, edges, w, alpha)

    truth_by_key = dict(zip(qdf.split.astype(str) + "|" + qdf.query_id, qdf.truth))
    grid_alpha = tc.get("alphas", [0.0, 0.2, 0.4, 0.6, 0.8])
    grid_tau = tc.get("tau_e", [0.5, 0.7, 0.9])
    scales = np.exp(np.arange(-1.5, 1.51, 0.25))
    best = {}
    for sp_ in SPECIES:
        res = []
        for alpha in grid_alpha:
            for tau_e in (grid_tau if alpha > 0 else [1.0]):
                aps = {s: [] for s in range(len(scales))}
                for r, s2, f in folds:
                    if s2 != sp_:
                        continue
                    keys, outp = run_fold(post_cv, pcv[(pcv.split == r) & (pcv.species == sp_)], f, r, alpha, tau_e,
                                          tc.get("consensus", True))
                    for si, sc in enumerate(scales):
                        aps[si].append(np.mean([_ap(l, p, truth_by_key[kk], sc) for kk, (l, p) in zip(keys, outp)]))
                for si, sc in enumerate(scales):
                    res.append((alpha, tau_e, float(np.log(sc)), float(np.mean(aps[si])), aps[si]))
        df = pd.DataFrame(res, columns=["alpha", "tau_e", "log_new_scale", "mapv", "per_split"])
        b = df.loc[df.mapv.idxmax()]
        off0 = df[df.alpha == 0]; off = off0.loc[off0.mapv.idxmax()]
        best[sp_] = dict(alpha=float(b.alpha), tau_e=float(b.tau_e), log_new_scale=float(b.log_new_scale),
                         map=float(b.mapv), per_split=list(b.per_split), map_off=float(off.mapv),
                         per_split_off=list(off.per_split),
                         grid=[(float(x.alpha), float(x.tau_e), float(x.log_new_scale), [float(v) for v in x.per_split])
                               for x in df.itertuples()])
        log(f"[trans] {sp_}: best alpha={b.alpha} tau_e={b.tau_e} new_scale=e^{b.log_new_scale:.1f} "
            f"MAP={b.mapv:.4f} (alpha=0: {off.mapv:.4f})")
    # test predictions
    preds = {}
    for r, sp_, f in tfolds:
        bb = best[sp_]
        keys, outp = run_fold(post_te, pte[pte.species == sp_], f, r, bb["alpha"], bb["tau_e"], tc.get("consensus", True))
        for kk, (l, p) in zip(keys, outp):
            preds[kk.split("|", 1)[1]] = _rank(l, p, np.exp(bb["log_new_scale"]))
            # export for team blending: final label probabilities (same new_scale as the ranking), renormalised
            ll, pp = list(l), np.array(p, float).copy()
            if NEW in ll:
                pp[ll.index(NEW)] *= np.exp(bb["log_new_scale"])
            EXPORT_PROBS[kk.split("|", 1)[1]] = dict(zip(ll, pp / pp.sum()))
    return preds, best, pcv
