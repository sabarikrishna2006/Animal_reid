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
from .models import oof_predict, fit_full, predict_bag
from .sift_match import FEATS as SF
from .splits import test_fold

EXPORT_CV = {}      # CV split -> query id -> {label: probability}
EXPORT_PROBS = {}   # test query id -> {label: probability}, filled by apply_transductive

KP_FEATS = ["kp_qq", "kp_qq_sym", "kp_qq_rank"]
SIFT_BG_FEATS = ["sift_rot", "sift_scale", "sift_is_bg", "sift_inl_clean"]
SIFT_BG_V2_FEATS = ["sift_inlh", "sift_abs_scale", "sift_is_zero_rot"]
SW_FEATS = ["cos_sw", "cos_fl512", "rank_sw_ij", "rank_sw_ji", "krec_qq_sw"]
RICH_FEATS = ["cos_lda_qq", "krec_qq_l2", "krec_qq_lda", "post_dot", "kp_qq_min", "kp_qq_log"]
PAIR_FEATS = ["cos_l2", "cos_pcaw", "rank_ij", "rank_ji", "same_side", "opp_side", "kp_min", "dlognorm",
              "sift_inl", "sift_inl_strict", "sift_mutual", "sift_ratio", "sift_overkp", "sift_spread",
              "top1_same", "p1_min", "species_is_leo"]


def _krec_qq(Zq, Zg, k1=20, k2=6, lam=0.3):
    """Unsupervised k-reciprocal Jaccard similarity among queries (nq x nq) over [Zq; Zg]."""
    nq = len(Zq)
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
    Vq = V[:nq]
    jaccard = np.ones((nq, nq), dtype=np.float32)
    for i in range(nq):
        nz = np.nonzero(Vq[i])[0]
        if len(nz):
            mn = np.minimum(Vq[i, nz][None, :], Vq[:, nz]).sum(axis=1)
            jaccard[i] = 1.0 - mn / (2.0 - mn)
    sim = 1.0 - ((1.0 - lam) * jaccard + lam * orig_dist[:nq, :nq])
    np.fill_diagonal(sim, -np.inf)
    return sim


def _rich_mats(fold, ctx):
    from .retrieval import fit_lda
    Zq_l2, Zg_l2 = ctx.emb["l2"][fold.query_rows], ctx.emb["l2"][fold.gallery_rows]
    f = fit_lda(Zg_l2, fold.gallery_labels, gamma=0.3, d_out=256)
    Zq_lda, Zg_lda = f(Zq_l2), f(Zg_l2)
    S_lda = Zq_lda @ Zq_lda.T; np.fill_diagonal(S_lda, -np.inf)
    Kr_l2 = _krec_qq(Zq_l2, Zg_l2)
    Kr_lda = _krec_qq(Zq_lda, Zg_lda)
    top_kr = np.argsort(-Kr_lda, axis=1)[:, :5]
    return S_lda, Kr_l2, Kr_lda, top_kr


def _sw_mats(fold, ctx):
    from .retrieval import fit_sw_and_fullda
    Zq_l2, Zg_l2 = ctx.emb["l2"][fold.query_rows], ctx.emb["l2"][fold.gallery_rows]
    projs = fit_sw_and_fullda(Zg_l2, fold.gallery_labels, reg=0.5, d_lda=512)
    Zq_sw, Zg_sw = projs["sw"](Zq_l2), projs["sw"](Zg_l2)
    Zq_fl = projs["fl512"](Zq_l2)
    S_sw = Zq_sw @ Zq_sw.T; np.fill_diagonal(S_sw, -np.inf)
    S_fl = Zq_fl @ Zq_fl.T; np.fill_diagonal(S_fl, -np.inf)
    order = np.argsort(-S_sw, axis=1)
    rank_sw = np.empty_like(order)
    rank_sw[np.arange(len(S_sw))[:, None], order] = np.arange(S_sw.shape[1])[None, :]
    Kr_sw = _krec_qq(Zq_sw, Zg_sw, k1=10, k2=4, lam=0.3)
    top_sw = order[:, :5]
    top_kr = np.argsort(-Kr_sw, axis=1)[:, :5]
    return S_sw, S_fl, rank_sw, Kr_sw, top_sw, top_kr


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


def _pair_table(fold, ctx, k, post, split, kpg=None, rich=None, sift_bg=False, swm=None, sift_bg_v2=False):
    Z = ctx.emb["l2"][fold.query_rows]; Zp = ctx.emb["pcaw"][fold.query_rows]
    S = Z @ Z.T; np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    rank = np.empty_like(order); rank[np.arange(len(S))[:, None], order] = np.arange(S.shape[1])[None, :]
    nn = order[:, :k]
    rows = []
    qr, qids = fold.query_rows, fold.query_ids
    side, kp, ln = ctx.side, ctx.kp, ctx.lognorm
    kscore, ktop = kpg if kpg is not None else ({}, {})
    S_lda, Kr_l2, Kr_lda, top_kr = rich if rich is not None else (None, None, None, None)
    S_sw, S_fl, rank_sw, Kr_sw, top_sw, top_kr_sw = swm if swm is not None else (None, None, None, None, None, None)
    post_d = {qids[i]: dict(zip(*post[f"{split}|{qids[i]}"])) for i in range(len(qr))} if rich is not None else None
    for i in range(len(qr)):
        li, pi_ = post[f"{split}|{qids[i]}"]
        cand = list(nn[i]) + [j for j in ktop.get(i, []) if j not in set(nn[i])]
        if top_kr is not None:
            cset = set(cand)
            cand += [int(j) for j in top_kr[i] if int(j) not in cset]
        if swm is not None:
            cset = set(cand)
            for j in list(top_sw[i]) + list(top_kr_sw[i]):
                if int(j) not in cset:
                    cand.append(int(j)); cset.add(int(j))
        for j in cand:
            if j == i:
                continue
            lj, pj = post[f"{split}|{qids[j]}"]
            f = ctx.pc.get(qr[i], qr[j]) if ctx.pc is not None else None
            if f is None:
                f = np.full(len(SF), np.nan, np.float32)
            si, sj = side[qr[i]], side[qr[j]]
            kp1 = kscore.get((i, j), 0.0); kp2 = kscore.get((j, i), 0.0)
            r_extra = ()
            if rich is not None:
                di, dj = post_d[qids[i]], post_d[qids[j]]
                dot = sum(v * dj.get(l, 0.0) for l, v in di.items() if l != NEW)
                r_extra = (float(S_lda[i, j]), float(Kr_l2[i, j]), float(Kr_lda[i, j]),
                           float(dot), float(min(kp1, kp2)), float(np.log1p(kp1)))
            bg_extra = ()
            if sift_bg:
                rt = float(f[SF.index("abs_rot")]); sc_ = float(f[SF.index("log_scale")])
                sp_ = float(f[SF.index("spread_min")]); inl_ = float(f[SF.index("inl_aff")])
                isbg = float(np.isfinite(rt) and np.isfinite(sp_) and rt < 0.50 and sp_ < 0.25)
                bg_extra = (rt, sc_, isbg, 0.0 if isbg else inl_)
            bg2_extra = ()
            if sift_bg_v2:
                rt2 = float(f[SF.index("abs_rot")]); sc2 = float(f[SF.index("log_scale")])
                bg2_extra = (float(f[SF.index("inl_h")]), abs(sc2) if np.isfinite(sc2) else np.nan,
                             float(np.isfinite(rt2) and rt2 < 0.15))
            sw_extra = ()
            if swm is not None:
                sw_extra = (float(S_sw[i, j]), float(S_fl[i, j]), int(rank_sw[i, j]), int(rank_sw[j, i]), float(Kr_sw[i, j]))
            rows.append((i, j, S[i, j], float(Zp[i] @ Zp[j]), rank[i, j], rank[j, i],
                         float(si == sj and si in "LR"), float({si, sj} == {"L", "R"}),
                         min(kp[qr[i]], kp[qr[j]]), abs(ln[qr[i]] - ln[qr[j]]),
                         f[SF.index("inl_aff")], f[SF.index("inl_aff_strict")], f[SF.index("n_mutual")],
                         f[SF.index("inl_ratio")], f[SF.index("inl_over_kp")], f[SF.index("spread_min")],
                         float(li[int(np.argmax(pi_))] == lj[int(np.argmax(pj))]), float(min(pi_.max(), pj.max())),
                         int(fold.species == "leopard"))
                        + ((kp1, kp2, ktop[i].index(j) if j in ktop.get(i, []) else 99) if kpg is not None else ())
                        + r_extra + bg_extra + bg2_extra + sw_extra)
    cols = (["i", "j"] + PAIR_FEATS + (KP_FEATS if kpg is not None else []) + (RICH_FEATS if rich is not None else [])
            + (SIFT_BG_FEATS if sift_bg else []) + (SIFT_BG_V2_FEATS if sift_bg_v2 else []) + (SW_FEATS if swm is not None else []))
    return pd.DataFrame(rows, columns=cols)


def _propagate(n, labels_per_q, probs_per_q, edges, w, alpha, iters=30, return_conn=False):
    """Returns list of (labels, probs) after propagation over a union label space."""
    if alpha == 0 or len(edges) == 0:
        out = list(zip(labels_per_q, probs_per_q))
        return (out, np.zeros(n, dtype=bool)) if return_conn else out
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
    return (out, d > 0) if return_conn else out


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
        from .kpgraph import query_graph
        import hashlib, pickle as _pk
        from .io import P as _P
        np_kp = int(tc.get("kp_nprobe", 8))
        use_hnsw = bool(tc.get("kp_hnsw", False))
        for r, sp_, f in folds + tfolds:
            h = hashlib.md5(np.asarray(f.query_rows).tobytes()).hexdigest()[:12]
            sfx = "_hnsw" if use_hnsw else (f"_np{np_kp}" if np_kp != 8 else "")
            cp = _P(cfg, "cache") / "kpgraph" / f"{sp_}_{h}{sfx}.pkl"
            if cp.exists():
                KP[(r, sp_)] = _pk.loads(cp.read_bytes())
            else:
                KP[(r, sp_)] = query_graph(f, ctx, nprobe=np_kp, use_hnsw=use_hnsw); cp.parent.mkdir(parents=True, exist_ok=True); cp.write_bytes(_pk.dumps(KP[(r, sp_)]))
        log(f"[trans] keypoint query graph built for {len(KP)} folds ({time.time() - t0:.0f}s)")
    RICH = {}
    if tc.get("trans_rich", False):
        for r, sp_, f in folds + tfolds:
            RICH[(r, sp_)] = _rich_mats(f, ctx)
        log(f"[trans] rich query-query LDA + k-reciprocal built for {len(RICH)} folds ({time.time() - t0:.0f}s)")
    SWM = {}
    if tc.get("trans_sw", False):
        for r, sp_, f in folds + tfolds:
            SWM[(r, sp_)] = _sw_mats(f, ctx)
        log(f"[trans] sw2152 + fl512 query-query affinity built for {len(SWM)} folds ({time.time() - t0:.0f}s)")
    sift_bg = bool(tc.get("sift_bg", False))
    sift_bg_v2 = bool(tc.get("sift_bg_v2", False))
    PF = PAIR_FEATS + (KP_FEATS if KP else []) + (RICH_FEATS if RICH else []) + (SIFT_BG_FEATS if sift_bg else []) + (SIFT_BG_V2_FEATS if sift_bg_v2 else []) + (SW_FEATS if SWM else [])
    if ctx.pc is not None:
        pairs = []
        for r, sp_, f in folds + tfolds:
            pairs += query_neighbour_pairs(f, ctx, k)
            if KP:
                pairs += [(f.query_rows[i], f.query_rows[j]) for i, js in KP[(r, sp_)][1].items() for j in js]
            if RICH:
                top_kr = RICH[(r, sp_)][3]
                pairs += [(f.query_rows[i], f.query_rows[int(j)]) for i in range(len(top_kr)) for j in top_kr[i]]
            if SWM:
                for i in range(len(SWM[(r, sp_)][4])):
                    for j in SWM[(r, sp_)][4][i]:
                        pairs.append((f.query_rows[i], f.query_rows[int(j)]))
                    for j in SWM[(r, sp_)][5][i]:
                        pairs.append((f.query_rows[i], f.query_rows[int(j)]))
        ctx.pc.ensure(pairs)
    post_cv = _posteriors(cv, qdf, T, beta)
    post_te = _posteriors(test, tq, T, beta)
    # optional: blend another pipeline's probabilities into the prior P0 BEFORE propagation (team-blend experiment)
    import os, pickle as _pk
    if os.environ.get("BLEND_CV"):
        wb = float(os.environ.get("BLEND_W", "0.5"))
        oc = _pk.load(open(os.environ["BLEND_CV"], "rb")); ot = _pk.load(open(os.environ["BLEND_TEST"], "rb"))

        def _mix(post, other):
            for key in post:
                labs, pr = post[key]; o = other(key)
                d = {l: (1 - wb) * p for l, p in zip(labs, pr)}
                for l, p in o.items():
                    d[l] = d.get(l, 0.0) + wb * p
                post[key] = (list(d.keys()), np.array(list(d.values()), float))
        _mix(post_cv, lambda k: oc[int(k.split("|", 1)[0])][0][k.split("|", 1)[1]])
        _mix(post_te, lambda k: ot[k.split("|", 1)[1]])
        log(f"[trans] P0 = {1 - wb:.2f} * own + {wb:.2f} * external probabilities")
    ident = data.meta.identity.values
    ptabs = []
    for r, sp_, f in folds:
        pt = _pair_table(f, ctx, k, post_cv, r, KP.get((r, sp_)), RICH.get((r, sp_)),
                         sift_bg=sift_bg, swm=SWM.get((r, sp_)), sift_bg_v2=sift_bg_v2)
        pt["y"] = (ident[f.query_rows[pt.i.values]] == ident[f.query_rows[pt.j.values]]).astype(int)
        pt["group"] = ident[f.query_rows[pt.i.values]]
        pt["split"] = r; pt["species"] = sp_
        ptabs.append(pt)
    pcv = pd.concat(ptabs, ignore_index=True)
    bag_seeds = tc.get("bag_seeds", None)
    bag_test = bool(tc.get("bag_test", False))
    if bag_test:
        pcv["p_same"], _, pm_bag = oof_predict(pcv, PF, "y", pcv.group.values, kind="lgb", return_models=True, seeds=bag_seeds)
    else:
        pcv["p_same"], _ = oof_predict(pcv, PF, "y", pcv.group.values, kind="lgb", seeds=bag_seeds)
    from sklearn.metrics import roc_auc_score
    log(f"[trans] pair model OOF AUC={roc_auc_score(pcv.y, pcv.p_same):.4f}, "
        f"AUC cos only={roc_auc_score(pcv.y, pcv.cos_l2):.4f}, pairs={len(pcv)} ({time.time() - t0:.0f}s)")
    tt = []
    for r, sp_, f in tfolds:
        pt = _pair_table(f, ctx, k, post_te, r, KP.get((r, sp_)), RICH.get((r, sp_)),
                         sift_bg=sift_bg, swm=SWM.get((r, sp_)), sift_bg_v2=sift_bg_v2)
        pt["split"] = r; pt["species"] = sp_
        tt.append(pt)
    pte = pd.concat(tt, ignore_index=True)
    if bag_test:
        pte["p_same"] = predict_bag(pm_bag, pte[PF].to_numpy(np.float32))
    else:
        pm = fit_full(pcv, PF, "y", pcv.group.values, kind="lgb")
        pte["p_same"] = pm.predict(pte[PF].to_numpy(np.float32))

    conn_only = bool(tc.get("conn_new_scale", False))

    def run_fold(post, ptab, f, r, alpha, tau_e, consensus):
        keys = [f"{r}|{q}" for q in f.query_ids]
        labs = [post[kk][0] for kk in keys]; prs = [post[kk][1] for kk in keys]
        m = ptab.p_same.values > tau_e
        edges = ptab[["i", "j"]].values[m]; w = ptab.p_same.values[m]
        if consensus and len(edges):
            edges, w = _consensus_filter(len(keys), edges, w, tau_e)
        outp, conn = _propagate(len(keys), labs, prs, edges, w, alpha, return_conn=True)
        return keys, outp, conn

    truth_by_key = dict(zip(qdf.split.astype(str) + "|" + qdf.query_id, qdf.truth))
    grid_alpha = tc.get("alphas", [0.0, 0.2, 0.4, 0.6, 0.8])
    grid_tau = tc.get("tau_e", [0.5, 0.7, 0.9])
    scales = np.exp(np.arange(-1.5, 1.51, 0.25))
    nested_tune = bool(tc.get("nested_tune", False))
    outer_picks = {}
    best = {}
    for sp_ in SPECIES:
        res = []
        sp_splits = [r for r, s2, _ in folds if s2 == sp_]
        for alpha in grid_alpha:
            for tau_e in (grid_tau if alpha > 0 else [1.0]):
                aps = {s: [] for s in range(len(scales))}
                for r, s2, f in folds:
                    if s2 != sp_:
                        continue
                    keys, outp, conn = run_fold(post_cv, pcv[(pcv.split == r) & (pcv.species == sp_)], f, r, alpha, tau_e,
                                                tc.get("consensus", True))
                    for si, sc in enumerate(scales):
                        aps[si].append(np.mean([_ap(l, p, truth_by_key[kk], sc if (not conn_only or c) else 1.0)
                                                for kk, (l, p), c in zip(keys, outp, conn)]))
                for si, sc in enumerate(scales):
                    res.append((alpha, tau_e, float(np.log(sc)), float(np.mean(aps[si])), aps[si]))
        df = pd.DataFrame(res, columns=["alpha", "tau_e", "log_new_scale", "mapv", "per_split"])
        off0 = df[df.alpha == 0]; off = off0.loc[off0.mapv.idxmax()]
        if nested_tune and len(sp_splits) > 1:
            ps_mat = np.array(df.per_split.tolist())
            nested_ps, picks = [], []
            for idx_r, r_id in enumerate(sp_splits):
                train_mean = np.delete(ps_mat, idx_r, axis=1).mean(1)
                br = df.iloc[int(np.argmax(train_mean))]
                nested_ps.append(float(ps_mat[int(np.argmax(train_mean)), idx_r]))
                pk = dict(alpha=float(br.alpha), tau_e=float(br.tau_e), log_new_scale=float(br.log_new_scale))
                outer_picks[(r_id, sp_)] = pk
                picks.append(pk)
            b_al = float(np.median([p["alpha"] for p in picks]))
            b_ta = float(np.median([p["tau_e"] for p in picks]))
            b_sc = float(np.median([p["log_new_scale"] for p in picks]))
            best[sp_] = dict(alpha=b_al, tau_e=b_ta, log_new_scale=b_sc,
                             map=float(np.mean(nested_ps)), per_split=nested_ps,
                             map_off=float(off.mapv), per_split_off=list(off.per_split),
                             grid=[(float(x.alpha), float(x.tau_e), float(x.log_new_scale), [float(v) for v in x.per_split])
                                   for x in df.itertuples()])
            log(f"[trans] {sp_} NESTED: consensus alpha={b_al:.2f} tau_e={b_ta:.2f} new_scale=e^{b_sc:.2f} "
                f"(conn_only={conn_only}) nested_MAP={np.mean(nested_ps):.4f} (alpha=0: {off.mapv:.4f})")
        else:
            b = df.loc[df.mapv.idxmax()]
            best[sp_] = dict(alpha=float(b.alpha), tau_e=float(b.tau_e), log_new_scale=float(b.log_new_scale),
                             map=float(b.mapv), per_split=list(b.per_split), map_off=float(off.mapv),
                             per_split_off=list(off.per_split),
                             grid=[(float(x.alpha), float(x.tau_e), float(x.log_new_scale), [float(v) for v in x.per_split])
                                   for x in df.itertuples()])
            log(f"[trans] {sp_}: best alpha={b.alpha} tau_e={b.tau_e} new_scale=e^{b.log_new_scale:.1f} "
                f"(conn_only={conn_only}) MAP={b.mapv:.4f} (alpha=0: {off.mapv:.4f})")
    # export: out-of-fold CV probabilities with each species' chosen alpha / tau_e / new_scale (for team-blend tuning)
    for r, sp_, f in folds:
        bb = outer_picks.get((r, sp_), best[sp_])
        keys, outp, conn = run_fold(post_cv, pcv[(pcv.split == r) & (pcv.species == sp_)], f, r, bb["alpha"], bb["tau_e"],
                                    tc.get("consensus", True))
        sc_b = np.exp(bb["log_new_scale"])
        for kk, (l, p), c in zip(keys, outp, conn):
            ll, pp = list(l), np.array(p, float).copy()
            if NEW in ll:
                pp[ll.index(NEW)] *= (sc_b if (not conn_only or c) else 1.0)
            EXPORT_CV.setdefault(r, {})[kk.split("|", 1)[1]] = dict(zip(ll, pp / pp.sum()))
    # test predictions
    preds = {}
    for r, sp_, f in tfolds:
        bb = best[sp_]
        keys, outp, conn = run_fold(post_te, pte[pte.species == sp_], f, r, bb["alpha"], bb["tau_e"], tc.get("consensus", True))
        sc_b = np.exp(bb["log_new_scale"])
        log(f"[trans] test {sp_}: connected={int(conn.sum())}/{len(conn)} ({conn.mean():.3f})")
        for kk, (l, p), c in zip(keys, outp, conn):
            sc_i = sc_b if (not conn_only or c) else 1.0
            preds[kk.split("|", 1)[1]] = _rank(l, p, sc_i)
            # export for team blending: final label probabilities (same new_scale as the ranking), renormalised
            ll, pp = list(l), np.array(p, float).copy()
            if NEW in ll:
                pp[ll.index(NEW)] *= sc_i
            EXPORT_PROBS[kk.split("|", 1)[1]] = dict(zip(ll, pp / pp.sum()))
    return preds, best, pcv
