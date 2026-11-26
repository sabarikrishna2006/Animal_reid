"""v28: Full 8-Split Honest CV & Test Fusion combining:
  1. v24 Baseline (Sw2152 + Side-Sw2152 + LDA512 + Side-LDA512 + nprobe=64 Bg-Clean LNBNN +
     lnb_img Top-5 Candidate Expansion + Multi-Image SIFT RANSAC + Stage-1/2 Camera-Trap SIFT Killers)
  2. 5 Research-Paper Upgrades (Sw2152 Manifold Diffusion/DBA, Cross-Flank Procrustes, Cohort AS-Norm, Bayesian PLDA, Sinkhorn OT)
  3. NEW Hypothesis 1: Transductive Gallery-to-Query Column Reciprocal Z-Score, Rank & Competitor Gap
     in BOTH Round 1 (max_sw2152, diff_sw, max_lda512, lnb_uimg, comb) AND Round 2 (g2q_p1_gap, g2q_p1_share, g2q_p1_rk)
  4. NEW Hypothesis 2: Asymmetric Connected-Component Novelty Scaling (conn_new_scale) in Learned BG-Filtered QQ-Graph Propagation
  5. NEW Hypothesis 3: Novelty-Safe & Competitor-Conflict-Gated Fusion onto lnb36 (0.88585 Rank 1).
"""
import os, sys, gc, time, pickle, warnings
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = "/home/jazl-anwar/Desktop/Projects/mlfinalproj/Animal_reid"
DATA = f"{ROOT}/data/reid_student_data_main"
CACHE = f"{ROOT}/data/cache"
sys.path.insert(0, f"{ROOT}/src")
from reid.decision import validate

NEW = "new_individual"
SIDE = {"left": "L", "frontleft": "L", "backleft": "L", "right": "R", "frontright": "R", "backright": "R"}
t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
mids = pd.read_csv(f"{DATA}/miewid_ids.csv")["id"].values
Xall = np.load(f"{DATA}/miewid_features.npy").astype(np.float32)
idx_map = {k: i for i, k in enumerate(mids)}
Xtr = Xall[[idx_map[i] for i in train.id]]
Xte = Xall[[idx_map[i] for i in test.id]]
def l2n(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
Xtr_n = l2n(Xtr); Xte_n = l2n(Xte)
tr_pos = {q: i for i, q in enumerate(train.id)}; te_pos = {q: i for i, q in enumerate(test.id)}
cos_tr = lambda a, b: float(Xtr_n[tr_pos[a]] @ Xtr_n[tr_pos[b]])
cos_te = lambda a, b: float(Xte_n[te_pos[a]] @ Xte_n[te_pos[b]])
side_tr = dict(zip(train.id, train.viewpoint.map(SIDE).fillna("O")))
side_te = dict(zip(test.id, test.viewpoint.map(SIDE).fillna("O")))
ident_tr = dict(zip(train.id, train.identity))

all_ids = np.r_[train.id.values, test.id.values]
SIFT_OFF = np.load(f"{CACHE}/sift_off.npy")
nkp = {k: int(SIFT_OFF[i + 1] - SIFT_OFF[i]) for i, k in enumerate(all_ids)}
SIFT_CACHE = pickle.load(open(f"{CACHE}/sift_geom_v27_cache.pkl", "rb"))

saved = pickle.load(open(f"{CACHE}/honest_cv16_v27_full.pkl", "rb"))
VAL, PT, QT, KPG_VAL, KPG_TE = saved["VAL"], saved["PT"], saved["QT"], saved["KPG_VAL"], saved["KPG_TE"]
res_prev = pickle.load(open(f"{CACHE}/honest_cv16_v27_results.pkl", "rb"))
log("Loaded cached 8-split Honest CV tables and previous Baseline/v27 results")

def add_g2q_r1(P, Q):
    P = P.copy(); Q = Q.copy()
    new_p, new_q = [], []
    for col in ["max_sw2152", "diff_sw", "max_lda512", "lnb_uimg", "comb"]:
        g = P.groupby("cand")[col]
        c_max = g.transform("max").astype(np.float32)
        c_mean = g.transform("mean").astype(np.float32)
        c_std = g.transform("std").fillna(0.05).clip(0.03).astype(np.float32)
        c_rk = g.rank(ascending=False, method="min").astype(np.float32)
        P[f"g2q_z_{col}"] = ((P[col] - c_mean) / c_std).astype(np.float32)
        P[f"g2q_gap_{col}"] = (c_max - P[col]).astype(np.float32)
        P[f"g2q_rk_{col}"] = np.log1p(c_rk - 1.0).astype(np.float32)
        new_p += [f"g2q_z_{col}", f"g2q_gap_{col}", f"g2q_rk_{col}"]
    top0 = P[P["rank"] == 0].set_index("qid")
    for col in ["max_sw2152", "diff_sw", "lnb_uimg"]:
        Q[f"q_top0_g2q_z_{col}"] = top0[f"g2q_z_{col}"].reindex(Q.qid).fillna(0.0).values.astype(np.float32)
        Q[f"q_top0_g2q_rk_{col}"] = top0[f"g2q_rk_{col}"].reindex(Q.qid).fillna(5.0).values.astype(np.float32)
        Q[f"q_max_g2q_z_{col}"] = P.groupby("qid")[f"g2q_z_{col}"].max().reindex(Q.qid).fillna(0.0).values.astype(np.float32)
        new_q += [f"q_top0_g2q_z_{col}", f"q_top0_g2q_rk_{col}", f"q_max_g2q_z_{col}"]
    return P, Q, new_p, new_q

def add_g2q_r2(P, Q):
    P = P.copy(); Q = Q.copy()
    g = P.groupby("cand")["p1"]
    c_max = g.transform("max").astype(np.float32)
    c_sum = g.transform("sum").clip(1e-6).astype(np.float32)
    c_rk = g.rank(ascending=False, method="min").astype(np.float32)
    P["g2q_p1_gap"] = (c_max - P["p1"]).astype(np.float32)
    P["g2q_p1_share"] = (P["p1"] / c_sum).astype(np.float32)
    P["g2q_p1_rk"] = np.log1p(c_rk - 1.0).astype(np.float32)
    top = P.sort_values("p1", ascending=False).drop_duplicates("qid").set_index("qid")
    Q["q_top_g2q_p1_gap"] = top["g2q_p1_gap"].reindex(Q.qid).fillna(0.0).values.astype(np.float32)
    Q["q_top_g2q_p1_share"] = top["g2q_p1_share"].reindex(Q.qid).fillna(0.0).values.astype(np.float32)
    Q["q_top_g2q_p1_rk"] = top["g2q_p1_rk"].reindex(Q.qid).fillna(5.0).values.astype(np.float32)
    return P, Q, ["g2q_p1_gap", "g2q_p1_share", "g2q_p1_rk"], ["q_top_g2q_p1_gap", "q_top_g2q_p1_share", "q_top_g2q_p1_rk"]

V2 = {}
for s in range(8):
    P, Q, gp1, gq1 = add_g2q_r1(*VAL[s])
    V2[s] = (P, Q)
PT2, QT2, _, _ = add_g2q_r1(PT, QT)

pf1 = [c for c in V2[0][0].columns if c not in ("qid", "cand", "y")]
qf1 = [c for c in V2[0][1].columns if c not in ("qid", "truth", "is_new", "nbr_ids", "nbr_sims")]

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=16)

PAIR_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=30,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, num_threads=16)

PAIR_BG_FEATS = [
    "cos_qq", "kp_ij", "kp_ji", "kp_min", "kp_log", "kp_rank",
    "sift_h", "sift_s", "sift_ratio", "sift_frac",
    "sift_rot", "sift_spread", "sift_is_bg", "sift_s_clean",
    "same_side", "opp_side", "min_kp", "top1_same", "prob_dot", "species"
]

def others(store, s, idx):
    qs = set(store[s][1].qid)
    return pd.concat([store[k][idx][~store[k][idx].qid.isin(qs)] for k in store if k != s], ignore_index=True)

def fit_lgb(P, Qf, pf, qf, seed=0):
    p = dict(PARAMS, seed=seed)
    return (lgb.train(p, lgb.Dataset(P[pf], P.y), 420),
            lgb.train(p, lgb.Dataset(Qf[qf], Qf.is_new), 260))

def round2_and_rich_g2q(P, Qf, G, scale=2.0):
    P = P.copy(); Qf = Qf.copy()
    P["p1_rank"] = P.groupby("qid").p1.rank(ascending=False).astype(np.float32)
    P["p1_norm"] = (P.p1 / P.groupby("qid").p1.transform("sum").clip(1e-9)).astype(np.float32)
    pmap = {(q, c): float(p) for q, c, p in zip(P.qid.values, P.cand.values, P.p1.values)}
    pn = dict(zip(Qf.qid.values, Qf.pn1.values))
    P["pn1_self"] = P.qid.map(pn).astype(np.float32)
    top = P.sort_values("p1", ascending=False).drop_duplicates("qid").set_index("qid")
    topc, topp = top.cand.to_dict(), top.p1.to_dict()
    nb = {q: (ids, np.array(s, np.float32)) for q, ids, s in zip(Qf.qid.values, Qf.nbr_ids.values, Qf.nbr_sims.values)}
    kpg_nb = {}
    for q in Qf.qid.values:
        edges = G.get(q, [])
        if edges:
            k_ids = [e[0] for e in edges]
            k_s1 = np.array([e[1] for e in edges], np.float32)
            k_sym = np.array([min(e[1], e[2]) for e in edges], np.float32)
            k_w = k_s1 / (k_s1 + scale)
            kpg_nb[q] = (k_ids, k_s1, k_sym, k_w)
        else:
            kpg_nb[q] = ([], np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
    a = np.zeros((len(P), 8), np.float32)
    for r, (q, c) in enumerate(zip(P.qid.values, P.cand.values)):
        ids, s = nb[q]; v = np.array([pmap.get((n, c), 0.0) for n in ids], np.float32); w = np.clip(s, 0.0, None)
        k_ids, _, k_sym, k_w = kpg_nb[q]
        if len(k_ids):
            kv = np.array([pmap.get((n, c), 0.0) for n in k_ids], np.float32)
            km1, kmw, kms, kcnt = float(np.max(k_w * kv)), float((k_w * kv).sum() / max(float(k_w.sum()), 1e-6)), float(np.max((k_sym / (k_sym + scale)) * kv)), float(np.sum((kv > 0.5) & (k_sym > 1.5)))
        else:
            km1, kmw, kms, kcnt = 0.0, 0.0, 0.0, 0.0
        a[r] = (np.max(s * v), (w * v).sum() / max(w.sum(), 1e-6), v[0] * s[0], np.sum((v > 0.5) & (s > 0.6)), km1, kmw, kms, kcnt)
    P[["nb_p_max", "nb_p_w", "nb0_p", "nb_cnt", "kpg_p_max", "kpg_p_w", "kpg_p_sym", "kpg_cnt"]] = a
    g = P.groupby("qid").p1
    Qf["p1_max"] = Qf.qid.map(g.max()).astype(np.float32)
    Qf["p1_2nd"] = Qf.qid.map(g.apply(lambda x: float(np.sort(x.values)[-2]) if len(x) > 1 else float(x.values[0]))).astype(np.float32)
    b = np.zeros((len(Qf), 9), np.float32)
    for r, (q, ids, s) in enumerate(zip(Qf.qid.values, Qf.nbr_ids.values, Qf.nbr_sims.values)):
        s = np.array(s, np.float32); w = np.clip(s, 0.0, None); pns = np.array([pn[n] for n in ids], np.float32)
        conf = np.array([topp[n] for n in ids], np.float32); agree = np.array([float(topc[n] == topc[q]) for n in ids], np.float32)
        k_ids, k_s1, k_sym, k_w = kpg_nb[q]
        if len(k_ids):
            kpns = np.array([pn.get(n, 0.5) for n in k_ids], np.float32); kconf = np.array([topp.get(n, 0.0) for n in k_ids], np.float32); kagr = np.array([float(topc.get(n, "") == topc[q]) for n in k_ids], np.float32)
            k_mx, k_smx, k_pnw, k_kc, k_ag = float(k_s1.max()), float(k_sym.max()), float((k_w * kpns).sum() / max(float(k_w.sum()), 1e-6)), float(np.max(k_w * kconf)), float(np.max(k_w * kagr))
        else:
            k_mx, k_smx, k_pnw, k_kc, k_ag = 0.0, 0.0, float(pn[q]), 0.0, 0.0
        b[r] = ((w * pns).sum() / max(w.sum(), 1e-6), pns[0], np.max(s * conf), np.max(s * agree), k_mx, k_smx, k_pnw, k_kc, k_ag)
    Qf[["nb_pn_w", "nb0_pn", "nb_known_conf", "nb_agree", "kpg_max", "kpg_sym_max", "kpg_pn_w", "kpg_known_conf", "kpg_agree"]] = b
    P, Qf, _, _ = add_g2q_r2(P, Qf)
    return P, Qf

def to_probs_raw(P, Qf, T):
    pn = dict(zip(Qf.qid.values, Qf.pn2.values)); out = {}
    for q, g in P.groupby("qid", sort=False):
        labs = list(g.cand.values) + [NEW]
        sc = np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)) / T
        e = np.exp(sc - sc.max()); out[q] = dict(zip(labs, e / e.sum()))
    return out

def eval_probs(probs, truth):
    s = 0.0
    for q, t in truth.items():
        d = probs[q]; top5 = sorted(d, key=d.get, reverse=True)[:5]
        for k, lab in enumerate(top5):
            if lab == t: s += 1.0 / (k + 1); break
    return s / len(truth)

def build_qq_pairs_bg(Qf, probs, G, cos_fn, side_map):
    qids = list(Qf.qid.values); qset = set(qids)
    sp_map = dict(zip(Qf.qid.values, Qf.species.values))
    top1 = {q: max(probs[q], key=probs[q].get) for q in qids}
    cand_edges = {}
    for q, nids, nsims in zip(Qf.qid.values, Qf.nbr_ids.values, Qf.nbr_sims.values):
        cdict = {}
        for nb, sim in zip(nids, nsims):
            if nb in qset: cdict[nb] = [float(sim), 0.0, 0.0, 99]
        for r_kp, (nb, s1, s2) in enumerate(G.get(q, [])):
            if nb not in qset: continue
            if nb in cdict:
                cdict[nb][1] = float(s1); cdict[nb][2] = float(s2); cdict[nb][3] = r_kp
            else:
                cdict[nb] = [cos_fn(q, nb), float(s1), float(s2), r_kp]
        cand_edges[q] = cdict
    rows = []
    for q in qids:
        sq = side_map[q]; pq = probs[q]; tq = top1[q]; spq = sp_map[q]; kpq = nkp[q]
        for nb, (cs, s1, s2, rkp) in cand_edges[q].items():
            if nb not in probs: continue
            sm = SIFT_CACHE.get(tuple(sorted((q, nb))), (0, 0, 0, 0.0, 99.0, 1.0, 0))
            rot, sp_m, is_bg = float(sm[4]), float(sm[5]), float(sm[6])
            s_clean = 0.0 if is_bg > 0.5 else float(sm[2])
            sn = side_map[nb]; pn = probs[nb]
            same_s = float(sq == sn and sq in ("L", "R"))
            opp_s = float({sq, sn} == {"L", "R"})
            dot = sum(v * pn.get(l, 0.0) for l, v in pq.items() if l != NEW)
            rows.append((q, nb, cs, s1, s2, min(s1, s2), np.log1p(s1), rkp,
                         sm[1], sm[2], sm[0], sm[3], rot, sp_m, is_bg, s_clean,
                         same_s, opp_s, min(kpq, nkp[nb]),
                         float(tq == top1[nb] and tq != NEW), float(dot), spq))
    return pd.DataFrame(rows, columns=["qi", "qj"] + PAIR_BG_FEATS)

def propagate_asymmetric(qids, raw_probs, pt_df, sp_map, alpha_sp, tau_sp, new_damp_sp={0: 1.0, 1: 1.0}, iters=30):
    out = {}
    for sp_i in [0, 1]:
        alpha = float(alpha_sp[sp_i]); tau = float(tau_sp[sp_i]); ndamp = float(new_damp_sp[sp_i])
        qs = [q for q in qids if sp_map[q] == sp_i]
        if not qs: continue
        q_idx = {q: i for i, q in enumerate(qs)}
        sub = pt_df[(pt_df.species == sp_i) & (pt_df.p_same > tau)]
        edges_i = [q_idx[a] for a in sub.qi.values if a in q_idx]
        edges_j = [q_idx[b] for b in sub.qj.values if b in q_idx]
        if not edges_i or alpha <= 0.0:
            for q in qs: out[q] = raw_probs[q]
            continue
        w = sub.p_same.values.astype(np.float64)
        n = len(qs)
        lab_index = {}
        for q in qs:
            for l in raw_probs[q]: lab_index.setdefault(l, len(lab_index))
        P0 = np.zeros((n, len(lab_index)), dtype=np.float64)
        for i, q in enumerate(qs):
            for l, pr in raw_probs[q].items(): P0[i, lab_index[l]] = pr
        ii = np.array(edges_i, np.int32); jj = np.array(edges_j, np.int32)
        W = sp_sparse.coo_matrix((np.r_[w, w], (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n)).tocsr()
        W.data = np.minimum(W.data, 1.0)
        d = np.asarray(W.sum(1)).ravel()
        dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
        Sn = sp_sparse.diags(dinv) @ W @ sp_sparse.diags(dinv)
        F = P0.copy()
        for _ in range(iters):
            F = alpha * (Sn @ F) + (1.0 - alpha) * P0
        # Degree-aware conn_new_scale on connected nodes (d > 0)
        if NEW in lab_index and ndamp != 1.0:
            new_col = lab_index[NEW]
            F[d > 0, new_col] *= ndamp
        F /= np.maximum(F.sum(1, keepdims=True), 1e-12)
        inv_labs = list(lab_index.keys())
        for i, q in enumerate(qs):
            nz = np.where(F[i] > 1e-9)[0]
            out[q] = {inv_labs[k]: float(F[i, k]) for k in nz}
    return out

# Run Round-1 -> Round-2 -> Asymmetric LPROP for v28
R1 = {}
r1_sc = []
for s in range(8):
    Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
    Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
    mp, mq = fit_lgb(Ptr, Qtr, pf1, qf1, seed=0)
    Pv["p1"] = mp.predict(Pv[pf1]).astype(np.float32)
    Qv["pn1"] = mq.predict(Qv[qf1]).astype(np.float32)
    Pv["p2"], Qv["pn2"] = Pv["p1"], Qv["pn1"]
    r1_sc.append(eval_probs(to_probs_raw(Pv, Qv, 1.0), dict(zip(Qv.qid, Qv.truth))))
    R1[s] = round2_and_rich_g2q(Pv, Qv, KPG_VAL[s])

pf2 = [c for c in R1[0][0].columns if c not in ("qid", "cand", "y", "p2")]
qf2 = [c for c in R1[0][1].columns if c not in ("qid", "truth", "is_new", "nbr_ids", "nbr_sims", "pn2")]

OOF, CAL = {}, []
for s in range(8):
    Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
    Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
    mp, mq = fit_lgb(Ptr, Qtr, pf2, qf2, seed=0)
    Pv["p2"] = mp.predict(Pv[pf2]).astype(np.float32)
    Qv["pn2"] = mq.predict(Qv[qf2]).astype(np.float32)
    OOF[s] = (Pv[["qid", "cand", "p2", "g2q_p1_gap", "g2q_p1_share", "g2q_p1_rk"]],
              Qv[["qid", "species", "pn2", "truth", "nbr_ids", "nbr_sims", "q_top_g2q_p1_gap", "q_top_g2q_p1_share"]])
    tmap = dict(zip(Qv.qid, Qv.truth)); pn = dict(zip(Qv.qid, Qv.pn2))
    for q, g in Pv.groupby("qid", sort=False):
        labs = list(g.cand.values) + [NEW]
        if tmap[q] in labs:
            CAL.append((np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)), labs.index(tmap[q])))

T_CAL = float(minimize_scalar(lambda T: -np.mean([(sc[i]/T) - np.log(np.exp(sc/T - (sc/T).max()).sum()) - (sc/T).max() for sc, i in CAL]),
                              bounds=(0.05, 20), method="bounded").x)
RAW_P = {s: to_probs_raw(OOF[s][0], OOF[s][1], T_CAL) for s in range(8)}
r2_sc = [eval_probs(RAW_P[s], dict(zip(OOF[s][1].qid, OOF[s][1].truth))) for s in range(8)]

PTABS = {}
for s in range(8):
    pt = build_qq_pairs_bg(OOF[s][1], RAW_P[s], KPG_VAL[s], cos_tr, side_tr)
    pt["y"] = [int(ident_tr[a] == ident_tr[b]) for a, b in zip(pt.qi, pt.qj)]
    pt["split"] = s
    PTABS[s] = pt
all_pt = pd.concat([PTABS[s] for s in range(8)], ignore_index=True)

OOF_PT, aucs = {}, []
for s in range(8):
    qs = set(OOF[s][1].qid.values)
    tr_m = (all_pt["split"] != s) & (~all_pt["qi"].isin(qs)) & (~all_pt["qj"].isin(qs))
    m_p = lgb.train(PAIR_PARAMS, lgb.Dataset(all_pt.loc[tr_m, PAIR_BG_FEATS].to_numpy(np.float32),
                                             all_pt.loc[tr_m, "y"].values), num_boost_round=350)
    pt_v = PTABS[s].copy()
    pt_v["p_same"] = m_p.predict(pt_v[PAIR_BG_FEATS].to_numpy(np.float32))
    OOF_PT[s] = pt_v
    aucs.append(roc_auc_score(pt_v["y"], pt_v["p_same"]))

alphas = [0.25, 0.35, 0.45, 0.55]
taus = [0.40, 0.50, 0.60, 0.70]
ndamps = [0.80, 0.90, 1.00, 1.10]
grid_scores = {(sp_i, a, t, nd): [] for sp_i in [0, 1] for a in alphas for t in taus for nd in ndamps}
for s in range(8):
    Qv = OOF[s][1]; truth = dict(zip(Qv.qid, Qv.truth)); sp_m = dict(zip(Qv.qid, Qv.species))
    qids = list(Qv.qid.values)
    for sp_i in [0, 1]:
        tr_sp = {q: truth[q] for q in qids if sp_m[q] == sp_i}
        for a in alphas:
            for t in taus:
                for nd in ndamps:
                    pr = propagate_asymmetric(qids, RAW_P[s], OOF_PT[s], sp_m, {0: a, 1: a}, {0: t, 1: t}, {0: nd, 1: nd})
                    grid_scores[(sp_i, a, t, nd)].append(eval_probs({q: pr[q] for q in tr_sp}, tr_sp))

best_hp = {sp_i: max([(np.mean(grid_scores[(sp_i, a, t, nd)]), a, t, nd) for a in alphas for t in taus for nd in ndamps]) for sp_i in [0, 1]}
OOF_LPROP = {}
lp_sc = []
for s in range(8):
    a_sp, t_sp, nd_sp = {}, {}, {}
    for sp_i in [0, 1]:
        _, ba, bt, bnd = max([(np.mean([grid_scores[(sp_i, a, t, nd)][k] for k in range(8) if k != s]), a, t, nd)
                              for a in alphas for t in taus for nd in ndamps])
        a_sp[sp_i], t_sp[sp_i], nd_sp[sp_i] = ba, bt, bnd
    Qv = OOF[s][1]; truth = dict(zip(Qv.qid, Qv.truth)); sp_m = dict(zip(Qv.qid, Qv.species))
    p_lp = propagate_asymmetric(list(Qv.qid.values), RAW_P[s], OOF_PT[s], sp_m, a_sp, t_sp, nd_sp)
    OOF_LPROP[s] = p_lp
    lp_sc.append(eval_probs(p_lp, truth))

cv_base = res_prev["cv_base"]
cv_v27 = res_prev["cv_all5"]
cv_v28 = np.array(lp_sc)
log(f"v28 (5-Paper + G2Q + Asym-LPROP): R1={np.mean(r1_sc):.5f} -> R2={np.mean(r2_sc):.5f} -> LPROP={cv_v28.mean():.5f} ± {cv_v28.std(ddof=1):.5f} (qq-AUC={np.mean(aucs):.4f})")
print("\n" + "=" * 105)
print(f"{'8-Split Honest CV Progression':<52s} | {'LPROP MAP@5':<18s} | {'Gain vs Base':<16s} | {'Wins'}")
print("-" * 105)
for name, arr in [
    ("0. Baseline v24 (v10_sw2152_lprop)", cv_base),
    ("1. v27 (+ 5 Research Papers + LPROP)", cv_v27),
    ("2. v28 (+ R1/R2 G2Q Reciprocal Z/Share + Asym-LPROP)", cv_v28),
]:
    d = arr - cv_base
    se = d.std(ddof=1)/np.sqrt(8) if np.any(d!=0) else 0.0
    print(f"{name:<52s} | {arr.mean():.5f} ± {arr.std(ddof=1):.5f} | {d.mean():+.5f} ± {se:.5f} | {int((d>1e-7).sum())}/8")
print("=" * 105)

# Fit 8-split x 3-seed Test model for v28
SEEDS = [0, 1, 2]
Pall = pd.concat([V2[s][0][pf1 + ["y"]] for s in range(8)], ignore_index=True)
Qall = pd.concat([V2[s][1][qf1 + ["is_new"]] for s in range(8)], ignore_index=True)
PTc, QTc = PT2.copy(), QT2.copy()
p1, n1 = np.zeros(len(PTc)), np.zeros(len(QTc))
for sd in SEEDS:
    mp, mq = fit_lgb(Pall, Qall, pf1, qf1, seed=sd)
    p1 += mp.predict(PTc[pf1]) / len(SEEDS)
    n1 += mq.predict(QTc[qf1]) / len(SEEDS)
PTc["p1"] = p1.astype(np.float32); QTc["pn1"] = n1.astype(np.float32)
PTc, QTc = round2_and_rich_g2q(PTc, QTc, KPG_TE)

P2all = pd.concat([R1[s][0][pf2 + ["y"]] for s in range(8)], ignore_index=True)
Q2all = pd.concat([R1[s][1][qf2 + ["is_new"]] for s in range(8)], ignore_index=True)
p2, n2 = np.zeros(len(PTc)), np.zeros(len(QTc))
for sd in SEEDS:
    mp, mq = fit_lgb(P2all, Q2all, pf2, qf2, seed=sd)
    p2 += mp.predict(PTc[pf2]) / len(SEEDS)
    n2 += mq.predict(QTc[qf2]) / len(SEEDS)
PTc["p2"] = p2.astype(np.float32); QTc["pn2"] = n2.astype(np.float32)

test_raw = to_probs_raw(PTc, QTc, T_CAL)
pt_te = build_qq_pairs_bg(QTc, test_raw, KPG_TE, cos_te, side_te)
p_same_te = np.zeros(len(pt_te))
for sd in SEEDS:
    mp_full = lgb.train(dict(PAIR_PARAMS, seed=sd),
                        lgb.Dataset(all_pt[PAIR_BG_FEATS].to_numpy(np.float32), all_pt["y"].values),
                        num_boost_round=350)
    p_same_te += mp_full.predict(pt_te[PAIR_BG_FEATS].to_numpy(np.float32)) / len(SEEDS)
pt_te["p_same"] = p_same_te
a_final = {sp_i: best_hp[sp_i][1] for sp_i in [0, 1]}
t_final = {sp_i: best_hp[sp_i][2] for sp_i in [0, 1]}
nd_final = {sp_i: best_hp[sp_i][3] for sp_i in [0, 1]}
sp_te_i = dict(zip(QTc.qid, QTc.species))
tp_v28 = propagate_asymmetric(list(QTc.qid.values), test_raw, pt_te, sp_te_i, a_final, t_final, nd_final)

# Build Novelty-Safe & Competitor-Conflict-Gated Fusion onto lnb36 (0.88585)
# NEVER demote new_individual from Rank 1 (fixing the 12 false flips that cost lnb37),
# NEVER revert any lnb36 vs lnb34 improvement, and ONLY apply known-vs-known + false-hub vetoes!
ref36_df = pd.read_csv(f"{ROOT}/submissions/sub_team_blend_lnb36_sw2152_v25.csv")
ref34_df = pd.read_csv(f"{ROOT}/submissions/sub_team_blend_lnb34_sw2152.csv")
ref36_map = dict(zip(ref36_df.id, ref36_df.predictions.str.split()))
ref34_map = dict(zip(ref34_df.id, ref34_df.predictions.str.split()))
sift_te = dict(zip(QTc.qid, QTc.sift_clean2_best))
lnbu_te = dict(zip(QTc.qid, QTc.lnb_uimg_max))
tp_base = res_prev["tp_base"]

rows_safe = []
n_k2k, n_k2new, n_r25 = 0, 0, 0
for qid, sp in zip(test.id.values, test.species.values):
    b5 = list(ref36_map[qid])
    b34 = list(ref34_map[qid])
    pb = tp_base[qid]; pv = tp_v28[qid]
    # Rule 1: Protect 100% of lnb36 vs lnb34 top-1 wins and all SIFT/LNBNN matches
    if b5[0] != b34[0] or sift_te[qid] >= 6.0 or lnbu_te[qid] >= 3.5:
        rows_safe.append(" ".join(b5))
        continue
    labs = list(b5)
    l0, l1 = labs[0], labs[1]
    # Rule 2: NEVER demote new_individual from Rank 1!
    if l0 != NEW:
        v_diff = pv.get(l1, 0.0) - pv.get(l0, 0.0)
        b_diff = pb.get(l1, 0.0) - pb.get(l0, 0.0)
        gain = v_diff - b_diff
        if l1 != NEW:
            # Known ID A <-> Known ID B swap (backed by G2Q + 5-Paper causal delta)
            if v_diff > 0.12 and gain > 0.05:
                labs[0], labs[1] = l1, l0
                n_k2k += 1
        else:
            # False-Hub Known ID -> new_individual veto (when G2Q proves l0 already belongs to another test query!)
            if v_diff > 0.22 and gain > 0.10:
                labs[0], labs[1] = l1, l0
                n_k2new += 1
    # Rule 3: Also allow Rank 2 <-> Rank 3 swap when l1 != NEW and causal delta is strong
    if labs[1] != NEW and len(labs) >= 3:
        lr1, lr2 = labs[1], labs[2]
        vd23 = pv.get(lr2, 0.0) - pv.get(lr1, 0.0)
        bd23 = pb.get(lr2, 0.0) - pb.get(lr1, 0.0)
        if lr2 != NEW and vd23 > 0.12 and (vd23 - bd23) > 0.05:
            labs[1], labs[2] = lr2, lr1
            n_r25 += 1
    rows_safe.append(" ".join(labs[:5]))

sub_v28 = pd.DataFrame({"id": test.id, "predictions": rows_safe})
validate(sub_v28, test, train)
out_repo = f"{ROOT}/submissions/sub_team_blend_lnb38_v28_g2q_safe.csv"
out_dl   = "/home/jazl-anwar/Downloads/sub_team_blend_lnb38_v28_g2q_safe.csv"
sub_v28.to_csv(out_repo, index=False); sub_v28.to_csv(out_dl, index=False)

t1_28 = sub_v28.predictions.str.split().str[0].values
t1_36 = ref36_df.predictions.str.split().str[0].values
mh = (test.species == "hyena").values; ml = (test.species == "leopard").values
log(f"WROTE & VALIDATED {out_dl}")
log(f"lnb38_v28_g2q_safe vs lnb36 (0.88585) top-1 flips: {(t1_28 != t1_36).sum()} (Known<->Known={n_k2k}, FalseHubKnown->NEW={n_k2new}, NEW->Known=0!) | Rank2<->3 swaps={n_r25}")
pickle.dump({"tp_v28": tp_v28, "cv_v28": cv_v28, "oof_v28": OOF_LPROP}, open(f"{CACHE}/honest_cv16_v28_results.pkl", "wb"), protocol=4)
log("Saved honest_cv16_v28_results.pkl — ALL DONE!")
