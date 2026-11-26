"""Upgrade Dev's winning dev_v10_fullbg_lprop (lnb30, LB 0.88403) with our zero-leakage
gallery-only PCA(512)->LDA(0.5) + Side-Conditioned PCA(512)->LDA(0.3) + r_g_diff hubness features.
1. Loads cached dev_v10_fullbg_v2.pkl (16 splits) and adds 8 strictly gallery-only LDA512/Side-LDA features.
2. Runs leak-free Round-1 OOF (16 splits) -> Round-2 OOF (3 seeds on splits 0-7, 1 seed on 8-15).
3. Runs strictly nested leave-one-split-out learned bg-filtered propagation (0.9777 AUC qq-pair model).
4. Evaluates head-to-head against lnb30 (dev_v10_fullbg_lprop) on 8-split Honest CV & Shared Blend CV.
5. Fits full 16-split 3-seed test model + learned prop -> exports test_probs_dev_v10_lda512_lprop.pkl."""
import os, gc, pickle, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

from core import load, l2n, val_splits, NEW
from features import SIDE
import model2
from exp_kpg import others
from exp_kpg_v2 import rich_feats
from exp_blend_check import to_probs_raw, eval_probs, blend_dict
from build_dev_v10 import load_memmap_sift, FastSiftCache, SEEDS
from make_dev_bgclean import ScorerDir
from make_dev_v10_fullbg import PF1_ALL, QF1_ALL, PF2_ALL, QF2_ALL, upgrade_table
from make_dev_v10_lprop import PAIR_BG_FEATS, build_qq_pairs_bg
from exp_dev_learned_prop import propagate_learned, PAIR_PARAMS

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")

LDA512_P = ["max_lda512", "max_slda512", "gap_lda512", "gap_slda512"]
LDA512_Q = ["q_lda512_max", "q_lda512_gap", "q_slda512_max", "q_slda512_gap"]

PF1_L = PF1_ALL + LDA512_P
QF1_L = QF1_ALL + LDA512_Q
PF2_L = PF2_ALL + LDA512_P
QF2_L = QF2_ALL + LDA512_Q


def _rg_diff(Z, glab, K=10):
    """100% gallery-only different-identity hubness: mean top-K cosine to gallery images of OTHER identities."""
    Sg = Z @ Z.T
    same = (glab[:, None] == glab[None, :])
    Sg = np.where(same, -2.0, Sg)
    k = min(K, Sg.shape[1] - 1)
    if k <= 0:
        return np.zeros(len(Z), dtype=np.float32)
    return np.partition(Sg, -k, axis=1)[:, -k:].mean(1).astype(np.float32)


def add_lda512_feats(P, Q, gal, Xg, qry, Xq):
    """Attach 100% gallery-only PCA(512)->LDA(0.5) & Side-Conditioned PCA(512)->LDA(0.3) + r_g_diff hubness."""
    q_info = {}
    Xg_n = l2n(Xg); Xq_n = l2n(Xq)
    for sp in ["hyena", "leopard"]:
        gm = (gal.species == sp).values; qm = (qry.species == sp).values
        G = gal[gm].reset_index(drop=True); Qdf = qry[qm].reset_index(drop=True)
        Zg0 = Xg_n[gm]; Zq0 = Xq_n[qm]
        glab = G.identity.values
        gside = G.viewpoint.map(SIDE).fillna("O").values
        glab_side = np.array([f"{a}|{b}" for a, b in zip(glab, gside)])

        m0 = Zg0.mean(0)
        nc = min(512, len(Zg0) - 1, Zg0.shape[1])
        pca = PCA(nc, random_state=0).fit(Zg0 - m0)
        g5, q5 = pca.transform(Zg0 - m0), pca.transform(Zq0 - m0)

        l5 = LDA(solver="eigen", shrinkage=0.5).fit(g5, glab)
        Zg_l, Zq_l = l2n(l5.transform(g5)), l2n(l5.transform(q5))
        ls = LDA(solver="eigen", shrinkage=0.3).fit(g5, glab_side)
        Zg_s, Zq_s = l2n(ls.transform(g5)), l2n(ls.transform(q5))

        rg_l = _rg_diff(Zg_l, glab, K=10)
        rg_s = _rg_diff(Zg_s, glab, K=10)
        Sl = (Zq_l @ Zg_l.T) - 0.50 * rg_l[None, :]
        Ss = (Zq_s @ Zg_s.T) - 0.75 * rg_s[None, :]

        names, inv = np.unique(glab, return_inverse=True)
        nid = len(names); nq = len(Qdf)
        members = [np.where(inv == j)[0] for j in range(nid)]
        Il = np.full((nq, nid), -9.0, np.float32); Is = np.full((nq, nid), -9.0, np.float32)
        for j, cols in enumerate(members):
            Il[:, j] = Sl[:, cols].max(1)
            Is[:, j] = Ss[:, cols].max(1)
        top2_l = -np.sort(-Il, axis=1)[:, :2]
        top2_s = -np.sort(-Is, axis=1)[:, :2]
        id_pos = {n: j for j, n in enumerate(names)}
        for i, qid in enumerate(Qdf.id.values):
            m1 = float(top2_l[i, 0]); m2 = float(top2_l[i, 1]) if nid > 1 else m1
            s1 = float(top2_s[i, 0]); s2 = float(top2_s[i, 1]) if nid > 1 else s1
            q_info[qid] = (id_pos, Il[i], Is[i], m1, m1 - m2, s1, s1 - s2)

    P = P.copy(); Q = Q.copy()
    p_rows = []
    for q, c in zip(P.qid.values, P.cand.values):
        id_pos, il, is_, m1, mg, s1, sg = q_info[q]
        j = id_pos.get(c, -1)
        if j >= 0:
            vl, vs = float(il[j]), float(is_[j])
            p_rows.append((vl, vs, m1 - vl, s1 - vs))
        else:
            p_rows.append((-9.0, -9.0, 9.0, 9.0))
    P[LDA512_P] = np.array(p_rows, dtype=np.float32)

    q_rows = []
    for q in Q.qid.values:
        _, _, _, m1, mg, s1, sg = q_info[q]
        q_rows.append((m1, mg, s1, sg))
    Q[LDA512_Q] = np.array(q_rows, dtype=np.float32)
    return P, Q


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    sp_te = dict(zip(te.id, te.species))
    splits = val_splits(tr, l2n(Xtr), range(16))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}
    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    sc_raw = pickle.load(open(f"{H}/sift_pair_cache.pkl", "rb"))
    sc_bg = pickle.load(open(f"{H}/sift_bg_cache.pkl", "rb"))

    # 1. Attach 100% gallery-only LDA512 + Side-LDA to cached dev_v10_fullbg_v2.pkl
    v2_l_path = f"{H}/dev_v10_lda512_v2.pkl"
    if os.path.exists(v2_l_path):
        V2 = pickle.load(open(v2_l_path, "rb"))
        print(f"Loaded cached {v2_l_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        V2_base = pickle.load(open(f"{H}/dev_v10_fullbg_v2.pkl", "rb"))
        V2 = {}
        for s, (gal, q, truth) in enumerate(splits):
            P0, Q0 = V2_base[s]
            V2[s] = add_lda512_feats(P0, Q0, tr[gal].reset_index(drop=True), Xtr[gal],
                                     tr[q].reset_index(drop=True), Xtr[q])
            if (s + 1) % 4 == 0:
                print(f"  attached lda512 to {s+1}/16 splits ({time.time()-t0:.0f}s)", flush=True)
        del V2_base; gc.collect()
        pickle.dump(V2, open(v2_l_path, "wb"), protocol=4)
        print(f"Cached {v2_l_path} ({time.time()-t0:.0f}s)", flush=True)

    # 2. Leak-free Round-1 OOF (16 splits)
    r1_l_path = f"{H}/dev_v10_lda512_r1.pkl"
    if os.path.exists(r1_l_path):
        R1 = pickle.load(open(r1_l_path, "rb"))
        print(f"Loaded cached {r1_l_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        R1, r1_scores = {}, []
        for s in range(16):
            Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
            Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
            mp, mq = model2.fit(Ptr, Qtr, PF1_L, QF1_L, seed=0)
            Pv["p1"] = mp.predict(Pv[PF1_L]); Qv["pn1"] = mq.predict(Qv[QF1_L])
            if s < 8:
                r1_scores.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
            Pv, Qv = model2.round2_feats(Pv, Qv)
            R1[s] = rich_feats(Pv, Qv, G[s], scale=2.0)
        pickle.dump(R1, open(r1_l_path, "wb"), protocol=4)
        print(f"R1 MEAN (splits 0-7): {np.mean(r1_scores):.4f} (v10_fullbg was 0.8732) ({time.time()-t0:.0f}s)", flush=True)

    # 3. Leak-free Round-2 OOF (splits 0-7: 3 seeds; splits 8-15: 1 seed)
    OOF, CAL = {}, []
    for s in range(16):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        sds = SEEDS if s < 8 else [0]
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in sds:
            mp, mq = model2.fit(Ptr, Qtr, PF2_L, QF2_L, seed=sd)
            p2 += mp.predict(Pv[PF2_L]) / len(sds)
            n2 += mq.predict(Qv[QF2_L]) / len(sds)
        Pv["p2"] = p2; Qv["pn2"] = n2
        OOF[s] = (Pv[["qid", "cand", "p2"]], Qv[["qid", "species", "pn2", "truth", "nbr_ids", "nbr_sims"]])
        if s < 8:
            tmap = dict(zip(Qv.qid, Qv.truth)); pn = dict(zip(Qv.qid, Qv.pn2))
            for q, g in Pv.groupby("qid", sort=False):
                labs = list(g.cand.values) + [NEW]
                if tmap[q] in labs:
                    CAL.append((np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)), labs.index(tmap[q])))

    def nll(T):
        return -np.mean([(sc[i] / T) - np.log(np.exp(sc / T - (sc / T).max()).sum()) - (sc / T).max() for sc, i in CAL])
    T_CAL = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    RAW_P = {s: to_probs_raw(OOF[s][0], OOF[s][1], T_CAL) for s in range(16)}
    print(f"Built OOF raw P0 (T_CAL={T_CAL:.3f}) in {time.time()-t0:.0f}s", flush=True)

    # 4. Learned bg-filtered qq-pair propagation (strictly nested OOF)
    D, XY, off, pos, nkp = load_memmap_sift()
    SC = FastSiftCache(D, XY, off, pos)
    side_tr = dict(zip(tr.id, tr.viewpoint.map(SIDE).fillna("O")))
    side_te = dict(zip(te.id, te.viewpoint.map(SIDE).fillna("O")))
    ident_tr = dict(zip(tr.id, tr.identity))

    Xtr_n = l2n(Xtr); Xte_n = l2n(Xte)
    tr_pos = {q: i for i, q in enumerate(tr.id)}; te_pos = {q: i for i, q in enumerate(te.id)}
    cos_tr = lambda a, b: float(Xtr_n[tr_pos[a]] @ Xtr_n[tr_pos[b]])
    cos_te = lambda a, b: float(Xte_n[te_pos[a]] @ Xte_n[te_pos[b]])

    PTABS = {}
    for s in range(16):
        pt = build_qq_pairs_bg(OOF[s][1], RAW_P[s], G[s], cos_tr, side_tr, nkp, SC, sc_bg)
        pt["y"] = [int(ident_tr[a] == ident_tr[b]) for a, b in zip(pt.qi, pt.qj)]
        pt["split"] = s
        PTABS[s] = pt

    all_pt = pd.concat([PTABS[s] for s in range(16)], ignore_index=True)
    OOF_PT, aucs = {}, []
    for s in range(8):
        qs = set(OOF[s][1].qid.values)
        tr_m = (all_pt["split"] != s) & (~all_pt["qi"].isin(qs)) & (~all_pt["qj"].isin(qs))
        m_p = lgb.train(PAIR_PARAMS, lgb.Dataset(all_pt.loc[tr_m, PAIR_BG_FEATS].to_numpy(np.float32), all_pt.loc[tr_m, "y"].values), num_boost_round=350)
        pt_v = PTABS[s].copy()
        pt_v["p_same"] = m_p.predict(pt_v[PAIR_BG_FEATS].to_numpy(np.float32))
        OOF_PT[s] = pt_v
        aucs.append(roc_auc_score(pt_v["y"], pt_v["p_same"]))
    print(f"Leak-free qq-pair OOF AUC (splits 0-7): {np.mean(aucs):.4f} ({time.time()-t0:.0f}s)", flush=True)

    alphas = [0.25, 0.35, 0.45, 0.55]
    taus = [0.40, 0.50, 0.60, 0.70]
    grid_scores = {(sp_i, a, t): [] for sp_i in [0, 1] for a in alphas for t in taus}
    for s in range(8):
        Qv = OOF[s][1]; truth = dict(zip(Qv.qid, Qv.truth)); sp_m = dict(zip(Qv.qid, Qv.species))
        qids = list(Qv.qid.values)
        for sp_i in [0, 1]:
            tr_sp = {q: truth[q] for q in qids if sp_m[q] == sp_i}
            for a in alphas:
                for t in taus:
                    pr = propagate_learned(qids, RAW_P[s], OOF_PT[s], sp_m, {0: a, 1: a}, {0: t, 1: t})
                    grid_scores[(sp_i, a, t)].append(eval_probs({q: pr[q] for q in tr_sp}, tr_sp))

    best_hp = {}
    for sp_i in [0, 1]:
        best_hp[sp_i] = max([(np.mean(grid_scores[(sp_i, a, t)]), a, t) for a in alphas for t in taus])
        print(f"  species {sp_i}: best alpha={best_hp[sp_i][1]}, tau_e={best_hp[sp_i][2]} -> sp_MAP={best_hp[sp_i][0]:.5f}", flush=True)

    # Head-to-head strictly nested 8-split Honest CV vs lnb30 (dev_v10_fullbg_lprop)
    SAB = pickle.load(open(SAB_CV, "rb"))
    O1 = pickle.load(open(f"{H}/oof_probs_dev_bgclean_prop.pkl", "rb"))
    O2 = pickle.load(open(f"{H}/oof_probs_dev_bgclean2_prop.pkl", "rb"))
    O3 = pickle.load(open(f"{H}/oof_probs_dev_bgclean3_prop.pkl", "rb"))
    OL_OLD = pickle.load(open(f"{H}/oof_probs_dev_v10_fullbg_lprop.pkl", "rb"))
    OOF_LPROP = {}
    s_old, b_old, s_new, b_new = [], [], [], []
    m_old, bm_old, m_new, bm_new, m_both, bm_both = [], [], [], [], [], []
    for s in range(8):
        a_sp, t_sp = {}, {}
        for sp_i in [0, 1]:
            _, ba, bt = max([(np.mean([grid_scores[(sp_i, a, t)][k] for k in range(8) if k != s]), a, t)
                             for a in alphas for t in taus])
            a_sp[sp_i], t_sp[sp_i] = ba, bt
        Qv = OOF[s][1]; truth = dict(zip(Qv.qid, Qv.truth)); sp_m = dict(zip(Qv.qid, Qv.species))
        p_lp = propagate_learned(list(Qv.qid.values), RAW_P[s], OOF_PT[s], sp_m, a_sp, t_sp)
        OOF_LPROP[s] = p_lp
        s_old.append(eval_probs(OL_OLD[s], truth)); b_old.append(eval_probs(blend_dict(OL_OLD[s], SAB[s], 0.5), truth))
        s_new.append(eval_probs(p_lp, truth)); b_new.append(eval_probs(blend_dict(p_lp, SAB[s], 0.5), truth))
        pm_o, pm_n, pm_b = {}, {}, {}
        for q in p_lp:
            labs = set(O1[s][q]) | set(O2[s][q]) | set(O3[s][q]) | set(OL_OLD[s][q]) | set(p_lp[q])
            b13 = lambda l: (O1[s][q].get(l, 0) + O2[s][q].get(l, 0) + O3[s][q].get(l, 0)) / 6.0
            pm_o[q] = {l: b13(l) + 0.5 * OL_OLD[s][q].get(l, 0) for l in labs}
            pm_n[q] = {l: b13(l) + 0.5 * p_lp[q].get(l, 0) for l in labs}
            pm_b[q] = {l: b13(l) + 0.25 * OL_OLD[s][q].get(l, 0) + 0.25 * p_lp[q].get(l, 0) for l in labs}
        m_old.append(eval_probs(pm_o, truth)); bm_old.append(eval_probs(blend_dict(pm_o, SAB[s], 0.5), truth))
        m_new.append(eval_probs(pm_n, truth)); bm_new.append(eval_probs(blend_dict(pm_n, SAB[s], 0.5), truth))
        m_both.append(eval_probs(pm_b, truth)); bm_both.append(eval_probs(blend_dict(pm_b, SAB[s], 0.5), truth))

    print("\n=== STRICTLY NESTED 8-SPLIT HONEST CV: v10_fullbg_lprop vs v10_lda512_lprop ===", flush=True)
    print(f"  solo  v10_lprop : {np.mean(s_old):.5f} -> {np.mean(s_new):.5f} ({np.mean(s_new)-np.mean(s_old):+.5f}, wins {(np.array(s_new)>np.array(s_old)).sum()}/8)", flush=True)
    print(f"  blend v10_lprop : {np.mean(b_old):.5f} -> {np.mean(b_new):.5f} ({np.mean(b_new)-np.mean(b_old):+.5f}, wins {(np.array(b_new)>np.array(b_old)).sum()}/8)", flush=True)
    print(f"  lnb30-mix solo  : {np.mean(m_old):.5f} -> {np.mean(m_new):.5f} ({np.mean(m_new)-np.mean(m_old):+.5f}, wins {(np.array(m_new)>np.array(m_old)).sum()}/8)", flush=True)
    print(f"  lnb30-mix blend : {np.mean(bm_old):.5f} -> {np.mean(bm_new):.5f} ({np.mean(bm_new)-np.mean(bm_old):+.5f}, wins {(np.array(bm_new)>np.array(bm_old)).sum()}/8)", flush=True)
    print(f"  50bg+25old+25new: solo={np.mean(m_both):.5f} ({np.mean(m_both)-np.mean(m_old):+.5f}) | blend={np.mean(bm_both):.5f} ({np.mean(bm_both)-np.mean(bm_old):+.5f})", flush=True)

    # 5. Full 16-split 3-seed test fit
    lnb_dir = f"{H}/lnbnn_np64"
    base_S = {sp: ScorerDir(lnb_dir, sp, zip(tr.id[tr.species == sp], tr.identity[tr.species == sp])) for sp in ["hyena", "leopard"]}
    PT0, QT0 = pickle.load(open(f"{H}/test_v10.pkl", "rb"))
    sc_test = {sp: (S, np.ones(len(S.gal), dtype=bool)) for sp, S in base_S.items()}
    PT, QT = upgrade_table(PT0, QT0, tr, Xtr, te, Xte, sp_te, sc_test, sc_raw, sc_bg)
    PT, QT = add_lda512_feats(PT, QT, tr, Xtr, te, Xte)

    Pall = pd.concat([V2[s][0][PF1_L + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_L + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_L, QF1_L, seed=sd)
        p1 += mp.predict(PT[PF1_L]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_L]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()

    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)

    P2all = pd.concat([R1[s][0][PF2_L + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_L + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()
    p2, n2 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_L, QF2_L, seed=sd)
        p2 += mp.predict(PT[PF2_L]) / len(SEEDS)
        n2 += mq.predict(QT[QF2_L]) / len(SEEDS)
    PT["p2"] = p2; QT["pn2"] = n2

    test_raw = to_probs_raw(PT, QT, T_CAL)
    pt_te = build_qq_pairs_bg(QT, test_raw, Gt, cos_te, side_te, nkp, SC, sc_bg)
    p_same_te = np.zeros(len(pt_te))
    for sd in SEEDS:
        mp_full = lgb.train(dict(PAIR_PARAMS, seed=sd), lgb.Dataset(all_pt[PAIR_BG_FEATS].to_numpy(np.float32), all_pt["y"].values), num_boost_round=350)
        p_same_te += mp_full.predict(pt_te[PAIR_BG_FEATS].to_numpy(np.float32)) / len(SEEDS)
    pt_te["p_same"] = p_same_te

    a_final = {sp_i: best_hp[sp_i][1] for sp_i in [0, 1]}
    t_final = {sp_i: best_hp[sp_i][2] for sp_i in [0, 1]}
    sp_te_i = dict(zip(QT.qid, QT.species))
    probs_lprop = propagate_learned(list(QT.qid.values), test_raw, pt_te, sp_te_i, a_final, t_final)

    pickle.dump(test_raw, open(f"{H}/test_probs_dev_v10_lda512_raw.pkl", "wb"))
    pickle.dump(probs_lprop, open(f"{H}/test_probs_dev_v10_lda512_lprop.pkl", "wb"))
    pickle.dump(OOF_LPROP, open(f"{H}/oof_probs_dev_v10_lda512_lprop.pkl", "wb"))

    for sp in ["hyena", "leopard"]:
        ids = te.id[te.species == sp].values
        pnew = np.mean([probs_lprop[q].get(NEW, 0.0) for q in ids])
        n1_ = np.mean([max(probs_lprop[q], key=probs_lprop[q].get) == NEW for q in ids])
        print(f"test dev_v10_lda512_lprop {sp:7s}: mean_pnew={pnew:.4f} new@1={n1_:.4f}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
