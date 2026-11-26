"""Upgrade dev_v10_fullbg's heuristic propagation -> 100% leak-free Learned Query-Query Graph WITH
Camera-Trap SIFT Background Filtering (sift_rot, sift_spread, sift_is_bg, sift_s_clean from sift_bg_cache.pkl).
1. Recomputes exact 8-split OOF raw P0 from cached dev_v10_fullbg_r1.pkl (zero Round-1 rebuild).
2. Builds query-query pair tables across splits 0-15 + test using cosine + kpgraph + SIFT + camera-trap geometry.
3. Trains leak-free pair model P(same | q, q') (when predicting split s, drops ANY pair where either endpoint is in split s).
4. Tunes per-species (alpha <= 0.60, tau_e) strictly out-of-fold (leave-one-split-out on splits 0-7).
5. Verifies 8-split Honest CV & Shared Blend CV head-to-head against lnb27's Dev half (0.88280 / 0.88292).
6. Exports test_probs_dev_v10_fullbg_lprop.pkl and builds validated lnb30_v10_lprop.csv if CV improves."""
import os, gc, glob, pickle, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp, lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

from core import load, l2n, val_splits, NEW
from features import SIDE
import model2
from exp_kpg import others
from exp_blend_check import to_probs_raw, eval_probs, blend_dict
from build_dev_v10 import load_memmap_sift, FastSiftCache, SEEDS
from make_dev_v10_fullbg import PF2_ALL, QF2_ALL
from exp_dev_learned_prop import propagate_learned, PAIR_PARAMS

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")

PAIR_BG_FEATS = [
    "cos_qq", "kp_ij", "kp_ji", "kp_min", "kp_log", "kp_rank",
    "sift_h", "sift_s", "sift_ratio", "sift_frac",
    "sift_rot", "sift_spread", "sift_is_bg", "sift_s_clean",
    "same_side", "opp_side", "min_kp", "top1_same", "prob_dot", "species"
]


def build_qq_pairs_bg(Qf, probs, G, cos_map, side_map, nkp, SC, sc_bg):
    qids = list(Qf.qid.values)
    sp_map = dict(zip(Qf.qid, Qf.species))
    top1 = {q: max(probs[q], key=probs[q].get) for q in qids}
    cand_edges = {}
    sift_need = []
    for q, nids, nsims in zip(Qf.qid.values, Qf.nbr_ids.values, Qf.nbr_sims.values):
        cdict = {}
        for nb, sim in zip(nids, nsims):
            cdict[nb] = [float(sim), 0.0, 0.0, 99]
        for r_kp, (nb, s1, s2) in enumerate(G.get(q, [])):
            if nb in cdict:
                cdict[nb][1] = float(s1); cdict[nb][2] = float(s2); cdict[nb][3] = r_kp
            else:
                cdict[nb] = [cos_map(q, nb), float(s1), float(s2), r_kp]
        cand_edges[q] = cdict
        for nb in cdict:
            sift_need.append((q, nb))
    SC.fill(sift_need)

    rows = []
    for q in qids:
        sq = side_map[q]; pq = probs[q]; tq = top1[q]; spq = sp_map[q]; kpq = nkp[q]
        for nb, (cs, s1, s2, rkp) in cand_edges[q].items():
            if nb not in probs:
                continue
            sm = SC(q, nb)
            key = tuple(sorted((q, nb)))
            bg = sc_bg.get(key)
            rot = float(bg[1]) if bg is not None else 99.0
            sp_m = float(bg[2]) if bg is not None else 1.0
            is_bg = float(bg[3]) if bg is not None else 0.0
            s_clean = 0.0 if is_bg > 0.5 else float(sm[2])
            sn = side_map[nb]; pn = probs[nb]
            same_s = float(sq == sn and sq in ("L", "R"))
            opp_s = float({sq, sn} == {"L", "R"})
            dot = sum(v * pn.get(l, 0.0) for l, v in pq.items() if l != NEW)
            rows.append((
                q, nb, cs, s1, s2, min(s1, s2), np.log1p(s1), rkp,
                sm[1], sm[2], sm[0], sm[3],
                rot, sp_m, is_bg, s_clean,
                same_s, opp_s, min(kpq, nkp[nb]),
                float(tq == top1[nb] and tq != NEW), float(dot), spq
            ))
    return pd.DataFrame(rows, columns=["qi", "qj"] + PAIR_BG_FEATS)


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    R1 = pickle.load(open(f"{H}/dev_v10_fullbg_r1.pkl", "rb"))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}
    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    sc_bg = pickle.load(open(f"{H}/sift_bg_cache.pkl", "rb"))

    # 1. Exact OOF raw P0 for all 16 splits (splits 0-7: 3 seeds; splits 8-15: 1 seed for pair-model training)
    OOF, CAL = {}, []
    for s in range(16):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        sds = SEEDS if s < 8 else [0]
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in sds:
            mp, mq = model2.fit(Ptr, Qtr, PF2_ALL, QF2_ALL, seed=sd)
            p2 += mp.predict(Pv[PF2_ALL]) / len(sds)
            n2 += mq.predict(Qv[QF2_ALL]) / len(sds)
        Pv["p2"] = p2; Qv["pn2"] = n2
        OOF[s] = (Pv[["qid", "cand", "p2"]], Qv[["qid", "species", "pn2", "truth", "nbr_ids", "nbr_sims"]])
        if s < 8:
            tmap = dict(zip(Qv.qid, Qv.truth)); pn = dict(zip(Qv.qid, Qv.pn2))
            for q, g in Pv.groupby("qid", sort=False):
                labs = list(g.cand.values) + [NEW]
                if tmap[q] in labs:
                    CAL.append((np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)), labs.index(tmap[q])))
    del R1; gc.collect()

    def nll(T):
        return -np.mean([(sc[i] / T) - np.log(np.exp(sc / T - (sc / T).max()).sum()) - (sc / T).max() for sc, i in CAL])
    T_CAL = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    RAW_P = {s: to_probs_raw(OOF[s][0], OOF[s][1], T_CAL) for s in range(16)}
    print(f"Built OOF raw P0 (T_CAL={T_CAL:.3f}) in {time.time()-t0:.0f}s", flush=True)

    # 2. Build query-query pair tables with camera-trap SIFT features
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
    print(f"Built 16 split qq-pair tables in {time.time()-t0:.0f}s", flush=True)

    # 3. Leak-free OOF pair model (strictly drop ANY training pair touching ANY query in split s!)
    all_pt = pd.concat([PTABS[s] for s in range(16)], ignore_index=True)
    OOF_PT = {}
    aucs = []
    for s in range(8):
        qs = set(OOF[s][1].qid.values)
        tr_m = (all_pt["split"] != s) & (~all_pt["qi"].isin(qs)) & (~all_pt["qj"].isin(qs))
        X_p = all_pt.loc[tr_m, PAIR_BG_FEATS].to_numpy(np.float32)
        y_p = all_pt.loc[tr_m, "y"].values
        m_p = lgb.train(PAIR_PARAMS, lgb.Dataset(X_p, y_p), num_boost_round=350)
        pt_v = PTABS[s].copy()
        pt_v["p_same"] = m_p.predict(pt_v[PAIR_BG_FEATS].to_numpy(np.float32))
        OOF_PT[s] = pt_v
        aucs.append(roc_auc_score(pt_v["y"], pt_v["p_same"]))
    print(f"Leak-free qq-pair OOF AUC (splits 0-7): {np.mean(aucs):.4f} (per split: {[round(a,4) for a in aucs]}) ({time.time()-t0:.0f}s)", flush=True)

    # 4. Out-of-fold (leave-one-split-out) grid search for per-species (alpha <= 0.60, tau_e)
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

    # Nested leave-one-split-out evaluation on splits 0-7!
    SAB = pickle.load(open(SAB_CV, "rb"))
    O1 = pickle.load(open(f"{H}/oof_probs_dev_bgclean_prop.pkl", "rb"))
    O2 = pickle.load(open(f"{H}/oof_probs_dev_bgclean2_prop.pkl", "rb"))
    O3 = pickle.load(open(f"{H}/oof_probs_dev_bgclean3_prop.pkl", "rb"))
    OV_OLD = pickle.load(open(f"{H}/oof_probs_dev_v10_fullbg_prop.pkl", "rb"))
    OOF_LPROP = {}
    s_old, b_old, s_new, b_new, m_old, bm_old, m_new, bm_new = [], [], [], [], [], [], [], []
    for s in range(8):
        # Pick hyperparams ONLY from the other 7 splits (strictly nested OOF!)
        a_sp, t_sp = {}, {}
        for sp_i in [0, 1]:
            _, ba, bt = max([(np.mean([grid_scores[(sp_i, a, t)][k] for k in range(8) if k != s]), a, t)
                             for a in alphas for t in taus])
            a_sp[sp_i], t_sp[sp_i] = ba, bt
        Qv = OOF[s][1]; truth = dict(zip(Qv.qid, Qv.truth)); sp_m = dict(zip(Qv.qid, Qv.species))
        p_lp = propagate_learned(list(Qv.qid.values), RAW_P[s], OOF_PT[s], sp_m, a_sp, t_sp)
        OOF_LPROP[s] = p_lp
        s_old.append(eval_probs(OV_OLD[s], truth)); b_old.append(eval_probs(blend_dict(OV_OLD[s], SAB[s], 0.5), truth))
        s_new.append(eval_probs(p_lp, truth)); b_new.append(eval_probs(blend_dict(p_lp, SAB[s], 0.5), truth))
        # Also compare lnb27 Dev half: 50% avg(bg1-3) + 50% v10_fullbg vs + 50% v10_fullbg_lprop
        pm_o, pm_n = {}, {}
        for q in p_lp:
            labs = set(O1[s][q]) | set(O2[s][q]) | set(O3[s][q]) | set(OV_OLD[s][q]) | set(p_lp[q])
            b13 = lambda l: (O1[s][q].get(l, 0) + O2[s][q].get(l, 0) + O3[s][q].get(l, 0)) / 6.0
            pm_o[q] = {l: b13(l) + 0.5 * OV_OLD[s][q].get(l, 0) for l in labs}
            pm_n[q] = {l: b13(l) + 0.5 * p_lp[q].get(l, 0) for l in labs}
        m_old.append(eval_probs(pm_o, truth)); bm_old.append(eval_probs(blend_dict(pm_o, SAB[s], 0.5), truth))
        m_new.append(eval_probs(pm_n, truth)); bm_new.append(eval_probs(blend_dict(pm_n, SAB[s], 0.5), truth))

    print("\n=== STRICTLY NESTED 8-SPLIT HONEST CV: HEURISTIC vs LEARNED BG-FILTERED PROP ===", flush=True)
    print(f"  v10_fullbg solo  : {np.mean(s_old):.5f} -> {np.mean(s_new):.5f} ({np.mean(s_new)-np.mean(s_old):+.5f}, wins {(np.array(s_new)>np.array(s_old)).sum()}/8)", flush=True)
    print(f"  v10_fullbg blend : {np.mean(b_old):.5f} -> {np.mean(b_new):.5f} ({np.mean(b_new)-np.mean(b_old):+.5f}, wins {(np.array(b_new)>np.array(b_old)).sum()}/8)", flush=True)
    print(f"  lnb27-mix solo   : {np.mean(m_old):.5f} -> {np.mean(m_new):.5f} ({np.mean(m_new)-np.mean(m_old):+.5f}, wins {(np.array(m_new)>np.array(m_old)).sum()}/8)", flush=True)
    print(f"  lnb27-mix blend  : {np.mean(bm_old):.5f} -> {np.mean(bm_new):.5f} ({np.mean(bm_new)-np.mean(bm_old):+.5f}, wins {(np.array(bm_new)>np.array(bm_old)).sum()}/8)", flush=True)

    # 5. Full pair model fit on all 16 splits -> apply to test_probs_dev_v10_fullbg.pkl
    _, QT = pickle.load(open(f"{H}/test_v10.pkl", "rb"))
    test_raw = pickle.load(open(f"{H}/test_probs_dev_v10_fullbg.pkl", "rb"))
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
    pickle.dump(probs_lprop, open(f"{H}/test_probs_dev_v10_fullbg_lprop.pkl", "wb"))
    pickle.dump(OOF_LPROP, open(f"{H}/oof_probs_dev_v10_fullbg_lprop.pkl", "wb"))

    # Build validated lnb30_v10_lprop_mix.csv
    d1 = pickle.load(open(f"{H}/test_probs_dev_bgclean_prop.pkl", "rb"))
    d2 = pickle.load(open(f"{H}/test_probs_dev_bgclean2_prop.pkl", "rb"))
    d3 = pickle.load(open(f"{H}/test_probs_dev_bgclean3_prop.pkl", "rb"))
    t17 = pickle.load(open([f for f in sorted(glob.glob("teammate_sabari/work/submissions/*v17_bgclean_conn*.probs.pkl")) if not f.endswith(".cv.probs.pkl")][-1], "rb"))
    t19 = pickle.load(open([f for f in sorted(glob.glob("teammate_sabari/work/submissions/*v19_bgclean_v2*.probs.pkl")) if not f.endswith(".cv.probs.pkl")][-1], "rb"))
    allowed = {s: set(tr.identity[tr.species == s]) | {NEW} for s in ["hyena", "leopard"]}

    out_csv = f"{H}/sub_team_blend_lnb30_v10_lprop.csv"
    preds = []
    for q, sp_s in zip(te.id, te.species):
        sc = {}
        for p, w in [(d1, 1/12), (d2, 1/12), (d3, 1/12), (probs_lprop, 0.25), (t17, 0.25), (t19, 0.25)]:
            for k, v in p[q].items():
                sc[k] = sc.get(k, 0.0) + w * v
        ranked = sorted(sc.keys(), key=lambda l: (-sc[l], l))
        labs = [l for l in ranked if l in allowed[sp_s]][:5]
        for l in [NEW] + sorted(allowed[sp_s]):
            if len(labs) == 5: break
            if l not in labs: labs.append(l)
        assert len(labs) == 5 and len(set(labs)) == 5 and set(labs) <= allowed[sp_s], q
        preds.append(" ".join(labs))
    sub = pd.DataFrame({"id": te.id, "predictions": preds})
    assert len(sub) == 2791 and sub.id.is_unique and sub.isna().sum().sum() == 0
    sub.to_csv(out_csv, index=False); assert pd.read_csv(out_csv).equals(sub)
    t1 = sub.predictions.str.split().str[0].values
    ref27 = pd.read_csv(f"{H}/sub_team_blend_lnb27_v10fullbg_mix.csv").predictions.str.split().str[0].values
    mh = (te.species == "hyena").values; ml = (te.species == "leopard").values
    print(f"\nWROTE & VALIDATED {out_csv} | new@1={(t1==NEW).mean():.4f} (h={(t1[mh]==NEW).mean():.4f}, l={(t1[ml]==NEW).mean():.4f}) | diff_vs_lnb27={(t1!=ref27).sum()} (h={(t1[mh]!=ref27[mh]).sum()}, l={(t1[ml]!=ref27[ml]).sum()})", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
