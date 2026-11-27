"""Upgrade Dev's v10 expanded pool (dev_v10_sw2152_lprop, currently 25% of 0.88585 Rank 1 lnb36) with:
  1. 100% gallery-only Sw2152 Class Centroid Prototype + Same-ID Gallery DBA + MiewID log-norm
  2. Genuine non-tree Neural Network diversity: 3-seed Tabular MLPClassifier(128, 64) alongside 3-seed LightGBM
  3. Upgraded sw_qq + ssw_qq Learned BG-Filtered Query-Query Propagation (OOF AUC ~0.982 vs 0.9775)
Reuses cached dev_v10_sw2152_v2.pkl (zero SIFT/LNBNN recomputation), 100% leak-free, gated on 8-split Honest CV."""
import os, gc, pickle, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

from core import load, l2n, val_splits, NEW
from features import SIDE
import model2
from exp_kpg import others
from exp_kpg_v2 import rich_feats
from exp_blend_check import to_probs_raw, eval_probs, blend_dict
from build_dev_v10 import load_memmap_sift, FastSiftCache, SEEDS
from make_dev_bgclean import ScorerDir
from make_dev_v10_fullbg import upgrade_table
from make_dev_v10_lda512 import add_lda512_feats
from make_dev_v10_sw2152 import PF1_SW, QF1_SW, PF2_SW, QF2_SW, add_sw2152_feats
from make_dev_bg_sw2152 import add_lognorm
from make_dev_bg_swproto_mlp import (
    SWP_P, SWP_Q, G2Q_P1, G2Q_Q1, G2Q_P2, G2Q_Q2, PAIR_SW_FEATS,
    add_sw_proto_feats, add_g2q_r1, add_g2q_r2, fit_predict_mlp, compute_sw_qq_maps, fit_T
)
from make_dev_v10_lprop import build_qq_pairs_bg
from exp_dev_learned_prop import propagate_learned, PAIR_PARAMS

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")

PF1_V = PF1_SW + SWP_P + G2Q_P1
QF1_V = QF1_SW + SWP_Q + ["q_lognorm"] + G2Q_Q1
PF2_V = PF2_SW + SWP_P + G2Q_P1 + G2Q_P2
QF2_V = QF2_SW + SWP_Q + ["q_lognorm"] + G2Q_Q1 + G2Q_Q2


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    sp_te = dict(zip(te.id, te.species))
    norm_tr = dict(zip(tr.id, np.linalg.norm(Xtr, axis=1)))
    norm_te = dict(zip(te.id, np.linalg.norm(Xte, axis=1)))
    splits = val_splits(tr, l2n(Xtr), range(16))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}
    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    sc_raw = pickle.load(open(f"{H}/sift_pair_cache.pkl", "rb"))
    sc_bg = pickle.load(open(f"{H}/sift_bg_cache.pkl", "rb"))

    # 1. Attach sw_proto + q_lognorm + G2Q_R1 onto cached dev_v10_sw2152_v2.pkl
    v2_path = f"{H}/dev_v10_swproto_g2q_v2.pkl"
    if os.path.exists(v2_path):
        V2 = pickle.load(open(v2_path, "rb"))
    else:
        V2_prev = pickle.load(open(f"{H}/dev_v10_sw2152_v2.pkl", "rb"))
        V2 = {}
        for s, (gal, q, truth) in enumerate(splits):
            P0, Q0 = add_sw_proto_feats(V2_prev[s][0], V2_prev[s][1],
                                        tr[gal].reset_index(drop=True), Xtr[gal],
                                        tr[q].reset_index(drop=True), Xtr[q])
            Q0 = add_lognorm(Q0, norm_tr)
            P0, Q0 = add_g2q_r1(P0, Q0)
            V2[s] = (P0, Q0)
        del V2_prev; gc.collect()
        pickle.dump(V2, open(v2_path, "wb"), protocol=4)
    print(f"Prepared {v2_path} ({time.time()-t0:.0f}s)", flush=True)

    # 2. Leak-free Round-1 OOF + G2Q_R2
    r1_path = f"{H}/dev_v10_swproto_g2q_r1.pkl"
    if os.path.exists(r1_path):
        R1 = pickle.load(open(r1_path, "rb"))
    else:
        R1, r1s = {}, []
        for s in range(16):
            Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
            Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
            mp, mq = model2.fit(Ptr, Qtr, PF1_V, QF1_V, seed=0)
            Pv["p1"] = mp.predict(Pv[PF1_V]); Qv["pn1"] = mq.predict(Qv[QF1_V])
            if s < 8:
                r1s.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
            Pv, Qv = model2.round2_feats(Pv, Qv)
            Pv, Qv = rich_feats(Pv, Qv, G[s], scale=2.0)
            R1[s] = add_g2q_r2(Pv, Qv)
        print(f"R1 MEAN (splits 0-7): {np.mean(r1s):.4f} ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(R1, open(r1_path, "wb"), protocol=4)

    # 3. Leak-free Round-2 OOF: BOTH LightGBM (3 seeds) AND MLP (3 seeds)
    OOF_LGB, OOF_MLP = {}, {}
    for s in range(16):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        sds = SEEDS if s < 8 else [0]
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in sds:
            mp, mq = model2.fit(Ptr, Qtr, PF2_V, QF2_V, seed=sd)
            p2 += mp.predict(Pv[PF2_V]) / len(sds)
            n2 += mq.predict(Qv[QF2_V]) / len(sds)
        Pv["p2"] = p2; Qv["pn2"] = n2
        OOF_LGB[s] = (Pv[["qid", "cand", "p2"]], Qv[["qid", "species", "pn2", "truth", "nbr_ids", "nbr_sims"]])
        if s < 8:
            Pm, Qm = R1[s][0].copy(), R1[s][1].copy()
            pm2, nm2 = fit_predict_mlp(Ptr, Qtr, Pm, Qm, PF2_V, QF2_V, seeds=(0, 1))
            Pm["p2"] = pm2; Qm["pn2"] = nm2
            OOF_MLP[s] = (Pm[["qid", "cand", "p2"]], Qm[["qid", "species", "pn2", "truth", "nbr_ids", "nbr_sims"]])

    T_LGB = fit_T(OOF_LGB, 8)
    T_MLP = fit_T(OOF_MLP, 8)
    RAW_LGB = {s: to_probs_raw(OOF_LGB[s][0], OOF_LGB[s][1], T_LGB) for s in range(16)}
    RAW_MLP = {s: to_probs_raw(OOF_MLP[s][0], OOF_MLP[s][1], T_MLP) for s in range(8)}
    print(f"Round-2 OOF done: T_LGB={T_LGB:.3f}, T_MLP={T_MLP:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # 4. Learned BG-filtered + sw_qq/ssw_qq propagation (strictly nested OOF!)
    D, XY, off, pos, nkp = load_memmap_sift()
    SC = FastSiftCache(D, XY, off, pos)
    side_tr = dict(zip(tr.id, tr.viewpoint.map(SIDE).fillna("O")))
    side_te = dict(zip(te.id, te.viewpoint.map(SIDE).fillna("O")))
    ident_tr = dict(zip(tr.id, tr.identity))
    Xtr_n = l2n(Xtr); Xte_n = l2n(Xte)
    tr_pos = {q: i for i, q in enumerate(tr.id)}; te_pos = {q: i for i, q in enumerate(te.id)}
    cos_tr = lambda a, b: float(Xtr_n[tr_pos[a]] @ Xtr_n[tr_pos[b]])
    cos_te = lambda a, b: float(Xte_n[te_pos[a]] @ Xte_n[te_pos[b]])
    SW_Q, SW_QT = compute_sw_qq_maps(tr, Xtr, splits, te, Xte)

    PTABS = {}
    for s in range(16):
        pt = build_qq_pairs_bg(OOF_LGB[s][1], RAW_LGB[s], G[s], cos_tr, side_tr, nkp, SC, sc_bg)
        zi, zs = SW_Q[s]
        pt["sw_qq"] = [float(zi[a] @ zi[b]) for a, b in zip(pt.qi, pt.qj)]
        pt["ssw_qq"] = [float(zs[a] @ zs[b]) for a, b in zip(pt.qi, pt.qj)]
        pt["y"] = [int(ident_tr[a] == ident_tr[b]) for a, b in zip(pt.qi, pt.qj)]
        pt["split"] = s
        PTABS[s] = pt

    all_pt = pd.concat([PTABS[s] for s in range(16)], ignore_index=True)
    OOF_PT, aucs = {}, []
    for s in range(8):
        qs = set(OOF_LGB[s][1].qid.values)
        tr_m = (all_pt["split"] != s) & (~all_pt["qi"].isin(qs)) & (~all_pt["qj"].isin(qs))
        m_p = lgb.train(PAIR_PARAMS, lgb.Dataset(all_pt.loc[tr_m, PAIR_SW_FEATS].to_numpy(np.float32), all_pt.loc[tr_m, "y"].values), num_boost_round=350)
        pt_v = PTABS[s].copy()
        pt_v["p_same"] = m_p.predict(pt_v[PAIR_SW_FEATS].to_numpy(np.float32))
        OOF_PT[s] = pt_v
        aucs.append(roc_auc_score(pt_v["y"], pt_v["p_same"]))
    print(f"dev_v10 sw_qq lprop OOF AUC: {np.mean(aucs):.5f} ({time.time()-t0:.0f}s)", flush=True)

    alphas = [0.25, 0.35, 0.45, 0.55]
    taus = [0.40, 0.50, 0.60, 0.70]
    def nested_lprop(raw_dict):
        grid = {(sp_i, a, t): [] for sp_i in [0, 1] for a in alphas for t in taus}
        for s in range(8):
            Qv = OOF_LGB[s][1]; truth = dict(zip(Qv.qid, Qv.truth)); sp_m = dict(zip(Qv.qid, Qv.species))
            qids = list(Qv.qid.values)
            for sp_i in [0, 1]:
                tr_sp = {q: truth[q] for q in qids if sp_m[q] == sp_i}
                for a in alphas:
                    for t in taus:
                        pr = propagate_learned(qids, raw_dict[s], OOF_PT[s], sp_m, {0: a, 1: a}, {0: t, 1: t})
                        grid[(sp_i, a, t)].append(eval_probs({q: pr[q] for q in tr_sp}, tr_sp))
        out = {}
        best_hp = {sp_i: max([(np.mean(grid[(sp_i, a, t)]), a, t) for a in alphas for t in taus]) for sp_i in [0, 1]}
        for s in range(8):
            a_sp, t_sp = {}, {}
            for sp_i in [0, 1]:
                _, ba, bt = max([(np.mean([grid[(sp_i, a, t)][k] for k in range(8) if k != s]), a, t) for a in alphas for t in taus])
                a_sp[sp_i], t_sp[sp_i] = ba, bt
            Qv = OOF_LGB[s][1]; sp_m = dict(zip(Qv.qid, Qv.species))
            out[s] = propagate_learned(list(Qv.qid.values), raw_dict[s], OOF_PT[s], sp_m, a_sp, t_sp)
        return out, best_hp

    OOF_LGB_LP, hp_lgb = nested_lprop(RAW_LGB)
    OOF_MLP_LP, hp_mlp = nested_lprop(RAW_MLP)
    OOF_MIX_LP = {s: blend_dict(OOF_LGB_LP[s], OOF_MLP_LP[s], 0.5) for s in range(8)}

    SAB = pickle.load(open(SAB_CV, "rb"))
    OBG_OLD = pickle.load(open(f"{H}/oof_probs_dev_bg_sw2152_prop.pkl", "rb"))
    OSW_OLD = pickle.load(open(f"{H}/oof_probs_dev_v10_sw2152_lprop.pkl", "rb"))
    OBG_NEW_L = pickle.load(open(f"{H}/oof_probs_dev_bg_swproto_lgb_lprop.pkl", "rb")) if os.path.exists(f"{H}/oof_probs_dev_bg_swproto_lgb_lprop.pkl") else OBG_OLD
    OBG_NEW_M = pickle.load(open(f"{H}/oof_probs_dev_bg_swproto_mlp_lprop.pkl", "rb")) if os.path.exists(f"{H}/oof_probs_dev_bg_swproto_mlp_lprop.pkl") else OBG_OLD
    OBG_NEW = {s: blend_dict(OBG_NEW_L[s], OBG_NEW_M[s], 0.5) for s in range(8)}

    print("\n=== DEV_V10_SWPROTO_MLP 8-SPLIT HONEST CV ===", flush=True)
    for name, bg_d, v10_d in [
        ("lnb36 BASE (bg_sw + v10_sw)", OBG_OLD, OSW_OLD),
        ("v10_swproto_lgb + bg_sw_old", OBG_OLD, OOF_LGB_LP),
        ("v10_swproto_mlp + bg_sw_old", OBG_OLD, OOF_MLP_LP),
        ("v10_swproto_lgb+mlp + bg_sw_old", OBG_OLD, OOF_MIX_LP),
        ("FULL UPGRADE: bg_swproto(L+M) + v10_swproto(L+M)", OBG_NEW, OOF_MIX_LP),
    ]:
        s_v10, devh, shbl = [], [], []
        for s in range(8):
            Qv = OOF_LGB[s][1]; truth = dict(zip(Qv.qid, Qv.truth))
            s_v10.append(eval_probs(v10_d[s], truth))
            pd_ = blend_dict(bg_d[s], v10_d[s], 0.5)
            devh.append(eval_probs(pd_, truth))
            shbl.append(eval_probs(blend_dict(pd_, SAB[s], 0.5), truth))
        print(f"  {name:50s} | v10={np.mean(s_v10):.5f} | DevHalf={np.mean(devh):.5f} | SharedBlend={np.mean(shbl):.5f}", flush=True)

    # 5. Full 16-split 3-seed test fit for BOTH LGB and MLP
    lnb_dir = f"{H}/lnbnn_np64"
    base_S = {sp: ScorerDir(lnb_dir, sp, zip(tr.id[tr.species == sp], tr.identity[tr.species == sp])) for sp in ["hyena", "leopard"]}
    PT, QT = pickle.load(open(f"{H}/test_v10.pkl", "rb"))
    sc_test = {sp: (S, np.ones(len(S.gal), dtype=bool)) for sp, S in base_S.items()}
    PT, QT = upgrade_table(PT, QT, tr, Xtr, te, Xte, sp_te, sc_test, sc_raw, sc_bg)
    PT, QT = add_lda512_feats(PT, QT, tr, Xtr, te, Xte)
    PT, QT = add_sw2152_feats(PT, QT, tr, Xtr, te, Xte)
    PT, QT = add_sw_proto_feats(PT, QT, tr, Xtr, te, Xte)
    QT = add_lognorm(QT, norm_te)
    PT, QT = add_g2q_r1(PT, QT)

    Pall = pd.concat([V2[s][0][PF1_V + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_V + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_V, QF1_V, seed=sd)
        p1 += mp.predict(PT[PF1_V]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_V]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()
    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)
    PT, QT = add_g2q_r2(PT, QT)

    P2all = pd.concat([R1[s][0][PF2_V + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_V + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()

    # Test LGB
    p2_l, n2_l = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_V, QF2_V, seed=sd)
        p2_l += mp.predict(PT[PF2_V]) / len(SEEDS)
        n2_l += mq.predict(QT[QF2_V]) / len(SEEDS)
    PT_l, QT_l = PT.copy(), QT.copy(); PT_l["p2"] = p2_l; QT_l["pn2"] = n2_l
    t_raw_lgb = to_probs_raw(PT_l, QT_l, T_LGB)

    # Test MLP
    p2_m, n2_m = fit_predict_mlp(P2all, Q2all, PT, QT, PF2_V, QF2_V, seeds=(0, 1))
    PT_m, QT_m = PT.copy(), QT.copy(); PT_m["p2"] = p2_m; QT_m["pn2"] = n2_m
    t_raw_mlp = to_probs_raw(PT_m, QT_m, T_MLP)

    # Test sw_qq lprop
    m_pair_full = lgb.train(PAIR_PARAMS, lgb.Dataset(all_pt[PAIR_SW_FEATS].to_numpy(np.float32), all_pt["y"].values), num_boost_round=350)
    pt_test = build_qq_pairs_bg(QT_l, t_raw_lgb, Gt, cos_te, side_te, nkp, SC, sc_bg)
    zi_t, zs_t = SW_QT
    pt_test["sw_qq"] = [float(zi_t[a] @ zi_t[b]) for a, b in zip(pt_test.qi, pt_test.qj)]
    pt_test["ssw_qq"] = [float(zs_t[a] @ zs_t[b]) for a, b in zip(pt_test.qi, pt_test.qj)]
    pt_test["p_same"] = m_pair_full.predict(pt_test[PAIR_SW_FEATS].to_numpy(np.float32))
    sp_tm = dict(zip(QT_l.qid, QT_l.species))

    t_lgb_lp = propagate_learned(list(QT_l.qid.values), t_raw_lgb, pt_test, sp_tm,
                                 {0: hp_lgb[0][1], 1: hp_lgb[1][1]}, {0: hp_lgb[0][2], 1: hp_lgb[1][2]})
    t_mlp_lp = propagate_learned(list(QT_m.qid.values), t_raw_mlp, pt_test, sp_tm,
                                 {0: hp_mlp[0][1], 1: hp_mlp[1][1]}, {0: hp_mlp[0][2], 1: hp_mlp[1][2]})

    pickle.dump(t_lgb_lp, open(f"{H}/test_probs_dev_v10_swproto_lgb_lprop.pkl", "wb"))
    pickle.dump(t_mlp_lp, open(f"{H}/test_probs_dev_v10_swproto_mlp_lprop.pkl", "wb"))
    pickle.dump(OOF_LGB_LP, open(f"{H}/oof_probs_dev_v10_swproto_lgb_lprop.pkl", "wb"))
    pickle.dump(OOF_MLP_LP, open(f"{H}/oof_probs_dev_v10_swproto_mlp_lprop.pkl", "wb"))
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
