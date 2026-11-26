"""Upgrade Dev's MiewID+LDA candidate pool (bg1-3, currently 25% of our 0.88424 Rank 1 submission)
with ALL our proven zero-leakage discoveries:
  1. Stage-2 Full-Frame Static Camera-Trap SIFT Killer (add_fullframe_bg)
  2. Gallery-only Different-Identity Hubness (r_g_diff)
  3. Full 2152-dim Within-Class Covariance Whitening (sw2152 + ssw2152, zero PCA truncation)
  4. Raw MiewID embedding log-norm (from Deep EDA: hyena ok=70.35 vs err=69.21)
Reuses cached dev_bgclean2_v2.pkl (zero SIFT/LNBNN recomputation!), 100% leak-free, strictly gated on 8-split Honest CV."""
import os, gc, pickle, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar

from core import load, l2n, val_splits, NEW
import model2
from exp_kpg import others
from exp_kpg_v2 import rich_feats
from exp_blend_check import to_probs_raw, propagate_probs, eval_probs, blend_dict
from make_kpg_probs import add_lnbnn_test
from make_dev_bgclean import ScorerDir, add_bg_lnb
from make_dev_bgclean2 import add_sift_bg_feats
from make_dev_bgclean3 import add_fullframe_bg, PF1_BG3, QF1_BG3, PF2_BG3, QF2_BG3
from make_dev_hubness import add_hubness_feats, HUB_P, HUB_Q
from make_dev_v10_sw2152 import add_sw2152_feats, SW_P, SW_Q

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")
SEEDS = [0, 1, 2]

NORM_Q = ["q_lognorm"]
PF1_BGSW = PF1_BG3 + HUB_P + SW_P
QF1_BGSW = QF1_BG3 + HUB_Q + SW_Q + NORM_Q
PF2_BGSW = PF2_BG3 + HUB_P + SW_P
QF2_BGSW = QF2_BG3 + HUB_Q + SW_Q + NORM_Q


def add_lognorm(Q, norm_map):
    Q = Q.copy()
    Q["q_lognorm"] = np.array([np.log1p(norm_map[q]) for q in Q.qid.values], dtype=np.float32)
    return Q


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    sp_te = dict(zip(te.id, te.species))
    norm_tr = dict(zip(tr.id, np.linalg.norm(Xtr, axis=1)))
    norm_te = dict(zip(te.id, np.linalg.norm(Xte, axis=1)))
    splits = val_splits(tr, l2n(Xtr), range(16))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}

    v2_path = f"{H}/dev_bg_sw2152_v2.pkl"
    if os.path.exists(v2_path):
        V2 = pickle.load(open(v2_path, "rb"))
        print(f"Loaded cached {v2_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        V2_prev = pickle.load(open(f"{H}/dev_bgclean2_v2.pkl", "rb"))
        V2 = {}
        for s, (gal, q, truth) in enumerate(splits):
            P0, Q0 = add_fullframe_bg(V2_prev[s][0], V2_prev[s][1])
            P0, Q0 = add_hubness_feats(P0, Q0, tr[gal].reset_index(drop=True), Xtr[gal],
                                       tr[q].reset_index(drop=True), Xtr[q])
            P0, Q0 = add_sw2152_feats(P0, Q0, tr[gal].reset_index(drop=True), Xtr[gal],
                                      tr[q].reset_index(drop=True), Xtr[q])
            Q0 = add_lognorm(Q0, norm_tr)
            V2[s] = (P0, Q0)
            if (s + 1) % 4 == 0:
                print(f"  upgraded {s+1}/16 splits ({time.time()-t0:.0f}s)", flush=True)
        del V2_prev; gc.collect()
        pickle.dump(V2, open(v2_path, "wb"), protocol=4)
        print(f"Cached {v2_path} ({time.time()-t0:.0f}s)", flush=True)

    # Leak-free Round-1 OOF
    R1, r1_scores = {}, []
    for s in range(16):
        Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
        Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
        mp, mq = model2.fit(Ptr, Qtr, PF1_BGSW, QF1_BGSW, seed=0)
        Pv["p1"] = mp.predict(Pv[PF1_BGSW]); Qv["pn1"] = mq.predict(Qv[QF1_BGSW])
        if s < 8:
            r1_scores.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
        Pv, Qv = model2.round2_feats(Pv, Qv)
        R1[s] = rich_feats(Pv, Qv, G[s], scale=2.0)
    print(f"R1 MEAN (splits 0-7): {np.mean(r1_scores):.4f} (bg3 was 0.8639, dev_hub was 0.8659) ({time.time()-t0:.0f}s)", flush=True)

    # Leak-free Round-2 OOF (splits 0-7, 3 seeds)
    OOF, CAL = {}, []
    for s in range(8):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in SEEDS:
            mp, mq = model2.fit(Ptr, Qtr, PF2_BGSW, QF2_BGSW, seed=sd)
            p2 += mp.predict(Pv[PF2_BGSW]) / len(SEEDS)
            n2 += mq.predict(Qv[QF2_BGSW]) / len(SEEDS)
        Pv["p2"] = p2; Qv["pn2"] = n2
        OOF[s] = (Pv[["qid", "cand", "p2"]], Qv[["qid", "species", "pn2", "truth", "nbr_ids", "nbr_sims"]])
        tmap = dict(zip(Qv.qid, Qv.truth)); pn = dict(zip(Qv.qid, Qv.pn2))
        for q, g in Pv.groupby("qid", sort=False):
            labs = list(g.cand.values) + [NEW]
            if tmap[q] in labs:
                CAL.append((np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)), labs.index(tmap[q])))

    def nll(T):
        return -np.mean([(sc[i] / T) - np.log(np.exp(sc / T - (sc / T).max()).sum()) - (sc / T).max() for sc, i in CAL])
    T_CAL = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    print(f"Fitted T_CAL = {T_CAL:.3f} ({time.time()-t0:.0f}s)", flush=True)

    SAB = pickle.load(open(SAB_CV, "rb"))
    O1 = pickle.load(open(f"{H}/oof_probs_dev_bgclean_prop.pkl", "rb"))
    O2 = pickle.load(open(f"{H}/oof_probs_dev_bgclean2_prop.pkl", "rb"))
    O3 = pickle.load(open(f"{H}/oof_probs_dev_bgclean3_prop.pkl", "rb"))
    OSW = pickle.load(open(f"{H}/oof_probs_dev_v10_sw2152_lprop.pkl", "rb"))

    OOF_PROBS = {}
    s_solo, b_solo, s_old, b_old, s_new, b_new, s_mix, b_mix = [], [], [], [], [], [], [], []
    for s in range(8):
        Pv, Qv = OOF[s]; truth = dict(zip(Qv.qid, Qv.truth))
        p_raw = to_probs_raw(Pv, Qv, T_CAL)
        p_prop = propagate_probs(p_raw, Qv, G[s], alpha=0.30, kp_thr=2.0, cos_thr=0.72)
        OOF_PROBS[s] = p_prop
        s_solo.append(eval_probs(p_prop, truth))
        b_solo.append(eval_probs(blend_dict(p_prop, SAB[s], 0.5), truth))

        # Compare against lnb34 Dev half: 50% avg(bg1-3) + 50% v10_sw2152_lprop
        p_o, p_n, p_m = {}, {}, {}
        for q in p_prop:
            labs = set(O1[s][q]) | set(O2[s][q]) | set(O3[s][q]) | set(p_prop[q]) | set(OSW[s][q])
            b13 = lambda l: (O1[s][q].get(l, 0) + O2[s][q].get(l, 0) + O3[s][q].get(l, 0)) / 3.0
            p_o[q] = {l: 0.5 * b13(l) + 0.5 * OSW[s][q].get(l, 0) for l in labs}
            p_n[q] = {l: 0.5 * p_prop[q].get(l, 0) + 0.5 * OSW[s][q].get(l, 0) for l in labs}
            p_m[q] = {l: 0.25 * b13(l) + 0.25 * p_prop[q].get(l, 0) + 0.5 * OSW[s][q].get(l, 0) for l in labs}
        s_old.append(eval_probs(p_o, truth)); b_old.append(eval_probs(blend_dict(p_o, SAB[s], 0.5), truth))
        s_new.append(eval_probs(p_n, truth)); b_new.append(eval_probs(blend_dict(p_n, SAB[s], 0.5), truth))
        s_mix.append(eval_probs(p_m, truth)); b_mix.append(eval_probs(blend_dict(p_m, SAB[s], 0.5), truth))

    print("\n=== DEV_BG_SW2152 HONEST CV SUMMARY (splits 0-7, 3 seeds) ===", flush=True)
    print(f"  dev_bg_sw2152 solo             : solo={np.mean(s_solo):.5f} (bg3=0.87570, hub=0.87670) | blend={np.mean(b_solo):.5f} (bg3=0.87960, hub=0.87995)", flush=True)
    print(f"  lnb34 (50%bg1-3 + 50%v10_sw)   : solo={np.mean(s_old):.5f} | blend={np.mean(b_old):.5f}", flush=True)
    print(f"  50%bg_sw2152 + 50%v10_sw       : solo={np.mean(s_new):.5f} ({np.mean(s_new)-np.mean(s_old):+.5f}, wins {(np.array(s_new)>np.array(s_old)).sum()}/8) | blend={np.mean(b_new):.5f} ({np.mean(b_new)-np.mean(b_old):+.5f}, wins {(np.array(b_new)>np.array(b_old)).sum()}/8)", flush=True)
    print(f"  25%bg1-3+25%bg_sw+50%v10_sw    : solo={np.mean(s_mix):.5f} ({np.mean(s_mix)-np.mean(s_old):+.5f}, wins {(np.array(s_mix)>np.array(s_old)).sum()}/8) | blend={np.mean(b_mix):.5f} ({np.mean(b_mix)-np.mean(b_old):+.5f}, wins {(np.array(b_mix)>np.array(b_old)).sum()}/8)", flush=True)

    # Full 16-split 3-seed test fit
    sc_raw = pickle.load(open(f"{H}/sift_pair_cache.pkl", "rb"))
    sc_bg = pickle.load(open(f"{H}/sift_bg_cache.pkl", "rb"))
    lnb_dir = f"{H}/lnbnn_np64"
    base_S = {sp: ScorerDir(lnb_dir, sp, zip(tr.id[tr.species == sp], tr.identity[tr.species == sp])) for sp in ["hyena", "leopard"]}
    PT, QT = pickle.load(open(f"{H}/test_feats.pkl", "rb"))
    PT, QT = add_lnbnn_test(PT, QT, tr, te)
    sc_test = {sp: (S, np.ones(len(S.gal), dtype=bool)) for sp, S in base_S.items()}
    PT, QT = add_bg_lnb(PT, QT, sp_te, sc_test)
    PT, QT = add_sift_bg_feats(PT, QT, tr, Xtr, te, Xte, sc_raw, sc_bg)
    PT, QT = add_fullframe_bg(PT, QT)
    PT, QT = add_hubness_feats(PT, QT, tr, Xtr, te, Xte)
    PT, QT = add_sw2152_feats(PT, QT, tr, Xtr, te, Xte)
    QT = add_lognorm(QT, norm_te)

    Pall = pd.concat([V2[s][0][PF1_BGSW + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_BGSW + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_BGSW, QF1_BGSW, seed=sd)
        p1 += mp.predict(PT[PF1_BGSW]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_BGSW]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()

    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)

    P2all = pd.concat([R1[s][0][PF2_BGSW + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_BGSW + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()
    p2, n2 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_BGSW, QF2_BGSW, seed=sd)
        p2 += mp.predict(PT[PF2_BGSW]) / len(SEEDS)
        n2 += mq.predict(QT[QF2_BGSW]) / len(SEEDS)
    PT["p2"] = p2; QT["pn2"] = n2

    probs_raw = to_probs_raw(PT, QT, T_CAL)
    probs_prop = propagate_probs(probs_raw, QT, Gt, alpha=0.30, kp_thr=2.0, cos_thr=0.72)
    pickle.dump(probs_raw, open(f"{H}/test_probs_dev_bg_sw2152.pkl", "wb"))
    pickle.dump(probs_prop, open(f"{H}/test_probs_dev_bg_sw2152_prop.pkl", "wb"))
    pickle.dump(OOF_PROBS, open(f"{H}/oof_probs_dev_bg_sw2152_prop.pkl", "wb"))

    for sp in ["hyena", "leopard"]:
        ids = te.id[te.species == sp].values
        pnew = np.mean([probs_prop[q].get(NEW, 0.0) for q in ids])
        n1_ = np.mean([max(probs_prop[q], key=probs_prop[q].get) == NEW for q in ids])
        print(f"test dev_bg_sw2152 {sp:7s}: mean_pnew={pnew:.4f} new@1={n1_:.4f}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
