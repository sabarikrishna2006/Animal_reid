"""Upgrade Dev's pipeline (dev_bgclean2 -> dev_bgclean3) with the Full-Frame Static Camera-Trap Killer:
Flags 0-deg rotation matches whose unique-ID LNBNN share (lnb_uimg_share = lnb_uimg / lnb_img) < 0.35
(92% FP rate on leopard, 100% FP rate on hyena, keeping 99.2%+ of true matches).
Reuses cached dev_bgclean2_v2.pkl (zero SIFT/LNBNN recomputation!), 100% leak-free, strictly gated on 8-split Honest CV."""
import os, gc, pickle, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar

from core import load, l2n, val_splits, NEW
import model2
from exp_kpg import PF1, QF1, PF2, QF2, others
from exp_kpg_v2 import KB_P_RICH, KB_Q_RICH, rich_feats
from exp_lnb_img import NEW_P, NEW_Q
from exp_blend_check import to_probs_raw, propagate_probs, eval_probs, blend_dict
from make_kpg_probs import add_lnbnn_test
from make_dev_bgclean import ScorerDir, add_bg_lnb, BG_P, BG_Q
from make_dev_bgclean2 import add_sift_bg_feats, SBG_P, SBG_Q

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")
SEEDS = [0, 1, 2]

SBG2_P = ["sift_s_clean2", "sift_h_clean2", "sift_is_bg2"]
SBG2_Q = ["sift_clean2_best", "sift_clean2_top1", "sift_top1_is_bg2"]

PF1_BG3 = PF1 + NEW_P + BG_P + SBG_P + SBG2_P
QF1_BG3 = QF1 + NEW_Q + BG_Q + SBG_Q + SBG2_Q
PF2_BG3 = PF2 + KB_P_RICH + NEW_P + BG_P + SBG_P + SBG2_P
QF2_BG3 = QF2 + KB_Q_RICH + NEW_Q + BG_Q + SBG_Q + SBG2_Q


def add_fullframe_bg(P, Q):
    P = P.copy(); Q = Q.copy()
    full_bg = (P["sift_rot_best"].values < 0.50) & (P["lnb_uimg_share"].values < 0.35) & (P["sift_s"].values >= 6)
    is_bg2 = (P["sift_is_bg"].values > 0.5) | full_bg
    P["sift_is_bg2"] = is_bg2.astype(np.float32)
    P["sift_s_clean2"] = np.where(is_bg2, 0.0, P["sift_s_clean"].values).astype(np.float32)
    P["sift_h_clean2"] = np.where(is_bg2, 0.0, P["sift_h_clean"].values).astype(np.float32)
    qmax = P.groupby("qid")["sift_s_clean2"].max()
    top0 = P[P["rank"] == 0].set_index("qid")
    Q["sift_clean2_best"] = qmax.reindex(Q.qid.values).fillna(0.0).values.astype(np.float32)
    Q["sift_clean2_top1"] = top0["sift_s_clean2"].reindex(Q.qid.values).fillna(0.0).values.astype(np.float32)
    Q["sift_top1_is_bg2"] = top0["sift_is_bg2"].reindex(Q.qid.values).fillna(0.0).values.astype(np.float32)
    return P, Q


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    sp_te = dict(zip(te.id, te.species))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}

    V2_prev = pickle.load(open(f"{H}/dev_bgclean2_v2.pkl", "rb"))
    V2 = {s: add_fullframe_bg(V2_prev[s][0], V2_prev[s][1]) for s in range(16)}
    del V2_prev; gc.collect()
    print(f"Added full-frame camera-trap features to all 16 splits in {time.time()-t0:.1f}s", flush=True)

    # Leak-free Round-1 OOF
    R1, r1_scores = {}, []
    for s in range(16):
        Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
        Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
        mp, mq = model2.fit(Ptr, Qtr, PF1_BG3, QF1_BG3, seed=0)
        Pv["p1"] = mp.predict(Pv[PF1_BG3]); Qv["pn1"] = mq.predict(Qv[QF1_BG3])
        if s < 8:
            r1_scores.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
        Pv, Qv = model2.round2_feats(Pv, Qv)
        R1[s] = rich_feats(Pv, Qv, G[s], scale=2.0)
    print(f"R1 MEAN (splits 0-7): {np.mean(r1_scores):.4f} (bg2 was 0.8634) ({time.time()-t0:.0f}s)", flush=True)

    # Leak-free Round-2 OOF (splits 0-7, 3 seeds)
    OOF, CAL = {}, []
    for s in range(8):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in SEEDS:
            mp, mq = model2.fit(Ptr, Qtr, PF2_BG3, QF2_BG3, seed=sd)
            p2 += mp.predict(Pv[PF2_BG3]) / len(SEEDS)
            n2 += mq.predict(Qv[QF2_BG3]) / len(SEEDS)
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
    OOF_PROBS = {}
    s3, b3, s_ens, b_ens = [], [], [], []
    for s in range(8):
        Pv, Qv = OOF[s]; truth = dict(zip(Qv.qid, Qv.truth))
        p_raw = to_probs_raw(Pv, Qv, T_CAL)
        p_prop = propagate_probs(p_raw, Qv, G[s], alpha=0.30, kp_thr=2.0, cos_thr=0.72)
        OOF_PROBS[s] = p_prop
        # 3-way avg(bg1, bg2, bg3)
        p_avg = {}
        for q in p_prop:
            labs = set(O1[s][q]) | set(O2[s][q]) | set(p_prop[q])
            p_avg[q] = {l: (O1[s][q].get(l, 0.0) + O2[s][q].get(l, 0.0) + p_prop[q].get(l, 0.0)) / 3.0 for l in labs}
        s3.append(eval_probs(p_prop, truth))
        b3.append(eval_probs(blend_dict(p_prop, SAB[s], 0.5), truth))
        s_ens.append(eval_probs(p_avg, truth))
        b_ens.append(eval_probs(blend_dict(p_avg, SAB[s], 0.5), truth))
    print("\n=== DEV_BGCLEAN3 HONEST CV SUMMARY (splits 0-7, 3 seeds) ===", flush=True)
    print(f"  dev_bgclean3         : solo={np.mean(s3):.5f} (bg2=0.87542) | blend={np.mean(b3):.5f} (bg2=0.87895)", flush=True)
    print(f"  avg(bg1, bg2, bg3)   : solo={np.mean(s_ens):.5f} (bg1+2=0.87591) | blend={np.mean(b_ens):.5f} (bg1+2=0.87925)", flush=True)

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

    Pall = pd.concat([V2[s][0][PF1_BG3 + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_BG3 + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_BG3, QF1_BG3, seed=sd)
        p1 += mp.predict(PT[PF1_BG3]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_BG3]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()

    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)

    P2all = pd.concat([R1[s][0][PF2_BG3 + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_BG3 + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()
    p2, n2 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_BG3, QF2_BG3, seed=sd)
        p2 += mp.predict(PT[PF2_BG3]) / len(SEEDS)
        n2 += mq.predict(QT[QF2_BG3]) / len(SEEDS)
    PT["p2"] = p2; QT["pn2"] = n2

    probs_raw = to_probs_raw(PT, QT, T_CAL)
    probs_prop = propagate_probs(probs_raw, QT, Gt, alpha=0.30, kp_thr=2.0, cos_thr=0.72)
    pickle.dump(probs_raw, open(f"{H}/test_probs_dev_bgclean3.pkl", "wb"))
    pickle.dump(probs_prop, open(f"{H}/test_probs_dev_bgclean3_prop.pkl", "wb"))
    pickle.dump(OOF_PROBS, open(f"{H}/oof_probs_dev_bgclean3_prop.pkl", "wb"))

    for sp in ["hyena", "leopard"]:
        ids = te.id[te.species == sp].values
        pnew = np.mean([probs_prop[q].get(NEW, 0.0) for q in ids])
        n1_ = np.mean([max(probs_prop[q], key=probs_prop[q].get) == NEW for q in ids])
        print(f"test dev_bgclean3 {sp:7s}: mean_pnew={pnew:.4f} new@1={n1_:.4f}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
