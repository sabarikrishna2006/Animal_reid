"""Upgrade Dev's v10 pipeline (val_v10_0..15.pkl + test_v10.pkl, which includes lnb_img top-5 candidate expansion +
lnb_img-guided SIFT + k-reciprocal) with ALL 4 proven zero-leakage discoveries:
  1. High-recall nprobe=64 LNBNN + lnb_uimg + lnb_rimg (wipes out shared camera-trap background keypoints)
  2. Stage-1 Corner Camera-Trap SIFT Killer (abs_rot < 0.50 & spread_min < 0.25) from cached sift_bg_cache.pkl
  3. Stage-2 Full-Frame Camera-Trap SIFT Killer (abs_rot < 0.50 & lnb_uimg_share < 0.35)
  4. Different-Identity Gallery Hubness (r_g_diff) in MiewID & Gallery-LDA (100% gallery-only)
Fixes the Missing-Candidate Veto Bug (17 high-confidence Sabari matches absent from Dev's top-20 MiewID list)
WITHOUT the camera-trap false positives that hurt earlier dev9 (lnb9).
Evaluates leak-free 8-split Honest CV + Shared Blend CV, then fits 16 splits x 3 seeds -> test_probs_dev_v10_fullbg_prop.pkl."""
import os, gc, pickle, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar

from core import load, l2n, val_splits, NEW
from features import embed
import model2
from exp_kpg import PF1, QF1, PF2, QF2, others
from exp_kpg_v2 import KB_P_RICH, KB_Q_RICH, rich_feats
from exp_lnb_img import NEW_P, NEW_Q
from exp_blend_check import to_probs_raw, propagate_probs, eval_probs, blend_dict
from build_dev_v10 import KREC_P, KREC_Q, K_CAND, K_SIFT, IMG_PER_ID, DEEP_R, DEEP_IMG, LNB_CAND_K
from make_dev_bgclean import ScorerDir, add_bg_lnb, BG_P, BG_Q
from make_dev_bgclean2 import SBG_P, SBG_Q
from make_dev_bgclean3 import add_fullframe_bg, SBG2_P, SBG2_Q
from make_dev_hubness import add_hubness_feats, HUB_P, HUB_Q

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")
SEEDS = [0, 1, 2]

PF1_ALL = PF1 + NEW_P + KREC_P + BG_P + SBG_P + SBG2_P + HUB_P
QF1_ALL = QF1 + NEW_Q + KREC_Q + BG_Q + SBG_Q + SBG2_Q + HUB_Q
PF2_ALL = PF2 + KB_P_RICH + NEW_P + KREC_P + BG_P + SBG_P + SBG2_P + HUB_P
QF2_ALL = QF2 + KB_Q_RICH + NEW_Q + KREC_Q + BG_Q + SBG_Q + SBG2_Q + HUB_Q


def add_sift_bg_v10(P, Q, gal, Xg, qry, Xq, sc_raw, sc_bg):
    """Lookup camera-trap SIFT geometry for every (qid, cand) in v10 tables using cached sift_bg_cache.pkl."""
    n_img = lambda r: DEEP_IMG if r < DEEP_R else IMG_PER_ID
    pair_feats = {}
    q_best = {}
    # Group P candidates per qid so we also cover lnb_img-expanded candidates (rank >= K_CAND)
    cands_by_q = P.groupby("qid")["cand"].apply(list).to_dict()
    for sp in ["hyena", "leopard"]:
        gm = (gal.species == sp).values; qm = (qry.species == sp).values
        G = gal[gm].reset_index(drop=True); Qdf = qry[qm].reset_index(drop=True)
        (cg, cq), _ = embed(Xg[gm], G.identity.values, Xq[qm])
        Sc = cq @ cg.T
        names, inv = np.unique(G.identity.values, return_inverse=True)
        id_pos = {n: j for j, n in enumerate(names)}
        members = [np.where(inv == j)[0] for j in range(len(names))]
        g_ids = G.id.values; q_ids = Qdf.id.values
        for i, qid in enumerate(q_ids):
            cands = cands_by_q.get(qid, [])
            best_clean_q = 0.0
            for r, cname in enumerate(cands):
                j = id_pos.get(cname, -1)
                if j < 0:
                    pair_feats[(qid, cname)] = (-1.0, -1.0, 0.0, -1.0, 99.0, 1.0)
                    continue
                cols = members[j]
                # Check all gallery images of this candidate that exist in sc_raw (covers both MiewID top and lnb_img-guided top!)
                s_clean_max = 0.0; h_clean_max = 0.0; hits_clean = 0; found_any = False
                best_raw = -1.0; best_is_bg = 0.0; best_rot = 99.0; best_sp = 1.0
                for t in cols:
                    key = tuple(sorted((qid, g_ids[t])))
                    raw = sc_raw.get(key)
                    if raw is None:
                        continue
                    found_any = True
                    nh, na = float(raw[1]), float(raw[2])
                    bg = sc_bg.get(key)
                    is_bg = int(bg[3]) if bg is not None else 0
                    rot = float(bg[1]) if bg is not None else 99.0
                    sp_m = float(bg[2]) if bg is not None else 1.0
                    if na > best_raw:
                        best_raw = na; best_is_bg = float(is_bg); best_rot = rot; best_sp = sp_m
                    if not is_bg:
                        if na > s_clean_max: s_clean_max = na
                        if nh > h_clean_max: h_clean_max = nh
                        if na >= 8.0: hits_clean += 1
                if not found_any:
                    pair_feats[(qid, cname)] = (-1.0, -1.0, 0.0, -1.0, 99.0, 1.0)
                else:
                    pair_feats[(qid, cname)] = (s_clean_max, h_clean_max, best_is_bg, float(hits_clean), best_rot, best_sp)
                    if s_clean_max > best_clean_q:
                        best_clean_q = s_clean_max
            q_best[qid] = best_clean_q

    P = P.copy(); Q = Q.copy()
    arr = np.array([pair_feats.get((q, c), (-1.0, -1.0, 0.0, -1.0, 99.0, 1.0))
                    for q, c in zip(P.qid.values, P.cand.values)], dtype=np.float32)
    P[SBG_P] = arr
    top0 = P[P["rank"] == 0].set_index("qid")
    Q["sift_clean_best"] = np.array([q_best.get(q, 0.0) for q in Q.qid.values], dtype=np.float32)
    Q["sift_clean_top1"] = top0["sift_s_clean"].reindex(Q.qid.values).fillna(0.0).values.astype(np.float32)
    Q["sift_top1_is_bg"] = top0["sift_is_bg"].reindex(Q.qid.values).fillna(0.0).values.astype(np.float32)
    return P, Q


def upgrade_table(P0, Q0, gal, Xg, qry, Xq, sp_map, scorers, sc_raw, sc_bg):
    # Drop old nprobe=16 NEW_P / NEW_Q before recomputing with nprobe=64 + lnb_uimg + lnb_rimg
    P = P0.drop(columns=[c for c in NEW_P if c in P0.columns]).copy()
    Q = Q0.drop(columns=[c for c in NEW_Q if c in Q0.columns]).copy()
    P, Q = add_bg_lnb(P, Q, sp_map, scorers)
    P, Q = add_sift_bg_v10(P, Q, gal, Xg, qry, Xq, sc_raw, sc_bg)
    P, Q = add_fullframe_bg(P, Q)
    P, Q = add_hubness_feats(P, Q, gal, Xg, qry, Xq)
    return P, Q


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    sp_tr = dict(zip(tr.id, tr.species))
    sp_te = dict(zip(te.id, te.species))
    splits = val_splits(tr, l2n(Xtr), range(16))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}
    sc_raw = pickle.load(open(f"{H}/sift_pair_cache.pkl", "rb"))
    sc_bg = pickle.load(open(f"{H}/sift_bg_cache.pkl", "rb"))
    lnb_dir = f"{H}/lnbnn_np64"
    base_S = {sp: ScorerDir(lnb_dir, sp, zip(tr.id[tr.species == sp], tr.identity[tr.species == sp])) for sp in ["hyena", "leopard"]}

    v2_path = f"{H}/dev_v10_fullbg_v2.pkl"
    if os.path.exists(v2_path):
        V2 = pickle.load(open(v2_path, "rb"))
        print(f"Loaded cached {v2_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        V2 = {}
        for s, (gal, q, truth) in enumerate(splits):
            P0, Q0 = pickle.load(open(f"{H}/val_v10_{s}.pkl", "rb"))
            sc_s = {sp: (base_S[sp], np.isin(base_S[sp].gal, tr.id[gal].values)) for sp in base_S}
            V2[s] = upgrade_table(P0, Q0, tr[gal].reset_index(drop=True), Xtr[gal],
                                  tr[q].reset_index(drop=True), Xtr[q], sp_tr, sc_s, sc_raw, sc_bg)
            if (s + 1) % 4 == 0:
                print(f"  upgraded {s+1}/16 splits ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(V2, open(v2_path, "wb"), protocol=4)
        print(f"Cached {v2_path} ({time.time()-t0:.0f}s)", flush=True)

    # Leak-free Round-1 OOF
    r1_path = f"{H}/dev_v10_fullbg_r1.pkl"
    if os.path.exists(r1_path):
        R1 = pickle.load(open(r1_path, "rb"))
        print(f"Loaded cached {r1_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        R1, r1_scores = {}, []
        for s in range(16):
            Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
            Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
            mp, mq = model2.fit(Ptr, Qtr, PF1_ALL, QF1_ALL, seed=0)
            Pv["p1"] = mp.predict(Pv[PF1_ALL]); Qv["pn1"] = mq.predict(Qv[QF1_ALL])
            if s < 8:
                r1_scores.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
            Pv, Qv = model2.round2_feats(Pv, Qv)
            R1[s] = rich_feats(Pv, Qv, G[s], scale=2.0)
        pickle.dump(R1, open(r1_path, "wb"), protocol=4)
        print(f"R1 MEAN (splits 0-7): {np.mean(r1_scores):.4f} (dev_hub was 0.8659, dev9 was 0.8698) ({time.time()-t0:.0f}s)", flush=True)

    # Leak-free Round-2 OOF (splits 0-7, 3 seeds)
    OOF, CAL = {}, []
    for s in range(8):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in SEEDS:
            mp, mq = model2.fit(Ptr, Qtr, PF2_ALL, QF2_ALL, seed=sd)
            p2 += mp.predict(Pv[PF2_ALL]) / len(SEEDS)
            n2 += mq.predict(Qv[QF2_ALL]) / len(SEEDS)
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
    OOF_PROBS = {}
    sv, bv, sm, bm = [], [], [], []
    for s in range(8):
        Pv, Qv = OOF[s]; truth = dict(zip(Qv.qid, Qv.truth))
        p_raw = to_probs_raw(Pv, Qv, T_CAL)
        p_prop = propagate_probs(p_raw, Qv, G[s], alpha=0.30, kp_thr=2.0, cos_thr=0.72)
        OOF_PROBS[s] = p_prop
        sv.append(eval_probs(p_prop, truth))
        bv.append(eval_probs(blend_dict(p_prop, SAB[s], 0.5), truth))
        # Mix 50% avg(bg1,bg2,bg3) + 50% dev_v10_fullbg (diversity + rescued candidates!)
        pmix = {}
        for q in p_prop:
            labs = set(O1[s][q]) | set(O2[s][q]) | set(O3[s][q]) | set(p_prop[q])
            pmix[q] = {l: (O1[s][q].get(l, 0) + O2[s][q].get(l, 0) + O3[s][q].get(l, 0)) / 6.0 + 0.5 * p_prop[q].get(l, 0) for l in labs}
        sm.append(eval_probs(pmix, truth))
        bm.append(eval_probs(blend_dict(pmix, SAB[s], 0.5), truth))

    print("\n=== DEV_V10_FULLBG HONEST CV SUMMARY (splits 0-7, 3 seeds) ===", flush=True)
    print(f"  dev_v10_fullbg solo            : solo={np.mean(sv):.5f} | blend={np.mean(bv):.5f} (dev9 was 0.8786 / 0.8826)", flush=True)
    print(f"  50% avg(bg1-3) + 50% v10_fullbg: solo={np.mean(sm):.5f} | blend={np.mean(bm):.5f} (bg1-3 was 0.8760 / 0.8793)", flush=True)

    # Full 16-split 3-seed test fit
    PT0, QT0 = pickle.load(open(f"{H}/test_v10.pkl", "rb"))
    sc_test = {sp: (S, np.ones(len(S.gal), dtype=bool)) for sp, S in base_S.items()}
    PT, QT = upgrade_table(PT0, QT0, tr, Xtr, te, Xte, sp_te, sc_test, sc_raw, sc_bg)

    Pall = pd.concat([V2[s][0][PF1_ALL + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_ALL + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_ALL, QF1_ALL, seed=sd)
        p1 += mp.predict(PT[PF1_ALL]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_ALL]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()

    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)

    P2all = pd.concat([R1[s][0][PF2_ALL + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_ALL + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()
    p2, n2 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_ALL, QF2_ALL, seed=sd)
        p2 += mp.predict(PT[PF2_ALL]) / len(SEEDS)
        n2 += mq.predict(QT[QF2_ALL]) / len(SEEDS)
    PT["p2"] = p2; QT["pn2"] = n2

    probs_raw = to_probs_raw(PT, QT, T_CAL)
    probs_prop = propagate_probs(probs_raw, QT, Gt, alpha=0.30, kp_thr=2.0, cos_thr=0.72)
    pickle.dump(probs_raw, open(f"{H}/test_probs_dev_v10_fullbg.pkl", "wb"))
    pickle.dump(probs_prop, open(f"{H}/test_probs_dev_v10_fullbg_prop.pkl", "wb"))
    pickle.dump(OOF_PROBS, open(f"{H}/oof_probs_dev_v10_fullbg_prop.pkl", "wb"))

    for sp in ["hyena", "leopard"]:
        ids = te.id[te.species == sp].values
        pnew = np.mean([probs_prop[q].get(NEW, 0.0) for q in ids])
        n1_ = np.mean([max(probs_prop[q], key=probs_prop[q].get) == NEW for q in ids])
        print(f"test dev_v10_fullbg {sp:7s}: mean_pnew={pnew:.4f} new@1={n1_:.4f}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
