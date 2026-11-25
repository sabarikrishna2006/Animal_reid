"""Upgrade Dev's pipeline (dev_bgclean -> dev_bgclean2) with the 2nd half of our Camera-Trap Background Killer:
Adds camera-trap SIFT geometric background filtering (sift_s_clean, sift_h_clean, sift_is_bg, sift_hits_clean,
sift_rot_best, sift_spread_best + query-level sift_clean_best, sift_clean_top1, sift_top1_is_bg) on top of
dev_bgclean's cached dev_bgclean_v2.pkl (which already has np64 + lnb_uimg + lnb_rimg).
100% leak-free, zero external data, strictly gated on 8-split Honest CV + Shared Blend CV."""
import os, gc, json, pickle, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar

from core import load, l2n, val_splits, NEW
from features import embed, K_CAND, K_SIFT, IMG_PER_ID, DEEP_R, DEEP_IMG
import model2
from exp_kpg import PF1, QF1, PF2, QF2, others
from exp_kpg_v2 import KB_P_RICH, KB_Q_RICH, rich_feats
from exp_lnb_img import NEW_P, NEW_Q
from exp_blend_check import to_probs_raw, propagate_probs, eval_probs, blend_dict
from make_kpg_probs import add_lnbnn_test
from make_dev_bgclean import ScorerDir, add_bg_lnb, BG_P, BG_Q

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")
SEEDS = [0, 1, 2]

SBG_P = ["sift_s_clean", "sift_h_clean", "sift_is_bg", "sift_hits_clean", "sift_rot_best", "sift_spread_best"]
SBG_Q = ["sift_clean_best", "sift_clean_top1", "sift_top1_is_bg"]

PF1_BG2 = PF1 + NEW_P + BG_P + SBG_P
QF1_BG2 = QF1 + NEW_Q + BG_Q + SBG_Q
PF2_BG2 = PF2 + KB_P_RICH + NEW_P + BG_P + SBG_P
QF2_BG2 = QF2 + KB_Q_RICH + NEW_Q + BG_Q + SBG_Q


def add_sift_bg_feats(P, Q, gal, Xg, qry, Xq, sc_raw, sc_bg):
    """For each (qid, cand) in P, inspect the exact same top gallery images checked by features.py
    and compute camera-trap background-filtered SIFT features (100% leak-free using only gal)."""
    n_img = lambda r: DEEP_IMG if r < DEEP_R else IMG_PER_ID
    pair_feats = {}
    q_best = {}
    for sp in ["hyena", "leopard"]:
        gm = (gal.species == sp).values; qm = (qry.species == sp).values
        G = gal[gm].reset_index(drop=True); Qdf = qry[qm].reset_index(drop=True)
        (cg, cq), (lg, lq) = embed(Xg[gm], G.identity.values, Xq[qm])
        Sc = cq @ cg.T; Sl = lq @ lg.T
        names, inv = np.unique(G.identity.values, return_inverse=True)
        nid = len(names)
        members = [np.where(inv == j)[0] for j in range(nid)]
        Ic = np.full((len(Qdf), nid), -9, np.float32); Il = Ic.copy()
        for j, cols in enumerate(members):
            Ic[:, j] = Sc[:, cols].max(1); Il[:, j] = Sl[:, cols].max(1)
        comb = Ic + Il
        cand = np.argsort(-comb, 1)[:, :K_CAND]
        g_ids = G.id.values; q_ids = Qdf.id.values
        for i in range(len(Qdf)):
            qid = q_ids[i]
            best_clean_q = 0.0
            for r, j in enumerate(cand[i]):
                cname = names[j]
                if r >= K_SIFT:
                    pair_feats[(qid, cname)] = (-1.0, -1.0, 0.0, -1.0, 99.0, 1.0)
                    continue
                cols = members[j]
                top = cols[np.argsort(-Sc[i, cols])[:n_img(r)]]
                s_clean_max = 0.0; h_clean_max = 0.0; hits_clean = 0
                best_raw = -1.0; best_is_bg = 0.0; best_rot = 99.0; best_sp = 1.0
                for t in top:
                    key = tuple(sorted((qid, g_ids[t])))
                    raw = sc_raw.get(key)
                    if raw is None:
                        continue
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
                pair_feats[(qid, cname)] = (s_clean_max, h_clean_max, best_is_bg, float(hits_clean), best_rot, best_sp)
                if s_clean_max > best_clean_q:
                    best_clean_q = s_clean_max
            q_best[qid] = best_clean_q

    P = P.copy()
    arr = np.array([pair_feats.get((q, c), (-1.0, -1.0, 0.0, -1.0, 99.0, 1.0))
                    for q, c in zip(P.qid.values, P.cand.values)], dtype=np.float32)
    P[SBG_P] = arr
    top0 = P[P["rank"] == 0].set_index("qid")
    Q = Q.copy()
    Q["sift_clean_best"] = np.array([q_best.get(q, 0.0) for q in Q.qid.values], dtype=np.float32)
    Q["sift_clean_top1"] = top0["sift_s_clean"].reindex(Q.qid.values).fillna(0.0).values.astype(np.float32)
    Q["sift_top1_is_bg"] = top0["sift_is_bg"].reindex(Q.qid.values).fillna(0.0).values.astype(np.float32)
    return P, Q


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    sp_te = dict(zip(te.id, te.species))
    splits = val_splits(tr, l2n(Xtr), range(16))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}
    sc_raw = pickle.load(open(f"{H}/sift_pair_cache.pkl", "rb"))
    sc_bg = pickle.load(open(f"{H}/sift_bg_cache.pkl", "rb"))

    v2_path = f"{H}/dev_bgclean2_v2.pkl"
    if os.path.exists(v2_path):
        V2 = pickle.load(open(v2_path, "rb"))
        print(f"Loaded cached {v2_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        V2_prev = pickle.load(open(f"{H}/dev_bgclean_v2.pkl", "rb"))
        V2 = {}
        for s, (gal, q, truth) in enumerate(splits):
            V2[s] = add_sift_bg_feats(V2_prev[s][0], V2_prev[s][1], tr[gal].reset_index(drop=True), Xtr[gal],
                                      tr[q].reset_index(drop=True), Xtr[q], sc_raw, sc_bg)
        del V2_prev; gc.collect()
        pickle.dump(V2, open(v2_path, "wb"), protocol=4)
        print(f"Added SIFT bg-clean features to all 16 splits in {time.time()-t0:.1f}s", flush=True)

    # Leak-free Round-1 OOF
    r1_path = f"{H}/dev_bgclean2_r1.pkl"
    if os.path.exists(r1_path):
        R1 = pickle.load(open(r1_path, "rb"))
        print(f"Loaded cached {r1_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        R1, r1_scores = {}, []
        for s in range(16):
            Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
            Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
            mp, mq = model2.fit(Ptr, Qtr, PF1_BG2, QF1_BG2, seed=0)
            Pv["p1"] = mp.predict(Pv[PF1_BG2]); Qv["pn1"] = mq.predict(Qv[QF1_BG2])
            if s < 8:
                r1_scores.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
            Pv, Qv = model2.round2_feats(Pv, Qv)
            R1[s] = rich_feats(Pv, Qv, G[s], scale=2.0)
        pickle.dump(R1, open(r1_path, "wb"), protocol=4)
        print(f"R1 MEAN (splits 0-7): {np.mean(r1_scores):.4f} (dev_bgclean was 0.8627) ({time.time()-t0:.0f}s)", flush=True)

    # Leak-free Round-2 OOF (splits 0-7, 3 seeds) + head-to-head vs dev_bgclean
    OOF, CAL = {}, []
    for s in range(8):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in SEEDS:
            mp, mq = model2.fit(Ptr, Qtr, PF2_BG2, QF2_BG2, seed=sd)
            p2 += mp.predict(Pv[PF2_BG2]) / len(SEEDS)
            n2 += mq.predict(Qv[QF2_BG2]) / len(SEEDS)
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
    PREV = pickle.load(open(f"{H}/oof_probs_dev_bgclean_prop.pkl", "rb"))
    res = {k: [] for k in ["prev_solo", "new_solo", "prev_blend", "new_blend"]}
    OOF_PROBS = {}
    for s in range(8):
        Pv, Qv = OOF[s]; truth = dict(zip(Qv.qid, Qv.truth))
        p_raw = to_probs_raw(Pv, Qv, T_CAL)
        p_prop = propagate_probs(p_raw, Qv, G[s], alpha=0.30, kp_thr=2.0, cos_thr=0.72)
        OOF_PROBS[s] = p_prop
        res["prev_solo"].append(eval_probs(PREV[s], truth))
        res["new_solo"].append(eval_probs(p_prop, truth))
        res["prev_blend"].append(eval_probs(blend_dict(PREV[s], SAB[s], 0.5), truth))
        res["new_blend"].append(eval_probs(blend_dict(p_prop, SAB[s], 0.5), truth))
        print(f"split {s}: solo {res['prev_solo'][-1]:.4f}->{res['new_solo'][-1]:.4f} | "
              f"blend {res['prev_blend'][-1]:.4f}->{res['new_blend'][-1]:.4f}", flush=True)

    ps, ns = np.array(res["prev_solo"]), np.array(res["new_solo"])
    pb, nb = np.array(res["prev_blend"]), np.array(res["new_blend"])
    print("\n=== DEV_BGCLEAN2 HONEST CV SUMMARY (splits 0-7, 3 seeds) ===", flush=True)
    print(f"  solo  : {ps.mean():.4f} -> {ns.mean():.4f} ({(ns-ps).mean():+.4f}, wins {(ns>ps).sum()}/8)", flush=True)
    print(f"  blend : {pb.mean():.4f} -> {nb.mean():.4f} ({(nb-pb).mean():+.4f}, wins {(nb>pb).sum()}/8)", flush=True)

    # Full 16-split 3-seed test fit
    lnb_dir = f"{H}/lnbnn_np64"
    base_S = {sp: ScorerDir(lnb_dir, sp, zip(tr.id[tr.species == sp], tr.identity[tr.species == sp])) for sp in ["hyena", "leopard"]}
    PT, QT = pickle.load(open(f"{H}/test_feats.pkl", "rb"))
    PT, QT = add_lnbnn_test(PT, QT, tr, te)
    sc_test = {sp: (S, np.ones(len(S.gal), dtype=bool)) for sp, S in base_S.items()}
    PT, QT = add_bg_lnb(PT, QT, sp_te, sc_test)
    PT, QT = add_sift_bg_feats(PT, QT, tr, Xtr, te, Xte, sc_raw, sc_bg)

    Pall = pd.concat([V2[s][0][PF1_BG2 + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_BG2 + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_BG2, QF1_BG2, seed=sd)
        p1 += mp.predict(PT[PF1_BG2]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_BG2]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()

    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)

    P2all = pd.concat([R1[s][0][PF2_BG2 + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_BG2 + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()
    p2, n2 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_BG2, QF2_BG2, seed=sd)
        p2 += mp.predict(PT[PF2_BG2]) / len(SEEDS)
        n2 += mq.predict(QT[QF2_BG2]) / len(SEEDS)
    PT["p2"] = p2; QT["pn2"] = n2

    probs_raw = to_probs_raw(PT, QT, T_CAL)
    probs_prop = propagate_probs(probs_raw, QT, Gt, alpha=0.30, kp_thr=2.0, cos_thr=0.72)
    pickle.dump(probs_raw, open(f"{H}/test_probs_dev_bgclean2.pkl", "wb"))
    pickle.dump(probs_prop, open(f"{H}/test_probs_dev_bgclean2_prop.pkl", "wb"))
    pickle.dump(OOF_PROBS, open(f"{H}/oof_probs_dev_bgclean2_prop.pkl", "wb"))

    for sp in ["hyena", "leopard"]:
        ids = te.id[te.species == sp].values
        pnew = np.mean([probs_prop[q].get(NEW, 0.0) for q in ids])
        n1_ = np.mean([max(probs_prop[q], key=probs_prop[q].get) == NEW for q in ids])
        print(f"test dev_bgclean2 {sp:7s}: mean_pnew={pnew:.4f} new@1={n1_:.4f}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
