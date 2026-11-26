"""Upgrade Dev's pipeline (dev_bgclean3 -> dev_hub) with Different-Identity Gallery Hubness (r_g_diff):
For each gallery image g, r_g_diff = mean top-10 cosine of g to DIFFERENT-identity gallery images (100% gallery-only,
zero leakage!). Removes embedding hubness in both centered MiewID (+0.0160 leopard top-1, +0.0145 hyena AUC)
and Gallery-LDA (+0.0084 hyena / +0.0080 leopard top-1, +0.0081 / +0.0106 AUC).
Reuses cached dev_bgclean2_v2.pkl (zero SIFT/LNBNN recomputation!), monotone LightGBM only, strictly CV-gated."""
import os, gc, pickle, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar

from core import load, l2n, val_splits, NEW
from features import embed
import model2
from exp_kpg import others
from exp_kpg_v2 import rich_feats
from exp_blend_check import to_probs_raw, propagate_probs, eval_probs, blend_dict
from make_kpg_probs import add_lnbnn_test
from make_dev_bgclean import ScorerDir, add_bg_lnb
from make_dev_bgclean2 import add_sift_bg_feats
from make_dev_bgclean3 import add_fullframe_bg, PF1_BG3, QF1_BG3, PF2_BG3, QF2_BG3

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")
SEEDS = [0, 1, 2]

HUB_P = ["hub_cos", "hub_lda", "hub_cos_gap", "hub_lda_gap", "best_g_hub_cos", "best_g_hub_lda"]
HUB_Q = ["q_hub_cos_max", "q_hub_cos_gap", "q_hub_lda_max", "q_hub_lda_gap"]

PF1_HUB = PF1_BG3 + HUB_P
QF1_HUB = QF1_BG3 + HUB_Q
PF2_HUB = PF2_BG3 + HUB_P
QF2_HUB = QF2_BG3 + HUB_Q


def _rg_diff(Zg, glab, K=10):
    Sgg = Zg @ Zg.T
    Sgg_diff = np.where(glab[:, None] == glab[None, :], -2.0, Sgg)
    k = min(K, max(1, len(Zg) - 1))
    return np.partition(Sgg_diff, -k, axis=1)[:, -k:].mean(1).astype(np.float32)


def add_hubness_feats(P, Q, gal, Xg, qry, Xq):
    """100% gallery-only different-identity hubness features for MiewID & LDA."""
    pair_map = {}
    q_map = {}
    for sp in ["hyena", "leopard"]:
        gm = (gal.species == sp).values; qm = (qry.species == sp).values
        G = gal[gm].reset_index(drop=True); Qdf = qry[qm].reset_index(drop=True)
        glab = G.identity.values
        (cg, cq), (lg, lq) = embed(Xg[gm], glab, Xq[qm])
        rg_c = _rg_diff(cg, glab, K=10)
        rg_l = _rg_diff(lg, glab, K=10)
        Sc = cq @ cg.T; Sl = lq @ lg.T
        Hc = Sc - 0.50 * rg_c[None, :]
        Hl = Sl - 0.75 * rg_l[None, :]

        names, inv = np.unique(glab, return_inverse=True)
        nid = len(names)
        members = [np.where(inv == j)[0] for j in range(nid)]
        Ihc = np.full((len(Qdf), nid), -9.0, np.float32)
        Ihl = np.full((len(Qdf), nid), -9.0, np.float32)
        Brc = np.zeros((len(Qdf), nid), np.float32)
        Brl = np.zeros((len(Qdf), nid), np.float32)
        for j, cols in enumerate(members):
            sub_c = Hc[:, cols]; idx_c = sub_c.argmax(1)
            Ihc[:, j] = sub_c[np.arange(len(Qdf)), idx_c]
            Brc[:, j] = rg_c[cols[idx_c]]
            sub_l = Hl[:, cols]; idx_l = sub_l.argmax(1)
            Ihl[:, j] = sub_l[np.arange(len(Qdf)), idx_l]
            Brl[:, j] = rg_l[cols[idx_l]]

        # Query-level top-1 & gap
        sc_top2 = -np.sort(-Ihc, axis=1)[:, :2]
        sl_top2 = -np.sort(-Ihl, axis=1)[:, :2]
        q_ids = Qdf.id.values
        id_pos = {n: j for j, n in enumerate(names)}
        for i, qid in enumerate(q_ids):
            mc1 = float(sc_top2[i, 0]); mc2 = float(sc_top2[i, 1]) if nid > 1 else mc1
            ml1 = float(sl_top2[i, 0]); ml2 = float(sl_top2[i, 1]) if nid > 1 else ml1
            q_map[qid] = (mc1, mc1 - mc2, ml1, ml1 - ml2)
            pair_map[qid] = (id_pos, Ihc[i], Ihl[i], Brc[i], Brl[i], mc1, ml1)

    rows = []
    for q, c in zip(P.qid.values, P.cand.values):
        id_pos, ihc, ihl, brc, brl, mc1, ml1 = pair_map[q]
        j = id_pos.get(c, -1)
        if j >= 0:
            vc, vl = float(ihc[j]), float(ihl[j])
            rows.append((vc, vl, mc1 - vc, ml1 - vl, float(brc[j]), float(brl[j])))
        else:
            rows.append((-9.0, -9.0, 9.0, 9.0, 0.0, 0.0))
    P = P.copy(); Q = Q.copy()
    P[HUB_P] = np.array(rows, dtype=np.float32)
    Q[HUB_Q] = np.array([q_map[q] for q in Q.qid.values], dtype=np.float32)
    return P, Q


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    sp_te = dict(zip(te.id, te.species))
    splits = val_splits(tr, l2n(Xtr), range(16))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}

    V2_prev = pickle.load(open(f"{H}/dev_bgclean2_v2.pkl", "rb"))
    V2 = {}
    for s, (gal, q, truth) in enumerate(splits):
        Ps, Qs = add_fullframe_bg(V2_prev[s][0], V2_prev[s][1])
        V2[s] = add_hubness_feats(Ps, Qs, tr[gal].reset_index(drop=True), Xtr[gal], tr[q].reset_index(drop=True), Xtr[q])
    del V2_prev; gc.collect()
    print(f"Added full-frame + gallery-hubness features to all 16 splits in {time.time()-t0:.1f}s", flush=True)

    # Leak-free Round-1 OOF
    R1, r1_scores = {}, []
    for s in range(16):
        Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
        Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
        mp, mq = model2.fit(Ptr, Qtr, PF1_HUB, QF1_HUB, seed=0)
        Pv["p1"] = mp.predict(Pv[PF1_HUB]); Qv["pn1"] = mq.predict(Qv[QF1_HUB])
        if s < 8:
            r1_scores.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
        Pv, Qv = model2.round2_feats(Pv, Qv)
        R1[s] = rich_feats(Pv, Qv, G[s], scale=2.0)
    print(f"R1 MEAN (splits 0-7): {np.mean(r1_scores):.4f} (bg3 was 0.8641) ({time.time()-t0:.0f}s)", flush=True)

    # Leak-free Round-2 OOF (splits 0-7, 3 seeds)
    OOF, CAL = {}, []
    for s in range(8):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in SEEDS:
            mp, mq = model2.fit(Ptr, Qtr, PF2_HUB, QF2_HUB, seed=sd)
            p2 += mp.predict(Pv[PF2_HUB]) / len(SEEDS)
            n2 += mq.predict(Qv[QF2_HUB]) / len(SEEDS)
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
    sh, bh, s4, b4, s3h, b3h = [], [], [], [], [], []
    for s in range(8):
        Pv, Qv = OOF[s]; truth = dict(zip(Qv.qid, Qv.truth))
        p_raw = to_probs_raw(Pv, Qv, T_CAL)
        p_prop = propagate_probs(p_raw, Qv, G[s], alpha=0.30, kp_thr=2.0, cos_thr=0.72)
        OOF_PROBS[s] = p_prop
        sh.append(eval_probs(p_prop, truth))
        bh.append(eval_probs(blend_dict(p_prop, SAB[s], 0.5), truth))
        p4, p3h = {}, {}
        for q in p_prop:
            labs = set(O1[s][q]) | set(O2[s][q]) | set(O3[s][q]) | set(p_prop[q])
            p4[q] = {l: 0.25 * (O1[s][q].get(l, 0) + O2[s][q].get(l, 0) + O3[s][q].get(l, 0) + p_prop[q].get(l, 0)) for l in labs}
            p3h[q] = {l: 0.5 * O3[s][q].get(l, 0) + 0.5 * p_prop[q].get(l, 0) for l in labs}
        s4.append(eval_probs(p4, truth))
        b4.append(eval_probs(blend_dict(p4, SAB[s], 0.5), truth))
        s3h.append(eval_probs(p3h, truth))
        b3h.append(eval_probs(blend_dict(p3h, SAB[s], 0.5), truth))

    print("\n=== DEV_HUB HONEST CV SUMMARY (splits 0-7, 3 seeds) ===", flush=True)
    print(f"  dev_hub solo          : solo={np.mean(sh):.5f} (bg3=0.87571) | blend={np.mean(bh):.5f} (bg3=0.87956)", flush=True)
    print(f"  50/50(bg3, dev_hub)   : solo={np.mean(s3h):.5f} | blend={np.mean(b3h):.5f}", flush=True)
    print(f"  avg(bg1,2,3,dev_hub)  : solo={np.mean(s4):.5f} (was 0.87600) | blend={np.mean(b4):.5f} (was 0.87926)", flush=True)

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

    Pall = pd.concat([V2[s][0][PF1_HUB + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_HUB + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_HUB, QF1_HUB, seed=sd)
        p1 += mp.predict(PT[PF1_HUB]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_HUB]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()

    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)

    P2all = pd.concat([R1[s][0][PF2_HUB + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_HUB + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()
    p2, n2 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_HUB, QF2_HUB, seed=sd)
        p2 += mp.predict(PT[PF2_HUB]) / len(SEEDS)
        n2 += mq.predict(QT[QF2_HUB]) / len(SEEDS)
    PT["p2"] = p2; QT["pn2"] = n2

    probs_raw = to_probs_raw(PT, QT, T_CAL)
    probs_prop = propagate_probs(probs_raw, QT, Gt, alpha=0.30, kp_thr=2.0, cos_thr=0.72)
    pickle.dump(probs_raw, open(f"{H}/test_probs_dev_hub.pkl", "wb"))
    pickle.dump(probs_prop, open(f"{H}/test_probs_dev_hub_prop.pkl", "wb"))
    pickle.dump(OOF_PROBS, open(f"{H}/oof_probs_dev_hub_prop.pkl", "wb"))

    for sp in ["hyena", "leopard"]:
        ids = te.id[te.species == sp].values
        pnew = np.mean([probs_prop[q].get(NEW, 0.0) for q in ids])
        n1_ = np.mean([max(probs_prop[q], key=probs_prop[q].get) == NEW for q in ids])
        print(f"test dev_hub {sp:7s}: mean_pnew={pnew:.4f} new@1={n1_:.4f}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
