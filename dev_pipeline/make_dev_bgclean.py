"""Upgrade Dev's pipeline (dev7 -> dev_bgclean) with our Camera-Trap Background Killer (100% leak-free):
1. High-recall nprobe=64 LNBNN store (solution/lnbnn_np64) for lnb_img, lnb_k3, AND background-suppressed LNBNN.
2. lnb_uimg: 2nd-different-identity margin max(0, d_id2 - d) aggregated per gallery image -> max over identity.
   Wipes out static camera-trap background keypoints shared across multiple animals!
3. lnb_rimg: strict cross-identity Lowe ratio (d < 0.85 * d_id2) image-level margin.
4. Keeps dev7's EXACT candidate pool (preserving 100% Dev <-> Sabari diversity!).
Evaluates leak-free 8-split Honest CV + Shared Blend CV, then fits 16 splits x 3 seeds -> test_probs_dev_bgclean_prop.pkl."""
import os, gc, json, pickle, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar

from core import load, l2n, val_splits, NEW
from lnbnn import Scorer
import model2
from exp_kpg import PF1, QF1, PF2, QF2, others
from exp_kpg_v2 import KB_P_RICH, KB_Q_RICH, rich_feats
from exp_lnb_img import NEW_P, NEW_Q
from exp_blend_check import to_probs_raw, propagate_probs, eval_probs, blend_dict
from make_kpg_probs import add_lnbnn_test

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")
OLD_CV = os.path.join(H, "probs_lnb", "cv_probs.pkl")
SEEDS = [0, 1, 2]

BG_P = ["lnb_uimg", "lnb_uimg_rank", "lnb_uimg_rel", "lnb_uimg_log", "lnb_uimg_share", "lnb_rimg", "lnb_rimg_rel"]
BG_Q = ["lnb_uimg_max", "lnb_uimg_gap", "lnb_uimg_agree", "lnb_uimg_over_img", "lnb_rimg_max", "lnb_rimg_gap"]

PF1_BG = PF1 + NEW_P + BG_P
QF1_BG = QF1 + NEW_Q + BG_Q
PF2_BG = PF2 + KB_P_RICH + NEW_P + BG_P
QF2_BG = QF2 + KB_Q_RICH + NEW_Q + BG_Q


class ScorerDir(Scorer):
    def __init__(self, d, sp, gallery_labels):
        m = json.load(open(f"{d}/{sp}_meta.json"))
        self.gal = np.array(m["gallery"])
        self.qpos = {q: i for i, q in enumerate(m["queries"])}
        self.qoff = np.array(m["qoff"])
        self.NB = np.load(f"{d}/{sp}_nb.npy", mmap_mode="r")
        self.DI = np.load(f"{d}/{sp}_d.npy", mmap_mode="r")
        lab = dict(gallery_labels)
        self.names, self.lab = np.unique([lab.get(g, "?") for g in self.gal], return_inverse=True)
        self.gpos = {g: i for i, g in enumerate(self.gal)}


def scores_all_bg(S, qid, gmask):
    """Returns (out_img, out_k3, out_uimg, out_rimg) over S.names using nprobe=64 stored NN lists."""
    i = S.qpos[qid]; a, b = S.qoff[i], S.qoff[i + 1]
    n_id, n_gal = len(S.names), len(S.gal)
    out_img = np.zeros(n_id, np.float32)
    out_k3 = np.zeros(n_id, np.float32)
    out_uimg = np.zeros(n_id, np.float32)
    out_rimg = np.zeros(n_id, np.float32)
    if b == a:
        return out_img, out_k3, out_uimg, out_rimg
    nb = np.asarray(S.NB[a:b]); d = np.asarray(S.DI[a:b], np.float32)
    ok = (nb >= 0) & gmask[np.maximum(nb, 0)]
    if qid in S.gpos:
        ok &= nb != S.gpos[qid]
    rank = np.cumsum(ok, 1)

    # 1. Image-level LNBNN (K=10)
    has_bg10 = rank[:, -1] >= 11
    bg10 = np.where(has_bg10, d[np.arange(len(d)), np.argmax(rank >= 11, 1)], d.max(1))
    r, c = np.nonzero(ok & (rank <= 10))
    if len(r):
        g_idx = nb[r, c]
        _, first = np.unique(r.astype(np.int64) * n_gal + g_idx, return_index=True)
        img_sc = np.zeros(n_gal, np.float32)
        np.add.at(img_sc, g_idx[first], np.maximum(bg10[r[first]] - d[r[first], c[first]], 0.0))
        np.maximum.at(out_img, S.lab, img_sc)

    # 2. Strict K=3 identity-level LNBNN
    has_bg3 = rank[:, -1] >= 4
    bg3 = np.where(has_bg3, d[np.arange(len(d)), np.argmax(rank >= 4, 1)], d.max(1))
    r, c = np.nonzero(ok & (rank <= 3))
    if len(r):
        lab = S.lab[nb[r, c]]
        _, first = np.unique(r.astype(np.int64) * n_id + lab, return_index=True)
        np.add.at(out_k3, lab[first], np.maximum(bg3[r[first]] - d[r[first], c[first]], 0.0))

    # 3. Camera-Trap Background-Suppressed LNBNN (lnb_uimg & lnb_rimg)
    has1 = rank[:, -1] >= 1
    c1 = np.argmax(ok, 1)
    rows_all = np.arange(len(d))
    g1 = nb[rows_all, c1]
    id1 = np.where(has1, S.lab[np.maximum(g1, 0)], -1)
    lab_mat = S.lab[np.maximum(nb, 0)]
    ok_diff = ok & (lab_mat != id1[:, None])
    has2 = ok_diff.any(1)
    c2 = np.argmax(ok_diff, 1)
    d2 = np.where(has2, d[rows_all, c2], d.max(1))

    ok_same = ok & (lab_mat == id1[:, None]) & (d < d2[:, None]) & has2[:, None]
    rs, cs = np.nonzero(ok_same)
    if len(rs):
        gs = nb[rs, cs]
        _, f_g = np.unique(rs.astype(np.int64) * n_gal + gs, return_index=True)
        rs1, cs1, gs1 = rs[f_g], cs[f_g], gs[f_g]
        v_m = np.maximum(d2[rs1] - d[rs1, cs1], 0.0)
        img_u = np.zeros(n_gal, np.float32)
        np.add.at(img_u, gs1, v_m)
        np.maximum.at(out_uimg, S.lab, img_u)
        v_r = np.where(d[rs1, cs1] < 0.85 * d2[rs1], v_m, 0.0)
        img_r = np.zeros(n_gal, np.float32)
        np.add.at(img_r, gs1, v_r)
        np.maximum.at(out_rimg, S.lab, img_r)

    return out_img, out_k3, out_uimg, out_rimg


def add_bg_lnb(P, Q, sp_of, scorers):
    pair, qrow = {}, []
    for qid in Q.qid:
        S, gm = scorers[sp_of[qid]]
        im, k3, uim, rim = scores_all_bg(S, qid, gm)
        o_im = np.argsort(-im); mx_im = im[o_im[0]]; m2_im = im[o_im[1]] if len(im) > 1 else 0.0
        rk_im = np.empty(len(im), int); rk_im[o_im] = np.arange(len(im))
        o_k3 = np.argsort(-k3); mx_k3 = k3[o_k3[0]]; m2_k3 = k3[o_k3[1]] if len(k3) > 1 else 0.0
        o_ui = np.argsort(-uim); mx_ui = uim[o_ui[0]]; m2_ui = uim[o_ui[1]] if len(uim) > 1 else 0.0
        rk_ui = np.empty(len(uim), int); rk_ui[o_ui] = np.arange(len(uim))
        o_ri = np.argsort(-rim); mx_ri = rim[o_ri[0]]; m2_ri = rim[o_ri[1]] if len(rim) > 1 else 0.0
        pair[qid] = (S, im, rk_im, mx_im, k3, mx_k3, uim, rk_ui, mx_ui, rim, mx_ri)
        qrow.append(dict(
            qid=qid, lnb_img_max=mx_im, lnb_img_gap=mx_im - m2_im, lnb_img_top=S.names[o_im[0]],
            lnb_k3_max=mx_k3, lnb_k3_gap=mx_k3 - m2_k3,
            lnb_uimg_max=mx_ui, lnb_uimg_gap=mx_ui - m2_ui, lnb_uimg_top=S.names[o_ui[0]],
            lnb_uimg_over_img=mx_ui / max(float(mx_im), 1e-6),
            lnb_rimg_max=mx_ri, lnb_rimg_gap=mx_ri - m2_ri,
        ))
    rows_new, rows_bg = [], []
    for q, c, lnb_v in zip(P.qid.values, P.cand.values, P.lnb.values):
        S, im, rk_im, mx_im, k3, mx_k3, uim, rk_ui, mx_ui, rim, mx_ri = pair[q]
        j = np.searchsorted(S.names, c)
        ok = j < len(S.names) and S.names[j] == c
        vi = float(im[j]) if ok else 0.0
        vk = float(k3[j]) if ok else 0.0
        vu = float(uim[j]) if ok else 0.0
        vr = float(rim[j]) if ok else 0.0
        rows_new.append((
            vi, rk_im[j] if ok else 999, vi / mx_im if mx_im > 0 else 0.0, np.log1p(vi),
            vi / max(float(lnb_v), 1e-6) if lnb_v > 0 else 0.0,
            vk, vk / mx_k3 if mx_k3 > 0 else 0.0,
        ))
        rows_bg.append((
            vu, rk_ui[j] if ok else 999, vu / mx_ui if mx_ui > 0 else 0.0, np.log1p(vu),
            vu / max(vi, 1e-6) if vi > 0 else 0.0,
            vr, vr / mx_ri if mx_ri > 0 else 0.0,
        ))
    P = P.copy()
    P[NEW_P] = np.array(rows_new, dtype=np.float32)
    P[BG_P] = np.array(rows_bg, dtype=np.float32)
    Qd = pd.DataFrame(qrow)
    top_cos = P[P["rank"] == 0].set_index("qid").cand
    Qd["lnb_img_agree"] = (Qd.lnb_img_top.values == top_cos.reindex(Qd.qid).values).astype(np.float32)
    Qd["lnb_uimg_agree"] = (Qd.lnb_uimg_top.values == top_cos.reindex(Qd.qid).values).astype(np.float32)
    Q = Q.merge(Qd.drop(columns=["lnb_img_top", "lnb_uimg_top"]), on="qid", how="left")
    return P, Q


def to_probs_sp(P, Qf, T_map, sp_lookup):
    pn = dict(zip(Qf.qid, Qf.pn2))
    out = {}
    for q, g in P.groupby("qid", sort=False):
        labs = list(g.cand.values) + [NEW]
        T = T_map[sp_lookup[q]] if isinstance(T_map, dict) else T_map
        sc = np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)) / T
        e = np.exp(sc - sc.max())
        out[q] = dict(zip(labs, e / e.sum()))
    return out


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, _ = load()
    sp_tr = dict(zip(tr.id, tr.species))
    sp_te = dict(zip(te.id, te.species))
    splits = val_splits(tr, l2n(Xtr), range(16))
    lnb_dir = f"{H}/lnbnn_np64"
    base_S = {sp: ScorerDir(lnb_dir, sp, zip(tr.id[tr.species == sp], tr.identity[tr.species == sp])) for sp in ["hyena", "leopard"]}
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}

    # 1. Build V2 (all 16 splits with np64 lnb_img + lnb_k3 + lnb_uimg + lnb_rimg), cached
    v2_path = f"{H}/dev_bgclean_v2.pkl"
    if os.path.exists(v2_path):
        V2 = pickle.load(open(v2_path, "rb"))
        print(f"Loaded cached {v2_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        V2 = {}
        for s, (gal, q, truth) in enumerate(splits):
            P0, Q0 = pickle.load(open(f"{H}/val_lnb_{s}.pkl", "rb"))
            sc_s = {sp: (base_S[sp], np.isin(base_S[sp].gal, tr.id[gal].values)) for sp in base_S}
            V2[s] = add_bg_lnb(P0, Q0, sp_tr, sc_s)
        pickle.dump(V2, open(v2_path, "wb"), protocol=4)
        print(f"Computed & cached np64 + bgclean LNBNN for all 16 splits in {time.time()-t0:.1f}s", flush=True)

    # 2. Leak-free Round-1 OOF for all 16 splits -> R1, cached
    r1_path = f"{H}/dev_bgclean_r1.pkl"
    if os.path.exists(r1_path):
        R1 = pickle.load(open(r1_path, "rb"))
        print(f"Loaded cached {r1_path} ({time.time()-t0:.1f}s)", flush=True)
    else:
        R1, r1_scores = {}, []
        for s in range(16):
            Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
            Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
            mp, mq = model2.fit(Ptr, Qtr, PF1_BG, QF1_BG, seed=0)
            Pv["p1"] = mp.predict(Pv[PF1_BG]); Qv["pn1"] = mq.predict(Qv[QF1_BG])
            if s < 8:
                r1_scores.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
            Pv, Qv = model2.round2_feats(Pv, Qv)
            R1[s] = rich_feats(Pv, Qv, G[s], scale=2.0)
        pickle.dump(R1, open(r1_path, "wb"), protocol=4)
        print(f"R1 MEAN (splits 0-7): {np.mean(r1_scores):.4f} (was 0.8594 in dev7) ({time.time()-t0:.0f}s)", flush=True)

    # 3. Leak-free Round-2 OOF (splits 0-7, 3 seeds) + Shared CV comparison vs dev7
    OOF, CAL, CAL_SP = {}, [], {"hyena": [], "leopard": []}
    for s in range(8):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in SEEDS:
            mp, mq = model2.fit(Ptr, Qtr, PF2_BG, QF2_BG, seed=sd)
            p2 += mp.predict(Pv[PF2_BG]) / len(SEEDS)
            n2 += mq.predict(Qv[QF2_BG]) / len(SEEDS)
        Pv["p2"] = p2; Qv["pn2"] = n2
        OOF[s] = (Pv[["qid", "cand", "p2"]], Qv[["qid", "species", "pn2", "truth", "nbr_ids", "nbr_sims"]])
        tmap = dict(zip(Qv.qid, Qv.truth)); pn = dict(zip(Qv.qid, Qv.pn2))
        for q, g in Pv.groupby("qid", sort=False):
            labs = list(g.cand.values) + [NEW]
            if tmap[q] in labs:
                item = (np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)), labs.index(tmap[q]))
                CAL.append(item)
                CAL_SP[sp_tr[q]].append(item)

    def fit_T(items):
        def nll(T):
            return -np.mean([(sc[i] / T) - np.log(np.exp(sc / T - (sc / T).max()).sum()) - (sc / T).max() for sc, i in items])
        return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)

    T_CAL = fit_T(CAL)
    T_SP = {sp: fit_T(CAL_SP[sp]) for sp in ["hyena", "leopard"]}
    print(f"Fitted T_CAL = {T_CAL:.3f} | per-species T = {T_SP} ({time.time()-t0:.0f}s)", flush=True)

    SAB = pickle.load(open(SAB_CV, "rb"))
    res = {k: [] for k in ["dev_bg", "dev_bg_prop", "blend_bg_prop", "blend_bg_sp"]}
    OOF_PROBS = {}
    for s in range(8):
        Pv, Qv = OOF[s]; truth = dict(zip(Qv.qid, Qv.truth))
        p_raw = to_probs_raw(Pv, Qv, T_CAL)
        p_prop = propagate_probs(p_raw, Qv, G[s], alpha=0.30, kp_thr=2.0, cos_thr=0.72)
        p_sp = propagate_probs(to_probs_sp(Pv, Qv, T_SP, sp_tr), Qv, G[s], alpha=0.30, kp_thr=2.0, cos_thr=0.72)
        OOF_PROBS[s] = p_prop
        res["dev_bg"].append(eval_probs(p_raw, truth))
        res["dev_bg_prop"].append(eval_probs(p_prop, truth))
        res["blend_bg_prop"].append(eval_probs(blend_dict(p_prop, SAB[s], 0.5), truth))
        res["blend_bg_sp"].append(eval_probs(blend_dict(p_sp, SAB[s], 0.5), truth))
        print(f"split {s}: dev_bg {res['dev_bg'][-1]:.4f} | +prop {res['dev_bg_prop'][-1]:.4f} | "
              f"blend {res['blend_bg_prop'][-1]:.4f} | blend_sp {res['blend_bg_sp'][-1]:.4f}", flush=True)
    print("\n=== DEV_BGCLEAN HONEST CV SUMMARY (splits 0-7, 3 seeds) ===", flush=True)
    print(f"  dev_bg        : {np.mean(res['dev_bg']):.4f} (dev7 was 0.8705)", flush=True)
    print(f"  dev_bg_prop   : {np.mean(res['dev_bg_prop']):.4f} (dev7 was 0.8715)", flush=True)
    print(f"  blend_bg_prop : {np.mean(res['blend_bg_prop']):.4f} (dev7 was 0.8764)", flush=True)
    print(f"  blend_bg_sp   : {np.mean(res['blend_bg_sp']):.4f}", flush=True)

    # 4. Build test features + Full 16-split 3-seed fit -> test_probs_dev_bgclean_prop.pkl
    PT, QT = pickle.load(open(f"{H}/test_feats.pkl", "rb"))
    PT, QT = add_lnbnn_test(PT, QT, tr, te)
    sc_test = {sp: (S, np.ones(len(S.gal), dtype=bool)) for sp, S in base_S.items()}
    PT, QT = add_bg_lnb(PT, QT, sp_te, sc_test)

    Pall = pd.concat([V2[s][0][PF1_BG + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_BG + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_BG, QF1_BG, seed=sd)
        p1 += mp.predict(PT[PF1_BG]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_BG]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()
    print(f"Round 1 full test done ({time.time()-t0:.0f}s)", flush=True)

    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)

    P2all = pd.concat([R1[s][0][PF2_BG + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_BG + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()
    p2, n2 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_BG, QF2_BG, seed=sd)
        p2 += mp.predict(PT[PF2_BG]) / len(SEEDS)
        n2 += mq.predict(QT[QF2_BG]) / len(SEEDS)
    PT["p2"] = p2; QT["pn2"] = n2
    print(f"Round 2 full test done ({time.time()-t0:.0f}s)", flush=True)

    probs_raw = to_probs_raw(PT, QT, T_CAL)
    probs_prop = propagate_probs(probs_raw, QT, Gt, alpha=0.30, kp_thr=2.0, cos_thr=0.72)
    pickle.dump(probs_raw, open(f"{H}/test_probs_dev_bgclean.pkl", "wb"))
    pickle.dump(probs_prop, open(f"{H}/test_probs_dev_bgclean_prop.pkl", "wb"))
    pickle.dump(OOF_PROBS, open(f"{H}/oof_probs_dev_bgclean_prop.pkl", "wb"))

    for sp in ["hyena", "leopard"]:
        ids = te.id[te.species == sp].values
        pnew = np.mean([probs_prop[q].get(NEW, 0.0) for q in ids])
        n1_ = np.mean([max(probs_prop[q], key=probs_prop[q].get) == NEW for q in ids])
        print(f"test dev_bgclean {sp:7s}: mean_pnew={pnew:.4f} new@1={n1_:.4f}", flush=True)
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
