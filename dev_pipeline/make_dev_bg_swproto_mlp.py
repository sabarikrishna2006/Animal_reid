"""Upgrade Dev's MiewID+LDA pool (dev_bg_sw2152, currently 25% of 0.88585 Rank 1 lnb36) with:
  1. 100% gallery-only Sw2152 Class Centroid Prototype + Same-ID Gallery DBA features (proto_sw2152, dba_sw2152)
  2. Genuine non-tree Neural Network diversity: 3-seed Tabular MLPClassifier(128, 64) alongside 3-seed LightGBM
  3. Upgraded sw_qq + ssw_qq Learned BG-Filtered Query-Query Propagation (strictly nested OOF, AUC ~0.982)
Reuses cached dev_bg_sw2152_v2.pkl (zero SIFT/LNBNN recomputation), 100% leak-free, gated on 8-split Honest CV."""
import os, gc, pickle, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from core import load, l2n, val_splits, NEW
from features import SIDE
import model2
from exp_kpg import others
from exp_kpg_v2 import rich_feats
from exp_blend_check import to_probs_raw, propagate_probs, eval_probs, blend_dict
from build_dev_v10 import load_memmap_sift, FastSiftCache
from make_kpg_probs import add_lnbnn_test
from make_dev_bgclean import ScorerDir, add_bg_lnb
from make_dev_bgclean2 import add_sift_bg_feats
from make_dev_bgclean3 import add_fullframe_bg
from make_dev_hubness import add_hubness_feats
from make_dev_v10_lda512 import _rg_diff
from make_dev_v10_sw2152 import _sw_whiten_2152, add_sw2152_feats
from make_dev_bg_sw2152 import PF1_BGSW, QF1_BGSW, PF2_BGSW, QF2_BGSW, add_lognorm
from make_dev_v10_lprop import PAIR_BG_FEATS, build_qq_pairs_bg
from exp_dev_learned_prop import propagate_learned, PAIR_PARAMS

H = os.path.dirname(os.path.abspath(__file__))
SAB_CV = os.path.join(H, "..", "teammate_sabari", "work", "submissions", "sub_20261121-012442_sub_v8_oursplits.cv.probs.pkl")
SEEDS = [0, 1, 2]

SWP_P = ["max_proto_sw2152", "max_dba_sw2152", "gap_proto_sw2152", "gap_dba_sw2152"]
SWP_Q = ["q_proto_sw2152_max", "q_proto_sw2152_gap", "q_dba_sw2152_max", "q_dba_sw2152_gap"]

G2Q_COLS_R1 = ["max_sw2152", "max_proto_sw2152", "lnb_uimg", "comb"]
G2Q_P1 = [f"g2q_{m}_{c}" for c in G2Q_COLS_R1 for m in ("z", "gap", "rk")]
G2Q_Q1 = [f"q_{m}_g2q_{c}" for c in ("max_sw2152", "max_proto_sw2152", "lnb_uimg") for m in ("top0_z", "top0_rk", "max_z")]
G2Q_P2 = ["g2q_p1_gap", "g2q_p1_share", "g2q_p1_rk"]
G2Q_Q2 = ["q_top_g2q_p1_gap", "q_top_g2q_p1_share", "q_top_g2q_p1_rk"]

PF1_NEW = PF1_BGSW + SWP_P + G2Q_P1
QF1_NEW = QF1_BGSW + SWP_Q + G2Q_Q1
PF2_NEW = PF2_BGSW + SWP_P + G2Q_P1 + G2Q_P2
QF2_NEW = QF2_BGSW + SWP_Q + G2Q_Q1 + G2Q_Q2
PAIR_SW_FEATS = PAIR_BG_FEATS + ["sw_qq", "ssw_qq"]


def add_g2q_r1(P, Q):
    """100% unlabeled transductive gallery-to-query column reciprocal Z-score, gap, and rank."""
    P = P.copy(); Q = Q.copy()
    for col in G2Q_COLS_R1:
        g = P.groupby("cand")[col]
        c_max = g.transform("max").astype(np.float32)
        c_mean = g.transform("mean").astype(np.float32)
        c_std = g.transform("std").fillna(0.05).clip(0.03).astype(np.float32)
        c_rk = g.rank(ascending=False, method="min").astype(np.float32)
        P[f"g2q_z_{col}"] = ((P[col] - c_mean) / c_std).astype(np.float32)
        P[f"g2q_gap_{col}"] = (c_max - P[col]).astype(np.float32)
        P[f"g2q_rk_{col}"] = np.log1p(c_rk - 1.0).astype(np.float32)
    top0 = P.sort_values("rank").drop_duplicates("qid").set_index("qid")
    for col in ("max_sw2152", "max_proto_sw2152", "lnb_uimg"):
        Q[f"q_top0_z_g2q_{col}"] = top0[f"g2q_z_{col}"].reindex(Q.qid).fillna(0.0).values.astype(np.float32)
        Q[f"q_top0_rk_g2q_{col}"] = top0[f"g2q_rk_{col}"].reindex(Q.qid).fillna(5.0).values.astype(np.float32)
        Q[f"q_max_z_g2q_{col}"] = P.groupby("qid")[f"g2q_z_{col}"].max().reindex(Q.qid).fillna(0.0).values.astype(np.float32)
    return P, Q


def add_g2q_r2(P, Q):
    """Transductive gallery-to-query Round-1 probability competition (gap, share, rank)."""
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
    return P, Q


def add_sw_proto_feats(P, Q, gal, Xg, qry, Xq):
    """100% gallery-only Sw2152 Class Centroid Prototype + Same-ID Gallery DBA features."""
    q_info = {}
    Xg_n = l2n(Xg); Xq_n = l2n(Xq)
    for sp in ["hyena", "leopard"]:
        gm = (gal.species == sp).values; qm = (qry.species == sp).values
        G = gal[gm].reset_index(drop=True); Qdf = qry[qm].reset_index(drop=True)
        Zg0 = Xg_n[gm]; Zq0 = Xq_n[qm]; m0 = Zg0.mean(0)
        glab = G.identity.values
        Zgi, Zqi = _sw_whiten_2152(Zg0, Zq0, m0, glab, gamma=0.5)

        names, inv = np.unique(glab, return_inverse=True)
        nid = len(names); nq = len(Qdf)
        members = [np.where(inv == j)[0] for j in range(nid)]
        Pmat = l2n(np.vstack([Zgi[cols].mean(0) for cols in members]).astype(np.float32))
        Ip = (Zqi @ Pmat.T).astype(np.float32)

        Zgd = l2n((0.70 * Zgi + 0.30 * Pmat[inv]).astype(np.float32))
        Sd = (Zqi @ Zgd.T) - 0.50 * _rg_diff(Zgd, glab, K=10)[None, :]
        Id = np.full((nq, nid), -9.0, np.float32)
        for j, cols in enumerate(members):
            Id[:, j] = Sd[:, cols].max(1)

        top2_p = -np.sort(-Ip, axis=1)[:, :2]
        top2_d = -np.sort(-Id, axis=1)[:, :2]
        id_pos = {n: j for j, n in enumerate(names)}
        for i, qid in enumerate(Qdf.id.values):
            p1 = float(top2_p[i, 0]); p2 = float(top2_p[i, 1]) if nid > 1 else p1
            d1 = float(top2_d[i, 0]); d2 = float(top2_d[i, 1]) if nid > 1 else d1
            q_info[qid] = (id_pos, Ip[i], Id[i], p1, p1 - p2, d1, d1 - d2)

    P = P.copy(); Q = Q.copy()
    p_rows = []
    for q, c in zip(P.qid.values, P.cand.values):
        id_pos, ip, id_, p1, pg, d1, dg = q_info[q]
        j = id_pos.get(c, -1)
        if j >= 0:
            vp, vd = float(ip[j]), float(id_[j])
            p_rows.append((vp, vd, p1 - vp, d1 - vd))
        else:
            p_rows.append((-9.0, -9.0, 9.0, 9.0))
    P[SWP_P] = np.array(p_rows, dtype=np.float32)
    Q[SWP_Q] = np.array([(q_info[q][3], q_info[q][4], q_info[q][5], q_info[q][6]) for q in Q.qid.values], dtype=np.float32)
    return P, Q


def fit_predict_mlp(Ptr, Qtr, Pv, Qv, pf, qf, seeds=(0, 1)):
    def prep(tr_df, va_df):
        Xtr_ = np.nan_to_num(tr_df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Xva_ = np.nan_to_num(va_df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sc = StandardScaler().fit(Xtr_)
        return np.clip(sc.transform(Xtr_), -6.0, 6.0), np.clip(sc.transform(Xva_), -6.0, 6.0)

    Xp_tr, Xp_va = prep(Ptr[pf], Pv[pf])
    Xq_tr, Xq_va = prep(Qtr[qf], Qv[qf])
    p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
    for sd in seeds:
        mp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=0.05,
                           batch_size=1024, learning_rate_init=1.5e-3, max_iter=25,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=4, random_state=sd)
        mq = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.10,
                           batch_size=256, learning_rate_init=1.5e-3, max_iter=35,
                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, random_state=sd)
        mp.fit(Xp_tr, Ptr.y.values)
        mq.fit(Xq_tr, Qtr.is_new.values)
        p2 += mp.predict_proba(Xp_va)[:, 1] / len(seeds)
        n2 += mq.predict_proba(Xq_va)[:, 1] / len(seeds)
    return p2, n2


def compute_sw_qq_maps(tr, Xtr, splits, te, Xte):
    Xtr_n = l2n(Xtr); Xte_n = l2n(Xte)
    sw_q = {}
    for s in range(16):
        gal, q, _ = splits[s]
        Gq = tr[gal].reset_index(drop=True); Qq = tr[q].reset_index(drop=True)
        zi, zs = {}, {}
        for sp in ["hyena", "leopard"]:
            gm = (Gq.species == sp).values; qm = (Qq.species == sp).values
            Zg0, Zq0 = Xtr_n[gal][gm], Xtr_n[q][qm]; m0 = Zg0.mean(0)
            glab = Gq.identity.values[gm]
            gside = Gq.viewpoint.map(SIDE).fillna("O").values[gm]
            glab_s = np.array([f"{a}|{b}" if b in ("L", "R") else a for a, b in zip(glab, gside)])
            _, Zqi = _sw_whiten_2152(Zg0, Zq0, m0, glab, gamma=0.5)
            _, Zqs = _sw_whiten_2152(Zg0, Zq0, m0, glab_s, gamma=0.5)
            for qid, vi, vs in zip(Qq.id.values[qm], Zqi, Zqs):
                zi[qid] = vi; zs[qid] = vs
        sw_q[s] = (zi, zs)
    zi_t, zs_t = {}, {}
    for sp in ["hyena", "leopard"]:
        gm = (tr.species == sp).values; qm = (te.species == sp).values
        Zg0, Zq0 = Xtr_n[gm], Xte_n[qm]; m0 = Zg0.mean(0)
        glab = tr.identity.values[gm]
        gside = tr.viewpoint.map(SIDE).fillna("O").values[gm]
        glab_s = np.array([f"{a}|{b}" if b in ("L", "R") else a for a, b in zip(glab, gside)])
        _, Zqi = _sw_whiten_2152(Zg0, Zq0, m0, glab, gamma=0.5)
        _, Zqs = _sw_whiten_2152(Zg0, Zq0, m0, glab_s, gamma=0.5)
        for qid, vi, vs in zip(te.id.values[qm], Zqi, Zqs):
            zi_t[qid] = vi; zs_t[qid] = vs
    return sw_q, (zi_t, zs_t)


def fit_T(OOF_dict, n_splits=8):
    CAL = []
    for s in range(n_splits):
        Pv, Qv = OOF_dict[s]
        tmap = dict(zip(Qv.qid, Qv.truth)); pn = dict(zip(Qv.qid, Qv.pn2))
        for q, g in Pv.groupby("qid", sort=False):
            labs = list(g.cand.values) + [NEW]
            if tmap[q] in labs:
                CAL.append((np.log(np.clip(np.r_[g.p2.values, pn[q]], 1e-6, None)), labs.index(tmap[q])))
    def nll(T):
        return -np.mean([(sc[i] / T) - np.log(np.exp(sc / T - (sc / T).max()).sum()) - (sc / T).max() for sc, i in CAL])
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


if __name__ == "__main__":
    t0 = time.time()
    tr, te, Xtr, Xte = load()
    sp_te = dict(zip(te.id, te.species))
    norm_tr = dict(zip(tr.id, np.linalg.norm(Xtr, axis=1)))
    norm_te = dict(zip(te.id, np.linalg.norm(Xte, axis=1)))
    splits = val_splits(tr, l2n(Xtr), range(16))
    G = {s: pickle.load(open(f"{H}/kpg_val_{s}.pkl", "rb")) for s in range(16)}
    Gt = pickle.load(open(f"{H}/kpg_test.pkl", "rb"))
    sc_bg = pickle.load(open(f"{H}/sift_bg_cache.pkl", "rb"))

    # 1. Attach gallery-only sw_proto + transductive G2Q_R1 onto cached dev_bg_sw2152_v2.pkl
    v2_path = f"{H}/dev_bg_swproto_g2q_v2.pkl"
    if os.path.exists(v2_path):
        V2 = pickle.load(open(v2_path, "rb"))
    else:
        V2_prev = pickle.load(open(f"{H}/dev_bg_sw2152_v2.pkl", "rb"))
        V2 = {}
        for s, (gal, q, truth) in enumerate(splits):
            P0, Q0 = add_sw_proto_feats(V2_prev[s][0], V2_prev[s][1],
                                        tr[gal].reset_index(drop=True), Xtr[gal],
                                        tr[q].reset_index(drop=True), Xtr[q])
            P0, Q0 = add_g2q_r1(P0, Q0)
            V2[s] = (P0, Q0)
        del V2_prev; gc.collect()
        pickle.dump(V2, open(v2_path, "wb"), protocol=4)
    print(f"Prepared {v2_path} ({time.time()-t0:.0f}s)", flush=True)

    # 2. Leak-free Round-1 OOF + G2Q_R2
    r1_path = f"{H}/dev_bg_swproto_g2q_r1.pkl"
    if os.path.exists(r1_path):
        R1 = pickle.load(open(r1_path, "rb"))
    else:
        R1, r1s = {}, []
        for s in range(16):
            Ptr, Qtr = others(V2, s, 0), others(V2, s, 1)
            Pv, Qv = V2[s][0].copy(), V2[s][1].copy()
            mp, mq = model2.fit(Ptr, Qtr, PF1_NEW, QF1_NEW, seed=0)
            Pv["p1"] = mp.predict(Pv[PF1_NEW]); Qv["pn1"] = mq.predict(Qv[QF1_NEW])
            if s < 8:
                r1s.append(model2.score(model2.rank(Pv, Qv, "p1", "pn1", 1.0), Qv))
            Pv, Qv = model2.round2_feats(Pv, Qv)
            Pv, Qv = rich_feats(Pv, Qv, G[s], scale=2.0)
            R1[s] = add_g2q_r2(Pv, Qv)
        print(f"R1 MEAN (splits 0-7): {np.mean(r1s):.4f} (bg_sw2152 was 0.8700) ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(R1, open(r1_path, "wb"), protocol=4)

    # 3. Leak-free Round-2 OOF: BOTH LightGBM (3 seeds, splits 0-15) AND MLP (3 seeds, splits 0-7)
    OOF_LGB, OOF_MLP = {}, {}
    for s in range(16):
        Ptr, Qtr = others(R1, s, 0), others(R1, s, 1)
        Pv, Qv = R1[s][0].copy(), R1[s][1].copy()
        sds = SEEDS if s < 8 else [0]
        p2, n2 = np.zeros(len(Pv)), np.zeros(len(Qv))
        for sd in sds:
            mp, mq = model2.fit(Ptr, Qtr, PF2_NEW, QF2_NEW, seed=sd)
            p2 += mp.predict(Pv[PF2_NEW]) / len(sds)
            n2 += mq.predict(Qv[QF2_NEW]) / len(sds)
        Pv["p2"] = p2; Qv["pn2"] = n2
        OOF_LGB[s] = (Pv[["qid", "cand", "p2"]], Qv[["qid", "species", "pn2", "truth", "nbr_ids", "nbr_sims"]])
        if s < 8:
            Pm, Qm = R1[s][0].copy(), R1[s][1].copy()
            pm2, nm2 = fit_predict_mlp(Ptr, Qtr, Pm, Qm, PF2_NEW, QF2_NEW, seeds=(0, 1))
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
    print(f"dev_bg sw_qq lprop OOF AUC: {np.mean(aucs):.5f} ({time.time()-t0:.0f}s)", flush=True)

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
    OSW = pickle.load(open(f"{H}/oof_probs_dev_v10_sw2152_lprop.pkl", "rb"))

    print("\n=== DEV_BG_SWPROTO_MLP 8-SPLIT HONEST CV ===", flush=True)
    for name, d in [
        ("OLD dev_bg_sw2152_prop", OBG_OLD),
        ("NEW dev_bg_swproto_lgb_lprop", OOF_LGB_LP),
        ("NEW dev_bg_swproto_mlp_lprop", OOF_MLP_LP),
        ("NEW 50%lgb + 50%mlp (bg_swproto)", OOF_MIX_LP),
    ]:
        solos, devh, shbl = [], [], []
        for s in range(8):
            Qv = OOF_LGB[s][1]; truth = dict(zip(Qv.qid, Qv.truth))
            solos.append(eval_probs(d[s], truth))
            pd_ = blend_dict(d[s], OSW[s], 0.5)
            devh.append(eval_probs(pd_, truth))
            shbl.append(eval_probs(blend_dict(pd_, SAB[s], 0.5), truth))
        print(f"  {name:34s} | solo={np.mean(solos):.5f} | DevHalf={np.mean(devh):.5f} | SharedBlend={np.mean(shbl):.5f}", flush=True)

    # 5. Full 16-split 3-seed test fit for BOTH LGB and MLP
    sc_raw = pickle.load(open(f"{H}/sift_pair_cache.pkl", "rb"))
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
    PT, QT = add_sw_proto_feats(PT, QT, tr, Xtr, te, Xte)
    PT, QT = add_g2q_r1(PT, QT)

    Pall = pd.concat([V2[s][0][PF1_NEW + ["y"]] for s in range(16)], ignore_index=True)
    Qall = pd.concat([V2[s][1][QF1_NEW + ["is_new"]] for s in range(16)], ignore_index=True)
    del V2; gc.collect()
    p1, n1 = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(Pall, Qall, PF1_NEW, QF1_NEW, seed=sd)
        p1 += mp.predict(PT[PF1_NEW]) / len(SEEDS)
        n1 += mq.predict(QT[QF1_NEW]) / len(SEEDS)
    PT["p1"] = p1; QT["pn1"] = n1
    del Pall, Qall; gc.collect()
    PT, QT = model2.round2_feats(PT, QT)
    PT, QT = rich_feats(PT, QT, Gt, scale=2.0)
    PT, QT = add_g2q_r2(PT, QT)

    P2all = pd.concat([R1[s][0][PF2_NEW + ["y"]] for s in range(16)], ignore_index=True)
    Q2all = pd.concat([R1[s][1][QF2_NEW + ["is_new"]] for s in range(16)], ignore_index=True)
    del R1; gc.collect()

    # Test LGB
    p2_l, n2_l = np.zeros(len(PT)), np.zeros(len(QT))
    for sd in SEEDS:
        mp, mq = model2.fit(P2all, Q2all, PF2_NEW, QF2_NEW, seed=sd)
        p2_l += mp.predict(PT[PF2_NEW]) / len(SEEDS)
        n2_l += mq.predict(QT[QF2_NEW]) / len(SEEDS)
    PT_l, QT_l = PT.copy(), QT.copy(); PT_l["p2"] = p2_l; QT_l["pn2"] = n2_l
    t_raw_lgb = to_probs_raw(PT_l, QT_l, T_LGB)

    # Test MLP
    p2_m, n2_m = fit_predict_mlp(P2all, Q2all, PT, QT, PF2_NEW, QF2_NEW, seeds=(0, 1))
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

    pickle.dump(t_lgb_lp, open(f"{H}/test_probs_dev_bg_swproto_lgb_lprop.pkl", "wb"))
    pickle.dump(t_mlp_lp, open(f"{H}/test_probs_dev_bg_swproto_mlp_lprop.pkl", "wb"))
    pickle.dump(OOF_LGB_LP, open(f"{H}/oof_probs_dev_bg_swproto_lgb_lprop.pkl", "wb"))
    pickle.dump(OOF_MLP_LP, open(f"{H}/oof_probs_dev_bg_swproto_mlp_lprop.pkl", "wb"))
    print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
