"""Evaluate 5 research-paper metric learning & transductive graph upgrades on the Corrected 8-Split
Honest 1-Query-Per-Encounter CV (enc_thr=0.8, new_frac=0.1, q_frac=0.4, per_enc=1, leak-free others()):
  1. Two-Covariance Bayesian PLDA & Multi-Shot Prototype Scoring (Ioffe ECCV 2006 / Prince & Elder ICCV 2007)
  2. Cohort-Excluded Adaptive Symmetric Score Normalization (AS-Norm, Cumani / Matejka Interspeech)
  3. Manifold Diffusion / Mutual alpha-QE + Gallery DBA & Query-Graph Votes in Sw2152 Space (Iscen CVPR 2017)
  4. Orthogonal Procrustes Cross-Flank Alignment (R_{L<->R} in Sw2152-PCA128, Schonemann 1966)
  5. Sinkhorn-Knopp Partial Optimal Transport Dual-Softmax (Cuturi NeurIPS 2013 / Sarlin SuperGlue CVPR 2020)
Combined with High-Recall nprobe=64 FAISS LNBNN (lnb, lnb_img, lnb_k3, lnb_uimg, lnb_rimg) and Sw2152/LDA512.
"""
import os, gc, json, time, pickle, warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import lightgbm as lgb

warnings.filterwarnings("ignore")
N_THREADS = min(16, os.cpu_count() or 8)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "reid_student_data_main")
CACHE = os.path.join(ROOT, "data", "cache")
LNB_DIR = os.path.join(CACHE, "lnbnn")
NEW = "new_individual"
SIDE = {"left": "L", "frontleft": "L", "backleft": "L", "right": "R", "frontright": "R", "backright": "R"}


def l2n(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _rg_diff(Z, glab, K=10):
    glab = np.asarray(glab, dtype=object)
    Sg = Z @ Z.T
    same = (glab[:, None] == glab[None, :])
    Sg = np.where(same, -2.0, Sg)
    k = min(K, max(1, Sg.shape[1] - 1))
    return np.partition(Sg, -k, axis=1)[:, -k:].mean(1).astype(np.float32)


def _sw_whiten_2152(Zg, Zq, m0, labels_sw, gamma=0.5):
    names_sw, inv_sw = np.unique(labels_sw, return_inverse=True)
    cnt = np.bincount(inv_sw)
    valid = cnt[inv_sw] >= 2
    mu = np.zeros((len(names_sw), Zg.shape[1]), dtype=np.float32)
    for j in np.where(cnt >= 2)[0]:
        mu[j] = Zg[inv_sw == j].mean(0)
    Zw = (Zg[valid] - mu[inv_sw[valid]]).astype(np.float64)
    Sw = (Zw.T @ Zw) / max(len(Zw), 1)
    ew, vw = np.linalg.eigh(Sw)
    ew = np.maximum(ew, 0.0)
    scale = (1.0 / np.sqrt((1.0 - gamma) * ew + gamma * ew.mean())).astype(np.float32)
    vw32 = vw.astype(np.float32)
    return l2n(((Zg - m0) @ vw32) * scale[None, :]), l2n(((Zq - m0) @ vw32) * scale[None, :])


class FastLNBScorer:
    def __init__(self, sp, train_df):
        m = json.load(open(f"{LNB_DIR}/{sp}_meta.json"))
        self.gal = np.array(m["gallery"])
        self.qpos = {q: i for i, q in enumerate(m["queries"])}
        self.qoff = np.array(m["qoff"])
        self.NB = np.array(np.load(f"{LNB_DIR}/{sp}_nb.npy"))
        self.DI = np.array(np.load(f"{LNB_DIR}/{sp}_d.npy"))
        lab = dict(zip(train_df.id, train_df.identity))
        self.names, self.lab = np.unique([lab[g] for g in self.gal], return_inverse=True)
        self.gpos = {g: i for i, g in enumerate(self.gal)}

    def score_one(self, qid, gmask):
        i = self.qpos[qid]; a, b = int(self.qoff[i]), int(self.qoff[i + 1])
        n_id, n_gal = len(self.names), len(self.gal)
        z = lambda: np.zeros(n_id, np.float32)
        if b <= a:
            return z(), z(), z(), z(), z()
        nb = self.NB[a:b]
        d = self.DI[a:b].astype(np.float32)
        ok = (nb >= 0) & gmask[np.maximum(nb, 0)]
        if qid in self.gpos:
            ok = ok & (nb != self.gpos[qid])
        rank = np.cumsum(ok, axis=1)
        rows_all = np.arange(len(d))
        d_max = d.max(axis=1)
        out_id10, out_img, out_k3, out_uimg, out_rimg = z(), z(), z(), z(), z()
        lab_mat = self.lab[np.maximum(nb, 0)]

        has_bg10 = rank[:, -1] >= 11
        bg10 = np.where(has_bg10, d[rows_all, np.argmax(rank >= 11, axis=1)], d_max)
        r10, c10 = np.nonzero(ok & (rank <= 10))
        if len(r10):
            l10 = lab_mat[r10, c10]
            _, f_id = np.unique(r10.astype(np.int64) * n_id + l10, return_index=True)
            np.add.at(out_id10, l10[f_id], np.maximum(bg10[r10[f_id]] - d[r10[f_id], c10[f_id]], 0.0))
            g10 = nb[r10, c10]
            _, f_g = np.unique(r10.astype(np.int64) * n_gal + g10, return_index=True)
            img_sc = np.zeros(n_gal, np.float32)
            np.add.at(img_sc, g10[f_g], np.maximum(bg10[r10[f_g]] - d[r10[f_g], c10[f_g]], 0.0))
            np.maximum.at(out_img, self.lab, img_sc)

        has_bg3 = rank[:, -1] >= 4
        bg3 = np.where(has_bg3, d[rows_all, np.argmax(rank >= 4, axis=1)], d_max)
        r3, c3 = np.nonzero(ok & (rank <= 3))
        if len(r3):
            l3 = lab_mat[r3, c3]
            _, f3 = np.unique(r3.astype(np.int64) * n_id + l3, return_index=True)
            np.add.at(out_k3, l3[f3], np.maximum(bg3[r3[f3]] - d[r3[f3], c3[f3]], 0.0))

        has1 = rank[:, -1] >= 1
        c1 = np.argmax(ok, axis=1)
        g1 = nb[rows_all, c1]
        id1 = np.where(has1, self.lab[np.maximum(g1, 0)], -1)
        ok_diff = ok & (lab_mat != id1[:, None])
        has2 = ok_diff.any(axis=1)
        c2 = np.argmax(ok_diff, axis=1)
        d2 = np.where(has2, d[rows_all, c2], d_max)
        ok_same = ok & (lab_mat == id1[:, None]) & (d < d2[:, None]) & has2[:, None]
        rs, cs = np.nonzero(ok_same)
        if len(rs):
            gs = nb[rs, cs]
            _, fg = np.unique(rs.astype(np.int64) * n_gal + gs, return_index=True)
            rs1, cs1, gs1 = rs[fg], cs[fg], gs[fg]
            vm = np.maximum(d2[rs1] - d[rs1, cs1], 0.0)
            img_u = np.zeros(n_gal, np.float32)
            np.add.at(img_u, gs1, vm)
            np.maximum.at(out_uimg, self.lab, img_u)
            vr = np.where(d[rs1, cs1] < 0.85 * d2[rs1], vm, 0.0)
            img_r = np.zeros(n_gal, np.float32)
            np.add.at(img_r, gs1, vr)
            np.maximum.at(out_rimg, self.lab, img_r)
        return out_id10, out_img, out_k3, out_uimg, out_rimg

    def score_batch(self, qids, gmask, col_idx):
        with ThreadPoolExecutor(N_THREADS) as ex:
            res = list(ex.map(lambda q: self.score_one(q, gmask), qids))
        return (
            np.stack([x[0] for x in res])[:, col_idx],
            np.stack([x[1] for x in res])[:, col_idx],
            np.stack([x[2] for x in res])[:, col_idx],
            np.stack([x[3] for x in res])[:, col_idx],
            np.stack([x[4] for x in res])[:, col_idx],
        )


BASE_P = [
    "sp_id", "cos_raw", "cos_hub", "gap_raw", "gap_hub", "rank_raw", "n_gal",
    "max_lda512", "max_slda512", "gap_lda512", "gap_slda512",
    "max_sw2152", "max_ssw2152", "gap_sw2152", "gap_ssw2152",
    "lnb", "lnb_rel", "lnb_log", "lnb_img", "lnb_img_rel", "lnb_k3",
    "lnb_uimg", "lnb_uimg_rel", "lnb_uimg_share", "lnb_rimg", "lnb_rimg_rel",
]
BASE_Q = [
    "sp_id", "q_raw_max", "q_raw_gap", "q_hub_max", "q_hub_gap",
    "q_lda512_max", "q_lda512_gap", "q_slda512_max", "q_slda512_gap",
    "q_sw2152_max", "q_sw2152_gap", "q_ssw2152_max", "q_ssw2152_gap",
    "q_lnb_max", "q_lnb_gap", "q_lnb_img_max", "q_lnb_img_gap", "q_lnb_k3_max",
    "q_lnb_uimg_max", "q_lnb_uimg_gap", "q_lnb_rimg_max", "q_lnb_rimg_gap", "q_nkp",
]
P1_P = ["plda_post", "plda_max", "gap_plda_post", "gap_plda_max", "proto_sw2152", "top2_sw2152"]
P1_Q = ["q_plda_post_max", "q_plda_post_gap", "q_plda_max_max", "q_plda_max_gap", "q_proto_sw_max", "q_proto_sw_gap"]
P2_P = ["asn_sw2152", "asn_lda512", "asn_plda", "asn_lnb_uimg", "gap_asn_sw", "gap_asn_lnb"]
P2_Q = ["q_asn_sw_max", "q_asn_sw_gap", "q_asn_plda_max", "q_asn_plda_gap", "q_asn_lnb_max", "q_asn_lnb_gap"]
P3_P = ["diff_sw2152", "gap_diff_sw", "qq_max_sw", "qq_nbr_vote"]
P3_Q = ["q_diff_sw_max", "q_diff_sw_gap", "q_qq_max_sw"]
P4_P = ["xflank_cos", "gap_xflank", "has_same_side", "opp_side_only"]
P4_Q = ["q_xflank_max", "q_xflank_gap"]
P5_P = ["ot_sw2152", "ot_plda", "gap_ot_sw"]
P5_Q = ["q_ot_sw_max", "q_ot_sw_gap"]
PAP_P = P1_P + P2_P + P3_P + P4_P + P5_P
PAP_Q = P1_Q + P2_Q + P3_Q + P4_Q + P5_Q

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=N_THREADS)


def others(store, s, idx):
    qs = set(store[s][1].qid)
    return pd.concat([store[k][idx][~store[k][idx].qid.isin(qs)] for k in store if k != s], ignore_index=True)


def rank_preds(P, Qf, pcol="p1", ncol="pn1", w=1.0):
    pn = dict(zip(Qf.qid, Qf[ncol])); out = {}
    for qid, g in P.groupby("qid", sort=False):
        s = g[pcol].values; o = np.argsort(-s); lst = []; placed = False
        for j in o[:5]:
            if not placed and s[j] < w * pn[qid]:
                lst.append(NEW); placed = True
            lst.append(g.cand.values[j])
        if not placed:
            lst.append(NEW)
        out[qid] = lst[:5]
    return out


def map5(preds, truth):
    s = 0.0
    for p, t in zip(preds, truth):
        for k, lab in enumerate(p[:5]):
            if lab == t:
                s += 1.0 / (k + 1)
                break
    return s / len(truth)


if __name__ == "__main__":
    t0 = time.time()
    VAL = pickle.load(open(f"{CACHE}/honest_cv_v27_tables.pkl", "rb"))
    configs = [
        ("0. Baseline v24 (Sw2152 + LDA512 + nprobe64 LNBNN)", BASE_P, BASE_Q),
        ("1. + Two-Covariance Bayesian PLDA & Prototype", BASE_P + P1_P, BASE_Q + P1_Q),
        ("2. + Cohort-Excluded AS-Norm", BASE_P + P2_P, BASE_Q + P2_Q),
        ("3. + Sw2152 Manifold Diffusion / QE + DBA", BASE_P + P3_P, BASE_Q + P3_Q),
        ("4. + Orthogonal Procrustes Cross-Flank Alignment", BASE_P + P4_P, BASE_Q + P4_Q),
        ("5. + Sinkhorn-Knopp Partial Optimal Transport", BASE_P + P5_P, BASE_Q + P5_Q),
        ("6. Combined 5-Paper Model (v27)", BASE_P + PAP_P, BASE_Q + PAP_Q),
    ]
    base_scores = None
    for name, pf, qf in configs:
        scores = []
        for s in range(8):
            Ptr, Qtr = others(VAL, s, 0), others(VAL, s, 1)
            Pv, Qv = VAL[s][0].copy(), VAL[s][1].copy()
            pp, pn = np.zeros(len(Pv)), np.zeros(len(Qv))
            for sd in [0, 1, 2]:
                mp = lgb.train(dict(PARAMS, seed=sd), lgb.Dataset(Ptr[pf], Ptr.y), 600)
                mq = lgb.train(dict(PARAMS, seed=sd), lgb.Dataset(Qtr[qf], Qtr.is_new), 400)
                pp += mp.predict(Pv[pf]) / 3.0
                pn += mq.predict(Qv[qf]) / 3.0
            Pv["p1"], Qv["pn1"] = pp, pn
            pr = rank_preds(Pv, Qv, "p1", "pn1", 1.0)
            scores.append(map5([pr[q] for q in Qv.qid], Qv.truth))
        arr = np.array(scores)
        if base_scores is None:
            base_scores = arr
        diff = arr - base_scores
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if np.any(diff != 0) else 0.0
        wins = int((diff > 1e-7).sum())
        print(f"{name:50s} | {arr.mean():.5f} ± {arr.std(ddof=1):.5f} | {diff.mean():+.5f} ± {se:.5f} | {wins}/8")
