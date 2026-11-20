"""EDA part 3: encounter (burst) structure, train vs test geometry, viewpoint compatibility, whitening quick check."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

RAW = Path(r"e:\ML_projectr\data\reid_student_data_main")
tr = pd.read_csv(RAW / "train.csv"); te = pd.read_csv(RAW / "test.csv")
X = np.load(RAW / "miewid_features.npy").astype(np.float64); ntr = len(tr)
l2 = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
U = l2(X); Utr, Ute = U[:ntr], U[ntr:]
q = lambda a: np.percentile(a, [5, 25, 50, 75, 95]).round(3).tolist()

print("== NN-cosine geometry: train->train(LOO, any id) vs test->test vs test->train")
for sp in ["hyena", "leopard"]:
    mtr = (tr.species == sp).values; mte = (te.species == sp).values
    G, T = Utr[mtr], Ute[mte]
    Sgg = G @ G.T; np.fill_diagonal(Sgg, -np.inf); Stt = T @ T.T; np.fill_diagonal(Stt, -np.inf)
    print(f"{sp}: train->train NN {q(Sgg.max(1))}\n{' '*len(sp)}  test->test  NN {q(Stt.max(1))}\n{' '*len(sp)}  test->train NN {q((T @ G.T).max(1))}")
    # subsample train to test size to remove gallery-size effect on train->train
    rng = np.random.default_rng(0); sub = rng.choice(mtr.sum(), mte.sum(), replace=False)
    Ss = Sgg[np.ix_(sub, sub)]
    print(f"{' '*len(sp)}  train(sub n={mte.sum()})->itself NN {q(Ss.max(1))}")

print("\n== encounter structure inside identities (avg-linkage on cosine distance)")
for sp in ["hyena", "leopard"]:
    g = tr[tr.species == sp]
    for thr in (0.2, 0.35, 0.5, 0.6):
        ncl = []
        for ident, gi in g.groupby("identity"):
            if len(gi) < 2: ncl.append(1); continue
            Z = Utr[gi.index.values]; D = np.clip(1 - Z @ Z.T, 0, 2); np.fill_diagonal(D, 0)
            lab = fcluster(linkage(squareform(D, checks=False), "average"), thr, "distance")
            ncl.append(lab.max())
        ncl = np.array(ncl); nimg = g.identity.value_counts().reindex(g.groupby("identity").size().index).values
        print(f"{sp} thr={thr}: clusters/identity mean={ncl.mean():.2f}, identities with >=2 clusters={(ncl>=2).sum()}/{len(ncl)}, "
              f"images per cluster mean={nimg.sum()/ncl.sum():.2f}")

print("\n== 'cross-encounter' LOO: NN accuracy when same-identity images with cos > c are removed from gallery")
for sp in ["hyena", "leopard"]:
    m = (tr.species == sp).values; Z = Utr[m]; y = tr.identity.values[m]
    S = Z @ Z.T; np.fill_diagonal(S, -np.inf); same = y[:, None] == y[None, :]
    for c in (1.01, 0.8, 0.6, 0.5):
        S2 = S.copy(); S2[same & (S > c)] = -np.inf
        still = (np.where(same, S2, -np.inf).max(1) > -np.inf)  # has a remaining sibling
        acc = (y[S2.argmax(1)] == y)[still].mean()
        print(f"{sp} remove same-id cos>{c}: queries with sibling left={still.sum()}, top1 acc={acc:.3f}, "
              f"top1 cos of known {q(S2.max(1)[still])}")

print("\n== viewpoint-pair compatibility (same-identity mean cosine / #pairs), per species")
vp_rows = []
for sp in ["hyena", "leopard"]:
    g = tr[tr.species == sp].copy(); g["vp"] = g.viewpoint.fillna("NaN")
    Z = Utr[g.index.values]; y = g.identity.values; v = g.vp.values
    S = Z @ Z.T; iu = np.triu_indices(len(y), 1); same = (y[:, None] == y[None, :])[iu]
    a, b = np.sort(np.c_[v[iu[0]], v[iu[1]]], axis=1).T
    df = pd.DataFrame({"a": a, "b": b, "s": S[iu], "same": same})
    t = df.groupby(["a", "b"]).apply(lambda d: pd.Series({"same_mean": d.s[d.same].mean(), "same_n": d.same.sum(),
                                                           "diff_mean": d.s[~d.same].mean(),
                                                           "auc": roc_auc_score(d.same, d.s) if 0 < d.same.sum() < len(d) else np.nan}),
                                     include_groups=False)
    t = t[t.same_n >= 30].sort_values("same_n", ascending=False)
    print(f"-- {sp}\n", t.round(3).head(20).to_string())
    t["species"] = sp; vp_rows.append(t.reset_index())
pd.concat(vp_rows).to_csv(r"e:\ML_projectr\eda\vp_compat.csv", index=False)

print("\n== quick: PCA-whitening (fit on train U test features, per species) pair AUC and cross-encounter top1 (cos>0.6 removed)")
sp_all = np.array(list(tr.species) + list(te.species))
for sp in ["hyena", "leopard"]:
    msp = sp_all == sp; Xs = X[msp]; mu = Xs.mean(0)
    Uc, s, Vt = np.linalg.svd(Xs - mu, full_matrices=False)
    ev = s**2 / (len(Xs) - 1)
    m = (tr.species == sp).values; y = tr.identity.values[m]; Xq = X[:ntr][m]
    Sraw = Utr[m] @ Utr[m].T; same = y[:, None] == y[None, :]
    def score(Z, name):
        S = Z @ Z.T; iu = np.triu_indices(len(y), 1)
        rng = np.random.default_rng(0); idx = rng.choice(len(iu[0]), min(len(iu[0]), 1_500_000), replace=False)
        auc = roc_auc_score(same[iu][idx], S[iu][idx])
        S = S.copy(); np.fill_diagonal(S, -np.inf); S[same & (Sraw > 0.6)] = -np.inf
        Sr = Sraw.copy(); np.fill_diagonal(Sr, -np.inf); Sr[same & (Sraw > 0.6)] = -np.inf
        still = np.where(same, Sr, -np.inf).max(1) > -np.inf
        acc = (y[S.argmax(1)] == y)[still].mean()
        print(f"{sp:8s} {name:30s} pairAUC={auc:.4f}  xenc-top1={acc:.4f}")
    score(Utr[m], "raw L2")
    for d in (128, 256, 512, 1024):
        for eps in (0.0, 0.1, 1.0):
            W = Vt[:d].T / np.sqrt(ev[:d] + eps * ev.mean())
            score(l2((Xq - mu) @ W), f"PCAw d={d} eps={eps}")
