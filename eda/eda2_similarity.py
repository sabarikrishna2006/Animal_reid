"""EDA part 2: cosine structure, normalisation choice, LOO retrieval, novelty signal, test-test repeats."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

RAW = Path(r"e:\ML_projectr\data\reid_student_data_main")
tr = pd.read_csv(RAW / "train.csv"); te = pd.read_csv(RAW / "test.csv")
ids = pd.read_csv(RAW / "miewid_ids.csv")["id"]; X = np.load(RAW / "miewid_features.npy").astype(np.float64)
ntr = len(tr)
Xtr, Xte = X[:ntr], X[ntr:]  # verified: miew order = train then test

def l2(a): return a / np.linalg.norm(a, axis=1, keepdims=True)

# ---- id-order check (ids should carry no info)
same_next = (tr.identity.values[1:] == tr.identity.values[:-1]).mean()
p = tr.identity.value_counts(normalize=True); print("P(consecutive train rows same identity) = %.4f, expected if shuffled = %.4f" % (same_next, (p**2).sum()))

# ---- norm vs other things
nrm = np.linalg.norm(X, axis=1)
print("norm by species/split:")
print(pd.DataFrame({"n": nrm, "sp": list(tr.species) + list(te.species), "split": ["tr"]*ntr + ["te"]*len(te)}).groupby(["sp", "split"]).n.describe()[["mean", "50%", "std"]].round(2))

def loo_eval(Z, name, df=tr):
    """Leave-one-out top-1 identity accuracy + same/diff AUC, per species, using images with >=1 sibling."""
    out = {}
    for sp in ["hyena", "leopard"]:
        m = (df.species == sp).values
        Zs, y = Z[m], df.identity.values[m]
        S = Zs @ Zs.T; np.fill_diagonal(S, -np.inf)
        nn = S.argmax(1)
        has_sib = pd.Series(y).map(pd.Series(y).value_counts()).values > 1
        acc = (y[nn] == y)[has_sib].mean()
        # map@5 over identities (max agg) for images with sibling (closed-set LOO)
        iu = np.triu_indices(len(y), 1)
        same = (y[:, None] == y[None, :])[iu]; s = S[iu]
        rng = np.random.default_rng(0); idx = rng.choice(len(s), min(len(s), 2_000_000), replace=False)
        auc = roc_auc_score(same[idx], s[idx])
        out[sp] = (acc, auc)
    print(f"{name:28s} LOO top1 hyena={out['hyena'][0]:.4f} leopard={out['leopard'][0]:.4f} | pair AUC hyena={out['hyena'][1]:.4f} leopard={out['leopard'][1]:.4f}")

print("\n== normalisation choice (LOO NN on train, closed set)")
loo_eval(l2(Xtr), "raw L2")
mu = X.mean(0); loo_eval(l2(Xtr - mu), "center(all) + L2")
sd = X.std(0); loo_eval(l2((Xtr - mu) / sd), "standardise + L2")
loo_eval(l2(np.sqrt(np.maximum(Xtr, 0)) - 0) if (Xtr.min() >= 0) else l2(np.sign(Xtr) * np.sqrt(np.abs(Xtr))), "signed sqrt + L2")
# per-species centering
Z = np.empty_like(X)
sp_all = np.array(list(tr.species) + list(te.species))
for sp in ["hyena", "leopard"]:
    m = sp_all == sp; Z[m] = X[m] - X[m].mean(0)
loo_eval(l2(Z[:ntr]), "center(per species) + L2")

# ---- work in raw L2 from here
U = l2(X); Utr, Ute = U[:ntr], U[ntr:]
print("\n== same vs diff identity cosine (raw L2), per species")
for sp in ["hyena", "leopard"]:
    m = (tr.species == sp).values; Z = Utr[m]; y = tr.identity.values[m]
    S = Z @ Z.T; iu = np.triu_indices(len(y), 1); same = (y[:, None] == y[None, :])[iu]; s = S[iu]
    q = lambda a: np.percentile(a, [1, 5, 25, 50, 75, 95, 99]).round(3).tolist()
    print(f"{sp}: same n={same.sum()} pct[1,5,25,50,75,95,99]={q(s[same])}")
    print(f"{' '*len(sp)}  diff n={(~same).sum()} pct={q(s[~same])}")

print("\n== top-1 cosine to train gallery: train LOO (known, has sibling) vs train singletons-LOO (proxy new) vs test")
rows = []
for sp in ["hyena", "leopard"]:
    mtr = (tr.species == sp).values; mte = (te.species == sp).values
    G = Utr[mtr]; y = tr.identity.values[mtr]
    S = G @ G.T; np.fill_diagonal(S, -np.inf)
    cnt = pd.Series(y).map(pd.Series(y).value_counts()).values
    loo_top1 = S.max(1)
    # "new" proxy: top-1 to OTHER identities only (as if identity were removed from gallery)
    S2 = S.copy(); S2[y[:, None] == y[None, :]] = -np.inf; new_proxy = S2.max(1)
    St = Ute[mte] @ G.T; te_top1 = St.max(1)
    q = lambda a: np.percentile(a, [5, 25, 50, 75, 95]).round(3).tolist()
    print(f"{sp}: known LOO top1 {q(loo_top1[cnt > 1])} | new-proxy top1 {q(new_proxy)} | TEST top1 {q(te_top1)}")
    rows.append((sp, loo_top1[cnt > 1], new_proxy, te_top1))
    # simple mixture estimate: test top1 as mixture of known & new-proxy distributions (method of moments over histogram, NNLS)
    from scipy.optimize import nnls
    bins = np.linspace(0, 1, 41)
    hk = np.histogram(loo_top1[cnt > 1], bins, density=True)[0]; hn = np.histogram(new_proxy, bins, density=True)[0]
    ht = np.histogram(te_top1, bins, density=True)[0]
    w, res = nnls(np.c_[hk, hn], ht); w = w / w.sum()
    print(f"   NNLS mixture of [known, new-proxy] histograms -> pi_new ~= {w[1]:.3f} (resid {res:.2f})")
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, random_state=0).fit(te_top1[:, None]); lo = g.means_.ravel().argmin()
    print(f"   GMM-2 on test top1: means={g.means_.ravel().round(3)}, weights={g.weights_.round(3)} -> pi_new(low comp) ~= {g.weights_[lo]:.3f}")
np.savez(r"e:\ML_projectr\eda\top1_dists.npz", **{f"{sp}_{k}": a for sp, a1, a2, a3 in rows for k, a in zip(["known", "newproxy", "test"], [a1, a2, a3])})

print("\n== test repeats: is nearest neighbour (over train U test pool) another test image?")
for sp in ["hyena", "leopard"]:
    mtr = (tr.species == sp).values; mte = (te.species == sp).values
    T = Ute[mte]; G = Utr[mtr]
    Stt = T @ T.T; np.fill_diagonal(Stt, -np.inf); Stg = T @ G.T
    nn_test = Stt.max(1) > Stg.max(1)
    print(f"{sp}: share NN is test = {nn_test.mean():.3f};  share with a test neighbour cos>0.8 = {(Stt.max(1) > 0.8).mean():.3f}, "
          f"and among those best train cos < 0.6: {((Stt.max(1) > 0.8) & (Stg.max(1) < 0.6)).mean():.3f}")
    # train analogue: share whose NN (over train) is same identity, for reference of what cos>0.8 means
print("\n== near duplicates (raw L2 cos > 0.97 / > 0.99)")
for sp in ["hyena", "leopard"]:
    m = (tr.species == sp).values; Z = Utr[m]; y = tr.identity.values[m]
    S = Z @ Z.T; iu = np.triu_indices(len(y), 1); s = S[iu]; same = (y[:, None] == y[None, :])[iu]
    for t in (0.95, 0.97, 0.99):
        k = s > t
        print(f"{sp} train-train cos>{t}: pairs={k.sum()}, diff-identity pairs={(k & ~same).sum()}")
    mte = (te.species == sp).values
    St = Ute[mte] @ Z.T
    print(f"{sp} test images with a train image cos>0.97: {(St.max(1) > 0.97).sum()} / {mte.sum()}")
