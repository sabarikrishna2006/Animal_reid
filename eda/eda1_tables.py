"""EDA part 1: integrity, label structure, viewpoints, embedding sanity."""
import numpy as np, pandas as pd
from pathlib import Path

RAW = Path(r"e:\ML_projectr\data\reid_student_data_main")
tr = pd.read_csv(RAW / "train.csv")
te = pd.read_csv(RAW / "test.csv")
ids = pd.read_csv(RAW / "miewid_ids.csv")["id"]
X = np.load(RAW / "miewid_features.npy")
sub = pd.read_csv(RAW / "sample_submission.csv")

print("== shapes", tr.shape, te.shape, ids.shape, X.shape, X.dtype, sub.shape)
print("dup ids train/test/miew:", tr.id.duplicated().sum(), te.id.duplicated().sum(), ids.duplicated().sum())
print("train ids in miew:", tr.id.isin(ids).all(), " test ids in miew:", te.id.isin(ids).all(),
      " miew covers exactly:", set(ids) == set(tr.id) | set(te.id))
print("miew order = train then test?", list(ids) == list(tr.id) + list(te.id))
print("sample_sub ids == test ids:", list(sub.id) == list(te.id))
print("NaN/inf in X:", np.isnan(X).sum(), np.isinf(X).sum())
print("nulls train:\n", tr.isna().sum().to_string(), "\nnulls test:\n", te.isna().sum().to_string())

print("\n== species counts")
print(pd.concat([tr.species.value_counts().rename("train"), te.species.value_counts().rename("test")], axis=1))
print("identity prefix vs species:", tr.groupby("species").identity.apply(lambda s: s.str[:3].unique().tolist()).to_dict())
print("any identity in 2 species:", (tr.groupby("identity").species.nunique() > 1).sum())

print("\n== identities / shots per species")
for sp, g in tr.groupby("species"):
    c = g.identity.value_counts()
    bins = pd.cut(c, [0, 1, 2, 5, 10, 20, 10**6], labels=["1", "2", "3-5", "6-10", "11-20", ">20"])
    print(f"{sp}: images={len(g)} identities={c.size} mean={c.mean():.2f} median={c.median()} max={c.max()}")
    print("   identities per shot bin:", bins.value_counts().sort_index().to_dict())
    print("   images per shot bin    :", c.groupby(bins, observed=False).sum().to_dict())
    print("   top 5:", c.head(5).to_dict())

print("\n== viewpoints (train vs test, by species)")
vp = pd.concat([tr.assign(split="train"), te.assign(split="test")])
vp["viewpoint"] = vp.viewpoint.fillna("NaN")
print(pd.crosstab([vp.species, vp.viewpoint], vp.split).to_string())

side = {"left": "L", "frontleft": "L", "backleft": "L", "right": "R", "frontright": "R", "backright": "R"}
tr["side"] = tr.viewpoint.map(side).fillna("N")
for sp, g in tr.groupby("species"):
    s = g.groupby("identity").side.agg(lambda x: set(x))
    multi = g.identity.value_counts()
    multi = multi[multi >= 2].index
    s = s[s.index.isin(multi)]
    print(f"{sp} (ids with >=2 imgs: {len(s)}): both L&R={sum(('L' in x and 'R' in x) for x in s)}, "
          f"only L={sum(x=={'L'} for x in s)}, only R={sum(x=={'R'} for x in s)}, "
          f"single-viewpoint ids={sum(g[g.identity.isin(multi)].groupby('identity').viewpoint.nunique()==1)}")

print("\n== embeddings")
n = np.linalg.norm(X, axis=1)
print("norm: min %.4f p1 %.4f median %.4f p99 %.4f max %.4f" % (n.min(), *np.percentile(n, [1, 50, 99]), n.max()))
print("feature mean abs %.4f, value range [%.3f, %.3f]" % (np.abs(X.mean(0)).mean(), X.min(), X.max()))
print("zero-variance dims:", (X.std(0) < 1e-8).sum())
# exact duplicate vectors
_, inv, cnt = np.unique(X.round(5), axis=0, return_inverse=True, return_counts=True)
print("exact-duplicate embedding rows:", (cnt > 1).sum(), "groups covering", cnt[cnt > 1].sum(), "rows")
# PCA spectrum
Xc = X - X.mean(0)
s = np.linalg.svd(Xc[::2], compute_uv=False) ** 2
cum = np.cumsum(s) / s.sum()
print("PCA dims for 50/80/90/95/99% variance:", [int(np.searchsorted(cum, q) + 1) for q in (.5, .8, .9, .95, .99)])
print("top-10 explained var ratios:", np.round(s[:10] / s.sum(), 4).tolist())

# species from embedding (sanity)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
row = {i: k for k, i in enumerate(ids)}
Xtr = X[[row[i] for i in tr.id]]
print("LR species-from-embedding 5-fold acc:",
      cross_val_score(LogisticRegression(max_iter=2000, C=1.0), Xtr, tr.species, cv=5).mean().round(4))
