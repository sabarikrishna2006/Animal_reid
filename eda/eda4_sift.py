"""EDA part 4: SIFT keypoint stats + pilot of RANSAC verification vs cosine on shortlisted pairs."""
import numpy as np, pandas as pd, time
from pathlib import Path
import cv2
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

D = Path(r"e:\ML_projectr\data")
RAW = D / "reid_student_data_main"

def sift_dir(i): return D / ("sift_features_train.npz" if i.startswith("tr_") else "sift_features_test.npz")
def load(i):
    return np.load(sift_dir(i) / f"{i}_xy.npy"), np.load(sift_dir(i) / f"{i}_desc.npy")

def root(d):
    d = d.astype(np.float32); d /= d.sum(1, keepdims=True) + 1e-7; return np.sqrt(d)

def match_pair(args):
    a, b = args
    cv2.setNumThreads(1)
    xa, da = load(a); xb, db = load(b)
    out = dict(kp_a=len(xa), kp_b=len(xb), mutual=0, inl_aff=0, inl_h=0)
    if len(xa) < 2 or len(xb) < 2: return out
    ra, rb = root(da), root(db)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m1 = bf.knnMatch(ra, rb, k=2); m2 = bf.knnMatch(rb, ra, k=2)
    f = {m[0].queryIdx: m[0].trainIdx for m in m1 if len(m) == 2 and m[0].distance < 0.8 * m[1].distance}
    r = {m[0].queryIdx: m[0].trainIdx for m in m2 if len(m) == 2 and m[0].distance < 0.8 * m[1].distance}
    pairs = [(i, j) for i, j in f.items() if r.get(j) == i]
    out["ratio"] = len(f); out["mutual"] = len(pairs)
    if len(pairs) >= 4:
        pa = xa[[p[0] for p in pairs]].astype(np.float32); pb = xb[[p[1] for p in pairs]].astype(np.float32)
        side = max(xa.max(), xb.max(), 1.0)
        thr = 0.02 * side
        M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=2000)
        out["inl_aff"] = int(inl.sum()) if inl is not None else 0
        H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, thr)
        out["inl_h"] = int(inl.sum()) if inl is not None else 0
    return out

if __name__ == "__main__":
    tr = pd.read_csv(RAW / "train.csv"); te = pd.read_csv(RAW / "test.csv")
    allids = list(tr.id) + list(te.id)
    # ---- keypoint stats
    t0 = time.time()
    rows = []
    for i in allids:
        xy, de = load(i)
        rows.append((i, len(xy), *(xy.max(0) if len(xy) else (np.nan, np.nan)), *(xy.min(0) if len(xy) else (np.nan, np.nan)),
                     de.dtype == np.uint8 and (de.shape[1] == 128 if de.ndim == 2 else len(de) == 0), xy.dtype))
    kp = pd.DataFrame(rows, columns=["id", "n", "xmax", "ymax", "xmin", "ymin", "desc_ok", "xy_dtype"])
    print(f"loaded all SIFT in {time.time()-t0:.0f}s; desc dtype/shape ok: {kp.desc_ok.all()}; xy dtypes: {kp.xy_dtype.astype(str).unique()}")
    meta = pd.concat([tr.assign(split="train"), te.assign(split="test")]).reset_index(drop=True)
    kp = kp.merge(meta, on="id")
    print("xy range: x [%.1f, %.1f], y [%.1f, %.1f]" % (kp.xmin.min(), kp.xmax.max(), kp.ymin.min(), kp.ymax.max()))
    print("\nkeypoint count describe by species/split:")
    print(kp.groupby(["species", "split"]).n.describe(percentiles=[.05, .25, .5]).round(1).to_string())
    print("\nshare n==0 / n<50 / n<500 by species/split:")
    print(kp.groupby(["species", "split"]).n.agg(zero=lambda s: (s == 0).mean(), lt50=lambda s: (s < 50).mean(),
                                                   lt500=lambda s: (s < 500).mean()).round(3).to_string())
    print("\nmedian kp by viewpoint (species):")
    print(kp.assign(viewpoint=kp.viewpoint.fillna("NaN")).groupby(["species", "viewpoint"]).n.median().unstack(0).to_string())
    # bounding extent of keypoints as crude cut-out size / aspect
    kp["ext_x"] = kp.xmax - kp.xmin; kp["ext_y"] = kp.ymax - kp.ymin
    print("\nkeypoint extent (approx cut-out size) median by species/split:")
    print(kp.groupby(["species", "split"])[["xmax", "ymax"]].median().to_string())
    kp.to_csv(r"e:\ML_projectr\eda\kp_stats.csv", index=False)

    # ---- pilot: shortlisted pairs (train-train), cosine vs RANSAC inliers
    X = np.load(RAW / "miewid_features.npy").astype(np.float64); U = X / np.linalg.norm(X, axis=1, keepdims=True)
    Utr = U[:len(tr)]
    rng = np.random.default_rng(0)
    jobs, meta_rows = [], []
    for sp in ["hyena", "leopard"]:
        idx = np.where(tr.species == sp)[0]
        Z = Utr[idx]; y = tr.identity.values[idx]
        S = Z @ Z.T; np.fill_diagonal(S, -np.inf)
        qs = rng.choice(len(idx), 250, replace=False)
        for qi in qs:
            for r, gj in enumerate(np.argsort(-S[qi])[:12]):
                jobs.append((tr.id.values[idx[qi]], tr.id.values[idx[gj]]))
                meta_rows.append((sp, S[qi, gj], r, y[qi] == y[gj],
                                  tr.viewpoint.values[idx[qi]], tr.viewpoint.values[idx[gj]]))
    t0 = time.time()
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(match_pair, jobs, chunksize=50))
    dt = time.time() - t0
    print(f"\nmatched {len(jobs)} pairs in {dt:.0f}s ({1000*dt/len(jobs)*10:.1f} ms/pair/core)")
    P = pd.concat([pd.DataFrame(meta_rows, columns=["sp", "cos", "rank", "same", "vq", "vg"]), pd.DataFrame(res)], axis=1)
    P.to_csv(r"e:\ML_projectr\eda\sift_pilot.csv", index=False)
    side = {"left": "L", "frontleft": "L", "backleft": "L", "right": "R", "frontright": "R", "backright": "R"}
    P["sameside"] = P.vq.map(side).fillna("N") == P.vg.map(side).fillna("N")
    for sp, g in P.groupby("sp"):
        print(f"\n-- {sp}: pairs={len(g)}, positive rate={g.same.mean():.3f}")
        print("inliers(aff) median same=%.0f diff=%.0f | p90 diff=%.0f" % (g.inl_aff[g.same].median(), g.inl_aff[~g.same].median(), g.inl_aff[~g.same].quantile(.9)))
        for name, sub in [("all", g), ("cos<0.6 (cross-encounter regime)", g[g.cos < 0.6]), ("cos>=0.6", g[g.cos >= 0.6])]:
            if sub.same.nunique() < 2: continue
            Xf = np.c_[sub.cos, np.log1p(sub.inl_aff), np.log1p(sub.inl_h), np.log1p(sub.mutual)]
            lr = cross_val_predict(LogisticRegression(max_iter=1000), Xf, sub.same, cv=5, method="predict_proba")[:, 1]
            print(f"  {name:34s} n={len(sub):5d} pos={sub.same.mean():.2f} AUC cos={roc_auc_score(sub.same, sub.cos):.3f} "
                  f"inl_aff={roc_auc_score(sub.same, sub.inl_aff):.3f} inl_h={roc_auc_score(sub.same, sub.inl_h):.3f} "
                  f"mutual={roc_auc_score(sub.same, sub.mutual):.3f} LR(all)={roc_auc_score(sub.same, lr):.3f}")
        # inliers by same-side vs cross-side for positives
        pos = g[g.same]
        print("  positives: median inl_aff same-side=%.0f cross-side=%.0f (n=%d/%d)" % (
            pos.inl_aff[pos.sameside].median(), pos.inl_aff[~pos.sameside].median(), pos.sameside.sum(), (~pos.sameside).sum()))
