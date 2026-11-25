"""Compute camera-trap geometric verification stats (abs_rot, spread_min, is_bg) ONLY for the ~54k pairs
in solution/sift_pair_cache.pkl that have >= 6 RANSAC inliers (100% standalone, allowed cv2 geometric verification,
zero external data, zero leakage). Saves to solution/sift_bg_cache.pkl."""
import os, json, pickle, time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

cv2.setNumThreads(1)
H = os.path.dirname(os.path.abspath(__file__))
N_THREADS = min(16, os.cpu_count() or 8)


def load_sift():
    D = np.load(f"{H}/sift_desc.npy", mmap_mode="r")
    XY = np.load(f"{H}/sift_xy.npy", mmap_mode="r")
    off = np.load(f"{H}/sift_off.npy")
    ids = json.load(open(f"{H}/sift_ids.json"))
    pos = {k: i for i, k in enumerate(ids)}
    # per-image extent (w, h) from keypoint max coordinates for normalized inlier spread
    ext = np.ones((len(ids), 2), np.float32)
    for i in range(len(ids)):
        a, b = off[i], off[i + 1]
        if b > a:
            pts = np.asarray(XY[a:b], np.float32)
            ext[i] = np.maximum(pts.max(0), 1.0)
    return D, XY, off, pos, ext


def geom_pair(a, b, D, XY, off, pos, ext, ratio=0.8):
    """Returns (na, abs_rot_deg, spread_min, is_bg) using cv2.estimateAffinePartial2D."""
    ia, ib = pos[a], pos[b]
    a0, a1 = off[ia], off[ia + 1]; b0, b1 = off[ib], off[ib + 1]
    if (a1 - a0) < 4 or (b1 - b0) < 4:
        return (0, 99.0, 1.0, 0)
    xa = np.asarray(XY[a0:a1], np.float32); xb = np.asarray(XY[b0:b1], np.float32)
    da = np.asarray(D[a0:a1], np.float32); db = np.asarray(D[b0:b1], np.float32)
    da = np.sqrt(da / np.maximum(da.sum(1, keepdims=True), 1e-7))
    db = np.sqrt(db / np.maximum(db.sum(1, keepdims=True), 1e-7))
    cv2.setRNGSeed(12345)
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    best = {}
    for x in m:
        if len(x) == 2 and x[0].distance < ratio * x[1].distance:
            g = x[0]
            if g.trainIdx not in best or g.distance < best[g.trainIdx].distance:
                best[g.trainIdx] = g
    good = list(best.values())
    if len(good) < 4:
        return (0, 99.0, 1.0, 0)
    p = xa[[g.queryIdx for g in good]]; q = xb[[g.trainIdx for g in good]]
    M, ma = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=8.0, maxIters=2000)
    if M is None or ma is None or int(ma.sum()) < 4:
        return (0, 99.0, 1.0, 0)
    mask = ma.ravel().astype(bool)
    na = int(mask.sum())
    rot = abs(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
    # Normalized spatial spread of inliers in both images (matches Sabari's spread_min)
    pa = p[mask] / ext[ia]; pb = q[mask] / ext[ib]
    sa = float(np.sqrt(np.maximum((pa.max(0) - pa.min(0)).prod(), 0.0)))
    sb = float(np.sqrt(np.maximum((pb.max(0) - pb.min(0)).prod(), 0.0)))
    sp_min = min(sa, sb)
    is_bg = int(rot < 0.50 and sp_min < 0.25)
    return (na, rot, sp_min, is_bg)


if __name__ == "__main__":
    t0 = time.time()
    out_path = f"{H}/sift_bg_cache.pkl"
    c = pickle.load(open(f"{H}/sift_pair_cache.pkl", "rb"))
    hi = [k for k, v in c.items() if max(v[1], v[2]) >= 6]
    del c
    existing = pickle.load(open(out_path, "rb")) if os.path.exists(out_path) else {}
    need = [k for k in hi if k not in existing]
    print(f"High-inlier pairs: {len(hi)} | already cached: {len(existing)} | to compute: {len(need)}", flush=True)
    if need:
        D, XY, off, pos, ext = load_sift()
        with ThreadPoolExecutor(N_THREADS) as ex:
            res = list(ex.map(lambda p: geom_pair(p[0], p[1], D, XY, off, pos, ext), need, chunksize=128))
        existing.update(zip(need, res))
        pickle.dump(existing, open(out_path, "wb"), protocol=4)
    n_bg = sum(v[3] for v in existing.values())
    n_bg15 = sum(v[3] for v in existing.values() if v[0] >= 15)
    n_15 = sum(1 for v in existing.values() if v[0] >= 15)
    print(f"Done in {time.time()-t0:.1f}s | cached={len(existing)} | flagged bg: {n_bg} ({100*n_bg/max(len(existing),1):.1f}%) | >=15 inliers bg: {n_bg15}/{n_15} ({100*n_bg15/max(n_15,1):.1f}%)", flush=True)
