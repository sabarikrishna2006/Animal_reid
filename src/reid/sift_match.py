"""Stage B: matching + geometric verification of the PROVIDED SIFT keypoints.

RootSIFT (L1-normalise, sqrt) -> unit-L2 vectors, so squared Euclidean = 2 - 2 * dot product.
Nearest neighbours via one matrix product; Lowe ratio test in both directions, keep mutual matches;
RANSAC partial affine (4 dof, no reflection) + homography via OpenCV. All features are symmetric in (a, b)
except where noted, and NaN when undefined (e.g. 0 keypoints).
"""
import numpy as np
import cv2

FEATS = ["kp_min", "kp_max", "n_ratio", "n_mutual", "inl_aff", "inl_h", "inl_ratio", "inl_over_kp",
         "inl_sim", "log_scale", "abs_rot", "spread_min", "spread_max", "inl_aff_strict"]
NF = len(FEATS)


def rootsift(desc):
    d = desc.astype(np.float32)
    d /= d.sum(1, keepdims=True) + 1e-7
    return np.sqrt(d)


def _ratio_nn(Sim, ratio):
    """For each row: index of best column and whether it passes the ratio test (on Euclidean distances)."""
    if Sim.shape[1] < 2:
        return np.zeros(Sim.shape[0], int), np.zeros(Sim.shape[0], bool)
    part = np.argpartition(-Sim, 1, axis=1)[:, :2]
    s = np.take_along_axis(Sim, part, 1)
    swap = s[:, 1] > s[:, 0]
    best = np.where(swap, part[:, 1], part[:, 0])
    s1 = np.maximum(s.max(1), -1); s2 = s.min(1)
    d1 = np.sqrt(np.maximum(2 - 2 * s1, 0)); d2 = np.sqrt(np.maximum(2 - 2 * s2, 0))
    return best, d1 < ratio * d2


def _spread(pts, allpts):
    sa = allpts.std(0).mean()
    return float(pts.std(0).mean() / sa) if sa > 0 else np.nan


def match_features(xa, ra, xb, rb, ratio=0.8, ransac_frac=0.02):
    """xa: (n,2) keypoints, ra: (n,128) RootSIFT. Returns feature vector of length NF."""
    f = np.full(NF, np.nan, np.float32)
    na, nb = len(xa), len(xb)
    f[0], f[1] = min(na, nb), max(na, nb)
    if na < 2 or nb < 2:
        return f
    Sim = ra @ rb.T
    ba, oka = _ratio_nn(Sim, ratio)
    bb, okb = _ratio_nn(Sim.T, ratio)
    f[2] = oka.sum() + okb.sum()
    ia = np.where(oka & okb[ba] & (bb[ba] == np.arange(na)))[0]
    ib = ba[ia]
    f[3] = len(ia)
    f[4:6] = 0
    if len(ia) < 4:
        return f
    pa = xa[ia].astype(np.float32); pb = xb[ib].astype(np.float32)
    side = float(max(xa.max(), xb.max(), 1.0))
    thr = ransac_frac * side
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=2000,
                                         confidence=0.995)
    if M is not None and inl is not None:
        inl = inl.ravel().astype(bool)
        k = int(inl.sum())
        f[4] = k
        f[6] = k / len(ia)
        f[7] = k / min(na, nb)
        if k:
            f[8] = float(Sim[ia[inl], ib[inl]].mean())
        sc = float(np.hypot(M[0, 0], M[1, 0]))
        f[9] = np.log(max(sc, 1e-6))
        f[10] = abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if k >= 3:
            s1, s2 = _spread(pa[inl], xa), _spread(pb[inl], xb)
            f[11], f[12] = min(s1, s2), max(s1, s2)
        # strict: inliers only counted if transform is plausible (scale within [0.5, 2], rotation < 45 deg)
        f[13] = k if (0.5 <= sc <= 2.0 and f[10] < 45) else 0
    H, inlh = cv2.findHomography(pa, pb, cv2.RANSAC, thr)
    if H is not None and inlh is not None:
        f[5] = int(inlh.sum())
    return f
