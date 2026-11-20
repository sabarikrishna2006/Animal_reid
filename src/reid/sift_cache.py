"""Label-free pair cache: (row_a, row_b) with row_a < row_b -> SIFT pair features. Shared by all folds and test.

Which pairs get USED is decided per fold from fold-legal information; the cache content itself is label-free.
Missing pairs are computed in parallel (ProcessPoolExecutor) and appended.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .io import P
from .sift_match import NF, match_features, rootsift

_W = {}


def _init(cfg):
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(1)
    from .io import Sift
    _W["sift"] = Sift(cfg)
    _W["ratio"] = cfg["sift"]["ratio"]; _W["frac"] = cfg["sift"]["ransac_frac"]


def _work(group):
    """group: (a, [b1, b2, ...]) -> (a, bs, feats)"""
    a, bs = group
    s = _W["sift"]
    xa, da = s[a]
    ra = rootsift(da) if len(da) else da.astype(np.float32)
    out = np.zeros((len(bs), NF), np.float32)
    for j, b in enumerate(bs):
        xb, db = s[b]
        rb = rootsift(db) if len(db) else db.astype(np.float32)
        out[j] = match_features(xa, ra, xb, rb, _W["ratio"], _W["frac"])
    return a, bs, out


class PairCache:
    def __init__(self, cfg):
        self.cfg = cfg
        tag = f"r{cfg['sift']['ratio']}_f{cfg['sift']['ransac_frac']}"
        self.path = P(cfg, "cache") / f"sift_pairs_{tag}.npz"
        self.map = {}
        if self.path.exists():
            z = np.load(self.path)
            for (a, b), f in zip(z["keys"], z["feats"]):
                self.map[(int(a), int(b))] = f

    @staticmethod
    def key(a, b):
        return (a, b) if a < b else (b, a)

    def ensure(self, pairs, workers=None, verbose=True):
        need = {self.key(int(a), int(b)) for a, b in pairs if a != b}
        need = [k for k in need if k not in self.map]
        if not need:
            return 0
        groups = {}
        for a, b in need:
            groups.setdefault(a, []).append(b)
        items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        t0 = time.time()
        workers = workers or self.cfg["sift"]["workers"]
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(self.cfg,)) as ex:
            for a, bs, feats in ex.map(_work, items, chunksize=8):
                for b, f in zip(bs, feats):
                    self.map[(a, b)] = f
        if verbose:
            print(f"  [sift] computed {len(need)} pairs in {time.time() - t0:.0f}s "
                  f"({1000 * (time.time() - t0) / len(need):.2f} ms/pair wall)", flush=True)
        self.save()
        return len(need)

    def save(self):
        keys = np.array(list(self.map.keys()), np.int32).reshape(-1, 2)
        feats = np.array(list(self.map.values()), np.float32).reshape(-1, NF)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, feats=feats)
        os.replace(tmp, self.path)

    def get(self, a, b):
        return self.map.get(self.key(int(a), int(b)))
