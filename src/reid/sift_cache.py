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
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(1)
    from .io import Sift
    _W["sift"] = Sift(cfg)
    _W["rs"] = {}
    _W["ratio"] = cfg["sift"]["ratio"]; _W["frac"] = cfg["sift"]["ransac_frac"]


def _get_rs(idx):
    rs = _W["rs"]
    v = rs.get(idx)
    if v is None:
        if len(rs) >= 500:
            rs.clear()
        x, d = _W["sift"][idx]
        v = (x, rootsift(d) if len(d) else d.astype(np.float32))
        rs[idx] = v
    return v


def _work(group):
    """group: (a, [b1, b2, ...]) -> (a, bs, feats)"""
    a, bs = group
    xa, ra = _get_rs(a)
    out = np.zeros((len(bs), NF), np.float32)
    for j, b in enumerate(bs):
        xb, rb = _get_rs(b)
        out[j] = match_features(xa, ra, xb, rb, _W["ratio"], _W["frac"])
    return a, bs, out


def _pack_pair(a, b):
    a, b = int(a), int(b)
    return ((a << 24) | b) if a < b else ((b << 24) | a)


class PairCache:
    def __init__(self, cfg):
        self.cfg = cfg
        tag = f"r{cfg['sift']['ratio']}_f{cfg['sift']['ransac_frac']}"
        self.path = P(cfg, "cache") / f"sift_pairs_{tag}.npz"
        self._keys64 = np.zeros(0, dtype=np.int64)
        self._feats = np.zeros((0, NF), dtype=np.float32)
        self._pending = {}
        self._disk_size = -1
        self._merge_disk()

    @staticmethod
    def key(a, b):
        return (a, b) if a < b else (b, a)

    def _merge_disk(self):
        if not self.path.exists():
            return
        sz = self.path.stat().st_size
        if sz == self._disk_size and len(self._keys64) > 0:
            return
        z = np.load(self.path)
        k = z["keys"]
        f = np.asarray(z["feats"], dtype=np.float32).reshape(-1, NF)
        z.close()
        if len(k):
            k64 = ((np.minimum(k[:, 0], k[:, 1]).astype(np.int64) << 24)
                   | np.maximum(k[:, 0], k[:, 1]).astype(np.int64))
            if len(self._keys64):
                all_k = np.concatenate([self._keys64, k64])
                all_f = np.concatenate([self._feats, f], axis=0)
            else:
                all_k, all_f = k64, f
            _, uniq_idx = np.unique(all_k, return_index=True)
            self._keys64 = all_k[uniq_idx]
            self._feats = all_f[uniq_idx]
        self._disk_size = sz

    def ensure(self, pairs, workers=None, verbose=True):
        self._merge_disk()
        arr = np.asarray(pairs, dtype=np.int64)
        if len(arr) == 0:
            return 0
        arr = arr[arr[:, 0] != arr[:, 1]]
        if len(arr) == 0:
            return 0
        req_k = np.unique((np.minimum(arr[:, 0], arr[:, 1]) << 24) | np.maximum(arr[:, 0], arr[:, 1]))
        if len(self._keys64):
            pos = np.searchsorted(self._keys64, req_k)
            pos_c = np.minimum(pos, len(self._keys64) - 1)
            miss_mask = (pos >= len(self._keys64)) | (self._keys64[pos_c] != req_k)
            req_k = req_k[miss_mask]
        need_k = [int(k) for k in req_k if int(k) not in self._pending]
        if not need_k:
            return 0
        groups = {}
        for k in need_k:
            groups.setdefault(k >> 24, []).append(k & 0xFFFFFF)
        items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        t0 = time.time()
        workers = workers or int(os.environ.get("REID_SIFT_WORKERS", 0)) or self.cfg["sift"]["workers"]
        done = 0
        next_log = 100_000
        next_save = 300_000
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(self.cfg,)) as ex:
            for a, bs, feats in ex.map(_work, items, chunksize=8):
                for b, f in zip(bs, feats):
                    self._pending[(int(a) << 24) | int(b)] = f
                done += len(bs)
                if verbose and done >= next_log:
                    dt = time.time() - t0
                    print(f"  [sift] {done}/{len(need_k)} pairs ({1000 * dt / done:.2f} ms/pair, {dt:.0f}s)", flush=True)
                    next_log += 100_000
                if done >= next_save:
                    self.save()
                    next_save += 300_000
        if verbose:
            print(f"  [sift] computed {len(need_k)} pairs in {time.time() - t0:.0f}s "
                  f"({1000 * (time.time() - t0) / len(need_k):.2f} ms/pair wall)", flush=True)
        self.save()
        return len(need_k)

    def save(self):
        self._merge_disk()
        if self._pending:
            pk = np.array(list(self._pending.keys()), dtype=np.int64)
            pf = np.array(list(self._pending.values()), dtype=np.float32).reshape(-1, NF)
            if len(self._keys64):
                all_k = np.concatenate([self._keys64, pk])
                all_f = np.concatenate([self._feats, pf], axis=0)
            else:
                all_k, all_f = pk, pf
            # Keep pending values on collision by placing them first via reverse or stable merge
            order = np.argsort(all_k, kind="stable")
            all_k, all_f = all_k[order], all_f[order]
            uniq = np.r_[True, all_k[1:] != all_k[:-1]]
            self._keys64 = all_k[uniq]
            self._feats = all_f[uniq]
            self._pending.clear()
        keys = np.column_stack([self._keys64 >> 24, self._keys64 & 0xFFFFFF]).astype(np.int32)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, feats=self._feats)
        os.replace(tmp, self.path)
        self._disk_size = self.path.stat().st_size

    def get(self, a, b):
        if a == b:
            return None
        k = _pack_pair(a, b)
        v = self._pending.get(k)
        if v is not None:
            return v
        if len(self._keys64) == 0:
            return None
        pos = int(np.searchsorted(self._keys64, k))
        if pos < len(self._keys64) and self._keys64[pos] == k:
            return self._feats[pos]
        return None
