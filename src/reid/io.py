"""The ONLY module that reads raw competition files.

Returns aligned arrays: row i of the embedding matrix corresponds to meta.iloc[i] (train rows first, then test).
SIFT features (20k small .npy files) are packed once into three arrays in data/cache for fast, mmap-able access.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    base = json.loads((ROOT / "configs" / "base.json").read_text())
    if path:
        over = json.loads(Path(path).read_text())
        base = deep_merge(base, over)
    return base


def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def P(cfg, key):
    return ROOT / cfg["paths"][key]


@dataclass
class Data:
    meta: pd.DataFrame        # id, species, viewpoint, identity (NaN for test), split ('train'/'test'), row
    X: np.ndarray             # raw MiewID (n, 2152) float32, aligned with meta
    ntr: int

    @property
    def train(self):
        return self.meta.iloc[: self.ntr]

    @property
    def test(self):
        return self.meta.iloc[self.ntr:]


def load_data(cfg) -> Data:
    raw = P(cfg, "raw")
    tr = pd.read_csv(raw / "train.csv", dtype=str)
    te = pd.read_csv(raw / "test.csv", dtype=str)
    ids = pd.read_csv(raw / "miewid_ids.csv", dtype=str)["id"]
    X = np.load(raw / "miewid_features.npy")
    assert X.shape == (len(ids), 2152) and X.dtype == np.float32
    assert not tr.id.duplicated().any() and not te.id.duplicated().any() and not ids.duplicated().any()
    assert set(ids) == set(tr.id) | set(te.id) and len(ids) == len(tr) + len(te)
    assert np.isfinite(X).all()
    meta = pd.concat([tr.assign(split="train"), te.assign(split="test", identity=np.nan)], ignore_index=True)
    # align X to meta order
    pos = pd.Series(np.arange(len(ids)), index=ids.values)
    X = X[pos.loc[meta.id].values]
    meta["row"] = np.arange(len(meta))
    for sp in meta.species.unique():
        assert sp in ("hyena", "leopard"), sp
    assert (meta[meta.split == "train"].groupby("identity").species.nunique() == 1).all()
    return Data(meta=meta, X=X, ntr=len(tr))


# ---------------------------------------------------------------- SIFT
def _sift_paths(cfg):
    c = P(cfg, "cache")
    return c / "sift_xy.npy", c / "sift_desc.npy", c / "sift_off.npy"


def build_sift_pack(cfg, data: Data):
    """Pack every image's keypoints into contiguous arrays in meta row order (run once)."""
    pxy, pde, poff = _sift_paths(cfg)
    if poff.exists():
        return
    pxy.parent.mkdir(parents=True, exist_ok=True)
    xs, ds, off = [], [], [0]
    # each SIFT source may be the original .npz FILE or a folder of extracted .npy files
    src = {}
    for key in ("sift_train_dir", "sift_test_dir"):
        p = P(cfg, key)
        src[key] = np.load(p) if p.is_file() else p

    def get(key, sid, kind):
        s = src[key]
        return s[f"{sid}_{kind}"] if not isinstance(s, Path) else np.load(s / f"{sid}_{kind}.npy")

    for i, sid in enumerate(data.meta.id):
        key = "sift_train_dir" if sid.startswith("tr_") else "sift_test_dir"
        xy = get(key, sid, "xy"); de = get(key, sid, "desc")
        if len(xy) == 0:
            xy = np.zeros((0, 2), np.float32); de = np.zeros((0, 128), np.uint8)
        assert xy.dtype == np.float32 and xy.shape[1] == 2 and de.dtype == np.uint8 and de.shape == (len(xy), 128)
        assert len(xy) == 0 or (xy.min() >= 0 and xy.max() <= 512)
        xs.append(xy); ds.append(de); off.append(off[-1] + len(xy))
    np.save(pxy, np.concatenate(xs)); np.save(pde, np.concatenate(ds)); np.save(poff, np.array(off, np.int64))


class Sift:
    """Lazy accessor: Sift(cfg)[row] -> (xy, desc). Arrays are memory-mapped."""

    def __init__(self, cfg):
        pxy, pde, poff = _sift_paths(cfg)
        self.xy = np.load(pxy, mmap_mode="r"); self.desc = np.load(pde, mmap_mode="r"); self.off = np.load(poff)

    def __getitem__(self, row):
        a, b = self.off[row], self.off[row + 1]
        return np.asarray(self.xy[a:b]), np.asarray(self.desc[a:b])

    def n_kp(self):
        return np.diff(self.off)
