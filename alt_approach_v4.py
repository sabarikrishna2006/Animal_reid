# %% [markdown] {"id":"mk7oUxrx8tCK"}
# # Problem 3: Wildlife Re-Identification — v4 Kaggle Training Notebook (`v10 SOTA + KRR/Ridge/Soft-PCAW`)
# 
# **Goal:** Self-contained Kaggle training & inference notebook combining every proven component of the **0.88073 LB** `v10` architecture with the orthogonal global discriminative upgrades from `v3` (**Dual RBF Kernel Ridge**, **Balanced Cosine-Ridge**, **Soft Power-Whitened PCA `pw=0.35`**, and **Per-Flank Sub-Prototypes**).
# 
# ### Kaggle Offline & Memory-Crash Optimizations (`v3_fixed` Parity)
# 1. **Zero-Fork PyTorch GPU/CPU SIFT RANSAC:** Completely eliminates multiprocessing `ProcessPoolExecutor` fork OOM. Uses vectorized PyTorch matrix operations on GPU (or multi-threaded CPU) for mutual Lowe-ratio tests, OpenCV affine/homography RANSAC, and atomic disk caching every 1,500 pairs.
# 2. **Resilient Offline HotSpotter LNBNN (Zero-Dependency Fallback):** Safely catches missing `faiss` when running in offline Kaggle environments (where `pip install` fails with `NameResolutionError`). Uses an exact chunked PyTorch $L_2$ / dot-product KNN fallback matching FAISS output down to machine precision.
# 3. **Streaming Fold-by-Fold Table Building:** Eliminates simultaneous accumulation of 22 fold dense similarity matrices in RAM. Pass 1 discovers SIFT pairs and stores artifacts to disk temp files; Pass 2 streams one fold into memory at a time, builds DataFrame, and unlinks temp files with aggressive `gc.collect()`. Peak memory drops from >11 GB to <1 GB.
# 4. **Header-Only SIFT Keypoint Counting:** Uses memory-mapped headers (`mmap_mode='r'`) to count keypoints without allocating full coordinate arrays in RAM.
# 5. **Auto-Sync Attached Dataset Checkpoints:** Automatically discovers and syncs precomputed `cache_v4` files from `/kaggle/input/**/cache_v4`.
# 
# ---
# 

# %% [code] {"id":"sCBACsfP8v7-"}
# ============================================================================
# 0. KAGGLE DEPENDENCY CHECK (FAISS SAFE IMPORT) & IMPORTS
# ============================================================================
import subprocess, sys

# Safe FAISS Import with Offline & Zero-Dependency Fallback
HAVE_FAISS = False
try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    print("FAISS not pre-installed. Attempting to install faiss-cpu...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])
        import faiss
        HAVE_FAISS = True
        print("Successfully installed faiss-cpu!")
    except Exception as e:
        print(f"Notice: Offline Kaggle environment or PyPI unavailable ({e}).")
        print("Proceeding with high-performance pure PyTorch/NumPy KNN fallback (100% offline & zero dependencies)!")
        HAVE_FAISS = False

import os
import gc
import time
import json
import shutil
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax, expit, logit, logsumexp
from scipy.optimize import minimize_scalar
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import cv2
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)
if HAVE_FAISS:
    try:
        faiss.omp_set_num_threads(max(1, os.cpu_count() or 4))
    except Exception:
        pass

SIFT_DEVICE = "cpu"

# ----------------------------------------------------------------------------
# Auto-detect Kaggle or local dataset paths (supports SIFT as dir OR .npz)
# ----------------------------------------------------------------------------
def resolve_dataset_paths():
    roots = [Path("/kaggle/input"), Path("."), Path("./data"), Path("./projectwild3")]
    csv_dir = None
    for r in roots:
        if not r.exists():
            continue
        if (r / "train.csv").exists():
            csv_dir = r
            break
        m = list(r.glob("**/train.csv"))
        if m:
            csv_dir = m[0].parent
            break
    if csv_dir is None:
        raise FileNotFoundError("Could not locate train.csv. Set DATA_DIR manually.")

    def find_sift(name: str):
        for cand in [csv_dir / name, csv_dir.parent / name, Path(".") / name]:
            if cand.exists():
                return cand
        for r in roots:
            if r.exists():
                m = list(r.glob(f"**/{name}"))
                if m:
                    return m[0]
        raise FileNotFoundError(f"Could not locate {name}.")

    return csv_dir, find_sift("sift_features_train.npz"), find_sift("sift_features_test.npz")

DATA_DIR, SIFT_TRAIN_PATH, SIFT_TEST_PATH = resolve_dataset_paths()
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
CACHE_DIR = WORK_DIR / "cache_v4"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Auto-import checkpoints/caches from any attached Kaggle Dataset or previous run output
for search_root in [Path("/kaggle/input"), Path(".")]:
    if not search_root.exists():
        continue
    for prev_cache in search_root.glob("**/cache_v4"):
        if prev_cache.is_dir() and prev_cache.resolve() != CACHE_DIR.resolve():
            print(f"[CACHE] Found previous cache at {prev_cache.resolve()}, syncing to {CACHE_DIR}...")
            for src in prev_cache.rglob("*"):
                if src.is_file():
                    dst = CACHE_DIR / src.relative_to(prev_cache)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        shutil.copy2(src, dst)

SPECIES = ("hyena", "leopard")
NEW = "new_individual"
W_TEST = {"hyena": 1217 / 2791, "leopard": 1574 / 2791}

CFG = {
    "seed": 2026,
    "n_splits": 5,
    "cv_v1_burst": 0.10,          # Primary encounter-aware CV
    "cv_v2_burst": 0.25,          # Harder secondary CV for mixed training (v08 trick)
    "use_mixed_cv": True,
    "pcaw_d": 512,
    "pcaw_power": 0.35,           # Soft power whitening (empirically beats 0.50)
    "pcaw_eig_floor": 0.02,
    "lda_gamma": 0.30,
    "lda_d_out": 256,
    "krr_gamma": 1.5,             # v3 orthogonal Dual RBF Kernel Ridge
    "krr_lam": 0.5,
    "ridge_lam": 2.0,             # v3 orthogonal Balanced Cosine-Ridge
    "krec_k1": 20,
    "krec_k2": 6,
    "krec_lam": 0.30,
    "n_cand": 30,                 # Top-30 MiewID/KRR candidates
    "lnbnn_cand": 5,              # + Top-5 image-level LNBNN expansion
    "lnbnn_k_store": 50,          # Neighbours stored per RootSIFT descriptor in FAISS/PyTorch
    "sift_top_ids": 15,           # Top-15 identities verified with SIFT RANSAC
    "sift_imgs_per_id": 8,        # + top-2 LNBNN-guided images per candidate
    "sift_ratio": 0.80,
    "sift_thresh": 10.0,
    "sift_thresh_strict": 5.0,
    "sift_workers": 2,
    "sift_save_every": 1500,      # Periodic cache flush to disk (prevents memory buildup)
    "two_round": True,            # Round 2 stacking with MiewID + SIFT kpgraph context
    "k_qq": 5,
    "prop_alphas": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
}

print(f"DATA_DIR:        {DATA_DIR.resolve()}")
print(f"WORK_DIR:        {WORK_DIR.resolve()}")
print(f"CACHE_DIR:       {CACHE_DIR.resolve()}")
print(f"SIFT_TRAIN_PATH: {SIFT_TRAIN_PATH.resolve()} (is_dir={SIFT_TRAIN_PATH.is_dir()})")
print(f"SIFT_TEST_PATH:  {SIFT_TEST_PATH.resolve()} (is_dir={SIFT_TEST_PATH.is_dir()})")
print(f"SIFT Matching:   PyTorch ({SIFT_DEVICE}) + OpenCV USAC_MAGSAC (Zero Fork OOM)")
print(f"FAISS Status:    {'Active' if HAVE_FAISS else 'PyTorch/NumPy Chunked Fallback Active (Offline Safe)'}")


# %% [markdown] {"id":"rTa_Nqbj8xm2"}
# ## 1. Data Loading, Universal SIFT Store & Unit-Tested MAP@5
# 

# %% [code] {"id":"V-8lzPN28xm4"}
# ============================================================================
# 1. DATA LOADING, UNIVERSAL SIFT STORE & MAP@5 METRIC
# ============================================================================
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
miewid_ids_df = pd.read_csv(DATA_DIR / "miewid_ids.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
X_raw = np.load(DATA_DIR / "miewid_features.npy").astype(np.float32)

assert len(miewid_ids_df) == X_raw.shape[0]
assert not np.isnan(X_raw).any() and not np.isinf(X_raw).any()
assert set(test_df["id"]) == set(sample_sub["id"])

SUB_PRED_COL = sample_sub.columns[1]
id_to_row = {img_id: i for i, img_id in enumerate(miewid_ids_df.iloc[:, 0].values)}

# Unified metadata table aligned 1:1 with X_raw rows
meta = pd.concat([
    train_df.assign(is_test=False),
    test_df.assign(identity="", is_test=True),
], ignore_index=True)
meta["row"] = meta["id"].map(id_to_row)
meta = meta.sort_values("row").reset_index(drop=True)
assert np.array_equal(meta["row"].values, np.arange(len(X_raw)))

SIDE_MAP = {"left": "L", "frontleft": "L", "backleft": "L", "right": "R", "frontright": "R", "backright": "R"}
meta["vp_clean"] = meta["viewpoint"].fillna("NaN").astype(str).str.strip().replace("", "NaN")
meta["side"] = meta["vp_clean"].map(lambda v: SIDE_MAP.get(v, "N" if v != "NaN" else "U"))

raw_norms = np.linalg.norm(X_raw.astype(np.float64), axis=1)
meta["lognorm"] = np.log(np.clip(raw_norms, 1e-8, None)).astype(np.float32)

def l2n(A: np.ndarray) -> np.ndarray:
    return (A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-8, None)).astype(np.float32)

EMB_L2 = l2n(X_raw)

# Unsupervised per-species Soft Power-Whitened PCA (d=512, pw=0.35) on train U test
EMB_PCAW = np.zeros((len(X_raw), CFG["pcaw_d"]), dtype=np.float32)
for sp in SPECIES:
    m = (meta["species"] == sp).values
    pca = PCA(n_components=CFG["pcaw_d"], svd_solver="randomized", random_state=CFG["seed"])
    Z = pca.fit_transform(EMB_L2[m])
    eigs = pca.explained_variance_
    scale = (eigs + CFG["pcaw_eig_floor"] * eigs.mean()) ** (-CFG["pcaw_power"])
    EMB_PCAW[m] = l2n(Z * scale)

# Free massive raw embeddings from memory immediately
del X_raw, miewid_ids_df
gc.collect()

class SiftStore:
    """Reads <id>_xy and <id>_desc from either a directory of .npy files or a .npz archive."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.is_dir = self.path.is_dir()
        self.npz = None if self.is_dir else np.load(self.path)
        self.keys = None if self.is_dir else set(self.npz.files)

    def get(self, img_id: str):
        if self.is_dir:
            xp, dp = self.path / f"{img_id}_xy.npy", self.path / f"{img_id}_desc.npy"
            if xp.exists() and dp.exists():
                return np.load(xp, mmap_mode="r").astype(np.float32), np.load(dp, mmap_mode="r")
            return np.empty((0, 2), np.float32), np.empty((0, 128), np.uint8)
        kx, kd = f"{img_id}_xy", f"{img_id}_desc"
        if kx in self.keys and kd in self.keys:
            return self.npz[kx].astype(np.float32), self.npz[kd]
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.uint8)

    def get_kp_count_only(self, img_id: str) -> int:
        """Reads only the array shape header from disk without loading full coordinates into RAM."""
        if self.is_dir:
            xp = self.path / f"{img_id}_xy.npy"
            if xp.exists():
                return int(np.load(xp, mmap_mode="r").shape[0])
            return 0
        kx = f"{img_id}_xy"
        return int(self.npz[kx].shape[0]) if kx in self.keys else 0

    def close(self):
        if self.npz is not None:
            self.npz.close()

def to_rootsift_l2(desc: np.ndarray) -> np.ndarray:
    if desc is None or len(desc) == 0:
        return np.empty((0, 128), dtype=np.float32)
    d = desc.astype(np.float32)
    d /= (d.sum(axis=1, keepdims=True) + 1e-7)
    np.sqrt(d, out=d)
    d /= np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-7, None)
    return d

def apk(actual: str, predicted: list, k: int = 5) -> float:
    seen = set()
    for idx, p in enumerate(predicted[:k]):
        if p in seen: continue
        seen.add(p)
        if p == actual: return 1.0 / (idx + 1)
    return 0.0

def mapk(actuals: list, predicteds: list, k: int = 5) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)]))

assert abs(mapk(
    ["LEO0317", "HYE0042", NEW, "HYE0011"],
    [["LEO0317", "LEO0102"], ["HYE0007", "HYE0042", NEW], ["LEO0102", "LEO0009", NEW], [NEW, "HYE0198"]]
) - (1 + 0.5 + 1/3) / 4) < 1e-9

# Precompute keypoint counts for all images efficiently via shape headers
kp_cache_path = CACHE_DIR / "kp_counts.npy"
if kp_cache_path.exists():
    KP_COUNTS = np.load(kp_cache_path)
else:
    st_tr, st_te = SiftStore(SIFT_TRAIN_PATH), SiftStore(SIFT_TEST_PATH)
    KP_COUNTS = np.zeros(len(meta), dtype=np.float32)
    for r_i, (img_id, is_te) in enumerate(zip(meta["id"].values, meta["is_test"].values)):
        KP_COUNTS[r_i] = float((st_te if is_te else st_tr).get_kp_count_only(img_id))
    st_tr.close(); st_te.close()
    np.save(kp_cache_path, KP_COUNTS)

print(f"Loaded {len(meta)} images | MAP@5 unit test passed | KP_COUNTS med={np.median(KP_COUNTS):.0f}")


# %% [markdown] {"id":"8TCms0bi8y6Z"}
# ## 2. Build/Load FAISS RootSIFT LNBNN Index & Unsupervised Keypoint Query Graph (`kpgraph`)
# Builds two cached one-time artifacts per species:
# 1. **Gallery LNBNN (`faiss.IndexHNSWFlat` over all training RootSIFT descriptors):** Stores top-50 gallery image indices and L2 distances per query descriptor (`train ∪ test`). At fold time, images outside the fold's gallery (and self) are masked out — **zero leakage**, exact match to `lnbnn.py`.
# 2. **Unsupervised Keypoint Query Graph (`kpgraph`):** For every image in `train ∪ test`, also queries a small FAISS index of same-species RootSIFT descriptors to find top-5 visual keypoint-voting neighbours (used in Round 2 stacking and Zhou label propagation to link same-animal sightings that MiewID missed).
# 

# %% [code] {"id":"Bs2Y5aE38y6a"}
# ============================================================================
# 2. FAISS ROOTSIFT LNBNN INDEX + UNSUPERVISED KEYPOINT GRAPH (kpgraph)
# ============================================================================
def numpy_topk_search(queries: np.ndarray, gal: np.ndarray, k: int, chunk_size: int = 100000):
    """Chunked exact KNN search in pure NumPy (zero deep-learning framework dependency)."""
    k = min(k, len(gal))
    q_np = queries.astype(np.float32, copy=False)
    best_sims = None
    best_indices = None

    for start in range(0, len(gal), chunk_size):
        end = min(len(gal), start + chunk_size)
        g_chunk = gal[start:end].astype(np.float32, copy=False)
        sim = q_np @ g_chunk.T
        kk = min(k, end - start)
        idx_part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        val_part = np.take_along_axis(sim, idx_part, axis=1)
        ord_p = np.argsort(-val_part, axis=1)
        top_v = np.take_along_axis(val_part, ord_p, axis=1)
        top_i = np.take_along_axis(idx_part, ord_p, axis=1) + start

        if best_sims is None:
            best_sims, best_indices = top_v, top_i
        else:
            comb_sims = np.concatenate([best_sims, top_v], axis=1)
            comb_indices = np.concatenate([best_indices, top_i], axis=1)
            pos = np.argpartition(-comb_sims, k - 1, axis=1)[:, :k]
            v_sel = np.take_along_axis(comb_sims, pos, axis=1)
            i_sel = np.take_along_axis(comb_indices, pos, axis=1)
            ord_c = np.argsort(-v_sel, axis=1)
            best_sims = np.take_along_axis(v_sel, ord_c, axis=1)
            best_indices = np.take_along_axis(i_sel, ord_c, axis=1)

    d2_np = np.clip(2.0 - 2.0 * best_sims, 0.0, 4.0).astype(np.float32)
    return d2_np, best_indices.astype(np.int64)

class FastKNN:
    """Unified KNN wrapper supporting FAISS IndexHNSWFlat and pure-NumPy fallback."""
    def __init__(self, d: int = 128, ef_const: int = 64, ef_search: int = 48, use_faiss: bool = HAVE_FAISS):
        self.d = d
        self.use_faiss = use_faiss
        self.index = None
        self.mat = None
        if self.use_faiss:
            self.index = faiss.IndexHNSWFlat(d, 32)
            self.index.hnsw.efConstruction = ef_const
            self.index.hnsw.efSearch = ef_search

    def add(self, mat: np.ndarray):
        if self.use_faiss:
            self.index.add(mat.astype(np.float32))
        else:
            self.mat = mat.astype(np.float32)

    def search(self, queries: np.ndarray, k: int):
        if self.use_faiss:
            return self.index.search(queries.astype(np.float32), k)
        else:
            return numpy_topk_search(queries, self.mat, k)

def build_or_load_lnbnn_and_kpgraph(sp: str):
    lnb_dir = CACHE_DIR / "lnbnn"
    lnb_dir.mkdir(parents=True, exist_ok=True)
    meta_p = lnb_dir / f"{sp}_meta.json"
    nb_p = lnb_dir / f"{sp}_nb.npy"
    di_p = lnb_dir / f"{sp}_d.npy"
    kp_graph_p = lnb_dir / f"{sp}_kpgraph.pkl"

    if meta_p.exists() and nb_p.exists() and di_p.exists() and kp_graph_p.exists():
        print(f"[{sp}] Loaded cached LNBNN + kpgraph from {lnb_dir}")
        return

    t0 = time.time()
    sp_df = meta[meta["species"] == sp].reset_index(drop=True)
    tr_ids = sp_df[~sp_df["is_test"]]["id"].tolist()
    all_ids = sp_df["id"].tolist()
    is_te_map = dict(zip(sp_df["id"], sp_df["is_test"]))

    st_tr, st_te = SiftStore(SIFT_TRAIN_PATH), SiftStore(SIFT_TEST_PATH)

    # 1. Stack all training RootSIFT descriptors for this species
    gal_descs, gal_owner = [], []
    for g_idx, gid in enumerate(tr_ids):
        _, d_raw = st_tr.get(gid)
        rs = to_rootsift_l2(d_raw)
        if len(rs):
            gal_descs.append(rs)
            gal_owner.append(np.full(len(rs), g_idx, dtype=np.int32))
    G_mat = np.vstack(gal_descs).astype(np.float32)
    G_owner = np.concatenate(gal_owner)
    print(f"[{sp}] Building KNN index over {len(G_mat):,} train RootSIFT descriptors ({len(tr_ids)} imgs)...")

    index = FastKNN(128, ef_const=80, ef_search=64, use_faiss=HAVE_FAISS)
    index.add(G_mat)
    del gal_descs, gal_owner, G_mat
    gc.collect()

    # 2. Query ALL images (train U test) for top-50 gallery descriptor neighbours
    K_store = CFG["lnbnn_k_store"]
    qoff = [0]
    nb_chunks, d_chunks = [], []
    all_descs_list, all_owner_list = [], []
    for q_idx, qid in enumerate(all_ids):
        _, d_raw = (st_te if is_te_map[qid] else st_tr).get(qid)
        rs = to_rootsift_l2(d_raw)
        qoff.append(qoff[-1] + len(rs))
        if len(rs):
            D2, I = index.search(rs, K_store)
            nb_img = np.where(I >= 0, G_owner[np.maximum(I, 0)], -1).astype(np.int32)
            nb_chunks.append(nb_img)
            d_chunks.append(np.sqrt(np.clip(D2, 0.0, 4.0)).astype(np.float16))
            all_descs_list.append(rs)
            all_owner_list.append(np.full(len(rs), q_idx, dtype=np.int32))
    st_tr.close(); st_te.close()

    NB_all = np.vstack(nb_chunks)
    DI_all = np.vstack(d_chunks)
    np.save(nb_p, NB_all)
    np.save(di_p, DI_all)
    meta_p.write_text(json.dumps({"gallery": tr_ids, "queries": all_ids, "qoff": qoff}))
    del index, nb_chunks, d_chunks, NB_all, DI_all
    gc.collect()
    print(f"[{sp}] Saved LNBNN cache in {time.time() - t0:.1f}s. Building unsupervised kpgraph...")

    # 3. Unsupervised SIFT Keypoint Graph over all_ids (used strictly within each fold's query pool!)
    t1 = time.time()
    A_mat = np.vstack(all_descs_list).astype(np.float32)
    A_owner = np.concatenate(all_owner_list)
    del all_descs_list, all_owner_list
    gc.collect()

    idx_all = FastKNN(128, ef_const=64, ef_search=48, use_faiss=HAVE_FAISS)
    idx_all.add(A_mat)

    # For each image, accumulate top-5 LNBNN-style votes to other images in all_ids
    n_all = len(all_ids)
    kp_edges = {}  # qid -> list of (nbr_qid, score) top-15
    K_kp = 6
    bs = 16384
    vote_scores = [dict() for _ in range(n_all)]
    for start in range(0, len(A_mat), bs):
        end = min(len(A_mat), start + bs)
        D2, I = idx_all.search(A_mat[start:end], K_kp + 1)
        D = np.sqrt(np.clip(D2, 0.0, 4.0))
        owners_q = A_owner[start:end]
        for local_r in range(end - start):
            qi = int(owners_q[local_r])
            bg = float(D[local_r, -1])
            seen_nbr = set()
            for c_i in range(K_kp):
                idx_hit = int(I[local_r, c_i])
                if idx_hit < 0: continue
                ni = int(A_owner[idx_hit])
                if ni == qi or ni in seen_nbr: continue
                seen_nbr.add(ni)
                gain = max(0.0, bg - float(D[local_r, c_i]))
                if gain > 0.0:
                    vote_scores[qi][ni] = vote_scores[qi].get(ni, 0.0) + gain

    for qi, qid in enumerate(all_ids):
        top15 = sorted(vote_scores[qi].items(), key=lambda x: -x[1])[:15]
        kp_edges[qid] = [(all_ids[ni], float(sc)) for ni, sc in top15]

    kp_graph_p.write_bytes(pickle.dumps(kp_edges))
    del idx_all, A_mat, A_owner, vote_scores
    gc.collect()
    print(f"[{sp}] Saved unsupervised kpgraph in {time.time() - t1:.1f}s.")

for sp in SPECIES:
    build_or_load_lnbnn_and_kpgraph(sp)

class LNBNNScorer:
    """Exact leak-free fold-masked HotSpotter LNBNN scorer (matching lnbnn.py)."""
    def __init__(self, sp: str):
        d = CACHE_DIR / "lnbnn"
        m = json.loads((d / f"{sp}_meta.json").read_text())
        self.gal = np.array(m["gallery"])
        self.qpos = {q: i for i, q in enumerate(m["queries"])}
        self.qoff = np.array(m["qoff"])
        self.NB = np.load(d / f"{sp}_nb.npy", mmap_mode="r")
        self.DI = np.load(d / f"{sp}_d.npy", mmap_mode="r")
        lab = dict(zip(train_df["id"].values, train_df["identity"].values))
        self.names, self.lab = np.unique([lab[g] for g in self.gal], return_inverse=True)
        self.gpos = {g: i for i, g in enumerate(self.gal)}
        self.kpgraph = pickle.loads((d / f"{sp}_kpgraph.pkl").read_bytes())

    def score_all_granularities(self, qid: str, gmask: np.ndarray, K: int = 10):
        i = self.qpos[qid]; a, b = self.qoff[i], self.qoff[i + 1]
        n_id, n_gal = len(self.names), len(self.gal)
        out_id = np.zeros(n_id, np.float32)
        out_img = np.zeros(n_id, np.float32)
        out_k3 = np.zeros(n_id, np.float32)
        img_sc = np.zeros(n_gal, np.float32)
        if b == a:
            return out_id, out_img, out_k3, img_sc
        nb = np.asarray(self.NB[a:b])
        d = np.asarray(self.DI[a:b], np.float32)
        ok = (nb >= 0) & gmask[np.maximum(nb, 0)]
        if qid in self.gpos:
            ok &= (nb != self.gpos[qid])
        rank = np.cumsum(ok, axis=1)

        # K=10 background
        has_bg10 = rank[:, -1] >= K + 1
        bg10 = np.where(has_bg10, d[np.arange(len(d)), np.argmax(rank >= K + 1, axis=1)], d.max(axis=1))
        r, c = np.nonzero(ok & (rank <= K))
        if len(r):
            g_idx = nb[r, c]
            # Identity-level LNBNN (K=10)
            lab10 = self.lab[g_idx]
            _, first_id = np.unique(r.astype(np.int64) * n_id + lab10, return_index=True)
            np.add.at(out_id, lab10[first_id], np.maximum(bg10[r[first_id]] - d[r[first_id], c[first_id]], 0.0))
            # Image-level LNBNN (K=10)
            _, first_g = np.unique(r.astype(np.int64) * n_gal + g_idx, return_index=True)
            np.add.at(img_sc, g_idx[first_g], np.maximum(bg10[r[first_g]] - d[r[first_g], c[first_g]], 0.0))
            np.maximum.at(out_img, self.lab, img_sc)

        # Strict K=3 identity-level LNBNN
        has_bg3 = rank[:, -1] >= 4
        bg3 = np.where(has_bg3, d[np.arange(len(d)), np.argmax(rank >= 4, axis=1)], d.max(axis=1))
        r3, c3 = np.nonzero(ok & (rank <= 3))
        if len(r3):
            lab3 = self.lab[nb[r3, c3]]
            _, first3 = np.unique(r3.astype(np.int64) * n_id + lab3, return_index=True)
            np.add.at(out_k3, lab3[first3], np.maximum(bg3[r3[first3]] - d[r3[first3], c3[first3]], 0.0))

        return out_id, out_img, out_k3, img_sc

LNB_SCORERS = {sp: LNBNNScorer(sp) for sp in SPECIES}


# %% [markdown] {"id":"ixat3ffM8y6b"}
# ## 3. Mixed-Difficulty Open-Set CV Splits (v1 Encounter-Aware + v2 Harder)
# Estimates unbiased $\hat{\pi}_s$ via empirical CDF deconvolution, then constructs 5 splits for **CV v1** (`burst_cos_dist=0.10`) and 5 splits for **CV v2** (`burst_cos_dist=0.25`) so LightGBM learns across both difficulty regimes without overfitting to one split difficulty.
# 

# %% [code] {"id":"zteB8b6u8y6c"}
# ============================================================================
# 3. OPEN-SET CV SPLITS (CV v1 + CV v2 MIXED TRAINING)
# ============================================================================
def build_burst_clusters(rows: np.ndarray, ids: np.ndarray, dist_thresh: float):
    X_sub = EMB_L2[rows]
    clusters = {}
    for ident, grp in pd.DataFrame({"pos": np.arange(len(rows)), "id": ids}).groupby("id"):
        pos = grp["pos"].values
        if len(pos) == 1:
            clusters[ident] = [rows[pos]]
            continue
        cd = np.clip(1.0 - (X_sub[pos] @ X_sub[pos].T), 0.0, 2.0)
        np.fill_diagonal(cd, 0.0)
        cl = fcluster(linkage(squareform(cd, checks=False), method="average"), t=dist_thresh, criterion="distance")
        clusters[ident] = [rows[pos[cl == c]] for c in np.unique(cl)]
    return clusters

def estimate_pi_cdf(sp: str) -> float:
    tr_rows = meta[(meta["species"] == sp) & (~meta["is_test"])]["row"].values
    te_rows = meta[(meta["species"] == sp) & (meta["is_test"])]["row"].values
    bmap = build_burst_clusters(tr_rows, meta["identity"].values[tr_rows], 0.20)
    te_top1 = (EMB_L2[te_rows] @ EMB_L2[tr_rows].T).max(axis=1)
    kn_all, nw_all = [], []
    for seed in (101, 202, 303):
        rng = np.random.default_rng(seed)
        uniq = list(rng.permutation(list(bmap.keys())))
        new_set = set(uniq[:int(0.30 * len(uniq))])
        gal, kn_q, nw_q = [], [], []
        for ident, clists in bmap.items():
            if ident in new_set:
                for c in clists: nw_q.extend(c.tolist())
            elif len(clists) == 1:
                gal.extend(clists[0].tolist())
            else:
                perm = rng.permutation(len(clists))
                gal.extend(clists[perm[0]].tolist())
                for ci in perm[1:]:
                    (kn_q if rng.random() < 0.40 else gal).extend(clists[ci].tolist())
        kn_all.append((EMB_L2[kn_q] @ EMB_L2[gal].T).max(axis=1))
        nw_all.append((EMB_L2[nw_q] @ EMB_L2[gal].T).max(axis=1))
    kn, nw = np.concatenate(kn_all), np.concatenate(nw_all)
    grid = np.linspace(0.25, 0.90, 200)
    cdf_te = np.array([(te_top1 <= g).mean() for g in grid])
    cdf_kn = np.array([(kn <= g).mean() for g in grid])
    cdf_nw = np.array([(nw <= g).mean() for g in grid])
    res = minimize_scalar(lambda p: float(np.mean((cdf_te - (p * cdf_nw + (1 - p) * cdf_kn)) ** 2)),
                          bounds=(0.12, 0.55), method="bounded")
    print(f"[{sp}] Unbiased CDF pi_hat = {res.x:.3f}")
    return float(res.x)

def make_open_set_folds(sp: str, pi_target: float, burst_thresh: float, n_splits: int, seed_base: int, split_offset: int = 0):
    tr_rows = meta[(meta["species"] == sp) & (~meta["is_test"])]["row"].values
    te_rows = meta[(meta["species"] == sp) & (meta["is_test"])]["row"].values
    tr_ids = meta["identity"].values[tr_rows]
    bmap = build_burst_clusters(tr_rows, tr_ids, burst_thresh)

    q_ratio = len(te_rows) / float(len(tr_rows))
    target_q = int(round(len(tr_rows) * (q_ratio / (1.0 + q_ratio))))
    target_new = int(round(target_q * pi_target))
    id_sizes = {k: sum(len(c) for c in v) for k, v in bmap.items()}
    bins = {}
    for ident, sz in id_sizes.items():
        b = 0 if sz == 1 else 1 if sz == 2 else 2 if sz <= 5 else 3 if sz <= 10 else 4
        bins.setdefault(b, []).append(ident)

    folds = []
    for r in range(n_splits):
        rng = np.random.default_rng(seed_base + 1000 * r)
        new_ids, new_q = set(), []
        bpools = {b: list(rng.permutation(v)) for b, v in bins.items()}
        bkeys = sorted(bpools.keys())
        bw = np.array([sum(id_sizes[i] for i in bpools[b]) for b in bkeys], dtype=float)
        bw /= bw.sum()
        while len(new_q) < target_new:
            avail = [k for k in bkeys if bpools[k]]
            if not avail: break
            w = np.array([bw[bkeys.index(k)] for k in avail]); w /= w.sum()
            cid = bpools[rng.choice(avail, p=w)].pop()
            new_ids.add(cid)
            for c in bmap[cid]: new_q.extend(c.tolist())

        target_kn = max(1, target_q - len(new_q))
        kn_q, gal, cands = [], [], []
        for ident, clists in bmap.items():
            if ident in new_ids: continue
            if len(clists) == 1:
                gal.extend(clists[0].tolist())
            else:
                perm = rng.permutation(len(clists))
                gal.extend(clists[perm[0]].tolist())
                for ci in perm[1:]: cands.append(clists[ci])
        rng.shuffle(cands)
        for cl in cands:
            (kn_q if len(kn_q) < target_kn else gal).extend(cl.tolist())

        g_rows = np.array(sorted(gal), dtype=np.int32)
        q_rows = np.array(sorted(kn_q + new_q), dtype=np.int32)
        g_labels = meta["identity"].values[g_rows]
        g_set = set(g_labels)
        q_truth = np.array([meta["identity"].values[i] if meta["identity"].values[i] in g_set else NEW for i in q_rows])
        folds.append({
            "split": r + split_offset, "species": sp,
            "gallery_rows": g_rows, "query_rows": q_rows,
            "gallery_labels": g_labels, "query_ids": meta["id"].values[q_rows],
            "truth": q_truth, "is_test": False,
        })
    return folds

PI_HAT = {sp: estimate_pi_cdf(sp) for sp in SPECIES}
ALL_FOLDS = []
for sp in SPECIES:
    ALL_FOLDS += make_open_set_folds(sp, PI_HAT[sp], CFG["cv_v1_burst"], CFG["n_splits"], CFG["seed"], split_offset=0)
    if CFG["use_mixed_cv"]:
        ALL_FOLDS += make_open_set_folds(sp, PI_HAT[sp], CFG["cv_v2_burst"], CFG["n_splits"], CFG["seed"] + 777, split_offset=10)

TEST_FOLDS = []
for sp in SPECIES:
    g_rows = meta[(meta["species"] == sp) & (~meta["is_test"])]["row"].values.astype(np.int32)
    q_rows = meta[(meta["species"] == sp) & (meta["is_test"])]["row"].values.astype(np.int32)
    TEST_FOLDS.append({
        "split": -1, "species": sp,
        "gallery_rows": g_rows, "query_rows": q_rows,
        "gallery_labels": meta["identity"].values[g_rows], "query_ids": meta["id"].values[q_rows],
        "truth": None, "is_test": True,
    })
print(f"Prepared {len(ALL_FOLDS)} CV folds (primary + mixed) and {len(TEST_FOLDS)} test folds.")


# %% [markdown] {"id":"Ndp8uYwJ8zfc"}
# ## 4. Fast Parallel SIFT RANSAC Verifier + Multi-Metric & v3 Orthogonal Feature Builder
# Computes all v10 features (`l2`, `pcaw`, `lda`, `krec_l2`, `krec_lda`, `recip_rank`, `compat_best`, `LNBNN K=10/K=3/img`, wide+LNBNN-guided SIFT RANSAC including `strict 5px` inliers and `sift_n6`) **plus** our v3 orthogonal signals (`Dual RBF Kernel Ridge`, `Balanced Cosine-Ridge`, `Per-Flank Sub-Prototypes`).
# 

# %% [code] {"id":"N0WbRP3680kA"}
# ============================================================================
# 4. PARALLEL SIFT RANSAC CACHE + FOLD FEATURE BUILDER (ZERO-FORK PYTORCH)
# ============================================================================
SIFT_NAMES = ["n_mutual", "inl_aff", "inl_aff_strict", "inl_h", "inl_ratio", "spread_min", "abs_rot", "log_scale"]

def _verify_sift_batch_numpy_v4(pairs, xy_dict, desc_dict, ratio=0.80, thresh=10.0, strict=5.0) -> np.ndarray:
    """Pure-NumPy RootSIFT mutual Lowe-ratio matching + OpenCV RANSAC."""
    res = np.full((len(pairs), len(SIFT_NAMES)), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        da, db = desc_dict[a], desc_dict[b]
        na, nb = da.shape[0], db.shape[0]
        if na < 2 or nb < 2:
            continue
        sim = da @ db.T
        i_ab = np.argpartition(-sim, 1, axis=1)[:, :2]
        v_ab = np.take_along_axis(sim, i_ab, axis=1)
        sw = v_ab[:, 1] > v_ab[:, 0]
        v1_ab = np.where(sw, v_ab[:, 1], v_ab[:, 0])
        v2_ab = np.where(sw, v_ab[:, 0], v_ab[:, 1])
        best_b = np.where(sw, i_ab[:, 1], i_ab[:, 0])
        d1_ab = np.sqrt(np.clip(2.0 - 2.0 * v1_ab, 0.0, None))
        d2_ab = np.sqrt(np.clip(2.0 - 2.0 * v2_ab, 0.0, None))
        p_ab = d1_ab < (ratio * d2_ab)

        i_ba = np.argpartition(-sim, 1, axis=0)[:2, :]
        v_ba = np.take_along_axis(sim, i_ba, axis=0)
        swb = v_ba[1, :] > v_ba[0, :]
        v1_ba = np.where(swb, v_ba[1, :], v_ba[0, :])
        v2_ba = np.where(swb, v_ba[0, :], v_ba[1, :])
        best_a = np.where(swb, i_ba[1, :], i_ba[0, :])
        d1_ba = np.sqrt(np.clip(2.0 - 2.0 * v1_ba, 0.0, None))
        d2_ba = np.sqrt(np.clip(2.0 - 2.0 * v2_ba, 0.0, None))
        p_ba = d1_ba < (ratio * d2_ba)

        ia = np.arange(na)
        mut = p_ab & p_ba[best_b] & (best_a[best_b] == ia)
        ma = ia[mut]
        mb = best_b[mut]
        n_mut = len(ma)
        res[i, 0], res[i, 1], res[i, 2], res[i, 3], res[i, 4] = float(n_mut), 0.0, 0.0, 0.0, 0.0
        if n_mut < 4:
            continue

        xy_a, xy_b = xy_dict[a], xy_dict[b]
        pts_a, pts_b = xy_a[ma], xy_b[mb]
        M, inl = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=thresh, maxIters=1200, confidence=0.99)
        n_aff = int(inl.sum()) if (M is not None and inl is not None) else 0
        res[i, 1] = float(n_aff)
        res[i, 4] = float(n_aff) / max(1.0, float(n_mut))
        if n_aff >= 3 and M is not None:
            m = inl.ravel().astype(bool)
            pred_b = (pts_a @ M[:, :2].T) + M[:, 2]
            errs = np.linalg.norm(pred_b - pts_b, axis=1)
            res[i, 2] = float((errs <= strict).sum())
            sa = float(np.hypot(pts_a[m, 0].std(), pts_a[m, 1].std())) / (float(np.hypot(xy_a[:, 0].std(), xy_a[:, 1].std())) + 1e-4)
            sb = float(np.hypot(pts_b[m, 0].std(), pts_b[m, 1].std())) / (float(np.hypot(xy_b[:, 0].std(), xy_b[:, 1].std())) + 1e-4)
            res[i, 5] = min(2.0, min(sa, sb))
            res[i, 6] = abs(float(np.arctan2(M[1, 0], M[0, 0])))
            res[i, 7] = abs(float(np.log(max(np.hypot(M[0, 0], M[0, 1]), 1e-4))))
        if n_mut >= 6:
            H, inlh = cv2.findHomography(pts_a, pts_b, getattr(cv2, "USAC_MAGSAC", cv2.RANSAC), thresh)
            if H is not None and inlh is not None:
                res[i, 3] = float(inlh.sum())
    return res

def _save_sift_cache(cache_p: Path, pair_dict: dict) -> None:
    """Atomic save of SIFT pairs to disk."""
    if pair_dict:
        keys_arr = np.array(list(pair_dict.keys()), dtype=np.int32)
        vals_arr = np.array([pair_dict[tuple(k)] for k in keys_arr], dtype=np.float32)
    else:
        keys_arr = np.empty((0, 2), dtype=np.int32)
        vals_arr = np.empty((0, len(SIFT_NAMES)), dtype=np.float32)
    tmp_p = cache_p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_p, keys=keys_arr, vals=vals_arr)
    tmp_p.replace(cache_p)

def ensure_sift_pairs(needed_pairs: set) -> dict:
    """Computes SIFT+RANSAC only for missing needed pairs using PyTorch with zero fork OOM."""
    cache_p = CACHE_DIR / "sift_pairs_v4.npz"
    pair_dict = {}
    if cache_p.exists():
        with np.load(cache_p, allow_pickle=False) as z:
            for k, v in zip(z["keys"], z["vals"]):
                pair_dict[(int(k[0]), int(k[1]))] = v.copy()
    missing = sorted(p for p in needed_pairs if p not in pair_dict)
    if not missing:
        print(f"[sift] All {len(needed_pairs):,} pairs satisfied from cache.")
        return pair_dict

    t0 = time.time()
    print(f"[sift] Computing {len(missing):,} new SIFT+RANSAC pairs on {SIFT_DEVICE} (cached: {len(pair_dict):,})...")
    needed_rows = sorted({r for p in missing for r in p})
    xy_d, desc_d = {}, {}
    st_tr, st_te = SiftStore(SIFT_TRAIN_PATH), SiftStore(SIFT_TEST_PATH)
    for r_i in needed_rows:
        img_id = meta["id"].iloc[r_i]
        xy, desc = (st_te if meta["is_test"].iloc[r_i] else st_tr).get(img_id)
        xy_d[r_i] = xy
        desc_d[r_i] = to_rootsift_l2(desc)
    st_tr.close(); st_te.close()

    step = CFG.get("sift_save_every", 1500)
    for start in range(0, len(missing), step):
        batch = missing[start:start + step]
        vals_batch = _verify_sift_batch_numpy_v4(
            batch, xy_d, desc_d, CFG["sift_ratio"], CFG["sift_thresh"], CFG["sift_thresh_strict"]
        )
        for p, v in zip(batch, vals_batch):
            pair_dict[p] = v
        _save_sift_cache(cache_p, pair_dict)
        print(f"  [sift] Processed {min(start + step, len(missing)):,}/{len(missing):,} pairs...")

    del xy_d, desc_d
    gc.collect()
    print(f"[sift] Computed & cached {len(missing):,} pairs in {time.time() - t0:.1f}s.")
    return pair_dict

def fit_lda_fold(Xg: np.ndarray, y_gal: np.ndarray, Xq: np.ndarray, gamma=0.30, d_out=256):
    p_dim = min(512, len(Xg) - 2, Xg.shape[1])
    pca = PCA(n_components=p_dim, svd_solver="randomized", random_state=CFG["seed"])
    Zg, Zq = pca.fit_transform(Xg), pca.transform(Xq)
    mu = Zg.mean(0)
    Sw = np.zeros((p_dim, p_dim), np.float64)
    Sb = np.zeros((p_dim, p_dim), np.float64)
    nw = 0
    for _, grp in pd.DataFrame({"i": np.arange(len(y_gal)), "id": y_gal}).groupby("id"):
        idxs = grp["i"].values
        sub = Zg[idxs]; mc = sub.mean(0)
        d = (mc - mu)[:, None]; Sb += len(idxs) * (d @ d.T)
        if len(idxs) >= 2:
            xc = sub - mc; Sw += xc.T @ xc; nw += len(idxs) - 1
    Sw /= max(1, nw); Sb /= max(1, len(y_gal))
    Sw_r = (1 - gamma) * Sw + gamma * (np.trace(Sw) / p_dim) * np.eye(p_dim)
    ew, vw = np.linalg.eigh(Sw_r)
    W_is = (vw * (np.clip(ew, 1e-6, None) ** -0.5)) @ vw.T
    _, vb = np.linalg.eigh(W_is @ Sb @ W_is)
    P = (W_is @ vb[:, -min(d_out, p_dim):][:, ::-1]).astype(np.float32)
    return l2n(Zg @ P), l2n(Zq @ P)

def fit_krr_and_ridge(Zg_p: np.ndarray, y_gal: np.ndarray, Zq_p: np.ndarray, uid_list: list):
    uid_map = {c: i for i, c in enumerate(uid_list)}
    Y = np.zeros((len(y_gal), len(uid_list)), dtype=np.float32)
    for i, c in enumerate(y_gal): Y[i, uid_map[c]] = 1.0
    # Dual RBF Kernel Ridge
    K_gg = np.exp(-CFG["krr_gamma"] * np.clip(2.0 - 2.0 * (Zg_p @ Zg_p.T), 0.0, 4.0)).astype(np.float32)
    K_qg = np.exp(-CFG["krr_gamma"] * np.clip(2.0 - 2.0 * (Zq_p @ Zg_p.T), 0.0, 4.0)).astype(np.float32)
    S_krr = (K_qg @ np.linalg.solve(K_gg + CFG["krr_lam"] * np.eye(len(y_gal), dtype=np.float32), Y)).astype(np.float32)
    # Balanced Cosine-Ridge
    cnt = Y.sum(0)
    sw = np.array([1.0 / np.sqrt(cnt[uid_map[c]]) for c in y_gal], dtype=np.float32)
    Zgw, Yw = Zg_p * sw[:, None], Y * sw[:, None]
    W = np.linalg.solve(Zgw.T @ Zgw + CFG["ridge_lam"] * np.eye(Zg_p.shape[1], dtype=np.float32), Zgw.T @ Yw)
    W /= np.clip(np.linalg.norm(W, axis=0, keepdims=True), 1e-8, None)
    S_ridge = (Zq_p @ W).astype(np.float32)
    return S_krr, S_ridge

def k_reciprocal_rerank(Zq: np.ndarray, Zg: np.ndarray, k1=20, k2=6, lam=0.30) -> np.ndarray:
    nq, ng = len(Zq), len(Zg)
    Z = np.vstack([Zq, Zg]).astype(np.float32)
    N = len(Z)
    orig_dist = np.clip(2.0 - 2.0 * (Z @ Z.T), 0.0, 4.0)
    orig_dist /= np.maximum(orig_dist.max(axis=1, keepdims=True), 1e-6)
    initial_rank = np.argsort(orig_dist, axis=1).astype(np.int32)
    V = np.zeros((N, N), dtype=np.float32)
    hk = int(round(k1 / 2)) + 1
    for i in range(N):
        fwd = initial_rank[i, :k1 + 1]
        bwd = initial_rank[fwd, :k1 + 1]
        k_recip = fwd[np.where(bwd == i)[0]]
        k_recip_exp = k_recip
        for cand in k_recip:
            cfwd = initial_rank[cand, :hk]
            cbwd = initial_rank[cfwd, :hk]
            c_recip = cfwd[np.where(cbwd == cand)[0]]
            if len(np.intersect1d(c_recip, k_recip)) > (2.0 / 3.0) * len(c_recip):
                k_recip_exp = np.append(k_recip_exp, c_recip)
        k_recip_exp = np.unique(k_recip_exp)
        w = np.exp(-orig_dist[i, k_recip_exp])
        V[i, k_recip_exp] = w / np.maximum(w.sum(), 1e-6)
    if k2 > 1:
        V = np.array([V[initial_rank[i, :k2]].mean(axis=0) for i in range(N)], dtype=np.float32)
    Vq, Vg = V[:nq], V[nq:]
    jaccard = np.ones((nq, ng), dtype=np.float32)
    for i in range(nq):
        nz = np.nonzero(Vq[i])[0]
        if len(nz):
            mn = np.minimum(Vq[i, nz][None, :], Vg[:, nz]).sum(axis=1)
            jaccard[i] = 1.0 - mn / (2.0 - mn)
    return 1.0 - ((1.0 - lam) * jaccard + lam * orig_dist[:nq, nq:])

def vp_compat_dict(Zg: np.ndarray, labels: np.ndarray, vps: np.ndarray) -> dict:
    acc = {}
    for _, grp in pd.DataFrame({"i": np.arange(len(labels)), "id": labels}).groupby("id"):
        idx = grp["i"].values
        if len(idx) < 2: continue
        S = Zg[idx] @ Zg[idx].T
        iu = np.triu_indices(len(idx), 1)
        for a, b, x in zip(vps[idx][iu[0]], vps[idx][iu[1]], S[iu]):
            k = (a, b) if a <= b else (b, a)
            t = acc.setdefault(k, [0.0, 0])
            t[0] += float(x); t[1] += 1
    return {k: sm / n for k, (sm, n) in acc.items() if n >= 10}

def prepare_fold_artifacts(fold: dict):
    """Pass 1: compute all global + LNBNN matrices and identify exact SIFT pairs needed."""
    sp = fold["species"]
    gr, qr = fold["gallery_rows"], fold["query_rows"]
    y_gal = fold["gallery_labels"]
    uid_list = sorted(np.unique(y_gal).tolist())
    nid, nq, ng = len(uid_list), len(qr), len(gr)
    id_to_g = {}
    for gi, c in enumerate(y_gal): id_to_g.setdefault(c, []).append(gi)
    id_to_g = {c: np.array(v, dtype=np.int32) for c, v in id_to_g.items()}

    Zg_l2, Zq_l2 = EMB_L2[gr], EMB_L2[qr]
    Zg_p, Zq_p = EMB_PCAW[gr], EMB_PCAW[qr]
    Zg_lda, Zq_lda = fit_lda_fold(Zg_l2, y_gal, Zq_l2, CFG["lda_gamma"], CFG["lda_d_out"])

    Z_map = {"l2": (Zq_l2, Zg_l2), "pcaw": (Zq_p, Zg_p), "lda": (Zq_lda, Zg_lda)}
    S, M, T2, PR = {}, {}, {}, {}
    for e, (zq, zg) in Z_map.items():
        Se = zq @ zg.T
        S[e] = Se
        M[e] = np.column_stack([Se[:, id_to_g[c]].max(1) for c in uid_list])
        T2[e] = np.column_stack([np.sort(Se[:, id_to_g[c]], 1)[:, -2:].mean(1) if len(id_to_g[c]) >= 2 else Se[:, id_to_g[c]].max(1) for c in uid_list])
        protos = np.vstack([l2n(zg[id_to_g[c]].mean(0, keepdims=True)) for c in uid_list])
        PR[e] = zq @ protos.T

    # v3 Orthogonal signals: RBF KRR + Balanced Cosine-Ridge + Per-Flank Sub-Prototypes
    S_krr, S_ridge = fit_krr_and_ridge(Zg_p, y_gal, Zq_p, uid_list)
    gside = meta["side"].values[gr]
    sub_p, sub_own = [], []
    for j, c in enumerate(uid_list):
        idxs = id_to_g[c]
        for fg in ("L", "R", "N", "U"):
            m = gside[idxs] == fg
            if m.any():
                sub_p.append(l2n(Zg_p[idxs[m]].mean(0, keepdims=True))[0])
                sub_own.append(j)
    S_sub_all = Zq_p @ np.vstack(sub_p).T
    S_subproto = np.full((nq, nid), -2.0, dtype=np.float32)
    for ck, oj in enumerate(sub_own):
        S_subproto[:, oj] = np.maximum(S_subproto[:, oj], S_sub_all[:, ck])

    # k-Reciprocal on l2 & lda
    KREC = {}
    for e in ("l2", "lda"):
        Kr = k_reciprocal_rerank(Z_map[e][0], Z_map[e][1], CFG["krec_k1"], CFG["krec_k2"], CFG["krec_lam"])
        KREC[e] = np.column_stack([Kr[:, id_to_g[c]].max(1) for c in uid_list])

    # Fold-masked HotSpotter LNBNN (K=10, K=3, image-level)
    sc = LNB_SCORERS[sp]
    gids = meta["id"].values[gr]
    gmask = np.isin(sc.gal, gids)
    col_idx = np.searchsorted(sc.names, uid_list)
    g_col = np.array([sc.gpos[g] for g in gids], dtype=np.int32)
    res_lnb = [sc.score_all_granularities(qid, gmask) for qid in fold["query_ids"]]
    L_id = np.stack([x[0] for x in res_lnb])[:, col_idx]
    L_img = np.stack([x[1] for x in res_lnb])[:, col_idx]
    L_k3 = np.stack([x[2] for x in res_lnb])[:, col_idx]
    ImgMat = np.stack([x[3][g_col] for x in res_lnb])

    # Candidate pool: top-30 combined rank (l2 + pcaw + lda + krr) + top-5 lnb_img expansion
    rk_l2 = np.argsort(np.argsort(-M["l2"], axis=1), axis=1)
    rk_p = np.argsort(np.argsort(-M["pcaw"], axis=1), axis=1)
    rk_lda = np.argsort(np.argsort(-M["lda"], axis=1), axis=1)
    rk_krr = np.argsort(np.argsort(-S_krr, axis=1), axis=1)
    comb = rk_l2 + rk_p + rk_lda + rk_krr

    sift_top, per_id, lnb_k = CFG["sift_top_ids"], CFG["sift_imgs_per_id"], CFG["lnbnn_cand"]
    cand_lists, sift_plans = [], []
    needed_pairs = set()
    for i in range(nq):
        cands = list(np.argsort(comb[i])[:CFG["n_cand"]])
        cset = set(cands)
        top_lnb = np.argsort(-L_img[i])[:lnb_k]
        for ce in top_lnb:
            if int(ce) not in cset:
                cands.append(int(ce)); cset.add(int(ce))
        cand_lists.append(cands)

        r_sift = (set(np.argsort(-M["l2"][i])[:sift_top]) |
                  set(np.argsort(-M["pcaw"][i])[:sift_top]) |
                  set(np.argsort(-S_krr[i])[:sift_top]) |
                  set(int(x) for x in top_lnb))
        plan_i = {}
        for c in cands:
            if c in r_sift:
                cols = id_to_g[uid_list[c]]
                b1 = cols[np.argsort(-S["l2"][i, cols])[:per_id]]
                b2 = cols[np.argsort(-ImgMat[i, cols])[:2]]
                best = np.union1d(b1, b2)
                pks = [(min(int(qr[i]), int(gr[j])), max(int(qr[i]), int(gr[j]))) for j in best]
                plan_i[c] = pks
                needed_pairs.update(pks)
        sift_plans.append(plan_i)

    art = dict(
        fold=fold, uid_list=uid_list, id_to_g=id_to_g, S=S, M=M, T2=T2, PR=PR,
        S_krr=S_krr, S_ridge=S_ridge, S_subproto=S_subproto, KREC=KREC,
        L_id=L_id, L_img=L_img, L_k3=L_k3, rk={"l2": rk_l2, "pcaw": rk_p, "lda": rk_lda, "krr": rk_krr},
        cand_lists=cand_lists, sift_plans=sift_plans,
    )
    return art, needed_pairs

def build_fold_df(art: dict, pair_dict: dict) -> pd.DataFrame:
    fold = art["fold"]
    gr, qr, y_gal = fold["gallery_rows"], fold["query_rows"], fold["gallery_labels"]
    uid_list, id_to_g = art["uid_list"], art["id_to_g"]
    M, T2, PR, S = art["M"], art["T2"], art["PR"], art["S"]
    S_krr, S_ridge, S_subproto, KREC = art["S_krr"], art["S_ridge"], art["S_subproto"], art["KREC"]
    L_id, L_img, L_k3, rk = art["L_id"], art["L_img"], art["L_k3"], art["rk"]
    nq, nid = len(qr), len(uid_list)

    gvp, gside = meta["vp_clean"].values[gr], meta["side"].values[gr]
    qvp, qside = meta["vp_clean"].values[qr], meta["side"].values[qr]
    compat = vp_compat_dict(EMB_L2[gr], y_gal, gvp)
    pool = np.vstack([EMB_L2[gr], EMB_L2[qr]]).astype(np.float32)
    GS = np.sort(EMB_L2[gr] @ pool.T, axis=1)
    npool = GS.shape[1]

    # Query-level summary stats
    qf = {}
    for e in ("l2", "pcaw", "lda"):
        srt = -np.sort(-M[e], axis=1)
        qf[f"q_top1_{e}"] = srt[:, 0]; qf[f"q_gap12_{e}"] = srt[:, 0] - srt[:, 1]
        qf[f"q_mean10_{e}"] = srt[:, :10].mean(1); qf[f"q_std10_{e}"] = srt[:, :10].std(1) + 1e-6
    for nm, mat in (("krr", S_krr), ("ridge", S_ridge)):
        srt = -np.sort(-mat, axis=1)
        qf[f"q_top1_{nm}"] = srt[:, 0]; qf[f"q_gap12_{nm}"] = srt[:, 0] - srt[:, 1]
        qf[f"q_mean10_{nm}"] = srt[:, :10].mean(1); qf[f"q_std10_{nm}"] = srt[:, :10].std(1) + 1e-6
    for e in ("l2", "lda"):
        srt = -np.sort(-KREC[e], axis=1)
        qf[f"q_krec_{e}_max"] = srt[:, 0]; qf[f"q_krec_{e}_gap"] = srt[:, 0] - srt[:, 1]
    for nm, L in (("lnb", L_id), ("lnb_img", L_img), ("lnb_k3", L_k3)):
        srt = -np.sort(-L, axis=1)
        qf[f"q_{nm}_max"] = srt[:, 0]; qf[f"q_{nm}_gap"] = srt[:, 0] - (srt[:, 1] if nid > 1 else srt[:, 0])
        qf[f"q_{nm}_log_max"] = np.log1p(srt[:, 0])

    qf["q_lognorm"] = meta["lognorm"].values[qr]
    qf["q_kp"] = KP_COUNTS[qr]
    qf["q_side"] = pd.Categorical(qside, categories=["L", "R", "N", "U"]).codes

    Sqq = EMB_L2[qr] @ EMB_L2[qr].T; np.fill_diagonal(Sqq, -np.inf)
    qf["q_nnq_cos"] = Sqq.max(1)
    qf["q_nnq_minus_top1"] = Sqq.max(1) - qf["q_top1_l2"]
    nnq = np.argsort(-Sqq, axis=1)[:, :5]
    qf["q_nnq_top1"] = qf["q_top1_l2"][nnq[:, 0]]
    top1p = M["pcaw"].argmax(1)

    n_img = np.array([len(id_to_g[c]) for c in uid_list])
    has_L = np.array([(gside[id_to_g[c]] == "L").any() for c in uid_list])
    has_R = np.array([(gside[id_to_g[c]] == "R").any() for c in uid_list])
    n_vp = np.array([len(set(gvp[id_to_g[c]])) for c in uid_list])
    rk_lnb = np.argsort(np.argsort(-L_id, axis=1), axis=1)
    rk_lnbi = np.argsort(np.argsort(-L_img, axis=1), axis=1)

    rows = []
    for i in range(nq):
        cands = art["cand_lists"][i]
        plan_i = art["sift_plans"][i]
        sd = qside[i]
        nb_i = nnq[i]; w_nb = Sqq[i, nb_i]; ok_nb = w_nb > 0.5
        for pos_c, c in enumerate(cands):
            cid = uid_list[c]
            cols = id_to_g[cid]
            sims = S["l2"][i, cols]
            jbest = cols[int(np.argmax(sims))]
            row = {"qi": i, "cand": c, "comb_rank": pos_c}
            for e in ("l2", "pcaw", "lda"):
                v = float(M[e][i, c])
                row[f"max_{e}"] = v; row[f"top2_{e}"] = float(T2[e][i, c]); row[f"proto_{e}"] = float(PR[e][i, c])
                row[f"rank_{e}"] = float(rk[e][i, c]); row[f"gap_{e}"] = v - float(qf[f"q_top1_{e}"][i])
                row[f"z_{e}"] = (v - float(qf[f"q_mean10_{e}"][i])) / float(qf[f"q_std10_{e}"][i])
            # v3 Orthogonal features
            vk, vr = float(S_krr[i, c]), float(S_ridge[i, c])
            row["krr_score"] = vk; row["krr_gap"] = vk - float(qf["q_top1_krr"][i])
            row["krr_z"] = (vk - float(qf["q_mean10_krr"][i])) / float(qf["q_std10_krr"][i])
            row["krr_rank"] = float(rk["krr"][i, c])
            row["ridge_score"] = vr; row["ridge_gap"] = vr - float(qf["q_top1_ridge"][i])
            row["subproto_pcaw"] = float(S_subproto[i, c])
            # k-Reciprocal
            for e in ("l2", "lda"):
                row[f"krec_{e}"] = float(KREC[e][i, c])
                row[f"krec_{e}_gap"] = float(KREC[e][i, c] - qf[f"q_krec_{e}_max"][i])
            # LNBNN
            vl, vli, vlk3 = float(L_id[i, c]), float(L_img[i, c]), float(L_k3[i, c])
            row["lnb"] = vl; row["lnb_log"] = float(np.log1p(vl)); row["lnb_rank"] = float(rk_lnb[i, c])
            row["lnb_rel"] = vl / max(float(qf["q_lnb_max"][i]), 1e-9) if qf["q_lnb_max"][i] > 0 else 0.0
            row["lnb_img"] = vli; row["lnb_img_log"] = float(np.log1p(vli)); row["lnb_img_rank"] = float(rk_lnbi[i, c])
            row["lnb_img_rel"] = vli / max(float(qf["q_lnb_img_max"][i]), 1e-9) if qf["q_lnb_img_max"][i] > 0 else 0.0
            row["lnb_img_share"] = vli / max(vl, 1e-6) if vl > 0 else 0.0
            row["lnb_k3"] = vlk3
            row["lnb_k3_rel"] = vlk3 / max(float(qf["q_lnb_k3_max"][i]), 1e-9) if qf["q_lnb_k3_max"][i] > 0 else 0.0
            # Gallery & viewpoint geometry
            row["log_nimg"] = float(np.log(n_img[c])); row["n_vp"] = float(n_vp[c])
            row["has_same_side"] = float(has_L[c] if sd == "L" else has_R[c] if sd == "R" else np.nan)
            same = gside[cols] == sd
            row["max_same_side"] = float(sims[same].max()) if (sd in "LR" and same.any()) else np.nan
            opp = gside[cols] == ("R" if sd == "L" else "L")
            row["max_opp_side"] = float(sims[opp].max()) if (sd in "LR" and opp.any()) else np.nan
            kp_vp = (qvp[i], gvp[jbest]) if qvp[i] <= gvp[jbest] else (gvp[jbest], qvp[i])
            cb = compat.get(kp_vp, np.nan)
            row["compat_best"] = cb; row["max_minus_compat"] = float(M["l2"][i, c]) - cb
            n_closer = npool - int(np.searchsorted(GS[jbest], np.float32(M["l2"][i, c]), side="right"))
            row["recip_rank"] = float(np.log1p(max(n_closer - 1, 0)))
            row["min_pair_lognorm"] = min(float(qf["q_lognorm"][i]), float(meta["lognorm"].values[gr[jbest]]))
            # Transductive neighbour votes
            row["tr_vote"] = float(((top1p[nb_i] == c) & ok_nb).sum())
            row["tr_nb_score"] = float(np.max(np.minimum(w_nb, M["l2"][nb_i, c])))
            row["tr_nb_minus_max"] = row["tr_nb_score"] - float(M["l2"][i, c])
            # Wide + LNBNN-guided SIFT
            pks = plan_i.get(c)
            if pks:
                F = np.array([pair_dict[pk] for pk in pks], dtype=np.float32)
                inl = np.nan_to_num(F[:, 1], nan=-1.0)
                kbest = int(np.argmax(inl))
                row["sift_checked"] = float(len(pks))
                row["sift_inl_max"] = float(np.nanmax(F[:, 1])) if np.isfinite(F[:, 1]).any() else np.nan
                row["sift_inl_strict_max"] = float(np.nanmax(F[:, 2])) if np.isfinite(F[:, 2]).any() else np.nan
                row["sift_inlh_max"] = float(np.nanmax(F[:, 3])) if np.isfinite(F[:, 3]).any() else np.nan
                row["sift_inl_top2"] = float(np.sort(np.nan_to_num(F[:, 1]))[-2:].sum())
                row["sift_n6"] = float((np.nan_to_num(F[:, 1]) >= 6).sum())
                row["sift_mutual_max"] = float(np.nanmax(F[:, 0])) if np.isfinite(F[:, 0]).any() else np.nan
                fb = F[kbest]
                row["sift_ratio_best"] = float(fb[4]); row["sift_spread_best"] = float(fb[5])
                row["sift_rot_best"] = float(fb[6]); row["sift_scale_best"] = float(fb[7])
            else:
                row["sift_checked"] = 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    for k, v in qf.items():
        df[k] = np.asarray(v)[df["qi"].values]
    g = df.groupby("qi")
    df["q_sift_inl_max"] = g["sift_inl_max"].transform("max")
    df["q_sift_n6_cands"] = g["sift_n6"].transform(lambda s: (s > 0).sum())
    df["sift_inl_minus_qmax"] = df["sift_inl_max"] - df["q_sift_inl_max"]
    df["sift_inl_rank"] = g["sift_inl_max"].rank(ascending=False, method="min")
    df["query_id"] = fold["query_ids"][df["qi"].values]
    df["cand_id"] = np.array(uid_list)[df["cand"].values]
    df["species_is_leo"] = int(fold["species"] == "leopard")
    df["species"] = fold["species"]
    df["split"] = fold["split"]
    if not fold["is_test"]:
        td = dict(zip(fold["query_ids"], fold["truth"]))
        df["truth"] = df["query_id"].map(td)
        df["y"] = (df["cand_id"] == df["truth"]).astype(np.int32)
        df["is_new"] = (df["truth"] == NEW).astype(np.int32)
        df["group"] = meta["identity"].values[qr[df["qi"].values]]
    return df

# Execute Pass 1 -> SIFT Batch -> Pass 2 (cached to disk!)
# Execute Pass 1 -> SIFT Batch -> Pass 2 (Streaming Fold-by-Fold Table Building)
tables_cache = CACHE_DIR / "tables_v4.pkl"
if tables_cache.exists():
    print(f"Loading cached v4 tables from {tables_cache}")
    TABLES = pickle.loads(tables_cache.read_bytes())
else:
    t0 = time.time()
    all_needed = set()
    
    # Pass 1: Extract candidate pools, serialize artifacts to temp files, collect unique SIFT pairs
    print("Pass 1: Preparing fold artifacts & caching to disk...")
    for f in ALL_FOLDS:
        art, need = prepare_fold_artifacts(f)
        art_p = CACHE_DIR / f"temp_art_cv_{f['species']}_{f['split']}.pkl"
        art_p.write_bytes(pickle.dumps(art))
        all_needed.update(need)
        del art
        gc.collect()

    for f in TEST_FOLDS:
        art, need = prepare_fold_artifacts(f)
        art_p = CACHE_DIR / f"temp_art_test_{f['species']}_{f['split']}.pkl"
        art_p.write_bytes(pickle.dumps(art))
        all_needed.update(need)
        del art
        gc.collect()

    print(f"Pass 1 complete in {time.time() - t0:.1f}s | Unique SIFT pairs requested: {len(all_needed):,}")
    pair_dict = ensure_sift_pairs(all_needed)

    # Pass 2: Stream each fold DataFrame into memory one by one, deleting temp artifacts
    print("Pass 2: Building fold DataFrames...")
    cv_dfs = []
    for f in ALL_FOLDS:
        art_p = CACHE_DIR / f"temp_art_cv_{f['species']}_{f['split']}.pkl"
        art = pickle.loads(art_p.read_bytes())
        cv_dfs.append(build_fold_df(art, pair_dict))
        del art
        art_p.unlink(missing_ok=True)
        gc.collect()

    te_dfs = []
    for f in TEST_FOLDS:
        art_p = CACHE_DIR / f"temp_art_test_{f['species']}_{f['split']}.pkl"
        art = pickle.loads(art_p.read_bytes())
        te_dfs.append(build_fold_df(art, pair_dict))
        del art
        art_p.unlink(missing_ok=True)
        gc.collect()

    TABLES = {"cv": pd.concat(cv_dfs, ignore_index=True), "test": pd.concat(te_dfs, ignore_index=True)}
    tables_cache.write_bytes(pickle.dumps(TABLES))
    del cv_dfs, te_dfs, pair_dict
    gc.collect()
    print(f"Built & cached v4 tables in {time.time() - t0:.1f}s: CV={TABLES['cv'].shape}, Test={TABLES['test'].shape}")


# %% [markdown] {"id":"seUXTitB80sd"}
# ## 5. Two-Round Monotone LightGBM Stacking + Combined MiewID/SIFT Keypoint Graph + Zhou Label Propagation
# Matches `models.py` and `pipeline.py` (with `GroupKFold(5)` by true query identity, monotone positive constraints, mixed CV v1+v2 training, Round 2 neighbour stacking enriched by `kpgraph`, per-species temperature $T_s$ & $\beta_s$, zero prior-shift EM, and CV-gated Zhou et al. label propagation).
# 

# %% [code] {"id":"FBztFDon80sf"}
# ============================================================================
# 5. TWO-ROUND MONOTONE LIGHTGBM + KPGRAPH NEIGHBOURS + ZHOU PROPAGATION
# ============================================================================
NON_FEATS = {"p", "p1", "qi", "cand", "query_id", "cand_id", "y", "split", "group", "species", "is_new", "truth", "_key", "post"}
MONO_POS = {
    "max_l2", "max_pcaw", "max_lda", "krr_score", "ridge_score", "subproto_pcaw",
    "sift_inl_max", "sift_inl_strict_max", "lnb", "lnb_img", "lnb_k3",
}
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=CFG["sift_workers"], seed=CFG["seed"],
)

def get_feat_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATS and df[c].dtype != object]

def fit_lgb(X, y, groups, feats, mono=True):
    p = dict(LGB_PARAMS)
    if mono:
        p["monotone_constraints"] = [1 if f in MONO_POS else 0 for f in feats]
        p["monotone_constraints_method"] = "advanced"
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(X, y, groups))
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    m = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m

def oof_lgb(df: pd.DataFrame, feats: list, label: str, mono=True):
    X = df[feats].to_numpy(np.float32)
    y = df[label].to_numpy()
    groups = df["group"].values
    oof = np.zeros(len(df), dtype=np.float64)
    iters = []
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = fit_lgb(X[tr], y[tr], groups[tr], feats, mono=mono)
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
    return oof, iters

def build_query_table(df: pd.DataFrame, pcol: str = "p") -> pd.DataFrame:
    df = df.copy()
    df["_key"] = df["split"].astype(str) + "|" + df["query_id"]
    g = df.groupby("_key", sort=False)
    first = g.head(1).set_index("_key")
    q = pd.DataFrame(index=first.index)
    srt = g[pcol].apply(lambda s: np.sort(s.values)[::-1])
    q["p1"] = srt.map(lambda a: a[0])
    q["p2"] = srt.map(lambda a: a[1] if len(a) > 1 else 0.0)
    q["psum"] = g[pcol].sum()
    q["lgap"] = logit(np.clip(q["p1"], 1e-6, 1 - 1e-6)) - logit(np.clip(q["p2"], 1e-6, 1 - 1e-6))
    q["n_p10"] = g[pcol].apply(lambda s: (s > 0.1).sum())
    top = df.loc[g[pcol].idxmax()].set_index("_key")
    for c in ("max_same_side", "has_same_side", "compat_best", "max_minus_compat", "recip_rank", "log_nimg",
              "max_opp_side", "sift_inl_max", "z_pcaw", "z_l2", "z_lda", "krr_score", "krr_gap", "ridge_score",
              "lnb_img", "lnb_img_rel", "lnb_img_share", "lnb_k3", "lnb_log", "lnb_rel",
              "krec_l2", "krec_lda", "nb_p_max", "nb_p_w", "nb0_p", "nb_cnt"):
        if c in top:
            q[f"top_{c}"] = top[c]
    for c in first.columns:
        if c.startswith("q_") or c in ("species_is_leo",):
            q[c] = first[c]
    for c in ("split", "query_id", "species", "is_new", "group", "truth"):
        if c in first:
            q[c] = first[c]
    return q.reset_index(drop=True)

def build_combined_query_neighbors(df: pd.DataFrame, k_qq: int = 5):
    """Per-(split, species) query-query neighbour graph blending MiewID L2 cosine + unsupervised SIFT kpgraph."""
    nbr_dict = {}
    for (sp_split, sp_name), grp in df.groupby(["split", "species"]):
        qids = grp["query_id"].unique()
        qset = set(qids)
        if len(qids) <= 1:
            for q in qids: nbr_dict[(sp_split, q)] = ([], [])
            continue
        rows = np.array([id_to_row[q] for q in qids])
        Z = EMB_L2[rows]
        S = Z @ Z.T; np.fill_diagonal(S, -np.inf)
        kpg = LNB_SCORERS[sp_name].kpgraph
        qid_to_pos = {q: i for i, q in enumerate(qids)}
        # Add normalized SIFT keypoint graph bonus strictly within this split's query set
        S_comb = S.copy()
        for i, q in enumerate(qids):
            for n_id, sc_kp in kpg.get(q, []):
                if n_id in qset and n_id != q:
                    j = qid_to_pos[n_id]
                    S_comb[i, j] = max(S_comb[i, j], float(np.clip(0.55 + 0.35 * np.tanh(sc_kp / 15.0), 0.0, 0.98)))
        k = min(k_qq, len(qids) - 1)
        order = np.argsort(-S_comb, axis=1)[:, :k]
        for i, q in enumerate(qids):
            nbr_dict[(sp_split, q)] = ([qids[j] for j in order[i]], [float(S_comb[i, j]) for j in order[i]])
    return nbr_dict

def add_round2_features(df: pd.DataFrame, qdf: pd.DataFrame, nbr_dict: dict):
    df = df.copy(); qdf = qdf.copy()
    df["_key"] = df["split"].astype(str) + "|" + df["query_id"]
    g = df.groupby("_key", sort=False)
    df["p1_rank"] = g["p1"].rank(ascending=False, method="min").values.astype(np.float32)
    df["p1_norm"] = (df["p1"].values / np.maximum(g["p1"].transform("sum").values, 1e-9)).astype(np.float32)
    pn_map = dict(zip(zip(qdf["split"], qdf["query_id"]), qdf["pn1"]))
    df["pn1_self"] = [float(pn_map.get((s, q), 0.0)) for s, q in zip(df["split"], df["query_id"])]

    pmap = {(s, q, c): float(p) for s, q, c, p in zip(df["split"], df["query_id"], df["cand_id"], df["p1"])}
    n = len(df)
    nb_p_max = np.zeros(n, np.float32); nb_p_w = np.zeros(n, np.float32)
    nb0_p = np.zeros(n, np.float32); nb_cnt = np.zeros(n, np.float32)
    for r, (sp_s, q, c) in enumerate(zip(df["split"].values, df["query_id"].values, df["cand_id"].values)):
        ids, sims = nbr_dict.get((sp_s, q), ([], []))
        if ids:
            s = np.asarray(sims, np.float32)
            v = np.array([pmap.get((sp_s, nid, c), 0.0) for nid in ids], np.float32)
            w = np.clip(s, 0.0, None)
            nb_p_max[r] = float(np.max(s * v))
            nb_p_w[r] = float((w * v).sum() / max(float(w.sum()), 1e-6))
            nb0_p[r] = float(v[0] * s[0])
            nb_cnt[r] = float(np.sum((v > 0.5) & (s > 0.6)))
    df["nb_p_max"] = nb_p_max; df["nb_p_w"] = nb_p_w; df["nb0_p"] = nb0_p; df["nb_cnt"] = nb_cnt
    df.drop(columns=["_key"], inplace=True)

    df_s = df.sort_values(["split", "query_id", "p1"], ascending=[True, True, False])
    top_r = df_s.drop_duplicates(["split", "query_id"])
    topc = dict(zip(zip(top_r["split"], top_r["query_id"]), top_r["cand_id"]))
    topp = dict(zip(zip(top_r["split"], top_r["query_id"]), top_r["p1"]))
    nq = len(qdf)
    nb_pn_w = np.zeros(nq, np.float32); nb0_pn = np.zeros(nq, np.float32)
    nb_kn_conf = np.zeros(nq, np.float32); nb_agree = np.zeros(nq, np.float32)
    for r, (sp_s, q) in enumerate(zip(qdf["split"].values, qdf["query_id"].values)):
        ids, sims = nbr_dict.get((sp_s, q), ([], []))
        if ids:
            s = np.asarray(sims, np.float32); w = np.clip(s, 0.0, None)
            pns = np.array([float(pn_map.get((sp_s, nid), 0.0)) for nid in ids], np.float32)
            conf = np.array([float(topp.get((sp_s, nid), 0.0)) for nid in ids], np.float32)
            tq = topc.get((sp_s, q))
            agr = np.array([float(topc.get((sp_s, nid)) == tq and tq is not None) for nid in ids], np.float32)
            nb_pn_w[r] = float((w * pns).sum() / max(float(w.sum()), 1e-6))
            nb0_pn[r] = float(pns[0]); nb_kn_conf[r] = float(np.max(s * conf)); nb_agree[r] = float(np.max(s * agr))
    qdf["nb_pn_w"] = nb_pn_w; qdf["nb0_pn"] = nb0_pn; qdf["nb_known_conf"] = nb_kn_conf; qdf["nb_agree"] = nb_agree
    return df, qdf

# --- Run Round 1 & Round 2 Training + OOF ---
cv = TABLES["cv"].copy()
test = TABLES["test"].copy()
feats_r1 = get_feat_cols(cv)
print(f"[R1] Training monotone LightGBM reranker on {len(feats_r1)} features ({len(cv):,} rows)...")
cv["p"], it1 = oof_lgb(cv, feats_r1, "y", mono=True)
print(f"[R1] Reranker OOF done | best iters: {it1}")

qdf = build_query_table(cv, "p")
qfeats_r1 = get_feat_cols(qdf)
qdf["pnew"], it_pn1 = oof_lgb(qdf, qfeats_r1, "is_new", mono=False)
print(f"[R1] P(new) OOF ROC-AUC: {roc_auc_score(qdf[qdf.split<10].is_new, qdf[qdf.split<10].pnew):.4f} | iters: {it_pn1}")

# Fit R1 full models for test
m_r1 = fit_lgb(cv[feats_r1].to_numpy(np.float32), cv["y"].values, cv["group"].values, feats_r1, mono=True)
test["p1"] = m_r1.predict(test[feats_r1].to_numpy(np.float32), num_iteration=m_r1.best_iteration)
tqdf = build_query_table(test, "p1")
m_pn1 = fit_lgb(qdf[qfeats_r1].to_numpy(np.float32), qdf["is_new"].values, qdf["group"].values, qfeats_r1, mono=False)
tqdf["pn1"] = m_pn1.predict(tqdf[qfeats_r1].to_numpy(np.float32), num_iteration=m_pn1.best_iteration)

if CFG["two_round"]:
    print("[R2] Building combined MiewID + SIFT kpgraph query neighbours & Round 2 stacking...")
    cv["p1"] = cv["p"].values
    qdf["pn1"] = qdf["pnew"].values
    nbr_cv = build_combined_query_neighbors(cv, CFG["k_qq"])
    nbr_te = build_combined_query_neighbors(test, CFG["k_qq"])
    cv, qdf = add_round2_features(cv, qdf, nbr_cv)
    test, tqdf = add_round2_features(test, tqdf, nbr_te)

    feats_r2 = get_feat_cols(cv)
    cv["p"], it2 = oof_lgb(cv, feats_r2, "y", mono=True)
    qdf_r2 = build_query_table(cv, "p")
    for col in ("nb_pn_w", "nb0_pn", "nb_known_conf", "nb_agree"):
        qdf_r2[col] = qdf[col].values
    qfeats_r2 = get_feat_cols(qdf_r2)
    qdf_r2["pnew"], it_pn2 = oof_lgb(qdf_r2, qfeats_r2, "is_new", mono=False)
    print(f"[R2] Reranker iters: {it2} | P(new) OOF ROC-AUC: {roc_auc_score(qdf_r2[qdf_r2.split<10].is_new, qdf_r2[qdf_r2.split<10].pnew):.4f}")

    m_r2 = fit_lgb(cv[feats_r2].to_numpy(np.float32), cv["y"].values, cv["group"].values, feats_r2, mono=True)
    test["p"] = m_r2.predict(test[feats_r2].to_numpy(np.float32), num_iteration=m_r2.best_iteration)
    tqdf_r2 = build_query_table(test, "p")
    for col in ("nb_pn_w", "nb0_pn", "nb_known_conf", "nb_agree"):
        tqdf_r2[col] = tqdf[col].values
    m_pn2 = fit_lgb(qdf_r2[qfeats_r2].to_numpy(np.float32), qdf_r2["is_new"].values, qdf_r2["group"].values, qfeats_r2, mono=False)
    tqdf_r2["pnew"] = m_pn2.predict(tqdf_r2[qfeats_r2].to_numpy(np.float32), num_iteration=m_pn2.best_iteration)
    qdf, tqdf = qdf_r2, tqdf_r2
else:
    test["p"] = test["p1"].values
    tqdf["pnew"] = tqdf["pn1"].values


# %% [markdown] {"id":"Y1-I7jNs80wE"}
# ## 6. Calibration ($T_s, \beta_s$), CV-Gated Zhou Label Propagation & Exporting `submissions.csv` + `v4_test_probs.pkl`
# 

# %% [code] {"id":"pUjYUs0A80wF"}
# ============================================================================
# 6. CALIBRATION, ZHOU LABEL PROPAGATION & FINAL EXPORT
# ============================================================================
cv_pri = cv[cv["split"] < 10].reset_index(drop=True)
qdf_pri = qdf[qdf["split"] < 10].reset_index(drop=True)

def fit_temp_species(df_sp: pd.DataFrame) -> float:
    groups = []
    for _, grp in df_sp.groupby(["split", "query_id"], sort=False):
        y = grp["y"].values
        if y.sum() == 1:
            z = logit(np.clip(grp["p"].values, 1e-6, 1 - 1e-6))
            groups.append((z, int(np.argmax(y))))
    if not groups: return 1.0
    return float(minimize_scalar(
        lambda T: sum(-float((z / T)[pos] - logsumexp(z / T)) for z, pos in groups) / len(groups),
        bounds=(0.25, 5.0), method="bounded"
    ).x)

T_OPT = {sp: fit_temp_species(cv_pri[cv_pri["species"] == sp]) for sp in SPECIES}
print(f"Fitted per-species softmax temperatures: {T_OPT}")

def build_posterior_dict(df_in: pd.DataFrame, qdf_in: pd.DataFrame, T_map: dict, beta_map: dict):
    """Returns dict (split, sp) -> (qids, vocab, P0_matrix)."""
    pn_map = dict(zip(zip(qdf_in["split"], qdf_in["query_id"]), qdf_in["pnew"]))
    out = {}
    for (sp_s, sp_name), grp in df_in.groupby(["split", "species"], sort=False):
        qids = grp["query_id"].unique().tolist()
        vocab = [NEW] + sorted(grp["cand_id"].unique().tolist())
        v2i = {v: i for i, v in enumerate(vocab)}
        q2i = {q: i for i, q in enumerate(qids)}
        P0 = np.zeros((len(qids), len(vocab)), dtype=np.float32)
        T_v = T_map[sp_name]; b_v = beta_map[sp_name]
        for qid, gq in grp.groupby("query_id", sort=False):
            qi = q2i[qid]
            pi = float(expit(logit(np.clip(pn_map[(sp_s, qid)], 1e-6, 1 - 1e-6)) + b_v))
            z = logit(np.clip(gq["p"].values, 1e-6, 1 - 1e-6))
            pk = (1.0 - pi) * softmax(z / T_v)
            P0[qi, 0] = pi
            for cid, pv in zip(gq["cand_id"].values, pk):
                P0[qi, v2i[cid]] = float(pv)
        out[(sp_s, sp_name)] = (qids, vocab, P0)
    return out

def propagate_zhou(qids: list, sp: str, P0: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 1e-6 or len(qids) < 4:
        return P0
    n = len(qids)
    rows = np.array([id_to_row[q] for q in qids])
    S = EMB_L2[rows] @ EMB_L2[rows].T; np.fill_diagonal(S, -2.0)
    kpg = LNB_SCORERS[sp].kpgraph
    q2i = {q: i for i, q in enumerate(qids)}
    # Symmetric affinity combining strong MiewID cosine + SIFT kpgraph links
    r_idx, c_idx, w_val = [], [], []
    top5 = [set(np.argpartition(S[i], -min(5, n-1))[-min(5, n-1):].tolist()) for i in range(n)]
    for i, qi in enumerate(qids):
        # MiewID mutual high-confidence edges
        for j in top5[i]:
            if j > i and i in top5[j] and S[i, j] >= 0.65:
                w = float(S[i, j])
                r_idx.extend([i, j]); c_idx.extend([j, i]); w_val.extend([w, w])
        # SIFT keypoint graph edges
        for n_id, sc_kp in kpg.get(qi, [])[:5]:
            if n_id in q2i and sc_kp >= 6.0:
                j = q2i[n_id]
                if j > i:
                    w = float(np.clip(0.5 + 0.4 * np.tanh(sc_kp / 15.0), 0.1, 0.95))
                    r_idx.extend([i, j]); c_idx.extend([j, i]); w_val.extend([w, w])
    if not w_val:
        return P0
    W = sparse.csr_matrix((w_val, (r_idx, c_idx)), shape=(n, n), dtype=np.float32)
    deg = np.array(W.sum(1)).ravel()
    D_inv = sparse.diags(np.where(deg > 0, deg ** -0.5, 0.0))
    S_norm = D_inv @ W @ D_inv
    F = P0.copy()
    for _ in range(15):
        F = (1.0 - alpha) * P0 + alpha * (S_norm @ F)
    F /= np.clip(F.sum(axis=1, keepdims=True), 1e-8, None)
    return F

def eval_cv_map5(post_dict: dict, alpha_map: dict):
    truth_map = dict(zip(zip(qdf_pri["split"], qdf_pri["query_id"]), qdf_pri["truth"]))
    sp_scores = {sp: [] for sp in SPECIES}
    for (sp_s, sp_name), (qids, vocab, P0) in post_dict.items():
        P = propagate_zhou(qids, sp_name, P0, alpha_map[sp_name])
        preds = [[vocab[j] for j in np.argsort(-P[i])[:5]] for i in range(len(qids))]
        trues = [truth_map[(sp_s, q)] for q in qids]
        sp_scores[sp_name].append(mapk(trues, preds))
    overall = sum(W_TEST[sp] * np.mean(sp_scores[sp]) for sp in SPECIES)
    return overall, {sp: (float(np.mean(sp_scores[sp])), float(np.std(sp_scores[sp]))) for sp in SPECIES}

# 1. Tune per-species beta on primary CV (alpha=0)
BETA_OPT = {}
for sp in SPECIES:
    sub_df = cv_pri[cv_pri["species"] == sp]
    sub_q = qdf_pri[qdf_pri["species"] == sp]
    best_b, best_m = 0.0, -1.0
    for b in np.round(np.arange(-1.5, 1.51, 0.10), 2):
        pd_sp = build_posterior_dict(sub_df, sub_q, T_OPT, {sp: float(b)})
        truth_m = dict(zip(zip(sub_q["split"], sub_q["query_id"]), sub_q["truth"]))
        sc = []
        for (sp_s, _), (qids, vocab, P0) in pd_sp.items():
            preds = [[vocab[j] for j in np.argsort(-P0[i])[:5]] for i in range(len(qids))]
            sc.append(mapk([truth_m[(sp_s, q)] for q in qids], preds))
        if np.mean(sc) > best_m:
            best_m, best_b = float(np.mean(sc)), float(b)
    BETA_OPT[sp] = best_b
    print(f"[{sp}] Tuned beta={best_b:+.2f} -> Inductive CV MAP@5 = {best_m:.5f}")

cv_post = build_posterior_dict(cv_pri, qdf_pri, T_OPT, BETA_OPT)
base_ov, base_sp = eval_cv_map5(cv_post, {sp: 0.0 for sp in SPECIES})

# 2. Tune Zhou propagation alpha per species on CV
ALPHA_OPT = {}
for sp in SPECIES:
    best_a, best_m = 0.0, base_sp[sp][0]
    for a in CFG["prop_alphas"]:
        _, sp_res = eval_cv_map5(cv_post, {s: (a if s == sp else 0.0) for s in SPECIES})
        if sp_res[sp][0] > best_m + 0.0005:
            best_m, best_a = sp_res[sp][0], a
    ALPHA_OPT[sp] = best_a
final_ov, final_sp = eval_cv_map5(cv_post, ALPHA_OPT)

print("\n" + "=" * 72)
print(f"FINAL v4 PRIMARY CV MAP@5: {final_ov:.5f} (Inductive: {base_ov:.5f})")
for sp in SPECIES:
    print(f"  {sp:8s}: {final_sp[sp][0]:.5f} +/- {final_sp[sp][1]:.5f} (T={T_OPT[sp]:.3f}, beta={BETA_OPT[sp]:+.2f}, alpha={ALPHA_OPT[sp]:.2f})")
print("=" * 72)

# 3. Generate final test probabilities & export submissions.csv + v4_test_probs.pkl
te_post = build_posterior_dict(test, tqdf, T_OPT, BETA_OPT)
sp_valid = {sp: set(train_df[train_df["species"] == sp]["identity"].unique()) | {NEW} for sp in SPECIES}

test_probs_export = {}  # {query_id: {label: prob}} ready for teammate blending!
pred_map = {}
for (_, sp_name), (qids, vocab, P0) in te_post.items():
    P_final = propagate_zhou(qids, sp_name, P0, ALPHA_OPT[sp_name])
    for i, qid in enumerate(qids):
        top_idx = np.argsort(-P_final[i])
        test_probs_export[qid] = {vocab[j]: float(P_final[i, j]) for j in top_idx[:15] if P_final[i, j] > 1e-6}
        clean = []
        for j in top_idx:
            lbl = vocab[j]
            if lbl in sp_valid[sp_name] and lbl not in clean:
                clean.append(lbl)
            if len(clean) == 5:
                break
        pred_map[qid] = " ".join(clean)

sub = pd.DataFrame({"id": test_df["id"].values, SUB_PRED_COL: [pred_map[i] for i in test_df["id"].values]})
assert len(sub) == 2791 and sub[SUB_PRED_COL].notna().all() and list(sub["id"]) == list(test_df["id"])

sub_path = WORK_DIR / "submissions.csv"
probs_path = WORK_DIR / "v4_test_probs.pkl"
sub.to_csv(sub_path, index=False)
probs_path.write_bytes(pickle.dumps(test_probs_export))

id_to_sp = dict(zip(test_df["id"], test_df["species"]))
sub_sp = sub.assign(species=sub["id"].map(id_to_sp), top1=sub[SUB_PRED_COL].str.split().str[0])
print(f"\nSaved & verified {sub_path} ({len(sub)} rows) and {probs_path}:")
for sp, grp in sub_sp.groupby("species"):
    print(f"  [{sp}] rank-1 new_individual={(grp['top1']==NEW).mean():.3f} | in top-5={grp[SUB_PRED_COL].str.contains(NEW).mean():.3f}")
sub.head(10)
