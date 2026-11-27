"""Builds the LNBNN neighbour lists used by configs with retrieval.lnbnn = true (HotSpotter-style one-vs-many keypoint
matching). For every train and test image, each RootSIFT descriptor gets its 50 nearest training descriptors of the same
species (faiss-cpu). Default builds fast HNSW index (99.1% top-10 recall) into data/cache/lnbnn_hnsw or --out.
Usage:
  py scripts/04_build_lnbnn.py --out data/cache/lnbnn_hnsw --mode hnsw
  py scripts/04_build_lnbnn.py --out data/cache/lnbnn --mode ivf --nprobe 64
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import faiss
from reid.io import load_config, load_data, Sift

KNN = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=("hnsw", "ivf"), default="hnsw")
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--ef-search", type=int, default=64)
    a = ap.parse_args()
    faiss.omp_set_num_threads(min(20, os.cpu_count() or 8))
    cfg = load_config(); data = load_data(cfg); S = Sift(cfg)
    default_dir = "data/cache/lnbnn_hnsw" if a.mode == "hnsw" else cfg["paths"].get("lnbnn", "data/cache/lnbnn")
    out = Path(a.out or default_dir); out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    row = dict(zip(data.meta.id, data.meta.row))

    def rs(i):
        d = S[row[i]][1].astype(np.float32).reshape(-1, 128)
        return np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1e-7))

    tr, te = data.train, data.test
    for sp in ("hyena", "leopard"):
        t0 = time.time()
        gids = tr.id[tr.species == sp].values; qids = np.r_[gids, te.id[te.species == sp].values]
        X = [rs(g) for g in gids]
        owner = np.repeat(np.arange(len(gids)), [len(x) for x in X]).astype(np.int32)
        X = np.concatenate(X)
        if a.mode == "hnsw":
            index = faiss.IndexHNSWFlat(128, 32)
            index.hnsw.efConstruction = 64
            index.add(X)
            index.hnsw.efSearch = a.ef_search
        else:
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(128), 128, 2048 if sp == "hyena" else 4096)
            index.train(X[np.random.RandomState(0).choice(len(X), min(len(X), 300000), replace=False)])
            index.add(X); index.nprobe = a.nprobe
        print(f"{sp}: index ({a.mode}) built in {time.time() - t0:.0f}s", flush=True)
        qlen = [len(S[row[q]][1]) for q in qids]; qoff = np.r_[0, np.cumsum(qlen)]
        NB = np.lib.format.open_memmap(out / f"{sp}_nb.npy", "w+", np.int32, (int(qoff[-1]), KNN))
        DI = np.lib.format.open_memmap(out / f"{sp}_d.npy", "w+", np.float16, (int(qoff[-1]), KNN))
        for b in range(0, len(qids), 400):
            e = min(b + 400, len(qids))
            if qoff[e] == qoff[b]:
                continue
            d, i = index.search(np.concatenate([rs(q) for q in qids[b:e]]), KNN)
            NB[qoff[b]:qoff[e]] = np.where(i >= 0, owner[np.maximum(i, 0)], -1)
            DI[qoff[b]:qoff[e]] = np.sqrt(np.maximum(d, 0))
        NB.flush(); DI.flush()
        (out / f"{sp}_meta.json").write_text(json.dumps({"gallery": gids.tolist(), "queries": qids.tolist(),
                                                         "qoff": qoff.tolist()}))
        print(f"{sp}: {len(X)} gallery descriptors, {len(qids)} query images, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
