"""Builds the LNBNN neighbour lists used by configs with retrieval.lnbnn = true (HotSpotter-style one-vs-many keypoint
matching). For every train and test image, each RootSIFT descriptor gets its 50 nearest training descriptors of the same
species (faiss-cpu IVF index). Output: data/cache/lnbnn/{species}_nb.npy, _d.npy, _meta.json (~1.3 GB, ~8 min).
Same format as Dev's notebook, so either side's cache can be copied over.
Usage: py scripts/04_build_lnbnn.py      (after 00_audit.py)"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import faiss
from reid.io import load_config, load_data, Sift

KNN = 50


def main():
    cfg = load_config(); data = load_data(cfg); S = Sift(cfg)
    out = Path(cfg["paths"].get("lnbnn", "data/cache/lnbnn")); out = out if out.is_absolute() else ROOT / out
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
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(128), 128, 2048 if sp == "hyena" else 4096)
        index.train(X[np.random.RandomState(0).choice(len(X), min(len(X), 300000), replace=False)])
        index.add(X); index.nprobe = 12
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
