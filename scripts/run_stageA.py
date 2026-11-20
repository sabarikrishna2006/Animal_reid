"""Gates 2-3: threshold-rule retrieval baselines on the open-set CV.

For each variant: identity score per (query, identity), new_individual inserted after identities whose score >= tau,
tau tuned per species on CV (mean over 5 splits). Reports CV MAP@5 per species and test-weighted overall.
Usage: py scripts/run_stageA.py [variant ...]
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from reid.io import load_config, load_data
from reid.splits import build_splits
from reid.retrieval import embed_unsupervised, IdentityIndex, baseline_predict, fit_lda, l2n
from reid.metrics import map_at_5
from reid.tracking import new_exp_id, save_experiment, append_registry
from reid import SPECIES

W_TEST = {"hyena": 1217 / 2791, "leopard": 1574 / 2791}
TAUS = np.round(np.arange(0.05, 0.90, 0.01), 2)

import os
cfg = load_config(os.environ.get("REID_CFG")); data = load_data(cfg)
splits = build_splits(data, cfg)
meta, X = data.meta, data.X


def scores_for(variant, fold):
    """Returns (identity score matrix q x n_ids, ids)."""
    kind, agg = variant.split("|")
    if kind.startswith("lda"):
        g = float(kind.split(":")[1]) if ":" in kind else 0.2
        U = EMB["l2"]
        f = fit_lda(U[fold.gallery_rows], fold.gallery_labels, gamma=g)
        Zg, Zq = f(U[fold.gallery_rows]), f(U[fold.query_rows])
    else:
        if kind not in EMB and kind.startswith("pcaw"):
            _, d_, e_ = kind.split(":"); EMB[kind] = embed_unsupervised(X, meta, "pcaw", d=int(d_), eps=float(e_))
        U = EMB[kind]
        Zg, Zq = U[fold.gallery_rows], U[fold.query_rows]
    S = Zq @ Zg.T
    idx = IdentityIndex(fold.gallery_labels)
    if agg == "max":
        M = idx.max(S)
    elif agg.startswith("top"):
        M = idx.sorted_desc_k(S, int(agg[3:]))
    elif agg.startswith("lse"):
        M = idx.logsumexp(S, float(agg[3:]))
    elif agg == "proto":
        M = Zq @ idx.prototypes(Zg).T
    elif agg == "maxproto":
        M = 0.5 * idx.max(S) + 0.5 * (Zq @ idx.prototypes(Zg).T)
    return M, idx.ids


EMB = {"l2": embed_unsupervised(X, meta, "l2"), "std": embed_unsupervised(X, meta, "std")}
variants = sys.argv[1:] or ["l2|max"]
for v in variants:
    kind = v.split("|")[0]
    if kind.startswith("qe") and kind not in EMB:
        _, base, k, al = kind.split(":")
        if base not in EMB:
            EMB[base] = embed_unsupervised(X, meta, "pcaw", d=512, eps=1.0) if base == "pcaw" else EMB["l2"]
        EMB[kind] = embed_unsupervised(X, meta, "qe", base=EMB[base], k=int(k), alpha=float(al))
    if kind.startswith("pcaw") and kind not in EMB:
        _, d, eps = kind.split(":")
        EMB[kind] = embed_unsupervised(X, meta, "pcaw", d=int(d), eps=float(eps))
    t0 = time.time()
    res = {sp: np.zeros((len(splits), len(TAUS))) for sp in SPECIES}
    recall = {sp: [] for sp in SPECIES}
    for r, entry in enumerate(splits):
        for sp in SPECIES:
            fold, truth = entry[sp]
            M, ids = scores_for(v, fold)
            td = truth.as_dict()
            top20 = np.argsort(-M, axis=1)[:, :20]
            known = truth.labels != "new_individual"
            recall[sp].append(np.mean([truth.labels[i] in set(ids[top20[i]]) for i in np.where(known)[0]]))
            for t, tau in enumerate(TAUS):
                preds = baseline_predict(M, ids, tau)
                res[sp][r, t] = map_at_5(td, dict(zip(fold.query_ids, preds)))
    best = {sp: int(res[sp].mean(0).argmax()) for sp in SPECIES}
    per_split = {sp: res[sp][:, best[sp]] for sp in SPECIES}
    overall = sum(W_TEST[sp] * per_split[sp] for sp in SPECIES)
    m = {"variant": v, "cv_mean": float(overall.mean()), "cv_std": float(overall.std(ddof=1)),
         "per_split": overall.tolist(),
         **{f"map_{sp}": float(per_split[sp].mean()) for sp in SPECIES},
         **{f"tau_{sp}": float(TAUS[best[sp]]) for sp in SPECIES},
         **{f"recall20_{sp}": float(np.mean(recall[sp])) for sp in SPECIES},
         "seconds": time.time() - t0}
    print(f"{v:22s} CV={m['cv_mean']:.4f}±{m['cv_std']:.4f} hyena={m['map_hyena']:.4f}(tau {m['tau_hyena']}) "
          f"leopard={m['map_leopard']:.4f}(tau {m['tau_leopard']}) recall@20 h={m['recall20_hyena']:.3f} "
          f"l={m['recall20_leopard']:.3f} [{m['seconds']:.0f}s]")
    exp_id = new_exp_id("A_" + v.replace("|", "_").replace(":", "-"))
    save_experiment(cfg, exp_id, m)
    append_registry(cfg, {"exp_id": exp_id, "date": exp_id[:8], "parent_exp": "", "change": f"stageA {v}",
                          "cv_mean": round(m["cv_mean"], 5), "cv_std": round(m["cv_std"], 5),
                          "map_hyena": round(m["map_hyena"], 5), "map_leopard": round(m["map_leopard"], 5)})
    json.dump(per_split and {k: (val.tolist() if hasattr(val, 'tolist') else val) for k, val in per_split.items()},
              open(Path(cfg["paths"]["experiments"]) / exp_id / "per_split_species.json", "w"))
