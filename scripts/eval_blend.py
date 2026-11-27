"""Honest nested leave-one-split-out evaluation & blend verifier over CV probability exports.

Usage:
  py scripts/eval_blend.py --cv-probs data/cache/cv_probs_v25.pkl data/cache/cv_probs_v26.pkl --config configs/exp_v25_hnsw_rescue.json
"""
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from reid import NEW, SPECIES
from reid.io import load_config, load_data
from reid.metrics import ap_at_5
from reid.pipeline import W_TEST
from reid.splits import build_splits


def _ap_dict(probs, new_scale, truth):
    pr = dict(probs)
    if NEW in pr and new_scale != 1.0:
        pr[NEW] = pr[NEW] * new_scale
    top5 = [k for k, _ in sorted(pr.items(), key=lambda t: -t[1])[:5]]
    return ap_at_5(truth, top5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-probs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--config", default="configs/exp_v25_hnsw_rescue.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    data = load_data(cfg)
    splits = build_splits(data, cfg)
    probs_list = [pickle.load(open(p, "rb")) for p in a.cv_probs]
    names = a.names or [Path(p).stem for p in a.cv_probs]
    assert len(probs_list) == len(names)

    # Collect per-(split, species) query ground truths
    truths = {(r, sp): splits[r][sp][1].as_dict() for r in range(len(splits)) for sp in SPECIES}

    def eval_single(pr_map):
        per_sp = {sp: [] for sp in SPECIES}
        overall = []
        for r in range(len(splits)):
            ov = 0.0
            for sp in SPECIES:
                td = truths[(r, sp)]
                sc = float(np.mean([_ap_dict(pr_map[r][q], 1.0, t) for q, t in td.items()]))
                per_sp[sp].append(sc)
                ov += W_TEST[sp] * sc
            overall.append(ov)
        return np.array(overall), {sp: np.array(per_sp[sp]) for sp in SPECIES}

    singles = [eval_single(p) for p in probs_list]
    for nm, (ov, spm) in zip(names, singles):
        print(f"[{nm:22s}] CV={ov.mean():.5f} ± {ov.std(ddof=1):.5f} "
              f"(h={spm['hyena'].mean():.5f}, l={spm['leopard'].mean():.5f})")

    if len(probs_list) == 2:
        # Paired deltas across models
        d = singles[1][0] - singles[0][0]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\nPaired {names[1]} - {names[0]}: mean={d.mean():+.5f}, SE={se:.5f}, "
              f"criterion(>max(0.002, SE))={d.mean() > max(0.002, se)}")

        p1, p2 = probs_list
        # Equal 0.5/0.5 mix
        eq_map = {}
        for r in range(len(splits)):
            eq_map[r] = {}
            for sp in SPECIES:
                for q in truths[(r, sp)]:
                    labs = set(p1[r][q]) | set(p2[r][q])
                    eq_map[r][q] = {l: 0.5 * p1[r][q].get(l, 0.0) + 0.5 * p2[r][q].get(l, 0.0) for l in labs}
        ov_eq, sp_eq = eval_single(eq_map)
        best_single = singles[0][0] if singles[0][0].mean() >= singles[1][0].mean() else singles[1][0]
        d_eq = ov_eq - best_single
        se_eq = d_eq.std(ddof=1) / np.sqrt(len(d_eq))
        print(f"[Blend 0.50/0.50      ] CV={ov_eq.mean():.5f} ± {ov_eq.std(ddof=1):.5f} "
              f"(h={sp_eq['hyena'].mean():.5f}, l={sp_eq['leopard'].mean():.5f}) | "
              f"delta_vs_best_single={d_eq.mean():+.5f} (SE={se_eq:.5f})")

        # Nested leave-one-split-out weight optimization
        ws = np.round(np.arange(0.0, 1.01, 0.1), 2)
        nested_sp = {sp: [] for sp in SPECIES}
        picked_w = {sp: [] for sp in SPECIES}
        for sp in SPECIES:
            mat = np.zeros((len(ws), len(splits)))
            for wi, w in enumerate(ws):
                for r in range(len(splits)):
                    td = truths[(r, sp)]
                    aps = []
                    for q, t in td.items():
                        labs = set(p1[r][q]) | set(p2[r][q])
                        mix = {l: float(w) * p1[r][q].get(l, 0.0) + (1.0 - float(w)) * p2[r][q].get(l, 0.0) for l in labs}
                        aps.append(_ap_dict(mix, 1.0, t))
                    mat[wi, r] = np.mean(aps)
            for r in range(len(splits)):
                tr_m = np.delete(mat, r, axis=1).mean(1)
                wi_b = int(np.argmax(tr_m))
                nested_sp[sp].append(float(mat[wi_b, r]))
                picked_w[sp].append(float(ws[wi_b]))
        ov_nes = np.array([W_TEST["hyena"] * nested_sp["hyena"][r] + W_TEST["leopard"] * nested_sp["leopard"][r]
                           for r in range(len(splits))])
        d_nes = ov_nes - best_single
        se_nes = d_nes.std(ddof=1) / np.sqrt(len(d_nes))
        print(f"[Nested Blend ({names[0]} w)] CV={ov_nes.mean():.5f} ± {ov_nes.std(ddof=1):.5f} "
              f"(h={np.mean(nested_sp['hyena']):.5f}, l={np.mean(nested_sp['leopard']):.5f}) | "
              f"delta_vs_best_single={d_nes.mean():+.5f} (SE={se_nes:.5f}) | outer_w={picked_w}")


if __name__ == "__main__":
    main()
