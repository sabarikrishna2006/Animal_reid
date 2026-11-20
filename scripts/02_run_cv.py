"""Run the full pipeline on the open-set CV and log an experiment.
Usage: py scripts/02_run_cv.py --config configs/exp_X.json [--parent EXP_ID] [--change "..."]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--parent", default="")
    ap.add_argument("--change", default="")
    a = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_compliance.py")], check=True)
    import numpy as np
    from reid.io import load_config
    from reid.pipeline import prepare, build_tables, run_second_level, decide, evaluate, tune_beta
    from reid.tracking import new_exp_id, save_experiment, append_registry
    from reid import SPECIES

    cfg = load_config(a.config)
    t0 = time.time()
    data, ctx, splits = prepare(cfg)
    tabs = build_tables(cfg, data, ctx, splits)
    cv = tabs["cv"]
    t_tab = time.time() - t0
    trans_on = cfg["transductive"].get("enabled", False)
    from reid.pipeline import load_extra_cv
    out = run_second_level(cv, cfg, test=tabs["test"] if trans_on else None, extra=load_extra_cv(cfg))
    cv, qdf, T = out["cv"], out["qdf"], out["T"]
    results = {}
    for method in cfg["model"].get("methods", ["softmax", "direct"]):
        beta = tune_beta(cv, qdf, "pnew", T, method)
        b = {sp: beta[sp][0] for sp in SPECIES}
        preds = decide(cv, qdf, "pnew", b, T, method)
        summ, _ = evaluate(cv, preds, qdf, "pnew")
        summ.update(beta=b, T=T)
        results[method] = summ
        print(f"[{method:7s}] CV MAP@5 = {summ['cv_mean']:.4f} ± {summ['cv_std']:.4f} | hyena {summ['map5_hyena']:.4f} "
              f"leopard {summ['map5_leopard']:.4f} | known h/l {summ['map5_known_hyena']:.3f}/{summ['map5_known_leopard']:.3f}"
              f" | newAUC h/l {summ['new_auc_hyena']:.3f}/{summ['new_auc_leopard']:.3f} | beta {b} T {np.round(list(T.values()), 2)}")
    if cfg["transductive"].get("enabled", False):
        from reid.transductive import apply_transductive
        from reid.pipeline import W_TEST
        method = cfg["model"].get("final_method", "softmax")
        b = results[method]["beta"]
        import pickle
        from reid.io import P
        sl = {k: out[k] for k in ("test", "tqdf", "feats", "qfeats")}
        sl.update(cv=cv, qdf=qdf, T=T, beta=b)
        (P(cfg, "cache") / f"second_level_{cfg.get('exp_slug')}.pkl").write_bytes(pickle.dumps(sl))
        _, best_t, _ = apply_transductive(cfg, ctx, data, splits, cv, qdf, out["test"], out["tqdf"], T, b)
        ps = np.array([sum(W_TEST[sp] * best_t[sp]["per_split"][r] for sp in SPECIES) for r in range(len(splits))])
        po = np.array([sum(W_TEST[sp] * best_t[sp]["per_split_off"][r] for sp in SPECIES) for r in range(len(splits))])
        d = ps - po
        results["transductive"] = dict(results[method], cv_mean=float(ps.mean()), cv_std=float(ps.std(ddof=1)),
                                       per_split=ps.tolist(), trans=best_t,
                                       map5_hyena=best_t["hyena"]["map"], map5_leopard=best_t["leopard"]["map"])
        print(f"[trans  ] CV MAP@5 = {ps.mean():.4f} ± {ps.std(ddof=1):.4f} | off: {po.mean():.4f} | "
              f"paired delta {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})")
    best = max(results, key=lambda k: results[k]["cv_mean"])
    m = dict(results[best], method=best, all_methods={k: {kk: v[kk] for kk in ("cv_mean", "cv_std")} for k, v in results.items()},
             seconds_tables=t_tab, seconds_total=time.time() - t0, n_rows=len(cv), n_feats=len(out["feats"]))
    exp_id = new_exp_id(cfg.get("exp_slug", "exp"))
    d = save_experiment(cfg, exp_id, m, notes=a.change)
    np.savez_compressed(d / "oof.npz", p=cv.p.values, pnew=qdf.pnew.values)
    # feature importance
    try:
        from reid.models import fit_full
        imp = fit_full(cv, out["feats"], "y", cv.group.values).model.feature_importance("gain")
        fi = sorted(zip(out["feats"], imp), key=lambda t: -t[1])
        (d / "feature_importance.json").write_text(json.dumps([(f, float(v)) for f, v in fi], indent=1))
        print("top features:", [f for f, _ in fi[:12]])
    except Exception as e:
        print("feature importance failed:", e)
    append_registry(cfg, {"exp_id": exp_id, "date": exp_id[:8], "parent_exp": a.parent, "change": a.change or cfg.get("exp_slug"),
                          "cv_mean": round(m["cv_mean"], 5), "cv_std": round(m["cv_std"], 5),
                          "map_hyena": round(m["map5_hyena"], 5), "map_leopard": round(m["map5_leopard"], 5),
                          "known_map": round((m["map5_known_hyena"] + m["map5_known_leopard"]) / 2, 5),
                          "new_auc": round((m["new_auc_hyena"] + m["new_auc_leopard"]) / 2, 5)})
    print(f"saved {exp_id} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
