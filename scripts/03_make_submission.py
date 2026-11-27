"""The ONLY way predictions are produced. Fits second-level models on all CV tables, predicts test, writes submission.
Usage:
  py scripts/03_make_submission.py --config configs/exp_X.json
  py scripts/03_make_submission.py --baseline        # Gate 2 raw-cosine threshold baseline
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--blend", nargs="+", metavar="PROBS.pkl", default=None,
                    help="team/ensemble blend: weighted average of 2+ {test_id: {label: prob}} files")
    ap.add_argument("--w", type=float, default=0.5, help="weight of first file for 2-file --blend (default 0.5)")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="weights matching --blend files")
    a = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_compliance.py")], check=True)
    import numpy as np, pandas as pd
    from reid.io import load_config, load_data, P
    from reid.decision import write_submission
    from reid.tracking import new_exp_id, save_experiment, append_registry
    from reid import SPECIES, NEW

    cfg = load_config(a.config)
    t0 = time.time()
    if a.blend:
        import pickle
        assert len(a.blend) >= 2, "--blend requires at least 2 probability files"
        data = load_data(cfg)
        probs_list = [pickle.load(open(p, "rb")) for p in a.blend]
        for p in probs_list:
            assert set(p) == set(data.test.id), "every probability file must cover all test ids"
        if a.weights is not None:
            assert len(a.weights) == len(probs_list), "--weights must match --blend count"
            ws = np.asarray(a.weights, dtype=float)
        elif len(probs_list) == 2:
            ws = np.array([a.w, 1.0 - a.w], dtype=float)
        else:
            ws = np.ones(len(probs_list), dtype=float)
        ws = ws / ws.sum()
        allowed = data.train.groupby("species").identity.apply(set).to_dict()
        sp = dict(zip(data.test.id, data.test.species))
        preds = {}
        for q in data.test.id:
            labs = set().union(*(p[q].keys() for p in probs_list))
            sc = {l: sum(float(w) * p[q].get(l, 0.0) for w, p in zip(ws, probs_list)) for l in labs}
            ranked = [l for l in sorted(labs, key=lambda l: (-sc[l], l)) if l == NEW or l in allowed[sp[q]]][:5]
            for l in [NEW] + sorted(allowed[sp[q]]):
                if len(ranked) == 5:
                    break
                if l not in ranked:
                    ranked.append(l)
            preds[q] = ranked
        exp_id = new_exp_id("sub_team_blend" + (("_" + a.name) if a.name else ""))
        info = {"blend": [str(p) for p in a.blend], "weights": ws.tolist()}
    elif a.baseline:
        from reid.retrieval import embed_unsupervised, IdentityIndex, baseline_predict
        from reid.splits import test_fold
        data = load_data(cfg); U = embed_unsupervised(data.X, data.meta, "l2")
        preds = {}
        for sp in SPECIES:
            f = test_fold(data, sp)
            idx = IdentityIndex(f.gallery_labels)
            M = idx.max(U[f.query_rows] @ U[f.gallery_rows].T)
            for qid, labs in zip(f.query_ids, baseline_predict(M, idx.ids, cfg["retrieval"]["tau"][sp])):
                preds[qid] = labs
        exp_id = new_exp_id("sub_baseline_rawcos")
        info = {"tau": cfg["retrieval"]["tau"]}
    else:
        from reid.pipeline import prepare, build_tables, run_second_level, decide, tune_beta, evaluate, load_extra_cv, attach_optional_features
        from reid.calibration import em_prior, prior_shift
        data, ctx, splits = prepare(cfg)
        tabs = build_tables(cfg, data, ctx, splits)
        extra = load_extra_cv(cfg)
        cv_in, test_in, extra = attach_optional_features(cfg, data, ctx, splits, tabs["cv"], tabs["test"], extra)
        if cfg["model"].get("sw_proto", False):
            from reid.pipeline import attach_sw_proto
            cv_in, test_in, extra = attach_sw_proto(cfg, data, ctx, splits, cv_in, test_in, extra)
        import gc
        del tabs
        gc.collect()
        if cfg["model"].get("paradigm") == "dual":
            from reid.dual_model import run_dual_pipeline
            out = run_dual_pipeline(cfg, data, ctx, splits, cv_in, test=test_in, extra=extra)
        else:
            out = run_second_level(cv_in, cfg, test=test_in, extra=extra)
        cv, qdf, T, test, tq = out["cv"], out["qdf"], out["T"], out["test"], out["tqdf"]
        method = cfg["model"].get("final_method", "softmax")
        beta = tune_beta(cv, qdf, "pnew", T, method)
        b = {sp: beta[sp][0] for sp in SPECIES}
        cvp = decide(cv, qdf, "pnew", b, T, method)
        summ, _ = evaluate(cv, cvp, qdf, "pnew")
        info = {"cv_mean": summ["cv_mean"], "cv_std": summ["cv_std"], "beta": b, "T": T, "method": method}
        # novelty-rate diagnostics + optional prior-shift correction
        for sp in SPECIES:
            pi_cv = float(qdf[qdf.species == sp].is_new.mean())
            pt = tq[tq.species == sp].pnew.values
            pi_em = float(em_prior(pt, pi_cv))
            info[f"pi_cv_{sp}"] = pi_cv; info[f"pi_test_mean_pnew_{sp}"] = float(pt.mean()); info[f"pi_test_em_{sp}"] = pi_em
            if cfg.get("calibration", {}).get("prior_shift", False):
                tq.loc[tq.species == sp, "pnew"] = prior_shift(pt, pi_cv, pi_em)
        print("novelty:", {k: round(v, 3) for k, v in info.items() if k.startswith("pi_")})
        if cfg["transductive"].get("enabled", False):
            from reid.transductive import apply_transductive
            from reid.pipeline import W_TEST
            preds, best, _ = apply_transductive(cfg, ctx, data, splits, cv, qdf, test, tq, T, b)
            info["transductive"] = best
            ps = np.array([sum(W_TEST[sp] * best[sp]["per_split"][r] for sp in SPECIES) for r in range(len(splits))])
            info["cv_mean"], info["cv_std"] = float(ps.mean()), float(ps.std(ddof=1))
            print(f"CV with transductive: {ps.mean():.4f} ± {ps.std(ddof=1):.4f}")
        else:
            tp = decide(test, tq, "pnew", b, T, method)
            preds = {k.split("|", 1)[1]: v for k, v in tp.items()}
        exp_id = new_exp_id("sub_" + cfg.get("exp_slug", "exp") + (("_" + a.name) if a.name else ""))
    path = P(cfg, "submissions") / f"sub_{exp_id}.csv"
    sub = write_submission(preds, data.test, data.train, path)
    if not a.baseline and not a.blend and cfg["transductive"].get("enabled", False):
        import pickle
        from reid.transductive import EXPORT_PROBS
        assert set(EXPORT_PROBS) == set(data.test.id), "probability export incomplete"
        pickle.dump(EXPORT_PROBS, open(path.with_suffix(".probs.pkl"), "wb"))
        print(f"wrote {path.with_suffix('.probs.pkl')} ({len(EXPORT_PROBS)} test queries)")
        from reid.transductive import EXPORT_CV
        pickle.dump(EXPORT_CV, open(path.with_name(path.stem + ".cv.probs.pkl"), "wb"))
        print(f"wrote CV probs: { {r: len(v) for r, v in EXPORT_CV.items()} }")
    first = sub.predictions.str.split(" ").str[0]
    info["frac_new_at_rank1"] = {sp: float((first[data.test.species.values == sp] == NEW).mean()) for sp in SPECIES}
    info["seconds"] = time.time() - t0
    save_experiment(cfg, exp_id, info)
    append_registry(cfg, {"exp_id": exp_id, "date": exp_id[:8], "change": f"SUBMISSION -> {path.name}",
                          "cv_mean": round(info.get("cv_mean", float("nan")), 5)})
    print(f"wrote {path} | new@rank1 {info['frac_new_at_rank1']} | {json.dumps({k: v for k, v in info.items() if k in ('cv_mean', 'beta')}, default=str)}")


if __name__ == "__main__":
    main()
