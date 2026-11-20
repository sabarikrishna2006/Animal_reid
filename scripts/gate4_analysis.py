"""Gate 4 analysis on cached tables: calibration (reliability, ECE, NLL) + per-feature-group ablation.
Usage: py scripts/gate4_analysis.py --config configs/exp_g4_sift.json [--ablate]
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reid.io import load_config
from reid.pipeline import prepare, build_tables, run_second_level, tune_beta, decide, evaluate, W_TEST
from reid.calibration import softmax_groups, shift_logit
from scipy.special import logit
from reid import SPECIES

GROUPS = {
    "sift": lambda c: c.startswith("sift_") or c.startswith("q_sift") or c == "sift_inl_rank",
    "viewpoint": lambda c: c in ("max_same_side", "has_same_side", "max_opp_side", "compat_best", "max_minus_compat", "q_side"),
    "lda": lambda c: c.endswith("_lda"),
    "pcaw": lambda c: c.endswith("_pcaw"),
    "reciprocity": lambda c: c == "recip_rank",
    "gallery_stats": lambda c: c in ("log_nimg", "n_vp"),
    "query_quality": lambda c: c in ("q_lognorm", "q_kp"),
}


def ece(p, y, bins=10):
    e, rows = 0.0, []
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return e, rows


def calib(cfg, cv, qdf, T, beta):
    out = {}
    df = cv.copy(); df["_key"] = df.split.astype(str) + "|" + df.query_id
    gid = pd.factorize(df["_key"])[0]
    df["post"] = softmax_groups(logit(np.clip(df.p.values, 1e-6, 1 - 1e-6)), gid, df.species.map(T).values)
    qk = qdf.split.astype(str) + "|" + qdf.query_id
    pn = pd.Series(shift_logit(qdf.pnew.values, qdf.species.map(beta).values), index=qk.values)
    df["pi"] = df["_key"].map(pn)
    df["pc"] = df.post * (1 - df.pi)
    top = df.loc[df.groupby("_key").pc.idxmax()]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, p, y) in zip(axs, [("top-1 identity P(c)", top.pc.values, top.y.values),
                                       ("P(new)", shift_logit(qdf.pnew.values, qdf.species.map(beta).values), qdf.is_new.values)]):
        e, rows = ece(p, y)
        nll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
        out[name] = {"ece": float(e), "nll": float(nll), "mean_p": float(p.mean()), "rate": float(y.mean())}
        r = np.array(rows)
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
        ax.plot(r[:, 0], r[:, 1], "o-", color="#2a78d6", lw=2, ms=6)
        ax.set_title(f"{name}: ECE={e:.3f}, NLL={nll:.3f}", loc="left")
        ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "figures" / "fig4_reliability.png", dpi=130)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--ablate", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    data, ctx, splits = prepare(cfg)
    tabs = build_tables(cfg, data, ctx, splits)
    res = {}

    def run(drop, label):
        c = json.loads(json.dumps(cfg)); c["model"]["drop_features"] = drop
        o = run_second_level(tabs["cv"], c, log=lambda *x: None)
        beta = {sp: v[0] for sp, v in tune_beta(o["cv"], o["qdf"], "pnew", o["T"], "softmax").items()}
        pr = decide(o["cv"], o["qdf"], "pnew", beta, o["T"], "softmax")
        s, _ = evaluate(o["cv"], pr, o["qdf"], "pnew")
        print(f"{label:28s} CV={s['cv_mean']:.4f}±{s['cv_std']:.4f} hyena={s['map5_hyena']:.4f} leopard={s['map5_leopard']:.4f} "
              f"newAUC h/l={s['new_auc_hyena']:.3f}/{s['new_auc_leopard']:.3f}", flush=True)
        return s, o, beta

    s_full, o, beta = run([], "full")
    res["full"] = s_full
    res["calibration"] = calib(cfg, o["cv"], o["qdf"], o["T"], beta)
    print("calibration:", json.dumps(res["calibration"], indent=1))
    if a.ablate:
        cols = [c for c in tabs["cv"].columns]
        for g, fn in GROUPS.items():
            drop = [c for c in cols if fn(c)]
            s, _, _ = run(drop, f"- {g} ({len(drop)} feats)")
            d = np.array(s["per_split"]) - np.array(s_full["per_split"])
            res[f"minus_{g}"] = {"cv_mean": s["cv_mean"], "delta": float(d.mean()),
                                 "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                 "map5_hyena": s["map5_hyena"], "map5_leopard": s["map5_leopard"]}
            print(f"    delta vs full {d.mean():+.4f} (SE {d.std(ddof=1)/np.sqrt(len(d)):.4f})", flush=True)
    (ROOT / "reports" / "gate4_analysis.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
