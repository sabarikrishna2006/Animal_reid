"""Final team blend: average calibrated label probabilities from Dev's and Sabari's pipelines.

Builds our final submitted blends from saved calibrated {test_id: {label: prob}} pickles:
  lnb32  ->  Dev half: 1/3 avg(bg3, dev_hub) + 2/3 dev_v10_fullbg_lprop      |  Sabari half: 50/50(v17, v19)
  lnb34  ->  Dev half: 1/2 avg(bg1, bg2, bg3) + 1/2 dev_v10_sw2152_lprop     |  Sabari half: avg(v17, v19, v24)
  lnb36  ->  Dev half: 1/2 dev_bg_sw2152_prop + 1/2 dev_v10_sw2152_lprop     |  Sabari half: avg(v17, v19, v23, v24, v25)
Each half gets 50% of the final weight. No training happens here.

Usage (from the repo root):
  py dev_pipeline/make_team_blend.py --recipe lnb36 --dev-dir dev_pipeline --sab-dir submissions
"""
import argparse, glob, os, pickle
import numpy as np, pandas as pd
from core import load, NEW

H = os.path.dirname(os.path.abspath(__file__))

# (source, file key, final weight). "dev" keys are file names in --dev-dir,
# "sab" keys are experiment slugs matched against *<slug>*.probs.pkl in --sab-dir.
RECIPES = {
    "lnb32": [
        ("dev", "test_probs_dev_bgclean3_prop.pkl", 1 / 12),
        ("dev", "test_probs_dev_hub_prop.pkl", 1 / 12),
        ("dev", "test_probs_dev_v10_fullbg_lprop.pkl", 1 / 3),
        ("sab", "v17_bgclean_conn", 1 / 4),
        ("sab", "v19_bgclean_v2", 1 / 4),
    ],
    "lnb34": [
        ("dev", "test_probs_dev_bgclean_prop.pkl", 1 / 12),
        ("dev", "test_probs_dev_bgclean2_prop.pkl", 1 / 12),
        ("dev", "test_probs_dev_bgclean3_prop.pkl", 1 / 12),
        ("dev", "test_probs_dev_v10_sw2152_lprop.pkl", 1 / 4),
        ("sab", "v17_bgclean_conn", 1 / 6),
        ("sab", "v19_bgclean_v2", 1 / 6),
        ("sab", "v24_fullbg_sw2152", 1 / 6),
    ],
    "lnb36": [
        ("dev", "test_probs_dev_bg_sw2152_prop.pkl", 1 / 4),
        ("dev", "test_probs_dev_v10_sw2152_lprop.pkl", 1 / 4),
        ("sab", "v17_bgclean_conn", 1 / 10),
        ("sab", "v19_bgclean_v2", 1 / 10),
        ("sab", "v23_fullbg_lda512", 1 / 10),
        ("sab", "v24_fullbg_sw2152", 1 / 10),
        ("sab", "v25_fullbg_sw2152_lda512", 1 / 10),
    ],
    "lnb39": [
        ("dev", "test_probs_dev_bg_swproto_lgb_lprop.pkl", 1 / 8),
        ("dev", "test_probs_dev_bg_swproto_mlp_lprop.pkl", 1 / 8),
        ("dev", "test_probs_dev_v10_swproto_lgb_lprop.pkl", 1 / 8),
        ("dev", "test_probs_dev_v10_swproto_mlp_lprop.pkl", 1 / 8),
        ("sab", "v17_bgclean_conn", 1 / 10),
        ("sab", "v19_bgclean_v2", 1 / 10),
        ("sab", "v24_fullbg_sw2152", 1 / 10),
        ("sab", "v25_fullbg_sw2152_lda512", 1 / 10),
        ("sab", "v26_fullbg_sw2152_proto", 1 / 10),
    ],
}
OUT_NAME = {
    "lnb32": "sub_team_blend_lnb32_lprop67.csv",
    "lnb34": "sub_team_blend_lnb34_sw2152.csv",
    "lnb36": "sub_team_blend_lnb36_sw2152_v25.csv",
    "lnb39": "sub_team_blend_lnb39_swproto_g2q_mlp.csv",
}


def load_sab(sab_dir, slug):
    files = [f for f in sorted(glob.glob(os.path.join(sab_dir, f"*{slug}*.probs.pkl"))) if not f.endswith(".cv.probs.pkl")]
    assert files, f"no test probs for {slug} in {sab_dir}"
    return pickle.load(open(files[-1], "rb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    ap.add_argument("--dev-dir", default=H)
    ap.add_argument("--sab-dir", default=os.path.join(H, "..", "submissions"))
    ap.add_argument("--out-dir", default=H)
    a = ap.parse_args()

    tr, te, _, _ = load()
    allowed = {s: set(tr.identity[tr.species == s]) | {NEW} for s in ["hyena", "leopard"]}
    parts = []
    for src, key, w in RECIPES[a.recipe]:
        p = pickle.load(open(os.path.join(a.dev_dir, key), "rb")) if src == "dev" else load_sab(a.sab_dir, key)
        parts.append((p, w))
    assert abs(sum(w for _, w in parts) - 1.0) < 1e-9

    preds = []
    for q, sp in zip(te.id, te.species):
        sc = {}
        for p, w in parts:
            for k, v in p[q].items():
                sc[k] = sc.get(k, 0.0) + w * v
        ranked = sorted(sc, key=lambda l: (-sc[l], l))
        labs = [l for l in ranked if l in allowed[sp]][:5]
        for l in [NEW] + sorted(allowed[sp]):          # pad to 5 valid same-species labels
            if len(labs) == 5:
                break
            if l not in labs:
                labs.append(l)
        assert len(labs) == 5 and len(set(labs)) == 5 and set(labs) <= allowed[sp], q
        preds.append(" ".join(labs))

    sub = pd.DataFrame({"id": te.id, "predictions": preds})
    assert len(sub) == len(te) and sub.id.is_unique and sub.isna().sum().sum() == 0
    out = os.path.join(a.out_dir, OUT_NAME[a.recipe])
    sub.to_csv(out, index=False)
    top1 = sub.predictions.str.split().str[0].values
    for sp in ["hyena", "leopard"]:
        m = (te.species == sp).values
        print(f"{sp:7s}: new@1={np.mean(top1[m] == NEW):.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
