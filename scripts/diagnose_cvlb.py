"""Which CV split settings reproduce the leaderboard? Simulate the EXACT submitted baseline (raw cosine max,
tau h=0.49 / l=0.52) and the all-new submission under many split settings."""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from reid.io import load_config, load_data, ROOT
from reid.splits import encounter_clusters, make_split, l2n
from reid.retrieval import IdentityIndex, baseline_predict
from reid.metrics import map_at_5
from reid import NEW

W = {"hyena": 1217 / 2791, "leopard": 1574 / 2791}
TAU = {"hyena": 0.49, "leopard": 0.52}
cfg = load_config(); d = load_data(cfg)
U = l2n(d.X.astype(np.float64)); tr, te = d.train, d.test
rows = []
for sp in ("hyena", "leopard"):
    m = (tr.species == sp).values; R = tr.row.values[m]; lab = tr.identity.values[m]; ids = tr.id.values[m]
    Ug = U[R]; S = Ug @ Ug.T
    te_top1 = (U[te.row.values[(te.species == sp).values]] @ Ug.T).max(1)
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        cl = encounter_clusters(Ug, lab, thr)
        for p_move, f_new, cap in itertools.product((0.2, 0.35, 0.5), (0.1, 0.2, 0.3, 0.4), (40, 10**6)):
            res = []
            for seed in range(3):
                isq, truth = make_split(lab, cl, seed, p_move, f_new, cap, sp)
                g = ~isq
                idx = IdentityIndex(lab[g]); M = idx.max(S[np.ix_(isq, g)])
                preds = baseline_predict(M, idx.ids, TAU[sp])
                tq = truth[isq]
                mp = map_at_5(dict(zip(ids[isq], tq)), dict(zip(ids[isq], preds)))
                known = tq != NEW
                mk = map_at_5(dict(zip(ids[isq][known], tq[known])), {i: p for i, p, k in zip(ids[isq], preds, known) if k})
                res.append((mp, mk, np.mean(tq == NEW), ks_2samp(S[np.ix_(isq, g)].max(1), te_top1).statistic))
            r = np.mean(res, 0)
            rows.append(dict(species=sp, enc_thr=thr, p_move=p_move, f_new=f_new, cap=cap, base_map=r[0], known_map=r[1], pi=r[2], ks=r[3]))
df = pd.DataFrame(rows); df.to_csv(ROOT / "reports" / "cvlb_diagnosis.csv", index=False)
# combine species with test weights for configs sharing (enc_thr, p_move, cap) per species independently -> show per species
for sp, g in df.groupby("species"):
    print(f"\n== {sp}: configs ordered by KS (baseline map, known-only map, pi)")
    print(g.sort_values("ks").head(12).round(3).to_string(index=False))
    print(f"   -- configs with pi in [0.2,0.4] and lowest baseline map:")
    print(g[(g.pi > 0.2) & (g.pi < 0.4)].sort_values("base_map").head(6).round(3).to_string(index=False))
