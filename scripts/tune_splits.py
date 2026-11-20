"""Gate 2a: choose CV split parameters so CV queries look like the test set.

Target: the distribution of each query's top-1 cosine to the gallery (raw L2 MiewID). We minimise the
Kolmogorov-Smirnov distance between CV queries and test queries, per species, averaged over 5 seeds.
The chosen f_new_id implies a novelty rate pi_cv; that is our (biased-but-consistent) estimate of pi_test.
"""
import sys, itertools, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from reid.io import load_config, load_data, ROOT
from reid.splits import encounter_clusters, make_split, l2n
from reid import NEW

cfg = load_config(); d = load_data(cfg)
U = l2n(d.X.astype(np.float64)); tr, te = d.train, d.test
rows = []
for sp in ("hyena", "leopard"):
    m = (tr.species == sp).values; R = tr.row.values[m]; lab = tr.identity.values[m]
    Ug = U[R]; S = Ug @ Ug.T
    T = U[te.row.values[(te.species == sp).values]]
    te_top1 = (T @ Ug.T).max(1)
    nte, ntr = len(T), len(R)
    for thr in (0.0, 0.1, 0.2, 0.3, 0.4):
        cl = encounter_clusters(Ug, lab, thr)
        for p_move, f_new in itertools.product((0.1, 0.2, 0.3, 0.4, 0.5), (0.1, 0.2, 0.3, 0.4, 0.5)):
            ks, pis, ratios = [], [], []
            for seed in range(5):
                isq, truth = make_split(lab, cl, seed, p_move, f_new, 40, sp)
                top1 = S[np.ix_(isq, ~isq)].max(1)
                ks.append(ks_2samp(top1, te_top1).statistic)
                pis.append(np.mean(truth[isq] == NEW)); ratios.append(isq.sum() / (~isq).sum())
            rows.append(dict(species=sp, enc_thr=thr, p_move=p_move, f_new_id=f_new, ks=np.mean(ks), ks_sd=np.std(ks),
                             pi_cv=np.mean(pis), q_over_g=np.mean(ratios)))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "reports" / "split_tuning.csv", index=False)
print("test Q/G ratio: hyena %.2f leopard %.2f" % (1217 / 1912, 1574 / 5251))
for sp, g in df.groupby("species"):
    print(f"\n== {sp}: best 12 by KS")
    print(g.sort_values("ks").head(12).round(3).to_string(index=False))
    print(f"   pi_cv among top-10 configs: {g.sort_values('ks').head(10).pi_cv.mean():.3f} "
          f"(range {g.sort_values('ks').head(10).pi_cv.min():.2f}-{g.sort_values('ks').head(10).pi_cv.max():.2f})")
for sp, g in df.groupby("species"):
    gg = g[g.q_over_g <= 1.0]
    print(f"\n== {sp}: best by KS with Q/G <= 1.0")
    print(gg.sort_values("ks").head(8).round(3).to_string(index=False))
    print("best KS per enc_thr:", gg.groupby("enc_thr").ks.min().round(3).to_dict())
