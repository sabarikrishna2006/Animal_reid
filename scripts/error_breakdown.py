"""Where is MAP@5 lost? Uses the cached second-level output of an experiment slug.
Usage: py scripts/error_breakdown.py SLUG"""
import sys, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from reid.pipeline import decide
from reid import NEW
sl = pickle.loads((ROOT / "data/cache" / f"second_level_{sys.argv[1]}.pkl").read_bytes())
cv, qdf = sl["cv"], sl["qdf"]
pr = decide(cv, qdf, "pnew", sl["beta"], sl["T"], "softmax")
incand = cv.groupby(cv.split.astype(str) + "|" + cv.query_id).y.max()
q = qdf.assign(key=qdf.split.astype(str) + "|" + qdf.query_id)
rows = []
for k, t, sp in zip(q.key, q.truth, q.species):
    p = pr[k]; ap = next((1 / i for i, l in enumerate(p, 1) if l == t), 0.0)
    if t == NEW:
        cat = "new: correct@1" if p[0] == NEW else ("new: in top5 not 1st" if NEW in p else "new: missing")
    else:
        if not incand[k]: cat = "known: not in candidates"
        elif p[0] == t: cat = "known: correct@1"
        elif p[0] == NEW: cat = "known: NEW ranked first"
        else: cat = "known: other identity first"
    rows.append((sp, cat, ap))
D = pd.DataFrame(rows, columns=["sp", "cat", "ap"])
D["loss"] = 1 - D.ap
for sp, g in D.groupby("sp"):
    t = g.groupby("cat").agg(n=("ap", "size"), mean_ap=("ap", "mean"), loss=("loss", "sum"))
    t["share_of_queries"] = t.n / len(g); t["share_of_total_loss"] = t.loss / g.loss.sum()
    print(f"== {sp}: MAP {g.ap.mean():.4f}  (total loss {g.loss.sum():.0f} over {len(g)} queries)")
    print(t.round(3).to_string())
