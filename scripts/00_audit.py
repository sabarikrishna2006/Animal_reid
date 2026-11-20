"""Gate 1: integrity audit + pack SIFT into data/cache. Detailed EDA lives in eda/ and reports/eda.md."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from reid.io import load_config, load_data, build_sift_pack, Sift

cfg = load_config()
t0 = time.time()
d = load_data(cfg)
print(f"loaded meta {d.meta.shape}, X {d.X.shape} in {time.time()-t0:.1f}s")
build_sift_pack(cfg, d)
s = Sift(cfg)
assert len(s.off) == len(d.meta) + 1
n = s.n_kp()
for sp in ("hyena", "leopard"):
    for split in ("train", "test"):
        m = ((d.meta.species == sp) & (d.meta.split == split)).values
        print(f"{sp:8s} {split:5s} images={m.sum():5d} kp median={np.median(n[m]):.0f} zero={np.mean(n[m]==0):.3f} <50={np.mean(n[m]<50):.3f}")
xy, de = s[0]
print("row0", d.meta.id[0], xy.shape, de.dtype, f"total {time.time()-t0:.0f}s")
print("AUDIT OK")
