"""Unit and leakage-invariance tests for v25 metric/rescue features, HNSW kpgraph, and Paradigm 2 dual_model."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from reid import retrieval, kpgraph, dual_model, sift_cache, sift_match, transductive


class TestV25AndDual(unittest.TestCase):
    def test_fit_sw_and_fullda_norm_and_leakage_free(self):
        rng = np.random.RandomState(42)
        ng, nq, d = 120, 20, 64
        # Build separable multi-class gallery in 64d (actually uses 64d as mock MiewID)
        labels = np.repeat([f"ID{i:02d}" for i in range(20)], 6)
        gside = np.array(["L" if i % 2 == 0 else "R" for i in range(ng)])
        centers = rng.randn(20, d).astype(np.float32)
        Zg = retrieval.l2n(centers[np.repeat(np.arange(20), 6)] + 0.15 * rng.randn(ng, d).astype(np.float32))
        Zq1 = retrieval.l2n(rng.randn(nq, d).astype(np.float32))
        Zq2 = Zq1.copy()
        Zq2[5:] += 0.5  # change other queries; query 0 projection must be invariant

        p1 = retrieval.fit_sw_and_fullda(Zg, labels, reg=0.5, d_lda=32, gside=gside)
        p2 = retrieval.fit_sw_and_fullda(Zg, labels, reg=0.5, d_lda=32, gside=gside)
        for k in ("cl2", "sw", "ssw", "fl512"):
            E1 = p1[k](Zq1)
            E2 = p2[k](Zq2)
            norms = np.linalg.norm(E1, axis=1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))
            # Query 0 projection depends strictly on Zg, never on other query rows
            self.assertTrue(np.allclose(E1[:5], E2[:5], atol=1e-5))

    def test_vp_compat_stats(self):
        rng = np.random.RandomState(7)
        Zg = retrieval.l2n(rng.randn(40, 16).astype(np.float32))
        labels = np.repeat(["A", "B"], 20)
        vps = np.array(["left"] * 20 + ["right"] * 20)
        stats = retrieval.vp_compat_stats(Zg, labels, vps, min_n=5)
        self.assertIn(("left", "left"), stats)
        self.assertIn(("right", "right"), stats)
        self.assertGreaterEqual(stats[("left", "left")][1], 0.05)

    def test_dual_oof_ensemble_and_lambdarank(self):
        rng = np.random.RandomState(123)
        n_q = 40
        n_c = 6
        rows = []
        for qi in range(n_q):
            grp = f"ID{qi // 2:02d}"
            sp_ = qi % 2
            for ci in range(n_c):
                y = int(ci == 0 and qi < 30)
                f1 = rng.randn() + (2.0 if y else 0.0)
                f2 = rng.randn() + (1.0 if y else -0.5)
                rows.append((sp_, f"q_{qi:03d}", f"{sp_}|q_{qi:03d}", f"c_{ci}", grp, y, f1, f2))
        df = pd.DataFrame(rows, columns=["split", "query_id", "_qkey", "cand_id", "group", "y", "f1", "f2"])
        oof, mods = dual_model._oof_ensemble(df, ["f1", "f2"], "y", df["group"].values,
                                             qkeys_col="_qkey", seeds=(2026,), use_rank=True)
        self.assertEqual(len(oof["mix"]), len(df))
        self.assertTrue(np.isfinite(oof["mix"]).all())
        self.assertTrue(((oof["mix"] > 0) & (oof["mix"] < 1)).all())
        pred = dual_model._predict_ensemble(mods, df[["f1", "f2"]].to_numpy(np.float32),
                                            qk=df["_qkey"].values, use_rank=True)
        self.assertTrue(np.isfinite(pred["mix"]).all())

        qdf = df.drop_duplicates("_qkey").copy()
        qdf["species"] = ["hyena" if i % 2 == 0 else "leopard" for i in range(len(qdf))]
        qdf["is_new"] = [int(i >= 15) for i in range(len(qdf))]
        pn_oof, pn_mods = dual_model._oof_pnew_per_species(qdf, ["f1", "f2"], seeds=(2026,))
        self.assertEqual(len(pn_oof), len(qdf))
        self.assertTrue(((pn_oof > 0) & (pn_oof < 1)).all())
        pn_te = dual_model._predict_pnew_per_species(pn_mods, qdf, ["f1", "f2"])
        self.assertEqual(len(pn_te), len(qdf))
        self.assertTrue(np.isfinite(pn_te).all())

    def test_pair_cache_compact_merge(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = {"paths": {"cache": td}, "sift": {"ratio": 0.8, "ransac_frac": 0.02, "workers": 1}}
            pc1 = sift_cache.PairCache(cfg)
            f1 = np.arange(sift_match.NF, dtype=np.float32)
            f2 = f1 + 10.0
            pc1._pending[sift_cache._pack_pair(3, 1)] = f1
            pc1.save()
            pc2 = sift_cache.PairCache(cfg)
            self.assertTrue(np.allclose(pc2.get(1, 3), f1))
            self.assertTrue(np.allclose(pc2.get(3, 1), f1))
            self.assertIsNone(pc2.get(1, 2))
            pc2._pending[sift_cache._pack_pair(2, 5)] = f2
            pc2.save()
            # pc1 misses (2,5) in RAM until _merge_disk / save merges disk without wiping (2,5)
            pc1._pending[sift_cache._pack_pair(7, 8)] = f1 + 20.0
            pc1.save()
            pc3 = sift_cache.PairCache(cfg)
            self.assertTrue(np.allclose(pc3.get(1, 3), f1))
            self.assertTrue(np.allclose(pc3.get(2, 5), f2))
            self.assertTrue(np.allclose(pc3.get(7, 8), f1 + 20.0))

    def test_transductive_feature_names_unique(self):
        pf = (transductive.PAIR_FEATS + transductive.KP_FEATS + transductive.RICH_FEATS
              + transductive.SIFT_BG_FEATS + transductive.SIFT_BG_V2_FEATS + transductive.SW_FEATS)
        self.assertEqual(len(pf), len(set(pf)))


if __name__ == "__main__":
    unittest.main()
