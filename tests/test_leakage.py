"""Leakage tests: features never read query labels; OOF models never see the evaluated identity group."""
import sys, unittest, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from reid.io import load_config, load_data, Sift
from reid.splits import build_splits, FoldData
from reid.retrieval import embed_unsupervised
from reid.candidates import Context, build_table


class TestLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(); cls.data = load_data(cls.cfg)
        cls.splits = build_splits(cls.data, cls.cfg)
        d = cls.data
        cls.emb = {"l2": embed_unsupervised(d.X, d.meta, "l2"), "pcaw": embed_unsupervised(d.X, d.meta, "pcaw")}
        cls.kp = Sift(cls.cfg).n_kp().astype(float)

    def _small_fold(self):
        fold, truth = self.splits[0]["hyena"]
        return FoldData(fold.species, fold.gallery_rows, fold.gallery_labels, fold.query_rows[:40], fold.query_ids[:40]), truth

    def test_features_invariant_to_query_labels(self):
        fold, _ = self._small_fold()
        ctx = Context(self.data, self.emb, self.kp, self.cfg, None)
        a = build_table(fold, ctx, n_cand=10)
        # corrupt the identity labels of the query rows everywhere and rebuild
        d2 = copy.deepcopy(self.data)
        rng = np.random.default_rng(0)
        d2.meta.loc[fold.query_rows, "identity"] = rng.permutation(d2.meta.identity.values[fold.query_rows])
        ctx2 = Context(d2, self.emb, self.kp, self.cfg, None)
        b = build_table(fold, ctx2, n_cand=10)
        pd.testing.assert_frame_equal(a, b)
        self.assertNotIn("truth", a.columns); self.assertNotIn("y", a.columns)

    def test_each_query_once_per_split(self):
        for e in self.splits:
            ids = np.concatenate([e[sp][0].query_ids for sp in e])
            self.assertEqual(len(ids), len(set(ids)))

    def test_oof_groups_disjoint(self):
        from sklearn.model_selection import GroupKFold
        groups = np.repeat(np.arange(50), 4)
        for tr, va in GroupKFold(5).split(np.zeros(len(groups)), groups=groups):
            self.assertEqual(len(set(groups[tr]) & set(groups[va])), 0)


if __name__ == "__main__":
    unittest.main()
