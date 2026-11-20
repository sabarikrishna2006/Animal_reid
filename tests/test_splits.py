import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from reid.io import load_config, load_data
from reid.splits import build_splits, encounter_clusters, l2n
from reid import NEW, SPECIES


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(); cls.data = load_data(cls.cfg)
        cls.splits = build_splits(cls.data, cls.cfg)

    def test_disjoint_and_new_ids_absent_from_gallery(self):
        for e in self.splits:
            for sp, (fold, truth) in e.items():
                self.assertEqual(len(set(fold.gallery_rows) & set(fold.query_rows)), 0)
                new_rows = fold.query_rows[truth.labels == NEW]
                new_ids = set(self.data.meta.identity.values[new_rows])
                self.assertEqual(len(new_ids & set(fold.gallery_labels)), 0)
                known = truth.labels != NEW
                # every known query's identity keeps >= 1 gallery image
                self.assertTrue(set(truth.labels[known]) <= set(fold.gallery_labels))
                # union = all train images of the species, species never mixed
                tr = self.data.train
                allrows = set(tr.row[tr.species == sp])
                self.assertEqual(set(fold.gallery_rows) | set(fold.query_rows), allrows)
                self.assertTrue((self.data.meta.species.values[fold.query_rows] == sp).all())

    def test_encounters_never_split(self):
        U = l2n(self.data.X.astype(np.float64)); tr = self.data.train
        for sp in SPECIES:
            m = (tr.species == sp).values
            thr = self.cfg["cv"]["enc_thr"]; thr = thr[sp] if isinstance(thr, dict) else thr
            cl = encounter_clusters(U[tr.row.values[m]], tr.identity.values[m], thr)
            rows = tr.row.values[m]
            for e in self.splits[:2]:
                fold, _ = e[sp]
                isq = np.isin(rows, fold.query_rows)
                for c in np.unique(cl):
                    v = isq[cl == c]
                    self.assertTrue(v.all() or (~v).all(), c)


if __name__ == "__main__":
    unittest.main()
