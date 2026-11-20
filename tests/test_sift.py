import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from reid.sift_match import match_features, rootsift, FEATS


class TestSiftMatch(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 300
        self.xa = rng.uniform(0, 500, (self.n, 2)).astype(np.float32)
        self.da = rng.integers(0, 60, (self.n, 128)).astype(np.uint8)
        th, s, t = np.radians(10), 0.9, np.array([20, -15])
        R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        self.xb = (self.xa @ R.T + t + rng.normal(0, 1.0, (self.n, 2))).astype(np.float32)
        self.db = np.clip(self.da.astype(int) + rng.integers(-3, 4, self.da.shape), 0, 255).astype(np.uint8)
        self.rng = rng

    def test_true_transform_gives_many_inliers(self):
        f = match_features(self.xa, rootsift(self.da), self.xb, rootsift(self.db))
        inl = f[FEATS.index("inl_aff")]
        self.assertGreater(inl, 0.8 * self.n)
        self.assertAlmostEqual(np.exp(f[FEATS.index("log_scale")]), 0.9, places=1)
        self.assertLess(abs(f[FEATS.index("abs_rot")] - 10), 2)

    def test_random_pair_gives_few_inliers(self):
        xr = self.rng.uniform(0, 500, (self.n, 2)).astype(np.float32)
        dr = self.rng.integers(0, 60, (self.n, 128)).astype(np.uint8)
        f = match_features(self.xa, rootsift(self.da), xr, rootsift(dr))
        self.assertLess(np.nan_to_num(f[FEATS.index("inl_aff")]), 8)

    def test_empty_keypoints_are_nan_not_zero(self):
        e = np.zeros((0, 2), np.float32)
        f = match_features(e, np.zeros((0, 128), np.float32), self.xb, rootsift(self.db))
        self.assertTrue(np.isnan(f[FEATS.index("inl_aff")]))
        self.assertEqual(f[FEATS.index("kp_min")], 0)


if __name__ == "__main__":
    unittest.main()
