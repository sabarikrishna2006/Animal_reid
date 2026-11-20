import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from reid.metrics import map_at_5
from reid.decision import rank_labels
from reid import NEW


class TestMetric(unittest.TestCase):
    def test_official_worked_example(self):
        truth = {"A": "LEO0317", "B": "HYE0042", "C": NEW, "D": "HYE0011"}
        preds = {"A": ["LEO0317", "LEO0102"], "B": ["HYE0007", "HYE0042", NEW],
                 "C": ["LEO0102", "LEO0009", NEW], "D": [NEW, "HYE0198"]}
        self.assertAlmostEqual(map_at_5(truth, preds), 0.458333, places=4)

    def test_sorting_by_probability_is_optimal(self):
        # expected score of every permutation <= sorted order (exchange argument)
        import itertools
        probs = {"A": .45, NEW: .30, "B": .15, "C": .06, "D": .04}
        exp = lambda order: sum(probs[l] / k for k, l in enumerate(order, 1))
        best = rank_labels(["A", "B", "C", "D"], [.45, .15, .06, .04], .30)
        self.assertEqual(best, ["A", NEW, "B", "C", "D"])
        self.assertAlmostEqual(exp(best), 0.673, places=6)
        for perm in itertools.permutations(probs):
            self.assertLessEqual(exp(perm), exp(best) + 1e-12)


if __name__ == "__main__":
    unittest.main()
