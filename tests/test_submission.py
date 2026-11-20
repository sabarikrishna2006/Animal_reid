import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from reid.io import load_config, load_data
from reid.decision import validate


class TestSubmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data(load_config())

    def test_all_written_submissions_valid(self):
        for p in (ROOT / "submissions").glob("sub_*.csv"):
            validate(pd.read_csv(p, dtype=str), self.data.test, self.data.train)

    def test_validator_rejects_cross_species_label(self):
        sub = pd.DataFrame({"id": self.data.test.id.values, "predictions": "new_individual"})
        validate(sub, self.data.test, self.data.train)
        i = (self.data.test.species == "hyena").values.argmax()
        sub.loc[i, "predictions"] = "LEO0017"
        with self.assertRaises(AssertionError):
            validate(sub, self.data.test, self.data.train)


if __name__ == "__main__":
    unittest.main()
