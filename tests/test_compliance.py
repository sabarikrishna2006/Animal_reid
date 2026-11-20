import sys, unittest, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_compliance import scan


class TestCompliance(unittest.TestCase):
    def test_no_forbidden_imports_or_cv2_calls(self):
        self.assertEqual(scan(ROOT), [])

    def test_every_submission_traces_to_registry(self):
        reg = ROOT / "experiments" / "registry.csv"
        subs = list((ROOT / "submissions").glob("sub_*.csv"))
        if not subs:
            return
        rows = list(csv.DictReader(open(reg)))
        names = {r["change"].split("-> ")[-1] for r in rows if r["change"].startswith("SUBMISSION")}
        for s in subs:
            self.assertIn(s.name, names, f"{s.name} has no registry row")
            exp = ROOT / "experiments" / s.stem[len("sub_"):]
            self.assertTrue((exp / "metrics.json").exists(), exp)


if __name__ == "__main__":
    unittest.main()
