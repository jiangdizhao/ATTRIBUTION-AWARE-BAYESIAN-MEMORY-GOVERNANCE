import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import abmg_stage1_c1_beta_vae as core  # noqa: E402
import abmg_stage1_c1_beta_vae_v2 as v2  # noqa: E402


class DynamicScaleAuditV2Tests(unittest.TestCase):
    def test_beta_schedule_reaches_target_on_epoch_five(self):
        vals = [core.beta_at_epoch(4.0, e, 5) for e in range(6)]
        self.assertEqual(vals[0], 0.0)
        self.assertAlmostEqual(vals[1], 0.8)
        self.assertAlmostEqual(vals[4], 3.2)
        self.assertEqual(vals[5], 4.0)

    def test_balanced_quota_sums_exactly(self):
        cats = [f"c{i}" for i in range(24)]
        q = v2._quota(1024, cats)
        self.assertEqual(sum(q.values()), 1024)
        self.assertLessEqual(max(q.values()) - min(q.values()), 1)

    def test_parse_modes(self):
        self.assertEqual(v2._parse_modes("raw,scalar"), ["raw", "scalar"])
        self.assertEqual(v2._parse_modes("raw,raw"), ["raw"])
        with self.assertRaises(ValueError):
            v2._parse_modes("whiten")

    def test_participation_ratio(self):
        self.assertAlmostEqual(v2._participation_ratio([1.0, 0.0, 0.0]), 1.0)
        self.assertAlmostEqual(v2._participation_ratio([1.0, 1.0, 1.0]), 3.0)
        self.assertEqual(v2._participation_ratio([0.0, 0.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
