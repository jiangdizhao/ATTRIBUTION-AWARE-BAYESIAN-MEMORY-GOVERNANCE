import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from abmg_e3_normal_state_audit import (
    _contamination_plan,
    _fixed_normal_stream_split,
    _select_state_support,
    parse_cv_folds,
)


class TestE3NormalStateAudit(unittest.TestCase):
    def test_parse_cv_folds(self):
        self.assertEqual(parse_cv_folds("all", 4), (0, 1, 2, 3))
        self.assertEqual(parse_cv_folds("2,0,2", 4), (0, 2))
        with self.assertRaises(ValueError):
            parse_cv_folds("4", 4)

    def test_state_support_is_deterministic(self):
        x = np.arange(60, dtype=np.float64).reshape(20, 3)
        ids = [f"id-{i}" for i in range(20)]
        a, ai = _select_state_support(x, ids, 4, seed=7)
        b, bi = _select_state_support(x, ids, 4, seed=7)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(ai, bi)
        self.assertEqual(len(set(ai)), 4)

    def test_normal_stream_and_sentinel_are_disjoint(self):
        x = np.arange(180, dtype=np.float64).reshape(60, 3)
        ids = [f"id-{i}" for i in range(60)]
        q, qi, s, si = _fixed_normal_stream_split(
            x,
            ids,
            max_updates=32,
            stream_seed=11,
            min_sentinel=8,
        )
        self.assertEqual(len(q), 32)
        self.assertEqual(len(s), 28)
        self.assertFalse(set(qi) & set(si))
        self.assertEqual(set(qi) | set(si), set(ids))

    def test_contamination_plan_has_exact_count_and_is_deterministic(self):
        a = _contamination_plan(
            32,
            0.10,
            n_defects=50,
            seed=19,
        )
        b = _contamination_plan(
            32,
            0.10,
            n_defects=50,
            seed=19,
        )
        self.assertEqual(a, b)
        self.assertEqual(len(a), round(0.10 * 32))
        self.assertTrue(all(0 <= p < 32 for p in a))
        self.assertTrue(all(0 <= d < 50 for d in a.values()))

    def test_zero_noise_has_no_contamination(self):
        self.assertEqual(
            _contamination_plan(32, 0.0, n_defects=0, seed=1),
            {},
        )


if __name__ == "__main__":
    unittest.main()
