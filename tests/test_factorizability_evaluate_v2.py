import unittest
from collections import Counter

import numpy as np

from scripts.abmg_factorizability_evaluate_v2 import (
    CORE4,
    _eligible_labels,
    within_category_label_shuffle,
)


class FactorizabilityEvaluateV2Tests(unittest.TestCase):
    def test_within_category_shuffle_preserves_each_category_histogram(self):
        labels = np.array(["AK", "AK", "HS", "QS", "AK", "ZW", "ZW", "HS"], dtype=object)
        cats = np.array(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
        out1 = within_category_label_shuffle(labels, cats, seed=7)
        out2 = within_category_label_shuffle(labels, cats, seed=7)
        self.assertTrue(np.array_equal(out1, out2))
        for cat in sorted(set(cats.tolist())):
            idx = np.flatnonzero(cats == cat)
            self.assertEqual(Counter(labels[idx].tolist()), Counter(out1[idx].tolist()))

    def test_core4_scope_is_fixed_not_fold_adaptive(self):
        labels = np.array(
            ["AK"] * 4 + ["HS"] * 4 + ["QS"] * 4 + ["ZW"] * 4 + ["PS"] * 4,
            dtype=object,
        )
        train0 = np.array([True] * 10 + [False] * 10)
        test0 = ~train0
        # With min count 1, at least one core label is absent on one side in this
        # constructed split, so v2 must invalidate the fixed core4 task rather
        # than silently shrinking it.
        selected = _eligible_labels(
            "core4",
            labels,
            train0,
            test0,
            min_train_per_label=1,
            min_test_per_label=1,
        )
        self.assertEqual(selected, [])

    def test_core4_constant(self):
        self.assertEqual(tuple(CORE4), ("AK", "HS", "QS", "ZW"))


if __name__ == "__main__":
    unittest.main()
