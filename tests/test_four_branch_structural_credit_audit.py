import unittest
from copy import deepcopy

import numpy as np

from scripts.abmg_four_branch_structural_credit_audit import (
    BRANCHES,
    CORE4,
    GLOBAL_CREDIT_WEIGHT,
    apply_branch_update,
    credit_vector,
    expected_changed_factors,
    make_derangement,
    score_weighted_memory,
)
from scripts.abmg_sequential_local_update_audit import init_memory, score_memory


class FourBranchStructuralCreditAuditTests(unittest.TestCase):
    def setUp(self):
        self.X = np.eye(4, dtype=np.float64)
        self.support = {
            label: np.asarray([i], dtype=np.int64)
            for i, label in enumerate(CORE4)
        }
        self.base = init_memory(self.X, self.support)
        self.prior = {
            "mu0": np.zeros(4, dtype=np.float64),
            "var": np.ones(4, dtype=np.float64) * 0.2,
        }
        self.vector = np.asarray([0.8, 0.2, 0.0, 0.0], dtype=np.float64)
        self.shuffle = make_derangement(CORE4, seed=7)

    def test_derangement_is_deterministic_and_has_no_fixed_points(self):
        a = make_derangement(CORE4, seed=7)
        b = make_derangement(CORE4, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(CORE4))
        self.assertEqual(set(a.values()), set(CORE4))
        for src, dst in a.items():
            self.assertNotEqual(src, dst)

    def test_learning_branches_have_matched_total_credit_mass(self):
        self.assertAlmostEqual(GLOBAL_CREDIT_WEIGHT, 1.0 / len(CORE4))
        for branch in BRANCHES:
            credit = credit_vector(branch, "AK", self.shuffle)
            expected = 0.0 if branch == "no_update" else 1.0
            self.assertAlmostEqual(sum(credit.values()), expected)
        self.assertTrue(
            all(
                np.isclose(v, 1.0 / len(CORE4))
                for v in credit_vector("global_broadcast", "AK", self.shuffle).values()
            )
        )

    def test_branch_write_footprints_and_effective_count_mass(self):
        for branch in BRANCHES:
            memory = deepcopy(self.base)
            changed = apply_branch_update(
                memory, branch, "AK", self.vector, self.shuffle
            )
            self.assertEqual(
                changed,
                expected_changed_factors(branch, "AK", self.shuffle),
            )
            total_delta = sum(
                float(memory[g]["n"]) - float(self.base[g]["n"])
                for g in CORE4
            )
            expected_mass = 0.0 if branch == "no_update" else 1.0
            self.assertAlmostEqual(total_delta, expected_mass)

        global_memory = deepcopy(self.base)
        apply_branch_update(
            global_memory, "global_broadcast", "AK", self.vector, self.shuffle
        )
        for label in CORE4:
            self.assertAlmostEqual(
                float(global_memory[label]["n"]),
                float(self.base[label]["n"]) + 0.25,
            )

    def test_correct_and_shuffled_write_to_different_single_addresses(self):
        correct = deepcopy(self.base)
        shuffled = deepcopy(self.base)
        c_changed = apply_branch_update(
            correct, "correct_address", "AK", self.vector, self.shuffle
        )
        s_changed = apply_branch_update(
            shuffled, "shuffled_address", "AK", self.vector, self.shuffle
        )
        self.assertEqual(c_changed, ("AK",))
        self.assertEqual(s_changed, (self.shuffle["AK"],))
        self.assertNotEqual(c_changed, s_changed)

    def test_weighted_scorer_matches_existing_scorer_for_integer_counts(self):
        q = np.asarray(
            [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]],
            dtype=np.float64,
        )
        for model in ("diag_shrunk", "bayes_predictive"):
            pred_old, score_old = score_memory(
                q, self.base, self.prior, 1.0, model
            )
            pred_new, score_new = score_weighted_memory(
                q, self.base, self.prior, 1.0, model
            )
            np.testing.assert_array_equal(pred_old, pred_new)
            np.testing.assert_allclose(score_old, score_new, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
