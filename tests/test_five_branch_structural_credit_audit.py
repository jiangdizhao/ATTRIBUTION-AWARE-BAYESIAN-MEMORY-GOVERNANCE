import unittest
from copy import deepcopy

import numpy as np

from scripts import abmg_four_branch_structural_credit_audit as four
from scripts.abmg_five_branch_structural_credit_audit import (
    BRANCHES,
    CORE4,
    FULL_BROADCAST_CREDIT_WEIGHT,
    MASS_MATCHED_CREDIT_WEIGHT,
    _five_branch_hooks,
    apply_branch_update,
    credit_vector,
    expected_changed_factors,
)
from scripts.abmg_sequential_local_update_audit import init_memory


class FiveBranchStructuralCreditAuditTests(unittest.TestCase):
    def setUp(self):
        self.X = np.eye(4, dtype=np.float64)
        self.support = {
            label: np.asarray([i], dtype=np.int64)
            for i, label in enumerate(CORE4)
        }
        self.base = init_memory(self.X, self.support)
        self.vector = np.asarray([0.8, 0.2, 0.0, 0.0], dtype=np.float64)
        self.shuffle = four.make_derangement(CORE4, seed=7)

    def test_five_branch_names_are_explicit(self):
        self.assertEqual(
            BRANCHES,
            (
                "no_update",
                "global_broadcast",
                "full_broadcast",
                "correct_address",
                "shuffled_address",
            ),
        )

    def test_credit_mass_distinguishes_two_broadcast_controls(self):
        masses = {
            branch: sum(credit_vector(branch, "AK", self.shuffle).values())
            for branch in BRANCHES
        }
        self.assertAlmostEqual(MASS_MATCHED_CREDIT_WEIGHT, 1.0 / len(CORE4))
        self.assertAlmostEqual(FULL_BROADCAST_CREDIT_WEIGHT, 1.0)
        self.assertAlmostEqual(masses["no_update"], 0.0)
        self.assertAlmostEqual(masses["global_broadcast"], 1.0)
        self.assertAlmostEqual(masses["full_broadcast"], float(len(CORE4)))
        self.assertAlmostEqual(masses["correct_address"], 1.0)
        self.assertAlmostEqual(masses["shuffled_address"], 1.0)

    def test_full_and_mass_matched_broadcast_touch_all_addresses_at_different_strength(self):
        mass_matched = deepcopy(self.base)
        full = deepcopy(self.base)
        changed_mass = apply_branch_update(
            mass_matched, "global_broadcast", "AK", self.vector, self.shuffle
        )
        changed_full = apply_branch_update(
            full, "full_broadcast", "AK", self.vector, self.shuffle
        )
        self.assertEqual(changed_mass, tuple(CORE4))
        self.assertEqual(changed_full, tuple(CORE4))
        for label in CORE4:
            self.assertAlmostEqual(
                float(mass_matched[label]["n"]),
                float(self.base[label]["n"]) + 0.25,
            )
            self.assertAlmostEqual(
                float(full[label]["n"]),
                float(self.base[label]["n"]) + 1.0,
            )

    def test_write_footprints_match_predeclared_rules(self):
        for branch in BRANCHES:
            memory = deepcopy(self.base)
            changed = apply_branch_update(
                memory, branch, "AK", self.vector, self.shuffle
            )
            self.assertEqual(
                changed,
                expected_changed_factors(branch, "AK", self.shuffle),
            )
        self.assertEqual(
            expected_changed_factors("correct_address", "AK", self.shuffle),
            ("AK",),
        )
        self.assertEqual(
            expected_changed_factors("shuffled_address", "AK", self.shuffle),
            (self.shuffle["AK"],),
        )

    def test_hooks_are_temporary_and_do_not_mutate_original_four_branch_module(self):
        original_branches = four.BRANCHES
        original_credit = four.credit_vector
        with _five_branch_hooks():
            self.assertEqual(four.BRANCHES, BRANCHES)
            self.assertIs(four.credit_vector, credit_vector)
            self.assertEqual(
                sum(four.credit_vector("full_broadcast", "AK", self.shuffle).values()),
                float(len(CORE4)),
            )
        self.assertEqual(four.BRANCHES, original_branches)
        self.assertIs(four.credit_vector, original_credit)


if __name__ == "__main__":
    unittest.main()
