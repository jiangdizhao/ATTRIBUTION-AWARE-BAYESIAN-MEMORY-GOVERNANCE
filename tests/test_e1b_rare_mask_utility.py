import sys
import unittest
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from abmg_e1b_rare_mask_utility import (
    CategoryCorrectionMemory,
    OnlinePrototypeBank,
    corrected_patch_scores,
    fixed_nested_mask_schedules,
    parse_mask_budgets,
)


class TestE1BRareMaskUtility(unittest.TestCase):
    def test_parse_mask_budgets(self):
        self.assertEqual(
            parse_mask_budgets("0.01,0.0025,0.005,0.005"),
            (0.0025, 0.005, 0.01),
        )
        with self.assertRaises(ValueError):
            parse_mask_budgets("0")
        with self.assertRaises(ValueError):
            parse_mask_budgets("1.1")

    def test_mask_schedules_are_exact_and_nested(self):
        schedules = fixed_nested_mask_schedules(
            1000,
            (0.01, 0.02, 0.05),
            seed=7,
        )
        self.assertEqual(len(schedules[0.01]), 10)
        self.assertEqual(len(schedules[0.02]), 20)
        self.assertEqual(len(schedules[0.05]), 50)
        self.assertTrue(schedules[0.01].issubset(schedules[0.02]))
        self.assertTrue(schedules[0.02].issubset(schedules[0.05]))

    def test_online_prototype_bank_merges_after_capacity(self):
        bank = OnlinePrototypeBank(2, torch.device("cpu"))
        x = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.1],
            ],
            dtype=torch.float32,
        )
        bank.update(x)
        self.assertEqual(bank.size, 2)
        self.assertEqual(bank.n_observations, 3)
        self.assertAlmostEqual(float(bank.counts.sum()), 3.0, places=6)

    def test_suppression_penalizes_negative_like_patch(self):
        memory = CategoryCorrectionMemory.create(2, torch.device("cpu"))
        memory.negative.update(torch.tensor([[1.0, 0.0]]))

        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        baseline = torch.tensor([0.5, 0.4], dtype=torch.float32)
        corrected = corrected_patch_scores(
            baseline,
            features,
            memory,
            "suppress",
        )
        self.assertLess(float(corrected[0] - corrected[1]), float(baseline[0] - baseline[1]))

    def test_bidirectional_promotes_positive_and_suppresses_negative(self):
        memory = CategoryCorrectionMemory.create(2, torch.device("cpu"))
        memory.positive.update(torch.tensor([[1.0, 0.0]]))
        memory.negative.update(torch.tensor([[0.0, 1.0]]))

        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        baseline = torch.tensor([0.45, 0.45], dtype=torch.float32)
        # Give the baseline non-zero spread so the scale-matched correction is active.
        baseline = torch.tensor([0.50, 0.40], dtype=torch.float32)
        corrected = corrected_patch_scores(
            baseline,
            features,
            memory,
            "bidirectional",
        )
        self.assertGreater(float(corrected[0] - corrected[1]), float(baseline[0] - baseline[1]))

    def test_empty_memory_leaves_scores_unchanged(self):
        memory = CategoryCorrectionMemory.create(2, torch.device("cpu"))
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        baseline = torch.tensor([0.6, 0.2], dtype=torch.float32)
        corrected = corrected_patch_scores(
            baseline,
            features,
            memory,
            "bidirectional",
        )
        self.assertTrue(torch.allclose(corrected, baseline))


if __name__ == "__main__":
    unittest.main()
