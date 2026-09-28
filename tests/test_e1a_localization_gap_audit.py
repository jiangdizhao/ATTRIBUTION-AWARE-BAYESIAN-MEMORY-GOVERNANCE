import sys
import unittest
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from abmg_e1a_localization_gap_audit import (
    localization_metrics,
    summarise_spatial_rows,
)


class TestE1ALocalizationGapAudit(unittest.TestCase):
    def test_hit_precision_recall_and_first_hit_rank(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        mask = torch.tensor([False, False, True, True, False])
        row, idx = localization_metrics(scores, mask, k=3)
        self.assertEqual(idx.tolist(), [0, 1, 2])
        self.assertTrue(row["sensor_hit"])
        self.assertAlmostEqual(row["sensor_precision"], 1.0 / 3.0)
        self.assertAlmostEqual(row["sensor_recall"], 0.5)
        self.assertEqual(row["first_hit_rank"], 3)
        self.assertFalse(row["geometry_failure"])

    def test_miss_still_reports_first_hit_rank(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        mask = torch.tensor([False, False, False, True, False])
        row, _ = localization_metrics(scores, mask, k=3)
        self.assertFalse(row["sensor_hit"])
        self.assertEqual(row["first_hit_rank"], 4)
        self.assertEqual(row["sensor_precision"], 0.0)
        self.assertEqual(row["sensor_recall"], 0.0)

    def test_empty_transformed_mask_is_geometry_failure(self):
        scores = torch.tensor([0.9, 0.8, 0.7])
        mask = torch.tensor([False, False, False])
        row, _ = localization_metrics(scores, mask, k=2)
        self.assertTrue(row["geometry_failure"])
        self.assertIsNone(row["sensor_hit"])
        self.assertIsNone(row["first_hit_rank"])

    def test_summary_excludes_geometry_failure_from_sensor_rates(self):
        rows = [
            {
                "mask_available": True,
                "geometry_failure": False,
                "sensor_hit": True,
                "sensor_precision": 0.5,
                "sensor_recall": 1.0,
                "first_hit_rank": 1,
            },
            {
                "mask_available": True,
                "geometry_failure": False,
                "sensor_hit": False,
                "sensor_precision": 0.0,
                "sensor_recall": 0.0,
                "first_hit_rank": 10,
            },
            {
                "mask_available": True,
                "geometry_failure": True,
                "sensor_hit": None,
                "sensor_precision": None,
                "sensor_recall": None,
                "first_hit_rank": None,
            },
        ]
        s = summarise_spatial_rows(rows)
        self.assertEqual(s["n_geometry_failures"], 1)
        self.assertAlmostEqual(s["sensor_hit_at_8"], 0.5)
        self.assertAlmostEqual(s["mean_sensor_precision_at_8"], 0.25)
        self.assertAlmostEqual(s["mean_sensor_recall_at_8"], 0.5)
        self.assertAlmostEqual(s["median_first_hit_rank"], 5.5)


if __name__ == "__main__":
    unittest.main()
