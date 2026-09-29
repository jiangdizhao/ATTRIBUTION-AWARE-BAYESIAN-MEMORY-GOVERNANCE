import sys
import unittest
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from abmg_e1b_finalize import (
    deterministic_normal_split,
    persistence_bucket,
    topk_mean,
)
from abmg_stage0_foundation import Stage0Record


class TestE1BFinalize(unittest.TestCase):
    def test_persistence_buckets(self):
        self.assertEqual(persistence_bucket(1), "1-10")
        self.assertEqual(persistence_bucket(10), "1-10")
        self.assertEqual(persistence_bucket(11), "11-25")
        self.assertEqual(persistence_bucket(25), "11-25")
        self.assertEqual(persistence_bucket(26), "26-50")
        self.assertEqual(persistence_bucket(50), "26-50")
        self.assertEqual(persistence_bucket(51), "51+")
        with self.assertRaises(ValueError):
            persistence_bucket(0)

    def test_normal_split_is_deterministic_disjoint_and_complete(self):
        rows = [
            Stage0Record(
                image_id=f"id-{i}",
                category="cat",
                split="test",
                image_path=f"/tmp/{i}.png",
                relative_path=f"{i}.png",
                is_good=True,
                defect_source="OK",
                mask_path=None,
                json_file="cat.json",
            )
            for i in range(10)
        ]
        a1, b1 = deterministic_normal_split(rows, 7)
        a2, b2 = deterministic_normal_split(rows, 7)
        self.assertEqual([x.image_id for x in a1], [x.image_id for x in a2])
        self.assertEqual([x.image_id for x in b1], [x.image_id for x in b2])
        self.assertFalse(set(x.image_id for x in a1) & set(x.image_id for x in b1))
        self.assertEqual(
            set(x.image_id for x in a1 + b1),
            set(x.image_id for x in rows),
        )

    def test_normal_split_skips_tiny_sets(self):
        rows = [
            Stage0Record(
                image_id=f"id-{i}",
                category="cat",
                split="test",
                image_path=f"/tmp/{i}.png",
                relative_path=f"{i}.png",
                is_good=True,
                defect_source="OK",
                mask_path=None,
                json_file="cat.json",
            )
            for i in range(3)
        ]
        a, b = deterministic_normal_split(rows, 7)
        self.assertEqual(a, [])
        self.assertEqual(b, [])

    def test_topk_mean(self):
        scores = torch.tensor([0.1, 0.4, 0.3, 0.2])
        self.assertAlmostEqual(topk_mean(scores, 2), 0.35, places=6)


if __name__ == "__main__":
    unittest.main()
