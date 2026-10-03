import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from abmg_stage0_foundation import Stage0Record
from abmg_e2_address_sidecar_benchmark import (
    _routing_signature,
    _signature_stability,
    _split_normal_train_images,
    parse_cv_folds,
)


def rec(cat: str, idx: int) -> Stage0Record:
    return Stage0Record(
        image_id=f"{cat}-{idx}",
        category=cat,
        split="train",
        image_path=f"/tmp/{cat}-{idx}.png",
        relative_path=f"{cat}/{idx}.png",
        is_good=True,
        defect_source="OK",
        mask_path=None,
        json_file=f"{cat}.json",
    )


class TestE2AddressBenchmark(unittest.TestCase):
    def test_parse_cv_folds(self):
        self.assertEqual(parse_cv_folds("all", 4), (0, 1, 2, 3))
        self.assertEqual(parse_cv_folds("3,1,1", 4), (1, 3))
        with self.assertRaises(ValueError):
            parse_cv_folds("4", 4)

    def test_normal_split_is_category_local_and_disjoint(self):
        rows = [rec("a", i) for i in range(10)] + [
            rec("b", i) for i in range(10)
        ]
        train, val = _split_normal_train_images(
            rows,
            ["a", "b"],
            val_fraction=0.2,
            max_train_images_per_category=4,
            max_val_images_per_category=2,
            seed=7,
        )
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 4)
        self.assertFalse(
            set(x.image_id for x in train) & set(x.image_id for x in val)
        )
        self.assertEqual(
            {x.category for x in train},
            {"a", "b"},
        )
        self.assertEqual(
            {x.category for x in val},
            {"a", "b"},
        )

    def test_routing_signature_is_row_scale_invariant(self):
        scores = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, 0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        a = _routing_signature(scores)
        b = _routing_signature(5.0 * scores + 17.0)
        self.assertTrue(np.allclose(a, b, atol=1e-7))

    def test_signature_stability_positive_for_separated_sources(self):
        y = np.asarray(
            ["AK", "AK", "HS", "HS", "QS", "QS", "ZW", "ZW"],
            dtype=object,
        )
        scores = np.asarray(
            [
                [5, 0, 0, 0],
                [4, 0, 0, 0],
                [0, 5, 0, 0],
                [0, 4, 0, 0],
                [0, 0, 5, 0],
                [0, 0, 4, 0],
                [0, 0, 0, 5],
                [0, 0, 0, 4],
            ],
            dtype=np.float64,
        )
        out = _signature_stability(y, scores)
        self.assertGreater(out["source_signature_gap"], 0.5)


if __name__ == "__main__":
    unittest.main()
