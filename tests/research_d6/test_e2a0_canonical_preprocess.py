from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts/research_d6/p2/e2a"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from canonical_preprocess import build_canonical  # noqa: E402
from feature_interface import (  # noqa: E402
    dinov2_to_shared_interface,
    dinov3_to_shared_interface,
)


class CanonicalPreprocessTests(unittest.TestCase):
    def test_registered_shape_and_dtype(self):
        result = build_canonical()
        self.assertEqual(result["tensor"].shape, (1, 3, 448, 448))
        self.assertEqual(result["tensor"].dtype, np.float32)
        self.assertTrue(np.isfinite(result["tensor"]).all())

    def test_center_letterbox_and_zero_normalized_padding(self):
        result = build_canonical()
        tensor = result["tensor"]
        self.assertTrue(np.all(tensor[:, :, :56, :] == 0.0))
        self.assertTrue(np.all(tensor[:, :, 392:, :] == 0.0))
        self.assertFalse(np.all(tensor[:, :, 56:392, :] == 0.0))

    def test_patch_center_masks(self):
        result = build_canonical()
        self.assertEqual(result["mask_v2"].shape, (1024,))
        self.assertEqual(int(result["mask_v2"].sum()), 768)
        self.assertEqual(result["mask_v3"].shape, (784,))
        self.assertEqual(int(result["mask_v3"].sum()), 588)

    def test_deterministic_synthetic_tensor(self):
        first = build_canonical()
        second = build_canonical()
        self.assertEqual(first["metadata"]["canonical_array_sha256"],
                         second["metadata"]["canonical_array_sha256"])
        self.assertTrue(np.array_equal(first["tensor"], second["tensor"]))

    def test_shared_interface_for_both_backbones(self):
        rng = np.random.default_rng(7)
        mask_v2 = np.ones(1024, dtype=bool)
        mask_v2[-256:] = False
        v2 = dinov2_to_shared_interface({
            "x_norm_clstoken": rng.normal(size=(1, 1024)).astype(np.float32),
            "x_norm_regtokens": rng.normal(size=(1, 4, 1024)).astype(np.float32),
            "x_norm_patchtokens": rng.normal(size=(1, 1024, 1024)).astype(np.float32),
        }, mask_v2)
        v3 = dinov3_to_shared_interface(
            rng.normal(size=(1, 1 + 4 + 784, 1024)).astype(np.float32),
            np.ones(784, dtype=bool),
        )
        self.assertEqual(v2["cls_unit"].shape, (1, 1024))
        self.assertEqual(v2["register_unit"].shape, (1, 4, 1024))
        self.assertEqual(v2["patch_unit"].shape, (1, 1024, 1024))
        self.assertEqual(v3["patch_unit"].shape, (1, 784, 1024))
        self.assertEqual(int(v2["valid_patch_mask"].sum()), 768)
        self.assertAlmostEqual(float(np.linalg.norm(v2["patch_unit"][0, 0])), 1.0, places=6)
        self.assertTrue(np.isfinite(v3["patch_unit"]).all())

    def test_shared_interface_rejects_inconsistent_shape(self):
        with self.assertRaises(ValueError):
            dinov3_to_shared_interface(
                np.ones((1, 1 + 4 + 10, 8), dtype=np.float32),
                np.ones(9, dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
