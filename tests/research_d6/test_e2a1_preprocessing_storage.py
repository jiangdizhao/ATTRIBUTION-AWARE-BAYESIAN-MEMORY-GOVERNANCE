from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "scripts/research_d6/p2/e2a1"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from audit_inputs import audit_registered_inputs  # noqa: E402
from preprocess import decode_and_preprocess, image_to_canonical  # noqa: E402
from storage import write_bundle_atomic  # noqa: E402


class E2A1PreprocessTests(unittest.TestCase):
    @staticmethod
    def rgb_image(width: int, height: int) -> Image.Image:
        y, x = np.mgrid[0:height, 0:width]
        arr = np.stack(((x * 13) % 256, (y * 19) % 256, ((x + y) * 7) % 256), axis=-1).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_landscape_round_half_up_and_mask_geometry(self):
        tensor, metadata = image_to_canonical(self.rgb_image(9, 6))
        self.assertEqual(tensor.shape, (1, 3, 448, 448))
        self.assertEqual(metadata["resized_width"], 448)
        self.assertEqual(metadata["resized_height"], 299)
        self.assertEqual(metadata["content_rect_xyxy"], [0, 74, 448, 373])
        self.assertEqual(metadata["padding_ltrb"], [0, 74, 0, 75])
        self.assertEqual(metadata["mask_dinov2"].shape, (1024,))
        self.assertEqual(metadata["mask_dinov3"].shape, (784,))

    def test_portrait_and_square_geometry(self):
        _, portrait = image_to_canonical(self.rgb_image(6, 9))
        self.assertEqual(portrait["resized_width"], 299)
        self.assertEqual(portrait["resized_height"], 448)
        self.assertEqual(portrait["content_rect_xyxy"], [74, 0, 373, 448])
        _, square = image_to_canonical(self.rgb_image(7, 7))
        self.assertEqual(square["content_rect_xyxy"], [0, 0, 448, 448])
        self.assertTrue(square["mask_dinov2"].all())
        self.assertTrue(square["mask_dinov3"].all())

    def test_odd_padding_extra_pixel_goes_right_or_bottom(self):
        _, horizontal_pad = image_to_canonical(self.rgb_image(5, 2))
        self.assertEqual(horizontal_pad["resized_height"], 179)
        self.assertEqual(horizontal_pad["padding_ltrb"], [0, 134, 0, 135])
        _, vertical_pad = image_to_canonical(self.rgb_image(2, 5))
        self.assertEqual(vertical_pad["resized_width"], 179)
        self.assertEqual(vertical_pad["padding_ltrb"], [134, 0, 135, 0])

    def test_extreme_aspect_ratio_with_no_patch_centers_blocks(self):
        with self.assertRaisesRegex(ValueError, "No valid patch centers"):
            image_to_canonical(self.rgb_image(10000, 20))

    def test_color_mode_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "only RGB is registered"):
            image_to_canonical(Image.new("L", (12, 8), 127))

    def test_exif_orientation_is_applied_before_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oriented.jpg"
            image = self.rgb_image(9, 5)
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, format="JPEG", exif=exif)
            _, metadata = decode_and_preprocess(path)
        self.assertEqual(metadata["exif_orientation_tag"], 6)
        self.assertEqual((metadata["source_width_before_exif"], metadata["source_height_before_exif"]), (9, 5))
        self.assertEqual((metadata["source_width_after_exif"], metadata["source_height_after_exif"]), (5, 9))

    def test_preprocessing_is_deterministic(self):
        image = self.rgb_image(63, 42)
        first, meta1 = image_to_canonical(image)
        second, meta2 = image_to_canonical(image)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(meta1["mask_dinov2"], meta2["mask_dinov2"])
        np.testing.assert_array_equal(meta1["mask_dinov3"], meta2["mask_dinov3"])
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.all(first[:, :, :meta1["content_rect_xyxy"][1], :] == 0.0))


class E2A1StorageTests(unittest.TestCase):
    def test_atomic_bundle_roundtrip_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "I-test"
            arrays = {"features": np.arange(12, dtype=np.float32).reshape(3, 4),
                      "mask": np.array([True, False, True])}
            identity = {"model": "synthetic"}
            metadata = {"schema": "test"}
            _, reused_first = write_bundle_atomic(target, arrays, identity, metadata)
            record, reused_second = write_bundle_atomic(target, arrays, identity, metadata)
            self.assertFalse(reused_first)
            self.assertTrue(reused_second)
            self.assertEqual(record["arrays"]["features"]["shape"], [3, 4])
            np.testing.assert_array_equal(np.load(target / "features.npy", allow_pickle=False), arrays["features"])

    def test_atomic_bundle_blocks_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "I-test"
            arrays = {"features": np.ones((2, 2), dtype=np.float32)}
            write_bundle_atomic(target, arrays, {"model": "A"}, {"schema": "test"})
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                write_bundle_atomic(target, arrays, {"model": "B"}, {"schema": "test"})

    def test_atomic_bundle_blocks_corrupt_existing_content(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "I-test"
            arrays = {"features": np.ones((2, 2), dtype=np.float32)}
            write_bundle_atomic(target, arrays, {"model": "A"}, {"schema": "test"})
            (target / "features.npy").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "file SHA mismatch"):
                write_bundle_atomic(target, arrays, {"model": "A"}, {"schema": "test"})


class E2A1WhitelistTests(unittest.TestCase):
    def test_registered_input_whitelist_and_source_hashes(self):
        rows, summary = audit_registered_inputs()
        self.assertEqual(len(rows), 102)
        self.assertEqual(summary["role_counts"], {"D-fit": 54, "D-dev": 48})
        self.assertTrue(summary["all_source_hashes_verified"])
        self.assertTrue(summary["all_paths_under_registered_roots"])
        self.assertFalse(summary["labels_read"])


if __name__ == "__main__":
    unittest.main()
