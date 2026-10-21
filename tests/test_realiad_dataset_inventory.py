"""Isolated synthetic metadata tests; never access user's Real-IAD dataset."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "abmg_realiad_dataset_inventory.py"
spec = importlib.util.spec_from_file_location("realiad_inventory", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RealIADInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "realiad"
        self.json_dir = self.root / "jsons" / "realiad_jsons"
        self.json_dir.mkdir(parents=True)
        (self.json_dir / "metal.json").write_text(json.dumps({
            "meta": {"normal_class": "OK", "prefix": "metal/"},
            "train": [
                {"category": "metal", "anomaly_class": "OK", "image_path": "train/good1.png"},
                {"category": "metal", "anomaly_class": "OK", "image_path": "train/good2.png"},
                {"category": "metal", "anomaly_class": "scratch", "image_path": "train/scratch.png"},
            ],
            "test": [
                {"category": "metal", "anomaly_class": "OK", "image_path": "test/good.png"},
                {"category": "metal", "anomaly_class": "scratch", "image_path": "test/scratch1.png", "mask_path": "mask/s1.png"},
                {"category": "metal", "defect_type": "scratch", "image_path": "test/scratch2.png"},
                {"category": "metal", "anomaly_class": "stain", "image_path": "test/stain.png"},
                {"category": "metal", "image_path": "test/unknownlabel.png"},
            ],
        }), encoding="utf-8")
        (self.json_dir / "plastic.json").write_text(json.dumps({
            "meta": {"normal_class": "good"},
            "train": [{"anomaly_class": "good", "image_path": "p_train.png"}],
            "test": [{"anomaly_class": "crack", "image_path": "p_crack.png", "mask_path": "mask/c.png"}],
        }), encoding="utf-8")

    def test_exact_totals_types_products_and_splits(self):
        report = module.inventory(self.root, self.json_dir)
        o = report["overall"]
        self.assertEqual(report["n_products"], 2)
        self.assertEqual(report["n_distinct_defect_types"], 3)
        self.assertEqual((o["train_normal"], o["train_defect"], o["test_normal"], o["test_defect"]), (3, 1, 2, 4))
        self.assertEqual((o["normal"], o["defect"], o["total"]), (5, 5, 10))
        self.assertEqual(report["n_implicit_normal_labels"], 1)
        self.assertEqual((o["test_defect_with_mask"], o["test_defect_without_mask"]), (2, 2))
        names = {r["product"]: r for r in report["per_product"]}
        self.assertEqual(names["metal"]["defect"], 4)
        self.assertEqual(names["plastic"]["defect"], 1)
        pairs = {(r["product"],r["defect_type"]):r for r in report["per_product_defect_type"]}
        self.assertEqual(pairs[("metal","scratch")]["defect"], 3)
        self.assertEqual(pairs[("metal","stain")]["defect"], 1)
        self.assertEqual(pairs[("plastic","crack")]["defect"], 1)
        self.assertEqual(report["issues"]["counts"]["implicit_normal_label"], 1)

    def test_outputs_roundtrip_and_csv(self):
        report = module.inventory(self.root, self.json_dir)
        paths = module.write_outputs(report, Path(self.tmp.name) / "out")
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(p.exists() for p in paths))
        self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8"))["overall"]["total"], 10)
        self.assertEqual(len(paths[3].read_text(encoding="utf-8-sig").splitlines()), 4)

    def test_verify_files_reports_missing_then_finds_alternate_layout(self):
        (self.root / "realiad_1024" / "metal" / "train").mkdir(parents=True)
        (self.root / "realiad_1024" / "metal" / "train" / "good1.png").touch()
        report = module.inventory(self.root, self.json_dir, verify_files=True)
        self.assertEqual(report["issues"]["counts"]["image_file_missing"], 9)
        self.assertEqual(report["issues"]["counts"]["mask_file_missing"], 2)

    def test_malformed_split_is_not_silently_skipped(self):
        (self.json_dir / "metal.json").write_text(json.dumps({"train": {}, "test": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected 'train' array"):
            module.inventory(self.root, self.json_dir)

    def test_duplicate_and_mismatch_warnings(self):
        (self.json_dir / "metal.json").write_text(json.dumps({
            "train": [{"category": "other", "anomaly_class": "OK", "image_path": "a.png"},
                      {"category": "other", "anomaly_class": "OK", "image_path": "a.png"}],
            "test": [{"category": "other", "anomaly_class": "scratch", "image_path": "a.png"}],
        }), encoding="utf-8")
        report = module.inventory(self.root, self.json_dir)
        self.assertEqual(report["issues"]["counts"]["category_filename_mismatch"], 3)
        self.assertEqual(report["issues"]["counts"]["duplicate_image_in_split"], 1)
        self.assertEqual(report["issues"]["counts"]["image_in_both_splits"], 1)
        self.assertEqual(report["overall"]["total"], 5)


if __name__ == "__main__":
    unittest.main()
