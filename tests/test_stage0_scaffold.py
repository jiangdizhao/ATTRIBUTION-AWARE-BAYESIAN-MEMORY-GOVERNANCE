import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import abmg_stage0_foundation as s0  # noqa: E402


class Stage0UtilityTests(unittest.TestCase):
    def test_parse_layers_range_and_list(self):
        self.assertEqual(s0.parse_layers("4-6"), (4, 5, 6))
        self.assertEqual(s0.parse_layers("6,4,5,5"), (4, 5, 6))

    def test_default_grid_is_28_by_28(self):
        cfg = s0.Stage0Config()
        self.assertEqual(cfg.grid_hw, (28, 28))

    def test_paper_support_variants(self):
        img = Image.new("RGB", (32, 32), (10, 20, 30))
        variants = s0.paper_support_variants(img)
        self.assertEqual(len(variants), 6)
        self.assertTrue(all(v.size == (32, 32) for v in variants))

    def test_feature_payload_does_not_contain_ground_truth(self):
        payload = s0.FrozenFeaturePayload(
            image_id="abc",
            category="toy",
            split="test",
            relative_path="toy/a.png",
            patch_features=torch.zeros(4, 8),
            global_feature=torch.zeros(8),
            grid_hw=(2, 2),
            model_name="dummy",
            layers=(1, 2),
            resize_size=32,
            crop_size=28,
            layer_fusion="mean",
        ).as_torch_dict()
        self.assertNotIn("label", payload)
        self.assertNotIn("is_good", payload)
        self.assertNotIn("defect_source", payload)
        self.assertNotIn("mask_path", payload)


class Stage0RealIADIndexTests(unittest.TestCase):
    def _make_dataset(self, root: Path, json_dir: Path):
        category = "toy"
        img_dir = root / "realiad_1024" / category
        img_dir.mkdir(parents=True)
        json_dir.mkdir(parents=True)

        for name in ("normal.png", "defect.png"):
            Image.new("RGB", (16, 16), (128, 128, 128)).save(img_dir / name)
        Image.new("L", (16, 16), 255).save(img_dir / "defect_mask.png")

        doc = {
            "meta": {"prefix": f"{category}/", "normal_class": "OK"},
            "train": [
                {
                    "category": category,
                    "anomaly_class": "OK",
                    "image_path": "normal.png",
                    "mask_path": None,
                }
            ],
            "test": [
                {
                    "category": category,
                    "anomaly_class": "scratch",
                    "image_path": "defect.png",
                    "mask_path": "defect_mask.png",
                }
            ],
        }
        with (json_dir / f"{category}.json").open("w", encoding="utf-8") as f:
            json.dump(doc, f)

    def test_official_realiad_layout_is_resolved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "Real-IAD"
            json_dir = root / "realiad_jsons_sv"
            self._make_dataset(root, json_dir)
            records = s0.load_realiad_records(str(root), str(json_dir))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(Path(r.image_path).is_file() for r in records))
            defect = next(r for r in records if not r.is_good)
            self.assertEqual(defect.defect_source, "scratch")
            self.assertTrue(Path(defect.mask_path).is_file())

    def test_support_selection_is_reproducible(self):
        records = []
        for i in range(8):
            records.append(
                s0.Stage0Record(
                    image_id=str(i),
                    category="toy",
                    split="train",
                    image_path=f"/tmp/{i}.png",
                    relative_path=f"toy/{i}.png",
                    is_good=True,
                    defect_source="OK",
                    mask_path=None,
                    json_file="toy.json",
                )
            )
        a = [r.image_id for r in s0.select_supports(records, "toy", 4, seed=7)]
        b = [r.image_id for r in s0.select_supports(records, "toy", 4, seed=7)]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)


if __name__ == "__main__":
    unittest.main()
