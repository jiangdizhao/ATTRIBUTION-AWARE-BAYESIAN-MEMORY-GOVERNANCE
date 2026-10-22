"""Stage-3 P0 contract tests. Synthetic fixtures only; no Real-IAD/DINO imports."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "stage3_p0_validate.py"
spec = importlib.util.spec_from_file_location("stage3_p0_validate", PATH)
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)


def trace(action="classify", revealed=None, **overrides):
    e = {
        "time": 0, "image_id": "unit-image", "action": action,
        "pre": {"score": 0.12, "threshold": 0.15},
        "revealed": {} if revealed is None else revealed,
        "paid_binary": action != "classify",
        "paid_rich_incremental": action == "query_rich",
        "prior_binary_query": action == "query_rich",
    }
    e.update(overrides)
    return e


class Stage3P0Tests(unittest.TestCase):
    def test_canonical_folds_cover_every_product_once_and_20_source_cv_groups(self):
        folds = p0.load_json(ROOT / "configs" / "realiad_folds_v0.json")
        out = p0.validate_folds(folds)
        self.assertEqual(len(out), 5)
        self.assertEqual(sum(len(f["source_cv"]) for f in out), 20)
        for f in out:
            target = set(f["sealed_target"])
            for inner in f["source_cv"]:
                self.assertFalse(target & set(inner["development"]))
                self.assertFalse(target & set(inner["validation"]))
                self.assertEqual(len(inner["development"]), 18)
                self.assertEqual(len(inner["validation"]), 6)

    def test_duplicate_target_product_is_rejected(self):
        folds = p0.load_json(ROOT / "configs" / "realiad_folds_v0.json")
        bad = copy.deepcopy(folds)
        bad["folds"][1]["target_categories"][0] = bad["folds"][0]["target_categories"][0]
        with self.assertRaises(ValueError):
            p0.validate_folds(bad)

    def test_draft_manifest_valid_but_not_ready_for_g0(self):
        report = p0.check_manifest()
        self.assertFalse(report["errors"], report["errors"])
        self.assertTrue(report["structurally_valid"])
        self.assertFalse(report["g0_ready"])
        self.assertGreater(len(report["pending_evidence"]), 0)
        self.assertEqual(report["source_cv_partitions_validated"], 20)
        strict = p0.check_manifest(strict=True)
        self.assertTrue(strict["errors"])
        self.assertFalse(strict["g0_ready"])

    def test_unqueried_classify_never_reveals_labels(self):
        p0.validate_agent_event(trace())
        with self.assertRaises(ValueError):
            p0.validate_agent_event(trace(revealed={"binary": 0}))
        with self.assertRaises(ValueError):
            p0.validate_agent_event(trace(gt_mask="hidden_mask"))

    def test_binary_allows_only_paid_image_level_label(self):
        p0.validate_agent_event(trace("query_binary", {"binary": 0}))
        p0.validate_agent_event(trace("query_binary", {"binary": 1}))
        with self.assertRaises(ValueError):
            p0.validate_agent_event(trace("query_binary", {"binary": 1, "mask_ref": "oracle"}))
        with self.assertRaises(ValueError):
            p0.validate_agent_event(trace("query_binary", {"binary": 1}, paid_binary=False))
        with self.assertRaises(ValueError):
            p0.validate_agent_event(trace("query_binary", {"binary": True}))

    def test_rich_only_after_paid_defect_binary_and_valid_mask(self):
        rich = {"binary": 1, "defect_type": "scratch", "mask_ref": "paid-sample-mask"}
        p0.validate_agent_event(trace("query_rich", rich))
        for wrong in (
            {"binary": 0, "defect_type": "scratch", "mask_ref": "mask"},
            {"binary": 1, "defect_type": "scratch"},
            {"binary": 1, "defect_type": "", "mask_ref": "mask"},
        ):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                p0.validate_agent_event(trace("query_rich", wrong))
        with self.assertRaises(ValueError):
            p0.validate_agent_event(trace("query_rich", rich, prior_binary_query=False))
        with self.assertRaises(ValueError):
            p0.validate_agent_event(trace("query_rich", rich, paid_rich_incremental=False))

    def test_evaluator_only_fields_rejected_even_when_nested(self):
        with self.assertRaises(ValueError):
            p0.validate_agent_event(trace("query_binary", {"binary": 1},
                                          extra={"hidden": {"target_gt": 1}}))


if __name__ == "__main__":
    unittest.main()
