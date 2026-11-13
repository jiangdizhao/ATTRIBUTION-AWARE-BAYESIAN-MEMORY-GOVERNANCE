from __future__ import annotations

import csv
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research_d6" / "p1a" / "extract_features.py"
spec = importlib.util.spec_from_file_location("p1a_extract_features", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class P1AManifestTests(unittest.TestCase):
    def test_exact_authorized_input_count_and_roles(self):
        rows = module.make_input_rows(ROOT)
        self.assertEqual(len(rows), 102)
        self.assertEqual(len({row["image_id"] for row in rows}), 102)
        self.assertEqual(sum(row["data_role"] == "D-fit" for row in rows), 54)
        self.assertEqual(sum(row["data_role"] == "D-dev" for row in rows), 48)
        self.assertEqual({row["dataset"] for row in rows}, {"aristrops/mvtec-concepts", "Real-IAD"})
        self.assertTrue(all(row["source_image_kind"] == "original_real_image" for row in rows))

    def test_manifest_does_not_import_label_fields(self):
        rows = module.make_input_rows(ROOT)
        forbidden = {"state", "label", "defect_type", "source_group_type", "view_anomaly_class"}
        self.assertFalse(forbidden.intersection(rows[0]))

    def test_ddev_ids_are_in_both_fixed_tapes(self):
        selected = {row["image_id"] for row in module.make_input_rows(ROOT) if row["data_role"] == "D-dev"}
        tape_path = ROOT / "research_pilot" / "LP_W1_D5R_IMAGE_PILOT_V2_2V" / "p1_gate0_draft" / "D6_P1_WX_role_map_feedback_tape.csv"
        with tape_path.open(encoding="utf-8", newline="") as f:
            tape = list(csv.DictReader(f))
        for protocol in ("W", "X"):
            # The tape deliberately contains no D-fit rows; all 48 registered D-dev IDs
            # appear as a feedback query, same-group exclusion, or future candidate.
            protocol_rows = [row for row in tape if row["protocol"] == protocol]
            used = {row["image_id"] for row in protocol_rows if row["feedback_or_future_role"] in {
                "feedback_query", "same_group_feedback_view_excluded_unqueried", "future_eval_candidate"
            }}
            self.assertEqual(used, selected)


if __name__ == "__main__":
    unittest.main()
