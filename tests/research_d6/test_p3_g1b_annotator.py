from __future__ import annotations

import csv
import hashlib
import io
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from scripts.research_d6.p3_g1b import annotate, g1b_core as core


def synthetic_bundle():
    roles = []
    mapping = {}
    for slot in range(1, 7):
        role_name = "P3_G1A_D_CAL" if slot <= 2 else "P3_G1A_D_EVAL"
        for group_index in range(10):
            group_id = f"synthetic-group-{slot:02d}-{group_index:02d}"
            product = f"synthetic-product-{slot:02d}"
            for view_index, view in enumerate(("C1", "C3")):
                image_id = f"synthetic-image-{slot:02d}-{group_index:02d}-{view_index}"
                kind = "REPRESENTATIVE_CANDIDATE" if group_index < 6 else "STRESS_POOL_CANDIDATE"
                role = {
                    "opaque_image_id": image_id,
                    "opaque_group_id": group_id,
                    "product_slot": str(slot),
                    "study_role": role_name,
                    "view": view,
                    "candidate_kind": kind,
                    "role_reason": "SYNTHETIC",
                }
                roles.append(role)
                mapping[image_id] = {
                    "opaque_image_id": image_id,
                    "opaque_group_id": group_id,
                    "product": product,
                    "product_slot": str(slot),
                    "study_role": role_name,
                    "view": view,
                    "candidate_kind": kind,
                    "source_image_path": f"G:\\dataset\\realiad\\synthetic\\{image_id}.png",
                    "source_sha256": hashlib.sha256(image_id.encode()).hexdigest(),
                    "source_size_bytes": "1",
                    "exposure_status": "NEW_STUDY_DEV_EXPOSED",
                    "g1a_read_success": "YES",
                    "g1a_final_path_verified": "YES",
                }
    return roles, mapping


class G1BCoreTests(unittest.TestCase):
    def setUp(self):
        self.roles, self.mapping = synthetic_bundle()
        self.queue, self.repeats, self.summary = core.build_queue(self.roles, self.mapping)

    def test_01_unique_images_120_and_groups_60(self):
        self.assertEqual(len({row["opaque_image_id"] for row in self.roles}), 120)
        self.assertEqual(len({(self.mapping[row["opaque_image_id"]]["product"], row["opaque_group_id"]) for row in self.roles}), 60)

    def test_02_two_views_per_group(self):
        groups = defaultdict(list)
        for row in self.roles:
            groups[(row["product_slot"], row["opaque_group_id"])].append(row)
        self.assertTrue(all(len(rows) == 2 and len({r["view"] for r in rows}) == 2 for rows in groups.values()))

    def test_03_dcal_and_deval_cardinality(self):
        counts = Counter(row["study_role"] for row in self.roles)
        self.assertEqual(counts, {"P3_G1A_D_CAL": 40, "P3_G1A_D_EVAL": 80})

    def test_04_product_roles_do_not_cross(self):
        by_product = defaultdict(set)
        for row in self.roles:
            by_product[row["product_slot"]].add(row["study_role"])
        self.assertEqual(sum(len(value) == 1 for value in by_product.values()), 6)

    def test_05_g0_repeat_hash_selects_exactly_first_twelve_groups(self):
        groups = {(self.mapping[r["opaque_image_id"]]["product"], r["opaque_group_id"]) for r in self.roles}
        expected = sorted(groups, key=lambda g: core.order_key("REPEAT_GROUP", g[0] + core.SEP + g[1]))[:12]
        actual = {(r["product"], r["opaque_group_id"]) for r in self.summary["repeated_groups"]}
        self.assertEqual(actual, set(expected))

    def test_06_repeat_groups_have_twenty_four_unique_views(self):
        self.assertEqual(len(self.repeats), 24)
        self.assertEqual(len({row["opaque_image_id"] for row in self.repeats}), 24)

    def test_07_exactly_144_queue_presentations(self):
        self.assertEqual(len(self.queue), 144)
        self.assertEqual(self.summary["presentation_count"], 144)

    def test_08_exactly_120_unique_queue_images(self):
        self.assertEqual(len({row["opaque_image_id"] for row in self.queue}), 120)

    def test_09_each_image_has_one_first_and_at_most_one_repeat(self):
        by_image = defaultdict(list)
        for row in self.queue:
            by_image[row["opaque_image_id"]].append(row)
        self.assertEqual(sum(len(rows) == 2 for rows in by_image.values()), 24)
        self.assertTrue(all(Counter(r["presentation_kind"] for r in rows) in (Counter({"FIRST": 1}), Counter({"FIRST": 1, "REPEAT": 1})) for rows in by_image.values()))

    def test_10_repeat_is_not_adjacent_to_first(self):
        self.assertGreater(self.summary["min_first_repeat_distance_positions"], 1)
        self.assertTrue(all(abs(int(row["first_order"]) - int(row["repeat_order"])) > 1 for row in self.repeats))

    def test_11_presentation_ids_are_unique_and_not_image_ids(self):
        ids = {row["presentation_id"] for row in self.queue}
        self.assertEqual(len(ids), 144)
        self.assertTrue(ids.isdisjoint({row["opaque_image_id"] for row in self.queue}))

    def test_12_queue_mismatch_fails_closed(self):
        broken = [dict(row) for row in self.queue]
        broken[0]["study_role"] = "P3_G1A_D_EVAL" if broken[0]["study_role"] == "P3_G1A_D_CAL" else "P3_G1A_D_CAL"
        with self.assertRaises(core.IntegrityError):
            core.validate_queue(broken, self.mapping)

    def test_13_allowed_states_and_blank_are_distinct(self):
        self.assertEqual(core.VALID_STATES, {"present", "absent_assessable", "unobservable_or_uncertain", "not_applicable"})
        row = {key: "" for key in core.ANNOTATION_FIELDS}
        row.update({"presentation_id": "synthetic-blank", "attribute_state": ""})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "annotations.csv"
            core.atomic_write_csv(path, [row])
            self.assertEqual(core.read_csv(path)[0]["attribute_state"], "")

    def test_14_four_state_labels_round_trip_without_collapsing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "annotations.csv"
            rows = []
            for index, state in enumerate(("present", "absent_assessable", "unobservable_or_uncertain", "not_applicable")):
                row = {key: "" for key in core.ANNOTATION_FIELDS}
                row.update({"presentation_id": f"p-{index}", "attribute_state": state})
                rows.append(row)
            core.atomic_write_csv(path, rows)
            self.assertEqual({r["attribute_state"] for r in core.read_csv(path)}, core.VALID_STATES)

    def test_15_source_hash_mismatch_blocks(self):
        with self.assertRaises(core.IntegrityError):
            core.verify_image_hash("0" * 64, "f" * 64)

    def test_16_source_hash_match_passes(self):
        value = hashlib.sha256(b"synthetic image").hexdigest()
        core.verify_image_hash(value, value)

    def test_17_deval_file_is_not_accessible_to_dcal_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "D-cal").mkdir()
            (root / "D-eval").mkdir()
            core.atomic_write_csv(root / "D-cal" / "annotations.csv", [])
            core.atomic_write_csv(root / "D-eval" / "annotations.csv", [])
            with self.assertRaises(PermissionError):
                core.load_dcal_annotations(root / "D-eval" / "annotations.csv", root)

    def test_18_dcal_loader_reads_only_its_allowlisted_empty_template(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "D-cal").mkdir()
            core.atomic_write_csv(root / "D-cal" / "annotations.csv", [])
            self.assertEqual(core.load_dcal_annotations(root / "D-cal" / "annotations.csv", root), [])

    def test_19_invalid_state_is_rejected_by_dcal_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "D-cal").mkdir()
            row = {key: "" for key in core.ANNOTATION_FIELDS}
            row.update({"presentation_id": "p", "attribute_state": "defect"})
            core.atomic_write_csv(root / "D-cal" / "annotations.csv", [row])
            with self.assertRaises(core.IntegrityError):
                core.load_dcal_annotations(root / "D-cal" / "annotations.csv", root)

    def test_20_single_writer_lock_blocks_second_process(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "session.lock"
            first, second = core.SingleWriterLock(path), core.SingleWriterLock(path)
            first.acquire()
            try:
                with self.assertRaises(RuntimeError):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()

    def test_21_interrupted_atomic_replace_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "value.csv"
            path.write_bytes(b"old-valid-data")
            def fail_replace(_source, _dest):
                raise OSError("synthetic replace interruption")
            with self.assertRaises(OSError):
                core.atomic_write_bytes(path, b"new-data", replace=fail_replace)
            self.assertEqual(path.read_bytes(), b"old-valid-data")

    def test_22_old_v2_output_path_is_forbidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            with self.assertRaises(core.IntegrityError):
                core.assert_output_within_run(root / "LP_W1_D5R_IMAGE_PILOT_V2_2V" / "annotations.csv", root)

    def test_23_exif_orientation_is_applied_before_render(self):
        image = Image.new("RGB", (40, 20), "white")
        exif = Image.Exif()
        exif[274] = 6
        payload = io.BytesIO()
        image.save(payload, format="JPEG", exif=exif)
        oriented = annotate.oriented_image(payload.getvalue())
        self.assertEqual(oriented.size, (20, 40))

    def test_24_gui_synthetic_render_save_restore_and_blind_repeat(self):
        result = annotate._synthetic_session()
        self.assertTrue(result["rendered_canvas_items"])
        self.assertTrue(result["save_restore"])
        self.assertTrue(result["blind_repeat_independent"])
        self.assertTrue(result["zoom_pan_preserved_identity_and_label"])
        self.assertTrue(result["hidden_metadata_not_visible"])

    def test_25_real_annotation_requires_separate_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            authorization = run_dir / "annotation_authorization.json"
            with self.assertRaises((FileNotFoundError, core.IntegrityError, PermissionError)):
                annotate.production_preflight(run_dir, authorization)


if __name__ == "__main__":
    unittest.main(verbosity=2)
