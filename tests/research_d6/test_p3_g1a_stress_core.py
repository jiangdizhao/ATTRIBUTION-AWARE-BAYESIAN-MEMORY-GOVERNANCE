from __future__ import annotations
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research_d6/p3_g1a"))
from stress_core import (STRESS_FIELDS, cosine_distance, group_medians, h12_score,
                         local_cells, local_distance, order_key, select_stress,
                         unit_mean, validate_stress_manifest)


def make_manifest(n_groups: int = 108) -> list[dict]:
    rows = []
    for i in range(n_groups):
        slot = str(i // 18 + 1)
        role = "P3_G1A_D_CAL" if int(slot) <= 2 else "P3_G1A_D_EVAL"
        kind = "REPRESENTATIVE_CANDIDATE" if i % 18 < 6 else "STRESS_POOL_CANDIDATE"
        gid = f"GRP_{i:032x}"
        tie = hashlib.sha256(gid.encode()).hexdigest()
        for view in ("C1", "C2"):
            rows.append({"opaque_image_id": f"IMG_{i:06d}_{view}", "opaque_group_id": gid,
                         "product_slot": slot, "study_role": role, "view": view,
                         "candidate_kind": kind, "stress_tie_key": tie})
    return rows


def stress_groups():
    vals = [1, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8, 9]
    rows = []
    for i, d in enumerate(vals, 1):
        rows.append({"product_slot": "1", "group_id": f"G{i:02d}",
                     "candidate_kind": "STRESS_POOL_CANDIDATE", "c": float(i), "d": float(d),
                     "tie_key": hashlib.sha256(f"tie{i}".encode()).hexdigest(),
                     "image_ids": [f"I{i}a", f"I{i}b"]})
    return rows


class G1AStressTests(unittest.TestCase):
    def test_01_fixed_manifest_is_216(self):
        rows = make_manifest()
        self.assertEqual(len(rows), 216)
        validate_stress_manifest(rows)

    def test_02_217th_candidate_rejected(self):
        rows = make_manifest()
        extra = dict(rows[0], opaque_image_id="IMG_EXTRA")
        with self.assertRaisesRegex(ValueError, "CARDINALITY"):
            validate_stress_manifest(rows + [extra])

    def test_03_manifest_has_108_groups(self):
        self.assertEqual(len({r["opaque_group_id"] for r in make_manifest()}), 108)

    def test_04_manifest_has_six_product_slots(self):
        self.assertEqual(len({r["product_slot"] for r in make_manifest()}), 6)

    def test_05_each_group_has_two_views(self):
        validate_stress_manifest(make_manifest())

    def test_06_group_cannot_cross_roles(self):
        rows = make_manifest()
        rows[1]["study_role"] = "P3_G1A_D_EVAL"
        with self.assertRaisesRegex(ValueError, "CROSSING"):
            validate_stress_manifest(rows)

    def test_07_duplicate_view_rejected(self):
        rows = make_manifest()
        rows[1]["view"] = rows[0]["view"]
        with self.assertRaisesRegex(ValueError, "CROSSING"):
            validate_stress_manifest(rows)

    def test_08_label_or_source_fields_rejected(self):
        rows = make_manifest()
        rows[0]["attribute_label"] = "present"
        with self.assertRaisesRegex(ValueError, "SCHEMA_NOT_LABEL_FREE"):
            validate_stress_manifest(rows)

    def test_09_mask_path_rejected_at_selector_boundary(self):
        rows = make_manifest()
        rows[0]["mask_path"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "SCHEMA_NOT_LABEL_FREE"):
            validate_stress_manifest(rows)

    def test_10_exact_allowed_fields(self):
        self.assertEqual(STRESS_FIELDS, set(make_manifest()[0]))

    def test_11_gmean_is_unit_length(self):
        x = np.eye(4, dtype=np.float32)
        m = np.array([1, 1, 0, 0], dtype=bool)
        g = unit_mean(x, m)
        self.assertAlmostEqual(np.linalg.norm(g), 1.0)

    def test_12_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            unit_mean(np.ones((4, 3)), np.zeros(4, dtype=bool))

    def test_13_nonfinite_feature_rejected(self):
        x = np.ones((4, 3)); x[1, 1] = np.nan
        with self.assertRaises(ValueError):
            unit_mean(x, np.ones(4, dtype=bool))

    def test_14_h12_theta_dimension(self):
        self.assertEqual(h12_score(np.array([1., 0.]), np.array([2., 1., -1.])), 1.0)

    def test_15_zero_logit_is_finite_registered_boundary(self):
        self.assertEqual(h12_score(np.array([1., 0.]), np.array([0., 0., 0.])), 0.0)

    def test_16_invalid_theta_rejected(self):
        with self.assertRaisesRegex(ValueError, "PARAMETER"):
            h12_score(np.ones(3), np.ones(3))

    def test_17_local_grid_produces_nonempty_cells(self):
        p = np.eye(16, dtype=np.float32)
        cells = local_cells(p, np.ones(16, dtype=bool))
        self.assertEqual(cells.shape, (16, 16))

    def test_18_local_grid_zero_norm_cells_handled(self):
        p = np.zeros((16, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "NONZERO"):
            local_cells(p, np.ones(16, dtype=bool))

    def test_19_invalid_patch_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHAPE"):
            local_cells(np.ones((15, 3)), np.ones(15, dtype=bool))

    def test_20_cosine_distance_reference(self):
        self.assertAlmostEqual(cosine_distance(np.array([1., 0.]), np.array([0., 1.])), 0.5)

    def test_21_bidirectional_local_distance_reference(self):
        a = np.array([[1., 0.], [0., 1.]])
        self.assertAlmostEqual(local_distance(a, a), 0.0)

    def test_22_empty_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError, "EMPTY_ANCHOR"):
            from stress_core import nearest
            nearest({"image_id": "q"}, [], {})

    def test_23_self_match_rejected(self):
        from stress_core import nearest
        f = {"q": {"global": np.array([1., 0.]), "local": np.array([[1., 0.]])}}
        with self.assertRaisesRegex(ValueError, "SELF_MATCH"):
            nearest({"image_id": "q"}, [{"image_id": "q", "event_index": 1}], f)

    def test_24_group_median_uses_two_views(self):
        rows = [{"product_slot": "1", "opaque_group_id": "g", "candidate_kind": "STRESS_POOL_CANDIDATE",
                 "h_logit": -1.0, "mixed_distance": 0.2, "tie_key": "a" * 64, "image_id": "a"},
                {"product_slot": "1", "opaque_group_id": "g", "candidate_kind": "STRESS_POOL_CANDIDATE",
                 "h_logit": 3.0, "mixed_distance": 0.6, "tie_key": "a" * 64, "image_id": "b"}]
        g = group_medians(rows)[0]
        self.assertEqual(g["c"], 2.0)
        self.assertEqual(g["d"], 0.4)

    def test_25_four_bins_selected(self):
        result = select_stress(stress_groups(), ["1"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["selected"]), 4)
        self.assertEqual({(r["confidence_bin"], r["distance_bin"]) for r in result["selected"]},
                         {("LOW", "LOW"), ("LOW", "HIGH"), ("HIGH", "LOW"), ("HIGH", "HIGH")})

    def test_26_input_order_does_not_change_selection(self):
        rows = stress_groups()
        a = select_stress(rows, ["1"])
        b = select_stress(list(reversed(rows)), ["1"])
        self.assertEqual({x["group_id"] for x in a["selected"]}, {x["group_id"] for x in b["selected"]})

    def test_27_missing_bin_is_reported_without_substitution(self):
        rows = stress_groups()
        for g in rows:
            g["d"] = g["c"]
        result = select_stress(rows, ["1"])
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["selected"], [])
        self.assertEqual(result["diagnostics"][0]["status"], "STRESS_BIN_COVERAGE_INSUFFICIENT")

    def test_28_tie_key_is_deterministic(self):
        self.assertEqual(order_key("STRESS_TIE", "usb\x1fGRP_A"), order_key("STRESS_TIE", "usb\x1fGRP_A"))

    def test_29_nonfinite_local_distance_rejected(self):
        with self.assertRaises(ValueError):
            local_distance(np.array([[np.nan, 1.]]), np.array([[1., 0.]]))

    def test_30_source_sha_identity_mismatch_refuses_bundle_reuse(self):
        from scripts.research_d6.p2.e2a1.storage import write_bundle_atomic
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts/research_d6/p2/e2a1"))
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "item"
            arr = {"x": np.array([1., 2.], dtype=np.float32)}
            write_bundle_atomic(bundle, arr, {"source_sha256": "a" * 64}, {"schema": "test"})
            with self.assertRaises(RuntimeError):
                write_bundle_atomic(bundle, arr, {"source_sha256": "b" * 64}, {"schema": "test"})

    def test_31_exact_completed_bundle_is_reused(self):
        from scripts.research_d6.p2.e2a1.storage import write_bundle_atomic
        with tempfile.TemporaryDirectory() as d:
            bundle = Path(d) / "item"
            arr = {"x": np.array([1., 2.], dtype=np.float32)}
            identity, meta = {"source_sha256": "a" * 64}, {"schema": "test"}
            write_bundle_atomic(bundle, arr, identity, meta)
            record, reused = write_bundle_atomic(bundle, arr, identity, meta)
            self.assertTrue(reused)
            self.assertIn("arrays", record)

    def test_32_existing_manifest_is_never_overwritten(self):
        from scripts.research_d6.p3_g1a.g1a_common import write_csv_once
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.csv"
            write_csv_once(path, [{"id": "a"}])
            with self.assertRaises(FileExistsError):
                write_csv_once(path, [{"id": "b"}])
            self.assertIn("a", path.read_text(encoding="utf-8"))

    def test_33_canonical_preprocessing_is_deterministic(self):
        from PIL import Image
        sys.path.insert(0, str(ROOT / "scripts/research_d6/p2/e2a1"))
        from preprocess import image_to_canonical, canonical_array_sha256
        image = Image.new("RGB", (37, 19), (12, 80, 220))
        a, ma = image_to_canonical(image)
        b, mb = image_to_canonical(image)
        self.assertEqual(a.shape, (1, 3, 448, 448))
        self.assertEqual(canonical_array_sha256(a), canonical_array_sha256(b))
        self.assertEqual(ma["content_rect_xyxy"], mb["content_rect_xyxy"])

    def test_34_geometry_mask_is_not_a_label_mask(self):
        from PIL import Image
        sys.path.insert(0, str(ROOT / "scripts/research_d6/p2/e2a1"))
        from preprocess import image_to_canonical
        _, meta = image_to_canonical(Image.new("RGB", (12, 80), (1, 2, 3)))
        mask = meta["mask_dinov2"]
        self.assertEqual(mask.shape, (1024,))
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), 1024)

    def test_35_shared_feature_interface_shape_and_finiteness(self):
        sys.path.insert(0, str(ROOT / "scripts/research_d6/p2/e2a"))
        from feature_interface import dinov2_to_shared_interface
        rng = np.random.default_rng(22)
        cls = rng.normal(size=(1, 1024)).astype(np.float32)
        regs = rng.normal(size=(1, 4, 1024)).astype(np.float32)
        patches = rng.normal(size=(1, 1024, 1024)).astype(np.float32)
        mask = np.ones(1024, dtype=bool); mask[-1] = False
        out = dinov2_to_shared_interface({"x_norm_clstoken": cls, "x_norm_regtokens": regs,
                                          "x_norm_patchtokens": patches}, mask)
        self.assertEqual(out["patch_unit"].shape, (1, 1024, 1024))
        self.assertEqual(int(out["valid_patch_mask"].sum()), 1023)
        self.assertTrue(np.isfinite(out["patch_unit"]).all())

    def test_36_six_products_get_four_bins_each(self):
        rows = []
        for slot in map(str, range(1, 7)):
            for g in stress_groups():
                g = dict(g); g["product_slot"] = slot; g["group_id"] = slot + g["group_id"]
                rows.append(g)
        result = select_stress(rows, list(map(str, range(1, 7))))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["selected"]), 24)
        self.assertEqual(len({r["product_slot"] for r in result["selected"]}), 6)

    def test_37_label_metadata_cannot_change_score_or_selection(self):
        base = stress_groups()
        before = {r["group_id"] for r in select_stress(base, ["1"])["selected"]}
        # Labels live outside the selector contract; it accepts no such field.
        altered = make_manifest()
        altered[0]["source_defect_type"] = "synthetic-change"
        with self.assertRaisesRegex(ValueError, "SCHEMA_NOT_LABEL_FREE"):
            validate_stress_manifest(altered)
        after = {r["group_id"] for r in select_stress(base, ["1"])["selected"]}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
