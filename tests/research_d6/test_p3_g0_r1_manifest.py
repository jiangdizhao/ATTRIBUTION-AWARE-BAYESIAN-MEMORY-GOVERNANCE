"""Synthetic identity-only validation tests; no research data are used."""
from __future__ import annotations
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.research_d6.p3_g0 import validate_identity_manifest as v


def fixture(products=6, groups=10, views=2):
    rows = []
    for p in range(products):
        for g in range(groups):
            group = p * 100 + g + 1
            for view in range(views):
                image = group * 100 + view + 1
                rows.append({
                    "opaque_image_id": f"IMG_{image:032x}", "product": f"synthetic_{p}",
                    "opaque_group_id": f"GRP_{group:032x}", "view": f"C{view+1}",
                    "source_split": "test", "source_sha256": f"{image:064x}",
                    "exposure_status": "METADATA_EXPOSED_ONLY", "allowed_research_role": "D_CAL_OR_D_EVAL",
                    "permission_authority": "fictional_custodian", "permission_record_version": "test_v1",
                    "permission_status": "ACTIVE", "exclusion_status": "CLEAR",
                    "exclusion_check_scope": "DFIT_DDEV_P2_AND_PROTECTED",
                    "exclusion_authority": "fictional_custodian", "exclusion_record_version": "test_v1",
                    "near_duplicate_status": "CLEAR_WITHIN_SCOPE", "group_conflict_status": "CLEAR_WITHIN_SCOPE",
                    "object_identity_confidence": "INFERRED", "source_manifest_version": "fictional_v1",
                })
    return rows


class ManifestTests(unittest.TestCase):
    def test_valid_capacity(self):
        r = v.validate_rows(fixture(), "D_EVAL")
        self.assertEqual((r["status"], r["confirmed_product_count"], r["nominal_group_count"]), ("PASS", 6, 60))
        self.assertEqual(r["object_identity_confidence_group_counts"], {"INFERRED": 60})

    def test_duplicate_image_id_blocks(self):
        rows = fixture(); rows[1]["opaque_image_id"] = rows[0]["opaque_image_id"]
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "BLOCKED"); self.assertIn("DUPLICATE_IMAGE_ID", r["reason_group_counts"])

    def test_duplicate_sha_makes_groups_ineligible(self):
        rows = fixture(); rows[1]["source_sha256"] = rows[0]["source_sha256"]
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "INELIGIBLE"); self.assertIn("DUPLICATE_SOURCE_SHA", r["reason_group_counts"])

    def test_missing_sha_is_unknown(self):
        rows = fixture(); rows[0]["source_sha256"] = ""
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "UNKNOWN"); self.assertIn("MISSING_SOURCE_SHA", r["reason_group_counts"])

    def test_one_view_is_ineligible(self):
        rows = fixture(); del rows[1]
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "INELIGIBLE"); self.assertIn("FEWER_THAN_TWO_VIEWS", r["reason_group_counts"])

    def test_duplicate_view_blocks(self):
        rows = fixture(); rows[1]["view"] = rows[0]["view"]
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "BLOCKED"); self.assertIn("DUPLICATE_VIEW_IN_GROUP", r["reason_group_counts"])

    def test_group_identity_crosses_products(self):
        rows = fixture(); rows[20]["opaque_group_id"] = rows[0]["opaque_group_id"]
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "BLOCKED"); self.assertIn("GROUP_ID_ACROSS_PRODUCTS", r["reason_group_counts"])

    def test_protected_conflict_excluded(self):
        rows = fixture(); rows[0]["exclusion_status"] = "CONFLICT"
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "INELIGIBLE"); self.assertIn("PROTECTED_EXCLUSION_CONFLICT", r["reason_group_counts"])

    def test_unknown_exclusion_never_confirmed(self):
        rows = fixture(); rows[0]["exclusion_status"] = rows[1]["exclusion_status"] = "UNKNOWN"
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "UNKNOWN"); self.assertEqual(r["confirmed_product_count"], 5)

    def test_clear_needs_scope_and_custodian_record(self):
        rows = fixture(); rows[0]["exclusion_check_scope"] = rows[1]["exclusion_check_scope"] = "PARTIAL"
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "UNKNOWN"); self.assertIn("EXCLUSION_SCOPE_INCOMPLETE", r["reason_group_counts"])

    def test_clear_without_exclusion_authority_is_unknown(self):
        rows = fixture(); rows[0]["exclusion_authority"] = rows[1]["exclusion_authority"] = ""
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "UNKNOWN"); self.assertIn("EXCLUSION_ATTESTATION_MISSING", r["reason_group_counts"])

    def test_inconsistent_group_roles_block(self):
        rows = fixture(); rows[0]["allowed_research_role"] = "D_CAL"
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "BLOCKED"); self.assertIn("INCONSISTENT_GROUP_METADATA", r["reason_group_counts"])

    def test_near_duplicate_conflict_and_unknown_exposure(self):
        rows = fixture(); rows[0]["near_duplicate_status"] = rows[1]["near_duplicate_status"] = "CONFLICT"
        self.assertEqual(v.validate_rows(rows, "D_EVAL")["status"], "INELIGIBLE")
        rows = fixture(); rows[0]["exposure_status"] = rows[1]["exposure_status"] = "NOT_KNOWN_EXPOSED"
        self.assertEqual(v.validate_rows(rows, "D_EVAL")["status"], "UNKNOWN")

    def test_existing_p2_product_is_not_confirmed(self):
        rows = fixture()
        for row in rows:
            if row["product"] == "synthetic_0": row["product"] = "toothbrush"
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "INELIGIBLE"); self.assertIn("P2_DEVELOPMENT_PRODUCT", r["reason_group_counts"])

    def test_wrong_role_ineligible(self):
        rows = fixture(); rows[0]["allowed_research_role"] = rows[1]["allowed_research_role"] = "D_CAL"
        r = v.validate_rows(rows, "D_EVAL")
        self.assertEqual(r["status"], "INELIGIBLE"); self.assertIn("ROLE_UNAUTHORIZED", r["reason_group_counts"])

    def test_insufficient_products(self):
        self.assertEqual(v.validate_rows(fixture(products=5), "D_EVAL")["status"], "INELIGIBLE")

    def test_insufficient_groups(self):
        self.assertEqual(v.validate_rows(fixture(groups=9), "D_EVAL")["status"], "INELIGIBLE")

    def test_missing_manifest_does_not_discover(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("no read allowed")):
            self.assertEqual(v.validate_file(None, "D_EVAL", False, "", "")["status"], "MISSING_AUTHORIZED_MANIFEST")
            self.assertEqual(v.validate_file(Path("missing.identity-only.csv"), "D_EVAL", True, "synthetic", "0"*64)["status"], "MISSING_AUTHORIZED_MANIFEST")

    def test_deterministic_results(self):
        rows = fixture(); self.assertEqual(v.validate_rows(rows, "D_CAL"), v.validate_rows(list(reversed(rows)), "D_CAL"))

    def test_real_images_and_raw_json_are_rejected_before_read(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("no read allowed")):
            for name in ("image.png", "realiad_product.json", "image.identity-only.png"):
                self.assertEqual(v.validate_file(Path(name), "D_EVAL", True, "synthetic", "0"*64)["status"], "BLOCKED")

    def test_missing_explicit_authorization_blocks_before_read(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("no read allowed")):
            self.assertEqual(v.validate_file(Path("synthetic.identity-only.csv"), "D_EVAL", False, "synthetic", "0"*64)["status"], "BLOCKED")

    def test_synthetic_csv_and_json(self):
        rows = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            csvpath = Path(tmp)/"synthetic.identity-only.csv"
            with csvpath.open("w", encoding="utf8", newline="") as f:
                writer=csv.DictWriter(f,fieldnames=v.FIELDS); writer.writeheader(); writer.writerows(rows)
            jpath = Path(tmp)/"synthetic.identity-only.json"
            jpath.write_text(json.dumps({"schema":"lp.p3.identity-only.v1","rows":rows}),encoding="utf8")
            for path in (csvpath,jpath):
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(v.validate_file(path,"D_EVAL",True,"synthetic",sha)["status"],"PASS")
                self.assertEqual(v.validate_file(path,"D_EVAL",True,"synthetic","0"*64)["status"],"BLOCKED")

    def test_forbidden_extra_field_blocks(self):
        rows=fixture(); rows[0]["defect_type"]="fake"
        self.assertEqual(v.validate_rows(rows,"D_EVAL")["status"],"BLOCKED")

    def test_pixel_exposed_and_unknown_near_duplicates_do_not_pass(self):
        rows=fixture(); rows[0]["exposure_status"]=rows[1]["exposure_status"]="PIXEL_EXPOSED_DEV"
        self.assertEqual(v.validate_rows(rows,"D_EVAL")["status"],"INELIGIBLE")
        rows=fixture(); rows[0]["near_duplicate_status"]=rows[1]["near_duplicate_status"]="UNKNOWN"
        self.assertEqual(v.validate_rows(rows,"D_EVAL")["status"],"UNKNOWN")


if __name__ == "__main__": unittest.main()
