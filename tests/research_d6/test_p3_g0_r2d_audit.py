from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from scripts.research_d6.p3_g0_r2d import r2d_core as r


def fixture():
    products = []
    for i in range(12):
        role = "TENTATIVE_D_CAL" if i < 2 else "TENTATIVE_D_EVAL" if i < 6 else "BACKUP_PRODUCT"
        products.append({"product": f"p{i}", "conditional_role": role})
    groups, rows = [], []
    for i in range(6):
        role = products[i]["conditional_role"]
        for j in range(18):
            gid = f"GRP_{i:02}_{j:02}"
            kind = "REPRESENTATIVE_CANDIDATE" if j < 6 else "STRESS_POOL_CANDIDATE"
            groups.append({"opaque_group_id": gid, "product": f"p{i}", "conditional_role": role,
                           "candidate_kind": kind, "eligibility": "UNKNOWN"})
            for view in ("C1", "C2"):
                n = len(rows)
                rows.append({"opaque_image_id": f"IMG_{n:04}", "product": f"p{i}", "opaque_group_id": gid,
                             "view": view, "conditional_role": role, "candidate_kind": kind,
                             "eligibility": "UNKNOWN", "source_image_path": fr"G:\dataset\realiad\p{i}\{gid}_{view}.jpg",
                             "source_sha256_status": "UNKNOWN", "future_file_access_permission": "NOT_AUTHORIZED",
                             "historical_exposure_status": "UNKNOWN", "protected_exclusion_status": "UNKNOWN"})
    return products, groups, rows


class CountedStream(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


class R2DTests(unittest.TestCase):
    def test_authorization_record_identifier(self):
        self.assertEqual(r.AUTHORIZATION, "RESEARCH_LEAD_CHAT_APPROVAL_20261121_R2D_SHA_ONLY")

    def test_wrong_authorization_blocks_before_read(self):
        stream = CountedStream(b"fake")
        with self.assertRaisesRegex(r.R2DError, "SHA_AUTHORIZATION_REQUIRED"):
            r.gated_hash_stream(stream, authorization="", protection_status="CLEARED_FOR_SHA_READ", path_safe=True)
        self.assertEqual(stream.read_calls, 0)

    def test_pinned_whitelist_hash_mismatch_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "whitelist.csv"
            path.write_bytes(b"synthetic")
            with self.assertRaisesRegex(r.R2DError, "PINNED_FILE_SHA_MISMATCH"):
                r.verify_file_sha(path, "0" * 64)

    def test_selection_receipt_output_hash_mismatch_blocks(self):
        with self.assertRaisesRegex(r.R2DError, "RECORDED_OUTPUT_HASH_MISMATCH"):
            r.verify_recorded_output_hash("a" * 64, "b" * 64)

    def test_exact_216_row_candidate_frame_passes_synthetic_validation(self):
        p, g, rows = fixture()
        result = r.validate_whitelist_rows(rows, p, g)
        self.assertEqual((result["image_count"], result["group_count"], result["product_count"]), (216, 108, 6))

    def test_217th_row_blocks(self):
        p, g, rows = fixture()
        rows.append(dict(rows[-1], opaque_image_id="IMG_EXTRA"))
        with self.assertRaisesRegex(r.R2DError, "WHITELIST_ROW_COUNT"):
            r.validate_whitelist_rows(rows, p, g)

    def test_duplicate_opaque_image_id_blocks(self):
        p, g, rows = fixture()
        rows[-1]["opaque_image_id"] = rows[0]["opaque_image_id"]
        with self.assertRaisesRegex(r.R2DError, "DUPLICATE_OPAQUE_IMAGE_ID"):
            r.validate_whitelist_rows(rows, p, g)

    def test_path_traversal_blocks(self):
        with self.assertRaisesRegex(r.R2DError, "PATH_TRAVERSAL"):
            r.validate_lexical_path(r"G:\dataset\realiad\p0\..\secret.jpg", "p0")

    def test_other_drive_blocks(self):
        with self.assertRaisesRegex(r.R2DError, "PATH_DRIVE_OR_UNC"):
            r.validate_lexical_path(r"H:\dataset\realiad\p0\x.jpg", "p0")

    def test_unc_path_blocks(self):
        with self.assertRaisesRegex(r.R2DError, "PATH_DRIVE_OR_UNC"):
            r.validate_lexical_path(r"\\server\share\p0\x.jpg", "p0")

    def test_reparse_point_component_blocks(self):
        self.assertFalse(r.validate_no_reparse_components([False, True, False]))
        self.assertTrue(r.validate_no_reparse_components([False, False, False]))

    def test_known_protected_conflict_is_never_clear(self):
        _, _, rows = fixture()
        row = dict(rows[0], protected_exclusion_status="KNOWN_PROTECTED_CONFLICT")
        self.assertEqual(r.classify_protection(row)[0], "KNOWN_PROTECTED_CONFLICT")
        stream = CountedStream(b"image")
        with self.assertRaisesRegex(r.R2DError, "PROTECTION_NOT_CLEARED"):
            r.gated_hash_stream(stream, authorization=r.AUTHORIZATION,
                                protection_status="KNOWN_PROTECTED_CONFLICT", path_safe=True)
        self.assertEqual(stream.read_calls, 0)

    def test_unknown_protection_skips_without_read(self):
        _, _, rows = fixture()
        result = r.protected_preflight(rows)
        self.assertEqual(len(result), 216)
        self.assertTrue(all(x["protection_status"] == "PROTECTION_UNKNOWN" for x in result))
        self.assertTrue(all(x["sha_status"] == "SHA_SKIPPED_PROTECTION_UNKNOWN" and x["bytes_read"] == 0 for x in result))
        stream = CountedStream(b"image")
        with self.assertRaisesRegex(r.R2DError, "PROTECTION_NOT_CLEARED"):
            r.gated_hash_stream(stream, authorization=r.AUTHORIZATION,
                                protection_status="PROTECTION_UNKNOWN", path_safe=True)
        self.assertEqual(stream.read_calls, 0)

    def test_item_scoped_synthetic_clear_enters_hash_gate(self):
        _, _, rows = fixture()
        row = rows[0]
        evidence = {"opaque_image_id": row["opaque_image_id"], "scope": "ITEM_LEVEL_PROTECTED_EXCLUSION",
                    "source_version": "synthetic-v1", "authorization": "synthetic-only",
                    "evidence_id": "fixture", "status": "CLEARED_FOR_SHA_READ"}
        state, _ = r.classify_protection(row, evidence)
        stream = CountedStream(b"synthetic bytes")
        digest, size = r.gated_hash_stream(stream, authorization=r.AUTHORIZATION,
                                           protection_status=state, path_safe=True, block_size=3)
        self.assertEqual(digest, r.sha_bytes(b"synthetic bytes"))
        self.assertEqual(size, len(b"synthetic bytes"))
        self.assertGreater(stream.read_calls, 1)

    def test_zero_clear_means_zero_real_candidate_read_calls(self):
        _, _, rows = fixture()
        clear_count = sum(r.classify_protection(row)[0] == "CLEARED_FOR_SHA_READ" for row in rows)
        reads = 0
        for row in rows:
            status, _ = r.classify_protection(row)
            if status == "CLEARED_FOR_SHA_READ":
                reads += 1
        self.assertEqual((clear_count, reads), (0, 0))

    def test_streaming_sha_matches_reference_without_decoder(self):
        payload = b"synthetic image bytes" * 1000
        stream = CountedStream(payload)
        digest, size = r.gated_hash_stream(stream, authorization=r.AUTHORIZATION,
                                           protection_status="CLEARED_FOR_SHA_READ", path_safe=True,
                                           block_size=97)
        self.assertEqual(digest, r.sha_bytes(payload))
        self.assertEqual(size, len(payload))
        self.assertGreater(stream.read_calls, 1)

    def test_file_change_during_read_rejects_digest(self):
        token = (1, 100)
        stream = CountedStream(b"synthetic")
        with self.assertRaisesRegex(r.R2DError, "FILE_CHANGED_DURING_READ"):
            r.gated_hash_stream(stream, authorization=r.AUTHORIZATION,
                                protection_status="CLEARED_FOR_SHA_READ", path_safe=True,
                                before_token=token, after_metadata=lambda: (2, 100))

    def test_exact_duplicate_detection(self):
        result = r.exact_duplicate_groups([{"sha256": "x", "opaque_group_id": "g1"},
                                           {"sha256": "x", "opaque_group_id": "g2"},
                                           {"sha256": "y", "opaque_group_id": "g3"}])
        self.assertEqual(result, {"duplicate_file_count": 2, "affected_group_count": 2, "duplicate_sha_count": 1})

    def test_different_sha_only_reports_no_exact_match(self):
        result = r.compare_sha([{"sha256": "a"}], ["b"])
        self.assertEqual(result["exact_matches"], 0)
        self.assertEqual(result["comparison_status"], "COMPARED")

    def test_no_candidate_sha_is_not_reported_as_zero_overlap(self):
        result = r.compare_sha([{"sha256": ""}], ["known-history"])
        self.assertEqual(result["comparison_status"], "NO_CANDIDATE_SHA")

    def test_sha_success_does_not_promote_eligibility(self):
        self.assertEqual(r.eligibility_after_sha("CLEARED_FOR_SHA_READ", {"near_duplicate": "UNKNOWN"}), "UNKNOWN")
        self.assertEqual(r.eligibility_after_sha("CLEARED_FOR_SHA_READ", None), "UNKNOWN")

    def test_no_auto_promotion_to_formal_roles(self):
        p, g, rows = fixture()
        rows[0]["conditional_role"] = "D_CAL"
        with self.assertRaises(r.R2DError):
            r.validate_whitelist_rows(rows, p, g)

    def test_public_summary_does_not_leak_path_or_ids(self):
        summary = r.public_summary({"status": "R2D_BLOCKED_PROTECTION", "whitelist_count": 216,
                                    "source_image_path": r"G:\private\x.jpg", "opaque_image_id": "private"})
        self.assertNotIn("source_image_path", summary)
        self.assertNotIn("opaque_image_id", summary)

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "existing"
            path.mkdir()
            with self.assertRaisesRegex(FileExistsError, "OUTPUT_ALREADY_EXISTS"):
                r.assert_output_absent(path)


if __name__ == "__main__":
    unittest.main()
