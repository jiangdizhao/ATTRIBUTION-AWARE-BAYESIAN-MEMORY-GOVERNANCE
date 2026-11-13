from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1B = ROOT / "scripts/research_d6/p1b"
sys.path.insert(0, str(P1B))

import r1_score_audit as audit  # noqa: E402
from run_experiment import (  # noqa: E402
    APPROVED_BASELINE_COMMIT,
    EXPECTED_BRANCH,
    INPUT_MANIFEST_SHA,
    capture_preflight,
    git_identity,
    require_fingerprint,
    validate_revision_policy,
)


class P1BR1AuditTests(unittest.TestCase):
    def test_committed_head_passes_ancestry_revision_policy(self):
        branch, head = git_identity()
        revision = validate_revision_policy(branch, head)
        self.assertEqual(branch, EXPECTED_BRANCH)
        self.assertEqual(revision["execution_head"], head)
        self.assertEqual(revision["approved_baseline_commit"], APPROVED_BASELINE_COMMIT)
        preflight = capture_preflight()
        self.assertEqual(preflight["git_sha"], head)
        self.assertEqual(preflight["historical_original_run_starting_head"], APPROVED_BASELINE_COMMIT)
        self.assertEqual(
            preflight["reviewed_p1b_archived_implementation_sha256"][
                "scripts/research_d6/p1b/run_experiment.py"
            ],
            "af3f221e885a51aa14d9adbdbde8c8ddf30b163713730e58329460cf7fbce54e",
        )
        self.assertEqual(
            set(preflight["p1b_implementation_sha256"]), {"run_experiment.py", "core.py"}
        )

    def test_wrong_branch_and_non_ancestor_are_rejected(self):
        branch, head = git_identity()
        with self.assertRaisesRegex(RuntimeError, "unexpected branch"):
            validate_revision_policy("wrong-branch", head, lambda _a, _d: True)
        with self.assertRaisesRegex(RuntimeError, "is not an ancestor"):
            validate_revision_policy(branch, "0" * 40, lambda _a, _d: False)

    def test_incompatible_source_fingerprint_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "input manifest fingerprint mismatch"):
            require_fingerprint("0" * 64, INPUT_MANIFEST_SHA, "P1-A input manifest")

    def test_public_product_aggregate_matches_original_metrics(self):
        self.assertEqual(audit.verify_published_aggregate(), 432)
        rows = audit.build_public_aggregate()
        self.assertEqual(len(rows), 432)
        self.assertEqual(set(rows[0]), set(audit.PUBLIC_COLUMNS))
        self.assertFalse(
            {"image_id", "sample_key", "attribute_state", "binary_target", "score", "prediction"}
            & set(rows[0])
        )

    def test_logged_score_diagnostic_reproduces_global_threshold_transition(self):
        diagnostic = audit.build_score_diagnostics()
        self.assertEqual(len(diagnostic["summary"]), 60)
        self.assertEqual(len(diagnostic["per_image"]), 1260)
        self.assertEqual(len(diagnostic["products"]), 270)
        w0 = [r for r in diagnostic["summary"]
              if r["protocol"] == "W" and r["checkpoint"] == 0]
        w3 = [r for r in diagnostic["summary"]
              if r["protocol"] == "W" and r["checkpoint"] == 3]
        x12 = [r for r in diagnostic["summary"]
               if r["protocol"] == "X" and r["checkpoint"] == 12]
        self.assertEqual({r["predicted_positive"] for r in w0}, {21})
        self.assertEqual({r["predicted_positive"] for r in w3}, {0})
        self.assertEqual({r["predicted_positive"] for r in x12}, {21})
        self.assertEqual({r["correct_to_wrong_from_previous_selected_checkpoint"] for r in w3}, {14})
        self.assertEqual({r["wrong_to_correct_from_previous_selected_checkpoint"] for r in w3}, {7})

    def test_original_p1b_result_files_retain_pinned_hashes(self):
        hashes = audit.verify_original_run_artifacts()
        self.assertEqual(set(hashes), set(audit.ORIGINAL_RUN_FILE_SHA256))
        self.assertEqual(hashes, audit.ORIGINAL_RUN_FILE_SHA256)


if __name__ == "__main__":
    unittest.main()
