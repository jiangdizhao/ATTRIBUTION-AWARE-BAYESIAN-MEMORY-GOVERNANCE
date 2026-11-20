"""Create the private, immutable G1B queue and empty role-separated templates."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from scripts.research_d6.p3_g1b import g1b_core as core

ROOT = Path(__file__).resolve().parents[3]
G1A = ROOT / core.G1A_ROOT_REL
RUN = ROOT / core.RUN_REL
PUBLIC_DIR = ROOT / "docs/research_d6/p3_g1b"
TEST_FILE = ROOT / "tests/research_d6/test_p3_g1b_annotator.py"
DEFINITION_SOURCE = ROOT / "research_pilot/LP_W1_D4B/attribute_definition_card_v1_1.md"
CODE_FILES = [
    ROOT / "scripts/research_d6/p3_g1b/g1b_core.py",
    ROOT / "scripts/research_d6/p3_g1b/prepare_run.py",
    ROOT / "scripts/research_d6/p3_g1b/annotate.py",
    TEST_FILE,
]


def _csv_fields(rows: list[dict], fallback: list[str]) -> list[str]:
    return list(rows[0]) if rows else fallback


def _verify_git() -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != "8108ba33c3107224e1aa7bb203a584132b5d9241" or branch != "stage3-eg-famar-execution":
        raise core.IntegrityError("G1B_EXPECTED_GIT_IDENTITY_MISMATCH")
    return {"head": head, "branch": branch}


def _verify_g1a_status() -> dict:
    receipt = json.loads((G1A / "final_execution_receipt.json").read_text(encoding="utf-8"))
    selection = json.loads((G1A / "selection_summary.json").read_text(encoding="utf-8"))
    run_summary = json.loads((G1A / "run_summary.json").read_text(encoding="utf-8"))
    if (receipt.get("status") != "P3_G1A_READY_FOR_ANNOTATION_APPROVAL"
            or selection.get("status") != "PASS"
            or run_summary.get("training_or_update") is not False
            or receipt.get("new_attribute_labels_read") is not False
            or receipt.get("gt_defect_masks_read") is not False):
        raise core.IntegrityError("G1A_RECEIPT_STATUS_INVALID")
    if receipt.get("development_roles_sha256") not in (None, core.EXPECTED_G1A_ROLES_SHA256):
        raise core.IntegrityError("G1A_RECEIPT_ROLE_HASH_MISMATCH")
    code_root = ROOT / "scripts/research_d6/p3_g1a"
    for filename, expected in receipt.get("code_sha256", {}).items():
        code_path = code_root / filename
        if not code_path.is_file() or core.sha256_file(code_path) != expected:
            raise core.IntegrityError("G1A_CODE_SHA_MISMATCH")
    if selection.get("development_roles_sha256") != core.sha256_file(G1A / "development_roles.csv"):
        raise core.IntegrityError("G1A_SELECTION_ROLE_HASH_MISMATCH")
    if selection.get("unlabeled_scores_sha256") != core.sha256_file(G1A / "unlabeled_scores.csv"):
        raise core.IntegrityError("G1A_SELECTION_SCORE_HASH_MISMATCH")
    if selection.get("stress_selection_sha256") != core.sha256_file(G1A / "stress_selection.csv"):
        raise core.IntegrityError("G1A_SELECTION_STRESS_HASH_MISMATCH")
    if run_summary.get("selection_summary_sha256") != core.sha256_file(G1A / "selection_summary.json"):
        raise core.IntegrityError("G1A_SELECTION_SUMMARY_HASH_MISMATCH")
    return {
        "receipt": receipt,
        "selection_summary": selection,
        "run_summary": run_summary,
    }


def prepare() -> dict:
    git = _verify_git()
    if RUN.exists():
        raise FileExistsError("G1B_PRIVATE_RUN_OUTPUT_ALREADY_EXISTS")
    if PUBLIC_DIR.exists():
        raise FileExistsError("G1B_PUBLIC_REPORT_DIRECTORY_ALREADY_EXISTS")
    if not TEST_FILE.is_file():
        raise core.IntegrityError("G1B_TEST_FILE_MISSING")
    core.require_git_ignored(ROOT, core.RUN_REL)

    g1a_status = _verify_g1a_status()
    roles, mapping, _, input_summary = core.load_g1a_metadata(G1A)
    queue, repeats, queue_summary = core.build_queue(roles, mapping)
    core.validate_queue(queue, mapping)

    # Metadata-only preparation: create private output root and restrict its
    # ACL before any manifests or templates are written.
    RUN.mkdir(parents=True, exist_ok=False)
    core.restrict_windows_acl(RUN)
    (RUN / "D-cal").mkdir()
    (RUN / "D-eval").mkdir()
    if not DEFINITION_SOURCE.is_file():
        raise core.IntegrityError("FROZEN_ATTRIBUTE_DEFINITION_MISSING")
    definition_bytes = DEFINITION_SOURCE.read_bytes()
    definition_sha = core.sha256_bytes(definition_bytes)
    core.atomic_write_bytes(RUN / "attribute_definition_card_v1_1.md", definition_bytes)

    image_fields = [
        "opaque_image_id", "opaque_group_id", "product", "product_slot", "study_role",
        "view", "candidate_kind", "source_image_path", "source_sha256", "source_size_bytes",
        "exposure_status", "g1a_read_success", "g1a_final_path_verified",
    ]
    image_rows = [mapping[row["opaque_image_id"]] for row in roles]
    image_sha = core.write_csv_once(RUN / "image_mapping.csv", image_rows, image_fields)
    queue_sha = core.write_csv_once(
        RUN / "annotation_queue.csv", queue,
        ["presentation_order", "presentation_id", "opaque_image_id", "opaque_group_id", "product",
         "product_slot", "study_role", "view", "candidate_kind", "presentation_kind",
         "first_presentation_id", "queue_nonce"],
    )
    repeat_sha = core.write_csv_once(
        RUN / "blind_repeat_mapping.csv", repeats,
        ["opaque_image_id", "opaque_group_id", "product", "product_slot", "study_role", "view",
         "first_presentation_id", "repeat_presentation_id", "first_order", "repeat_order"],
    )
    repeat_group_sha = core.write_csv_once(
        RUN / "blind_repeat_groups.csv", queue_summary["repeated_groups"],
        ["product", "opaque_group_id", "study_role", "view_count", "selection_domain"],
    )
    selected_exposure = []
    g1a_exposure = {row["opaque_image_id"]: row for row in core.read_csv(G1A / "exposure_ledger.csv")}
    for item in image_rows:
        exposure = g1a_exposure[item["opaque_image_id"]]
        selected_exposure.append({
            "opaque_image_id": item["opaque_image_id"],
            "opaque_group_id": item["opaque_group_id"],
            "product": item["product"],
            "study_role": item["study_role"],
            "view": item["view"],
            "g1a_source_read_status": "SOURCE_READ_AND_SHA_VERIFIED_DURING_G1A",
            "new_study_exposure_status": "NEW_STUDY_DEV_EXPOSED",
            "g1b_human_presentation_status": "NOT_PRESENTED",
            "source_sha256": exposure["source_sha256"],
            "old_protected_intersection": "UNKNOWN",
        })
    exposure_sha = core.write_csv_once(
        RUN / "exposure_ledger.csv", selected_exposure,
        ["opaque_image_id", "opaque_group_id", "product", "study_role", "view",
         "g1a_source_read_status", "new_study_exposure_status", "g1b_human_presentation_status",
         "source_sha256", "old_protected_intersection"],
    )
    core.write_csv_once(RUN / "D-cal" / "annotations.csv", [], core.ANNOTATION_FIELDS)
    core.write_csv_once(RUN / "D-eval" / "annotations.csv", [], core.ANNOTATION_FIELDS)
    (RUN / "D-cal" / "annotation_corrections.jsonl").write_bytes(b"")
    (RUN / "D-eval" / "annotation_corrections.jsonl").write_bytes(b"")
    core.write_json_once(RUN / "session_timing.json", {"schema": "lp.d6.p3.g1b.session-timing.v1", "presentations": {}})

    input_hashes = {
        name: core.sha256_file(G1A / name)
        for name in [
            "development_roles.csv", "authorized_images.csv", "exposure_ledger.csv",
            "unlabeled_scores.csv", "stress_selection.csv", "final_execution_receipt.json",
            "selection_summary.json", "run_summary.json",
        ]
    }
    code_hashes = {str(path.relative_to(ROOT).as_posix()): core.sha256_file(path) for path in CODE_FILES}
    private_receipt = {
        "schema": "lp.d6.p3.g1b.input-freeze-receipt.v1",
        "status": "PREP_READY_NO_REAL_ANNOTATION_AUTHORIZATION",
        "git_identity": git,
        "g1a_input_directory": core.G1A_ROOT_REL.as_posix(),
        "g1a_input_sha256": input_hashes,
        "g1a_summary": input_summary,
        "g1a_execution_status": g1a_status["receipt"]["status"],
        "study_split": {"D-cal": {"views": 40, "groups": 20, "products": 2}, "D-eval": {"views": 80, "groups": 40, "products": 4}},
        "queue": {
            "unique_images": 120,
            "groups": 60,
            "first_presentations": 120,
            "repeat_groups": 12,
            "repeat_presentations": 24,
            "total_presentations": 144,
            "image_mapping_sha256": image_sha,
            "annotation_queue_sha256": queue_sha,
            "blind_repeat_mapping_sha256": repeat_sha,
            "blind_repeat_groups_sha256": repeat_group_sha,
            "exposure_ledger_sha256": exposure_sha,
            **{k: v for k, v in queue_summary.items() if k != "repeated_groups"},
        },
        "attribute": "surface_abrasion_visible",
        "attribute_definition_version": "v1.1",
        "attribute_definition_source_sha256": definition_sha,
        "attribute_definition_snapshot_sha256": core.sha256_file(RUN / "attribute_definition_card_v1_1.md"),
        "allowed_states": sorted(core.VALID_STATES),
        "annotation_authorized": False,
        "real_image_bytes_opened_during_g1b_preparation": 0,
        "new_attribute_labels_read": False,
        "dcal_deval_outputs_separate": True,
        "deval_sealed_from_dcal_loader": True,
        "old_protected_intersection": "UNKNOWN",
        "code_sha256": code_hashes,
    }
    receipt_sha = core.write_json_once(RUN / "input_freeze_receipt.json", private_receipt)
    core.write_json_once(RUN / "annotation_authorization.template.json", {
        "schema": "lp.d6.p3.g1b.annotation-authorization.v1",
        "status": "NOT_AUTHORIZED",
        "approved_by": "",
        "approved_at_utc": "",
        "authorization_reference": "",
        "input_freeze_receipt_sha256": receipt_sha,
        "scope": "G1B_D_CAL_AND_D_EVAL_MANUAL_PRESENTATION_ONLY",
    })

    # Sanity-check access-control on the newly created private tree before
    # recording preparation success.
    import subprocess
    acl_text = subprocess.check_output(["icacls", str(RUN)], text=True, errors="replace")
    for forbidden in ("Authenticated Users", "BUILTIN\\Users", "S-1-5-11", "S-1-5-32-545"):
        if forbidden.casefold() in acl_text.casefold():
            raise core.IntegrityError("PRIVATE_OUTPUT_ACL_IS_BROAD")

    return {
        "git": git,
        "summary": input_summary,
        "queue": queue_summary,
        "receipt_sha256": receipt_sha,
        "queue_sha256": queue_sha,
        "repeat_mapping_sha256": repeat_sha,
        "exposure_sha256": exposure_sha,
        "roles_by_product_role": {
            f"{slot}:{role}": count
            for (slot, role), count in Counter((mapping[row["opaque_image_id"]]["product_slot"], row["study_role"]) for row in roles).items()
        },
    }


if __name__ == "__main__":
    try:
        print(json.dumps(prepare(), sort_keys=True, indent=2))
    except Exception as exc:
        # Do not echo private paths or source identifiers in terminal diagnostics.
        print(f"G1B_PREPARATION_FAILED:{type(exc).__name__}:{exc}", file=sys.stderr)
        raise
