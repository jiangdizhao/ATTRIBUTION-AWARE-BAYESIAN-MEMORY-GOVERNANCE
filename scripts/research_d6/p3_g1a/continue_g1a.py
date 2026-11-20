"""Continue the same authorized G1A run using verified canonical caches."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import g1a_common as c
from run_g1a import save_public_reports

PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")


def run() -> dict:
    started = time.monotonic()
    identity_path = c.OUT / "run_identity.json"
    if not c.OUT.is_dir() or not identity_path.is_file():
        raise RuntimeError("NO_PRIOR_G1A_RUN_TO_CONTINUE")
    if (c.OUT / "run_summary.json").exists() or (c.OUT / "continuation_summary.json").exists():
        raise FileExistsError("G1A_ALREADY_FINALIZED; preserve outputs")
    git = c.current_git()
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (identity.get("authorization") != c.AUTHORIZATION or identity.get("head") != git["head"]
            or identity.get("whitelist_sha256") != c.WHITELIST_SHA
            or identity.get("selection_receipt_sha256") != c.SELECTION_SHA
            or c.sha_file(c.OUT / "authorized_images.csv") != identity.get("authorized_images_manifest_sha256")
            or c.sha_file(c.OUT / "stress_input_manifest.csv") != identity.get("stress_input_manifest_sha256")):
        raise RuntimeError("PRIOR_G1A_INPUT_IDENTITY_MISMATCH")
    prior = json.loads((c.OUT / "extraction_summary.json").read_text(encoding="utf-8"))
    if (prior.get("sha_success") != 216 or prior.get("canonical_success") != 216
            or prior.get("feature_success") != 0 or prior.get("new_attribute_labels_read") is not False
            or prior.get("gt_masks_read") is not False):
        raise RuntimeError("PRIOR_EXTRACTION_NOT_RESUMABLE")
    failures = c.read_csv(c.OUT / "failures.csv")
    if len(failures) != 216 or any("feature_interface" not in r["reason_code"] for r in failures):
        raise RuntimeError("PRIOR_FAILURES_NOT_SINGLE_KNOWN_ADAPTER_IMPORT")
    if ((c.OUT / "history").exists() and any((c.OUT / "history").iterdir())
            and not (c.OUT / "history" / "history_validation.json").is_file()):
        raise FileExistsError("HISTORY_OUTPUT_EXISTS; inspect before continuation")
    if ((c.OUT / "unlabeled_scores.csv").exists() or (c.OUT / "selection_summary.json").exists()) \
            and not (c.OUT / "selection_summary.json").is_file():
        raise FileExistsError("SELECTOR_OUTPUT_ALREADY_EXISTS")

    log_rows = []
    steps = [
        (c.OUT / "extraction_summary_retry_01.json",
         [str(PYTHON), str(HERE / "resume_features.py")]),
        (c.OUT / "history" / "history_validation.json",
         [str(PYTHON), str(HERE / "validate_history.py"), "--output", str(c.OUT / "history")]),
        (c.OUT / "selection_summary.json",
         [str(PYTHON), str(HERE / "select_stress.py"), "--output", str(c.OUT),
          "--feature-manifest", "feature_manifest_retry_01.csv"]),
    ]
    for i, (completion_marker, command) in enumerate(steps, 1):
        if completion_marker.is_file():
            log_rows.append({"script": Path(command[1]).name, "status": "ALREADY_COMPLETE_VERIFIED_BY_OUTPUT"})
            continue
        child_start = time.monotonic()
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                              timeout=max(1, c.MAX_RUNTIME_SECONDS - int(time.monotonic() - started)))
        log_path = c.OUT / f"continuation_child_{i:02d}.log"
        if log_path.exists():
            raise FileExistsError("CONTINUATION_LOG_EXISTS")
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
        log_rows.append({"script": Path(command[1]).name, "returncode": proc.returncode,
                         "elapsed_seconds": time.monotonic() - child_start,
                         "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
                         "stderr_sha256": hashlib.sha256(proc.stderr.encode()).hexdigest()})
        if proc.returncode:
            failure = {"status": "P3_G1A_PARTIAL", "script": Path(command[1]).name,
                       "returncode": proc.returncode,
                       "reason_code": "CONTINUATION_CHILD_FAILED", "elapsed_seconds": time.monotonic() - started,
                       "new_attribute_labels_read": False, "gt_masks_read": False,
                       "protected_storage_accessed": False, "training_or_update": False}
            c.write_json_once(c.OUT / "continuation_failure.json", failure)
            raise RuntimeError("CONTINUATION_CHILD_FAILED")
        c.check_output_budget(c.OUT)

    retry = json.loads((c.OUT / "extraction_summary_retry_01.json").read_text(encoding="utf-8"))
    history = json.loads((c.OUT / "history" / "history_validation.json").read_text(encoding="utf-8"))
    selection = json.loads((c.OUT / "selection_summary.json").read_text(encoding="utf-8"))
    roles = c.read_csv(c.OUT / "development_roles.csv")
    input_rows = c.read_csv(c.OUT / "authorized_images.csv")
    stress_input = c.read_csv(c.OUT / "stress_input_manifest.csv")
    product_by_slot = {r["product_slot"]: input_rows[i]["product"]
                       for i, r in enumerate(stress_input)}
    if len({r["opaque_image_id"] for r in input_rows}) != 216 or len(roles) % 2:
        raise RuntimeError("FINAL_ROLE_SCOPE_INVALID")
    groups = {(r["product_slot"], r["opaque_group_id"], r["study_role"]) for r in roles}
    role_counts = {}
    for r in roles:
        key = (r["product_slot"], r["opaque_group_id"], r["study_role"])
        role_counts[key] = role_counts.get(key, 0) + 1
    if any(value != 2 for value in role_counts.values()):
        raise RuntimeError("FINAL_ROLE_GROUP_VIEW_SPLIT")
    full = (retry["feature_success"] == 216 and retry["feature_failure_count"] == 0
            and history["status"] == "PASS" and history["historical_anchor_count"] == 5
            and selection["status"] == "PASS" and len(groups) == 60 and len(roles) == 120
            and len(selection["per_product_bin_diagnostics"]) == 6
            and all(x["status"] == "PASS" for x in selection["per_product_bin_diagnostics"]))
    status = "P3_G1A_READY_FOR_ANNOTATION_APPROVAL" if full else "P3_G1A_PARTIAL"

    exposure = c.read_csv(c.OUT / "exposure_ledger.csv")
    sha_by_product = {}
    for r in exposure:
        sha_by_product.setdefault(r["product"], 0)
        sha_by_product[r["product"]] += int(r["read_success"] == "YES")
    feature_ids = {r["opaque_image_id"] for r in c.read_csv(c.OUT / "feature_manifest_retry_01.csv")}
    feature_by_product = {name: 0 for name in sha_by_product}
    for r in input_rows:
        feature_by_product[r["product"]] += int(r["opaque_image_id"] in feature_ids)
    rep_groups_by_product = {}
    selected_groups_by_product = {}
    for r in roles:
        product = product_by_slot[r["product_slot"]]
        selected_groups_by_product.setdefault(product, set()).add(r["opaque_group_id"])
        if r["candidate_kind"] == "REPRESENTATIVE_CANDIDATE":
            rep_groups_by_product.setdefault(product, set()).add(r["opaque_group_id"])
    diagnostic = {product_by_slot[r["product_slot"]]: r for r in selection["per_product_bin_diagnostics"]}
    product_rows, bin_rows = [], []
    for product in sorted(sha_by_product):
        d = diagnostic.get(product, {})
        product_rows.append({"product": product, "sha_success": sha_by_product[product],
                             "feature_success": feature_by_product[product],
                             "representative_groups": len(rep_groups_by_product.get(product, set())),
                             "stress_bins": d.get("status", "NOT_RUN"),
                             "development_groups": len(selected_groups_by_product.get(product, set()))})
        bin_rows.append({"product": product, "status": d.get("status", "NOT_RUN"),
                         "bin_counts": d.get("bin_counts", {})})
    dcal = sum(role == "P3_G1A_D_CAL" for _, _, role in groups)
    deval = sum(role == "P3_G1A_D_EVAL" for _, _, role in groups)
    result = {"status": status, "branch": git["branch"], "head": git["head"],
              "authorized_views": 216, "sha_success": prior["sha_success"],
              "sha_failure": prior["sha_failure"], "feature_success": retry["feature_success"],
              "history_status": history["status"], "h12_max_abs_score_delta": history["max_abs_score_delta"],
              "anchor_count": history["historical_anchor_count"], "development_groups": len(groups),
              "development_views": len(roles), "d_cal_groups": dcal, "d_eval_groups": deval,
              "products_with_all_stress_bins": sum(x["status"] == "PASS" for x in selection["per_product_bin_diagnostics"]),
              "private_output_relpath": str(c.OUT.relative_to(ROOT)).replace("\\", "/"),
              "output_bytes": c.check_output_budget(c.OUT),
              "gpu_peak_allocated_bytes": retry["peak_allocated_bytes"],
              "elapsed_seconds": time.monotonic() - started + prior["resources"]["elapsed_seconds"],
              "new_attribute_labels_read": False, "gt_masks_read": False,
              "protected_storage_accessed": False, "training_or_update": False}
    final_payload = {**result, "continuation": "verified-canonical-cache-only",
                     "first_attempt_feature_import_failure_preserved": True,
                     "children": log_rows, "retry_feature_manifest": retry["feature_manifest"],
                     "retry_feature_manifest_sha256": retry["feature_manifest_sha256"],
                     "selection_summary_sha256": c.sha_file(c.OUT / "selection_summary.json"),
                     "source_images_reopened_during_continuation": False}
    c.write_json_once(c.OUT / "run_summary.json", final_payload)
    save_public_reports(result, product_rows, bin_rows)
    c.write_json_once(c.OUT / "continuation_summary.json", final_payload)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
