"""Continue a failed G1A feature stage from verified canonical bundles only.

This recovery path does not reopen source images. It preserves all first-attempt
files and writes separately named receipts/manifests for the retry.
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1a_common as c
from extract_features import CHECKPOINT_SHA, SOURCE_TREE_SHA, FEATURE_SCHEMA, load_model, forward_one, check_repeat


def run(output: Path) -> dict:
    torch = None
    import torch as _torch
    import torchvision
    from PIL import __version__ as pillow_version
    if (_torch.__version__ != "2.0.1+cu117" or torchvision.__version__ != "0.15.2+cu117"
            or np.__version__ != "1.24.1" or pillow_version != "9.3.0"):
        raise RuntimeError("DINOV2_REGISTERED_RUNTIME_MISMATCH")
    torch = _torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from extract_features import gpu_status
    gpu_before = gpu_status()
    total = torch.cuda.get_device_properties("cuda:0").total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 12 * 1024**3 / total), device=torch.device("cuda:0"))

    entries = c.read_csv(output / "authorized_images.csv")
    initial_exposures = c.read_csv(output / "exposure_ledger.csv")
    canonical_manifest = c.read_csv(output / "canonical_manifest.csv")
    if len(entries) != 216 or len(initial_exposures) != 216 or len(canonical_manifest) != 216:
        raise RuntimeError("RESUME_CANONICAL_SCOPE_CARDINALITY")
    exposure_by_id = {r["opaque_image_id"]: r for r in initial_exposures}
    canonical_by_id = {r["opaque_image_id"]: r for r in canonical_manifest}
    stress_rows = c.read_csv(output / "stress_input_manifest.csv")
    stress_by_id = {r["opaque_image_id"]: r for r in stress_rows}
    if set(exposure_by_id) != {r["opaque_image_id"] for r in entries} or set(canonical_by_id) != set(exposure_by_id):
        raise RuntimeError("RESUME_CANONICAL_IDENTITY_SET")
    if set(stress_by_id) != set(exposure_by_id):
        raise RuntimeError("RESUME_STRESS_INPUT_IDENTITY_SET")
    if any(r["read_success"] != "YES" or r["decoded"] != "YES" for r in initial_exposures):
        raise RuntimeError("RESUME_REQUIRES_COMPLETE_CANONICAL_READ")

    # Verify all prior canonical arrays and their source SHA binding before any
    # feature inference. Only opaque IDs and source hashes are read here.
    sys.path.insert(0, str(c.ROOT / "scripts/research_d6/p2/e2a1"))
    from preprocess import canonical_array_sha256
    features_input = []
    for row in entries:
        image_id = row["opaque_image_id"]
        expected = canonical_by_id[image_id]
        exposure = exposure_by_id[image_id]
        base = output / "canonical" / "items" / image_id
        record_path = base / "record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("identity", {}).get("opaque_image_id") != image_id
                or record.get("identity", {}).get("source_sha256") != exposure["source_sha256"]
                or exposure["source_sha256"] != expected["source_sha256"]
                or record.get("metadata", {}).get("schema") != "d6.e2a1.canonical_rgb_letterbox.v1"):
            raise RuntimeError("RESUME_CANONICAL_IDENTITY_MISMATCH")
        arrays = {}
        for name in ("canonical_tensor", "valid_patch_mask_dinov2"):
            spec = record["arrays"][name]
            path = base / spec["file"]
            if c.sha_file(path) != spec["file_sha256"]:
                raise RuntimeError("RESUME_CANONICAL_FILE_SHA_MISMATCH")
            arr = np.load(path, allow_pickle=False)
            if c.sha_array(arr) != spec["array_sha256"]:
                raise RuntimeError("RESUME_CANONICAL_ARRAY_SHA_MISMATCH")
            arrays[name] = arr
        tensor = np.ascontiguousarray(arrays["canonical_tensor"][None, ...], dtype=np.float32)
        mask = np.ascontiguousarray(arrays["valid_patch_mask_dinov2"], dtype=np.bool_)
        if tensor.shape != (1, 3, 448, 448) or mask.shape != (1024,) or not np.isfinite(tensor).all():
            raise RuntimeError("RESUME_CANONICAL_SHAPE_OR_VALUE")
        if canonical_array_sha256(tensor) != expected["canonical_array_sha256"]:
            raise RuntimeError("RESUME_CANONICAL_HASH_BINDING")
        features_input.append((row, exposure, tensor, mask))

    model, runtime = load_model(torch, np, sys)
    runtime["initial_gpu"] = gpu_before
    runtime["resume_source"] = "verified_canonical_bundles_only"
    feature_start = time.monotonic()
    feature_rows, feature_failures, completion_rows = [], [], []
    smoke_metrics = None
    for index, (row, exposure, tensor, mask) in enumerate(features_input):
        image_id = row["opaque_image_id"]
        try:
            arrays = forward_one(model, tensor, mask, torch)
            if smoke_metrics is None:
                repeated = forward_one(model, tensor, mask, torch)
                smoke_metrics = check_repeat(arrays, repeated)
            identity_feature = {"study": "LP-D6-P3-G1A", "opaque_image_id": image_id,
                                "source_sha256": exposure["source_sha256"],
                                "canonical_array_sha256": canonical_array_sha256(tensor),
                                "model": "dinov2_vitl14_reg", "checkpoint_sha256": CHECKPOINT_SHA,
                                "source_tree_sha256": SOURCE_TREE_SHA, "feature_interface": FEATURE_SCHEMA}
            metadata = {"token_normalization": "each CLS/register/patch independently L2-normalized FP32",
                        "valid_patch_count": int(mask.sum()), "patch_count": 1024,
                        "feature_shapes": {k: list(v.shape) for k, v in arrays.items()}}
            record, reused = __import__("storage").write_bundle_atomic(
                output / "dinov2" / "items" / image_id, arrays, identity_feature, metadata)
            record_sha = c.sha_file(output / "dinov2" / "items" / image_id / "record.json")
            completion_rows.append({"opaque_image_id": image_id, "source_sha256": exposure["source_sha256"],
                                    "feature_record_sha256": record_sha, "status": "COMPLETE",
                                    "feature_bundle_reused": "YES" if reused else "NO",
                                    "completed_utc": datetime.now(timezone.utc).isoformat()})
            feature_rows.append({"opaque_image_id": image_id, "opaque_group_id": row["opaque_group_id"],
                                 "product_slot": stress_by_id[image_id]["product_slot"],
                                 "view": row["view"], "candidate_kind": row["candidate_kind"],
                                 "source_sha256": exposure["source_sha256"],
                                 "canonical_array_sha256": canonical_array_sha256(tensor),
                                 "record_sha256": record_sha,
                                 "cls_array_sha256": record["arrays"]["cls_unit"]["array_sha256"],
                                 "register_array_sha256": record["arrays"]["register_unit"]["array_sha256"],
                                 "patch_array_sha256": record["arrays"]["patch_unit"]["array_sha256"],
                                 "mask_array_sha256": record["arrays"]["valid_patch_mask"]["array_sha256"],
                                 "valid_patch_count": int(mask.sum()), "dtype": "float32",
                                 "backbone": "dinov2_vitl14_reg", "checkpoint_sha256": CHECKPOINT_SHA,
                                 "feature_interface": FEATURE_SCHEMA})
            c.check_output_budget(output)
        except Exception as exc:
            feature_failures.append({"opaque_image_id": image_id,
                                     "reason_code": type(exc).__name__ + ":" + str(exc).split(":")[0]})
    feature_manifest_path = output / "feature_manifest_retry_01.csv"
    exposure_path = output / "exposure_ledger_retry_01.csv"
    failure_path = output / "feature_failures_retry_01.csv"
    completion_path = output / "item_feature_receipts_retry_01.csv"
    c.write_csv_once(feature_manifest_path, feature_rows)
    c.write_csv_once(failure_path, feature_failures, ["opaque_image_id", "reason_code"])
    c.write_csv_once(completion_path, completion_rows)
    by_id = {r["opaque_image_id"]: r for r in feature_rows}
    exposure_rows = []
    for row in initial_exposures:
        updated = dict(row)
        f = by_id.get(row["opaque_image_id"])
        updated["used_for_model"] = "YES" if f else "NO"
        updated["feature_record_sha256"] = f["record_sha256"] if f else ""
        exposure_rows.append(updated)
    exposure_sha = c.write_csv_once(exposure_path, exposure_rows)
    gpu_after = gpu_status()
    result = {"status": "P3_G1A_FEATURES_PASS" if len(feature_rows) == 216 and not feature_failures else "P3_G1A_PARTIAL",
              "input_views": 216, "source_images_reopened": False,
              "canonical_bundles_verified": 216, "feature_success": len(feature_rows),
              "feature_failure_count": len(feature_failures),
              "feature_manifest": feature_manifest_path.name, "feature_manifest_sha256": c.sha_file(feature_manifest_path),
              "exposure_manifest_sha256": exposure_sha, "feature_failures_sha256": c.sha_file(failure_path),
              "completion_receipts_sha256": c.sha_file(completion_path), "checkpoint_sha256": CHECKPOINT_SHA,
              "source_tree_sha256": SOURCE_TREE_SHA, "runtime": runtime,
              "gpu_initial": gpu_before, "gpu_final": gpu_after,
              "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
              "output_bytes": c.check_output_budget(output), "elapsed_seconds": time.monotonic() - feature_start,
              "smoke_repeat_metrics": smoke_metrics, "new_attribute_labels_read": False,
              "defect_type_read": False, "gt_masks_read": False, "training_or_update": False}
    c.write_json_once(output / "extraction_summary_retry_01.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(c.OUT), ensure_ascii=False, indent=2))
