"""Hash, canonicalize and extract frozen DINOv2 tokens for the fixed G1A list."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g1a_common import OUT, ROOT, check_output_budget, read_csv, sha_array, sha_file, write_csv_once, write_json_once
from secure_io import read_and_hash_whitelisted

E2A1_DIR = ROOT / "scripts/research_d6/p2/e2a1"
E2A1_OUT = ROOT / "outputs/research_d6/p2/e2a1"
CHECKPOINT_SHA = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
SOURCE_TREE_SHA = "7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c"
PREPROCESS_SCHEMA = "d6.e2a1.canonical_rgb_letterbox.v1"
FEATURE_SCHEMA = "d6.e2.shared_spatial_tokens.v1"


def gpu_status() -> dict:
    import subprocess
    proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
                           "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=20)
    lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    if len(lines) != 1:
        raise RuntimeError("GPU_COUNT_MISMATCH")
    parts = [x.strip() for x in lines[0].split(",")]
    state = {"name": parts[0], "total_mib": int(parts[1]), "free_mib": int(parts[2]),
             "used_mib": int(parts[3]), "utilization_percent": int(parts[4])}
    if state["name"] != "NVIDIA GeForce RTX 3090 Ti" or state["free_mib"] < 10240:
        raise RuntimeError("GPU_RESOURCE_GATE")
    return state


def canonicalize(blob: bytes):
    sys.path.insert(0, str(E2A1_DIR))
    from PIL import Image, ImageOps
    from preprocess import image_to_canonical
    with Image.open(BytesIO(blob)) as source:
        source.load()
        original = {"format": source.format, "mode": source.mode, "size": list(source.size),
                    "exif_orientation": int(source.getexif().get(274, 1))}
        oriented = ImageOps.exif_transpose(source)
        if oriented.mode != "RGB":
            raise ValueError("NON_RGB_SOURCE")
        tensor, meta = image_to_canonical(oriented)
    return tensor, meta, original


def load_model(torch, np, sys):
    sys.path.insert(0, str(ROOT / "scripts"))
    from research_d6.p1a import extract_features as p1a
    from contracts import DINO2_MODEL, FEATURE_INTERFACE_SCHEMA, DINO2_CHECKPOINT_SHA256, DINO2_SOURCE_TREE_SHA256
    from storage import sha256_file
    checkpoint_sha = p1a.sha256_file(p1a.CHECKPOINT)
    source_sha, source_count, source_bytes = p1a.tree_sha256(p1a.HUB_SOURCE)
    if (checkpoint_sha, source_sha) != (CHECKPOINT_SHA, SOURCE_TREE_SHA):
        raise RuntimeError("FROZEN_DINOV2_IDENTITY_MISMATCH")
    if (DINO2_MODEL != "dinov2_vitl14_reg" or DINO2_CHECKPOINT_SHA256 != CHECKPOINT_SHA
            or DINO2_SOURCE_TREE_SHA256 != SOURCE_TREE_SHA or FEATURE_INTERFACE_SCHEMA != FEATURE_SCHEMA):
        raise RuntimeError("E2A1_CONTRACT_MISMATCH")
    sensor = p1a.load_frozen_sensor(torch, device="cpu")
    sensor.cleanup()
    model = sensor.backbone.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        if p.dtype != torch.float32:
            raise RuntimeError("MODEL_PARAMETER_NOT_FP32")
    model = model.to("cuda:0")
    model.eval()
    runtime = {"model": DINO2_MODEL, "checkpoint_sha256": checkpoint_sha,
               "source_tree_sha256": source_sha, "source_tree_files": source_count,
               "source_tree_bytes": source_bytes, "torch": torch.__version__,
               "torchvision": __import__("torchvision").__version__, "numpy": np.__version__,
               "python": platform.python_version(), "device": "cuda:0", "precision": "FP32",
               "tf32": False, "autocast": False, "batch_size": 1,
               "feature_interface": FEATURE_SCHEMA}
    return model, runtime


def forward_one(model, tensor_np: np.ndarray, mask: np.ndarray, torch):
    # The shared token adapter is registered under p2/e2, not under e2a1.
    sys.path.insert(0, str(ROOT / "scripts/research_d6/p2/e2a"))
    from feature_interface import dinov2_to_shared_interface
    tensor = torch.from_numpy(np.ascontiguousarray(tensor_np.copy())).to("cuda:0")
    if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("INPUT_TENSOR_NOT_FINITE_FP32")
    with torch.inference_mode():
        output = model.forward_features(tensor)
    raw = {name: output[name].detach().float().cpu().contiguous().numpy()
           for name in ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens")}
    result = dinov2_to_shared_interface(raw, mask)
    arrays = {"cls_unit": np.ascontiguousarray(result["cls_unit"][0], dtype=np.float32),
              "register_unit": np.ascontiguousarray(result["register_unit"][0], dtype=np.float32),
              "patch_unit": np.ascontiguousarray(result["patch_unit"][0], dtype=np.float32),
              "valid_patch_mask": np.ascontiguousarray(result["valid_patch_mask"][0], dtype=np.bool_)}
    expected = {"cls_unit": (1024,), "register_unit": (4, 1024), "patch_unit": (1024, 1024),
                "valid_patch_mask": (1024,)}
    for name, array in arrays.items():
        if array.shape != expected[name] or (array.dtype != np.bool_ if name == "valid_patch_mask" else array.dtype != np.float32):
            raise RuntimeError("FEATURE_SHAPE_OR_DTYPE_MISMATCH")
        if name != "valid_patch_mask" and not np.isfinite(array).all():
            raise RuntimeError("NONFINITE_FEATURE")
    return arrays


def check_repeat(first: dict, second: dict) -> dict:
    metrics = {}
    for name in ("cls_unit", "register_unit", "patch_unit"):
        delta = float(np.max(np.abs(first[name] - second[name])))
        cos = np.sum(first[name] * second[name], axis=-1)
        min_cos = float(np.min(cos))
        metrics[name] = {"max_abs_delta": delta, "min_cosine": min_cos}
        if delta > 1e-3 or min_cos < 0.999:
            raise RuntimeError("DINOV2_REPEAT_MISMATCH")
    return metrics


def run(output: Path) -> dict:
    started = time.monotonic()
    sys.path.insert(0, str(E2A1_DIR))
    from preprocess import canonical_array_sha256
    from storage import write_bundle_atomic
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise RuntimeError("DINOV2_RUNTIME_IMPORT_FAILED") from exc
    if (torch.__version__ != "2.0.1+cu117" or torchvision.__version__ != "0.15.2+cu117"
            or np.__version__ != "1.24.1" or __import__("PIL").__version__ != "9.3.0"):
        raise RuntimeError("DINOV2_REGISTERED_RUNTIME_MISMATCH")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    initial_gpu = gpu_status()
    total = torch.cuda.get_device_properties("cuda:0").total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 12 * 1024**3 / total), device=torch.device("cuda:0"))

    entries = read_csv(output / "authorized_images.csv")
    if not entries or len(entries) > 216 or len({r["opaque_image_id"] for r in entries}) != len(entries):
        raise RuntimeError("INPUT_ALLOWLIST_CARDINALITY")
    canonical_root = output / "canonical" / "items"
    feature_root = output / "dinov2" / "items"
    receipt_root = output / "item_receipts"
    canonical_rows, feature_rows, exposures, failures = [], [], [], []
    successful_canonical = []
    for index, row in enumerate(entries):
        exposure = {"opaque_image_id": row["opaque_image_id"], "opaque_group_id": row["opaque_group_id"],
                    "product": row["product"], "view": row["view"], "read_type": "SHA_AND_CANONICAL_DECODE",
                    "read_success": "NO", "source_sha256": "", "source_size_bytes": "", "file_id": "",
                    "last_write_filetime": "", "final_path_verified": "NO", "reparse_check": "NOT_RUN",
                    "decoded": "NO", "used_for_model": "NO", "preprocess_schema": PREPROCESS_SCHEMA,
                    "content_rect_xyxy": "", "valid_patch_count": "", "canonical_array_sha256": "",
                    "canonical_bundle_reused": "NO"}
        exposures.append(exposure)
        if time.monotonic() - started > 2 * 60 * 60:
            failures.append({"opaque_image_id": row["opaque_image_id"], "status": "FAILED_TIME_BUDGET",
                             "reason_code": "TIME_BUDGET_EXCEEDED"})
            continue
        source = None
        try:
            blob, source = read_and_hash_whitelisted(row["source_image_path"])
            exposure.update({"read_success": "YES", "source_sha256": source["sha256"],
                             "source_size_bytes": source["size_bytes"], "file_id": source["file_id"],
                             "last_write_filetime": source["last_write_filetime"],
                             "final_path_verified": "YES", "reparse_check": "PASS"})
            tensor, prep, source_meta = canonicalize(blob)
            if tensor.shape != (1, 3, 448, 448) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
                raise RuntimeError("CANONICAL_TENSOR_INVALID")
            if prep["schema"] != "d6.e2a1.canonical_rgb_letterbox.v1":
                raise RuntimeError("CANONICAL_SCHEMA_MISMATCH")
            canonical_hash = canonical_array_sha256(tensor)
            arrays = {"canonical_tensor": tensor[0],
                      "valid_patch_mask_dinov2": prep["mask_dinov2"],
                      "valid_patch_mask_dinov3": prep["mask_dinov3"]}
            identity = {"study": "LP-D6-P3-G1A", "opaque_image_id": row["opaque_image_id"],
                        "source_sha256": source["sha256"], "source_size_bytes": source["size_bytes"],
                        "preprocess_schema": prep["schema"], "preprocess_version": prep["preprocess_version"]}
            meta = {k: v for k, v in prep.items() if not k.startswith("mask_")}
            _, canonical_reused = write_bundle_atomic(canonical_root / row["opaque_image_id"], arrays, identity, meta)
            check_output_budget(output)
            exposure.update({"decoded": "YES", "preprocess_schema": prep["schema"],
                             "content_rect_xyxy": json.dumps(prep["content_rect_xyxy"], separators=(",", ":")),
                             "valid_patch_count": prep["valid_patch_counts"]["mask_dinov2"],
                             "canonical_array_sha256": canonical_hash,
                             "canonical_bundle_reused": "YES" if canonical_reused else "NO",
                             "used_for_model": "PENDING"})
            canonical_rows.append({"opaque_image_id": row["opaque_image_id"], "opaque_group_id": row["opaque_group_id"],
                                   "product": row["product"], "view": row["view"],
                                   "conditional_role": row["conditional_role"], "candidate_kind": row["candidate_kind"],
                                   "source_sha256": source["sha256"], "source_size_bytes": source["size_bytes"],
                                   "canonical_array_sha256": canonical_hash, "preprocess_schema": prep["schema"],
                                   "source_format": source_meta["format"], "source_mode": source_meta["mode"],
                                   "source_size": json.dumps(source_meta["size"], separators=(",", ":")),
                                   "canonical_bundle_reused": canonical_reused})
            successful_canonical.append((row, tensor, prep, source, identity, meta))
        except Exception as exc:
            failures.append({"opaque_image_id": row["opaque_image_id"], "status": "FAILED_IMAGE_OR_PATH",
                             "reason_code": type(exc).__name__ + ":" + str(exc).split(":")[0]})

    if not successful_canonical:
        raise RuntimeError("NO_CANONICAL_IMAGE_SUCCEEDED")
    model, runtime = load_model(torch, np, sys)
    runtime["initial_gpu"] = initial_gpu
    runtime["allocator_limit_bytes"] = 12 * 1024**3
    runtime["max_output_bytes"] = 3 * 1024**3
    feature_start = time.monotonic()
    smoke_metrics = None
    for i, (row, tensor, prep, source, identity, meta) in enumerate(successful_canonical):
        if time.monotonic() - started > 2 * 60 * 60:
            failures.append({"opaque_image_id": row["opaque_image_id"], "status": "FAILED_TIME_BUDGET",
                             "reason_code": "TIME_BUDGET_EXCEEDED_DURING_FEATURE_EXTRACTION"})
            continue
        try:
            mask = prep["mask_dinov2"]
            arrays = forward_one(model, tensor, mask, torch)
            if smoke_metrics is None:
                repeated = forward_one(model, tensor, mask, torch)
                smoke_metrics = check_repeat(arrays, repeated)
            identity_feature = {"study": "LP-D6-P3-G1A", "opaque_image_id": row["opaque_image_id"],
                                "source_sha256": source["sha256"], "canonical_array_sha256": canonical_array_sha256(tensor),
                                "model": "dinov2_vitl14_reg", "checkpoint_sha256": CHECKPOINT_SHA,
                                "source_tree_sha256": SOURCE_TREE_SHA, "feature_interface": FEATURE_SCHEMA}
            feature_meta = {"token_normalization": "each CLS/register/patch independently L2-normalized FP32",
                            "valid_patch_count": int(mask.sum()), "patch_count": 1024,
                            "feature_shapes": {k: list(v.shape) for k, v in arrays.items()}}
            record, reused = write_bundle_atomic(feature_root / row["opaque_image_id"], arrays,
                                                  identity_feature, feature_meta)
            if not record.get("arrays"):
                raise RuntimeError("FEATURE_BUNDLE_INCOMPLETE")
            recpath = receipt_root / row["opaque_image_id"] / "item_completion.json"
            recpath.parent.mkdir(parents=True, exist_ok=True)
            item_receipt = {"opaque_image_id": row["opaque_image_id"], "source_sha256": source["sha256"],
                            "canonical_array_sha256": canonical_array_sha256(tensor),
                            "feature_record_sha256": sha_file(feature_root / row["opaque_image_id"] / "record.json"),
                            "feature_reused_after_integrity_check": reused, "valid_patch_count": int(mask.sum()),
                            "status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat()}
            if recpath.exists():
                prior = json.loads(recpath.read_text(encoding="utf-8"))
                if prior != item_receipt:
                    raise RuntimeError("EXISTING_ITEM_RECEIPT_MISMATCH")
            else:
                with recpath.open("x", encoding="utf-8") as f:
                    json.dump(item_receipt, f, ensure_ascii=False, indent=2)
            feature_rows.append({"opaque_image_id": row["opaque_image_id"], "opaque_group_id": row["opaque_group_id"],
                                 "product": row["product"], "view": row["view"], "conditional_role": row["conditional_role"],
                                 "candidate_kind": row["candidate_kind"], "source_sha256": source["sha256"],
                                 "canonical_array_sha256": canonical_array_sha256(tensor),
                                 "record_sha256": item_receipt["feature_record_sha256"],
                                 "cls_array_sha256": record["arrays"]["cls_unit"]["array_sha256"],
                                 "register_array_sha256": record["arrays"]["register_unit"]["array_sha256"],
                                 "patch_array_sha256": record["arrays"]["patch_unit"]["array_sha256"],
                                 "mask_array_sha256": record["arrays"]["valid_patch_mask"]["array_sha256"],
                                 "valid_patch_count": int(mask.sum()), "dtype": "float32",
                                 "backbone": "dinov2_vitl14_reg", "checkpoint_sha256": CHECKPOINT_SHA,
                                 "feature_interface": FEATURE_SCHEMA})
            exposure_by_id = {x["opaque_image_id"]: x for x in exposures}
            exposure_by_id[row["opaque_image_id"]]["used_for_model"] = "YES"
            check_output_budget(output)
        except Exception as exc:
            failures.append({"opaque_image_id": row["opaque_image_id"], "status": "FAILED_FEATURE",
                             "reason_code": type(exc).__name__ + ":" + str(exc).split(":")[0]})
    elapsed = time.monotonic() - started
    final_gpu = gpu_status()
    for row in exposures:
        if row["used_for_model"] == "PENDING":
            row["used_for_model"] = "YES" if any(f["opaque_image_id"] == row["opaque_image_id"]
                                                       for f in feature_rows) else "NO"
    # Keep a per-item terminal receipt for every authorized identity, including failures.
    failure_by_id = {r["opaque_image_id"]: r for r in failures}
    success_by_id = {r["opaque_image_id"]: r for r in feature_rows}
    for entry in entries:
        item_id = entry["opaque_image_id"]
        recpath = receipt_root / item_id / "item_completion.json"
        if recpath.exists():
            continue
        status = "COMPLETE" if item_id in success_by_id else failure_by_id.get(item_id, {}).get("status", "FAILED_CANONICAL_OR_FEATURE")
        receipt = {"opaque_image_id": item_id, "source_sha256": next((x["source_sha256"] for x in exposures if x["opaque_image_id"] == item_id), ""),
                   "status": status, "reason_code": failure_by_id.get(item_id, {}).get("reason_code", ""),
                   "completed_utc": datetime.now(timezone.utc).isoformat()}
        recpath.parent.mkdir(parents=True, exist_ok=True)
        with recpath.open("x", encoding="utf-8") as f:
            json.dump(receipt, f, ensure_ascii=False, indent=2)
    # Only successful canonical rows go into manifests; failed views remain in failures.
    canonical_hash_manifest = write_csv_once(output / "canonical_manifest.csv", canonical_rows)
    feature_hash_manifest = write_csv_once(output / "feature_manifest.csv", feature_rows)
    exposure_hash = write_csv_once(output / "exposure_ledger.csv", exposures)
    failure_hash = write_csv_once(output / "failures.csv", failures,
                                  ["opaque_image_id", "status", "reason_code"])
    resource = {"gpu_initial": initial_gpu, "gpu_final": final_gpu,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
                "free_disk_bytes_after": int(__import__("shutil").disk_usage(output).free),
                "output_bytes": check_output_budget(output), "output_cap_bytes": 3 * 1024**3,
                "elapsed_seconds": elapsed, "feature_seconds": time.monotonic() - feature_start,
                "processes_terminated": False, "smoke_repeat_metrics": smoke_metrics}
    sha_success = sum(row["read_success"] == "YES" for row in exposures)
    result = {"status": "P3_G1A_PARTIAL" if failures else "P3_G1A_FEATURES_PASS",
              "input_views": len(entries), "canonical_success": len(canonical_rows),
              "sha_success": sha_success, "sha_failure": len(entries) - sha_success,
              "feature_success": len(feature_rows), "failure_count": len(failures),
              "canonical_manifest_sha256": canonical_hash_manifest,
              "feature_manifest_sha256": feature_hash_manifest, "exposure_ledger_sha256": exposure_hash,
              "failures_sha256": failure_hash, "runtime_identity": runtime, "resources": resource,
              "new_attribute_labels_read": False, "gt_masks_read": False,
              "protected_storage_accessed": False, "training_or_update": False}
    write_json_once(output / "extraction_summary.json", result)
    return result


if __name__ == "__main__":
    try:
        result = run(OUT)
        print(json.dumps({k: result[k] for k in ("status", "input_views", "canonical_success", "feature_success", "failure_count")}, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "P3_G1A_BLOCKED_RUNTIME", "reason_code": type(exc).__name__ + ":" + str(exc).split(":")[0]}))
        raise
