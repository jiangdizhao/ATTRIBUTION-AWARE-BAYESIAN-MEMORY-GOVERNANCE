"""Bounded, resumable FP32 GPU extraction shared by the two isolated backends."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from contracts import (ALLOCATOR_CAP_BYTES, FEATURE_INTERFACE_SCHEMA, GPU_MIN_FREE_BYTES,
                       INPUT_SIZE, OUT, OUTPUT_CAP_BYTES, ROOT)
from preprocess import canonical_array_sha256
from storage import sha256_file, write_bundle_atomic, write_manifest_once


CANONICAL_FIELDS = [
    "image_id", "input_order_index", "source_sha256", "canonical_relpath",
    "canonical_file_sha256", "canonical_array_sha256", "canonical_dtype", "canonical_shape",
    "mask_dinov2_relpath", "mask_dinov2_sha256", "mask_dinov2_count",
    "mask_dinov3_relpath", "mask_dinov3_sha256", "mask_dinov3_count",
    "content_rect_xyxy", "preprocess_version", "preprocess_code_sha256",
    "bundle_reused_after_validation",
]


def gpu_preflight() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True, timeout=20,
    )
    lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one registered GPU; found {len(lines)}")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 5:
        raise RuntimeError("Cannot parse nvidia-smi device query")
    process_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=True, text=True, capture_output=True, timeout=20,
    )
    process_lines = [line for line in process_result.stdout.splitlines() if line.strip()]
    return {
        "gpu_name": parts[0],
        "memory_total_mib": int(parts[1]),
        "memory_free_mib": int(parts[2]),
        "memory_used_mib": int(parts[3]),
        "utilization_percent": int(parts[4]),
        "compute_process_count": len(process_lines),
        "per_process_memory_visible": any("N/A" not in row for row in process_lines),
        "processes_terminated": False,
    }


def check_output_cap() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    if total > OUTPUT_CAP_BYTES:
        raise RuntimeError(f"E2-A1 output already exceeds registered cap: {total} > {OUTPUT_CAP_BYTES}")


def _safe_canonical_path(relpath: str) -> Path:
    path = (OUT / relpath).resolve(strict=True)
    base = (OUT / "canonical").resolve(strict=True)
    if not path.is_relative_to(base):
        raise RuntimeError("Canonical manifest points outside the E2-A1 canonical directory")
    return path


def load_canonical_rows(mode: str) -> list[dict[str, str]]:
    filename = "canonical_smoke_manifest.csv" if mode == "smoke" else "canonical_manifest.csv"
    path = OUT / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required canonical input manifest is missing: {filename}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_count = 2 if mode == "smoke" else 102
    if len(rows) != expected_count or len({row["image_id"] for row in rows}) != expected_count:
        raise RuntimeError(f"Canonical {mode} manifest must contain {expected_count} unique images")
    return rows


def load_canonical_item(row: dict[str, str], backend: str) -> tuple[np.ndarray, np.ndarray]:
    tensor_path = _safe_canonical_path(row["canonical_relpath"])
    if sha256_file(tensor_path) != row["canonical_file_sha256"]:
        raise RuntimeError(f"Canonical tensor file SHA mismatch for {row['image_id']}")
    tensor = np.load(tensor_path, allow_pickle=False)
    if tensor.shape != (1, 3, INPUT_SIZE, INPUT_SIZE) or tensor.dtype != np.float32:
        raise RuntimeError(f"Canonical tensor shape/dtype mismatch for {row['image_id']}")
    if not np.isfinite(tensor).all() or canonical_array_sha256(tensor) != row["canonical_array_sha256"]:
        raise RuntimeError(f"Canonical tensor array identity/value mismatch for {row['image_id']}")
    mask_rel = row[f"mask_{backend}_relpath"]
    mask_path = _safe_canonical_path(mask_rel)
    if sha256_file(mask_path) != row[f"mask_{backend}_sha256"]:
        raise RuntimeError(f"Canonical patch mask file SHA mismatch for {row['image_id']}")
    mask = np.load(mask_path, allow_pickle=False)
    patch_count = 1024 if backend == "dinov2" else 784
    if mask.shape != (patch_count,) or mask.dtype != np.bool_ or int(mask.sum()) != int(row[f"mask_{backend}_count"]):
        raise RuntimeError(f"Canonical patch mask shape/dtype/count mismatch for {row['image_id']}")
    if not mask.any():
        raise RuntimeError(f"Canonical patch mask contains no valid patches for {row['image_id']}")
    return tensor, mask


def _feature_arrays(record: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = {
        "cls_unit": record["cls_unit"][0],
        "register_unit": record["register_unit"][0],
        "patch_unit": record["patch_unit"][0],
        "valid_patch_mask": record["valid_patch_mask"][0],
    }
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite shared feature array: {name}")
    if arrays["cls_unit"].shape != (1024,) or arrays["register_unit"].shape != (4, 1024):
        raise RuntimeError("Shared CLS/register feature shapes do not match the registered interface")
    expected_patches = 1024 if record["backbone"] == "dinov2_vitl14_reg" else 784
    if arrays["patch_unit"].shape != (expected_patches, 1024):
        raise RuntimeError("Shared patch feature shape does not match the registered backbone")
    if abs(float(np.linalg.norm(arrays["cls_unit"])) - 1.0) > 1e-5:
        raise RuntimeError("CLS unit-norm check failed")
    for name in ("register_unit", "patch_unit"):
        error = float(np.max(np.abs(np.linalg.norm(arrays[name], axis=-1) - 1.0)))
        if error > 1e-5:
            raise RuntimeError(f"Per-token unit-norm check failed for {name}: {error}")
    return arrays


def _check_repeated(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    result = {}
    for name in ("cls_unit", "register_unit", "patch_unit"):
        a, b = first[name], second[name]
        max_abs = float(np.max(np.abs(a - b)))
        cosine = np.sum(a * b, axis=-1)
        result[f"{name}_max_abs"] = max_abs
        result[f"{name}_min_cosine"] = float(np.min(cosine))
        if max_abs > 1e-3 or float(np.min(cosine)) < 0.999:
            raise RuntimeError(f"Repeated GPU extraction mismatch in {name}: {result}")
    return result


def _read_array_specs(bundle_dir: Path) -> tuple[dict[str, Any], str]:
    record_path = bundle_dir / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return record, sha256_file(record_path)


def _feature_manifest_rows(rows: list[dict[str, str]], item_dir: Path,
                           identity: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        bundle = item_dir / row["image_id"]
        record, record_sha = _read_array_specs(bundle)
        if record["identity"] != identity_for_row(identity, row):
            raise RuntimeError(f"Feature identity changed for {row['image_id']}")
        specs = record["arrays"]
        result.append({
            "image_id": row["image_id"],
            "source_sha256": row["source_sha256"],
            "canonical_array_sha256": row["canonical_array_sha256"],
            "canonical_file_sha256": row["canonical_file_sha256"],
            "record_sha256": record_sha,
            "cls_file_sha256": specs["cls_unit"]["file_sha256"],
            "register_file_sha256": specs["register_unit"]["file_sha256"],
            "patch_file_sha256": specs["patch_unit"]["file_sha256"],
            "mask_file_sha256": specs["valid_patch_mask"]["file_sha256"],
            "cls_shape": json.dumps(specs["cls_unit"]["shape"], separators=(",", ":")),
            "register_shape": json.dumps(specs["register_unit"]["shape"], separators=(",", ":")),
            "patch_shape": json.dumps(specs["patch_unit"]["shape"], separators=(",", ":")),
            "mask_shape": json.dumps(specs["valid_patch_mask"]["shape"], separators=(",", ":")),
            "valid_patch_count": int(np.load(bundle / specs["valid_patch_mask"]["file"], allow_pickle=False).sum()),
            "dtype": "float32",
            "feature_interface_schema": FEATURE_INTERFACE_SCHEMA,
            "model": identity["model"],
            "weights_sha256": identity["weights_sha256"],
            "extractor_identity_sha256": identity["extractor_identity_sha256"],
        })
    return result


def identity_for_row(base: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    return {
        **base,
        "image_id": row["image_id"],
        "source_sha256": row["source_sha256"],
        "canonical_file_sha256": row["canonical_file_sha256"],
        "canonical_array_sha256": row["canonical_array_sha256"],
    }


def run_backend(*, backend: str, mode: str, torch: Any, runtime: dict[str, Any],
                model_loader: Callable[[], Any], forward: Callable[[Any, Any, np.ndarray], dict[str, Any]],
                warmups: int = 0) -> dict[str, Any]:
    """Run one model subprocess against only canonical tensors and write verified bundles."""
    if backend not in ("dinov2", "dinov3") or mode not in ("smoke", "all"):
        raise ValueError("Unregistered backend or extraction mode")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the registered model environment")
    gpu = gpu_preflight()
    if gpu["memory_free_mib"] * 1024**2 < GPU_MIN_FREE_BYTES:
        raise RuntimeError("Less than the registered 10 GiB free-VRAM gate; model not loaded")
    if gpu["gpu_name"] != "NVIDIA GeForce RTX 3090 Ti":
        raise RuntimeError(f"Unexpected GPU: {gpu['gpu_name']}")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    total_vram = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP_BYTES / total_vram, device=device)

    rows = load_canonical_rows(mode)
    item_dir = OUT / backend / "items"
    report_path = OUT / backend / ("smoke_report.json" if mode == "smoke" else "run_report.json")
    manifest_path = OUT / backend / ("smoke_feature_manifest.csv" if mode == "smoke" else "feature_manifest.csv")
    if report_path.exists():
        raise FileExistsError(f"Existing {mode} report found; refusing to rerun or overwrite: {report_path.name}")

    start_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    load_started = time.perf_counter()
    model = model_loader()
    model_load_seconds = time.perf_counter() - load_started
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Model parameters are not all FP32")
    model = model.to(device)
    model.eval()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    per_image: list[dict[str, Any]] = []
    reused_count = 0
    for row in rows:
        tensor_np, mask = load_canonical_item(row, backend)
        tensor = torch.from_numpy(tensor_np.copy()).to(device)
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Model input is not finite FP32 for {row['image_id']}")
        if warmups:
            for _ in range(warmups):
                with torch.inference_mode():
                    _ = forward(model, tensor, mask)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            shared_first = forward(model, tensor, mask)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        first_arrays = _feature_arrays(shared_first)
        repeat_metrics = None
        if mode == "smoke":
            with torch.inference_mode():
                shared_repeat = forward(model, tensor, mask)
            torch.cuda.synchronize(device)
            repeat_metrics = _check_repeated(first_arrays, _feature_arrays(shared_repeat))

        identity = identity_for_row(runtime, row)
        metadata = {
            "feature_interface_schema": FEATURE_INTERFACE_SCHEMA,
            "token_normalization": "each CLS, register and patch token L2-normalized independently in FP32",
            "valid_patch_count": int(mask.sum()),
            "patch_count": int(mask.size),
            "feature_shapes": {name: list(value.shape) for name, value in first_arrays.items()},
        }
        _, reused = write_bundle_atomic(item_dir / row["image_id"], first_arrays, identity, metadata)
        reused_count += int(reused)
        per_image.append({
            "image_id": row["image_id"],
            "elapsed_seconds": elapsed,
            "valid_patch_count": int(mask.sum()),
            "reused_after_identity_and_sha_validation": reused,
            "repeat_gpu_metrics": repeat_metrics,
        })
        check_output_cap()

    elapsed_total = time.perf_counter() - run_started
    fields = [
        "image_id", "source_sha256", "canonical_array_sha256", "canonical_file_sha256",
        "record_sha256", "cls_file_sha256", "register_file_sha256", "patch_file_sha256",
        "mask_file_sha256", "cls_shape", "register_shape", "patch_shape", "mask_shape",
        "valid_patch_count", "dtype", "feature_interface_schema", "model", "weights_sha256",
        "extractor_identity_sha256",
    ]
    manifest_rows = _feature_manifest_rows(rows, item_dir, runtime)
    write_manifest_once(manifest_path, manifest_rows, fields)
    torch.cuda.synchronize(device)
    end_gpu = gpu_preflight()
    report = {
        "status": "E2A1_SMOKE_PASS" if mode == "smoke" else "E2A1_BACKBONE_CACHE_PASS",
        "backend": backend,
        "mode": mode,
        "start_utc": start_utc,
        "execution_head": runtime["execution_head"],
        "model": runtime["model"],
        "weights_sha256": runtime["weights_sha256"],
        "runtime_identity_sha256": runtime["extractor_identity_sha256"],
        "torch_version": torch.__version__,
        "python_version": runtime["python_version"],
        "device": str(device),
        "precision": "FP32; autocast disabled; TF32 disabled",
        "initial_gpu": gpu,
        "final_gpu": end_gpu,
        "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "model_load_seconds": model_load_seconds,
        "processed_images": len(rows),
        "elapsed_seconds_including_model_load": elapsed_total,
        "feature_generation_seconds_sum": float(sum(item["elapsed_seconds"] for item in per_image)),
        "images_per_second_total": float(len(rows) / elapsed_total) if elapsed_total > 0 else None,
        "images_per_second_feature_generation": float(len(rows) / sum(item["elapsed_seconds"] for item in per_image)) if sum(item["elapsed_seconds"] for item in per_image) > 0 else None,
        "reused_validated_items": reused_count,
        "failed_items": 0,
        "feature_manifest_sha256": sha256_file(manifest_path),
        "runtime_identity": runtime,
        "input_ids_and_per_image_times": per_image,
        "labels_read": False,
        "network_enabled": False,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    with report_path.open("xb") as stream:
        stream.write(payload.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    check_output_cap()
    return report
