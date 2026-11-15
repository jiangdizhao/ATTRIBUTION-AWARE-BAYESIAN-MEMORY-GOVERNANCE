"""Post-extraction checks joining only the registered whitelist and cache indexes."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from contracts import FEATURE_INTERFACE_SCHEMA, OUT
from extract_core import identity_for_row
from preprocess import canonical_array_sha256, patch_center_mask
from storage import array_sha256, sha256_file


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_cache(authorized_rows: list[dict[str, str]], runtime_by_backend: dict[str, dict[str, Any]]) -> dict[str, Any]:
    canonical_rows = read_csv(OUT / "canonical_manifest.csv")
    ids = [row["image_id"] for row in authorized_rows]
    if len(canonical_rows) != 102 or [row["image_id"] for row in canonical_rows] != ids:
        raise RuntimeError("Canonical manifest does not exactly join the ordered 102-image whitelist")
    if len({row["image_id"] for row in canonical_rows}) != 102:
        raise RuntimeError("Canonical manifest contains duplicate IDs")
    expected_source = {row["image_id"]: row["source_sha256"] for row in authorized_rows}

    valid_counts = {"dinov2": [], "dinov3": []}
    content_sizes: Counter[tuple[int, int]] = Counter()
    for row in canonical_rows:
        image_id = row["image_id"]
        if row["source_sha256"] != expected_source[image_id]:
            raise RuntimeError(f"Canonical source join mismatch: {image_id}")
        directory = OUT / "canonical" / image_id
        record_path = directory / "record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if sha256_file(record_path) == "":
            raise RuntimeError(f"Canonical record checksum missing: {image_id}")
        rect = tuple(json.loads(row["content_rect_xyxy"]))
        metadata = record["metadata"]
        if tuple(metadata["content_rect_xyxy"]) != rect:
            raise RuntimeError(f"Canonical content rectangle join mismatch: {image_id}")
        tensor_path = directory / "canonical_tensor.npy"
        tensor = np.load(tensor_path, allow_pickle=False)
        if (sha256_file(tensor_path) != row["canonical_file_sha256"]
                or canonical_array_sha256(tensor) != row["canonical_array_sha256"]
                or tensor.shape != (1, 3, 448, 448) or tensor.dtype != np.float32
                or not np.isfinite(tensor).all()):
            raise RuntimeError(f"Canonical tensor verification failed: {image_id}")
        for backend, patch_size, expected_length in (("dinov2", 14, 1024), ("dinov3", 16, 784)):
            mask_name = f"mask_{backend}"
            mask_path = directory / f"{mask_name}.npy"
            mask = np.load(mask_path, allow_pickle=False)
            expected_mask = patch_center_mask(rect, patch_size)
            if (mask.dtype != np.bool_ or mask.shape != (expected_length,)
                    or not np.array_equal(mask, expected_mask)
                    or sha256_file(mask_path) != row[f"mask_{backend}_sha256"]):
                raise RuntimeError(f"Patch mask does not match content geometry: {image_id}/{backend}")
            valid_counts[backend].append(int(mask.sum()))
        content_sizes[(metadata["resized_width"], metadata["resized_height"])] += 1

    model_results = {}
    for backend, patch_length in (("dinov2", 1024), ("dinov3", 784)):
        runtime = runtime_by_backend[backend]
        manifest_path = OUT / backend / "feature_manifest.csv"
        rows = read_csv(manifest_path)
        if len(rows) != 102 or [r["image_id"] for r in rows] != ids:
            raise RuntimeError(f"{backend} feature manifest does not exactly cover the whitelist")
        expected_model_dirs = set(ids)
        actual_model_dirs = {p.name for p in (OUT / backend / "items").iterdir() if p.is_dir()}
        if actual_model_dirs != expected_model_dirs:
            raise RuntimeError(f"{backend} feature item directory set contains missing/extra IDs")
        for row in rows:
            image_id = row["image_id"]
            canonical = canonical_rows[ids.index(image_id)]
            if (row["source_sha256"] != expected_source[image_id]
                    or row["canonical_array_sha256"] != canonical["canonical_array_sha256"]
                    or row["canonical_file_sha256"] != canonical["canonical_file_sha256"]):
                raise RuntimeError(f"{backend} canonical/source identity join mismatch: {image_id}")
            bundle = OUT / backend / "items" / image_id
            record_path = bundle / "record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            expected_identity = identity_for_row(runtime, canonical)
            if record.get("identity") != expected_identity:
                raise RuntimeError(f"{backend} model identity mismatch: {image_id}")
            if row["record_sha256"] != sha256_file(record_path):
                raise RuntimeError(f"{backend} record manifest SHA mismatch: {image_id}")
            specs = record["arrays"]
            for name in ("cls_unit", "register_unit", "patch_unit", "valid_patch_mask"):
                spec = specs[name]
                path = bundle / spec["file"]
                array = np.load(path, allow_pickle=False)
                if sha256_file(path) != spec["file_sha256"] or not np.isfinite(array).all():
                    raise RuntimeError(f"{backend} output file SHA/finite check failed: {image_id}/{name}")
                if array.dtype.name != spec["dtype"] or list(array.shape) != spec["shape"]:
                    raise RuntimeError(f"{backend} output dtype/shape check failed: {image_id}/{name}")
                if array_sha256(array) != spec["array_sha256"]:
                    raise RuntimeError(f"{backend} output array digest check failed: {image_id}/{name}")
                if name in ("cls_unit", "register_unit", "patch_unit"):
                    if array.dtype != np.float32:
                        raise RuntimeError(f"{backend} feature is not FP32: {image_id}/{name}")
                    norm_error = (abs(float(np.linalg.norm(array)) - 1.0) if name == "cls_unit"
                                  else float(np.max(np.abs(np.linalg.norm(array, axis=-1) - 1.0))))
                    if norm_error > 1e-5:
                        raise RuntimeError(f"{backend} token normalization check failed: {image_id}/{name}")
                elif array.dtype != np.bool_:
                    raise RuntimeError(f"{backend} valid mask is not boolean: {image_id}")
            expected_shapes = {
                "cls_unit": [1024], "register_unit": [4, 1024],
                "patch_unit": [patch_length, 1024], "valid_patch_mask": [patch_length],
            }
            for name, shape in expected_shapes.items():
                if specs[name]["shape"] != shape:
                    raise RuntimeError(f"{backend} expected shape failure: {image_id}/{name}")
            for manifest_name, spec_name in (("cls_shape", "cls_unit"), ("register_shape", "register_unit"),
                                             ("patch_shape", "patch_unit"), ("mask_shape", "valid_patch_mask")):
                if json.loads(row[manifest_name]) != specs[spec_name]["shape"]:
                    raise RuntimeError(f"{backend} manifest shape does not match bundle: {image_id}/{spec_name}")
            canonical_mask = np.load(OUT / canonical[f"mask_{backend}_relpath"], allow_pickle=False)
            feature_mask = np.load(bundle / specs["valid_patch_mask"]["file"], allow_pickle=False)
            if not np.array_equal(canonical_mask, feature_mask):
                raise RuntimeError(f"{backend} feature mask differs from canonical mask: {image_id}")
            if int(row["valid_patch_count"]) != int(canonical_mask.sum()):
                raise RuntimeError(f"{backend} feature manifest mask count mismatch: {image_id}")
            if row["feature_interface_schema"] != FEATURE_INTERFACE_SCHEMA:
                raise RuntimeError(f"{backend} feature interface version mismatch: {image_id}")
            if row["weights_sha256"] != runtime["weights_sha256"]:
                raise RuntimeError(f"{backend} weight SHA mismatch: {image_id}")
        model_results[backend] = {
            "image_count": len(rows),
            "feature_manifest_sha256": sha256_file(manifest_path),
            "weights_sha256": runtime["weights_sha256"],
            "valid_patch_count_min": min(valid_counts[backend]),
            "valid_patch_count_max": max(valid_counts[backend]),
        }
    if not set(ids) == {row["image_id"] for row in read_csv(OUT / "dinov2/feature_manifest.csv")} == {row["image_id"] for row in read_csv(OUT / "dinov3/feature_manifest.csv")}:
        raise RuntimeError("The DINOv2 and DINOv3 image ID sets differ")
    return {
        "image_count": 102,
        "role_counts": {"D-fit": 54, "D-dev": 48},
        "source_join_verified": True,
        "canonical_tensor_count": 102,
        "both_backbone_image_sets_equal": True,
        "valid_patch_counts": {
            backend: {"min": min(values), "max": max(values), "mean": float(np.mean(values))}
            for backend, values in valid_counts.items()
        },
        "resized_dimensions_unique_count": len(content_sizes),
        "model_results": model_results,
    }
