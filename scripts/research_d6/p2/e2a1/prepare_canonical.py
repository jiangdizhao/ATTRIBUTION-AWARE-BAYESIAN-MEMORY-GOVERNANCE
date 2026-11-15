"""Prepare only the P1-A-registered D-fit/D-dev canonical inputs."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import PIL

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_inputs import audit_registered_inputs, sha256_file
from contracts import OUT, PREPROCESS_VERSION, ROOT
from preprocess import canonical_array_sha256, decode_and_preprocess
from storage import write_bundle_atomic, write_manifest_once


def code_sha256(path: Path) -> str:
    return sha256_file(path)


def _copy_whitelist_bytes() -> None:
    source = ROOT / "outputs/research_d6/p1a_features/input_manifest.csv"
    target = OUT / "registered_input_manifest.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise RuntimeError("Existing E2-A1 whitelist copy differs from registered P1-A; refusing overwrite")
    else:
        temp = OUT / ".registered_input_manifest.csv.tmp"
        with temp.open("xb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()


def _preprocess_one(row: dict[str, str], index: int) -> dict[str, Any]:
    image_id = row["image_id"]
    if not re.fullmatch(r"I-[a-f0-9]+", image_id):
        raise RuntimeError(f"Unexpected image ID format in registered whitelist: {image_id}")
    source_path = Path(row["image_path"]).resolve(strict=True)
    before_sha = sha256_file(source_path)
    if before_sha.lower() != row["source_sha256"].lower():
        raise RuntimeError(f"Source image SHA mismatch before decode: {image_id}")
    tensor, decoded = decode_and_preprocess(source_path)
    after_sha = sha256_file(source_path)
    if after_sha.lower() != before_sha.lower():
        raise RuntimeError(f"Source image changed while being decoded: {image_id}")
    masks = {
        "mask_dinov2": decoded.pop("mask_dinov2"),
        "mask_dinov3": decoded.pop("mask_dinov3"),
    }
    metadata = {
        **decoded,
        "image_id": image_id,
        "source_sha256": before_sha,
        "input_manifest_sha256": "820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63",
        "input_order_index": index,
        "canonical_array_sha256": canonical_array_sha256(tensor),
        "mask_counts": {key: int(value.sum()) for key, value in masks.items()},
    }
    identity = {
        "image_id": image_id,
        "source_sha256": before_sha,
        "input_manifest_sha256": metadata["input_manifest_sha256"],
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_code_sha256": code_sha256(Path(__file__).resolve()),
    }
    arrays = {"canonical_tensor": tensor, **masks}
    bundle_dir = OUT / "canonical" / image_id
    record, reused = write_bundle_atomic(bundle_dir, arrays, identity, metadata)
    tensor_spec = record["arrays"]["canonical_tensor"]
    v2_spec = record["arrays"]["mask_dinov2"]
    v3_spec = record["arrays"]["mask_dinov3"]
    return {
        "image_id": image_id,
        "input_order_index": index,
        "source_sha256": before_sha,
        "canonical_relpath": (bundle_dir / tensor_spec["file"]).relative_to(OUT).as_posix(),
        "canonical_file_sha256": tensor_spec["file_sha256"],
        "canonical_array_sha256": tensor_spec["array_sha256"],
        "canonical_dtype": tensor_spec["dtype"],
        "canonical_shape": json.dumps(tensor_spec["shape"], separators=(",", ":")),
        "mask_dinov2_relpath": (bundle_dir / v2_spec["file"]).relative_to(OUT).as_posix(),
        "mask_dinov2_sha256": v2_spec["file_sha256"],
        "mask_dinov2_count": int(masks["mask_dinov2"].sum()),
        "mask_dinov3_relpath": (bundle_dir / v3_spec["file"]).relative_to(OUT).as_posix(),
        "mask_dinov3_sha256": v3_spec["file_sha256"],
        "mask_dinov3_count": int(masks["mask_dinov3"].sum()),
        "content_rect_xyxy": json.dumps(metadata["content_rect_xyxy"], separators=(",", ":")),
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_code_sha256": identity["preprocess_code_sha256"],
        "bundle_reused_after_validation": str(reused).lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "all"), required=True)
    args = parser.parse_args()
    if np.__version__ != "1.24.1" or PIL.__version__ != "9.3.0":
        raise RuntimeError(f"Preprocessing environment differs from registered Pillow/NumPy: {PIL.__version__}/{np.__version__}")
    rows, summary = audit_registered_inputs()
    _copy_whitelist_bytes()
    audit_path = OUT / "source_whitelist_audit.json"
    audit_payload = {**summary, "scope": "registered P1-A whitelist only", "pixels_decoded": False}
    encoded = json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n"
    if audit_path.exists():
        if audit_path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("Existing source whitelist audit differs; refusing overwrite")
    else:
        audit_path.write_text(encoded, encoding="utf-8")

    if args.mode == "smoke":
        selected = [row for row in rows if row["data_role"] == "D-fit"][:2]
        if len(selected) != 2:
            raise RuntimeError("Could not select two D-fit smoke images using frozen manifest order")
        manifest_name = "canonical_smoke_manifest.csv"
    else:
        selected = rows
        manifest_name = "canonical_manifest.csv"

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        output_rows.append(_preprocess_one(row, index))
    fields = list(output_rows[0])
    manifest_sha = write_manifest_once(OUT / manifest_name, output_rows, fields)
    print(json.dumps({
        "status": "CANONICAL_SMOKE_READY" if args.mode == "smoke" else "CANONICAL_102_READY",
        "mode": args.mode,
        "processed_count": len(output_rows),
        "input_order": "P1-A frozen input manifest order; smoke is first two D-fit rows",
        "manifest_file": manifest_name,
        "manifest_sha256": manifest_sha,
        "preprocess_version": PREPROCESS_VERSION,
        "labels_read": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
