"""Validate the registered 102-row input whitelist without reading labels."""
from __future__ import annotations

import hashlib
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any

from contracts import (INPUT_MANIFEST_SHA256, P1A_FEATURE_MANIFEST_SHA256,
                       ROOT, WX_TAPE_SHA256)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_p1a_module():
    path = ROOT / "scripts/research_d6/p1a/extract_features.py"
    spec = importlib.util.spec_from_file_location("e2a1_p1a_whitelist", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the registered P1-A input builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_registered_inputs(root: Path = ROOT) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Rebuild via P1-A, compare to its fixed manifest, and hash only listed files."""
    p1a = _load_p1a_module()
    manifest_path = root / "outputs/research_d6/p1a_features/input_manifest.csv"
    feature_manifest = root / "outputs/research_d6/p1a_features/feature_manifest.csv"
    tape_path = root / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/p1_gate0_draft/D6_P1_WX_role_map_feedback_tape.csv"
    if sha256_file(manifest_path) != INPUT_MANIFEST_SHA256:
        raise RuntimeError("Registered P1-A input manifest SHA does not match")
    if sha256_file(feature_manifest) != P1A_FEATURE_MANIFEST_SHA256:
        raise RuntimeError("Registered P1-A feature manifest SHA does not match")
    if sha256_file(tape_path) != WX_TAPE_SHA256:
        raise RuntimeError("Registered P1 W/X feedback tape SHA does not match")

    rows = p1a.make_input_rows(root)
    p1a.validate_input_rows(rows, verify_sha=True)
    stored_rows = p1a.csv_rows(manifest_path)
    if rows != stored_rows:
        raise RuntimeError("P1-A-generated whitelist rows differ from the frozen input manifest")

    allowed_roots = [
        (root / "research_pilot/LP_W1_D4LITE/images").resolve(),
        (root / "research_pilot/LP_W1_D4B/images").resolve(),
        Path(r"G:\dataset\realiad\images").resolve(),
    ]
    for row in rows:
        path = Path(row["image_path"]).resolve(strict=True)
        if not any(path.is_relative_to(allowed) for allowed in allowed_roots):
            raise RuntimeError(f"Whitelist path outside registered roots: {row['image_id']}")
        if row["image_id"] == "" or row["source_sha256"] == "":
            raise RuntimeError("Whitelist contains an empty image identity or source digest")
        if row["data_role"] == "D-fit":
            if row["dataset"] != "aristrops/mvtec-concepts" or row["exposure_status"] != "PILOT_EXPOSED":
                raise RuntimeError("D-fit source/exposure authorization is inconsistent")
            if row["authorization_status"] not in {
                "pilot_only_exclude_from_future_confirmation",
                "D4B annotation pilot only; exclude from future confirmation",
            }:
                raise RuntimeError("D-fit authorization status is outside the registered pilot scope")
        elif row["data_role"] == "D-dev":
            if row["dataset"] != "Real-IAD" or row["exposure_status"] != "NEW_STUDY_PILOT_EXPOSED":
                raise RuntimeError("D-dev source/exposure authorization is inconsistent")
            if row["authorization_status"] != "AUTHORIZED_LIMITED_DEV_PILOT":
                raise RuntimeError("D-dev authorization status is outside the registered pilot scope")
        else:
            raise RuntimeError(f"Unregistered data role: {row['data_role']}")

    roles = Counter(row["data_role"] for row in rows)
    source_shas = [row["source_sha256"].lower() for row in rows]
    source_paths = [str(Path(row["image_path"]).resolve()).casefold() for row in rows]
    if len(rows) != 102 or len({row["image_id"] for row in rows}) != 102:
        raise RuntimeError("Whitelist is not 102 unique image IDs")
    if dict(roles) != {"D-fit": 54, "D-dev": 48}:
        raise RuntimeError(f"Whitelist role counts are wrong: {dict(roles)}")
    if len(set(source_shas)) != 102 or len(set(source_paths)) != 102:
        raise RuntimeError("Whitelist has duplicate source files or SHA-256 values")
    summary = {
        "row_count": len(rows),
        "unique_image_ids": len({row["image_id"] for row in rows}),
        "unique_source_sha256": len(set(source_shas)),
        "role_counts": dict(roles),
        "dataset_role_counts": {
            dataset: {role: sum(row["dataset"] == dataset and row["data_role"] == role for row in rows)
                      for role in ("D-fit", "D-dev")}
            for dataset in sorted({row["dataset"] for row in rows})
        },
        "input_manifest_sha256": sha256_file(manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "wx_tape_sha256": sha256_file(tape_path),
        "all_source_hashes_verified": True,
        "all_paths_under_registered_roots": True,
        "authorization_status_counts": dict(Counter(row["authorization_status"] for row in rows)),
        "exposure_status_counts": dict(Counter(row["exposure_status"] for row in rows)),
        "labels_read": False,
    }
    return rows, summary
