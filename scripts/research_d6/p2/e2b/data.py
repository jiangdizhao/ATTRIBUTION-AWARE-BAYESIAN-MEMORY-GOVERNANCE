"""Read-only E2B input joins; labels and evaluator metadata stay outside learners."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stable_tree_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "bytes": 0, "sha256": ""}
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    )
    base = path.parent if path.is_file() else path
    digest = hashlib.sha256()
    total = 0
    for item in files:
        digest.update(item.relative_to(base).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_file(item).encode("ascii") + b"\n")
        total += item.stat().st_size
    return {"exists": True, "file_count": len(files), "bytes": total,
            "sha256": digest.hexdigest()}


def _load_e2a1_verifier():
    # Load E2-A1's verifier and its local contracts under their expected import names.
    e2a1_dir = str(config.ROOT / "scripts/research_d6/p2/e2a1")
    if e2a1_dir not in sys.path:
        sys.path.insert(0, e2a1_dir)
    return importlib.import_module("verify_cache")


def verify_frozen_cache() -> dict[str, Any]:
    """Recheck manifests and every cached array without touching source images or labels."""
    e2a = config.E2A1_ROOT
    p1a_input_path = config.P1A_INPUT_MANIFEST
    p1a_feature_path = config.P1A_FEATURE_MANIFEST
    tape_path = config.TAPE_PATH
    if sha256_file(p1a_input_path) != config.EXPECTED["p1a_input_manifest_sha256"]:
        raise RuntimeError("P1-A input manifest identity mismatch")
    if sha256_file(p1a_feature_path) != config.EXPECTED["p1a_feature_manifest_sha256"]:
        raise RuntimeError("P1-A feature manifest identity mismatch")
    if sha256_file(tape_path) != config.EXPECTED["wx_tape_sha256"]:
        raise RuntimeError("W/X tape identity mismatch")

    run_provenance = json.loads((e2a / "run_provenance.json").read_text(encoding="utf-8"))
    task_result = json.loads((e2a / "task_result.json").read_text(encoding="utf-8"))
    if task_result.get("status") != "E2A1_PASS" or task_result.get("source_labels_read") is not False:
        raise RuntimeError("E2-A1 did not leave a valid label-free cache receipt")
    if (run_provenance.get("input_manifest_sha256") != config.EXPECTED["p1a_input_manifest_sha256"]
            or run_provenance.get("p1a_feature_manifest_sha256") != config.EXPECTED["p1a_feature_manifest_sha256"]
            or run_provenance.get("wx_tape_sha256") != config.EXPECTED["wx_tape_sha256"]
            or run_provenance.get("labels_read") is not False
            or run_provenance.get("protected_data_opened") is not False):
        raise RuntimeError("E2-A1 provenance is not consistent with the frozen inputs/boundary")

    registered = read_csv(e2a / "registered_input_manifest.csv")
    original_inputs = read_csv(p1a_input_path)
    if registered != original_inputs:
        raise RuntimeError("E2-A1 registered input rows differ from the P1-A frozen whitelist")
    if len(registered) != 102 or Counter(r["data_role"] for r in registered) != {"D-fit": 54, "D-dev": 48}:
        raise RuntimeError("Registered input count/role split is incorrect")

    canonical_path = e2a / "canonical_manifest.csv"
    if sha256_file(canonical_path) != config.EXPECTED["canonical_manifest_sha256"]:
        raise RuntimeError("Canonical manifest SHA-256 mismatch")
    runtime_by_backend: dict[str, dict] = {}
    manifest_hashes = {"canonical": sha256_file(canonical_path)}
    for backend in config.BACKBONES:
        report = json.loads((e2a / backend / "run_report.json").read_text(encoding="utf-8"))
        if report.get("status") != "E2A1_BACKBONE_CACHE_PASS" or report.get("labels_read") is not False:
            raise RuntimeError(f"E2-A1 {backend} receipt failed")
        if report.get("processed_images") != 102 or report.get("failed_items") != 0:
            raise RuntimeError(f"E2-A1 {backend} image count/failure receipt is invalid")
        runtime_by_backend[backend] = report["runtime_identity"]
        manifest_path = e2a / backend / "feature_manifest.csv"
        manifest_hashes[backend] = sha256_file(manifest_path)
        if manifest_hashes[backend] != config.EXPECTED[f"{backend}_feature_manifest_sha256"]:
            raise RuntimeError(f"E2-A1 {backend} feature manifest SHA-256 mismatch")

    registered_by_id = {r["image_id"]: r for r in registered}
    p1a_features = read_csv(p1a_feature_path)
    if len(p1a_features) != 102 or [r["image_id"] for r in p1a_features] != [r["image_id"] for r in registered]:
        raise RuntimeError("P1-A feature manifest image order mismatch")
    for feature_row in p1a_features:
        source = registered_by_id[feature_row["image_id"]]
        if feature_row["source_sha256"] != source["source_sha256"]:
            raise RuntimeError("P1-A feature/source SHA join mismatch")

    # The delegated E2-A1 verifier checks every canonical and feature array, its hash,
    # shape, dtype, finite values, token norms, source join and patch-mask geometry.
    verifier = _load_e2a1_verifier()
    verified = verifier.verify_cache(registered, runtime_by_backend)
    if verified.get("image_count") != 102 or verified.get("source_join_verified") is not True:
        raise RuntimeError("Independent E2-A1 cache verifier returned an incomplete result")
    return {
        "cache_status": "PASS",
        "input_manifest_sha256": config.EXPECTED["p1a_input_manifest_sha256"],
        "p1a_feature_manifest_sha256": config.EXPECTED["p1a_feature_manifest_sha256"],
        "wx_tape_sha256": config.EXPECTED["wx_tape_sha256"],
        "manifest_hashes": manifest_hashes,
        "verified_cache": verified,
        "registered_rows": registered,
        "registered_by_id": registered_by_id,
    }


def validate_tape(registered_by_id: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    rows = read_csv(config.TAPE_PATH)
    if len(rows) != 240:
        raise RuntimeError(f"W/X tape must have 240 exposure rows, found {len(rows)}")
    summary: dict[str, dict] = {}
    for protocol in ("X", "W"):
        subset = [r for r in rows if r["protocol"] == protocol]
        feedback = [r for r in subset if r["feedback_or_future_role"] == "feedback_query"]
        future = [r for r in subset if r["feedback_or_future_role"] == "future_eval_candidate"]
        excluded = [r for r in subset if r["feedback_or_future_role"] not in {"feedback_query", "future_eval_candidate"}]
        if len(feedback) != 12 or len(future) != 24:
            raise RuntimeError(f"Protocol {protocol} has an unexpected tape event/future count")
        if len({r["event_id"] for r in feedback}) != 12 or len({r["image_id"] for r in feedback}) != 12:
            raise RuntimeError(f"Protocol {protocol} feedback events are not unique")
        if len({r["image_id"] for r in future}) != 24:
            raise RuntimeError(f"Protocol {protocol} future candidate IDs are not unique")
        if {r["image_id"] for r in feedback} & {r["image_id"] for r in future}:
            raise RuntimeError(f"Protocol {protocol} query/future image IDs overlap")
        if {r["sample_key"] for r in feedback} & {r["sample_key"] for r in future}:
            raise RuntimeError(f"Protocol {protocol} query/future sample groups overlap")
        if protocol == "X" and {r["product"] for r in feedback} & {r["product"] for r in future}:
            raise RuntimeError("Protocol X feedback/future product sets overlap")
        active_rows = feedback + future
        for row in active_rows:
            source = registered_by_id.get(row["image_id"])
            if source is None or source["data_role"] != "D-dev":
                raise RuntimeError("W/X feedback/future ID is outside the registered D-dev whitelist")
            for field in ("product", "sample_key", "view"):
                if source[field] != row[field]:
                    raise RuntimeError(f"W/X tape {field} does not match registered input metadata")
            if source["exposure_status"] != row["exposure_status"]:
                raise RuntimeError("W/X tape exposure status differs from registered input metadata")
        feedback.sort(key=lambda r: int(r["stream_order"]))
        if [int(r["stream_order"]) for r in feedback] != list(range(1, 13)):
            raise RuntimeError(f"Protocol {protocol} stream order is not 1..12")
        summary[protocol] = {
            "feedback_events": 12,
            "future_views": 24,
            "excluded_rows": len(excluded),
            "feedback_products": sorted({r["product"] for r in feedback}),
            "future_products": sorted({r["product"] for r in future}),
            "feedback_image_ids": [r["image_id"] for r in feedback],
            "future_image_ids": [r["image_id"] for r in future],
        }
    return rows, summary


def first_state_by_item(queue_path: Path, annotation_path: Path,
                        permitted_image_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
    queue = read_csv(queue_path)
    annotations = read_csv(annotation_path)
    by_presentation = {r["presentation_id"]: r for r in annotations}
    if len(by_presentation) != len(annotations) or len(queue) != len(annotations):
        raise RuntimeError(f"Queue/annotation presentation count mismatch in {queue_path.parent}")
    first: dict[str, dict[str, Any]] = {}
    for qrow in sorted(queue, key=lambda r: int(r.get("presentation_order", r.get("presentation_order_index", "0")))):
        pid, image_id = qrow["presentation_id"], qrow["item_id"]
        annotation = by_presentation.get(pid)
        if annotation is None or annotation.get("item_id", image_id) != image_id:
            raise RuntimeError("Annotation presentation does not match frozen queue")
        state = annotation["state"]
        if state not in config.VALID_STATES:
            raise RuntimeError("Invalid annotation state encountered")
        if permitted_image_ids is not None and image_id not in permitted_image_ids:
            continue
        first.setdefault(image_id, {
            "state": state,
            "first_presentation_id": pid,
            "first_queue_order": int(qrow.get("presentation_order", qrow.get("presentation_order_index", "0"))),
            "presentation_count": 0,
        })
        first[image_id]["presentation_count"] += 1
    return first


def load_dfit_labels(dfit_ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict]:
    lite = first_state_by_item(*config.ANNOTATION_SOURCES["D4-Lite"])
    d4b = first_state_by_item(*config.ANNOTATION_SOURCES["D4B"])
    merged = {**lite, **d4b}
    if set(merged) != set(dfit_ids) or len(merged) != 54:
        raise RuntimeError("D-fit first annotation labels do not exactly cover the 54 registered images")
    states = Counter(v["state"] for v in merged.values())
    expected = Counter({"present": 27, "absent_assessable": 25, "unobservable_or_uncertain": 2})
    if states != expected:
        raise RuntimeError(f"D-fit state counts differ from the registered source: {dict(states)}")
    binary_n = sum(config.VALID_STATES[s] is not None for s in (v["state"] for v in merged.values()))
    if binary_n != 52:
        raise RuntimeError("D-fit binary-valid count is not 52")
    return merged, {"total": len(merged), "states": dict(states), "binary_valid": binary_n}


def load_tape_labels(allowed_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict]:
    queue_path, annotations_path = config.ANNOTATION_SOURCES["D-dev"]
    states = first_state_by_item(queue_path, annotations_path, permitted_image_ids=allowed_ids)
    if set(states) != allowed_ids:
        missing = len(allowed_ids - set(states))
        raise RuntimeError(f"Tape-referenced D-dev labels missing for {missing} items")
    return states, {"tape_referenced_items": len(states), "states": dict(Counter(x["state"] for x in states.values()))}


def read_source_type_metadata(expected_id_order: list[str]) -> dict[str, str]:
    """Read only ID and evaluator-only source-type columns from the preserved P1-B join."""
    path = config.P1B_JOINED_METADATA
    if sha256_file(path) != config.EXPECTED["p1b_joined_source_metadata_sha256"]:
        raise RuntimeError("P1-B evaluator metadata fingerprint mismatch")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_index = header.index("image_id")
        source_index = header.index("source_type_evaluator_only")
        result = {}
        for row in reader:
            if len(row) != len(header):
                raise RuntimeError("P1-B evaluator metadata row width mismatch")
            image_id = row[id_index]
            if image_id in result:
                raise RuntimeError("P1-B evaluator metadata contains duplicate IDs")
            result[image_id] = row[source_index]
    if list(result) != expected_id_order:
        raise RuntimeError("P1-B evaluator metadata ID order differs from P1-A")
    return result


def load_feature_bundle(backend: str, image_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Read feature arrays only; identity metadata never enters the returned learner payload."""
    if backend not in config.BACKBONES:
        raise ValueError("unknown backbone")
    manifest = read_csv(config.E2A1_ROOT / backend / "feature_manifest.csv")
    rows_by_id = {r["image_id"]: r for r in manifest}
    if len(rows_by_id) != 102:
        raise RuntimeError(f"{backend} feature manifest is not unique")
    result: dict[str, dict[str, np.ndarray]] = {}
    for image_id in image_ids:
        row = rows_by_id.get(image_id)
        if row is None:
            raise RuntimeError("Requested feature ID is missing from frozen cache")
        bundle = config.E2A1_ROOT / backend / "items" / image_id
        record = json.loads((bundle / "record.json").read_text(encoding="utf-8"))
        arrays = {}
        for name in ("cls_unit", "patch_unit", "valid_patch_mask"):
            spec = record["arrays"][name]
            arrays[name] = np.load(bundle / spec["file"], mmap_mode="r", allow_pickle=False)
        result[image_id] = arrays
    return result


def source_attribute_crosstab(metadata_by_id: dict[str, dict[str, str]],
                              states_by_id: dict[str, dict[str, Any]],
                              source_type_by_id: dict[str, str],
                              role_by_id: dict[str, str]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for image_id, label in states_by_id.items():
        counter[(role_by_id[image_id], source_type_by_id.get(image_id, "UNKNOWN"), label["state"])] += 1
    rows = []
    for (role, source_type, state), count in sorted(counter.items()):
        rows.append({"data_role": role, "source_type_evaluator_only": source_type,
                     "attribute_state_evaluator_only": state, "count": count})
    return rows

