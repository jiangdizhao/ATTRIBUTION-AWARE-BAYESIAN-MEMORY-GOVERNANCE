"""Run the bounded CPU-only LP-D6-P1-B matched baseline replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set before NumPy/scikit-learn imports. This module never imports torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from core import (
    VALID_STATES, FeedbackItem, FutureItem, FrozenLinearPredictor, LogisticRefitLearner,
    RecursiveRidgeLearner, SGDReplayLearner, binary_target, instantiate_refit,
    instantiate_ridge, instantiate_sgd, run_prequential,
)


ROOT = Path(__file__).resolve().parents[3]
P1A_DIR = ROOT / "outputs/research_d6/p1a_features"
D4LITE = ROOT / "research_pilot/LP_W1_D4LITE"
D4B = ROOT / "research_pilot/LP_W1_D4B"
V2 = ROOT / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V"
TAPE_PATH = V2 / "p1_gate0_draft/D6_P1_WX_role_map_feedback_tape.csv"
FEATURE_INDEX_SHA = "ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d"
INPUT_MANIFEST_SHA = "820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63"
TAPE_SHA = "13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1"
CHECKPOINT_SHA = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
HUB_TREE_SHA = "7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c"
PREPROCESS_SHA = "4d3a5906bacafde057f806bbe53ccf24e4960574837e9c47bd273233223ec56e"
APPROVED_BASELINE_COMMIT = "815e5c801be408d08807089d9b48eb372e9258a6"
HISTORICAL_ORIGINAL_RUN_STARTING_HEAD = APPROVED_BASELINE_COMMIT
REVIEWED_P1B_ARCHIVE_COMMIT = "743ee24854465c3f4c8079aaaa74a6e3dd93ad44"
REVIEWED_P1B_ARCHIVE_SHA256 = {
    "scripts/research_d6/p1b/core.py": "f45aafa175dc4691ea505a2b130f01dd20d76f9987d676772d47f2ab656f9f58",
    "scripts/research_d6/p1b/run_experiment.py": "af3f221e885a51aa14d9adbdbde8c8ddf30b163713730e58329460cf7fbce54e",
}
EXPECTED_BRANCH = "stage3-eg-famar-execution"
ALPHAS = (1e-5, 1e-4, 1e-3)
LAMBDAS = (0.1, 1.0, 10.0)
SGD_SEEDS = (0, 1)
MAX_RUNTIME_SECONDS = 3600.0
MAX_OUTPUT_BYTES = 200 * 1024 * 1024
VALID_STATES = set(VALID_STATES)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_scalar(v) for k, v in row.items()})


def json_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_identity() -> tuple[str, str]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return branch, head


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    """Return ancestry status; git errors are not treated as a valid negative result."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    require(result.returncode in (0, 1),
            f"could not verify Git ancestry ({result.returncode}): {result.stderr.strip()}")
    return result.returncode == 0


def validate_revision_policy(
    branch: str,
    head: str,
    ancestry_check: Any | None = None,
) -> dict[str, str]:
    """Accept descendants of the approved baseline while pinning the research branch."""
    require(branch == EXPECTED_BRANCH, f"unexpected branch: {branch}")
    check_ancestor = ancestry_check or git_is_ancestor
    require(check_ancestor(APPROVED_BASELINE_COMMIT, head),
            f"approved baseline {APPROVED_BASELINE_COMMIT} is not an ancestor of execution HEAD {head}")
    return {
        "branch": branch,
        "execution_head": head,
        "approved_baseline_commit": APPROVED_BASELINE_COMMIT,
    }


def require_fingerprint(actual: str, expected: str, label: str) -> None:
    require(actual == expected, f"{label} fingerprint mismatch")


def archived_p1b_implementation_sha256() -> dict[str, str]:
    """Hash the reviewed implementation as stored in its archived commit."""
    result_hashes: dict[str, str] = {}
    for rel_path, expected in REVIEWED_P1B_ARCHIVE_SHA256.items():
        result = subprocess.run(
            ["git", "show", f"{REVIEWED_P1B_ARCHIVE_COMMIT}:{rel_path}"],
            cwd=ROOT, capture_output=True, check=False,
        )
        require(result.returncode == 0,
                f"reviewed P1-B source unavailable at {REVIEWED_P1B_ARCHIVE_COMMIT}:{rel_path}")
        actual = sha_bytes(result.stdout)
        require_fingerprint(actual, expected, f"archived P1-B source {rel_path}")
        result_hashes[rel_path] = actual
    return result_hashes


def current_p1b_implementation_sha256() -> dict[str, str]:
    return {
        "run_experiment.py": sha_file(Path(__file__).resolve()),
        "core.py": sha_file(Path(__file__).with_name("core.py")),
    }


def _source_records() -> dict[str, dict[str, Any]]:
    """Rebuild the authorized whitelist from the three frozen source manifests."""
    sources = [
        (D4LITE / "pilot_manifest.csv", "D-fit", "local_image", "sha256"),
        (D4B / "source_manifest.csv", "D-fit", "local_image", "actual_sha256"),
        (V2 / "selection_manifest.csv", "D-dev", "image_path", "sha256"),
    ]
    records: dict[str, dict[str, Any]] = {}
    for manifest_path, role, path_field, sha_field in sources:
        for row in read_csv(manifest_path):
            image_id = row["item_id"]
            require(image_id not in records, "duplicate image ID across approved source manifests")
            local_path = Path(row[path_field])
            if not local_path.is_absolute():
                local_path = manifest_path.parent / local_path
            dataset = row.get("dataset", "Real-IAD")
            if role == "D-fit":
                require(row.get("image_kind") == "original_real_image", "non-original D-fit asset")
                require(row.get("synthetic", "false").lower() == "false", "synthetic D-fit asset")
            else:
                require("AUTHORIZED" in row.get("authorization_status", ""), "D-dev source lacks pilot authorization")
                require(row.get("exposure_status") == "NEW_STUDY_PILOT_EXPOSED", "D-dev exposure status mismatch")
            records[image_id] = {
                "image_id": image_id, "role": role, "path": local_path,
                "sha256": row[sha_field], "dataset": dataset,
                "source_manifest": manifest_path, "source_row": row,
            }
    require(len(records) == 102, f"approved source manifests yield {len(records)} images, expected 102")
    return records


def capture_preflight() -> dict[str, Any]:
    """Integrity-only gate. Reads authorized image bytes only for SHA verification."""
    branch, head = git_identity()
    revision = validate_revision_policy(branch, head)
    reviewed_archive_sha = archived_p1b_implementation_sha256()
    current_implementation_sha = current_p1b_implementation_sha256()
    input_path = P1A_DIR / "input_manifest.csv"
    feature_index_path = P1A_DIR / "feature_manifest.csv"
    provenance_path = P1A_DIR / "run_provenance.json"
    status_path = P1A_DIR / "extraction_status.json"
    input_rows = read_csv(input_path)
    feature_rows = read_csv(feature_index_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    require(len(input_rows) == 102 and len({r["image_id"] for r in input_rows}) == 102,
            "P1-A input manifest is not exactly 102 unique IDs")
    require(len(feature_rows) == 102 and len({r["image_id"] for r in feature_rows}) == 102,
            "P1-A feature manifest is not exactly 102 unique IDs")
    require(Counter(r["data_role"] for r in input_rows) == {"D-fit": 54, "D-dev": 48},
            "P1-A role counts are not 54 D-fit / 48 D-dev")
    require(status.get("status") == "PASS" and status.get("completed_unique_images") == 102,
            "P1-A extraction status is not PASS/102")
    require_fingerprint(sha_file(input_path), INPUT_MANIFEST_SHA, "P1-A input manifest")
    require_fingerprint(sha_file(feature_index_path), FEATURE_INDEX_SHA, "P1-A feature index")
    sources = _source_records()
    in_by_id = {r["image_id"]: r for r in input_rows}
    feature_by_id = {r["image_id"]: r for r in feature_rows}
    require(set(in_by_id) == set(sources) == set(feature_by_id),
            "independently reconstructed IDs differ from P1-A cache")
    require_fingerprint(provenance.get("checkpoint_sha256", ""), CHECKPOINT_SHA, "DINO checkpoint")
    require_fingerprint(provenance.get("hub_source_tree_sha256", ""), HUB_TREE_SHA, "DINO Hub source tree")
    require_fingerprint(provenance.get("preprocess_fingerprint", ""), PREPROCESS_SHA, "preprocessing")
    p1a_script = ROOT / "scripts/research_d6/p1a/extract_features.py"
    current_p1a_script_sha = sha_file(p1a_script)
    require(all(r["adapter_sha256"] == current_p1a_script_sha for r in feature_rows),
            "P1-A adapter code hash differs from cached feature records")

    source_digests: dict[str, str] = {}
    feature_digests: dict[str, str] = {}
    total_source_bytes = 0
    max_norm_error = 0.0
    for image_id, source in sources.items():
        row = in_by_id[image_id]
        feat = feature_by_id[image_id]
        require(row["data_role"] == source["role"] == feat["data_role"], "role mismatch")
        require(row["source_sha256"] == source["sha256"] == feat["source_sha256"], "source SHA disagreement")
        require(row["dataset"] == source["dataset"], "dataset identity mismatch")
        if source["role"] == "D-fit":
            src = source["source_row"]
            require(row["product"] == src["product"], "D-fit product mismatch")
            require(row["source_split"] == src["source_partition"], "D-fit source split mismatch")
            require(row["dataset_revision"] == src["dataset_revision"], "D-fit dataset revision mismatch")
            require(row["source_relpath"].replace("\\", "/") == src["source_path"].replace("\\", "/"),
                    "D-fit relative source path mismatch")
            if source["source_manifest"].name == "source_manifest.csv":
                require(src["actual_sha256"] == src["remote_sha256"], "D4B downloaded object differs from pinned source")
        else:
            src = source["source_row"]
            require(row["product"] == src["product"] and row["sample_key"] == src["sample_key"],
                    "D-dev product/sample-key mismatch")
            require(row["view"] == src["view_id"] and row["source_split"] == src["source_split"],
                    "D-dev view/split mismatch")
            require(row["source_relpath"].replace("\\", "/") == src["image_relpath"].replace("\\", "/"),
                    "D-dev relative path mismatch")
            require(row["authorization_status"] == src["authorization_status"] and
                    row["exposure_status"] == src["exposure_status"], "D-dev authorization/exposure mismatch")
        require(source["path"].is_file(), "authorized source image path is missing")
        actual_source_sha = sha_file(source["path"])
        require(actual_source_sha == source["sha256"], "authorized source image SHA mismatch")
        source_digests[image_id] = actual_source_sha
        total_source_bytes += source["path"].stat().st_size

        feature_rel = Path(feat["feature_relpath"])
        require(not feature_rel.is_absolute() and ".." not in feature_rel.parts,
                "feature path escapes the P1-A output directory")
        feature_path = P1A_DIR / feature_rel
        require(feature_path.is_file(), "feature payload missing")
        actual_feature_sha = sha_file(feature_path)
        require(actual_feature_sha == feat["feature_sha256"], "feature payload SHA mismatch")
        values = np.load(feature_path, allow_pickle=False)
        require(values.dtype == np.float32 and values.shape == (1024,), "feature dtype/dimension mismatch")
        require(np.isfinite(values).all(), "feature contains nonfinite values")
        norm = float(np.linalg.norm(values.astype(np.float64)))
        max_norm_error = max(max_norm_error, abs(norm - 1.0))
        require(abs(norm - 1.0) <= 1e-5, "feature is not L2 normalized")
        require(feat["feature_dtype"] == "float32" and feat["feature_shape"] == "[1024]" and
                feat["device"] == "cpu" and feat["precision"] == "float32" and
                feat["completion_status"] == "COMPLETE", "feature metadata mismatch")
        require(feat["checkpoint_sha256"] == CHECKPOINT_SHA and
                feat["hub_source_tree_sha256"] == HUB_TREE_SHA and
                feat["preprocess_fingerprint"] == PREPROCESS_SHA,
                "feature provenance fingerprint mismatch")
        feature_digests[image_id] = actual_feature_sha

    tape_rows = read_csv(TAPE_PATH)
    tape_sha = sha_file(TAPE_PATH)
    require_fingerprint(tape_sha, TAPE_SHA, "controlled W/X tape")
    for protocol in ("W", "X"):
        rows = [r for r in tape_rows if r["protocol"] == protocol]
        queries = [r for r in rows if r["feedback_or_future_role"] == "feedback_query"]
        futures = [r for r in rows if r["feedback_or_future_role"] == "future_eval_candidate"]
        qids = {r["image_id"] for r in queries}
        fids = {r["image_id"] for r in futures}
        require(len(queries) == 12 and len(qids) == 12 and len(futures) == 24 and len(fids) == 24,
                f"{protocol} tape event/future count mismatch")
        require(qids.isdisjoint(fids), f"{protocol} image appears as both feedback and future")
        require(qids | fids <= set(feature_by_id), f"{protocol} tape ID has no P1-A feature")
        require(all(feature_by_id[i]["data_role"] == "D-dev" for i in qids | fids),
                f"{protocol} tape role is not D-dev")
        query_groups = {(r["product"], r["sample_key"]) for r in queries}
        future_groups = {(r["product"], r["sample_key"]) for r in futures}
        require(query_groups.isdisjoint(future_groups), f"{protocol} sample-key crosses feedback/future")
        require(sorted(int(r["stream_order"]) for r in queries) == list(range(1, 13)),
                f"{protocol} feedback order mismatch")
        require(len({r["event_id"] for r in queries}) == 12, f"{protocol} repeated feedback event")
        require(all(r["revealed_fields"] == "surface_abrasion_visible_state_after_prediction" for r in queries),
                f"{protocol} revealed field mismatch")
        require(all(r["feedback_cost_status"] == "HISTORICAL_ANNOTATION_COST_UNKNOWN_ASSUMED_ONE_EVENT_UNIT" for r in queries),
                f"{protocol} feedback cost status mismatch")
        group_role: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            group_role.setdefault((row["product"], row["sample_key"]), set()).add(row["group_role"])
        require(all(len(v) == 1 for v in group_role.values()), f"{protocol} group views cross roles")
        require(len(group_role) == 24, f"{protocol} tape does not cover exactly 24 registered groups")
        if protocol == "X":
            require({r["product"] for r in queries} == {"porcelain_doll", "u_block", "usb_adaptor"},
                    "Protocol X feedback products changed")
            require({r["product"] for r in futures} == {"sim_card_set", "toothbrush", "transistor1"},
                    "Protocol X future products changed")

    manifest_paths = [
        D4LITE / "pilot_manifest.csv", D4B / "source_manifest.csv",
        V2 / "selection_manifest.csv", P1A_DIR / "input_manifest.csv",
        P1A_DIR / "feature_manifest.csv", P1A_DIR / "run_provenance.json",
        P1A_DIR / "extraction_status.json", ROOT / "scripts/research_d6/p1a/extract_features.py",
        D4LITE / "annotation_queue.csv", D4LITE / "annotations.csv",
        D4B / "annotation_queue.csv", D4B / "annotations.csv",
        V2 / "annotation_queue.csv", V2 / "annotations.csv", TAPE_PATH,
    ]
    manifest_digests = {str(p.relative_to(ROOT)): sha_file(p) for p in manifest_paths}
    adapter_rows = {r["adapter_sha256"] for r in feature_rows}
    require(len(adapter_rows) == 1, "multiple P1-A extractor versions in the cache")
    snapshot_material = {
        "manifest_digests": manifest_digests,
        "source_digests": source_digests,
        "feature_digests": feature_digests,
    }
    snapshot_sha = sha_bytes(json.dumps(snapshot_material, sort_keys=True).encode("utf-8"))
    return {
        "status": "PASS", "branch": branch, "git_sha": head,
        "execution_head": revision["execution_head"],
        "approved_baseline_commit": revision["approved_baseline_commit"],
        "historical_original_run_starting_head": HISTORICAL_ORIGINAL_RUN_STARTING_HEAD,
        "reviewed_p1b_archive_commit": REVIEWED_P1B_ARCHIVE_COMMIT,
        "reviewed_p1b_archived_implementation_sha256": reviewed_archive_sha,
        "p1b_implementation_sha256": current_implementation_sha,
        "input_manifest_sha256": sha_file(input_path),
        "feature_manifest_sha256": sha_file(feature_index_path),
        "feedback_tape_sha256": tape_sha,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "hub_source_tree_sha256": provenance["hub_source_tree_sha256"],
        "preprocess_fingerprint": provenance["preprocess_fingerprint"],
        "adapter_code_sha256": next(iter(adapter_rows)),
        "n_dfit": 54, "n_ddev": 48, "n_total": 102,
        "source_bytes_hashed": total_source_bytes,
        "max_feature_norm_error": max_norm_error,
        "w_query_ids": sum(r["protocol"] == "W" and r["feedback_or_future_role"] == "feedback_query" for r in tape_rows),
        "x_query_ids": sum(r["protocol"] == "X" and r["feedback_or_future_role"] == "feedback_query" for r in tape_rows),
        "w_future_ids": sum(r["protocol"] == "W" and r["feedback_or_future_role"] == "future_eval_candidate" for r in tape_rows),
        "x_future_ids": sum(r["protocol"] == "X" and r["feedback_or_future_role"] == "future_eval_candidate" for r in tape_rows),
        "snapshot_sha256": snapshot_sha,
        "manifest_digests": manifest_digests,
        "source_digests": source_digests,
        "feature_digests": feature_digests,
        "snapshot_material": snapshot_material,
        "input_rows": input_rows,
        "feature_rows": feature_rows,
        "tape_rows": tape_rows,
        "sources": sources,
    }


def first_annotation_states(queue_path: Path, annotations_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Select the first queued presentation per item; repeated presentations never add events."""
    queue = read_csv(queue_path)
    annotations = read_csv(annotations_path)
    by_pid = {r["presentation_id"]: r for r in annotations}
    require(len(by_pid) == len(annotations), f"duplicate presentation_id in {annotations_path.name}")
    order_field = "presentation_order"
    ordered = sorted(queue, key=lambda r: int(r[order_field]))
    require(len(ordered) == len(annotations), f"queue/annotation presentation count mismatch in {queue_path.parent}")
    seen_presentations: set[str] = set()
    first: dict[str, dict[str, Any]] = {}
    repeats: Counter[str] = Counter()
    for qrow in ordered:
        pid = qrow["presentation_id"]
        require(pid not in seen_presentations and pid in by_pid, "queue presentation missing or duplicated")
        seen_presentations.add(pid)
        annotation = by_pid[pid]
        require(annotation["item_id"] == qrow["item_id"], "queue/annotation item ID mismatch")
        state = annotation["state"]
        require(state in VALID_STATES, f"invalid state in approved annotations: {state}")
        item_id = qrow["item_id"]
        repeats[item_id] += 1
        if item_id not in first:
            first[item_id] = {
                "image_id": item_id, "state": state,
                "first_presentation_id": pid,
                "first_queue_order": int(qrow[order_field]),
                "presentations_for_item": 1,
            }
        else:
            first[item_id]["presentations_for_item"] += 1
    require(seen_presentations == set(by_pid), "unmatched formal annotation presentation")
    return first, dict(repeats)


def load_human_labels(integrity: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    d4l, r1 = first_annotation_states(D4LITE / "annotation_queue.csv", D4LITE / "annotations.csv")
    d4b, r2 = first_annotation_states(D4B / "annotation_queue.csv", D4B / "annotations.csv")
    ddev, r3 = first_annotation_states(V2 / "annotation_queue.csv", V2 / "annotations.csv")
    labels = {**d4l, **d4b, **ddev}
    require(len(labels) == 102, f"first-presentation annotations cover {len(labels)} IDs, expected 102")
    require(set(labels) == {r["image_id"] for r in integrity["input_rows"]},
            "first-presentation annotations do not exactly cover approved features")
    counts = Counter(("D-fit" if image_id in d4l or image_id in d4b else "D-dev", row["state"])
                     for image_id, row in labels.items())
    require(counts == Counter({("D-fit", "present"): 27, ("D-fit", "absent_assessable"): 25,
                               ("D-fit", "unobservable_or_uncertain"): 2,
                               ("D-dev", "present"): 25, ("D-dev", "absent_assessable"): 19,
                               ("D-dev", "unobservable_or_uncertain"): 4}),
            f"first-label counts differ from Gate0; observed={dict(counts)}")
    return labels, {**r1, **r2, **r3}


def source_type_for(image_id: str, source: dict[str, Any]) -> str:
    row = source["source_row"]
    if source["role"] == "D-dev":
        return row.get("view_anomaly_class_metadata_only", "UNKNOWN")
    if "original_source_category" in row:
        return row["original_source_category"]
    parts = row.get("source_path", "").replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else "UNKNOWN"


def build_items(integrity: dict[str, Any], labels: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    feature_rows = {r["image_id"]: r for r in integrity["feature_rows"]}
    input_rows = {r["image_id"]: r for r in integrity["input_rows"]}
    features: dict[str, np.ndarray] = {}
    joined: list[dict[str, Any]] = []
    for image_id, source in integrity["sources"].items():
        feat_row = feature_rows[image_id]
        input_row = input_rows[image_id]
        x = np.load(P1A_DIR / feat_row["feature_relpath"], allow_pickle=False)
        features[image_id] = x
        label = labels[image_id]
        joined.append({
            "image_id": image_id, "data_role": source["role"],
            "dataset": input_row["dataset"], "product": input_row["product"],
            "sample_key": input_row["sample_key"], "view": input_row["view"],
            "source_split": input_row["source_split"],
            "source_type_evaluator_only": source_type_for(image_id, source),
            "attribute_state_first_presentation": label["state"],
            "binary_target_evaluator_only": binary_target(label["state"]),
            "first_presentation_id": label["first_presentation_id"],
            "first_queue_order": label["first_queue_order"],
            "presentation_count_collapsed": label["presentations_for_item"],
            "source_sha256": input_row["source_sha256"],
            "feature_sha256": feat_row["feature_sha256"],
        })
    return features, joined


def build_role_items(protocol: str, tape_rows: list[dict[str, str]], features: dict[str, np.ndarray],
                     labels: dict[str, dict[str, Any]]) -> tuple[list[FeedbackItem], list[FutureItem]]:
    rows = [r for r in tape_rows if r["protocol"] == protocol]
    feedback: list[FeedbackItem] = []
    future: list[FutureItem] = []
    for row in rows:
        role = row["feedback_or_future_role"]
        image_id = row["image_id"]
        if role == "feedback_query":
            feedback.append(FeedbackItem(
                event_id=row["event_id"], image_id=image_id, product=row["product"],
                sample_key=row["sample_key"], stream_order=int(row["stream_order"]),
                state=labels[image_id]["state"], x=features[image_id],
            ))
        elif role == "future_eval_candidate":
            future.append(FutureItem(
                image_id=image_id, product=row["product"], sample_key=row["sample_key"],
                state=labels[image_id]["state"], x=features[image_id],
            ))
    feedback.sort(key=lambda x: x.stream_order)
    return feedback, future


def _record_state(store: dict[str, np.ndarray], prefix: str, learner: Any) -> list[str]:
    fields = []
    for name, value in learner.export_state().items():
        key = f"{prefix}__{name}"
        store[key] = np.asarray(value)
        fields.append(key)
    return fields


def run_pair(protocol: str, pair_id: str, b0_name: str, learner_name: str, learner: Any,
             init_seconds: float, feedback: list[FeedbackItem], future: list[FutureItem],
             state_store: dict[str, np.ndarray], artifacts: dict[str, list[dict[str, Any]]],
             run_started_process: float, max_runtime: float) -> None:
    if time.process_time() - run_started_process >= max_runtime:
        raise TimeoutError("60-minute cumulative CPU experiment runtime limit reached")
    config_id = pair_id
    seed = getattr(learner, "seed", None)
    initial_digest = learner.state_digest()
    initial_state = learner.export_state()
    threshold = learner.threshold
    # All linear learner families expose coefficients/intercept; Ridge uses augmented w.
    if isinstance(learner, RecursiveRidgeLearner):
        b0 = FrozenLinearPredictor.from_linear(
            learner.w[:-1], learner.w[-1], threshold,
            learner.family, config_id,
        )
    else:
        if isinstance(learner, SGDReplayLearner):
            coef = learner.model.coef_[0]
            intercept = learner.model.intercept_[0]
        else:
            coef = learner.model.coef_[0]
            intercept = learner.model.intercept_[0]
        b0 = FrozenLinearPredictor.from_linear(coef, intercept, threshold, learner.family, config_id)
    b0_digest = b0.state_digest()
    if isinstance(learner, RecursiveRidgeLearner):
        require(np.array_equal(b0.coef, np.asarray(learner.w[:-1], dtype=np.float64)) and
                b0.intercept == float(learner.w[-1]), "ridge B0 coefficient copy differs from initialized state")
        require(np.allclose(b0.predict_scores(np.stack([i.x for i in future])),
                            learner.predict_scores(np.stack([i.x for i in future])), atol=1e-6, rtol=1e-6),
                "ridge B0 copy differs from initialized model")
    else:
        require(np.array_equal(b0.coef, np.asarray(coef, dtype=np.float64)) and
                b0.intercept == float(intercept), "logistic B0 coefficient copy differs from initialized state")
        require(np.allclose(b0.predict_scores(np.stack([i.x for i in future])),
                            learner.predict_scores(np.stack([i.x for i in future])), atol=1e-6, rtol=1e-6),
                "logistic B0 copy differs from initialized model")

    def reveal(event_id: str) -> str:
        matches = [e for e in feedback if e.event_id == event_id]
        require(len(matches) == 1, "event has no unique evaluator reveal")
        return matches[0].state

    common = {"protocol": protocol, "config_id": config_id, "seed": seed, "pair_id": pair_id}
    start = time.perf_counter()
    p, m, pm, gm, ev = run_prequential(
        b0, feedback, future, reveal, method=b0_name, **common,
    )
    b0_wall = time.perf_counter() - start
    require(b0.state_digest() == b0_digest, "B0 no-update state changed")
    artifacts["predictions"].extend(p)
    artifacts["metrics"].extend(m)
    artifacts["product_metrics"].extend(pm)
    artifacts["group_metrics"].extend(gm)
    artifacts["events"].extend(ev)
    prefix = f"{protocol}__{config_id}"
    b0_fields = _record_state(state_store, prefix + "__b0_snapshot", b0)
    init_fields = _record_state(state_store, prefix + "__initial", learner)
    artifacts["attempts"].append({
        "protocol": protocol, "method": b0_name, "control_method": "",
        "config_id": config_id, "pair_id": pair_id, "seed": seed,
        "status": "PASS", "initialization_seconds": 0.0,
        "b0_replay_seconds": b0_wall, "learner_replay_seconds": 0.0,
        "state_digest_initial": initial_digest,
        "state_digest_b0": b0_digest,
        "state_digest_final": b0.state_digest(),
        "state_payload_bytes_final": b0.state_payload_bytes(),
        "logistic_n_iter_final": "", "logistic_converged_final": "",
        "initial_state_arrays": ";".join(b0_fields),
        "final_state_arrays": ";".join(b0_fields),
        "error": "",
    })

    start = time.perf_counter()
    p, m, pm, gm, ev = run_prequential(
        learner, feedback, future, reveal, method=learner_name, **common,
    )
    learner_wall = time.perf_counter() - start
    artifacts["predictions"].extend(p)
    artifacts["metrics"].extend(m)
    artifacts["product_metrics"].extend(pm)
    artifacts["group_metrics"].extend(gm)
    artifacts["events"].extend(ev)
    final_fields = _record_state(state_store, prefix + "__final", learner)
    artifacts["attempts"].append({
        "protocol": protocol, "method": learner_name, "control_method": b0_name,
        "config_id": config_id, "pair_id": pair_id, "seed": seed,
        "status": "PASS", "initialization_seconds": init_seconds,
        "b0_replay_seconds": b0_wall, "learner_replay_seconds": learner_wall,
        "state_digest_initial": initial_digest,
        "state_digest_b0": b0_digest,
        "state_digest_final": learner.state_digest(),
        "state_payload_bytes_final": learner.state_payload_bytes(),
        "logistic_n_iter_final": getattr(learner, "n_iter", ""),
        "logistic_converged_final": getattr(learner, "converged", ""),
        "initial_state_arrays": ";".join(init_fields),
        "final_state_arrays": ";".join(final_fields),
        "error": "",
    })


def add_paired_gains(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["protocol"], r["pair_id"], int(r["checkpoint"])): r
              for r in metrics if r["method"].startswith("B0-")}
    rows: list[dict[str, Any]] = []
    for row in metrics:
        if row["method"].startswith("B0-"):
            continue
        base = by_key[(row["protocol"], row["pair_id"], int(row["checkpoint"]))]
        def diff(key: str) -> float | None:
            a, b = row.get(key), base.get(key)
            return None if a is None or b is None else float(a) - float(b)
        rows.append({
            "protocol": row["protocol"], "method": row["method"],
            "config_id": row["config_id"], "seed": row["seed"],
            "pair_id": row["pair_id"], "checkpoint": row["checkpoint"],
            "paired_control": base["method"],
            "ba_gain_vs_paired_b0": diff("ba_product_macro"),
            "ap_gain_vs_paired_b0_including_degenerate": diff("ap_product_macro_including_degenerate"),
            "ap_gain_vs_paired_b0_paired_class": diff("ap_product_macro_paired_class"),
            "harmful_flips_from_previous_checkpoint": row["harmful_flips_from_previous_checkpoint"],
            "future_groups": row["future_groups"],
            "future_binary_views": row["future_binary_views"],
        })
    return rows


def source_attribute_rows(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((r["data_role"], r["source_type_evaluator_only"], r["attribute_state_first_presentation"])
                     for r in joined)
    return [{"data_role": role, "source_type_evaluator_only": source_type,
             "human_attribute_state": state, "n": n}
            for (role, source_type, state), n in sorted(counts.items())]


def package_versions() -> dict[str, str]:
    names = ("numpy", "scikit-learn", "scipy", "threadpoolctl", "joblib", "psutil")
    return {name: importlib.metadata.version(name) for name in names}


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-dir", default="outputs/research_d6/p1b/run_20261113_01")
    parser.add_argument("--max-runtime-seconds", type=float, default=MAX_RUNTIME_SECONDS)
    args = parser.parse_args()

    integrity = capture_preflight()
    print("INDEPENDENT_CACHE_GATE=PASS")
    print(f"  IDs={integrity['n_total']} D-fit={integrity['n_dfit']} D-dev={integrity['n_ddev']}")
    print(f"  source_bytes_hashed={integrity['source_bytes_hashed']} max_norm_error={integrity['max_feature_norm_error']:.3g}")
    print(f"  tape_sha256={integrity['feedback_tape_sha256']}")
    if args.preflight_only:
        return 0

    run_dir = (ROOT / args.run_dir).resolve()
    require(run_dir.is_relative_to(ROOT), "P1-B output path escapes repository")
    if run_dir.exists():
        require(not any(run_dir.iterdir()), f"refusing to overwrite nonempty output directory: {run_dir}")
    else:
        run_dir.mkdir(parents=True)
    run_started_wall = time.perf_counter()
    run_started_process = time.process_time()
    peak_rss = psutil.Process().memory_info().rss
    labels, repeat_counts = load_human_labels(integrity)
    features, joined = build_items(integrity, labels)
    by_id_input = {r["image_id"]: r for r in integrity["input_rows"]}
    dfit_ids = [r["image_id"] for r in integrity["input_rows"] if r["data_role"] == "D-fit"]
    fit_ids = [i for i in dfit_ids if binary_target(labels[i]["state"]) is not None]
    x_fit = np.stack([features[i] for i in fit_ids]).astype(np.float32)
    y_fit = np.asarray([binary_target(labels[i]["state"]) for i in fit_ids], dtype=np.int64)
    require(len(fit_ids) == 52 and Counter(y_fit.tolist()) == Counter({1: 27, 0: 25}),
            "D-fit initialization is not exactly 27 positive / 25 negative binary labels")
    require(not any(labels[i]["state"] in {"unobservable_or_uncertain", "not_applicable"} for i in fit_ids),
            "nonbinary D-fit labels entered initialization")

    write_csv(run_dir / "joined_human_annotations.csv", joined)
    write_csv(run_dir / "source_attribute_diagnostic.csv", source_attribute_rows(joined))
    artifacts: dict[str, list[dict[str, Any]]] = {
        "predictions": [], "metrics": [], "product_metrics": [],
        "group_metrics": [], "events": [], "attempts": [],
    }
    state_store: dict[str, np.ndarray] = {}
    process = psutil.Process()

    def execute(protocol: str, method: str, pair_id: str, b0_name: str,
                learner_factory: Any) -> None:
        if time.process_time() - run_started_process >= args.max_runtime_seconds:
            raise TimeoutError("cumulative CPU runtime budget exhausted")
        feedback, future = build_role_items(protocol, integrity["tape_rows"], features, labels)
        init_start = time.perf_counter()
        learner, fit_seconds = learner_factory()
        if fit_seconds is None:
            fit_seconds = time.perf_counter() - init_start
        run_pair(protocol, pair_id, b0_name, method, learner, fit_seconds,
                 feedback, future, state_store, artifacts, run_started_process,
                 args.max_runtime_seconds)
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        if isinstance(learner, LogisticRefitLearner) and not learner.converged:
            artifacts["attempts"][-1]["status"] = "PARTIAL_MAX_ITER"
            artifacts["attempts"][-1]["error"] = "LBFGS reached fixed max_iter; no retuning performed"

    failures: list[dict[str, Any]] = []
    with threadpool_limits(limits=2):
        for protocol in ("X", "W"):
            for alpha in ALPHAS:
                for seed in SGD_SEEDS:
                    config = f"sgd_alpha_{alpha:.0e}_seed{seed}"
                    pair = f"sgd_alpha_{alpha:.0e}_seed{seed}"
                    try:
                        execute(protocol, "B1-SGD-Replay", pair, "B0-Logistic-SGD",
                                lambda a=alpha, s=seed: (instantiate_sgd(a, s, x_fit, y_fit), None))
                    except Exception as exc:
                        failures.append({"protocol": protocol, "method": "B1-SGD-Replay",
                                         "config_id": config, "seed": seed,
                                         "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            for alpha in ALPHAS:
                config = f"refit_alpha_{alpha:.0e}"
                try:
                    execute(protocol, "B1-Refit", config, "B0-Logistic-Refit",
                            lambda a=alpha: instantiate_refit(a, x_fit, y_fit))
                except Exception as exc:
                    failures.append({"protocol": protocol, "method": "B1-Refit",
                                     "config_id": config, "seed": "", "status": "FAILED",
                                     "error": f"{type(exc).__name__}: {exc}"})
            for lam in LAMBDAS:
                config = f"ridge_lambda_{lam:g}"
                try:
                    execute(protocol, "B2-Recursive-Ridge", config, "B0-Ridge",
                            lambda l=lam: (instantiate_ridge(l, x_fit, y_fit), None))
                except Exception as exc:
                    failures.append({"protocol": protocol, "method": "B2-Recursive-Ridge",
                                     "config_id": config, "seed": "", "status": "FAILED",
                                     "error": f"{type(exc).__name__}: {exc}"})
    attempts = artifacts["attempts"] + failures
    paired = add_paired_gains(artifacts["metrics"])

    # Re-hash the complete source/feature whitelist and registered manifests after execution.
    post_integrity = capture_preflight()
    require(post_integrity["snapshot_sha256"] == integrity["snapshot_sha256"],
            "P1-A source, manifest, or feature asset changed during P1-B")
    require(post_integrity["snapshot_material"] == integrity["snapshot_material"],
            "P1-A asset digest map changed during P1-B")

    write_csv(run_dir / "future_and_query_predictions.csv", artifacts["predictions"])
    write_csv(run_dir / "checkpoint_metrics.csv", artifacts["metrics"])
    write_csv(run_dir / "product_metrics.csv", artifacts["product_metrics"])
    write_csv(run_dir / "group_metrics.csv", artifacts["group_metrics"])
    write_csv(run_dir / "feedback_event_log.csv", artifacts["events"])
    write_csv(run_dir / "paired_gains.csv", paired)
    write_csv(run_dir / "attempted_configurations.csv", attempts)
    state_path = run_dir / "model_states.npz"
    np.savez_compressed(state_path, **state_store)

    metrics_by_pair = {(r["protocol"], r["pair_id"], r["method"], int(r["checkpoint"])): r
                       for r in artifacts["metrics"]}
    state_index = []
    for attempt in artifacts["attempts"]:
        proto, pair = attempt["protocol"], attempt["pair_id"]
        initial = metrics_by_pair[(proto, pair, attempt["method"], 0)]
        final = metrics_by_pair[(proto, pair, attempt["method"], 12)]
        state_index.append({
            "protocol": proto, "pair_id": pair, "method": attempt["method"],
            "initial_state_sha256": initial["state_sha256"],
            "final_state_sha256": final["state_sha256"],
            "state_archive": "model_states.npz",
            "initial_arrays": attempt["initial_state_arrays"],
            "final_arrays": attempt["final_state_arrays"],
        })
    write_csv(run_dir / "model_state_index.csv", state_index)
    input_digests = {
        "d4lite_annotations_sha256": sha_file(D4LITE / "annotations.csv"),
        "d4lite_queue_sha256": sha_file(D4LITE / "annotation_queue.csv"),
        "d4b_annotations_sha256": sha_file(D4B / "annotations.csv"),
        "d4b_queue_sha256": sha_file(D4B / "annotation_queue.csv"),
        "v2_annotations_sha256": sha_file(V2 / "annotations.csv"),
        "v2_queue_sha256": sha_file(V2 / "annotation_queue.csv"),
    }
    elapsed_wall = time.perf_counter() - run_started_wall
    elapsed_cpu = time.process_time() - run_started_process
    final_status = "PASS" if not failures and all(a["status"] == "PASS" for a in attempts if "status" in a) else "PARTIAL"
    package_names = package_versions()
    provenance = {
        "schema": "lp.d6.p1b.run_provenance.v1", "status": final_status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": integrity["branch"], "git_sha": integrity["git_sha"],
        "execution_head": integrity["execution_head"],
        "approved_baseline_commit": integrity["approved_baseline_commit"],
        "historical_original_run_starting_head": integrity["historical_original_run_starting_head"],
        "reviewed_p1b_archive_commit": integrity["reviewed_p1b_archive_commit"],
        "reviewed_p1b_archived_implementation_sha256": integrity["reviewed_p1b_archived_implementation_sha256"],
        "p1b_implementation_sha256_at_execution": integrity["p1b_implementation_sha256"],
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "packages": package_names, "device": "cpu", "cuda_visible_devices": "",
        "torch_imported": False, "thread_limit": 2,
        "feature_manifest_sha256": integrity["feature_manifest_sha256"],
        "input_manifest_sha256": integrity["input_manifest_sha256"],
        "feedback_tape_sha256": integrity["feedback_tape_sha256"],
        "checkpoint_sha256": integrity["checkpoint_sha256"],
        "hub_source_tree_sha256": integrity["hub_source_tree_sha256"],
        "preprocess_fingerprint": integrity["preprocess_fingerprint"],
        "input_snapshot_sha256_before": integrity["snapshot_sha256"],
        "input_snapshot_sha256_after": post_integrity["snapshot_sha256"],
        "annotation_input_sha256": input_digests,
        "candidate_settings": {"sgd_alpha": ALPHAS, "sgd_eta0": 0.01,
                               "sgd_seeds": SGD_SEEDS, "replay_capacity": 12,
                               "refit_alpha": ALPHAS, "refit_C_mapping": "C=1/(alpha*n_binary_training_rows)",
                               "ridge_lambda": LAMBDAS},
        "update_protocol": "fixed Gate0 W/X tape; score query before reveal; update after reveal; then score fixed future set",
        "d_fit_binary_n": len(fit_ids), "d_fit_present_n": int(np.sum(y_fit == 1)),
        "d_fit_absent_n": int(np.sum(y_fit == 0)),
        "w_feedback_events": 12, "w_future_views": 24,
        "x_feedback_events": 12, "x_future_views": 24,
        "attempted_learner_trajectories": sum(not a["method"].startswith("B0-") for a in attempts),
        "attempted_paired_b0_trajectories": sum(a["method"].startswith("B0-") for a in attempts),
        "failed_configurations": failures,
        "elapsed_wall_seconds": elapsed_wall, "elapsed_process_cpu_seconds": elapsed_cpu,
        "peak_observed_rss_bytes": peak_rss, "artifact_bytes": 0,
        "artifact_limit_bytes": MAX_OUTPUT_BYTES,
        "feature_and_source_inputs_unchanged": True,
        "annotation_presentation_repeats_collapsed_to_first_per_item": True,
        "annotation_presentations_and_unique_items": {
            "D4-Lite": {"presentations": len(read_csv(D4LITE / "annotation_queue.csv")),
                        "unique_items": len(read_csv(D4LITE / "pilot_manifest.csv"))},
            "D4B": {"presentations": len(read_csv(D4B / "annotation_queue.csv")),
                    "unique_items": len(read_csv(D4B / "source_manifest.csv"))},
            "D-dev": {"presentations": len(read_csv(V2 / "annotation_queue.csv")),
                      "unique_items": len(read_csv(V2 / "selection_manifest.csv"))},
        },
        "learner_input_contract": "feature vectors; only registered binary feedback label after reveal; no product/source/group fields",
    }
    integrity_public = {k: v for k, v in integrity.items()
                        if k not in {"manifest_digests", "source_digests", "feature_digests", "snapshot_material",
                                     "input_rows", "feature_rows", "tape_rows", "sources"}}
    (run_dir / "preflight_integrity_report.json").write_text(
        json.dumps(integrity_public, indent=2, ensure_ascii=False), encoding="utf-8")
    integrity_detail = {
        "manifest_digests": integrity["manifest_digests"],
        "source_sha256_by_image_id": integrity["source_digests"],
        "feature_sha256_by_image_id": integrity["feature_digests"],
        "snapshot_sha256": integrity["snapshot_sha256"],
    }
    (run_dir / "input_integrity_details.json").write_text(
        json.dumps(integrity_detail, indent=2, ensure_ascii=False), encoding="utf-8")
    for _ in range(3):
        provenance["artifact_bytes"] = directory_bytes(run_dir)
        (run_dir / "run_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
        if directory_bytes(run_dir) == provenance["artifact_bytes"]:
            break
    artifact_bytes = directory_bytes(run_dir)
    require(artifact_bytes <= MAX_OUTPUT_BYTES,
            f"P1-B artifacts exceed 200 MiB limit: {artifact_bytes} bytes")
    print(f"P1B_STATUS={final_status}")
    print(f"RESULT_DIR={run_dir}")
    print(f"TRAJECTORIES={len(attempts)} FAILED={len(failures)}")
    print(f"WALL_SECONDS={elapsed_wall:.2f} CPU_SECONDS={elapsed_cpu:.2f} PEAK_RSS={peak_rss} OUTPUT_BYTES={artifact_bytes}")
    return 0 if final_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
