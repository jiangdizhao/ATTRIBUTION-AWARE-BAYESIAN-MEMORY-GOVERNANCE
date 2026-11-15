"""Read-only E2-B-R1 attribution and MIL spatial diagnostics.

Run in a fresh process with the pinned research interpreter. This module reads
only frozen annotation CSVs, E2-A1 feature caches, and archived E2-B-R1 records.
It never imports an optimizer or image/mask reader and never fits a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "outputs/research_d6/p2/e2b/r1_run_20261116_01"
AGG = RUN / "aggregates/session_001"
E2A1 = ROOT / "outputs/research_d6/p2/e2a1"
OUT = ROOT / "outputs/research_d6/p2/e2b_r1_diagnostics/attribution_spatial_v1"
EXPECTED_HEAD = "cd1d611bac860a5c82b014ffc518a6afa3a0f500"
EXPECTED_BRANCH = "stage3-eg-famar-execution"
SCORE_TOL = 1e-8
SPATIAL_IDENTITY_TOL = 1e-10
JENSEN_TOL = 1e-12
TAU = 1.0
THRESHOLD = 0.0
STATE_TO_Y = {"present": 1, "absent_assessable": 0,
              "unobservable_or_uncertain": None, "not_applicable": None}
METRICS = ("ba", "ap_paired", "roc_auc")
ATTR_OUTPUT_DIR = "outputs/research_d6/p2/e2b_r1_diagnostics/attribution_spatial_v1"


class IntegrityError(RuntimeError):
    """A source identity, saved-state, or numerical contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def score_global(theta: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Frozen E2-B linear score, with bias in the last parameter slot."""
    params = np.asarray(theta, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or params.ndim != 1 or params.shape != (x.shape[1] + 1,):
        raise ValueError("global scoring shape mismatch")
    result = x @ params[:-1] + params[-1]
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite global scores")
    return result


def mil_spatial_components(bag: np.ndarray, theta: np.ndarray, tau: float = TAU,
                           valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    """Return original LME score, same-weight patch-mean counterfactual and spatial term."""
    params = np.asarray(theta, dtype=np.float64)
    z = np.asarray(bag, dtype=np.float64)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
        if z.ndim != 2 or len(mask) != len(z):
            raise ValueError("MIL patch mask shape mismatch")
        z = z[mask]
    if tau <= 0 or z.ndim != 2 or z.shape[1] + 1 != len(params) or len(z) == 0:
        raise ValueError("MIL scoring requires a nonempty valid bag and matching parameters")
    if not np.isfinite(z).all() or not np.isfinite(params).all():
        raise FloatingPointError("nonfinite MIL features or parameters")

    weights, bias = params[:-1], float(params[-1])
    patch_scores = z @ weights
    maximum = float(np.max(patch_scores / tau))
    exp_scaled = np.exp(patch_scores / tau - maximum)
    denominator = float(np.sum(exp_scaled))
    if not math.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("invalid LogMeanExp denominator")
    alpha = exp_scaled / denominator
    lme = tau * (maximum + math.log(denominator) - math.log(len(z)))
    mean_score = float(np.mean(patch_scores))
    full_score = float(bias + lme)
    simple_mean_score = float(bias + mean_score)
    delta = float(lme - mean_score)
    reconstructed = simple_mean_score + delta
    error = abs(full_score - reconstructed)
    if error > SPATIAL_IDENTITY_TOL:
        raise IntegrityError(f"MIL spatial identity error {error} exceeds {SPATIAL_IDENTITY_TOL}")
    if delta < -JENSEN_TOL:
        raise IntegrityError(f"MIL Jensen nonnegativity failed: Delta_spatial={delta}")
    entropy = float(-np.sum(alpha * np.log(np.maximum(alpha, np.finfo(np.float64).tiny))))
    effective = float(np.exp(entropy))
    ordered = np.sort(alpha)[::-1]
    return {
        "score_full": full_score,
        "score_mean": simple_mean_score,
        "delta_spatial": max(delta, 0.0) if delta < 0.0 else delta,
        "reconstruction_error": error,
        "n_valid_patches": int(len(z)),
        "patch_entropy": entropy,
        "effective_patch_count": effective,
        "effective_patch_fraction": effective / len(z),
        "top1_mass": float(np.sum(ordered[:1])),
        "top5_mass": float(np.sum(ordered[:min(5, len(ordered))])),
        "top10_mass": float(np.sum(ordered[:min(10, len(ordered))])),
        "alpha": alpha,
    }


def score_mil(bags: Iterable[np.ndarray], theta: np.ndarray, tau: float = TAU) -> np.ndarray:
    return np.asarray([mil_spatial_components(bag, theta, tau)["score_full"] for bag in bags],
                      dtype=np.float64)


def four_state_parameters(theta0: np.ndarray, theta12: np.ndarray) -> dict[str, np.ndarray]:
    p0 = np.asarray(theta0, dtype=np.float64)
    p12 = np.asarray(theta12, dtype=np.float64)
    if p0.ndim != 1 or p12.shape != p0.shape or len(p0) < 2:
        raise ValueError("endpoint parameter shape mismatch")
    return {
        "F00": p0.copy(),
        "F01": np.concatenate([p0[:-1], p12[-1:]]),
        "F10": np.concatenate([p12[:-1], p0[-1:]]),
        "F11": p12.copy(),
    }


def shapley_two_factor(f00: float, f01: float, f10: float, f11: float) -> dict[str, float]:
    vals = np.asarray([f00, f01, f10, f11], dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("Shapley inputs must be finite")
    phi_w = 0.5 * ((f10 - f00) + (f11 - f01))
    phi_b = 0.5 * ((f01 - f00) + (f11 - f10))
    total = f11 - f00
    residual = phi_w + phi_b - total
    return {"phi_weight": float(phi_w), "phi_bias": float(phi_b),
            "total_delta": float(total), "additivity_residual": float(residual)}


def select_endpoint_pair(checkpoints: Iterable[int]) -> tuple[int, int] | None:
    """Select only real endpoints; never synthesize cp12 for an incomplete run."""
    values = set(int(x) for x in checkpoints)
    if 0 in values and 12 in values:
        return (0, 12)
    return None


def require_score_match(recomputed: np.ndarray, saved: np.ndarray, *, tol: float = SCORE_TOL) -> float:
    left = np.asarray(recomputed, dtype=np.float64).reshape(-1)
    right = np.asarray(saved, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise IntegrityError("score reconstruction shape/finiteness mismatch")
    delta = float(np.max(np.abs(left - right))) if len(left) else 0.0
    if delta > tol:
        raise IntegrityError(f"score reconstruction delta {delta} exceeds frozen tolerance {tol}")
    return delta


def verify_identity(expected: dict[str, Any], actual: dict[str, Any], where: str) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise IntegrityError(f"identity mismatch at {where}: field={key}")


def metric_summary(items: list[dict[str, Any]], scores: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(items) or not np.isfinite(values).all():
        raise ValueError("metric score/item count or finiteness mismatch")
    y_all = [STATE_TO_Y[item["state"]] for item in items]
    binary = np.asarray([value is not None for value in y_all], dtype=bool)
    y = np.asarray([value for value in y_all if value is not None], dtype=np.int8)
    binary_scores = values[binary]
    pred = (values >= THRESHOLD).astype(np.int8)
    binary_pred = pred[binary]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        groups[str(item["product"])].append(i)
    product_rows: list[dict[str, Any]] = []
    ba_vals: list[float] = []
    ap_vals: list[float] = []
    auc_vals: list[float] = []
    degenerate_ap = 0
    for product, indices in sorted(groups.items()):
        bix = [i for i in indices if y_all[i] is not None]
        yy = np.asarray([y_all[i] for i in bix], dtype=np.int8)
        ss = values[bix]
        pp = pred[bix]
        n_pos, n_neg = int(np.sum(yy == 1)), int(np.sum(yy == 0))
        ba = ap = auc = None
        if n_pos and n_neg:
            ba = 0.5 * (float(np.mean(pp[yy == 1] == 1)) + float(np.mean(pp[yy == 0] == 0)))
            ap = float(average_precision_score(yy, ss))
            auc = float(roc_auc_score(yy, ss))
            ba_vals.append(ba)
            ap_vals.append(ap)
            auc_vals.append(auc)
        elif n_pos:
            ap = float(average_precision_score(yy, ss))
            degenerate_ap += 1
        product_rows.append({
            "product": product,
            "n_views": len(indices),
            "n_binary": len(bix),
            "present": n_pos,
            "absent": n_neg,
            "uncertain": sum(items[i]["state"] == "unobservable_or_uncertain" for i in indices),
            "not_applicable": sum(items[i]["state"] == "not_applicable" for i in indices),
            "ba": ba,
            "ap": ap,
            "roc_auc": auc,
            "ba_status": "defined" if n_pos and n_neg else "undefined_missing_class",
            "ap_status": "defined_both_classes" if n_pos and n_neg else
                         "degenerate_all_positive" if n_pos else "undefined_no_positive",
            "roc_auc_status": "defined" if n_pos and n_neg else "undefined_missing_class",
        })
    result = {
        "n_views": len(items),
        "n_binary": int(np.sum(binary)),
        "present": int(np.sum(y == 1)),
        "absent": int(np.sum(y == 0)),
        "uncertain": sum(item["state"] == "unobservable_or_uncertain" for item in items),
        "not_applicable": sum(item["state"] == "not_applicable" for item in items),
        "product_count": len(groups),
        "ba_product_count": len(ba_vals),
        "ba_product_macro": float(np.mean(ba_vals)) if ba_vals else None,
        "ap_paired_product_count": len(ap_vals),
        "ap_product_macro_paired": float(np.mean(ap_vals)) if ap_vals else None,
        "ap_degenerate_product_count": degenerate_ap,
        "roc_auc_product_count": len(auc_vals),
        "roc_auc_product_macro": float(np.mean(auc_vals)) if auc_vals else None,
        "positive_prediction_rate_binary": float(np.mean(binary_pred == 1)) if len(binary_pred) else None,
    }
    return result, product_rows


def verify_r1_bundle(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record_path = path / "record.json"
    state_path_default = path / "state.npz"
    if not record_path.is_file():
        raise IntegrityError(f"missing R1 checkpoint record: {path}")
    record = read_json(record_path)
    state_path = path / str(record.get("state_file", state_path_default.name))
    if not state_path.is_file():
        raise IntegrityError(f"missing R1 checkpoint state: {path}")
    state_sha = sha256_file(state_path)
    if state_sha != record.get("state_npz_sha256"):
        raise IntegrityError(f"R1 state checksum mismatch: {path}")
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    expected_record = hashlib.sha256(canonical_json_bytes({
        "metadata": body, "state_npz_sha256": record.get("state_npz_sha256")
    })).hexdigest()
    if expected_record != record.get("record_sha256"):
        raise IntegrityError(f"R1 record checksum mismatch: {path}")
    with np.load(state_path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    theta = np.ascontiguousarray(arrays["theta"], dtype=np.float64)
    raw_theta_hash = hashlib.sha256(theta.tobytes()).hexdigest()
    shaped_hash = hashlib.sha256(repr(theta.shape).encode("ascii") + theta.tobytes()).hexdigest()
    if record.get("theta_sha256") not in {raw_theta_hash, shaped_hash}:
        raise IntegrityError(f"R1 parameter checksum mismatch: {path}")
    arrays["theta"] = theta
    return record, arrays


def read_first_annotation_states(queue_path: Path, annotations_path: Path) -> dict[str, str]:
    queue = read_csv(queue_path)
    annotations = read_csv(annotations_path)
    by_presentation = {row["presentation_id"]: row for row in annotations}
    if len(by_presentation) != len(annotations) or len(queue) != len(annotations):
        raise IntegrityError(f"annotation/queue cardinality mismatch: {queue_path.parent}")
    order_key = "presentation_order" if queue and "presentation_order" in queue[0] else "presentation_order_index"
    states: dict[str, str] = {}
    for qrow in sorted(queue, key=lambda row: int(row[order_key])):
        annotation = by_presentation.get(qrow["presentation_id"])
        if annotation is None or annotation.get("item_id", qrow["item_id"]) != qrow["item_id"]:
            raise IntegrityError(f"annotation queue mapping mismatch: {queue_path.parent}")
        state = annotation.get("state", "")
        if state not in STATE_TO_Y:
            raise IntegrityError(f"invalid frozen annotation state in {annotations_path}")
        states.setdefault(qrow["item_id"], state)
    return states


def current_input_hashes() -> dict[str, str]:
    paths = {
        "D-dev_annotations": "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/annotations.csv",
        "D-dev_queue": "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/annotation_queue.csv",
        "D4-Lite_annotations": "research_pilot/LP_W1_D4LITE/annotations.csv",
        "D4-Lite_queue": "research_pilot/LP_W1_D4LITE/annotation_queue.csv",
        "D4B_annotations": "research_pilot/LP_W1_D4B/annotations.csv",
        "D4B_queue": "research_pilot/LP_W1_D4B/annotation_queue.csv",
        "p1a_feature_manifest": "outputs/research_d6/p1a_features/feature_manifest.csv",
        "p1a_input_manifest": "outputs/research_d6/p1a_features/input_manifest.csv",
        "p1b_source_metadata": "outputs/research_d6/p1b/run_20261113_01/joined_human_annotations.csv",
        "tape": "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/p1_gate0_draft/D6_P1_WX_role_map_feedback_tape.csv",
    }
    return {key: sha256_file(ROOT / relative) for key, relative in paths.items()}


def preflight() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing attribution/spatial output: {OUT}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if (head, branch) != (EXPECTED_HEAD, EXPECTED_BRANCH):
        raise IntegrityError(f"archive identity mismatch: head={head}, branch={branch}")
    ignored = subprocess.run(["git", "check-ignore", "-q", ATTR_OUTPUT_DIR], cwd=ROOT)
    if ignored.returncode != 0:
        raise IntegrityError("private output path is not Git-ignored")

    identity = read_json(RUN / "run_identity.json")
    result = read_json(RUN / "run_result.json")
    manifest = read_json(AGG / "aggregate_manifest.json")
    if identity.get("branch") != EXPECTED_BRANCH or identity.get("head") != "072e11adaffd85cf64b85b4708befe45cef92018":
        raise IntegrityError("historical R1 identity differs from archived run")
    if result.get("status") != "E2B_R1_PARTIAL" or manifest.get("status") != "E2B_R1_PARTIAL":
        raise IntegrityError("unexpected R1 status")
    if (result.get("completed_learner_trajectories"), result.get("expected_learner_trajectories"),
            result.get("completed_b0_controls"), result.get("expected_b0_controls"),
            result.get("checkpoint_rows")) != (27, 28, 28, 28, 725):
        raise IntegrityError("R1 trajectory/checkpoint counts differ from archived contract")
    if result.get("input_hashes_unchanged") is not True or result.get("protected_assets_unchanged") is not True:
        raise IntegrityError("R1 run receipt has a failed input/protected-assets check")
    for filename, expected in manifest.get("files", {}).items():
        actual = sha256_file(AGG / filename)
        if actual != expected:
            raise IntegrityError(f"R1 aggregate file checksum mismatch: {filename}")
    current_hashes = current_input_hashes()
    for key, actual in current_hashes.items():
        if identity.get("input_hashes", {}).get(key) != actual:
            raise IntegrityError(f"frozen R1 input hash mismatch: {key}")

    input_manifest_path = ROOT / "outputs/research_d6/p1a_features/input_manifest.csv"
    cache_input_path = E2A1 / "registered_input_manifest.csv"
    canonical_path = E2A1 / "canonical_manifest.csv"
    if sha256_file(input_manifest_path) != identity.get("input_manifest_sha256"):
        raise IntegrityError("P1-A input manifest does not match R1 identity")
    if sha256_file(cache_input_path) != identity.get("input_manifest_sha256"):
        raise IntegrityError("E2-A1 registered input manifest does not match R1 identity")
    if sha256_file(canonical_path) != identity.get("canonical_manifest_sha256"):
        raise IntegrityError("E2-A1 canonical manifest does not match R1 identity")
    feature_manifests = {}
    for backend, key in (("dinov2", "feature_manifest_sha256"),
                         ("dinov3", "dinov3_feature_manifest_sha256")):
        p = E2A1 / backend / "feature_manifest.csv"
        actual = sha256_file(p)
        if actual != identity.get(key):
            raise IntegrityError(f"E2-A1 {backend} feature manifest identity mismatch")
        feature_manifests[backend] = actual

    input_rows = read_csv(input_manifest_path)
    if len(input_rows) != 102 or len({row["image_id"] for row in input_rows}) != 102:
        raise IntegrityError("registered feature input manifest must contain 102 unique items")
    roles = Counter(row["data_role"] for row in input_rows)
    if roles != Counter({"D-fit": 54, "D-dev": 48}):
        raise IntegrityError(f"registered input role counts changed: {roles}")
    return {
        "head": head,
        "branch": branch,
        "r1_execution_head": identity["head"],
        "r1_run_identity_sha256": sha256_file(RUN / "run_identity.json"),
        "aggregate_manifest_sha256": sha256_file(AGG / "aggregate_manifest.json"),
        "aggregate_file_count": len(manifest["files"]),
        "aggregate_files_verified": len(manifest["files"]),
        "input_hashes": current_hashes,
        "input_manifest_sha256": identity["input_manifest_sha256"],
        "canonical_manifest_sha256": identity["canonical_manifest_sha256"],
        "feature_manifest_sha256": feature_manifests,
        "trajectory_rows": result["trajectory_rows"],
        "checkpoint_rows": result["checkpoint_rows"],
        "complete_learner_trajectories": result["completed_learner_trajectories"],
        "complete_b0_controls": result["completed_b0_controls"],
        "registered_input_counts": dict(roles),
    }


def load_feature_cache(backend: str, input_rows: list[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    feature_rows = read_csv(E2A1 / backend / "feature_manifest.csv")
    feature_by_id = {row["image_id"]: row for row in feature_rows}
    source_by_id = {row["image_id"]: row for row in input_rows}
    if len(feature_by_id) != 102 or set(feature_by_id) != set(source_by_id):
        raise IntegrityError(f"{backend} feature cache does not cover the registered input set")
    cache: dict[str, dict[str, np.ndarray]] = {}
    for image_id in source_by_id:
        feature_row = feature_by_id[image_id]
        source_row = source_by_id[image_id]
        if feature_row["source_sha256"] != source_row["source_sha256"]:
            raise IntegrityError(f"{backend} feature/source identity mismatch")
        item_dir = E2A1 / backend / "items" / image_id
        record_path = item_dir / "record.json"
        record = read_json(record_path)
        if sha256_file(record_path) != feature_row["record_sha256"]:
            raise IntegrityError(f"{backend} item record hash mismatch")
        ident = record.get("identity", {})
        verify_identity({
            "image_id": image_id,
            "source_sha256": source_row["source_sha256"],
            "registered_input_manifest_sha256": sha256_file(E2A1 / "registered_input_manifest.csv"),
            "model": feature_row["model"],
            "weights_sha256": feature_row["weights_sha256"],
            "extractor_identity_sha256": feature_row["extractor_identity_sha256"],
        }, ident, f"E2-A1/{backend}/{image_id}")
        arrays: dict[str, np.ndarray] = {}
        mapping = {"cls_unit": "cls_file_sha256", "patch_unit": "patch_file_sha256",
                   "valid_patch_mask": "mask_file_sha256"}
        for name, manifest_key in mapping.items():
            spec = record.get("arrays", {}).get(name)
            if not spec:
                raise IntegrityError(f"missing {backend} feature array {name}")
            path = item_dir / spec["file"]
            if sha256_file(path) != spec["file_sha256"] or spec["file_sha256"] != feature_row[manifest_key]:
                raise IntegrityError(f"{backend} feature file SHA mismatch: {name}")
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(arr.shape) != spec["shape"] or sha256_array(arr) != spec["array_sha256"]:
                raise IntegrityError(f"{backend} feature array identity mismatch: {name}")
            arrays[name] = np.asarray(arr)
        cls = arrays["cls_unit"]
        patches = arrays["patch_unit"]
        mask = np.asarray(arrays["valid_patch_mask"], dtype=np.bool_).reshape(-1)
        if cls.ndim != 1 or patches.ndim != 2 or patches.shape[1] != len(cls) or len(mask) != len(patches):
            raise IntegrityError(f"{backend} cached feature shapes disagree")
        if not mask.any() or not np.isfinite(cls).all() or not np.isfinite(patches[mask]).all():
            raise IntegrityError(f"{backend} cached features contain invalid values")
        if int(np.sum(mask)) != int(feature_row["valid_patch_count"]):
            raise IntegrityError(f"{backend} valid patch count mismatch")
        cache[image_id] = {
            "cls": np.asarray(cls, dtype=np.float32),
            "patches": np.asarray(patches[mask], dtype=np.float32),
            "mask": mask,
            "registered_patch_count": int(len(mask)),
        }
    return cache


def global_feature(method: str, feature: dict[str, np.ndarray]) -> np.ndarray:
    cls = np.asarray(feature["cls"], dtype=np.float64).reshape(-1)
    if method == "G-CLS":
        return cls
    patches = np.asarray(feature["patches"], dtype=np.float64)
    mean = patches.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise IntegrityError("cached valid patch mean has invalid norm")
    mean /= norm
    if method == "G-Mean":
        return mean
    if method == "G-Concat":
        combined = np.concatenate([cls, mean])
        norm = float(np.linalg.norm(combined))
        if not math.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise IntegrityError("cached concatenated feature has invalid norm")
        return combined / norm
    raise ValueError(f"unknown frozen Global readout: {method}")


def prediction_change_counts(items: list[dict[str, Any]], before: np.ndarray,
                             after: np.ndarray) -> dict[str, int]:
    b = np.asarray(before, dtype=np.float64).reshape(-1) >= THRESHOLD
    a = np.asarray(after, dtype=np.float64).reshape(-1) >= THRESHOLD
    if len(items) != len(b) or len(a) != len(b):
        raise ValueError("prediction count mismatch")
    binary_ix = [i for i, row in enumerate(items) if STATE_TO_Y[row["state"]] is not None]
    corrected = harmful = 0
    for i in binary_ix:
        y = STATE_TO_Y[items[i]["state"]]
        if b[i] != a[i]:
            corrected += int(int(a[i]) == y)
            harmful += int(int(b[i]) == y)
    return {
        "threshold_flips_all": int(np.sum(b != a)),
        "threshold_flips_binary": sum(int(b[i] != a[i]) for i in binary_ix),
        "corrected_binary_flips": corrected,
        "harmful_binary_flips": harmful,
    }


def ranking_change_counts(items: list[dict[str, Any]], full: np.ndarray,
                          mean: np.ndarray) -> dict[str, int]:
    y = np.asarray([STATE_TO_Y[row["state"]] if STATE_TO_Y[row["state"]] is not None else -1
                    for row in items], dtype=np.int8)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    full_delta = np.asarray(full, dtype=np.float64)[pos, None] - np.asarray(full, dtype=np.float64)[neg][None, :]
    mean_delta = np.asarray(mean, dtype=np.float64)[pos, None] - np.asarray(mean, dtype=np.float64)[neg][None, :]
    full_sign, mean_sign = np.sign(full_delta), np.sign(mean_delta)
    return {
        "positive_negative_pairs": int(full_delta.size),
        "order_discordances": int(np.sum(full_sign != mean_sign)),
        "full_ties": int(np.sum(full_sign == 0)),
        "mean_ties": int(np.sum(mean_sign == 0)),
    }


def parse_frozen_sources(input_rows: list[dict[str, str]]) -> tuple[
        dict[str, str], dict[str, str], dict[str, dict[str, str]], dict[str, list[str]], list[str]]:
    dfit_states: dict[str, str] = {}
    for qrel, arel in (
        ("research_pilot/LP_W1_D4LITE/annotation_queue.csv", "research_pilot/LP_W1_D4LITE/annotations.csv"),
        ("research_pilot/LP_W1_D4B/annotation_queue.csv", "research_pilot/LP_W1_D4B/annotations.csv"),
    ):
        current = read_first_annotation_states(ROOT / qrel, ROOT / arel)
        if set(dfit_states) & set(current):
            raise IntegrityError("duplicate D-fit item across frozen label sources")
        dfit_states.update(current)
    ddev_states = read_first_annotation_states(
        ROOT / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/annotation_queue.csv",
        ROOT / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/annotations.csv")
    source_by_id = {row["image_id"]: row for row in input_rows}
    dfit_ids = [row["image_id"] for row in input_rows if row["data_role"] == "D-fit"]
    ddev_ids = [row["image_id"] for row in input_rows if row["data_role"] == "D-dev"]
    if set(dfit_ids) != set(dfit_states) or set(ddev_ids) != set(ddev_states):
        raise IntegrityError("frozen human annotations do not exactly cover registered D-fit/D-dev")
    dfit_binary_ids = [image_id for image_id in dfit_ids if STATE_TO_Y[dfit_states[image_id]] is not None]
    dfit_y = Counter(STATE_TO_Y[dfit_states[image_id]] for image_id in dfit_binary_ids)
    if dfit_y != Counter({1: 27, 0: 25}):
        raise IntegrityError(f"D-fit binary support changed: {dfit_y}")
    tape = read_csv(ROOT / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/p1_gate0_draft/D6_P1_WX_role_map_feedback_tape.csv")
    future_ids = {
        protocol: [row["image_id"] for row in tape
                   if row["protocol"] == protocol and row["feedback_or_future_role"] == "future_eval_candidate"]
        for protocol in ("W", "X")
    }
    if any(len(ids) != 24 or len(set(ids)) != 24 for ids in future_ids.values()):
        raise IntegrityError("registered W/X future tape must have 24 unique views each")
    return dfit_states, ddev_states, source_by_id, future_ids, dfit_binary_ids


def validate_future_items(record: dict[str, Any], protocol: str, source_by_id: dict[str, dict[str, str]],
                          ddev_states: dict[str, str], expected_ids: list[str]) -> list[dict[str, Any]]:
    saved = record.get("item_predictions", [])
    saved_ids = [item.get("image_id") for item in saved]
    if len(saved_ids) != 24 or len(set(saved_ids)) != 24 or set(saved_ids) != set(expected_ids):
        raise IntegrityError(f"future identity mismatch for protocol {protocol}")
    items = []
    for item in saved:
        image_id = item["image_id"]
        source = source_by_id[image_id]
        state = ddev_states[image_id]
        if (item.get("product"), item.get("sample_key"), item.get("view")) != (
                source["product"], source["sample_key"], source["view"]):
            raise IntegrityError("future item metadata differs from registered manifest")
        if item.get("attribute_state_evaluator_only") != state:
            raise IntegrityError("future saved state differs from frozen human first label")
        if item.get("binary_target_evaluator_only") != STATE_TO_Y[state]:
            raise IntegrityError("future saved target differs from frozen human first label")
        items.append({"image_id": image_id, "product": source["product"],
                      "sample_key": source["sample_key"], "view": source["view"],
                      "state": state})
    return items


def scores_for(method: str, theta: np.ndarray, ids: list[str], cache: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    if method.startswith("G-"):
        features = np.stack([global_feature(method, cache[image_id]) for image_id in ids])
        return score_global(theta, features)
    bags = [cache[image_id]["patches"] for image_id in ids]
    return score_mil(bags, theta, TAU)


def checkpoint_dirs(trajectory_id: str, side: str) -> list[Path]:
    suffix = f"__{side}"
    if not trajectory_id.endswith(suffix):
        raise IntegrityError(f"trajectory ID does not encode its registered side: {trajectory_id}")
    pair_id = trajectory_id[:-len(suffix)]
    base = RUN / "trajectories" / pair_id / side
    return sorted(base.glob("checkpoint_[0-9][0-9]"))


def checkpoint_map(trajectory_id: str, side: str) -> dict[int, Path]:
    return {int(path.name[-2:]): path for path in checkpoint_dirs(trajectory_id, side)}


def compact_stats(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(np.mean(x)), "median": float(np.median(x)),
            "std": float(np.std(x)), "min": float(np.min(x)), "max": float(np.max(x)),
            "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75))}


def corr_or_none(x: list[float], y: list[float]) -> float | None:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _bundle_for(trajectory_id: str, side: str, checkpoint: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    suffix = f"__{side}"
    if not trajectory_id.endswith(suffix):
        raise IntegrityError(f"trajectory ID does not encode its registered side: {trajectory_id}")
    pair_id = trajectory_id[:-len(suffix)]
    return verify_r1_bundle(RUN / "trajectories" / pair_id / side / f"checkpoint_{checkpoint:02d}")


def run_diagnostics() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    identity = preflight()
    run_identity = read_json(RUN / "run_identity.json")
    input_rows = read_csv(ROOT / "outputs/research_d6/p1a_features/input_manifest.csv")
    dfit_states, ddev_states, source_by_id, future_ids_by_protocol, dfit_binary_ids = parse_frozen_sources(input_rows)
    dfit_items = [{"image_id": image_id, "product": source_by_id[image_id]["product"],
                   "sample_key": source_by_id[image_id]["sample_key"],
                   "view": source_by_id[image_id]["view"], "state": dfit_states[image_id]}
                  for image_id in dfit_binary_ids]
    statuses = read_csv(AGG / "trajectory_status.csv")
    if len(statuses) != 56:
        raise IntegrityError("R1 aggregate does not contain 56 registered trajectories")
    status_by_id = {row["trajectory_id"]: row for row in statuses}
    if len(status_by_id) != 56:
        raise IntegrityError("duplicate R1 trajectory status")
    learners = [row for row in statuses if row["side"] == "learner"]
    controls = [row for row in statuses if row["side"] == "control"]
    if len(learners) != 28 or len(controls) != 28:
        raise IntegrityError("R1 learner/control counts do not match 28 pairs")
    complete_learners = [row for row in learners if row["status"] == "PASS" and int(row["last_checkpoint"]) == 12]
    failed_learners = [row for row in learners if row["status"] != "PASS"]
    if len(complete_learners) != 27 or len(failed_learners) != 1:
        raise IntegrityError("R1 completion/failure profile differs from the registered output")
    failure_status = failed_learners[0]
    if (failure_status["trajectory_id"] != "dinov2__G-CLS__X__learner"
            or int(failure_status["last_checkpoint"]) != 9):
        raise IntegrityError("R1 failed trajectory identity/checkpoint boundary changed")
    if any(row["status"] != "PASS" or int(row["last_checkpoint"]) != 12 for row in controls):
        raise IntegrityError("one or more frozen B0 controls are incomplete")

    # Read and validate each frozen DINO feature bundle once per backbone.
    all_ids = [row["image_id"] for row in input_rows]
    attribution_rows: list[dict[str, Any]] = []
    attribution_product_rows: list[dict[str, Any]] = []
    attribution_item_rows: list[dict[str, Any]] = []
    score_rebuild_rows: list[dict[str, Any]] = []
    blocked_tracks: list[dict[str, Any]] = []
    spatial_item_rows: list[dict[str, Any]] = []
    spatial_pair_change_rows: list[dict[str, Any]] = []
    mil_counterfactual_rows: list[dict[str, Any]] = []
    mil_counterfactual_product_rows: list[dict[str, Any]] = []
    private_parameters: dict[str, np.ndarray] = {}
    max_score_delta = 0.0
    max_spatial_identity_error = 0.0
    min_delta_spatial = math.inf
    b0_state_checks = 0

    for backend in ("dinov2", "dinov3"):
        cache = load_feature_cache(backend, input_rows)

        # Verify all 28 original B0 trajectories remain frozen and still pair with learner cp0.
        for control in [row for row in controls if row["backbone"] == backend]:
            tid = control["trajectory_id"]
            cpoints = checkpoint_map(tid, "control")
            if sorted(cpoints) != list(range(13)):
                raise IntegrityError(f"B0 checkpoint sequence is incomplete: {tid}")
            c0_rec, c0_state = verify_r1_bundle(cpoints[0])
            c0_theta = c0_state["theta"]
            c0_scores = np.asarray(c0_state["future_scores"], dtype=np.float64).reshape(-1)
            pair_id = control["pair_id"]
            # The paired learner identity is looked up by pair_id without consulting outcomes.
            learner_row = next(row for row in learners if row["pair_id"] == pair_id)
            l0_rec, l0_state = _bundle_for(learner_row["trajectory_id"], "learner", 0)
            if not np.array_equal(c0_theta, l0_state["theta"]) or not np.array_equal(c0_scores, l0_state["future_scores"]):
                raise IntegrityError(f"B0 and learner checkpoint-0 initialization differ: {pair_id}")
            items = validate_future_items(c0_rec, control["protocol"], source_by_id, ddev_states,
                                          future_ids_by_protocol[control["protocol"]])
            ids = [item["image_id"] for item in items]
            c0_rebuilt = scores_for(control["method"], c0_theta, ids, cache)
            delta = require_score_match(c0_rebuilt, c0_scores, tol=SCORE_TOL)
            max_score_delta = max(max_score_delta, delta)
            score_rebuild_rows.append({"trajectory_id": tid, "side": "control", "checkpoint": 0,
                                       "n_future": len(ids), "max_abs_score_delta": delta, "status": "PASS"})
            for checkpoint, path in cpoints.items():
                rec, arr = verify_r1_bundle(path)
                if not np.array_equal(arr["theta"], c0_theta) or not np.array_equal(arr["future_scores"], c0_scores):
                    raise IntegrityError(f"frozen B0 state/score changed at checkpoint {checkpoint}: {pair_id}")
                if rec.get("identity") != c0_rec.get("identity") or rec.get("checkpoint") != checkpoint:
                    raise IntegrityError(f"B0 checkpoint identity mismatch: {pair_id}/{checkpoint}")
                b0_state_checks += 1

        # Endpoint parameter reconstruction and the four-state Shapley-style analysis.
        for status in [row for row in complete_learners if row["backbone"] == backend]:
            tid = status["trajectory_id"]
            pmap = checkpoint_map(tid, "learner")
            if select_endpoint_pair(pmap) != (0, 12):
                raise IntegrityError(f"complete trajectory lacks true cp0/cp12: {tid}")
            rec0, arr0 = verify_r1_bundle(pmap[0])
            rec12, arr12 = verify_r1_bundle(pmap[12])
            theta0, theta12 = arr0["theta"], arr12["theta"]
            if rec0["identity"] != rec12["identity"] or rec0["identity"].get("head") != run_identity["head"]:
                raise IntegrityError(f"trajectory model identity changed across endpoints: {tid}")
            if rec0.get("checkpoint") != 0 or rec12.get("checkpoint") != 12:
                raise IntegrityError(f"endpoint checkpoint metadata mismatch: {tid}")
            items = validate_future_items(rec0, status["protocol"], source_by_id, ddev_states,
                                          future_ids_by_protocol[status["protocol"]])
            items12 = validate_future_items(rec12, status["protocol"], source_by_id, ddev_states,
                                            future_ids_by_protocol[status["protocol"]])
            ids = [item["image_id"] for item in items]
            if ids != [item["image_id"] for item in items12]:
                raise IntegrityError(f"endpoint future ordering changed: {tid}")
            state_scores: dict[str, np.ndarray] = {}
            for checkpoint, rec, arrays, theta in ((0, rec0, arr0, theta0), (12, rec12, arr12, theta12)):
                rebuilt = scores_for(status["method"], theta, ids, cache)
                saved = np.asarray(arrays["future_scores"], dtype=np.float64).reshape(-1)
                try:
                    delta = require_score_match(rebuilt, saved, tol=SCORE_TOL)
                except IntegrityError as exc:
                    blocked_tracks.append({"trajectory_id": tid, "checkpoint": checkpoint,
                                           "reason": str(exc), "status": "BLOCKED_SCORE_RECONSTRUCTION"})
                    state_scores = {}
                    break
                max_score_delta = max(max_score_delta, delta)
                state_scores[f"cp{checkpoint}"] = rebuilt
                score_rebuild_rows.append({"trajectory_id": tid, "side": "learner", "checkpoint": checkpoint,
                                           "n_future": len(ids), "max_abs_score_delta": delta,
                                           "status": "PASS"})
            if not state_scores:
                continue

            params = four_state_parameters(theta0, theta12)
            scorer_scores = {
                key: (score_global(value, np.stack([global_feature(status["method"], cache[i]) for i in ids]))
                      if status["method"].startswith("G-") else
                      score_mil([cache[i]["patches"] for i in ids], value, TAU))
                for key, value in params.items()
            }
            # Ensure the scorer dispatch and checkpoint reconstruction use identical feature pathways.
            if (np.max(np.abs(scorer_scores["F00"] - state_scores["cp0"])) > SCORE_TOL
                    or np.max(np.abs(scorer_scores["F11"] - state_scores["cp12"])) > SCORE_TOL):
                raise IntegrityError(f"four-state endpoint score reconstruction mismatch: {tid}")
            private_parameters[f"{tid}__theta0"] = theta0
            private_parameters[f"{tid}__theta12"] = theta12
            metric_by_state: dict[str, dict[str, Any]] = {}
            for key, scores in scorer_scores.items():
                metrics, product_rows = metric_summary(items, scores)
                metric_by_state[key] = metrics
                for row in product_rows:
                    attribution_product_rows.append({"trajectory_id": tid, "state": key, **row})
                for i, (item, score) in enumerate(zip(items, scores)):
                    attribution_item_rows.append({"trajectory_id": tid, "backbone": backend,
                        "method": status["method"], "seed": status["seed"], "protocol": status["protocol"],
                        "four_state": key, **item, "score": float(score),
                        "prediction": int(score >= THRESHOLD)})
            row = {"trajectory_id": tid, "pair_id": status["pair_id"], "backbone": backend,
                   "method": status["method"], "seed": status["seed"], "protocol": status["protocol"],
                   "n_future": len(items), "n_binary": metric_by_state["F00"]["n_binary"]}
            for metric, key in (("ba", "ba_product_macro"), ("ap_paired", "ap_product_macro_paired"),
                                ("roc_auc", "roc_auc_product_macro")):
                vals = {state: metric_by_state[state][key] for state in ("F00", "F01", "F10", "F11")}
                if any(value is None for value in vals.values()):
                    raise IntegrityError(f"required evaluator metric undefined for {tid}/{metric}")
                shapley = shapley_two_factor(vals["F00"], vals["F01"], vals["F10"], vals["F11"])
                if abs(shapley["additivity_residual"]) > 1e-12:
                    raise IntegrityError(f"Shapley additivity failed for {tid}/{metric}")
                for state, value in vals.items():
                    row[f"{metric}_{state}"] = value
                row[f"{metric}_phi_weight"] = shapley["phi_weight"]
                row[f"{metric}_phi_bias"] = shapley["phi_bias"]
                row[f"{metric}_delta"] = shapley["total_delta"]
                row[f"{metric}_additivity_residual"] = shapley["additivity_residual"]
            flip_info = prediction_change_counts(items, scorer_scores["F00"], scorer_scores["F11"])
            row.update(flip_info)
            row["positive_rate_F00"] = metric_by_state["F00"]["positive_prediction_rate_binary"]
            row["positive_rate_F11"] = metric_by_state["F11"]["positive_prediction_rate_binary"]
            attribution_rows.append(row)

        # Failed learner: audit only actually persisted checkpoints 0--9, never synthesize cp12.
        for status in [row for row in failed_learners if row["backbone"] == backend]:
            tid = status["trajectory_id"]
            pmap = checkpoint_map(tid, "learner")
            if sorted(pmap) != list(range(10)):
                raise IntegrityError("failed trajectory does not have precisely stored checkpoints 0--9")
            for checkpoint, path in pmap.items():
                rec, arrays = verify_r1_bundle(path)
                items = validate_future_items(rec, status["protocol"], source_by_id, ddev_states,
                                              future_ids_by_protocol[status["protocol"]])
                scores = scores_for(status["method"], arrays["theta"],
                                    [item["image_id"] for item in items], cache)
                delta = require_score_match(scores, arrays["future_scores"], tol=SCORE_TOL)
                max_score_delta = max(max_score_delta, delta)
                score_rebuild_rows.append({"trajectory_id": tid, "side": "learner",
                    "checkpoint": checkpoint, "n_future": len(items), "max_abs_score_delta": delta,
                    "status": "PASS_FAILED_TRACK_CHECKPOINT"})

        # MIL spatial diagnostics use all 16 complete MIL learner paths at cp0/cp12.
        mil_statuses = [row for row in complete_learners
                        if row["backbone"] == backend and row["method"].startswith("M-")]
        if len(mil_statuses) != 8:
            raise IntegrityError(f"expected 8 complete MIL trajectories for {backend}")
        for status in mil_statuses:
            tid = status["trajectory_id"]
            for checkpoint in (0, 12):
                rec, arrays = _bundle_for(tid, "learner", checkpoint)
                ids = [item["image_id"] for item in rec["item_predictions"]]
                items = validate_future_items(rec, status["protocol"], source_by_id, ddev_states,
                                              future_ids_by_protocol[status["protocol"]])
                if checkpoint == 0:
                    expected_ids = ids
                elif ids != expected_ids:
                    raise IntegrityError(f"MIL cp0/cp12 future order changed: {tid}")
                future_full_scores: np.ndarray | None = None
                future_mean_scores: np.ndarray | None = None
                for split, split_items in (("D-fit", dfit_items), ("future", items)):
                    split_ids = [item["image_id"] for item in split_items]
                    bags = [cache[i]["patches"] for i in split_ids]
                    split_full_scores: list[float] = []
                    split_mean_scores: list[float] = []
                    for item, bag in zip(split_items, bags):
                        result = mil_spatial_components(bag, arrays["theta"], TAU)
                        split_full_scores.append(float(result["score_full"]))
                        split_mean_scores.append(float(result["score_mean"]))
                        max_spatial_identity_error = max(max_spatial_identity_error,
                                                         result["reconstruction_error"])
                        min_delta_spatial = min(min_delta_spatial, result["delta_spatial"])
                        spatial_item_rows.append({
                            "trajectory_id": tid, "backbone": backend, "method": status["method"],
                            "seed": status["seed"], "protocol": status["protocol"],
                            "checkpoint": checkpoint, "split": split, **item,
                            "n_valid_patches": result["n_valid_patches"],
                            "score_full": result["score_full"], "score_mean": result["score_mean"],
                            "delta_spatial": result["delta_spatial"],
                            "spatial_reconstruction_error": result["reconstruction_error"],
                            "patch_entropy": result["patch_entropy"],
                            "effective_patch_count": result["effective_patch_count"],
                            "effective_patch_fraction": result["effective_patch_fraction"],
                            "top1_mass": result["top1_mass"], "top5_mass": result["top5_mass"],
                            "top10_mass": result["top10_mass"],
                        })
                    if split == "future":
                        future_full_scores = np.asarray(split_full_scores, dtype=np.float64)
                        future_mean_scores = np.asarray(split_mean_scores, dtype=np.float64)
                        delta_score = require_score_match(future_full_scores, arrays["future_scores"], tol=SCORE_TOL)
                        max_score_delta = max(max_score_delta, delta_score)
                # Same-trained-parameter mean pooling counterfactual on future bags only.
                if future_full_scores is None or future_mean_scores is None:
                    raise IntegrityError(f"future spatial scorer did not run: {tid}/cp{checkpoint}")
                full_scores, mean_scores = future_full_scores, future_mean_scores
                full_metrics, full_products = metric_summary(items, full_scores)
                mean_metrics, mean_products = metric_summary(items, mean_scores)
                rank = ranking_change_counts(items, full_scores, mean_scores)
                changes = prediction_change_counts(items, full_scores, mean_scores)
                mil_counterfactual_rows.append({
                    "trajectory_id": tid, "backbone": backend, "method": status["method"],
                    "seed": status["seed"], "protocol": status["protocol"], "checkpoint": checkpoint,
                    "n_views": full_metrics["n_views"], "n_binary": full_metrics["n_binary"],
                    "ba_products": full_metrics["ba_product_count"],
                    "ap_products": full_metrics["ap_paired_product_count"],
                    "auc_products": full_metrics["roc_auc_product_count"],
                    "ba_full": full_metrics["ba_product_macro"], "ba_mean": mean_metrics["ba_product_macro"],
                    "ba_full_minus_mean": full_metrics["ba_product_macro"] - mean_metrics["ba_product_macro"],
                    "ap_full": full_metrics["ap_product_macro_paired"],
                    "ap_mean": mean_metrics["ap_product_macro_paired"],
                    "ap_full_minus_mean": full_metrics["ap_product_macro_paired"] - mean_metrics["ap_product_macro_paired"],
                    "auc_full": full_metrics["roc_auc_product_macro"],
                    "auc_mean": mean_metrics["roc_auc_product_macro"],
                    "auc_full_minus_mean": full_metrics["roc_auc_product_macro"] - mean_metrics["roc_auc_product_macro"],
                    "positive_rate_full": full_metrics["positive_prediction_rate_binary"],
                    "positive_rate_mean": mean_metrics["positive_prediction_rate_binary"],
                    **rank, **changes,
                })
                for full_row, mean_row in zip(full_products, mean_products):
                    mil_counterfactual_product_rows.append({
                        "trajectory_id": tid, "backbone": backend, "method": status["method"],
                        "seed": status["seed"], "protocol": status["protocol"],
                        "checkpoint": checkpoint, "product": full_row["product"],
                        "full_ba": full_row["ba"], "mean_ba": mean_row["ba"],
                        "full_ap": full_row["ap"], "mean_ap": mean_row["ap"],
                        "full_auc": full_row["roc_auc"], "mean_auc": mean_row["roc_auc"],
                        "present": full_row["present"], "absent": full_row["absent"],
                        "uncertain": full_row["uncertain"], "not_applicable": full_row["not_applicable"],
                    })

        del cache

    if b0_state_checks != 28 * 13:
        raise IntegrityError(f"expected 364 frozen B0 state checks, got {b0_state_checks}")
    if len(attribution_rows) + len(blocked_tracks) != 27:
        raise IntegrityError("checkpoint endpoint analysis did not account for all 27 complete learner tracks")
    if len(attribution_rows) != 27:
        # A score mismatch is an explicit per-track block, not a relaxed tolerance.
        overall_status = "D2_PARTIAL"
    else:
        overall_status = "D2_PASS"
    if len(spatial_item_rows) != 2432 and not blocked_tracks:
        raise IntegrityError(f"unexpected spatial bag count: {len(spatial_item_rows)}")

    # Aggregate two-factor attribution by protocol, model identity and method.
    attr_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attribution_rows:
        attr_groups[(row["protocol"], row["backbone"], row["method"], str(row["seed"]))].append(row)
    attribution_group_rows = []
    for (protocol, backbone, method, seed), rows in sorted(attr_groups.items()):
        values = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
                  for key in ("ba_delta", "ba_phi_weight", "ba_phi_bias", "ap_paired_delta",
                              "ap_paired_phi_weight", "ap_paired_phi_bias", "roc_auc_delta",
                              "roc_auc_phi_weight", "roc_auc_phi_bias")}
        weight_dom = bias_dom = opposed = ties = exact_near_zero = 0
        for row in rows:
            w, b, total = row["ba_phi_weight"], row["ba_phi_bias"], row["ba_delta"]
            if abs(total) <= 1e-12:
                exact_near_zero += 1
            if abs(w) > abs(b) + 1e-12:
                weight_dom += 1
            elif abs(b) > abs(w) + 1e-12:
                bias_dom += 1
            else:
                ties += 1
            if w * b < -1e-24:
                opposed += 1
        attribution_group_rows.append({
            "protocol": protocol, "backbone": backbone, "method": method, "seed": seed,
            "n_trajectories": len(rows),
            **{f"mean_{key}": float(np.mean(value)) for key, value in values.items()},
            "weight_magnitude_dominant_count": weight_dom,
            "bias_magnitude_dominant_count": bias_dom,
            "opposing_contribution_count": opposed,
            "magnitude_tie_count": ties,
            "exact_near_zero_total_ba_count": exact_near_zero,
            "corrected_binary_flips": sum(int(row["corrected_binary_flips"]) for row in rows),
            "harmful_binary_flips": sum(int(row["harmful_binary_flips"]) for row in rows),
        })

    attribution_protocol_rows = []
    for protocol in ("W", "X"):
        rows = [row for row in attribution_rows if row["protocol"] == protocol]
        if not rows:
            continue
        ba_w = np.asarray([row["ba_phi_weight"] for row in rows], dtype=np.float64)
        ba_b = np.asarray([row["ba_phi_bias"] for row in rows], dtype=np.float64)
        ba_d = np.asarray([row["ba_delta"] for row in rows], dtype=np.float64)
        attribution_protocol_rows.append({
            "protocol": protocol, "n_trajectories": len(rows),
            "mean_ba_total_gain": float(np.mean(ba_d)),
            "mean_ba_weight_shapley": float(np.mean(ba_w)),
            "mean_ba_bias_shapley": float(np.mean(ba_b)),
            "weight_magnitude_dominant_count": sum(abs(row["ba_phi_weight"]) > abs(row["ba_phi_bias"]) + 1e-12 for row in rows),
            "bias_magnitude_dominant_count": sum(abs(row["ba_phi_bias"]) > abs(row["ba_phi_weight"]) + 1e-12 for row in rows),
            "opposing_contribution_count": sum(row["ba_phi_weight"] * row["ba_phi_bias"] < -1e-24 for row in rows),
            "magnitude_tie_count": sum(abs(abs(row["ba_phi_weight"]) - abs(row["ba_phi_bias"])) <= 1e-12 for row in rows),
            "near_zero_total_gain_count_abs_le_1e_12": sum(abs(row["ba_delta"]) <= 1e-12 for row in rows),
            "corrected_binary_flips": sum(int(row["corrected_binary_flips"]) for row in rows),
            "harmful_binary_flips": sum(int(row["harmful_binary_flips"]) for row in rows),
        })

    # Aggregate spatial descriptors; product/state-specific records remain private.
    spatial_group_fields = ("split", "backbone", "method", "protocol", "checkpoint", "state")
    spatial_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in spatial_item_rows:
        spatial_groups[tuple(row[key] for key in spatial_group_fields)].append(row)
    spatial_summary_rows = []
    for keys, rows in sorted(spatial_groups.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        delta_vals = [float(row["delta_spatial"]) for row in rows]
        spatial_summary_rows.append({
            **dict(zip(spatial_group_fields, keys)),
            **{f"delta_{key}": value for key, value in compact_stats(delta_vals).items()},
            "mean_n_valid_patches": float(np.mean([row["n_valid_patches"] for row in rows])),
            "mean_effective_patch_count": float(np.mean([row["effective_patch_count"] for row in rows])),
            "mean_effective_patch_fraction": float(np.mean([row["effective_patch_fraction"] for row in rows])),
            "mean_patch_entropy": float(np.mean([row["patch_entropy"] for row in rows])),
            "mean_top1_mass": float(np.mean([row["top1_mass"] for row in rows])),
            "mean_top5_mass": float(np.mean([row["top5_mass"] for row in rows])),
            "mean_top10_mass": float(np.mean([row["top10_mass"] for row in rows])),
            "corr_delta_vs_effective_fraction": corr_or_none(delta_vals,
                [float(row["effective_patch_fraction"]) for row in rows]),
            "corr_delta_vs_top1_mass": corr_or_none(delta_vals,
                [float(row["top1_mass"]) for row in rows]),
        })
    spatial_product_rows = []
    product_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in spatial_item_rows:
        key = (row["split"], row["backbone"], row["method"], row["protocol"],
               row["checkpoint"], row["product"], row["state"])
        product_groups[key].append(row)
    for keys, rows in sorted(product_groups.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        spatial_product_rows.append({
            **dict(zip(("split", "backbone", "method", "protocol", "checkpoint", "product", "state"), keys)),
            **{f"delta_{key}": value for key, value in compact_stats(
                [float(row["delta_spatial"]) for row in rows]).items()},
        })

    # Delta-spatial changes are joined only within the same frozen image and trajectory.
    by_state_item = {(row["trajectory_id"], row["checkpoint"], row["split"], row["image_id"]): row
                     for row in spatial_item_rows}
    for row in spatial_item_rows:
        if row["checkpoint"] != 0:
            continue
        peer = by_state_item.get((row["trajectory_id"], 12, row["split"], row["image_id"]))
        if peer is not None:
            spatial_pair_change_rows.append({
                "trajectory_id": row["trajectory_id"], "backbone": row["backbone"],
                "method": row["method"], "protocol": row["protocol"], "split": row["split"],
                "image_id": row["image_id"], "product": row["product"], "state": row["state"],
                "delta_cp0": row["delta_spatial"], "delta_cp12": peer["delta_spatial"],
                "delta_change_cp12_minus_cp0": peer["delta_spatial"] - row["delta_spatial"],
            })

    counterfactual_summary_rows = []
    cf_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mil_counterfactual_rows:
        cf_groups[(row["protocol"], row["backbone"], row["method"])].append(row)
    for (protocol, backbone, method), rows in sorted(cf_groups.items()):
        counterfactual_summary_rows.append({
            "protocol": protocol, "backbone": backbone, "method": method,
            "state_count": len(rows),
            **{f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in (
                "ba_full_minus_mean", "ap_full_minus_mean", "auc_full_minus_mean",
                "positive_rate_full", "positive_rate_mean")},
            "order_discordances": sum(row["order_discordances"] for row in rows),
            "positive_negative_pairs": sum(row["positive_negative_pairs"] for row in rows),
            "threshold_flips": sum(row["threshold_flips_binary"] for row in rows),
            "corrected_flips": sum(row["corrected_binary_flips"] for row in rows),
            "harmful_flips": sum(row["harmful_binary_flips"] for row in rows),
        })

    # Private run summary stores only aggregate output plus identities; detailed IDs and states stay in CSV/NPZ.
    summary = {
        "schema": "lp.d6.p2.e2b.r1.attribution_spatial.summary.v1",
        "status": overall_status,
        "identity": identity,
        "scope": {
            "learner_endpoint_tracks_expected": 27,
            "learner_endpoint_tracks_analyzed": len(attribution_rows),
            "learner_endpoint_tracks_blocked": len(blocked_tracks),
            "failed_trajectory_checkpoints_reconstructed": 10,
            "b0_controls": 28,
            "b0_checkpoint_states_verified_unchanged": b0_state_checks,
            "mil_learner_trajectories": 16,
            "mil_spatial_bag_rows": len(spatial_item_rows),
            "mil_counterfactual_state_rows": len(mil_counterfactual_rows),
            "max_abs_reconstructed_model_score_delta": max_score_delta,
            "frozen_score_tolerance": SCORE_TOL,
            "max_spatial_identity_error": max_spatial_identity_error,
            "spatial_identity_tolerance": SPATIAL_IDENTITY_TOL,
            "minimum_delta_spatial": min_delta_spatial,
            "jensen_tolerance": JENSEN_TOL,
        },
        "attribution_by_protocol": attribution_protocol_rows,
        "mil_counterfactual_by_protocol_backbone_method": counterfactual_summary_rows,
        "blocked_tracks": blocked_tracks,
    }
    outputs = {
        "attribution_per_trajectory.csv": attribution_rows,
        "attribution_by_protocol_backbone_method_seed.csv": attribution_group_rows,
        "attribution_by_protocol.csv": attribution_protocol_rows,
        "attribution_product_metrics.csv": attribution_product_rows,
        "attribution_item_scores.csv": attribution_item_rows,
        "score_reconstruction_checks.csv": score_rebuild_rows,
        "blocked_tracks.csv": blocked_tracks,
        "mil_spatial_per_bag.csv": spatial_item_rows,
        "mil_spatial_summary.csv": spatial_summary_rows,
        "mil_spatial_by_product.csv": spatial_product_rows,
        "mil_spatial_cp0_cp12_item_changes.csv": spatial_pair_change_rows,
        "mil_mean_pooling_counterfactual.csv": mil_counterfactual_rows,
        "mil_mean_pooling_by_product.csv": mil_counterfactual_product_rows,
        "mil_mean_pooling_summary.csv": counterfactual_summary_rows,
    }
    # Create the private directory only after all identity and numerical checks pass.
    OUT.mkdir(parents=True, exist_ok=False)
    for filename, rows in outputs.items():
        write_csv(OUT / filename, rows)
    np.savez_compressed(OUT / "mil_model_endpoints.npz", **{
        key: value for key, value in private_parameters.items()
        if "__theta" in key and "M-" in key
    })
    summary["diagnostic_code_sha256"] = sha256_file(Path(__file__))
    (OUT / "summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
    (OUT / "input_identity.json").write_bytes(canonical_json_bytes(identity) + b"\n")
    return summary, outputs, private_parameters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true",
                        help="verify archive/cache identities without creating output or scoring features")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(), ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    summary, _, _ = run_diagnostics()
    print(json.dumps({
        "status": summary["status"],
        "output": str(OUT),
        "complete_attribution_tracks": summary["scope"]["learner_endpoint_tracks_analyzed"],
        "blocked_tracks": summary["scope"]["learner_endpoint_tracks_blocked"],
        "b0_states_verified": summary["scope"]["b0_checkpoint_states_verified_unchanged"],
        "mil_spatial_bag_rows": summary["scope"]["mil_spatial_bag_rows"],
        "counterfactual_rows": summary["scope"]["mil_counterfactual_state_rows"],
        "max_score_delta": summary["scope"]["max_abs_reconstructed_model_score_delta"],
        "max_spatial_identity_error": summary["scope"]["max_spatial_identity_error"],
    }, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if summary["status"] == "D2_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
