"""Run the bounded LP-D6-P2-E1 full-history proximal headroom experiment."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Must precede NumPy/SciPy/scikit-learn imports. No Torch/DINO/GPU code is used.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import psutil
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[4]
P1B_DIR = ROOT / "scripts/research_d6/p1b"
if str(P1B_DIR) not in sys.path:
    sys.path.insert(0, str(P1B_DIR))

import core as p1b_core  # noqa: E402
import run_experiment as p1b_runner  # noqa: E402

from proximal_logistic import (  # noqa: E402
    GAMMAS,
    GRAD_INF_TOL,
    LAMBDA,
    ProximalLogistic,
    SolverFailure,
    hessian_vector_product,
    objective_and_gradient,
)


P1A_DIR = ROOT / "outputs/research_d6/p1a_features"
P1_RUN_DIR = ROOT / "outputs/research_d6/p1b/run_20261113_01"
P1_R1_DIR = ROOT / "outputs/research_d6/p1b_r1"
P1_HASH_MANIFEST = P1_R1_DIR / "original_run_artifact_hashes.json"
G0_DIR = ROOT / "docs/research_d6/p2/g0"
RUN_DIR = ROOT / "outputs/research_d6/p2/e1/run_20261115_01"
MAX_CPU_SECONDS = 1800.0
MAX_RSS_BYTES = 1024 * 1024 * 1024
MAX_OUTPUT_BYTES = 100 * 1024 * 1024
CHECKPOINTS = tuple(range(13))
VERIFY_CHECKPOINTS = (0, 3, 6, 12)
NEWTON_MAX_RUNS = 16
NEWTON_OPTIONS = {"xtol": 1e-12, "maxiter": 10000}
NEWTON_OBJECTIVE_ATOL = 1e-8
NEWTON_MAX_FUTURE_SCORE_ABS = 1e-5
INITIAL_SCORE_ATOL = 1e-6
INITIAL_SCORE_RTOL = 1e-6


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
        writer.writerows(rows)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0, "sha256": ""}
    if path.is_file():
        files = [path]
        base = path.parent
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file() and
                       "__pycache__" not in p.parts and p.suffix != ".pyc")
        base = path
    digest = hashlib.sha256()
    total = 0
    for file in files:
        relative = file.relative_to(base).as_posix().encode("utf-8")
        file_sha = sha_file(file)
        size = file.stat().st_size
        digest.update(relative + b"\0" + file_sha.encode("ascii") + b"\n")
        total += size
    return {"exists": True, "files": len(files), "bytes": total, "sha256": digest.hexdigest()}


PROTECTED_TREES = {
    "p1a_cache": P1A_DIR,
    "p1b_original_run": P1_RUN_DIR,
    "p1b_r1_audit": P1_R1_DIR,
    "p1a_code": ROOT / "scripts/research_d6/p1a",
    "p1b_code": P1B_DIR,
    "p1a_docs": ROOT / "docs/research_d6/p1a",
    "p1b_docs": ROOT / "docs/research_d6/p1b",
    "p1b_r1_docs": ROOT / "docs/research_d6/p1b_r1",
    "p2_g0_docs": G0_DIR,
    "p1a_test": ROOT / "tests/research_d6/test_p1a_manifest.py",
    "p1b_baseline_test": ROOT / "tests/research_d6/test_p1b_baselines.py",
    "p1b_r1_test": ROOT / "tests/research_d6/test_p1b_r1_audit.py",
}


def protected_snapshot() -> dict[str, dict[str, Any]]:
    return {name: tree_fingerprint(path) for name, path in PROTECTED_TREES.items()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_feedback_future_roles(feedback: Iterable[Any], future: Iterable[Any]) -> None:
    feedback = list(feedback)
    future = list(future)
    query_ids = [item.image_id for item in feedback]
    future_ids = [item.image_id for item in future]
    require(len(query_ids) == len(set(query_ids)), "feedback image repeated in one protocol")
    require(len(future_ids) == len(set(future_ids)), "future image repeated in one protocol")
    require(set(query_ids).isdisjoint(future_ids), "feedback and future image crossover")
    query_groups = {(item.product, item.sample_key) for item in feedback}
    future_groups = {(item.product, item.sample_key) for item in future}
    require(query_groups.isdisjoint(future_groups), "feedback and future sample-key group crossover")


def validate_protocol_support(feedback: list[Any], future: list[Any], protocol: str) -> dict[str, Any]:
    validate_feedback_future_roles(feedback, future)
    require(len(feedback) == 12 and len(future) == 24, f"{protocol} feedback/future counts changed")
    require([item.stream_order for item in feedback] == list(range(1, 13)), f"{protocol} stream order changed")
    fcount = Counter(item.state for item in feedback)
    expected = ({"present": 7, "absent_assessable": 5} if protocol == "X"
                else {"present": 5, "absent_assessable": 6, "unobservable_or_uncertain": 1})
    require(dict(fcount) == expected, f"{protocol} feedback states differ from registered first labels")
    binary_future = sum(p1b_core.binary_target(item.state) is not None for item in future)
    require(binary_future == 21, f"{protocol} binary future support is {binary_future}, expected 21")
    if protocol == "X":
        require({i.product for i in feedback} == {"porcelain_doll", "u_block", "usb_adaptor"},
                "X feedback product roles changed")
        require({i.product for i in future} == {"sim_card_set", "toothbrush", "transistor1"},
                "X future product roles changed")
    return {
        "feedback_events": len(feedback),
        "feedback_binary": sum(p1b_core.binary_target(item.state) is not None for item in feedback),
        "feedback_state_counts": dict(fcount),
        "future_views": len(future),
        "future_binary": binary_future,
        "future_groups": len({(i.product, i.sample_key) for i in future}),
        "feedback_groups": len({(i.product, i.sample_key) for i in feedback}),
        "role_group_overlap": 0,
    }


def validate_original_p1b_hashes() -> dict[str, str]:
    expected = json.loads(P1_HASH_MANIFEST.read_text(encoding="utf-8"))["files"]
    observed: dict[str, str] = {}
    for name, value in expected.items():
        path = P1_RUN_DIR / name
        require(path.is_file(), f"original P1-B artifact missing: {name}")
        observed[name] = sha_file(path)
        require(observed[name] == value, f"original P1-B artifact hash mismatch: {name}")
    require(len(expected) == 15, f"expected 15 original P1-B files, found {len(expected)}")
    require(set(p.name for p in P1_RUN_DIR.iterdir() if p.is_file()) == set(expected),
            "original P1-B run file inventory changed")
    return observed


def load_registered_anchor(x_fit: np.ndarray, y_fit: np.ndarray) -> dict[str, Any]:
    state_path = P1_RUN_DIR / "model_states.npz"
    index_rows = read_csv(P1_RUN_DIR / "model_state_index.csv")
    metrics = read_csv(P1_RUN_DIR / "checkpoint_metrics.csv")
    idx_by_key = {(r["protocol"], r["method"], r["pair_id"]): r for r in index_rows}
    with np.load(state_path, allow_pickle=False) as archive:
        prefix = "X__refit_alpha_1e-04__initial__"
        b0_prefix = "X__refit_alpha_1e-04__b0_snapshot__"
        initial_w_matrix = archive[prefix + "coef"].copy()
        initial_b_array = archive[prefix + "intercept"].copy()
        base_x = archive[prefix + "base_X"].copy()
        base_y = archive[prefix + "base_y"].copy()
        b0_w = archive[b0_prefix + "coef"].copy()
        b0_b = archive[b0_prefix + "intercept"].copy()
        b0_threshold = archive[b0_prefix + "threshold"].copy()
        final_states: dict[str, dict[str, np.ndarray]] = {}
        for protocol in ("X", "W"):
            p = f"{protocol}__refit_alpha_1e-04__final__"
            final_states[protocol] = {
                name: archive[p + name].copy()
                for name in ("coef", "intercept", "paid_X", "paid_y", "paid_event_keys_utf8_hex")
            }
    require(initial_w_matrix.shape == (1, 1024) and initial_b_array.shape == (1,),
            "archived P1-B alpha=1e-4 initialization has an unexpected shape")
    require(np.array_equal(initial_w_matrix.reshape(-1), b0_w), "archived B0 and Refit coefficient arrays differ")
    require(np.array_equal(initial_b_array, b0_b), "archived B0 and Refit intercept arrays differ")
    require(np.array_equal(b0_threshold, np.asarray([0.0])), "archived B0 threshold is not zero logit")
    require(base_x.dtype == np.float32 and np.array_equal(base_x, np.asarray(x_fit, dtype=np.float32)),
            "archived Logistic D-fit feature history differs from authorized 52 binary records")
    require(base_y.dtype == np.int8 and np.array_equal(base_y.astype(np.int64), np.asarray(y_fit, dtype=np.int64)),
            "archived Logistic D-fit label history differs from authorized first labels")
    w0 = initial_w_matrix.reshape(-1).astype(np.float64)
    b0 = float(initial_b_array[0])
    anchor_value, anchor_grad = objective_and_gradient(
        np.concatenate([w0, [b0]]), x_fit, y_fit, w0, lam=LAMBDA, gamma=0.0
    )
    require(np.isfinite(anchor_value) and np.isfinite(anchor_grad).all(), "registered B0 objective/gradient is nonfinite")

    # Independently reproduce P1-B's learner state digest convention.
    digest_fn = p1b_core.array_digest
    init_digest = digest_fn(initial_w_matrix.astype(np.float64), initial_b_array.astype(np.float64))
    for protocol in ("X", "W"):
        idx = idx_by_key[(protocol, "B1-Refit", "refit_alpha_1e-04")]
        require(idx["initial_state_sha256"] == init_digest,
                f"{protocol} P1-B full-history Refit initial state digest mismatch")
        state = final_states[protocol]
        extra = b"\0".join(bytes.fromhex(v) for v in state["paid_event_keys_utf8_hex"].tolist())
        final_digest = digest_fn(
            state["coef"].astype(np.float64),
            state["intercept"].astype(np.float64),
            state["paid_X"].astype(np.float32),
            state["paid_y"].astype(np.int8),
            extra=extra,
        )
        require(idx["final_state_sha256"] == final_digest,
                f"{protocol} P1-B full-history Refit final state digest mismatch")
        matching_metric = [r for r in metrics if r["protocol"] == protocol and
                           r["method"] == "B1-Refit" and r["config_id"] == "refit_alpha_1e-04" and
                           int(r["checkpoint"]) == 12]
        require(len(matching_metric) == 1 and matching_metric[0]["state_sha256"] == final_digest,
                f"{protocol} archived final model state does not match checkpoint 12")
    return {
        "w0": w0,
        "b0": b0,
        "b0_threshold": float(b0_threshold[0]),
        "base_x": base_x,
        "base_y": base_y,
        "anchor_objective": anchor_value,
        "anchor_gradient_inf": float(np.max(np.abs(anchor_grad))),
        "anchor_gradient_vector": anchor_grad.copy(),
        "initial_learner_state_sha256": init_digest,
        "final_states": final_states,
        "state_index": idx_by_key,
    }


def _baseline_rows(protocol: str, name: str, config: str) -> list[dict[str, str]]:
    rows = read_csv(P1_RUN_DIR / name)
    return [r for r in rows if r.get("protocol") == protocol and
            r.get("method") == "B1-Refit" and r.get("config_id") == config]


def _binary_truth_by_id(future: list[Any]) -> dict[str, int]:
    result = {}
    for item in future:
        target = p1b_core.binary_target(item.state)
        if target is not None:
            result[item.image_id] = int(target)
    return result


def _flip_counts(
    previous: dict[str, int] | None,
    current: dict[str, int],
    truth: dict[str, int],
) -> tuple[int, int]:
    if previous is None:
        return 0, 0
    harmful = corrected = 0
    for image_id, y in truth.items():
        was_correct = previous[image_id] == y
        is_correct = current[image_id] == y
        harmful += int(was_correct and not is_correct)
        corrected += int((not was_correct) and is_correct)
    return harmful, corrected


def evaluate_scores(scores: np.ndarray, future: list[Any], threshold: float = 0.0) -> dict[str, Any]:
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    metrics, products, groups = p1b_core.summarize_future(score_values, future, threshold)
    truths = np.asarray([p1b_core.binary_target(i.state) for i in future], dtype=object)
    eligible = np.asarray([v is not None for v in truths], dtype=bool)
    y_binary = np.asarray([int(v) for v in truths[eligible]], dtype=np.int8)
    s_binary = score_values[eligible]
    predictions = (score_values >= threshold).astype(np.int8)
    aucs = []
    for product in sorted({i.product for i in future}):
        indices = [n for n, item in enumerate(future)
                   if item.product == product and p1b_core.binary_target(item.state) is not None]
        y = np.asarray([p1b_core.binary_target(future[n].state) for n in indices], dtype=np.int8)
        if np.unique(y).size == 2:
            aucs.append(float(roc_auc_score(y, score_values[indices])))
    predictions_by_id = {
        item.image_id: int(predictions[n])
        for n, item in enumerate(future)
        if p1b_core.binary_target(item.state) is not None
    }
    return {
        "metrics": metrics,
        "products": products,
        "groups": groups,
        "predictions": predictions_by_id,
        "truth": {item.image_id: int(p1b_core.binary_target(item.state))
                  for item in future if p1b_core.binary_target(item.state) is not None},
        "positive_prediction_fraction": float(np.mean(predictions[eligible] == 1)),
        "roc_auc_product_macro": float(np.mean(aucs)) if aucs else None,
        "roc_auc_product_count": len(aucs),
        "roc_auc_product_values": {
            product: float(roc_auc_score(
                [p1b_core.binary_target(i.state) for i in future
                 if i.product == product and p1b_core.binary_target(i.state) is not None],
                [score_values[n] for n, i in enumerate(future)
                 if i.product == product and p1b_core.binary_target(i.state) is not None],
            ))
            for product in sorted({i.product for i in future})
            if len({p1b_core.binary_target(i.state) for i in future
                    if i.product == product and p1b_core.binary_target(i.state) is not None}) == 2
        },
        "binary_truth_n": int(y_binary.size),
        "binary_positive_n": int(np.sum(y_binary == 1)),
        "binary_negative_n": int(np.sum(y_binary == 0)),
    }


def weight_diagnostics(w: np.ndarray, b: float, w0: np.ndarray, b0: float) -> dict[str, float]:
    w = np.asarray(w, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    d = w - w0
    norm = float(np.linalg.norm(w))
    anchor_norm = float(np.linalg.norm(w0))
    parallel = float(np.dot(d, w0) / anchor_norm) if anchor_norm else 0.0
    perpendicular = float(np.linalg.norm(d - (np.dot(d, w0) / (anchor_norm ** 2) * w0
                                                   if anchor_norm else np.zeros_like(d))))
    denominator = norm * anchor_norm
    return {
        "weight_cosine_to_w0": float(np.dot(w, w0) / denominator) if denominator else None,
        "weight_norm": norm,
        "delta_norm": float(np.linalg.norm(d)),
        "intercept_shift": float(b - b0),
        "weight_norm_ratio_to_w0": float(norm / anchor_norm) if anchor_norm else None,
        "delta_parallel_to_w0_norm": abs(parallel),
        "delta_perpendicular_to_w0_norm": perpendicular,
    }


def _append_evaluation_rows(
    *,
    protocol: str,
    gamma: float,
    source: str,
    checkpoint: int,
    scores: np.ndarray,
    future: list[Any],
    w: np.ndarray | None,
    b: float | None,
    w0: np.ndarray,
    b0: float,
    objective: float | None,
    grad_inf: float | None,
    update_latency_ms: float | None,
    fit_iterations: int | None,
    function_evaluations: int | None,
    optimization_count: int,
    state_digest: str,
    state_payload_bytes: int | None,
    previous_predictions: dict[str, int] | None,
    initial_predictions: dict[str, int] | None,
    previous_harmful_total: int,
    previous_corrected_total: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int], int, int]:
    evaluated = evaluate_scores(scores, future)
    harmful, corrected = _flip_counts(previous_predictions, evaluated["predictions"], evaluated["truth"])
    initial_harmful, initial_corrected = _flip_counts(initial_predictions, evaluated["predictions"], evaluated["truth"])
    model_diag = (weight_diagnostics(w, float(b), w0, b0) if w is not None and b is not None else {})
    metric = {
        "protocol": protocol,
        "gamma": gamma,
        "lambda": LAMBDA,
        "trajectory_source": source,
        "checkpoint": checkpoint,
        "feedback_event_units_consumed": checkpoint,
        "binary_feedback_count": optimization_count,
        "future_views": evaluated["metrics"]["future_views"],
        "future_binary_views": evaluated["metrics"]["future_binary_views"],
        "future_positive": evaluated["binary_positive_n"],
        "future_negative": evaluated["binary_negative_n"],
        "ba_product_macro": evaluated["metrics"]["ba_product_macro"],
        "ba_eligible_product_count": evaluated["metrics"]["ba_eligible_product_count"],
        "ap_product_macro_paired_class": evaluated["metrics"]["ap_product_macro_paired_class"],
        "ap_paired_class_product_count": evaluated["metrics"]["ap_paired_class_product_count"],
        "ap_product_macro_including_degenerate": evaluated["metrics"]["ap_product_macro_including_degenerate"],
        "roc_auc_product_macro": evaluated["roc_auc_product_macro"],
        "roc_auc_product_count": evaluated["roc_auc_product_count"],
        "positive_prediction_fraction": evaluated["positive_prediction_fraction"],
        "harmful_flips_from_previous_checkpoint": harmful,
        "corrected_flips_from_previous_checkpoint": corrected,
        "cumulative_harmful_transitions": previous_harmful_total + harmful,
        "cumulative_corrected_transitions": previous_corrected_total + corrected,
        "net_harmful_views_vs_checkpoint0": initial_harmful,
        "net_corrected_views_vs_checkpoint0": initial_corrected,
        "objective": objective,
        "gradient_inf_norm": grad_inf,
        "l2_regularization_term": (0.5 * LAMBDA * float(np.dot(w, w)) if w is not None else None),
        "proximal_regularization_term": (0.5 * gamma * float(np.dot(w - w0, w - w0))
                                         if w is not None else None),
        "update_latency_ms": update_latency_ms,
        "fit_iterations": fit_iterations,
        "function_evaluations": function_evaluations,
        "optimization_count": optimization_count,
        "state_sha256": state_digest,
        "state_payload_bytes": state_payload_bytes,
        **model_diag,
    }
    product_rows = []
    auc_by_product = evaluated["roc_auc_product_values"]
    for row in evaluated["products"]:
        product_rows.append({
            "protocol": protocol,
            "gamma": gamma,
            "trajectory_source": source,
            "checkpoint": checkpoint,
            **row,
            "roc_auc": auc_by_product.get(row["product"]),
        })
    return metric, product_rows, evaluated["predictions"], previous_harmful_total + harmful, previous_corrected_total + corrected


def _score_rows(protocol: str, method: str, config_id: str) -> list[dict[str, str]]:
    rows = read_csv(P1_RUN_DIR / "future_and_query_predictions.csv")
    return [r for r in rows if r["protocol"] == protocol and r["method"] == method and
            r["config_id"] == config_id and r["role"] == "future_eval"]


def verify_registered_initial_scores(
    features: dict[str, np.ndarray],
    protocols: dict[str, tuple[list[Any], list[Any]]],
    w0: np.ndarray,
    b0: float,
) -> dict[str, float]:
    max_abs_by_protocol: dict[str, float] = {}
    for protocol, (_, future) in protocols.items():
        predicted = {item.image_id: float(item.x.astype(np.float64) @ w0 + b0) for item in future}
        for baseline_method in ("B0-Logistic-Refit", "B1-Refit"):
            original = _score_rows(protocol, baseline_method, "refit_alpha_1e-04")
            original = [r for r in original if int(r["checkpoint"]) == 0]
            by_id = {r["image_id"]: float(r["score"]) for r in original}
            require(set(by_id) == set(predicted), f"{protocol} P1-B {baseline_method} checkpoint-0 future IDs differ")
            max_abs = max(abs(predicted[i] - by_id[i]) for i in predicted)
            require(np.allclose(
                np.asarray([predicted[i] for i in sorted(predicted)]),
                np.asarray([by_id[i] for i in sorted(predicted)]),
                atol=INITIAL_SCORE_ATOL,
                rtol=INITIAL_SCORE_RTOL,
            ), f"{protocol} checkpoint-0 scores disagree with preserved {baseline_method}")
            max_abs_by_protocol[f"{protocol}_{baseline_method}"] = max_abs
    return max_abs_by_protocol


def _record_future_predictions(
    destination: list[dict[str, Any]], protocol: str, gamma: float, source: str,
    checkpoint: int, scores: np.ndarray, future: list[Any],
) -> None:
    predictions = (np.asarray(scores, dtype=np.float64) >= 0.0).astype(np.int8)
    for item, score, prediction in zip(future, scores, predictions):
        target = p1b_core.binary_target(item.state)
        destination.append({
            "protocol": protocol,
            "gamma": gamma,
            "trajectory_source": source,
            "checkpoint": checkpoint,
            "image_id": item.image_id,
            "product": item.product,
            "sample_key": item.sample_key,
            "attribute_state_evaluator_only": item.state,
            "binary_target_evaluator_only": target,
            "score": float(score),
            "threshold": 0.0,
            "prediction": int(prediction),
        })


def replay_proximal(
    *,
    protocol: str,
    gamma: float,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    w0: np.ndarray,
    b0: float,
    feedback: list[Any],
    future: list[Any],
    cpu_start: float,
    max_cpu_seconds: float = MAX_CPU_SECONDS,
    reveal_state: Any | None = None,
) -> dict[str, Any]:
    validate_feedback_future_roles(feedback, future)
    learner = ProximalLogistic(x_fit, y_fit, w0, b0, gamma=gamma)
    rows: dict[str, list[dict[str, Any]]] = {"metrics": [], "products": [], "events": [], "predictions": []}
    states: dict[int, dict[str, np.ndarray]] = {}
    previous_predictions: dict[str, int] | None = None
    initial_predictions: dict[str, int] | None = None
    cumulative_harmful = cumulative_corrected = 0
    fit_count = 0
    previous_update_ms: float | None = None
    previous_iterations: int | None = None
    previous_nfev: int | None = None

    def checkpoint(cp: int) -> None:
        nonlocal previous_predictions, initial_predictions, cumulative_harmful, cumulative_corrected
        score_start = time.perf_counter()
        scores = learner.scores(np.stack([item.x for item in future]))
        inference_ms = (time.perf_counter() - score_start) * 1000.0
        objective, grad_inf = learner.objective_gradient()
        metric, product_rows, current_predictions, cumulative_harmful, cumulative_corrected = _append_evaluation_rows(
            protocol=protocol,
            gamma=gamma,
            source="new_proximal_lbfgsb",
            checkpoint=cp,
            scores=scores,
            future=future,
            w=learner.w,
            b=learner.b,
            w0=w0,
            b0=b0,
            objective=objective,
            grad_inf=grad_inf,
            update_latency_ms=previous_update_ms,
            fit_iterations=previous_iterations,
            function_evaluations=previous_nfev,
            optimization_count=fit_count,
            state_digest=learner.state_digest(),
            state_payload_bytes=learner.state_payload_bytes(),
            previous_predictions=previous_predictions,
            initial_predictions=initial_predictions,
            previous_harmful_total=cumulative_harmful,
            previous_corrected_total=cumulative_corrected,
        )
        # cumulative columns must add this transition exactly once.
        harmful = int(metric["harmful_flips_from_previous_checkpoint"])
        corrected = int(metric["corrected_flips_from_previous_checkpoint"])
        metric["cumulative_harmful_transitions"] = (rows["metrics"][-1]["cumulative_harmful_transitions"] if rows["metrics"] else 0) + harmful
        metric["cumulative_corrected_transitions"] = (rows["metrics"][-1]["cumulative_corrected_transitions"] if rows["metrics"] else 0) + corrected
        metric["future_inference_ms"] = inference_ms
        rows["metrics"].append(metric)
        rows["products"].extend(product_rows)
        _record_future_predictions(rows["predictions"], protocol, gamma, "new_proximal_lbfgsb", cp, scores, future)
        if cp == 0:
            initial_predictions = current_predictions.copy()
        previous_predictions = current_predictions.copy()
        states[cp] = learner.state_arrays()

    checkpoint(0)
    for event in sorted(feedback, key=lambda item: item.stream_order):
        if time.process_time() - cpu_start > max_cpu_seconds:
            raise TimeoutError("30-minute cumulative CPU experiment ceiling reached")
        state_before = learner.state_digest()
        prediction_start = time.perf_counter()
        pre_score = float(learner.scores(event.x.reshape(1, -1))[0])
        pre_prediction = int(pre_score >= 0.0)
        prediction_ms = (time.perf_counter() - prediction_start) * 1000.0
        # The first access to the revealed state is deliberately after prediction.
        revealed_state = reveal_state(event) if reveal_state is not None else event.state
        target = p1b_core.binary_target(revealed_state)
        update_ms = 0.0
        fit_result = None
        if target is not None:
            fit_result = learner.observe(event.x, target, event.event_id)
            update_ms = fit_result.elapsed_seconds * 1000.0
            fit_count += 1
            previous_update_ms = update_ms
            previous_iterations = fit_result.iterations
            previous_nfev = fit_result.function_evaluations
        state_after = learner.state_digest()
        if target is None:
            require(state_before == state_after, "Uncertain/N-A event changed learner state")
            previous_update_ms = 0.0
            previous_iterations = 0
            previous_nfev = 0
        rows["events"].append({
            "protocol": protocol,
            "gamma": gamma,
            "event_id": event.event_id,
            "stream_order": event.stream_order,
            "image_id": event.image_id,
            "product": event.product,
            "sample_key": event.sample_key,
            "prediction_timing": "before_reveal",
            "pre_reveal_score": pre_score,
            "pre_reveal_prediction": pre_prediction,
            "revealed_state_evaluator_only": revealed_state,
            "binary_target_if_revealed": target,
            "feedback_cost_unit_assumed": 1,
            "update_applied": target is not None,
            "update_latency_ms": update_ms,
            "pre_reveal_prediction_latency_ms": prediction_ms,
            "state_sha256_before": state_before,
            "state_sha256_after": state_after,
            "state_payload_bytes_after": learner.state_payload_bytes(),
            "solver_status": fit_result.status if fit_result else "NO_BINARY_UPDATE",
            "objective_after_update": fit_result.objective if fit_result else None,
            "gradient_inf_after_update": fit_result.gradient_inf_norm if fit_result else None,
            "solver_iterations": fit_result.iterations if fit_result else 0,
        })
        checkpoint(event.stream_order)
    expected_updates = sum(p1b_core.binary_target(event.state) is not None for event in feedback)
    require(fit_count == expected_updates, f"{protocol} proximal update count differs from binary feedback history")
    require(learner.optimization_count == fit_count, "fit counter disagrees with valid binary feedback count")
    return {"rows": rows, "states": states, "fit_count": fit_count, "learner": learner}


def _baseline_replay(
    *, protocol: str, feedback: list[Any], future: list[Any], features: dict[str, np.ndarray], w0: np.ndarray, b0: float,
    final_state: dict[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {"metrics": [], "products": [], "predictions": [], "events": []}
    baseline_predictions = _score_rows(protocol, "B1-Refit", "refit_alpha_1e-04")
    require(len(baseline_predictions) == 13 * 24, f"{protocol} preserved gamma=0 prediction log incomplete")
    checkpoint_metrics = [r for r in read_csv(P1_RUN_DIR / "checkpoint_metrics.csv")
                          if r["protocol"] == protocol and r["method"] == "B1-Refit" and
                          r["config_id"] == "refit_alpha_1e-04"]
    product_metrics = [r for r in read_csv(P1_RUN_DIR / "product_metrics.csv")
                       if r["protocol"] == protocol and r["method"] == "B1-Refit" and
                       r["config_id"] == "refit_alpha_1e-04"]
    events = [r for r in read_csv(P1_RUN_DIR / "feedback_event_log.csv")
              if r["protocol"] == protocol and r["method"] == "B1-Refit" and
              r["config_id"] == "refit_alpha_1e-04"]
    require(len(checkpoint_metrics) == 13 and len(product_metrics) == 13 * len({i.product for i in future}) and
            len(events) == 12, f"{protocol} preserved P1-B metric/event coverage mismatch")
    events_by_order = sorted(events, key=lambda r: int(r["stream_order"]))
    require([int(r["stream_order"]) for r in events_by_order] == list(range(1, 13)),
            f"{protocol} preserved P1-B feedback order is incomplete")
    for registered, preserved in zip(sorted(feedback, key=lambda e: e.stream_order), events_by_order):
        require(registered.event_id == preserved["event_id"] and
                registered.image_id == preserved["image_id"] and
                registered.state == preserved["revealed_state_evaluator_only"] and
                (p1b_core.binary_target(registered.state) is not None) == (preserved["update_applied"].lower() == "true"),
                f"{protocol} preserved P1-B gamma=0 event stream differs from the registered tape")
    by_cp_metrics = {int(r["checkpoint"]): r for r in checkpoint_metrics}
    by_cp_predictions: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in baseline_predictions:
        by_cp_predictions[int(row["checkpoint"])].append(row)
    by_cp_products: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in product_metrics:
        by_cp_products[int(row["checkpoint"])].append(row)
    initial_preds = None
    previous_preds = None
    cumulative_harmful = cumulative_corrected = 0
    truth = _binary_truth_by_id(future)
    final_w = final_state["coef"].reshape(-1).astype(np.float64)
    final_b = float(final_state["intercept"][0])
    for cp in CHECKPOINTS:
        pred_rows = sorted(by_cp_predictions[cp], key=lambda r: r["image_id"])
        scores_by_id = {r["image_id"]: float(r["score"]) for r in pred_rows}
        predictions_by_id = {r["image_id"]: int(r["prediction"]) for r in pred_rows
                             if r["binary_target"] != ""}
        future_order_ids = [item.image_id for item in future]
        scores = np.asarray([scores_by_id[i] for i in future_order_ids], dtype=np.float64)
        evaluated = evaluate_scores(scores, future)
        harmful, corrected = _flip_counts(previous_preds, evaluated["predictions"], truth)
        net_harmful, net_corrected = _flip_counts(initial_preds, evaluated["predictions"], truth)
        cumulative_harmful += harmful
        cumulative_corrected += corrected
        if cp == 0:
            initial_preds = evaluated["predictions"].copy()
        previous_preds = evaluated["predictions"].copy()
        saved_metric = by_cp_metrics[cp]
        require(abs(float(saved_metric["ba_product_macro"]) - evaluated["metrics"]["ba_product_macro"]) < 1e-12,
                f"{protocol} reused gamma=0 BA no longer matches P1-B log")
        require(abs(float(saved_metric["ap_product_macro_paired_class"]) -
                    float(evaluated["metrics"]["ap_product_macro_paired_class"])) < 1e-12,
                f"{protocol} reused gamma=0 paired AP no longer matches P1-B log")
        if cp == 0:
            w, b = w0, b0
        elif cp == 12:
            w, b = final_w, final_b
        else:
            w = b = None
        objective = grad_inf = None
        if cp in (0, 12):
            # P1-B state is preserved and never refit here. Recompute diagnostics only.
            feedback_rows = [r for r in events if int(r["stream_order"]) <= cp and
                             r["binary_target_if_revealed"] in {"0", "1"}]
            x_history = [features[r["image_id"]] for r in feedback_rows]
            y_history = [int(r["binary_target_if_revealed"]) for r in feedback_rows]
            # D-fit values are already registered in the cached initial state; append only
            # preserved legal feedback features for the endpoint diagnostic.
            with np.load(P1_RUN_DIR / "model_states.npz", allow_pickle=False) as p1states:
                base_X = p1states["X__refit_alpha_1e-04__initial__base_X"].astype(np.float64)
                base_y = p1states["X__refit_alpha_1e-04__initial__base_y"].astype(np.int8)
            if x_history:
                X = np.vstack([base_X, np.stack(x_history).astype(np.float64)])
                y = np.concatenate([base_y, np.asarray(y_history, dtype=np.int8)])
            else:
                X, y = base_X, base_y
            theta = np.concatenate([w, [b]])
            objective, grad = objective_and_gradient(theta, X, y, w0, lam=LAMBDA, gamma=0.0)
            grad_inf = float(np.max(np.abs(grad)))
        metric = {
            "protocol": protocol,
            "gamma": 0.0,
            "lambda": LAMBDA,
            "trajectory_source": "p1b_archived_full_history_refit_reused",
            "checkpoint": cp,
            "feedback_event_units_consumed": cp,
            "binary_feedback_count": sum(r["binary_target_if_revealed"] in {"0", "1"}
                                          for r in events if int(r["stream_order"]) <= cp),
            "future_views": evaluated["metrics"]["future_views"],
            "future_binary_views": evaluated["metrics"]["future_binary_views"],
            "future_positive": evaluated["binary_positive_n"],
            "future_negative": evaluated["binary_negative_n"],
            "ba_product_macro": evaluated["metrics"]["ba_product_macro"],
            "ba_eligible_product_count": evaluated["metrics"]["ba_eligible_product_count"],
            "ap_product_macro_paired_class": evaluated["metrics"]["ap_product_macro_paired_class"],
            "ap_paired_class_product_count": evaluated["metrics"]["ap_paired_class_product_count"],
            "ap_product_macro_including_degenerate": evaluated["metrics"]["ap_product_macro_including_degenerate"],
            "roc_auc_product_macro": evaluated["roc_auc_product_macro"],
            "roc_auc_product_count": evaluated["roc_auc_product_count"],
            "positive_prediction_fraction": evaluated["positive_prediction_fraction"],
            "harmful_flips_from_previous_checkpoint": harmful,
            "corrected_flips_from_previous_checkpoint": corrected,
            "cumulative_harmful_transitions": cumulative_harmful,
            "cumulative_corrected_transitions": cumulative_corrected,
            "net_harmful_views_vs_checkpoint0": net_harmful,
            "net_corrected_views_vs_checkpoint0": net_corrected,
            "objective": objective,
            "gradient_inf_norm": grad_inf,
            "update_latency_ms": None,
            "fit_iterations": None,
            "function_evaluations": None,
            "optimization_count": sum(r["binary_target_if_revealed"] in {"0", "1"}
                                       for r in events if int(r["stream_order"]) <= cp),
            "state_sha256": saved_metric["state_sha256"],
            "state_payload_bytes": saved_metric["state_payload_bytes"],
            "state_parameters_available": cp in (0, 12),
            "objective_gradient_available": cp in (0, 12),
            "l2_regularization_term": (0.5 * LAMBDA * float(np.dot(w, w)) if w is not None else None),
            "proximal_regularization_term": 0.0 if w is not None else None,
            **(weight_diagnostics(w, b, w0, b0) if w is not None else {}),
        }
        rows["metrics"].append(metric)
        for product_row in by_cp_products[cp]:
            rows["products"].append({
                "protocol": protocol,
                "gamma": 0.0,
                "trajectory_source": "p1b_archived_full_history_refit_reused",
                **product_row,
                "roc_auc": evaluated["roc_auc_product_values"].get(product_row["product"]),
            })
        _record_future_predictions(rows["predictions"], protocol, 0.0,
                                   "p1b_archived_full_history_refit_reused", cp, scores, future)
    rows["events"] = events
    return rows


def _newton_reference(
    X: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x64 = np.asarray(X, dtype=np.float64)
    y8 = np.asarray(y, dtype=np.int8)
    init = np.zeros(x64.shape[1] + 1, dtype=np.float64)
    result = minimize(
        lambda theta, xx, yy, anchor: objective_and_gradient(
            theta, xx, yy, anchor, lam=LAMBDA, gamma=gamma
        ),
        init,
        args=(x64, y8, w0),
        method="Newton-CG",
        jac=True,
        hessp=lambda theta, p, xx, yy, anchor: hessian_vector_product(
            theta, p, xx, yy, lam=LAMBDA, gamma=gamma
        ),
        options=dict(NEWTON_OPTIONS),
    )
    theta = np.asarray(result.x, dtype=np.float64)
    value, gradient = objective_and_gradient(theta, x64, y8, w0, lam=LAMBDA, gamma=gamma)
    return theta, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "objective": value,
        "gradient_inf_norm": float(np.max(np.abs(gradient))),
    }


def numerical_cross_checks(
    *,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    protocols: dict[str, tuple[list[Any], list[Any]]],
    trajectories: dict[tuple[str, float], dict[str, Any]],
    w0: np.ndarray,
    cpu_start: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for protocol in ("X", "W"):
        feedback, future = protocols[protocol]
        states = trajectories[(protocol, GAMMAS[0])]["states"]
        for gamma in GAMMAS:
            states = trajectories[(protocol, gamma)]["states"]
            for checkpoint_id in VERIFY_CHECKPOINTS:
                if time.process_time() - cpu_start > MAX_CPU_SECONDS:
                    raise TimeoutError("30-minute cumulative CPU experiment ceiling reached during Newton-CG checks")
                events = [event for event in feedback if event.stream_order <= checkpoint_id and
                          p1b_core.binary_target(event.state) is not None]
                if events:
                    X = np.vstack([np.asarray(x_fit, dtype=np.float64),
                                   np.stack([event.x for event in events]).astype(np.float64)])
                    y = np.concatenate([np.asarray(y_fit, dtype=np.int8),
                                        np.asarray([p1b_core.binary_target(event.state) for event in events], dtype=np.int8)])
                else:
                    X = np.asarray(x_fit, dtype=np.float64)
                    y = np.asarray(y_fit, dtype=np.int8)
                state = states[checkpoint_id]
                theta_fit = np.concatenate([state["coef"].reshape(-1), state["intercept"].reshape(-1)])
                reference, diag = _newton_reference(X, y, w0, gamma)
                lbfgs_obj, lbfgs_grad = objective_and_gradient(theta_fit, X, y, w0,
                                                                lam=LAMBDA, gamma=gamma)
                reference_scores = np.stack([item.x for item in future]).astype(np.float64) @ reference[:-1] + reference[-1]
                lbfgs_scores = np.stack([item.x for item in future]).astype(np.float64) @ theta_fit[:-1] + theta_fit[-1]
                max_score_diff = float(np.max(np.abs(reference_scores - lbfgs_scores)))
                objective_diff = abs(float(diag["objective"]) - float(lbfgs_obj))
                passed = (diag["success"] and diag["gradient_inf_norm"] <= GRAD_INF_TOL and
                          float(np.max(np.abs(lbfgs_grad))) <= GRAD_INF_TOL and
                          objective_diff <= NEWTON_OBJECTIVE_ATOL and
                          max_score_diff <= NEWTON_MAX_FUTURE_SCORE_ABS)
                checks.append({
                    "protocol": protocol,
                    "gamma": gamma,
                    "checkpoint": checkpoint_id,
                    "training_rows": int(len(y)),
                    "newton_success": diag["success"],
                    "newton_message": diag["message"],
                    "newton_iterations": diag["iterations"],
                    "newton_function_evaluations": diag["function_evaluations"],
                    "newton_objective": diag["objective"],
                    "lbfgsb_objective": float(lbfgs_obj),
                    "objective_abs_difference": objective_diff,
                    "newton_gradient_inf_norm": diag["gradient_inf_norm"],
                    "lbfgsb_gradient_inf_norm": float(np.max(np.abs(lbfgs_grad))),
                    "max_future_score_abs_difference": max_score_diff,
                    "passed": bool(passed),
                })
    require(len(checks) <= NEWTON_MAX_RUNS, "numerical verification exceeded 16 independent solves")
    return checks


def build_reports_data(
    *,
    support: dict[str, dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    product_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    fits: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    integrity: dict[str, Any],
    anchor: dict[str, Any],
    initial_score_diffs: dict[str, float],
    protected_before: dict[str, dict[str, Any]],
    protected_after: dict[str, dict[str, Any]],
    elapsed_wall: float,
    elapsed_cpu: float,
    peak_rss: int,
    artifact_bytes: int,
    output_path: Path,
) -> dict[str, Any]:
    cp12 = [r for r in metric_rows if int(r["checkpoint"]) == 12]
    summary = []
    for row in cp12:
        summary.append({
            "protocol": row["protocol"],
            "gamma": row["gamma"],
            "source": row["trajectory_source"],
            "BA": row["ba_product_macro"],
            "AP_paired": row["ap_product_macro_paired_class"],
            "AUC_macro": row["roc_auc_product_macro"],
            "harmful_from_0": row["net_harmful_views_vs_checkpoint0"],
            "corrected_from_0": row["net_corrected_views_vs_checkpoint0"],
        })
    product_support = []
    for protocol in ("X", "W"):
        for row in product_rows:
            if row.get("protocol") == protocol and float(row["gamma"]) == 0.0 and int(row["checkpoint"]) == 12:
                product_support.append({
                    "protocol": protocol,
                    "product": row["product"],
                    "present": int(row["present"]),
                    "absent": int(row["absent"]),
                    "uncertain": int(row["uncertain"]),
                    "n_binary": int(row["n_binary"]),
                    "BA_defined": row["ba_status"],
                    "AP_status": row["ap_status"],
                })
    return {
        "status": "PASS" if all(r.get("status") == "PASS" for r in fits) and all(c["passed"] for c in checks) else "PARTIAL",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "execution_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "scikit-learn", "threadpoolctl", "psutil")},
        "device": "CPU only",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_imported": "torch" in sys.modules,
        "thread_limit": 2,
        "feature_cache": {"records": integrity["n_total"], "d_fit": integrity["n_dfit"], "d_dev": integrity["n_ddev"],
                           "feature_index_sha256": integrity["feature_manifest_sha256"],
                           "input_manifest_sha256": integrity["input_manifest_sha256"],
                           "feedback_tape_sha256": integrity["feedback_tape_sha256"],
                           "snapshot_sha256": integrity["snapshot_sha256"]},
        "feedback_future_support": support,
        "d_fit_binary": {"n": int(len(anchor["base_y"])), "present": int(np.sum(anchor["base_y"] == 1)),
                         "absent": int(np.sum(anchor["base_y"] == 0))},
        "anchor": {"initial_learner_state_sha256": anchor["initial_learner_state_sha256"],
                   "objective": anchor["anchor_objective"],
                   "gradient_inf_norm": anchor["anchor_gradient_inf"],
                   "b0_frozen": True,
                   "initial_scores_max_abs_diff_vs_p1b": initial_score_diffs,
                   "source": "preserved P1-B full-history Logistic alpha=1e-4 initial state"},
        "optimizer": {"lambda": LAMBDA, "gamma": [0.0, *GAMMAS],
                      "lbfgsb_options": {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 10000, "maxls": 50, "maxcor": 10},
                      "gradient_inf_success_threshold": GRAD_INF_TOL,
                      "newton_cg_verification_runs": len(checks),
                      "newton_verification_all_pass": all(c["passed"] for c in checks),
                      "newton_objective_atol": NEWTON_OBJECTIVE_ATOL,
                      "newton_max_future_score_abs": NEWTON_MAX_FUTURE_SCORE_ABS},
        "fit_summary": {"proximal_optimization_count": sum(int(r["valid_binary_update_count"]) for r in fits),
                        "successes": sum(r.get("status") == "PASS" for r in fits),
                        "failures": sum(r.get("status") != "PASS" for r in fits),
                        "configurations": fits},
        "checkpoint12_summary": summary,
        "product_support": product_support,
        "elapsed_wall_seconds": elapsed_wall,
        "elapsed_process_cpu_seconds": elapsed_cpu,
        "peak_observed_rss_bytes": peak_rss,
        "rss_target_bytes": MAX_RSS_BYTES,
        "rss_target_exceeded": peak_rss > MAX_RSS_BYTES,
        "artifact_bytes": artifact_bytes,
        "artifact_limit_bytes": MAX_OUTPUT_BYTES,
        "protected_tree_fingerprints_before": protected_before,
        "protected_tree_fingerprints_after": protected_after,
        "protected_trees_unchanged": protected_before == protected_after,
        "output_directory": str(output_path.relative_to(ROOT)).replace("\\", "/"),
    }


def main() -> int:
    if Path(sys.executable).resolve() != Path(r"D:\anaconda3\envs\torch\python.exe").resolve():
        raise RuntimeError("Use the authorized D:\\anaconda3\\envs\\torch\\python.exe interpreter")
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported; E1 requires CPU-only cached-feature execution")
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() ==
            "stage3-eg-famar-execution", "unexpected research branch")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() ==
            "a50459f6061a951ca8aa7d6f34bd0863b6145ebd", "starting HEAD differs from E1 authorization")
    if RUN_DIR.exists():
        require(not any(RUN_DIR.iterdir()), f"refusing to overwrite nonempty E1 output directory: {RUN_DIR}")
    else:
        RUN_DIR.mkdir(parents=True)
    start_wall = time.perf_counter()
    cpu_start = time.process_time()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    protected_before = protected_snapshot()
    original_p1b_hashes = validate_original_p1b_hashes()
    require(len(original_p1b_hashes) == 15, "original P1-B archive hash verification incomplete")

    with threadpool_limits(limits=2):
        integrity = p1b_runner.capture_preflight()
        labels, _repeat_counts = p1b_runner.load_human_labels(integrity)
        features, _joined = p1b_runner.build_items(integrity, labels)
        dfit_ids = [r["image_id"] for r in integrity["input_rows"] if r["data_role"] == "D-fit"]
        fit_ids = [i for i in dfit_ids if p1b_core.binary_target(labels[i]["state"]) is not None]
        x_fit = np.stack([features[i] for i in fit_ids]).astype(np.float32)
        y_fit = np.asarray([p1b_core.binary_target(labels[i]["state"]) for i in fit_ids], dtype=np.int8)
        require(len(x_fit) == 52 and Counter(y_fit.tolist()) == Counter({1: 27, 0: 25}),
                "D-fit does not contain exactly the registered 52 binary records")
        protocols: dict[str, tuple[list[Any], list[Any]]] = {}
        support: dict[str, dict[str, Any]] = {}
        for protocol in ("X", "W"):
            feedback, future = p1b_runner.build_role_items(protocol, integrity["tape_rows"], features, labels)
            support[protocol] = validate_protocol_support(feedback, future, protocol)
            protocols[protocol] = feedback, future

        anchor = load_registered_anchor(x_fit, y_fit)
        initial_score_diffs = verify_registered_initial_scores(
            features, protocols, anchor["w0"], anchor["b0"]
        )
        if anchor["anchor_gradient_inf"] > GRAD_INF_TOL:
            post_integrity = p1b_runner.capture_preflight()
            protected_after = protected_snapshot()
            require(post_integrity["snapshot_sha256"] == integrity["snapshot_sha256"],
                    "P1-A source/feature/manifests changed before the compatibility gate stopped")
            require(protected_before == protected_after,
                    "a protected P1-A/P1-B/G0 tree changed before the compatibility gate stopped")
            require(len(validate_original_p1b_hashes()) == 15,
                    "original P1-B run artifact verification failed at the compatibility gate")
            gate = {
                "status": "BLOCKED_BEFORE_FITTING",
                "reason": "HISTORICAL_SKLEARN_INITIALIZATION_EXCEEDS_REGISTERED_GRADIENT_TOLERANCE",
                "protocol_scope": "LP-D6-P2-E1 only",
                "execution_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "interpreter": sys.executable,
                "device": "CPU only; no Torch import; no GPU use",
                "integrity_gate": "PASS",
                "d_fit_binary_n": int(len(y_fit)),
                "d_fit_present_n": int(np.sum(y_fit == 1)),
                "d_fit_absent_n": int(np.sum(y_fit == 0)),
                "lambda": LAMBDA,
                "gamma_anchor_check": 0.0,
                "historical_sklearn_mapping": {"alpha": 1e-4, "n": int(len(y_fit)),
                                                "C": float(1.0 / (1e-4 * len(y_fit)))},
                "registered_objective_at_anchor": float(anchor["anchor_objective"]),
                "registered_gradient_inf_at_anchor": float(anchor["anchor_gradient_inf"]),
                "gradient_inf_tolerance": GRAD_INF_TOL,
                "gradient_intercept_at_anchor": float(anchor["anchor_gradient_vector"][-1]),
                "gradient_weight_inf_at_anchor": float(np.max(np.abs(anchor["anchor_gradient_vector"][:-1]))),
                "anchor_weight_norm": float(np.linalg.norm(anchor["w0"])),
                "anchor_intercept": float(anchor["b0"]),
                "anchor_state_digest": anchor["initial_learner_state_sha256"],
                "p1b_checkpoint0_max_abs_score_difference": initial_score_diffs,
                "p1b_initial_and_final_state_digests_verified": True,
                "original_p1b_files_hash_verified": 15,
                "p1a_snapshot_sha256_before_and_after": integrity["snapshot_sha256"],
                "p1b_original_artifacts_and_g0_trees_unchanged": True,
                "proximal_fits_run": 0,
                "gamma_zero_refits_run": 0,
                "newton_cg_verification_runs": 0,
                "newton_cg_not_started_due_to_initialization_gate": True,
                "future_labels_used_for_fit_or_gate": False,
                "note": "The saved P1-B initialization and B0 arrays were reused unchanged. No alternative initializer, optimizer setting, or scientific model fit was attempted.",
            }
            (RUN_DIR / "anchor_compatibility_gate.json").write_text(
                json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("D6_P2_E1_STATUS=BLOCKED_BEFORE_FITTING")
            print("reason=HISTORICAL_SKLEARN_INITIALIZATION_EXCEEDS_REGISTERED_GRADIENT_TOLERANCE")
            print("gradient_inf_norm=" + f"{anchor['anchor_gradient_inf']:.12g}")
            print("gradient_tolerance=" + f"{GRAD_INF_TOL:.12g}")
            print("proximal_fits=0")
            print("newton_cg_checks=0")
            return 2
        all_metric_rows: list[dict[str, Any]] = []
        all_product_rows: list[dict[str, Any]] = []
        all_event_rows: list[dict[str, Any]] = []
        all_prediction_rows: list[dict[str, Any]] = []
        all_fit_rows: list[dict[str, Any]] = []
        trajectories: dict[tuple[str, float], dict[str, Any]] = {}
        state_store: dict[str, np.ndarray] = {
            "registered_anchor_w0": anchor["w0"].astype(np.float64),
            "registered_anchor_b0": np.asarray([anchor["b0"]], dtype=np.float64),
        }
        for protocol in ("X", "W"):
            feedback, future = protocols[protocol]
            baseline = _baseline_replay(
                protocol=protocol,
                feedback=feedback,
                future=future,
                features=features,
                w0=anchor["w0"],
                b0=anchor["b0"],
                final_state=anchor["final_states"][protocol],
            )
            all_metric_rows.extend(baseline["metrics"])
            all_product_rows.extend(baseline["products"])
            all_prediction_rows.extend(baseline["predictions"])
            all_event_rows.extend(baseline["events"])
            for gamma in GAMMAS:
                trajectory = replay_proximal(
                    protocol=protocol,
                    gamma=gamma,
                    x_fit=x_fit,
                    y_fit=y_fit,
                    w0=anchor["w0"],
                    b0=anchor["b0"],
                    feedback=feedback,
                    future=future,
                    cpu_start=cpu_start,
                )
                trajectories[(protocol, gamma)] = trajectory
                all_metric_rows.extend(trajectory["rows"]["metrics"])
                all_product_rows.extend(trajectory["rows"]["products"])
                all_prediction_rows.extend(trajectory["rows"]["predictions"])
                all_event_rows.extend(trajectory["rows"]["events"])
                all_fit_rows.append({
                    "protocol": protocol,
                    "gamma": gamma,
                    "status": "PASS",
                    "valid_binary_update_count": trajectory["fit_count"],
                    "event_cost_units": 12,
                    "solver": "L-BFGS-B",
                    "max_iterations": max(int(r["fit_iterations"] or 0) for r in trajectory["rows"]["metrics"]),
                    "max_gradient_inf_norm": max(float(r["gradient_inf_norm"]) for r in trajectory["rows"]["metrics"]),
                })
                require(trajectory["fit_count"] == (12 if protocol == "X" else 11),
                        f"{protocol} proximal update count differs from the registered tape")
                for checkpoint_id, arrays in trajectory["states"].items():
                    prefix = f"{protocol}__gamma_{gamma:.0e}__checkpoint_{checkpoint_id:02d}__"
                    for name, value in arrays.items():
                        state_store[prefix + name] = value
                peak_rss = max(peak_rss, process.memory_info().rss)
                if time.process_time() - cpu_start > MAX_CPU_SECONDS:
                    raise TimeoutError("30-minute cumulative CPU experiment ceiling reached")

        checks = numerical_cross_checks(
            x_fit=x_fit,
            y_fit=y_fit,
            protocols=protocols,
            trajectories=trajectories,
            w0=anchor["w0"],
            cpu_start=cpu_start,
        )
        write_csv(RUN_DIR / "checkpoint_metrics.csv", all_metric_rows)
        write_csv(RUN_DIR / "product_metrics.csv", all_product_rows)
        write_csv(RUN_DIR / "feedback_event_log.csv", all_event_rows)
        write_csv(RUN_DIR / "future_predictions.csv", all_prediction_rows)
        write_csv(RUN_DIR / "proximal_fit_summary.csv", all_fit_rows)
        write_csv(RUN_DIR / "newton_cg_verification.csv", checks)
        np.savez_compressed(RUN_DIR / "model_states.npz", **state_store)

        # Numerical objective/gradient checks at every registered selected checkpoint.
        objective_checks = []
        for protocol in ("X", "W"):
            feedback, _future = protocols[protocol]
            for gamma in GAMMAS:
                trajectory = trajectories[(protocol, gamma)]
                for cp in VERIFY_CHECKPOINTS:
                    params = trajectory["states"][cp]
                    events = [e for e in feedback if e.stream_order <= cp and
                              p1b_core.binary_target(e.state) is not None]
                    if events:
                        X = np.vstack([x_fit.astype(np.float64), np.stack([e.x for e in events]).astype(np.float64)])
                        y = np.concatenate([y_fit, np.asarray([p1b_core.binary_target(e.state) for e in events], dtype=np.int8)])
                    else:
                        X, y = x_fit.astype(np.float64), y_fit
                    theta = np.concatenate([params["coef"].reshape(-1), params["intercept"].reshape(-1)])
                    value, gradient = objective_and_gradient(theta, X, y, anchor["w0"],
                                                              lam=LAMBDA, gamma=gamma)
                    objective_checks.append({
                        "protocol": protocol,
                        "gamma": gamma,
                        "checkpoint": cp,
                        "training_rows": len(y),
                        "objective": value,
                        "gradient_inf_norm": float(np.max(np.abs(gradient))),
                        "finite": bool(np.isfinite(value) and np.isfinite(gradient).all()),
                    })
        write_csv(RUN_DIR / "objective_gradient_checks.csv", objective_checks)
        require(all(row["finite"] and row["gradient_inf_norm"] <= GRAD_INF_TOL for row in objective_checks),
                "selected objective/gradient checkpoint verification failed")
        require(len(checks) == 16 and all(row["passed"] for row in checks),
                "independent Newton-CG verification did not pass all 16 registered checks")
        require(sum(int(r["valid_binary_update_count"]) for r in all_fit_rows) == 46,
                "new proximal optimization count differs from the registered maximum 46")

        artifact_bytes = sum(p.stat().st_size for p in RUN_DIR.rglob("*") if p.is_file())
        require(artifact_bytes <= MAX_OUTPUT_BYTES, "E1 artifacts exceed the 100 MiB limit")
        protected_after = protected_snapshot()
        require(protected_before == protected_after, "a protected P1-A/P1-B/G0 tree changed during E1")
        post_integrity = p1b_runner.capture_preflight()
        require(post_integrity["snapshot_sha256"] == integrity["snapshot_sha256"],
                "P1-A source/feature/manifests changed during E1")
        validate_original_p1b_hashes()

    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu = time.process_time() - cpu_start
    peak_rss = max(peak_rss, process.memory_info().rss)
    if elapsed_cpu > MAX_CPU_SECONDS:
        raise TimeoutError("30-minute cumulative CPU experiment ceiling exceeded")
    provenance = build_reports_data(
        support=support,
        metric_rows=all_metric_rows,
        product_rows=all_product_rows,
        events=all_event_rows,
        fits=all_fit_rows,
        checks=checks,
        integrity=integrity,
        anchor=anchor,
        initial_score_diffs=initial_score_diffs,
        protected_before=protected_before,
        protected_after=protected_after,
        elapsed_wall=elapsed_wall,
        elapsed_cpu=elapsed_cpu,
        peak_rss=peak_rss,
        artifact_bytes=artifact_bytes,
        output_path=RUN_DIR,
    )
    provenance["original_p1b_files_hash_verified"] = len(original_p1b_hashes)
    provenance["p1b_input_snapshot_sha256_after"] = post_integrity["snapshot_sha256"]
    (RUN_DIR / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("D6_P2_E1_STATUS=" + provenance["status"])
    print("proximal_fits=" + str(provenance["fit_summary"]["proximal_optimization_count"]))
    print("newton_cg_checks=" + str(len(checks)))
    print("cpu_seconds=" + f"{elapsed_cpu:.3f}")
    print("peak_rss_bytes=" + str(peak_rss))
    print("artifact_bytes=" + str(artifact_bytes))
    return 0 if provenance["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

