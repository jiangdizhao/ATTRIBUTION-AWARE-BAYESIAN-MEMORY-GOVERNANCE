"""Guarded CPU-only LP-D6-P2-E2-B profile and fixed 28-trajectory runner."""
from __future__ import annotations

import os

# Set numerical/GPU boundaries before importing NumPy, SciPy or sklearn.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from data import (
    first_state_by_item,
    load_dfit_labels,
    load_feature_bundle,
    load_tape_labels,
    read_csv,
    read_source_type_metadata,
    sha256_file,
    stable_tree_identity,
    validate_tape,
    verify_frozen_cache,
    write_csv,
)
from evaluator import source_type_state_counts, summarize
from model import (
    build_global_readout,
    fit_global,
    fit_mil,
    mil_initial_parameters,
    mil_objective_parts,
    parameter_digest,
    predict_global,
    predict_mil,
    valid_bag,
)
from pipeline import run_trajectory


CPU_BUDGET_SECONDS = 30 * 60
EXPECTED_SOLVE_COUNTS = {"global": 144, "mil": 192, "total": 336}


class RSSSampler:
    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except psutil.Error:
            pass


def json_dump(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, default=_json_default)
        stream.write("\n")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def json_safe(data):
    return json.loads(json.dumps(data, default=_json_default, allow_nan=False))


def code_hashes() -> dict[str, str]:
    paths = sorted(SCRIPT_DIR.glob("*.py")) + [ROOT / "tests/research_d6/test_e2b_numerics.py"]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def code_fingerprint() -> str:
    raw = json.dumps(code_hashes(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ensure_identity() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if (head, branch) != (config.EXPECTED_HEAD, config.EXPECTED_BRANCH):
        raise RuntimeError(f"Unexpected E2-B starting identity: {branch}/{head}")


def ensure_output_root() -> None:
    probe = (config.OUTPUT_ROOT / ".gitignore_probe").relative_to(ROOT).as_posix()
    check = subprocess.run(["git", "check-ignore", "-q", probe], cwd=ROOT)
    if check.returncode != 0:
        raise RuntimeError("E2-B output root is not Git-ignored")


def run_synthetic_tests() -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "2"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.research_d6.test_e2b_numerics", "-v"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=300,
    )
    transcript = result.stdout + result.stderr
    if result.returncode != 0 or "Ran 15 tests" not in transcript or "OK" not in transcript:
        raise RuntimeError(f"Synthetic E2-B checks failed before data-label access:\n{transcript}")
    return {"return_code": result.returncode, "test_count": 15,
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            "transcript": transcript}


def fixed_solver_options() -> dict[str, float | int]:
    return dict(config.SOLVER)


def prepare_preflight() -> tuple[dict, list[dict[str, str]], dict]:
    ensure_identity()
    ensure_output_root()
    cache = verify_frozen_cache()
    tape_rows, tape_summary = validate_tape(cache["registered_by_id"])
    return cache, tape_rows, tape_summary


def dfit_ids_in_order(cache: dict) -> list[str]:
    return [row["image_id"] for row in cache["registered_rows"] if row["data_role"] == "D-fit"]


def profile_only() -> int:
    if config.OUTPUT_ROOT.exists() and (config.OUTPUT_ROOT / "bounded_profile.json").exists():
        raise FileExistsError("bounded profile already exists; refusing to overwrite")
    tests = run_synthetic_tests()
    cache, _tape, tape_summary = prepare_preflight()
    dfit_ids = dfit_ids_in_order(cache)
    dfit_states, dfit_summary = load_dfit_labels(dfit_ids)
    y = np.asarray([config.VALID_STATES[dfit_states[i]["state"]]
                    for i in dfit_ids if config.VALID_STATES[dfit_states[i]["state"]] is not None], dtype=np.int8)
    binary_ids = [i for i in dfit_ids if config.VALID_STATES[dfit_states[i]["state"]] is not None]
    features = load_feature_bundle(config.PROFILE_BACKBONE, binary_ids)
    bags = [valid_bag(features[i]["patch_unit"], features[i]["valid_patch_mask"]) for i in binary_ids]
    initial = mil_initial_parameters(bags[0].shape[1], config.PROFILE_SEED)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    with threadpool_limits(limits=config.THREAD_LIMIT), RSSSampler() as memory:
        result = fit_mil(
            bags, y, initial, config.LAMBDA, rho=0.0, tau=config.TAU,
            margin=config.MARGIN, kappa=config.KAPPA, options=fixed_solver_options(),
        )
    process_cpu = time.process_time() - cpu_start
    elapsed = time.perf_counter() - wall_start
    # The full 336-solve count includes this D-fit configuration, reused in the main matrix.
    projected_seconds = process_cpu * EXPECTED_SOLVE_COUNTS["total"] * 2.0
    resource_status = "PASS" if projected_seconds <= CPU_BUDGET_SECONDS else "OVER_BUDGET"
    status = "PASS" if result["accepted"] and resource_status == "PASS" else "BLOCKED"

    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    init_path = config.OUTPUT_ROOT / "profile_initialization.npz"
    if init_path.exists():
        raise FileExistsError("profile initialization already exists; refusing to overwrite")
    np.savez_compressed(init_path,
                        initial_theta=initial.astype(np.float64),
                        fitted_theta=np.asarray(result["theta"], dtype=np.float64))
    profile = {
        "schema": "lp.d6.p2.e2b.bounded_profile.v1",
        "status": status,
        "resource_gate": resource_status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_head": config.EXPECTED_HEAD,
        "branch": config.EXPECTED_BRANCH,
        "mode": "D-fit-only; no future labels joined or evaluated",
        "synthetic_tests": {k: v for k, v in tests.items() if k != "transcript"},
        "cache_manifest_sha256": cache["manifest_hashes"],
        "input_manifest_sha256": cache["input_manifest_sha256"],
        "feature_manifest_p1a_sha256": cache["p1a_feature_manifest_sha256"],
        "wx_tape_sha256": config.EXPECTED["wx_tape_sha256"],
        "dfit_label_summary": dfit_summary,
        "dfit_binary_bags": len(binary_ids),
        "tape_roles_checked_without_reading_annotation_values": tape_summary,
        "configuration": {
            "backbone": config.PROFILE_BACKBONE, "method": config.PROFILE_METHOD,
            "seed": config.PROFILE_SEED, "lambda": config.LAMBDA, "tau": config.TAU,
            "loss": "mean bag BCE + lambda/2 ||w||^2", "precision": "FP64 solver; FP32 source cache",
            "initialization": config.MIL_INIT_WEIGHT_STD, "bias": config.MIL_INIT_BIAS,
            "optimizer": config.SOLVER, "threads": config.THREAD_LIMIT,
        },
        "solver": {k: v for k, v in result.items() if k != "theta"},
        "elapsed_wall_seconds": elapsed,
        "process_cpu_seconds": process_cpu,
        "sampled_peak_rss_bytes": memory.peak,
        "expected_full_matrix_solver_count_including_profile": EXPECTED_SOLVE_COUNTS,
        "conservative_full_matrix_projection_seconds": projected_seconds,
        "cpu_planning_window_seconds": CPU_BUDGET_SECONDS,
        "profile_initialization_file": init_path.name,
        "profile_initialization_sha256": sha256_file(init_path),
        "code_hashes": code_hashes(),
        "code_fingerprint": code_fingerprint(),
        "future_labels_read": False,
        "device": "CPU-only; CUDA_VISIBLE_DEVICES empty",
    }
    json_dump(config.OUTPUT_ROOT / "bounded_profile.json", profile)
    print(f"PROFILE_STATUS={status}")
    print(f"PROFILE_SOLVER={result['status']} GRAD_INF={result['gradient_inf_norm']:.6e}")
    print(f"PROJECTED_SECONDS={projected_seconds:.2f} WINDOW_SECONDS={CPU_BUDGET_SECONDS}")
    print(f"PEAK_RSS_BYTES={memory.peak}")
    return 0 if status == "PASS" else 2


def annotation_input_hashes() -> dict[str, str]:
    values = {"tape": sha256_file(config.TAPE_PATH),
              "p1a_input_manifest": sha256_file(config.P1A_INPUT_MANIFEST),
              "p1a_feature_manifest": sha256_file(config.P1A_FEATURE_MANIFEST),
              "p1b_source_metadata": sha256_file(config.P1B_JOINED_METADATA)}
    for source, (queue, annotations) in config.ANNOTATION_SOURCES.items():
        values[f"{source}_queue"] = sha256_file(queue)
        values[f"{source}_annotations"] = sha256_file(annotations)
    return values


def build_protocol_rows(tape_rows: list[dict[str, str]], protocol: str,
                        state_by_id: dict[str, dict[str, Any]], source_by_id: dict[str, str],
                        registered_by_id: dict[str, dict[str, str]]) -> tuple[list[dict], list[dict]]:
    rows = [r for r in tape_rows if r["protocol"] == protocol]
    feedback_tape = sorted((r for r in rows if r["feedback_or_future_role"] == "feedback_query"),
                           key=lambda r: int(r["stream_order"]))
    future_tape = [r for r in rows if r["feedback_or_future_role"] == "future_eval_candidate"]
    if len(feedback_tape) != 12 or len(future_tape) != 24:
        raise RuntimeError("protocol row count changed after preflight")
    event_states = Counter(state_by_id[r["image_id"]]["state"] for r in feedback_tape)
    future_states = Counter(state_by_id[r["image_id"]]["state"] for r in future_tape)
    expected_event = (Counter({"present": 7, "absent_assessable": 5}) if protocol == "X"
                      else Counter({"present": 5, "absent_assessable": 6,
                                    "unobservable_or_uncertain": 1}))
    expected_future = (Counter({"present": 12, "absent_assessable": 9,
                                "unobservable_or_uncertain": 3}) if protocol == "X"
                       else Counter({"present": 14, "absent_assessable": 7,
                                     "unobservable_or_uncertain": 3}))
    if event_states != expected_event:
        raise RuntimeError(f"{protocol} feedback labels differ from frozen tape audit: {dict(event_states)}")
    if future_states != expected_future:
        raise RuntimeError(f"{protocol} future support differs from frozen tape audit: {dict(future_states)}")
    events = []
    for row in feedback_tape:
        meta = registered_by_id[row["image_id"]]
        events.append({
            "event_id": row["event_id"], "image_id": row["image_id"],
            "sample_key": row["sample_key"], "product": row["product"],
            "view": row["view"], "stream_order": int(row["stream_order"]),
            "features": None,
            "feedback_cost_status": row["feedback_cost_status"],
            "revealed_fields": row["revealed_fields"],
            "exposure_status": meta["exposure_status"],
        })
    future = []
    for row in future_tape:
        meta = registered_by_id[row["image_id"]]
        future.append({
            "image_id": row["image_id"], "product": row["product"],
            "sample_key": row["sample_key"], "view": row["view"],
            "source_type_evaluator_only": source_by_id.get(row["image_id"], "UNKNOWN"),
            "source_split_evaluator_only": meta["source_split"],
        })
    return events, future


def _fit_record(context: dict, result: dict, stage: str, checkpoint: int | None,
                training_rows: int, pair_count: int | None = None) -> dict:
    record = {**context, "stage": stage,
              "checkpoint": "" if checkpoint is None else checkpoint,
              "training_rows": training_rows,
              "status": result["status"],
              "solver_success": result["solver_success"],
              "solver_status": result["solver_status"],
              "message": result["message"],
              "iterations": result["iterations"],
              "function_evaluations": result["function_evaluations"],
              "objective": result["objective"],
              "gradient_inf_norm": result["gradient_inf_norm"],
              "elapsed_seconds": result["elapsed_seconds"],
              "finite": result["finite"]}
    if pair_count is not None:
        record["unique_positive_negative_pairs"] = pair_count
    return record


def _flip_metrics(checkpoint_rows: list[dict], state_by_id: dict[str, dict[str, Any]]) -> list[dict]:
    by_cp = {int(r["checkpoint"]): r for r in checkpoint_rows}
    initial_items = {r["image_id"]: r for r in by_cp[0]["item_predictions"]}
    result = []
    for cp in sorted(by_cp):
        current = {r["image_id"]: r for r in by_cp[cp]["item_predictions"]}
        previous = {r["image_id"]: r for r in by_cp[cp - 1]["item_predictions"]} if cp > 0 and cp - 1 in by_cp else {}
        metrics = {"checkpoint": cp,
                   "harmful_vs_previous": 0, "corrected_vs_previous": 0,
                   "prediction_flips_vs_previous": 0,
                   "harmful_vs_checkpoint0": 0, "corrected_vs_checkpoint0": 0,
                   "prediction_flips_vs_checkpoint0": 0}
        for image_id, item in current.items():
            y = config.VALID_STATES[state_by_id[image_id]["state"]]
            if y is None:
                continue
            now_correct = int(item["prediction"]) == y
            base_correct = int(initial_items[image_id]["prediction"]) == y
            if int(item["prediction"]) != int(initial_items[image_id]["prediction"]):
                metrics["prediction_flips_vs_checkpoint0"] += 1
            if base_correct and not now_correct:
                metrics["harmful_vs_checkpoint0"] += 1
            elif not base_correct and now_correct:
                metrics["corrected_vs_checkpoint0"] += 1
            if image_id in previous:
                prev_pred = int(previous[image_id]["prediction"])
                prev_correct = prev_pred == y
                if int(item["prediction"]) != prev_pred:
                    metrics["prediction_flips_vs_previous"] += 1
                if prev_correct and not now_correct:
                    metrics["harmful_vs_previous"] += 1
                elif not prev_correct and now_correct:
                    metrics["corrected_vs_previous"] += 1
        result.append(metrics)
    return result


def flatten_trajectory(trajectory: dict, *, backend: str, method: str, seed,
                       control_method: str, future_rows: list[dict],
                       state_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    checks = trajectory["checkpoints"]
    flips = {r["checkpoint"]: r for r in _flip_metrics(checks, state_by_id)}
    checkpoint_rows, product_rows, prediction_rows = [], [], []
    for cp in checks:
        metrics = {k: v for k, v in cp["metrics"].items()}
        metrics.update({
            "protocol": trajectory["protocol"], "backbone": backend,
            "method": method, "seed": "" if seed is None else seed,
            "control_method": control_method,
            "trajectory_id": trajectory["trajectory_id"],
            "checkpoint": cp["checkpoint"], "events_seen": cp["events_seen"],
            "binary_updates": cp["binary_updates"],
            "training_rows": cp["training_rows"],
            "solver_status": cp["solver_record"].get("status", "INITIAL_STATE"),
            "gradient_inf_norm": cp["solver_record"].get("gradient_inf_norm", ""),
            "fit_seconds": cp["fit_seconds"],
            "state_sha256": cp["state_sha256"],
            "status": cp["status"],
            **flips[cp["checkpoint"]],
        })
        checkpoint_rows.append(metrics)
        for row in cp["product_metrics"]:
            product_rows.append({
                "protocol": trajectory["protocol"], "backbone": backend,
                "method": method, "seed": "" if seed is None else seed,
                "trajectory_id": trajectory["trajectory_id"],
                "checkpoint": cp["checkpoint"], **row,
            })
        for row in cp["item_predictions"]:
            prediction_rows.append({
                "protocol": trajectory["protocol"], "backbone": backend,
                "method": method, "seed": "" if seed is None else seed,
                "control_method": control_method,
                "trajectory_id": trajectory["trajectory_id"],
                "checkpoint": cp["checkpoint"], **row,
            })
    events = [{"backbone": backend, "method": method,
               "seed": "" if seed is None else seed,
               "control_method": control_method, **row} for row in trajectory["events"]]
    by_cp = {int(r["checkpoint"]): r for r in checks}
    event_effects = []
    flip_by_cp = {r["checkpoint"]: r for r in _flip_metrics(checks, state_by_id)}
    for event in trajectory["events"]:
        cp = int(event["stream_order"])
        effect = flip_by_cp.get(cp, {})
        event_effects.append({
            "protocol": trajectory["protocol"], "backbone": backend,
            "method": method, "seed": "" if seed is None else seed,
            "stream_order": cp, "revealed_state": event["revealed_state_after_prediction"],
            "updated": event["updated"], "status": event["status"],
            "harmful_vs_previous": effect.get("harmful_vs_previous", "NO_VALID_POST_EVENT_CHECKPOINT"),
            "corrected_vs_previous": effect.get("corrected_vs_previous", "NO_VALID_POST_EVENT_CHECKPOINT"),
            "prediction_flips_vs_previous": effect.get("prediction_flips_vs_previous", "NO_VALID_POST_EVENT_CHECKPOINT"),
        })
    return checkpoint_rows, product_rows, prediction_rows, events, event_effects


def main_run() -> int:
    ensure_identity()
    ensure_output_root()
    if not config.OUTPUT_ROOT.is_dir():
        raise RuntimeError("E2-B bounded profile must pass before the main run")
    profile_path = config.OUTPUT_ROOT / "bounded_profile.json"
    profile_npz_path = config.OUTPUT_ROOT / "profile_initialization.npz"
    if not profile_path.is_file() or not profile_npz_path.is_file():
        raise RuntimeError("bounded D-fit profile artifacts are missing")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("status") != "PASS" or profile.get("resource_gate") != "PASS":
        raise RuntimeError("bounded profile did not pass both solver and resource gates")
    if profile.get("code_fingerprint") != code_fingerprint():
        raise RuntimeError("E2-B code changed after profiling; refusing an unmatched run")
    if config.RUN_DIR.exists():
        raise FileExistsError("E2-B main run directory exists; refusing overwrite/resume")

    cache, tape_rows, tape_summary = prepare_preflight()
    input_hashes_before = annotation_input_hashes()
    protected_before = {rel: stable_tree_identity(ROOT / rel) for rel in config.PROTECTED_PATHS}
    dfit_ids = dfit_ids_in_order(cache)
    dfit_states, dfit_summary = load_dfit_labels(dfit_ids)
    active_ids = {r["image_id"] for r in tape_rows
                  if r["feedback_or_future_role"] in {"feedback_query", "future_eval_candidate"}}
    dev_state_by_id, dev_summary = load_tape_labels(active_ids)
    state_by_id = {**dfit_states, **dev_state_by_id}
    source_types = read_source_type_metadata([r["image_id"] for r in cache["registered_rows"]])
    registered_by_id = cache["registered_by_id"]
    role_by_id = {r["image_id"]: r["data_role"] for r in cache["registered_rows"]}

    # Only now, after the bounded profile has passed, create the main output directory.
    config.RUN_DIR.mkdir(parents=True, exist_ok=False)
    (config.OUTPUT_ROOT / ".gitignore_probe").write_text("ignored\n", encoding="utf-8")
    (config.OUTPUT_ROOT / ".gitignore_probe").unlink()

    support_rows = []
    protocol_data = {}
    for protocol in ("X", "W"):
        events, future_rows = build_protocol_rows(tape_rows, protocol, state_by_id,
                                                  source_types, registered_by_id)
        support_rows.append({"protocol": protocol, "role": "feedback",
                             **dict(Counter(state_by_id[e["image_id"]]["state"] for e in events))})
        support_rows.append({"protocol": protocol, "role": "future",
                             **dict(Counter(state_by_id[r["image_id"]]["state"] for r in future_rows))})
        protocol_data[protocol] = (events, future_rows)

    code_before = code_hashes()
    all_solver_rows: list[dict] = []
    all_checkpoint_rows: list[dict] = []
    all_product_rows: list[dict] = []
    all_prediction_rows: list[dict] = []
    all_event_rows: list[dict] = []
    all_event_effect_rows: list[dict] = []
    trajectory_status_rows: list[dict] = []
    state_arrays: dict[str, np.ndarray] = {}
    main_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_peak = psutil.Process().memory_info().rss

    with threadpool_limits(limits=config.THREAD_LIMIT), RSSSampler() as sampler:
        for backend in config.BACKBONES:
            active_order = [r["image_id"] for r in cache["registered_rows"]
                            if r["image_id"] in set(dfit_ids) | active_ids]
            bundles = load_feature_bundle(backend, active_order)
            global_features: dict[str, dict[str, np.ndarray]] = {r: {} for r in config.GLOBAL_READOUTS}
            mil_bags: dict[str, np.ndarray] = {}
            for image_id in active_order:
                data = bundles[image_id]
                for readout in config.GLOBAL_READOUTS:
                    global_features[readout][image_id] = build_global_readout(
                        readout, data["cls_unit"], data["patch_unit"], data["valid_patch_mask"]
                    )
                mil_bags[image_id] = valid_bag(data["patch_unit"], data["valid_patch_mask"])
            y_dfit = np.asarray([config.VALID_STATES[dfit_states[i]["state"]]
                                 for i in dfit_ids if config.VALID_STATES[dfit_states[i]["state"]] is not None], dtype=np.int8)
            binary_dfit_ids = [i for i in dfit_ids if config.VALID_STATES[dfit_states[i]["state"]] is not None]

            initial_global: dict[str, np.ndarray] = {}
            global_contexts: dict[str, dict] = {}
            for readout in config.GLOBAL_READOUTS:
                context = {"backbone": backend, "method": readout, "seed": ""}
                matrix = np.stack([global_features[readout][i] for i in binary_dfit_ids])
                fit = fit_global(matrix, y_dfit, None, config.LAMBDA, fixed_solver_options())
                record = _fit_record(context, fit, "D-fit_initialization", None, len(y_dfit))
                all_solver_rows.append(record)
                if not fit["accepted"]:
                    global_contexts[readout] = {"status": fit["status"], "fit": fit}
                    trajectory_status_rows.extend({
                        "protocol": protocol, "backbone": backend, "method": readout,
                        "seed": "", "status": "INITIALIZATION_FAILED", "last_checkpoint": "",
                    } for protocol in ("X", "W"))
                    continue
                initial_global[readout] = fit["theta"].copy()
                global_contexts[readout] = {"status": "PASS", "fit": fit}
                state_arrays[f"{backend}__{readout}__initial_theta"] = fit["theta"].copy()

            initial_mil: dict[tuple[str, int], dict[str, Any]] = {}
            profile_arrays = np.load(profile_npz_path, allow_pickle=False)
            profiled_initial = np.asarray(profile_arrays["initial_theta"], dtype=np.float64)
            profiled_fitted = np.asarray(profile_arrays["fitted_theta"], dtype=np.float64)
            profiled_hash = profile["profile_initialization_sha256"]
            if sha256_file(profile_npz_path) != profiled_hash:
                raise RuntimeError("profile initialization artifact checksum mismatch")
            for seed in config.MIL_SEEDS:
                start = mil_initial_parameters(mil_bags[binary_dfit_ids[0]].shape[1], seed)
                initial_hash = parameter_digest(start)
                for method in config.MIL_READOUTS:
                    context = {"backbone": backend, "method": method, "seed": seed}
                    if backend == config.PROFILE_BACKBONE and method == config.PROFILE_METHOD and seed == config.PROFILE_SEED:
                        if parameter_digest(start) != parameter_digest(profiled_initial):
                            raise RuntimeError("profile and main M-BCE seed-0 initial parameters differ")
                        fit = {"theta": profiled_fitted.copy(), "accepted": True,
                               "status": "PASS_PROFILE_REUSED", "solver_success": True,
                               "solver_status": 0, "message": "bounded D-fit profile reused",
                               "iterations": profile["solver"]["iterations"],
                               "function_evaluations": profile["solver"]["function_evaluations"],
                               "objective": profile["solver"]["objective"],
                               "gradient_inf_norm": profile["solver"]["gradient_inf_norm"],
                               "finite": profile["solver"]["finite"],
                               "elapsed_seconds": 0.0}
                        stage = "D-fit_initialization_reused_profile"
                    else:
                        fit = fit_mil(
                            [mil_bags[i] for i in binary_dfit_ids], y_dfit, start,
                            config.LAMBDA, rho=config.RHO if method == "M-Rank-Softplus" else 0.0,
                            tau=config.TAU, margin=config.MARGIN, kappa=config.KAPPA,
                            options=fixed_solver_options(),
                        )
                        stage = "D-fit_initialization"
                    detail = mil_objective_parts(
                        fit["theta"], [mil_bags[i] for i in binary_dfit_ids], y_dfit,
                        config.LAMBDA,
                        rho=config.RHO if method == "M-Rank-Softplus" else 0.0,
                        tau=config.TAU, margin=config.MARGIN, kappa=config.KAPPA,
                    ) if fit["finite"] else None
                    record = _fit_record(context, fit, stage, None, len(y_dfit),
                                         None if detail is None else detail.pair_count)
                    record["paired_start_parameter_sha256"] = initial_hash
                    if detail is not None:
                        record.update({"bce_loss": detail.bce, "l2_penalty": detail.l2,
                                       "rank_loss": detail.rank, "total_objective_recomputed": detail.total})
                    all_solver_rows.append(record)
                    if not fit["accepted"]:
                        initial_mil[(method, seed)] = {"status": fit["status"], "fit": fit,
                                                       "initial_sha": initial_hash}
                        for protocol in ("X", "W"):
                            trajectory_status_rows.append({"protocol": protocol, "backbone": backend,
                                                           "method": method, "seed": seed,
                                                           "status": "INITIALIZATION_FAILED",
                                                           "last_checkpoint": ""})
                    else:
                        initial_mil[(method, seed)] = {"status": "PASS", "fit": fit,
                                                       "initial_sha": initial_hash}
                        state_arrays[f"{backend}__{method}__seed{seed}__initial_theta"] = fit["theta"].copy()

            for method in config.GLOBAL_READOUTS:
                if global_contexts[method]["status"] != "PASS":
                    continue
                initial_theta = initial_global[method]
                for protocol in ("X", "W"):
                    events_raw, future_rows = protocol_data[protocol]
                    future_features = [global_features[method][r["image_id"]] for r in future_rows]
                    events = [{**e, "features": global_features[method][e["image_id"]]} for e in events_raw]
                    state_map = state_by_id
                    evaluator = lambda scores, fr=future_rows, sm=state_map: summarize(scores, fr, sm)
                    reveal = lambda event_id, er=events: state_map[next(e["image_id"] for e in er if e["event_id"] == event_id)]["state"]
                    x_fit = [global_features[method][i] for i in binary_dfit_ids]

                    def make_global_fit(context, protocol_name):
                        def callback(features, labels, theta, checkpoint):
                            matrix = np.stack(features)
                            fit_result = fit_global(matrix, labels, theta, config.LAMBDA, fixed_solver_options())
                            all_solver_rows.append(_fit_record(context, fit_result, "online_full_history",
                                                               checkpoint, len(labels)))
                            return fit_result
                        return callback

                    fit_callback = make_global_fit({"backbone": backend, "method": method,
                                                    "seed": "", "protocol": protocol}, protocol)
                    for update_enabled, label in ((False, f"B0-{method}"), (True, method)):
                        trajectory_id = f"{backend}__{method}__{protocol}__{label}"
                        trajectory = run_trajectory(
                            protocol=protocol, trajectory_id=trajectory_id,
                            initial_theta=initial_theta,
                            initial_features=x_fit,
                            initial_labels=y_dfit,
                            initial_image_ids=binary_dfit_ids,
                            feedback_events=events, reveal_state=reveal,
                            fit_history=fit_callback,
                            predict_one=lambda t, x: float(predict_global(t, np.asarray(x)[None, :])[0]),
                            predict_future=lambda t, xs: predict_global(t, np.stack(xs)),
                            future_features=future_features, future_rows=future_rows,
                            evaluate=evaluator, update_enabled=update_enabled,
                        )
                        control_name = "" if not update_enabled else f"B0-{method}"
                        flattened = flatten_trajectory(trajectory, backend=backend, method=label,
                                                       seed=None, control_method=control_name,
                                                       future_rows=future_rows, state_by_id=state_by_id)
                        all_checkpoint_rows.extend(flattened[0]); all_product_rows.extend(flattened[1])
                        all_prediction_rows.extend(flattened[2]); all_event_rows.extend(flattened[3])
                        all_event_effect_rows.extend(flattened[4])
                        trajectory_status_rows.append({"protocol": protocol, "backbone": backend,
                                                       "method": label, "seed": "",
                                                       "status": trajectory["status"],
                                                       "last_checkpoint": max(c["checkpoint"] for c in trajectory["checkpoints"])})
                        for cp in trajectory["checkpoints"]:
                            state_arrays[f"{trajectory_id}__cp{cp['checkpoint']:02d}__theta"] = cp["theta"].copy()

            for method in config.MIL_READOUTS:
                rho = config.RHO if method == "M-Rank-Softplus" else 0.0
                for seed in config.MIL_SEEDS:
                    init = initial_mil[(method, seed)]
                    if init["status"] != "PASS":
                        continue
                    initial_theta = init["fit"]["theta"]
                    initial_features = [mil_bags[i] for i in binary_dfit_ids]
                    for protocol in ("X", "W"):
                        events_raw, future_rows = protocol_data[protocol]
                        future_features = [mil_bags[r["image_id"]] for r in future_rows]
                        events = [{**e, "features": mil_bags[e["image_id"]]} for e in events_raw]
                        evaluator = lambda scores, fr=future_rows, sm=state_by_id: summarize(scores, fr, sm)
                        reveal = lambda event_id, er=events: state_by_id[next(e["image_id"] for e in er if e["event_id"] == event_id)]["state"]

                        def make_mil_fit(context, method_name, protocol_name):
                            def callback(features, labels, theta, checkpoint):
                                result = fit_mil(
                                    features, labels, theta, config.LAMBDA, rho=rho,
                                    tau=config.TAU, margin=config.MARGIN, kappa=config.KAPPA,
                                    options=fixed_solver_options(),
                                )
                                details = mil_objective_parts(
                                    result["theta"], features, labels, config.LAMBDA,
                                    rho=rho, tau=config.TAU, margin=config.MARGIN, kappa=config.KAPPA,
                                ) if result["finite"] else None
                                row = _fit_record(context, result, "online_full_history",
                                                  checkpoint, len(labels),
                                                  None if details is None else details.pair_count)
                                if details is not None:
                                    row.update({"bce_loss": details.bce, "l2_penalty": details.l2,
                                                "rank_loss": details.rank,
                                                "total_objective_recomputed": details.total})
                                all_solver_rows.append(row)
                                return result
                            return callback

                        fit_callback = make_mil_fit({"backbone": backend, "method": method,
                                                    "seed": seed, "protocol": protocol}, method, protocol)
                        for update_enabled, label in ((False, f"B0-{method}"), (True, method)):
                            trajectory_id = f"{backend}__{method}__seed{seed}__{protocol}__{label}"
                            trajectory = run_trajectory(
                                protocol=protocol, trajectory_id=trajectory_id,
                                initial_theta=initial_theta,
                                initial_features=initial_features,
                                initial_labels=y_dfit,
                                initial_image_ids=binary_dfit_ids,
                                feedback_events=events, reveal_state=reveal,
                                fit_history=fit_callback,
                                predict_one=lambda t, bag: float(predict_mil(t, [bag], config.TAU)[0]),
                                predict_future=lambda t, bags: predict_mil(t, bags, config.TAU),
                                future_features=future_features, future_rows=future_rows,
                                evaluate=evaluator, update_enabled=update_enabled,
                            )
                            control_name = "" if not update_enabled else f"B0-{method}"
                            flattened = flatten_trajectory(trajectory, backend=backend, method=label,
                                                           seed=seed, control_method=control_name,
                                                           future_rows=future_rows, state_by_id=state_by_id)
                            all_checkpoint_rows.extend(flattened[0]); all_product_rows.extend(flattened[1])
                            all_prediction_rows.extend(flattened[2]); all_event_rows.extend(flattened[3])
                            all_event_effect_rows.extend(flattened[4])
                            trajectory_status_rows.append({"protocol": protocol, "backbone": backend,
                                                           "method": label, "seed": seed,
                                                           "status": trajectory["status"],
                                                           "last_checkpoint": max(c["checkpoint"] for c in trajectory["checkpoints"])})
                            for cp in trajectory["checkpoints"]:
                                state_arrays[f"{trajectory_id}__cp{cp['checkpoint']:02d}__theta"] = cp["theta"].copy()
            del bundles, global_features, mil_bags
            rss_peak = max(rss_peak, sampler.peak)

    main_wall = time.perf_counter() - main_start
    main_cpu = time.process_time() - cpu_start
    rss_peak = max(rss_peak, sampler.peak)

    # Evaluator-only source/type diagnostic; run after every model trajectory is complete.
    all_dev_labels = first_state_by_item(*config.ANNOTATION_SOURCES["D-dev"])
    all_labels = {**dfit_states, **all_dev_labels}
    all_id_order = [r["image_id"] for r in cache["registered_rows"]]
    source_rows = []
    for image_id in all_id_order:
        source_rows.append({"image_id": image_id,
                            "data_role": role_by_id[image_id],
                            "source_type_evaluator_only": source_types.get(image_id, "UNKNOWN"),
                            "attribute_state_evaluator_only": all_labels[image_id]["state"]})
    crosstab = Counter((r["data_role"], r["source_type_evaluator_only"],
                        r["attribute_state_evaluator_only"]) for r in source_rows)
    crosstab_rows = [{"data_role": role, "source_type_evaluator_only": source_type,
                      "attribute_state_evaluator_only": state, "count": count}
                     for (role, source_type, state), count in sorted(crosstab.items())]

    # Ensure no input, annotation or protected asset changed while the experiment ran.
    input_hashes_after = annotation_input_hashes()
    protected_after = {rel: stable_tree_identity(ROOT / rel) for rel in config.PROTECTED_PATHS}
    code_after = code_hashes()
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("E2-B changed a registered annotation, tape, or manifest input")
    if protected_after != protected_before:
        raise RuntimeError("E2-B changed a protected P1/E1/E2-A asset")
    if code_after != code_before:
        raise RuntimeError("E2-B code changed during execution")

    # Flatten all requested artifacts without exposing rows in version-controlled reports.
    write_csv(config.RUN_DIR / "checkpoint_metrics.csv", all_checkpoint_rows)
    write_csv(config.RUN_DIR / "product_metrics.csv", all_product_rows)
    write_csv(config.RUN_DIR / "future_predictions.csv", all_prediction_rows)
    write_csv(config.RUN_DIR / "feedback_event_log.csv", all_event_rows)
    write_csv(config.RUN_DIR / "event_effects.csv", all_event_effect_rows)
    write_csv(config.RUN_DIR / "solver_log.csv", all_solver_rows)
    write_csv(config.RUN_DIR / "trajectory_status.csv", trajectory_status_rows)
    write_csv(config.RUN_DIR / "protocol_support.csv", support_rows)
    write_csv(config.RUN_DIR / "source_attribute_crosstab.csv", crosstab_rows)
    np.savez_compressed(config.RUN_DIR / "model_states.npz", **state_arrays)

    learner_statuses = [r for r in trajectory_status_rows if not str(r["method"]).startswith("B0-")]
    failed_trajectories = [r for r in learner_statuses if r["status"] != "PASS"]
    solver_failures = [r for r in all_solver_rows if r["status"] not in {"PASS", "PASS_PROFILE_REUSED"}]
    status = "E2B_PASS" if not failed_trajectories and not solver_failures else "E2B_PARTIAL"
    final_provenance = {
        "schema": "lp.d6.p2.e2b.run.v1",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_head": config.EXPECTED_HEAD,
        "branch": config.EXPECTED_BRANCH,
        "mode": "CPU-only; frozen features; no DINO load or E2-A1 cache writes",
        "cache": cache["manifest_hashes"],
        "input_manifest_sha256": cache["input_manifest_sha256"],
        "p1a_feature_manifest_sha256": cache["p1a_feature_manifest_sha256"],
        "wx_tape_sha256": config.EXPECTED["wx_tape_sha256"],
        "p1b_evaluator_source_metadata_sha256": config.EXPECTED["p1b_joined_source_metadata_sha256"],
        "annotation_input_sha256": input_hashes_before,
        "dfit_summary": dfit_summary,
        "tape_label_summary": dev_summary,
        "protocol_metadata": tape_summary,
        "config": {
            "backbones": config.BACKBONES, "global_readouts": config.GLOBAL_READOUTS,
            "mil_readouts": config.MIL_READOUTS, "mil_seeds": config.MIL_SEEDS,
            "lambda": config.LAMBDA, "tau": config.TAU, "margin": config.MARGIN,
            "kappa": config.KAPPA, "rho": config.RHO,
            "mil_init": config.MIL_INIT_WEIGHT_STD, "mil_bias": config.MIL_INIT_BIAS,
            "solver": config.SOLVER, "thread_limit": config.THREAD_LIMIT,
            "predict_threshold": 0.0,
            "update_schedule": "predict query -> save state/score -> reveal -> append binary only -> full-history refit -> evaluator-only future score",
        },
        "trajectory_count_expected": 28,
        "learner_trajectories_completed": len(learner_statuses) - len(failed_trajectories),
        "learner_trajectories_failed_or_partial": len(failed_trajectories),
        "checkpoint_count": len(all_checkpoint_rows),
        "solver_fit_count": len(all_solver_rows),
        "solver_failures": solver_failures,
        "failed_trajectories": failed_trajectories,
        "elapsed_wall_seconds": main_wall,
        "elapsed_process_cpu_seconds": main_cpu,
        "sampled_peak_rss_bytes": rss_peak,
        "protected_assets_unchanged": True,
        "input_hashes_unchanged": True,
        "code_hashes": code_after,
        "code_fingerprint": code_fingerprint(),
        "profile_code_fingerprint": profile["code_fingerprint"],
        "source_metadata_used_by_learner": False,
        "future_labels_used_by_learner": False,
        "source_labels_as_attribute_targets": False,
        "model_training_or_feature_extraction": "attribute learners only; no backbone fine-tuning or extraction",
    }
    json_dump(config.RUN_DIR / "run_provenance.json", json_safe(final_provenance))
    json_dump(config.RUN_DIR / "protected_assets_before.json", protected_before)
    json_dump(config.RUN_DIR / "protected_assets_after.json", protected_after)
    json_dump(config.RUN_DIR / "input_hashes_before.json", input_hashes_before)
    json_dump(config.RUN_DIR / "input_hashes_after.json", input_hashes_after)
    output_bytes = sum(p.stat().st_size for p in config.RUN_DIR.rglob("*") if p.is_file())
    final_result = {
        "status": status,
        "expected_trajectory_count": 28,
        "completed_learner_trajectories": len(learner_statuses) - len(failed_trajectories),
        "failed_or_partial_learner_trajectories": len(failed_trajectories),
        "solver_fit_count": len(all_solver_rows),
        "solver_failures": solver_failures,
        "protected_assets_unchanged": True,
        "input_hashes_unchanged": True,
        "elapsed_wall_seconds": main_wall,
        "elapsed_process_cpu_seconds": main_cpu,
        "sampled_peak_rss_bytes": rss_peak,
        "output_bytes": output_bytes,
    }
    json_dump(config.RUN_DIR / "task_result.json", final_result)
    print(f"E2B_STATUS={status}")
    print(f"TRAJECTORIES={len(learner_statuses)} COMPLETED={final_result['completed_learner_trajectories']} FAILED={len(failed_trajectories)}")
    print(f"SOLVER_FITS={len(all_solver_rows)} FAILURES={len(solver_failures)}")
    print(f"WALL_SECONDS={main_wall:.2f} CPU_SECONDS={main_cpu:.2f} PEAK_RSS={rss_peak} OUTPUT_BYTES={output_bytes}")
    return 0 if status == "E2B_PASS" else 2


def cli() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile-only", action="store_true")
    group.add_argument("--run-main", action="store_true")
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=config.THREAD_LIMIT):
            return profile_only() if args.profile_only else main_run()
    except Exception as exc:
        print(f"E2B_BLOCKED={type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(cli())
