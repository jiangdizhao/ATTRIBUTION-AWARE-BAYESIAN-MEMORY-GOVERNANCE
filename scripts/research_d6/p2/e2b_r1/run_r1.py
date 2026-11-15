"""Supervisor and resumable executor for the authorized LP-D6-P2-E2-B-R1 matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# This task is explicitly CPU-only; establish the child-process runtime before NumPy loads.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "2"

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
E2B_DIR = ROOT / "scripts/research_d6/p2/e2b"
if str(E2B_DIR) not in sys.path:
    sys.path.insert(0, str(E2B_DIR))

import config as base  # noqa: E402
import data as base_data  # noqa: E402
import evaluator as base_evaluator  # noqa: E402
import model as base_model  # noqa: E402
from r1_core import (  # noqa: E402
    ExecutionStop, IntegrityError, canonical_bytes, event_record, jsonable, read_json,
    run_checkpointed_trajectory, sha256_bytes, sha256_file, verify_bundle,
    write_bundle, write_event_stage, write_json_atomic, write_json_exclusive,
)

EXPECTED_HEAD = "072e11adaffd85cf64b85b4708befe45cef92018"
EXPECTED_BRANCH = "stage3-eg-famar-execution"
EXPECTED_R0_HEAD = "1e6a602cb8b788bca03a61773ef442a55f3fa7e2"
FIXED_PYTHON = Path(r"D:\anaconda3\envs\torch\python.exe")
OUTPUT_ROOT = ROOT / "outputs/research_d6/p2/e2b/r1_run_20261116_01"
PROFILE_JSON = base.OUTPUT_ROOT / "bounded_profile.json"
PROFILE_NPZ = base.OUTPUT_ROOT / "profile_initialization.npz"
MAX_WALL_SECONDS = 7200.0
MAX_CPU_SECONDS = 7200.0
RSS_SOFT_LIMIT = 6 * 1024**3
THREAD_LIMIT = 2

FROZEN_FILES = [
    "scripts/research_d6/p2/e2b/config.py",
    "scripts/research_d6/p2/e2b/model.py",
    "scripts/research_d6/p2/e2b/data.py",
    "scripts/research_d6/p2/e2b/evaluator.py",
    "scripts/research_d6/p2/e2b/pipeline.py",
    "scripts/research_d6/p2/e2b/run_e2b.py",
    "tests/research_d6/test_e2b_numerics.py",
]
R0_PROFILE_CODE_FILES = FROZEN_FILES[:6] + [FROZEN_FILES[-1]]
PROTECTED_PATHS = tuple(base.PROTECTED_PATHS) + (
    "outputs/research_d6/p2/e2b/bounded_profile.json",
    "outputs/research_d6/p2/e2b/profile_initialization.npz",
)


class RSSSampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak = psutil.Process().memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, process.memory_info().rss)
            except psutil.Error:
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.peak = max(self.peak, psutil.Process().memory_info().rss)
        except psutil.Error:
            pass


class Budget:
    def __init__(self, wall_seconds: float, cpu_seconds: float, sampler: RSSSampler):
        self.wall_seconds = float(wall_seconds)
        self.cpu_seconds = float(cpu_seconds)
        self.sampler = sampler
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()

    @property
    def wall_used(self) -> float:
        return time.perf_counter() - self.start_wall

    @property
    def cpu_used(self) -> float:
        return time.process_time() - self.start_cpu

    def check(self, stage: str):
        if self.sampler.peak >= RSS_SOFT_LIMIT:
            raise ExecutionStop(f"RSS_SOFT_LIMIT at {stage}: {self.sampler.peak} bytes")
        if self.wall_used >= self.wall_seconds:
            raise ExecutionStop(f"WALL_TIME_LIMIT at {stage}: {self.wall_used:.3f}s")
        if self.cpu_used >= self.cpu_seconds:
            raise ExecutionStop(f"PROCESS_CPU_LIMIT at {stage}: {self.cpu_used:.3f}s")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def current_branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def check_start_identity():
    actual = (current_branch(), current_head())
    expected = (EXPECTED_BRANCH, EXPECTED_HEAD)
    if actual != expected:
        raise IntegrityError(f"R1 start identity mismatch: got {actual}, expected {expected}")
    if Path(sys.executable).resolve() != FIXED_PYTHON.resolve():
        raise IntegrityError(f"R1 must use {FIXED_PYTHON}; got {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise IntegrityError("CUDA must be disabled for R1")


def check_frozen_files_against_head() -> dict[str, str]:
    result = {}
    for rel in FROZEN_FILES:
        local = (ROOT / rel).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{EXPECTED_HEAD}:{rel}"], cwd=ROOT)
        local_sha = hashlib.sha256(local).hexdigest()
        committed_sha = hashlib.sha256(committed).hexdigest()
        if local_sha != committed_sha:
            raise IntegrityError(f"frozen R0 file differs from expected HEAD: {rel}")
        result[rel] = local_sha
    return result


def ignored_path(relative: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", relative], cwd=ROOT).returncode == 0


def stable_tree_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0, "tree_sha256": ""}
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    )
    base_path = path.parent if path.is_file() else path
    digest = hashlib.sha256()
    total = 0
    for item in files:
        digest.update(item.relative_to(base_path).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_file(item).encode("ascii") + b"\n")
        total += item.stat().st_size
    return {"exists": True, "files": len(files), "bytes": total,
            "tree_sha256": digest.hexdigest()}


def frozen_code_hashes() -> dict[str, str]:
    return {rel: sha256_file(ROOT / rel) for rel in FROZEN_FILES}


def r1_code_hashes() -> dict[str, str]:
    source_paths = sorted(p for p in (ROOT / "scripts/research_d6/p2/e2b_r1").glob("*.py"))
    test_path = ROOT / "tests/research_d6/test_e2b_r1.py"
    if test_path.exists():
        source_paths.append(test_path)
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in source_paths}


def code_identity() -> dict[str, Any]:
    frozen = frozen_code_hashes()
    r1 = r1_code_hashes()
    digest = sha256_bytes(canonical_bytes({"frozen_r0": frozen, "r1": r1}))
    return {"frozen_r0": frozen, "r1": r1, "code_fingerprint": digest}


def r0_profile_code_hashes() -> dict[str, str]:
    return {rel: sha256_file(ROOT / rel) for rel in R0_PROFILE_CODE_FILES}


def ensure_output_ignored():
    relative = OUTPUT_ROOT.relative_to(ROOT).as_posix() + "/r1_ignore_probe.txt"
    if not ignored_path(relative):
        raise IntegrityError(f"R1 output path must be Git-ignored: {relative}")


def run_synthetic_tests() -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(THREAD_LIMIT)
    command = [sys.executable, "-m", "unittest",
               "tests.research_d6.test_e2b_numerics", "tests.research_d6.test_e2b_r1", "-v"]
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=300)
    transcript = proc.stdout + proc.stderr
    if proc.returncode != 0 or "Ran 22 tests" not in transcript or "OK" not in transcript:
        raise RuntimeError(f"R1 synthetic gate failed:\n{transcript}")
    return {"return_code": proc.returncode, "test_count": 22,
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            "transcript": transcript}


def profile_hashes() -> dict[str, str]:
    if not PROFILE_JSON.is_file() or not PROFILE_NPZ.is_file():
        raise IntegrityError("R0 profile receipt or saved initialization is missing")
    return {"bounded_profile_json_sha256": sha256_file(PROFILE_JSON),
            "profile_initialization_npz_sha256": sha256_file(PROFILE_NPZ)}


def load_dfit_only(cache: dict[str, Any]):
    dfit_ids = [r["image_id"] for r in cache["registered_rows"] if r["data_role"] == "D-fit"]
    dfit_states, summary = base_data.load_dfit_labels(dfit_ids)
    binary_ids = [image_id for image_id in dfit_ids
                  if base.VALID_STATES[dfit_states[image_id]["state"]] is not None]
    labels = np.asarray([base.VALID_STATES[dfit_states[i]["state"]] for i in binary_ids], dtype=np.int8)
    return dfit_ids, dfit_states, binary_ids, labels, summary


def validate_old_profile(cache: dict[str, Any], dfit_ids: list[str],
                         binary_ids: list[str], y_dfit: np.ndarray) -> dict[str, Any]:
    hashes = profile_hashes()
    receipt = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    if receipt.get("status") != "BLOCKED" or receipt.get("resource_gate") != "OVER_BUDGET":
        raise IntegrityError("R0 profile status is not the preserved BLOCKED/OVER_BUDGET outcome")
    if receipt.get("execution_head") != EXPECTED_R0_HEAD or receipt.get("branch") != EXPECTED_BRANCH:
        raise IntegrityError("R0 profile provenance does not match its original run")
    if receipt.get("future_labels_read") is not False or "no future labels" not in receipt.get("mode", ""):
        raise IntegrityError("R0 profile does not certify the no-future-label boundary")
    if receipt.get("code_hashes") != r0_profile_code_hashes():
        raise IntegrityError("R0 profile code hashes do not match committed E2-B scientific code")
    if receipt.get("cache_manifest_sha256") != {
            "canonical": base.EXPECTED["canonical_manifest_sha256"],
            "dinov2": base.EXPECTED["dinov2_feature_manifest_sha256"],
            "dinov3": base.EXPECTED["dinov3_feature_manifest_sha256"]}:
        raise IntegrityError("R0 profile cache identity does not match the frozen E2-A1 cache")
    if (receipt.get("input_manifest_sha256") != base.EXPECTED["p1a_input_manifest_sha256"]
            or receipt.get("feature_manifest_p1a_sha256") != base.EXPECTED["p1a_feature_manifest_sha256"]
            or receipt.get("wx_tape_sha256") != base.EXPECTED["wx_tape_sha256"]):
        raise IntegrityError("R0 profile input identity mismatch")
    solver = receipt.get("solver", {})
    if (solver.get("accepted") is not True or solver.get("solver_success") is not True
            or solver.get("finite") is not True
            or float(solver.get("gradient_inf_norm", float("inf"))) > base.SOLVER["accept_gradient_inf"]):
        raise IntegrityError("R0 profile solver receipt failed the frozen numerical acceptance gate")
    if receipt.get("dfit_label_summary") != {
            "total": 54, "states": {"present": 27, "absent_assessable": 25,
                                      "unobservable_or_uncertain": 2}, "binary_valid": 52}:
        raise IntegrityError("R0 profile D-fit label support differs from the registered data")
    if receipt.get("dfit_binary_bags") != 52 or len(dfit_ids) != 54 or len(binary_ids) != 52:
        raise IntegrityError("R0 profile training data count mismatch")

    with np.load(PROFILE_NPZ, allow_pickle=False) as stored:
        required = {"initial_theta", "fitted_theta"}
        if set(stored.files) != required:
            raise IntegrityError("R0 profile initialization archive fields changed")
        initial = np.asarray(stored["initial_theta"], dtype=np.float64).copy()
        fitted = np.asarray(stored["fitted_theta"], dtype=np.float64).copy()
    if hashes["profile_initialization_npz_sha256"] != receipt.get("profile_initialization_sha256"):
        raise IntegrityError("R0 profile initialization SHA mismatch")
    if not np.isfinite(initial).all() or not np.isfinite(fitted).all():
        raise IntegrityError("R0 profile parameters are nonfinite")

    if receipt["configuration"].get("backbone") != "dinov2" or receipt["configuration"].get("method") != "M-BCE" or receipt["configuration"].get("seed") != 0:
        raise IntegrityError("R0 profile configuration is not the registered DINOv2 M-BCE seed-0 fit")
    features = base_data.load_feature_bundle("dinov2", binary_ids)
    bags = [base_model.valid_bag(features[i]["patch_unit"], features[i]["valid_patch_mask"])
            for i in binary_ids]
    expected_initial = base_model.mil_initial_parameters(bags[0].shape[1], 0)
    if not np.array_equal(initial, expected_initial):
        raise IntegrityError("R0 profile seed-0 initialization differs from the registered RNG")
    if fitted.shape != expected_initial.shape:
        raise IntegrityError("R0 profile fitted parameter dimensionality mismatch")
    recomputed = base_model.mil_objective_parts(
        fitted, bags, y_dfit, base.LAMBDA, rho=0.0, tau=base.TAU,
        margin=base.MARGIN, kappa=base.KAPPA,
    )
    objective_delta = abs(recomputed.total - float(solver["objective"]))
    gradient_delta = abs(float(np.max(np.abs(recomputed.gradient))) - float(solver["gradient_inf_norm"]))
    if objective_delta > 1e-12 or gradient_delta > max(1e-10, 1e-7 * abs(float(solver["gradient_inf_norm"]))):
        raise IntegrityError(f"R0 profile objective/gradient recomputation mismatch: {objective_delta}, {gradient_delta}")
    return {
        "profile_hashes": hashes, "profile_status_preserved": receipt["status"],
        "resource_gate_preserved": receipt["resource_gate"],
        "r0_code_hashes_verified": receipt["code_hashes"],
        "recomputed_objective": recomputed.total,
        "stored_objective": float(solver["objective"]),
        "objective_absolute_delta": objective_delta,
        "recomputed_gradient_inf": float(np.max(np.abs(recomputed.gradient))),
        "stored_gradient_inf": float(solver["gradient_inf_norm"]),
        "gradient_inf_absolute_delta": gradient_delta,
        "initial_theta_sha256": base_model.parameter_digest(initial),
        "fitted_theta_sha256": base_model.parameter_digest(fitted),
        "solver_record": solver,
        "profile_initial_theta": initial, "profile_fitted_theta": fitted,
        "dfit_binary_ids": binary_ids,
    }


def protected_snapshot() -> dict[str, Any]:
    return {rel: stable_tree_identity(ROOT / rel) for rel in PROTECTED_PATHS}


def input_hash_snapshot() -> dict[str, str]:
    values = {
        "tape": sha256_file(base.TAPE_PATH),
        "p1a_input_manifest": sha256_file(base.P1A_INPUT_MANIFEST),
        "p1a_feature_manifest": sha256_file(base.P1A_FEATURE_MANIFEST),
        "p1b_source_metadata": sha256_file(base.P1B_JOINED_METADATA),
    }
    for source, (queue_path, annotation_path) in base.ANNOTATION_SOURCES.items():
        values[f"{source}_queue"] = sha256_file(queue_path)
        values[f"{source}_annotations"] = sha256_file(annotation_path)
    return values


def _record_self_sha(record: dict[str, Any]) -> dict[str, Any]:
    value = jsonable(dict(record))
    value["record_sha256"] = sha256_bytes(canonical_bytes(value))
    return value


def verify_self_sha(record: dict[str, Any], where: str):
    expected = sha256_bytes(canonical_bytes({k: v for k, v in record.items() if k != "record_sha256"}))
    if record.get("record_sha256") != expected:
        raise IntegrityError(f"record SHA mismatch: {where}")


def immutable_receipt(path: Path, record: dict[str, Any], *, identity_key: str = "identity"):
    value = _record_self_sha(record)
    if path.exists():
        existing = read_json(path)
        verify_self_sha(existing, str(path))
        if existing.get(identity_key) != value.get(identity_key):
            raise IntegrityError(f"immutable receipt identity mismatch: {path}")
        if existing != value:
            raise IntegrityError(f"immutable receipt differs from requested run: {path}")
        return existing
    write_json_exclusive(path, value)
    return value


def validate_existing_run(code: dict[str, Any], profile_identity: dict[str, Any],
                          input_hashes: dict[str, str]):
    if OUTPUT_ROOT.exists() and not (OUTPUT_ROOT / "run_identity.json").exists():
        raise IntegrityError(f"R1 output exists without an identity receipt: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "lp.d6.p2.e2b.r1.identity.v1",
        "branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD,
        "code_fingerprint": code["code_fingerprint"],
        "frozen_code_hashes": code["frozen_r0"], "r1_code_hashes": code["r1"],
        "feature_manifest_sha256": base.EXPECTED["dinov2_feature_manifest_sha256"],
        "dinov3_feature_manifest_sha256": base.EXPECTED["dinov3_feature_manifest_sha256"],
        "canonical_manifest_sha256": base.EXPECTED["canonical_manifest_sha256"],
        "input_manifest_sha256": base.EXPECTED["p1a_input_manifest_sha256"],
        "p1a_feature_manifest_sha256": base.EXPECTED["p1a_feature_manifest_sha256"],
        "tape_sha256": base.EXPECTED["wx_tape_sha256"],
        "profile_hashes": profile_identity["profile_hashes"],
        "input_hashes": input_hashes,
        "profile_initial_theta_sha256": profile_identity["initial_theta_sha256"],
        "profile_fitted_theta_sha256": profile_identity["fitted_theta_sha256"],
        "solver": base.SOLVER, "lambda": base.LAMBDA, "tau": base.TAU,
        "margin": base.MARGIN, "kappa": base.KAPPA, "rho": base.RHO,
        "backbones": list(base.BACKBONES), "global_readouts": list(base.GLOBAL_READOUTS),
        "mil_readouts": list(base.MIL_READOUTS), "mil_seeds": list(base.MIL_SEEDS),
    }
    identity_path = OUTPUT_ROOT / "run_identity.json"
    if identity_path.exists():
        old = read_json(identity_path)
        verify_self_sha(old, str(identity_path))
        if old != _record_self_sha(identity):
            raise IntegrityError("existing R1 output identity does not match this source/profile")
    else:
        immutable_receipt(identity_path, identity)

    override = {
        "schema": "lp.d6.p2.e2b.r1.resource-override.v1",
        "identity": {"head": EXPECTED_HEAD, "code_fingerprint": code["code_fingerprint"]},
        "previous_status": "BLOCKED / OVER_BUDGET",
        "authorization": "E2B_R1_RESOURCE_OVERRIDE_APPROVED",
        "previous_cpu_planning_window_seconds": 1800,
        "new_cpu_time_limit_seconds": MAX_CPU_SECONDS,
        "new_wall_time_limit_seconds": MAX_WALL_SECONDS,
        "previous_projection_process_cpu_seconds": 3979.5,
        "registered_trajectory_count": 28,
        "matched_b0_control_count": 28,
        "loss_optimizer_data_evaluation_unchanged": True,
        "profile_hashes": profile_identity["profile_hashes"],
        "created_utc": "2026-11-16",
    }
    override_path = OUTPUT_ROOT / "resource_override_receipt.json"
    if override_path.exists():
        old = read_json(override_path)
        verify_self_sha(old, str(override_path))
        if {k: v for k, v in old.items() if k not in {"record_sha256", "created_utc"}} != {
                **{k: v for k, v in override.items() if k != "created_utc"}}:
            raise IntegrityError("resource override receipt changed")
    else:
        immutable_receipt(override_path, override)
    return identity


def ensure_initialization(path: Path, identity: dict[str, Any], *, initial: np.ndarray,
                          fit_callable, budget: Budget, profile_reuse: dict[str, Any] | None = None,
                          details_callable=None) -> dict[str, Any]:
    if path.exists():
        record = verify_bundle(path, identity)
        arrays = record.pop("_arrays")
        record["initial_theta"] = arrays["initial_theta"]
        record["theta"] = arrays["theta"]
        return record

    budget.check(f"before_initialization_{path.name}")
    if profile_reuse is not None:
        theta = np.asarray(profile_reuse["profile_fitted_theta"], dtype=np.float64).copy()
        fit_record = dict(profile_reuse["solver_record"])
        fit_record.update({"accepted": True, "status": "PASS_R0_PROFILE_REUSED",
                           "solver_success": True, "finite": True,
                           "elapsed_seconds": 0.0, "reused_profile": True})
        profile_used = profile_reuse["profile_hashes"]
    else:
        fit_record = fit_callable()
        theta = np.asarray(fit_record["theta"], dtype=np.float64).copy()
        profile_used = None
    details = details_callable(theta) if details_callable is not None and fit_record.get("finite", True) else None
    metadata = {
        "schema": "lp.d6.p2.e2b.r1.initialization.v1", "identity": identity,
        "checkpoint": 0, "accepted": bool(fit_record.get("accepted", False)),
        "fit_record": {k: jsonable(v) for k, v in fit_record.items() if k != "theta"},
        "initial_theta_sha256": base_model.parameter_digest(np.asarray(initial, dtype=np.float64)),
        "theta_sha256": base_model.parameter_digest(theta),
        "training_history_rows": 52,
        "training_history_image_ids_sha256": identity["dfit_binary_ids_sha256"],
        "profile_reuse": profile_used,
        "objective_components": details,
    }
    bundle = write_bundle(path, metadata, {
        "initial_theta": np.asarray(initial, dtype=np.float64), "theta": theta,
    })
    bundle["initial_theta"] = np.asarray(initial, dtype=np.float64)
    bundle["theta"] = theta
    budget.check(f"after_initialization_{path.name}")
    return bundle


def make_initialization_identity(run_identity: dict[str, Any], backend: str,
                                 method: str, seed: int | None,
                                 dfit_binary_ids: list[str]) -> dict[str, Any]:
    return {
        "run_code_fingerprint": run_identity["code_fingerprint"],
        "head": run_identity["head"], "backend": backend, "method": method,
        "seed": "" if seed is None else int(seed),
        "dfit_image_count": 52,
        "dfit_binary_ids_sha256": sha256_bytes(canonical_bytes(dfit_binary_ids)),
        "p1a_input_manifest_sha256": base.EXPECTED["p1a_input_manifest_sha256"],
        "feature_manifest_sha256": base.EXPECTED[f"{backend}_feature_manifest_sha256"],
        "solver": base.SOLVER, "lambda": base.LAMBDA, "rho": base.RHO if method == "M-Rank-Softplus" else 0.0,
        "tau": base.TAU, "margin": base.MARGIN, "kappa": base.KAPPA,
        "init_protocol": base.MIL_INIT_WEIGHT_STD if method.startswith("M-") else "all-zero",
    }


def initialization_path(backend: str, method: str, seed: int | None) -> Path:
    seed_name = "seed_none" if seed is None else f"seed_{int(seed)}"
    method_name = method.replace("-", "_")
    return OUTPUT_ROOT / "initializations" / backend / method_name / seed_name


def fit_details_for_mil(theta: np.ndarray, bags: list[np.ndarray], y: np.ndarray,
                        method: str) -> dict[str, Any]:
    rho = base.RHO if method == "M-Rank-Softplus" else 0.0
    parts = base_model.mil_objective_parts(
        theta, bags, y, base.LAMBDA, rho=rho, tau=base.TAU,
        margin=base.MARGIN, kappa=base.KAPPA,
    )
    return {"total": parts.total, "bce": parts.bce, "l2": parts.l2,
            "rank": parts.rank, "pair_count": parts.pair_count,
            "gradient_inf_norm_recomputed": float(np.max(np.abs(parts.gradient)))}


def execute_pair(*, backend: str, method: str, seed: int | None, protocol: str,
                 initial_theta: np.ndarray, initial_features: list[Any],
                 initial_labels: np.ndarray, initial_ids: list[str],
                 events: list[dict[str, Any]], future_rows: list[dict[str, Any]],
                 future_features: list[Any], feedback_state_by_id: dict[str, dict[str, Any]],
                 future_state_by_id: dict[str, dict[str, Any]],
                 input_hashes: dict[str, str], run_identity: dict[str, Any],
                 budget: Budget, predict_one, predict_future, fit_history) -> tuple[dict, dict]:
    pair_id = f"{backend}__{method}__{('seed'+str(seed)+'__') if seed is not None else ''}{protocol}"
    feedback_by_event = {row["event_id"]: feedback_state_by_id[row["image_id"]]["state"]
                         for row in events}

    def reveal(event_id: str) -> str:
        # Only the current feedback event's state is exposed to the trajectory engine.
        return feedback_by_event[event_id]

    def evaluate(scores: np.ndarray):
        return base_evaluator.summarize(scores, future_rows, future_state_by_id)

    init_sha = base_model.parameter_digest(initial_theta)
    statuses = []
    outputs = {}
    for side, update_enabled in (("control", False), ("learner", True)):
        trajectory_id = f"{pair_id}__{side}"
        side_identity = {
            "schema": "lp.d6.p2.e2b.r1.trajectory-identity.v1",
            "run_code_fingerprint": run_identity["code_fingerprint"],
            "head": EXPECTED_HEAD, "trajectory_id": trajectory_id, "pair_id": pair_id,
            "side": side, "backbone": backend, "method": method,
            "seed": "" if seed is None else int(seed), "protocol": protocol,
            "initial_theta_sha256": init_sha,
            "feature_manifest_sha256": base.EXPECTED[f"{backend}_feature_manifest_sha256"],
            "tape_sha256": base.EXPECTED["wx_tape_sha256"],
            "input_hashes": input_hashes,
            "solver": base.SOLVER, "lambda": base.LAMBDA,
            "rho": base.RHO if method == "M-Rank-Softplus" else 0.0,
            "tau": base.TAU, "margin": base.MARGIN, "kappa": base.KAPPA,
            "prediction_threshold": 0.0,
        }
        output = OUTPUT_ROOT / "trajectories" / pair_id / side

        def fit_callback(features, labels, theta, checkpoint):
            budget.check(f"before_fit_{trajectory_id}_event_{checkpoint}")
            return fit_history(features, labels, theta, checkpoint)

        result = run_checkpointed_trajectory(
            output_dir=output, identity=side_identity,
            trajectory_id=trajectory_id, side=side, protocol=protocol,
            initial_theta=initial_theta, initial_features=initial_features,
            initial_labels=initial_labels.tolist(), initial_image_ids=initial_ids,
            feedback_events=events, reveal_state=reveal,
            fit_history=fit_callback, predict_one=predict_one,
            predict_future=predict_future, future_features=future_features,
            future_rows=future_rows, evaluate=evaluate,
            update_enabled=update_enabled, resource_check=budget.check,
        )
        statuses.append({"trajectory_id": trajectory_id, "side": side,
                         "status": result["status"],
                         "last_checkpoint": result.get("last_checkpoint", 0)})
        outputs[side] = result
        if result["status"] == "SOLVER_FAILED":
            break
    return outputs, {"pair_id": pair_id, "backbone": backend, "method": method,
                     "seed": "" if seed is None else seed, "protocol": protocol,
                     "statuses": statuses}


def _training_labels_and_features(binary_ids: list[str], dfit_states: dict[str, dict[str, Any]],
                                  feature_by_id: dict[str, Any]):
    y = np.asarray([base.VALID_STATES[dfit_states[i]["state"]] for i in binary_ids], dtype=np.int8)
    return [feature_by_id[i] for i in binary_ids], y


def _trajectory_specs():
    for backend in base.BACKBONES:
        for method in base.GLOBAL_READOUTS:
            for protocol in ("X", "W"):
                yield backend, method, None, protocol
        for method in base.MIL_READOUTS:
            for seed in base.MIL_SEEDS:
                for protocol in ("X", "W"):
                    yield backend, method, seed, protocol


def prepare_authorized_inputs():
    check_start_identity()
    frozen = check_frozen_files_against_head()
    cache = base_data.verify_frozen_cache()
    if cache.get("cache_status") != "PASS":
        raise IntegrityError("E2-A1 frozen cache verifier did not pass")
    tape_rows, tape_summary = base_data.validate_tape(cache["registered_by_id"])
    dfit_ids, dfit_states, binary_dfit_ids, y_dfit, dfit_summary = load_dfit_only(cache)
    profile_validation = validate_old_profile(cache, dfit_ids, binary_dfit_ids, y_dfit)
    hashes = input_hash_snapshot()
    active_ids = {r["image_id"] for r in tape_rows
                  if r["feedback_or_future_role"] in {"feedback_query", "future_eval_candidate"}}
    dev_states, dev_summary = base_data.load_tape_labels(active_ids)
    if len(dev_states) != len(active_ids):
        raise IntegrityError("authorized W/X labels do not exactly cover active tape IDs")
    source_types = base_data.read_source_type_metadata(
        [r["image_id"] for r in cache["registered_rows"]]
    )
    role_by_id = {r["image_id"]: r["data_role"] for r in cache["registered_rows"]}
    state_by_id = {**dfit_states, **dev_states}
    protocol_data = {}
    label_support = []
    for protocol in ("X", "W"):
        # build_protocol_rows lives in the old execution module because it also enforces
        # its frozen support counts; import lazily to avoid changing the R0 module.
        from importlib import import_module
        r0 = import_module("run_e2b")
        events, futures = r0.build_protocol_rows(
            tape_rows, protocol, state_by_id, source_types, cache["registered_by_id"]
        )
        for row in events:
            row["features"] = None
        protocol_data[protocol] = (events, futures)
        label_support.append({
            "protocol": protocol, "role": "feedback",
            **dict(Counter(dev_states[e["image_id"]]["state"] for e in events)),
        })
        label_support.append({
            "protocol": protocol, "role": "future",
            **dict(Counter(dev_states[r["image_id"]]["state"] for r in futures)),
        })
    return {
        "frozen_code_hashes": frozen, "cache": cache, "tape_rows": tape_rows,
        "tape_summary": tape_summary, "dfit_ids": dfit_ids,
        "dfit_states": dfit_states, "binary_dfit_ids": binary_dfit_ids,
        "y_dfit": y_dfit, "dfit_summary": dfit_summary,
        "profile_validation": profile_validation, "input_hashes": hashes,
        "active_ids": active_ids, "dev_states": dev_states,
        "dev_summary": dev_summary, "source_types": source_types,
        "role_by_id": role_by_id, "protocol_data": protocol_data,
        "label_support": label_support,
    }


def _write_preflight_receipts(inputs: dict[str, Any], run_identity: dict[str, Any]):
    protected_path = OUTPUT_ROOT / "protected_assets_before.json"
    current_protected = protected_snapshot()
    if protected_path.exists():
        old = read_json(protected_path)
        verify_self_sha(old, str(protected_path))
        if old.get("assets") != current_protected:
            raise IntegrityError("protected P1/E1/E2-A0/E2-A1 assets changed since R1 began")
    else:
        immutable_receipt(protected_path, {"schema": "lp.d6.p2.e2b.r1.protected-snapshot.v1",
                                           "assets": current_protected})
    input_path = OUTPUT_ROOT / "input_hashes_before.json"
    if input_path.exists():
        old = read_json(input_path)
        verify_self_sha(old, str(input_path))
        if old.get("input_hashes") != inputs["input_hashes"]:
            raise IntegrityError("annotation, tape or manifest inputs changed since R1 began")
    else:
        immutable_receipt(input_path, {"schema": "lp.d6.p2.e2b.r1.input-hashes.v1",
                                       "input_hashes": inputs["input_hashes"]})
    profile_path = OUTPUT_ROOT / "profile_revalidation.json"
    profile_meta = {k: v for k, v in inputs["profile_validation"].items()
                    if k not in {"profile_initial_theta", "profile_fitted_theta", "dfit_binary_ids"}}
    profile_meta["identity"] = {"code_fingerprint": run_identity["code_fingerprint"],
                                 "profile_hashes": inputs["profile_validation"]["profile_hashes"]}
    immutable_receipt(profile_path, {"schema": "lp.d6.p2.e2b.r1.profile-revalidation.v1",
                                     "identity": profile_meta["identity"],
                                     "validation": profile_meta})
    preflight = {
        "schema": "lp.d6.p2.e2b.r1.data-preflight.v1",
        "identity": {"code_fingerprint": run_identity["code_fingerprint"],
                     "head": EXPECTED_HEAD},
        "cache_status": inputs["cache"]["cache_status"],
        "manifest_hashes": inputs["cache"]["manifest_hashes"],
        "registered_image_count": len(inputs["cache"]["registered_rows"]),
        "data_roles": dict(Counter(r["data_role"] for r in inputs["cache"]["registered_rows"])),
        "tape_role_summary": inputs["tape_summary"],
        "dfit_summary": inputs["dfit_summary"], "ddev_tape_label_summary": inputs["dev_summary"],
        "protocol_label_support_evaluator_only": inputs["label_support"],
        "source_metadata_used_by_learner": False,
        "future_labels_passed_to_learner": False,
    }
    immutable_receipt(OUTPUT_ROOT / "data_preflight.json", preflight)


def _make_initialization(backend: str, method: str, seed: int | None,
                         bags_by_id: dict[str, np.ndarray],
                         global_by_method: dict[str, dict[str, np.ndarray]],
                         inputs: dict[str, Any], run_identity: dict[str, Any],
                         budget: Budget, profile_validation: dict[str, Any]) -> dict[str, Any]:
    identity = make_initialization_identity(
        run_identity, backend, method, seed, inputs["binary_dfit_ids"]
    )
    path = initialization_path(backend, method, seed)
    if method in base.GLOBAL_READOUTS:
        initial = np.zeros(global_by_method[method][inputs["binary_dfit_ids"][0]].shape[0] + 1,
                           dtype=np.float64)
        x0 = np.stack([global_by_method[method][i] for i in inputs["binary_dfit_ids"]])
        y = inputs["y_dfit"]
        fit_callable = lambda: base_model.fit_global(x0, y, initial, base.LAMBDA, base.SOLVER)
        bundle = ensure_initialization(path, identity, initial=initial,
                                       fit_callable=fit_callable, budget=budget)
    else:
        start = base_model.mil_initial_parameters(bags_by_id[inputs["binary_dfit_ids"][0]].shape[1], int(seed))
        method_rho = base.RHO if method == "M-Rank-Softplus" else 0.0
        train_bags = [bags_by_id[i] for i in inputs["binary_dfit_ids"]]
        y = inputs["y_dfit"]

        def fit_mil():
            return base_model.fit_mil(
                train_bags, y, start, base.LAMBDA, rho=method_rho,
                tau=base.TAU, margin=base.MARGIN, kappa=base.KAPPA, options=base.SOLVER,
            )

        reuse = None
        if backend == "dinov2" and method == "M-BCE" and seed == 0:
            reuse = {
                "profile_fitted_theta": profile_validation["profile_fitted_theta"],
                "profile_hashes": profile_validation["profile_hashes"],
                "solver_record": profile_validation["solver_record"],
            }
        details = lambda theta: fit_details_for_mil(theta, train_bags, y, method)
        bundle = ensure_initialization(path, identity, initial=start, fit_callable=fit_mil,
                                       budget=budget, profile_reuse=reuse,
                                       details_callable=details)
    if bundle["identity"] != jsonable(identity):
        raise IntegrityError(f"initialization identity mismatch: {backend}/{method}/{seed}")
    return bundle


def _check_mil_seed_pair(starts: dict[tuple[str, int], str], backend: str,
                         seed: int, method: str, bundle: dict[str, Any]):
    digest = base_model.parameter_digest(bundle["initial_theta"])
    key = (backend, int(seed))
    if key in starts and starts[key] != digest:
        raise IntegrityError(f"M-BCE/M-Rank initial parameter mismatch: {key}")
    starts[key] = digest


def _run_backend(backend: str, inputs: dict[str, Any], run_identity: dict[str, Any], budget: Budget,
                 profile_validation: dict[str, Any]) -> list[dict[str, Any]]:
    registered = inputs["cache"]["registered_rows"]
    dfit_ids = set(inputs["dfit_ids"])
    active_ids = inputs["active_ids"]
    requested = dfit_ids | active_ids
    active_order = [r["image_id"] for r in registered if r["image_id"] in requested]
    bundles = base_data.load_feature_bundle(backend, active_order)
    global_features: dict[str, dict[str, np.ndarray]] = {m: {} for m in base.GLOBAL_READOUTS}
    mil_bags: dict[str, np.ndarray] = {}
    for image_id in active_order:
        budget.check(f"features_{backend}_{image_id}")
        item = bundles[image_id]
        for readout in base.GLOBAL_READOUTS:
            global_features[readout][image_id] = base_model.build_global_readout(
                readout, item["cls_unit"], item["patch_unit"], item["valid_patch_mask"]
            )
        mil_bags[image_id] = base_model.valid_bag(item["patch_unit"], item["valid_patch_mask"])

    dfit_ids = inputs["dfit_ids"]
    binary_ids = inputs["binary_dfit_ids"]
    y_dfit = inputs["y_dfit"]
    feedback_states = {image_id: inputs["dev_states"][image_id]
                       for events, _ in inputs["protocol_data"].values()
                       for image_id in (e["image_id"] for e in events)}
    mil_initials: dict[tuple[str, int], dict[str, Any]] = {}
    global_initials: dict[str, dict[str, Any]] = {}
    methods = []

    for method in base.GLOBAL_READOUTS:
        budget.check(f"before_global_initialization_{backend}_{method}")
        init = _make_initialization(backend, method, None, mil_bags, global_features,
                                    inputs, run_identity, budget, profile_validation)
        global_initials[method] = init

    seed_starts: dict[tuple[str, int], str] = {}
    for seed in base.MIL_SEEDS:
        for method in base.MIL_READOUTS:
            budget.check(f"before_mil_initialization_{backend}_{method}_seed{seed}")
            init = _make_initialization(backend, method, seed, mil_bags, global_features,
                                        inputs, run_identity, budget, profile_validation)
            _check_mil_seed_pair(seed_starts, backend, int(seed), method, init)
            mil_initials[(method, int(seed))] = init

    y_dfit = np.asarray(y_dfit, dtype=np.int8)
    for method in base.GLOBAL_READOUTS:
        init = global_initials[method]
        if not init.get("accepted"):
            for protocol in ("X", "W"):
                methods.append({"backbone": backend, "method": method, "seed": "",
                                "protocol": protocol, "status": "INITIALIZATION_FAILED",
                                "last_checkpoint": 0, "side": "learner"})
            continue
        theta0 = np.asarray(init["theta"], dtype=np.float64)
        initial_features = [global_features[method][i] for i in binary_ids]
        x_index = {image_id: global_features[method][image_id] for image_id in active_order}
        for protocol in ("X", "W"):
            events_raw, future_rows = inputs["protocol_data"][protocol]
            events = [{**row, "features": x_index[row["image_id"]]} for row in events_raw]
            future_features = [x_index[row["image_id"]] for row in future_rows]
            future_states = {r["image_id"]: inputs["dev_states"][r["image_id"]]
                             for r in future_rows}

            def make_fit(features, labels, initial, checkpoint):
                return base_model.fit_global(np.stack(features), labels, initial,
                                             base.LAMBDA, base.SOLVER)

            results, status = execute_pair(
                backend=backend, method=method, seed=None, protocol=protocol,
                initial_theta=theta0, initial_features=initial_features,
                initial_labels=y_dfit, initial_ids=binary_ids, events=events,
                future_rows=future_rows, future_features=future_features,
                feedback_state_by_id=feedback_states, future_state_by_id=future_states,
                input_hashes=inputs["input_hashes"], run_identity=run_identity,
                budget=budget,
                predict_one=lambda theta, x: float(base_model.predict_global(theta, np.asarray(x)[None, :])[0]),
                predict_future=lambda theta, xs: base_model.predict_global(theta, np.stack(xs)),
                fit_history=make_fit,
            )
            methods.extend(status["statuses"])
            if status["statuses"] and any(r["status"] == "SOLVER_FAILED" for r in status["statuses"]):
                methods.append({"backbone": backend, "method": method, "seed": "",
                                "protocol": protocol, "status": "SOLVER_FAILED", "side": "learner"})

    for method in base.MIL_READOUTS:
        rho = base.RHO if method == "M-Rank-Softplus" else 0.0
        for seed in base.MIL_SEEDS:
            init = mil_initials[(method, int(seed))]
            if not init.get("accepted"):
                for protocol in ("X", "W"):
                    methods.append({"backbone": backend, "method": method, "seed": seed,
                                    "protocol": protocol, "status": "INITIALIZATION_FAILED",
                                    "last_checkpoint": 0, "side": "learner"})
                continue
            theta0 = np.asarray(init["theta"], dtype=np.float64)
            initial_bags = [mil_bags[i] for i in binary_ids]
            for protocol in ("X", "W"):
                events_raw, future_rows = inputs["protocol_data"][protocol]
                events = [{**row, "features": mil_bags[row["image_id"]]} for row in events_raw]
                future_features = [mil_bags[row["image_id"]] for row in future_rows]
                future_states = {r["image_id"]: inputs["dev_states"][r["image_id"]]
                                 for r in future_rows}

                def make_fit(features, labels, initial, checkpoint):
                    return base_model.fit_mil(
                        features, labels, initial, base.LAMBDA, rho=rho,
                        tau=base.TAU, margin=base.MARGIN, kappa=base.KAPPA,
                        options=base.SOLVER,
                    )

                results, status = execute_pair(
                    backend=backend, method=method, seed=int(seed), protocol=protocol,
                    initial_theta=theta0, initial_features=initial_bags,
                    initial_labels=y_dfit, initial_ids=binary_ids, events=events,
                    future_rows=future_rows, future_features=future_features,
                    feedback_state_by_id=feedback_states, future_state_by_id=future_states,
                    input_hashes=inputs["input_hashes"], run_identity=run_identity,
                    budget=budget,
                    predict_one=lambda theta, bag: float(base_model.predict_mil(theta, [bag], base.TAU)[0]),
                    predict_future=lambda theta, bags: base_model.predict_mil(theta, bags, base.TAU),
                    fit_history=make_fit,
                )
                methods.extend(status["statuses"])
    del bundles, global_features, mil_bags
    return methods


def _verify_completion_file(path: Path, identity: dict[str, Any],
                            checkpoints: list[dict[str, Any]]):
    complete = read_json(path)
    verify_self_sha(complete, str(path))
    if complete.get("identity") != jsonable(identity):
        raise IntegrityError(f"trajectory completion identity mismatch: {path}")
    if complete.get("checkpoint_record_sha256") != [cp["record_sha256"] for cp in checkpoints]:
        raise IntegrityError(f"trajectory completion checkpoint list mismatch: {path}")
    if complete.get("checkpoint_count") != 13 or complete.get("last_checkpoint") != 12:
        raise IntegrityError(f"trajectory completion counts invalid: {path}")


def _load_and_audit_trajectory(path: Path, *, expected_code_fingerprint: str):
    if not path.exists():
        return {"status": "NOT_STARTED", "checkpoints": [], "record": None,
                "solver_failure": None}
    failure_path = path / "terminal_failure.json"
    failure = None
    if failure_path.exists():
        failure = read_json(failure_path)
        verify_self_sha(failure, str(failure_path))
    cp_paths = sorted(path.glob("checkpoint_[0-9][0-9]"))
    numbers = [int(p.name[-2:]) for p in cp_paths]
    if numbers != list(range(len(numbers))):
        raise IntegrityError(f"audit found checkpoint gap: {path}")
    checkpoints = [verify_bundle(p, expected_checkpoint=i) for i, p in enumerate(cp_paths)]
    if checkpoints:
        identity = checkpoints[0].get("identity")
        if identity.get("run_code_fingerprint") != expected_code_fingerprint:
            raise IntegrityError(f"trajectory code fingerprint mismatch: {path}")
        for i, cp in enumerate(checkpoints):
            if cp.get("identity") != identity or cp.get("checkpoint") != i:
                raise IntegrityError(f"trajectory checkpoint identity mismatch: {path}/{i}")
            theta = np.asarray(cp["_arrays"]["theta"], dtype=np.float64)
            scores = np.asarray(cp["_arrays"]["future_scores"], dtype=np.float64)
            if not np.isfinite(theta).all() or not np.isfinite(scores).all():
                raise IntegrityError(f"nonfinite saved state or score: {path}/{i}")
            if base_model.parameter_digest(theta) != cp.get("theta_sha256"):
                # Legacy R1 checkpoint identity uses the raw array digest.
                if hashlib.sha256(theta.tobytes()).hexdigest() != cp.get("theta_sha256"):
                    raise IntegrityError(f"saved parameter SHA mismatch: {path}/{i}")
            items = cp.get("item_predictions", [])
            if len(items) != len(scores):
                raise IntegrityError(f"saved evaluator predictions do not match scores: {path}/{i}")
            state_map = {r["image_id"]: {"state": r["attribute_state_evaluator_only"]} for r in items}
            future_rows = [{k: r[k] for k in ("image_id", "product", "sample_key", "view",
                                              "source_type_evaluator_only") if k in r}
                           for r in items]
            recomputed, product_rows, item_rows = base_evaluator.summarize(scores, future_rows, state_map)
            if canonical_bytes(recomputed) != canonical_bytes(cp["metrics"]):
                raise IntegrityError(f"evaluator summary mismatch: {path}/{i}")
            if canonical_bytes(product_rows) != canonical_bytes(cp["product_metrics"]):
                raise IntegrityError(f"product evaluator summary mismatch: {path}/{i}")
            if canonical_bytes(item_rows) != canonical_bytes(cp["item_predictions"]):
                raise IntegrityError(f"item evaluator summary mismatch: {path}/{i}")
            if i > 0:
                event = cp.get("event")
                event_path = path / "events" / f"event_{i:02d}.json"
                persisted = event_record(event_path, identity, str(event["event_id"]), i)
                if persisted is None or persisted.get("record_sha256") != cp.get("event_record_sha256"):
                    raise IntegrityError(f"saved feedback receipt does not match checkpoint: {event_path}")
        if (checkpoints[0].get("identity") is None
                or checkpoints[0].get("initial_theta_sha256") != checkpoints[0].get("theta_sha256")):
            raise IntegrityError(f"checkpoint 0 is not its recorded initialization: {path}")
    complete_path = path / "complete.json"
    if complete_path.exists():
        if len(checkpoints) != 13:
            raise IntegrityError(f"complete trajectory is missing checkpoints: {path}")
        _verify_completion_file(complete_path, checkpoints[0]["identity"], checkpoints)
        status = "SOLVER_FAILED" if failure else "PASS"
    elif failure:
        if failure.get("identity", {}).get("run_code_fingerprint") != expected_code_fingerprint:
            raise IntegrityError(f"terminal failure code fingerprint mismatch: {path}")
        status = "SOLVER_FAILED"
    elif checkpoints:
        status = f"PARTIAL_LAST_CHECKPOINT_{len(checkpoints)-1}"
    else:
        status = "NOT_STARTED"
    return {"status": status, "checkpoints": checkpoints,
            "record": checkpoints[-1] if checkpoints else None,
            "identity": checkpoints[0].get("identity") if checkpoints else None,
            "solver_failure": failure}


def _event_effect_rows(checkpoints: list[dict[str, Any]], *, backend: str,
                       method: str, seed: Any, protocol: str, side: str):
    if not checkpoints:
        return []
    pred_by_cp = {
        cp["checkpoint"]: {item["image_id"]: item for item in cp.get("item_predictions", [])}
        for cp in checkpoints
    }
    rows = []
    for cp_idx in sorted(pred_by_cp):
        current = pred_by_cp[cp_idx]
        previous = pred_by_cp.get(cp_idx - 1, {})
        initial = pred_by_cp[0]
        counts = {"harmful_vs_previous": 0, "corrected_vs_previous": 0,
                  "flips_vs_previous": 0, "harmful_vs_checkpoint0": 0,
                  "corrected_vs_checkpoint0": 0, "flips_vs_checkpoint0": 0}
        for image_id, item in current.items():
            target = item.get("binary_target_evaluator_only")
            if target is None:
                continue
            pred = int(item["prediction"])
            init_pred = int(initial[image_id]["prediction"])
            if pred != init_pred:
                counts["flips_vs_checkpoint0"] += 1
            if pred == int(target) and init_pred != int(target):
                counts["corrected_vs_checkpoint0"] += 1
            elif pred != int(target) and init_pred == int(target):
                counts["harmful_vs_checkpoint0"] += 1
            if image_id in previous:
                prev_pred = int(previous[image_id]["prediction"])
                if pred != prev_pred:
                    counts["flips_vs_previous"] += 1
                if pred == int(target) and prev_pred != int(target):
                    counts["corrected_vs_previous"] += 1
                elif pred != int(target) and prev_pred == int(target):
                    counts["harmful_vs_previous"] += 1
        rows.append({"protocol": protocol, "backbone": backend, "method": method,
                     "seed": "" if seed is None else seed, "side": side,
                     "checkpoint": cp_idx, **counts})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None):
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def audit_outputs(inputs: dict[str, Any], run_identity: dict[str, Any], session_id: str,
                  worker_state: dict[str, Any], protected_after: dict[str, Any],
                  input_hashes_after: dict[str, str]) -> dict[str, Any]:
    expected_code = run_identity["code_fingerprint"]
    checkpoint_rows, product_rows, prediction_rows = [], [], []
    event_rows, effect_rows, solver_rows, trajectory_rows, state_arrays = [], [], [], [], {}
    source_attr_counts = []
    all_status = []
    audited: dict[str, dict[str, Any]] = {}

    for backend, method, seed, protocol in _trajectory_specs():
        pair_id = f"{backend}__{method}__{('seed'+str(seed)+'__') if seed is not None else ''}{protocol}"
        for side in ("control", "learner"):
            trajectory_id = f"{pair_id}__{side}"
            path = OUTPUT_ROOT / "trajectories" / pair_id / side
            audited[trajectory_id] = _load_and_audit_trajectory(path, expected_code_fingerprint=expected_code)
            outcome = audited[trajectory_id]
            trajectory_rows.append({"trajectory_id": trajectory_id, "pair_id": pair_id,
                                    "backbone": backend, "method": method,
                                    "seed": "" if seed is None else seed,
                                    "protocol": protocol, "side": side,
                                    "status": outcome["status"],
                                    "last_checkpoint": len(outcome["checkpoints"])-1 if outcome["checkpoints"] else "",
                                    "solver_failure_event": outcome["solver_failure"].get("failure_event", "")
                                    if outcome["solver_failure"] else ""})
            if side == "learner":
                all_status.append(outcome["status"])
            for cp in outcome["checkpoints"]:
                meta = {k: v for k, v in cp.items() if k != "_arrays"}
                state = cp["_arrays"]["theta"]
                future_scores = cp["_arrays"]["future_scores"]
                prefix = f"{trajectory_id}__cp{cp['checkpoint']:02d}"
                state_arrays[prefix + "__theta"] = state
                state_arrays[prefix + "__future_scores"] = future_scores
                metrics = dict(cp["metrics"])
                checkpoint_rows.append({"trajectory_id": trajectory_id, "pair_id": pair_id,
                    "backbone": backend, "method": method, "seed": "" if seed is None else seed,
                    "protocol": protocol, "side": side, "checkpoint": cp["checkpoint"],
                    "events_seen": cp["events_seen"], "binary_updates": cp["binary_updates"],
                    "training_history_rows": cp["training_history_rows"],
                    "training_history_sha256": cp["training_history_sha256"],
                    "theta_sha256": cp["theta_sha256"],
                    "fit_status": cp.get("fit_record", {}).get("status", "INITIAL_STATE"),
                    "fit_seconds": cp.get("fit_seconds", 0.0), **metrics})
                product_rows.extend({"trajectory_id": trajectory_id, "backbone": backend,
                    "method": method, "seed": "" if seed is None else seed,
                    "protocol": protocol, "side": side, "checkpoint": cp["checkpoint"], **r}
                    for r in cp.get("product_metrics", []))
                prediction_rows.extend({"trajectory_id": trajectory_id, "backbone": backend,
                    "method": method, "seed": "" if seed is None else seed,
                    "protocol": protocol, "side": side, "checkpoint": cp["checkpoint"], **r}
                    for r in cp.get("item_predictions", []))
                if cp.get("event"):
                    event_rows.append({"trajectory_id": trajectory_id, "backbone": backend,
                        "method": method, "seed": "" if seed is None else seed,
                        "protocol": protocol, "side": side, **cp["event"]})
                fit_record = cp.get("fit_record", {})
                if cp["checkpoint"] > 0 and "iterations" in fit_record:
                    solver_rows.append({"trajectory_id": trajectory_id, "backbone": backend,
                        "method": method, "seed": "" if seed is None else seed,
                        "protocol": protocol, "checkpoint": cp["checkpoint"],
                        "training_history_rows": cp["training_history_rows"], **fit_record})
            if outcome["solver_failure"]:
                fail = outcome["solver_failure"]
                solver_rows.append({"trajectory_id": trajectory_id, "backbone": backend,
                    "method": method, "seed": "" if seed is None else seed,
                    "protocol": protocol, "checkpoint": fail.get("failure_event"),
                    "training_history_rows": "candidate rejected", **fail.get("fit_record", {})})
            event_effect = _event_effect_rows(outcome["checkpoints"], backend=backend,
                method=method, seed=seed, protocol=protocol, side=side)
            for row in event_effect:
                cp_ix = int(row["checkpoint"])
                if cp_ix > 0:
                    event = outcome["checkpoints"][cp_ix].get("event") or {}
                    effect_rows.append({"trajectory_id": trajectory_id, "backbone": backend,
                        "method": method, "seed": "" if seed is None else seed,
                        "protocol": protocol, "side": side,
                        "event_id": event.get("event_id", ""),
                        "image_id": event.get("image_id", ""),
                        "stream_order": cp_ix,
                        "feedback_state_after_prediction": event.get("feedback_state_after_prediction", ""),
                        **{k: row[k] for k in ("harmful_vs_previous", "corrected_vs_previous",
                                               "flips_vs_previous", "harmful_vs_checkpoint0",
                                               "corrected_vs_checkpoint0", "flips_vs_checkpoint0")}})

    # Initial fit receipts are also independently hash-checked and reported as solver rows.
    init_root = OUTPUT_ROOT / "initializations"
    if init_root.exists():
        for path in sorted(init_root.rglob("seed_none")) + sorted(init_root.rglob("seed_0")) + sorted(init_root.rglob("seed_1")):
            if not path.is_dir() or not (path / "record.json").exists():
                continue
            init = verify_bundle(path)
            ident = init["identity"]
            record = init["fit_record"]
            solver_rows.append({"stage": "D-fit_initialization", "backbone": ident["backend"],
                "method": ident["method"], "seed": ident["seed"], "protocol": "",
                "checkpoint": 0, "training_history_rows": init["training_history_rows"],
                "initial_theta_sha256": init["initial_theta_sha256"],
                "theta_sha256": init["theta_sha256"], **record,
                "profile_reuse": bool(init.get("profile_reuse"))})

    # A matched control must retain exactly the learner's checkpoint-0 state and scores.
    pair_checks = []
    for backend, method, seed, protocol in _trajectory_specs():
        pair_id = f"{backend}__{method}__{('seed'+str(seed)+'__') if seed is not None else ''}{protocol}"
        control_id, learner_id = f"{pair_id}__control", f"{pair_id}__learner"
        c, l = audited[control_id], audited[learner_id]
        if c["checkpoints"] and l["checkpoints"]:
            c0, l0 = c["checkpoints"][0], l["checkpoints"][0]
            if c0["theta_sha256"] != l0["theta_sha256"]:
                raise IntegrityError(f"B0 and learner initial states differ: {pair_id}")
            np.testing.assert_array_equal(c0["_arrays"]["future_scores"], l0["_arrays"]["future_scores"])
            for cp in c["checkpoints"]:
                if cp["theta_sha256"] != c0["theta_sha256"]:
                    raise IntegrityError(f"B0 changed state: {pair_id}, checkpoint {cp['checkpoint']}")
            pair_checks.append({"pair_id": pair_id, "status": "PASS_INITIAL_MATCH_AND_FROZEN_B0"})
        else:
            pair_checks.append({"pair_id": pair_id, "status": "INCOMPLETE_NOT_COMPARABLE"})

    ddev_labels = base_data.first_state_by_item(*base.ANNOTATION_SOURCES["D-dev"])
    all_labels = {**inputs["dfit_states"], **ddev_labels}
    if set(all_labels) != set(inputs["role_by_id"]):
        raise IntegrityError("evaluator-only source/attribute audit does not cover the registered 102 items")
    crosstab = base_data.source_attribute_crosstab(
        inputs["role_by_id"], all_labels, inputs["source_types"], inputs["role_by_id"]
    )
    support_rows = inputs["label_support"]
    completed_learners = sum(row["status"] == "PASS" for row in trajectory_rows if row["side"] == "learner")
    completed_controls = sum(row["status"] == "PASS" for row in trajectory_rows if row["side"] == "control")
    failures = [row for row in trajectory_rows if row["status"] == "SOLVER_FAILED"]
    status = ("E2B_R1_PASS" if completed_learners == 28 and completed_controls == 28
              and not failures and worker_state.get("stop_reason") == "COMPLETED"
              and int(worker_state.get("sampled_peak_rss_bytes", 0)) < RSS_SOFT_LIMIT
              and protected_after == worker_state.get("protected_before")
              and input_hashes_after == inputs["input_hashes"] else "E2B_R1_PARTIAL")

    aggregate_root = OUTPUT_ROOT / "aggregates" / f"session_{session_id}"
    if aggregate_root.exists():
        old = read_json(aggregate_root / "aggregate_manifest.json")
        verify_self_sha(old, str(aggregate_root / "aggregate_manifest.json"))
        raise IntegrityError(f"aggregate already exists for this session: {aggregate_root}")
    temp = aggregate_root.parent / f".{aggregate_root.name}.pending-{os.getpid()}"
    temp.mkdir(parents=True, exist_ok=False)
    try:
        _write_csv(temp / "checkpoint_metrics.csv", checkpoint_rows)
        _write_csv(temp / "product_metrics.csv", product_rows)
        _write_csv(temp / "future_predictions.csv", prediction_rows)
        _write_csv(temp / "feedback_event_log.csv", event_rows)
        _write_csv(temp / "event_effects.csv", effect_rows)
        _write_csv(temp / "solver_log.csv", solver_rows)
        _write_csv(temp / "trajectory_status.csv", trajectory_rows)
        _write_csv(temp / "protocol_support.csv", support_rows)
        _write_csv(temp / "source_attribute_crosstab.csv", crosstab)
        np.savez_compressed(temp / "model_states.npz", **state_arrays)
        file_hashes = {p.name: sha256_file(p) for p in sorted(temp.iterdir()) if p.is_file()}
        manifest = _record_self_sha({
            "schema": "lp.d6.p2.e2b.r1.aggregate.v1", "identity": run_identity,
            "session_id": session_id, "status": status,
            "files": file_hashes, "expected_learner_trajectories": 28,
            "completed_learner_trajectories": completed_learners,
            "completed_b0_controls": completed_controls,
            "trajectory_rows": len(trajectory_rows), "checkpoint_rows": len(checkpoint_rows),
            "pair_checks": pair_checks,
        })
        with (temp / "aggregate_manifest.json").open("xb") as stream:
            stream.write(canonical_bytes(manifest) + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        aggregate_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, aggregate_root)
    except Exception:
        raise
    result = {
        "schema": "lp.d6.p2.e2b.r1.result.v1", "status": status,
        "expected_learner_trajectories": 28,
        "completed_learner_trajectories": completed_learners,
        "expected_b0_controls": 28, "completed_b0_controls": completed_controls,
        "trajectory_rows": len(trajectory_rows), "checkpoint_rows": len(checkpoint_rows),
        "solver_failures": failures, "pair_checks": pair_checks,
        "aggregate_directory": aggregate_root.relative_to(ROOT).as_posix(),
        "stop_reason": worker_state.get("stop_reason"),
        "protected_assets_unchanged": protected_after == worker_state.get("protected_before"),
        "input_hashes_unchanged": input_hashes_after == inputs["input_hashes"],
    }
    run_result_path = OUTPUT_ROOT / "run_result.json"
    write_json_atomic(run_result_path, _record_self_sha(result))
    return result


def worker_main() -> int:
    check_start_identity()
    if os.environ.get("E2B_R1_SUPERVISED") != "1":
        raise IntegrityError("R1 worker must be launched by the resource supervisor")
    ensure_output_ignored()
    session_id = os.environ.get("E2B_R1_SESSION_ID", "")
    if not session_id:
        raise IntegrityError("worker session ID was not supplied by the supervisor")
    wall_limit = float(os.environ.get("E2B_R1_WORKER_WALL_SECONDS", "0"))
    cpu_limit = float(os.environ.get("E2B_R1_WORKER_CPU_SECONDS", "0"))
    if wall_limit <= 0 or cpu_limit <= 0:
        raise IntegrityError("worker received an invalid remaining resource budget")
    code = code_identity()
    with RSSSampler() as sampler, threadpool_limits(limits=THREAD_LIMIT):
        budget = Budget(wall_limit, cpu_limit, sampler)
        inputs = prepare_authorized_inputs()
        budget.check("after_read_only_input_and_profile_validation")
        run_identity = validate_existing_run(code, inputs["profile_validation"], inputs["input_hashes"])
        _write_preflight_receipts(inputs, run_identity)
        protected_before = read_json(OUTPUT_ROOT / "protected_assets_before.json")["assets"]
        worker_state = {
            "session_id": session_id, "stop_reason": "COMPLETED",
            "protected_before": protected_before,
        }
        trajectory_statuses = []
        stop_reason = "COMPLETED"
        try:
            for backend in base.BACKBONES:
                budget.check(f"before_backbone_{backend}")
                trajectory_statuses.extend(_run_backend(
                    backend, inputs, run_identity, budget, inputs["profile_validation"]
                ))
        except ExecutionStop as exc:
            stop_reason = f"RESOURCE_STOP: {exc}"
        worker_state["stop_reason"] = stop_reason
        worker_state["trajectory_statuses_this_session"] = trajectory_statuses
        worker_state["elapsed_wall_seconds"] = budget.wall_used
        worker_state["process_cpu_seconds"] = budget.cpu_used
        worker_state["sampled_peak_rss_bytes"] = sampler.peak
        worker_state["profile_revalidation"] = {
            k: v for k, v in inputs["profile_validation"].items()
            if k not in {"profile_initial_theta", "profile_fitted_theta", "dfit_binary_ids"}
        }
        input_after = input_hash_snapshot()
        protected_after = protected_snapshot()
        if input_after != inputs["input_hashes"]:
            raise IntegrityError("registered labels, tape or manifests changed during R1")
        if protected_after != protected_before:
            raise IntegrityError("protected P1/E1/E2-A0/E2-A1/profile assets changed during R1")
        result = audit_outputs(inputs, run_identity, session_id, worker_state,
                               protected_after, input_after)
        worker_state["result_status"] = result["status"]
        worker_state["aggregate_directory"] = result["aggregate_directory"]
        worker_path = OUTPUT_ROOT / "worker_sessions" / f"session_{session_id}.json"
        immutable_receipt(worker_path, {
            "schema": "lp.d6.p2.e2b.r1.worker-session.v1",
            "identity": {"head": EXPECTED_HEAD, "code_fingerprint": code["code_fingerprint"]},
            "worker_state": worker_state, "result": result,
        })
        print(f"R1_WORKER_STATUS={result['status']}")
        print(f"LEARNER_TRAJECTORIES={result['completed_learner_trajectories']}/28")
        print(f"B0_CONTROLS={result['completed_b0_controls']}/28")
        print(f"WALL_SECONDS={budget.wall_used:.3f} PROCESS_CPU_SECONDS={budget.cpu_used:.3f}")
        print(f"PEAK_RSS_BYTES={sampler.peak}")
        print(f"AGGREGATE={result['aggregate_directory']}")
        if stop_reason != "COMPLETED":
            return 10
        return 0 if result["status"] == "E2B_R1_PASS" else 20


def _self_test_command() -> list[str]:
    return [sys.executable, "-m", "unittest",
            "tests.research_d6.test_e2b_numerics", "tests.research_d6.test_e2b_r1", "-v"]


def _child_cpu_and_rss(proc: subprocess.Popen) -> tuple[float, int]:
    try:
        p = psutil.Process(proc.pid)
        cpu = p.cpu_times().user + p.cpu_times().system
        rss = p.memory_info().rss
        return cpu, rss
    except psutil.Error:
        return 0.0, 0


def _run_monitored(command: list[str], env: dict[str, str], *,
                   remaining_wall: float, remaining_cpu: float,
                   parent_cpu_base: float, session_wall_start: float,
                   prior_wall: float, prior_cpu: float,
                   sample_peak: int) -> dict[str, Any]:
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    child_cpu_peak = 0.0
    process_peak_rss = int(sample_peak)
    output_parts = []
    stop_reason = "EXITED"
    try:
        while proc.poll() is None:
            child_cpu, rss = _child_cpu_and_rss(proc)
            child_cpu_peak = max(child_cpu_peak, child_cpu)
            process_peak_rss = max(process_peak_rss, rss)
            total_wall = prior_wall + (time.perf_counter() - session_wall_start)
            parent_cpu = max(0.0, time.process_time() - parent_cpu_base)
            total_cpu = prior_cpu + parent_cpu + child_cpu_peak
            if total_wall >= MAX_WALL_SECONDS or total_wall >= prior_wall + remaining_wall:
                stop_reason = "WALL_TIME_HARD_LIMIT"
                proc.kill()
                break
            if total_cpu >= MAX_CPU_SECONDS or total_cpu >= prior_cpu + remaining_cpu:
                stop_reason = "PROCESS_CPU_HARD_LIMIT"
                proc.kill()
                break
            time.sleep(0.05)
        try:
            stdout, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
        if stdout:
            output_parts.append(stdout)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return {"return_code": proc.returncode, "output": "".join(output_parts),
            "child_process_cpu_seconds": child_cpu_peak,
            "sampled_peak_rss_bytes": process_peak_rss,
            "stop_reason": stop_reason}


def _finished_sessions() -> list[dict[str, Any]]:
    sessions = OUTPUT_ROOT / "supervisor_sessions"
    if not sessions.exists():
        return []
    starts = sorted(sessions.glob("session_*.started.json"))
    finished = sorted(sessions.glob("session_*.finished.json"))
    finish_ids = {p.name.replace(".finished.json", "") for p in finished}
    open_ids = {p.name.replace(".started.json", "") for p in starts} - finish_ids
    if open_ids:
        raise IntegrityError(f"unclosed supervisor session requires review: {sorted(open_ids)[0]}")
    rows = []
    for path in finished:
        record = read_json(path)
        verify_self_sha(record, str(path))
        if record.get("identity", {}).get("head") != EXPECTED_HEAD:
            raise IntegrityError(f"supervisor session HEAD mismatch: {path}")
        rows.append(record)
    return rows


def supervisor_main() -> int:
    check_start_identity()
    ensure_output_ignored()
    code = code_identity()
    # The immutable run identity is made before labels are read; the worker must still
    # independently recompute and pass the historical D-fit-only profile receipt.
    hashes = profile_hashes()
    with np.load(PROFILE_NPZ, allow_pickle=False) as stored:
        initial = np.asarray(stored["initial_theta"], dtype=np.float64).copy()
        fitted = np.asarray(stored["fitted_theta"], dtype=np.float64).copy()
    light_profile_identity = {"profile_hashes": hashes,
        "initial_theta_sha256": base_model.parameter_digest(initial),
        "fitted_theta_sha256": base_model.parameter_digest(fitted)}
    inputs_hash = input_hash_snapshot()
    run_identity = validate_existing_run(code, light_profile_identity, inputs_hash)
    session_dir = OUTPUT_ROOT / "supervisor_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    prior = _finished_sessions()
    prior_wall = sum(float(r["elapsed_wall_seconds"]) for r in prior)
    prior_cpu = sum(float(r["process_cpu_seconds"]) for r in prior)
    prior_peak = max([int(r.get("sampled_peak_rss_bytes", 0)) for r in prior] or [0])
    if prior_wall >= MAX_WALL_SECONDS or prior_cpu >= MAX_CPU_SECONDS:
        raise ExecutionStop("previous R1 sessions exhausted the approved cumulative resource budget")
    if prior_peak >= RSS_SOFT_LIMIT:
        raise ExecutionStop("a previous R1 session crossed the 6 GiB RSS soft-stop gate")

    session_number = len(prior) + 1
    session_id = f"{session_number:03d}"
    start_path = session_dir / f"session_{session_id}.started.json"
    finish_path = session_dir / f"session_{session_id}.finished.json"
    session_started = time.perf_counter()
    wall_start_utc = now_utc()
    parent_cpu_base = time.process_time()
    started = {
        "schema": "lp.d6.p2.e2b.r1.supervisor-session-start.v1",
        "identity": {"head": EXPECTED_HEAD, "branch": EXPECTED_BRANCH,
                     "code_fingerprint": code["code_fingerprint"]},
        "session_id": session_id, "started_utc": wall_start_utc,
        "prior_wall_seconds": prior_wall, "prior_cpu_seconds": prior_cpu,
        "remaining_wall_seconds": MAX_WALL_SECONDS - prior_wall,
        "remaining_cpu_seconds": MAX_CPU_SECONDS - prior_cpu,
        "resource_limits": {"wall_seconds": MAX_WALL_SECONDS,
                            "process_cpu_seconds": MAX_CPU_SECONDS,
                            "rss_soft_limit_bytes": RSS_SOFT_LIMIT},
    }
    immutable_receipt(start_path, started)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(THREAD_LIMIT)
    env["E2B_R1_SESSION_ID"] = session_id

    test_remaining_wall = min(MAX_WALL_SECONDS - prior_wall, MAX_CPU_SECONDS - prior_cpu)
    test = _run_monitored(_self_test_command(), env,
        remaining_wall=test_remaining_wall, remaining_cpu=test_remaining_wall,
        parent_cpu_base=parent_cpu_base, session_wall_start=session_started,
        prior_wall=prior_wall, prior_cpu=prior_cpu, sample_peak=prior_peak)
    test_transcript = test["output"]
    test_pass = (test["return_code"] == 0 and "Ran 22 tests" in test_transcript
                 and "OK" in test_transcript)
    test_seconds = time.perf_counter() - session_started
    own_cpu = max(0.0, time.process_time() - parent_cpu_base)
    current_wall = prior_wall + test_seconds
    current_cpu = prior_cpu + own_cpu + test["child_process_cpu_seconds"]
    current_peak = max(prior_peak, test["sampled_peak_rss_bytes"])
    if not test_pass:
        finish = _record_self_sha({
            "schema": "lp.d6.p2.e2b.r1.supervisor-session-finish.v1",
            "identity": started["identity"], "session_id": session_id,
            "started_utc": wall_start_utc, "finished_utc": now_utc(),
            "elapsed_wall_seconds": test_seconds,
            "process_cpu_seconds": own_cpu + test["child_process_cpu_seconds"],
            "sampled_peak_rss_bytes": current_peak,
            "synthetic_test_count": 22, "synthetic_tests_passed": False,
            "test_transcript_sha256": hashlib.sha256(test_transcript.encode("utf-8")).hexdigest(),
            "worker_return_code": None, "stop_reason": "SYNTHETIC_TEST_GATE_FAILED",
        })
        write_json_exclusive(finish_path, finish)
        print(test_transcript)
        print("E2B_R1_BLOCKED=SYNTHETIC_TEST_GATE_FAILED")
        return 3
    if (test["stop_reason"] != "EXITED" or current_wall >= MAX_WALL_SECONDS
            or current_cpu >= MAX_CPU_SECONDS or current_peak >= RSS_SOFT_LIMIT):
        finish = _record_self_sha({
            "schema": "lp.d6.p2.e2b.r1.supervisor-session-finish.v1",
            "identity": started["identity"], "session_id": session_id,
            "started_utc": wall_start_utc, "finished_utc": now_utc(),
            "elapsed_wall_seconds": test_seconds,
            "process_cpu_seconds": own_cpu + test["child_process_cpu_seconds"],
            "sampled_peak_rss_bytes": current_peak,
            "synthetic_test_count": 22, "synthetic_tests_passed": True,
            "test_transcript_sha256": hashlib.sha256(test_transcript.encode("utf-8")).hexdigest(),
            "worker_return_code": None, "stop_reason": test["stop_reason"],
        })
        write_json_exclusive(finish_path, finish)
        raise ExecutionStop("resource limit reached during pre-training synthetic gate")

    reserve = 180.0
    worker_wall = max(0.0, MAX_WALL_SECONDS - current_wall - reserve)
    worker_cpu = max(0.0, MAX_CPU_SECONDS - current_cpu - reserve)
    if worker_wall <= 1 or worker_cpu <= 1:
        finish = _record_self_sha({
            "schema": "lp.d6.p2.e2b.r1.supervisor-session-finish.v1",
            "identity": started["identity"], "session_id": session_id,
            "started_utc": wall_start_utc, "finished_utc": now_utc(),
            "elapsed_wall_seconds": test_seconds,
            "process_cpu_seconds": own_cpu + test["child_process_cpu_seconds"],
            "sampled_peak_rss_bytes": current_peak,
            "synthetic_test_count": 22, "synthetic_tests_passed": True,
            "test_transcript_sha256": hashlib.sha256(test_transcript.encode("utf-8")).hexdigest(),
            "worker_return_code": None, "stop_reason": "INSUFFICIENT_REMAINING_BUDGET",
        })
        write_json_exclusive(finish_path, finish)
        raise ExecutionStop("less than 180 seconds remain for audited worker finalization")

    env["E2B_R1_SUPERVISED"] = "1"
    env["E2B_R1_WORKER_WALL_SECONDS"] = f"{worker_wall:.6f}"
    env["E2B_R1_WORKER_CPU_SECONDS"] = f"{worker_cpu:.6f}"
    command = [sys.executable, str(ROOT / "scripts/research_d6/p2/e2b_r1/run_r1.py"), "--worker"]
    worker = _run_monitored(command, env,
        remaining_wall=MAX_WALL_SECONDS - prior_wall,
        remaining_cpu=MAX_CPU_SECONDS - prior_cpu,
        parent_cpu_base=parent_cpu_base, session_wall_start=session_started,
        prior_wall=prior_wall, prior_cpu=prior_cpu,
        sample_peak=current_peak)
    elapsed = time.perf_counter() - session_started
    supervisor_cpu = max(0.0, time.process_time() - parent_cpu_base)
    peak = max(current_peak, worker["sampled_peak_rss_bytes"])
    worker_result_path = OUTPUT_ROOT / "worker_sessions" / f"session_{session_id}.json"
    worker_result = read_json(worker_result_path) if worker_result_path.exists() else None
    worker_cpu = worker["child_process_cpu_seconds"]
    if worker_result:
        worker_cpu = max(worker_cpu, float(worker_result.get("worker_state", {}).get("process_cpu_seconds", 0.0)))
    cpu_total = supervisor_cpu + test["child_process_cpu_seconds"] + worker_cpu
    stop_reason = worker["stop_reason"]
    if stop_reason == "EXITED":
        stop_reason = "WORKER_COMPLETED" if worker["return_code"] == 0 else f"WORKER_EXIT_{worker['return_code']}"
    finish = _record_self_sha({
        "schema": "lp.d6.p2.e2b.r1.supervisor-session-finish.v1",
        "identity": started["identity"], "session_id": session_id,
        "started_utc": wall_start_utc, "finished_utc": now_utc(),
        "elapsed_wall_seconds": elapsed, "process_cpu_seconds": cpu_total,
        "sampled_peak_rss_bytes": peak,
        "synthetic_test_count": 22, "synthetic_tests_passed": True,
        "test_transcript_sha256": hashlib.sha256(test_transcript.encode("utf-8")).hexdigest(),
        "worker_return_code": worker["return_code"], "worker_stop_reason": worker["stop_reason"],
        "stop_reason": stop_reason,
        "worker_result": worker_result.get("result") if worker_result else None,
    })
    write_json_exclusive(finish_path, finish)
    total_wall = prior_wall + elapsed
    total_cpu = prior_cpu + cpu_total
    write_json_atomic(OUTPUT_ROOT / "budget_summary.json", _record_self_sha({
        "schema": "lp.d6.p2.e2b.r1.budget-summary.v1",
        "identity": started["identity"], "wall_limit_seconds": MAX_WALL_SECONDS,
        "process_cpu_limit_seconds": MAX_CPU_SECONDS,
        "cumulative_wall_seconds": total_wall,
        "cumulative_process_cpu_seconds": total_cpu,
        "remaining_wall_seconds": max(0.0, MAX_WALL_SECONDS-total_wall),
        "remaining_process_cpu_seconds": max(0.0, MAX_CPU_SECONDS-total_cpu),
        "peak_rss_bytes": peak, "sessions": len(prior)+1,
        "latest_worker_result": worker_result.get("result") if worker_result else None,
    }))
    print(test_transcript)
    print(worker["output"])
    print(f"R1_SESSION={session_id} WALL={elapsed:.2f}s CPU={cpu_total:.2f}s PEAK_RSS={peak}")
    if worker["return_code"] == 0:
        return 0
    if worker["return_code"] == 20:
        return 20
    if worker["return_code"] == 10 or worker["stop_reason"] != "EXITED":
        return 10
    return 3


def cli() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    try:
        if args.worker:
            return worker_main()
        return supervisor_main()
    except ExecutionStop as exc:
        print(f"E2B_R1_STOPPED={exc}", file=sys.stderr)
        return 10
    except Exception as exc:
        print(f"E2B_R1_BLOCKED={type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(cli())
