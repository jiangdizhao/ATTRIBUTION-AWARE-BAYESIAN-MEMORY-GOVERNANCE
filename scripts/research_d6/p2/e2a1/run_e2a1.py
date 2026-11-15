"""Guarded LP-D6-P2-E2-A1 runner: whitelist -> two-image smoke -> 102 cache."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_inputs import audit_registered_inputs, sha256_file
from contracts import (ALLOCATOR_CAP_BYTES, DINO2_CHECKPOINT_SHA256, DINO2_INTERPRETER,
                       DINO2_SOURCE_TREE_SHA256, DINO3_CONFIG_SHA256, DINO3_INTERPRETER,
                       DINO3_PROCESSOR_SHA256, DINO3_REVISION, DINO3_WEIGHTS_SHA256,
                       EXPECTED_BRANCH, EXPECTED_HEAD, INPUT_MANIFEST_SHA256, MAX_RUN_SECONDS,
                       OUT, OUTPUT_CAP_BYTES, ROOT as CONTRACT_ROOT, WX_TAPE_SHA256)
from verify_cache import verify_cache


PROTECTED_PATHS = [
    "outputs/research_d6/p1a_features",
    "outputs/research_d6/p1b/run_20261113_01",
    "outputs/research_d6/p1b_r1",
    "outputs/research_d6/p2/e1/run_20261115_r1_01",
    "outputs/research_d6/p2/e1_r2",
    "outputs/research_d6/p2/e2a0",
    "docs/research_d6/p1a",
    "docs/research_d6/p1b",
    "docs/research_d6/p1b_r1",
    "docs/research_d6/p2/g0",
    "docs/research_d6/p2/e1_r1",
    "docs/research_d6/p2/e1_r2",
    "docs/research_d6/p2/e2a0",
    "scripts/research_d6/p1a",
    "scripts/research_d6/p1b",
    "scripts/research_d6/p2/e1",
    "scripts/research_d6/p2/e2a",
    "tests/research_d6/test_e2a0_canonical_preprocess.py",
]


def stable_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def tree_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0, "sha256": ""}
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and ".git" not in item.parts and item.suffix != ".pyc"
    )
    base = path.parent if path.is_file() else path
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        digest.update(item.relative_to(base).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_file(item).encode("ascii") + b"\n")
        total_bytes += item.stat().st_size
    return {"exists": True, "files": len(files), "bytes": total_bytes, "sha256": digest.hexdigest()}


def protected_snapshot() -> dict[str, Any]:
    return {rel: tree_identity(ROOT / rel) for rel in PROTECTED_PATHS}


def backbone_identity() -> dict[str, Any]:
    p1a_path = ROOT / "scripts/research_d6/p1a/extract_features.py"
    spec = importlib.util.spec_from_file_location("e2a1_protected_p1a_identity", p1a_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import P1-A's local model identity routines")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = module.sha256_file(module.CHECKPOINT)
    source_sha, source_files, source_bytes = module.tree_sha256(module.HUB_SOURCE)
    d3_snapshot = (Path.home() / ".cache/huggingface/hub" /
                   "models--facebook--dinov3-vitl16-pretrain-lvd1689m" / "snapshots" / DINO3_REVISION)
    d3_hashes = {name: sha256_file(d3_snapshot / name)
                 for name in ("model.safetensors", "config.json", "preprocessor_config.json")}
    return {
        "dinov2_checkpoint_sha256": checkpoint,
        "dinov2_source_tree_sha256": source_sha,
        "dinov2_source_tree_files": source_files,
        "dinov2_source_tree_bytes": source_bytes,
        "dinov3_snapshot_revision": DINO3_REVISION,
        "dinov3_snapshot_sha256": d3_hashes,
    }


def source_code_identity() -> dict[str, str]:
    code_paths = sorted(SCRIPT_DIR.glob("*.py"))
    code_paths.extend([
        ROOT / "tests/research_d6/test_e2a0_canonical_preprocess.py",
        ROOT / "tests/research_d6/test_e2a1_preprocessing_storage.py",
        ROOT / "scripts/research_d6/p2/e2a/feature_interface.py",
        ROOT / "scripts/research_d6/p1a/extract_features.py",
    ])
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(set(code_paths))}


def gpu_status() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True, timeout=20,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one registered GPU, found {len(lines)}")
    parts = [part.strip() for part in lines[0].split(",")]
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=True, text=True, capture_output=True, timeout=20,
    )
    process_lines = [line for line in processes.stdout.splitlines() if line.strip()]
    return {
        "gpu_name": parts[0], "memory_total_mib": int(parts[1]), "memory_free_mib": int(parts[2]),
        "memory_used_mib": int(parts[3]), "utilization_percent": int(parts[4]),
        "compute_process_count": len(process_lines),
        "per_process_memory_visible": any("N/A" not in line for line in process_lines),
        "processes_terminated": False,
    }


def storage_estimate() -> dict[str, int]:
    n = 102
    canonical_each = 1 * 3 * 448 * 448 * 4 + 1024 + 784
    d2_each = 1024 * 4 + 4 * 1024 * 4 + 1024 * 1024 * 4 + 1024
    d3_each = 1024 * 4 + 4 * 1024 * 4 + 784 * 1024 * 4 + 784
    canonical = n * canonical_each
    dino2, dino3 = n * d2_each, n * d3_each
    total = canonical + dino2 + dino3
    return {"canonical_bytes_estimate": canonical, "dinov2_bytes_estimate": dino2,
            "dinov3_bytes_estimate": dino3, "arrays_total_bytes_estimate": total,
            "registered_output_cap_bytes": OUTPUT_CAP_BYTES}


def write_json_once(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Existing file differs; refusing overwrite: {path.name}")
    else:
        with path.open("xb") as stream:
            stream.write(encoded.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())


def run_child(label: str, interpreter: Path, script: Path, args: list[str], timeout: float) -> dict[str, Any]:
    env = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        env.pop(key, None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    started = time.perf_counter()
    result = subprocess.run([str(interpreter), str(script), *args], cwd=ROOT, env=env,
                            text=True, capture_output=True, timeout=max(1.0, timeout))
    elapsed = time.perf_counter() - started
    log_stem = f"{label}.log"
    (OUT / f"{log_stem}").write_text(result.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.log").write_text(result.stderr, encoding="utf-8")
    return {"label": label, "return_code": result.returncode, "elapsed_seconds": elapsed,
            "stdout_log": log_stem, "stderr_log": f"{label}.stderr.log",
            "interpreter": str(interpreter), "script": script.relative_to(ROOT).as_posix()}


def run_tests() -> list[dict[str, Any]]:
    commands = [
        ("a0-canonical-tests", ["-m", "unittest", "discover", "-s", "tests/research_d6",
                                "-p", "test_e2a0_canonical_preprocess.py", "-v"]),
        ("e2a1-tests", ["-m", "unittest", "discover", "-s", "tests/research_d6",
                         "-p", "test_e2a1_preprocessing_storage.py", "-v"]),
    ]
    results = []
    for label, args in commands:
        started = time.perf_counter()
        result = subprocess.run([str(DINO2_INTERPRETER), *args], cwd=ROOT, text=True,
                                capture_output=True, timeout=300)
        results.append({"label": label, "return_code": result.returncode,
                        "elapsed_seconds": time.perf_counter() - started,
                        "stdout": result.stdout, "stderr": result.stderr})
    return results


def main() -> int:
    task_started = time.perf_counter()
    deadline = task_started + MAX_RUN_SECONDS
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if (head, branch) != (EXPECTED_HEAD, EXPECTED_BRANCH):
        raise RuntimeError(f"Unexpected start identity: {branch}/{head}")
    if ROOT != CONTRACT_ROOT:
        raise RuntimeError("Repository roots in runner and contracts differ")
    if OUT.exists():
        raise FileExistsError("E2-A1 output exists; refusing overwrite or implicit resume at run root")
    ignore_check = subprocess.run(["git", "check-ignore", "-q", (OUT / "probe").relative_to(ROOT).as_posix()],
                                  cwd=ROOT)
    if ignore_check.returncode != 0:
        raise RuntimeError("E2-A1 output root is not Git-ignored; refusing to store protected caches")
    if not DINO2_INTERPRETER.is_file() or not DINO3_INTERPRETER.is_file():
        raise FileNotFoundError("One or both registered Conda interpreters are missing")
    a0_test = ROOT / "tests/research_d6/test_e2a0_canonical_preprocess.py"
    if not a0_test.is_file():
        raise FileNotFoundError("A0 canonical preprocessing test is absent; E2-A1 is blocked")
    a0_tree_entry = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD",
                                    "tests/research_d6/test_e2a0_canonical_preprocess.py"],
                                   cwd=ROOT, text=True, capture_output=True, check=True)
    a0_git_status = subprocess.run(["git", "status", "--short", "--",
                                    "tests/research_d6/test_e2a0_canonical_preprocess.py"],
                                   cwd=ROOT, text=True, capture_output=True, check=True)
    a0_test_archival = {
        "file_exists": True,
        "tracked_at_expected_head": bool(a0_tree_entry.stdout.strip()),
        "working_tree_status": a0_git_status.stdout.strip(),
        "sha256": sha256_file(a0_test),
        "archival_note": "Local A0 six-test file exists but is absent from the expected HEAD tree; it is retained unmodified and remains available for the next archive. Not staged automatically.",
    }

    # The source and cache identities are locked before any registered image is decoded.
    inputs, input_summary = audit_registered_inputs()
    if (input_summary["input_manifest_sha256"], input_summary["feature_manifest_sha256"],
            input_summary["wx_tape_sha256"]) != (INPUT_MANIFEST_SHA256,
                                                 "ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d",
                                                 WX_TAPE_SHA256):
        raise RuntimeError("Registered input/feature/tape identity preflight failed")
    if (len(inputs) != 102 or sum(r["data_role"] == "D-fit" for r in inputs) != 54
            or sum(r["data_role"] == "D-dev" for r in inputs) != 48):
        raise RuntimeError("Registered input split is not exactly 54 D-fit + 48 D-dev")
    estimate = storage_estimate()
    if estimate["arrays_total_bytes_estimate"] > OUTPUT_CAP_BYTES:
        raise RuntimeError("Registered storage estimate exceeds the 1.5 GiB output cap")
    gpu_before = gpu_status()
    if gpu_before["gpu_name"] != "NVIDIA GeForce RTX 3090 Ti":
        raise RuntimeError(f"Unexpected registered GPU: {gpu_before['gpu_name']}")
    if gpu_before["memory_free_mib"] * 1024**2 < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB GPU memory is free; no model will be loaded")

    protected_before = protected_snapshot()
    backbones_before = backbone_identity()
    if backbones_before["dinov2_checkpoint_sha256"] != DINO2_CHECKPOINT_SHA256 or backbones_before["dinov2_source_tree_sha256"] != DINO2_SOURCE_TREE_SHA256:
        raise RuntimeError("DINOv2 checkpoint/source tree identity differs from the registered values")
    if backbones_before["dinov3_snapshot_revision"] != DINO3_REVISION or backbones_before["dinov3_snapshot_sha256"] != {
        "model.safetensors": DINO3_WEIGHTS_SHA256,
        "config.json": DINO3_CONFIG_SHA256,
        "preprocessor_config.json": DINO3_PROCESSOR_SHA256,
    }:
        raise RuntimeError("DINOv3 local snapshot identity differs from the registered values")
    code_before = source_code_identity()

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "protected_assets_before.json").write_text(json.dumps(protected_before, indent=2), encoding="utf-8")
    (OUT / "backbone_identity_before.json").write_text(json.dumps(backbones_before, indent=2), encoding="utf-8")
    provenance = {
        "schema": "d6.e2a1.run.v1",
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "execution_head": head,
        "branch": branch,
        "research_direction": "A_CONTINUE_FEASIBILITY",
        "input_manifest_sha256": input_summary["input_manifest_sha256"],
        "p1a_feature_manifest_sha256": input_summary["feature_manifest_sha256"],
        "wx_tape_sha256": input_summary["wx_tape_sha256"],
        "input_counts": {"total": 102, "D-fit": 54, "D-dev": 48},
        "a0_test_archival_status": a0_test_archival,
        "authorization_status_counts": input_summary["authorization_status_counts"],
        "exposure_status_counts": input_summary["exposure_status_counts"],
        "smoke_selection_rule": "first two D-fit rows in frozen P1-A input manifest order; labels not consulted",
        "preprocess_code_and_version_sha256": code_before,
        "preprocess_schema": "d6.e2a1.canonical_rgb_letterbox.v1",
        "feature_interface_schema": "d6.e2.shared_spatial_tokens.v1",
        "resize_and_exif_policy": "Pillow 9.3.0 RGB BILINEAR single-pass; round-half-up dimensions; EXIF transpose before size; odd padding remainder to right/bottom",
        "models": backbones_before,
        "precision": "FP32; autocast disabled; TF32 disabled",
        "batch_size": 1,
        "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
        "gpu_time_budget_seconds": MAX_RUN_SECONDS,
        "output_cap_bytes": OUTPUT_CAP_BYTES,
        "storage_estimate": estimate,
        "labels_read": False,
        "source_scope": "Only the 102 rows in the verified P1-A whitelist; no directory discovery",
        "protected_data_opened": False,
    }
    (OUT / "run_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "protected_assets_before.json").write_text(json.dumps(protected_before, indent=2), encoding="utf-8")

    command_results: list[dict[str, Any]] = []
    test_results = run_tests()
    for result in test_results:
        (OUT / f"{result['label']}.stdout.log").write_text(result["stdout"], encoding="utf-8")
        (OUT / f"{result['label']}.stderr.log").write_text(result["stderr"], encoding="utf-8")
        result.pop("stdout")
        result.pop("stderr")
    if any(result["return_code"] != 0 for result in test_results):
        raise RuntimeError(f"Pre-image unit tests failed: {test_results}")
    if source_code_identity() != code_before:
        raise RuntimeError("E2-A1 code changed after protocol identity was frozen")

    completed: dict[str, Any] = {"dinov2_smoke": None, "dinov3_smoke": None,
                                 "dinov2_full": None, "dinov3_full": None,
                                 "canonical_smoke": None, "canonical_full": None}
    try:
        remaining = deadline - time.perf_counter()
        command = run_child("canonical-smoke", DINO2_INTERPRETER,
                            SCRIPT_DIR / "prepare_canonical.py", ["--mode", "smoke"], remaining)
        command_results.append(command)
        if command["return_code"] != 0:
            raise RuntimeError("Two-image canonical preprocessing smoke failed")
        completed["canonical_smoke"] = "PASS"

        for key, backend, interpreter, script in [
            ("dinov2_smoke", "dinov2", DINO2_INTERPRETER, SCRIPT_DIR / "extract_dinov2.py"),
            ("dinov3_smoke", "dinov3", DINO3_INTERPRETER, SCRIPT_DIR / "extract_dinov3.py"),
        ]:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Two-hour GPU budget elapsed before smoke completed")
            command = run_child(key.replace("_", "-"), interpreter, script, ["--mode", "smoke"], deadline - time.perf_counter())
            command_results.append(command)
            if command["return_code"] != 0:
                raise RuntimeError(f"{backend} two-image GPU smoke failed; see private run log")
            report_path = OUT / backend / "smoke_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "E2A1_SMOKE_PASS" or report.get("processed_images") != 2:
                raise RuntimeError(f"{backend} smoke receipt failed validation")
            completed[key] = report

        if source_code_identity() != code_before:
            raise RuntimeError("E2-A1 code changed after smoke; stopping before full cache")
        command = run_child("canonical-all", DINO2_INTERPRETER,
                            SCRIPT_DIR / "prepare_canonical.py", ["--mode", "all"], deadline - time.perf_counter())
        command_results.append(command)
        if command["return_code"] != 0:
            raise RuntimeError("Full canonical preprocessing failed")
        completed["canonical_full"] = "PASS"

        for key, backend, interpreter, script in [
            ("dinov2_full", "dinov2", DINO2_INTERPRETER, SCRIPT_DIR / "extract_dinov2.py"),
            ("dinov3_full", "dinov3", DINO3_INTERPRETER, SCRIPT_DIR / "extract_dinov3.py"),
        ]:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Two-hour GPU budget elapsed before both caches completed")
            command = run_child(key.replace("_", "-"), interpreter, script, ["--mode", "all"], deadline - time.perf_counter())
            command_results.append(command)
            if command["return_code"] != 0:
                raise RuntimeError(f"{backend} full 102-image cache failed; see private run log")
            report_path = OUT / backend / "run_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "E2A1_BACKBONE_CACHE_PASS" or report.get("processed_images") != 102:
                raise RuntimeError(f"{backend} full cache receipt failed validation")
            completed[key] = report

        final_rows, final_input_summary = audit_registered_inputs()
        if final_input_summary != input_summary:
            raise RuntimeError("Registered source whitelist, manifests or source hashes changed during extraction")
        if source_code_identity() != code_before:
            raise RuntimeError("E2-A1 source code identity changed during extraction")
        cache_check = verify_cache(final_rows, {
            "dinov2": completed["dinov2_full"]["runtime_identity"],
            "dinov3": completed["dinov3_full"]["runtime_identity"],
        })
        protected_after = protected_snapshot()
        backbones_after = backbone_identity()
        (OUT / "protected_assets_after.json").write_text(json.dumps(protected_after, indent=2), encoding="utf-8")
        (OUT / "backbone_identity_after.json").write_text(json.dumps(backbones_after, indent=2), encoding="utf-8")
        if protected_after != protected_before:
            raise RuntimeError("Protected P1/E1/E2-A0 assets changed during E2-A1")
        if backbones_after != backbones_before:
            raise RuntimeError("Backbone source or weight identity changed during E2-A1")
        if sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) > OUTPUT_CAP_BYTES:
            raise RuntimeError("E2-A1 output exceeded registered 1.5 GiB cap")
        final = {
            "status": "E2A1_PASS",
            "execution_head": head,
            "branch": branch,
            "image_count": 102,
            "role_counts": {"D-fit": 54, "D-dev": 48},
            "canonical_tensor_count": 102,
            "dinov2_feature_count": completed["dinov2_full"]["processed_images"],
            "dinov3_feature_count": completed["dinov3_full"]["processed_images"],
            "model_run_reports": {
                "dinov2": "dinov2/run_report.json",
                "dinov3": "dinov3/run_report.json",
            },
            "cache_check": cache_check,
            "protected_assets_unchanged": True,
            "backbone_identity_unchanged": True,
            "source_manifest_unchanged": True,
            "source_labels_read": False,
            "full_feature_cache_created": True,
            "training_or_e2b": False,
            "output_bytes": sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()),
            "output_cap_bytes": OUTPUT_CAP_BYTES,
            "task_elapsed_seconds": time.perf_counter() - task_started,
            "test_results": test_results,
            "commands": command_results,
        }
        (OUT / "task_result.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in final.items() if key not in ("test_results", "commands")}, indent=2))
        return 0
    except Exception as exc:
        # Preserve a bounded local failure receipt. Successful per-image atomic
        # bundles remain available for identity-checked recovery, never overwrite.
        partial_counts = {}
        for backend in ("dinov2", "dinov3"):
            item_root = OUT / backend / "items"
            partial_counts[backend] = len([path for path in item_root.iterdir() if path.is_dir()]) if item_root.exists() else 0
        status = "E2A1_PARTIAL" if any(partial_counts.values()) else "E2A1_BLOCKED"
        failure = {
            "status": status,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "execution_head": head,
            "branch": branch,
            "partial_item_directory_counts": partial_counts,
            "commands": command_results,
            "source_labels_read": False,
            "training_or_e2b": False,
            "task_elapsed_seconds": time.perf_counter() - task_started,
        }
        failure_path = OUT / "task_failure.json"
        if not failure_path.exists():
            failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
