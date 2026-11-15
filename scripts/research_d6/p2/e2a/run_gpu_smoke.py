"""Orchestrate isolated E2-A0 preprocessing/tests and the two offline GPU smokes."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
EXPECTED_HEAD = "2d33f899c7c17157376b8be1bf9c311114ff2344"
EXPECTED_BRANCH = "stage3-eg-famar-execution"
OUT_DIR = ROOT / "outputs/research_d6/p2/e2a0"
PY_A = Path(r"D:\anaconda3\envs\torch\python.exe")
PY_B = Path(r"D:\anaconda3\envs\dinov3\python.exe")
GPU_GATE_BYTES = 10 * 1024**3
SNAPSHOT_PATHS = [
    "outputs/research_d6/p1a_features",
    "outputs/research_d6/p1b/run_20261113_01",
    "outputs/research_d6/p1b_r1",
    "outputs/research_d6/p2/e1/run_20261115_r1_01",
    "outputs/research_d6/p2/e1_r2",
    "docs/research_d6/p1a",
    "docs/research_d6/p1b",
    "docs/research_d6/p1b_r1",
    "docs/research_d6/p2/g0",
    "docs/research_d6/p2/e1_r1",
    "docs/research_d6/p2/e1_r2",
    "scripts/research_d6/p1a",
    "scripts/research_d6/p1b",
    "scripts/research_d6/p2/e1",
]
DINO2_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
DINO2_SOURCE = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
DINO3_SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m"
                  / "snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": 0, "bytes": 0, "sha256": ""}
    if path.is_file():
        items = [path]
        base = path.parent
    else:
        items = sorted(p for p in path.rglob("*") if p.is_file() and
                       "__pycache__" not in p.parts and ".git" not in p.parts and p.suffix != ".pyc")
        base = path
    digest = hashlib.sha256()
    total = 0
    for item in items:
        file_hash = sha256_file(item)
        total += item.stat().st_size
        digest.update(item.relative_to(base).as_posix().encode("utf-8") + b"\0" +
                      file_hash.encode("ascii") + b"\n")
    return {"exists": True, "files": len(items), "bytes": total, "sha256": digest.hexdigest()}


def protected_snapshot() -> dict[str, Any]:
    return {rel: tree_manifest(ROOT / rel) for rel in SNAPSHOT_PATHS}


def backbone_identity() -> dict[str, Any]:
    dino3_assets = {name: sha256_file(DINO3_SNAPSHOT / name)
                    for name in ("model.safetensors", "config.json", "preprocessor_config.json")}
    return {
        "dinov2_checkpoint_sha256": sha256_file(DINO2_CHECKPOINT),
        "dinov2_source_tree": tree_manifest(DINO2_SOURCE),
        "dinov3_snapshot_revision": DINO3_SNAPSHOT.name,
        "dinov3_snapshot_assets_sha256": dino3_assets,
    }


def gpu_status() -> dict[str, Any]:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        cwd=ROOT, check=True, text=True, capture_output=True, timeout=15,
    )
    lines = [line.strip() for line in q.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected a single GPU, found {len(lines)}")
    parts = [x.strip() for x in lines[0].split(",")]
    p = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        cwd=ROOT, check=True, text=True, capture_output=True, timeout=15,
    )
    process_rows = [x for x in p.stdout.splitlines() if x.strip()]
    return {
        "gpu_name": parts[0], "memory_total_mib": int(parts[1]),
        "memory_free_mib": int(parts[2]), "memory_used_mib": int(parts[3]),
        "utilization_percent": int(parts[4]),
        "compute_process_query_entries": len(process_rows),
        "per_process_memory_visible": any("N/A" not in x for x in process_rows),
        "processes_terminated": False,
    }


def child_env(offline: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        env.pop(key, None)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return env


def run_child(label: str, interpreter: Path, script: Path, output_base: str,
              offline: bool = False, timeout: int = 600) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        [str(interpreter), str(script)], cwd=ROOT, env=child_env(offline),
        text=True, capture_output=True, timeout=timeout,
    )
    (OUT_DIR / f"{output_base}_stdout.log").write_text(result.stdout, encoding="utf-8")
    (OUT_DIR / f"{output_base}_stderr.log").write_text(result.stderr, encoding="utf-8")
    return {
        "label": label, "interpreter": str(interpreter), "script": script.relative_to(ROOT).as_posix(),
        "return_code": result.returncode, "elapsed_wall_seconds": time.perf_counter() - started,
        "stdout_log": f"{output_base}_stdout.log", "stderr_log": f"{output_base}_stderr.log",
    }


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != EXPECTED_HEAD or branch != EXPECTED_BRANCH:
        raise RuntimeError(f"unexpected execution identity: {branch}/{head}")
    if OUT_DIR.exists():
        raise FileExistsError("E2-A0 output directory already exists; refusing to overwrite")
    if not PY_A.is_file() or not PY_B.is_file():
        raise FileNotFoundError("one or both researcher-specified Conda interpreters are missing")

    # Read-only identity audit before any run; no source image or private label is opened.
    before = protected_snapshot()
    weights_before = backbone_identity()
    p1a_prov = json.loads((ROOT / "outputs/research_d6/p1a_features/run_provenance.json").read_text(encoding="utf-8"))
    if p1a_prov.get("checkpoint_sha256") != weights_before["dinov2_checkpoint_sha256"]:
        raise RuntimeError("DINOv2 checkpoint hash differs from P1-A provenance")
    if p1a_prov.get("hub_source_tree_sha256") != weights_before["dinov2_source_tree"]["sha256"]:
        raise RuntimeError("DINOv2 source tree hash differs from P1-A provenance")

    initial_gpu = gpu_status()
    if initial_gpu["memory_free_mib"] * 1024**2 < GPU_GATE_BYTES:
        raise RuntimeError("E2-A0 stopped before model loading: free VRAM is below 10 GiB")

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUT_DIR / "protected_assets_before.json").write_text(
        json.dumps(before, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "backbone_identity_before.json").write_text(
        json.dumps(weights_before, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    command_log: list[dict[str, Any]] = []
    preprocess = run_child("canonical-preprocess", PY_A, ROOT / "scripts/research_d6/p2/e2a/canonical_preprocess.py",
                           "canonical_preprocess", timeout=120)
    command_log.append(preprocess)
    if preprocess["return_code"] != 0:
        final_status = "E2A0_BLOCKED"
        stop_reason = "canonical preprocessing process failed"
    else:
        test_started = time.perf_counter()
        test_result = subprocess.run(
            [str(PY_A), "-m", "unittest", "discover", "-s", str(ROOT / "tests/research_d6"),
             "-p", "test_e2a0_canonical_preprocess.py"],
            cwd=ROOT, env=child_env(), text=True, capture_output=True, timeout=120,
        )
        (OUT_DIR / "canonical_unit_tests_stdout.log").write_text(test_result.stdout, encoding="utf-8")
        (OUT_DIR / "canonical_unit_tests_stderr.log").write_text(test_result.stderr, encoding="utf-8")
        unit = {"label": "canonical-unit-tests", "interpreter": str(PY_A),
                "return_code": test_result.returncode,
                "elapsed_wall_seconds": time.perf_counter() - test_started,
                "stdout_log": "canonical_unit_tests_stdout.log",
                "stderr_log": "canonical_unit_tests_stderr.log"}
        command_log.append(unit)
        if test_result.returncode != 0:
            final_status = "E2A0_BLOCKED"
            stop_reason = "canonical preprocessing unit tests failed; no model was loaded"
        else:
            # The two backbones are run in independent interpreters, sequentially.
            d2 = run_child("dinov2", PY_A, ROOT / "scripts/research_d6/p2/e2a/smoke_dinov2.py",
                           "dinov2", timeout=600)
            command_log.append(d2)
            d2_result = read_json(OUT_DIR / "dinov2_smoke_results.json")
            if d2_result and d2_result.get("failure_scope") == "global_gpu":
                final_status = "E2A0_BLOCKED"
                stop_reason = "DINOv2 reported a global GPU gate/CUDA/OOM blocker; DINOv3 was not started"
                d3_result = None
            else:
                d3 = run_child("dinov3", PY_B, ROOT / "scripts/research_d6/p2/e2a/smoke_dinov3.py",
                               "dinov3", offline=True, timeout=600)
                command_log.append(d3)
                d3_result = read_json(OUT_DIR / "dinov3_smoke_results.json")
                stop_reason = None
                statuses = [d2_result.get("status") if d2_result else "NO_RESULT",
                            d3_result.get("status") if d3_result else "NO_RESULT"]
                if statuses == ["E2A0_PASS", "E2A0_PASS"]:
                    final_status = "E2A0_PASS"
                elif any(s in ("E2A0_BLOCKED", "PREPROCESS_CONTRACT_CONFLICT") for s in statuses):
                    final_status = "E2A0_BLOCKED"
                    stop_reason = "one or both backbone smokes are blocked; see per-model result files"
                else:
                    final_status = "E2A0_PARTIAL"

    # Read-only post audit; hashes cover P1/E1/G0/R1/R2 assets and cached backbone identity.
    after = protected_snapshot()
    weights_after = backbone_identity()
    protected_unchanged = before == after
    backbone_unchanged = weights_before == weights_after
    (OUT_DIR / "protected_assets_after.json").write_text(
        json.dumps(after, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "backbone_identity_after.json").write_text(
        json.dumps(weights_after, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not protected_unchanged or not backbone_unchanged:
        final_status = "E2A0_BLOCKED"
        stop_reason = "protected research assets or backbone files changed during smoke"

    final = {
        "status": final_status,
        "failure_or_stop_reason": stop_reason,
        "execution_head": head, "branch": branch,
        "driver_python": sys.executable, "driver_python_version": platform.python_version(),
        "environment_a": str(PY_A), "environment_b": str(PY_B),
        "initial_gpu": initial_gpu,
        "pre_smoke_snapshot_count": len(before),
        "protected_assets_unchanged": protected_unchanged,
        "backbone_assets_unchanged": backbone_unchanged,
        "processes_terminated": False,
        "network_enabled_for_hf": False,
        "full_102_image_features_created": False,
        "commands": command_log,
        "model_results": {
            "dinov2": read_json(OUT_DIR / "dinov2_smoke_results.json"),
            "dinov3": read_json(OUT_DIR / "dinov3_smoke_results.json"),
        },
    }
    (OUT_DIR / "run_gpu_smoke_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in final.items() if k != "model_results"}, indent=2, ensure_ascii=False))
    return 0 if final_status == "E2A0_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
