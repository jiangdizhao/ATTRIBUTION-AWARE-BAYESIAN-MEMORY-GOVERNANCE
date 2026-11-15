"""Standalone offline FP32 CPU/GPU smoke for the registered DINOv2 ViT-L/14 Reg model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import torchvision
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "outputs/research_d6/p2/e2a0"
CANONICAL = OUT_DIR / "canonical_tensor.npy"
CANONICAL_MANIFEST = OUT_DIR / "canonical_manifest.json"
RESULT_PATH = OUT_DIR / "dinov2_smoke_results.json"
TOKENS_PATH = OUT_DIR / "dinov2_smoke_tokens.npz"
EXPECTED_CHECKPOINT_SHA = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
EXPECTED_SOURCE_TREE_SHA = "7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c"
MEMORY_GATE_BYTES = 10 * 1024**3
ALLOCATOR_CAP_BYTES = 12 * 1024**3
NORMALIZED_MAX_ABS_TOL = 1e-3
CLS_COSINE_MIN = 0.9999
PATCH_COSINE_MIN = 0.999


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def mi_b(value: str) -> int:
    return int(value.strip().split()[0])


def gpu_preflight() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=15,
    )
    lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one target GPU, found {len(lines)}")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 5:
        raise RuntimeError("could not parse nvidia-smi GPU query")
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], check=True, text=True, capture_output=True, timeout=15,
    )
    process_rows = [line for line in process_query.stdout.splitlines() if line.strip()]
    # WDDM may enumerate processes while withholding per-process GPU memory. Names
    # are deliberately not copied into the persisted report.
    return {
        "gpu_name": parts[0], "memory_total_mib": int(parts[1]),
        "memory_free_mib": int(parts[2]), "memory_used_mib": int(parts[3]),
        "utilization_percent": int(parts[4]),
        "compute_process_query_entries": len(process_rows),
        "per_process_memory_visible": any("N/A" not in line for line in process_rows),
        "processes_terminated": False,
    }


def to_numpy_groups(output: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    keys = {"cls": "x_norm_clstoken", "register": "x_norm_regtokens", "patch": "x_norm_patchtokens"}
    result = {name: output[key].detach().cpu().contiguous().numpy().copy() for name, key in keys.items()}
    return result


def normalized(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), np.float32(1e-12))
    return x / denom


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an, bn = normalized(a), normalized(b)
    return np.sum(an * bn, axis=-1, dtype=np.float32)


def comparison(cpu: dict[str, np.ndarray], gpu: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    raw = {name: float(np.max(np.abs(cpu[name] - gpu[name]))) for name in cpu}
    normed_cpu = {name: normalized(value) for name, value in cpu.items()}
    normed_gpu = {name: normalized(value) for name, value in gpu.items()}
    normalized_max_abs = {
        name: float(np.max(np.abs(normed_cpu[name] - normed_gpu[name]))) for name in cpu
    }
    patch_cos = cosine_rows(cpu["patch"], gpu["patch"]).reshape(-1)
    valid_cos = patch_cos[mask.astype(bool)]
    cls_cos = float(cosine_rows(cpu["cls"], gpu["cls"]).reshape(-1)[0])
    all_norm_max = max(normalized_max_abs.values())
    return {
        "raw_max_abs_difference": raw,
        "l2_normalized_max_abs_difference": normalized_max_abs,
        "l2_normalized_max_abs_difference_all_tokens": all_norm_max,
        "cls_cosine_similarity": cls_cos,
        "valid_patch_count": int(mask.sum()),
        "valid_patch_cosine_min": float(np.min(valid_cos)),
        "valid_patch_cosine_mean": float(np.mean(valid_cos)),
        "valid_patch_cosine_max": float(np.max(valid_cos)),
        "all_outputs_finite": bool(all(np.isfinite(v).all() for v in (*cpu.values(), *gpu.values()))),
    }


def main() -> int:
    require_no_output = not RESULT_PATH.exists() and not TOKENS_PATH.exists()
    if not require_no_output:
        raise FileExistsError("DINOv2 smoke result already exists; refusing to overwrite")
    if not CANONICAL.is_file() or not CANONICAL_MANIFEST.is_file():
        raise FileNotFoundError("canonical synthetic tensor is missing")

    # These gates precede CUDA initialization, model loading, or GPU allocation.
    try:
        gpu = gpu_preflight()
    except Exception as exc:
        write_result({"status": "E2A0_BLOCKED", "failure_scope": "global_gpu",
                      "reason": f"GPU preflight failed: {type(exc).__name__}: {exc}"})
        return 2
    if gpu["memory_free_mib"] * 1024**2 < MEMORY_GATE_BYTES:
        write_result({"status": "E2A0_BLOCKED", "failure_scope": "global_gpu",
                      "reason": "less than 10 GiB free VRAM before model load", "gpu_before": gpu})
        return 2
    if torch.__version__ != "2.0.1+cu117" or torchvision.__version__ != "0.15.2+cu117":
        write_result({"status": "E2A0_BLOCKED", "failure_scope": "model_environment",
                      "reason": "DINOv2 interpreter package versions differ from registered environment",
                      "torch": torch.__version__, "torchvision": torchvision.__version__})
        return 2
    if not torch.cuda.is_available():
        write_result({"status": "E2A0_BLOCKED", "failure_scope": "global_gpu",
                      "reason": "torch.cuda.is_available() is false", "gpu_before": gpu})
        return 2

    try:
        p1a_path = ROOT / "scripts/research_d6/p1a/extract_features.py"
        spec = importlib.util.spec_from_file_location("e2a0_p1a_identity", p1a_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load P1-A identity loader")
        p1a = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p1a)
        checkpoint_sha = p1a.sha256_file(p1a.CHECKPOINT)
        source_sha, source_files, source_bytes = p1a.tree_sha256(p1a.HUB_SOURCE)
        if checkpoint_sha != EXPECTED_CHECKPOINT_SHA or source_sha != EXPECTED_SOURCE_TREE_SHA:
            raise RuntimeError("DINOv2 local checkpoint/source identity differs from P1-A registered identity")

        manifest = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
        canonical_sha_file = sha256_file(CANONICAL)
        if canonical_sha_file != manifest["canonical_tensor_file_sha256"]:
            raise RuntimeError("canonical tensor file SHA does not match its manifest")
        pixel_array = np.load(CANONICAL, allow_pickle=False)
        if pixel_array.shape != (1, 3, 448, 448) or pixel_array.dtype != np.float32:
            raise RuntimeError("canonical tensor shape/dtype mismatch")

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP_BYTES / props.total_memory, device=device)
        input_cpu = torch.from_numpy(pixel_array.copy())
        if input_cpu.dtype != torch.float32 or not torch.isfinite(input_cpu).all():
            raise RuntimeError("canonical tensor must be finite FP32")

        scripts_dir = ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        started = time.perf_counter()
        sensor = p1a.load_frozen_sensor(torch, device="cpu")
        load_seconds = time.perf_counter() - started
        model = sensor.backbone.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("DINOv2 model parameters are not all FP32")

        with threadpool_limits(limits=2), torch.inference_mode():
            started = time.perf_counter()
            cpu_output = to_numpy_groups(model.forward_features(input_cpu))
            cpu_seconds = time.perf_counter() - started

        expected_shapes = {"cls": (1, 1024), "register": (1, 4, 1024), "patch": (1, 1024, 1024)}
        for name, shape in expected_shapes.items():
            if cpu_output[name].shape != shape:
                raise RuntimeError(f"DINOv2 {name} shape mismatch: {cpu_output[name].shape}")
        if not all(np.isfinite(x).all() for x in cpu_output.values()):
            raise RuntimeError("DINOv2 CPU output contains nonfinite values")

        started = time.perf_counter()
        model = model.to(device)
        input_gpu = input_cpu.to(device)
        torch.cuda.synchronize(device)
        transfer_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats(device)

        def forward_gpu() -> tuple[dict[str, np.ndarray], float]:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = model.forward_features(input_gpu)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            return to_numpy_groups(output), elapsed

        for _ in range(2):
            warm, _ = forward_gpu()
            if not all(np.isfinite(x).all() for x in warm.values()):
                raise RuntimeError("DINOv2 GPU warm-up output contains nonfinite values")
        gpu_runs, gpu_times = [], []
        for _ in range(5):
            groups, seconds = forward_gpu()
            gpu_runs.append(groups)
            gpu_times.append(seconds)
        gpu_output = gpu_runs[0]
        mask = np.load(OUT_DIR / "dino_v2_patch_mask.npy", allow_pickle=False).astype(bool)
        if mask.shape != (1024,) or int(mask.sum()) != 768:
            raise RuntimeError("DINOv2 patch mask identity/count mismatch")
        cpu_gpu = comparison(cpu_output, gpu_output, mask)
        repeat_metrics = [comparison(gpu_output, other, mask) for other in gpu_runs[1:]]
        repeat_max_abs = {
            group: max(float(np.max(np.abs(gpu_output[group] - other[group]))) for other in gpu_runs[1:])
            for group in gpu_output
        }
        repeat_norm_max_abs = max(m["l2_normalized_max_abs_difference_all_tokens"] for m in repeat_metrics)
        repeat_cls_cos_min = min(m["cls_cosine_similarity"] for m in repeat_metrics)
        repeat_patch_cos_min = min(m["valid_patch_cosine_min"] for m in repeat_metrics)

        checks = {
            "registered_token_shapes": True,
            "all_outputs_finite": cpu_gpu["all_outputs_finite"],
            "patch_mask_matches_patch_order": True,
            "normalized_max_abs_le_1e-3": cpu_gpu["l2_normalized_max_abs_difference_all_tokens"] <= NORMALIZED_MAX_ABS_TOL,
            "cls_cosine_ge_0.9999": cpu_gpu["cls_cosine_similarity"] >= CLS_COSINE_MIN,
            "valid_patch_min_cosine_ge_0.999": cpu_gpu["valid_patch_cosine_min"] >= PATCH_COSINE_MIN,
            "repeat_normalized_max_abs_le_1e-3": repeat_norm_max_abs <= NORMALIZED_MAX_ABS_TOL,
            "repeat_cls_cosine_ge_0.9999": repeat_cls_cos_min >= CLS_COSINE_MIN,
            "repeat_valid_patch_min_cosine_ge_0.999": repeat_patch_cos_min >= PATCH_COSINE_MIN,
        }
        identity = {
            "model": "dinov2_vitl14_reg",
            "checkpoint_sha256": checkpoint_sha,
            "hub_source_tree_sha256": source_sha,
            "hub_source_file_count": source_files,
            "hub_source_total_bytes": source_bytes,
            "p1a_extractor_code_sha256": sha256_file(p1a_path),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "device": str(device), "precision": "FP32; autocast disabled; TF32 disabled",
            "patch_size": 14, "model_embed_dim": int(model.embed_dim),
            "final_output_norm": "official forward_features x_norm_* after model final LayerNorm; these are not unit-L2 tokens",
        }
        payload = {
            "status": "E2A0_PASS" if all(checks.values()) else "E2A0_PARTIAL",
            "failure_scope": None if all(checks.values()) else "numerical_or_output_tolerance",
            "identity": identity,
            "gpu_before": gpu,
            "input": {
                "canonical_tensor_file_sha256": canonical_sha_file,
                "canonical_array_sha256": manifest["canonical_array_sha256"],
                "shape": list(pixel_array.shape), "dtype": pixel_array.dtype.name,
                "source_kind": "synthetic_only",
            },
            "token_shapes": {name: list(value.shape) for name, value in cpu_output.items()},
            "patch_grid": [32, 32], "patch_mask_valid": int(mask.sum()),
            "processor_path": "none; canonical NCHW RGB FP32 tensor passed directly to forward_features",
            "load_seconds": load_seconds, "cpu_forward_seconds": cpu_seconds,
            "cpu_to_gpu_transfer_seconds": transfer_seconds,
            "gpu_warmup_runs": 2, "gpu_timed_runs": len(gpu_runs),
            "gpu_forward_seconds": {"min": float(min(gpu_times)), "median": float(np.median(gpu_times)),
                                    "mean": float(np.mean(gpu_times)), "max": float(max(gpu_times))},
            "gpu_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
            "cpu_vs_gpu": cpu_gpu,
            "gpu_repeat_consistency": {
                "compared_repeats": len(gpu_runs) - 1,
                "raw_max_abs_difference": repeat_max_abs,
                "normalized_max_abs_difference_all_tokens": repeat_norm_max_abs,
                "cls_cosine_min": repeat_cls_cos_min,
                "valid_patch_cosine_min": repeat_patch_cos_min,
            },
            "checks": checks,
        }
        np.savez_compressed(TOKENS_PATH,
                            cpu_cls=cpu_output["cls"], cpu_register=cpu_output["register"], cpu_patch=cpu_output["patch"],
                            gpu_cls=gpu_output["cls"], gpu_register=gpu_output["register"], gpu_patch=gpu_output["patch"])
        write_result(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if all(checks.values()) else 1
    except torch.cuda.OutOfMemoryError as exc:
        payload = {"status": "E2A0_BLOCKED", "failure_scope": "global_gpu",
                   "reason": f"CUDA out of memory under registered FP32/input/cap; no fallback used: {exc}",
                   "gpu_before": gpu}
        write_result(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2
    except Exception as exc:
        payload = {"status": "E2A0_BLOCKED", "failure_scope": "model_local",
                   "reason": f"{type(exc).__name__}: {exc}", "gpu_before": gpu}
        write_result(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
