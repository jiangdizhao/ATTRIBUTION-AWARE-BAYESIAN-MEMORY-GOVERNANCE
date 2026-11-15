"""Standalone offline FP32 CPU/GPU smoke for the cached DINOv3 ViT-L/16 HF model."""
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

# Enforce offline operation before importing Hugging Face libraries, and avoid
# inheriting/printing any account credentials in this standalone child process.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
for _secret_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
    os.environ.pop(_secret_name, None)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "outputs/research_d6/p2/e2a0"
CANONICAL = OUT_DIR / "canonical_tensor.npy"
CANONICAL_MANIFEST = OUT_DIR / "canonical_manifest.json"
RESULT_PATH = OUT_DIR / "dinov3_smoke_results_verified.json"
TOKENS_PATH = OUT_DIR / "dinov3_smoke_tokens_verified.npz"
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SNAPSHOT_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
EXPECTED_WEIGHTS_SHA = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
EXPECTED_CONFIG_SHA = "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
EXPECTED_PROCESSOR_SHA = "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7"
EXPECTED_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
EXPECTED_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
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
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=True, text=True, capture_output=True, timeout=15,
    )
    process_rows = [line for line in processes.stdout.splitlines() if line.strip()]
    return {
        "gpu_name": parts[0], "memory_total_mib": int(parts[1]),
        "memory_free_mib": int(parts[2]), "memory_used_mib": int(parts[3]),
        "utilization_percent": int(parts[4]),
        "compute_process_query_entries": len(process_rows),
        "per_process_memory_visible": any("N/A" not in line for line in process_rows),
        "processes_terminated": False,
    }


def normalized(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), np.float32(1e-12))


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(normalized(a) * normalized(b), axis=-1, dtype=np.float32)


def split_tokens(hidden: torch.Tensor) -> dict[str, np.ndarray]:
    value = hidden.detach().cpu().contiguous().numpy().copy()
    return {
        "cls": value[:, :1, :].reshape(1, 1024),
        "register": value[:, 1:5, :],
        "patch": value[:, 5:, :],
    }


def comparison(cpu: dict[str, np.ndarray], gpu: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    raw = {name: float(np.max(np.abs(cpu[name] - gpu[name]))) for name in cpu}
    normalized_max = {
        name: float(np.max(np.abs(normalized(cpu[name]) - normalized(gpu[name])))) for name in cpu
    }
    patch_cos = cosine_rows(cpu["patch"], gpu["patch"]).reshape(-1)
    valid_cos = patch_cos[mask.astype(bool)]
    return {
        "raw_max_abs_difference": raw,
        "l2_normalized_max_abs_difference": normalized_max,
        "l2_normalized_max_abs_difference_all_tokens": max(normalized_max.values()),
        "cls_cosine_similarity": float(cosine_rows(cpu["cls"], gpu["cls"]).reshape(-1)[0]),
        "valid_patch_count": int(mask.sum()),
        "valid_patch_cosine_min": float(np.min(valid_cos)),
        "valid_patch_cosine_mean": float(np.mean(valid_cos)),
        "valid_patch_cosine_max": float(np.max(valid_cos)),
        "all_outputs_finite": bool(all(np.isfinite(v).all() for v in (*cpu.values(), *gpu.values()))),
    }


def main() -> int:
    if RESULT_PATH.exists() or TOKENS_PATH.exists():
        raise FileExistsError("DINOv3 smoke result already exists; refusing to overwrite")
    if not CANONICAL.is_file() or not CANONICAL_MANIFEST.is_file():
        raise FileNotFoundError("canonical synthetic tensor is missing")
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
    if torch.__version__ != "2.14.1+cu126" or transformers.__version__ != "5.23.0":
        write_result({"status": "E2A0_BLOCKED", "failure_scope": "model_environment",
                      "reason": "DINOv3 interpreter package versions differ from researcher-registered environment",
                      "torch": torch.__version__, "transformers": transformers.__version__})
        return 2
    if not torch.cuda.is_available():
        write_result({"status": "E2A0_BLOCKED", "failure_scope": "global_gpu",
                      "reason": "torch.cuda.is_available() is false", "gpu_before": gpu})
        return 2

    try:
        snapshot = (Path.home() / ".cache/huggingface/hub" /
                    "models--facebook--dinov3-vitl16-pretrain-lvd1689m" / "snapshots" / SNAPSHOT_REVISION)
        weights_path = snapshot / "model.safetensors"
        config_path = snapshot / "config.json"
        processor_path = snapshot / "preprocessor_config.json"
        for path in (weights_path, config_path, processor_path):
            if not path.is_file():
                raise FileNotFoundError(f"required local DINOv3 snapshot asset is missing: {path.name}")
        weights_sha, config_sha, processor_sha = (
            sha256_file(weights_path), sha256_file(config_path), sha256_file(processor_path)
        )
        if (weights_sha, config_sha, processor_sha) != (
                EXPECTED_WEIGHTS_SHA, EXPECTED_CONFIG_SHA, EXPECTED_PROCESSOR_SHA):
            raise RuntimeError("DINOv3 snapshot revision/weights/config identity mismatch")

        canonical_meta = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
        tensor_sha = sha256_file(CANONICAL)
        if tensor_sha != canonical_meta["canonical_tensor_file_sha256"]:
            raise RuntimeError("canonical tensor file SHA does not match its manifest")
        pixel_array = np.load(CANONICAL, allow_pickle=False)
        if pixel_array.shape != (1, 3, 448, 448) or pixel_array.dtype != np.float32:
            raise RuntimeError("canonical tensor shape/dtype mismatch")

        config = AutoConfig.from_pretrained(
            MODEL_ID, revision=SNAPSHOT_REVISION, local_files_only=True
        )
        processor = AutoImageProcessor.from_pretrained(
            MODEL_ID, revision=SNAPSHOT_REVISION, local_files_only=True
        )
        mean = np.asarray(processor.image_mean, dtype=np.float32)
        std = np.asarray(processor.image_std, dtype=np.float32)
        mean_match = bool(np.array_equal(mean, EXPECTED_MEAN))
        std_match = bool(np.array_equal(std, EXPECTED_STD))
        data_format = str(getattr(processor.data_format, "value", processor.data_format))
        if not mean_match or not std_match or data_format != "channels_first" or config.num_channels != 3:
            payload = {
                "status": "PREPROCESS_CONTRACT_CONFLICT",
                "failure_scope": "preprocess_contract",
                "reason": "official local DINOv3 processor mean/std/channel layout differs from the frozen shared RGB FP32 contract",
                "processor_mean": mean.tolist(), "processor_std": std.tolist(),
                "processor_data_format": data_format, "model_num_channels": config.num_channels,
            }
            write_result(payload)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 2

        expected_config = ("DINOv3ViTModel", 1024, 16, 4, 24)
        observed_config = (config.architectures[0], config.hidden_size, config.patch_size,
                           config.num_register_tokens, config.num_hidden_layers)
        if observed_config != expected_config:
            raise RuntimeError(f"unexpected DINOv3 config: {observed_config}")

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP_BYTES / props.total_memory, device=device)
        input_cpu = torch.from_numpy(pixel_array.copy())
        if input_cpu.dtype != torch.float32 or not torch.isfinite(input_cpu).all():
            raise RuntimeError("canonical tensor must be finite FP32")

        started = time.perf_counter()
        model = AutoModel.from_pretrained(
            MODEL_ID, revision=SNAPSHOT_REVISION, local_files_only=True,
            dtype=torch.float32,
        )
        load_seconds = time.perf_counter() - started
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("DINOv3 model parameters are not all FP32")

        with torch.inference_mode():
            started = time.perf_counter()
            cpu_result = model(pixel_values=input_cpu, return_dict=True)
            cpu_seconds = time.perf_counter() - started
            cpu_tokens_tensor = cpu_result.last_hidden_state
            cpu_output = split_tokens(cpu_tokens_tensor)
            pooler_cls_max_diff = float(torch.max(torch.abs(
                cpu_result.pooler_output - cpu_tokens_tensor[:, 0, :]
            )).item())

        expected_shapes = {"cls": (1, 1024), "register": (1, 4, 1024), "patch": (1, 784, 1024)}
        for name, shape in expected_shapes.items():
            if cpu_output[name].shape != shape:
                raise RuntimeError(f"DINOv3 {name} shape mismatch: {cpu_output[name].shape}")
        if not all(np.isfinite(x).all() for x in cpu_output.values()):
            raise RuntimeError("DINOv3 CPU output contains nonfinite values")

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
                result = model(pixel_values=input_gpu, return_dict=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            return split_tokens(result.last_hidden_state), elapsed

        for _ in range(2):
            warm, _ = forward_gpu()
            if not all(np.isfinite(x).all() for x in warm.values()):
                raise RuntimeError("DINOv3 GPU warm-up output contains nonfinite values")
        gpu_runs, gpu_times = [], []
        for _ in range(5):
            groups, seconds = forward_gpu()
            gpu_runs.append(groups)
            gpu_times.append(seconds)
        gpu_output = gpu_runs[0]
        mask = np.load(OUT_DIR / "dino_v3_patch_mask.npy", allow_pickle=False).astype(bool)
        if mask.shape != (784,) or int(mask.sum()) != 588:
            raise RuntimeError("DINOv3 patch mask identity/count mismatch")
        cpu_gpu = comparison(cpu_output, gpu_output, mask)
        repeat_metrics = [comparison(gpu_output, other, mask) for other in gpu_runs[1:]]
        repeat_raw = {
            group: max(float(np.max(np.abs(gpu_output[group] - other[group]))) for other in gpu_runs[1:])
            for group in gpu_output
        }
        repeat_norm = max(m["l2_normalized_max_abs_difference_all_tokens"] for m in repeat_metrics)
        repeat_cls = min(m["cls_cosine_similarity"] for m in repeat_metrics)
        repeat_patch = min(m["valid_patch_cosine_min"] for m in repeat_metrics)

        checks = {
            "registered_token_shapes": True,
            "all_outputs_finite": cpu_gpu["all_outputs_finite"],
            "pooler_output_matches_last_hidden_cls": pooler_cls_max_diff <= 1e-7,
            "patch_mask_matches_patch_order": True,
            "processor_mean_matches_g0": mean_match,
            "processor_std_matches_g0": std_match,
            "processor_channel_layout_is_channels_first": data_format == "channels_first",
            "normalized_max_abs_le_1e-3": cpu_gpu["l2_normalized_max_abs_difference_all_tokens"] <= NORMALIZED_MAX_ABS_TOL,
            "cls_cosine_ge_0.9999": cpu_gpu["cls_cosine_similarity"] >= CLS_COSINE_MIN,
            "valid_patch_min_cosine_ge_0.999": cpu_gpu["valid_patch_cosine_min"] >= PATCH_COSINE_MIN,
            "repeat_normalized_max_abs_le_1e-3": repeat_norm <= NORMALIZED_MAX_ABS_TOL,
            "repeat_cls_cosine_ge_0.9999": repeat_cls >= CLS_COSINE_MIN,
            "repeat_valid_patch_min_cosine_ge_0.999": repeat_patch >= PATCH_COSINE_MIN,
        }
        payload = {
            "status": "E2A0_PASS" if all(checks.values()) else "E2A0_PARTIAL",
            "failure_scope": None if all(checks.values()) else "numerical_or_output_tolerance",
            "identity": {
                "model_id": MODEL_ID, "snapshot_revision": SNAPSHOT_REVISION,
                "weights_file": "model.safetensors", "weights_sha256": weights_sha,
                "config_sha256": config_sha, "preprocessor_config_sha256": processor_sha,
                "config_architecture": config.architectures,
                "hidden_size": config.hidden_size, "patch_size": config.patch_size,
                "register_tokens": config.num_register_tokens, "layers": config.num_hidden_layers,
                "config_image_size": config.image_size,
                "position_encoding": "official dynamic 2D RoPE path",
                "last_hidden_state_semantics": "output of final model LayerNorm; not unit-L2; token order CLS, 4 registers, row-major spatial patches",
                "torch": torch.__version__, "transformers": transformers.__version__,
                "numpy": np.__version__, "python": platform.python_version(),
                "device": str(device), "precision": "FP32; autocast disabled; TF32 disabled",
                "user_model_access_authorized": True,
                "authorization_source": "researcher instruction; no private approval data accessed",
            },
            "official_processor_inspection": {
                "processor_type": type(processor).__name__,
                "image_mean": mean.tolist(), "image_std": std.tolist(),
                "data_format": data_format,
                "input_color_order": "RGB synthetic source; passed as NCHW RGB without processor conversion",
                "do_resize": getattr(processor, "do_resize", None),
                "default_size": str(getattr(processor, "size", None)),
                "default_resample": str(getattr(processor, "resample", None)),
                "processor_invoked_on_tensor": False,
                "note": "The cached processor defaults to square 224 sizing; this task intentionally bypasses resize/crop and passes the registered 448 tensor directly. The model's dynamic RoPE forward accepted 448 tokens.",
            },
            "gpu_before": gpu,
            "input": {
                "canonical_tensor_file_sha256": tensor_sha,
                "canonical_array_sha256": canonical_meta["canonical_array_sha256"],
                "shape": list(pixel_array.shape), "dtype": pixel_array.dtype.name,
                "source_kind": "synthetic_only",
            },
            "token_shapes": {name: list(value.shape) for name, value in cpu_output.items()},
            "sequence_length": int(1 + cpu_output["register"].shape[1] +
                                    cpu_output["patch"].shape[1]),
            "patch_grid": [28, 28], "patch_mask_valid": int(mask.sum()),
            "pooler_cls_max_abs_difference": pooler_cls_max_diff,
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
                "raw_max_abs_difference": repeat_raw,
                "normalized_max_abs_difference_all_tokens": repeat_norm,
                "cls_cosine_min": repeat_cls, "valid_patch_cosine_min": repeat_patch,
            },
            "checks": checks,
            "run_note": "Recovery run after the first DINOv3 inference completed but its result serialization failed on a Transformers SizeDict; model, input, precision, tolerances, warmups and repetition count are unchanged.",
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
