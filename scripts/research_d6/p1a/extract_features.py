#!/usr/bin/env python3
"""Restartable, low-resource D6-P1-A extraction of frozen global DINO CLS features.

The default device is CPU and masks CUDA before importing PyTorch. The optional
guarded-cuda mode is explicit and never selected automatically. No labels are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "research_d6" / "p1a_features"
HUB_SOURCE = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
CHECKPOINT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
MODEL_NAME = "dinov2_vitl14_reg"
LAYERS = tuple(range(4, 19))
RESIZE_SIZE = 448
CROP_SIZE = 392
BATCH_SIZE = 1
OUTPUT_DTYPE = "float32"
SMOKE_ATOL = 1e-6
SMOKE_RTOL = 1e-5

INPUT_FIELDS = [
    "image_id", "data_role", "dataset", "dataset_revision", "product",
    "sample_key", "view", "source_split", "source_relpath", "image_path",
    "source_sha256", "source_image_kind", "authorization_status", "exposure_status",
]
FEATURE_FIELDS = [
    "image_id", "data_role", "dataset", "dataset_revision", "source_sha256",
    "feature_relpath", "feature_sha256", "feature_dtype", "feature_shape",
    "device", "precision", "checkpoint_sha256", "hub_source_tree_sha256",
    "reference_code_sha256", "adapter_sha256", "preprocess_fingerprint",
    "completion_status", "completed_utc",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, stable_json(value) + b"\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    from io import StringIO

    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buf.getvalue().encode("utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tree_sha256(root: Path) -> tuple[str, int, int]:
    if not root.is_dir():
        raise FileNotFoundError(f"Torch Hub source directory is absent: {root}")
    entries = sorted(
        p for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts and p.suffix != ".pyc"
    )
    digest = hashlib.sha256()
    total = 0
    for path in entries:
        file_hash = sha256_file(path)
        total += path.stat().st_size
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel + b"\0" + file_hash.encode("ascii") + b"\n")
    return digest.hexdigest(), len(entries), total


def read_csv(path: Path) -> list[dict[str, str]]:
    return csv_rows(path)


def make_input_rows(root: Path = ROOT) -> list[dict[str, str]]:
    """Build the exact 54 D-fit + 48 D-dev white-list without reading annotation files."""
    pilot = root / "research_pilot"
    d4l_rows = read_csv(pilot / "LP_W1_D4LITE" / "pilot_manifest.csv")
    d4b_rows = read_csv(pilot / "LP_W1_D4B" / "source_manifest.csv")
    ddev_rows = read_csv(pilot / "LP_W1_D5R_IMAGE_PILOT_V2_2V" / "selection_manifest.csv")
    exposure_rows = read_csv(pilot / "LP_W1_D5R_IMAGE_PILOT_V2_2V" / "exposure_ledger.csv")
    exp_by_id = {r["item_id"]: r for r in exposure_rows}
    if len(d4l_rows) != 36 or len(d4b_rows) != 18 or len(ddev_rows) != 48:
        raise ValueError(f"Expected 36 D4-Lite, 18 D4B, 48 D-dev rows; got {len(d4l_rows)}, {len(d4b_rows)}, {len(ddev_rows)}")

    rows: list[dict[str, str]] = []
    for record, folder, sha_field, auth_field in [
        *((r, pilot / "LP_W1_D4LITE", "sha256", "future_eval_use") for r in d4l_rows),
        *((r, pilot / "LP_W1_D4B", "actual_sha256", "future_use") for r in d4b_rows),
    ]:
        if record.get("synthetic", "").lower() != "false" or record.get("image_kind") != "original_real_image":
            raise ValueError(f"D-fit row is not an authorized original real image: {record.get('item_id')}")
        image_path = (folder / record["local_image"]).resolve()
        rows.append({
            "image_id": record["item_id"], "data_role": "D-fit", "dataset": record["dataset"],
            "dataset_revision": record["dataset_revision"], "product": record["product"],
            "sample_key": "", "view": "", "source_split": record["source_partition"],
            "source_relpath": record["source_path"], "image_path": str(image_path),
            "source_sha256": record[sha_field].lower(), "source_image_kind": record["image_kind"],
            "authorization_status": record[auth_field], "exposure_status": "PILOT_EXPOSED",
        })

    for record in ddev_rows:
        exposure = exp_by_id.get(record["item_id"])
        if not exposure or exposure["exposure_status"] != "NEW_STUDY_PILOT_EXPOSED":
            raise ValueError(f"D-dev image is absent from the authorized exposure ledger: {record['item_id']}")
        if (record["sha256"].lower() != exposure["sha256"].lower()
                or record["sample_key"] != exposure["sample_key"]
                or record["view_id"] != exposure["view_id"]):
            raise ValueError(f"D-dev selection/exposure identity mismatch: {record['item_id']}")
        rows.append({
            "image_id": record["item_id"], "data_role": "D-dev", "dataset": exposure["dataset"],
            "dataset_revision": exposure["revision"], "product": record["product"],
            "sample_key": record["sample_key"], "view": record["view_id"],
            "source_split": record["source_split"], "source_relpath": record["image_relpath"],
            "image_path": str(Path(record["image_path"]).resolve()), "source_sha256": record["sha256"].lower(),
            "source_image_kind": "original_real_image", "authorization_status": record["authorization_status"],
            "exposure_status": exposure["exposure_status"],
        })

    ids = [r["image_id"] for r in rows]
    if len(rows) != 102 or len(set(ids)) != 102:
        raise ValueError(f"White-list must have 102 distinct image IDs; got {len(rows)} rows / {len(set(ids))} unique")
    if any(r["dataset"] not in {"aristrops/mvtec-concepts", "Real-IAD"} for r in rows):
        raise ValueError("White-list contains an unapproved data source")
    return rows


def validate_input_rows(rows: list[dict[str, str]], verify_sha: bool) -> None:
    if len(rows) != 102 or len({r["image_id"] for r in rows}) != 102:
        raise ValueError("Expected exactly 102 unique registered image IDs")
    counts = {role: sum(r["data_role"] == role for r in rows) for role in ("D-fit", "D-dev")}
    if counts != {"D-fit": 54, "D-dev": 48}:
        raise ValueError(f"Wrong role counts: {counts}")
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Authorized source image is missing: {row['image_id']}")
        if row["source_image_kind"] != "original_real_image":
            raise ValueError(f"Non-original image in white-list: {row['image_id']}")
        if verify_sha:
            actual = sha256_file(path)
            if actual.lower() != row["source_sha256"].lower():
                raise ValueError(f"Source SHA-256 mismatch for {row['image_id']}: {actual}")


def prepare_input_manifest(out: Path) -> Path:
    rows = make_input_rows()
    validate_input_rows(rows, verify_sha=True)
    manifest_path = out / "input_manifest.csv"
    if manifest_path.exists():
        existing = manifest_path.read_bytes()
        from io import StringIO

        buf = StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=INPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        if existing != buf.getvalue().encode("utf-8"):
            raise FileExistsError("Existing input_manifest.csv differs; refusing to overwrite")
    else:
        atomic_csv(manifest_path, rows, INPUT_FIELDS)
    return manifest_path


def source_code_sha256() -> dict[str, str]:
    source_root = ROOT / "scripts"
    return {
        "reference_encoder": sha256_file(source_root / "vmb_visionad_new.py"),
        "stage0_adapter": sha256_file(source_root / "abmg_stage0_foundation.py"),
        "p1a_adapter": sha256_file(Path(__file__).resolve()),
    }


class PeakRSS:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        process = psutil.Process()
        self.peak = process.memory_info().rss

        def sample() -> None:
            while not self.stop_event.wait(self.interval):
                try:
                    self.peak = max(self.peak, process.memory_info().rss)
                except Exception:
                    pass

        self.thread = threading.Thread(target=sample, name="p1a-rss-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> int:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        try:
            import psutil
            self.peak = max(self.peak, psutil.Process().memory_info().rss)
        except ImportError:
            pass
        return self.peak


def nvidia_smi_free_bytes() -> tuple[int, int]:
    command = ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    first = result.stdout.strip().splitlines()[0].split(",")
    return int(first[0].strip()) * 1024 * 1024, int(first[1].strip()) * 1024 * 1024


def configure_torch(device_name: str, threads: int):
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if device_name == "cpu":
        device = torch.device("cpu")
        return torch, device, None
    free_before, total_before = nvidia_smi_free_bytes()
    if free_before < 8 * 1024**3:
        raise RuntimeError(f"Guarded CUDA refused: nvidia-smi free memory is below 8 GiB ({free_before} bytes)")
    if not torch.cuda.is_available():
        raise RuntimeError("Guarded CUDA refused: torch reports CUDA unavailable")
    device = torch.device("cuda:0")
    total_torch = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, (3.5 * 1024**3) / total_torch)
    setter = getattr(torch.cuda, "set_per_process_memory_fraction", None)
    if setter is None:
        raise RuntimeError("Guarded CUDA refused: allocator limit API is unavailable")
    setter(fraction, device=device)
    return torch, device, {"free_before_bytes": free_before, "total_before_bytes": total_before}


def load_frozen_sensor(torch, device):
    """Load only from the cached local hub snapshot and cached checkpoint; never download."""
    if not HUB_SOURCE.is_dir() or not CHECKPOINT.is_file():
        raise FileNotFoundError("Cached DINO source/checkpoint is missing; network download is disabled")
    sys.path.insert(0, str(ROOT / "scripts"))
    from abmg_stage0_foundation import PaperAlignedFrozenSensor, Stage0Config

    original_load = torch.hub.load
    original_state_loader = torch.hub.load_state_dict_from_url

    def local_only_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != "facebookresearch/dinov2" or model != MODEL_NAME:
            raise RuntimeError(f"Offline loader refused unregistered Torch Hub request: {repo_or_dir} / {model}")
        kwargs.pop("source", None)
        return original_load(str(HUB_SOURCE), model, *args, source="local", **kwargs)

    def cached_state_loader(url, *args, **kwargs):
        from urllib.parse import urlparse
        if Path(urlparse(url).path).name != CHECKPOINT.name:
            raise RuntimeError(f"Offline loader refused unexpected checkpoint URL: {url}")
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(CHECKPOINT)
        return torch.load(CHECKPOINT, map_location=kwargs.get("map_location", "cpu"))

    torch.hub.load = local_only_load
    torch.hub.load_state_dict_from_url = cached_state_loader
    try:
        config = Stage0Config(
            model_name=MODEL_NAME, layers=LAYERS, resize_size=RESIZE_SIZE,
            crop_size=CROP_SIZE, patch_size=14, layer_fusion="mean", cache_dtype="float32",
        )
        sensor = PaperAlignedFrozenSensor(config, device=str(device), use_fp16=False)
    finally:
        torch.hub.load = original_load
        torch.hub.load_state_dict_from_url = original_state_loader
    sensor.backbone.model.eval()
    for parameter in sensor.backbone.model.parameters():
        parameter.requires_grad_(False)
    return sensor


def install_global_cls_hook(sensor):
    """Replace the reference's 15 retained block hooks with a block-18 CLS-only hook."""
    import torch
    import torch.nn.functional as F

    backbone = sensor.backbone
    backbone.cleanup()
    backbone._clear_feats()
    captured: list[Any] = []

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured.append(output.detach())

    handle = backbone.model.blocks[max(LAYERS)].register_forward_hook(hook)

    def extract(image, device):
        captured.clear()
        x = backbone._preprocess(image).unsqueeze(0)
        if x.dtype != torch.float32:
            raise TypeError(f"Preprocessor returned {x.dtype}, expected FP32")
        x = x.to(device)
        with torch.inference_mode():
            _ = backbone.model(x)
            if len(captured) != 1:
                raise RuntimeError(f"Expected one block-{max(LAYERS)} hook result, got {len(captured)}")
            cls = captured[0][:, 0, :].float()
            feature = F.normalize(cls, dim=-1)[0].detach().cpu().contiguous()
        del x
        return feature

    return extract, handle


def load_image(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def validate_vector(feature, expected_dim: int | None = None) -> None:
    import torch
    if not isinstance(feature, torch.Tensor) or feature.ndim != 1:
        raise ValueError("Feature must be a one-dimensional torch tensor")
    if feature.dtype != torch.float32 or feature.device.type != "cpu":
        raise ValueError(f"Feature must be CPU float32; got {feature.dtype} on {feature.device}")
    if expected_dim is not None and feature.numel() != expected_dim:
        raise ValueError(f"Feature width changed: {feature.numel()} != {expected_dim}")
    if not torch.isfinite(feature).all().item():
        raise ValueError("Feature contains NaN or Inf")
    norm = float(torch.linalg.vector_norm(feature).item())
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"Expected L2-normalized CLS feature; norm={norm}")


def runtime_identity(torch, sensor, device, input_manifest: Path, threads: int) -> dict[str, Any]:
    hub_hash, hub_files, hub_bytes = tree_sha256(HUB_SOURCE)
    code_hashes = source_code_sha256()
    preprocess = {
        "model_name": MODEL_NAME, "checkpoint_sha256": sha256_file(CHECKPOINT),
        "hub_source_tree_sha256": hub_hash, "layers_zero_indexed": list(LAYERS),
        "resize_size": RESIZE_SIZE, "crop_size": CROP_SIZE, "patch_size": 14,
        "color": "RGB", "mean": list(sensor.backbone.mean), "std": list(sensor.backbone.std),
        "feature": "raw block-18 CLS token, then torch.nn.functional.normalize(dim=-1)",
        "reference": "DINOv2MultiLayerBackbone.encode_batch global output",
        "precision": "float32", "stored_dtype": OUTPUT_DTYPE,
        "batch_size": BATCH_SIZE,
    }
    return {
        "schema": "d6.p1a.run.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": sha256_file(input_manifest),
        "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": preprocess["checkpoint_sha256"],
        "hub_source_path": str(HUB_SOURCE), "hub_source_tree_sha256": hub_hash,
        "hub_source_files": hub_files, "hub_source_bytes": hub_bytes,
        "model_name": MODEL_NAME, "device": str(device), "precision": "float32",
        "python_executable": sys.executable, "python_version": sys.version.split()[0],
        "torch_version": torch.__version__, "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": __import__("numpy").__version__, "pillow_version": __import__("PIL").__version__,
        "cpu_threads": threads, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unchanged>"),
        "source_code_sha256": code_hashes, "preprocess": preprocess,
        "preprocess_fingerprint": hashlib.sha256(stable_json(preprocess)).hexdigest(),
    }


def save_feature_item(out: Path, row: dict[str, str], feature, identity: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    validate_vector(feature)
    final_dir = out / "features" / row["image_id"]
    if final_dir.exists():
        return verify_feature_item(out, row, identity)
    temp_dir = final_dir.parent / f".tmp-{row['image_id']}-{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)
    arr = feature.detach().cpu().numpy().astype(np.float32, copy=False)
    npy_path = temp_dir / "feature.npy"
    with npy_path.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    feature_hash = sha256_file(npy_path)
    record = {
        "image_id": row["image_id"], "data_role": row["data_role"],
        "dataset": row["dataset"], "dataset_revision": row["dataset_revision"],
        "source_sha256": row["source_sha256"], "feature_relpath": str((final_dir / "feature.npy").relative_to(out)),
        "feature_sha256": feature_hash, "feature_dtype": "float32", "feature_shape": list(arr.shape),
        "device": identity["device"], "precision": "float32",
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "hub_source_tree_sha256": identity["hub_source_tree_sha256"],
        "reference_code_sha256": identity["source_code_sha256"]["reference_encoder"],
        "adapter_sha256": identity["source_code_sha256"]["p1a_adapter"],
        "preprocess_fingerprint": identity["preprocess_fingerprint"],
        "completion_status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(temp_dir / "record.json", record)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_dir, final_dir)
    return record


def verify_feature_item(out: Path, row: dict[str, str], identity: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    final_dir = out / "features" / row["image_id"]
    feature_path = final_dir / "feature.npy"
    record_path = final_dir / "record.json"
    if not feature_path.is_file() or not record_path.is_file():
        raise FileExistsError(f"Incomplete existing feature item; refusing overwrite: {row['image_id']}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    checks = {
        "image_id": row["image_id"], "data_role": row["data_role"],
        "source_sha256": row["source_sha256"], "checkpoint_sha256": identity["checkpoint_sha256"],
        "hub_source_tree_sha256": identity["hub_source_tree_sha256"],
        "adapter_sha256": identity["source_code_sha256"]["p1a_adapter"],
        "preprocess_fingerprint": identity["preprocess_fingerprint"],
        "device": identity["device"], "feature_dtype": OUTPUT_DTYPE,
        "completion_status": "COMPLETE",
    }
    for key, expected in checks.items():
        if record.get(key) != expected:
            raise ValueError(f"Existing feature identity mismatch for {row['image_id']} field {key}; refusing overwrite")
    if sha256_file(feature_path) != record.get("feature_sha256"):
        raise ValueError(f"Existing feature SHA mismatch; refusing overwrite: {row['image_id']}")
    arr = np.load(feature_path, allow_pickle=False)
    if arr.dtype != np.float32 or list(arr.shape) != record.get("feature_shape") or not np.isfinite(arr).all():
        raise ValueError(f"Existing feature values/shape invalid: {row['image_id']}")
    norm = float(np.linalg.norm(arr))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"Existing feature is not normalized: {row['image_id']} norm={norm}")
    return record


def write_feature_index(out: Path, input_rows: list[dict[str, str]], identity: dict[str, Any]) -> int:
    result = []
    for row in input_rows:
        final_dir = out / "features" / row["image_id"]
        if final_dir.exists():
            record = verify_feature_item(out, row, identity)
            result.append({field: record.get(field, "") for field in FEATURE_FIELDS})
    atomic_csv(out / "feature_manifest.csv", result, FEATURE_FIELDS)
    return len(result)


def check_existing_run(out: Path, identity: dict[str, Any]) -> None:
    run_path = out / "run_provenance.json"
    if not run_path.exists():
        return
    old = json.loads(run_path.read_text(encoding="utf-8"))
    stable_keys = [
        "input_manifest_sha256", "checkpoint_sha256", "hub_source_tree_sha256",
        "model_name", "device", "precision", "source_code_sha256",
        "preprocess_fingerprint", "torch_version", "torchvision_version",
    ]
    for key in stable_keys:
        if old.get(key) != identity.get(key):
            raise ValueError(f"Existing run provenance differs at {key}; refusing to mix or overwrite features")


def run_smoke(out: Path, rows: list[dict[str, str]], identity: dict[str, Any], sensor, device, guard_state):
    import torch
    row = rows[0]
    path = Path(row["image_path"])
    if sha256_file(path).lower() != row["source_sha256"].lower():
        raise ValueError(f"Smoke source hash mismatch: {row['image_id']}")
    rss = PeakRSS()
    rss.start()
    image = load_image(path)
    start = time.perf_counter()
    with torch.inference_mode():
        _patches, reference = sensor.encode_pil_batch([image])
    ref_elapsed = time.perf_counter() - start
    reference_feature = reference[0].detach().cpu().float().contiguous()
    del _patches, reference
    sensor.backbone.cleanup()
    slim, handle = install_global_cls_hook(sensor)
    start = time.perf_counter()
    slim_feature = slim(image, device)
    slim_elapsed = time.perf_counter() - start
    handle.remove()
    image.close()
    validate_vector(reference_feature)
    validate_vector(slim_feature, expected_dim=reference_feature.numel())
    max_abs = float(torch.max(torch.abs(reference_feature - slim_feature)).item())
    max_rel = float((torch.linalg.vector_norm(reference_feature - slim_feature) / torch.linalg.vector_norm(reference_feature)).item())
    equivalent = torch.allclose(reference_feature, slim_feature, atol=SMOKE_ATOL, rtol=SMOKE_RTOL)
    if not equivalent:
        raise RuntimeError(f"Slim CLS path is not canonical: max_abs={max_abs}, relative_l2={max_rel}")
    if guard_state is not None:
        free_after, _ = nvidia_smi_free_bytes()
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        if allocated > int(3.5 * 1024**3) or free_after < 4 * 1024**3:
            raise RuntimeError("Guarded CUDA smoke failed memory safety; GPU extraction stopped")
    else:
        free_after = None
        allocated = reserved = None
    record = {
        "schema": "d6.p1a.smoke.v1", "image_id": row["image_id"],
        "reference_seconds": ref_elapsed, "slim_seconds": slim_elapsed,
        "estimated_full_102_seconds": slim_elapsed * 102,
        "feature_dim": int(slim_feature.numel()), "dtype": "float32",
        "norm": float(torch.linalg.vector_norm(slim_feature).item()),
        "all_finite": bool(torch.isfinite(slim_feature).all().item()),
        "reference_equivalent": bool(equivalent), "atol": SMOKE_ATOL, "rtol": SMOKE_RTOL,
        "max_abs_difference": max_abs, "relative_l2_difference": max_rel,
        "peak_rss_bytes": rss.stop(), "device": str(device),
        "cuda_free_after_bytes": free_after, "cuda_peak_allocated_bytes": allocated,
        "cuda_peak_reserved_bytes": reserved,
    }
    atomic_json(out / "smoke_result.json", record)
    if equivalent:
        save_feature_item(out, row, slim_feature, identity)
        write_feature_index(out, rows, identity)
    return record, slim, handle


def cuda_guard_after_image(torch, device, initial_free: int) -> None:
    free_now, _ = nvidia_smi_free_bytes()
    allocated = int(torch.cuda.max_memory_allocated(device))
    if free_now < 4 * 1024**3:
        raise RuntimeError("Guarded CUDA stopped: system free VRAM fell below 4 GiB")
    if allocated > int(3.5 * 1024**3):
        raise RuntimeError("Guarded CUDA stopped: PyTorch allocated memory exceeded 3.5 GiB")
    if initial_free - free_now > 8 * 1024**3:
        raise RuntimeError("Guarded CUDA stopped: free-memory change is unexpectedly large; protect other GPU work")


def extract_all(out: Path, input_rows: list[dict[str, str]], identity: dict[str, Any], sensor, device, guard_state, limit: int | None, model_load_peak_rss: int = 0) -> dict[str, Any]:
    import torch
    smoke_path = out / "smoke_result.json"
    if not smoke_path.is_file():
        raise RuntimeError("Run --mode smoke and pass the equivalence gate before extraction")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not smoke.get("reference_equivalent") or smoke.get("device") != str(device) or smoke.get("dtype") != "float32":
        raise RuntimeError("Stored smoke result does not authorize this feature identity")
    provenance_path = out / "run_provenance.json"
    if provenance_path.exists():
        check_existing_run(out, identity)
    else:
        atomic_json(provenance_path, identity)
    slim, handle = install_global_cls_hook(sensor)
    completed_before = write_feature_index(out, input_rows, identity)
    expected_ids = {row["image_id"] for row in input_rows}
    feature_root = out / "features"
    if feature_root.exists():
        unexpected = [p.name for p in feature_root.iterdir() if p.is_dir() and not p.name.startswith(".tmp-") and p.name not in expected_ids]
        if unexpected:
            raise ValueError(f"Unexpected feature identities in output directory: {unexpected[:5]}")
    selected_rows = input_rows if limit is None else input_rows[:limit]
    rss = PeakRSS()
    rss.start()
    start_all = time.perf_counter()
    done = 0
    initial_free = guard_state["free_before_bytes"] if guard_state else None
    for i, row in enumerate(selected_rows, 1):
        path = Path(row["image_path"])
        if sha256_file(path).lower() != row["source_sha256"].lower():
            raise ValueError(f"Source identity changed for {row['image_id']}; stopping")
        final_dir = out / "features" / row["image_id"]
        if final_dir.exists():
            verify_feature_item(out, row, identity)
            print(f"SKIP {i}/{len(selected_rows)} {row['image_id']} verified", flush=True)
            continue
        image = load_image(path)
        t0 = time.perf_counter()
        with torch.inference_mode():
            feature = slim(image, device)
        elapsed = time.perf_counter() - t0
        image.close()
        validate_vector(feature)
        save_feature_item(out, row, feature, identity)
        done += 1
        write_feature_index(out, input_rows, identity)
        if guard_state:
            cuda_guard_after_image(torch, device, initial_free)
        print(f"DONE {i}/{len(selected_rows)} {row['image_id']} {elapsed:.2f}s", flush=True)
        del feature
    handle.remove()
    peak_rss = rss.stop()
    total_elapsed = time.perf_counter() - start_all
    completed = write_feature_index(out, input_rows, identity)
    result = {
        "schema": "d6.p1a.extraction_status.v1", "status": "PASS" if completed == 102 else "PARTIAL",
        "completed_unique_images": completed, "target_unique_images": 102,
        "newly_computed_this_call": done, "completed_before_call": completed_before,
        "device": str(device), "dtype": "float32", "feature_dim": json.loads((out / "features" / input_rows[0]["image_id"] / "record.json").read_text(encoding="utf-8"))["feature_shape"][0] if completed else None,
        "elapsed_seconds_this_call": total_elapsed, "peak_rss_bytes_this_call": max(peak_rss, model_load_peak_rss),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "hub_source_tree_sha256": identity["hub_source_tree_sha256"],
        "feature_manifest_sha256": sha256_file(out / "feature_manifest.csv") if (out / "feature_manifest.csv").is_file() else None,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(out / "extraction_status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "smoke", "extract"], required=True)
    parser.add_argument("--device", choices=["cpu", "guarded-cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None, help="Optional bounded partial run; default is all 102")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()

    if args.mode == "prepare":
        out.mkdir(parents=True, exist_ok=True)
        manifest = prepare_input_manifest(out)
        print(f"Prepared {manifest} ({len(csv_rows(manifest))} rows), SHA256={sha256_file(manifest)}")
        return 0

    # CPU-first: mask the GPU before PyTorch is imported or any model code is loaded.
    device_name = "cpu" if args.device == "cpu" else "guarded-cuda"
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    input_manifest = out / "input_manifest.csv"
    if not input_manifest.is_file():
        raise FileNotFoundError("Run --mode prepare first")
    if args.mode == "extract" and (not (out / "smoke_result.json").is_file() or not (out / "run_provenance.json").is_file()):
        raise RuntimeError("Run --mode smoke and pass the equivalence gate before extraction")
    rows = csv_rows(input_manifest)
    validate_input_rows(rows, verify_sha=False)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    device_name_for_torch = "cpu" if device_name == "cpu" else "cuda"
    torch, device, guard_state = configure_torch(device_name_for_torch, args.threads)
    rss = PeakRSS()
    rss.start()
    sensor = load_frozen_sensor(torch, device)
    identity = runtime_identity(torch, sensor, device, input_manifest, args.threads)
    check_existing_run(out, identity)
    if args.mode == "smoke":
        record, _slim, handle = run_smoke(out, rows, identity, sensor, device, guard_state)
        handle.remove()
        record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss.stop())
        atomic_json(out / "smoke_result.json", record)
        atomic_json(out / "run_provenance.json", identity)
        print(json.dumps(record, indent=2))
        return 0
    model_load_peak = rss.stop()
    result = extract_all(out, rows, identity, sensor, device, guard_state, args.max_images, model_load_peak)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"P1A_STOP: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
