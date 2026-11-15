"""Isolated local-only DINOv2 ViT-L/14 Register E2-A1 extractor."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import torchvision

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from contracts import (DINO2_CHECKPOINT_SHA256, DINO2_MODEL, DINO2_SOURCE_TREE_SHA256,
                       EXPECTED_BRANCH, EXPECTED_HEAD, FEATURE_INTERFACE_SCHEMA,
                       INPUT_MANIFEST_SHA256, ROOT)
from extract_core import run_backend
from storage import sha256_file, stable_json_bytes

A0_INTERFACE_DIR = ROOT / "scripts/research_d6/p2/e2a"
if str(A0_INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(A0_INTERFACE_DIR))
from feature_interface import dinov2_to_shared_interface


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load local module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "all"), required=True)
    args = parser.parse_args()
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != EXPECTED_HEAD or branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Unexpected execution identity: {branch}/{head}")
    if torch.__version__ != "2.0.1+cu117" or torchvision.__version__ != "0.15.2+cu117":
        raise RuntimeError(f"Unexpected DINOv2 environment: torch={torch.__version__}, torchvision={torchvision.__version__}")
    if np.__version__ != "1.24.1":
        raise RuntimeError(f"Unexpected DINOv2 NumPy version: {np.__version__}")

    p1a = load_module("e2a1_p1a_model_loader", ROOT / "scripts/research_d6/p1a/extract_features.py")
    checkpoint_sha = p1a.sha256_file(p1a.CHECKPOINT)
    source_sha, source_files, source_bytes = p1a.tree_sha256(p1a.HUB_SOURCE)
    if (checkpoint_sha, source_sha) != (DINO2_CHECKPOINT_SHA256, DINO2_SOURCE_TREE_SHA256):
        raise RuntimeError("DINOv2 local checkpoint/source identity mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("DINOv2 environment reports CUDA unavailable")

    script_sha = sha256_file(Path(__file__).resolve())
    core_sha = sha256_file(SCRIPT_DIR / "extract_core.py")
    interface_sha = sha256_file(A0_INTERFACE_DIR / "feature_interface.py")
    runtime = {
        "execution_head": head,
        "branch": branch,
        "model": DINO2_MODEL,
        "weights_sha256": checkpoint_sha,
        "registered_input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "source_tree_sha256": source_sha,
        "source_tree_file_count": source_files,
        "source_tree_bytes": source_bytes,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "feature_interface_schema": FEATURE_INTERFACE_SCHEMA,
        "backend_script_sha256": script_sha,
        "extract_core_sha256": core_sha,
        "feature_interface_code_sha256": interface_sha,
    }
    runtime["extractor_identity_sha256"] = hashlib.sha256(stable_json_bytes(runtime)).hexdigest()

    sensor_holder: dict[str, Any] = {}

    def model_loader():
        scripts_dir = ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        sensor = p1a.load_frozen_sensor(torch, device="cpu")
        # P1-A's multilevel hooks are not part of the E2 final-layer contract.
        sensor.cleanup()
        sensor_holder["sensor"] = sensor
        return sensor.backbone.model

    def forward(model, input_gpu, mask):
        output = model.forward_features(input_gpu)
        raw = {
            "x_norm_clstoken": output["x_norm_clstoken"].detach().float().cpu().contiguous().numpy(),
            "x_norm_regtokens": output["x_norm_regtokens"].detach().float().cpu().contiguous().numpy(),
            "x_norm_patchtokens": output["x_norm_patchtokens"].detach().float().cpu().contiguous().numpy(),
        }
        return dinov2_to_shared_interface(raw, mask)

    report = run_backend(backend="dinov2", mode=args.mode, torch=torch,
                         runtime=runtime, model_loader=model_loader, forward=forward,
                         warmups=1 if args.mode == "smoke" else 0)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
