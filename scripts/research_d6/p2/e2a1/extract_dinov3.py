"""Isolated offline Hugging Face DINOv3 ViT-L/16 E2-A1 extractor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

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
from transformers import AutoConfig, AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from contracts import (DINO3_CONFIG_SHA256, DINO3_MODEL, DINO3_PROCESSOR_SHA256,
                       DINO3_REVISION, DINO3_WEIGHTS_SHA256, EXPECTED_BRANCH,
                       EXPECTED_HEAD, FEATURE_INTERFACE_SCHEMA, INPUT_MANIFEST_SHA256, ROOT)
from extract_core import run_backend
from storage import sha256_file, stable_json_bytes

A0_INTERFACE_DIR = ROOT / "scripts/research_d6/p2/e2a"
if str(A0_INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(A0_INTERFACE_DIR))
from feature_interface import dinov3_to_shared_interface


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "all"), required=True)
    args = parser.parse_args()
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != EXPECTED_HEAD or branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Unexpected execution identity: {branch}/{head}")
    if torch.__version__ != "2.14.1+cu126" or transformers.__version__ != "5.23.0":
        raise RuntimeError(f"Unexpected DINOv3 environment: torch={torch.__version__}, transformers={transformers.__version__}")
    if np.__version__ != "2.4.6":
        raise RuntimeError(f"Unexpected DINOv3 NumPy version: {np.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("DINOv3 environment reports CUDA unavailable")

    snapshot = (Path.home() / ".cache/huggingface/hub" /
                "models--facebook--dinov3-vitl16-pretrain-lvd1689m" / "snapshots" / DINO3_REVISION)
    weights_path, config_path, processor_path = (snapshot / "model.safetensors", snapshot / "config.json",
                                                  snapshot / "preprocessor_config.json")
    weights_sha, config_sha, processor_sha = (sha256_file(weights_path), sha256_file(config_path),
                                               sha256_file(processor_path))
    if (weights_sha, config_sha, processor_sha) != (
            DINO3_WEIGHTS_SHA256, DINO3_CONFIG_SHA256, DINO3_PROCESSOR_SHA256):
        raise RuntimeError("DINOv3 local model snapshot identity mismatch")
    config = AutoConfig.from_pretrained(DINO3_MODEL, revision=DINO3_REVISION, local_files_only=True)
    expected_config = ("DINOv3ViTModel", 1024, 16, 4, 24)
    actual_config = (config.architectures[0], config.hidden_size, config.patch_size,
                     config.num_register_tokens, config.num_hidden_layers)
    if actual_config != expected_config:
        raise RuntimeError(f"Unexpected DINOv3 model configuration: {actual_config}")

    script_sha = sha256_file(Path(__file__).resolve())
    core_sha = sha256_file(SCRIPT_DIR / "extract_core.py")
    interface_sha = sha256_file(A0_INTERFACE_DIR / "feature_interface.py")
    runtime = {
        "execution_head": head,
        "branch": branch,
        "model": DINO3_MODEL,
        "snapshot_revision": DINO3_REVISION,
        "weights_sha256": weights_sha,
        "registered_input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "config_sha256": config_sha,
        "processor_config_sha256": processor_sha,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "feature_interface_schema": FEATURE_INTERFACE_SCHEMA,
        "backend_script_sha256": script_sha,
        "extract_core_sha256": core_sha,
        "feature_interface_code_sha256": interface_sha,
        "position_encoding": "official dynamic 2D RoPE",
        "token_order": "CLS, four register tokens, row-major spatial patches",
    }
    runtime["extractor_identity_sha256"] = hashlib.sha256(stable_json_bytes(runtime)).hexdigest()

    def model_loader():
        model = AutoModel.from_pretrained(
            DINO3_MODEL, revision=DINO3_REVISION, local_files_only=True, dtype=torch.float32
        )
        return model

    def forward(model, input_gpu, mask):
        output = model(pixel_values=input_gpu, return_dict=True).last_hidden_state
        hidden = output.detach().float().cpu().contiguous().numpy()
        return dinov3_to_shared_interface(hidden, mask)

    report = run_backend(backend="dinov3", mode=args.mode, torch=torch,
                         runtime=runtime, model_loader=model_loader, forward=forward,
                         warmups=1 if args.mode == "smoke" else 0)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
