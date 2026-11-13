# D6-P1 Gate0 — Runtime Readiness

**Checked:** 2026-11-13. Checks import package metadata and GPU availability only. No DINO model was instantiated and no checkpoint or feature tensor was loaded.

## Runtime matrix

| Conda environment | Python executable | Python | torch / torchvision | CUDA / GPU | Other relevant packages | Assessment |
|---|---|---:|---|---|---|---|
| Active `base` | `D:\anaconda3\python.exe` | 3.11.9 | Not installed | Not available | NumPy 1.26.4, pandas 1.5.3, Pillow 12.0.0, scikit-learn 1.3.0 | Current shell interpreter cannot run DINO inference |
| `flybody` | `D:\anaconda3\envs\flybody\python.exe` | 3.10.21 | Not installed | Not available | NumPy 1.26.4, Pillow 12.3.0; pandas/scikit-learn absent | Not suitable |
| `textgen` | `D:\anaconda3\envs\textgen\python.exe` | 3.10.9 | torch 2.0.1, torchvision 0.15.2 | CUDA build 11.7; available; RTX 3090 Ti | NumPy 1.26.4, pandas 2.1.1, Pillow 9.5.0; scikit-learn absent | CUDA exists, but lacks B1 dependency |
| `torch` | **`D:\anaconda3\envs\torch\python.exe`** | 3.11.14 | torch 2.0.1+cu117, torchvision 0.15.2+cu117 | CUDA build 11.7; available; RTX 3090 Ti | NumPy 1.24.1, pandas 2.1.0, Pillow 9.3.0, scikit-learn 1.3.0 | **Recommended interpreter candidate** |

The active PowerShell process is in Conda `base`, despite the earlier `(torch)` prompt report. No environment was changed and no package was installed, upgraded, or removed. Use the absolute `torch` interpreter path if a later run is approved; do not assume shell activation.

## GPU reading

Device name is `NVIDIA GeForce RTX 3090 Ti`. `torch.cuda.get_device_properties(0)` reports 25,756,696,576 total bytes; `torch.cuda.mem_get_info()` reported 24,445,452,288 free bytes during the `torch` environment check. `nvidia-smi` reported 24,564 MiB total and 8,989 MiB free in a nearby check. The free-memory readings disagree substantially and are time-sensitive. Treat available VRAM as **not yet stable/confirmed** and recheck immediately before any authorized extraction; this report does not establish a safe batch size.

## DINO configuration, checkpoint and reproducibility

| Item | Finding |
|---|---|
| Project config | `configs/stage0_realiad_single_view.json`: `dinov2_vitl14_reg`, frozen; resize 448, crop 392, patch size 14/grid 28×28, blocks 4–18 mean fusion |
| Encoder code | `scripts/vmb_visionad_new.py` calls `torch.hub.load("facebookresearch/dinov2", model_name)`; `scripts/abmg_stage0_foundation.py` exposes frozen `encode_pil_batch()` outputs for patch and global features |
| Local weight file | `C:\Users\rico\.cache\torch\hub\checkpoints\dinov2_vitl14_reg4_pretrain.pth`, 1,217,607,321 bytes; presence/name/size checked only; weights were not loaded or hashed |
| Torch Hub source | A local `facebookresearch_dinov2_main` source directory exists, but it has no Git metadata; its exact source commit is UNKNOWN |
| Checkpoint pin | No checkpoint SHA-256 is recorded in the Stage3 experiment manifest (`checkpoint_sha256` is null). Exact weight identity is UNKNOWN until a later approved integrity pin |
| Environment pin | `requirements-stage0.txt` lists unpinned packages. No D6 environment lock/conda export was found among the audited project records |

**Conclusion:** a CUDA-capable PyTorch environment and a matching-named local DINO checkpoint are present. This makes inference technically plausible, but a reproducible compatible DINO runtime is not fully certified: the hub source revision and checkpoint hash are not pinned, the feature dimension has not been verified, and no model-load smoke test was authorized. The recommended interpreter is `D:\anaconda3\envs\torch\python.exe`, subject to a later version/checkpoint pin and memory recheck.

## Feature readiness

P0's cache metadata describes 36,465 Real-IAD train-normal records over 30 categories with 64 sampled patch features per image and global features declared. It excludes defects and does not cover the 54 D-fit images or 48 selected D-dev views. The D6 annotated-item feature cache does not exist. The current encoder API can produce global and spatial patch outputs, but the existing dataset/cache path is not a ready D6 adapter. New feature extraction would be required for the exact 102 candidate images and must be separately authorized; no image, mask, weight, or tensor was opened here.
