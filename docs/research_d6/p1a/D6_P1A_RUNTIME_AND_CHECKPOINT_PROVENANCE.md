# D6-P1-A Runtime and Checkpoint Provenance

**Run date:** 2026-11-13.  
**Repository:** `stage3-eg-famar-execution`, HEAD `a59432092ec824713de3c1cfec334c7e9ce105e1` at start of task. The working tree had pre-existing untracked researcher materials; no tracked file was changed.

## Runtime used

| Field | Measured value |
|---|---|
| Python executable | `D:\anaconda3\envs\torch\python.exe` |
| Python | 3.11.14 |
| PyTorch | 2.0.1+cu117; CUDA build 11.7 |
| torchvision | 0.15.2+cu117 |
| NumPy | 1.24.1 |
| Pillow | 9.3.0 |
| scikit-learn | 1.3.0 |
| Execution device | CPU, FP32, batch size 1, two CPU threads |
| `CUDA_VISIBLE_DEVICES` | Empty before PyTorch import |

No package was installed or changed. The P1-A process did not call CUDA APIs, create a CUDA model, or launch a GPU kernel. The other GPU project's utilization was not measured during extraction; its exact resource use is therefore UNKNOWN, while the D6 feature process was constrained to CPU by the process environment. No GPU process was stopped or modified.

The cached DINO source emitted an xFormers-unavailable warning and used its standard PyTorch attention path. Both the reference and CLS-only adapter used this same path, same CPU, and same FP32 weights for the equivalence check.

## DINO identity and feature definition

| Item | Provenance |
|---|---|
| Model name | `dinov2_vitl14_reg` |
| Local checkpoint | `C:\Users\rico\.cache\torch\hub\checkpoints\dinov2_vitl14_reg4_pretrain.pth` |
| Checkpoint size | 1,217,607,321 bytes |
| Checkpoint SHA-256 | `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` |
| Cached Hub source | `C:\Users\rico\.cache\torch\hub\facebookresearch_dinov2_main` |
| Hub source snapshot | 174 files, 2,654,538 bytes; deterministic tree SHA-256 `7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c` |
| Hub Git revision | UNKNOWN: local source cache has no `.git` metadata |
| Weight loading | Offline local source/checkpoint only; no download; DINO Hub model construction uses strict state-dict loading |
| Preprocessing | RGB; resize 448; center crop 392; mean `(0.485, 0.456, 0.406)`; std `(0.229, 0.224, 0.225)` |
| Feature | Raw block-18 CLS token, then L2 normalization; 1024 values, stored FP32 |
| Preprocessing fingerprint | `4d3a5906bacafde057f806bbe53ccf24e4960574837e9c47bd273233223ec56e` |

Code SHA-256 values:

| Code | SHA-256 |
|---|---|
| Existing reference encoder `scripts/vmb_visionad_new.py` | `0430959e86ba75b029341aef003655e6a3ac7d08df22f3f48a04e0b2d7fc39c9` |
| Existing Stage0 adapter `scripts/abmg_stage0_foundation.py` | `6e287288aec74994f995071c1486bf909ecb2c399bebe0e936c1beb12347bdd5` |
| New isolated P1-A adapter `scripts/research_d6/p1a/extract_features.py` | `034a5650365ce972909801ad96f417fde4a878ff0369d7a5f8e8e852c8c47d33` |

The DINO source/checkpoint identities, Python/PyTorch versions, preprocessing, device, precision, feature dimension, and per-image output hashes are recorded in the ignored local run and feature manifests. The local Hub source commit is not known; its content-tree digest is the reproducibility pin for this run.

## Resource and safety record

The CPU CLS-only path measured 4.31 seconds per image in the smoke test. The 100-image continuation took 458.29 seconds. Peak process RSS was 2.80 GB during the reference/equivalence smoke and 1.77 GB during the feature-only continuation. No GPU memory statistics were sampled during P1-A because the extraction process masked CUDA and never initialized a CUDA context. Guarded CUDA remains untested and was not selected.

P1-A did not read the remaining 72 Real-IAD views, masks, additional JSON, sealed targets, Probe, Mint/PCB, D-conf, or other HOLD material. It used only the 54 D-fit and 48 D-dev paths in the local input manifest. No D6-P1-B, P2, or P3 model operation occurred.
