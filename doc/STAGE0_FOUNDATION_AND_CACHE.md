# Stage 0A / 0B — Foundation and Frozen DINO Cache

This document defines the first implementation checkpoint for the Attribution-Aware Bayesian Memory Governance project.

## Scientific purpose

Stage 0 is deliberately narrow. It does **not** re-implement the complete predecessor v22 controller. It establishes a stable visual substrate that later representation and memory experiments can share.

- **Stage 0A**: run a thin VisionAD-aligned few-shot anomaly baseline on Real-IAD and write reproducible per-item scores plus category-level AUROC/AP.
- **Stage 0B**: cache the frozen DINOv2 patch representation once, with target ground truth separated from the feature payload.

The next experiment can then compare `R0 = frozen DINO`, `R1 = beta=1 VAE`, and `R_beta = beta>1 VAE` without repeatedly running DINO and without changing the sensory evidence between conditions.

## Configuration basis

The implementation refers to both the local paper `doc/Search is All You Need for Few-shot Anomaly Detection.pdf` and the paper's released VisionAD code. The default foundation profile is:

| Item | Stage-0 default |
|---|---|
| Backbone | DINOv2-Register ViT-L/14 |
| Online backbone update | none; frozen |
| Resize | 448 |
| Center crop | 392 |
| Patch size | 14 |
| Patch grid | 28 x 28 |
| Selected transformer blocks | 4..18, zero-indexed |
| Layer fusion | mean |
| Support expansion | identity + 90/180/270 rotation + vertical/horizontal flip |
| Patch anomaly | 1 - nearest cosine similarity |
| Image anomaly score | mean of top 1% anomaly-map pixels |
| Query view | **identity only** for the ABMG principal protocol |

The last row is intentional. VisionAD also studies pseudo-multi-view query fusion, but the ABMG scaffold pre-registers a single-view protocol for the first decisive experiments. Stage 0 therefore keeps the paper-aligned sensor configuration while removing pseudo-multi-view as a confounder.

The machine-readable profile is `configs/stage0_realiad_single_view.json`.

## Data-access boundary

The cache is split into three pieces:

1. `features/<category>/<split>/<image_id>.pt`
   - frozen patch descriptors
   - frozen global descriptor
   - geometry/config metadata
   - **no labels, defect type, or mask**
2. `manifest_public.jsonl`
   - image id, category, split, paths, cache path
   - safe for later representation experiments
3. `manifest_offline_gt.jsonl`
   - binary label, defect source/type, mask path
   - **offline evaluator only**

This separation is deliberate: later target-time code should never need to load the offline-GT manifest.

## 1. Checkout

```bash
git fetch origin
git checkout stage0-foundation-cache
```

## 2. Syntax/small unit tests

```bash
python -m py_compile scripts/abmg_stage0_foundation.py
python -m unittest discover -s tests -p 'test_stage0*.py' -v
```

## 3. Inspect your Real-IAD layout first

The script understands both the modified layout used by the predecessor scripts and the official VisionAD-style Real-IAD layout.

```bash
python scripts/abmg_stage0_foundation.py inspect \
  --root /PATH/TO/Real-IAD \
  --json-dir /PATH/TO/realiad_jsons_sv
```

Do not proceed if `missing_images` is non-zero. If you are not using the single-view JSON protocol, pass the directory you actually use, but record that deviation before running the later C1 experiment.

## 4. Stage 0A smoke test

Start with one category and a small number of test images:

```bash
python scripts/abmg_stage0_foundation.py continuity \
  --root /PATH/TO/Real-IAD \
  --json-dir /PATH/TO/realiad_jsons_sv \
  --classes audiojack \
  --shots 4 \
  --seed 0 \
  --max-test-per-class 32 \
  --out-dir outputs/stage0_smoke
```

Expected artifacts:

```text
outputs/stage0_smoke/
├── dataset_inventory.json
├── resolved_config.json
├── per_item_scores.csv
└── summary.json
```

A successful smoke test means the backbone loads, the support bank is built, all requested query images are scored, and `summary.json` is produced. The smoke-test AUROC is not a scientific result because the stream is truncated.

## 5. Stage 0A full continuity run

```bash
python scripts/abmg_stage0_foundation.py continuity \
  --root /PATH/TO/Real-IAD \
  --json-dir /PATH/TO/realiad_jsons_sv \
  --shots 4 \
  --seed 0 \
  --out-dir outputs/stage0_full
```

For an initial local check it is acceptable to run a few categories first. Do not tune the sensor against target-category labels. The purpose is numerical continuity and a stable reference trajectory, not leaderboard optimization.

## 6. Stage 0B cache smoke test

```bash
python scripts/abmg_stage0_foundation.py cache \
  --root /PATH/TO/Real-IAD \
  --json-dir /PATH/TO/realiad_jsons_sv \
  --classes audiojack \
  --max-items 16 \
  --batch-size 2 \
  --cache-dir cache/stage0_dino_smoke
```

Inspect one payload:

```bash
python - <<'PY'
import torch
from pathlib import Path
p = next(Path('cache/stage0_dino_smoke/features').rglob('*.pt'))
x = torch.load(p, map_location='cpu')
print(p)
print(x.keys())
print(x['patch_features'].shape, x['patch_features'].dtype)
print(x['global_feature'].shape, x['global_feature'].dtype)
print(x['grid_hw'])
assert 'label' not in x
assert 'defect_source' not in x
assert 'mask_path' not in x
PY
```

With the default ViT-L/14 profile the fused patch tensor should have a 28 x 28 patch grid, i.e. `784` patch descriptors per image, with feature dimension `1024`.

## 7. Stage 0B full cache

```bash
python scripts/abmg_stage0_foundation.py cache \
  --root /PATH/TO/Real-IAD \
  --json-dir /PATH/TO/realiad_jsons_sv \
  --batch-size 2 \
  --dtype float16 \
  --resume \
  --cache-dir cache/realiad_dino_single_view
```

The cache command is restart-safe with `--resume`. The DINO backbone remains frozen and the cache is identity-view only.

Expected structure:

```text
cache/realiad_dino_single_view/
├── cache_metadata.json
├── cache_completion.json
├── dataset_inventory.json
├── manifest_public.jsonl
├── manifest_offline_gt.jsonl
└── features/
    └── <category>/
        ├── train/
        └── test/
```

## 8. Frozen category folds

`configs/realiad_folds_v0.json` freezes five target folds of six Real-IAD categories. For each fold, the other 24 categories are source categories.

Later Stage-1 representation training must obey:

- fit VAE/beta-VAE only on permitted **source-category normal features**;
- choose beta and other representation hyperparameters only from source-side validation;
- freeze the factorizer before target streaming;
- use target masks/defect-source labels only for offline diagnostics/evaluation unless explicitly revealed by the experimental feedback condition.

## 9. What is intentionally absent from Stage 0

Do not add these before the Stage-1/2 representation gate:

- beta-VAE/VAE training;
- NIG Bayesian normal memory;
- responsibility vectors;
- factor-addressed writes;
- source-centroid memory;
- factor correction;
- end-to-end query controller;
- target-time DINO fine-tuning;
- multi-view target fusion.

This keeps Stage 0 as the controlled sensor and data-interface boundary for the new research.
