# Stage 0B compact cache policy

## Why the original full cache is not viable

The Stage-0A substrate produces a 28 x 28 patch grid from DINOv2-Register ViT-L/14. With mean layer fusion the feature dimension is 1024. A full fp16 grid therefore requires approximately:

```
784 patches * 1024 dimensions * 2 bytes = 1,605,632 bytes ~= 1.53 MiB / image
```

For roughly 151k Real-IAD train+test images this is over 220 GiB before filesystem/container overhead. The earlier one-file-per-image cache also creates about 151k files. This is unnecessary for the ABMG Stage-1 representation experiment.

## Revised Stage-0B policy

Stage 0B is split into two roles.

### Persistent source-training cache

Persist only the data needed to train the source-side factorizer:

- Real-IAD `train` split by default;
- normal samples only by default;
- frozen DINO features only;
- deterministic patch subsampling, default 64 of 784 patches per image;
- fp16 by default;
- sharded storage, default 1024 images per shard;
- no label, defect-source or mask fields inside feature shards.

This preserves a large, reproducible patch-level training pool while reducing disk use by more than an order of magnitude.

### Target/test evaluation

Do **not** persist full patch grids for all target/test images. During Stage-1 evaluation, each target batch will be passed through frozen DINO once, and the resulting full patch tensor will be fed in memory to all compared representations (`R0`, `R1`, and the beta-VAE conditions) before the tensor is released. This preserves identical sensory evidence across methods without a >200 GiB cache.

Full patch grids remain available transiently for factor maps, spatial attribution, and anomaly scoring. Only persistent storage is reduced.

## Recommended command

After deleting the aborted old cache, run:

```bash
python scripts/abmg_stage0_compact_cache.py \
  --root /PATH/TO/Real-IAD \
  --json-dir /PATH/TO/realiad_jsons_sv \
  --patches-per-image 64 \
  --batch-size 2 \
  --images-per-shard 1024 \
  --dtype float16 \
  --resume \
  --cache-dir cache/realiad_dino_source_normal_compact
```

The script prints a storage estimate before extracting features. Under the default mean-fused 1024-dimensional representation, 64 patches per image require roughly 128 KiB of patch storage per image plus one 2 KiB global feature, before small metadata overhead.

## Why patch subsampling is acceptable for Stage-1 training

The factorizer treats DINO patch descriptors as independent feature-space training observations. It does not require the complete 28 x 28 spatial grid for every source training image. Spatial structure is required later for addressability/spatial-consistency evaluation, and those target grids are computed in full on demand.

Patch selection is deterministic from `(patch_seed, image_id)`, so all later source-side experiments see the same patch sample. The selected patch index is stored in each shard.

## Resume safety

Shards are written to a temporary file and renamed only after `torch.save` succeeds. Therefore a disk-full or interrupted write cannot leave a truncated final shard that `--resume` would incorrectly treat as complete.

This differs from the original one-file-per-image implementation, whose `--resume` check was only `Path.is_file()` and could therefore skip a partially written file after an interrupted save.

## Outputs

```
cache/realiad_dino_source_normal_compact/
  cache_metadata.json
  cache_completion.json
  dataset_inventory.json
  manifest_public.jsonl
  shards/
    shard_00000.pt
    shard_00001.pt
    ...
```

A shard contains:

```python
{
    "image_ids": [...],
    "categories": [...],
    "relative_paths": [...],
    "patch_features": Tensor[N, 64, 1024],
    "global_features": Tensor[N, 1024],
    "patch_indices": Tensor[N, 64],
    "grid_hw": (28, 28),
}
```

No target label, mask, or defect-source information is embedded in the feature shard.

## Scientific boundary

This change is a storage/experiment-infrastructure decision, not a representation-method change. The frozen DINO configuration remains the Stage-0A configuration. The Stage-1 factorizer still learns from DINO patch descriptors. Target evaluation still uses complete DINO patch grids; it simply does not materialize every target grid on disk.
