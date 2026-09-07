#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact Stage-0B cache for ABMG.

Why this exists
---------------
Caching every 28x28 DINOv2 ViT-L/14 patch grid for every Real-IAD image as one
``.pt`` file is prohibitively large: 784 patches x 1024 dims x fp16 is about
1.53 MiB per image before container overhead.  At ~151k images that is >220 GiB.

For Stage 1 representation training we do not need a persistent full-grid cache
for every target image.  We only need a reproducible source-side pool of normal
DINO patch descriptors.  This script therefore:

* defaults to the Real-IAD TRAIN split only;
* defaults to verified normal samples only;
* deterministically samples a small number of patches per image;
* stores many images in one shard rather than one file per image;
* writes shards atomically (tmp -> rename), making --resume safe;
* keeps labels, masks and defect-source metadata out of feature shards.

Target/test images should be processed online in later Stage-1 evaluation so all
R0/R1/R_beta conditions see the same DINO tensor in the same process without a
persistent >200 GiB target cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import torch

# scripts/ is on sys.path when this file is run directly.
from abmg_stage0_foundation import (
    PaperAlignedFrozenSensor,
    Stage0Config,
    Stage0Record,
    config_fingerprint,
    inventory,
    load_realiad_records,
    parse_csv_set,
    parse_layers,
    set_seed,
)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _chunks(xs: Sequence[Stage0Record], n: int) -> Iterator[Sequence[Stage0Record]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _patch_seed(image_id: str, seed: int) -> int:
    h = hashlib.sha1(f"{seed}|{image_id}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**63 - 1)


def _sample_patch_indices(image_id: str, n_patches: int, keep: int, seed: int) -> torch.Tensor:
    keep = min(int(keep), int(n_patches))
    g = torch.Generator(device="cpu")
    g.manual_seed(_patch_seed(image_id, seed))
    return torch.randperm(int(n_patches), generator=g)[:keep]


def _atomic_torch_save(payload: Dict[str, Any], final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        torch.save(payload, tmp)
        os.replace(tmp, final_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _estimate_bytes(n_images: int, patches_per_image: int, dim: int, dtype: str) -> int:
    bpe = 2 if dtype == "float16" else 4
    # patch payload + one global feature per image; metadata overhead is small.
    return int(n_images) * (int(patches_per_image) * int(dim) + int(dim)) * bpe


def _human_gib(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GiB"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ABMG compact Stage-0B source-normal DINO patch cache"
    )
    p.add_argument("--root", required=True, help="Real-IAD root directory")
    p.add_argument("--json-dir", required=True, help="Directory containing Real-IAD JSON files")
    p.add_argument("--classes", default="", help="Optional comma-separated category subset")
    p.add_argument("--splits", default="train", help="Default train; comma-separated if needed")
    p.add_argument(
        "--include-defects",
        action="store_true",
        help="Include defects. OFF by default because Stage-1 factorizer training is source-normal only.",
    )

    p.add_argument("--model-name", default="dinov2_vitl14_reg")
    p.add_argument("--layers", default="4-18")
    p.add_argument("--resize-size", type=int, default=448)
    p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--layer-fusion", choices=["mean", "concat"], default="mean")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-fp16", action="store_true")

    p.add_argument(
        "--patches-per-image",
        type=int,
        default=64,
        help="Deterministic patch subsample per image. 64/784 is usually ample for Stage-1 VAE training.",
    )
    p.add_argument("--patch-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2, help="DINO inference batch size")
    p.add_argument(
        "--images-per-shard",
        type=int,
        default=1024,
        help="Number of images represented in each .pt shard",
    )
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-items", type=int, default=0, help="0 = all; useful for smoke test")
    p.add_argument("--cache-dir", default="cache/realiad_dino_source_normal_compact")
    return p


def main() -> int:
    args = build_parser().parse_args()
    set_seed(args.patch_seed)

    classes = parse_csv_set(args.classes)
    splits = parse_csv_set(args.splits) or {"train"}
    records = load_realiad_records(args.root, args.json_dir, classes)
    selected = [r for r in records if r.split in splits]
    if not args.include_defects:
        selected = [r for r in selected if r.is_good]
    selected = sorted(selected, key=lambda r: (r.category, r.split, r.relative_path))
    if args.max_items > 0:
        selected = selected[: args.max_items]

    inv = inventory(selected)
    if inv["missing_images"]:
        raise FileNotFoundError(
            f"{inv['missing_images']} selected image files are missing; run Stage-0 inspect first"
        )
    if not selected:
        raise RuntimeError("No records selected")

    cfg = Stage0Config(
        model_name=args.model_name,
        layers=parse_layers(args.layers),
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        layer_fusion=args.layer_fusion,
        support_augmentation="paper_geometric",
        query_view="identity",
        top_fraction=0.01,
        knn=1,
        shots=4,
        support_seed=0,
        cache_dtype=args.dtype,
    )
    fingerprint = config_fingerprint(cfg)
    out = Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ViT-L/14 fused patch dim is 1024 under mean fusion. For concat we infer
    # after the first batch, but 1024 remains a conservative estimate only for
    # default mean mode.
    estimate_dim = 1024 if args.layer_fusion == "mean" else 1024 * len(cfg.layers)
    est = _estimate_bytes(len(selected), args.patches_per_image, estimate_dim, args.dtype)
    print(f"selected images: {len(selected)}")
    print(f"patches/image: {args.patches_per_image}")
    print(f"estimated raw tensor storage: {_human_gib(est)}")
    print("(actual shard files add a small amount of metadata overhead)")

    metadata = {
        "schema": "abmg.compact_source_patch_cache.v1",
        "purpose": "Stage-1 source-normal factorizer training; not target evaluation",
        "config": asdict(cfg),
        "config_fingerprint": fingerprint,
        "splits": sorted(splits),
        "include_defects": bool(args.include_defects),
        "patches_per_image": int(args.patches_per_image),
        "patch_seed": int(args.patch_seed),
        "images_per_shard": int(args.images_per_shard),
        "n_selected": len(selected),
        "estimated_raw_tensor_bytes": est,
        "data_access": {
            "feature_shards": "sensor-derived features only; no labels, masks, or defect-source fields",
            "target_policy": "do not persist full target patch grids; process target batches on demand",
        },
    }
    _json_dump(metadata, out / "cache_metadata.json")
    _json_dump(inv, out / "dataset_inventory.json")

    manifest = [
        {
            "schema": "abmg.compact_source_manifest.v1",
            "image_id": r.image_id,
            "category": r.category,
            "split": r.split,
            "relative_path": r.relative_path,
            "config_fingerprint": fingerprint,
        }
        for r in selected
    ]
    _write_jsonl(manifest, out / "manifest_public.jsonl")

    sensor = PaperAlignedFrozenSensor(cfg, device=args.device, use_fp16=not args.no_fp16)
    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    n_done = 0
    n_skipped = 0
    shard_paths: List[str] = []
    try:
        for shard_idx, shard_records in enumerate(_chunks(selected, args.images_per_shard)):
            final_path = out / "shards" / f"shard_{shard_idx:05d}.pt"
            shard_paths.append(str(final_path))
            if args.resume and final_path.is_file():
                n_skipped += len(shard_records)
                print(f"skip complete shard {shard_idx:05d}: {len(shard_records)} images")
                continue

            patch_chunks: List[torch.Tensor] = []
            global_chunks: List[torch.Tensor] = []
            patch_index_chunks: List[torch.Tensor] = []
            image_ids: List[str] = []
            categories: List[str] = []
            relative_paths: List[str] = []

            for batch_records in _chunks(shard_records, args.batch_size):
                patch, global_feat = sensor.encode_path_batch(batch_records)
                patch = patch.detach().cpu()
                global_feat = global_feat.detach().cpu().to(save_dtype)

                for i, rec in enumerate(batch_records):
                    idx = _sample_patch_indices(
                        rec.image_id,
                        patch.shape[1],
                        args.patches_per_image,
                        args.patch_seed,
                    )
                    patch_chunks.append(patch[i, idx].to(save_dtype).contiguous())
                    global_chunks.append(global_feat[i].contiguous())
                    patch_index_chunks.append(idx.to(torch.int16))
                    image_ids.append(rec.image_id)
                    categories.append(rec.category)
                    relative_paths.append(rec.relative_path)

            patches = torch.stack(patch_chunks, dim=0)
            globals_ = torch.stack(global_chunks, dim=0)
            patch_indices = torch.stack(patch_index_chunks, dim=0)

            payload = {
                "schema": "abmg.compact_source_patch_shard.v1",
                "config_fingerprint": fingerprint,
                "shard_index": shard_idx,
                "image_ids": image_ids,
                "categories": categories,
                "relative_paths": relative_paths,
                "patch_features": patches,
                "global_features": globals_,
                "patch_indices": patch_indices,
                "grid_hw": cfg.grid_hw,
                "patches_per_image": int(patches.shape[1]),
                "feature_dim": int(patches.shape[2]),
                "dtype": args.dtype,
            }
            _atomic_torch_save(payload, final_path)
            n_done += len(shard_records)
            print(
                f"wrote shard {shard_idx:05d}: images={len(shard_records)} "
                f"done={n_done} skipped={n_skipped} / {len(selected)} "
                f"shape={tuple(patches.shape)}"
            )
    finally:
        sensor.cleanup()

    completion = {
        "schema": "abmg.compact_source_patch_cache_completion.v1",
        "config_fingerprint": fingerprint,
        "n_selected": len(selected),
        "n_written_this_run": n_done,
        "n_skipped_existing": n_skipped,
        "complete": (n_done + n_skipped == len(selected)),
        "n_shards": len(shard_paths),
        "shards": shard_paths,
    }
    _json_dump(completion, out / "cache_completion.json")
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
