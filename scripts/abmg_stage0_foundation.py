#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 0A/0B foundation for Attribution-Aware Bayesian Memory Governance.

Stage 0A
--------
Create a thin, reproducible VisionAD-aligned Real-IAD baseline that establishes
numerical continuity without importing the old v22 controller stack.

Stage 0B
--------
Cache the frozen DINOv2 patch representation once so later R0/R1/R_beta
experiments operate on identical sensory evidence without repeatedly running the
backbone.

The defaults are deliberately aligned with the VisionAD paper/public reference
implementation where that is compatible with the ABMG scaffold:
  * DINOv2-Register ViT-L/14
  * 448 resize -> 392 center crop
  * transformer blocks 4..18 (0-indexed)
  * mean fusion over selected transformer layers
  * support geometric augmentation: 90/180/270 rotations + vertical/horizontal flip
  * image anomaly score: mean of the top 1% anomaly-map pixels

The ABMG decisive experiments use a SINGLE QUERY VIEW.  Therefore this file does
not reproduce VisionAD's pseudo-multi-view query fusion by default.  The purpose
of Stage 0 is a stable sensor substrate for the new paper, not a second full
VisionAD reimplementation.

Important data-access rule
--------------------------
The feature payload contains no target labels, defect-source labels, or masks.
Those are written to a separate ``manifest_offline_gt.jsonl`` file for offline
evaluation only.  Future target-time code should consume ``manifest_public`` and
feature payloads, never ``manifest_offline_gt``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score

# Running ``python scripts/abmg_stage0_foundation.py`` puts scripts/ on sys.path,
# so the predecessor sensor module remains importable without package installation.
from vmb_visionad_new import DINOv2MultiLayerBackbone, patch_nn_anomaly_map


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_LAYERS = tuple(range(4, 19))


# -----------------------------------------------------------------------------
# Configuration and records
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage0Config:
    model_name: str = "dinov2_vitl14_reg"
    layers: Tuple[int, ...] = DEFAULT_LAYERS
    resize_size: int = 448
    crop_size: int = 392
    patch_size: int = 14
    layer_fusion: str = "mean"
    support_augmentation: str = "paper_geometric"
    query_view: str = "identity"
    top_fraction: float = 0.01
    knn: int = 1
    shots: int = 4
    support_seed: int = 0
    cache_dtype: str = "float16"

    @property
    def grid_hw(self) -> Tuple[int, int]:
        if self.crop_size % self.patch_size != 0:
            raise ValueError(
                f"crop_size={self.crop_size} must be divisible by patch_size={self.patch_size}"
            )
        side = self.crop_size // self.patch_size
        return side, side


@dataclass(frozen=True)
class Stage0Record:
    image_id: str
    category: str
    split: str
    image_path: str
    relative_path: str
    is_good: bool
    defect_source: str
    mask_path: Optional[str]
    json_file: str


@dataclass
class FrozenFeaturePayload:
    image_id: str
    category: str
    split: str
    relative_path: str
    patch_features: torch.Tensor
    global_feature: torch.Tensor
    grid_hw: Tuple[int, int]
    model_name: str
    layers: Tuple[int, ...]
    resize_size: int
    crop_size: int
    layer_fusion: str

    def as_torch_dict(self) -> Dict[str, Any]:
        # Deliberately no label, defect_source, or mask in this payload.
        return {
            "schema": "abmg.frozen_dino.v1",
            "image_id": self.image_id,
            "category": self.category,
            "split": self.split,
            "relative_path": self.relative_path,
            "patch_features": self.patch_features,
            "global_feature": self.global_feature,
            "grid_hw": tuple(int(x) for x in self.grid_hw),
            "model_name": self.model_name,
            "layers": tuple(int(x) for x in self.layers),
            "resize_size": int(self.resize_size),
            "crop_size": int(self.crop_size),
            "layer_fusion": self.layer_fusion,
        }


# -----------------------------------------------------------------------------
# Reproducibility and parsing
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def parse_layers(spec: str) -> Tuple[int, ...]:
    """Parse ``4-18`` or ``4,5,6`` into a sorted unique tuple."""
    spec = str(spec or "").strip()
    if not spec:
        return DEFAULT_LAYERS
    vals: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            step = 1 if hi >= lo else -1
            vals.extend(range(lo, hi + step, step))
        else:
            vals.append(int(part))
    out = tuple(sorted(set(vals)))
    if not out:
        raise ValueError(f"No layer indices parsed from: {spec!r}")
    return out


def parse_csv_set(spec: str) -> Optional[set[str]]:
    xs = {x.strip() for x in str(spec or "").split(",") if x.strip()}
    return xs or None


def config_fingerprint(cfg: Stage0Config) -> str:
    raw = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def stable_image_id(category: str, split: str, relative_path: str) -> str:
    key = f"{category}|{split}|{relative_path}".replace("\\", "/")
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


# -----------------------------------------------------------------------------
# Real-IAD protocol reader
# -----------------------------------------------------------------------------


def _resolve_existing(root: Path, prefix: str, rel: Optional[str], *, kind: str) -> Optional[Path]:
    if rel is None or str(rel).strip() == "":
        return None
    rel_s = str(rel).replace("\\", "/").lstrip("/")
    prefix_s = str(prefix or "").replace("\\", "/").strip("/")

    # Support both the modified layout used by the predecessor scripts and the
    # official Real-IAD layout documented by VisionAD.
    candidates: List[Path] = []
    if kind == "image":
        candidates.extend(
            [
                root / "images" / prefix_s / rel_s,
                root / "realiad_1024" / prefix_s / rel_s,
                root / prefix_s / rel_s,
                root / rel_s,
            ]
        )
    else:
        candidates.extend(
            [
                root / "images" / prefix_s / rel_s,
                root / "realiad_1024" / prefix_s / rel_s,
                root / prefix_s / rel_s,
                root / rel_s,
            ]
        )

    for p in candidates:
        if p.is_file():
            return p
    # Return the first canonical candidate even if missing so the inventory can
    # report a useful expected location.
    return candidates[0] if candidates else None


def load_realiad_records(
    root: str,
    json_dir: str,
    classes: Optional[set[str]] = None,
) -> List[Stage0Record]:
    root_p = Path(root).expanduser().resolve()
    json_p = Path(json_dir).expanduser().resolve()
    if not json_p.is_dir():
        raise FileNotFoundError(f"Real-IAD JSON directory not found: {json_p}")

    records: List[Stage0Record] = []
    for jf in sorted(json_p.glob("*.json")):
        category_from_file = jf.stem
        if classes is not None and category_from_file not in classes:
            continue
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta", {}) or {}
        prefix = str(meta.get("prefix", "") or "")
        normal_class = str(meta.get("normal_class", "OK") or "OK")

        for split in ("train", "test"):
            for rec in data.get(split, []) or []:
                category = str(rec.get("category", category_from_file) or category_from_file)
                if classes is not None and category not in classes:
                    continue
                anomaly_class = str(
                    rec.get("anomaly_class", rec.get("defect_type", normal_class)) or normal_class
                )
                image_rel = str(rec.get("image_path", "") or "")
                mask_rel = rec.get("mask_path", None)
                image_path = _resolve_existing(root_p, prefix, image_rel, kind="image")
                mask_path = _resolve_existing(root_p, prefix, mask_rel, kind="mask")

                rel_key = "/".join(x for x in [prefix.strip("/"), image_rel.lstrip("/")] if x)
                records.append(
                    Stage0Record(
                        image_id=stable_image_id(category, split, rel_key),
                        category=category,
                        split=split,
                        image_path=str(image_path) if image_path is not None else "",
                        relative_path=rel_key,
                        is_good=(anomaly_class == normal_class),
                        defect_source=anomaly_class,
                        mask_path=str(mask_path) if mask_path is not None else None,
                        json_file=jf.name,
                    )
                )
    return records


def inventory(records: Sequence[Stage0Record]) -> Dict[str, Any]:
    cats = sorted({r.category for r in records})
    by_cat: Dict[str, Dict[str, int]] = {}
    missing_images = 0
    missing_masks = 0
    for c in cats:
        rs = [r for r in records if r.category == c]
        by_cat[c] = {
            "train": sum(r.split == "train" for r in rs),
            "test": sum(r.split == "test" for r in rs),
            "normal": sum(r.is_good for r in rs),
            "defect": sum(not r.is_good for r in rs),
        }
    for r in records:
        if not r.image_path or not Path(r.image_path).is_file():
            missing_images += 1
        if (not r.is_good) and r.mask_path and not Path(r.mask_path).is_file():
            missing_masks += 1
    return {
        "n_records": len(records),
        "n_categories": len(cats),
        "categories": cats,
        "missing_images": missing_images,
        "missing_declared_masks": missing_masks,
        "by_category": by_cat,
    }


# -----------------------------------------------------------------------------
# Paper-aligned frozen sensor
# -----------------------------------------------------------------------------


def paper_support_variants(img: Image.Image) -> List[Image.Image]:
    """Geometric support expansion used by the public VisionAD reference code.

    Identity + 90/180/270 degree rotations + vertical + horizontal flips.
    """
    return [
        img,
        img.transpose(Image.Transpose.ROTATE_90),
        img.transpose(Image.Transpose.ROTATE_180),
        img.transpose(Image.Transpose.ROTATE_270),
        ImageOps.flip(img),
        ImageOps.mirror(img),
    ]


def select_supports(
    records: Sequence[Stage0Record],
    category: str,
    shots: int,
    seed: int,
) -> List[Stage0Record]:
    candidates = sorted(
        [r for r in records if r.category == category and r.split == "train" and r.is_good],
        key=lambda r: r.relative_path,
    )
    if len(candidates) < int(shots):
        raise RuntimeError(
            f"category={category!r} has only {len(candidates)} normal train images; shots={shots}"
        )
    gen = torch.Generator(device="cpu")
    # Category-specific offset keeps support draws independent while reproducible.
    cat_offset = int(hashlib.sha1(category.encode("utf-8")).hexdigest()[:8], 16)
    gen.manual_seed(int(seed) + cat_offset)
    idx = torch.randperm(len(candidates), generator=gen)[: int(shots)].tolist()
    return [candidates[i] for i in idx]


class PaperAlignedFrozenSensor:
    """Thin single-view VisionAD-style nearest-neighbour sensor.

    This is intentionally smaller than the predecessor v22 system.  It exposes
    the frozen feature representation and a continuity anomaly score/map only.
    """

    def __init__(self, cfg: Stage0Config, device: str = "cuda", use_fp16: bool = True):
        self.cfg = cfg
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16)
        self.backbone = DINOv2MultiLayerBackbone(
            model_name=cfg.model_name,
            layers=cfg.layers,
            img_size=cfg.crop_size,
            resize_size=cfg.resize_size,
            device=device,
            use_fp16=use_fp16,
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_pil_batch(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_layers, global_features = self.backbone.encode_batch(images)
        if self.cfg.layer_fusion == "mean":
            # Public VisionAD DADE reference implementation uses mean fusion over
            # the selected transformer blocks.
            patch = torch.stack([x.float() for x in patch_layers], dim=0).mean(dim=0)
        elif self.cfg.layer_fusion == "concat":
            patch = torch.cat([x.float() for x in patch_layers], dim=-1)
        else:
            raise ValueError(f"Unsupported layer_fusion={self.cfg.layer_fusion!r}")
        patch = F.normalize(patch, dim=-1)
        global_features = F.normalize(global_features.float(), dim=-1)
        return patch, global_features

    @torch.no_grad()
    def encode_path_batch(self, records: Sequence[Stage0Record]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = [Image.open(r.image_path).convert("RGB") for r in records]
        try:
            return self.encode_pil_batch(images)
        finally:
            for im in images:
                im.close()

    @torch.no_grad()
    def build_category_bank(self, supports: Sequence[Stage0Record]) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for rec in supports:
            base = Image.open(rec.image_path).convert("RGB")
            try:
                variants = (
                    paper_support_variants(base)
                    if self.cfg.support_augmentation == "paper_geometric"
                    else [base]
                )
                p, _ = self.encode_pil_batch(variants)
                chunks.append(p.reshape(-1, p.shape[-1]).detach().cpu())
            finally:
                base.close()
        bank = torch.cat(chunks, dim=0)
        dtype = torch.float16 if (self.use_fp16 and self.device.type == "cuda") else torch.float32
        return F.normalize(bank.float(), dim=-1).to(self.device, dtype=dtype)

    @torch.no_grad()
    def infer_one(self, bank: torch.Tensor, rec: Stage0Record) -> Tuple[float, torch.Tensor]:
        img = Image.open(rec.image_path).convert("RGB")
        try:
            p, _ = self.encode_pil_batch([img])
        finally:
            img.close()
        q = p[0]
        amap = patch_nn_anomaly_map(
            q,
            bank,
            self.cfg.grid_hw,
            (self.cfg.crop_size, self.cfg.crop_size),
            knn=int(self.cfg.knn),
            chunk=4096,
        )
        flat = amap.flatten()
        k = max(1, int(round(float(self.cfg.top_fraction) * flat.numel())))
        score = float(torch.topk(flat, k=k, largest=True).values.mean().item())
        return score, amap.detach().cpu()

    def cleanup(self) -> None:
        self.backbone.cleanup()


# -----------------------------------------------------------------------------
# Stage 0A: continuity baseline
# -----------------------------------------------------------------------------


def run_continuity(args: argparse.Namespace) -> int:
    classes = parse_csv_set(args.classes)
    records = load_realiad_records(args.root, args.json_dir, classes)
    inv = inventory(records)
    if inv["missing_images"]:
        raise FileNotFoundError(
            f"{inv['missing_images']} indexed image files are missing. Run the inspect subcommand first."
        )

    cfg = Stage0Config(
        model_name=args.model_name,
        layers=parse_layers(args.layers),
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        layer_fusion=args.layer_fusion,
        support_augmentation=args.support_augmentation,
        top_fraction=args.top_fraction,
        knn=args.knn,
        shots=args.shots,
        support_seed=args.seed,
        cache_dtype="float16" if not args.no_fp16 else "float32",
    )
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump({"schema": "abmg.stage0.config.v1", **asdict(cfg)}, out_dir / "resolved_config.json")
    _json_dump(inv, out_dir / "dataset_inventory.json")

    categories = sorted({r.category for r in records})
    if args.max_classes and args.max_classes > 0:
        categories = categories[: args.max_classes]

    sensor = PaperAlignedFrozenSensor(cfg, device=args.device, use_fp16=not args.no_fp16)
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "schema": "abmg.stage0.continuity.v1",
        "config_fingerprint": config_fingerprint(cfg),
        "categories": {},
        "paper_alignment": {
            "backbone": "DINOv2-Register ViT-L/14 by default",
            "preprocess": "resize 448 then center crop 392 by default",
            "layers": "blocks 4..18 by default",
            "layer_fusion": "mean",
            "support_augmentation": "identity+90/180/270+vertical/horizontal flip",
            "query_protocol": "single-view identity (ABMG principal protocol)",
            "score": "mean top 1% pixels",
        },
    }

    try:
        for ci, category in enumerate(categories, 1):
            supports = select_supports(records, category, cfg.shots, cfg.support_seed)
            print(f"[{ci}/{len(categories)}] {category}: building support bank from {len(supports)} shots")
            bank = sensor.build_category_bank(supports)
            tests = [r for r in records if r.category == category and r.split == "test"]
            if args.max_test_per_class and args.max_test_per_class > 0:
                tests = tests[: args.max_test_per_class]

            ys: List[int] = []
            ss: List[float] = []
            for j, rec in enumerate(tests, 1):
                score, _ = sensor.infer_one(bank, rec)
                y = 0 if rec.is_good else 1
                ys.append(y)
                ss.append(score)
                rows.append(
                    {
                        "category": category,
                        "image_id": rec.image_id,
                        "relative_path": rec.relative_path,
                        "label_offline_only": y,
                        "defect_source_offline_only": rec.defect_source,
                        "score": score,
                    }
                )
                if args.progress_every > 0 and j % args.progress_every == 0:
                    print(f"  {j}/{len(tests)}")

            if len(set(ys)) >= 2:
                auroc = float(roc_auc_score(ys, ss))
                ap = float(average_precision_score(ys, ss))
            else:
                auroc = float("nan")
                ap = float("nan")
            summary["categories"][category] = {
                "n_test": len(tests),
                "n_normal": int(sum(y == 0 for y in ys)),
                "n_defect": int(sum(y == 1 for y in ys)),
                "image_auroc": auroc,
                "image_ap": ap,
                "support_image_ids": [r.image_id for r in supports],
            }
            print(f"  image AUROC={auroc:.4f} AP={ap:.4f}")

            del bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        sensor.cleanup()

    valid_aurocs = [
        v["image_auroc"]
        for v in summary["categories"].values()
        if np.isfinite(v["image_auroc"])
    ]
    valid_aps = [
        v["image_ap"] for v in summary["categories"].values() if np.isfinite(v["image_ap"])
    ]
    summary["macro_image_auroc"] = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    summary["macro_image_ap"] = float(np.mean(valid_aps)) if valid_aps else float("nan")

    with (out_dir / "per_item_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "category",
            "image_id",
            "relative_path",
            "label_offline_only",
            "defect_source_offline_only",
            "score",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    _json_dump(summary, out_dir / "summary.json")
    print(f"Macro image AUROC: {summary['macro_image_auroc']:.4f}")
    print(f"Wrote Stage 0A outputs to: {out_dir}")
    return 0


# -----------------------------------------------------------------------------
# Stage 0B: frozen DINO cache
# -----------------------------------------------------------------------------


def _feature_cache_path(cache_dir: Path, rec: Stage0Record) -> Path:
    return cache_dir / "features" / rec.category / rec.split / f"{rec.image_id}.pt"


def _batched(xs: Sequence[Stage0Record], batch_size: int) -> Iterator[Sequence[Stage0Record]]:
    for i in range(0, len(xs), int(batch_size)):
        yield xs[i : i + int(batch_size)]


def run_cache(args: argparse.Namespace) -> int:
    classes = parse_csv_set(args.classes)
    records = load_realiad_records(args.root, args.json_dir, classes)
    splits = parse_csv_set(args.splits) or {"train", "test"}
    selected = [r for r in records if r.split in splits]
    if args.normal_only:
        selected = [r for r in selected if r.is_good]
    selected = sorted(selected, key=lambda r: (r.category, r.split, r.relative_path))
    if args.max_items and args.max_items > 0:
        selected = selected[: args.max_items]

    inv = inventory(selected)
    if inv["missing_images"]:
        raise FileNotFoundError(
            f"{inv['missing_images']} selected image files are missing. Run the inspect subcommand first."
        )

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
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema": "abmg.frozen_dino_cache.v1",
        "config": asdict(cfg),
        "config_fingerprint": fingerprint,
        "n_selected": len(selected),
        "splits": sorted(splits),
        "normal_only": bool(args.normal_only),
        "data_access": {
            "feature_payload": "sensor-derived fields only; no labels/masks/defect source",
            "manifest_public": "safe substrate metadata for representation experiments",
            "manifest_offline_gt": "OFFLINE EVALUATOR ONLY; never provide to target-time learner",
        },
    }
    _json_dump(metadata, cache_dir / "cache_metadata.json")
    _json_dump(inv, cache_dir / "dataset_inventory.json")

    # Build manifests before extraction. This makes interrupted runs auditable and
    # allows --resume without reconstructing label-bearing metadata from tensors.
    public_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []
    for rec in selected:
        feat_path = _feature_cache_path(cache_dir, rec)
        public_rows.append(
            {
                "schema": "abmg.cache_manifest_public.v1",
                "image_id": rec.image_id,
                "category": rec.category,
                "split": rec.split,
                "relative_path": rec.relative_path,
                "image_path": rec.image_path,
                "feature_path": str(feat_path),
                "config_fingerprint": fingerprint,
            }
        )
        gt_rows.append(
            {
                "schema": "abmg.cache_manifest_offline_gt.v1",
                "image_id": rec.image_id,
                "label": 0 if rec.is_good else 1,
                "defect_source": rec.defect_source,
                "mask_path": rec.mask_path,
                "offline_only": True,
            }
        )
    _append_jsonl(cache_dir / "manifest_public.jsonl", public_rows)
    _append_jsonl(cache_dir / "manifest_offline_gt.jsonl", gt_rows)

    use_fp16_backbone = not args.no_fp16
    sensor = PaperAlignedFrozenSensor(cfg, device=args.device, use_fp16=use_fp16_backbone)
    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    n_written = 0
    n_skipped = 0
    try:
        for bi, batch in enumerate(_batched(selected, args.batch_size), 1):
            todo = []
            for rec in batch:
                fp = _feature_cache_path(cache_dir, rec)
                if args.resume and fp.is_file():
                    n_skipped += 1
                else:
                    todo.append(rec)
            if not todo:
                continue

            patch, global_features = sensor.encode_path_batch(todo)
            patch = patch.detach().cpu().to(dtype=save_dtype)
            global_features = global_features.detach().cpu().to(dtype=save_dtype)
            for i, rec in enumerate(todo):
                fp = _feature_cache_path(cache_dir, rec)
                fp.parent.mkdir(parents=True, exist_ok=True)
                payload = FrozenFeaturePayload(
                    image_id=rec.image_id,
                    category=rec.category,
                    split=rec.split,
                    relative_path=rec.relative_path,
                    patch_features=patch[i].contiguous(),
                    global_feature=global_features[i].contiguous(),
                    grid_hw=cfg.grid_hw,
                    model_name=cfg.model_name,
                    layers=cfg.layers,
                    resize_size=cfg.resize_size,
                    crop_size=cfg.crop_size,
                    layer_fusion=cfg.layer_fusion,
                )
                torch.save(payload.as_torch_dict(), fp)
                n_written += 1
            if args.progress_every > 0 and (n_written + n_skipped) % args.progress_every < len(todo):
                print(
                    f"cached={n_written} skipped={n_skipped} / {len(selected)} "
                    f"last={todo[-1].category}/{todo[-1].split}"
                )
            del patch, global_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sensor.cleanup()

    completion = {
        "schema": "abmg.frozen_dino_cache_completion.v1",
        "config_fingerprint": fingerprint,
        "n_selected": len(selected),
        "n_written": n_written,
        "n_skipped_existing": n_skipped,
        "complete": (n_written + n_skipped == len(selected)),
    }
    _json_dump(completion, cache_dir / "cache_completion.json")
    print(json.dumps(completion, indent=2))
    return 0


# -----------------------------------------------------------------------------
# Dataset inspection / smoke validation
# -----------------------------------------------------------------------------


def run_inspect(args: argparse.Namespace) -> int:
    classes = parse_csv_set(args.classes)
    records = load_realiad_records(args.root, args.json_dir, classes)
    inv = inventory(records)
    print(json.dumps(inv, indent=2))
    if args.out:
        _json_dump(inv, Path(args.out))
    if inv["missing_images"]:
        print(
            "\nERROR: indexed images are missing. Supported layouts include "
            "root/images/<prefix>/..., root/realiad_1024/<prefix>/..., and root/<prefix>/...",
            file=sys.stderr,
        )
        return 2
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def add_common_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="Real-IAD root directory")
    p.add_argument("--json-dir", required=True, help="Directory containing per-category Real-IAD JSON files")
    p.add_argument("--classes", default="", help="Optional comma-separated category subset")


def add_common_backbone_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-name", default="dinov2_vitl14_reg")
    p.add_argument("--layers", default="4-18", help="Transformer blocks, e.g. 4-18 or 5,11,17,23")
    p.add_argument("--resize-size", type=int, default=448)
    p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--layer-fusion", choices=["mean", "concat"], default="mean")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-fp16", action="store_true", help="Run backbone in fp32")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ABMG Stage 0A/0B: paper-aligned Real-IAD foundation and frozen DINO feature cache"
    )
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inspect", help="Validate Real-IAD indexing/path layout without loading DINO")
    add_common_data_args(pi)
    pi.add_argument("--out", default="")
    pi.set_defaults(func=run_inspect)

    pc = sub.add_parser("continuity", help="Stage 0A: run thin VisionAD-aligned single-view baseline")
    add_common_data_args(pc)
    add_common_backbone_args(pc)
    pc.add_argument("--shots", type=int, default=4)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument(
        "--support-augmentation",
        choices=["paper_geometric", "none"],
        default="paper_geometric",
    )
    pc.add_argument("--top-fraction", type=float, default=0.01)
    pc.add_argument("--knn", type=int, default=1)
    pc.add_argument("--max-classes", type=int, default=0, help="0 = all")
    pc.add_argument("--max-test-per-class", type=int, default=0, help="0 = all")
    pc.add_argument("--progress-every", type=int, default=100)
    pc.add_argument("--out-dir", default="outputs/stage0_continuity")
    pc.set_defaults(func=run_continuity)

    pf = sub.add_parser("cache", help="Stage 0B: cache frozen single-view DINO descriptors")
    add_common_data_args(pf)
    add_common_backbone_args(pf)
    pf.add_argument("--splits", default="train,test", help="Comma-separated train,test")
    pf.add_argument("--normal-only", action="store_true")
    pf.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    pf.add_argument("--batch-size", type=int, default=2)
    pf.add_argument("--max-items", type=int, default=0, help="0 = all; useful for smoke tests")
    pf.add_argument("--resume", action="store_true")
    pf.add_argument("--progress-every", type=int, default=100)
    pf.add_argument("--cache-dir", default="cache/realiad_dino_single_view")
    pf.set_defaults(func=run_cache)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
