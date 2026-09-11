#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Factorizability audit for Attribution-Aware Bayesian Memory Governance.

Purpose
-------
This script is a diagnostic *before* changing the supervisor-approved ABMG
proposal.  It asks a narrower question than C1:

    Where does reusable defect-type structure live in the frozen DINO/v22
    sensory evidence?

It deliberately leaves the deployed anomaly detector unchanged.  The audit
compares several candidate evidence representations and evaluates whether
visual defect-source labels are predictable across *held-out product
categories*.  Target categories for the outer ABMG fold are never opened by
``extract`` or ``evaluate``.

The primary comparison is source-side and category-held-out:

  1. raw_patch
       Mean frozen fused DINO descriptor over selected suspicious patches.
  2. nn_residual
       Signed difference h - n* from the nearest normal support patch.
  3. abs_nn_residual
       Absolute nearest-normal residual.
  4. whitened_nn_residual
       Residual divided by normal-support coordinate scale.
  5. subspace_residual
       Component of h outside a PCA subspace fit only to normal support patches.
  6. layer_distance_profile
       Per-layer mean/max cosine NN distances to normal support patches.

Two patch-localisation conditions are supported:

  * oracle: ground-truth masks are used OFFLINE ONLY to isolate whether the
    representation itself contains defect-type information when the defect is
    localised correctly.
  * sensor: the frozen Stage-0 DINO nearest-normal evidence selects top-K
    suspicious patches, approximating the evidence available to v22 online.

The distinction is critical.  If oracle succeeds but sensor fails, localisation
is the bottleneck.  If both fail, the candidate representation itself is weak.
If both succeed, factorisation/routing is plausible without changing detection.

Ground-truth defect source labels are used only by ``inventory``/``evaluate``
and are marked as offline-only metadata in the extraction artefact.  No target
category labels, masks, or images are used by the source-side audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from abmg_stage0_foundation import (
    PaperAlignedFrozenSensor,
    Stage0Config,
    Stage0Record,
    config_fingerprint,
    load_realiad_records,
    paper_support_variants,
    parse_layers,
    select_supports,
    set_seed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_FOLDS = REPO_ROOT / "configs" / "realiad_folds_v0.json"
REPRESENTATION_NAMES = (
    "raw_patch",
    "nn_residual",
    "abs_nn_residual",
    "whitened_nn_residual",
    "subspace_residual",
    "layer_distance_profile",
)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def _sha_order(seed: int, key: str) -> str:
    return hashlib.sha1(f"{int(seed)}|{key}".encode("utf-8")).hexdigest()


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([float(x) for x in xs if np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _l2_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


@dataclass(frozen=True)
class FoldProtocol:
    outer_fold: int
    all_categories: Tuple[str, ...]
    target_categories: Tuple[str, ...]
    source_categories: Tuple[str, ...]
    source_cv_groups: Tuple[Tuple[str, ...], ...]


def load_fold_protocol(fold: int, folds_path: str | Path = DEFAULT_FOLDS) -> FoldProtocol:
    path = Path(folds_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    all_categories = tuple(str(x) for x in cfg["all_categories"])
    folds = {int(x["fold"]): tuple(str(c) for c in x["target_categories"]) for x in cfg["folds"]}
    if int(fold) not in folds:
        raise ValueError(f"fold={fold} not present in {path}")
    target = folds[int(fold)]
    source = tuple(c for c in all_categories if c not in set(target))
    cv_groups = tuple(folds[k] for k in sorted(folds) if k != int(fold))
    flat = [c for g in cv_groups for c in g]
    if sorted(flat) != sorted(source):
        raise RuntimeError("Source CV groups do not exactly partition source categories")
    return FoldProtocol(
        outer_fold=int(fold),
        all_categories=all_categories,
        target_categories=target,
        source_categories=source,
        source_cv_groups=cv_groups,
    )


def source_defect_records(records: Sequence[Stage0Record], source_categories: Sequence[str]) -> List[Stage0Record]:
    src = set(source_categories)
    return [r for r in records if r.category in src and r.split == "test" and (not r.is_good)]


def deterministic_stratified_defect_sample(
    records: Sequence[Stage0Record], *, max_per_type_per_category: int, seed: int
) -> List[Stage0Record]:
    groups: Dict[Tuple[str, str], List[Stage0Record]] = defaultdict(list)
    for r in records:
        groups[(r.category, r.defect_source)].append(r)
    out: List[Stage0Record] = []
    for key in sorted(groups):
        xs = sorted(groups[key], key=lambda r: _sha_order(seed, r.image_id))
        if int(max_per_type_per_category) > 0:
            xs = xs[: int(max_per_type_per_category)]
        out.extend(xs)
    return sorted(out, key=lambda r: (r.category, r.defect_source, r.relative_path))


def build_inventory(protocol: FoldProtocol, records: Sequence[Stage0Record]) -> Dict[str, Any]:
    defects = source_defect_records(records, protocol.source_categories)
    by_type = Counter(r.defect_source for r in defects)
    by_cat: Dict[str, Counter] = defaultdict(Counter)
    for r in defects:
        by_cat[r.category][r.defect_source] += 1

    cv: List[Dict[str, Any]] = []
    src = set(protocol.source_categories)
    for i, test_group in enumerate(protocol.source_cv_groups):
        test_cats = set(test_group)
        train_cats = src - test_cats
        train_labels = Counter(r.defect_source for r in defects if r.category in train_cats)
        test_labels = Counter(r.defect_source for r in defects if r.category in test_cats)
        shared = sorted(set(train_labels) & set(test_labels))
        covered = sum(v for k, v in test_labels.items() if k in set(shared))
        total = sum(test_labels.values())
        cv.append(
            {
                "source_cv_fold": i,
                "train_categories": sorted(train_cats),
                "test_categories": sorted(test_cats),
                "train_label_counts": dict(sorted(train_labels.items())),
                "test_label_counts": dict(sorted(test_labels.items())),
                "shared_labels": shared,
                "test_image_label_coverage": float(covered / total) if total else float("nan"),
            }
        )

    return {
        "schema": "abmg.factorizability.inventory.v1",
        "outer_fold": protocol.outer_fold,
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "n_source_defect_images": len(defects),
        "defect_type_counts": dict(sorted(by_type.items())),
        "defect_type_category_coverage": {
            d: int(sum(d in by_cat[c] for c in by_cat)) for d in sorted(by_type)
        },
        "by_category": {c: dict(sorted(by_cat[c].items())) for c in sorted(by_cat)},
        "source_category_cv": cv,
    }


def mask_to_patch_mask(mask_path: str, cfg: Stage0Config) -> torch.Tensor:
    """Apply the sensor geometry and max-pool a tiny mask to the patch grid."""
    p = Path(mask_path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    im = Image.open(p).convert("L")
    try:
        im = im.resize((cfg.resize_size, cfg.resize_size), resample=Image.Resampling.NEAREST)
        if cfg.crop_size != cfg.resize_size:
            left = (cfg.resize_size - cfg.crop_size) // 2
            top = (cfg.resize_size - cfg.crop_size) // 2
            im = im.crop((left, top, left + cfg.crop_size, top + cfg.crop_size))
        arr = np.asarray(im) > 0
    finally:
        im.close()
    gh, gw = cfg.grid_hw
    ph = cfg.crop_size // gh
    pw = cfg.crop_size // gw
    patch = arr.reshape(gh, ph, gw, pw).max(axis=(1, 3))
    return torch.from_numpy(patch.reshape(-1).astype(np.bool_))


def selected_indices(
    mode: str,
    patch_scores: torch.Tensor,
    oracle_mask: Optional[torch.Tensor],
    sensor_topk: int,
) -> Optional[torch.Tensor]:
    if mode == "oracle":
        if oracle_mask is None:
            return None
        idx = torch.nonzero(oracle_mask, as_tuple=False).flatten()
        return idx if idx.numel() else None
    if mode == "sensor":
        k = min(max(1, int(sensor_topk)), int(patch_scores.numel()))
        return torch.topk(patch_scores, k=k, largest=True).indices.cpu()
    raise ValueError(mode)


@dataclass
class CategorySupportState:
    fused_bank: torch.Tensor
    layer_banks: List[torch.Tensor]
    normal_mean: torch.Tensor
    normal_std: torch.Tensor
    pca_basis: torch.Tensor
    pca_explained_fraction: float
    support_image_ids: List[str]


def _deterministic_rows(x: torch.Tensor, n: int, seed: int, salt: str) -> torch.Tensor:
    if x.shape[0] <= int(n) or int(n) <= 0:
        return x
    g = torch.Generator(device="cpu")
    h = int(hashlib.sha1(f"{seed}|{salt}".encode("utf-8")).hexdigest()[:8], 16)
    g.manual_seed(int(seed) + h)
    idx = torch.randperm(x.shape[0], generator=g)[: int(n)].to(x.device)
    return x[idx]


@torch.no_grad()
def build_support_state(
    sensor: PaperAlignedFrozenSensor,
    supports: Sequence[Stage0Record],
    *,
    pca_rank: int,
    pca_max_patches: int,
    whiten_floor_ratio: float,
    seed: int,
    category: str,
) -> CategorySupportState:
    layer_chunks: Optional[List[List[torch.Tensor]]] = None
    fused_chunks: List[torch.Tensor] = []
    out_dtype = torch.float16 if sensor.device.type == "cuda" else torch.float32

    for rec in supports:
        base = Image.open(rec.image_path).convert("RGB")
        try:
            variants = paper_support_variants(base)
            layers, _ = sensor.backbone.encode_batch(variants)
        finally:
            base.close()
        if layer_chunks is None:
            layer_chunks = [[] for _ in layers]
        for li, x in enumerate(layers):
            layer_chunks[li].append(x.reshape(-1, x.shape[-1]).to(dtype=out_dtype))
        fused = sum(x.float() for x in layers) / float(len(layers))
        fused = F.normalize(fused, dim=-1)
        fused_chunks.append(fused.reshape(-1, fused.shape[-1]).to(dtype=out_dtype))

    if layer_chunks is None:
        raise RuntimeError(f"No support features for category={category}")
    fused_bank = torch.cat(fused_chunks, dim=0).contiguous()
    layer_banks = [torch.cat(xs, dim=0).contiguous() for xs in layer_chunks]

    sample = _deterministic_rows(fused_bank, int(pca_max_patches), seed, category).float()
    normal_mean = sample.mean(dim=0)
    std = sample.std(dim=0, unbiased=False)
    floor = float(std.median().item()) * float(whiten_floor_ratio)
    normal_std = std.clamp_min(max(floor, 1e-6))

    centered = sample - normal_mean
    q = min(int(pca_rank), int(centered.shape[0]) - 1, int(centered.shape[1]))
    if q <= 0:
        raise RuntimeError("PCA rank became non-positive")
    _, s, v = torch.pca_lowrank(centered, q=q, center=False, niter=2)
    total_ss = float(centered.square().sum().item())
    explained = float(s.square().sum().item() / max(total_ss, 1e-12))

    return CategorySupportState(
        fused_bank=fused_bank,
        layer_banks=layer_banks,
        normal_mean=normal_mean,
        normal_std=normal_std,
        pca_basis=v.float(),
        pca_explained_fraction=explained,
        support_image_ids=[r.image_id for r in supports],
    )


@torch.no_grad()
def nearest_cosine(query: torch.Tensor, bank: torch.Tensor, chunk: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    sims_out: List[torch.Tensor] = []
    idx_out: List[torch.Tensor] = []
    for i in range(0, query.shape[0], int(chunk)):
        q = query[i : i + int(chunk)].to(dtype=bank.dtype)
        s = q @ bank.T
        vals, idx = s.max(dim=1)
        sims_out.append(vals.float())
        idx_out.append(idx)
    return torch.cat(sims_out, dim=0), torch.cat(idx_out, dim=0)


@torch.no_grad()
def build_representation_vector(
    name: str,
    *,
    q_fused: torch.Tensor,
    q_layers: List[torch.Tensor],
    chosen_idx: torch.Tensor,
    nn_idx_all: torch.Tensor,
    support: CategorySupportState,
) -> torch.Tensor:
    idx_dev = chosen_idx.to(q_fused.device)
    q = q_fused[idx_dev].float()
    nn = support.fused_bank[nn_idx_all[idx_dev]].float()
    residual = q - nn

    if name == "raw_patch":
        return q.mean(dim=0)
    if name == "nn_residual":
        return residual.mean(dim=0)
    if name == "abs_nn_residual":
        return residual.abs().mean(dim=0)
    if name == "whitened_nn_residual":
        return (residual / support.normal_std).mean(dim=0)
    if name == "subspace_residual":
        centered = q - support.normal_mean
        proj = (centered @ support.pca_basis) @ support.pca_basis.T
        return (centered - proj).mean(dim=0)
    if name == "layer_distance_profile":
        means: List[torch.Tensor] = []
        maxima: List[torch.Tensor] = []
        for li, ql_all in enumerate(q_layers):
            ql = ql_all[idx_dev]
            best, _ = nearest_cosine(ql, support.layer_banks[li], chunk=max(1, int(ql.shape[0])))
            dist = 1.0 - best
            means.append(dist.mean().view(1))
            maxima.append(dist.max().view(1))
        return torch.cat(means + maxima, dim=0)
    raise ValueError(name)


def _append_rep(store: Dict[str, Dict[str, Any]], key: str, item_idx: int, vec: torch.Tensor) -> None:
    slot = store.setdefault(key, {"item_indices": [], "vectors": []})
    slot["item_indices"].append(int(item_idx))
    slot["vectors"].append(vec.detach().cpu().to(torch.float16))


def _patch_localisation_metrics(
    scores: torch.Tensor, mask: Optional[torch.Tensor], top_idx: torch.Tensor
) -> Dict[str, Any]:
    if mask is None:
        return {"mask_available": False}
    y = mask.cpu().numpy().astype(np.int64)
    s = scores.detach().cpu().numpy().astype(np.float64)
    pred = np.zeros_like(y, dtype=np.int64)
    pred[top_idx.detach().cpu().numpy()] = 1
    inter = int(np.logical_and(pred == 1, y == 1).sum())
    n_gt = int(y.sum())
    n_pred = int(pred.sum())
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan")
    return {
        "mask_available": True,
        "n_positive_patches": n_gt,
        "sensor_topk": n_pred,
        "sensor_hit_any": bool(inter > 0),
        "sensor_patch_precision": float(inter / max(n_pred, 1)),
        "sensor_patch_recall": float(inter / max(n_gt, 1)),
        "sensor_patch_auroc": auc,
    }


def run_inventory(args: argparse.Namespace) -> int:
    protocol = load_fold_protocol(args.fold, args.folds)
    records = load_realiad_records(args.root, args.json_dir, set(protocol.source_categories))
    inv = build_inventory(protocol, records)
    out = Path(args.out_dir) / f"fold_{args.fold}" / "inventory.json"
    _json_dump(inv, out)
    print(json.dumps(inv, indent=2, allow_nan=True))
    print(f"Wrote {out}")
    return 0


def run_extract(args: argparse.Namespace) -> int:
    protocol = load_fold_protocol(args.fold, args.folds)
    # HARD DATA-ACCESS BOUNDARY: only source category JSON files are read.
    records = load_realiad_records(args.root, args.json_dir, set(protocol.source_categories))
    defects_all = source_defect_records(records, protocol.source_categories)
    defects = deterministic_stratified_defect_sample(
        defects_all,
        max_per_type_per_category=args.max_per_type_per_category,
        seed=args.seed,
    )
    if not defects:
        raise RuntimeError("No source defect test records found")

    cfg = Stage0Config(
        model_name=args.model_name,
        layers=parse_layers(args.layers),
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        layer_fusion="mean",
        support_augmentation="paper_geometric",
        query_view="identity",
        top_fraction=0.01,
        knn=1,
        shots=args.shots,
        support_seed=args.support_seed,
        cache_dtype="float16" if not args.no_fp16 else "float32",
    )
    set_seed(args.seed)
    localizers = tuple(x.strip() for x in args.localizers.split(",") if x.strip())
    for x in localizers:
        if x not in {"oracle", "sensor"}:
            raise ValueError(f"Unknown localizer={x}")
    reps = tuple(x.strip() for x in args.representations.split(",") if x.strip())
    unknown = sorted(set(reps) - set(REPRESENTATION_NAMES))
    if unknown:
        raise ValueError(f"Unknown representations: {unknown}; allowed={REPRESENTATION_NAMES}")

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(
        {
            "schema": "abmg.factorizability.extract_config.v1",
            "outer_fold": args.fold,
            "target_categories_untouched": list(protocol.target_categories),
            "source_categories": list(protocol.source_categories),
            "source_cv_groups": [list(x) for x in protocol.source_cv_groups],
            "sensor": asdict(cfg),
            "sensor_fingerprint": config_fingerprint(cfg),
            "localizers": list(localizers),
            "representations": list(reps),
            "sensor_topk": args.sensor_topk,
            "pca_rank": args.pca_rank,
            "pca_max_patches": args.pca_max_patches,
            "whiten_floor_ratio": args.whiten_floor_ratio,
            "max_per_type_per_category": args.max_per_type_per_category,
            "seed": args.seed,
            "note": "Target category files are not read. Defect labels/masks are offline-only audit metadata.",
        },
        out_dir / "resolved_config.json",
    )

    items: List[Dict[str, Any]] = []
    for r in defects:
        items.append(
            {
                "image_id": r.image_id,
                "category": r.category,
                "relative_path": r.relative_path,
                "defect_source_offline_only": r.defect_source,
                "mask_path_offline_only": r.mask_path,
            }
        )
    item_index = {r.image_id: i for i, r in enumerate(defects)}
    by_category: Dict[str, List[Stage0Record]] = defaultdict(list)
    for r in defects:
        by_category[r.category].append(r)

    rep_store: Dict[str, Dict[str, Any]] = {}
    localisation_rows: List[Dict[str, Any]] = []
    support_meta: Dict[str, Any] = {}

    sensor = PaperAlignedFrozenSensor(cfg, device=args.device, use_fp16=not args.no_fp16)
    try:
        categories = sorted(by_category)
        for ci, category in enumerate(categories, 1):
            supports = select_supports(records, category, cfg.shots, cfg.support_seed)
            print(f"[{ci}/{len(categories)}] {category}: support={len(supports)}, defects={len(by_category[category])}")
            state = build_support_state(
                sensor,
                supports,
                pca_rank=args.pca_rank,
                pca_max_patches=args.pca_max_patches,
                whiten_floor_ratio=args.whiten_floor_ratio,
                seed=args.seed,
                category=category,
            )
            support_meta[category] = {
                "support_image_ids": state.support_image_ids,
                "n_fused_support_patches": int(state.fused_bank.shape[0]),
                "pca_rank": int(state.pca_basis.shape[1]),
                "pca_explained_fraction_on_support_sample": state.pca_explained_fraction,
            }

            cat_recs = by_category[category]
            for start in range(0, len(cat_recs), int(args.query_batch_size)):
                batch = cat_recs[start : start + int(args.query_batch_size)]
                images = [Image.open(r.image_path).convert("RGB") for r in batch]
                try:
                    layer_batch, _ = sensor.backbone.encode_batch(images)
                finally:
                    for im in images:
                        im.close()
                fused_batch = sum(x.float() for x in layer_batch) / float(len(layer_batch))
                fused_batch = F.normalize(fused_batch, dim=-1)

                for bi, rec in enumerate(batch):
                    qf = fused_batch[bi]
                    qlayers = [x[bi] for x in layer_batch]
                    best_sim, nn_idx = nearest_cosine(qf, state.fused_bank, chunk=args.nn_chunk)
                    patch_scores = 1.0 - best_sim
                    oracle_mask: Optional[torch.Tensor] = None
                    if rec.mask_path and Path(rec.mask_path).is_file():
                        try:
                            oracle_mask = mask_to_patch_mask(rec.mask_path, cfg)
                        except Exception as e:
                            print(f"  WARN mask failed for {rec.relative_path}: {e}")

                    sensor_idx = selected_indices("sensor", patch_scores, oracle_mask, args.sensor_topk)
                    assert sensor_idx is not None
                    loc = _patch_localisation_metrics(patch_scores, oracle_mask, sensor_idx)
                    loc.update(
                        {
                            "item_index": int(item_index[rec.image_id]),
                            "image_id": rec.image_id,
                            "category": rec.category,
                            "defect_source_offline_only": rec.defect_source,
                        }
                    )
                    localisation_rows.append(loc)

                    for localizer in localizers:
                        chosen = selected_indices(localizer, patch_scores, oracle_mask, args.sensor_topk)
                        if chosen is None:
                            continue
                        for rep_name in reps:
                            vec = build_representation_vector(
                                rep_name,
                                q_fused=qf,
                                q_layers=qlayers,
                                chosen_idx=chosen,
                                nn_idx_all=nn_idx,
                                support=state,
                            )
                            _append_rep(rep_store, f"{localizer}/{rep_name}", item_index[rec.image_id], vec)

                done = min(start + len(batch), len(cat_recs))
                if args.progress_every > 0 and (done % args.progress_every == 0 or done == len(cat_recs)):
                    print(f"  {done}/{len(cat_recs)}")

            del state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sensor.cleanup()

    packed_reps: Dict[str, Dict[str, Any]] = {}
    for key, slot in sorted(rep_store.items()):
        vectors = slot["vectors"]
        packed_reps[key] = {
            "item_indices": torch.tensor(slot["item_indices"], dtype=torch.int64),
            "features": torch.stack(vectors, dim=0) if vectors else torch.empty((0, 0), dtype=torch.float16),
        }

    artifact = {
        "schema": "abmg.factorizability.features.v1",
        "outer_fold": args.fold,
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "source_cv_groups": [list(x) for x in protocol.source_cv_groups],
        "sensor_fingerprint": config_fingerprint(cfg),
        "items": items,
        "representations": packed_reps,
        "localisation": localisation_rows,
        "support": support_meta,
        "offline_only_notice": "defect_source and mask fields are evaluator-only and must not enter target-time logic",
    }
    artifact_path = out_dir / "factorizability_features.pt"
    torch.save(artifact, artifact_path)

    aucs = [_to_float(x.get("sensor_patch_auroc")) for x in localisation_rows]
    hits = [1.0 if x.get("sensor_hit_any") else 0.0 for x in localisation_rows if x.get("mask_available")]
    recalls = [_to_float(x.get("sensor_patch_recall")) for x in localisation_rows if x.get("mask_available")]
    summary = {
        "schema": "abmg.factorizability.extract_summary.v1",
        "outer_fold": args.fold,
        "target_categories_untouched": list(protocol.target_categories),
        "n_sampled_source_defect_images": len(defects),
        "representations": {
            k: {
                "n": int(v["features"].shape[0]),
                "dim": int(v["features"].shape[1]) if v["features"].ndim == 2 else 0,
            }
            for k, v in packed_reps.items()
        },
        "sensor_localisation": {
            "n_with_masks": len(hits),
            "hit_at_k": float(np.mean(hits)) if hits else float("nan"),
            "mean_patch_recall_at_k": float(np.nanmean(recalls)) if recalls else float("nan"),
            "mean_per_image_patch_auroc": float(np.nanmean(aucs)) if aucs else float("nan"),
            "sensor_topk": int(args.sensor_topk),
        },
        "artifact": str(artifact_path),
    }
    _json_dump(summary, out_dir / "extract_summary.json")
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0


def _fit_centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(x_train)
    a = _l2_np(scaler.transform(x_train).astype(np.float64))
    b = _l2_np(scaler.transform(x_test).astype(np.float64))
    classes = np.array(sorted(set(y_train.tolist())))
    cents = []
    for c in classes:
        z = a[y_train == c].mean(axis=0, keepdims=True)
        cents.append(_l2_np(z)[0])
    C = np.stack(cents, axis=0)
    return classes[(b @ C.T).argmax(axis=1)]


def _fit_linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    scaler = StandardScaler().fit(x_train)
    a = scaler.transform(x_train)
    b = scaler.transform(x_test)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(a, y_train)
    return clf.predict(b)


def _shuffle_labels(y: np.ndarray, seed: int) -> np.ndarray:
    g = np.random.default_rng(int(seed))
    out = np.array(y, copy=True)
    g.shuffle(out)
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_representation(
    X: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    cv_groups: Sequence[Sequence[str]],
    *,
    min_train_per_label: int,
    min_test_per_label: int,
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    folds_out: List[Dict[str, Any]] = []
    all_source = set(categories.tolist())
    for fi, group in enumerate(cv_groups):
        test_cats = set(group)
        train_cats = all_source - test_cats
        train0 = np.array([c in train_cats for c in categories], dtype=bool)
        test0 = np.array([c in test_cats for c in categories], dtype=bool)
        tr_counts = Counter(labels[train0].tolist())
        te_counts = Counter(labels[test0].tolist())
        shared = sorted(
            c for c in set(tr_counts) & set(te_counts)
            if tr_counts[c] >= int(min_train_per_label) and te_counts[c] >= int(min_test_per_label)
        )
        shared_set = set(shared)
        train = train0 & np.array([y in shared_set for y in labels], dtype=bool)
        test = test0 & np.array([y in shared_set for y in labels], dtype=bool)
        total_test = int(test0.sum())
        if len(shared) < 2 or int(train.sum()) == 0 or int(test.sum()) == 0:
            folds_out.append(
                {
                    "source_cv_fold": fi,
                    "test_categories": sorted(test_cats),
                    "status": "insufficient_shared_labels",
                    "shared_labels": shared,
                    "test_label_coverage": float(test.sum() / max(total_test, 1)),
                }
            )
            continue
        xt, yt = X[train], labels[train]
        xv, yv = X[test], labels[test]
        pred_c = _fit_centroid_predict(xt, yt, xv)
        pred_l = _fit_linear_predict(xt, yt, xv, seed + fi)
        perm_f1: List[float] = []
        for pi in range(int(permutations)):
            ys = _shuffle_labels(yt, seed + 1000 * (fi + 1) + pi)
            if len(set(ys.tolist())) < 2:
                continue
            pp = _fit_linear_predict(xt, ys, xv, seed + 10000 + pi)
            perm_f1.append(float(f1_score(yv, pp, average="macro", zero_division=0)))
        pm, ps = _mean_std(perm_f1)
        folds_out.append(
            {
                "source_cv_fold": fi,
                "test_categories": sorted(test_cats),
                "status": "ok",
                "shared_labels": shared,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "n_total_test_before_label_filter": total_test,
                "test_label_coverage": float(test.sum() / max(total_test, 1)),
                "centroid": _metrics(yv, pred_c),
                "linear_probe": _metrics(yv, pred_l),
                "linear_probe_label_shuffle_macro_f1": {"n": len(perm_f1), "mean": pm, "std": ps},
            }
        )

    ok = [x for x in folds_out if x.get("status") == "ok"]
    aggregate: Dict[str, Any] = {"n_valid_cv_folds": len(ok)}
    for method in ("centroid", "linear_probe"):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            vals = [x[method][metric] for x in ok]
            m, s = _mean_std(vals)
            aggregate[f"{method}_{metric}_mean"] = m
            aggregate[f"{method}_{metric}_std"] = s
    aggregate["test_label_coverage_mean"] = _mean_std([x["test_label_coverage"] for x in ok])[0]
    aggregate["linear_probe_label_shuffle_macro_f1_mean"] = _mean_std(
        [x["linear_probe_label_shuffle_macro_f1"]["mean"] for x in ok]
    )[0]
    a = aggregate.get("linear_probe_macro_f1_mean", float("nan"))
    b = aggregate.get("linear_probe_label_shuffle_macro_f1_mean", float("nan"))
    aggregate["linear_probe_macro_f1_gain_over_shuffle"] = float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")
    return {"folds": folds_out, "aggregate": aggregate}


def run_evaluate(args: argparse.Namespace) -> int:
    path = Path(args.features)
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if artifact.get("schema") != "abmg.factorizability.features.v1":
        raise ValueError(f"Unexpected artifact schema: {artifact.get('schema')}")
    if int(artifact["outer_fold"]) != int(args.fold):
        raise ValueError(f"Artifact fold={artifact['outer_fold']} does not match --fold={args.fold}")

    items = artifact["items"]
    result: Dict[str, Any] = {
        "schema": "abmg.factorizability.evaluation.v1",
        "outer_fold": args.fold,
        "target_categories_untouched": artifact["target_categories_untouched"],
        "source_cv_groups": artifact["source_cv_groups"],
        "decision_scope": (
            "Diagnostic only. A representation is promising if defect-source structure generalises across "
            "held-out source product categories and exceeds label-shuffle controls. This does not by itself "
            "establish C1 or justify changing the supervisor proposal."
        ),
        "representations": {},
    }

    for key, slot in sorted(artifact["representations"].items()):
        idx = slot["item_indices"].numpy().astype(np.int64)
        X = slot["features"].float().numpy().astype(np.float64)
        cats = np.array([items[i]["category"] for i in idx], dtype=object)
        labels = np.array([items[i]["defect_source_offline_only"] for i in idx], dtype=object)
        print(f"Evaluating {key}: n={len(idx)} d={X.shape[1]}")
        result["representations"][key] = evaluate_representation(
            X,
            cats,
            labels,
            artifact["source_cv_groups"],
            min_train_per_label=args.min_train_per_label,
            min_test_per_label=args.min_test_per_label,
            permutations=args.permutations,
            seed=args.seed,
        )

    ranking = []
    for key, obj in result["representations"].items():
        agg = obj["aggregate"]
        ranking.append(
            {
                "representation": key,
                "linear_probe_macro_f1": agg.get("linear_probe_macro_f1_mean"),
                "centroid_macro_f1": agg.get("centroid_macro_f1_mean"),
                "gain_over_shuffle": agg.get("linear_probe_macro_f1_gain_over_shuffle"),
                "label_coverage": agg.get("test_label_coverage_mean"),
            }
        )
    ranking.sort(
        key=lambda x: -999.0 if not np.isfinite(_to_float(x["linear_probe_macro_f1"])) else _to_float(x["linear_probe_macro_f1"]),
        reverse=True,
    )
    result["ranking_by_linear_probe_macro_f1"] = ranking

    loc = artifact.get("localisation", [])
    masked = [x for x in loc if x.get("mask_available")]
    result["sensor_localisation"] = {
        "n_with_masks": len(masked),
        "hit_at_k": float(np.mean([1.0 if x.get("sensor_hit_any") else 0.0 for x in masked])) if masked else float("nan"),
        "mean_patch_recall_at_k": float(np.nanmean([_to_float(x.get("sensor_patch_recall")) for x in masked])) if masked else float("nan"),
        "mean_per_image_patch_auroc": float(np.nanmean([_to_float(x.get("sensor_patch_auroc")) for x in masked])) if masked else float("nan"),
    }

    out = Path(args.out_dir) / f"fold_{args.fold}" / "evaluation.json"
    _json_dump(result, out)
    print(json.dumps({"ranking": ranking, "sensor_localisation": result["sensor_localisation"]}, indent=2, allow_nan=True))
    print(f"Wrote {out}")
    return 0


def add_common_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--json-dir", type=str, required=True)
    p.add_argument("--folds", type=str, default=str(DEFAULT_FOLDS))
    p.add_argument("--out-dir", type=str, default="outputs/factorizability_audit")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory", help="CPU-only source defect-label/category coverage audit")
    add_common_dataset_args(p)
    p.set_defaults(func=run_inventory)

    p = sub.add_parser("extract", help="Extract frozen source-side candidate evidence representations")
    add_common_dataset_args(p)
    p.add_argument("--model-name", type=str, default="dinov2_vitl14_reg")
    p.add_argument("--layers", type=str, default="4-18")
    p.add_argument("--resize-size", type=int, default=448)
    p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--query-batch-size", type=int, default=2)
    p.add_argument("--nn-chunk", type=int, default=256)
    p.add_argument("--sensor-topk", type=int, default=16, help="v22-like suspicious patch budget")
    p.add_argument("--localizers", type=str, default="oracle,sensor")
    p.add_argument("--representations", type=str, default=",".join(REPRESENTATION_NAMES))
    p.add_argument("--pca-rank", type=int, default=64)
    p.add_argument("--pca-max-patches", type=int, default=4096)
    p.add_argument("--whiten-floor-ratio", type=float, default=0.10)
    p.add_argument("--max-per-type-per-category", type=int, default=32,
                   help="Deterministic source-side cap for diagnostic runtime; 0 means all defects")
    p.add_argument("--progress-every", type=int, default=16)
    p.set_defaults(func=run_extract)

    p = sub.add_parser("evaluate", help="Category-held-out source-side nearest-centroid and linear-probe audit")
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--features", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="outputs/factorizability_audit")
    p.add_argument("--min-train-per-label", type=int, default=8)
    p.add_argument("--min-test-per-label", type=int, default=2)
    p.add_argument("--permutations", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=run_evaluate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
