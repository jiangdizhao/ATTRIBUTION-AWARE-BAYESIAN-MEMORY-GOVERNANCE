#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch-aggregation smoke audit for ABMG factorizability.

This experiment follows the source-side factorizability smoke result where
oracle-localised raw DINO patches were strongly defect-type decodable but a
uniform mean over the detector's top-16 patches was substantially weaker.

The question here is deliberately narrow:

    Is the oracle -> sensor gap mainly caused by diluting a small number of
    informative anomalous patches with less relevant top-K patches?

The deployed frozen detector is not changed.  For each source defect image we
compute its frozen DINO patch descriptors and the usual nearest-normal patch
anomaly scores once, then derive multiple raw-patch evidence vectors:

  * oracle/raw_patch                       (offline mask ceiling)
  * sensor/raw_patch/k{1,2,4,8,16}/uniform
  * sensor/raw_patch/k{1,2,4,8,16}/score_softmax

For score_softmax, selected top-K patch scores a_p are pooled with

    w_p = softmax(temperature * a_p)
    z   = sum_p w_p h_p

The K=1 uniform and score_softmax vectors must be identical and act as an
engineering sanity check.

HARD DATA-ACCESS BOUNDARY
-------------------------
Only source-category Real-IAD JSON files are read.  Outer-fold target categories
are not opened.  Defect labels and masks are stored as OFFLINE-ONLY evaluator
metadata; they do not alter the detector or patch scores.

The output artifact intentionally uses schema ``abmg.factorizability.features.v1``
so it can be evaluated by ``abmg_factorizability_evaluate_v2.py`` with the same
Core-4 category-held-out linear/centroid probes and shuffle controls.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from abmg_stage0_foundation import (
    PaperAlignedFrozenSensor,
    Stage0Config,
    Stage0Record,
    config_fingerprint,
    load_realiad_records,
    parse_layers,
    select_supports,
    set_seed,
)
from abmg_factorizability_audit import (
    DEFAULT_FOLDS,
    deterministic_stratified_defect_sample,
    load_fold_protocol,
    mask_to_patch_mask,
    nearest_cosine,
    source_defect_records,
)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def parse_k_values(spec: str) -> Tuple[int, ...]:
    vals = sorted({int(x.strip()) for x in str(spec).split(",") if x.strip()})
    if not vals or any(k <= 0 for k in vals):
        raise ValueError(f"--k-values must contain positive integers, got {spec!r}")
    return tuple(vals)


def pool_selected_raw_patches(
    patch_features: torch.Tensor,
    patch_scores: torch.Tensor,
    selected_idx: torch.Tensor,
    *,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    """Pool selected raw DINO patches without changing the detector."""
    idx = selected_idx.to(patch_features.device)
    q = patch_features[idx].float()
    if q.ndim != 2 or q.shape[0] == 0:
        raise ValueError("selected patch tensor must be non-empty [K,D]")
    if mode == "uniform":
        return q.mean(dim=0)
    if mode == "score_softmax":
        s = patch_scores[selected_idx.to(patch_scores.device)].float()
        w = torch.softmax(float(temperature) * s, dim=0)
        return (w[:, None] * q).sum(dim=0)
    raise ValueError(f"Unknown pooling mode={mode!r}")


def _append_rep(
    store: Dict[str, Dict[str, Any]],
    key: str,
    item_idx: int,
    vec: torch.Tensor,
) -> None:
    slot = store.setdefault(key, {"item_indices": [], "vectors": []})
    slot["item_indices"].append(int(item_idx))
    slot["vectors"].append(vec.detach().cpu().to(torch.float16))


def _localisation_row(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    top_idx: torch.Tensor,
    k: int,
) -> Dict[str, Any]:
    if mask is None:
        return {"mask_available": False, "sensor_topk": int(k)}
    y = mask.cpu().numpy().astype(bool)
    pred = np.zeros_like(y, dtype=bool)
    pred[top_idx.detach().cpu().numpy()] = True
    inter = int(np.logical_and(pred, y).sum())
    n_gt = int(y.sum())
    return {
        "mask_available": True,
        "n_positive_patches": n_gt,
        "sensor_topk": int(k),
        "sensor_hit_any": bool(inter > 0),
        "sensor_patch_precision": float(inter / max(int(pred.sum()), 1)),
        "sensor_patch_recall": float(inter / max(n_gt, 1)),
    }


def _aggregate_localisation(rows_by_k: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in sorted(rows_by_k):
        rows = [r for r in rows_by_k[k] if r.get("mask_available")]
        nonempty = [r for r in rows if int(r.get("n_positive_patches", 0)) > 0]
        out[str(k)] = {
            "n_declared_masks": len(rows),
            "n_nonempty_after_sensor_geometry": len(nonempty),
            "hit_at_k_nonempty_geometry": (
                float(np.mean([1.0 if r.get("sensor_hit_any") else 0.0 for r in nonempty]))
                if nonempty
                else float("nan")
            ),
            "mean_patch_recall_at_k_nonempty_geometry": (
                float(np.mean([float(r["sensor_patch_recall"]) for r in nonempty]))
                if nonempty
                else float("nan")
            ),
            "mean_patch_precision_at_k_nonempty_geometry": (
                float(np.mean([float(r["sensor_patch_precision"]) for r in nonempty]))
                if nonempty
                else float("nan")
            ),
        }
    return out


def run(args: argparse.Namespace) -> int:
    protocol = load_fold_protocol(args.fold, args.folds)
    k_values = parse_k_values(args.k_values)
    pooling_modes = tuple(x.strip() for x in args.pooling.split(",") if x.strip())
    allowed_pooling = {"uniform", "score_softmax"}
    unknown = sorted(set(pooling_modes) - allowed_pooling)
    if unknown or not pooling_modes:
        raise ValueError(f"Unsupported --pooling values {unknown}; allowed={sorted(allowed_pooling)}")

    # HARD BOUNDARY: only source-category JSON files are loaded.
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

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(
        {
            "schema": "abmg.patch_aggregation_smoke.config.v1",
            "outer_fold": int(args.fold),
            "target_categories_untouched": list(protocol.target_categories),
            "source_categories": list(protocol.source_categories),
            "source_cv_groups": [list(x) for x in protocol.source_cv_groups],
            "sensor": asdict(cfg),
            "sensor_fingerprint": config_fingerprint(cfg),
            "k_values": list(k_values),
            "pooling_modes": list(pooling_modes),
            "score_softmax_temperature": float(args.score_temperature),
            "max_per_type_per_category": int(args.max_per_type_per_category),
            "seed": int(args.seed),
            "note": "Raw DINO patch aggregation audit only; detector and anomaly scores are unchanged.",
        },
        out_dir / "resolved_config.json",
    )

    items: List[Dict[str, Any]] = [
        {
            "image_id": r.image_id,
            "category": r.category,
            "relative_path": r.relative_path,
            "defect_source_offline_only": r.defect_source,
            "mask_path_offline_only": r.mask_path,
        }
        for r in defects
    ]
    item_index = {r.image_id: i for i, r in enumerate(defects)}
    by_category: Dict[str, List[Stage0Record]] = defaultdict(list)
    for r in defects:
        by_category[r.category].append(r)

    rep_store: Dict[str, Dict[str, Any]] = {}
    loc_rows_by_k: Dict[int, List[Dict[str, Any]]] = {k: [] for k in k_values}
    # Kept for evaluator-v2 compatibility; use largest K as the legacy localisation row.
    legacy_localisation: List[Dict[str, Any]] = []
    support_meta: Dict[str, Any] = {}

    sensor = PaperAlignedFrozenSensor(cfg, device=args.device, use_fp16=not args.no_fp16)
    try:
        categories = sorted(by_category)
        for ci, category in enumerate(categories, 1):
            supports = select_supports(records, category, cfg.shots, cfg.support_seed)
            bank = sensor.build_category_bank(supports)
            support_meta[category] = {
                "support_image_ids": [r.image_id for r in supports],
                "n_fused_support_patches": int(bank.shape[0]),
            }
            cat_recs = by_category[category]
            print(f"[{ci}/{len(categories)}] {category}: support={len(supports)}, defects={len(cat_recs)}")

            for start in range(0, len(cat_recs), int(args.query_batch_size)):
                batch = cat_recs[start : start + int(args.query_batch_size)]
                images = [Image.open(r.image_path).convert("RGB") for r in batch]
                try:
                    q_batch, _ = sensor.encode_pil_batch(images)
                finally:
                    for im in images:
                        im.close()

                for bi, rec in enumerate(batch):
                    qf = q_batch[bi]
                    best_sim, _ = nearest_cosine(qf, bank, chunk=args.nn_chunk)
                    patch_scores = 1.0 - best_sim

                    oracle_mask: Optional[torch.Tensor] = None
                    if rec.mask_path and Path(rec.mask_path).is_file():
                        try:
                            oracle_mask = mask_to_patch_mask(rec.mask_path, cfg)
                        except Exception as exc:
                            print(f"  WARN mask failed for {rec.relative_path}: {exc}")

                    # Offline oracle ceiling: mean only the true defect patches.
                    if oracle_mask is not None:
                        oracle_idx = torch.nonzero(oracle_mask, as_tuple=False).flatten()
                        if oracle_idx.numel() > 0:
                            oracle_vec = qf[oracle_idx.to(qf.device)].float().mean(dim=0)
                            _append_rep(
                                rep_store,
                                "oracle/raw_patch",
                                item_index[rec.image_id],
                                oracle_vec,
                            )

                    order = torch.argsort(patch_scores, descending=True).cpu()
                    for k in k_values:
                        kk = min(int(k), int(order.numel()))
                        idx = order[:kk]
                        loc = _localisation_row(patch_scores, oracle_mask, idx, kk)
                        loc.update(
                            {
                                "item_index": int(item_index[rec.image_id]),
                                "image_id": rec.image_id,
                                "category": rec.category,
                                "defect_source_offline_only": rec.defect_source,
                            }
                        )
                        loc_rows_by_k[int(k)].append(loc)

                        for mode in pooling_modes:
                            vec = pool_selected_raw_patches(
                                qf,
                                patch_scores,
                                idx,
                                mode=mode,
                                temperature=args.score_temperature,
                            )
                            key = f"sensor/raw_patch/k{k}/{mode}"
                            _append_rep(rep_store, key, item_index[rec.image_id], vec)

                    legacy_localisation.append(loc_rows_by_k[max(k_values)][-1])

                done = min(start + len(batch), len(cat_recs))
                if args.progress_every > 0 and (
                    done % int(args.progress_every) == 0 or done == len(cat_recs)
                ):
                    print(f"  {done}/{len(cat_recs)}")

            del bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sensor.cleanup()

    packed: Dict[str, Dict[str, Any]] = {}
    for key, slot in sorted(rep_store.items()):
        vecs = slot["vectors"]
        packed[key] = {
            "item_indices": torch.tensor(slot["item_indices"], dtype=torch.int64),
            "features": (
                torch.stack(vecs, dim=0)
                if vecs
                else torch.empty((0, 0), dtype=torch.float16)
            ),
        }

    artifact = {
        # Intentional compatibility with evaluator v2.
        "schema": "abmg.factorizability.features.v1",
        "experiment": "patch_aggregation_smoke_v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "source_cv_groups": [list(x) for x in protocol.source_cv_groups],
        "sensor_fingerprint": config_fingerprint(cfg),
        "items": items,
        "representations": packed,
        "localisation": legacy_localisation,
        "localisation_by_k": {str(k): v for k, v in loc_rows_by_k.items()},
        "support": support_meta,
        "aggregation": {
            "k_values": list(k_values),
            "pooling_modes": list(pooling_modes),
            "score_softmax_temperature": float(args.score_temperature),
        },
        "offline_only_notice": "defect_source and masks are evaluator-only and never enter sensor scoring",
    }
    artifact_path = out_dir / "patch_aggregation_features.pt"
    torch.save(artifact, artifact_path)

    summary = {
        "schema": "abmg.patch_aggregation_smoke.summary.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "n_sampled_source_defect_images": len(defects),
        "k_values": list(k_values),
        "pooling_modes": list(pooling_modes),
        "score_softmax_temperature": float(args.score_temperature),
        "representations": {
            k: {
                "n": int(v["features"].shape[0]),
                "dim": int(v["features"].shape[1]) if v["features"].ndim == 2 else 0,
            }
            for k, v in packed.items()
        },
        "sensor_localisation_by_k": _aggregate_localisation(loc_rows_by_k),
        "artifact": str(artifact_path),
        "sanity_note": "For k=1, uniform and score_softmax must be numerically identical apart from fp16 serialization.",
    }
    _json_dump(summary, out_dir / "patch_aggregation_summary.json")
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--json-dir", type=str, required=True)
    p.add_argument("--folds", type=str, default=str(DEFAULT_FOLDS))
    p.add_argument("--out-dir", type=str, default="outputs/patch_aggregation_smoke")
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
    p.add_argument("--k-values", type=str, default="1,2,4,8,16")
    p.add_argument("--pooling", type=str, default="uniform,score_softmax")
    p.add_argument(
        "--score-temperature",
        type=float,
        default=20.0,
        help="Softmax temperature applied to cosine-distance anomaly scores.",
    )
    p.add_argument(
        "--max-per-type-per-category",
        type=int,
        default=4,
        help="Deterministic source-side smoke cap; 0 means all source defects.",
    )
    p.add_argument("--progress-every", type=int, default=16)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
