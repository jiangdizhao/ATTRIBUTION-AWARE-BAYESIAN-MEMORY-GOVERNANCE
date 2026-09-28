#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1A: fixed Sensor vs offline Oracle localization diagnostic.

E1A does not learn. It asks two separate questions:

1) Spatial selection:
   Does the frozen deployment-style Sensor select the annotated defect patches?
   Primary Sensor condition is fixed to top-8 + score-softmax.

2) Downstream routing consequence:
   Using matched support image indices, how do Oracle-localized and
   Sensor-selected evidence differ under the existing diag_shrunk factor router?

Important: a factor-routing error is NOT itself called a spatial error.

Only source categories are opened. Outer target categories remain untouched.
Masks and defect-source labels are offline evaluator metadata only; masks never
change Sensor scores or top-8 selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score

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
from abmg_patch_aggregation_smoke import pool_selected_raw_patches
from abmg_prototype_addressability_audit import (
    CORE4,
    category_diverse_support_indices,
)
from abmg_memory_addressability_audit import fit_shared_diag_prior
from abmg_sequential_local_update_audit import init_memory, score_memory, true_margin


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def localization_metrics(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    k: int = 8,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Compute the four E1A spatial quantities.

    first_hit_rank is 1-based. Geometry failure means the transformed mask
    contains zero positive patches.
    """
    scores = scores.detach().float().cpu().reshape(-1)
    kk = min(int(k), int(scores.numel()))
    if kk <= 0:
        raise ValueError("k must be positive and scores must be non-empty")

    order = torch.argsort(scores, descending=True)
    top_idx = order[:kk]

    if mask is None:
        return {
            "mask_available": False,
            "n_positive_patches": None,
            "sensor_topk": kk,
            "sensor_hit": None,
            "sensor_precision": None,
            "sensor_recall": None,
            "first_hit_rank": None,
            "geometry_failure": None,
        }, top_idx

    y = mask.detach().cpu().bool().reshape(-1)
    if y.numel() != scores.numel():
        raise ValueError(
            f"mask has {y.numel()} patches but scores have {scores.numel()}"
        )

    n_pos = int(y.sum().item())
    if n_pos == 0:
        return {
            "mask_available": True,
            "n_positive_patches": 0,
            "sensor_topk": kk,
            "sensor_hit": None,
            "sensor_precision": None,
            "sensor_recall": None,
            "first_hit_rank": None,
            "geometry_failure": True,
        }, top_idx

    inter = int(y[top_idx].sum().item())
    positive_ranks = torch.nonzero(y[order], as_tuple=False).flatten()
    first_rank = int(positive_ranks[0].item()) + 1

    return {
        "mask_available": True,
        "n_positive_patches": n_pos,
        "sensor_topk": kk,
        "sensor_hit": bool(inter > 0),
        "sensor_precision": float(inter / kk),
        "sensor_recall": float(inter / n_pos),
        "first_hit_rank": first_rank,
        "geometry_failure": False,
    }, top_idx


def summarise_spatial_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    declared = [r for r in rows if r.get("mask_available") is True]
    nonempty = [r for r in declared if r.get("geometry_failure") is False]
    geometry = [r for r in declared if r.get("geometry_failure") is True]

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in nonempty if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    ranks = [
        int(r["first_hit_rank"])
        for r in nonempty
        if r.get("first_hit_rank") is not None
    ]
    hits = [
        bool(r["sensor_hit"])
        for r in nonempty
        if r.get("sensor_hit") is not None
    ]
    return {
        "n_rows": len(rows),
        "n_masks_available": len(declared),
        "n_geometry_failures": len(geometry),
        "geometry_failure_fraction": (
            float(len(geometry) / len(declared)) if declared else float("nan")
        ),
        "n_nonempty_masks": len(nonempty),
        "sensor_hit_at_8": float(np.mean(hits)) if hits else float("nan"),
        "mean_sensor_precision_at_8": mean("sensor_precision"),
        "mean_sensor_recall_at_8": mean("sensor_recall"),
        "median_first_hit_rank": (
            float(np.median(ranks)) if ranks else float("nan")
        ),
    }


def _routing_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=list(CORE4),
                average="macro",
                zero_division=0,
            )
        ),
    }


def _routing_cv(
    oracle_x: np.ndarray,
    sensor_x: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    cv_groups: Sequence[Sequence[str]],
    *,
    shots: int,
    seed: int,
    kappa0: float,
) -> Dict[str, Any]:
    """Paired downstream routing diagnostic.

    The same support image indices are used in Oracle and Sensor conditions.
    Each condition has its own representation-space prior/memory values because
    Oracle and Sensor evidence vectors differ.
    """
    wanted = set(CORE4)
    all_categories = set(categories.tolist())
    folds: List[Dict[str, Any]] = []
    item_rows: List[Dict[str, Any]] = []

    for fi, group in enumerate(cv_groups):
        test_cats = set(str(x) for x in group)
        train_cats = all_categories - test_cats

        train_all = np.asarray([c in train_cats for c in categories], dtype=bool)
        train_core = np.asarray(
            [(c in train_cats) and (y in wanted) for c, y in zip(categories, labels)],
            dtype=bool,
        )
        test_core = np.asarray(
            [(c in test_cats) and (y in wanted) for c, y in zip(categories, labels)],
            dtype=bool,
        )

        if not np.any(train_core) or not np.any(test_core):
            folds.append({"source_cv_fold": fi, "status": "empty_core4"})
            continue

        y_train = labels[train_core]
        c_train = categories[train_core]
        y_test = labels[test_core]

        if set(y_train.tolist()) != wanted or set(y_test.tolist()) != wanted:
            folds.append(
                {
                    "source_cv_fold": fi,
                    "status": "incomplete_core4",
                    "train_labels": sorted(set(y_train.tolist())),
                    "test_labels": sorted(set(y_test.tolist())),
                }
            )
            continue

        support_seed = int(seed) + 100_000 * fi + 17
        try:
            support = category_diverse_support_indices(
                y_train,
                c_train,
                CORE4,
                int(shots),
                support_seed,
            )
        except ValueError as exc:
            folds.append(
                {
                    "source_cv_fold": fi,
                    "status": "insufficient_support",
                    "error": str(exc),
                }
            )
            continue

        xo_train, xs_train = oracle_x[train_core], sensor_x[train_core]
        xo_test, xs_test = oracle_x[test_core], sensor_x[test_core]

        prior_o = fit_shared_diag_prior(oracle_x[train_all])
        prior_s = fit_shared_diag_prior(sensor_x[train_all])
        mem_o = init_memory(xo_train, support)
        mem_s = init_memory(xs_train, support)

        pred_o, score_o = score_memory(
            xo_test, mem_o, prior_o, float(kappa0), "diag_shrunk"
        )
        pred_s, score_s = score_memory(
            xs_test, mem_s, prior_s, float(kappa0), "diag_shrunk"
        )

        margin_o = true_margin(y_test, score_o)
        margin_s = true_margin(y_test, score_s)
        correct_o = pred_o == y_test
        correct_s = pred_s == y_test

        transitions = Counter()
        for oc, sc in zip(correct_o.tolist(), correct_s.tolist()):
            if oc and sc:
                transitions["oracle_correct_sensor_correct"] += 1
            elif oc and not sc:
                transitions["oracle_correct_sensor_wrong"] += 1
            elif (not oc) and sc:
                transitions["oracle_wrong_sensor_correct"] += 1
            else:
                transitions["oracle_wrong_sensor_wrong"] += 1

        test_original = original_indices[test_core]
        for j in range(len(y_test)):
            item_rows.append(
                {
                    "source_cv_fold": fi,
                    "item_index": int(test_original[j]),
                    "defect_source": str(y_test[j]),
                    "oracle_pred": str(pred_o[j]),
                    "sensor_pred": str(pred_s[j]),
                    "oracle_correct": bool(correct_o[j]),
                    "sensor_correct": bool(correct_s[j]),
                    "oracle_margin": float(margin_o[j]),
                    "sensor_margin": float(margin_s[j]),
                }
            )

        folds.append(
            {
                "source_cv_fold": fi,
                "status": "ok",
                "test_categories": sorted(test_cats),
                "n_train_core4": int(train_core.sum()),
                "n_test_core4": int(test_core.sum()),
                "support_categories": {
                    y: sorted(set(c_train[idx].tolist()))
                    for y, idx in support.items()
                },
                "oracle": _routing_metrics(y_test, pred_o),
                "sensor": _routing_metrics(y_test, pred_s),
                "transition_counts": dict(sorted(transitions.items())),
            }
        )

    ok = [f for f in folds if f.get("status") == "ok"]
    aggregate: Dict[str, Any] = {"n_valid_source_cv_folds": len(ok)}

    for condition in ("oracle", "sensor"):
        for metric in ("balanced_accuracy", "macro_f1"):
            vals = [float(f[condition][metric]) for f in ok]
            aggregate[f"{condition}_{metric}_mean"] = (
                float(np.mean(vals)) if vals else float("nan")
            )

    transitions = Counter()
    for f in ok:
        transitions.update(f["transition_counts"])
    n = sum(transitions.values())
    aggregate["transition_counts"] = dict(sorted(transitions.items()))
    aggregate["transition_fractions"] = {
        k: float(v / n) if n else float("nan")
        for k, v in sorted(transitions.items())
    }
    return {"folds": folds, "aggregate": aggregate, "items": item_rows}


def run(args: argparse.Namespace) -> int:
    if int(args.sensor_k) != 8:
        raise ValueError(
            "Primary E1A is fixed at Sensor top-8. "
            "Use another K only in an explicitly separate sensitivity analysis."
        )

    protocol = load_fold_protocol(args.fold, args.folds)
    records = load_realiad_records(
        args.root,
        args.json_dir,
        set(protocol.source_categories),
    )
    defects = deterministic_stratified_defect_sample(
        source_defect_records(records, protocol.source_categories),
        max_per_type_per_category=int(args.max_per_type_per_category),
        seed=int(args.seed),
    )
    if not defects:
        raise RuntimeError("No source defect test records found")

    cfg = Stage0Config(
        model_name=args.model_name,
        layers=parse_layers(args.layers),
        resize_size=int(args.resize_size),
        crop_size=int(args.crop_size),
        patch_size=int(args.patch_size),
        layer_fusion="mean",
        support_augmentation="paper_geometric",
        query_view="identity",
        top_fraction=0.01,
        knn=1,
        shots=int(args.shots),
        support_seed=int(args.support_seed),
        cache_dtype="float16" if not args.no_fp16 else "float32",
    )
    set_seed(int(args.seed))

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    by_category: Dict[str, List[Stage0Record]] = defaultdict(list)
    for i, rec in enumerate(defects):
        items.append(
            {
                "item_index": i,
                "image_id": rec.image_id,
                "category": rec.category,
                "relative_path": rec.relative_path,
                "defect_source_offline_only": rec.defect_source,
                "mask_path_offline_only": rec.mask_path,
            }
        )
        by_category[rec.category].append(rec)

    item_index = {rec.image_id: i for i, rec in enumerate(defects)}
    spatial_rows: List[Dict[str, Any]] = []
    paired_indices: List[int] = []
    paired_oracle: List[torch.Tensor] = []
    paired_sensor: List[torch.Tensor] = []
    paired_categories: List[str] = []
    paired_labels: List[str] = []
    support_meta: Dict[str, Any] = {}

    sensor = PaperAlignedFrozenSensor(
        cfg,
        device=args.device,
        use_fp16=not args.no_fp16,
    )

    try:
        categories = sorted(by_category)
        for ci, category in enumerate(categories, 1):
            supports = select_supports(
                records,
                category,
                cfg.shots,
                cfg.support_seed,
            )
            bank = sensor.build_category_bank(supports)
            support_meta[category] = {
                "support_image_ids": [r.image_id for r in supports],
                "n_fused_support_patches": int(bank.shape[0]),
            }

            cat_recs = by_category[category]
            print(
                f"[{ci}/{len(categories)}] {category}: "
                f"support={len(supports)}, defects={len(cat_recs)}"
            )

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
                    best_sim, _ = nearest_cosine(
                        qf,
                        bank,
                        chunk=int(args.nn_chunk),
                    )
                    patch_scores = 1.0 - best_sim

                    oracle_mask: Optional[torch.Tensor] = None
                    if rec.mask_path and Path(rec.mask_path).is_file():
                        try:
                            oracle_mask = mask_to_patch_mask(rec.mask_path, cfg)
                        except Exception as exc:
                            print(
                                f"  WARN mask failed for {rec.relative_path}: {exc}"
                            )

                    loc, top_idx = localization_metrics(
                        patch_scores,
                        oracle_mask,
                        k=int(args.sensor_k),
                    )
                    loc.update(
                        {
                            "item_index": int(item_index[rec.image_id]),
                            "image_id": rec.image_id,
                            "category": rec.category,
                            "defect_source_offline_only": rec.defect_source,
                        }
                    )
                    spatial_rows.append(loc)

                    sensor_vec = pool_selected_raw_patches(
                        qf,
                        patch_scores,
                        top_idx,
                        mode="score_softmax",
                        temperature=float(args.score_temperature),
                    )

                    if (
                        oracle_mask is not None
                        and loc.get("geometry_failure") is False
                    ):
                        oracle_idx = torch.nonzero(
                            oracle_mask,
                            as_tuple=False,
                        ).flatten()
                        oracle_vec = qf[
                            oracle_idx.to(qf.device)
                        ].float().mean(dim=0)

                        paired_indices.append(int(item_index[rec.image_id]))
                        paired_oracle.append(
                            oracle_vec.detach().cpu().to(torch.float16)
                        )
                        paired_sensor.append(
                            sensor_vec.detach().cpu().to(torch.float16)
                        )
                        paired_categories.append(str(rec.category))
                        paired_labels.append(str(rec.defect_source))

                done = min(start + len(batch), len(cat_recs))
                if args.progress_every > 0 and (
                    done % int(args.progress_every) == 0
                    or done == len(cat_recs)
                ):
                    print(f"  {done}/{len(cat_recs)}")

            del bank
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sensor.cleanup()

    if not paired_oracle:
        raise RuntimeError(
            "No paired non-empty-mask Oracle/Sensor evidence was produced"
        )

    oracle_tensor = torch.stack(paired_oracle, dim=0)
    sensor_tensor = torch.stack(paired_sensor, dim=0)
    oracle_x = oracle_tensor.float().numpy().astype(np.float64)
    sensor_x = sensor_tensor.float().numpy().astype(np.float64)
    route_categories = np.asarray(paired_categories, dtype=object)
    route_labels = np.asarray(paired_labels, dtype=object)
    original_indices = np.asarray(paired_indices, dtype=np.int64)

    routing = _routing_cv(
        oracle_x,
        sensor_x,
        route_categories,
        route_labels,
        original_indices,
        protocol.source_cv_groups,
        shots=int(args.routing_shots),
        seed=int(args.seed),
        kappa0=float(args.kappa0),
    )

    spatial_by_item = {int(r["item_index"]): r for r in spatial_rows}
    critical = [
        r
        for r in routing["items"]
        if r["oracle_correct"] and not r["sensor_correct"]
    ]
    critical_hits = [
        spatial_by_item[int(r["item_index"])]["sensor_hit"]
        for r in critical
        if int(r["item_index"]) in spatial_by_item
        and spatial_by_item[int(r["item_index"])].get("sensor_hit") is not None
    ]

    summary = {
        "schema": "abmg.e1a.localization_gap.summary.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "n_source_defect_images": len(defects),
        "protocol": {
            "sensor_topk": 8,
            "sensor_pooling": "score_softmax",
            "score_softmax_temperature": float(args.score_temperature),
            "oracle": "mean of all non-empty ground-truth-mask patches; offline only",
            "routing_model": "diag_shrunk",
            "routing_shots_per_core4_factor": int(args.routing_shots),
            "kappa0": float(args.kappa0),
            "routing_interpretation": (
                "Downstream diagnostic only; routing error is not itself "
                "classified as spatial error."
            ),
        },
        "spatial": summarise_spatial_rows(spatial_rows),
        "routing": routing["aggregate"],
        "oracle_correct_sensor_wrong": {
            "n": len(critical),
            "sensor_miss_fraction": (
                float(np.mean([not bool(x) for x in critical_hits]))
                if critical_hits
                else float("nan")
            ),
            "sensor_hit_fraction": (
                float(np.mean([bool(x) for x in critical_hits]))
                if critical_hits
                else float("nan")
            ),
        },
    }

    artifact = {
        "schema": "abmg.e1a.localization_gap.artifact.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "source_cv_groups": [list(x) for x in protocol.source_cv_groups],
        "sensor_config": asdict(cfg),
        "sensor_fingerprint": config_fingerprint(cfg),
        "items": items,
        "spatial_rows": spatial_rows,
        "paired_original_indices": torch.tensor(
            paired_indices,
            dtype=torch.int64,
        ),
        "oracle_features": oracle_tensor,
        "sensor_features": sensor_tensor,
        "paired_categories": paired_categories,
        "paired_labels_offline_only": paired_labels,
        "routing": routing,
        "support": support_meta,
        "offline_only_notice": (
            "Masks and defect-source labels are evaluator-only. "
            "They never alter Sensor scoring or top-8 selection."
        ),
    }

    torch.save(artifact, out_dir / "e1a_localization_gap.pt")
    _json_dump(
        summary,
        out_dir / "e1a_localization_gap_summary.json",
    )
    _json_dump(
        spatial_rows,
        out_dir / "e1a_spatial_rows.json",
    )
    _json_dump(
        routing,
        out_dir / "e1a_routing_rows.json",
    )
    _json_dump(
        {
            "schema": "abmg.e1a.localization_gap.config.v1",
            "outer_fold": int(args.fold),
            "sensor": asdict(cfg),
            "sensor_fingerprint": config_fingerprint(cfg),
            "sensor_k": int(args.sensor_k),
            "score_temperature": float(args.score_temperature),
            "routing_shots": int(args.routing_shots),
            "kappa0": float(args.kappa0),
            "max_per_type_per_category": int(
                args.max_per_type_per_category
            ),
            "seed": int(args.seed),
        },
        out_dir / "resolved_config.json",
    )

    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"Wrote E1A outputs to: {out_dir}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--json-dir", type=str, required=True)
    p.add_argument("--folds", type=str, default=str(DEFAULT_FOLDS))
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/e1a_localization_gap",
    )

    p.add_argument("--model-name", type=str, default="dinov2_vitl14_reg")
    p.add_argument("--layers", type=str, default="4-18")
    p.add_argument("--resize-size", type=int, default=448)
    p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--query-batch-size", type=int, default=2)
    p.add_argument("--nn-chunk", type=int, default=256)

    p.add_argument(
        "--sensor-k",
        type=int,
        default=8,
        help="Primary E1A protocol is fixed at 8.",
    )
    p.add_argument(
        "--score-temperature",
        type=float,
        default=20.0,
    )
    p.add_argument(
        "--routing-shots",
        type=int,
        default=8,
        help="Verified supports per Core-4 factor for diag_shrunk routing.",
    )
    p.add_argument("--kappa0", type=float, default=1.0)
    p.add_argument(
        "--max-per-type-per-category",
        type=int,
        default=0,
        help="0 = all source defects; use a small value only for smoke runs.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=16)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
