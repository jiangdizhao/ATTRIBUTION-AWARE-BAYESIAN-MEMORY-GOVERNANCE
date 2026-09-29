#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1B finalization: robustness, persistence, and normal collateral.

This script runs one bounded follow-up after the primary E1B experiment.

It fixes the already-tested mechanism to:
  - bidirectional mask-guided correction,
  - 0.5% mask budget,
  - Sensor top-8,
  - 8 positive + 8 negative prototypes per category,
  - the same frozen diag_shrunk downstream router.

It then asks only three remaining questions:

1) Robustness:
   Does the gain survive five predeclared random mask schedules?

2) Persistence:
   On future *unmasked* defect images, does the gain remain after the most
   recent mask feedback rather than disappearing immediately?

3) Normal collateral:
   After learning from sparse defect masks, does the correction memory push
   held-out normal images above a fixed baseline-normal threshold more often?

No v22 controller is used. No factor-memory update is allowed. Target categories
remain sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
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
    load_fold_protocol,
    mask_to_patch_mask,
    nearest_cosine,
    source_defect_records,
)
from abmg_patch_aggregation_smoke import pool_selected_raw_patches
from abmg_e1b_rare_mask_utility import (
    CategoryCorrectionMemory,
    RoutingAccumulator,
    SpatialAccumulator,
    _prepare_stream,
    _route_one,
    _spatial_row,
    build_fixed_routers,
    corrected_patch_scores,
    fixed_nested_mask_schedules,
)

PRIMARY_BUDGET = 0.005
PRIMARY_SENSOR_K = 8
PRIMARY_SCORE_TEMPERATURE = 20.0
PRIMARY_PROTOTYPES_PER_SIGN = 8
PRIMARY_ROUTING_SHOTS = 8
PRIMARY_KAPPA0 = 1.0
DEFAULT_MASK_SEEDS = (271828, 314159, 161803, 141421, 173205)
PERSISTENCE_BUCKETS = (
    ("1-10", 1, 10),
    ("11-25", 11, 25),
    ("26-50", 26, 50),
    ("51+", 51, None),
)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def parse_int_list(spec: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if not vals:
        raise ValueError("expected at least one integer")
    if len(set(vals)) != len(vals):
        raise ValueError("mask schedule seeds must be unique")
    return vals


def persistence_bucket(distance: int) -> str:
    d = int(distance)
    if d <= 0:
        raise ValueError("distance must be positive")
    for name, lo, hi in PERSISTENCE_BUCKETS:
        if d >= lo and (hi is None or d <= hi):
            return name
    raise RuntimeError("unreachable persistence bucket")


def _sha_key(seed: int, image_id: str) -> str:
    return hashlib.sha1(f"{int(seed)}|{image_id}".encode("utf-8")).hexdigest()


def deterministic_normal_split(
    records: Sequence[Stage0Record],
    seed: int,
) -> Tuple[List[Stage0Record], List[Stage0Record]]:
    """Deterministically split test normals 50/50 into calibration/sentinel."""
    xs = sorted(records, key=lambda r: _sha_key(seed, str(r.image_id)))
    if len(xs) < 4:
        return [], []
    n_cal = len(xs) // 2
    return xs[:n_cal], xs[n_cal:]


def topk_mean(scores: torch.Tensor, k: int = PRIMARY_SENSOR_K) -> float:
    s = scores.detach().float().reshape(-1)
    kk = min(max(1, int(k)), int(s.numel()))
    return float(torch.topk(s, k=kk, largest=True).values.mean().item())


class PersistenceAccumulator:
    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, float]] = {
            name: {
                "n_spatial": 0.0,
                "precision_delta_sum": 0.0,
                "recall_delta_sum": 0.0,
                "hit_delta_sum": 0.0,
                "n_routing": 0.0,
                "routing_correct_delta_sum": 0.0,
            }
            for name, _, _ in PERSISTENCE_BUCKETS
        }

    def add_spatial(
        self,
        distance: int,
        frozen_row: Mapping[str, Any],
        corrected_row: Mapping[str, Any],
    ) -> None:
        if (
            frozen_row.get("geometry_failure") is not False
            or corrected_row.get("geometry_failure") is not False
        ):
            return
        b = self.rows[persistence_bucket(distance)]
        b["n_spatial"] += 1.0
        b["precision_delta_sum"] += (
            float(corrected_row["precision"]) - float(frozen_row["precision"])
        )
        b["recall_delta_sum"] += (
            float(corrected_row["recall"]) - float(frozen_row["recall"])
        )
        b["hit_delta_sum"] += (
            float(bool(corrected_row["hit"])) - float(bool(frozen_row["hit"]))
        )

    def add_routing(
        self,
        distance: int,
        frozen_correct: bool,
        corrected_correct: bool,
    ) -> None:
        b = self.rows[persistence_bucket(distance)]
        b["n_routing"] += 1.0
        b["routing_correct_delta_sum"] += (
            float(bool(corrected_correct)) - float(bool(frozen_correct))
        )

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, _, _ in PERSISTENCE_BUCKETS:
            b = self.rows[name]
            ns = int(b["n_spatial"])
            nr = int(b["n_routing"])
            out[name] = {
                "n_future_unmasked_spatial": ns,
                "mean_precision_delta_vs_frozen": (
                    float(b["precision_delta_sum"] / ns) if ns else float("nan")
                ),
                "mean_recall_delta_vs_frozen": (
                    float(b["recall_delta_sum"] / ns) if ns else float("nan")
                ),
                "mean_hit_delta_vs_frozen": (
                    float(b["hit_delta_sum"] / ns) if ns else float("nan")
                ),
                "n_future_unmasked_routing": nr,
                "routing_accuracy_delta_vs_frozen": (
                    float(b["routing_correct_delta_sum"] / nr)
                    if nr
                    else float("nan")
                ),
            }
        return out


class NormalCollateralAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.baseline_fp = 0
        self.corrected_fp = 0
        self.score_shift_sum = 0.0
        self.categories = 0

    def add_category(
        self,
        threshold: float,
        baseline_scores: Sequence[float],
        corrected_scores: Sequence[float],
    ) -> None:
        if len(baseline_scores) != len(corrected_scores):
            raise ValueError("normal score arrays differ in length")
        if not baseline_scores:
            return
        self.categories += 1
        for b, c in zip(baseline_scores, corrected_scores):
            self.n += 1
            self.baseline_fp += int(float(b) > float(threshold))
            self.corrected_fp += int(float(c) > float(threshold))
            self.score_shift_sum += float(c) - float(b)

    def summary(self) -> Dict[str, Any]:
        if self.n == 0:
            return {
                "n_sentinel_normals": 0,
                "n_categories": 0,
                "baseline_fpr": float("nan"),
                "corrected_fpr": float("nan"),
                "fpr_delta": float("nan"),
                "mean_top8_score_shift": float("nan"),
            }
        b = float(self.baseline_fp / self.n)
        c = float(self.corrected_fp / self.n)
        return {
            "n_sentinel_normals": int(self.n),
            "n_categories": int(self.categories),
            "baseline_fpr": b,
            "corrected_fpr": c,
            "fpr_delta": float(c - b),
            "mean_top8_score_shift": float(self.score_shift_sum / self.n),
        }


def _mean_std(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(x) for x in vals if np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def run(args: argparse.Namespace) -> int:
    seeds = parse_int_list(args.mask_schedule_seeds)
    if len(seeds) < 3:
        raise ValueError("use at least 3 mask schedule seeds for robustness")
    if float(args.mask_budget) != PRIMARY_BUDGET:
        raise ValueError("E1B finalization is fixed at 0.5% mask budget")
    if int(args.sensor_k) != PRIMARY_SENSOR_K:
        raise ValueError("E1B finalization is fixed at top-8")
    if int(args.prototypes_per_sign) != PRIMARY_PROTOTYPES_PER_SIGN:
        raise ValueError("E1B finalization is fixed at 8 prototypes per sign")
    if int(args.routing_shots) != PRIMARY_ROUTING_SHOTS:
        raise ValueError("E1B finalization is fixed at 8 routing supports")
    if float(args.kappa0) != PRIMARY_KAPPA0:
        raise ValueError("E1B finalization is fixed at kappa0=1")

    protocol = load_fold_protocol(args.fold, args.folds)
    records = load_realiad_records(
        args.root,
        args.json_dir,
        set(protocol.source_categories),
    )
    defects = source_defect_records(records, protocol.source_categories)
    stream = _prepare_stream(defects, int(args.seed))
    if int(args.max_items) > 0:
        stream = stream[: int(args.max_items)]
    if not stream:
        raise RuntimeError("empty defect stream")
    n_stream = len(stream)

    schedules: Dict[int, set[int]] = {}
    for seed in seeds:
        schedules[int(seed)] = fixed_nested_mask_schedules(
            n_stream,
            (float(args.mask_budget),),
            int(seed),
        )[float(args.mask_budget)]

    routers, router_meta = build_fixed_routers(
        args.e1a_artifact,
        int(args.fold),
        seed=int(args.seed),
        shots=int(args.routing_shots),
        kappa0=float(args.kappa0),
    )

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

    frozen_spatial = SpatialAccumulator()
    frozen_routing = RoutingAccumulator()
    seed_spatial = {s: SpatialAccumulator() for s in seeds}
    seed_routing = {s: RoutingAccumulator() for s in seeds}
    persistence = {s: PersistenceAccumulator() for s in seeds}
    normal_collateral = {s: NormalCollateralAccumulator() for s in seeds}
    mask_usage = {
        s: {
            "scheduled_events": 0,
            "usable_events": 0,
            "positive_patch_observations": 0,
            "negative_patch_observations": 0,
        }
        for s in seeds
    }
    mask_events: List[Dict[str, Any]] = []

    by_category: Dict[str, List[Tuple[int, Stage0Record]]] = defaultdict(list)
    for pos, rec in enumerate(stream):
        by_category[str(rec.category)].append((int(pos), rec))

    sensor = PaperAlignedFrozenSensor(
        cfg,
        device=args.device,
        use_fp16=not args.no_fp16,
    )

    try:
        categories = sorted(by_category)
        for ci, category in enumerate(categories, 1):
            cat_rows = by_category[category]
            supports = select_supports(
                records,
                category,
                cfg.shots,
                cfg.support_seed,
            )
            normal_bank = sensor.build_category_bank(supports)

            memories = {
                s: CategoryCorrectionMemory.create(
                    int(args.prototypes_per_sign),
                    normal_bank.device,
                )
                for s in seeds
            }
            last_usable_mask_local: Dict[int, Optional[int]] = {
                s: None for s in seeds
            }

            print(
                f"[{ci}/{len(categories)}] {category}: "
                f"defects={len(cat_rows)}"
            )

            # -------------------------------------------------------------
            # Defect stream: score first, reveal feedback second.
            # -------------------------------------------------------------
            for start in range(0, len(cat_rows), int(args.query_batch_size)):
                block = cat_rows[start : start + int(args.query_batch_size)]
                batch_recs = [x[1] for x in block]
                images = [Image.open(r.image_path).convert("RGB") for r in batch_recs]
                try:
                    q_batch, _ = sensor.encode_pil_batch(images)
                finally:
                    for im in images:
                        im.close()

                for bi, (global_pos, rec) in enumerate(block):
                    local_idx = start + bi
                    qf = q_batch[bi]
                    best_sim, _ = nearest_cosine(
                        qf,
                        normal_bank,
                        chunk=int(args.nn_chunk),
                    )
                    s0 = 1.0 - best_sim

                    mask: Optional[torch.Tensor] = None
                    if rec.mask_path and Path(rec.mask_path).is_file():
                        try:
                            mask = mask_to_patch_mask(rec.mask_path, cfg)
                        except Exception as exc:
                            print(f"  WARN mask failed for {rec.relative_path}: {exc}")

                    base_row, base_top = _spatial_row(
                        s0,
                        mask,
                        int(args.sensor_k),
                    )
                    frozen_spatial.add(base_row)
                    base_vec = pool_selected_raw_patches(
                        qf,
                        s0,
                        base_top,
                        mode="score_softmax",
                        temperature=float(args.score_temperature),
                    )

                    true_source = str(rec.defect_source)
                    router = routers.get(str(rec.category))
                    route_eligible = (
                        router is not None
                        and true_source in set(("AK", "HS", "QS", "ZW"))
                        and base_row.get("geometry_failure") is False
                    )
                    base_pred: Optional[str] = None
                    if route_eligible:
                        base_pred = _route_one(
                            base_vec,
                            router,
                            float(args.kappa0),
                        )
                        frozen_routing.add(true_source, base_pred)

                    corrected_rows: Dict[int, Dict[str, Any]] = {}
                    corrected_preds: Dict[int, Optional[str]] = {}

                    for s in seeds:
                        corr_scores = corrected_patch_scores(
                            s0,
                            qf,
                            memories[s],
                            "bidirectional",
                        )
                        corr_row, corr_top = _spatial_row(
                            corr_scores,
                            mask,
                            int(args.sensor_k),
                        )
                        seed_spatial[s].add(corr_row)
                        corrected_rows[s] = corr_row

                        corr_vec = pool_selected_raw_patches(
                            qf,
                            corr_scores,
                            corr_top,
                            mode="score_softmax",
                            temperature=float(args.score_temperature),
                        )
                        pred: Optional[str] = None
                        if route_eligible:
                            pred = _route_one(
                                corr_vec,
                                router,
                                float(args.kappa0),
                            )
                            seed_routing[s].add(true_source, pred)
                        corrected_preds[s] = pred

                        # Persistence is evaluated only on future UNMASKED items.
                        current_is_mask_event = int(global_pos) in schedules[s]
                        last_idx = last_usable_mask_local[s]
                        if (not current_is_mask_event) and last_idx is not None:
                            dist = int(local_idx - last_idx)
                            if dist > 0:
                                persistence[s].add_spatial(
                                    dist,
                                    base_row,
                                    corr_row,
                                )
                                if (
                                    route_eligible
                                    and base_pred is not None
                                    and pred is not None
                                ):
                                    persistence[s].add_routing(
                                        dist,
                                        base_pred == true_source,
                                        pred == true_source,
                                    )

                    # Reveal full GT mask only after every branch predicts.
                    for s in seeds:
                        if int(global_pos) not in schedules[s]:
                            continue
                        mask_usage[s]["scheduled_events"] += 1
                        usable = (
                            mask is not None
                            and base_row.get("geometry_failure") is False
                        )
                        n_pos = 0
                        n_neg = 0
                        if usable:
                            y = mask.to(qf.device).bool().reshape(-1)
                            pos_idx = torch.nonzero(y, as_tuple=False).flatten()
                            top_dev = base_top.to(qf.device)
                            neg_idx = top_dev[~y[top_dev]]
                            pos_vec = qf[pos_idx]
                            neg_vec = qf[neg_idx]
                            n_pos = int(pos_vec.shape[0])
                            n_neg = int(neg_vec.shape[0])

                            memories[s].negative.update(neg_vec)
                            memories[s].positive.update(pos_vec)
                            last_usable_mask_local[s] = int(local_idx)

                            mask_usage[s]["usable_events"] += 1
                            mask_usage[s]["positive_patch_observations"] += n_pos
                            mask_usage[s]["negative_patch_observations"] += n_neg

                        mask_events.append(
                            {
                                "mask_schedule_seed": int(s),
                                "stream_position": int(global_pos),
                                "category_local_position": int(local_idx),
                                "image_id": rec.image_id,
                                "category": category,
                                "usable_nonempty_mask": bool(usable),
                                "n_positive_mask_patches": int(n_pos),
                                "n_negative_frozen_top8_patches": int(n_neg),
                            }
                        )

                done = min(start + len(block), len(cat_rows))
                if args.progress_every > 0 and (
                    done % int(args.progress_every) == 0
                    or done == len(cat_rows)
                ):
                    print(f"  defects {done}/{len(cat_rows)}")

            # -------------------------------------------------------------
            # Normal collateral: frozen threshold from calibration normals,
            # evaluated on separate normal sentinel normals.
            # -------------------------------------------------------------
            normals = [
                r
                for r in records
                if r.category == category
                and r.split == "test"
                and r.is_good
            ]
            cal_normals, sentinel_normals = deterministic_normal_split(
                normals,
                int(args.normal_split_seed),
            )
            if cal_normals and sentinel_normals:
                baseline_cal_scores: List[float] = []
                baseline_sentinel_scores: List[float] = []
                corrected_sentinel_scores: Dict[int, List[float]] = {
                    s: [] for s in seeds
                }

                all_normals = [
                    ("cal", r) for r in cal_normals
                ] + [
                    ("sentinel", r) for r in sentinel_normals
                ]

                for start in range(
                    0,
                    len(all_normals),
                    int(args.query_batch_size),
                ):
                    block = all_normals[
                        start : start + int(args.query_batch_size)
                    ]
                    images = [
                        Image.open(r.image_path).convert("RGB")
                        for _, r in block
                    ]
                    try:
                        q_batch, _ = sensor.encode_pil_batch(images)
                    finally:
                        for im in images:
                            im.close()

                    for bi, (split_name, _rec) in enumerate(block):
                        qf = q_batch[bi]
                        best_sim, _ = nearest_cosine(
                            qf,
                            normal_bank,
                            chunk=int(args.nn_chunk),
                        )
                        s0 = 1.0 - best_sim
                        base_score = topk_mean(s0, int(args.sensor_k))
                        if split_name == "cal":
                            baseline_cal_scores.append(base_score)
                        else:
                            baseline_sentinel_scores.append(base_score)
                            for s in seeds:
                                corr = corrected_patch_scores(
                                    s0,
                                    qf,
                                    memories[s],
                                    "bidirectional",
                                )
                                corrected_sentinel_scores[s].append(
                                    topk_mean(corr, int(args.sensor_k))
                                )

                threshold = float(
                    np.quantile(
                        np.asarray(baseline_cal_scores, dtype=np.float64),
                        float(args.normal_threshold_quantile),
                    )
                )
                for s in seeds:
                    normal_collateral[s].add_category(
                        threshold,
                        baseline_sentinel_scores,
                        corrected_sentinel_scores[s],
                    )

            del normal_bank
            del memories
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        sensor.cleanup()

    base_sp = frozen_spatial.summary()
    base_rt = frozen_routing.summary()

    per_seed: Dict[str, Any] = {}
    deltas = {
        "hit_at_8": [],
        "precision_at_8": [],
        "recall_at_8": [],
        "routing_macro_f1": [],
        "normal_fpr_delta": [],
    }

    for s in seeds:
        sp = seed_spatial[s].summary()
        rt = seed_routing[s].summary()
        collateral = normal_collateral[s].summary()
        d = {
            "hit_at_8": float(sp["hit_at_8"] - base_sp["hit_at_8"]),
            "precision_at_8": float(
                sp["mean_precision_at_8"] - base_sp["mean_precision_at_8"]
            ),
            "recall_at_8": float(
                sp["mean_recall_at_8"] - base_sp["mean_recall_at_8"]
            ),
            "routing_macro_f1": float(
                rt["macro_f1"] - base_rt["macro_f1"]
            ),
        }
        per_seed[str(s)] = {
            "mask_usage": mask_usage[s],
            "spatial": sp,
            "routing": rt,
            "delta_vs_frozen": d,
            "persistence_on_future_unmasked_items": persistence[s].summary(),
            "normal_collateral": collateral,
        }
        deltas["hit_at_8"].append(d["hit_at_8"])
        deltas["precision_at_8"].append(d["precision_at_8"])
        deltas["recall_at_8"].append(d["recall_at_8"])
        deltas["routing_macro_f1"].append(d["routing_macro_f1"])
        deltas["normal_fpr_delta"].append(collateral["fpr_delta"])

    robustness = {
        key: _mean_std(vals)
        for key, vals in deltas.items()
    }
    robustness["all_seeds_positive"] = {
        "precision_at_8": bool(all(x > 0 for x in deltas["precision_at_8"])),
        "recall_at_8": bool(all(x > 0 for x in deltas["recall_at_8"])),
        "routing_macro_f1": bool(all(x > 0 for x in deltas["routing_macro_f1"])),
    }

    summary = {
        "schema": "abmg.e1b.finalization.summary.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "n_source_defect_images_in_stream": int(n_stream),
        "fixed_condition": {
            "branch": "bidirectional",
            "mask_budget_fraction": float(args.mask_budget),
            "mask_budget_percent": float(100.0 * args.mask_budget),
            "mask_schedule_seeds": [int(x) for x in seeds],
            "sensor_topk": int(args.sensor_k),
            "score_temperature": float(args.score_temperature),
            "prototypes_per_sign": int(args.prototypes_per_sign),
            "routing_model": "fixed diag_shrunk",
            "routing_shots": int(args.routing_shots),
            "kappa0": float(args.kappa0),
        },
        "frozen": {
            "spatial": base_sp,
            "routing": base_rt,
        },
        "per_seed": per_seed,
        "robustness_across_mask_schedules": robustness,
        "persistence_definition": (
            "future unmasked defect images are grouped by number of same-category "
            "stream items since the most recent usable mask feedback"
        ),
        "normal_collateral_definition": {
            "normal_data": "source-category test normals only",
            "split": "deterministic 50/50 calibration/sentinel",
            "image_score": "mean of top-8 frozen/corrected patch anomaly scores",
            "threshold": (
                f"per-category baseline calibration-normal "
                f"{100.0 * float(args.normal_threshold_quantile):g}th percentile"
            ),
            "evaluation": (
                "same frozen threshold applied to baseline and corrected sentinel "
                "normal scores; diagnostic FPR shift, not a deployment threshold"
            ),
        },
        "router_reconstruction": router_meta,
    }

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(summary, out_dir / "e1b_finalization_summary.json")
    _json_dump(mask_events, out_dir / "e1b_finalization_mask_events.json")
    _json_dump(
        {
            "schema": "abmg.e1b.finalization.config.v1",
            "outer_fold": int(args.fold),
            "e1a_artifact": str(args.e1a_artifact),
            "sensor": asdict(cfg),
            "sensor_fingerprint": config_fingerprint(cfg),
            "mask_budget": float(args.mask_budget),
            "mask_schedule_seeds": [int(x) for x in seeds],
            "normal_split_seed": int(args.normal_split_seed),
            "normal_threshold_quantile": float(args.normal_threshold_quantile),
            "max_items": int(args.max_items),
        },
        out_dir / "resolved_config.json",
    )

    print(json.dumps(summary["robustness_across_mask_schedules"], indent=2))
    print(f"Wrote E1B finalization outputs to: {out_dir}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--json-dir", type=str, required=True)
    p.add_argument("--e1a-artifact", type=str, required=True)
    p.add_argument("--folds", type=str, default=str(DEFAULT_FOLDS))
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/e1b_finalization",
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

    p.add_argument("--sensor-k", type=int, default=PRIMARY_SENSOR_K)
    p.add_argument(
        "--score-temperature",
        type=float,
        default=PRIMARY_SCORE_TEMPERATURE,
    )
    p.add_argument(
        "--mask-budget",
        type=float,
        default=PRIMARY_BUDGET,
    )
    p.add_argument(
        "--mask-schedule-seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MASK_SEEDS),
    )
    p.add_argument(
        "--prototypes-per-sign",
        type=int,
        default=PRIMARY_PROTOTYPES_PER_SIGN,
    )
    p.add_argument(
        "--routing-shots",
        type=int,
        default=PRIMARY_ROUTING_SHOTS,
    )
    p.add_argument("--kappa0", type=float, default=PRIMARY_KAPPA0)
    p.add_argument("--normal-split-seed", type=int, default=424242)
    p.add_argument("--normal-threshold-quantile", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="0 = all source defects; use a small value only for smoke tests.",
    )
    p.add_argument("--progress-every", type=int, default=64)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
