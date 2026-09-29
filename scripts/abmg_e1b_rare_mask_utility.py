#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1B: sparse human-mask utility for future spatial evidence selection.

Scientific question
-------------------
Given the frozen E1A sensor, can a sparse set of *whole-image ground-truth
defect masks* improve top-8 patch selection on future images without changing
the visual backbone?

This experiment deliberately does NOT use the v22 controller. Mask feedback is
made sparse by a fixed, deterministic random schedule over stream positions.
The randomization chooses which IMAGE receives its dataset-provided mask; it
never randomizes pixels inside the mask.

Branches
--------
1. frozen
   No spatial memory update.

2. suppress
   At a mask-feedback event, store the frozen Sensor top-8 patches that lie
   outside the ground-truth mask as verified nuisance evidence.

3. bidirectional
   Store the same verified nuisance patches plus all positive mask patches as
   verified defect evidence.

For fairness, every learning branch at a given mask budget receives exactly the
same feedback packet. Feedback is revealed only AFTER that image has been
scored, so it can affect future images only.

Spatial memory
--------------
Each category and sign (+ defect / - nuisance) has a tiny online prototype bank.
The bank has K prototypes (default 8). It fills deterministically and then uses
nearest-prototype running-mean updates. No SGD and no backbone update.

For a future patch h, let s0 be its frozen nearest-normal anomaly score,
sim+ its maximum cosine similarity to positive prototypes, and sim- its maximum
cosine similarity to negative prototypes. Similarities are clipped to [0,1].
The correction is scale-matched to the current image:

    sigma = std_p(s0_p)

    suppress:      s = s0 - sigma * sim-
    bidirectional: s = s0 + sigma * (sim+ - sim-)

The primary correction strength is therefore fixed at one within-image score
standard deviation; there is no parameter sweep in E1B.

Downstream check
----------------
The factor router is frozen. It is reconstructed from the E1A artifact using
the same source-CV split, 8-shot supports, kappa0=1, and diag_shrunk memory.
Corrected top-8 evidence is passed through this unchanged router. Thus any
routing change is a downstream consequence of changed spatial evidence, not a
factor-memory update.

Data boundary
-------------
Only source categories from the outer fold are opened. Outer target categories
remain untouched. All available source defect images are used by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
    load_fold_protocol,
    mask_to_patch_mask,
    nearest_cosine,
    source_defect_records,
)
from abmg_patch_aggregation_smoke import pool_selected_raw_patches
from abmg_prototype_addressability_audit import CORE4, category_diverse_support_indices
from abmg_memory_addressability_audit import fit_shared_diag_prior
from abmg_sequential_local_update_audit import init_memory, score_memory


PRIMARY_SENSOR_K = 8
PRIMARY_SCORE_TEMPERATURE = 20.0
PRIMARY_CORRECTION_STRENGTH = 1.0
PRIMARY_ROUTING_SHOTS = 8
PRIMARY_KAPPA0 = 1.0
PRIMARY_PROTOTYPES_PER_SIGN = 8


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def parse_mask_budgets(spec: str) -> Tuple[float, ...]:
    vals = sorted({float(x.strip()) for x in str(spec).split(",") if x.strip()})
    if not vals:
        raise ValueError("mask budgets must not be empty")
    if any((x <= 0.0 or x > 1.0) for x in vals):
        raise ValueError("each mask budget must be in (0,1]")
    return tuple(vals)


def budget_tag(budget: float) -> str:
    return f"{100.0 * float(budget):g}pct".replace(".", "p")


def fixed_nested_mask_schedules(
    n_stream: int,
    budgets: Sequence[float],
    seed: int,
) -> Dict[float, Set[int]]:
    """Choose exact, nested stream positions without using labels/scores/masks."""
    n_stream = int(n_stream)
    if n_stream <= 0:
        raise ValueError("n_stream must be positive")
    budgets = tuple(sorted(float(x) for x in budgets))
    if not budgets:
        raise ValueError("budgets must not be empty")

    rng = np.random.default_rng(int(seed))
    priority = rng.permutation(n_stream)
    out: Dict[float, Set[int]] = {}
    for b in budgets:
        n = int(round(float(b) * n_stream))
        n = min(max(n, 1), n_stream)
        out[float(b)] = set(int(x) for x in priority[:n].tolist())
    return out


class OnlinePrototypeBank:
    """Tiny bounded online cosine-prototype bank."""

    def __init__(self, max_prototypes: int, device: torch.device):
        if int(max_prototypes) <= 0:
            raise ValueError("max_prototypes must be positive")
        self.max_prototypes = int(max_prototypes)
        self.device = device
        self.prototypes: Optional[torch.Tensor] = None
        self.counts: Optional[torch.Tensor] = None
        self.n_observations = 0

    @property
    def size(self) -> int:
        return 0 if self.prototypes is None else int(self.prototypes.shape[0])

    @torch.no_grad()
    def update(self, vectors: torch.Tensor) -> None:
        if vectors.numel() == 0:
            return
        x = F.normalize(vectors.detach().float().to(self.device), dim=1)

        for i in range(int(x.shape[0])):
            v = x[i]
            if self.prototypes is None:
                self.prototypes = v.unsqueeze(0)
                self.counts = torch.ones(1, device=self.device, dtype=torch.float32)
            elif self.size < self.max_prototypes:
                self.prototypes = torch.cat([self.prototypes, v.unsqueeze(0)], dim=0)
                self.counts = torch.cat(
                    [
                        self.counts,
                        torch.ones(1, device=self.device, dtype=torch.float32),
                    ],
                    dim=0,
                )
            else:
                sims = self.prototypes @ v
                j = int(torch.argmax(sims).item())
                c = float(self.counts[j].item())
                merged = (self.prototypes[j] * c + v) / (c + 1.0)
                self.prototypes[j] = F.normalize(merged.unsqueeze(0), dim=1)[0]
                self.counts[j] = c + 1.0
            self.n_observations += 1

    @torch.no_grad()
    def max_similarity(self, patch_features: torch.Tensor) -> torch.Tensor:
        n = int(patch_features.shape[0])
        if self.prototypes is None or self.size == 0:
            return torch.zeros(n, device=patch_features.device, dtype=torch.float32)
        q = F.normalize(patch_features.detach().float(), dim=1)
        p = self.prototypes.to(q.device)
        sim = (q @ p.T).max(dim=1).values
        return torch.clamp(sim, min=0.0, max=1.0)


@dataclass
class CategoryCorrectionMemory:
    positive: OnlinePrototypeBank
    negative: OnlinePrototypeBank

    @classmethod
    def create(
        cls,
        max_prototypes: int,
        device: torch.device,
    ) -> "CategoryCorrectionMemory":
        return cls(
            positive=OnlinePrototypeBank(max_prototypes, device),
            negative=OnlinePrototypeBank(max_prototypes, device),
        )


@torch.no_grad()
def corrected_patch_scores(
    baseline_scores: torch.Tensor,
    patch_features: torch.Tensor,
    memory: CategoryCorrectionMemory,
    branch: str,
) -> torch.Tensor:
    """Return corrected scores for one of the E1B learning branches."""
    if branch not in {"suppress", "bidirectional"}:
        raise ValueError(f"unknown learning branch={branch!r}")

    s0 = baseline_scores.detach().float()
    scale = torch.std(s0, unbiased=False)
    if not torch.isfinite(scale):
        raise ValueError("non-finite baseline score scale")

    sim_neg = memory.negative.max_similarity(patch_features)
    if branch == "suppress":
        delta = -sim_neg
    else:
        sim_pos = memory.positive.max_similarity(patch_features)
        delta = sim_pos - sim_neg

    return s0 + float(PRIMARY_CORRECTION_STRENGTH) * scale * delta


def _spatial_row(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int = PRIMARY_SENSOR_K,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    s = scores.detach().float().cpu().reshape(-1)
    kk = min(int(k), int(s.numel()))
    order = torch.argsort(s, descending=True)
    top_idx = order[:kk]

    if mask is None:
        return {
            "mask_available": False,
            "geometry_failure": None,
            "hit": None,
            "precision": None,
            "recall": None,
            "first_hit_rank": None,
        }, top_idx

    y = mask.detach().cpu().bool().reshape(-1)
    if y.numel() != s.numel():
        raise ValueError("mask and score patch counts differ")

    n_pos = int(y.sum().item())
    if n_pos == 0:
        return {
            "mask_available": True,
            "geometry_failure": True,
            "hit": None,
            "precision": None,
            "recall": None,
            "first_hit_rank": None,
        }, top_idx

    inter = int(y[top_idx].sum().item())
    positive_ranks = torch.nonzero(y[order], as_tuple=False).flatten()
    first_hit = int(positive_ranks[0].item()) + 1
    return {
        "mask_available": True,
        "geometry_failure": False,
        "hit": bool(inter > 0),
        "precision": float(inter / kk),
        "recall": float(inter / n_pos),
        "first_hit_rank": first_hit,
    }, top_idx


class SpatialAccumulator:
    def __init__(self) -> None:
        self.n_total = 0
        self.n_valid = 0
        self.n_geometry_failure = 0
        self.hit_sum = 0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.first_hit_ranks: List[int] = []

    def add(self, row: Mapping[str, Any]) -> None:
        self.n_total += 1
        if row.get("geometry_failure") is True:
            self.n_geometry_failure += 1
            return
        if row.get("geometry_failure") is not False:
            return
        self.n_valid += 1
        self.hit_sum += int(bool(row["hit"]))
        self.precision_sum += float(row["precision"])
        self.recall_sum += float(row["recall"])
        self.first_hit_ranks.append(int(row["first_hit_rank"]))

    def summary(self) -> Dict[str, Any]:
        n = self.n_valid
        return {
            "n_total": int(self.n_total),
            "n_valid_nonempty_masks": int(n),
            "n_geometry_failures": int(self.n_geometry_failure),
            "hit_at_8": float(self.hit_sum / n) if n else float("nan"),
            "mean_precision_at_8": (
                float(self.precision_sum / n) if n else float("nan")
            ),
            "mean_recall_at_8": (
                float(self.recall_sum / n) if n else float("nan")
            ),
            "median_first_hit_rank": (
                float(np.median(self.first_hit_ranks))
                if self.first_hit_ranks
                else float("nan")
            ),
        }


class RoutingAccumulator:
    def __init__(self) -> None:
        self.y: List[str] = []
        self.pred: List[str] = []

    def add(self, y: str, pred: str) -> None:
        self.y.append(str(y))
        self.pred.append(str(pred))

    def summary(self) -> Dict[str, Any]:
        if not self.y:
            return {
                "n": 0,
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
            }
        y = np.asarray(self.y, dtype=object)
        p = np.asarray(self.pred, dtype=object)
        return {
            "n": int(len(y)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(
                f1_score(
                    y,
                    p,
                    labels=list(CORE4),
                    average="macro",
                    zero_division=0,
                )
            ),
        }


@dataclass(frozen=True)
class FixedRouter:
    source_cv_fold: int
    test_categories: Tuple[str, ...]
    memory: Mapping[str, Mapping[str, Any]]
    prior: Mapping[str, np.ndarray]


def _load_e1a_sensor_pool(
    path: str | Path,
    outer_fold: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(Path(path), map_location="cpu")

    if artifact.get("schema") != "abmg.e1a.localization_gap.artifact.v1":
        raise ValueError(f"unexpected E1A artifact schema={artifact.get('schema')!r}")
    if int(artifact.get("outer_fold", -1)) != int(outer_fold):
        raise ValueError("E1A artifact outer fold does not match E1B --fold")

    paired_idx = artifact["paired_original_indices"].numpy().astype(np.int64)
    sensor_x = artifact["sensor_features"].float().numpy().astype(np.float64)
    items = artifact["items"]

    image_ids = np.asarray([str(items[i]["image_id"]) for i in paired_idx], dtype=object)
    categories = np.asarray([str(items[i]["category"]) for i in paired_idx], dtype=object)
    labels = np.asarray(
        [str(items[i]["defect_source_offline_only"]) for i in paired_idx],
        dtype=object,
    )
    return artifact, sensor_x, categories, labels, image_ids


def build_fixed_routers(
    e1a_path: str | Path,
    outer_fold: int,
    *,
    seed: int,
    shots: int = PRIMARY_ROUTING_SHOTS,
    kappa0: float = PRIMARY_KAPPA0,
) -> Tuple[Dict[str, FixedRouter], Dict[str, Any]]:
    artifact, X, categories, labels, _ = _load_e1a_sensor_pool(
        e1a_path,
        outer_fold,
    )
    cv_groups = artifact["source_cv_groups"]
    all_categories = set(categories.tolist())
    wanted = set(CORE4)

    by_category: Dict[str, FixedRouter] = {}
    meta: Dict[str, Any] = {"folds": []}

    for fi, group in enumerate(cv_groups):
        test_cats = set(str(x) for x in group)
        train_cats = all_categories - test_cats
        train_all = np.asarray([c in train_cats for c in categories], dtype=bool)
        train_core = np.asarray(
            [(c in train_cats) and (y in wanted) for c, y in zip(categories, labels)],
            dtype=bool,
        )
        xt = X[train_core]
        yt = labels[train_core]
        ct = categories[train_core]

        if set(yt.tolist()) != wanted:
            raise RuntimeError(f"source CV fold {fi} lacks Core-4 training coverage")

        support_seed = int(seed) + 100_000 * fi + 17
        support = category_diverse_support_indices(
            yt,
            ct,
            CORE4,
            int(shots),
            support_seed,
        )
        memory = init_memory(xt, support)
        prior = fit_shared_diag_prior(X[train_all])

        router = FixedRouter(
            source_cv_fold=int(fi),
            test_categories=tuple(sorted(test_cats)),
            memory=memory,
            prior=prior,
        )
        for cat in test_cats:
            if cat in by_category:
                raise RuntimeError(f"category {cat} appears in multiple source-CV groups")
            by_category[cat] = router

        meta["folds"].append(
            {
                "source_cv_fold": int(fi),
                "test_categories": sorted(test_cats),
                "n_train_core4": int(train_core.sum()),
                "n_train_background": int(train_all.sum()),
                "support_categories": {
                    y: sorted(set(ct[idx].tolist()))
                    for y, idx in support.items()
                },
            }
        )

    return by_category, meta


def _route_one(
    vector: torch.Tensor,
    router: FixedRouter,
    kappa0: float = PRIMARY_KAPPA0,
) -> str:
    x = vector.detach().float().cpu().numpy().astype(np.float64)[None, :]
    pred, _ = score_memory(
        x,
        router.memory,
        router.prior,
        float(kappa0),
        "diag_shrunk",
    )
    return str(pred[0])


def _condition_key(branch: str, budget: Optional[float] = None) -> str:
    if branch == "frozen":
        return "frozen"
    if budget is None:
        raise ValueError("learning condition needs a mask budget")
    return f"{branch}__{budget_tag(float(budget))}"


def _prepare_stream(
    defects: Sequence[Stage0Record],
    seed: int,
) -> List[Stage0Record]:
    """Shuffle within category, then concatenate categories.

    Because correction memory is category-local, interleaving categories would
    not change any category's state trajectory. Category blocks avoid repeatedly
    rebuilding large category-specific normal-support banks.
    """
    by_cat: Dict[str, List[Stage0Record]] = defaultdict(list)
    for r in defects:
        by_cat[str(r.category)].append(r)

    out: List[Stage0Record] = []
    for ci, cat in enumerate(sorted(by_cat)):
        rows = sorted(by_cat[cat], key=lambda r: str(r.image_id))
        rng = np.random.default_rng(int(seed) + 10_000 * (ci + 1))
        order = rng.permutation(len(rows))
        out.extend(rows[int(i)] for i in order.tolist())
    return out


def run(args: argparse.Namespace) -> int:
    budgets = parse_mask_budgets(args.mask_budgets)
    if int(args.sensor_k) != PRIMARY_SENSOR_K:
        raise ValueError("primary E1B is fixed at Sensor top-8")
    if float(args.score_temperature) != PRIMARY_SCORE_TEMPERATURE:
        raise ValueError("primary E1B fixes score-softmax temperature at 20")
    if int(args.prototypes_per_sign) != PRIMARY_PROTOTYPES_PER_SIGN:
        raise ValueError("primary E1B fixes 8 prototypes per sign")
    if int(args.routing_shots) != PRIMARY_ROUTING_SHOTS:
        raise ValueError("primary E1B fixes 8 routing supports per Core-4 factor")
    if float(args.kappa0) != PRIMARY_KAPPA0:
        raise ValueError("primary E1B fixes kappa0=1")

    protocol = load_fold_protocol(args.fold, args.folds)
    records = load_realiad_records(
        args.root,
        args.json_dir,
        set(protocol.source_categories),
    )
    defects = source_defect_records(records, protocol.source_categories)
    if not defects:
        raise RuntimeError("no source defect images found")

    stream = _prepare_stream(defects, int(args.seed))
    if int(args.max_items) > 0:
        stream = stream[: int(args.max_items)]
    n_stream = len(stream)
    if n_stream == 0:
        raise RuntimeError("empty E1B stream")

    schedules = fixed_nested_mask_schedules(
        n_stream,
        budgets,
        int(args.mask_schedule_seed),
    )

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

    condition_keys = ["frozen"]
    for b in budgets:
        condition_keys.extend(
            [
                _condition_key("suppress", b),
                _condition_key("bidirectional", b),
            ]
        )

    spatial = {k: SpatialAccumulator() for k in condition_keys}
    routing = {k: RoutingAccumulator() for k in condition_keys}
    transition_counts = {
        k: Counter()
        for k in condition_keys
        if k != "frozen"
    }
    mask_usage: Dict[str, Dict[str, Any]] = {
        k: {
            "scheduled_events": 0,
            "usable_events": 0,
            "positive_patch_observations": 0,
            "negative_patch_observations": 0,
        }
        for k in condition_keys
        if k != "frozen"
    }
    mask_events: List[Dict[str, Any]] = []

    # Stream is category-blocked for efficient normal-bank construction.
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

            # Each condition has independent category-local online state.
            memories: Dict[str, CategoryCorrectionMemory] = {}
            for b in budgets:
                for branch in ("suppress", "bidirectional"):
                    key = _condition_key(branch, b)
                    memories[key] = CategoryCorrectionMemory.create(
                        int(args.prototypes_per_sign),
                        normal_bank.device,
                    )

            print(
                f"[{ci}/{len(categories)}] {category}: "
                f"support={len(supports)}, stream={len(cat_rows)}"
            )

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
                            print(
                                f"  WARN mask failed for {rec.relative_path}: {exc}"
                            )

                    frozen_spatial, frozen_top = _spatial_row(
                        s0,
                        mask,
                        int(args.sensor_k),
                    )
                    spatial["frozen"].add(frozen_spatial)

                    frozen_vec = pool_selected_raw_patches(
                        qf,
                        s0,
                        frozen_top,
                        mode="score_softmax",
                        temperature=float(args.score_temperature),
                    )

                    per_condition_pred: Dict[str, str] = {}
                    true_source = str(rec.defect_source)
                    router = routers.get(str(rec.category))
                    route_eligible = (
                        router is not None
                        and true_source in set(CORE4)
                        and frozen_spatial.get("geometry_failure") is False
                    )
                    if route_eligible:
                        pred = _route_one(frozen_vec, router, float(args.kappa0))
                        routing["frozen"].add(true_source, pred)
                        per_condition_pred["frozen"] = pred

                    condition_top: Dict[str, torch.Tensor] = {}
                    for b in budgets:
                        for branch in ("suppress", "bidirectional"):
                            key = _condition_key(branch, b)
                            corr_scores = corrected_patch_scores(
                                s0,
                                qf,
                                memories[key],
                                branch,
                            )
                            row, top_idx = _spatial_row(
                                corr_scores,
                                mask,
                                int(args.sensor_k),
                            )
                            spatial[key].add(row)
                            condition_top[key] = top_idx

                            corr_vec = pool_selected_raw_patches(
                                qf,
                                corr_scores,
                                top_idx,
                                mode="score_softmax",
                                temperature=float(args.score_temperature),
                            )
                            if route_eligible:
                                pred = _route_one(
                                    corr_vec,
                                    router,
                                    float(args.kappa0),
                                )
                                routing[key].add(true_source, pred)
                                per_condition_pred[key] = pred

                    # Paired routing transitions relative to frozen.
                    if route_eligible:
                        base_correct = per_condition_pred["frozen"] == true_source
                        for key in transition_counts:
                            pred = per_condition_pred[key]
                            corr_correct = pred == true_source
                            if base_correct and corr_correct:
                                transition_counts[key]["correct_to_correct"] += 1
                            elif base_correct and not corr_correct:
                                transition_counts[key]["correct_to_wrong"] += 1
                            elif (not base_correct) and corr_correct:
                                transition_counts[key]["wrong_to_correct"] += 1
                            else:
                                transition_counts[key]["wrong_to_wrong"] += 1

                    # Feedback is revealed only AFTER all branches score this image.
                    active_budgets = [
                        float(b) for b in budgets if int(global_pos) in schedules[float(b)]
                    ]
                    if active_budgets:
                        usable = (
                            mask is not None
                            and frozen_spatial.get("geometry_failure") is False
                        )
                        n_pos = 0
                        n_neg = 0
                        pos_vec = torch.empty(
                            (0, qf.shape[1]),
                            device=qf.device,
                            dtype=qf.dtype,
                        )
                        neg_vec = torch.empty_like(pos_vec)

                        if usable:
                            y = mask.to(qf.device).bool().reshape(-1)
                            pos_idx = torch.nonzero(y, as_tuple=False).flatten()
                            frozen_top_dev = frozen_top.to(qf.device)
                            neg_idx = frozen_top_dev[~y[frozen_top_dev]]
                            pos_vec = qf[pos_idx]
                            neg_vec = qf[neg_idx]
                            n_pos = int(pos_vec.shape[0])
                            n_neg = int(neg_vec.shape[0])

                        event = {
                            "stream_position": int(global_pos),
                            "image_id": rec.image_id,
                            "category": rec.category,
                            "scheduled_budgets": active_budgets,
                            "usable_nonempty_mask": bool(usable),
                            "n_positive_mask_patches": int(n_pos),
                            "n_negative_frozen_top8_patches": int(n_neg),
                        }
                        mask_events.append(event)

                        for b in active_budgets:
                            for branch in ("suppress", "bidirectional"):
                                key = _condition_key(branch, b)
                                use = mask_usage[key]
                                use["scheduled_events"] += 1
                                if not usable:
                                    continue
                                use["usable_events"] += 1
                                use["negative_patch_observations"] += int(n_neg)
                                memories[key].negative.update(neg_vec)
                                if branch == "bidirectional":
                                    use["positive_patch_observations"] += int(n_pos)
                                    memories[key].positive.update(pos_vec)

                done = min(start + len(block), len(cat_rows))
                if args.progress_every > 0 and (
                    done % int(args.progress_every) == 0
                    or done == len(cat_rows)
                ):
                    print(f"  {done}/{len(cat_rows)}")

            del normal_bank
            del memories
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        sensor.cleanup()

    frozen_spatial_summary = spatial["frozen"].summary()
    frozen_routing_summary = routing["frozen"].summary()

    conditions: Dict[str, Any] = {
        "frozen": {
            "branch": "frozen",
            "mask_budget_fraction": 0.0,
            "spatial": frozen_spatial_summary,
            "routing": frozen_routing_summary,
        }
    }

    for b in budgets:
        for branch in ("suppress", "bidirectional"):
            key = _condition_key(branch, b)
            sp = spatial[key].summary()
            rt = routing[key].summary()
            trans = dict(sorted(transition_counts[key].items()))
            n_trans = sum(trans.values())
            conditions[key] = {
                "branch": branch,
                "mask_budget_fraction": float(b),
                "mask_budget_percent": float(100.0 * b),
                "mask_usage": mask_usage[key],
                "spatial": sp,
                "routing": rt,
                "delta_vs_frozen": {
                    "hit_at_8": float(sp["hit_at_8"] - frozen_spatial_summary["hit_at_8"]),
                    "mean_precision_at_8": float(
                        sp["mean_precision_at_8"]
                        - frozen_spatial_summary["mean_precision_at_8"]
                    ),
                    "mean_recall_at_8": float(
                        sp["mean_recall_at_8"]
                        - frozen_spatial_summary["mean_recall_at_8"]
                    ),
                    "routing_macro_f1": float(
                        rt["macro_f1"] - frozen_routing_summary["macro_f1"]
                    ),
                },
                "routing_transitions_vs_frozen": {
                    "counts": trans,
                    "fractions": {
                        k: float(v / n_trans) if n_trans else float("nan")
                        for k, v in trans.items()
                    },
                },
            }

    summary = {
        "schema": "abmg.e1b.rare_mask_utility.summary.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "n_source_defect_images_in_stream": int(n_stream),
        "protocol": {
            "stream": (
                "all source defect images; deterministic label-independent shuffle "
                "within category; categories processed in blocks because correction "
                "memory is category-local"
            ),
            "mask_schedule": (
                "fixed deterministic random IMAGE positions; nested budgets; "
                "mask revealed after prediction; no v22 controller"
            ),
            "mask_budgets": [float(x) for x in budgets],
            "sensor_topk": int(args.sensor_k),
            "sensor_pooling": "score_softmax",
            "score_softmax_temperature": float(args.score_temperature),
            "correction_memory": (
                f"{int(args.prototypes_per_sign)} online cosine prototypes per "
                "category per sign"
            ),
            "correction_strength": (
                "one within-image standard deviation of frozen patch scores"
            ),
            "negative_feedback": (
                "frozen Sensor top-8 patches outside the full ground-truth mask"
            ),
            "positive_feedback": "all positive patches in the full ground-truth mask",
            "factor_router": (
                "fixed E1A diag_shrunk router; 8 supports/Core-4 factor; kappa0=1; "
                "no factor-memory update"
            ),
        },
        "conditions": conditions,
        "router_reconstruction": router_meta,
    }

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(summary, out_dir / "e1b_rare_mask_summary.json")
    _json_dump(mask_events, out_dir / "e1b_mask_events.json")
    _json_dump(
        {
            "schema": "abmg.e1b.rare_mask_utility.config.v1",
            "outer_fold": int(args.fold),
            "e1a_artifact": str(args.e1a_artifact),
            "sensor": asdict(cfg),
            "sensor_fingerprint": config_fingerprint(cfg),
            "mask_budgets": [float(x) for x in budgets],
            "mask_schedule_seed": int(args.mask_schedule_seed),
            "stream_seed": int(args.seed),
            "prototypes_per_sign": int(args.prototypes_per_sign),
            "max_items": int(args.max_items),
        },
        out_dir / "resolved_config.json",
    )

    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"Wrote E1B outputs to: {out_dir}")
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
        default="outputs/e1b_rare_mask_utility",
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
        "--mask-budgets",
        type=str,
        default="0.0025,0.005,0.01",
        help="Fractions of defect images whose full masks are revealed after prediction.",
    )
    p.add_argument(
        "--mask-schedule-seed",
        type=int,
        default=271828,
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="0 = all source defect images; use a small value only for smoke tests.",
    )
    p.add_argument("--progress-every", type=int, default=64)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
