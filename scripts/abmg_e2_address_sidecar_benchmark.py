#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 address-representation benchmark for ABMG.

Only address geometry changes.

Frozen detection path
---------------------
image -> frozen DINO patch h -> frozen normal NN score -> top-8 + score-softmax

Candidate address path
----------------------
the SAME h -> optional sidecar f(h)=z_addr -> same spatial pooling
           -> same diag_shrunk Core-4 memory/router

No candidate sidecar can modify the raw DINO anomaly score, selected top-8
patches, or their score-softmax weights.

Data boundary
-------------
For each source-CV fold:
* 18 source product categories are development categories.
* 6 source product categories are held out for addressability evaluation.
* The outer six target categories remain unopened.
* Sidecars fit only NORMAL TRAIN patch descriptors from the 18 development
  categories. Defect-source labels and masks never fit a sidecar.
* Core-4 defect labels/masks are evaluator-only after the sidecar is frozen.

The screening default caps each (product, defect-source) cell at 64 examples.
Use --max-per-type-per-category 0 only for finalist/full confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

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
from abmg_prototype_addressability_audit import (
    CORE4,
    category_diverse_support_indices,
)
from abmg_memory_addressability_audit import fit_shared_diag_prior
from abmg_sequential_local_update_audit import (
    init_memory,
    score_memory,
    true_margin,
)
from abmg_e2_sidecars import (
    CANDIDATES,
    SidecarTrainConfig,
    AddressSidecar,
    build_sidecar,
    coordinate_participation_ratio,
    group_participation_ratio,
    sidecar_config_dict,
)


PRIMARY_SENSOR_K = 8
PRIMARY_SCORE_TEMPERATURE = 20.0
PRIMARY_ROUTER = "diag_shrunk"
PRIMARY_ROUTING_SHOTS = 8
PRIMARY_KAPPA0 = 1.0
PRIMARY_SUCCESS_DELTA_F1 = 0.02
PRIMARY_ORACLE_DROP_TOLERANCE = 0.02


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def _sha_order(seed: int, key: str) -> str:
    return hashlib.sha1(f"{int(seed)}|{key}".encode("utf-8")).hexdigest()


def parse_csv(spec: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in str(spec).split(",") if x.strip())


def parse_int_csv(spec: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if not vals:
        raise ValueError("integer list is empty")
    return vals


def parse_cv_folds(spec: str, n_folds: int) -> Tuple[int, ...]:
    s = str(spec).strip().lower()
    if s == "all":
        return tuple(range(int(n_folds)))
    vals = tuple(sorted(set(int(x) for x in parse_csv(s))))
    if not vals:
        raise ValueError("no source CV folds requested")
    if min(vals) < 0 or max(vals) >= int(n_folds):
        raise ValueError(f"source CV folds must be in [0,{n_folds-1}]")
    return vals


def _sample_patch_rows(
    q: torch.Tensor,
    n: int,
    seed: int,
    salt: str,
) -> torch.Tensor:
    n = min(int(n), int(q.shape[0]))
    if n <= 0:
        return q[:0]
    h = int(hashlib.sha1(f"{seed}|{salt}".encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + h)
    idx = torch.randperm(int(q.shape[0]), generator=g)[:n].to(q.device)
    return q[idx]


def _split_normal_train_images(
    records: Sequence[Stage0Record],
    categories: Sequence[str],
    *,
    val_fraction: float,
    max_train_images_per_category: int,
    max_val_images_per_category: int,
    seed: int,
) -> Tuple[List[Stage0Record], List[Stage0Record]]:
    train_rows: List[Stage0Record] = []
    val_rows: List[Stage0Record] = []
    for cat in sorted(set(str(x) for x in categories)):
        xs = [
            r
            for r in records
            if r.category == cat and r.split == "train" and r.is_good
        ]
        xs = sorted(xs, key=lambda r: _sha_order(seed, r.image_id))
        if len(xs) < 2:
            raise RuntimeError(
                f"Need >=2 normal train images for sidecar train/val: {cat}"
            )
        n_val = max(1, int(round(float(val_fraction) * len(xs))))
        n_val = min(n_val, len(xs) - 1)
        val = xs[:n_val]
        train = xs[n_val:]
        if int(max_train_images_per_category) > 0:
            train = train[: int(max_train_images_per_category)]
        if int(max_val_images_per_category) > 0:
            val = val[: int(max_val_images_per_category)]
        train_rows.extend(train)
        val_rows.extend(val)
    return train_rows, val_rows


@torch.no_grad()
def collect_normal_patch_dataset(
    sensor: PaperAlignedFrozenSensor,
    records: Sequence[Stage0Record],
    categories: Sequence[str],
    *,
    patches_per_image: int,
    val_fraction: float,
    max_train_images_per_category: int,
    max_val_images_per_category: int,
    seed: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    train_rows, val_rows = _split_normal_train_images(
        records,
        categories,
        val_fraction=val_fraction,
        max_train_images_per_category=max_train_images_per_category,
        max_val_images_per_category=max_val_images_per_category,
        seed=seed,
    )

    def encode_rows(rows: Sequence[Stage0Record], split_name: str) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for start in range(0, len(rows), int(batch_size)):
            batch = rows[start : start + int(batch_size)]
            q_batch, _ = sensor.encode_path_batch(batch)
            for bi, rec in enumerate(batch):
                samp = _sample_patch_rows(
                    q_batch[bi],
                    int(patches_per_image),
                    int(seed),
                    f"{split_name}|{rec.image_id}",
                )
                chunks.append(samp.detach().cpu().to(torch.float16))
        if not chunks:
            raise RuntimeError(f"No normal patches collected for {split_name}")
        return torch.cat(chunks, dim=0).contiguous()

    train_x = encode_rows(train_rows, "train")
    val_x = encode_rows(val_rows, "val")
    meta = {
        "n_train_images": len(train_rows),
        "n_val_images": len(val_rows),
        "n_train_patches": int(train_x.shape[0]),
        "n_val_patches": int(val_x.shape[0]),
        "patches_per_image": int(patches_per_image),
        "train_categories": sorted(set(str(x) for x in categories)),
        "labels_or_masks_used": False,
    }
    return train_x, val_x, meta


def _pool_sensor_address(
    z_patch: torch.Tensor,
    patch_scores: torch.Tensor,
    top_idx: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    idx = top_idx.to(z_patch.device)
    s = patch_scores[idx].float()
    w = torch.softmax(float(temperature) * s, dim=0)
    return torch.sum(w[:, None] * z_patch[idx].float(), dim=0)


def _pool_oracle_address(
    z_patch: torch.Tensor,
    oracle_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if oracle_mask is None:
        return None
    idx = torch.nonzero(oracle_mask, as_tuple=False).flatten()
    if idx.numel() == 0:
        return None
    return z_patch[idx.to(z_patch.device)].float().mean(dim=0)


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    return {
        "accuracy": float(np.mean(pred == y)),
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
        "per_class_f1": {
            str(label): float(
                f1_score(
                    y == label,
                    pred == label,
                    zero_division=0,
                )
            )
            for label in CORE4
        },
        "confusion_true_normalized": confusion_matrix(
            y,
            pred,
            labels=list(CORE4),
            normalize="true",
        ).tolist(),
    }


def _top2_recall(y: np.ndarray, scores: np.ndarray) -> float:
    lut = {str(yv): i for i, yv in enumerate(CORE4)}
    truth = np.asarray([lut[str(v)] for v in y], dtype=np.int64)
    top2 = np.argpartition(scores, -2, axis=1)[:, -2:]
    return float(np.mean([truth[i] in top2[i] for i in range(len(truth))]))


def _routing_signature(scores: np.ndarray) -> np.ndarray:
    """Scale-invariant 4-D routing signature for responsibility stability."""
    s = np.asarray(scores, dtype=np.float64)
    s = s - s.mean(axis=1, keepdims=True)
    sd = s.std(axis=1, keepdims=True)
    s = s / np.maximum(sd, 1e-8)
    n = np.linalg.norm(s, axis=1, keepdims=True)
    return s / np.maximum(n, 1e-12)


def _signature_stability(
    y: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    sig = _routing_signature(scores)
    centroids: Dict[str, np.ndarray] = {}
    for label in CORE4:
        m = y == label
        if np.any(m):
            c = sig[m].mean(axis=0)
            c = c / max(float(np.linalg.norm(c)), 1e-12)
            centroids[label] = c

    own: List[float] = []
    other: List[float] = []
    for i, label_obj in enumerate(y.tolist()):
        label = str(label_obj)
        if label not in centroids:
            continue
        own.append(float(sig[i] @ centroids[label]))
        competitors = [
            float(sig[i] @ c)
            for k, c in centroids.items()
            if k != label
        ]
        if competitors:
            other.append(max(competitors))
    own_m = float(np.mean(own)) if own else float("nan")
    other_m = float(np.mean(other)) if other else float("nan")
    return {
        "within_source_cosine_to_centroid": own_m,
        "max_other_source_cosine_to_centroid": other_m,
        "source_signature_gap": float(own_m - other_m),
    }


def _evaluate_routing_once(
    X: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    test_categories: Sequence[str],
    *,
    support_seed: int,
    shots: int,
    kappa0: float,
) -> Tuple[Dict[str, Any], Mapping[str, Mapping[str, Any]], Mapping[str, np.ndarray]]:
    wanted = set(CORE4)
    test_cats = set(str(x) for x in test_categories)
    all_cats = set(categories.tolist())
    train_cats = all_cats - test_cats

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
        raise RuntimeError("empty Core-4 train/test partition")

    xt = X[train_core]
    yt = labels[train_core]
    ct = categories[train_core]
    xv = X[test_core]
    yv = labels[test_core]

    if set(yt.tolist()) != wanted or set(yv.tolist()) != wanted:
        raise RuntimeError(
            "Core-4 coverage incomplete in source-CV fold: "
            f"train={sorted(set(yt.tolist()))}, test={sorted(set(yv.tolist()))}"
        )

    support = category_diverse_support_indices(
        yt,
        ct,
        CORE4,
        int(shots),
        int(support_seed),
    )
    prior = fit_shared_diag_prior(X[train_all])
    memory = init_memory(xt, support)
    pred, scores = score_memory(
        xv,
        memory,
        prior,
        float(kappa0),
        PRIMARY_ROUTER,
    )
    margins = true_margin(yv, scores)

    out = {
        "support_seed": int(support_seed),
        "n_train_background": int(train_all.sum()),
        "n_train_core4": int(train_core.sum()),
        "n_test_core4": int(test_core.sum()),
        "support_categories": {
            y: sorted(set(ct[idx].tolist()))
            for y, idx in support.items()
        },
        "metrics": _metrics(yv, pred),
        "top2_recall": _top2_recall(yv, scores),
        "mean_true_minus_best_wrong_margin": float(margins.mean()),
        "routing_signature": _signature_stability(yv, scores),
    }
    return out, memory, prior


def _finite_mean_std(xs: Iterable[float]) -> Tuple[float, float]:
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def _aggregate_seed_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_support_seeds": len(runs)}
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        m, s = _finite_mean_std(r["metrics"][key] for r in runs)
        out[f"{key}_mean"] = m
        out[f"{key}_std"] = s
    m, s = _finite_mean_std(r["top2_recall"] for r in runs)
    out["top2_recall_mean"] = m
    out["top2_recall_std"] = s
    m, s = _finite_mean_std(
        r["mean_true_minus_best_wrong_margin"] for r in runs
    )
    out["margin_mean"] = m
    out["margin_std"] = s
    m, s = _finite_mean_std(
        r["routing_signature"]["source_signature_gap"] for r in runs
    )
    out["source_signature_gap_mean"] = m
    out["source_signature_gap_std"] = s
    return out


def _address_efficiency(X: np.ndarray, candidate: str, corr_groups: int) -> Dict[str, Any]:
    t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    out = {
        "coordinate_participation_ratio": coordinate_participation_ratio(t),
        "address_dim": int(t.shape[1]),
    }
    if candidate == "corrvae":
        out["group_participation_ratio"] = group_participation_ratio(
            t, int(corr_groups)
        )
        out["n_groups"] = int(corr_groups)
    if candidate == "sparse_ae":
        out["fraction_abs_lt_1e-3"] = float(
            (t.abs() < 1e-3).float().mean().item()
        )
    return out


@torch.no_grad()
def _spatial_consistency(
    sensor: PaperAlignedFrozenSensor,
    sidecar: AddressSidecar,
    probe_records: Sequence[Stage0Record],
    memory: Mapping[str, Mapping[str, Any]],
    prior: Mapping[str, np.ndarray],
    cfg: Stage0Config,
    *,
    kappa0: float,
    device: torch.device,
) -> Dict[str, Any]:
    lut = {str(y): i for i, y in enumerate(CORE4)}
    aucs: List[float] = []
    inside_minus_outside: List[float] = []
    n_geometry_failure = 0

    for rec in probe_records:
        if rec.defect_source not in lut:
            continue
        if not rec.mask_path or not Path(rec.mask_path).is_file():
            continue
        q_batch, _ = sensor.encode_path_batch([rec])
        qf = q_batch[0]
        mask = mask_to_patch_mask(rec.mask_path, cfg)
        if int(mask.sum().item()) == 0:
            n_geometry_failure += 1
            continue

        z_patch = sidecar.transform(qf, device).detach().cpu().numpy()
        _, patch_scores = score_memory(
            z_patch,
            memory,
            prior,
            float(kappa0),
            PRIMARY_ROUTER,
        )
        true_col = lut[str(rec.defect_source)]
        s = patch_scores[:, true_col]
        y = mask.numpy().astype(np.int64)
        if len(np.unique(y)) >= 2:
            aucs.append(float(roc_auc_score(y, s)))
        inside = s[y == 1]
        outside = s[y == 0]
        if len(inside) and len(outside):
            inside_minus_outside.append(
                float(np.mean(inside) - np.mean(outside))
            )

    return {
        "n_probes": int(len(aucs)),
        "n_geometry_failures": int(n_geometry_failure),
        "true_factor_patch_auroc_mean": (
            float(np.mean(aucs)) if aucs else float("nan")
        ),
        "true_factor_score_inside_minus_outside_mean": (
            float(np.mean(inside_minus_outside))
            if inside_minus_outside
            else float("nan")
        ),
    }


def _select_spatial_probes(
    records: Sequence[Stage0Record],
    test_categories: Sequence[str],
    n: int,
    seed: int,
) -> List[Stage0Record]:
    test = set(str(x) for x in test_categories)
    xs = [
        r
        for r in records
        if r.category in test
        and r.defect_source in set(CORE4)
        and r.mask_path
        and Path(r.mask_path).is_file()
    ]
    xs = sorted(xs, key=lambda r: _sha_order(seed, r.image_id))
    return xs[: int(n)] if int(n) > 0 else []


def _fit_candidates(
    names: Sequence[str],
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    cfg: SidecarTrainConfig,
    device: torch.device,
) -> Tuple[Dict[str, AddressSidecar], Dict[str, Any]]:
    input_dim = int(train_x.shape[1])
    sidecars: Dict[str, AddressSidecar] = {}
    reports: Dict[str, Any] = {}
    for name in names:
        print(f"  fitting sidecar: {name}")
        sidecar = build_sidecar(name, input_dim, cfg)
        reports[name] = sidecar.fit(train_x, val_x, cfg, device)
        sidecars[name] = sidecar
    return sidecars, reports


@torch.no_grad()
def _extract_candidate_evidence(
    sensor: PaperAlignedFrozenSensor,
    sidecars: Mapping[str, AddressSidecar],
    records_all: Sequence[Stage0Record],
    defects: Sequence[Stage0Record],
    cfg: Stage0Config,
    *,
    sensor_k: int,
    temperature: float,
    query_batch_size: int,
    nn_chunk: int,
    device: torch.device,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, Any],
]:
    by_category: Dict[str, List[Stage0Record]] = defaultdict(list)
    for r in defects:
        by_category[r.category].append(r)

    sensor_store: Dict[str, List[torch.Tensor]] = {
        name: [] for name in sidecars
    }
    oracle_store: Dict[str, List[torch.Tensor]] = {
        name: [] for name in sidecars
    }
    categories: List[str] = []
    labels: List[str] = []
    image_ids: List[str] = []
    oracle_valid: List[bool] = []
    support_meta: Dict[str, Any] = {}

    for ci, category in enumerate(sorted(by_category), 1):
        supports = select_supports(
            records_all,
            category,
            cfg.shots,
            cfg.support_seed,
        )
        bank = sensor.build_category_bank(supports)
        support_meta[category] = {
            "support_image_ids": [r.image_id for r in supports],
            "n_normal_bank_patches": int(bank.shape[0]),
        }
        rows = by_category[category]
        print(
            f"  evidence [{ci}/{len(by_category)}] {category}: {len(rows)} defects"
        )

        for start in range(0, len(rows), int(query_batch_size)):
            batch = rows[start : start + int(query_batch_size)]
            q_batch, _ = sensor.encode_path_batch(batch)
            for bi, rec in enumerate(batch):
                qf = q_batch[bi]
                best_sim, _ = nearest_cosine(
                    qf,
                    bank,
                    chunk=int(nn_chunk),
                )
                patch_scores = 1.0 - best_sim
                kk = min(int(sensor_k), int(patch_scores.numel()))
                top_idx = torch.topk(
                    patch_scores,
                    k=kk,
                    largest=True,
                ).indices

                mask: Optional[torch.Tensor] = None
                if rec.mask_path and Path(rec.mask_path).is_file():
                    try:
                        mask = mask_to_patch_mask(rec.mask_path, cfg)
                    except Exception:
                        mask = None
                has_oracle = bool(mask is not None and int(mask.sum().item()) > 0)

                for name, sidecar in sidecars.items():
                    z_patch = sidecar.transform(qf, device)
                    svec = _pool_sensor_address(
                        z_patch,
                        patch_scores,
                        top_idx,
                        float(temperature),
                    )
                    sensor_store[name].append(
                        svec.detach().cpu().to(torch.float32)
                    )
                    ovec = _pool_oracle_address(z_patch, mask)
                    if ovec is None:
                        oracle_store[name].append(
                            torch.full(
                                (sidecar.output_dim,),
                                float("nan"),
                                dtype=torch.float32,
                            )
                        )
                    else:
                        oracle_store[name].append(
                            ovec.detach().cpu().to(torch.float32)
                        )

                categories.append(str(rec.category))
                labels.append(str(rec.defect_source))
                image_ids.append(str(rec.image_id))
                oracle_valid.append(has_oracle)

        del bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    evidence: Dict[str, Dict[str, np.ndarray]] = {}
    for name in sidecars:
        evidence[name] = {
            "sensor": torch.stack(sensor_store[name], dim=0).numpy(),
            "oracle": torch.stack(oracle_store[name], dim=0).numpy(),
        }

    return (
        evidence,
        np.asarray(categories, dtype=object),
        np.asarray(labels, dtype=object),
        image_ids,
        {
            "oracle_valid": np.asarray(oracle_valid, dtype=bool),
            "normal_support": support_meta,
        },
    )


def run_cv_fold(
    args: argparse.Namespace,
    *,
    source_cv_fold: int,
    protocol: Any,
    records: Sequence[Stage0Record],
    defects: Sequence[Stage0Record],
    candidate_names: Sequence[str],
    sensor: PaperAlignedFrozenSensor,
    cfg: Stage0Config,
    sidecar_cfg: SidecarTrainConfig,
    support_seeds: Sequence[int],
    out_dir: Path,
) -> Dict[str, Any]:
    test_categories = tuple(protocol.source_cv_groups[int(source_cv_fold)])
    train_categories = tuple(
        c for c in protocol.source_categories if c not in set(test_categories)
    )
    print(
        f"\n=== E2 source-CV fold {source_cv_fold}: "
        f"train={len(train_categories)} products, test={len(test_categories)} ==="
    )

    train_x, val_x, normal_meta = collect_normal_patch_dataset(
        sensor,
        records,
        train_categories,
        patches_per_image=int(args.patches_per_image),
        val_fraction=float(args.val_fraction),
        max_train_images_per_category=int(args.max_train_images_per_category),
        max_val_images_per_category=int(args.max_val_images_per_category),
        seed=int(args.model_seed) + 10_000 * int(source_cv_fold),
        batch_size=int(args.query_batch_size),
    )
    print(
        f"  normal sidecar data: train={tuple(train_x.shape)}, "
        f"val={tuple(val_x.shape)}"
    )

    fold_cfg = SidecarTrainConfig(
        address_dim=int(args.address_dim),
        hidden_dim=int(args.hidden_dim),
        bottleneck_dim=int(args.bottleneck_dim),
        residual_dim=int(args.residual_dim),
        corr_groups=int(args.corr_groups),
        batch_size=int(args.sidecar_batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        vae_kl_weight=float(args.vae_kl_weight),
        beta_tc_weight=float(args.beta_tc_weight),
        factor_tc_weight=float(args.factor_tc_weight),
        factor_disc_lr=float(args.factor_disc_lr),
        corr_cross_weight=float(args.corr_cross_weight),
        corr_group_sparse_weight=float(args.corr_group_sparse_weight),
        address_aux_weight=float(args.address_aux_weight),
        address_l1_weight=float(args.address_l1_weight),
        sparse_l1_weight=float(args.sparse_l1_weight),
        grad_clip=float(args.grad_clip),
        seed=int(args.model_seed) + 1000 * int(source_cv_fold),
    )
    sidecars, training = _fit_candidates(
        candidate_names,
        train_x,
        val_x,
        fold_cfg,
        sensor.device,
    )

    evidence, categories, labels, image_ids, extract_meta = (
        _extract_candidate_evidence(
            sensor,
            sidecars,
            records,
            defects,
            cfg,
            sensor_k=int(args.sensor_k),
            temperature=float(args.score_temperature),
            query_batch_size=int(args.query_batch_size),
            nn_chunk=int(args.nn_chunk),
            device=sensor.device,
        )
    )

    fold_result: Dict[str, Any] = {
        "schema": "abmg.e2.source_cv_fold.v1",
        "source_cv_fold": int(source_cv_fold),
        "train_categories": list(train_categories),
        "test_categories": list(test_categories),
        "sidecar_training_data": normal_meta,
        "training": training,
        "candidates": {},
    }

    probe_records = _select_spatial_probes(
        defects,
        test_categories,
        int(args.spatial_probe_count),
        int(args.seed) + int(source_cv_fold),
    )

    oracle_valid = np.asarray(extract_meta["oracle_valid"], dtype=bool)
    wanted = set(CORE4)
    test_core_mask = np.asarray(
        [
            (c in set(test_categories)) and (y in wanted)
            for c, y in zip(categories, labels)
        ],
        dtype=bool,
    )

    for name in candidate_names:
        result: Dict[str, Any] = {
            "output_dim": int(sidecars[name].output_dim),
            "sidecar_diagnostics": sidecars[name].diagnostics(),
            "sensor": {},
            "oracle": {},
        }
        first_sensor_memory = None
        first_sensor_prior = None

        sensor_runs = []
        for si, support_seed in enumerate(support_seeds):
            run, memory, prior = _evaluate_routing_once(
                evidence[name]["sensor"],
                categories,
                labels,
                test_categories,
                support_seed=int(support_seed) + 100_000 * int(source_cv_fold),
                shots=int(args.routing_shots),
                kappa0=float(args.kappa0),
            )
            sensor_runs.append(run)
            if si == 0:
                first_sensor_memory = memory
                first_sensor_prior = prior
        result["sensor"]["support_seed_runs"] = sensor_runs
        result["sensor"]["aggregate"] = _aggregate_seed_runs(sensor_runs)
        result["sensor"]["address_efficiency"] = _address_efficiency(
            evidence[name]["sensor"][test_core_mask],
            name,
            int(args.corr_groups),
        )

        # Oracle uses only rows whose transformed GT mask survived geometry.
        valid_oracle = oracle_valid & np.all(
            np.isfinite(evidence[name]["oracle"]),
            axis=1,
        )
        oracle_runs = []
        for support_seed in support_seeds:
            run, _, _ = _evaluate_routing_once(
                evidence[name]["oracle"][valid_oracle],
                categories[valid_oracle],
                labels[valid_oracle],
                test_categories,
                support_seed=int(support_seed) + 100_000 * int(source_cv_fold),
                shots=int(args.routing_shots),
                kappa0=float(args.kappa0),
            )
            oracle_runs.append(run)
        result["oracle"]["support_seed_runs"] = oracle_runs
        result["oracle"]["aggregate"] = _aggregate_seed_runs(oracle_runs)

        if (
            int(args.spatial_probe_count) > 0
            and first_sensor_memory is not None
            and first_sensor_prior is not None
        ):
            result["spatial_consistency"] = _spatial_consistency(
                sensor,
                sidecars[name],
                probe_records,
                first_sensor_memory,
                first_sensor_prior,
                cfg,
                kappa0=float(args.kappa0),
                device=sensor.device,
            )

        fold_result["candidates"][name] = result

    fold_dir = out_dir / f"cv_{source_cv_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(fold_result, fold_dir / "e2_cv_summary.json")
    _json_dump(
        {
            "schema": "abmg.e2.cv_manifest.v1",
            "source_cv_fold": int(source_cv_fold),
            "image_ids": image_ids,
            "categories": categories.tolist(),
            "defect_sources_offline_only": labels.tolist(),
            "oracle_valid": oracle_valid.tolist(),
        },
        fold_dir / "e2_eval_manifest.json",
    )
    print(f"Wrote {fold_dir / 'e2_cv_summary.json'}")
    return fold_result


def aggregate_cv_results(
    cv_results: Sequence[Dict[str, Any]],
    candidate_names: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in candidate_names:
        sensor_fold = []
        oracle_fold = []
        signature_fold = []
        spatial_auc = []
        for cv in cv_results:
            c = cv["candidates"][name]
            sensor_fold.append(float(c["sensor"]["aggregate"]["macro_f1_mean"]))
            oracle_fold.append(float(c["oracle"]["aggregate"]["macro_f1_mean"]))
            signature_fold.append(
                float(c["sensor"]["aggregate"]["source_signature_gap_mean"])
            )
            sp = c.get("spatial_consistency", {})
            if np.isfinite(float(sp.get("true_factor_patch_auroc_mean", float("nan")))):
                spatial_auc.append(float(sp["true_factor_patch_auroc_mean"]))

        out[name] = {
            "n_cv_folds": len(cv_results),
            "sensor_macro_f1_mean": float(np.mean(sensor_fold)),
            "sensor_macro_f1_std_across_cv": float(np.std(sensor_fold)),
            "sensor_macro_f1_by_cv": sensor_fold,
            "oracle_macro_f1_mean": float(np.mean(oracle_fold)),
            "oracle_macro_f1_std_across_cv": float(np.std(oracle_fold)),
            "oracle_macro_f1_by_cv": oracle_fold,
            "source_signature_gap_mean": float(np.mean(signature_fold)),
            "spatial_true_factor_patch_auroc_mean": (
                float(np.mean(spatial_auc)) if spatial_auc else float("nan")
            ),
        }

    raw = out.get("raw")
    if raw is not None:
        for name, r in out.items():
            delta_sensor = float(
                r["sensor_macro_f1_mean"] - raw["sensor_macro_f1_mean"]
            )
            oracle_drop = float(
                r["oracle_macro_f1_mean"] - raw["oracle_macro_f1_mean"]
            )
            positive_folds = int(
                sum(
                    cand > base
                    for cand, base in zip(
                        r["sensor_macro_f1_by_cv"],
                        raw["sensor_macro_f1_by_cv"],
                    )
                )
            )
            r["vs_raw"] = {
                "sensor_macro_f1_delta": delta_sensor,
                "oracle_macro_f1_delta": oracle_drop,
                "positive_sensor_cv_folds": positive_folds,
            }
            if len(cv_results) == 4:
                r["primary_gate"] = {
                    "delta_sensor_f1_at_least_0p02": bool(
                        delta_sensor >= PRIMARY_SUCCESS_DELTA_F1
                    ),
                    "positive_in_at_least_3_of_4_cv_folds": bool(
                        positive_folds >= 3
                    ),
                    "oracle_f1_drop_no_worse_than_0p02": bool(
                        oracle_drop >= -PRIMARY_ORACLE_DROP_TOLERANCE
                    ),
                    "pass": bool(
                        delta_sensor >= PRIMARY_SUCCESS_DELTA_F1
                        and positive_folds >= 3
                        and oracle_drop >= -PRIMARY_ORACLE_DROP_TOLERANCE
                    ),
                }
            else:
                r["primary_gate"] = {
                    "pass": None,
                    "reason": "requires all 4 source-CV folds",
                }
    return out


def run(args: argparse.Namespace) -> int:
    if int(args.sensor_k) != PRIMARY_SENSOR_K:
        raise ValueError("Primary E2 fixes Sensor top-k at 8")
    if float(args.score_temperature) != PRIMARY_SCORE_TEMPERATURE:
        raise ValueError("Primary E2 fixes score-softmax temperature at 20")
    if int(args.routing_shots) != PRIMARY_ROUTING_SHOTS:
        raise ValueError("Primary E2 screening fixes routing supports at 8/factor")
    if float(args.kappa0) != PRIMARY_KAPPA0:
        raise ValueError("Primary E2 fixes kappa0=1")

    candidate_names = parse_csv(args.candidates)
    unknown = sorted(set(candidate_names) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"Unknown candidates={unknown}; allowed={CANDIDATES}")
    if "raw" not in candidate_names:
        raise ValueError(
            "Every E2 comparison must include raw as the mandatory baseline"
        )

    protocol = load_fold_protocol(args.fold, args.folds)
    cv_folds = parse_cv_folds(
        args.source_cv_folds,
        len(protocol.source_cv_groups),
    )
    support_seeds = parse_int_csv(args.support_seeds)

    # HARD BOUNDARY: load only outer-fold source category JSONs.
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
        raise RuntimeError("No source defect images found")

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
        support_seed=int(args.normal_support_seed),
        cache_dtype="float16" if not args.no_fp16 else "float32",
    )
    set_seed(int(args.seed))

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar_cfg = SidecarTrainConfig(
        address_dim=int(args.address_dim),
        hidden_dim=int(args.hidden_dim),
        bottleneck_dim=int(args.bottleneck_dim),
        residual_dim=int(args.residual_dim),
        corr_groups=int(args.corr_groups),
        batch_size=int(args.sidecar_batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        vae_kl_weight=float(args.vae_kl_weight),
        beta_tc_weight=float(args.beta_tc_weight),
        factor_tc_weight=float(args.factor_tc_weight),
        factor_disc_lr=float(args.factor_disc_lr),
        corr_cross_weight=float(args.corr_cross_weight),
        corr_group_sparse_weight=float(args.corr_group_sparse_weight),
        address_aux_weight=float(args.address_aux_weight),
        address_l1_weight=float(args.address_l1_weight),
        sparse_l1_weight=float(args.sparse_l1_weight),
        grad_clip=float(args.grad_clip),
        seed=int(args.model_seed),
    )
    sidecar_cfg.validate()

    resolved = {
        "schema": "abmg.e2.config.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "source_categories": list(protocol.source_categories),
        "source_cv_folds": list(cv_folds),
        "candidates": list(candidate_names),
        "support_seeds": list(support_seeds),
        "max_per_type_per_category": int(args.max_per_type_per_category),
        "n_sampled_source_defects": int(len(defects)),
        "success_gate": {
            "sensor_macro_f1_delta_vs_raw": PRIMARY_SUCCESS_DELTA_F1,
            "positive_cv_folds_required": 3,
            "oracle_drop_tolerance": PRIMARY_ORACLE_DROP_TOLERANCE,
        },
        "sensor": asdict(cfg),
        "sensor_fingerprint": config_fingerprint(cfg),
        "sidecar": sidecar_config_dict(sidecar_cfg),
        "dual_path_invariant": (
            "raw DINO determines anomaly score/top8/weights; sidecar only "
            "transforms the same patch h for address routing"
        ),
    }
    _json_dump(resolved, out_dir / "resolved_config.json")

    sensor = PaperAlignedFrozenSensor(
        cfg,
        device=args.device,
        use_fp16=not args.no_fp16,
    )
    cv_results: List[Dict[str, Any]] = []
    try:
        for cv_fold in cv_folds:
            cv_results.append(
                run_cv_fold(
                    args,
                    source_cv_fold=int(cv_fold),
                    protocol=protocol,
                    records=records,
                    defects=defects,
                    candidate_names=candidate_names,
                    sensor=sensor,
                    cfg=cfg,
                    sidecar_cfg=sidecar_cfg,
                    support_seeds=support_seeds,
                    out_dir=out_dir,
                )
            )
    finally:
        sensor.cleanup()

    aggregate = aggregate_cv_results(cv_results, candidate_names)
    summary = {
        "schema": "abmg.e2.address_sidecar_benchmark.summary.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(protocol.target_categories),
        "n_cv_folds_completed": len(cv_results),
        "candidates": aggregate,
        "primary_question": (
            "Does an optional address representation improve held-out-product "
            "Core-4 routing while the raw DINO detector remains unchanged?"
        ),
    }
    _json_dump(summary, out_dir / "e2_summary.json")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"Wrote E2 summary to: {out_dir / 'e2_summary.json'}")
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
        default="outputs/e2_address_sidecar",
    )
    p.add_argument(
        "--candidates",
        type=str,
        default="raw,pca,ica",
    )
    p.add_argument(
        "--source-cv-folds",
        type=str,
        default="all",
        help="'all' or comma-separated source-CV fold indices 0..3.",
    )
    p.add_argument(
        "--support-seeds",
        type=str,
        default="0",
        help="Screening: 0. Finalists can use 0,1,2,3,4.",
    )

    # Frozen sensor.
    p.add_argument("--model-name", type=str, default="dinov2_vitl14_reg")
    p.add_argument("--layers", type=str, default="4-18")
    p.add_argument("--resize-size", type=int, default=448)
    p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--normal-support-seed", type=int, default=0)
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

    # Screening data volume.
    p.add_argument(
        "--max-per-type-per-category",
        type=int,
        default=64,
        help="Screening default=64. Use 0 only for finalist/full confirmation.",
    )
    p.add_argument("--patches-per-image", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-train-images-per-category", type=int, default=32)
    p.add_argument("--max-val-images-per-category", type=int, default=8)

    # Fixed factor router.
    p.add_argument("--routing-shots", type=int, default=PRIMARY_ROUTING_SHOTS)
    p.add_argument("--kappa0", type=float, default=PRIMARY_KAPPA0)
    p.add_argument("--spatial-probe-count", type=int, default=128)

    # Shared sidecar architecture/training.
    p.add_argument("--address-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--bottleneck-dim", type=int, default=256)
    p.add_argument("--residual-dim", type=int, default=64)
    p.add_argument("--corr-groups", type=int, default=8)
    p.add_argument("--sidecar-batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)

    # Fixed regularization coefficients: no held-out-label tuning in E2.
    p.add_argument("--vae-kl-weight", type=float, default=1e-3)
    p.add_argument("--beta-tc-weight", type=float, default=5e-3)
    p.add_argument("--factor-tc-weight", type=float, default=5e-3)
    p.add_argument("--factor-disc-lr", type=float, default=1e-3)
    p.add_argument("--corr-cross-weight", type=float, default=5e-3)
    p.add_argument("--corr-group-sparse-weight", type=float, default=1e-3)
    p.add_argument("--address-aux-weight", type=float, default=0.25)
    p.add_argument("--address-l1-weight", type=float, default=1e-4)
    p.add_argument("--sparse-l1-weight", type=float, default=1e-3)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-seed", type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
