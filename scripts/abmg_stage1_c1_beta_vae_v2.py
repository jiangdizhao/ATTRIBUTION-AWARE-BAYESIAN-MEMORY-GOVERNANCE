#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ABMG Stage-1/Stage-2 C1 experiment wrapper with revised dynamic scale audit.

The original comprehensive driver remains the implementation of train,
retention, addressability, compare, and aggregate. This wrapper supersedes only
the old one-shot ``scale-audit`` gate. The revised audit trains small paired
raw/scalar calibration models under the same beta warm-up used by the primary
experiment and measures the objective/gradient trajectory after the model has
started adapting.

Why v2 exists
-------------
The first audit compared reconstruction and beta*KL encoder gradients at random
initialisation with beta already fixed at 4. That can reject a representation
for a condition that never occurs in the actual run: the primary beta-VAE uses
beta warm-up from zero. V2 therefore makes the audit a short source-only
training-dynamics experiment and deliberately avoids an arbitrary pass/fail
ratio band.

For every command other than ``scale-audit``, arguments are delegated unchanged
to ``abmg_stage1_c1_beta_vae.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import abmg_stage1_c1_beta_vae as core


SCHEMA = "abmg.stage1_c1.v2.dynamic_scale_audit"
EPS = 1e-12


def _parse_modes(spec: str) -> List[str]:
    modes = [x.strip().lower() for x in str(spec).split(",") if x.strip()]
    if not modes:
        raise ValueError("--modes must contain raw and/or scalar")
    bad = [x for x in modes if x not in {"raw", "scalar"}]
    if bad:
        raise ValueError(f"Unsupported audit mode(s): {bad}")
    return list(dict.fromkeys(modes))


def _quota(total: int, categories: Sequence[str]) -> Dict[str, int]:
    """Balanced per-category image quotas whose sum is exactly ``total``."""
    cats = list(categories)
    if total <= 0:
        raise ValueError("Requested image count must be positive")
    base, rem = divmod(int(total), len(cats))
    return {c: base + (1 if i < rem else 0) for i, c in enumerate(cats)}


def _balanced_subset(
    cache: core.CacheInfo,
    source: Sequence[str],
    split: str,
    fold: int,
    val_fraction: float,
    total_images: int,
    seed: int,
) -> List[Tuple[str, str, torch.Tensor]]:
    """Pick a deterministic category-balanced image subset without patch leakage."""
    source_sorted = sorted(source)
    quota = _quota(total_images, source_sorted)
    chosen: Dict[str, List[Tuple[float, str, str, torch.Tensor]]] = {
        c: [] for c in source_sorted
    }

    for image_id, cat, feats in core.iter_cache_images(
        cache, set(source_sorted), split, fold, val_fraction
    ):
        q = quota[cat]
        if q <= 0:
            continue
        score = core.stable_fraction(
            f"stage1-v2-audit|seed={seed}|fold={fold}|split={split}|{image_id}"
        )
        bucket = chosen[cat]
        bucket.append((score, image_id, cat, feats))
        if len(bucket) > q:
            bucket.sort(key=lambda x: (x[0], x[1]))
            del bucket[q:]

    out: List[Tuple[str, str, torch.Tensor]] = []
    shortages: Dict[str, Dict[str, int]] = {}
    for cat in source_sorted:
        bucket = sorted(chosen[cat], key=lambda x: (x[0], x[1]))
        if len(bucket) < quota[cat]:
            shortages[cat] = {"requested": quota[cat], "available": len(bucket)}
        out.extend((image_id, cat, feats) for _, image_id, cat, feats in bucket)
    if shortages:
        raise RuntimeError(f"Not enough {split} images for balanced audit subset: {shortages}")

    out.sort(key=lambda x: (x[1], x[0]))
    return out


def _subset_stats(items: Sequence[Tuple[str, str, torch.Tensor]]) -> Dict[str, Any]:
    by_cat: Dict[str, int] = {}
    patch_counts = []
    for _, cat, feats in items:
        by_cat[cat] = by_cat.get(cat, 0) + 1
        patch_counts.append(int(feats.shape[0]))
    return {
        "n_images": len(items),
        "n_patches": int(sum(patch_counts)),
        "by_category": dict(sorted(by_cat.items())),
        "image_is_minimum_split_unit": True,
    }


def _image_batches(
    items: Sequence[Tuple[str, str, torch.Tensor]],
    image_batch_size: int,
    seed: int,
    shuffle: bool,
):
    idx = list(range(len(items)))
    if shuffle:
        random.Random(int(seed)).shuffle(idx)
    for start in range(0, len(idx), int(image_batch_size)):
        yield [items[i] for i in idx[start : start + int(image_batch_size)]]


def _to_patch_batch(
    batch: Sequence[Tuple[str, str, torch.Tensor]],
    device: torch.device,
    feature_dim: int,
) -> torch.Tensor:
    return (
        torch.stack([x[2] for x in batch], dim=0)
        .float()
        .to(device)
        .reshape(-1, feature_dim)
    )


def _encoder_parameters(model: core.FeatureVAE) -> Iterable[torch.nn.Parameter]:
    yield from model.encoder.parameters()
    yield from model.mu.parameters()
    yield from model.logvar.parameters()


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def _participation_ratio(values: Sequence[float]) -> float:
    """Continuous effective-count diagnostic: (sum x)^2 / sum x^2."""
    x = np.asarray(values, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    denom = float(np.square(x).sum())
    if denom <= EPS:
        return 0.0
    return float(np.square(x.sum()) / denom)


@torch.no_grad()
def _evaluate_subset(
    model: core.FeatureVAE,
    norm: core.Normalizer,
    items: Sequence[Tuple[str, str, torch.Tensor]],
    image_batch_size: int,
    device: torch.device,
    beta_target: float,
) -> Dict[str, Any]:
    model.eval()
    n_patch = 0
    rec_sum = 0.0
    kl_sum = 0.0
    cos_sum = 0.0
    kl_coord_sum = torch.zeros(model.latent_dim, dtype=torch.float64)
    mu_sum = torch.zeros(model.latent_dim, dtype=torch.float64)
    mu_sq_sum = torch.zeros(model.latent_dim, dtype=torch.float64)

    for batch in _image_batches(items, image_batch_size, seed=0, shuffle=False):
        x_raw = _to_patch_batch(batch, device, model.input_dim)
        x = norm.transform(x_raw)
        mu, logvar = model.encode(x)
        recon = model.decode(mu)
        rec, kl, kl_coord = core.vae_losses(x, recon, mu, logvar)
        recon_raw = norm.inverse(recon)
        cos = F.cosine_similarity(x_raw, recon_raw, dim=1).mean()

        n = int(x.shape[0])
        n_patch += n
        rec_sum += float(rec.item()) * n
        kl_sum += float(kl.item()) * n
        cos_sum += float(cos.item()) * n
        kl_coord_sum += kl_coord.detach().cpu().double() * n
        mu_cpu = mu.detach().cpu().double()
        mu_sum += mu_cpu.sum(0)
        mu_sq_sum += (mu_cpu * mu_cpu).sum(0)

    if n_patch == 0:
        raise RuntimeError("Empty validation subset")

    mean_mu = mu_sum / n_patch
    var_mu = (mu_sq_sum / n_patch - mean_mu * mean_mu).clamp_min(0.0)
    kl_coord = kl_coord_sum / n_patch
    rec_mean = rec_sum / n_patch
    kl_mean = kl_sum / n_patch
    return {
        "n_patches": n_patch,
        "rec_loss": rec_mean,
        "kl_loss": kl_mean,
        "target_beta_kl_loss": float(beta_target) * kl_mean,
        "target_betaKL_over_rec": float(beta_target) * kl_mean / max(rec_mean, EPS),
        "recon_cosine_original_space": cos_sum / n_patch,
        "per_coordinate_kl": kl_coord.float().tolist(),
        "per_coordinate_var_mu": var_mu.float().tolist(),
        "kl_participation_ratio": _participation_ratio(kl_coord.tolist()),
        "var_mu_participation_ratio": _participation_ratio(var_mu.tolist()),
        "sum_var_mu": float(var_mu.sum().item()),
    }


def _component_gradient_probe(
    model: core.FeatureVAE,
    norm: core.Normalizer,
    batch: Sequence[Tuple[str, str, torch.Tensor]],
    device: torch.device,
    beta_effective: float,
    noise_seed: int,
) -> Dict[str, float]:
    model.train()
    x_raw = _to_patch_batch(batch, device, model.input_dim)
    x = norm.transform(x_raw)

    torch.manual_seed(int(noise_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(noise_seed))
    recon, mu, logvar = model(x)
    rec, kl, _ = core.vae_losses(x, recon, mu, logvar)

    model.zero_grad(set_to_none=True)
    rec.backward(retain_graph=True)
    grad_rec = _grad_norm(_encoder_parameters(model))

    model.zero_grad(set_to_none=True)
    if beta_effective > 0.0:
        (float(beta_effective) * kl).backward()
        grad_beta_kl = _grad_norm(_encoder_parameters(model))
    else:
        grad_beta_kl = 0.0
    model.zero_grad(set_to_none=True)

    return {
        "probe_rec_loss": float(rec.detach().item()),
        "probe_kl_loss": float(kl.detach().item()),
        "probe_beta_kl_loss": float(beta_effective * kl.detach().item()),
        "probe_betaKL_over_rec": float(
            beta_effective * kl.detach().item() / max(float(rec.detach().item()), EPS)
        ),
        "encoder_grad_rec": grad_rec,
        "encoder_grad_betaKL": grad_beta_kl,
        "encoder_grad_ratio_betaKL_over_rec": grad_beta_kl / max(grad_rec, EPS),
    }


def _build_normalizer(
    mode: str,
    cache: core.CacheInfo,
    source: Sequence[str],
    fold: int,
    val_fraction: float,
) -> core.Normalizer:
    if mode == "raw":
        return core.Normalizer("raw", torch.zeros(cache.feature_dim), 1.0)
    return core.estimate_scalar_normalizer(
        cache, set(source), fold, val_fraction, max_images=0
    )


def _run_condition(
    mode: str,
    init_state: Dict[str, torch.Tensor],
    cache: core.CacheInfo,
    source: Sequence[str],
    train_items: Sequence[Tuple[str, str, torch.Tensor]],
    val_items: Sequence[Tuple[str, str, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    core.set_seed(args.seed)
    norm = _build_normalizer(mode, cache, source, args.fold, args.val_fraction)
    model = core.FeatureVAE(cache.feature_dim, args.latent_dim).to(device)
    model.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    probe_batch = next(
        _image_batches(
            val_items,
            min(args.probe_images, len(val_items)),
            seed=args.seed,
            shuffle=False,
        )
    )

    trajectory: List[Dict[str, Any]] = []

    initial_val = _evaluate_subset(
        model, norm, val_items, args.image_batch_size, device, args.beta
    )
    initial_probe = _component_gradient_probe(
        model, norm, probe_batch, device, 0.0, args.seed + 900000
    )
    trajectory.append(
        {
            "epoch": -1,
            "phase": "initial_untrained",
            "beta_effective": 0.0,
            **{f"val_{k}": v for k, v in initial_val.items() if not k.startswith("per_coordinate_")},
            **initial_probe,
        }
    )

    for epoch in range(args.epochs):
        model.train()
        beta_eff = core.beta_at_epoch(args.beta, epoch, args.beta_warmup_epochs)
        train_rec_sum = 0.0
        train_kl_sum = 0.0
        train_total_sum = 0.0
        n_patch = 0

        for step, batch in enumerate(
            _image_batches(
                train_items,
                args.image_batch_size,
                seed=args.seed * 100000 + epoch,
                shuffle=True,
            )
        ):
            x_raw = _to_patch_batch(batch, device, cache.feature_dim)
            x = norm.transform(x_raw)
            optimizer.zero_grad(set_to_none=True)
            noise_seed = args.seed * 10000000 + epoch * 10000 + step
            torch.manual_seed(int(noise_seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(noise_seed))
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                recon, mu, logvar = model(x)
                rec, kl, _ = core.vae_losses(x, recon, mu, logvar)
                loss = rec + beta_eff * kl
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            n = int(x.shape[0])
            n_patch += n
            train_rec_sum += float(rec.detach().item()) * n
            train_kl_sum += float(kl.detach().item()) * n
            train_total_sum += float(loss.detach().item()) * n

        val = _evaluate_subset(
            model, norm, val_items, args.image_batch_size, device, args.beta
        )
        probe = _component_gradient_probe(
            model,
            norm,
            probe_batch,
            device,
            beta_eff,
            args.seed + 900001 + epoch,
        )
        row = {
            "epoch": epoch,
            "phase": "calibration_training",
            "beta_effective": float(beta_eff),
            "train_rec": train_rec_sum / max(n_patch, 1),
            "train_kl": train_kl_sum / max(n_patch, 1),
            "train_beta_kl": float(beta_eff) * train_kl_sum / max(n_patch, 1),
            "train_total": train_total_sum / max(n_patch, 1),
            **{f"val_{k}": v for k, v in val.items() if not k.startswith("per_coordinate_")},
            **probe,
        }
        trajectory.append(row)
        print(json.dumps({"mode": mode, **row}))

    final_val = _evaluate_subset(
        model, norm, val_items, args.image_batch_size, device, args.beta
    )
    final_beta_eff = core.beta_at_epoch(
        args.beta, args.epochs - 1, args.beta_warmup_epochs
    )
    finite_keys = [
        "rec_loss",
        "kl_loss",
        "target_betaKL_over_rec",
        "recon_cosine_original_space",
        "kl_participation_ratio",
        "var_mu_participation_ratio",
        "sum_var_mu",
    ]
    finite = all(math.isfinite(float(final_val[k])) for k in finite_keys)

    return {
        "mode": mode,
        "normalization_scalar": float(norm.scalar),
        "epochs": int(args.epochs),
        "beta_target": float(args.beta),
        "beta_warmup_epochs": int(args.beta_warmup_epochs),
        "final_beta_effective": float(final_beta_eff),
        "reached_target_beta": bool(abs(final_beta_eff - float(args.beta)) < 1e-12),
        "all_key_final_metrics_finite": bool(finite),
        "final_validation": final_val,
        "trajectory": trajectory,
        "interpretation": {
            "gradient_ratio": (
                "Descriptive only. It is measured after each calibration epoch under the actual "
                "beta warm-up; there is no arbitrary 0.1-10 pass/fail band."
            ),
            "participation_ratio": (
                "Continuous effective-count diagnostic in [0, latent_dim]. Values near 1 mean "
                "information/variation is highly concentrated; values spread toward latent_dim "
                "mean broader latent usage. It is not a semantic disentanglement score."
            ),
            "collapse": (
                "Inspect total KL, sum Var(mu), per-coordinate KL/Var(mu), and participation ratios "
                "jointly. The script intentionally does not invent a single collapse threshold."
            ),
        },
    }


def run_dynamic_scale_audit(args: argparse.Namespace) -> int:
    folds = core.load_folds(Path(args.folds))
    source, target = core.fold_categories(folds, args.fold)
    cache = core.inspect_cache(Path(args.cache_dir))
    device = core.device_of(args.device)
    modes = _parse_modes(args.modes)
    core.set_seed(args.seed)

    train_items = _balanced_subset(
        cache,
        source,
        "train",
        args.fold,
        args.val_fraction,
        args.train_images,
        args.seed,
    )
    val_items = _balanced_subset(
        cache,
        source,
        "val",
        args.fold,
        args.val_fraction,
        args.val_images,
        args.seed + 1,
    )

    base = core.FeatureVAE(cache.feature_dim, args.latent_dim)
    init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    conditions = []
    for mode in modes:
        print(f"=== dynamic scale audit: fold={args.fold} mode={mode} ===")
        conditions.append(
            _run_condition(
                mode,
                init_state,
                cache,
                source,
                train_items,
                val_items,
                args,
                device,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema": SCHEMA,
        "fold": int(args.fold),
        "source_categories": source,
        "target_categories_hidden_from_audit": target,
        "cache_fingerprint": cache.fingerprint,
        "feature_dim": cache.feature_dim,
        "patches_per_image": cache.patches_per_image,
        "latent_dim": int(args.latent_dim),
        "image_level_protocol": (
            "Train/validation membership and audit subsampling are image-level. Complete cached "
            "patch sets remain together; flattening occurs only inside an optimizer batch."
        ),
        "full_split_stats": core.manifest_split_stats(
            cache, set(source), args.fold, args.val_fraction
        ),
        "calibration_train_subset": _subset_stats(train_items),
        "calibration_val_subset": _subset_stats(val_items),
        "paired_design": {
            "same_initial_weights": True,
            "same_train_images": True,
            "same_validation_images": True,
            "same_epoch_image_order": True,
            "same_stochastic_latent_seeds_per_step": True,
            "only_condition_difference": "input normalization mode",
        },
        "objective": "0.5*sum_feature_squared_error + beta_effective*KL",
        "beta_target": float(args.beta),
        "beta_warmup_epochs": int(args.beta_warmup_epochs),
        "epochs": int(args.epochs),
        "conditions": conditions,
        "decision_policy": {
            "automatic_pass_fail": False,
            "default_preference": "raw, because it changes the frozen DINO geometry least",
            "how_to_choose": (
                "Inspect the warm-up trajectory rather than the random-initialization gradient ratio. "
                "Reject a condition for genuine instability/non-finite values, failure to reconstruct, "
                "or clear posterior collapse. If raw is healthy, prefer raw. Scalar normalization should "
                "only be adopted if the dynamic evidence shows a material optimization advantage without "
                "worse reconstruction or latent collapse."
            ),
            "target_data_used_for_choice": False,
        },
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"fold_{args.fold}_scale_audit_v2.json"
    csv_path = out / f"fold_{args.fold}_scale_audit_v2_trajectory.csv"
    core.json_dump(report, json_path)

    csv_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        for row in condition["trajectory"]:
            csv_rows.append(
                {
                    "fold": args.fold,
                    "mode": condition["mode"],
                    "normalization_scalar": condition["normalization_scalar"],
                    **row,
                }
            )
    core.write_csv(csv_rows, csv_path)

    print(json.dumps(report, indent=2))
    print(f"wrote: {json_path}")
    print(f"wrote: {csv_path}")
    return 0


def build_scale_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ABMG revised source-only dynamic beta-VAE scale audit"
    )
    p.add_argument("scale-audit", nargs="?")
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--folds", default="configs/realiad_folds_v0.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--modes", default="raw,scalar", help="raw, scalar, or raw,scalar")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--beta-warmup-epochs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--train-images", type=int, default=1024)
    p.add_argument("--val-images", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--image-batch-size", type=int, default=32)
    p.add_argument("--probe-images", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-dir", default="outputs/stage1_scale_audit_v2")
    return p


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "scale-audit":
        args = build_scale_parser().parse_args()
        return run_dynamic_scale_audit(args)
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
