#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1-2 / C1 experiment for Attribution-Aware Bayesian Memory Governance.

This script implements the complete *factor-addressability* evidence path while
keeping later memory/query claims out of scope:

1) source-only reconstruction/KL scale audit (raw DINO vs optional scalar-normalised input),
2) fold-specific feature-space VAE / beta-VAE training with beta warm-up,
3) latent-health diagnostics (per-coordinate KL and posterior-mean variance),
4) held-out target anomaly-ranking retention (R0 vs reconstructed R1/R_beta),
5) offline C1 addressability diagnostics: source separability, responsibility
   stability, spatial consistency, coordinate baseline, and random-group control,
6) a compact comparison report.

Hard protocol rules
-------------------
* Five fixed Real-IAD category folds; each category is target exactly once.
* Factorizer training sees SOURCE-CATEGORY NORMAL TRAIN features only.
* Train/validation split is IMAGE-LEVEL: all 64 cached patches from one image
  stay together. Patches are flattened only inside an optimizer batch.
* DINO and the trained factorizer are frozen during target evaluation.
* Hidden target labels / defect-source labels / masks are used only by the
  evaluator after responsibilities/scores are computed.
* C1 uses a simple fixed Gaussian normal reference from the four target normal
  supports. NIG Bayesian memory is intentionally deferred to later claims.
* No automatic C1 pass/fail threshold is invented. The report exposes paired
  utility and addressability deltas for scientific interpretation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from abmg_stage0_foundation import (
    PaperAlignedFrozenSensor,
    Stage0Config,
    Stage0Record,
    config_fingerprint,
    load_realiad_records,
    paper_support_variants,
    select_supports,
    set_seed,
)
from vmb_visionad_new import patch_nn_anomaly_map, resize_and_center_crop


SCHEMA = "abmg.stage1_c1.v1"
EPS = 1e-8


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def chunks(xs: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def stable_fraction(key: str) -> float:
    v = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:15], 16)
    return v / float(16**15 - 1)


def safe_auc(y: Sequence[int], score: Sequence[float]) -> float:
    if len(set(int(x) for x in y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y: Sequence[int], score: Sequence[float]) -> float:
    if len(set(int(x) for x in y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def device_of(spec: str) -> torch.device:
    d = torch.device(spec)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return d


def git_sha_or_unknown() -> str:
    return os.environ.get("GIT_COMMIT", os.environ.get("GITHUB_SHA", "unknown"))


def torch_load(path: Any, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -----------------------------------------------------------------------------
# Fold protocol
# -----------------------------------------------------------------------------


def load_folds(path: Path) -> Dict[str, Any]:
    data = json_load(path)
    all_categories = list(data["all_categories"])
    folds = data["folds"]
    target_flat: List[str] = []
    for f in folds:
        target_flat.extend(list(f["target_categories"]))
    if len(folds) != 5 or len(all_categories) != 30:
        raise ValueError("Expected five folds over 30 Real-IAD categories")
    if sorted(target_flat) != sorted(all_categories) or len(set(target_flat)) != 30:
        raise ValueError("Fold invariant failed: every category must be target exactly once")
    return data


def fold_categories(folds: Dict[str, Any], fold: int) -> Tuple[List[str], List[str]]:
    all_categories = list(folds["all_categories"])
    matches = [x for x in folds["folds"] if int(x["fold"]) == int(fold)]
    if len(matches) != 1:
        raise ValueError(f"Fold {fold} not found uniquely")
    target = list(matches[0]["target_categories"])
    source = [c for c in all_categories if c not in set(target)]
    if len(source) != 24 or len(target) != 6:
        raise AssertionError("Expected 24 source and 6 target categories")
    return source, target


# -----------------------------------------------------------------------------
# Compact Stage-0B cache reader
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheInfo:
    cache_dir: str
    fingerprint: str
    feature_dim: int
    patches_per_image: int
    shard_paths: Tuple[str, ...]


def inspect_cache(cache_dir: Path) -> CacheInfo:
    meta = json_load(cache_dir / "cache_metadata.json")
    comp = json_load(cache_dir / "cache_completion.json")
    if not bool(comp.get("complete")):
        raise RuntimeError("Compact cache is not marked complete")
    fingerprint = str(meta["config_fingerprint"])
    if str(comp.get("config_fingerprint")) != fingerprint:
        raise RuntimeError("Cache metadata/completion fingerprints disagree")
    if bool(meta.get("include_defects", False)):
        raise RuntimeError("C1 factorizer cache must be source-normal only; include_defects=true was found")
    if set(meta.get("splits", [])) != {"train"}:
        raise RuntimeError(f"C1 compact cache must contain train split only, got {meta.get('splits')}")
    shard_dir = cache_dir / "shards"
    shard_paths = tuple(str(x) for x in sorted(shard_dir.glob("shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {shard_dir}")
    first = torch_load(shard_paths[0], map_location="cpu")
    if str(first.get("config_fingerprint")) != fingerprint:
        raise RuntimeError("Shard fingerprint disagrees with cache metadata")
    p = first["patch_features"]
    if p.ndim != 3:
        raise ValueError(f"Expected [N,P,D] patch_features, got {tuple(p.shape)}")
    return CacheInfo(
        cache_dir=str(cache_dir),
        fingerprint=fingerprint,
        feature_dim=int(p.shape[2]),
        patches_per_image=int(p.shape[1]),
        shard_paths=shard_paths,
    )


def image_split(image_id: str, fold: int, val_fraction: float) -> str:
    u = stable_fraction(f"stage1|fold={fold}|{image_id}")
    return "val" if u < float(val_fraction) else "train"


def iter_cache_images(
    cache: CacheInfo,
    allowed_categories: set[str],
    split: str,
    fold: int,
    val_fraction: float,
    shard_shuffle_seed: Optional[int] = None,
) -> Iterator[Tuple[str, str, torch.Tensor]]:
    paths = list(cache.shard_paths)
    if shard_shuffle_seed is not None:
        rnd = random.Random(int(shard_shuffle_seed))
        rnd.shuffle(paths)
    for fp in paths:
        shard = torch_load(fp, map_location="cpu")
        if str(shard.get("config_fingerprint")) != cache.fingerprint:
            raise RuntimeError(f"Fingerprint mismatch in {fp}")
        feats = shard["patch_features"]
        ids = shard["image_ids"]
        cats = shard["categories"]
        for i, (image_id, cat) in enumerate(zip(ids, cats)):
            if cat not in allowed_categories:
                continue
            if image_split(str(image_id), fold, val_fraction) != split:
                continue
            yield str(image_id), str(cat), feats[i]


def collect_image_batch(it: Iterator[Tuple[str, str, torch.Tensor]], batch_images: int):
    buf: List[Tuple[str, str, torch.Tensor]] = []
    for item in it:
        buf.append(item)
        if len(buf) >= batch_images:
            yield buf
            buf = []
    if buf:
        yield buf


def manifest_split_stats(cache: CacheInfo, source: set[str], fold: int, val_fraction: float) -> Dict[str, Any]:
    path = Path(cache.cache_dir) / "manifest_public.jsonl"
    counts = {"train_images": 0, "val_images": 0, "ignored_non_source_images": 0}
    by_category: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cat = str(row["category"])
            image_id = str(row["image_id"])
            if cat not in source:
                counts["ignored_non_source_images"] += 1
                continue
            sp = image_split(image_id, fold, val_fraction)
            counts[f"{sp}_images"] += 1
            by_category.setdefault(cat, {"train": 0, "val": 0})[sp] += 1
    counts["source_images"] = counts["train_images"] + counts["val_images"]
    return {
        **counts,
        "by_source_category": by_category,
        "unit": "image; all cached patches of an image inherit this split",
    }


# -----------------------------------------------------------------------------
# Factorizer and normalization
# -----------------------------------------------------------------------------


class FeatureVAE(nn.Module):
    def __init__(self, input_dim: int = 1024, latent_dim: int = 32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-12.0, 12.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


@dataclass
class Normalizer:
    mode: str
    mean: torch.Tensor
    scalar: float

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "raw":
            return x
        return (x - self.mean.to(x.device, x.dtype)) / float(self.scalar)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "raw":
            return x
        return x * float(self.scalar) + self.mean.to(x.device, x.dtype)

    def state(self) -> Dict[str, Any]:
        return {"mode": self.mode, "mean": self.mean.cpu(), "scalar": float(self.scalar)}

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "Normalizer":
        return Normalizer(str(state["mode"]), state["mean"].float().cpu(), float(state["scalar"]))


def estimate_scalar_normalizer(
    cache: CacheInfo,
    source: set[str],
    fold: int,
    val_fraction: float,
    max_images: int = 0,
) -> Normalizer:
    total = 0
    sum_x = torch.zeros(cache.feature_dim, dtype=torch.float64)
    sum_sq = 0.0
    for n_img, (_, _, feats) in enumerate(
        iter_cache_images(cache, source, "train", fold, val_fraction), start=1
    ):
        x = feats.float().reshape(-1, cache.feature_dim).double()
        sum_x += x.sum(0)
        sum_sq += float((x * x).sum().item())
        total += int(x.shape[0])
        if max_images > 0 and n_img >= max_images:
            break
    if total == 0:
        raise RuntimeError("No source-train features available for normalizer")
    mean = sum_x / total
    centered_ss = sum_sq - total * float((mean * mean).sum().item())
    scalar = math.sqrt(max(centered_ss / (total * cache.feature_dim), 1e-12))
    return Normalizer("scalar", mean.float(), scalar)


def make_normalizer(mode: str, cache: CacheInfo, source: set[str], fold: int, val_fraction: float) -> Normalizer:
    if mode == "raw":
        return Normalizer("raw", torch.zeros(cache.feature_dim), 1.0)
    if mode == "scalar":
        return estimate_scalar_normalizer(cache, source, fold, val_fraction)
    raise ValueError(mode)


def vae_losses(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    rec_each = 0.5 * (x - recon).pow(2).sum(dim=1)
    kl_each_coord = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    kl_each = kl_each_coord.sum(dim=1)
    return rec_each.mean(), kl_each.mean(), kl_each_coord.mean(dim=0)


def grad_l2(parameters: Iterable[torch.nn.Parameter]) -> float:
    s = 0.0
    for p in parameters:
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(s)


# -----------------------------------------------------------------------------
# A. Scale audit
# -----------------------------------------------------------------------------


def run_scale_audit(args: argparse.Namespace) -> int:
    folds = load_folds(Path(args.folds))
    source, target = fold_categories(folds, args.fold)
    cache = inspect_cache(Path(args.cache_dir))
    device = device_of(args.device)
    set_seed(args.seed)

    raw_norm = Normalizer("raw", torch.zeros(cache.feature_dim), 1.0)
    scalar_norm = estimate_scalar_normalizer(
        cache, set(source), args.fold, args.val_fraction, max_images=args.normalizer_max_images
    )

    audit_images: List[torch.Tensor] = []
    for _, _, feats in iter_cache_images(cache, set(source), "train", args.fold, args.val_fraction):
        audit_images.append(feats.float())
        if len(audit_images) >= args.audit_images:
            break
    if not audit_images:
        raise RuntimeError("No images available for scale audit")

    base = FeatureVAE(cache.feature_dim, args.latent_dim)
    init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    rows = []

    for mode, norm in [("raw", raw_norm), ("scalar", scalar_norm)]:
        model = FeatureVAE(cache.feature_dim, args.latent_dim).to(device)
        model.load_state_dict(init_state)
        model.train()
        rec_vals: List[float] = []
        kl_vals: List[float] = []
        rec_grad_vals: List[float] = []
        kl_grad_vals: List[float] = []

        for batch in chunks(audit_images, args.audit_image_batch):
            x = torch.stack(batch, 0).to(device).reshape(-1, cache.feature_dim)
            x = norm.transform(x)
            torch.manual_seed(args.seed + len(rec_vals))
            recon, mu, logvar = model(x)
            rec, kl, _ = vae_losses(x, recon, mu, logvar)

            model.zero_grad(set_to_none=True)
            rec.backward(retain_graph=True)
            rec_g = grad_l2(model.encoder.parameters())
            rec_g = math.sqrt(rec_g**2 + grad_l2(model.mu.parameters())**2 + grad_l2(model.logvar.parameters())**2)

            model.zero_grad(set_to_none=True)
            (args.audit_beta * kl).backward()
            kl_g = grad_l2(model.encoder.parameters())
            kl_g = math.sqrt(kl_g**2 + grad_l2(model.mu.parameters())**2 + grad_l2(model.logvar.parameters())**2)

            rec_vals.append(float(rec.detach().cpu()))
            kl_vals.append(float(kl.detach().cpu()))
            rec_grad_vals.append(rec_g)
            kl_grad_vals.append(kl_g)

        rows.append({
            "mode": mode,
            "n_images": len(audit_images),
            "n_patches": len(audit_images) * cache.patches_per_image,
            "scalar": float(norm.scalar),
            "rec_loss": float(np.mean(rec_vals)),
            "kl_loss": float(np.mean(kl_vals)),
            "beta": float(args.audit_beta),
            "beta_kl_loss": float(args.audit_beta * np.mean(kl_vals)),
            "loss_ratio_betaKL_over_rec": float(args.audit_beta * np.mean(kl_vals) / max(np.mean(rec_vals), EPS)),
            "encoder_grad_rec": float(np.mean(rec_grad_vals)),
            "encoder_grad_betaKL": float(np.mean(kl_grad_vals)),
            "grad_ratio_betaKL_over_rec": float(np.mean(kl_grad_vals) / max(np.mean(rec_grad_vals), EPS)),
        })
        rows[-1]["loss_scale_compatible"] = bool(
            args.compat_min <= rows[-1]["loss_ratio_betaKL_over_rec"] <= args.compat_max
        )
        rows[-1]["encoder_gradient_scale_compatible"] = bool(
            args.compat_min <= rows[-1]["grad_ratio_betaKL_over_rec"] <= args.compat_max
        )
        rows[-1]["preflight_compatible"] = bool(
            rows[-1]["loss_scale_compatible"] and rows[-1]["encoder_gradient_scale_compatible"]
        )
        del model

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA + ".scale_audit",
        "fold": args.fold,
        "source_categories": source,
        "target_categories_hidden_from_audit": target,
        "cache_fingerprint": cache.fingerprint,
        "latent_dim": args.latent_dim,
        "image_level_split": manifest_split_stats(cache, set(source), args.fold, args.val_fraction),
        "compatibility_band": [args.compat_min, args.compat_max],
        "note": "Compatibility means both beta*KL/reconstruction and encoder-gradient ratios are within the predeclared order-of-magnitude band. Prefer raw if it passes; use scalar only if raw fails and scalar passes. This is a numerical preflight gate, not evidence for C1.",
        "conditions": rows,
    }
    json_dump(report, out / f"fold_{args.fold}_scale_audit.json")
    write_csv(rows, out / f"fold_{args.fold}_scale_audit.csv")
    print(json.dumps(report, indent=2))
    return 0


# -----------------------------------------------------------------------------
# B-D. VAE training and latent health
# -----------------------------------------------------------------------------


def beta_at_epoch(beta: float, epoch0: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(beta)
    return float(beta) * min(1.0, max(0.0, epoch0 / float(warmup_epochs)))


@torch.no_grad()
def evaluate_vae_on_cache(
    model: FeatureVAE,
    norm: Normalizer,
    cache: CacheInfo,
    source: set[str],
    fold: int,
    val_fraction: float,
    image_batch_size: int,
    device: torch.device,
    beta: float,
    latent_sample_cap: int,
) -> Dict[str, Any]:
    model.eval()
    n_patch = 0
    rec_sum = 0.0
    kl_sum = 0.0
    cos_sum = 0.0
    kl_coord_sum = torch.zeros(model.latent_dim, dtype=torch.float64)
    mu_samples: List[torch.Tensor] = []
    n_mu = 0

    it = iter_cache_images(cache, source, "val", fold, val_fraction)
    for batch in collect_image_batch(it, image_batch_size):
        x_raw = torch.stack([b[2] for b in batch], 0).float().to(device).reshape(-1, cache.feature_dim)
        x = norm.transform(x_raw)
        mu, logvar = model.encode(x)
        recon = model.decode(mu)
        rec, kl, kl_coord = vae_losses(x, recon, mu, logvar)
        recon_raw = norm.inverse(recon)
        cos = F.cosine_similarity(x_raw, recon_raw, dim=1).mean()
        n = int(x.shape[0])
        rec_sum += float(rec.item()) * n
        kl_sum += float(kl.item()) * n
        cos_sum += float(cos.item()) * n
        kl_coord_sum += kl_coord.detach().cpu().double() * n
        n_patch += n
        if n_mu < latent_sample_cap:
            take = min(latent_sample_cap - n_mu, n)
            mu_samples.append(mu[:take].detach().cpu().float())
            n_mu += take

    if n_patch == 0:
        raise RuntimeError("Validation split is empty")
    mu_cat = torch.cat(mu_samples, 0) if mu_samples else torch.empty(0, model.latent_dim)
    var_mu = mu_cat.var(dim=0, unbiased=False) if len(mu_cat) else torch.zeros(model.latent_dim)
    mean_mu = mu_cat.mean(dim=0) if len(mu_cat) else torch.zeros(model.latent_dim)
    corr_offdiag = float("nan")
    if len(mu_cat) >= 2:
        z = (mu_cat - mu_cat.mean(0)) / (mu_cat.std(0, unbiased=False) + 1e-8)
        corr = (z.T @ z) / len(z)
        mask = ~torch.eye(model.latent_dim, dtype=torch.bool)
        corr_offdiag = float(corr[mask].abs().mean().item())

    kl_coord = (kl_coord_sum / n_patch).float()
    return {
        "n_val_patches": n_patch,
        "rec_loss": rec_sum / n_patch,
        "kl_loss": kl_sum / n_patch,
        "beta_objective": rec_sum / n_patch + beta * kl_sum / n_patch,
        "recon_cosine_original_space": cos_sum / n_patch,
        "mean_abs_offdiag_mu_corr": corr_offdiag,
        "per_coordinate_kl": kl_coord.tolist(),
        "per_coordinate_var_mu": var_mu.tolist(),
        "per_coordinate_mean_mu": mean_mu.tolist(),
    }


def run_train(args: argparse.Namespace) -> int:
    folds = load_folds(Path(args.folds))
    source, target = fold_categories(folds, args.fold)
    cache = inspect_cache(Path(args.cache_dir))
    device = device_of(args.device)
    set_seed(args.seed)

    norm = make_normalizer(args.normalization, cache, set(source), args.fold, args.val_fraction)
    model = FeatureVAE(cache.feature_dim, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    out = Path(args.out_dir) / f"fold_{args.fold}" / f"beta_{args.beta:g}" / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    train_log = out / "train.jsonl"
    if train_log.exists() and not args.resume:
        train_log.unlink()

    resolved = {
        "schema": SCHEMA + ".train_config",
        "fold": args.fold,
        "source_categories": source,
        "target_categories": target,
        "cache_fingerprint": cache.fingerprint,
        "feature_dim": cache.feature_dim,
        "patches_per_image": cache.patches_per_image,
        "latent_dim": args.latent_dim,
        "groups_default": 8,
        "group_dim_default": 4 if args.latent_dim == 32 else None,
        "beta": args.beta,
        "beta_warmup_epochs": args.beta_warmup_epochs,
        "normalization": args.normalization,
        "normalization_scalar": float(norm.scalar),
        "val_fraction_image_level": args.val_fraction,
        "image_level_split": manifest_split_stats(cache, set(source), args.fold, args.val_fraction),
        "image_batch_size": args.image_batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "seed": args.seed,
        "git_sha": git_sha_or_unknown(),
    }
    json_dump(resolved, out / "resolved_config.json")

    best = float("inf")
    best_epoch = -1
    best_path = out / "checkpoint_best.pt"
    start_epoch = 0
    last_path = out / "checkpoint_last.pt"
    if args.resume and last_path.is_file():
        state = torch_load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state.get("best_val", best))
        best_epoch = int(state.get("best_epoch", -1))
        print(f"resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        beta_eff = beta_at_epoch(args.beta, epoch, args.beta_warmup_epochs)
        rec_sum = kl_sum = total_sum = 0.0
        n_patch = 0
        it = iter_cache_images(
            cache, set(source), "train", args.fold, args.val_fraction,
            shard_shuffle_seed=args.seed * 100000 + epoch,
        )
        for batch in collect_image_batch(it, args.image_batch_size):
            x_raw = torch.stack([b[2] for b in batch], 0).float().to(device).reshape(-1, cache.feature_dim)
            x = norm.transform(x_raw)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                recon, mu, logvar = model(x)
                rec, kl, _ = vae_losses(x, recon, mu, logvar)
                loss = rec + beta_eff * kl
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            n = int(x.shape[0])
            rec_sum += float(rec.detach().item()) * n
            kl_sum += float(kl.detach().item()) * n
            total_sum += float(loss.detach().item()) * n
            n_patch += n

        val = evaluate_vae_on_cache(
            model, norm, cache, set(source), args.fold, args.val_fraction,
            args.image_batch_size, device, args.beta, args.latent_sample_cap,
        )
        row = {
            "epoch": epoch,
            "beta_effective": beta_eff,
            "train_rec": rec_sum / max(n_patch, 1),
            "train_kl": kl_sum / max(n_patch, 1),
            "train_total": total_sum / max(n_patch, 1),
            "val_rec": val["rec_loss"],
            "val_kl": val["kl_loss"],
            "val_beta_objective": val["beta_objective"],
            "val_recon_cosine_original_space": val["recon_cosine_original_space"],
            "val_mean_abs_offdiag_mu_corr": val["mean_abs_offdiag_mu_corr"],
        }
        append_jsonl(train_log, row)
        print(json.dumps(row))

        is_best = bool(val["beta_objective"] < best)
        if is_best:
            best = float(val["beta_objective"])
            best_epoch = epoch
        ckpt = {
            "schema": SCHEMA + ".checkpoint",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": norm.state(),
            "epoch": epoch,
            "best_val": best,
            "best_epoch": best_epoch,
            "fold": args.fold,
            "beta": float(args.beta),
            "seed": args.seed,
            "latent_dim": args.latent_dim,
            "input_dim": cache.feature_dim,
            "source_categories": source,
            "target_categories": target,
            "cache_fingerprint": cache.fingerprint,
            "config": resolved,
        }
        torch.save(ckpt, last_path)
        if is_best:
            torch.save(ckpt, best_path)

    state = torch_load(best_path, map_location=device)
    model.load_state_dict(state["model"])
    final = evaluate_vae_on_cache(
        model, norm, cache, set(source), args.fold, args.val_fraction,
        args.image_batch_size, device, args.beta, args.latent_sample_cap,
    )
    kl = np.asarray(final["per_coordinate_kl"], dtype=float)
    var_mu = np.asarray(final["per_coordinate_var_mu"], dtype=float)
    group_kl = []
    group_var = []
    if args.latent_dim % 8 == 0:
        gd = args.latent_dim // 8
        group_kl = [float(kl[g*gd:(g+1)*gd].sum()) for g in range(8)]
        group_var = [float(var_mu[g*gd:(g+1)*gd].sum()) for g in range(8)]
    summary = {
        "schema": SCHEMA + ".train_summary",
        "fold": args.fold,
        "beta": float(args.beta),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint": str(best_path),
        "normalization": args.normalization,
        "normalization_scalar": float(norm.scalar),
        "source_categories": source,
        "target_categories_untouched": target,
        "cache_fingerprint": cache.fingerprint,
        "validation": final,
        "canonical_8_group_kl": group_kl,
        "canonical_8_group_var_mu": group_var,
        "interpretation_note": "KL/Var(mu) are continuous collapse/usage diagnostics. Do not infer semantic defect meanings from latent coordinates/groups.",
    }
    json_dump(summary, out / "summary.json")
    print(json.dumps(summary, indent=2))
    return 0


# -----------------------------------------------------------------------------
# Checkpoint helpers and common target sensor functions
# -----------------------------------------------------------------------------


@dataclass
class FrozenFactorizer:
    name: str
    model: FeatureVAE
    norm: Normalizer
    fold: int
    beta: float
    latent_dim: int
    cache_fingerprint: str

    @torch.no_grad()
    def mu(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.norm.transform(raw.float())
        mu, _ = self.model.encode(x)
        return mu

    @torch.no_grad()
    def reconstruct(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.norm.transform(raw.float())
        mu, _ = self.model.encode(x)
        rec = self.model.decode(mu)
        return self.norm.inverse(rec)


def parse_named_checkpoints(specs: Sequence[str], device: torch.device, expected_fold: int) -> List[FrozenFactorizer]:
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--checkpoint must be NAME=PATH, e.g. R1=outputs/.../checkpoint_best.pt")
        name, path = spec.split("=", 1)
        state = torch_load(path, map_location=device)
        fold = int(state["fold"])
        if fold != int(expected_fold):
            raise ValueError(f"Checkpoint {path} fold={fold}, expected {expected_fold}")
        model = FeatureVAE(int(state["input_dim"]), int(state["latent_dim"])).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        out.append(FrozenFactorizer(
            name=name.strip(), model=model, norm=Normalizer.from_state(state["normalizer"]),
            fold=fold, beta=float(state["beta"]), latent_dim=int(state["latent_dim"]),
            cache_fingerprint=str(state["cache_fingerprint"]),
        ))
    if len({x.name for x in out}) != len(out):
        raise ValueError("Checkpoint names must be unique")
    return out


def stage0_cfg_from_args(args: argparse.Namespace) -> Stage0Config:
    return Stage0Config(
        model_name=args.model_name,
        layers=tuple(range(4, 19)),
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        patch_size=14,
        layer_fusion="mean",
        support_augmentation="paper_geometric",
        query_view="identity",
        top_fraction=args.top_fraction,
        knn=1,
        shots=args.shots,
        support_seed=0,
        cache_dtype="float16" if not args.no_fp16 else "float32",
    )


def support_feature_images(sensor: PaperAlignedFrozenSensor, supports: Sequence[Stage0Record]) -> torch.Tensor:
    pieces = []
    for rec in supports:
        base = Image.open(rec.image_path).convert("RGB")
        try:
            variants = paper_support_variants(base)
            p, _ = sensor.encode_pil_batch(variants)
            pieces.append(p.detach())
        finally:
            base.close()
    return torch.cat(pieces, dim=0)


def anomaly_score_from_patch(q: torch.Tensor, bank: torch.Tensor, cfg: Stage0Config) -> Tuple[float, torch.Tensor]:
    amap = patch_nn_anomaly_map(q, bank, cfg.grid_hw, (cfg.crop_size, cfg.crop_size), knn=1, chunk=4096)
    flat = amap.flatten()
    k = max(1, int(round(cfg.top_fraction * flat.numel())))
    score = float(torch.topk(flat, k=k, largest=True).values.mean().item())
    return score, amap


# -----------------------------------------------------------------------------
# E. Held-out anomaly information retention
# -----------------------------------------------------------------------------


def run_retention(args: argparse.Namespace) -> int:
    folds = load_folds(Path(args.folds))
    source, target = fold_categories(folds, args.fold)
    device = device_of(args.device)
    factors = parse_named_checkpoints(args.checkpoint, device, args.fold)
    cfg = stage0_cfg_from_args(args)
    fp = config_fingerprint(cfg)
    for fac in factors:
        if fac.cache_fingerprint != fp:
            raise RuntimeError(
                f"Checkpoint {fac.name} cache fingerprint {fac.cache_fingerprint} != target sensor {fp}. "
                "Use the identical Stage-0 sensor configuration."
            )

    records = load_realiad_records(args.root, args.json_dir, set(target))
    sensor = PaperAlignedFrozenSensor(cfg, device=args.device, use_fp16=not args.no_fp16)
    out = Path(args.out_dir) / f"fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    cat_summaries: Dict[str, Any] = {}

    try:
        for category in target:
            supports = select_supports(records, category, args.shots, args.support_seed)
            sfeat = support_feature_images(sensor, supports)
            raw_bank = F.normalize(sfeat.reshape(-1, sfeat.shape[-1]).float(), dim=-1)
            banks: Dict[str, torch.Tensor] = {"R0": raw_bank}
            for fac in factors:
                rec = fac.reconstruct(raw_bank)
                banks[fac.name] = F.normalize(rec.float(), dim=-1)

            tests = [r for r in records if r.category == category and r.split == "test"]
            if args.max_test_per_category > 0:
                tests = tests[: args.max_test_per_category]
            method_scores: Dict[str, List[float]] = {k: [] for k in banks}
            labels: List[int] = []

            for batch_records in chunks(tests, args.target_batch_size):
                qfeat, _ = sensor.encode_path_batch(batch_records)
                for bi, rec_meta in enumerate(batch_records):
                    qraw = qfeat[bi].float()
                    y = 0 if rec_meta.is_good else 1
                    labels.append(y)
                    sample_scores: Dict[str, float] = {}
                    score0, _ = anomaly_score_from_patch(qraw, banks["R0"], cfg)
                    sample_scores["R0"] = score0
                    method_scores["R0"].append(score0)
                    for fac in factors:
                        qrec = F.normalize(fac.reconstruct(qraw).float(), dim=-1)
                        score, _ = anomaly_score_from_patch(qrec, banks[fac.name], cfg)
                        sample_scores[fac.name] = score
                        method_scores[fac.name].append(score)
                    rows.append({
                        "fold": args.fold, "category": category, "image_id": rec_meta.image_id,
                        "relative_path": rec_meta.relative_path, "offline_label": y, **sample_scores,
                    })

            cat_summaries[category] = {}
            for method, scores in method_scores.items():
                cat_summaries[category][method] = {
                    "n": len(labels), "n_defect": int(sum(labels)), "n_normal": int(len(labels)-sum(labels)),
                    "auroc": safe_auc(labels, scores), "ap": safe_ap(labels, scores),
                }
            print(category, json.dumps(cat_summaries[category]))
    finally:
        sensor.cleanup()

    methods = ["R0"] + [f.name for f in factors]
    macro = {
        m: {
            "macro_auroc": finite_mean(cat_summaries[c][m]["auroc"] for c in target),
            "macro_ap": finite_mean(cat_summaries[c][m]["ap"] for c in target),
        } for m in methods
    }
    for m in methods[1:]:
        macro[m]["delta_macro_auroc_vs_R0"] = macro[m]["macro_auroc"] - macro["R0"]["macro_auroc"]
        macro[m]["delta_macro_ap_vs_R0"] = macro[m]["macro_ap"] - macro["R0"]["macro_ap"]
    report = {
        "schema": SCHEMA + ".retention",
        "fold": args.fold, "source_categories_not_evaluated": source, "target_categories": target,
        "sensor_fingerprint": fp, "support_seed": args.support_seed,
        "methods": methods, "macro": macro, "categories": cat_summaries,
        "note": "AUROC is the primary representation-retention metric; AP is secondary. No automatic non-inferiority margin is inferred from R0 variability.",
    }
    write_csv(rows, out / "per_item_scores.csv")
    json_dump(report, out / "retention_summary.json")
    print(json.dumps(report["macro"], indent=2))
    return 0


# -----------------------------------------------------------------------------
# F. C1 responsibility/addressability
# -----------------------------------------------------------------------------


def make_groups(latent_dim: int, grouping: str, seed: int) -> List[List[int]]:
    if grouping == "coordinate":
        return [[i] for i in range(latent_dim)]
    if latent_dim % 8 != 0:
        raise ValueError("canonical/random 8-group experiment requires latent_dim divisible by 8")
    idx = list(range(latent_dim))
    if grouping == "random":
        rnd = random.Random(seed)
        rnd.shuffle(idx)
    elif grouping != "canonical":
        raise ValueError(grouping)
    gd = latent_dim // 8
    return [idx[g*gd:(g+1)*gd] for g in range(8)]


def factor_patch_surprise(z: torch.Tensor, coord_mean: torch.Tensor, coord_var: torch.Tensor, groups: List[List[int]]) -> torch.Tensor:
    var = coord_var.clamp_min(1e-6)
    s = 0.5 * (torch.log(2.0 * math.pi * var) + (z - coord_mean).pow(2) / var)
    return torch.stack([s[..., g].sum(dim=-1) for g in groups], dim=-1)


def topmean_factor(patch_surprise: torch.Tensor, top_fraction: float) -> torch.Tensor:
    if patch_surprise.ndim == 2:
        patch_surprise = patch_surprise.unsqueeze(0)
    k = max(1, int(round(top_fraction * patch_surprise.shape[1])))
    return torch.topk(patch_surprise, k=k, dim=1, largest=True).values.mean(dim=1)


def support_normal_reference(fac: FrozenFactorizer, support_feats: torch.Tensor, groups: List[List[int]], top_fraction: float):
    v, p, d = support_feats.shape
    z = fac.mu(support_feats.reshape(-1, d)).reshape(v, p, fac.latent_dim)
    flat = z.reshape(-1, fac.latent_dim)
    mean = flat.mean(0)
    var = flat.var(0, unbiased=True).clamp_min(1e-6)
    ps = factor_patch_surprise(z, mean, var, groups)
    image_a = topmean_factor(ps, top_fraction)
    cal_mean = image_a.mean(0)
    cal_std = image_a.std(0, unbiased=True).clamp_min(1e-6)
    return mean, var, cal_mean, cal_std


def responsibility(image_a: torch.Tensor, cal_mean: torch.Tensor, cal_std: torch.Tensor, temperature_lambda: float) -> torch.Tensor:
    standardized = (image_a - cal_mean) / cal_std.clamp_min(1e-6)
    return torch.softmax(float(temperature_lambda) * standardized, dim=-1)


def preprocess_mask(mask_path: str, resize_size: int, crop_size: int, grid_hw: Tuple[int, int]) -> np.ndarray:
    im = Image.open(mask_path).convert("L")
    try:
        im = resize_and_center_crop(im, resize_size, crop_size)
        im = im.resize((grid_hw[1], grid_hw[0]), resample=Image.Resampling.NEAREST)
        return (np.asarray(im) > 0).astype(np.uint8)
    finally:
        im.close()


def stability_fast(r: np.ndarray, labels: Sequence[str]) -> Tuple[float, float, float, int, int]:
    if len(r) < 2:
        return float("nan"), float("nan"), float("nan"), 0, 0
    x = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), EPS)
    total_sum = x.sum(axis=0)
    total_pair_sum = (float(total_sum @ total_sum) - len(x)) / 2.0
    total_pairs = len(x) * (len(x) - 1) // 2
    within_sum = 0.0
    within_pairs = 0
    for lab in sorted(set(labels)):
        xi = x[np.asarray([y == lab for y in labels])]
        n = len(xi)
        if n < 2:
            continue
        s = xi.sum(axis=0)
        within_sum += (float(s @ s) - n) / 2.0
        within_pairs += n * (n - 1) // 2
    between_pairs = total_pairs - within_pairs
    between_sum = total_pair_sum - within_sum
    within = within_sum / within_pairs if within_pairs else float("nan")
    between = between_sum / between_pairs if between_pairs else float("nan")
    delta = within - between if math.isfinite(within) and math.isfinite(between) else float("nan")
    return within, between, delta, within_pairs, between_pairs


def centroid_split_indices(image_ids: Sequence[str], labels: Sequence[str], fraction: float, seed: int):
    train = np.zeros(len(labels), dtype=bool)
    eval_ = np.zeros(len(labels), dtype=bool)
    for lab in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == lab]
        idx = sorted(idx, key=lambda i: stable_fraction(f"centroid|{seed}|{image_ids[i]}"))
        if len(idx) < 2:
            continue
        n_train = min(len(idx)-1, max(1, int(round(len(idx) * fraction))))
        train[idx[:n_train]] = True
        eval_[idx[n_train:]] = True
    return train, eval_


def source_separability(r: np.ndarray, image_ids: Sequence[str], labels: Sequence[str], fraction: float, seed: int):
    train, eval_ = centroid_split_indices(image_ids, labels, fraction, seed)
    available = sorted({labels[i] for i in range(len(labels)) if train[i]})
    if len(available) < 2 or int(eval_.sum()) == 0:
        return {
            "accuracy": float("nan"), "macro_f1": float("nan"),
            "n_centroid": int(train.sum()), "n_eval": int(eval_.sum()), "n_sources": len(available),
        }
    centroids = {}
    for lab in available:
        x = r[np.asarray([train[i] and labels[i] == lab for i in range(len(labels))])]
        c = x.mean(axis=0)
        centroids[lab] = c / max(np.linalg.norm(c), EPS)
    y_true = []
    y_pred = []
    for i in np.where(eval_)[0]:
        if labels[i] not in centroids:
            continue
        x = r[i] / max(np.linalg.norm(r[i]), EPS)
        pred = max(available, key=lambda lab: float(x @ centroids[lab]))
        y_true.append(labels[i])
        y_pred.append(pred)
    if not y_true:
        return {
            "accuracy": float("nan"), "macro_f1": float("nan"),
            "n_centroid": int(train.sum()), "n_eval": 0, "n_sources": len(available),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_centroid": int(train.sum()), "n_eval": len(y_true), "n_sources": len(available),
    }


def run_addressability(args: argparse.Namespace) -> int:
    folds = load_folds(Path(args.folds))
    source, target = fold_categories(folds, args.fold)
    device = device_of(args.device)
    factors = parse_named_checkpoints(args.checkpoint, device, args.fold)
    cfg = stage0_cfg_from_args(args)
    fp = config_fingerprint(cfg)
    for fac in factors:
        if fac.cache_fingerprint != fp:
            raise RuntimeError(f"Checkpoint {fac.name} sensor/cache fingerprint mismatch")

    records = load_realiad_records(args.root, args.json_dir, set(target))
    sensor = PaperAlignedFrozenSensor(cfg, device=args.device, use_fp16=not args.no_fp16)
    out = Path(args.out_dir) / f"fold_{args.fold}" / args.grouping
    out.mkdir(parents=True, exist_ok=True)
    all_summary: Dict[str, Any] = {}

    try:
        for fac in factors:
            groups = make_groups(fac.latent_dim, args.grouping, args.grouping_seed)
            per_cat: Dict[str, Any] = {}
            per_item_rows: List[Dict[str, Any]] = []
            for category in target:
                supports = select_supports(records, category, args.shots, args.support_seed)
                sfeat = support_feature_images(sensor, supports)
                mean, var, cal_mean, cal_std = support_normal_reference(
                    fac, sfeat, groups, args.factor_top_fraction
                )
                defects = [
                    r for r in records
                    if r.category == category and r.split == "test" and not r.is_good
                ]
                if args.max_defects_per_category > 0:
                    defects = defects[: args.max_defects_per_category]

                resp_rows: List[np.ndarray] = []
                labels: List[str] = []
                ids: List[str] = []
                spatial_aucs: List[float] = []

                for batch_records in chunks(defects, args.target_batch_size):
                    qfeat, _ = sensor.encode_path_batch(batch_records)
                    b, p, d = qfeat.shape
                    z = fac.mu(qfeat.reshape(-1, d)).reshape(b, p, fac.latent_dim)
                    ps = factor_patch_surprise(z, mean, var, groups)
                    image_a = topmean_factor(ps, args.factor_top_fraction)
                    resp = responsibility(image_a, cal_mean, cal_std, args.responsibility_lambda)

                    for bi, rec_meta in enumerate(batch_records):
                        r_np = resp[bi].detach().cpu().numpy().astype(float)
                        resp_rows.append(r_np)
                        labels.append(rec_meta.defect_source)
                        ids.append(rec_meta.image_id)
                        gstar = int(np.argmax(r_np))
                        spatial_auc = float("nan")
                        if rec_meta.mask_path and Path(rec_meta.mask_path).is_file():
                            mask = preprocess_mask(
                                rec_meta.mask_path, cfg.resize_size, cfg.crop_size, cfg.grid_hw
                            ).reshape(-1)
                            fmap = ps[bi, :, gstar].detach().cpu().numpy().reshape(-1)
                            if len(np.unique(mask)) >= 2:
                                spatial_auc = float(roc_auc_score(mask, fmap))
                                spatial_aucs.append(spatial_auc)
                        row: Dict[str, Any] = {
                            "fold": args.fold, "method": fac.name, "beta": fac.beta,
                            "grouping": args.grouping, "category": category,
                            "image_id": rec_meta.image_id, "relative_path": rec_meta.relative_path,
                            "offline_defect_source": rec_meta.defect_source,
                            "max_responsibility_index": gstar,
                            "spatial_pixel_auroc": spatial_auc,
                        }
                        for gi, val in enumerate(r_np):
                            row[f"r_{gi}"] = float(val)
                        per_item_rows.append(row)

                if resp_rows:
                    R = np.stack(resp_rows, 0)
                    sep = source_separability(
                        R, ids, labels, args.centroid_train_fraction, args.centroid_seed
                    )
                    within, between, delta, npw, npb = stability_fast(R, labels)
                else:
                    sep = {
                        "accuracy": float("nan"), "macro_f1": float("nan"),
                        "n_centroid": 0, "n_eval": 0, "n_sources": 0,
                    }
                    within = between = delta = float("nan")
                    npw = npb = 0
                per_cat[category] = {
                    "n_defects": len(resp_rows),
                    "n_sources": len(set(labels)),
                    "source_separability": sep,
                    "responsibility_stability": {
                        "within_cosine": within,
                        "between_cosine": between,
                        "delta_within_minus_between": delta,
                        "n_within_pairs": npw,
                        "n_between_pairs": npb,
                    },
                    "spatial_consistency": {
                        "mean_pixel_auroc": finite_mean(spatial_aucs),
                        "n_masks_scored": len(spatial_aucs),
                    },
                }
                print(f"{fac.name} {category}: " + json.dumps(per_cat[category]))

            macro = {
                "macro_source_accuracy": finite_mean(
                    per_cat[c]["source_separability"]["accuracy"] for c in target
                ),
                "macro_source_f1": finite_mean(
                    per_cat[c]["source_separability"]["macro_f1"] for c in target
                ),
                "macro_stability_delta": finite_mean(
                    per_cat[c]["responsibility_stability"]["delta_within_minus_between"] for c in target
                ),
                "macro_spatial_pixel_auroc": finite_mean(
                    per_cat[c]["spatial_consistency"]["mean_pixel_auroc"] for c in target
                ),
            }
            all_summary[fac.name] = {
                "beta": fac.beta,
                "grouping": args.grouping,
                "groups": groups,
                "macro": macro,
                "categories": per_cat,
            }
            write_csv(per_item_rows, out / f"{fac.name}_per_item_responsibility.csv")
    finally:
        sensor.cleanup()

    report = {
        "schema": SCHEMA + ".addressability",
        "fold": args.fold,
        "source_categories_not_used_for_target_labels": source,
        "target_categories": target,
        "sensor_fingerprint": fp,
        "grouping": args.grouping,
        "grouping_seed": args.grouping_seed,
        "factor_top_fraction": args.factor_top_fraction,
        "responsibility_lambda": args.responsibility_lambda,
        "centroid_train_fraction": args.centroid_train_fraction,
        "methods": all_summary,
        "offline_evaluator_boundary": "defect_source and masks are accessed only after responsibility is computed; they never train or update the factorizer/normal reference.",
    }
    json_dump(report, out / "addressability_summary.json")
    print(json.dumps({k: v["macro"] for k, v in all_summary.items()}, indent=2))
    return 0


# -----------------------------------------------------------------------------
# G. One-fold C1 comparison
# -----------------------------------------------------------------------------


def run_compare(args: argparse.Namespace) -> int:
    ret = json_load(Path(args.retention_summary))
    addr = json_load(Path(args.addressability_summary))
    r1 = args.r1_name
    rb = args.rbeta_name
    if r1 not in ret["macro"] or rb not in ret["macro"]:
        raise KeyError("R1/Rbeta names missing from retention summary")
    if r1 not in addr["methods"] or rb not in addr["methods"]:
        raise KeyError("R1/Rbeta names missing from addressability summary")
    a1 = addr["methods"][r1]["macro"]
    ab = addr["methods"][rb]["macro"]
    result = {
        "schema": SCHEMA + ".comparison",
        "fold": ret["fold"],
        "R0_macro_auroc": ret["macro"]["R0"]["macro_auroc"],
        "R1_macro_auroc": ret["macro"][r1]["macro_auroc"],
        "Rbeta_macro_auroc": ret["macro"][rb]["macro_auroc"],
        "R1_delta_auroc_vs_R0": ret["macro"][r1].get("delta_macro_auroc_vs_R0"),
        "Rbeta_delta_auroc_vs_R0": ret["macro"][rb].get("delta_macro_auroc_vs_R0"),
        "R1_addressability": a1,
        "Rbeta_addressability": ab,
        "Rbeta_minus_R1": {
            "macro_source_accuracy": ab["macro_source_accuracy"] - a1["macro_source_accuracy"],
            "macro_source_f1": ab["macro_source_f1"] - a1["macro_source_f1"],
            "macro_stability_delta": ab["macro_stability_delta"] - a1["macro_stability_delta"],
            "macro_spatial_pixel_auroc": ab["macro_spatial_pixel_auroc"] - a1["macro_spatial_pixel_auroc"],
        },
        "decision_note": "Do not call C1 supported from latent independence alone. Evidence requires better operational addressability for Rbeta than R1 together with acceptable held-out anomaly-utility retention. This script deliberately does not invent the acceptable utility-loss threshold.",
    }
    json_dump(result, Path(args.out))
    print(json.dumps(result, indent=2))
    return 0


# -----------------------------------------------------------------------------
# H. Cross-fold aggregation and paired bootstrap CIs
# -----------------------------------------------------------------------------


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int, alpha: float = 0.05) -> Dict[str, float]:
    x = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if len(x) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n": int(len(x)),
    }


def run_aggregate(args: argparse.Namespace) -> int:
    retention_paths = list(args.retention_summary or [])
    address_paths = list(args.addressability_summary or [])
    if args.retention_root:
        retention_paths.extend(
            str(p) for p in sorted(Path(args.retention_root).glob("fold_*/retention_summary.json"))
        )
    if args.addressability_root:
        address_paths.extend(
            str(p) for p in sorted(Path(args.addressability_root).glob("fold_*/*/addressability_summary.json"))
        )
    retention_paths = sorted(set(retention_paths))
    address_paths = sorted(set(address_paths))
    retention_docs = [json_load(Path(p)) for p in retention_paths]
    address_docs = [json_load(Path(p)) for p in address_paths]
    if len(retention_docs) != 5:
        raise ValueError(f"Expected exactly five retention summaries, found {len(retention_docs)}")
    if not address_docs:
        raise ValueError("No addressability summaries found")

    r1, rb = args.r1_name, args.rbeta_name
    seen: Dict[str, int] = {}
    ret_rows: List[Dict[str, Any]] = []
    for d in retention_docs:
        fold = int(d["fold"])
        for c in d["target_categories"]:
            if c in seen:
                raise ValueError(f"Target category {c} appears in more than one retention fold")
            seen[c] = fold
            row = {"category": c, "fold": fold}
            for m in ["R0", r1, rb]:
                if m not in d["categories"][c]:
                    raise KeyError(f"{m} missing for {c}")
                row[f"{m}_auroc"] = float(d["categories"][c][m]["auroc"])
                row[f"{m}_ap"] = float(d["categories"][c][m]["ap"])
            row[f"{r1}_delta_auroc_vs_R0"] = row[f"{r1}_auroc"] - row["R0_auroc"]
            row[f"{rb}_delta_auroc_vs_R0"] = row[f"{rb}_auroc"] - row["R0_auroc"]
            ret_rows.append(row)
    if len(seen) != 30:
        raise ValueError(f"Expected 30 unique held-out target categories, got {len(seen)}")

    retention = {
        "R0_macro_auroc": finite_mean(r["R0_auroc"] for r in ret_rows),
        "R0_macro_ap": finite_mean(r["R0_ap"] for r in ret_rows),
        f"{r1}_macro_auroc": finite_mean(r[f"{r1}_auroc"] for r in ret_rows),
        f"{r1}_macro_ap": finite_mean(r[f"{r1}_ap"] for r in ret_rows),
        f"{rb}_macro_auroc": finite_mean(r[f"{rb}_auroc"] for r in ret_rows),
        f"{rb}_macro_ap": finite_mean(r[f"{rb}_ap"] for r in ret_rows),
        f"{r1}_delta_auroc_vs_R0_bootstrap": bootstrap_mean_ci(
            [r[f"{r1}_delta_auroc_vs_R0"] for r in ret_rows],
            args.bootstrap, args.bootstrap_seed,
        ),
        f"{rb}_delta_auroc_vs_R0_bootstrap": bootstrap_mean_ci(
            [r[f"{rb}_delta_auroc_vs_R0"] for r in ret_rows],
            args.bootstrap, args.bootstrap_seed + 1,
        ),
        f"{rb}_minus_{r1}_auroc_bootstrap": bootstrap_mean_ci(
            [r[f"{rb}_auroc"] - r[f"{r1}_auroc"] for r in ret_rows],
            args.bootstrap, args.bootstrap_seed + 2,
        ),
        f"{rb}_minus_{r1}_ap_bootstrap": bootstrap_mean_ci(
            [r[f"{rb}_ap"] - r[f"{r1}_ap"] for r in ret_rows],
            args.bootstrap, args.bootstrap_seed + 3,
        ),
    }

    by_grouping: Dict[str, List[Dict[str, Any]]] = {}
    for d in address_docs:
        by_grouping.setdefault(str(d["grouping"]), []).append(d)
    addr_out: Dict[str, Any] = {}
    for grouping, docs in by_grouping.items():
        folds_present = sorted(int(d["fold"]) for d in docs)
        if folds_present != [0, 1, 2, 3, 4]:
            raise ValueError(f"Grouping {grouping} must have folds 0..4; got {folds_present}")
        rows: List[Dict[str, Any]] = []
        cat_seen = set()
        for d in docs:
            fold = int(d["fold"])
            for c in d["target_categories"]:
                if c in cat_seen:
                    raise ValueError(f"{grouping}: category {c} repeated")
                cat_seen.add(c)
                row = {"category": c, "fold": fold}
                for m in [r1, rb]:
                    cm = d["methods"][m]["categories"][c]
                    row[f"{m}_source_accuracy"] = float(cm["source_separability"]["accuracy"])
                    row[f"{m}_source_f1"] = float(cm["source_separability"]["macro_f1"])
                    row[f"{m}_stability_delta"] = float(
                        cm["responsibility_stability"]["delta_within_minus_between"]
                    )
                    row[f"{m}_spatial_auroc"] = float(
                        cm["spatial_consistency"]["mean_pixel_auroc"]
                    )
                rows.append(row)
        if len(cat_seen) != 30:
            raise ValueError(f"{grouping}: expected 30 unique categories, got {len(cat_seen)}")

        metrics = ["source_accuracy", "source_f1", "stability_delta", "spatial_auroc"]
        block: Dict[str, Any] = {"n_categories": 30}
        for metric in metrics:
            v1 = [r[f"{r1}_{metric}"] for r in rows]
            vb = [r[f"{rb}_{metric}"] for r in rows]
            diff = [b - a for a, b in zip(v1, vb) if math.isfinite(a) and math.isfinite(b)]
            block[f"{r1}_{metric}"] = finite_mean(v1)
            block[f"{rb}_{metric}"] = finite_mean(vb)
            block[f"Rbeta_minus_R1_{metric}_bootstrap"] = bootstrap_mean_ci(
                diff, args.bootstrap, args.bootstrap_seed + len(block)
            )
        addr_out[grouping] = block

    grouping_controls: Dict[str, Any] = {}
    if "canonical" in addr_out and "random" in addr_out:
        grouping_controls["Rbeta_canonical_minus_random"] = {
            metric: addr_out["canonical"][f"{rb}_{metric}"] - addr_out["random"][f"{rb}_{metric}"]
            for metric in ["source_accuracy", "source_f1", "stability_delta", "spatial_auroc"]
        }
    if "canonical" in addr_out and "coordinate" in addr_out:
        grouping_controls["Rbeta_canonical_minus_coordinate"] = {
            metric: addr_out["canonical"][f"{rb}_{metric}"] - addr_out["coordinate"][f"{rb}_{metric}"]
            for metric in ["source_accuracy", "source_f1", "stability_delta", "spatial_auroc"]
        }

    report = {
        "schema": SCHEMA + ".cross_fold_aggregate",
        "target_rotation_verified": True,
        "n_unique_target_categories": 30,
        "retention": retention,
        "addressability": addr_out,
        "grouping_controls": grouping_controls,
        "negative_control_note": "If random grouping is similar to canonical grouping, the 8x4 grouping has not demonstrated special structure. Coordinate grouping is an explicit granularity baseline.",
        "decision_note": "C1 requires Rbeta to improve operational addressability over R1 while held-out anomaly utility remains acceptably retained. No post-hoc utility-loss tolerance is generated here.",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(ret_rows, out / "retention_30_category.csv")
    json_dump(report, out / "c1_cross_fold_summary.json")
    print(json.dumps(report, indent=2))
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def add_common_fold(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fold", type=int, required=True, choices=range(5))
    p.add_argument("--folds", default="configs/realiad_folds_v0.json")


def add_sensor_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True)
    p.add_argument("--json-dir", required=True)
    p.add_argument("--model-name", default="dinov2_vitl14_reg")
    p.add_argument("--resize-size", type=int, default=448)
    p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--top-fraction", type=float, default=0.01)
    p.add_argument("--target-batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-fp16", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ABMG Stage 1-2 / C1 beta-VAE experiment")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("scale-audit", help="Source-only reconstruction/KL and gradient scale audit")
    add_common_fold(a)
    a.add_argument("--cache-dir", required=True)
    a.add_argument("--latent-dim", type=int, default=32)
    a.add_argument("--audit-beta", type=float, default=4.0)
    a.add_argument("--compat-min", type=float, default=0.1)
    a.add_argument("--compat-max", type=float, default=10.0)
    a.add_argument("--audit-images", type=int, default=256)
    a.add_argument("--audit-image-batch", type=int, default=32)
    a.add_argument("--normalizer-max-images", type=int, default=4096)
    a.add_argument("--val-fraction", type=float, default=0.10)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--device", default="cuda")
    a.add_argument("--out-dir", default="outputs/stage1_scale_audit")
    a.set_defaults(func=run_scale_audit)

    t = sub.add_parser("train", help="Train one fold-specific VAE/beta-VAE")
    add_common_fold(t)
    t.add_argument("--cache-dir", required=True)
    t.add_argument("--beta", type=float, required=True)
    t.add_argument("--latent-dim", type=int, default=32)
    t.add_argument("--normalization", choices=["raw", "scalar"], required=True)
    t.add_argument("--val-fraction", type=float, default=0.10)
    t.add_argument("--image-batch-size", type=int, default=64)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--beta-warmup-epochs", type=int, default=5)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-5)
    t.add_argument("--grad-clip", type=float, default=5.0)
    t.add_argument("--latent-sample-cap", type=int, default=100000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--out-dir", default="outputs/stage1_factorizers")
    t.set_defaults(func=run_train)

    r = sub.add_parser("retention", help="Held-out target R0/R1/Rbeta anomaly-ranking retention")
    add_common_fold(r)
    add_sensor_args(r)
    r.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; repeat for R1 and Rbeta")
    r.add_argument("--max-test-per-category", type=int, default=0)
    r.add_argument("--out-dir", default="outputs/stage1_retention")
    r.set_defaults(func=run_retention)

    c = sub.add_parser("addressability", help="C1 source separability, stability, and spatial consistency")
    add_common_fold(c)
    add_sensor_args(c)
    c.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; repeat for R1 and Rbeta")
    c.add_argument("--grouping", choices=["canonical", "coordinate", "random"], default="canonical")
    c.add_argument("--grouping-seed", type=int, default=0)
    c.add_argument("--factor-top-fraction", type=float, default=0.01)
    c.add_argument("--responsibility-lambda", type=float, default=1.0)
    c.add_argument("--centroid-train-fraction", type=float, default=0.5)
    c.add_argument("--centroid-seed", type=int, default=0)
    c.add_argument("--max-defects-per-category", type=int, default=0)
    c.add_argument("--out-dir", default="outputs/stage2_c1")
    c.set_defaults(func=run_addressability)

    x = sub.add_parser("compare", help="Make one-fold C1 comparison from retention/addressability summaries")
    x.add_argument("--retention-summary", required=True)
    x.add_argument("--addressability-summary", required=True)
    x.add_argument("--r1-name", default="R1")
    x.add_argument("--rbeta-name", default="Rbeta")
    x.add_argument("--out", required=True)
    x.set_defaults(func=run_compare)

    g = sub.add_parser("aggregate", help="Aggregate all five held-out folds with paired category bootstrap CIs")
    g.add_argument("--retention-summary", action="append", default=[])
    g.add_argument("--addressability-summary", action="append", default=[])
    g.add_argument("--retention-root", default="")
    g.add_argument("--addressability-root", default="")
    g.add_argument("--r1-name", default="R1")
    g.add_argument("--rbeta-name", default="Rbeta")
    g.add_argument("--bootstrap", type=int, default=5000)
    g.add_argument("--bootstrap-seed", type=int, default=12345)
    g.add_argument("--out-dir", default="outputs/c1_aggregate")
    g.set_defaults(func=run_aggregate)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
