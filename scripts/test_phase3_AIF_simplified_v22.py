import os
# PATCH_ID: COVERAGE_TAIL_SUPPORT_SELECTION_R8K4P2 -- adds --support_select coverage_tail and related argparse flags.
# unified_v22: v21 harness + support-robust cold-start and early rescue controls.
# Based on v21: v20 stable v16-core + bounded memory package + sign-aware correction-bank write/retention.
# Use --policy simple_aif, hardcode_band, or query baselines: entropy/margin/novelty/random/periodic.
# Adds optional support sigma-floor guard, warm-up/audit query budget, fast upward sidecar rescue, and guarded_dual_anchor_v2_balanced_rescue.
import json
import csv
import math
import random
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
from scipy import stats
import torch


import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import gc
import re
import copy
import hashlib
import contextlib
import sys
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vmb_visionad_new import (
    kmeans_torch,
    DINOv2MultiLayerBackbone,
    IndustrialDatasetIndex,   # MVTec-AD style
    VisACSVIndex,             # VisA style
    RealIADJSONIndex,
    SupportAugConfig as VisionADSupportAugConfig,
    augment_support as visionad_augment_support
)

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    # Make initialization reproducible across repeated runs on the same machine.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # Required by CUDA deterministic algorithms for some GEMM kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic backend settings (may reduce throughput, but stabilizes init).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    # Disable TF32 for reproducible matmul / conv numerics.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def safe_makedirs(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ---------------------------------------------------------------------------
#  VisionAD-inspired additions (Phase-2):
#   - Support augmentation
#   - Pseudo multi-view transforms (precompute per-view support banks)
#   - Category-indexed / mode-indexed normal banks (CLS-space k-means)
#   - Layer + view feature fusion (map-level fusion)
# ---------------------------------------------------------------------------
# VisionAD-style pose coverage (critical for classes like "screw")
ROT_SET = [0, 45, 90, 135, 180, 225, 270]

def augment_support_list(base: Image.Image, cfg: Any, aug_per_image: int) -> List[Image.Image]:
    """
    Return [base] + aug_per_image augmented images.

    Key: discrete rotation coverage (VisionAD) rather than small random rotations only.
    We cycle through non-zero angles deterministically so even small aug_per_image
    gets meaningful coverage.
    """
    out = [base]
    if aug_per_image <= 0:
        return out

    rot_cycle = [a for a in ROT_SET if a != 0]  # identity already included as base

    for i in range(int(aug_per_image)):
        ang = rot_cycle[i % len(rot_cycle)]
        im = base.rotate(ang, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

        # keep the same "secondary" jitter as vmb_visionad_new.py
        if getattr(cfg, "rotation_deg", 0.0) > 0 and random.random() < float(getattr(cfg, "rotation_p", 0.0)):
            a2 = random.uniform(-float(cfg.rotation_deg), float(cfg.rotation_deg))
            im = im.rotate(a2, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

        tr = float(getattr(cfg, "translate", 0.0))
        if tr > 0:
            w, h = im.size
            dx = random.uniform(-tr, tr) * w
            dy = random.uniform(-tr, tr) * h
            im = im.transform(
                (w, h),
                Image.AFFINE,
                (1.0, 0.0, dx, 0.0, 1.0, dy),
                resample=Image.BICUBIC,
                fillcolor=(0, 0, 0),
            )

        if float(getattr(cfg, "hflip_p", 0.0)) > 0 and random.random() < float(cfg.hflip_p):
            im = ImageOps.mirror(im)

        out.append(im)

    return out


def augment_support_list_baseline(base: Image.Image, img_size: int, aug_per_image: int) -> List[Image.Image]:
    """Baseline-faithful VisionAD support augmentation without forced identity insertion."""
    support_aug = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.9, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    return [support_aug(base) for _ in range(int(aug_per_image))]


def make_support_refs(
    base: Image.Image,
    backbone: Any,
    support_aug_cfg: Any,
    aug_per_image: int,
    *,
    baseline_compat: bool,
    include_identity: bool,
) -> List[Image.Image]:
    if baseline_compat:
        refs = augment_support_list_baseline(base, int(backbone.img_size), int(aug_per_image))
        if include_identity:
            refs = [base] + refs
    else:
        refs = augment_support_list(base, support_aug_cfg, aug_per_image)
        if (not include_identity) and len(refs) > 0:
            refs = refs[1:]
    return refs


@dataclass(frozen=True)
class SupportAugConfig:
    aug_per_image: int = 4
    # tuned to be similar to Phase-0 (VisionAD-style)
    rrc_scale: Tuple[float, float] = (0.90, 1.00)
    rrc_ratio: Tuple[float, float] = (0.90, 1.10)

    # Rotation augmentation (important for classes like "screw")
    rotation_deg: float = 15.0      # degrees for RandomRotation
    rotation_p: float = 0.7         # probability to apply rotation

    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.05
    hflip_p: float = 0.5

# def make_support_aug(img_size: int, cfg: SupportAugConfig) -> transforms.Compose:
#     ops = [
#         transforms.RandomResizedCrop(img_size, scale=cfg.rrc_scale, ratio=cfg.rrc_ratio),
#     ]
#     if cfg.rotation_deg and cfg.rotation_deg > 0:
#         ops.append(
#             transforms.RandomApply(
#                 [transforms.RandomRotation(degrees=float(cfg.rotation_deg), fill=0)],
#                 p=float(cfg.rotation_p),
#             )
#         )
#     ops.extend([
#         transforms.ColorJitter(
#             brightness=cfg.jitter_brightness,
#             contrast=cfg.jitter_contrast,
#             saturation=cfg.jitter_saturation,
#             hue=cfg.jitter_hue,
#         ),
#         transforms.RandomHorizontalFlip(p=cfg.hflip_p),
#     ])
#     return transforms.Compose(ops)

@dataclass(frozen=True)
class ViewSpec:
    name: str
    kind: str  # "id", "xflip", "yflip", "posclamp", "rbswap"
    posclamp_low: int = 64  # only used for posclamp; 0..255

def apply_view_pil(img: Image.Image, v: ViewSpec) -> Image.Image:
    if v.kind == "id":
        return img
    if v.kind == "xflip":
        return ImageOps.mirror(img)
    if v.kind == "yflip":
        return ImageOps.flip(img)
    if v.kind == "rbswap":
        r, g, b = img.split()
        return Image.merge("RGB", (b, g, r))
    if v.kind == "posclamp":
        # simple, deterministic photometric projection:
        # clamp dark pixels upward so texture/edges may become more salient.
        arr = np.asarray(img).astype(np.uint8)
        low = int(max(0, min(255, v.posclamp_low)))
        arr = np.maximum(arr, low).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    if v.kind == "rot90":
        # PIL rotates 90 degrees counter-clockwise
        return img.transpose(Image.ROTATE_90)
    if v.kind == "rot180":
        return img.transpose(Image.ROTATE_180)
    if v.kind == "rot270":
        return img.transpose(Image.ROTATE_270)
    raise ValueError(f"Unknown view kind: {v.kind}")

def _grid_hw_from_npatches(n_patches: int, aspect: Optional[float] = None) -> Tuple[int, int]:
    """
    Infer a plausible (H, W) patch grid from n_patches.

    - If aspect is provided (H_img / W_img), choose factor-pair whose h/w best matches aspect.
    - Else choose factor-pair closest to square.

    This preserves 2D structure for non-square images (critical for localization).
    """
    if n_patches <= 0:
        return 1, max(1, int(n_patches))

    best = None
    best_cost = float("inf")

    # enumerate factor pairs
    r = int(math.sqrt(n_patches))
    for h in range(1, r + 1):
        if n_patches % h != 0:
            continue
        w = n_patches // h

        if aspect is None:
            cost = abs(h - w)  # closest to square
        else:
            cost = abs((h / w) - float(aspect))  # closest to image aspect

        if cost < best_cost:
            best_cost = cost
            best = (h, w)

    if best is None:
        return 1, n_patches
    return int(best[0]), int(best[1])


def map_patch_index_identity_to_view(idx: int, v: ViewSpec, H: int, W: int) -> int:
    """Map identity-grid patch index -> view-grid patch index for geometric views.

    Notes:
      - We assume patch tokens are laid out in row-major order on an HxW grid.
      - For rotations, correctness requires H==W (true for ViT patch grids with square inputs).
    """
    if H == 1:
        return idx
    r, c = divmod(int(idx), int(W))

    if v.kind == "xflip":
        c = (W - 1 - c)
    elif v.kind == "yflip":
        r = (H - 1 - r)
    elif v.kind == "rot180":
        r = (H - 1 - r)
        c = (W - 1 - c)
    elif v.kind == "rot90":
        # 90° CCW: (r,c) -> (H-1-c, r)
        if H == W:
            r, c = (H - 1 - c), r
        else:
            return idx
    elif v.kind == "rot270":
        # 270° CCW (90° CW): (r,c) -> (c, W-1-r)
        if H == W:
            r, c = c, (W - 1 - r)
        else:
            return idx

    # photometric views keep indices
    return r * W + c


def align_map_to_identity(vec: torch.Tensor, v: ViewSpec, H: int, W: int) -> torch.Tensor:
    # vec: [N] in *view* coordinates; return aligned to identity coordinates
    if H == 1:
        return vec
    m = vec.view(H, W)

    if v.kind == "xflip":
        m = torch.flip(m, dims=[1])
    elif v.kind == "yflip":
        m = torch.flip(m, dims=[0])
    elif v.kind == "rot90":
        # view is 90° CCW; invert by rotating 90° CW
        if H == W:
            m = torch.rot90(m, k=-1, dims=(0, 1))
    elif v.kind == "rot180":
        if H == W:
            m = torch.rot90(m, k=2, dims=(0, 1))
    elif v.kind == "rot270":
        # view is 270° CCW; invert by rotating 90° CCW
        if H == W:
            m = torch.rot90(m, k=1, dims=(0, 1))

    # photometric views: identity
    return m.reshape(-1)


@dataclass
class ModeConfig:
    enabled: bool = True
    K_modes: int = 3
    shots_per_mode: int = 4
    

def build_modes_from_train_good(
    backbone: DINOv2MultiLayerBackbone,
    train_good_paths: List[str],
    *,
    K_modes: int,
    seed: int,
) -> Tuple[torch.Tensor, List[int]]:
    """Return (centroids [K, C] on CPU, labels per path on CPU)."""
    if len(train_good_paths) == 0:
        raise RuntimeError("No train_good_paths for mode building.")
    # Stable order prevents path-order drift from changing k-means assignments.
    train_good_paths = sorted(train_good_paths)
    gfeats = []
    for p in train_good_paths:
        _, g = backbone.encode_path(p)
        gfeats.append(g.detach().cpu())
    X = torch.stack(gfeats, dim=0)  # [N, C], already normalized in Phase-0 backbone
    Kc = int(min(K_modes, X.shape[0]))
    device = backbone.device
    # Re-seed right before clustering to stabilize centroid initialization.
    set_seed(seed)
    Cc, labels = kmeans_torch(X.to(device), K=Kc, num_iters=20)
    return Cc.detach().cpu(), labels.detach().cpu().tolist()



def select_clean_support_paths_per_mode(
    backbone: DINOv2MultiLayerBackbone,
    train_good_paths: List[str],
    labels: List[int],
    centroids_cpu: torch.Tensor,
    *,
    shots_per_mode: int,
    seed: int,
    max_candidates_per_mode: int = 300,
    keep_quantile: float = 0.90,
) -> Dict[int, List[str]]:
    """Cluster-conditional inlier filtering for *support* selection.

    Deterministic variant: no shuffling. Candidates are ranked by (distance-to-centroid, path),
    then the closest inliers are chosen. This avoids run-to-run drift in initial support banks.
    """
    n_modes = int(centroids_cpu.shape[0]) if centroids_cpu.numel() else 1
    mode_to_paths: Dict[int, List[str]] = {}
    train_good_paths = list(train_good_paths)

    for k in range(n_modes):
        idxs = sorted(i for i, lab in enumerate(labels) if int(lab) == int(k))
        idxs = idxs[: min(len(idxs), int(max_candidates_per_mode))]

        if not idxs:
            mode_to_paths[int(k)] = []
            continue

        c = centroids_cpu[int(k)].float()
        ranked = []
        for i in idxs:
            pth = train_good_paths[i]
            try:
                _, g = backbone.encode_path(pth)
                g = g.detach().cpu().float()
                sim = float(torch.clamp(torch.dot(g, c), -1.0, 1.0).item())
                d = 1.0 - sim
            except Exception:
                d = 1e9
            ranked.append((float(d), str(pth)))

        import numpy as _np
        arr = _np.asarray([d for d, _ in ranked], dtype=_np.float32)
        thr = float(_np.quantile(arr, float(keep_quantile)))

        inliers = [(d, pth) for d, pth in ranked if float(d) <= thr]
        if len(inliers) < max(1, int(shots_per_mode)):
            thr = float(_np.quantile(arr, min(0.95, float(keep_quantile) + 0.05)))
            inliers = [(d, pth) for d, pth in ranked if float(d) <= thr]

        inliers.sort(key=lambda x: (x[0], x[1]))
        mode_to_paths[int(k)] = [pth for _, pth in inliers[: max(1, int(shots_per_mode))]]

    return mode_to_paths

def choose_mode(global_feat: torch.Tensor, centroids_cpu: torch.Tensor) -> int:
    """global_feat: [C] on any device, assumed L2-normalized; centroids_cpu: [K, C]"""
    if centroids_cpu.numel() == 0:
        return 0
    c = centroids_cpu.to(global_feat.device)
    sim = (global_feat.view(1, -1) @ c.T).squeeze(0)  # [K]
    return int(sim.argmax().item())

# ---------------------------------------------------------------------------
#  Two-Gaussian Bayes + Welford (copied from Phase-1, kept intentionally)
#  - We ONLY use this to compute p_defect for querying and for a stable decision rule.
#  - No online logistic regression.
# ---------------------------------------------------------------------------

@dataclass
class ClassStats:
    # Normal scores
    mu_N: float = 0.0
    m2_N: float = 1e-6   # Welford M2 accumulator
    n_N: int = 0

    # Anomaly scores
    mu_A: float = 0.0
    m2_A: float = 1e-6
    n_A: int = 0

    # Hyperparameters
    lambda_z: float = 3.0   # global z-margin
    theta_dyn: float = 0.5  # a score-space threshold (used only as fallback logistic center)

    def var_N(self) -> float:
        if self.n_N > 1:
            return self.m2_N / (self.n_N - 1)
        return 1e-4

    def var_A(self) -> float:
        if self.n_A > 1:
            return self.m2_A / (self.n_A - 1)
        return 1e-4

    def init_from_support(self, scores: List[float]) -> None:
        """Initialise normal stats from support scores only (no test data)."""
        if len(scores) == 0:
            return
        arr = np.asarray(scores, dtype=np.float32)
        self.n_N = int(arr.shape[0])
        self.mu_N = float(arr.mean())
        if self.n_N > 1:
            var = float(arr.var(ddof=1))
            self.m2_N = var * (self.n_N - 1)
        else:
            var = 1e-4
            self.m2_N = var
        self.theta_dyn = self.mu_N + self.lambda_z * math.sqrt(var)

    def _update_normal(self, d: float) -> None:
        self.n_N += 1
        delta = d - self.mu_N
        self.mu_N += delta / self.n_N
        delta2 = d - self.mu_N
        self.m2_N += delta * delta2

        var = self.var_N()
        self.theta_dyn = self.mu_N + self.lambda_z * math.sqrt(var)

    def _update_anomaly(self, d: float) -> None:
        self.n_A += 1
        delta = d - self.mu_A
        self.mu_A += delta / self.n_A
        delta2 = d - self.mu_A
        self.m2_A += delta * delta2

    def update(self, d: float, is_anomaly: bool) -> None:
        if is_anomaly:
            self._update_anomaly(d)
        else:
            self._update_normal(d)

    def prior_probs(self) -> Tuple[float, float]:
        """Empirical priors with pseudo-counts."""
        total = self.n_N + self.n_A + 2.0
        pN = (self.n_N + 1.0) / total
        pA = (self.n_A + 1.0) / total
        return pN, pA

    def posterior_defect(self, d: float) -> float:
        """
        Approximate p(defect | score). If we have enough anomaly samples,
        use two-Gaussian Bayes; otherwise, fall back to a logistic around theta_dyn.
        """
        pN, pA = self.prior_probs()
        varN = self.var_N()
        varA = self.var_A()

        if self.n_A >= 3:
            sigmaN = math.sqrt(varN)
            sigmaA = math.sqrt(varA)

            def gauss(x: float, mu: float, sigma: float) -> float:
                sigma = max(sigma, 1e-4)
                z = (x - mu) / sigma
                return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))

            likN = gauss(d, self.mu_N, sigmaN)
            likA = gauss(d, self.mu_A, sigmaA)
            num = likA * pA
            den = num + likN * pN + 1e-9
            return num / den

        # Fallback: logistic around theta_dyn (not "online calibrator", just a stable heuristic)
        k = 8.0  # slope
        x = d - self.theta_dyn
        return 1.0 / (1.0 + math.exp(-k * x))
    
# ---------------------------------------------------------------------------
#  Robust thresholding (Phase-2 v4):
#   - Final decision uses a normal-only, FPR-controlled threshold theta
#   - p_defect is a smooth uncertainty proxy around theta (used mainly for querying)
#   - theta/sigma track the stream using *confident normals* to avoid selection bias
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    # Operating point: theta = quantile(bufN, 1 - target_fpr)
    target_fpr: float = 0.05
    min_bufN: int = 30
    bufN_max: int = 200   # This must be a problem, is it too small ???  mvtec_ad and visa have at most 200 images/class, but realiad has nearly 4000 images/class !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Confident-normal criteria for adding into bufN without labels
    p_low: float = 0.05 # To make sure the unlabelled sample is most likely the real normal sample

    # Cold-start prior (anchored on support normal stats)
    z_prior_anom: float = 5.0

    # Posterior softness (k = slope_c / sigmaN)
    slope_c: float = 1.0

    # theta_unc should not depend only on the *current* rolling buffer length.
    # Keep theta() on a bounded window, but let confidence grow sub-linearly with
    # additional trustworthy normal evidence after the buffer is full.
    theta_unc_extra_w: float = 2.0
    theta_unc_extra_cap: int = 4000 # This should be large enough to cover the entire stream in RealIAD, but not too large to cause instability in smaller datasets like MVTec-AD and VisA.

    # v22 cold-start support guard. Disabled by default to preserve v21 behavior.
    enable_support_guard: bool = False
    support_sigma_floor: float = 0.035
    support_theta_z: float = 2.5
    support_cold_sigma_thr: float = 0.020


def _sigmoid(x: float) -> float:
    # numerically stable enough for our ranges
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _quantile_representative_scores(scores: List[float], capacity: int) -> List[float]:
    """Return a bounded, distribution-preserving subset of scalar scores.

    For global normal-threshold initialization the calibration set can be much
    larger than the sidecar reservoir.  Keeping the last N samples would make
    the sidecar depend on path order, so we keep approximately uniform quantile
    representatives instead.
    """
    xs = [float(x) for x in scores if np.isfinite(float(x))]
    cap = int(max(0, capacity))
    if cap <= 0 or len(xs) == 0:
        return []
    if len(xs) <= cap:
        return xs
    arr = np.sort(np.asarray(xs, dtype=np.float64))
    idx = np.linspace(0, len(arr) - 1, num=cap)
    idx = np.unique(np.round(idx).astype(int))
    return [float(arr[i]) for i in idx]


def _safe_normal_ppf(q: float) -> float:
    """Numerically safe standard-normal inverse CDF for z-score calibration."""
    q = float(np.clip(float(q), 1e-6, 1.0 - 1e-6))
    try:
        z = float(stats.norm.ppf(q))
    except Exception:
        from statistics import NormalDist
        z = float(NormalDist().inv_cdf(q))
    if not np.isfinite(z):
        from statistics import NormalDist
        z = float(NormalDist().inv_cdf(q))
    return float(z)


class NormalThresholdController:
    """Normal-only threshold controller.

    Maintains a rolling buffer of confident-normal scores bufN.
    - theta() returns quantile(bufN, 1 - target_fpr) once bufN is populated
    - otherwise uses a prior derived from support scores (muN + z*sigmaN)
    - posterior(score) returns sigmoid(k*(score - theta)) for query uncertainty
    """
    def __init__(self, cfg: ThresholdConfig):
        self.cfg = cfg
        self.bufN: List[float] = []
        self.muN0: float = 0.0
        self.sigmaN0: float = 1e-3
        self.theta0: float = 0.5
        self.global_threshold_init_mode: str = "support"
        self.support_guard_info: Dict[str, Any] = {}
        self.n_normals_total: int = 0
        self.n_normals_support: int = 0
        self.n_normals_calibration: int = 0
        self.n_normals_labeled: int = 0
        self.n_normals_unlabeled: int = 0

    def _append_bufN(self, score: float) -> None:
        self.bufN.append(float(score))
        if len(self.bufN) > int(self.cfg.bufN_max):
            self.bufN = self.bufN[-int(self.cfg.bufN_max):]

    def _record_normal(self, score: float, source: str) -> None:
        self._append_bufN(score)
        self.n_normals_total += 1
        if source == "support":
            self.n_normals_support += 1
        elif source == "calibration":
            self.n_normals_calibration += 1
        elif source == "labeled":
            self.n_normals_labeled += 1
        elif source == "unlabeled":
            self.n_normals_unlabeled += 1

    def effective_normal_count(self, extra_verified_normals: int = 0) -> float:
        """Effective sample size for theta uncertainty.

        theta() still uses the rolling buffer, but theta_unc should keep improving
        after bufN is full when more trustworthy normals are observed. Use a
        sub-linear extra term so confidence does not become unrealistically strong.
        """
        extra_verified_normals = int(max(0, extra_verified_normals))
        n_buf = min(int(self.cfg.bufN_max), len(self.bufN) + extra_verified_normals)
        n_total = self.n_normals_total + extra_verified_normals
        extra = max(0, int(n_total) - int(n_buf))
        if int(self.cfg.theta_unc_extra_cap) > 0:
            extra = min(extra, int(self.cfg.theta_unc_extra_cap))
        n_eff = float(max(1, n_buf)) + float(self.cfg.theta_unc_extra_w) * math.sqrt(float(extra))
        return float(max(1.0, n_eff))
    

    def init_from_support(self, support_scores: List[float]) -> None:
        if len(support_scores) == 0:
            self.muN0, self.sigmaN0, self.theta0 = 0.0, 1e-3, 0.5
            self.bufN = []
            self.n_normals_total = 0
            self.n_normals_support = 0
            self.n_normals_calibration = 0
            self.n_normals_labeled = 0
            self.n_normals_unlabeled = 0
            return

        arr = np.asarray(support_scores, dtype=np.float32)
        mu = float(arr.mean())
        sig_raw = float(arr.std(ddof=1)) if arr.shape[0] > 1 else float(arr.std())  # fallback
        sig_raw = float(max(sig_raw, 1e-6))
        guard_enabled = bool(getattr(self.cfg, "enable_support_guard", False))
        sig_eff = float(max(sig_raw, float(getattr(self.cfg, "support_sigma_floor", 0.0)))) if guard_enabled else sig_raw
        theta_raw = float(mu + 0.5 * float(self.cfg.z_prior_anom) * sig_raw)
        if guard_enabled:
            theta_guard = float(mu + float(getattr(self.cfg, "support_theta_z", 2.5)) * sig_eff)
            theta0 = float(max(theta_raw, theta_guard))
        else:
            theta_guard = theta_raw
            theta0 = theta_raw
        self.muN0 = mu
        self.sigmaN0 = sig_eff
        self.theta0 = theta0
        self.global_threshold_init_mode = "support"
        self.support_guard_info = {
            "enabled": bool(guard_enabled),
            "mu": float(mu),
            "sigma_raw": float(sig_raw),
            "sigma_eff": float(sig_eff),
            "sigma_floor": float(getattr(self.cfg, "support_sigma_floor", 0.0)),
            "cold_start_risk": bool(sig_raw < float(getattr(self.cfg, "support_cold_sigma_thr", 0.0))),
            "theta_raw": float(theta_raw),
            "theta_guard": float(theta_guard),
            "theta0": float(theta0),
        }

        # Optional: seed bufN with support scores (bounded)
        self.bufN = [float(x) for x in support_scores[-self.cfg.bufN_max:]]
        self.n_normals_support = len(self.bufN)
        self.n_normals_calibration = 0
        self.n_normals_total = len(self.bufN)
        self.n_normals_labeled = 0
        self.n_normals_unlabeled = 0

    def init_from_normal_calibration(
        self,
        support_scores: List[float],
        calibration_scores: List[float],
        mode: str = "raw_quantile",
    ) -> None:
        """Initialize the operating threshold from a normal-only training pool.

        Modes:
          - raw_quantile: previous behavior, theta0 = empirical Q_(1-target_fpr).
          - zscore_gaussian: Fix-B behavior, theta0 = mean + std * Phi^{-1}(1-target_fpr).

        The calibration samples are used only as scalar threshold statistics;
        they are not inserted into any visual memory bank.
        """
        mode = str(mode or "raw_quantile").lower().strip()
        if mode not in {"raw_quantile", "zscore_gaussian"}:
            raise ValueError(f"Unknown global threshold init mode: {mode}")

        if calibration_scores is None or len(calibration_scores) == 0:
            self.init_from_support(support_scores)
            return
        arr = np.asarray([float(x) for x in calibration_scores if np.isfinite(float(x))], dtype=np.float32)
        if arr.size == 0:
            self.init_from_support(support_scores)
            return

        q = float(np.clip(1.0 - float(self.cfg.target_fpr), 0.0, 1.0))
        zq = _safe_normal_ppf(q)
        mu = float(arr.mean())
        sig = float(arr.std(ddof=1)) if arr.shape[0] > 1 else float(arr.std())
        sig = float(max(sig, 1e-6))
        theta_quantile = float(np.quantile(arr, q))
        theta_gaussian = float(mu + sig * zq)
        theta0 = theta_gaussian if mode == "zscore_gaussian" else theta_quantile

        self.muN0 = mu
        self.sigmaN0 = sig
        self.theta0 = float(theta0)
        self.global_threshold_init_mode = mode
        self.support_guard_info = {
            "enabled": False,
            "init_source": "normal_train_pool",
            "mode": str(mode),
            "n_support_scores": int(len(support_scores) if support_scores is not None else 0),
            "n_calibration_scores": int(arr.shape[0]),
            "target_fpr": float(self.cfg.target_fpr),
            "quantile": float(q),
            "z_quantile": float(zq),
            "mu": float(mu),
            "sigma_raw": float(sig),
            "sigma_eff": float(sig),
            "theta_quantile": float(theta_quantile),
            "theta_gaussian": float(theta_gaussian),
            "theta0": float(theta0),
        }

        if mode == "raw_quantile":
            # Preserve the previous behavior: seed the rolling normal reservoir with
            # distribution-preserving representatives, so theta() returns the
            # empirical normal quantile.
            self.bufN = _quantile_representative_scores(arr.tolist(), int(self.cfg.bufN_max))
            self.n_normals_total = len(self.bufN)
            self.n_normals_calibration = len(self.bufN)
        else:
            # Fix-B behavior: keep the normalized operating point fixed at zq.
            # Do not seed bufN, otherwise theta() would immediately switch back
            # to an empirical raw quantile and defeat the z-score initializer.
            self.bufN = []
            self.n_normals_total = int(arr.shape[0])
            self.n_normals_calibration = int(arr.shape[0])
        self.n_normals_support = 0
        self.n_normals_labeled = 0
        self.n_normals_unlabeled = 0

    def theta(self) -> float:
        if len(self.bufN) >= int(self.cfg.min_bufN):
            q = 1.0 - float(self.cfg.target_fpr)
            return float(np.quantile(np.asarray(self.bufN, dtype=np.float32), q))  # Is this best solution ????????????????????????????
        return float(self.theta0)

    def sigmaN(self) -> float:
        if len(self.bufN) >= int(self.cfg.min_bufN):
            arr = np.asarray(self.bufN, dtype=np.float32)
            sig = float(arr.std(ddof=1)) if arr.shape[0] > 1 else float(arr.std())
            return float(max(sig, 1e-6))
        return float(max(self.sigmaN0, 1e-6))

    def posterior(self, score: float) -> float:
        theta = self.theta()
        sig = self.sigmaN()
        k = float(self.cfg.slope_c) / sig
        return float(_sigmoid(k * (float(score) - theta)))   # the smaller the sigma, the steeper the sigmoid function

    def observe_unlabeled(self, score: float, p_defect: float, pred_defect: bool) -> None:
        # Only accept *very confident* predicted normals to avoid contamination
        if (not pred_defect) and (float(p_defect) <= float(self.cfg.p_low)):
            # self.bufN.append(float(score))
            # if len(self.bufN) > int(self.cfg.bufN_max):
            #     self.bufN = self.bufN[-int(self.cfg.bufN_max):]
            self._record_normal(float(score), source="unlabeled")

    def observe_labeled(self, score: float, y_true: int) -> None:
        # If we queried and user says it's normal, it's safe to add.
        if int(y_true) == 0:
            # self.bufN.append(float(score))
            # if len(self.bufN) > int(self.cfg.bufN_max):
            #     self.bufN = self.bufN[-int(self.cfg.bufN_max):]
            self._record_normal(float(score), source="labeled")

    def controller_name(self) -> str:
        return "legacy_normal_only"

    def debug_state(self) -> Dict[str, Any]:
        # Keep the debug schema compatible with GuardedDualAnchorThresholdController.
        # Legacy normal-only thresholding has no defect reservoir or recent labeled
        # sidecar window, so those fields are reported as zero/default values.
        return {
            "controller": self.controller_name(),
            "bufN_len": int(len(self.bufN)),
            "bufA_len": 0,
            "recent_labeled_len": 0,
            "n_normals_total": int(self.n_normals_total),
            "n_normals_support": int(self.n_normals_support),
            "n_normals_calibration": int(getattr(self, "n_normals_calibration", 0)),
            "n_normals_labeled": int(self.n_normals_labeled),
            "n_normals_unlabeled": int(self.n_normals_unlabeled),
            "n_defects_labeled": 0,
            "theta": float(self.theta()),
            "sigmaN": float(self.sigmaN()),
            "support_guard": getattr(self, "support_guard_info", {}),
            "last_mode": "none",
            "last_move": 0.0,
            "last_theta_star": float(self.theta()),
            "last_accept": False,
            "last_eval": {},
        }

@dataclass
class GuardedThetaSidecarConfig:
    normal_capacity: int = 512
    defect_capacity: int = 512
    recent_capacity: int = 96
    min_normal_anchor: int = 32
    min_defect_anchor: int = 16
    min_recent_total: int = 24
    min_recent_per_class: int = 6
    update_every_labeled: int = 8

    q_normal: float = 0.95
    q_defect: float = 0.10
    delta_normal: float = 0.005
    delta_defect: float = 0.005
    sep_margin: float = 0.02

    lambda_fp: float = 1.0
    lambda_fn: float = 1.0
    lambda_move: float = 0.2
    accept_margin: float = 1e-4
    guard_delta_fp: float = 0.01
    guard_delta_fn: float = 0.01

    step_eta: float = 0.3
    step_up: float = 0.01
    step_down: float = 0.01

    candidate_radius: float = 0.02
    candidate_step: float = 0.005
    support_seed: bool = True

    # Normal-dominant fallback for high-FPR classes with too few queried defects.
    enable_normal_fallback: bool = True
    min_recent_normals_fallback: int = 16
    min_defect_anchor_fallback: int = 2
    normal_fallback_fpr_trigger: float = 0.25
    normal_fallback_guard_delta_fn: float = 0.01

    # v22 fast upward rescue from verified false-positive normals.
    enable_fast_upward_rescue: bool = False
    fast_rescue_min_fp_normals: int = 2
    fast_rescue_fpr_trigger: float = 0.35
    fast_rescue_q_normal: float = 0.95
    fast_rescue_delta_normal: float = 0.005
    fast_rescue_step_up: float = 0.030
    fast_rescue_guard_delta_fn: float = 0.020
    fast_rescue_max_recent: int = 64

    # v22++ guarded_dual_anchor_v2_balanced_rescue.
    # Disabled by default so the original v22 behavior is exactly recoverable.
    enable_balanced_rescue: bool = False
    balanced_min_normals: int = 8
    balanced_min_defects: int = 3
    balanced_defect_q: float = 0.15
    balanced_defect_margin: float = 0.010
    balanced_normal_q: float = 0.95
    balanced_up_trigger: float = 0.35
    balanced_down_trigger: float = 0.25
    balanced_exit_fpr: float = 0.20
    balanced_exit_fnr: float = 0.15
    balanced_anchor_gap: float = 0.015
    balanced_up_defect_slack: float = 0.020
    balanced_step_up: float = 0.020
    balanced_step_down: float = 0.020
    balanced_normal_step: float = 0.005
    balanced_rescue_eta: float = 1.0
    balanced_candidate_radius_normal: float = 0.020
    balanced_candidate_radius_rescue: float = 0.060
    balanced_candidate_step: float = 0.005
    balanced_guard_delta_fp_normal: float = 0.010
    balanced_guard_delta_fn_normal: float = 0.010
    balanced_guard_delta_fn_up: float = 0.050
    balanced_guard_delta_fp_down: float = 0.200
    balanced_lambda_fp: float = 1.0
    balanced_lambda_fn: float = 1.5
    balanced_lambda_move: float = 0.10
    balanced_lambda_q: float = 0.10
    balanced_lambda_osc: float = 0.05
    balanced_cooldown_steps: int = 150


class GuardedDualAnchorThresholdController:
    """Decoupled operating-point controller using only queried verified labels.

    The controller is intentionally sidecar-only: it never appears in the AIF objective.
    It proposes a new theta from labeled normal/defect score anchors, and only accepts
    the move when a recent labeled evaluation window shows a broad improvement with
    small tolerated degradation on either side.

    Patch: add a guarded normal-dominant fallback mode for classes like eraser where
    queried labels are heavily normal-skewed and defect labels are too sparse to satisfy
    the full dual-anchor gate. The fallback can only move theta upward and still applies
    an explicit do-no-harm FNR guard using whatever verified defect anchors exist.
    """
    def __init__(self, cfg: GuardedThetaSidecarConfig, fallback_cfg: ThresholdConfig):
        self.cfg = cfg
        self.fallback_cfg = fallback_cfg

        self.bufN: List[float] = []
        self.bufA: List[float] = []
        self.recent_labeled: List[Tuple[float, int]] = []

        self.muN0: float = 0.0
        self.sigmaN0: float = 1e-3
        self.theta0: float = 0.5
        self.theta_current: float = 0.5
        self.support_guard_info: Dict[str, Any] = {}
        self.fast_rescue_fp_normals: List[float] = []
        self.fast_rescue_updates: int = 0
        self.balanced_rescue_updates: int = 0
        self.balanced_rescue_up_updates: int = 0
        self.balanced_rescue_down_updates: int = 0
        self.balanced_cooldown_remaining: int = 0
        self.balanced_last_direction: int = 0

        self.n_normals_total: int = 0
        self.n_normals_support: int = 0
        self.n_normals_calibration: int = 0
        self.n_normals_labeled: int = 0
        self.n_normals_unlabeled: int = 0
        self.n_defects_labeled: int = 0

        self.labeled_since_update: int = 0
        self.last_move: float = 0.0
        self.last_theta_star: float = 0.5
        self.last_accept: bool = False
        self.last_eval: Dict[str, Any] = {}
        self.last_mode: str = "none"

    def controller_name(self) -> str:
        if bool(getattr(self.cfg, "enable_balanced_rescue", False)):
            return "guarded_dual_anchor_v2_balanced_rescue"
        return "guarded_dual_anchor"

    def _append_recent(self, score: float, y_true: int) -> None:
        self.recent_labeled.append((float(score), int(y_true)))
        if len(self.recent_labeled) > int(self.cfg.recent_capacity):
            self.recent_labeled = self.recent_labeled[-int(self.cfg.recent_capacity):]

    def _append_normal(self, score: float, source: str) -> None:
        self.bufN.append(float(score))
        if len(self.bufN) > int(self.cfg.normal_capacity):
            self.bufN = self.bufN[-int(self.cfg.normal_capacity):]
        self.n_normals_total += 1
        if source == "support":
            self.n_normals_support += 1
        elif source == "calibration":
            self.n_normals_calibration += 1
        elif source == "labeled":
            self.n_normals_labeled += 1
        elif source == "unlabeled":
            self.n_normals_unlabeled += 1

    def _append_defect(self, score: float) -> None:
        self.bufA.append(float(score))
        if len(self.bufA) > int(self.cfg.defect_capacity):
            self.bufA = self.bufA[-int(self.cfg.defect_capacity):]
        self.n_defects_labeled += 1

    def init_from_support(self, support_scores: List[float]) -> None:
        if len(support_scores) == 0:
            self.muN0, self.sigmaN0, self.theta0 = 0.0, 1e-3, 0.5
            self.theta_current = self.theta0
            self.bufN = []
            self.bufA = []
            self.recent_labeled = []
            self.n_normals_total = 0
            self.n_normals_support = 0
            self.n_normals_calibration = 0
            self.n_normals_labeled = 0
            self.n_normals_unlabeled = 0
            self.n_defects_labeled = 0
            self.labeled_since_update = 0
            self.fast_rescue_fp_normals = []
            self.fast_rescue_updates = 0
            self.balanced_rescue_updates = 0
            self.balanced_rescue_up_updates = 0
            self.balanced_rescue_down_updates = 0
            self.balanced_cooldown_remaining = 0
            self.balanced_last_direction = 0
            self.support_guard_info = {"enabled": bool(getattr(self.fallback_cfg, "enable_support_guard", False)), "empty_support": True}
            self.last_mode = "none"
            return

        arr = np.asarray(support_scores, dtype=np.float32)
        mu = float(arr.mean())
        sig_raw = float(arr.std(ddof=1)) if arr.shape[0] > 1 else float(arr.std())
        sig_raw = float(max(sig_raw, 1e-6))
        guard_enabled = bool(getattr(self.fallback_cfg, "enable_support_guard", False))
        sig_eff = float(max(sig_raw, float(getattr(self.fallback_cfg, "support_sigma_floor", 0.0)))) if guard_enabled else sig_raw
        theta_raw = float(mu + 0.5 * float(self.fallback_cfg.z_prior_anom) * sig_raw)
        if guard_enabled:
            theta_guard = float(mu + float(getattr(self.fallback_cfg, "support_theta_z", 2.5)) * sig_eff)
            theta0 = float(max(theta_raw, theta_guard))
        else:
            theta_guard = theta_raw
            theta0 = theta_raw
        self.muN0 = mu
        self.sigmaN0 = sig_eff
        self.theta0 = float(theta0)
        self.theta_current = float(self.theta0)
        self.support_guard_info = {
            "enabled": bool(guard_enabled),
            "mu": float(mu),
            "sigma_raw": float(sig_raw),
            "sigma_eff": float(sig_eff),
            "sigma_floor": float(getattr(self.fallback_cfg, "support_sigma_floor", 0.0)),
            "cold_start_risk": bool(sig_raw < float(getattr(self.fallback_cfg, "support_cold_sigma_thr", 0.0))),
            "theta_raw": float(theta_raw),
            "theta_guard": float(theta_guard),
            "theta0": float(theta0),
        }

        self.bufN = []
        if bool(self.cfg.support_seed):
            for x in support_scores[-int(self.cfg.normal_capacity):]:
                self._append_normal(float(x), source="support")

        self.bufA = []
        self.recent_labeled = []
        self.n_normals_calibration = 0
        self.n_normals_labeled = 0
        self.n_normals_unlabeled = 0
        self.n_defects_labeled = 0
        self.labeled_since_update = 0
        self.fast_rescue_fp_normals = []
        self.fast_rescue_updates = 0
        self.balanced_rescue_updates = 0
        self.balanced_rescue_up_updates = 0
        self.balanced_rescue_down_updates = 0
        self.balanced_cooldown_remaining = 0
        self.balanced_last_direction = 0
        self.last_mode = "none"

    def init_from_normal_calibration(
        self,
        support_scores: List[float],
        calibration_scores: List[float],
        mode: str = "raw_quantile",
    ) -> None:
        """Initialize theta from a normal-only training calibration pool.

        Modes:
          - raw_quantile: previous behavior, theta0 = empirical Q_(1-target_fpr).
          - zscore_gaussian: Fix-B behavior, theta0 = mean + std * Phi^{-1}(1-target_fpr).

        The fixed visual memory remains built from the few-shot support set.
        Calibration samples are represented only by scalar normal scores in the
        threshold sidecar, not by patch prototypes in any visual memory bank.
        """
        mode = str(mode or "raw_quantile").lower().strip()
        if mode not in {"raw_quantile", "zscore_gaussian"}:
            raise ValueError(f"Unknown global threshold init mode: {mode}")

        if calibration_scores is None or len(calibration_scores) == 0:
            self.init_from_support(support_scores)
            return
        arr = np.asarray([float(x) for x in calibration_scores if np.isfinite(float(x))], dtype=np.float32)
        if arr.size == 0:
            self.init_from_support(support_scores)
            return

        q = float(np.clip(1.0 - float(self.fallback_cfg.target_fpr), 0.0, 1.0))
        zq = _safe_normal_ppf(q)
        mu = float(arr.mean())
        sig = float(arr.std(ddof=1)) if arr.shape[0] > 1 else float(arr.std())
        sig = float(max(sig, 1e-6))
        theta_quantile = float(np.quantile(arr, q))
        theta_gaussian = float(mu + sig * zq)
        theta0 = theta_gaussian if mode == "zscore_gaussian" else theta_quantile

        self.muN0 = mu
        self.sigmaN0 = sig
        self.theta0 = float(theta0)
        self.theta_current = float(theta0)
        self.support_guard_info = {
            "enabled": False,
            "init_source": "normal_train_pool",
            "mode": str(mode),
            "n_support_scores": int(len(support_scores) if support_scores is not None else 0),
            "n_calibration_scores": int(arr.shape[0]),
            "target_fpr": float(self.fallback_cfg.target_fpr),
            "quantile": float(q),
            "z_quantile": float(zq),
            "mu": float(mu),
            "sigma_raw": float(sig),
            "sigma_eff": float(sig),
            "theta_quantile": float(theta_quantile),
            "theta_gaussian": float(theta_gaussian),
            "theta0": float(theta0),
        }

        self.bufN = []
        if bool(self.cfg.support_seed):
            calib_reps = _quantile_representative_scores(arr.tolist(), int(self.cfg.normal_capacity))
            for x in calib_reps:
                self._append_normal(float(x), source="calibration")

        self.bufA = []
        self.recent_labeled = []
        self.n_normals_support = 0
        self.n_normals_labeled = 0
        self.n_normals_unlabeled = 0
        self.n_defects_labeled = 0
        self.labeled_since_update = 0
        self.fast_rescue_fp_normals = []
        self.fast_rescue_updates = 0
        self.balanced_rescue_updates = 0
        self.balanced_rescue_up_updates = 0
        self.balanced_rescue_down_updates = 0
        self.balanced_cooldown_remaining = 0
        self.balanced_last_direction = 0
        self.last_mode = "none"

    def theta(self) -> float:
        return float(self.theta_current)

    def sigmaN(self) -> float:
        if len(self.bufN) >= 2:
            arr = np.asarray(self.bufN, dtype=np.float32)
            sig = float(arr.std(ddof=1)) if arr.shape[0] > 1 else float(arr.std())
            return float(max(sig, 1e-6))
        return float(max(self.sigmaN0, 1e-6))

    def posterior(self, score: float) -> float:
        sig = self.sigmaN()
        k = float(self.fallback_cfg.slope_c) / sig
        return float(_sigmoid(k * (float(score) - float(self.theta_current))))

    def observe_unlabeled(self, score: float, p_defect: float, pred_defect: bool) -> None:
        # Intentionally decoupled from unlabeled stream scores.
        return

    def effective_normal_count(self, extra_verified_normals: int = 0) -> float:
        n = len(self.bufN) + max(0, int(extra_verified_normals))
        return float(max(1, n))

    def _quantile_or_none(self, xs: List[float], q: float) -> Optional[float]:
        if len(xs) == 0:
            return None
        return float(np.quantile(np.asarray(xs, dtype=np.float32), float(q)))

    def _split_recent(self) -> Tuple[List[float], List[float]]:
        normals = [float(s) for s, y in self.recent_labeled if int(y) == 0]
        defects = [float(s) for s, y in self.recent_labeled if int(y) == 1]
        return normals, defects

    def _rates_from_lists(self, theta: float, normals: List[float], defects: List[float]) -> Dict[str, Any]:
        out = {
            "n_total": int(len(normals) + len(defects)),
            "n_normals": int(len(normals)),
            "n_defects": int(len(defects)),
            "FPR": 0.0,
            "FNR": 0.0,
        }
        if len(normals) > 0:
            out["FPR"] = float(sum(float(s) >= float(theta) for s in normals) / max(1, len(normals)))
        if len(defects) > 0:
            out["FNR"] = float(sum(float(s) < float(theta) for s in defects) / max(1, len(defects)))
        return out

    def _rates(self, theta: float) -> Dict[str, Any]:
        normals, defects = self._split_recent()
        return self._rates_from_lists(theta, normals, defects)

    def _rates_defect_buffer(self, theta: float) -> Dict[str, Any]:
        defects = [float(s) for s in self.bufA]
        return self._rates_from_lists(theta, [], defects)

    def _objective(self, theta: float, rates: Dict[str, Any]) -> float:
        return (
            float(self.cfg.lambda_fp) * float(rates["FPR"])
            + float(self.cfg.lambda_fn) * float(rates["FNR"])
            + float(self.cfg.lambda_move) * abs(float(theta) - float(self.theta_current))
        )

    def _candidate_thetas(self, mode: str = "full") -> List[float]:
        cands = {float(self.theta_current)}

        qN = None
        if len(self.bufN) >= max(1, int(self.cfg.min_normal_anchor)):
            qN = self._quantile_or_none(self.bufN, float(self.cfg.q_normal))
        elif len(self.bufN) > 0:
            qN = self._quantile_or_none(self.bufN, float(self.cfg.q_normal))

        qA = None
        if len(self.bufA) >= max(1, int(self.cfg.min_defect_anchor)):
            qA = self._quantile_or_none(self.bufA, float(self.cfg.q_defect))
        elif len(self.bufA) > 0:
            qA = self._quantile_or_none(self.bufA, float(self.cfg.q_defect))

        if qN is not None:
            cands.add(float(qN + float(self.cfg.delta_normal)))
        if (mode == "full") and (qA is not None):
            cands.add(float(qA - float(self.cfg.delta_defect)))

        if (mode == "full") and (qN is not None) and (qA is not None) and ((float(qA) - float(qN)) > float(self.cfg.sep_margin)):
            lo = float(qN + float(self.cfg.delta_normal))
            hi = float(qA - float(self.cfg.delta_defect))
            if lo <= hi:
                cands.add(float(0.5 * (lo + hi)))

        radius = float(max(0.0, self.cfg.candidate_radius))
        step = float(max(1e-6, self.cfg.candidate_step))
        kmax = int(round(radius / step))
        for k in range(-kmax, kmax + 1):
            cand = float(self.theta_current + k * step)
            if mode == "normal_fallback" and cand < float(self.theta_current):
                continue
            cands.add(cand)

        out = sorted(float(x) for x in cands if np.isfinite(x))
        return out

    def _can_evaluate_full(self) -> bool:
        if len(self.recent_labeled) < int(self.cfg.min_recent_total):
            return False
        normals, defects = self._split_recent()
        return (len(normals) >= int(self.cfg.min_recent_per_class)) and (len(defects) >= int(self.cfg.min_recent_per_class))

    def _can_evaluate_normal_fallback(self) -> bool:
        if not bool(self.cfg.enable_normal_fallback):
            return False
        if len(self.recent_labeled) < int(self.cfg.min_recent_total):
            return False
        normals, defects = self._split_recent()
        if len(normals) < int(self.cfg.min_recent_normals_fallback):
            return False
        # Fallback is specifically for defect-sparse recent windows.
        if len(defects) >= int(self.cfg.min_recent_per_class):
            return False
        if len(self.bufA) < int(self.cfg.min_defect_anchor_fallback):
            return False
        rates_now = self._rates(float(self.theta_current))
        return float(rates_now["FPR"]) >= float(self.cfg.normal_fallback_fpr_trigger)

    def _balanced_anchor_values(self) -> Dict[str, Any]:
        """Return normal/defect anchors for guarded_dual_anchor_v2_balanced_rescue.

        theta_N protects normal-side FPR: high enough to avoid over-rejecting verified normals.
        theta_D protects defect-side FNR: low enough to avoid accepting low-score verified defects.
        """
        min_n = int(max(1, getattr(self.cfg, "balanced_min_normals", 8)))
        min_d = int(max(1, getattr(self.cfg, "balanced_min_defects", 3)))
        qn = None
        qd = None
        theta_N = None
        theta_D = None
        if len(self.bufN) >= min_n:
            qn = self._quantile_or_none(self.bufN, float(getattr(self.cfg, "balanced_normal_q", self.cfg.q_normal)))
            if qn is not None:
                theta_N = float(qn) + float(getattr(self.cfg, "delta_normal", 0.005))
        elif len(self.bufN) > 0:
            qn = self._quantile_or_none(self.bufN, float(getattr(self.cfg, "balanced_normal_q", self.cfg.q_normal)))
            if qn is not None:
                theta_N = float(qn) + float(getattr(self.cfg, "delta_normal", 0.005))

        if len(self.bufA) >= min_d:
            qd = self._quantile_or_none(self.bufA, float(getattr(self.cfg, "balanced_defect_q", 0.15)))
            if qd is not None:
                theta_D = float(qd) - float(getattr(self.cfg, "balanced_defect_margin", 0.010))

        return {
            "theta_N": None if theta_N is None else float(theta_N),
            "theta_D": None if theta_D is None else float(theta_D),
            "qN": None if qn is None else float(qn),
            "qD": None if qd is None else float(qd),
            "n_normal_anchor": int(len(self.bufN)),
            "n_defect_anchor": int(len(self.bufA)),
            "min_normals": int(min_n),
            "min_defects": int(min_d),
        }

    def _balanced_eval_lists(self) -> Tuple[List[float], List[float], str, str]:
        """Choose stable evaluation lists for the balanced sidecar objective.

        Prefer recent labels when they are class-sufficient; otherwise use the larger
        sidecar buffers.  This lets the sidecar move early in cold starts without
        requiring a fully balanced recent window, while still keeping the decision
        label-driven.
        """
        recent_normals, recent_defects = self._split_recent()
        min_n = int(max(1, getattr(self.cfg, "balanced_min_normals", 8)))
        min_d = int(max(1, getattr(self.cfg, "balanced_min_defects", 3)))
        if len(recent_normals) >= min_n:
            normals = list(recent_normals)
            n_src = "recent"
        else:
            normals = [float(x) for x in self.bufN]
            n_src = "buffer"
        if len(recent_defects) >= min_d:
            defects = list(recent_defects)
            d_src = "recent"
        else:
            defects = [float(x) for x in self.bufA]
            d_src = "buffer"
        return normals, defects, n_src, d_src

    def _balanced_rates(self, theta: float) -> Dict[str, Any]:
        normals, defects, n_src, d_src = self._balanced_eval_lists()
        rates = self._rates_from_lists(float(theta), normals, defects)
        rates["normal_source"] = str(n_src)
        rates["defect_source"] = str(d_src)
        return rates

    def _balanced_objective(self, theta: float, rates: Dict[str, Any]) -> float:
        sig = float(max(self.sigmaN(), 1e-6))
        move = abs(float(theta) - float(self.theta_current)) / sig
        direction = 0
        if float(theta) > float(self.theta_current) + 1e-12:
            direction = 1
        elif float(theta) < float(self.theta_current) - 1e-12:
            direction = -1
        osc = 1.0 if (int(getattr(self, "balanced_last_direction", 0)) * int(direction) < 0) else 0.0
        # Query pressure is handled by the AIF budget/FN-audit cap; keep the term explicit
        # for logging/objective extensibility without requiring loop-level state injection.
        query_pressure = 0.0
        return (
            float(getattr(self.cfg, "balanced_lambda_fp", self.cfg.lambda_fp)) * float(rates.get("FPR", 0.0))
            + float(getattr(self.cfg, "balanced_lambda_fn", self.cfg.lambda_fn)) * float(rates.get("FNR", 0.0))
            + float(getattr(self.cfg, "balanced_lambda_move", self.cfg.lambda_move)) * float(move)
            + float(getattr(self.cfg, "balanced_lambda_q", 0.0)) * float(query_pressure)
            + float(getattr(self.cfg, "balanced_lambda_osc", 0.0)) * float(osc)
        )

    def _candidate_thetas_balanced(self, mode: str, anchors: Dict[str, Any]) -> List[float]:
        theta_now = float(self.theta_current)
        cands = {theta_now}
        theta_N = anchors.get("theta_N", None)
        theta_D = anchors.get("theta_D", None)
        if theta_N is not None:
            cands.add(float(theta_N))
        if theta_D is not None:
            cands.add(float(theta_D))
        if theta_N is not None and theta_D is not None:
            cands.add(float(0.5 * (float(theta_N) + float(theta_D))))

        if mode == "fast_upward_rescue":
            cands.add(theta_now + float(getattr(self.cfg, "balanced_step_up", 0.020)))
            qfp = self._quantile_or_none(self.fast_rescue_fp_normals, min(0.99, float(getattr(self.cfg, "fast_rescue_q_normal", 0.95)))) if len(self.fast_rescue_fp_normals) > 0 else None
            if qfp is not None:
                cands.add(float(qfp) + float(getattr(self.cfg, "fast_rescue_delta_normal", 0.005)))
        elif mode == "fast_downward_rescue":
            cands.add(theta_now - float(getattr(self.cfg, "balanced_step_down", 0.020)))
        else:
            cands.add(theta_now + float(getattr(self.cfg, "balanced_normal_step", 0.005)))
            cands.add(theta_now - float(getattr(self.cfg, "balanced_normal_step", 0.005)))

        radius = float(getattr(self.cfg, "balanced_candidate_radius_rescue", 0.060)) if mode in {"fast_upward_rescue", "fast_downward_rescue"} else float(getattr(self.cfg, "balanced_candidate_radius_normal", 0.020))
        step = float(max(1e-6, getattr(self.cfg, "balanced_candidate_step", 0.005)))
        kmax = int(max(0, round(radius / step)))
        for k in range(-kmax, kmax + 1):
            cand = float(theta_now + k * step)
            if mode == "fast_upward_rescue" and cand < theta_now:
                continue
            if mode == "fast_downward_rescue" and cand > theta_now:
                continue
            cands.add(cand)

        return sorted(float(x) for x in cands if np.isfinite(float(x)))

    def _select_balanced_mode(self, anchors: Dict[str, Any], force_hint: Optional[str] = None, prev_mode: Optional[str] = None) -> str:
        theta_now = float(self.theta_current)
        rates_recent = self._rates(theta_now)
        rates_bal = self._balanced_rates(theta_now)
        fpr = float(rates_recent.get("FPR", 0.0)) if int(rates_recent.get("n_normals", 0)) > 0 else float(rates_bal.get("FPR", 0.0))
        fnr = float(rates_recent.get("FNR", 0.0)) if int(rates_recent.get("n_defects", 0)) > 0 else float(rates_bal.get("FNR", 0.0))
        theta_D = anchors.get("theta_D", None)

        if force_hint == "upward":
            return "fast_upward_rescue"
        if force_hint == "downward" and theta_D is not None:
            return "fast_downward_rescue"

        last = str(prev_mode) if prev_mode is not None else str(getattr(self, "last_mode", ""))
        if last == "fast_upward_rescue" and fpr > float(getattr(self.cfg, "balanced_exit_fpr", 0.20)):
            return "fast_upward_rescue"
        if last == "fast_downward_rescue" and fnr > float(getattr(self.cfg, "balanced_exit_fnr", 0.15)) and theta_D is not None:
            return "fast_downward_rescue"

        if theta_D is not None and (fnr >= float(getattr(self.cfg, "balanced_down_trigger", 0.25)) or theta_now > float(theta_D) + float(getattr(self.cfg, "balanced_anchor_gap", 0.015))):
            return "fast_downward_rescue"
        if fpr >= float(getattr(self.cfg, "balanced_up_trigger", 0.35)):
            return "fast_upward_rescue"
        return "normal_guarded"

    def _maybe_balanced_update_theta(self, force_hint: Optional[str] = None) -> None:
        prev_mode = str(getattr(self, "last_mode", "none"))
        self.last_accept = False
        self.last_move = 0.0
        self.last_theta_star = float(self.theta_current)
        self.last_mode = "none"

        if int(getattr(self, "balanced_cooldown_remaining", 0)) > 0 and force_hint is None:
            self.balanced_cooldown_remaining = int(max(0, int(self.balanced_cooldown_remaining) - 1))
            self.last_eval = {
                "status": "balanced_rescue_cooldown",
                "cooldown_remaining": int(self.balanced_cooldown_remaining),
                "current_theta": float(self.theta_current),
            }
            return

        anchors = self._balanced_anchor_values()
        normals_eval, defects_eval, _, _ = self._balanced_eval_lists()
        if len(normals_eval) == 0 and len(defects_eval) == 0:
            self.last_eval = {"status": "balanced_insufficient_buffers", "current_theta": float(self.theta_current), "anchors": anchors}
            return

        mode = self._select_balanced_mode(anchors, force_hint=force_hint, prev_mode=prev_mode)
        self.last_mode = str(mode)
        current_rates = self._balanced_rates(float(self.theta_current))
        current_obj = self._balanced_objective(float(self.theta_current), current_rates)
        best_theta = float(self.theta_current)
        best_rates = current_rates
        best_obj = float(current_obj)

        for cand in self._candidate_thetas_balanced(mode=mode, anchors=anchors):
            cand = float(cand)
            rates = self._balanced_rates(cand)
            if mode == "fast_upward_rescue":
                # Upward rescue may reduce FPR, but must not cause unconstrained FNR growth.
                if float(rates.get("FNR", 0.0)) > float(current_rates.get("FNR", 0.0)) + float(getattr(self.cfg, "balanced_guard_delta_fn_up", 0.050)):
                    continue
                theta_D = anchors.get("theta_D", None)
                if theta_D is not None and cand > max(float(self.theta_current), float(theta_D) + float(getattr(self.cfg, "balanced_up_defect_slack", 0.020))) + 1e-12:
                    continue
            elif mode == "fast_downward_rescue":
                # Downward rescue is allowed to raise FPR, but only with a relaxed guard.
                if float(rates.get("FPR", 0.0)) > float(current_rates.get("FPR", 0.0)) + float(getattr(self.cfg, "balanced_guard_delta_fp_down", 0.200)):
                    continue
            else:
                if float(rates.get("FPR", 0.0)) > float(current_rates.get("FPR", 0.0)) + float(getattr(self.cfg, "balanced_guard_delta_fp_normal", self.cfg.guard_delta_fp)):
                    continue
                if float(rates.get("FNR", 0.0)) > float(current_rates.get("FNR", 0.0)) + float(getattr(self.cfg, "balanced_guard_delta_fn_normal", self.cfg.guard_delta_fn)):
                    continue

            obj = self._balanced_objective(cand, rates)
            if float(obj) < float(best_obj) - float(self.cfg.accept_margin):
                best_theta = cand
                best_rates = rates
                best_obj = float(obj)

        self.last_eval = {
            "status": "balanced_evaluated",
            "mode": str(mode),
            "current_theta": float(self.theta_current),
            "current_rates": current_rates,
            "current_obj": float(current_obj),
            "anchors": anchors,
            "best_theta_preclip": float(best_theta),
            "best_rates": best_rates,
            "best_obj": float(best_obj),
            "force_hint": None if force_hint is None else str(force_hint),
            "cooldown_remaining": int(getattr(self, "balanced_cooldown_remaining", 0)),
        }
        if abs(float(best_theta) - float(self.theta_current)) <= 1e-12:
            return

        eta = float(getattr(self.cfg, "balanced_rescue_eta", 1.0)) if mode in {"fast_upward_rescue", "fast_downward_rescue"} else float(self.cfg.step_eta)
        delta = float(eta * (float(best_theta) - float(self.theta_current)))
        if mode == "fast_upward_rescue":
            delta = float(np.clip(delta, 0.0, float(getattr(self.cfg, "balanced_step_up", 0.020))))
        elif mode == "fast_downward_rescue":
            delta = float(np.clip(delta, -float(getattr(self.cfg, "balanced_step_down", 0.020)), 0.0))
        else:
            ns = float(getattr(self.cfg, "balanced_normal_step", 0.005))
            delta = float(np.clip(delta, -ns, ns))
        if abs(delta) <= 1e-12:
            return

        self.last_theta_star = float(best_theta)
        self.theta_current = float(self.theta_current + delta)
        self.last_move = float(delta)
        self.last_accept = True
        direction = 1 if delta > 0 else -1
        self.balanced_last_direction = int(direction)
        if mode in {"fast_upward_rescue", "fast_downward_rescue"}:
            self.balanced_rescue_updates += 1
            if mode == "fast_upward_rescue":
                self.balanced_rescue_up_updates += 1
            else:
                self.balanced_rescue_down_updates += 1
            self.balanced_cooldown_remaining = int(max(0, getattr(self.cfg, "balanced_cooldown_steps", 150)))
        self.last_eval["accepted_theta"] = float(self.theta_current)
        self.last_eval["accepted_delta"] = float(delta)
        self.last_eval["cooldown_remaining_after"] = int(getattr(self, "balanced_cooldown_remaining", 0))

    def _maybe_balanced_fast_rescue(self, score: float, y_true: int) -> None:
        """Immediate two-sided rescue trigger from newly queried labels.

        Verified FP normals trigger upward rescue; verified FN defects trigger downward rescue.
        The actual move is still selected by the guarded dual-anchor objective.
        """
        if not bool(getattr(self.cfg, "enable_balanced_rescue", False)):
            return
        s = float(score)
        y = int(y_true)
        theta_now = float(self.theta_current)
        if y == 0 and s >= theta_now:
            self.fast_rescue_fp_normals.append(s)
            max_recent = int(max(1, getattr(self.cfg, "fast_rescue_max_recent", 64)))
            if len(self.fast_rescue_fp_normals) > max_recent:
                self.fast_rescue_fp_normals = self.fast_rescue_fp_normals[-max_recent:]
            if len(self.fast_rescue_fp_normals) >= int(max(1, getattr(self.cfg, "fast_rescue_min_fp_normals", 2))):
                self._maybe_balanced_update_theta(force_hint="upward")
        elif y == 1 and s < theta_now:
            self._maybe_balanced_update_theta(force_hint="downward")

    def _maybe_update_theta(self) -> None:
        if bool(getattr(self.cfg, "enable_balanced_rescue", False)):
            self._maybe_balanced_update_theta(force_hint=None)
            return

        self.last_accept = False
        self.last_move = 0.0
        self.last_theta_star = float(self.theta_current)
        self.last_mode = "none"

        mode = None
        if self._can_evaluate_full():
            mode = "full"
        elif self._can_evaluate_normal_fallback():
            mode = "normal_fallback"

        if mode is None:
            normals, defects = self._split_recent()
            self.last_eval = {
                "status": "insufficient_recent_labels",
                "recent_normals": int(len(normals)),
                "recent_defects": int(len(defects)),
                "bufA_len": int(len(self.bufA)),
                "recent_FPR_current": float(self._rates(float(self.theta_current))["FPR"]) if len(self.recent_labeled) > 0 else 0.0,
            }
            return

        self.last_mode = str(mode)
        current_rates = self._rates(float(self.theta_current))
        if mode == "normal_fallback":
            defect_guard_current = self._rates_defect_buffer(float(self.theta_current))
            current_obj = (
                float(self.cfg.lambda_fp) * float(current_rates["FPR"])
                + float(self.cfg.lambda_fn) * float(defect_guard_current["FNR"])
                + float(self.cfg.lambda_move) * 0.0
            )
        else:
            defect_guard_current = None
            current_obj = self._objective(float(self.theta_current), current_rates)

        best_theta = float(self.theta_current)
        best_rates = current_rates
        best_defect_guard = defect_guard_current
        best_obj = current_obj

        for cand in self._candidate_thetas(mode=mode):
            rates = self._rates(float(cand))
            if float(rates["FPR"]) > float(current_rates["FPR"]) + float(self.cfg.guard_delta_fp):
                continue

            if mode == "normal_fallback":
                defect_guard = self._rates_defect_buffer(float(cand))
                if float(defect_guard["FNR"]) > float(defect_guard_current["FNR"]) + float(self.cfg.normal_fallback_guard_delta_fn):
                    continue
                obj = (
                    float(self.cfg.lambda_fp) * float(rates["FPR"])
                    + float(self.cfg.lambda_fn) * float(defect_guard["FNR"])
                    + float(self.cfg.lambda_move) * abs(float(cand) - float(self.theta_current))
                )
            else:
                if float(rates["FNR"]) > float(current_rates["FNR"]) + float(self.cfg.guard_delta_fn):
                    continue
                defect_guard = None
                obj = self._objective(float(cand), rates)

            if obj < (best_obj - float(self.cfg.accept_margin)):
                best_theta = float(cand)
                best_rates = rates
                best_defect_guard = defect_guard
                best_obj = obj

        self.last_eval = {
            "status": "evaluated",
            "mode": str(mode),
            "current_theta": float(self.theta_current),
            "current_rates": current_rates,
            "current_obj": float(current_obj),
            "best_theta_preclip": float(best_theta),
            "best_rates": best_rates,
            "best_obj": float(best_obj),
        }
        if defect_guard_current is not None:
            self.last_eval["current_defect_guard"] = defect_guard_current
        if best_defect_guard is not None:
            self.last_eval["best_defect_guard"] = best_defect_guard

        if abs(float(best_theta) - float(self.theta_current)) <= 1e-12:
            return

        delta = float(self.cfg.step_eta) * (float(best_theta) - float(self.theta_current))
        delta = float(np.clip(delta, -float(self.cfg.step_down), float(self.cfg.step_up)))
        if mode == "normal_fallback":
            delta = float(max(0.0, delta))
        if abs(delta) <= 1e-12:
            return

        self.last_theta_star = float(best_theta)
        self.theta_current = float(self.theta_current + delta)
        self.last_move = float(delta)
        self.last_accept = True
        self.last_eval["accepted_theta"] = float(self.theta_current)
        self.last_eval["accepted_delta"] = float(delta)

    def _maybe_fast_upward_rescue(self, score: float, y_true: int) -> None:
        """v22 upward rescue for cold-start false-positive cascades."""
        if not bool(getattr(self.cfg, "enable_fast_upward_rescue", False)):
            return
        s = float(score); y = int(y_true); theta_now = float(self.theta_current)
        if y != 0 or s < theta_now:
            return
        self.fast_rescue_fp_normals.append(s)
        max_recent = int(max(1, getattr(self.cfg, "fast_rescue_max_recent", 64)))
        if len(self.fast_rescue_fp_normals) > max_recent:
            self.fast_rescue_fp_normals = self.fast_rescue_fp_normals[-max_recent:]
        if len(self.fast_rescue_fp_normals) < int(max(1, getattr(self.cfg, "fast_rescue_min_fp_normals", 2))):
            return
        rates_now = self._rates(theta_now)
        if float(rates_now.get("FPR", 0.0)) < float(getattr(self.cfg, "fast_rescue_fpr_trigger", 0.35)):
            return
        qn = self._quantile_or_none(self.bufN, float(getattr(self.cfg, "fast_rescue_q_normal", 0.95)))
        qfp = self._quantile_or_none(self.fast_rescue_fp_normals, min(0.99, float(getattr(self.cfg, "fast_rescue_q_normal", 0.95))))
        cand = theta_now
        if qn is not None:
            cand = max(cand, float(qn) + float(getattr(self.cfg, "fast_rescue_delta_normal", 0.005)))
        if qfp is not None:
            cand = max(cand, float(qfp) + float(getattr(self.cfg, "fast_rescue_delta_normal", 0.005)))
        cand = min(cand, theta_now + float(max(1e-6, getattr(self.cfg, "fast_rescue_step_up", 0.03))))
        if cand <= theta_now + 1e-12:
            return
        if len(self.bufA) > 0:
            cur_def = self._rates_defect_buffer(theta_now)
            new_def = self._rates_defect_buffer(cand)
            if float(new_def.get("FNR", 0.0)) > float(cur_def.get("FNR", 0.0)) + float(getattr(self.cfg, "fast_rescue_guard_delta_fn", 0.02)):
                self.last_eval = {"status": "fast_upward_rescue_rejected_defect_guard", "current_theta": theta_now, "candidate_theta": float(cand), "current_defect_guard": cur_def, "candidate_defect_guard": new_def, "fast_rescue_fp_normals": int(len(self.fast_rescue_fp_normals))}
                return
        self.last_mode = "fast_upward_rescue"; self.last_accept = True
        self.last_theta_star = float(cand); self.last_move = float(cand - theta_now)
        self.theta_current = float(cand); self.fast_rescue_updates += 1
        self.last_eval = {"status": "fast_upward_rescue_accepted", "current_theta": theta_now, "accepted_theta": float(self.theta_current), "accepted_delta": float(self.last_move), "recent_rates_before": rates_now, "fast_rescue_fp_normals": int(len(self.fast_rescue_fp_normals))}

    def observe_labeled(self, score: float, y_true: int) -> None:
        s = float(score)
        y = int(y_true)
        if y == 0:
            self._append_normal(s, source="labeled")
        else:
            self._append_defect(s)
        self._append_recent(s, y)
        if bool(getattr(self.cfg, "enable_balanced_rescue", False)):
            self._maybe_balanced_fast_rescue(s, y)
        else:
            self._maybe_fast_upward_rescue(s, y)
        self.labeled_since_update += 1
        if self.labeled_since_update >= int(max(1, self.cfg.update_every_labeled)):
            self._maybe_update_theta()
            self.labeled_since_update = 0

    def debug_state(self) -> Dict[str, Any]:
        return {
            "controller": self.controller_name(),
            "bufN_len": int(len(self.bufN)),
            "bufA_len": int(len(self.bufA)),
            "recent_labeled_len": int(len(self.recent_labeled)),
            "n_normals_total": int(self.n_normals_total),
            "n_normals_support": int(self.n_normals_support),
            "n_normals_calibration": int(getattr(self, "n_normals_calibration", 0)),
            "n_normals_labeled": int(self.n_normals_labeled),
            "n_defects_labeled": int(self.n_defects_labeled),
            "theta": float(self.theta_current),
            "theta0": float(self.theta0),
            "sigmaN": float(self.sigmaN()),
            "support_guard": getattr(self, "support_guard_info", {}),
            "fast_rescue_fp_normals": int(len(getattr(self, "fast_rescue_fp_normals", []))),
            "fast_rescue_updates": int(getattr(self, "fast_rescue_updates", 0)),
            "balanced_rescue_enabled": bool(getattr(self.cfg, "enable_balanced_rescue", False)),
            "balanced_rescue_updates": int(getattr(self, "balanced_rescue_updates", 0)),
            "balanced_rescue_up_updates": int(getattr(self, "balanced_rescue_up_updates", 0)),
            "balanced_rescue_down_updates": int(getattr(self, "balanced_rescue_down_updates", 0)),
            "balanced_cooldown_remaining": int(getattr(self, "balanced_cooldown_remaining", 0)),
            "balanced_anchors": self._balanced_anchor_values() if bool(getattr(self.cfg, "enable_balanced_rescue", False)) else {},
            "last_mode": str(self.last_mode),
            "last_move": float(self.last_move),
            "last_theta_star": float(self.last_theta_star),
            "last_accept": bool(self.last_accept),
            "last_eval": self.last_eval,
        }


# ---------------------------------------------------------------------------
#  Fixed-capacity codebooks (prevent memory explosion)
# ---------------------------------------------------------------------------

class FixedCapacityCodebook:
    """Fixed-capacity cosine-prototype codebook.

    v19 keeps the old novelty-gated behavior by default, but optionally enables
    utility-aware STM retention for dynamic-normal memory.  Under pressure, the
    utility-aware rule protects repeatedly useful hard-normal anchors and replaces
    the weakest local prototype rather than blindly overwriting the nearest one.
    """
    def __init__(self, K: int, feat_dim: int, device: torch.device, dtype: torch.dtype = torch.float16):
        self.K = int(K)
        self.device = device
        self.feat_dim = int(feat_dim)
        self.dtype = dtype
        self.protos = torch.empty((0, self.feat_dim), device=self.device, dtype=self.dtype)

        # v19 metadata used only when utility-aware STM retention is enabled.
        self.meta_count = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.meta_utility = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.meta_boundary = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.meta_fp_rescue = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.meta_coverage = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.meta_first_step = torch.empty((0,), device=self.device, dtype=torch.long)
        self.meta_last_step = torch.empty((0,), device=self.device, dtype=torch.long)
        self.meta_last_event = torch.empty((0,), device=self.device, dtype=torch.long)
        self.meta_distinct_events = torch.empty((0,), device=self.device, dtype=torch.float32)
        self.retention_counters: Dict[str, int] = {
            "touches": 0,
            "utility_replacements": 0,
            "utility_skips": 0,
            "utility_candidate_evals": 0,
            "covered_candidates": 0,
        }

    def size(self) -> int:
        return int(self.protos.shape[0])

    def _empty_meta(self, n: int) -> None:
        n = int(n)
        self.meta_count = torch.ones((n,), device=self.device, dtype=torch.float32)
        self.meta_utility = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self.meta_boundary = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self.meta_fp_rescue = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self.meta_coverage = torch.zeros((n,), device=self.device, dtype=torch.float32)
        self.meta_first_step = torch.full((n,), -1, device=self.device, dtype=torch.long)
        self.meta_last_step = torch.full((n,), -1, device=self.device, dtype=torch.long)
        self.meta_last_event = torch.full((n,), -1, device=self.device, dtype=torch.long)
        self.meta_distinct_events = torch.ones((n,), device=self.device, dtype=torch.float32)

    def _ensure_meta(self) -> None:
        n = self.size()
        if getattr(self, "meta_count", None) is None or int(self.meta_count.numel()) != n:
            self._empty_meta(n)

    def clear(self) -> None:
        self.protos = torch.empty((0, self.feat_dim), device=self.device, dtype=self.dtype)
        self._empty_meta(0)

    def set_prototypes(self, feats: torch.Tensor) -> None:
        if feats is None or feats.numel() == 0:
            self.clear()
            return
        feats = l2_normalize(feats.to(self.device)).to(dtype=self.dtype)
        if feats.shape[0] > self.K:
            feats = feats[: self.K]
        self.protos = feats.clone()
        self._empty_meta(self.protos.shape[0])

    @torch.no_grad()
    def nearest(self, q: torch.Tensor, relax=False, k=5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          d: [N] cosine distance (1 - max cos)
          idx: [N] argmax prototype index (or -1 if empty)
        """
        if self.size() == 0:
            d = torch.ones((q.shape[0],), device=q.device)
            idx = torch.full((q.shape[0],), -1, device=q.device, dtype=torch.long)
            return d, idx

        # compute in fp32 for stability, store in fp16 to save VRAM
        q = l2_normalize(q).float()
        p = l2_normalize(self.protos.float())
        sim = q @ p.T # [N, K]

        if relax:
            topk, idx = sim.topk(k=min(k, sim.shape[1]), dim=1)
            return (1.0 - topk).mean(dim=1), idx

        max_sim, idx = sim.max(dim=1)
        d = 1.0 - max_sim
        return d, idx

    def _meta_values(self, *, utility_score: Optional[float], outcome: Optional[str], boundary_value: Optional[float], coverage_gain: Optional[float]) -> Tuple[float, float, float, float]:
        u = float(np.clip(1.0 if utility_score is None else float(utility_score), 0.0, 1.0))
        b = float(np.clip(u if boundary_value is None else float(boundary_value), 0.0, 1.0))
        fp = 1.0 if str(outcome or "").upper() == "FP" else 0.0
        cov = float(np.clip(0.0 if coverage_gain is None else float(coverage_gain), 0.0, 1.0))
        return u, b, fp, cov

    def _append_meta(self, *, utility_score: Optional[float], event_id: Optional[int], step: Optional[int], outcome: Optional[str], boundary_value: Optional[float], coverage_gain: Optional[float]) -> None:
        u, b, fp, cov = self._meta_values(utility_score=utility_score, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain)
        st = int(-1 if step is None else step)
        ev = int(-1 if event_id is None else event_id)
        self.meta_count = torch.cat([self.meta_count, torch.tensor([1.0], device=self.device)])
        self.meta_utility = torch.cat([self.meta_utility, torch.tensor([u], device=self.device)])
        self.meta_boundary = torch.cat([self.meta_boundary, torch.tensor([b], device=self.device)])
        self.meta_fp_rescue = torch.cat([self.meta_fp_rescue, torch.tensor([fp], device=self.device)])
        self.meta_coverage = torch.cat([self.meta_coverage, torch.tensor([cov], device=self.device)])
        self.meta_first_step = torch.cat([self.meta_first_step, torch.tensor([st], device=self.device, dtype=torch.long)])
        self.meta_last_step = torch.cat([self.meta_last_step, torch.tensor([st], device=self.device, dtype=torch.long)])
        self.meta_last_event = torch.cat([self.meta_last_event, torch.tensor([ev], device=self.device, dtype=torch.long)])
        self.meta_distinct_events = torch.cat([self.meta_distinct_events, torch.tensor([1.0], device=self.device)])

    def _set_meta(self, j: int, *, utility_score: Optional[float], event_id: Optional[int], step: Optional[int], outcome: Optional[str], boundary_value: Optional[float], coverage_gain: Optional[float]) -> None:
        u, b, fp, cov = self._meta_values(utility_score=utility_score, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain)
        st = int(-1 if step is None else step)
        ev = int(-1 if event_id is None else event_id)
        self.meta_count[j] = 1.0
        self.meta_utility[j] = float(u)
        self.meta_boundary[j] = float(b)
        self.meta_fp_rescue[j] = float(fp)
        self.meta_coverage[j] = float(cov)
        self.meta_first_step[j] = int(st)
        self.meta_last_step[j] = int(st)
        self.meta_last_event[j] = int(ev)
        self.meta_distinct_events[j] = 1.0

    def _touch_meta(self, j: int, *, utility_score: Optional[float], event_id: Optional[int], step: Optional[int], outcome: Optional[str], boundary_value: Optional[float], coverage_gain: Optional[float], ema: float = 0.10) -> None:
        if j < 0 or j >= self.size():
            return
        u, b, fp, cov = self._meta_values(utility_score=utility_score, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain)
        ema = float(np.clip(float(ema), 0.0, 1.0))
        self.meta_count[j] += 1.0
        self.meta_utility[j] = (1.0 - ema) * self.meta_utility[j] + ema * float(u)
        self.meta_boundary[j] = (1.0 - ema) * self.meta_boundary[j] + ema * float(b)
        self.meta_fp_rescue[j] = torch.maximum(self.meta_fp_rescue[j] * (1.0 - 0.5 * ema), torch.tensor(float(fp), device=self.device))
        self.meta_coverage[j] = torch.maximum(self.meta_coverage[j] * (1.0 - 0.5 * ema), torch.tensor(float(cov), device=self.device))
        if step is not None:
            st = int(step)
            if int(self.meta_first_step[j].item()) < 0:
                self.meta_first_step[j] = st
            self.meta_last_step[j] = st
        if event_id is not None:
            ev = int(event_id)
            if int(self.meta_last_event[j].item()) != ev:
                self.meta_distinct_events[j] += 1.0
                self.meta_last_event[j] = ev
        self.retention_counters["touches"] = int(self.retention_counters.get("touches", 0)) + 1

    @torch.no_grad()
    def _retention_scores(
        self,
        idxs: torch.Tensor,
        *,
        current_step: Optional[int],
        cover_feats: Optional[torch.Tensor],
        cover_eps: float,
        recency_tau: float,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        self._ensure_meta()
        idxs = idxs.to(self.device).long()
        count = self.meta_count[idxs].clamp_min(1.0)
        repeat = torch.clamp(torch.log1p(count) / math.log1p(max(2.0, float(self.meta_count.max().item()) + 1.0)), 0.0, 1.0)
        utility = torch.clamp(self.meta_utility[idxs], 0.0, 1.0)
        boundary = torch.clamp(self.meta_boundary[idxs], 0.0, 1.0)
        fp = torch.clamp(self.meta_fp_rescue[idxs], 0.0, 1.0)
        cov = torch.clamp(self.meta_coverage[idxs], 0.0, 1.0)
        if current_step is None or recency_tau <= 0:
            recency = torch.ones_like(utility) * 0.5
        else:
            last = self.meta_last_step[idxs].float()
            age = torch.clamp(float(current_step) - last, min=0.0)
            recency = torch.exp(-age / float(recency_tau))

        covered = torch.zeros_like(utility)
        if cover_feats is not None and getattr(cover_feats, "numel", lambda: 0)() > 0 and self.size() > 0:
            P = l2_normalize(self.protos[idxs].float())
            C = l2_normalize(cover_feats.to(self.device).float())
            d_cover = 1.0 - torch.clamp(P @ C.T, -1.0, 1.0).max(dim=1).values
            covered = (d_cover <= float(cover_eps)).float()
        not_covered = 1.0 - covered

        redundancy = torch.zeros_like(utility)
        if idxs.numel() > 1:
            P = l2_normalize(self.protos[idxs].float())
            sim = torch.clamp(P @ P.T, -1.0, 1.0)
            sim.fill_diagonal_(-1.0)
            redundancy = torch.clamp(sim.max(dim=1).values, 0.0, 1.0)

        score = (
            0.30 * fp
            + 0.25 * utility
            + 0.20 * repeat
            + 0.10 * recency
            + 0.10 * not_covered
            + 0.10 * boundary
            + 0.05 * cov
            - 0.20 * redundancy
        )
        info = {"covered_candidates": int(covered.sum().item())}
        return score, info

    @torch.no_grad()
    def update(
        self,
        new_feats: torch.Tensor,
        *,
        tau_insert: float,
        tau_replace: float,
        allow_replace: bool = True,
        # v19 optional utility-aware retention metadata
        use_utility_retention: bool = False,
        utility_score: Optional[float] = None,
        event_id: Optional[int] = None,
        step: Optional[int] = None,
        outcome: Optional[str] = None,
        boundary_value: Optional[float] = None,
        coverage_gain: Optional[float] = None,
        cover_feats: Optional[torch.Tensor] = None,
        retention_local_k: int = 32,
        retention_cover_eps: float = 0.025,
        retention_recency_tau: float = 1024.0,
        retention_replace_margin: float = 0.00,
        retention_ema: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Insert/replace prototypes with novelty gating.
        - If not full: append sufficiently novel features; refresh metadata for repeated ones.
        - If full and utility retention is off: legacy nearest-prototype replacement.
        - If full and utility retention is on: replace weakest local prototype based on utility,
          recurrence, FP-rescue value, recency, LTM/buffer coverage, and redundancy.
        """
        out = {
            "n_in": int(new_feats.shape[0]) if new_feats is not None else 0,
            "n_added": 0,
            "n_replaced": 0,
            "n_refreshed": 0,
            "n_skipped": 0,
            "utility_retention": bool(use_utility_retention),
            "utility_replacements": 0,
            "utility_skips": 0,
            "covered_replace_candidates": 0,
        }

        if new_feats is None or new_feats.numel() == 0:
            return out

        self._ensure_meta()
        new_feats = l2_normalize(new_feats.to(self.device)).to(dtype=self.dtype)
        cur_step = int(step if step is not None else (event_id if event_id is not None else -1))
        cur_step_opt = None if cur_step < 0 else cur_step

        for i in range(new_feats.shape[0]):
            f = new_feats[i:i+1]  # [1, C]
            if self.size() == 0:
                self.protos = f.clone()
                self._append_meta(utility_score=utility_score, event_id=event_id, step=cur_step_opt, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain)
                out["n_added"] += 1
                continue

            q = l2_normalize(f).float()
            P = l2_normalize(self.protos.float())
            dist_all = 1.0 - torch.clamp(q @ P.T, -1.0, 1.0).squeeze(0)
            d0, idx0 = dist_all.min(dim=0)
            d0 = float(d0.item())
            idx0 = int(idx0.item())

            if self.size() < self.K:
                if d0 >= tau_insert:
                    self.protos = torch.cat([self.protos, f], dim=0)
                    self._append_meta(utility_score=utility_score, event_id=event_id, step=cur_step_opt, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain)
                    out["n_added"] += 1
                else:
                    self._touch_meta(idx0, utility_score=utility_score, event_id=event_id, step=cur_step_opt, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain, ema=retention_ema)
                    out["n_refreshed"] += 1
                continue

            # full
            if (not allow_replace) or d0 < tau_replace:
                # Repeated evidence should still protect the matched prototype.
                self._touch_meta(idx0, utility_score=utility_score, event_id=event_id, step=cur_step_opt, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain, ema=retention_ema)
                out["n_refreshed"] += 1
                continue

            if not bool(use_utility_retention):
                self.protos[idx0:idx0+1] = f
                self._set_meta(idx0, utility_score=utility_score, event_id=event_id, step=cur_step_opt, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain)
                out["n_replaced"] += 1
                continue

            # v19 utility-aware local replacement.
            k_local = int(max(1, min(int(retention_local_k), self.size())))
            local_idxs = torch.topk(dist_all, k=k_local, largest=False).indices
            scores, sinfo = self._retention_scores(
                local_idxs,
                current_step=cur_step_opt,
                cover_feats=cover_feats,
                cover_eps=float(retention_cover_eps),
                recency_tau=float(retention_recency_tau),
            )
            self.retention_counters["utility_candidate_evals"] = int(self.retention_counters.get("utility_candidate_evals", 0)) + int(k_local)
            self.retention_counters["covered_candidates"] = int(self.retention_counters.get("covered_candidates", 0)) + int(sinfo.get("covered_candidates", 0))
            out["covered_replace_candidates"] += int(sinfo.get("covered_candidates", 0))
            weakest_pos = int(scores.argmin().item())
            jrep = int(local_idxs[weakest_pos].item())
            weakest_score = float(scores[weakest_pos].item())

            u, b, fp, cov = self._meta_values(utility_score=utility_score, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain)
            cand_score = float(0.30 * fp + 0.25 * u + 0.20 * 0.35 + 0.10 * 1.0 + 0.10 * 1.0 + 0.10 * b + 0.05 * cov)
            if cand_score >= weakest_score + float(retention_replace_margin):
                self.protos[jrep:jrep+1] = f
                self._set_meta(jrep, utility_score=utility_score, event_id=event_id, step=cur_step_opt, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain)
                out["n_replaced"] += 1
                out["utility_replacements"] += 1
                self.retention_counters["utility_replacements"] = int(self.retention_counters.get("utility_replacements", 0)) + 1
            else:
                # Do not discard the evidence entirely; refresh the nearest anchor.
                self._touch_meta(idx0, utility_score=utility_score, event_id=event_id, step=cur_step_opt, outcome=outcome, boundary_value=boundary_value, coverage_gain=coverage_gain, ema=retention_ema)
                out["n_refreshed"] += 1
                out["n_skipped"] += 1
                out["utility_skips"] += 1
                self.retention_counters["utility_skips"] = int(self.retention_counters.get("utility_skips", 0)) + 1

        return out

    @torch.no_grad()
    def prune_covered_by(
        self,
        cover_feats: torch.Tensor,
        *,
        max_dist: float = 0.02,
        min_keep: int = 32,
    ) -> Dict[str, Any]:
        """Conservatively retire STM entries covered by LTM prototypes.

        This is optional and disabled by default in v18/v19. It should only be used
        after active LTM promotion is verified, because STM remains the primary
        fast learner.
        """
        out = {"before": int(self.size()), "after": int(self.size()), "removed": 0}
        if self.size() == 0 or cover_feats is None or cover_feats.numel() == 0:
            return out
        if self.size() <= int(max(0, min_keep)):
            return out
        self._ensure_meta()
        P = l2_normalize(self.protos.float())
        C = l2_normalize(cover_feats.to(self.device).float())
        d = 1.0 - torch.clamp(P @ C.T, -1.0, 1.0).max(dim=1).values
        keep = d > float(max_dist)
        # Preserve at least min_keep prototypes by keeping the least-covered entries.
        if int(keep.sum().item()) < int(min_keep):
            order = torch.argsort(d, descending=True)[: int(min_keep)]
            keep = torch.zeros_like(keep, dtype=torch.bool)
            keep[order] = True
        self.protos = self.protos[keep].to(dtype=self.dtype).clone()
        self.meta_count = self.meta_count[keep].clone()
        self.meta_utility = self.meta_utility[keep].clone()
        self.meta_boundary = self.meta_boundary[keep].clone()
        self.meta_fp_rescue = self.meta_fp_rescue[keep].clone()
        self.meta_coverage = self.meta_coverage[keep].clone()
        self.meta_first_step = self.meta_first_step[keep].clone()
        self.meta_last_step = self.meta_last_step[keep].clone()
        self.meta_last_event = self.meta_last_event[keep].clone()
        self.meta_distinct_events = self.meta_distinct_events[keep].clone()
        out["after"] = int(self.size())
        out["removed"] = int(out["before"] - out["after"])
        return out

    def utility_summary(self) -> Dict[str, Any]:
        self._ensure_meta()
        n = int(self.size())
        if n <= 0:
            return {"n": 0, **{k: int(v) for k, v in self.retention_counters.items()}}
        return {
            "n": n,
            "count_mean": float(self.meta_count.mean().item()),
            "utility_mean": float(self.meta_utility.mean().item()),
            "fp_rescue_mean": float(self.meta_fp_rescue.mean().item()),
            "boundary_mean": float(self.meta_boundary.mean().item()),
            "distinct_events_mean": float(self.meta_distinct_events.mean().item()),
            **{k: int(v) for k, v in self.retention_counters.items()},
        }


# ---------------------------------------------------------------------------
#  v18 Maturation buffer for dual-timescale STM -> LTM consolidation
# ---------------------------------------------------------------------------

class MaturationBuffer:
    """Maturation buffer between relaxed STM and stable LTM.

    The buffer is not directly used for retrieval. It accumulates candidate-level
    evidence for verified-normal patch motifs and promotes only mature,
    non-redundant candidates into LTM. Recurrence is counted across distinct
    queried events, not across many patches from one image.
    """

    def __init__(
        self,
        K: int,
        feat_dim: int,
        device: torch.device,
        *,
        merge_eps: float = 0.04,
        replace_eps: float = 0.12,
        max_event_prototypes: int = 8,
        event_merge_eps: float = 0.03,
    ):
        self.K = int(max(1, K))
        self.device = device
        self.feat_dim = int(feat_dim)
        self.merge_eps = float(max(1e-6, merge_eps))
        self.replace_eps = float(max(self.merge_eps, replace_eps))
        self.max_event_prototypes = int(max(1, max_event_prototypes))
        self.event_merge_eps = float(max(1e-6, event_merge_eps))
        self.protos = torch.zeros((self.K, self.feat_dim), device=self.device, dtype=torch.float32)
        self.count = torch.zeros((self.K,), device=self.device, dtype=torch.long)
        self.distinct_events = torch.zeros((self.K,), device=self.device, dtype=torch.long)
        self.usefulness_sum = torch.zeros((self.K,), device=self.device, dtype=torch.float32)
        self.boundary_sum = torch.zeros((self.K,), device=self.device, dtype=torch.float32)
        self.fp_rescue_sum = torch.zeros((self.K,), device=self.device, dtype=torch.float32)
        self.coverage_sum = torch.zeros((self.K,), device=self.device, dtype=torch.float32)
        self.last_event = torch.full((self.K,), -1, device=self.device, dtype=torch.long)
        self.first_step = torch.full((self.K,), -1, device=self.device, dtype=torch.long)
        self.last_step = torch.full((self.K,), -1, device=self.device, dtype=torch.long)
        self.promoted_count = torch.zeros((self.K,), device=self.device, dtype=torch.long)
        self.n = 0

    def size(self) -> int:
        return int(self.n)

    @staticmethod
    def _quality(
        distinct_events: torch.Tensor,
        usefulness_avg: torch.Tensor,
        fp_rescue_avg: Optional[torch.Tensor] = None,
        boundary_avg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rep = torch.log1p(distinct_events.float())
        q = 0.55 * rep + 0.35 * usefulness_avg.float()
        if fp_rescue_avg is not None:
            q = q + 0.25 * fp_rescue_avg.float()
        if boundary_avg is not None:
            q = q + 0.15 * boundary_avg.float()
        return q

    @torch.no_grad()
    def _event_level_prototypes(self, q: torch.Tensor) -> torch.Tensor:
        """Compress one queried image/layer/view into a few diverse candidates.

        This avoids fake maturation caused by many similar patches inside the
        same image. Recurrence is still counted only across distinct events.
        The input order is already top-k anomaly-map order, so the first pass
        greedily keeps non-near-duplicate high-value patches before using a
        farthest-point fill if needed.
        """
        q = l2_normalize(q.to(self.device).float())
        if q.numel() == 0:
            return q
        max_k = int(min(int(self.max_event_prototypes), int(q.shape[0])))
        selected: List[int] = []
        for i in range(int(q.shape[0])):
            if len(selected) >= max_k:
                break
            if not selected:
                selected.append(i)
                continue
            S = q[torch.tensor(selected, device=q.device, dtype=torch.long)]
            dmin = float((1.0 - torch.clamp(q[i:i+1] @ S.T, -1.0, 1.0)).min().item())
            if dmin >= float(self.event_merge_eps):
                selected.append(i)
        if len(selected) < max_k:
            sim = torch.clamp(q @ q.T, -1.0, 1.0)
            if not selected:
                selected = [int(sim.mean(dim=1).argmin().item())]
            min_dist = 1.0 - sim[torch.tensor(selected, device=q.device, dtype=torch.long)].min(dim=0).values
            while len(selected) < max_k:
                cand = int(min_dist.argmax().item())
                if cand in selected:
                    break
                selected.append(cand)
                min_dist = torch.minimum(min_dist, 1.0 - sim[cand])
        idx = torch.tensor(selected[:max_k], device=q.device, dtype=torch.long)
        return q[idx]

    @torch.no_grad()
    def update(
        self,
        q: torch.Tensor,
        *,
        usefulness: float,
        event_id: int,
        step: Optional[int] = None,
        outcome: Optional[str] = None,
        boundary_value: Optional[float] = None,
        coverage_gain: Optional[float] = None,
    ) -> Dict[str, int]:
        out = {"inserted": 0, "merged": 0, "replaced": 0, "skipped": 0, "n_in": int(q.shape[0]) if q is not None else 0, "n_event_protos": 0}
        if q is None or q.numel() == 0:
            return out
        q = self._event_level_prototypes(q)
        out["n_event_protos"] = int(q.shape[0])
        u = float(np.clip(float(usefulness), 0.0, 1.0))
        b = float(np.clip(float(u if boundary_value is None else boundary_value), 0.0, 1.0))
        cov = float(np.clip(float(0.0 if coverage_gain is None else coverage_gain), 0.0, 1.0))
        fp_rescue = 1.0 if str(outcome or "").upper() == "FP" else 0.0
        s = int(event_id if step is None else step)

        for i in range(int(q.shape[0])):
            f = q[i:i+1]
            if self.n == 0:
                self.protos[0:1] = f
                self.count[0] = 1
                self.distinct_events[0] = 1
                self.usefulness_sum[0] = u
                self.boundary_sum[0] = b
                self.fp_rescue_sum[0] = fp_rescue
                self.coverage_sum[0] = cov
                self.last_event[0] = int(event_id)
                self.first_step[0] = s
                self.last_step[0] = s
                self.promoted_count[0] = 0
                self.n = 1
                out["inserted"] += 1
                continue

            P = l2_normalize(self.protos[: self.n])
            d = 1.0 - torch.clamp(f @ P.T, -1.0, 1.0)
            d0, j = d.min(dim=1)
            d0 = float(d0.item())
            j = int(j.item())

            if d0 <= self.merge_eps:
                w_old = float(max(1, int(self.count[j].item())))
                proto = l2_normalize((P[j] * w_old + f.squeeze(0)) / (w_old + 1.0))
                self.protos[j] = proto
                self.count[j] += 1
                self.usefulness_sum[j] += u
                self.boundary_sum[j] += b
                self.fp_rescue_sum[j] += fp_rescue
                self.coverage_sum[j] += cov
                if int(self.last_event[j].item()) != int(event_id):
                    self.distinct_events[j] += 1
                    self.last_event[j] = int(event_id)
                self.last_step[j] = s
                out["merged"] += 1
                continue

            if self.n < self.K:
                k = int(self.n)
                self.protos[k:k+1] = f
                self.count[k] = 1
                self.distinct_events[k] = 1
                self.usefulness_sum[k] = u
                self.boundary_sum[k] = b
                self.fp_rescue_sum[k] = fp_rescue
                self.coverage_sum[k] = cov
                self.last_event[k] = int(event_id)
                self.first_step[k] = s
                self.last_step[k] = s
                self.promoted_count[k] = 0
                self.n += 1
                out["inserted"] += 1
                continue

            count = self.count[: self.n].clamp_min(1).float()
            usefulness_avg = self.usefulness_sum[: self.n] / count
            fp_avg = self.fp_rescue_sum[: self.n] / count
            boundary_avg = self.boundary_sum[: self.n] / count
            quality = self._quality(self.distinct_events[: self.n], usefulness_avg, fp_avg, boundary_avg)
            worst = int(quality.argmin().item())
            worst_q = float(quality[worst].item())
            cand_q = float(0.55 * math.log1p(1.0) + 0.35 * u + 0.25 * fp_rescue + 0.15 * b)
            if d0 >= self.replace_eps and cand_q > worst_q:
                self.protos[worst:worst+1] = f
                self.count[worst] = 1
                self.distinct_events[worst] = 1
                self.usefulness_sum[worst] = u
                self.boundary_sum[worst] = b
                self.fp_rescue_sum[worst] = fp_rescue
                self.coverage_sum[worst] = cov
                self.last_event[worst] = int(event_id)
                self.first_step[worst] = s
                self.last_step[worst] = s
                self.promoted_count[worst] = 0
                out["replaced"] += 1
            else:
                out["skipped"] += 1
        return out

    @torch.no_grad()
    def get_promotion_candidates(
        self,
        *,
        fixed_ref: Optional[torch.Tensor] = None,
        existing_ltm: Optional[torch.Tensor] = None,
        min_repeat: int = 2,
        min_usefulness: float = 0.15,
        score_threshold: float = 0.45,
        max_promote: int = 8,
        rescue_min_fp: float = 0.15,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        info = {
            "candidate_count": int(self.n),
            "eligible_count": 0,
            "promote_count": 0,
            "scores": [],
            "usefulness_avg": [],
            "boundary_avg": [],
            "fp_rescue_avg": [],
            "coverage_gain": [],
            "distinct_events": [],
            "redundancy_ltm": [],
            "promotion_rule": "maturation_buffer",
            "promote_indices": [],
        }
        if self.n == 0:
            return torch.empty((0, self.protos.shape[1]), device=self.device, dtype=torch.float32), info

        P = l2_normalize(self.protos[: self.n].float())
        count = self.count[: self.n].clamp_min(1).float()
        distinct = self.distinct_events[: self.n].float()
        usefulness_avg = torch.clamp(self.usefulness_sum[: self.n] / count, 0.0, 1.0)
        boundary_avg = torch.clamp(self.boundary_sum[: self.n] / count, 0.0, 1.0)
        fp_rescue_avg = torch.clamp(self.fp_rescue_sum[: self.n] / count, 0.0, 1.0)
        cov_stored = torch.clamp(self.coverage_sum[: self.n] / count, 0.0, 1.0)
        rep = torch.clamp(torch.log1p(distinct) / math.log1p(max(3, int(min_repeat) + 1)), 0.0, 1.0)

        if fixed_ref is not None and getattr(fixed_ref, 'numel', lambda: 0)() > 0:
            F = l2_normalize(fixed_ref.float().to(self.device))
            sim_fixed = torch.clamp(P @ F.T, -1.0, 1.0).max(dim=1).values
        else:
            sim_fixed = torch.zeros((P.shape[0],), device=self.device)

        if existing_ltm is not None and getattr(existing_ltm, 'numel', lambda: 0)() > 0:
            L = l2_normalize(existing_ltm.float().to(self.device))
            sim_ltm = torch.clamp(P @ L.T, -1.0, 1.0).max(dim=1).values
        else:
            sim_ltm = torch.zeros((P.shape[0],), device=self.device)

        # Coverage is strongest when a candidate is not already covered by fixed or LTM memory.
        cov_now = torch.clamp(1.0 - torch.maximum(sim_fixed, sim_ltm), 0.0, 1.0)
        cov = torch.maximum(cov_stored, cov_now)
        ltm_novelty = torch.clamp(1.0 - sim_ltm, 0.0, 1.0)
        redundancy_ltm = sim_ltm
        persistence = torch.clamp((self.last_step[: self.n].float() - self.first_step[: self.n].float()).clamp_min(0.0) / 256.0, 0.0, 1.0)

        score = (
            0.30 * rep
            + 0.20 * usefulness_avg
            + 0.20 * fp_rescue_avg
            + 0.15 * cov
            + 0.10 * boundary_avg
            + 0.05 * persistence
            - 0.20 * redundancy_ltm
        )

        # Main path: recurring verified-normal motif. Rescue path: repeated hard-normal pattern.
        recurrent_ok = distinct >= float(min_repeat)
        rescue_ok = (distinct >= max(1.0, float(min_repeat) - 1.0)) & (fp_rescue_avg >= float(rescue_min_fp)) & (boundary_avg >= 0.25)
        novel_to_ltm = ltm_novelty >= 0.012
        useful_ok = usefulness_avg >= float(min_usefulness)
        eligible = (recurrent_ok | rescue_ok) & useful_ok & (score >= float(score_threshold)) & novel_to_ltm

        info["eligible_count"] = int(eligible.sum().item())
        info["scores"] = [float(x) for x in score.detach().cpu().tolist()]
        info["usefulness_avg"] = [float(x) for x in usefulness_avg.detach().cpu().tolist()]
        info["boundary_avg"] = [float(x) for x in boundary_avg.detach().cpu().tolist()]
        info["fp_rescue_avg"] = [float(x) for x in fp_rescue_avg.detach().cpu().tolist()]
        info["coverage_gain"] = [float(x) for x in cov.detach().cpu().tolist()]
        info["redundancy_ltm"] = [float(x) for x in redundancy_ltm.detach().cpu().tolist()]
        info["distinct_events"] = [int(x) for x in self.distinct_events[: self.n].detach().cpu().tolist()]

        if not bool(eligible.any()):
            return torch.empty((0, P.shape[1]), device=self.device, dtype=torch.float32), info

        idx = torch.nonzero(eligible, as_tuple=False).view(-1)
        s = score[idx]
        order = idx[torch.argsort(s, descending=True)]
        order = order[: int(max(1, max_promote))]
        sel = P[order]
        info["promote_count"] = int(sel.shape[0])
        info["promote_indices"] = [int(i) for i in order.detach().cpu().tolist()]
        return sel, info


    @torch.no_grad()
    def get_retrieval_prototypes(
        self,
        *,
        min_repeat: int = 2,
        min_usefulness: float = 0.10,
        score_threshold: float = 0.25,
        max_protos: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Return maturity-gated provisional prototypes for optional retrieval.

        This is less strict than LTM promotion but still requires cross-event
        recurrence and non-trivial usefulness. The caller applies an additional
        distance penalty based on (1 - maturity_score), so buffer candidates can
        rescue hard normals without becoming an unrestricted second STM.
        """
        info: Dict[str, Any] = {
            "candidate_count": int(self.n),
            "eligible_count": 0,
            "retrieval_count": 0,
            "scores": [],
            "distinct_events": [],
            "usefulness_avg": [],
            "boundary_avg": [],
            "fp_rescue_avg": [],
        }
        if self.n == 0:
            return (
                torch.empty((0, self.feat_dim), device=self.device, dtype=torch.float32),
                torch.empty((0,), device=self.device, dtype=torch.float32),
                info,
            )

        P = l2_normalize(self.protos[: self.n].float())
        count = self.count[: self.n].clamp_min(1).float()
        distinct = self.distinct_events[: self.n].float()
        usefulness_avg = torch.clamp(self.usefulness_sum[: self.n] / count, 0.0, 1.0)
        boundary_avg = torch.clamp(self.boundary_sum[: self.n] / count, 0.0, 1.0)
        fp_rescue_avg = torch.clamp(self.fp_rescue_sum[: self.n] / count, 0.0, 1.0)

        rep = torch.clamp(torch.log1p(distinct) / math.log1p(max(3, int(min_repeat) + 1)), 0.0, 1.0)
        score = (
            0.45 * rep
            + 0.25 * usefulness_avg
            + 0.15 * boundary_avg
            + 0.15 * fp_rescue_avg
        )

        eligible = (
            (distinct >= float(max(1, min_repeat)))
            & (usefulness_avg >= float(min_usefulness))
            & (score >= float(score_threshold))
        )
        info["eligible_count"] = int(eligible.sum().item())
        info["scores"] = [float(x) for x in score.detach().cpu().tolist()]
        info["distinct_events"] = [int(x) for x in self.distinct_events[: self.n].detach().cpu().tolist()]
        info["usefulness_avg"] = [float(x) for x in usefulness_avg.detach().cpu().tolist()]
        info["boundary_avg"] = [float(x) for x in boundary_avg.detach().cpu().tolist()]
        info["fp_rescue_avg"] = [float(x) for x in fp_rescue_avg.detach().cpu().tolist()]

        if not bool(eligible.any()):
            return (
                torch.empty((0, P.shape[1]), device=self.device, dtype=torch.float32),
                torch.empty((0,), device=self.device, dtype=torch.float32),
                info,
            )

        idx = torch.nonzero(eligible, as_tuple=False).view(-1)
        order = idx[torch.argsort(score[idx], descending=True)]
        order = order[: int(max(1, max_protos))]
        sel = P[order]
        sel_score = torch.clamp(score[order], 0.0, 1.0)
        info["retrieval_count"] = int(sel.shape[0])
        return sel, sel_score, info



# Backward-compatible name used by older serialization / initialization code.
TemporalPrototypeTracker = MaturationBuffer

# ---------------------------------------------------------------------------
#  Correction Bank (Boundary Refinement) + Set-Membership Core-Normal Selection
# ---------------------------------------------------------------------------

@dataclass
class CorrectionConfig:
    """Local boundary refinement via signed patch-level corrections.

    This replaces the fragile idea of a 'defect prototype bank' as a class.
    Instead, we store sparse *mistake anchors*:
      - FP (pred defect, true normal): negative correction (push score down locally)
      - FN (pred normal, true defect): positive correction (push score up locally)

    Corrections are applied as an additive term on the patch scores:
        s(p) = dN(p) + defect_term(p) + corr_weight * C(p)

    Note: if enabled, you typically want to disable defect bank scoring/updates
    (keep_defect_bank=False) to avoid conflicting signals.
    """
    enabled: bool = True
    K_correction: int = 256

    corr_weight: float = 1.0

    # Correction magnitude derived from image-level margin |score-theta|
    beta_scale: float = 0.5
    beta_min: float = 0.02
    beta_max: float = 0.20

    # RBF kernel radius in cosine-distance units
    radius0: float = 0.10
    radius_min: float = 0.05
    radius_max: float = 0.25

    # Precision (confidence) weight per correction anchor
    kappa0: float = 1.0
    kappa_inc: float = 0.10
    kappa_max: float = 5.0

    # Merge anchors if very close (set-membership style compactness)
    merge_eps: float = 0.05

    # Store only if selected patch is sufficiently 'interesting'
    min_dN_for_store_fp: float = 0.10
    min_dN_for_store_fn: float = 0.12

    # If False, defect bank is considered redundant when corrections are enabled
    keep_defect_bank: bool = False

    # v21: outcome-aware, patch-weighted, sign-aware correction-bank updates.
    # These defaults are conservative and only affect already-queried samples.
    enable_sign_aware_write: bool = True
    enable_patch_weighted_beta: bool = True
    enable_sign_aware_retention: bool = True
    outcome_weight_fp: float = 1.30   # confirmed false positives get stronger negative local repair
    outcome_weight_fn: float = 1.20   # confirmed false negatives get stronger positive local repair
    outcome_weight_tn: float = 0.60   # true normals can refine normal-side boundary, but weakly
    outcome_weight_tp: float = 0.80   # true defects can refine defect-side boundary, but weakly
    patch_beta_floor: float = 0.75
    patch_beta_cap: float = 1.25
    patch_beta_power: float = 1.0
    corr_replace_margin: float = 0.0


class SignedCorrectionCodebook:
    """Fixed-capacity signed correction anchors with sign-aware retention.

    v21 keeps the original signed local residual field, but makes the write path
    less magnitude-only by adding:
      - outcome-aware correction strength (FP/FN/TN/TP weights are applied before update);
      - patch-local beta weighting from selected top-k patch distances;
      - sign-aware replacement, preferring to replace the weakest anchor of the same sign;
      - lightweight diagnostics for positive/negative anchors and outcome counts.
    """

    def __init__(self, K: int, feat_dim: int, device: torch.device, cfg: CorrectionConfig):
        self.K = int(K)
        self.device = device
        self.cfg = cfg

        self.protos = torch.zeros((self.K, feat_dim), device=self.device, dtype=torch.float32)
        self.beta = torch.zeros((self.K,), device=self.device, dtype=torch.float32)
        self.radius = torch.full((self.K,), float(cfg.radius0), device=self.device, dtype=torch.float32)
        self.kappa = torch.full((self.K,), float(cfg.kappa0), device=self.device, dtype=torch.float32)
        self.count = torch.zeros((self.K,), device=self.device, dtype=torch.long)

        # v21 metadata for sign-aware correction retention and diagnostics.
        self.outcome_fp = torch.zeros((self.K,), device=self.device, dtype=torch.long)
        self.outcome_fn = torch.zeros((self.K,), device=self.device, dtype=torch.long)
        self.outcome_tn = torch.zeros((self.K,), device=self.device, dtype=torch.long)
        self.outcome_tp = torch.zeros((self.K,), device=self.device, dtype=torch.long)
        self.last_step = torch.full((self.K,), -1, device=self.device, dtype=torch.long)
        self.n = 0
        self.counters: Dict[str, int] = {
            "inserted_total": 0,
            "merged_total": 0,
            "replaced_total": 0,
            "skipped_total": 0,
            "same_sign_replaced_total": 0,
            "cross_sign_replaced_total": 0,
            "same_sign_merge_total": 0,
        }

    def size(self) -> int:
        return int(self.n)

    @staticmethod
    def _cos_dist(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # q: [N, C], p: [M, C]  (both L2-normalized)
        sim = torch.clamp(q @ p.t(), -1.0, 1.0)
        return 1.0 - sim

    def _outcome_key(self, outcome: Optional[str]) -> str:
        o = str(outcome or "").upper()
        if o in {"FP", "FN", "TN", "TP"}:
            return o
        return "UNK"

    def _bump_outcome(self, j: int, outcome: Optional[str]) -> None:
        o = self._outcome_key(outcome)
        if o == "FP":
            self.outcome_fp[j] += 1
        elif o == "FN":
            self.outcome_fn[j] += 1
        elif o == "TN":
            self.outcome_tn[j] += 1
        elif o == "TP":
            self.outcome_tp[j] += 1

    def _write_slot(
        self,
        j: int,
        q_i: torch.Tensor,
        beta_i: torch.Tensor,
        radius_i: torch.Tensor,
        kappa_i: torch.Tensor,
        *,
        outcome: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        self.protos[j] = q_i
        self.beta[j] = beta_i
        self.radius[j] = radius_i
        self.kappa[j] = kappa_i
        self.count[j] = 1
        self.outcome_fp[j] = 0
        self.outcome_fn[j] = 0
        self.outcome_tn[j] = 0
        self.outcome_tp[j] = 0
        self._bump_outcome(j, outcome)
        self.last_step[j] = int(-1 if step is None else step)

    @torch.no_grad()
    def apply(self, q: torch.Tensor) -> torch.Tensor:
        """Return per-vector correction C(q) in the same units as patch scores."""
        if self.n == 0 or (not self.cfg.enabled):
            return torch.zeros((q.shape[0],), device=q.device, dtype=torch.float32)

        p = self.protos[: self.n]                  # [M, C]
        beta = self.beta[: self.n]                 # [M]
        rad = self.radius[: self.n].clamp_min(1e-6)  # [M]
        kappa = self.kappa[: self.n].clamp_min(1e-6) # [M]

        dist = self._cos_dist(q, p)                # [N, M]
        w = torch.exp(-(dist * dist) / (2.0 * (rad[None, :] ** 2))) * kappa[None, :]
        corr = (w * beta[None, :]).sum(dim=1) / (w.sum(dim=1) + 1e-6)
        return corr

    @torch.no_grad()
    def update(
        self,
        q: torch.Tensor,
        beta: torch.Tensor,
        *,
        radius: Optional[torch.Tensor] = None,
        kappa: Optional[torch.Tensor] = None,
        outcome: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, int]:
        """Insert/merge/replace anchors with sign-aware retention.

        Backward compatibility: if v21 sign-aware retention is disabled in cfg,
        this reduces to the old merge+insert+weakest-replace behavior.
        """
        if q.numel() == 0:
            return {"inserted": 0, "merged": 0, "replaced": 0, "skipped": 0,
                    "same_sign_replaced": 0, "cross_sign_replaced": 0}

        q = l2_normalize(q.to(self.device))
        beta = beta.to(self.device).float().clamp(-float(self.cfg.beta_max), float(self.cfg.beta_max))

        M = int(q.shape[0])
        if radius is None:
            radius = torch.full((M,), float(self.cfg.radius0), device=self.device, dtype=torch.float32)
        else:
            radius = radius.to(self.device).float()
        radius = radius.clamp(float(self.cfg.radius_min), float(self.cfg.radius_max))

        if kappa is None:
            kappa = torch.full((M,), float(self.cfg.kappa0), device=self.device, dtype=torch.float32)
        else:
            kappa = kappa.to(self.device).float()
        kappa = kappa.clamp(0.1, float(self.cfg.kappa_max))

        inserted = merged = replaced = skipped = 0
        same_sign_replaced = cross_sign_replaced = 0
        same_sign_merge = 0
        outcome_key = self._outcome_key(outcome)
        new_is_mistake = 1.0 if outcome_key in {"FP", "FN"} else 0.0

        for i in range(M):
            b_i = float(beta[i].item())
            if abs(b_i) < float(self.cfg.beta_min - 1e-5):
                skipped += 1
                continue

            if self.n == 0:
                self._write_slot(0, q[i], beta[i], radius[i], kappa[i], outcome=outcome, step=step)
                self.n = 1
                inserted += 1
                continue

            p = self.protos[: self.n]
            dist = self._cos_dist(q[i : i + 1], p).squeeze(0)   # [n]
            dmin, j = torch.min(dist, dim=0)
            j = int(j.item())
            dmin_val = float(dmin.item())

            same_sign = (float(self.beta[j].item()) * b_i) > 0.0

            if dmin_val <= float(self.cfg.merge_eps) and same_sign:
                c = float(self.count[j].item())
                self.protos[j] = l2_normalize((c * self.protos[j] + q[i]) / (c + 1.0))
                self.beta[j] = torch.clamp((c * self.beta[j] + beta[i]) / (c + 1.0),
                                           -float(self.cfg.beta_max), float(self.cfg.beta_max))
                self.radius[j] = torch.clamp((c * self.radius[j] + radius[i]) / (c + 1.0),
                                             float(self.cfg.radius_min), float(self.cfg.radius_max))
                self.kappa[j] = torch.clamp(self.kappa[j] + float(self.cfg.kappa_inc),
                                            0.1, float(self.cfg.kappa_max))
                self.count[j] += 1
                self._bump_outcome(j, outcome)
                if step is not None:
                    self.last_step[j] = int(step)
                merged += 1
                same_sign_merge += 1
                continue

            if self.n < self.K:
                self._write_slot(self.n, q[i], beta[i], radius[i], kappa[i], outcome=outcome, step=step)
                self.n += 1
                inserted += 1
                continue

            # Full bank. v21: prefer replacing the weakest anchor of the same sign.
            counts = self.count[: self.n].float()
            babs = self.beta[: self.n].abs()
            mistake_count = (self.outcome_fp[: self.n] + self.outcome_fn[: self.n]).float()
            same_mask = (self.beta[: self.n] * float(b_i)) > 0.0

            if bool(getattr(self.cfg, "enable_sign_aware_retention", True)) and bool(same_mask.any().item()):
                cand_idx = torch.nonzero(same_mask, as_tuple=False).view(-1)
            else:
                cand_idx = torch.arange(self.n, device=self.device, dtype=torch.long)

            # Larger score means more worth retaining. The weakest local/same-sign
            # correction anchor is replaced first.
            retention = (
                counts[cand_idx]
                + 1.25 * mistake_count[cand_idx]
                + 0.50 * torch.clamp(babs[cand_idx] / max(float(self.cfg.beta_max), 1e-6), 0.0, 1.0)
                + 0.25 * torch.clamp(self.kappa[: self.n][cand_idx] / max(float(self.cfg.kappa_max), 1e-6), 0.0, 1.0)
            )
            pos = int(torch.argmin(retention).item())
            jrep = int(cand_idx[pos].item())
            weakest_score = float(retention[pos].item())
            new_score = float(
                1.0
                + 1.25 * new_is_mistake
                + 0.50 * min(abs(b_i) / max(float(self.cfg.beta_max), 1e-6), 1.0)
                + 0.25 * min(float(kappa[i].item()) / max(float(self.cfg.kappa_max), 1e-6), 1.0)
            )

            replace_ok = True
            if bool(getattr(self.cfg, "enable_sign_aware_retention", True)):
                replace_ok = new_score >= weakest_score + float(getattr(self.cfg, "corr_replace_margin", 0.0))
            else:
                # Legacy set-membership style: only replace if new constraint is stronger.
                replace_ok = abs(b_i) > float(babs[jrep].item()) + 1e-6

            if replace_ok:
                was_same = (float(self.beta[jrep].item()) * b_i) > 0.0
                self._write_slot(jrep, q[i], beta[i], radius[i], kappa[i], outcome=outcome, step=step)
                replaced += 1
                if was_same:
                    same_sign_replaced += 1
                else:
                    cross_sign_replaced += 1
            else:
                skipped += 1

        self.counters["inserted_total"] = int(self.counters.get("inserted_total", 0)) + int(inserted)
        self.counters["merged_total"] = int(self.counters.get("merged_total", 0)) + int(merged)
        self.counters["replaced_total"] = int(self.counters.get("replaced_total", 0)) + int(replaced)
        self.counters["skipped_total"] = int(self.counters.get("skipped_total", 0)) + int(skipped)
        self.counters["same_sign_replaced_total"] = int(self.counters.get("same_sign_replaced_total", 0)) + int(same_sign_replaced)
        self.counters["cross_sign_replaced_total"] = int(self.counters.get("cross_sign_replaced_total", 0)) + int(cross_sign_replaced)
        self.counters["same_sign_merge_total"] = int(self.counters.get("same_sign_merge_total", 0)) + int(same_sign_merge)

        return {
            "inserted": int(inserted), "merged": int(merged), "replaced": int(replaced), "skipped": int(skipped),
            "same_sign_replaced": int(same_sign_replaced), "cross_sign_replaced": int(cross_sign_replaced),
            "same_sign_merged": int(same_sign_merge),
        }

    def summary(self) -> Dict[str, Any]:
        n = int(self.n)
        out: Dict[str, Any] = {"n": n, **{k: int(v) for k, v in self.counters.items()}}
        if n <= 0:
            out.update({
                "positive": 0, "negative": 0,
                "beta_pos_mean": 0.0, "beta_neg_mean": 0.0,
                "beta_abs_mean": 0.0,
                "fp_anchor_count": 0, "fn_anchor_count": 0,
                "tn_anchor_count": 0, "tp_anchor_count": 0,
            })
            return out
        b = self.beta[:n]
        pos = b > 0
        neg = b < 0
        out.update({
            "positive": int(pos.sum().item()),
            "negative": int(neg.sum().item()),
            "beta_pos_mean": float(b[pos].mean().item()) if bool(pos.any().item()) else 0.0,
            "beta_neg_mean": float(b[neg].mean().item()) if bool(neg.any().item()) else 0.0,
            "beta_abs_mean": float(b.abs().mean().item()),
            "count_mean": float(self.count[:n].float().mean().item()),
            "fp_anchor_count": int(self.outcome_fp[:n].sum().item()),
            "fn_anchor_count": int(self.outcome_fn[:n].sum().item()),
            "tn_anchor_count": int(self.outcome_tn[:n].sum().item()),
            "tp_anchor_count": int(self.outcome_tp[:n].sum().item()),
        })
        return out


@dataclass
class SetMembershipConfig:
    """Conservative self-labeling of *core* normals (no human) to improve bank quality.

    Motivation:
      - Queried-confirmed normals concentrate near the boundary and should not dominate the
        representation of 'typical normal' (your concern).
      - Use a bounded feasible region in global feature space (CLS embedding) to admit only
        very confident, in-manifold normals into the dynamic normal bank.
    """
    enabled: bool = True

    # Only accept self-labeled normals when posterior defect is low.
    # The selftrain_core comparator uses relaxed defaults so that the baseline
    # genuinely tests unsupervised online memory update instead of degenerating
    # into a static fixed-bank detector.
    core_p_defect_max: float = 0.30

    # Require score below theta by margin_sigma * sigmaN.
    core_margin_sigma: float = 0.5

    # Global feasible radius slack multiplier (computed from support)
    radius_slack: float = 2.00

    # Novelty threshold in global cosine-distance units (avoid only near-duplicates)
    core_global_novelty_min: float = 0.005

    # Max stored globals per (class, mode)
    core_max_buf: int = 256

    # How many typical patches per layer to store when accepted
    core_patches_per_layer: int = 16

    # Use separate taus for core-normal inserts.
    core_tau_insert: float = 0.25
    core_tau_replace: float = 0.40


class CoreNormalFeasibleSet:
    """Set-membership style feasible region in global (CLS) feature space."""

    def __init__(self, seed_globals: List[torch.Tensor], cfg: SetMembershipConfig):
        self.cfg = cfg
        self.buf: List[torch.Tensor] = []
        for g in seed_globals:
            if g is None:
                continue
            self.buf.append(g.detach().float().cpu())

        if len(self.buf) == 0:
            # fallback: will reject until we have some core normals
            self.center = None
            self.radius = 0.0
        else:
            G = torch.stack(self.buf, dim=0)  # [M, C], already normalized
            self.center = l2_normalize(G.mean(dim=0, keepdim=False))
            # radius from seed set + slack
            dist = 1.0 - torch.clamp(G @ self.center, -1.0, 1.0)
            self.radius = float(dist.max().item()) * float(cfg.radius_slack) + 1e-6

        # Diagnostics for the unsupervised self-training comparator.
        # These counters make it explicit whether the baseline actually writes
        # pseudo-normal evidence into dynamic memory, and why candidates are rejected.
        self.stats: Dict[str, Any] = {
            "seen": 0,
            "accepted": 0,
            "reject_disabled": 0,
            "reject_no_center": 0,
            "reject_p_defect": 0,
            "reject_margin": 0,
            "reject_radius": 0,
            "reject_novelty": 0,
            "last_d_center": None,
            "last_dmin_global": None,
            "last_margin_required": None,
            "last_score_margin": None,
            "radius": float(self.radius),
        }

    def _bump(self, key: str) -> None:
        self.stats[key] = int(self.stats.get(key, 0)) + 1

    def diagnostics(self) -> Dict[str, Any]:
        out = dict(self.stats)
        out["buf_size"] = int(len(self.buf))
        out["radius"] = float(self.radius)
        out["has_center"] = bool(self.center is not None)
        return out

    def _cos_dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: [C], b: [M, C]
        return 1.0 - torch.clamp(b @ a, -1.0, 1.0)

    def accept(self, g: torch.Tensor, *, score: float, theta: float, sigmaN: float, p_defect: float) -> bool:
        self._bump("seen")
        if not self.cfg.enabled:
            self._bump("reject_disabled")
            return False
        if self.center is None:
            self._bump("reject_no_center")
            return False

        if float(p_defect) > float(self.cfg.core_p_defect_max):
            self._bump("reject_p_defect")
            return False

        sigma = max(float(sigmaN), 1e-3)
        margin_required = float(self.cfg.core_margin_sigma) * sigma
        score_margin = float(theta) - float(score)
        self.stats["last_margin_required"] = float(margin_required)
        self.stats["last_score_margin"] = float(score_margin)
        if score_margin < margin_required:
            self._bump("reject_margin")
            return False

        g_cpu = g.detach().float().cpu()
        # already L2-normalized by backbone, but keep safe
        g_cpu = l2_normalize(g_cpu.unsqueeze(0)).squeeze(0)

        d_center = float((1.0 - torch.clamp(torch.dot(g_cpu, self.center), -1.0, 1.0)).item())
        self.stats["last_d_center"] = float(d_center)
        self.stats["radius"] = float(self.radius)
        if d_center > float(self.radius):
            self._bump("reject_radius")
            return False

        # novelty w.r.t stored core globals; the threshold only filters near duplicates.
        if len(self.buf) > 0:
            G = torch.stack(self.buf, dim=0)
            dmin = float(self._cos_dist(g_cpu, G).min().item())
            self.stats["last_dmin_global"] = float(dmin)
            if dmin < float(self.cfg.core_global_novelty_min):
                self._bump("reject_novelty")
                return False

        # accept -> update buffer and feasible region (compact)
        self.buf.append(g_cpu)
        if len(self.buf) > int(self.cfg.core_max_buf):
            # drop oldest (bounded memory)
            self.buf.pop(0)

        G = torch.stack(self.buf, dim=0)
        self.center = l2_normalize(G.mean(dim=0, keepdim=False))
        dist = 1.0 - torch.clamp(G @ self.center, -1.0, 1.0)
        self.radius = float(dist.max().item()) * float(self.cfg.radius_slack) + 1e-6
        self.stats["radius"] = float(self.radius)
        self._bump("accepted")
        return True

    def is_inlier(self, g: torch.Tensor, *, slack: float = 1.0) -> bool:
        """Check cluster-conditional membership without updating the feasible set.

        Used for contamination resistance: accept writes to normal/correction banks only
        when the *global* feature remains within the mode's feasible region.
        """
        if (not self.cfg.enabled) or (self.center is None):
            return False
        g_cpu = g.detach().float().cpu()
        g_cpu = l2_normalize(g_cpu.unsqueeze(0)).squeeze(0)
        d_center = float((1.0 - torch.clamp(torch.dot(g_cpu, self.center), -1.0, 1.0)).item())
        return d_center <= float(self.radius) * float(max(slack, 0.0))

# ---------------------------------------------------------------------------
#  Concept codebook: stable-ish cluster IDs for patches (bounded)
# ---------------------------------------------------------------------------

class ConceptCodebook:
    """
    Lightweight clustering-by-nearest:
    - keeps up to K concept centers
    - assigns each patch to nearest concept id
    Used only for RB statistics and gating.
    """
    def __init__(self, K: int, feat_dim: int, device: torch.device):
        self.cb = FixedCapacityCodebook(K=K, feat_dim=feat_dim, device=device)

    def size(self) -> int:
        return self.cb.size()

    @torch.no_grad()
    def assign_ids(self, q: torch.Tensor) -> List[int]:
        _, idx = self.cb.nearest(q)
        return [int(i) for i in idx.detach().cpu().tolist()]

    @torch.no_grad()
    def update(self, q: torch.Tensor, tau_insert: float = 0.15, tau_replace: float = 0.25) -> None:
        # allow replacement to keep concepts representative
        self.cb.update(q, tau_insert=tau_insert, tau_replace=tau_replace, allow_replace=True)

# ---------------------------------------------------------------------------
#  Patch VMB (dual-bank scoring)
#  Key design: stable baseline (distance-to-normal) + bounded defect affinity boost.
# ---------------------------------------------------------------------------

class PatchVMB:
    """
    Phase-2 PatchVMB with VisionAD-inspired improvements:
      1) Fixed normal bank + dynamic normal bank (per mode, per view)
      2) Support augmentation + Pseudo multi-view (banks are per view)
      3) Layer fusion (map-level) + view fusion (map-level)
      4) Optional mode-indexed (CLS-space k-means) routing inside a class
    Notes:
      - Concept codebook is shared per class+layer (keeps RB concept IDs stable).
      - Defect bank is shared per class+view+layer (not mode-specific).
      - Updates (FP/FN) can be applied across all views by mapping patch indices.
    """

    def __init__(
        self,
        device: str = "cuda",
        K_normal_fixed: int = 4096,
        K_normal_dyn: int = 1024,
        K_normal_ltm: int = 512,
        K_defect: int = 256,
        K_concept: int = 4096,
        knn_k_normal: int = 5,
        *,
        corr_cfg: Optional[CorrectionConfig] = None,
        use_defect_bank: bool = True,
    disable_defect_bank: bool = False,
        ltm_min_stm_size: int = 16,
        ltm_min_repeat: int = 3,
    ):
        self.device = torch.device(device)
        self.bank_dtype = torch.float16  # default fp16 banks
        self.K_normal_fixed = int(K_normal_fixed)
        self.K_normal_dyn = int(K_normal_dyn)
        self.K_normal_ltm = int(K_normal_ltm)
        self.K_defect = int(K_defect)
        self.K_concept = int(K_concept)
        self.knn_k_normal = int(knn_k_normal)
        self.ltm_min_stm_size = int(max(1, ltm_min_stm_size))
        self.ltm_min_repeat = int(max(2, ltm_min_repeat))
        # Correction bank (boundary refinement)
        self.corr_cfg = corr_cfg if corr_cfg is not None else CorrectionConfig()
        self.use_defect_bank = bool(use_defect_bank)
        if self.corr_cfg.enabled and (not self.corr_cfg.keep_defect_bank):
            # If corrections are enabled, defect prototypes are usually redundant/noisy.
            self.use_defect_bank = False


        # Shared concept codebook: class -> list[layer] -> ConceptCodebook
        self.concept_cb: Dict[str, List[ConceptCodebook]] = {}

        # Normal banks (fixed + dyn): class -> mode -> view -> list[layer] -> codebook
        self.normal_fixed: Dict[str, Dict[int, Dict[str, List[FixedCapacityCodebook]]]] = {}
        self.normal_dyn: Dict[str, Dict[int, Dict[str, List[FixedCapacityCodebook]]]] = {}
        self.normal_dyn_ltm: Dict[str, Dict[int, Dict[str, List[FixedCapacityCodebook]]]] = {}

        # Defect bank: class -> view -> list[layer] -> codebook
        self.defect_cb: Dict[str, Dict[str, List[FixedCapacityCodebook]]] = {}

        # Correction bank (signed): class -> view -> list[layer] -> SignedCorrectionCodebook
        self.corr_cb: Dict[str, Dict[str, List[SignedCorrectionCodebook]]] = {}


        # Per class: CLS-space centroids for modes (CPU tensor [K, C])
        self.mode_centroids: Dict[str, torch.Tensor] = {}
        self._ltm_pending_updates: Dict[str, int] = {}
        self._ltm_last_event: Dict[str, int] = {}

        # v8: pressure-driven, usefulness-aware LTM promotion
        self.ltm_pressure_ratio: float = 0.85
        self.ltm_min_pending_updates: int = 64
        self.ltm_min_usefulness: float = 0.25
        self.ltm_score_threshold: float = 0.55
        self.ltm_candidate_merge_eps: float = 0.06
        self.ltm_candidate_capacity: int = max(64, int(self.K_normal_ltm) * 8)
        # v9: keep LTM as a shadow sidecar first.
        # - STM remains the actual fast learner and retrieval path (v5-style).
        # - Temporal candidates are still tracked so we can measure whether repeated,
        #   useful motifs exist, but they do not yet alter retrieval geometry.
        self.shadow_ltm: bool = True
        self.enable_ltm_retrieval: bool = False
        self.enable_ltm_promotion: bool = False

        # v18: maturation-buffered STM -> LTM consolidation.
        # STM remains broad and active; this candidate buffer decides what is
        # mature enough to become stable LTM.  When enabled, promotion is driven
        # mainly by distinct-event recurrence, usefulness, FP-rescue value, and
        # coverage gain rather than by raw STM fullness alone.
        self.enable_maturation_buffer: bool = False
        self.mature_merge_radius: float = float(self.ltm_candidate_merge_eps)
        self.mature_event_merge_eps: float = 0.03
        self.mature_max_event_protos: int = 8
        self.mature_event_prototypes_per_layer: int = self.mature_max_event_protos  # legacy alias
        self.mature_min_events: int = int(max(2, self.ltm_min_repeat))
        self.mature_score_threshold: float = 0.45
        self.mature_score_thr: float = self.mature_score_threshold  # CLI alias
        self.mature_min_usefulness: float = 0.10
        self.mature_promote_every: int = 16
        self.mature_promote_max_per_layer: int = max(4, int(self.K_normal_ltm) // 8)
        self.mature_rescue_min_fp: float = 0.15
        self.mature_max_candidates: int = int(self.ltm_candidate_capacity)
        self.enable_stm_retirement: bool = False
        self.stm_retire_eps: float = 0.025
        self.stm_retire_ltm_eps: float = self.stm_retire_eps
        self.stm_retire_min_keep: int = max(16, int(self.K_normal_dyn) // 4)

        # Compact LTM retrieval should not be evaluated like a dense STM bank.
        # Default k=1 gives each mature prototype a fair nearest-neighbor chance.
        self.ltm_retrieval_k: int = 1

        # Optional v18_fix2: maturity-gated provisional retrieval from the
        # maturation buffer. Disabled by default so the buffer remains a pure
        # consolidation mechanism unless explicitly tested.
        self.enable_maturation_buffer_retrieval: bool = False
        self.mature_retrieval_min_events: int = 2
        self.mature_retrieval_min_usefulness: float = 0.10
        self.mature_retrieval_score_thr: float = 0.25
        self.mature_retrieval_penalty: float = 0.01
        self.mature_retrieval_max_protos: int = 256
        self.mature_retrieval_k: int = 1

        # v19: utility-aware STM retention. Disabled by default so v18 behavior
        # is exactly recoverable unless explicitly enabled.
        self.enable_utility_aware_stm_retention: bool = False
        self.stm_retention_local_k: int = 32
        self.stm_retention_recency_tau: float = 1024.0
        self.stm_retention_ltm_cover_eps: float = 0.025
        self.stm_retention_replace_margin: float = 0.03
        self.stm_retention_ema: float = 0.10

        # Accumulates patch-source statistics when LTM / buffer retrieval is active.
        self._ltm_retrieval_stats: Dict[str, Dict[str, int]] = {}

        self.normal_dyn_candidates: Dict[str, Dict[int, Dict[str, List[TemporalPrototypeTracker]]]] = {}

    def set_mode_centroids(self, cls_name: str, centroids_cpu: torch.Tensor) -> None:
        self.mode_centroids[cls_name] = centroids_cpu.detach().cpu()

    def get_mode_centroids(self, cls_name: str) -> torch.Tensor:
        return self.mode_centroids.get(cls_name, torch.empty((0, 0)))

    @staticmethod
    @torch.no_grad()
    def _select_diverse_prototypes(pool: torch.Tensor, K: int) -> torch.Tensor:
        pool = l2_normalize(pool.float())
        if pool.shape[0] <= int(K):
            return pool
        sim = torch.clamp(pool @ pool.T, -1.0, 1.0)
        mean_sim = sim.mean(dim=1)
        first = int(mean_sim.argmax().item())
        selected = [first]
        min_dist = 1.0 - sim[first]
        while len(selected) < int(K):
            cand = int(min_dist.argmax().item())
            if cand in selected:
                break
            selected.append(cand)
            min_dist = torch.minimum(min_dist, 1.0 - sim[cand])
        idx = torch.tensor(selected, device=pool.device, dtype=torch.long)
        sel = pool[idx]
        if sel.shape[0] > int(K):
            sel = sel[: int(K)]
        return sel

    @staticmethod
    @torch.no_grad()
    def _extract_recurrent_pattern_prototypes(
        stm_feats: torch.Tensor,
        *,
        min_repeat: int = 3,
        max_patterns: int = 128,
        existing_ltm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Extract stable pattern-level prototypes from STM using coarser grouped candidates.

        Instead of relying on raw patch-level micro-clusters, build a small set of coarse
        candidate groups over STM, then only promote groups that are sufficiently populated,
        reasonably compact, and not redundant with existing LTM prototypes.
        """
        X = l2_normalize(stm_feats.float())
        N = int(X.shape[0])
        info = {
            "radius": None,
            "recurrent_points": 0,
            "n_patterns": 0,
            "discard_only": False,
            "accepted_clusters": 0,
            "redundant_clusters": 0,
            "rejected_noncompact": 0,
            "candidate_groups": 0,
        }
        if N < max(6, int(min_repeat) * 2):
            return torch.empty((0, X.shape[1]), device=X.device, dtype=X.dtype), X, info

        sim = torch.clamp(X @ X.T, -1.0, 1.0)
        D = 1.0 - sim
        D.fill_diagonal_(10.0)
        nn1 = D.min(dim=1).values
        valid = nn1[torch.isfinite(nn1)]
        if valid.numel() == 0:
            return torch.empty((0, X.shape[1]), device=X.device, dtype=X.dtype), X, info

        # Coarser grouping radius than the old raw-patch matcher.
        base_radius = float(np.clip(float(torch.quantile(valid, 0.65).item()) * 2.75, 0.03, 0.16))
        info["radius"] = base_radius

        # Build coarse grouped candidates by clustering STM around a small set of centers.
        n_groups = int(np.clip(round(math.sqrt(N)), 4, max(4, min(int(max_patterns), N))))
        centers = PatchVMB._select_diverse_prototypes(X, n_groups)
        if centers.shape[0] == 0:
            return torch.empty((0, X.shape[1]), device=X.device, dtype=X.dtype), X, info

        # A few assignment/refinement rounds are enough and much more robust than raw micro-clusters.
        for _ in range(4):
            assign = torch.clamp(X @ centers.T, -1.0, 1.0).argmax(dim=1)
            new_centers = []
            for g in range(int(centers.shape[0])):
                idx = torch.nonzero(assign == g, as_tuple=False).view(-1)
                if idx.numel() == 0:
                    new_centers.append(centers[g])
                    continue
                proto = l2_normalize(X[idx].mean(dim=0, keepdim=True)).squeeze(0)
                new_centers.append(proto)
            centers = torch.stack(new_centers, dim=0)

        assign = torch.clamp(X @ centers.T, -1.0, 1.0).argmax(dim=1)
        info["candidate_groups"] = int(centers.shape[0])

        ltm_ref = None
        if existing_ltm is not None and getattr(existing_ltm, 'numel', lambda: 0)() > 0:
            ltm_ref = l2_normalize(existing_ltm.float())

        keep_mask = torch.ones((N,), device=X.device, dtype=torch.bool)
        protos: List[torch.Tensor] = []
        recurrent_points = 0

        # Process larger groups first so promotion favors repeated structure.
        group_order: List[Tuple[int, int]] = []
        for g in range(int(centers.shape[0])):
            idx = torch.nonzero(assign == g, as_tuple=False).view(-1)
            group_order.append((int(idx.numel()), g))
        group_order.sort(reverse=True)

        for group_size, g in group_order:
            if len(protos) >= int(max_patterns):
                break
            if group_size < int(min_repeat):
                continue
            idx = torch.nonzero(assign == g, as_tuple=False).view(-1)
            if idx.numel() < int(min_repeat):
                continue
            cluster = X[idx]
            recurrent_points += int(cluster.shape[0])

            proto = l2_normalize(cluster.mean(dim=0, keepdim=True)).squeeze(0)
            d_to_proto = 1.0 - torch.clamp(cluster @ proto.unsqueeze(1), -1.0, 1.0).squeeze(1)
            compact_q80 = float(torch.quantile(d_to_proto, 0.80).item()) if cluster.shape[0] > 1 else 0.0
            compact_mean = float(d_to_proto.mean().item())
            representative_idx = int(d_to_proto.argmin().item())
            rep_proto = cluster[representative_idx]

            compact_ok = (
                compact_q80 <= max(base_radius, 0.06) and
                compact_mean <= max(base_radius * 0.80, 0.04)
            )
            if not compact_ok:
                info["rejected_noncompact"] += 1
                continue

            redundant = False
            if ltm_ref is not None and ltm_ref.numel() > 0:
                d_ltm = 1.0 - torch.clamp(rep_proto.unsqueeze(0) @ ltm_ref.T, -1.0, 1.0)
                min_d_ltm = float(d_ltm.min().item())
                redundant = min_d_ltm < max(base_radius * 0.50, 0.012)

            if redundant:
                info["redundant_clusters"] += 1
                keep_mask[idx] = False
                continue

            protos.append(rep_proto)
            keep_mask[idx] = False
            info["accepted_clusters"] += 1

        proto_t = torch.stack(protos, dim=0) if len(protos) > 0 else torch.empty((0, X.shape[1]), device=X.device, dtype=X.dtype)
        residual = X[keep_mask]
        info["recurrent_points"] = int(recurrent_points)
        info["n_patterns"] = int(proto_t.shape[0])
        return proto_t, residual, info

    @staticmethod
    @torch.no_grad()
    def _compress_stm_residual(stm_feats: torch.Tensor, keep_k: int) -> torch.Tensor:
        X = l2_normalize(stm_feats.float())
        if X.shape[0] <= int(keep_k):
            return X
        return PatchVMB._select_diverse_prototypes(X, int(keep_k))

    @torch.no_grad()
    def shadow_evaluate_ltm(self, cls_name: str) -> Dict[str, Any]:
        """Non-mutating diagnostics for the shadow LTM sidecar.

        v9 intentionally does not let LTM prototypes affect retrieval yet. This
        method answers a narrower question: if we *were* to promote from the
        current temporal candidate tracker under the configured thresholds, how
        much promotable structure exists?
        """
        stats: Dict[str, Any] = {
            "shadow_mode": True,
            "eligible_count": 0,
            "promote_count": 0,
            "candidate_count": 0,
            "stm_total": 0,
            "ltm_total": 0,
            "pressure_triggered": False,
            "pending_updates": int(self._ltm_pending_updates.get(cls_name, 0)),
            "layer_details": [],
        }
        if cls_name not in self.normal_dyn:
            return stats

        pressure_k = max(int(self.ltm_min_stm_size), int(math.ceil(float(self.ltm_pressure_ratio) * float(max(1, self.K_normal_dyn)))))
        min_pending = int(self.mature_promote_every if self.enable_maturation_buffer else self.ltm_min_pending_updates)
        pending_ok = int(self._ltm_pending_updates.get(cls_name, 0)) >= int(max(1, min_pending))

        for mode, view_dict in self.normal_dyn[cls_name].items():
            for view, cbs in view_dict.items():
                for l, stm_cb in enumerate(cbs):
                    stm_n = int(stm_cb.size())
                    stats["stm_total"] += stm_n
                    stats["ltm_total"] += int(self.normal_dyn_ltm[cls_name][mode][view][l].size())
                    cand_cb = self.normal_dyn_candidates[cls_name][mode][view][l]
                    fixed_ref = None
                    fixed_cb = self.normal_fixed[cls_name][mode][view][l]
                    if fixed_cb.size() > 0:
                        fixed_ref = fixed_cb.protos.float()
                    cand_protos, pinfo = cand_cb.get_promotion_candidates(
                        fixed_ref=fixed_ref,
                        existing_ltm=None,
                        min_repeat=int(self.ltm_min_repeat),
                        min_usefulness=float(self.ltm_min_usefulness),
                        score_threshold=float(self.ltm_score_threshold),
                        max_promote=max(4, int(self.K_normal_ltm) // 8),
                    )
                    if stm_n >= int(pressure_k) and pending_ok:
                        stats["pressure_triggered"] = True
                    stats["candidate_count"] += int(pinfo.get("candidate_count", 0))
                    stats["eligible_count"] += int(pinfo.get("eligible_count", 0))
                    stats["promote_count"] += int(cand_protos.shape[0])
                    stats["layer_details"].append({
                        "mode": int(mode),
                        "view": str(view),
                        "layer": int(l),
                        "stm_n": int(stm_n),
                        "candidate_count": int(pinfo.get("candidate_count", 0)),
                        "eligible_count": int(pinfo.get("eligible_count", 0)),
                        "would_promote": int(cand_protos.shape[0]),
                    })
        return stats

    @torch.no_grad()
    def consolidate_normal_dyn(self, cls_name: str) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "cls_name": str(cls_name),
            "consolidated": False,
            "promoted_layers": 0,
            "stm_before": 0,
            "stm_after": 0,
            "ltm_before": 0,
            "ltm_after": 0,
            "skipped_small_stm": 0,
            "skip_not_full": 0,
            "discard_only_layers": 0,
            "layers": {},
            "pending_updates_before": int(self._ltm_pending_updates.get(cls_name, 0)),
            "pending_updates_after": int(self._ltm_pending_updates.get(cls_name, 0)),
            "maturation_buffer_enabled": bool(self.enable_maturation_buffer),
        }
        if cls_name not in self.normal_dyn:
            return stats

        pressure_k = max(int(self.ltm_min_stm_size), int(math.ceil(float(self.ltm_pressure_ratio) * float(max(1, self.K_normal_dyn)))))
        pending_threshold = int(self.mature_promote_every) if self.enable_maturation_buffer else int(self.ltm_min_pending_updates)
        pending_ok = int(self._ltm_pending_updates.get(cls_name, 0)) >= int(max(1, pending_threshold))

        for mode, view_dict in self.normal_dyn[cls_name].items():
            for view, layer_cbs in view_dict.items():
                for l, stm_cb in enumerate(layer_cbs):
                    ltm_cb = self.normal_dyn_ltm[cls_name][mode][view][l]
                    cand_cb = self.normal_dyn_candidates[cls_name][mode][view][l]
                    stm_n = int(stm_cb.size())
                    ltm_n = int(ltm_cb.size())
                    stats["stm_before"] += stm_n
                    stats["ltm_before"] += ltm_n
                    key = f"mode{int(mode)}::{view}::layer{int(l)}"

                    pressure_ok = bool(stm_n >= int(pressure_k))
                    if self.enable_maturation_buffer:
                        # v18: maturing candidates can be promoted once enough new verified-normal
                        # events have passed, even if STM is not close to full. This avoids the
                        # previous dormant-LTM failure mode.
                        pressure_ok = True
                    if not pressure_ok:
                        stats["skip_not_full"] += 1
                        stats["layers"][key] = {
                            "stm_before": stm_n,
                            "ltm_before": ltm_n,
                            "skipped": True,
                            "reason": "stm_not_under_pressure",
                            "candidate_count": int(cand_cb.size()),
                            "maturation_buffer": bool(self.enable_maturation_buffer),
                        }
                        continue
                    if not pending_ok:
                        stats["skip_not_full"] += 1
                        stats["layers"][key] = {
                            "stm_before": stm_n,
                            "ltm_before": ltm_n,
                            "skipped": True,
                            "reason": "not_enough_new_updates",
                            "candidate_count": int(cand_cb.size()),
                        }
                        continue

                    fixed_ref = self.normal_fixed[cls_name][mode][view][l].protos.float()
                    existing_ltm = ltm_cb.protos.float() if ltm_n > 0 else None
                    min_repeat = int(self.mature_min_events if self.enable_maturation_buffer else self.ltm_min_repeat)
                    min_use = float(self.mature_min_usefulness if self.enable_maturation_buffer else self.ltm_min_usefulness)
                    score_thr = float(getattr(self, "mature_score_threshold", getattr(self, "mature_score_thr", 0.45)) if self.enable_maturation_buffer else self.ltm_score_threshold)
                    max_promote = int(self.mature_promote_max_per_layer if self.enable_maturation_buffer else max(4, int(self.K_normal_ltm) // 8))
                    try:
                        pattern_protos, patt_stats = cand_cb.get_promotion_candidates(
                            fixed_ref=fixed_ref,
                            existing_ltm=existing_ltm,
                            min_repeat=min_repeat,
                            min_usefulness=min_use,
                            score_threshold=score_thr,
                            max_promote=max_promote,
                            rescue_min_fp=float(self.mature_rescue_min_fp),
                        )
                    except TypeError:
                        pattern_protos, patt_stats = cand_cb.get_promotion_candidates(
                            fixed_ref=fixed_ref,
                            existing_ltm=existing_ltm,
                            min_repeat=min_repeat,
                            min_usefulness=min_use,
                            score_threshold=score_thr,
                            max_promote=max_promote,
                        )

                    if pattern_protos.shape[0] == 0:
                        stats["layers"][key] = {
                            "stm_before": stm_n,
                            "ltm_before": ltm_n,
                            "candidate_count": patt_stats.get("candidate_count", 0),
                            "eligible_count": patt_stats.get("eligible_count", 0),
                            "promote_count": 0,
                            "stm_after": stm_n,
                            "ltm_after": ltm_n,
                            "skipped": False,
                            "discard_only": False,
                            "reason": "no_useful_recurrent_patterns",
                        }
                        stats["stm_after"] += stm_n
                        stats["ltm_after"] += ltm_n
                        continue

                    if ltm_n == 0:
                        pool = pattern_protos
                    else:
                        pool = torch.cat([ltm_cb.protos.float(), pattern_protos.float()], dim=0)
                    selected_ltm = self._select_diverse_prototypes(pool, int(self.K_normal_ltm))
                    ltm_cb.set_prototypes(selected_ltm)

                    retired = 0
                    stm_after_layer = stm_n
                    if self.enable_maturation_buffer and self.enable_stm_retirement and stm_cb.size() > 0 and ltm_cb.size() > 0:
                        retired = int(self._retire_stm_covered_by_ltm(stm_cb, ltm_cb))
                        stm_after_layer = int(stm_cb.size())

                    stats["promoted_layers"] += 1
                    stats["consolidated"] = True
                    stats["layers"][key] = {
                        "stm_before": stm_n,
                        "ltm_before": ltm_n,
                        "candidate_count": patt_stats.get("candidate_count", 0),
                        "eligible_count": patt_stats.get("eligible_count", 0),
                        "promote_count": int(pattern_protos.shape[0]),
                        "ltm_after": int(ltm_cb.size()),
                        "stm_after": int(stm_after_layer),
                        "stm_retired": int(retired),
                        "skipped": False,
                        "discard_only": False,
                        "promotion_info": {
                            "candidate_count": int(patt_stats.get("candidate_count", 0)),
                            "eligible_count": int(patt_stats.get("eligible_count", 0)),
                            "promote_count": int(patt_stats.get("promote_count", 0)),
                            "promotion_rule": str(patt_stats.get("promotion_rule", "")),
                        },
                    }
                    stats["stm_after"] += int(stm_after_layer)
                    stats["ltm_after"] += int(ltm_cb.size())

        if stats["consolidated"]:
            self._ltm_pending_updates[cls_name] = 0
        stats["pending_updates_after"] = int(self._ltm_pending_updates.get(cls_name, 0))
        if (not stats["consolidated"]) and stats["stm_after"] == 0 and stats["ltm_after"] == 0:
            stats["stm_after"] = stats["stm_before"]
            stats["ltm_after"] = stats["ltm_before"]
        return stats

    @torch.no_grad()
    def _retire_stm_covered_by_ltm(self, stm_cb: FixedCapacityCodebook, ltm_cb: FixedCapacityCodebook) -> int:
        """Conservatively retire STM entries already covered by LTM.

        Disabled by default. This should only run after successful LTM promotion,
        and keeps at least self.stm_retire_min_keep STM prototypes so the fast
        plastic memory does not lose rare evidence too aggressively.
        """
        if stm_cb.size() == 0 or ltm_cb.size() == 0:
            return 0
        X = l2_normalize(stm_cb.protos.float())
        L = l2_normalize(ltm_cb.protos.float())
        d = 1.0 - torch.clamp(X @ L.T, -1.0, 1.0).max(dim=1).values
        eps = float(getattr(self, "stm_retire_ltm_eps", getattr(self, "stm_retire_eps", 0.025)))
        keep = d > eps
        min_keep = int(max(0, min(int(getattr(self, "stm_retire_min_keep", 16)), int(X.shape[0]))))
        if int(keep.sum().item()) < min_keep:
            # Keep the least-covered prototypes as STM residuals.
            order = torch.argsort(d, descending=True)[:min_keep]
            keep = torch.zeros_like(keep, dtype=torch.bool)
            keep[order] = True
        retired = int(X.shape[0] - int(keep.sum().item()))
        if retired > 0:
            stm_cb.set_prototypes(X[keep])
        return retired

    @torch.no_grad()
    def memory_summary(self, cls_name: str) -> Dict[str, Any]:
        """Compact end-of-run memory diagnostics for paper tables/debugging."""
        out: Dict[str, Any] = {
            "class_name": str(cls_name),
            "maturation_buffer_enabled": bool(getattr(self, "enable_maturation_buffer", False)),
            "utility_aware_stm_retention_enabled": bool(getattr(self, "enable_utility_aware_stm_retention", False)),
            "stm_retention_config": {
                "local_k": int(getattr(self, "stm_retention_local_k", 32)),
                "recency_tau": float(getattr(self, "stm_retention_recency_tau", 1024.0)),
                "ltm_cover_eps": float(getattr(self, "stm_retention_ltm_cover_eps", 0.025)),
                "replace_margin": float(getattr(self, "stm_retention_replace_margin", 0.0)),
                "ema": float(getattr(self, "stm_retention_ema", 0.10)),
            },
            "enable_ltm_retrieval": bool(getattr(self, "enable_ltm_retrieval", False)),
            "enable_ltm_promotion": bool(getattr(self, "enable_ltm_promotion", False)),
            "ltm_retrieval_k": int(getattr(self, "ltm_retrieval_k", 1)),
            "enable_maturation_buffer_retrieval": bool(getattr(self, "enable_maturation_buffer_retrieval", False)),
            "mature_retrieval_penalty": float(getattr(self, "mature_retrieval_penalty", 0.01)),
            "normal_dyn_stm_total": 0,
            "normal_dyn_ltm_total": 0,
            "maturation_candidate_total": 0,
            "maturation_eligible_total": 0,
            "maturation_would_promote_total": 0,
            "ltm_pending_updates": int(self._ltm_pending_updates.get(cls_name, 0)),
            "per_view_layer": [],
            "stm_retention_stats": {
                "touches": 0,
                "utility_replacements": 0,
                "utility_skips": 0,
                "utility_candidate_evals": 0,
                "covered_candidates": 0,
                "utility_mean_weighted": 0.0,
                "fp_rescue_mean_weighted": 0.0,
                "count_mean_weighted": 0.0,
            },
        }
        out["ltm_retrieval_stats"] = copy.deepcopy(self._ltm_retrieval_stats.get(str(cls_name), {}))
        if cls_name not in self.normal_dyn:
            return out
        for mode, view_dict in self.normal_dyn.get(cls_name, {}).items():
            for view, layer_cbs in view_dict.items():
                for l, stm_cb in enumerate(layer_cbs):
                    ltm_cb = self.normal_dyn_ltm[cls_name][mode][view][l]
                    cand_cb = self.normal_dyn_candidates[cls_name][mode][view][l]
                    stm_n = int(stm_cb.size())
                    ltm_n = int(ltm_cb.size())
                    cand_n = int(cand_cb.size())
                    eligible = 0
                    would_promote = 0
                    try:
                        fixed_ref = self.normal_fixed[cls_name][mode][view][l].protos.float()
                        existing_ltm = ltm_cb.protos.float() if ltm_n > 0 else None
                        _, info = cand_cb.get_promotion_candidates(
                            fixed_ref=fixed_ref,
                            existing_ltm=existing_ltm,
                            min_repeat=int(getattr(self, "mature_min_events", self.ltm_min_repeat)) if self.enable_maturation_buffer else int(self.ltm_min_repeat),
                            min_usefulness=float(getattr(self, "mature_min_usefulness", self.ltm_min_usefulness)) if self.enable_maturation_buffer else float(self.ltm_min_usefulness),
                            score_threshold=float(getattr(self, "mature_score_threshold", getattr(self, "mature_score_thr", self.ltm_score_threshold))) if self.enable_maturation_buffer else float(self.ltm_score_threshold),
                            max_promote=int(getattr(self, "mature_promote_max_per_layer", max(4, int(self.K_normal_ltm) // 8))),
                        )
                        eligible = int(info.get("eligible_count", 0))
                        would_promote = int(info.get("promote_count", 0))
                    except Exception:
                        eligible = 0
                        would_promote = 0
                    out["normal_dyn_stm_total"] += stm_n
                    out["normal_dyn_ltm_total"] += ltm_n
                    out["maturation_candidate_total"] += cand_n
                    out["maturation_eligible_total"] += eligible
                    out["maturation_would_promote_total"] += would_promote
                    util_sum = {}
                    try:
                        util_sum = stm_cb.utility_summary()
                        rs = out["stm_retention_stats"]
                        for _k in ["touches", "utility_replacements", "utility_skips", "utility_candidate_evals", "covered_candidates"]:
                            rs[_k] = int(rs.get(_k, 0)) + int(util_sum.get(_k, 0))
                        n_util = int(util_sum.get("n", 0))
                        rs["utility_mean_weighted"] += float(util_sum.get("utility_mean", 0.0)) * n_util
                        rs["fp_rescue_mean_weighted"] += float(util_sum.get("fp_rescue_mean", 0.0)) * n_util
                        rs["count_mean_weighted"] += float(util_sum.get("count_mean", 0.0)) * n_util
                    except Exception:
                        util_sum = {}
                    out["per_view_layer"].append({
                        "mode": int(mode), "view": str(view), "layer": int(l),
                        "stm": stm_n, "ltm": ltm_n, "candidates": cand_n,
                        "eligible": eligible, "would_promote": would_promote,
                        "stm_retention": util_sum,
                    })
        try:
            rs = out.get("stm_retention_stats", {})
            denom = float(max(1, int(out.get("normal_dyn_stm_total", 0))))
            for _k in ["utility_mean_weighted", "fp_rescue_mean_weighted", "count_mean_weighted"]:
                rs[_k] = float(rs.get(_k, 0.0)) / denom
        except Exception:
            pass

        # v21: correction-bank diagnostics for local boundary refinement.
        corr_summary: Dict[str, Any] = {
            "enabled": bool(getattr(self.corr_cfg, "enabled", False)),
            "sign_aware_write": bool(getattr(self.corr_cfg, "enable_sign_aware_write", False)),
            "patch_weighted_beta": bool(getattr(self.corr_cfg, "enable_patch_weighted_beta", False)),
            "sign_aware_retention": bool(getattr(self.corr_cfg, "enable_sign_aware_retention", False)),
            "total": 0,
            "positive": 0,
            "negative": 0,
            "fp_anchor_count": 0,
            "fn_anchor_count": 0,
            "tn_anchor_count": 0,
            "tp_anchor_count": 0,
            "inserted_total": 0,
            "merged_total": 0,
            "replaced_total": 0,
            "skipped_total": 0,
            "same_sign_replaced_total": 0,
            "cross_sign_replaced_total": 0,
            "per_view_layer": [],
        }
        try:
            for vname, layer_cbs in self.corr_cb.get(cls_name, {}).items():
                for l, cb in enumerate(layer_cbs):
                    csum = cb.summary() if hasattr(cb, "summary") else {"n": int(cb.size())}
                    corr_summary["total"] += int(csum.get("n", 0))
                    corr_summary["positive"] += int(csum.get("positive", 0))
                    corr_summary["negative"] += int(csum.get("negative", 0))
                    for _k in ["fp_anchor_count", "fn_anchor_count", "tn_anchor_count", "tp_anchor_count",
                               "inserted_total", "merged_total", "replaced_total", "skipped_total",
                               "same_sign_replaced_total", "cross_sign_replaced_total"]:
                        corr_summary[_k] = int(corr_summary.get(_k, 0)) + int(csum.get(_k, 0))
                    corr_summary["per_view_layer"].append({"view": str(vname), "layer": int(l), **csum})
        except Exception:
            pass
        out["correction_bank"] = corr_summary
        return out


    def _init_concepts_if_needed(self, cls_name: str, feats_per_layer: List[torch.Tensor]) -> None:
        if cls_name in self.concept_cb:
            return
        n_layers = len(feats_per_layer)
        feat_dim = int(feats_per_layer[0].shape[1])
        self.concept_cb[cls_name] = [ConceptCodebook(self.K_concept, feat_dim, self.device) for _ in range(n_layers)]

    def _ensure_mode_view_normals(self, cls_name: str, mode: int, view: str, feats_per_layer: List[torch.Tensor]) -> None:
        self._init_concepts_if_needed(cls_name, feats_per_layer)
        n_layers = len(feats_per_layer)
        feat_dim = int(feats_per_layer[0].shape[1])

        if cls_name not in self.normal_fixed:
            self.normal_fixed[cls_name] = {}
            self.normal_dyn[cls_name] = {}
            self.normal_dyn_ltm[cls_name] = {}
            self.normal_dyn_candidates[cls_name] = {}

        if mode not in self.normal_fixed[cls_name]:
            self.normal_fixed[cls_name][mode] = {}
            self.normal_dyn[cls_name][mode] = {}
            self.normal_dyn_ltm[cls_name][mode] = {}
            self.normal_dyn_candidates[cls_name][mode] = {}

        if view not in self.normal_fixed[cls_name][mode]:
            self.normal_fixed[cls_name][mode][view] = [
                FixedCapacityCodebook(self.K_normal_fixed, feat_dim, self.device, dtype=self.bank_dtype) for _ in range(n_layers)
            ]
            self.normal_dyn[cls_name][mode][view] = [
                FixedCapacityCodebook(self.K_normal_dyn, feat_dim, self.device, dtype=self.bank_dtype) for _ in range(n_layers)
            ]
            self.normal_dyn_ltm[cls_name][mode][view] = [
                FixedCapacityCodebook(self.K_normal_ltm, feat_dim, self.device, dtype=self.bank_dtype) for _ in range(n_layers)
            ]
            self.normal_dyn_candidates[cls_name][mode][view] = [
                TemporalPrototypeTracker(self.ltm_candidate_capacity, feat_dim, self.device, merge_eps=self.ltm_candidate_merge_eps, max_event_prototypes=self.mature_max_event_protos, event_merge_eps=self.mature_event_merge_eps) for _ in range(n_layers)
            ]

        if cls_name not in self.defect_cb:
            self.defect_cb[cls_name] = {}

        if view not in self.defect_cb[cls_name]:
            self.defect_cb[cls_name][view] = [
                FixedCapacityCodebook(self.K_defect, feat_dim, self.device, dtype=self.bank_dtype) for _ in range(n_layers)
            ]

        if cls_name not in self.corr_cb:
            self.corr_cb[cls_name] = {}

        if view not in self.corr_cb[cls_name]:
            Kc = int(max(1, self.corr_cfg.K_correction))
            self.corr_cb[cls_name][view] = [
                SignedCorrectionCodebook(K=Kc, feat_dim=feat_dim, device=self.device, cfg=self.corr_cfg) for _ in range(n_layers)
            ]

    @torch.no_grad()
    def seed_concepts_from_support(
        self,
        cls_name: str,
        support_feats_identity: List[List[torch.Tensor]],
        *,
        concept_seed_patches: int = 2048,
        seed: int = 0,
    ) -> None:
        """Seed shared concept codebooks using identity-view support features."""
        assert len(support_feats_identity) > 0
        self._init_concepts_if_needed(cls_name, support_feats_identity[0])
        rng = np.random.RandomState(seed)
        n_layers = len(support_feats_identity[0])

        for l in range(n_layers):
            # Memory-safe sampling: avoid concatenating all patches from all support images.
            samples = []
            per_img_cap = int(max(64, min(4096, concept_seed_patches // max(1, len(support_feats_identity)))))
            for sf in support_feats_identity:
                t = l2_normalize(sf[l].to(self.device))
                m = int(t.shape[0])
                if m == 0:
                    continue
                k = min(per_img_cap, m)
                idx = rng.choice(m, size=k, replace=False)
                samples.append(t[torch.from_numpy(idx).to(device=self.device, dtype=torch.long)])
            if len(samples) == 0:
                continue
            pool = torch.cat(samples, dim=0)
            M = int(pool.shape[0])
            KC = min(int(concept_seed_patches), M)
            idxC = rng.choice(M, size=KC, replace=False)
            initC = pool[torch.from_numpy(idxC).to(device=self.device, dtype=torch.long)]
            self.concept_cb[cls_name][l].update(initC, tau_insert=0.10, tau_replace=0.30)

    @torch.no_grad()
    def build_fixed_normal_from_support(
        self,
        cls_name: str,
        support_feats: List[List[torch.Tensor]],
        *,
        mode: int,
        view: str,
        normal_seed_patches: int = 8192,
        seed: int = 0,
    ) -> None:
        """Seed FIXED normal bank for a specific (mode, view)."""
        assert len(support_feats) > 0
        self._ensure_mode_view_normals(cls_name, mode, view, support_feats[0])
        rng = np.random.RandomState(seed)
        n_layers = len(support_feats[0])

        for l in range(n_layers):
            # Memory-safe sampling: bound patch pool size.
            samples = []
            per_img_cap = int(max(128, min(8192, normal_seed_patches // max(1, len(support_feats)))))
            for sf in support_feats:
                t = l2_normalize(sf[l].to(self.device))
                m = int(t.shape[0])
                if m == 0:
                    continue
                k = min(per_img_cap, m)
                idx = rng.choice(m, size=k, replace=False)
                samples.append(t[torch.from_numpy(idx).to(device=self.device, dtype=torch.long)])
            if len(samples) == 0:
                continue
            pool = torch.cat(samples, dim=0)
            M = int(pool.shape[0])
            KN = min(int(normal_seed_patches), M)
            idxN = rng.choice(M, size=KN, replace=False)
            initN = pool[torch.from_numpy(idxN).to(device=self.device, dtype=torch.long)]
            # fixed bank: no replacement
            self.normal_fixed[cls_name][mode][view][l].update(initN, tau_insert=0.0, tau_replace=1.0, allow_replace=False)

    def _dN_min_fixed_dyn(
        self, cls_name: str, mode: int, view: str, layer: int, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return dN = min(d_fixed, d_STM[, d_LTM][, d_buffer_eff]).

        v18_fix2 keeps STM broad/plastic, makes compact LTM retrieval fair
        through a separate k, and optionally lets mature buffer candidates
        participate in retrieval with a maturity penalty.
        """
        cbF = self.normal_fixed[cls_name][mode][view][layer]
        cbS = self.normal_dyn[cls_name][mode][view][layer]
        cbL = self.normal_dyn_ltm[cls_name][mode][view][layer]

        d_fixed, _ = cbF.nearest(q, relax=True, k=self.knn_k_normal)
        d = d_fixed
        source_dists: List[torch.Tensor] = [d_fixed]
        source_names: List[str] = ["fixed"]

        d_stm = torch.full_like(d_fixed, 10.0)
        if cbS.size() > 0:
            d_stm, _ = cbS.nearest(q, relax=True, k=self.knn_k_normal)
            d = torch.minimum(d, d_stm)
        source_dists.append(d_stm)
        source_names.append("stm")

        if self.enable_ltm_retrieval and cbL.size() > 0:
            ltm_k = int(max(1, getattr(self, "ltm_retrieval_k", 1)))
            if ltm_k <= 1:
                d_ltm, _ = cbL.nearest(q, relax=False)
            else:
                d_ltm, _ = cbL.nearest(q, relax=True, k=ltm_k)
            d = torch.minimum(d, d_ltm)
            source_dists.append(d_ltm)
            source_names.append("ltm")

        buffer_active = False
        buffer_retrieval_info: Dict[str, Any] = {}
        if bool(getattr(self, "enable_maturation_buffer_retrieval", False)):
            cand_layers = self.normal_dyn_candidates.get(cls_name, {}).get(mode, {}).get(view, None)
            cand_cb = cand_layers[layer] if (cand_layers is not None and layer < len(cand_layers)) else None
            if cand_cb is not None and cand_cb.size() > 0:
                buf_protos, buf_scores, buffer_retrieval_info = cand_cb.get_retrieval_prototypes(
                    min_repeat=int(getattr(self, "mature_retrieval_min_events", 2)),
                    min_usefulness=float(getattr(self, "mature_retrieval_min_usefulness", 0.10)),
                    score_threshold=float(getattr(self, "mature_retrieval_score_thr", 0.25)),
                    max_protos=int(getattr(self, "mature_retrieval_max_protos", 256)),
                )
                if buf_protos.numel() > 0:
                    qn = l2_normalize(q).float()
                    P = l2_normalize(buf_protos.to(q.device).float())
                    scores = torch.clamp(buf_scores.to(q.device).float(), 0.0, 1.0)
                    dist = 1.0 - torch.clamp(qn @ P.T, -1.0, 1.0)
                    penalty = float(max(0.0, getattr(self, "mature_retrieval_penalty", 0.01)))
                    eff = dist + penalty * (1.0 - scores.view(1, -1))
                    k_buf = int(max(1, getattr(self, "mature_retrieval_k", 1)))
                    if k_buf <= 1:
                        d_buf = eff.min(dim=1).values
                    else:
                        kk = int(min(k_buf, eff.shape[1]))
                        d_buf = torch.topk(eff, k=kk, largest=False, dim=1).values.mean(dim=1)
                    d = torch.minimum(d, d_buf)
                    source_dists.append(d_buf)
                    source_names.append("buffer")
                    buffer_active = True

        if (self.enable_ltm_retrieval and cbL.size() > 0) or buffer_active:
            src = torch.stack(source_dists, dim=0).argmin(dim=0)
            st = self._ltm_retrieval_stats.setdefault(str(cls_name), {
                "patches": 0,
                "fixed_wins": 0,
                "stm_wins": 0,
                "ltm_wins": 0,
                "buffer_wins": 0,
                "buffer_active_patches": 0,
                "buffer_candidate_batches": 0,
                "buffer_candidate_total": 0,
            })
            for key in ["patches", "fixed_wins", "stm_wins", "ltm_wins", "buffer_wins",
                        "buffer_active_patches", "buffer_candidate_batches", "buffer_candidate_total"]:
                st.setdefault(key, 0)
            st["patches"] += int(src.numel())
            for si, name in enumerate(source_names):
                st[f"{name}_wins"] = int(st.get(f"{name}_wins", 0)) + int((src == si).sum().item())
            if buffer_active:
                st["buffer_active_patches"] += int(q.shape[0])
                st["buffer_candidate_batches"] += 1
                st["buffer_candidate_total"] += int(buffer_retrieval_info.get("retrieval_count", 0))

        return d, torch.full((q.shape[0],), -1, device=q.device, dtype=torch.long)

    @torch.no_grad()
    def score_single_view(
        self,
        cls_name: str,
        feats_per_layer: List[torch.Tensor],
        *,
        mode: int,
        view: str,
        topk_patches: int,
        mode_scoring: str,
        lam: float,
        tau_close: float,
        gamma: float,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Compute per-layer patch score maps for one view.
        Returns:
          s_maps: list[layer] of [N_patches] tensors (higher = more anomalous)
          per_layer_summary: for identity view only: topk concept ids + patch idx etc
        """
        self._ensure_mode_view_normals(cls_name, mode, view, feats_per_layer)
        n_layers = len(feats_per_layer)
        per_layer: Dict[str, Any] = {}

        s_maps: List[torch.Tensor] = []

        for l in range(n_layers):
            q = l2_normalize(feats_per_layer[l].to(self.device))  # [N, C]
            dN, _ = self._dN_min_fixed_dyn(cls_name, mode, view, l, q)

            # top-k by dN
            k = min(int(topk_patches), int(dN.numel()))
            top_dN, top_idx = torch.topk(dN, k=k, largest=True)

            # defect boost (bounded, only computed on top-k)
            defect_empty = ((not self.use_defect_bank) or (self.defect_cb[cls_name][view][l].size() == 0))
            if mode_scoring == "contrast":
                if defect_empty:
                    s_top = top_dN
                    top_dD = torch.ones_like(top_dN)
                else:
                    q_top = q[top_idx]
                    top_dD, _ = self.defect_cb[cls_name][view][l].nearest(q_top, relax=True)
                    s_top = top_dN - gamma * top_dD
            else:
                if defect_empty:
                    top_dD = torch.ones_like(top_dN)
                    boost = torch.zeros_like(top_dN)
                else:
                    q_top = q[top_idx]
                    top_dD, _ = self.defect_cb[cls_name][view][l].nearest(q_top, relax=True)
                    boost = torch.relu(torch.tensor(tau_close, device=self.device) - top_dD)
                s_top = top_dN + lam * boost

            # correction bank (boundary refinement): apply only on top-k for efficiency
            corr_top = torch.zeros_like(top_dN)
            if self.corr_cfg.enabled and (self.corr_cb[cls_name][view][l].size() > 0):
                q_top2 = q[top_idx]
                corr_top = self.corr_cb[cls_name][view][l].apply(q_top2)
                s_top = s_top + float(self.corr_cfg.corr_weight) * corr_top

            # full patch map: baseline dN + boost on top-k
            s_map = dN.clone()
            s_map[top_idx] = s_top
            s_maps.append(s_map)

            # For RB stats/gating we only need identity-view summary; caller decides.
            per_layer[f"layer{l}"] = {
                "topk_dN_max": float(top_dN.max().item()) if top_dN.numel() else 0.0,
                "topk_dN_mean": float(top_dN.mean().item()) if top_dN.numel() else 0.0,
                "topk_dD_min": float(top_dD.min().item()) if top_dD.numel() else 1.0,
                "corr_top_mean": float(corr_top.mean().item()) if corr_top.numel() else 0.0,
                "corr_top_max": float(corr_top.max().item()) if corr_top.numel() else 0.0,
                "corr_top_min": float(corr_top.min().item()) if corr_top.numel() else 0.0,
                "corr_bank_size": int(self.corr_cb[cls_name][view][l].size()) if self.corr_cfg.enabled else 0,
                "use_defect_bank": bool(self.use_defect_bank),
                "topk_patch_idx": [int(i) for i in top_idx.detach().cpu().tolist()],
            }

        return s_maps, per_layer

    @staticmethod
    def fuse_layers(
        s_maps: List[torch.Tensor],
        fusion: str = "mean",
    ) -> torch.Tensor:
        """Fuse per-layer maps -> a single patch map."""
        if len(s_maps) == 0:
            return torch.zeros((0,), device="cpu")
        S = torch.stack(s_maps, dim=0)  # [L, N]
        if fusion == "max":
            return S.max(dim=0).values
        # default: mean
        return S.mean(dim=0)

    @staticmethod
    def aggregate_map(
        s_map: torch.Tensor,
        *,
        agg: str = "topk_mean",
        topk: int = 64,
        quantile: float = 0.95,
    ) -> float:
        if s_map.numel() == 0:
            return 0.0
        v = s_map.flatten()
        if agg == "max":
            return float(v.max().item())
        k = min(int(topk), int(v.numel()))
        v_top = torch.topk(v, k=k, largest=True).values
        if agg == "spiky_quantile":
            return float(torch.quantile(v_top, torch.tensor(quantile, device=v_top.device)).item())
        # default: topk_mean
        return float(v_top.mean().item())

    @torch.no_grad()
    def score_sample_multiview(  # This function is important !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self,
        cls_name: str,
        view_to_feats: Dict[str, List[torch.Tensor]],
        *,
        mode: int,
        views: Dict[str, ViewSpec],
        topk_patches: int = 32,
        mode_scoring: str = "boost",
        lam: float = 0.8,
        tau_close: float = 0.20,
        gamma: float = 1.0,
        layer_fusion: str = "mean",
        view_fusion: str = "mean",
        img_agg: str = "topk_mean",
        img_topk: int = 64,
        img_quantile: float = 0.95,
        concept_assign_view: str = "id",
        img_hw: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, Dict[str, Any], Dict[str, List[torch.Tensor]]]:
        """
        Multi-view scoring:
          - compute per-view fused patch map (after layer fusion)
          - align each view map to identity coordinates (for flips)
          - fuse views at map-level
          - aggregate to score_img

        Returns:
          score_img
          A_map_stats: includes identity-view topk patch idx + concept ids (for RB)
          view_to_feats: returned so caller can update banks across views
        """
        assert concept_assign_view in view_to_feats, "concept_assign_view must exist in view_to_feats"
        # compute maps per view
        view_maps_aligned: List[torch.Tensor] = []
        per_view_debug: Dict[str, Any] = {}

        aspect = None
        if img_hw is not None:
            H_img, W_img = int(img_hw[0]), int(img_hw[1])
            if W_img > 0:
                aspect = float(H_img) / float(W_img)

        # grid size from concept_assign_view (assumed consistent)
        n_patches = int(view_to_feats[concept_assign_view][0].shape[0])
        H, W = _grid_hw_from_npatches(n_patches, aspect=aspect)

        identity_per_layer_summary: Optional[Dict[str, Any]] = None
        identity_topk_patch_idx: Optional[Dict[int, List[int]]] = None

        for vname, feats in view_to_feats.items():
            s_maps, per_layer_summary = self.score_single_view(
                cls_name, feats,
                mode=mode, view=vname,
                topk_patches=topk_patches,
                mode_scoring=mode_scoring,
                lam=lam, tau_close=tau_close, gamma=gamma,
            )
            fused = self.fuse_layers(s_maps, fusion=layer_fusion)  # [N]
            aligned = align_map_to_identity(fused, views[vname], H, W)  # [N]
            view_maps_aligned.append(aligned)

            per_view_debug[vname] = {
                "layer_summary": per_layer_summary,
                "fused_map_stats": {
                    "max": float(fused.max().item()) if fused.numel() else 0.0,
                    "mean": float(fused.mean().item()) if fused.numel() else 0.0,
                }
            }

            if vname == concept_assign_view:
                identity_per_layer_summary = per_layer_summary
                # use topk indices from each layer for updates
                identity_topk_patch_idx = {
                    int(k.replace("layer", "")): v["topk_patch_idx"]
                    for k, v in per_layer_summary.items()
                }

        # fuse views at map level
        if len(view_maps_aligned) == 0:
            fused_all = torch.zeros((0,), device=self.device)
        else:
            V = torch.stack(view_maps_aligned, dim=0)  # [V, N]
            if view_fusion == "sum":
                fused_all = V.sum(dim=0)
            elif view_fusion == "max":
                fused_all = V.max(dim=0).values
            else: # "mean" mode
                fused_all = V.mean(dim=0)


        score_img = self.aggregate_map(
            fused_all,
            agg=img_agg,
            topk=img_topk,
            quantile=img_quantile,
        )

        # identity-view concept ids for RB
        feats_id = view_to_feats[concept_assign_view]
        per_layer_for_rb: Dict[str, Any] = {}
        assert identity_per_layer_summary is not None and identity_topk_patch_idx is not None

        for lk, info in identity_per_layer_summary.items():
            layer = int(lk.replace("layer", ""))
            top_idx = torch.tensor(info["topk_patch_idx"], device=self.device, dtype=torch.long)
            q_layer = l2_normalize(feats_id[layer].to(self.device))
            q_top = q_layer[top_idx]
            concept_ids = self.concept_cb[cls_name][layer].assign_ids(q_top)
            per_layer_for_rb[lk] = {
                "score_layer_proxy": float(torch.topk(fused_all, k=min(64, fused_all.numel())).values.mean().item()) if fused_all.numel() else 0.0,
                "topk_dN_max": info["topk_dN_max"],
                "topk_dN_mean": info["topk_dN_mean"],
                "topk_dD_min": info["topk_dD_min"],
                "corr_top_mean": info.get("corr_top_mean", 0.0),
                "corr_top_max": info.get("corr_top_max", 0.0),
                "corr_top_min": info.get("corr_top_min", 0.0),
                "corr_bank_size": info.get("corr_bank_size", 0),
                "use_defect_bank": info.get("use_defect_bank", False),
                "topk_concept_ids": concept_ids,
                "topk_patch_idx": info["topk_patch_idx"],
            }

        A_map_stats = {
            "per_layer_summary": per_layer_for_rb,
            "debug": {"per_view": per_view_debug, "grid_hw": [H, W]},
        }
        return float(score_img), A_map_stats, view_to_feats


    @torch.no_grad()
    def score_sample_correction_debug_maps(
        self,
        cls_name: str,
        view_to_feats: Dict[str, List[torch.Tensor]],
        *,
        mode: int,
        views: Dict[str, ViewSpec],
        topk_patches: int = 32,
        mode_scoring: str = "boost",
        lam: float = 0.8,
        tau_close: float = 0.20,
        gamma: float = 1.0,
        layer_fusion: str = "mean",
        view_fusion: str = "mean",
        img_agg: str = "topk_mean",
        img_topk: int = 64,
        img_quantile: float = 0.95,
        concept_assign_view: str = "id",
        img_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Return faithful before/after correction maps for qualitative diagnostics.

        The normal scorer applies the signed correction bank only to the top-k
        anomalous patches for efficiency.  This method mirrors that path and
        exposes three map-level tensors in identity coordinates:
          - A_pre:  anomaly map before signed correction;
          - A_post: anomaly map after signed correction;
          - A_corr: the actually applied signed residual A_post - A_pre.

        This is diagnostic-only and does not mutate any memory.
        """
        assert concept_assign_view in view_to_feats, "concept_assign_view must exist in view_to_feats"
        self._ensure_mode_view_normals(cls_name, mode, concept_assign_view, view_to_feats[concept_assign_view])

        aspect = None
        if img_hw is not None:
            H_img, W_img = int(img_hw[0]), int(img_hw[1])
            if W_img > 0:
                aspect = float(H_img) / float(W_img)
        n_patches = int(view_to_feats[concept_assign_view][0].shape[0])
        H, W = _grid_hw_from_npatches(n_patches, aspect=aspect)

        pre_view_maps: List[torch.Tensor] = []
        post_view_maps: List[torch.Tensor] = []
        debug: Dict[str, Any] = {"per_view": {}, "grid_hw": [int(H), int(W)]}

        for vname, feats in view_to_feats.items():
            n_layers = len(feats)
            pre_layer_maps: List[torch.Tensor] = []
            post_layer_maps: List[torch.Tensor] = []
            per_layer: Dict[str, Any] = {}
            self._ensure_mode_view_normals(cls_name, mode, vname, feats)

            for l in range(n_layers):
                q = l2_normalize(feats[l].to(self.device))
                dN, _ = self._dN_min_fixed_dyn(cls_name, mode, vname, l, q)
                k = min(int(topk_patches), int(dN.numel()))
                top_dN, top_idx = torch.topk(dN, k=k, largest=True)

                defect_empty = ((not self.use_defect_bank) or (self.defect_cb[cls_name][vname][l].size() == 0))
                if mode_scoring == "contrast":
                    if defect_empty:
                        s_top_pre = top_dN.clone()
                        top_dD = torch.ones_like(top_dN)
                    else:
                        q_top = q[top_idx]
                        top_dD, _ = self.defect_cb[cls_name][vname][l].nearest(q_top, relax=True)
                        s_top_pre = top_dN - float(gamma) * top_dD
                else:
                    if defect_empty:
                        top_dD = torch.ones_like(top_dN)
                        boost = torch.zeros_like(top_dN)
                    else:
                        q_top = q[top_idx]
                        top_dD, _ = self.defect_cb[cls_name][vname][l].nearest(q_top, relax=True)
                        boost = torch.relu(torch.tensor(float(tau_close), device=self.device) - top_dD)
                    s_top_pre = top_dN + float(lam) * boost

                corr_top = torch.zeros_like(top_dN)
                if self.corr_cfg.enabled and (self.corr_cb[cls_name][vname][l].size() > 0):
                    q_top2 = q[top_idx]
                    corr_top = float(self.corr_cfg.corr_weight) * self.corr_cb[cls_name][vname][l].apply(q_top2)
                s_top_post = s_top_pre + corr_top

                pre_map = dN.clone()
                post_map = dN.clone()
                pre_map[top_idx] = s_top_pre
                post_map[top_idx] = s_top_post
                pre_layer_maps.append(pre_map)
                post_layer_maps.append(post_map)
                per_layer[f"layer{l}"] = {
                    "corr_top_mean": float(corr_top.mean().item()) if corr_top.numel() else 0.0,
                    "corr_top_max": float(corr_top.max().item()) if corr_top.numel() else 0.0,
                    "corr_top_min": float(corr_top.min().item()) if corr_top.numel() else 0.0,
                    "corr_bank_size": int(self.corr_cb[cls_name][vname][l].size()) if self.corr_cfg.enabled else 0,
                    "topk_patch_idx": [int(i) for i in top_idx.detach().cpu().tolist()],
                }

            pre_fused = self.fuse_layers(pre_layer_maps, fusion=layer_fusion)
            post_fused = self.fuse_layers(post_layer_maps, fusion=layer_fusion)
            pre_aligned = align_map_to_identity(pre_fused, views[vname], H, W)
            post_aligned = align_map_to_identity(post_fused, views[vname], H, W)
            pre_view_maps.append(pre_aligned)
            post_view_maps.append(post_aligned)
            debug["per_view"][vname] = {
                "layer_summary": per_layer,
                "pre_fused_max": float(pre_fused.max().item()) if pre_fused.numel() else 0.0,
                "post_fused_max": float(post_fused.max().item()) if post_fused.numel() else 0.0,
            }

        if len(pre_view_maps) == 0:
            fused_pre = torch.zeros((0,), device=self.device)
            fused_post = torch.zeros((0,), device=self.device)
        else:
            P = torch.stack(pre_view_maps, dim=0)
            Q = torch.stack(post_view_maps, dim=0)
            if view_fusion == "sum":
                fused_pre = P.sum(dim=0)
                fused_post = Q.sum(dim=0)
            elif view_fusion == "max":
                fused_pre = P.max(dim=0).values
                fused_post = Q.max(dim=0).values
            else:
                fused_pre = P.mean(dim=0)
                fused_post = Q.mean(dim=0)

        fused_corr = fused_post - fused_pre
        score_pre = self.aggregate_map(fused_pre, agg=img_agg, topk=img_topk, quantile=img_quantile)
        score_post = self.aggregate_map(fused_post, agg=img_agg, topk=img_topk, quantile=img_quantile)

        def _to_hw(x: torch.Tensor) -> np.ndarray:
            if x.numel() == 0:
                return np.zeros((int(H), int(W)), dtype=np.float32)
            return x.detach().float().cpu().reshape(int(H), int(W)).numpy().astype(np.float32)

        A_pre = _to_hw(fused_pre)
        A_post = _to_hw(fused_post)
        A_corr = _to_hw(fused_corr)
        return {
            "A_pre": A_pre,
            "A_post": A_post,
            "A_corr": A_corr,
            "A_diff": A_corr,
            "score_pre": float(score_pre),
            "score_post": float(score_post),
            "corr_min": float(np.nanmin(A_corr)) if A_corr.size else 0.0,
            "corr_max": float(np.nanmax(A_corr)) if A_corr.size else 0.0,
            "corr_mean": float(np.nanmean(A_corr)) if A_corr.size else 0.0,
            "corr_abs_mean": float(np.nanmean(np.abs(A_corr))) if A_corr.size else 0.0,
            "grid_hw": [int(H), int(W)],
            "debug": debug,
        }

    @torch.no_grad()
    def update_from_topk(
        self,
        cls_name: str,
        feats_per_layer: List[torch.Tensor],
        *,
        which: str,
        mode: int,
        view: str,
        layer_to_patch_idx: Dict[int, List[int]],
        tau_insert: float,
        tau_replace: float,
        event_id: Optional[int] = None,
        write_value: float = 1.0,
        outcome: Optional[str] = None,
        boundary_value: Optional[float] = None,
        coverage_gain: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update NORMAL-DYN or DEFECT banks with selected patch features."""
        self._ensure_mode_view_normals(cls_name, mode, view, feats_per_layer)
        out: Dict[str, Any] = {"which": which, "mode": int(mode), "view": view, "layers": {}}
        total_changes = 0

        for l, patch_idx in layer_to_patch_idx.items():
            if len(patch_idx) == 0:
                continue
            q = l2_normalize(feats_per_layer[l].to(self.device))
            idx_t = torch.tensor(patch_idx, device=self.device, dtype=torch.long)
            sel = q[idx_t]

            if which == "normal_dyn":
                cover_feats = None
                if bool(getattr(self, "enable_utility_aware_stm_retention", False)):
                    cover_chunks: List[torch.Tensor] = []
                    try:
                        ltm_cb = self.normal_dyn_ltm[cls_name][mode][view][l]
                        if ltm_cb.size() > 0:
                            cover_chunks.append(ltm_cb.protos.float())
                    except Exception:
                        pass
                    try:
                        cand_cb_for_cover = self.normal_dyn_candidates[cls_name][mode][view][l]
                        if bool(getattr(self, "enable_maturation_buffer_retrieval", False)) and cand_cb_for_cover.size() > 0:
                            buf_p, _buf_scores, _buf_info = cand_cb_for_cover.get_retrieval_prototypes(
                                min_repeat=int(getattr(self, "mature_retrieval_min_events", 2)),
                                min_usefulness=float(getattr(self, "mature_retrieval_min_usefulness", 0.10)),
                                score_threshold=float(getattr(self, "mature_retrieval_score_thr", 0.25)),
                                max_protos=int(getattr(self, "mature_retrieval_max_protos", 256)),
                            )
                            if buf_p.numel() > 0:
                                cover_chunks.append(buf_p.float())
                    except Exception:
                        pass
                    if len(cover_chunks) > 0:
                        cover_feats = torch.cat([c.to(self.device).float() for c in cover_chunks], dim=0)
                stats = self.normal_dyn[cls_name][mode][view][l].update(
                    sel,
                    tau_insert=tau_insert,
                    tau_replace=tau_replace,
                    allow_replace=True,
                    use_utility_retention=bool(getattr(self, "enable_utility_aware_stm_retention", False)),
                    utility_score=float(write_value),
                    event_id=event_id,
                    step=event_id,
                    outcome=outcome,
                    boundary_value=boundary_value,
                    coverage_gain=coverage_gain,
                    cover_feats=cover_feats,
                    retention_local_k=int(getattr(self, "stm_retention_local_k", 32)),
                    retention_cover_eps=float(getattr(self, "stm_retention_ltm_cover_eps", 0.025)),
                    retention_recency_tau=float(getattr(self, "stm_retention_recency_tau", 1024.0)),
                    retention_replace_margin=float(getattr(self, "stm_retention_replace_margin", 0.0)),
                    retention_ema=float(getattr(self, "stm_retention_ema", 0.10)),
                )
                if event_id is not None and cls_name in self.normal_dyn_candidates:
                    track_sel = sel
                    # Buffer update sees a compact event-level summary; STM still receives
                    # all selected patches above. This prevents one image from creating too
                    # many provisional candidates and faking maturity.
                    if self.enable_maturation_buffer:
                        # First collapse near-duplicate patches within this same event.
                        # This makes distinct-event recurrence meaningful: a single image
                        # cannot manufacture many independent maturation candidates.
                        event_eps = float(getattr(self, "mature_event_merge_eps", 0.0))
                        if event_eps > 0.0 and int(track_sel.shape[0]) > 1:
                            event_protos: List[torch.Tensor] = []
                            for _ii in range(int(track_sel.shape[0])):
                                _f = l2_normalize(track_sel[_ii:_ii+1].float()).squeeze(0)
                                if len(event_protos) == 0:
                                    event_protos.append(_f)
                                    continue
                                _P = l2_normalize(torch.stack(event_protos, dim=0))
                                _dmin = float((1.0 - torch.clamp(_f.view(1, -1) @ _P.T, -1.0, 1.0)).min().item())
                                if _dmin > event_eps:
                                    event_protos.append(_f)
                            if len(event_protos) > 0:
                                track_sel = torch.stack(event_protos, dim=0).to(device=sel.device)
                        if int(getattr(self, "mature_max_event_protos", 0)) > 0:
                            cap = int(max(1, self.mature_max_event_protos))
                            if int(track_sel.shape[0]) > cap:
                                track_sel = self._select_diverse_prototypes(track_sel.float(), cap)
                    track_stats = self.normal_dyn_candidates[cls_name][mode][view][l].update(
                        track_sel,
                        usefulness=float(write_value),
                        event_id=int(event_id),
                        step=int(event_id),
                        outcome=outcome,
                        boundary_value=boundary_value,
                        coverage_gain=coverage_gain,
                    )
                    stats["maturation_buffer_update"] = track_stats
                    # Backward-compatible key for old parsers.
                    stats["ltm_candidate_update"] = track_stats
            elif which == "defect":
                stats = self.defect_cb[cls_name][view][l].update(
                    sel, tau_insert=tau_insert, tau_replace=tau_replace, allow_replace=True
                )
            else:
                raise ValueError(f"Unknown bank: {which}")

            out["layers"][str(l)] = stats
            total_changes += int(stats.get("n_added", 0)) + int(stats.get("n_replaced", 0))

        if which == "normal_dyn":
            if event_id is not None and int(self._ltm_last_event.get(cls_name, -1)) != int(event_id):
                self._ltm_pending_updates[cls_name] = int(self._ltm_pending_updates.get(cls_name, 0)) + 1
                self._ltm_last_event[cls_name] = int(event_id)
            stm_sizes = [int(cb.size()) for mode_dict in self.normal_dyn.get(cls_name, {}).values() for view_dict in mode_dict.values() for cb in view_dict]
            out["ltm_pending_updates"] = int(self._ltm_pending_updates.get(cls_name, 0))
            pressure_k = max(int(self.ltm_min_stm_size), int(math.ceil(float(self.ltm_pressure_ratio) * float(max(1, self.K_normal_dyn)))))
            pending_now = int(self._ltm_pending_updates.get(cls_name, 0))
            pressure_trigger = any(sz >= int(pressure_k) for sz in stm_sizes) and pending_now >= int(self.ltm_min_pending_updates)
            mature_trigger = bool(self.enable_maturation_buffer) and pending_now >= int(self.mature_promote_every)
            trigger = bool(pressure_trigger or mature_trigger)
            out["ltm_trigger_debug"] = {
                "pressure_trigger": bool(pressure_trigger),
                "mature_trigger": bool(mature_trigger),
                "pending_updates": int(pending_now),
                "pressure_k": int(pressure_k),
                "enable_maturation_buffer": bool(self.enable_maturation_buffer),
            }
            if trigger:
                if self.enable_ltm_promotion:
                    out["consolidation"] = self.consolidate_normal_dyn(cls_name)
                    out["ltm_pending_updates"] = int(self._ltm_pending_updates.get(cls_name, 0))
                elif self.shadow_ltm:
                    out["shadow_ltm"] = self.shadow_evaluate_ltm(cls_name)

        return out


    @torch.no_grad()
    def select_core_patches(
        self,
        cls_name: str,
        feats_per_layer: List[torch.Tensor],
        *,
        mode: int,
        view: str,
        k_core: int,
    ) -> Dict[int, List[int]]:
        """Select *typical* patches (lowest dN) for core-normal self-label updates."""
        self._ensure_mode_view_normals(cls_name, mode, view, feats_per_layer)
        n_layers = len(feats_per_layer)
        layer_to_idx: Dict[int, List[int]] = {}

        for l in range(n_layers):
            q = l2_normalize(feats_per_layer[l].to(self.device))
            dN, _ = self._dN_min_fixed_dyn(cls_name, mode, view, l, q)
            k = min(int(k_core), int(dN.numel()))
            if k <= 0:
                layer_to_idx[l] = []
                continue
            idx = torch.topk(dN, k=k, largest=False).indices
            layer_to_idx[l] = [int(i) for i in idx.detach().cpu().tolist()]

        return layer_to_idx

    @torch.no_grad()
    def update_corrections_from_topk(
        self,
        cls_name: str,
        feats_per_layer: List[torch.Tensor],
        *,
        mode: int,
        view: str,
        layer_to_patch_idx: Dict[int, List[int]],
        sign: int,
        score_img: float,
        theta: float,
        strength_scale: float = 1.0,
        outcome: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update correction bank using selected patch features.

        v21 keeps the stable v20 query behavior, but improves the *post-query*
        correction write path.  The queried label is already known here, so the
        write can be sign-aware and outcome-aware without changing the controller.
        """
        self._ensure_mode_view_normals(cls_name, mode, view, feats_per_layer)
        out: Dict[str, Any] = {"view": view, "mode": int(mode), "layers": {}, "sign": int(sign), "outcome": str(outcome or "")}

        if (not self.corr_cfg.enabled):
            out["disabled"] = True
            return out

        outcome_key = str(outcome or "").upper()
        if bool(getattr(self.corr_cfg, "enable_sign_aware_write", True)):
            outcome_weight = {
                "FP": float(getattr(self.corr_cfg, "outcome_weight_fp", 1.30)),
                "FN": float(getattr(self.corr_cfg, "outcome_weight_fn", 1.20)),
                "TN": float(getattr(self.corr_cfg, "outcome_weight_tn", 0.60)),
                "TP": float(getattr(self.corr_cfg, "outcome_weight_tp", 0.80)),
            }.get(outcome_key, 1.0)
        else:
            outcome_weight = 1.0

        # Image-level margin still controls base strength, but no longer alone:
        # outcome_weight and local patch weighting make the write sign-aware and local.
        margin = abs(float(score_img) - float(theta))
        beta_abs_base = float(np.clip(self.corr_cfg.beta_scale * margin, self.corr_cfg.beta_min, self.corr_cfg.beta_max))
        beta_abs = float(np.clip(
            beta_abs_base * float(max(strength_scale, 1e-3)) * float(max(outcome_weight, 1e-3)),
            self.corr_cfg.beta_min,
            self.corr_cfg.beta_max,
        ))
        sign_val = 1.0 if int(sign) >= 0 else -1.0

        for l, patch_idx in layer_to_patch_idx.items():
            if len(patch_idx) == 0:
                continue

            q = l2_normalize(feats_per_layer[l].to(self.device))
            idx_t = torch.tensor(patch_idx, device=self.device, dtype=torch.long)
            sel = q[idx_t]

            dN_full, _ = self._dN_min_fixed_dyn(cls_name, mode, view, int(l), q)
            dN_sel = dN_full[idx_t]

            if int(sign) < 0:
                keep = dN_sel >= float(self.corr_cfg.min_dN_for_store_fp)
            else:
                keep = dN_sel >= float(self.corr_cfg.min_dN_for_store_fn)

            if keep.numel() == 0 or (not bool(keep.any().item())):
                continue

            sel_keep = sel[keep]
            dN_keep = dN_sel[keep]
            if sel_keep.numel() == 0:
                continue

            if bool(getattr(self.corr_cfg, "enable_patch_weighted_beta", True)) and int(dN_keep.numel()) > 1:
                lo = float(dN_keep.min().item())
                hi = float(dN_keep.max().item())
                if hi > lo + 1e-12:
                    z = torch.clamp((dN_keep - lo) / (hi - lo + 1e-12), 0.0, 1.0)
                else:
                    z = torch.ones_like(dN_keep) * 0.5
                power = float(max(0.25, getattr(self.corr_cfg, "patch_beta_power", 1.0)))
                z = torch.pow(z, power)
                floor = float(getattr(self.corr_cfg, "patch_beta_floor", 0.75))
                cap = float(getattr(self.corr_cfg, "patch_beta_cap", 1.25))
                patch_mult = torch.clamp(floor + (cap - floor) * z, min=min(floor, cap), max=max(floor, cap))
            else:
                patch_mult = torch.ones_like(dN_keep)

            beta_t = (sign_val * beta_abs * patch_mult).to(device=self.device, dtype=torch.float32)
            stats = self.corr_cb[cls_name][view][int(l)].update(
                sel_keep,
                beta_t,
                outcome=outcome_key,
                step=step,
            )
            out["layers"][str(l)] = {
                **stats,
                "outcome": str(outcome_key),
                "outcome_weight": float(outcome_weight),
                "beta_abs_base": float(beta_abs_base),
                "beta_abs_after_outcome": float(beta_abs),
                "beta_mean": float(beta_t.mean().item()) if beta_t.numel() else 0.0,
                "beta_abs_mean": float(beta_t.abs().mean().item()) if beta_t.numel() else 0.0,
                "beta_abs_min": float(beta_t.abs().min().item()) if beta_t.numel() else 0.0,
                "beta_abs_max": float(beta_t.abs().max().item()) if beta_t.numel() else 0.0,
                "strength_scale": float(strength_scale),
                "kept": int(sel_keep.shape[0]),
                "dN_sel_mean": float(dN_keep.mean().item()) if dN_keep.numel() else 0.0,
                "dN_sel_min": float(dN_keep.min().item()) if dN_keep.numel() else 0.0,
                "dN_sel_max": float(dN_keep.max().item()) if dN_keep.numel() else 0.0,
                "patch_beta_weighted": bool(getattr(self.corr_cfg, "enable_patch_weighted_beta", True)),
                "corr_bank_size": int(self.corr_cb[cls_name][view][int(l)].size()),
            }

        return out
        

# ---------------------------------------------------------------------------
#  ReasoningBank: store cards for queried samples + concept stats for RB gating
# ---------------------------------------------------------------------------

@dataclass
class ConceptStats:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def fp_rate(self) -> float:
        # Bayesian mean with Beta(1,1) prior
        t = self.total
        return float((self.fp + 1) / (t + 2)) if t > 0 else 0.5

    def fn_rate(self) -> float:
        # Bayesian mean with Beta(1,1) prior
        t = self.total
        return float((self.fn + 1) / (t + 2)) if t > 0 else 0.5

class ReasoningBank:
    def __init__(self):
        self.cards: List[Dict[str, Any]] = []
        self.class_stats: Dict[str, ClassStats] = {}
        # class -> layer -> concept_id -> ConceptStats
        self.concept_stats: Dict[str, Dict[int, Dict[int, ConceptStats]]] = {}

    def get_or_create_stats(self, cls_name: str) -> ClassStats:
        if cls_name not in self.class_stats:
            self.class_stats[cls_name] = ClassStats()
        return self.class_stats[cls_name]

    def _ensure(self, cls_name: str, layer: int) -> None:
        if cls_name not in self.concept_stats:
            self.concept_stats[cls_name] = {}
        if layer not in self.concept_stats[cls_name]:
            self.concept_stats[cls_name][layer] = {}

    def update_concepts(self, cls_name: str, layer: int, concept_ids: List[int], outcome: str) -> None:
        self._ensure(cls_name, layer)
        for cid in concept_ids:
            if cid < 0:
                continue
            if cid not in self.concept_stats[cls_name][layer]:
                self.concept_stats[cls_name][layer][cid] = ConceptStats()
            s = self.concept_stats[cls_name][layer][cid]
            if outcome == "TP":
                s.tp += 1
            elif outcome == "TN":
                s.tn += 1
            elif outcome == "FP":
                s.fp += 1
            elif outcome == "FN":
                s.fn += 1

    def get_concept_stats(self, cls_name: str, layer: int, cid: int) -> ConceptStats:
        self._ensure(cls_name, layer)
        return self.concept_stats[cls_name][layer].get(cid, ConceptStats())

    def add_card(self, card: Dict[str, Any]) -> None:
        self.cards.append(card)

    def save_json(self, path: str) -> None:
        safe_makedirs(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cards, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
#  RB gating: decide which concepts to promote into normal/defect bank
# ---------------------------------------------------------------------------

@dataclass
class RBGatingConfig:
    min_count: int = 3
    min_fp_rate: float = 0.6
    min_fn_rate: float = 0.6

def get_raw_topk_patches(
    A_map_stats: Dict[str, Any], 
    topk_keep: int = 16
) -> Dict[int, List[int]]:
    """Get the raw top-k patches directly, bypassing RB gating. Used for the Correction Bank."""
    layer_to_idx: Dict[int, List[int]] = {}
    per_layer = A_map_stats.get("per_layer_summary", {})
    for lk, info in per_layer.items():
        layer = int(lk.replace("layer", ""))
        patch_idx: List[int] = info.get("topk_patch_idx", [])
        layer_to_idx[layer] = patch_idx[:topk_keep]
    return layer_to_idx


# ---------------------------------------------------------------------------
#  Query policy (warm-start calibration + uncertainty + exploration/exploitation)
# ---------------------------------------------------------------------------

@dataclass
class QueryConfig:
    # Uncertainty band on p_defect (used only for querying)
    ambig_low: float = 0.4
    ambig_high: float = 0.6

    # Budgeting
    warmup_queries: int = 15
    warmup_steps: int = 25

    # Recent-score window for tail queries
    topq_quantile: float = 0.90
    botq_quantile: float = 0.10
    score_window: int = 200

    # Warm-start calibration: spend early queries to stabilize theta quickly
    calib_steps: int = 40
    calib_margin_sigma: float = 0.5  # query if |score - theta| <= margin*sigmaN

    # Exploration vs exploitation
    eps0: float = 0.25
    eps_decay: float = 50.0
    eps_min: float = 0.05


class QueryState:
    def __init__(self):
        self.used = 0
        self.recent_scores: List[float] = []
        self.calib_used = 0

        # v21-query-baselines: state for independent budget-matched baselines
        # (entropy/margin/novelty/random/periodic). This is separate from
        # SimpleAIFState so baseline policies do not inherit AIF utility terms.
        self.baseline_steps: int = 0
        self.baseline_used: int = 0
        self.recent_policy_scores: List[float] = []
        self.recent_policy_queries: List[int] = []

def update_query_state_score(qs: QueryState, qc: QueryConfig, score: float) -> None:
    qs.recent_scores.append(float(score))
    if len(qs.recent_scores) > qc.score_window:
        qs.recent_scores = qs.recent_scores[-qc.score_window:]


QUERY_BASELINE_POLICIES = {"entropy", "margin", "novelty", "random", "periodic"}


def _stable_uniform01(seed: int, step: int, salt: str = "query_baseline_random") -> float:
    """Deterministic pseudo-random number in [0, 1).

    This avoids coupling the random-query baseline to unrelated randomness such as
    support augmentation or PyTorch kernels.
    """
    msg = f"{int(seed)}:{int(step)}:{str(salt)}".encode("utf-8")
    h = hashlib.blake2b(msg, digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False) / float(2 ** 64)


def _policy_baseline_utility(
    *,
    policy: str,
    p_defect: float,
    score: float,
    theta: float,
    sigmaN: float,
    novelty: float,
) -> Tuple[float, Dict[str, Any]]:
    """Return a normalized query-worthiness score for non-AIF baselines.

    Higher utility means more likely to query. These utilities are deliberately
    simple and independent of the AIF EFE objective:
      - entropy: Bernoulli entropy of p(defect), normalized by log(2);
      - margin: closeness to the current score threshold;
      - novelty: global normal-coverage novelty percentile;
      - random/periodic: utility is diagnostic only, decision is made separately.
    """
    pol = str(policy).lower().strip()
    sig = float(max(float(sigmaN), 1e-6))
    if pol == "entropy":
        H = _entropy_bern(float(p_defect))
        util = float(np.clip(H / math.log(2.0), 0.0, 1.0))
        return util, {"utility_name": "entropy_norm", "entropy": float(H)}
    if pol == "margin":
        z = abs(float(score) - float(theta)) / sig
        util = float(np.clip(math.exp(-0.5 * z * z), 0.0, 1.0))
        return util, {"utility_name": "boundary_closeness", "margin_abs": float(abs(float(score) - float(theta))), "z_margin": float(z)}
    if pol == "novelty":
        util = float(np.clip(float(novelty), 0.0, 1.0))
        return util, {"utility_name": "coverage_novelty"}
    if pol in {"random", "periodic"}:
        return 0.0, {"utility_name": pol}
    raise ValueError(f"Unsupported query baseline policy: {policy}")


def observe_query_baseline_state(
    qs: QueryState,
    *,
    utility: float,
    queried: bool,
    simple_cfg: "SimpleAIFConfig",
) -> None:
    qs.baseline_steps += 1
    qs.baseline_used += int(bool(queried))
    qs.recent_policy_scores.append(float(utility))
    qs.recent_policy_queries.append(int(bool(queried)))
    win = int(max(1, getattr(simple_cfg, "local_qrate_window", 256)))
    if len(qs.recent_policy_scores) > win:
        qs.recent_policy_scores = qs.recent_policy_scores[-win:]
    if len(qs.recent_policy_queries) > win:
        qs.recent_policy_queries = qs.recent_policy_queries[-win:]


def select_query_baseline_action(
    *,
    policy: str,
    qs: QueryState,
    simple_cfg: "SimpleAIFConfig",
    seed: int,
    query_seed: Optional[int] = None,
    step: int = 0,
    p_defect: float,
    score: float,
    theta: float,
    sigmaN: float,
    novelty: float,
    novelty_info: Dict[str, Any],
    evidence: Dict[str, float],
) -> Tuple["AIFAction", Dict[str, Any]]:
    """Independent query-policy baselines for controller attribution.

    These policies keep the same downstream memory/sidecar/write mechanisms as
    the main method, but replace the AIF query decision itself. For entropy,
    margin, and novelty we use an online top-q quantile gate over recent utility
    scores so that observed query rate is approximately matched to target_qrate.
    Random and periodic use the same target_qrate directly.
    """
    pol = str(policy).lower().strip()
    target = float(np.clip(float(simple_cfg.target_qrate), 0.0, 1.0))
    rng_seed = int(seed if query_seed is None else query_seed)
    utility, util_info = _policy_baseline_utility(
        policy=pol,
        p_defect=float(p_defect),
        score=float(score),
        theta=float(theta),
        sigmaN=float(sigmaN),
        novelty=float(novelty),
    )

    reason = ""
    threshold = None

    # Budget accounting before the current decision.  qs.baseline_steps / used
    # contain only previous stream elements; n_next is the 1-indexed count after
    # the current element is processed.  These quantities are used below by the
    # hard budget guard for score-based baselines.
    n_next = int(qs.baseline_steps) + 1
    used_before = int(qs.baseline_used)
    desired_floor = int(math.floor(target * float(n_next)))
    desired_ceil = int(math.ceil(target * float(n_next)))
    budget_deficit_before = float(target * float(n_next) - float(used_before))
    budget_guard = "none"

    if target <= 0.0:
        queried = False
        reason = "target_qrate_zero"
    elif pol == "random":
        u = _stable_uniform01(int(rng_seed), int(step), salt="random_query_policy")
        queried = bool(u < target)
        reason = "random_uniform"
        util_info["random_u"] = float(u)
    elif pol == "periodic":
        period = int(max(1, round(1.0 / max(target, 1e-8))))
        queried = bool(((int(qs.baseline_steps) + 1) % period) == 0)
        reason = f"period_{period}"
        util_info["period"] = int(period)
    elif pol in {"entropy", "margin", "novelty"}:
        history = list(qs.recent_policy_scores)
        min_hist = int(max(16, min(64, getattr(simple_cfg, "local_qrate_window", 256) // 4)))
        if len(history) < min_hist:
            # Cold-start without borrowing AIF logic: spend the expected budget at
            # random until enough utility history exists for an online quantile.
            u = _stable_uniform01(int(rng_seed), int(step), salt=f"{pol}_cold_start")
            queried = bool(u < target)
            reason = f"cold_start_random_until_{min_hist}"
            util_info["random_u"] = float(u)
            util_info["min_history"] = int(min_hist)
        else:
            q = float(np.clip(1.0 - target, 0.0, 1.0))
            threshold = float(np.quantile(np.asarray(history, dtype=np.float32), q))
            queried = bool(float(utility) > float(threshold))
            reason = f"online_top_{target:.4f}_quantile"

            # Gentle local-budget guard to prevent pathological overshoot under
            # ties or nonstationary utility distributions. It suppresses only
            # marginal candidates; extreme utility values are still allowed.
            if len(qs.recent_policy_queries) > 0:
                local_q = float(sum(qs.recent_policy_queries)) / max(1.0, float(len(qs.recent_policy_queries)))
                if local_q > max(target * 1.5, target + 0.03) and float(utility) < 0.995:
                    queried = False
                    reason += "::budget_suppress"
                elif local_q < max(target * 0.5, target - 0.03):
                    lo_q = float(np.clip(1.0 - min(3.0 * target, 0.50), 0.0, 1.0))
                    lo_thr = float(np.quantile(np.asarray(history, dtype=np.float32), lo_q))
                    if float(utility) > lo_thr:
                        queried = True
                        reason += "::budget_catchup"
                        util_info["catchup_threshold"] = float(lo_thr)
        util_info["history_len"] = int(len(qs.recent_policy_scores))

        # Final-submission budget guard for deterministic score-based baselines.
        # The earlier online quantile gate could under-spend badly for novelty-only
        # policies on classes where coverage novelty becomes sparse.  For a fair
        # matched-budget comparison against AIF, entropy/margin/novelty must spend
        # approximately the requested label budget.  The guard does not change the
        # utility definition; it only forces catch-up when the policy is below the
        # cumulative target and suppresses extra queries when it is already above it.
        if target > 0.0:
            if used_before < desired_floor:
                if not bool(queried):
                    budget_guard = "force_catchup"
                    reason += "::budget_force_catchup"
                queried = True
            elif used_before >= max(1, desired_ceil):
                if bool(queried):
                    budget_guard = "hard_cap"
                    reason += "::budget_hard_cap"
                queried = False
    else:
        raise ValueError(f"Unsupported query baseline policy: {policy}")

    observe_query_baseline_state(qs, utility=float(utility), queried=bool(queried), simple_cfg=simple_cfg)
    action = AIFAction.QUERY_LABEL if bool(queried) else AIFAction.CLASSIFY
    info = {
        "controller": f"query_baseline_{pol}",
        "version": "v21_query_baselines_budget_guarded",
        "action": action.value,
        "target_qrate": float(target),
        "query_seed": None if query_seed is None else int(query_seed),
        "randomness_seed_used": int(rng_seed),
        "local_qrate_runtime": float(sum(qs.recent_policy_queries) / max(1.0, float(len(qs.recent_policy_queries)))) if len(qs.recent_policy_queries) else 0.0,
        "global_qrate_runtime": float(qs.baseline_used / max(1.0, float(qs.baseline_steps))),
        "budget_guard": str(budget_guard),
        "budget_n_next": int(n_next),
        "budget_used_before": int(used_before),
        "budget_desired_floor": int(desired_floor),
        "budget_desired_ceil": int(desired_ceil),
        "budget_deficit_before": float(budget_deficit_before),
        "utility": float(utility),
        "threshold": None if threshold is None else float(threshold),
        "reason": str(reason),
        "novelty": float(novelty),
        "novelty_info": novelty_info,
        "utility_info": util_info,
        "correction_evidence": {
            "corr_mean": float(evidence.get("corr_mean", 0.0)),
            "corr_max": float(evidence.get("corr_max", 0.0)),
            "corr_min": float(evidence.get("corr_min", 0.0)),
            "corr_def": float(evidence.get("corr_def", 0.0)),
            "corr_norm": float(evidence.get("corr_norm", 0.0)),
            "corr_bank_nonempty": bool(evidence.get("corr_bank_nonempty", False)),
            "corr_signal_dead": bool(evidence.get("corr_signal_dead", False)),
        },
        "feasible_actions": [AIFAction.CLASSIFY.value, AIFAction.QUERY_LABEL.value],
    }
    return action, info


# ---------------------------------------------------------------------------
#  Phase-3: Active Inference (AIF) controller
#   - Action selection via approximate Expected Free Energy (EFE)
#   - Actions: classify, query label, query region (mask), query new support
#   - Phase-3 fixes:
#       (1) s_rel explicitly includes calibration state (NormalThresholdController stats + uncertainty)
#       (2) contamination resistance:
#           - Bayesian feedback-channel reliability r_user
#           - two evidence channels (margin-to-theta + correction activation) to detect suspicious labels
#           - trust-weighted / gated writes to ALL banks
# ---------------------------------------------------------------------------

from enum import Enum


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _entropy_bern(p: float) -> float:
    p = _clamp01(p)
    p = min(max(p, 1e-8), 1.0 - 1e-8)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


class AIFAction(str, Enum):
    CLASSIFY = "classify"
    QUERY_LABEL = "query_label"
    QUERY_REGION = "query_region"
    QUERY_SUPPORT = "query_support"


class AIFRegime(str, Enum):
    CALIBRATION_POOR = "calibration_poor"
    COVERAGE_POOR = "coverage_poor"
    REASONING_POOR = "reasoning_poor"
    STABLE = "stable"



@dataclass
class SimpleAIFConfig:
    cost_fp: float = 1.0
    cost_fn: float = 2.5
    query_cost: float = 0.80
    beta_entropy: float = 0.30
    lambda_normal: float = 0.35
    lambda_correction: float = 0.15
    lambda_defect_side: float = 0.35

    # Adaptive label-budget control (local-window rather than cumulative rate)
    adapt_query_cost: bool = True
    target_qrate: float = 0.06
    query_cost_lr: float = 0.02
    min_query_cost: float = 0.05
    max_query_cost: float = 3.00
    local_qrate_window: int = 256

    # Feature-space coverage of normality
    novelty_k: int = 1
    coverage_ref_clip_low: float = 0.05
    coverage_ref_clip_high: float = 0.95

    # Optional local saliency multiplier for correction utility
    use_local_saliency: bool = False
    saliency_weight: float = 1.0

    # Correction-gap scaling
    correction_gap_scale: float = 0.05
    defect_weak_corr_scale: float = 0.05

    # Defect-side epistemic band on the normal side of theta
    defect_near_sigma: float = 0.5
    defect_far_sigma: float = 3.0
    defect_tau_sigma: float = 0.5
    defect_near_min_abs: float = 0.005
    defect_far_min_abs: float = 0.02
    defect_tau_min_abs: float = 0.005

    # Hybrid boundary bootstrap
    bootstrap_dynamic_target: int = 16
    bootstrap_max_steps: int = 400
    bootstrap_band_low_sigma: float = 1.0
    bootstrap_band_high_sigma: float = 1.0
    bootstrap_band_min_abs: float = 0.02
    bootstrap_top_tail_sigma: float = 2.5
    bootstrap_top_tail_min_abs: float = 0.08

    # v22 finite-stream cold-start rescue. Disabled by default.
    v22_enable_warmup_budget: bool = False
    v22_warmup_steps: int = 1000
    v22_warmup_qrate: float = 0.06
    v22_budget_guard: bool = True
    v22_enable_defect_audit: bool = False
    v22_audit_frac: float = 0.25
    v22_audit_min_p: float = 0.80
    v22_audit_score_sigma: float = 1.0

    # v22 guarded_dual_anchor_v2_balanced_rescue: bounded FN-audit side-channel.
    # This audits predicted-normal samples just below theta, where hidden FNs live.
    # It is intentionally separate from the high-confidence predicted-defect audit above.
    v22_enable_fn_audit: bool = False
    v22_fn_audit_band: float = 0.040
    v22_fn_audit_max_extra_qrate: float = 0.005
    v22_fn_audit_min_gap: int = 20
    v22_fn_audit_warmup_steps: int = 1500
    v22_fn_audit_min_p: float = 0.05



class SimpleAIFState:
    def __init__(self, cfg: "SimpleAIFConfig"):
        self.cfg = cfg
        self.query_cost_runtime = float(cfg.query_cost)
        self.steps = 0
        self.queries = 0
        self.recent_query_flags: List[int] = []

    def observe(self, queried: bool) -> None:
        self.steps += 1
        qflag = int(bool(queried))
        self.queries += qflag
        self.recent_query_flags.append(qflag)

        win = int(max(1, getattr(self.cfg, "local_qrate_window", 256)))
        if len(self.recent_query_flags) > win:
            self.recent_query_flags = self.recent_query_flags[-win:]

        if not bool(self.cfg.adapt_query_cost):
            return

        qrate_local = float(sum(self.recent_query_flags)) / max(1.0, float(len(self.recent_query_flags)))
        self.query_cost_runtime += float(self.cfg.query_cost_lr) * (qrate_local - float(self.cfg.target_qrate))
        self.query_cost_runtime = float(np.clip(
            self.query_cost_runtime,
            float(self.cfg.min_query_cost),
            float(self.cfg.max_query_cost),
        ))

    def local_qrate(self) -> float:
        if len(self.recent_query_flags) == 0:
            return 0.0
        return float(sum(self.recent_query_flags)) / max(1.0, float(len(self.recent_query_flags)))


class NormalCoverageMemory:
    """Track feature-space coverage of normality using global image features.

    We keep the initial support features fixed and append queried/verified normal
    features into a bounded dynamic queue. Novelty is reported as a percentile of
    nearest-neighbor distance relative to a *growing* reference distribution built
    from both support normals and later accepted dynamic normals. High percentile
    => weakly covered normal region.
    """

    def __init__(
        self,
        support_feats: torch.Tensor,
        ref_dists: torch.Tensor,
        *,
        max_dynamic: int = 1024,
    ):
        self.support_feats = l2_normalize(support_feats.float().cpu()) if support_feats.numel() else torch.empty((0, 0), dtype=torch.float32)
        self.support_ref_dists = ref_dists.float().cpu() if ref_dists.numel() else torch.empty((0,), dtype=torch.float32)
        self.dynamic_feats = torch.empty((0, self.support_feats.shape[1]), dtype=torch.float32) if self.support_feats.numel() else torch.empty((0, 0), dtype=torch.float32)
        self.dynamic_ref_dists = torch.empty((0,), dtype=torch.float32)
        self.max_dynamic = int(max(0, max_dynamic))
        self.last_debug: Dict[str, Any] = {}

    @classmethod
    def from_support_feats(
        cls,
        support_feats: List[torch.Tensor],
        *,
        max_dynamic: int = 1024,
    ) -> "NormalCoverageMemory":
        if len(support_feats) == 0:
            return cls(torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.float32), max_dynamic=max_dynamic)

        bank = torch.stack([l2_normalize(f.detach().float().cpu().view(-1)) for f in support_feats], dim=0)
        if bank.shape[0] <= 1:
            ref_d = torch.zeros((bank.shape[0],), dtype=torch.float32)
            return cls(bank, ref_d, max_dynamic=max_dynamic)

        sim = bank @ bank.T
        sim.fill_diagonal_(-1.0)
        max_sim, _ = sim.max(dim=1)
        ref_d = (1.0 - max_sim).clamp(min=0.0)
        return cls(bank, ref_d, max_dynamic=max_dynamic)

    def _combined_bank(self) -> torch.Tensor:
        if self.support_feats.numel() == 0:
            return self.dynamic_feats
        if self.dynamic_feats.numel() == 0:
            return self.support_feats
        return torch.cat([self.support_feats, self.dynamic_feats], dim=0)

    def _combined_ref(self) -> Tuple[torch.Tensor, str]:
        parts: List[torch.Tensor] = []
        if self.support_ref_dists.numel() > 0:
            parts.append(self.support_ref_dists)
        if self.dynamic_ref_dists.numel() > 0:
            parts.append(self.dynamic_ref_dists)
        if len(parts) == 0:
            return torch.tensor([0.0], dtype=torch.float32), "fallback_zero"
        if len(parts) == 1:
            if self.dynamic_ref_dists.numel() > 0 and self.support_ref_dists.numel() == 0:
                return parts[0], "dynamic_only"
            return parts[0], "support_only"
        return torch.cat(parts, dim=0), "support_plus_dynamic"

    def novelty_percentile(self, feat: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        bank = self._combined_bank()
        q = l2_normalize(feat.detach().float().cpu().view(1, -1))
        if bank.numel() == 0:
            info = {
                "distance_nn": 1.0,
                "nn_index": -1,
                "support_size": int(self.support_feats.shape[0]) if self.support_feats.ndim == 2 else 0,
                "dynamic_size": int(self.dynamic_feats.shape[0]) if self.dynamic_feats.ndim == 2 else 0,
                "combined_bank_size": 0,
                "ref_size": 0,
                "ref_source": "empty",
                "novelty_percentile": 1.0,
            }
            self.last_debug = info
            return 1.0, info

        sim = q @ bank.T
        max_sim, nn_idx = sim.max(dim=1)
        d_nn = float((1.0 - max_sim).item())

        ref, ref_source = self._combined_ref()
        pct = float((ref <= d_nn).float().mean().item())
        pct = float(np.clip(pct, 0.0, 1.0))

        info = {
            "distance_nn": d_nn,
            "nn_index": int(nn_idx.item()),
            "support_size": int(self.support_feats.shape[0]) if self.support_feats.ndim == 2 else 0,
            "dynamic_size": int(self.dynamic_feats.shape[0]) if self.dynamic_feats.ndim == 2 else 0,
            "ref_size": int(ref.numel()),
            "ref_source": str(ref_source),
            "combined_bank_size": int(bank.shape[0]),
            "novelty_percentile": float(pct),
        }
        self.last_debug = info
        return pct, info

    def observe_normal(self, feat: torch.Tensor) -> None:
        q = l2_normalize(feat.detach().float().cpu().view(1, -1))
        bank_before = self._combined_bank()
        if bank_before.numel() > 0:
            sim = q @ bank_before.T
            d_nn = float((1.0 - sim.max(dim=1).values).item())
            self.dynamic_ref_dists = torch.cat([self.dynamic_ref_dists, torch.tensor([d_nn], dtype=torch.float32)], dim=0)
        if self.dynamic_feats.numel() == 0:
            self.dynamic_feats = q
        else:
            self.dynamic_feats = torch.cat([self.dynamic_feats, q], dim=0)
        if self.max_dynamic > 0 and self.dynamic_feats.shape[0] > self.max_dynamic:
            self.dynamic_feats = self.dynamic_feats[-self.max_dynamic:]
            if self.dynamic_ref_dists.shape[0] > self.max_dynamic:
                self.dynamic_ref_dists = self.dynamic_ref_dists[-self.max_dynamic:]

def _simple_aif_risk_given_theta(p_defect: float, pred_defect: bool, cost_fp: float, cost_fn: float) -> float:
    p = _clamp01(p_defect)
    return (1.0 - p) * float(cost_fp) if bool(pred_defect) else p * float(cost_fn)


def choose_simple_aif_bootstrap_action(
    *,
    step: int,
    score: float,
    p_defect: float,
    theta: float,
    sigmaN: float,
    cfg: SimpleAIFConfig,
    dynamic_normal_count: int,
) -> Tuple[Optional[AIFAction], Optional[str], Dict[str, Any]]:
    """Hybrid bootstrap for v5.

    Goal: seed dynamic normal memory with boundary-adjacent verified normals before
    letting the v3-style free competition dominate. This intentionally suppresses
    top-tail TP queries that waste early query budget on obvious defects.
    """
    if int(dynamic_normal_count) >= int(cfg.bootstrap_dynamic_target):
        return None, None, {
            "bootstrap_active": False,
            "bootstrap_done": True,
            "dynamic_normal_count": int(dynamic_normal_count),
            "dynamic_target": int(cfg.bootstrap_dynamic_target),
        }

    if int(step) > int(cfg.bootstrap_max_steps):
        return None, None, {
            "bootstrap_active": False,
            "bootstrap_done": False,
            "bootstrap_expired": True,
            "dynamic_normal_count": int(dynamic_normal_count),
            "dynamic_target": int(cfg.bootstrap_dynamic_target),
        }

    sig = float(max(sigmaN, 1e-6))
    margin = float(score) - float(theta)
    band_low = max(float(cfg.bootstrap_band_min_abs), float(cfg.bootstrap_band_low_sigma) * sig)
    band_high = max(float(cfg.bootstrap_band_min_abs), float(cfg.bootstrap_band_high_sigma) * sig)
    top_tail = max(float(cfg.bootstrap_top_tail_min_abs), float(cfg.bootstrap_top_tail_sigma) * sig)

    info = {
        "bootstrap_active": True,
        "dynamic_normal_count": int(dynamic_normal_count),
        "dynamic_target": int(cfg.bootstrap_dynamic_target),
        "margin": float(margin),
        "band_low": float(band_low),
        "band_high": float(band_high),
        "top_tail": float(top_tail),
        "p_defect": float(p_defect),
        "theta": float(theta),
        "sigmaN": float(sig),
    }

    if float(margin) > float(top_tail):
        info["bootstrap_region"] = "top_tail_skip"
        return AIFAction.CLASSIFY, "bootstrap_skip_top_tail", info

    if (-float(band_low) <= float(margin) <= float(band_high)):
        info["bootstrap_region"] = "boundary_band"
        return AIFAction.QUERY_LABEL, "bootstrap_boundary_band", info

    info["bootstrap_region"] = "outside_band"
    return AIFAction.CLASSIFY, "bootstrap_outside_band", info


def select_informative_normal_write(
    *,
    score: float,
    theta: float,
    sigmaN: float,
    pred_defect: bool,
    novelty: float,
) -> Tuple[bool, int, str, Dict[str, Any]]:
    """Priority-weighted STM write plan.

    Important correction to v7: this is NOT reject-first. Every queried verified
    normal leaves some STM footprint; informative cases simply receive a larger
    patch budget and higher write value.
    """
    sig = float(max(sigmaN, 1e-6))
    margin = float(score) - float(theta)
    sig_eff = float(max(sig, 0.02))

    boundary = math.exp(-abs(margin) / sig_eff)
    fp_rescue = 1.0 if (bool(pred_defect) and (margin >= 0.0)) else 0.0
    novelty01 = float(np.clip(float(novelty), 0.0, 1.0))

    write_value = float(np.clip(0.45 * boundary + 0.35 * fp_rescue + 0.20 * novelty01, 0.0, 1.0))

    if fp_rescue >= 1.0:
        topk_keep = 24
        reason = "fp_rescue"
    elif write_value >= 0.60:
        topk_keep = 16
        reason = "boundary_or_novel"
    else:
        topk_keep = 8
        reason = "easy_but_keep_small"

    info = {
        "margin": float(margin),
        "sigmaN": float(sig),
        "boundary": float(boundary),
        "fp_rescue": float(fp_rescue),
        "novelty": float(novelty01),
        "reason": str(reason),
        "topk_keep": int(topk_keep),
        "write_value": float(write_value),
    }
    return True, int(topk_keep), str(reason), info


def compute_local_saliency_simple(A_map_stats: Dict[str, Any]) -> float:
    per_layer = A_map_stats.get("per_layer_summary", {}) or {}
    vals: List[float] = []
    for _lk, info in per_layer.items():
        if not isinstance(info, dict):
            continue
        dmax = float(info.get("topk_dN_max", 0.0))
        dmean = float(info.get("topk_dN_mean", 0.0))
        denom = max(abs(dmax), abs(dmean), 1e-6)
        vals.append(max(0.0, (dmax - dmean) / denom))
    if len(vals) == 0:
        return 1.0
    return float(np.clip(float(np.mean(vals)), 0.0, 1.0))


def compute_simple_correction_gap(
    *,
    score: float,
    theta: float,
    sigmaN: float,
    evidence: Dict[str, float],
    A_map_stats: Dict[str, Any],
    cfg: SimpleAIFConfig,
) -> Tuple[float, Dict[str, Any]]:
    sig = float(max(sigmaN, 1e-6))
    boundary = math.exp(-0.5 * (abs(float(score) - float(theta)) / sig) ** 2)
    corr_def = float(evidence.get("corr_def", 0.0))
    corr_norm = float(evidence.get("corr_norm", 0.0))
    corr_mag = max(corr_def, corr_norm)
    corr_gap = float(np.clip(corr_mag / max(float(cfg.correction_gap_scale), 1e-6), 0.0, 1.0))
    info = {
        "boundary": float(boundary),
        "corr_def": float(corr_def),
        "corr_norm": float(corr_norm),
        "corr_mag": float(corr_mag),
        "corr_gap": float(corr_gap),
    }
    return float(boundary * corr_gap), info




def compute_simple_defect_side_epistemic(
    *,
    p_defect: float,
    pred_defect: bool,
    score: float,
    theta: float,
    sigmaN: float,
    evidence: Dict[str, float],
    cfg: SimpleAIFConfig,
) -> Tuple[float, Dict[str, Any]]:
    """Defect-side epistemic utility for suspicious predicted normals.

    This term is high only when:
      - the current prediction is normal,
      - the posterior defect probability is not tiny,
      - the sample lies moderately below theta (not too far, not glued to theta),
      - and local correction evidence is still weak.
    """
    if bool(pred_defect):
        info = {
            "active": False,
            "reason": "pred_defect",
            "margin": float(score) - float(theta),
            "weak_corr": 0.0,
            "gate_far": 0.0,
            "gate_near": 0.0,
            "near_margin": 0.0,
            "far_margin": 0.0,
            "tau_margin": 0.0,
        }
        return 0.0, info

    sig = float(max(sigmaN, 1e-6))
    margin = float(score) - float(theta)

    near_margin = max(float(cfg.defect_near_min_abs), float(cfg.defect_near_sigma) * sig)
    far_margin = max(float(cfg.defect_far_min_abs), float(cfg.defect_far_sigma) * sig)
    tau_margin = max(float(cfg.defect_tau_min_abs), float(cfg.defect_tau_sigma) * sig)

    # Soft band on the normal side of theta:
    # - gate_far suppresses deep normals
    # - gate_near suppresses samples already handled by pure entropy at the threshold
    gate_far = float(_sigmoid((margin + far_margin) / max(tau_margin, 1e-6)))
    gate_near = float(_sigmoid(((-margin) - near_margin) / max(tau_margin, 1e-6)))

    corr_def = float(evidence.get("corr_def", 0.0))
    corr_norm = float(evidence.get("corr_norm", 0.0))
    corr_mag = max(corr_def, corr_norm)
    weak_corr = 1.0 - float(np.clip(
        corr_mag / max(float(cfg.defect_weak_corr_scale), 1e-6),
        0.0,
        1.0,
    ))

    value = float(_clamp01(p_defect) * gate_far * gate_near * weak_corr)

    info = {
        "active": True,
        "margin": float(margin),
        "p_defect": float(_clamp01(p_defect)),
        "weak_corr": float(weak_corr),
        "gate_far": float(gate_far),
        "gate_near": float(gate_near),
        "near_margin": float(near_margin),
        "far_margin": float(far_margin),
        "tau_margin": float(tau_margin),
        "corr_mag": float(corr_mag),
    }
    return value, info



def select_simple_aif_action(
    *,
    cfg: SimpleAIFConfig,
    state: SimpleAIFState,
    ablation: str,
    p_defect: float,
    score: float,
    theta: float,
    sigmaN: float,
    pred_defect: bool,
    novelty: float,
    novelty_info: Dict[str, Any],
    evidence: Dict[str, float],
    A_map_stats: Dict[str, Any],
) -> Tuple["AIFAction", Dict[str, Any]]:
    H = _entropy_bern(p_defect)
    G_classify = _simple_aif_risk_given_theta(
        p_defect=p_defect,
        pred_defect=pred_defect,
        cost_fp=float(cfg.cost_fp),
        cost_fn=float(cfg.cost_fn),
    )

    sig = float(max(sigmaN, 1e-6))
    boundary = math.exp(-0.5 * (abs(float(score) - float(theta)) / sig) ** 2)

    enable_normal_term = ablation in {"normal_dyn_only", "normal_dyn_plus_correction"}
    enable_corr_term = ablation in {"correction_only", "normal_dyn_plus_correction"}

    # Broad normal-memory value: keep the v9 mechanism unchanged so the patch remains narrow.
    U_normal = float(boundary * _clamp01(novelty)) if enable_normal_term else 0.0

    corr_gap_raw, corr_info = compute_simple_correction_gap(
        score=score,
        theta=theta,
        sigmaN=sigmaN,
        evidence=evidence,
        A_map_stats=A_map_stats,
        cfg=cfg,
    )
    local_saliency = compute_local_saliency_simple(A_map_stats) if bool(cfg.use_local_saliency) else 1.0
    U_correction = float(
        corr_gap_raw * (1.0 + float(cfg.saliency_weight) * max(0.0, local_saliency - 0.5))
    ) if enable_corr_term else 0.0

    # New in v10: defect-side epistemic utility for suspicious predicted normals.
    B_defect, defect_info = compute_simple_defect_side_epistemic(
        p_defect=p_defect,
        pred_defect=pred_defect,
        score=score,
        theta=theta,
        sigmaN=sigmaN,
        evidence=evidence,
        cfg=cfg,
    )

    G_query = (
        float(state.query_cost_runtime)
        - float(cfg.beta_entropy) * float(H)
        - float(cfg.lambda_defect_side) * float(B_defect)
        - float(cfg.lambda_normal) * float(U_normal)
        - float(cfg.lambda_correction) * float(U_correction)
    )

    action = AIFAction.QUERY_LABEL if float(G_query) < float(G_classify) else AIFAction.CLASSIFY
    info = {
        "controller": "simple_aif",
        "version": "v10",
        "action": action.value,
        "G": {
            AIFAction.CLASSIFY.value: float(G_classify),
            AIFAction.QUERY_LABEL.value: float(G_query),
        },
        "G_val": [float(G_classify), float(G_query)],
        "entropy": float(H),
        "risk_classify": float(G_classify),
        "query_cost_runtime": float(state.query_cost_runtime),
        "local_qrate_runtime": float(state.local_qrate()),
        "boundary_closeness": float(boundary),
        "novelty": float(novelty),
        "novelty_info": novelty_info,
        "B_defect": float(B_defect),
        "defect_info": defect_info,
        "U_normal": float(U_normal),
        "U_correction": float(U_correction),
        "correction_info": corr_info,
        "correction_evidence": {
            "corr_mean": float(evidence.get("corr_mean", 0.0)),
            "corr_max": float(evidence.get("corr_max", 0.0)),
            "corr_min": float(evidence.get("corr_min", 0.0)),
            "corr_def": float(evidence.get("corr_def", 0.0)),
            "corr_norm": float(evidence.get("corr_norm", 0.0)),
            "corr_bank_nonempty": bool(evidence.get("corr_bank_nonempty", False)),
            "corr_signal_dead": bool(evidence.get("corr_signal_dead", False)),
        },
        "local_saliency": float(local_saliency),
        "use_local_saliency": bool(cfg.use_local_saliency),
        "feasible_actions": [AIFAction.CLASSIFY.value, AIFAction.QUERY_LABEL.value],
        "ablation": str(ablation),
    }
    return action, info


@dataclass
class AIFConfig:
    # Preference weights (temporary defaults; user-approved)
    cost_fp: float = 1.0
    cost_fn: float = 5.0
    cost_query_label: float = 1.0
    cost_query_region: float = 2.0
    cost_query_support: float = 4.0

    # Weighting of EFE terms
    w_risk: float = 1.0
    w_query: float = 1.0
    w_epi: float = 1.0
    w_calib: float = 1.0
    w_rel: float = 1.0
    w_contam: float = 1.0

    # Calibration penalty (NormalThresholdController)
    calib_gap_cost: float = 1.0          # penalize insufficient bufN for stable theta
    calib_sigma_cost: float = 0.10       # mild penalty for large sigmaN
    calib_theta_unc_cost: float = 0.25   # penalize high theta uncertainty

    # Reliability penalty
    rel_untrust_cost: float = 1.0        # penalize low model reliability

    # Feedback-channel reliability (operator / label source)
    # Use a mild prior toward "usually correct" but not perfect.
    fb_alpha0: float = 8.0
    fb_beta0: float = 2.0

    # Suspicious-label detection (two evidence channels)
    z_strong: float = 2.5
    z_very_strong: float = 4.0
    corr_strong: float = 0.05

    # Trust gating for memory writes
    trust_min_write: float = 0.55

    # Action params
    region_k: int = 64
    support_budget_per_class: int = 8
    min_query_gain: float = 0.0

    # Epistemic gain decay (shrinks query incentive as calibration + coverage improve)
    epi_floor: float = 0.05          # minimum fraction of epistemic gain to keep (drift sentinel)
    tau_theta_unc: float = 0.010     # scale for theta uncertainty (sigmaN/sqrt(bufN_len))
    tau_novelty: float = 0.50        # novelty scale for coverage confidence
    tau_rb_error: float = 0.30       # RB error scale for coverage confidence
    epi_cal_w: float = 0.60          # weight of calibration confidence
    epi_cov_w: float = 0.40          # weight of coverage confidence

    # During explicit warm-up, prevent calibration confidence from saturating too early.
    warmup_max_cal_conf: float = 0.15

    # Regime identification / regime-conditioned control
    regime_theta_unc_high: float = 0.020
    regime_novelty_high: float = 0.80
    regime_rb_error_high: float = 0.25
    regime_conflict_high: float = 0.50
    regime_boundary_margin_sigma: float = 1.25
    support_query_pmax: float = 0.40
    normal_write_novelty_min: float = 0.70
    normal_write_margin_sigma: float = 1.25
    defect_write_margin_sigma: float = 1.50
    correction_theta_unc_scale: float = 0.020
    correction_mistake_bonus: float = 1.20
    action_bias_calib_label: float = 0.80
    action_bias_coverage_support: float = 0.90
    action_bias_reason_region: float = 1.10
    action_bias_stable_classify: float = 0.35


@dataclass
class BetaReliability:
    alpha: float = 1.0
    beta: float = 1.0

    def mean(self) -> float:
        return float(self.alpha / max(self.alpha + self.beta, 1e-8))

    def update(self, correct: bool, w: float = 1.0) -> None:
        w = float(max(0.0, w))
        if correct:
            self.alpha += w
        else:
            self.beta += w


@dataclass
class CalibBelief:
    """Calibration state as part of s_rel.

    We treat NormalThresholdController outputs as sufficient statistics for calibration,
    and store *uncertainty proxies* so AIF can act epistemically to stabilize calibration.
    """
    theta: float = 0.0
    sigmaN: float = 1e-3
    bufN_len: int = 0
    min_bufN: int = 0
    eff_n: float = 1.0
    theta_unc: float = 1.0

    def update(self, *, theta: float, sigmaN: float, bufN_len: int, min_bufN: int, eff_n: Optional[float] = None) -> None:
        self.theta = float(theta)
        self.sigmaN = float(max(sigmaN, 1e-6))
        self.bufN_len = int(bufN_len)
        self.min_bufN = int(min_bufN)
        # crude uncertainty: sigma / sqrt(n)
        #n = max(1, self.bufN_len)
        #self.theta_unc = float(self.sigmaN / math.sqrt(float(n)))
        self.eff_n = float(max(1.0, eff_n if eff_n is not None else max(1, self.bufN_len)))
        self.theta_unc = float(self.sigmaN / math.sqrt(float(self.eff_n)))

    def gap(self) -> float:
        if self.min_bufN <= 0:
            return 0.0
        g = max(0, int(self.min_bufN) - int(self.bufN_len))
        return float(g) / float(self.min_bufN)


@dataclass
class AIFBelief:
    # Reliability of the *current* vision decision pipeline for this class
    r_vmb: BetaReliability = field(default_factory=BetaReliability)

    # Reliability of the feedback channel (operator / dataset label source)
    r_user: BetaReliability = field(default_factory=lambda: BetaReliability(alpha=8.0, beta=2.0))

    # Calibration state (explicit in s_rel)
    calib: CalibBelief = field(default_factory=CalibBelief)


def compute_two_channel_evidence(*, score: float, theta: float, sigmaN: float, A_map_stats: Dict[str, Any]) -> Dict[str, float]:
    """Two evidence channels:
    - channel-1: margin-to-theta (normalized)
    - channel-2: correction activation statistics (signed)
    """
    sig = float(max(sigmaN, 1e-6))
    margin_z = (float(score) - float(theta)) / sig

    per_layer = A_map_stats.get('per_layer_summary', {}) or {}
    corr_means = []
    corr_maxs = []
    corr_mins = []
    for _lk, info in per_layer.items():
        if not isinstance(info, dict):
            continue
        corr_means.append(float(info.get('corr_top_mean', 0.0)))
        corr_maxs.append(float(info.get('corr_top_max', 0.0)))
        # corr_top_max can be negative if all are negative; min gives stronger normal-evidence
        #corr_mins.append(float(info.get('corr_top_max', 0.0)))
        corr_mins.append(float(info.get('corr_top_min', 0.0)))

    corr_mean = float(sum(corr_means) / max(1, len(corr_means))) if corr_means else 0.0
    corr_max = float(max(corr_maxs)) if corr_maxs else 0.0
    corr_min = float(min(corr_mins)) if corr_mins else 0.0

    corr_def = max(0.0, corr_max)          # positive correction indicates defect-side anchors
    corr_norm = max(0.0, -corr_min)        # negative correction indicates normal-side anchors

    corr_bank_sizes = [int(info.get('corr_bank_size', 0)) for info in per_layer.values() if isinstance(info, dict)]
    corr_bank_nonempty = bool(any(s > 0 for s in corr_bank_sizes))
    corr_signal_dead = bool(corr_bank_nonempty and (abs(corr_mean) <= 1e-12) and (abs(corr_max) <= 1e-12) and (abs(corr_min) <= 1e-12))

    return {
        'margin_z': float(margin_z),
        'corr_mean': float(corr_mean),
        'corr_max': float(corr_max),
        'corr_min': float(corr_min),
        'corr_def': float(corr_def),
        'corr_norm': float(corr_norm),
        'corr_bank_nonempty': bool(corr_bank_nonempty),
        'corr_signal_dead': bool(corr_signal_dead),
    }


def is_label_suspicious(*, y_user: int, evidence: Dict[str, float], cfg: AIFConfig) -> bool:
    """Detect suspicious feedback using TWO evidence channels.

    We only flag as suspicious when:
      - channel-1 is strongly confident in the opposite class
      - AND channel-2 supports the same opposite direction (when available)
    If correction evidence is absent/weak, require a very strong margin.
    """
    y = int(y_user)
    mz = float(evidence.get('margin_z', 0.0))
    cdef = float(evidence.get('corr_def', 0.0))
    cnorm = float(evidence.get('corr_norm', 0.0))

    if y == 0:
        # label says normal; suspicious if model strongly says defect
        if mz > float(cfg.z_strong) and cdef > float(cfg.corr_strong):
            return True
        if mz > float(cfg.z_very_strong) and cdef <= float(cfg.corr_strong):
            return True
    else:
        # label says defect; suspicious if model strongly says normal
        if mz < -float(cfg.z_strong) and cnorm > float(cfg.corr_strong):
            return True
        if mz < -float(cfg.z_very_strong) and cnorm <= float(cfg.corr_strong):
            return True
    return False




def compute_correction_strength(
    *,
    score: float,
    theta: float,
    sigmaN: float,
    novelty: float,
    rb_error: float,
    theta_unc: float,
    cfg: AIFConfig,
) -> float:
    sig = float(max(sigmaN, 1e-6))
    margin_z = abs(float(score) - float(theta)) / sig
    ambiguity = math.exp(-0.5 * (margin_z ** 2))  # strongest near boundary
    nov_term = _clamp01(float(novelty))
    rb_term = min(1.0, float(rb_error) / max(float(cfg.regime_rb_error_high), 1e-6))
    unc_term = min(1.0, float(theta_unc) / max(float(cfg.correction_theta_unc_scale), 1e-6))
    strength = 0.45 * ambiguity + 0.25 * nov_term + 0.20 * rb_term + 0.10 * unc_term
    return float(np.clip(strength, 0.35, 1.75))


def should_write_normal_from_feedback(
    *,
    outcome: str,
    novelty: float,
    score: float,
    theta: float,
    sigmaN: float,
    regime: AIFRegime,
    cfg: AIFConfig,
) -> bool:
    near_boundary = abs(float(score) - float(theta)) <= float(cfg.normal_write_margin_sigma) * float(max(sigmaN, 1e-6))
    if str(outcome) == 'FP':
        return True
    if regime == AIFRegime.COVERAGE_POOR and float(novelty) >= float(cfg.normal_write_novelty_min):
        return True
    return bool(near_boundary and (float(novelty) >= float(cfg.normal_write_novelty_min)))


class AIFController:
    def __init__(self, cfg: AIFConfig):
        self.cfg = cfg
        self._support_left: Dict[str, int] = {}
        self._beliefs: Dict[str, AIFBelief] = {}

    def belief(self, cls_name: str) -> AIFBelief:
        if cls_name not in self._beliefs:
            b = AIFBelief()
            # re-init r_user with configured prior
            b.r_user = BetaReliability(alpha=float(self.cfg.fb_alpha0), beta=float(self.cfg.fb_beta0))
            self._beliefs[cls_name] = b
        return self._beliefs[cls_name]

    def sync_calibration(self, cls_name: str, *, theta: float, sigmaN: float, bufN_len: int, min_bufN: int, eff_n: Optional[float] = None) -> None:
        b = self.belief(cls_name)
        b.calib.update(theta=float(theta), sigmaN=float(sigmaN), bufN_len=int(bufN_len), min_bufN=int(min_bufN), eff_n=eff_n)

    def support_left(self, cls_name: str) -> int:
        if cls_name not in self._support_left:
            self._support_left[cls_name] = int(self.cfg.support_budget_per_class)
        return int(self._support_left[cls_name])

    def consume_support(self, cls_name: str) -> None:
        self._support_left[cls_name] = max(0, self.support_left(cls_name) - 1)

    @staticmethod
    def _risk_given_theta(p_defect: float, pred_defect: bool, cost_fp: float, cost_fn: float) -> float:
        p = _clamp01(p_defect)
        if bool(pred_defect):
            return (1.0 - p) * float(cost_fp)
        return p * float(cost_fn)

    def _expected_calib_gain_one_normal(self, b: AIFBelief) -> float:
        g0 = b.calib.gap()
        # simulate adding one verified-normal
        tmp = CalibBelief(theta=b.calib.theta, sigmaN=b.calib.sigmaN, bufN_len=b.calib.bufN_len, min_bufN=b.calib.min_bufN, theta_unc=b.calib.theta_unc)
        tmp.update(theta=b.calib.theta, sigmaN=b.calib.sigmaN, bufN_len=b.calib.bufN_len + 1, min_bufN=b.calib.min_bufN)
        g1 = tmp.gap()
        return float(max(0.0, g0 - g1))
    
    @staticmethod
    def _expected_theta_unc_gain_one_normal(*, sigmaN: float, eff_n0: float, eff_n1: float) -> float:
        sig = float(max(sigmaN, 1e-6))
        n0 = float(max(1.0, eff_n0))
        n1 = float(max(n0 + 1e-6, eff_n1))
        unc0 = float(sig / math.sqrt(n0))
        unc1 = float(sig / math.sqrt(n1))
        return float(max(0.0, unc0 - unc1))
    

    def update_vmb_reliability(self, cls_name: str, *, correct: bool, w: float = 1.0) -> None:
        b = self.belief(cls_name)
        b.r_vmb.update(bool(correct), w=float(w))

    def update_user_reliability(self, cls_name: str, *, consistent: bool, w: float = 1.0) -> None:
        b = self.belief(cls_name)
        b.r_user.update(bool(consistent), w=float(w))

# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def auc_roc(scores: List[float], labels: List[int]) -> float:
    """Simple AUROC implementation without sklearn."""
    if len(scores) == 0:
        return 0.0
    # rank scores
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)

    labels_np = np.asarray(labels, dtype=np.int32)
    pos = labels_np == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    sum_ranks_pos = float(ranks[pos].sum())
    # Mann–Whitney U
    U = sum_ranks_pos - n_pos * (n_pos - 1) / 2.0
    return float(U / (n_pos * n_neg))


#------------------------------------------------------------------------------
# Newly added functions
#------------------------------------------------------------------------------

@torch.no_grad()
def load_rgb(path: str) -> Image.Image:
    # avoid file-handle leaks
    with Image.open(path) as im:
        return im.convert("RGB")

@torch.no_grad()
def encode_pil(backbone: Any, img: Image.Image, mode: str = 'concat') -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Compatibility wrapper: vmb_visionad_new backbone exposes encode_batch; older code used encode_pil."""
    if hasattr(backbone, "encode_pil"):
        feats_per_layer, g = backbone.encode_pil(img)
        feats = fuse_patch_features([t.detach() for t in feats_per_layer], mode) #fuse_patch_features(feats, mode)
        return feats, g.detach()
    patches_per_layer, globals_b = backbone.encode_batch([img])
    feats = [p[0].detach() for p in patches_per_layer]  # list[N,C]
    g = globals_b[0].detach()
    feats = fuse_patch_features(feats, mode)
    return feats, g

def fuse_patch_features(feats_per_layer: List[torch.Tensor], mode: str = "concat") -> List[torch.Tensor]:
    """VisionAD-style token-level feature fusion (1-group).

    Input: list of [N,C_l] tensors (same N).
    Output: list with a single [N,D] tensor (or unchanged if mode='none').
    """
    if mode == "none":
        return feats_per_layer
    if len(feats_per_layer) == 0:
        raise ValueError("No layer features to fuse")
    if mode == "concat":
        z = torch.cat(feats_per_layer, dim=-1)
        return [l2_normalize(z)]
    if mode == "sum":
        z = torch.stack(feats_per_layer, dim=0).sum(dim=0)
        return [l2_normalize(z)]
    raise ValueError(mode)

# ---------------------------------------------------------------------------
#  Main streaming experiment (Phase-2: RB-driven prototype updates + dual banks)
# ---------------------------------------------------------------------------


def compute_rb_error_rate(  #  estimates a risk score from ReasoningBank concept history for the current sample
    rb: ReasoningBank,
    cls_name: str,
    A_map_stats: Dict[str, Any],
    *,
    pred_defect: bool,
    min_count: int = 3,
) -> float:
    per_layer = A_map_stats.get("per_layer_summary", {})
    errs: List[float] = []
    for lk, info in per_layer.items():
        layer = int(lk.replace("layer", ""))
        for cid in info.get("topk_concept_ids", []):
            st = rb.get_concept_stats(cls_name, layer, int(cid))
            if st.total < min_count:
                continue
            errs.append(st.fp_rate() if pred_defect else st.fn_rate())
    return float(max(errs)) if errs else 0.0

def map_layer_patch_idx_to_view(
    layer_to_patch_idx: Dict[int, List[int]],
    v: ViewSpec,
    n_patches: int,
) -> Dict[int, List[int]]:
    H, W = _grid_hw_from_npatches(n_patches)
    out: Dict[int, List[int]] = {}
    for l, idxs in layer_to_patch_idx.items():
        out[l] = [map_patch_index_identity_to_view(int(i), v, H, W) for i in idxs]
    return out


def layer_to_topk_from_amap(A_map_stats: Dict[str, Any]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    per_layer = A_map_stats.get("per_layer_summary", {}) or {}
    for lk, info in per_layer.items():
        try:
            layer = int(str(lk).replace("layer", ""))
        except Exception:
            continue
        out[layer] = [int(i) for i in info.get("topk_patch_idx", [])]
    return out


def _qual_corr_sanitize_name(x: Any) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s or "item"


def _qual_corr_extract_rough_stats(A_map_stats: Dict[str, Any]) -> Dict[str, float]:
    vals_max: List[float] = []
    vals_min: List[float] = []
    vals_mean: List[float] = []
    for _lk, info in (A_map_stats.get("per_layer_summary", {}) or {}).items():
        vals_max.append(float(info.get("corr_top_max", 0.0)))
        vals_min.append(float(info.get("corr_top_min", 0.0)))
        vals_mean.append(float(info.get("corr_top_mean", 0.0)))
    return {
        "corr_top_max": float(max(vals_max)) if vals_max else 0.0,
        "corr_top_min": float(min(vals_min)) if vals_min else 0.0,
        "corr_top_mean": float(np.mean(vals_mean)) if vals_mean else 0.0,
    }


def _qual_corr_resize_map(A: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest/bilinear-free PIL resize helper for visualization only."""
    h, w = int(image_hw[0]), int(image_hw[1])
    arr = np.asarray(A, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    den = max(a_max - a_min, 1e-8)
    norm = ((arr - a_min) / den * 255.0).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(norm, mode="L").resize((w, h), resample=Image.BILINEAR)
    out = np.asarray(im).astype(np.float32) / 255.0
    return out * den + a_min


def _qual_corr_save_panel(
    *,
    img: Image.Image,
    A_pre: np.ndarray,
    A_corr: np.ndarray,
    A_post: np.ndarray,
    out_png: str,
    title: str,
) -> None:
    safe_makedirs(out_png)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    axes[0].imshow(img)
    axes[0].set_title("image")
    axes[0].axis("off")

    maps = [A_pre, A_corr, A_post, A_post - A_pre]
    titles = ["pre-correction", "signed residual", "post-correction", "post - pre"]
    cmaps = ["inferno", "coolwarm", "inferno", "coolwarm"]
    for ax, arr, ttl, cmap in zip(axes[1:], maps, titles, cmaps):
        arr = np.asarray(arr, dtype=np.float32)
        if cmap == "coolwarm":
            vmax = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
            vmax = max(vmax, 1e-6)
            im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(arr, cmap=cmap)
        ax.set_title(ttl)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def maybe_dump_correction_qual_example(
    *,
    qual_corr_dump_dir: Optional[str],
    qual_corr_counts: Dict[Tuple[str, str], int],
    qual_corr_records: List[Dict[str, Any]],
    qual_corr_max_per_bucket: int,
    qual_corr_min_abs: float,
    qual_corr_save_flips_only: bool,
    class_name: str,
    sample_path: str,
    img: Image.Image,
    t: int,
    global_t: int,
    y_true: int,
    outcome: str,
    theta: float,
    score_img: float,
    pred_defect: bool,
    maps: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not qual_corr_dump_dir:
        return None
    A_pre = np.asarray(maps.get("A_pre"), dtype=np.float32)
    A_post = np.asarray(maps.get("A_post"), dtype=np.float32)
    A_corr = np.asarray(maps.get("A_corr"), dtype=np.float32)
    score_pre = float(maps.get("score_pre", score_img))
    score_post = float(maps.get("score_post", score_img))
    corr_min = float(maps.get("corr_min", 0.0))
    corr_max = float(maps.get("corr_max", 0.0))
    corr_mean = float(maps.get("corr_mean", 0.0))
    corr_abs_mean = float(maps.get("corr_abs_mean", 0.0))
    pre_pred = bool(score_pre >= float(theta))
    post_pred = bool(score_post >= float(theta))

    bucket = None
    if int(y_true) == 0 and pre_pred and (not post_pred):
        bucket = "FP_to_TN_negative_rescue"
    elif int(y_true) == 1 and (not pre_pred) and post_pred:
        bucket = "FN_to_TP_positive_rescue"
    elif (not qual_corr_save_flips_only) and int(y_true) == 0 and corr_min <= -abs(float(qual_corr_min_abs)):
        bucket = "negative_normal_correction"
    elif (not qual_corr_save_flips_only) and int(y_true) == 1 and corr_max >= abs(float(qual_corr_min_abs)):
        bucket = "positive_defect_correction"
    if bucket is None:
        return None

    key = (str(class_name), str(bucket))
    if int(qual_corr_counts.get(key, 0)) >= int(max(1, qual_corr_max_per_bucket)):
        return None

    cls_safe = _qual_corr_sanitize_name(class_name)
    bucket_safe = _qual_corr_sanitize_name(bucket)
    base_dir = os.path.join(str(qual_corr_dump_dir), cls_safe, bucket_safe)
    os.makedirs(base_dir, exist_ok=True)
    stem = f"{cls_safe}_t{int(t):05d}_gt{int(y_true)}_{str(outcome)}_{bucket_safe}"
    out_png = os.path.join(base_dir, stem + ".png")
    out_npz = os.path.join(base_dir, stem + ".npz")
    out_json = os.path.join(base_dir, stem + ".json")
    title = (
        f"{class_name} t={t} outcome={outcome} theta={theta:.4f} "
        f"score_pre={score_pre:.4f} score_post={score_post:.4f} bucket={bucket}"
    )
    _qual_corr_save_panel(img=img, A_pre=A_pre, A_corr=A_corr, A_post=A_post, out_png=out_png, title=title)
    np.savez_compressed(out_npz, A_pre=A_pre, A_corr=A_corr, A_post=A_post, A_diff=A_post - A_pre)

    rec = {
        "class": str(class_name),
        "t": int(t),
        "global_t": int(global_t),
        "path": str(sample_path),
        "true": int(y_true),
        "outcome": str(outcome),
        "theta": float(theta),
        "score_img": float(score_img),
        "score_pre": float(score_pre),
        "score_post": float(score_post),
        "delta_score": float(score_post - score_pre),
        "pred": int(pred_defect),
        "pre_pred": int(pre_pred),
        "post_pred": int(post_pred),
        "corr_min": float(corr_min),
        "corr_max": float(corr_max),
        "corr_mean": float(corr_mean),
        "corr_abs_mean": float(corr_abs_mean),
        "bucket": str(bucket),
        "png_path": str(out_png),
        "npz_path": str(out_npz),
        "json_path": str(out_json),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    qual_corr_counts[key] = int(qual_corr_counts.get(key, 0)) + 1
    qual_corr_records.append(rec)
    return rec


def save_correction_qualitative_outputs(
    *,
    qual_corr_dump_dir: Optional[str],
    class_name: str,
    qual_corr_records: List[Dict[str, Any]],
    vmb: Any,
) -> Dict[str, Any]:
    if not qual_corr_dump_dir:
        return {}
    os.makedirs(str(qual_corr_dump_dir), exist_ok=True)
    cls_safe = _qual_corr_sanitize_name(class_name)
    examples_index = os.path.join(str(qual_corr_dump_dir), f"examples_index_{cls_safe}.csv")
    examples_index_all = os.path.join(str(qual_corr_dump_dir), "examples_index_all.csv")
    fieldnames = [
        "class", "t", "global_t", "path", "true", "outcome", "theta", "score_img",
        "score_pre", "score_post", "delta_score", "pred", "pre_pred", "post_pred",
        "corr_min", "corr_max", "corr_mean", "corr_abs_mean", "bucket", "png_path", "npz_path", "json_path",
    ]
    with open(examples_index, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in qual_corr_records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    append_all = os.path.exists(examples_index_all)
    with open(examples_index_all, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append_all:
            writer.writeheader()
        for r in qual_corr_records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    anchor_csv = os.path.join(str(qual_corr_dump_dir), f"anchor_sign_distribution_{cls_safe}.csv")
    anchor_csv_all = os.path.join(str(qual_corr_dump_dir), "anchor_sign_distribution_all.csv")
    mem = {}
    corr = {}
    try:
        mem = vmb.memory_summary(class_name)
        corr = mem.get("correction_bank", {}) if isinstance(mem, dict) else {}
    except Exception:
        corr = {}
    anchor_fields = [
        "class", "total_correction_anchors", "positive_anchors", "negative_anchors",
        "fp_anchor_count", "fn_anchor_count", "tn_anchor_count", "tp_anchor_count",
        "inserted_total", "merged_total", "replaced_total", "same_sign_replaced_total", "cross_sign_replaced_total",
    ]
    anchor_row = {
        "class": str(class_name),
        "total_correction_anchors": int(corr.get("total", 0) or 0),
        "positive_anchors": int(corr.get("positive", 0) or 0),
        "negative_anchors": int(corr.get("negative", 0) or 0),
        "fp_anchor_count": int(corr.get("fp_anchor_count", 0) or 0),
        "fn_anchor_count": int(corr.get("fn_anchor_count", 0) or 0),
        "tn_anchor_count": int(corr.get("tn_anchor_count", 0) or 0),
        "tp_anchor_count": int(corr.get("tp_anchor_count", 0) or 0),
        "inserted_total": int(corr.get("inserted_total", 0) or 0),
        "merged_total": int(corr.get("merged_total", 0) or 0),
        "replaced_total": int(corr.get("replaced_total", 0) or 0),
        "same_sign_replaced_total": int(corr.get("same_sign_replaced_total", 0) or 0),
        "cross_sign_replaced_total": int(corr.get("cross_sign_replaced_total", 0) or 0),
    }
    with open(anchor_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=anchor_fields)
        writer.writeheader()
        writer.writerow(anchor_row)
    append_anchor_all = os.path.exists(anchor_csv_all)
    with open(anchor_csv_all, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=anchor_fields)
        if not append_anchor_all:
            writer.writeheader()
        writer.writerow(anchor_row)

    return {
        "qual_corr_dump_dir": str(qual_corr_dump_dir),
        "examples_index_csv": examples_index,
        "examples_index_all_csv": examples_index_all,
        "anchor_sign_distribution_csv": anchor_csv,
        "anchor_sign_distribution_all_csv": anchor_csv_all,
        "n_examples_saved": int(len(qual_corr_records)),
        "anchor_sign_distribution": anchor_row,
    }



# ---------------------------------------------------------------------------
#  Experiment helpers: temporal post-processing, resume state, and panel/revisit
# ---------------------------------------------------------------------------

STREAM_RECORD_RE = re.compile(
    r"\[t=(?P<t>\d+)\]\s+mode=(?P<mode>-?\d+)\s+score=(?P<score>[-+0-9.eE]+)\s+"
    r"theta=(?P<theta>[-+0-9.eE]+)\s+p=(?P<p>[-+0-9.eE]+)\s+pred=(?P<pred>\d+)\s+"
    r"true=(?P<true>\d+)\s+outcome=(?P<outcome>[A-Z]+)\s+acc=(?P<acc>[-+0-9.eE]+)\s+"
    r"qrate=(?P<qrate>[-+0-9.eE]+)\s+regime=(?P<regime>[^\s]+)\s+action=(?P<action>[^\s]+)"
)


def parse_comma_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(',') if x.strip()]


def parse_stream_record(record_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not record_path or (not os.path.isfile(record_path)):
        return records
    with open(record_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = STREAM_RECORD_RE.search(line.strip())
            if not m:
                continue
            gd = m.groupdict()
            action = gd['action']
            rec = {
                't': int(gd['t']),
                'mode': int(gd['mode']),
                'score': float(gd['score']),
                'theta': float(gd['theta']),
                'p_defect': float(gd['p']),
                'pred': int(gd['pred']),
                'true': int(gd['true']),
                'outcome': gd['outcome'],
                'acc_cum': float(gd['acc']),
                'qrate_cum': float(gd['qrate']),
                'regime': gd['regime'],
                'action': action,
                'queried': bool('query_label' in action),
                'correct': bool(gd['outcome'] in ('TP', 'TN')),
            }
            records.append(rec)
    return records


def summarize_stream_records(
    records: List[Dict[str, Any]],
    *,
    later_frac: float = 0.5,
    early_n: int = 200,
) -> Dict[str, Any]:
    if len(records) == 0:
        return {'n_records': 0}

    n = len(records)
    scores = [float(r['score']) for r in records]
    labels = [int(r['true']) for r in records]
    queried = [1 if r['queried'] else 0 for r in records]
    correct = [1 if r['correct'] else 0 for r in records]

    split = int(max(1, min(n - 1, round(float(n) * float(later_frac))))) if n > 1 else 1
    early = records[:split]
    late = records[split:] if split < n else records[-1:]
    head = records[: min(int(max(1, early_n)), n)]

    def _window_metrics(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(rs) == 0:
            return {'n': 0, 'acc': 0.0, 'qrate': 0.0, 'auroc': 0.0}
        ys = [int(r['true']) for r in rs]
        ss = [float(r['score']) for r in rs]
        q = [1 if r['queried'] else 0 for r in rs]
        c = [1 if r['correct'] else 0 for r in rs]
        return {
            'n': int(len(rs)),
            'acc': float(sum(c) / max(1, len(c))),
            'qrate': float(sum(q) / max(1, len(q))),
            'auroc': float(auc_roc(ss, ys)),
        }

    return {
        'n_records': int(n),
        'first_window': _window_metrics(head),
        'early_window': _window_metrics(early),
        'late_window': _window_metrics(late),
        'later_window_gain_acc': float(_window_metrics(late)['acc'] - _window_metrics(early)['acc']),
        'later_window_gain_auroc': float(_window_metrics(late)['auroc'] - _window_metrics(early)['auroc']),
        'final_cumulative_acc': float(records[-1]['acc_cum']),
        'final_cumulative_qrate': float(records[-1]['qrate_cum']),
    }


def save_temporal_artifacts_for_run(
    *,
    record_path: str,
    out_json: str,
    class_name: str,
    policy: str,
    later_frac: float = 0.5,
    early_n: int = 200,
) -> Dict[str, Any]:
    records = parse_stream_record(record_path)
    prefix = str(Path(out_json).with_suffix(''))
    temporal_json = prefix + '_temporal.json'
    acc_png = prefix + '_temporal_acc.png'
    q_png = prefix + '_temporal_queries.png'
    summary = summarize_stream_records(records, later_frac=later_frac, early_n=early_n)
    summary['record_path'] = record_path
    summary['temporal_json'] = temporal_json
    summary['acc_plot'] = acc_png
    summary['query_plot'] = q_png
    summary['policy'] = str(policy)
    summary['class_name'] = str(class_name)

    safe_makedirs(temporal_json)
    with open(temporal_json, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'records': records}, f, indent=2, ensure_ascii=False)

    if len(records) > 0:
        xs = [r['t'] for r in records]
        acc = [r['acc_cum'] for r in records]
        qcum = np.cumsum([1 if r['queried'] else 0 for r in records]).tolist()

        plt.figure(figsize=(8, 4.5))
        plt.plot(xs, acc, label='cumulative accuracy')
        plt.xlabel('stream step')
        plt.ylabel('accuracy')
        plt.title(f'{class_name} | {policy} | cumulative accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_png, dpi=180)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.plot(xs, qcum, label='cumulative queries')
        plt.xlabel('stream step')
        plt.ylabel('queries')
        plt.title(f'{class_name} | {policy} | cumulative queries')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(q_png, dpi=180)
        plt.close()

    return summary


def save_temporal_comparison_plot(
    *,
    runs: Sequence[Tuple[str, List[Dict[str, Any]]]],
    out_prefix: str,
    class_name: str,
) -> Dict[str, str]:
    acc_png = out_prefix + '_compare_acc.png'
    q_png = out_prefix + '_compare_queries.png'
    plt.figure(figsize=(8, 4.5))
    for label, records in runs:
        if not records:
            continue
        xs = [r['t'] for r in records]
        ys = [r['acc_cum'] for r in records]
        plt.plot(xs, ys, label=label)
    plt.xlabel('stream step')
    plt.ylabel('accuracy')
    plt.title(f'{class_name} | cumulative accuracy comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_png, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for label, records in runs:
        if not records:
            continue
        xs = [r['t'] for r in records]
        qcum = np.cumsum([1 if r['queried'] else 0 for r in records]).tolist()
        plt.plot(xs, qcum, label=label)
    plt.xlabel('stream step')
    plt.ylabel('queries')
    plt.title(f'{class_name} | cumulative queries comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(q_png, dpi=180)
    plt.close()
    return {'acc_plot': acc_png, 'query_plot': q_png}


def _tensor_cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().cpu().clone()


def _fixed_cb_state(cb: FixedCapacityCodebook) -> Dict[str, Any]:
    state = {'protos': _tensor_cpu(cb.protos), 'K': int(cb.K), 'feat_dim': int(cb.feat_dim)}
    # v19 utility-aware STM retention metadata; harmless for fixed/LTM/defect banks.
    for name in ['meta_count', 'meta_utility', 'meta_boundary', 'meta_fp_rescue', 'meta_coverage',
                 'meta_first_step', 'meta_last_step', 'meta_last_event', 'meta_distinct_events']:
        if hasattr(cb, name):
            state[name] = _tensor_cpu(getattr(cb, name))
    if hasattr(cb, 'retention_counters'):
        state['retention_counters'] = dict(cb.retention_counters)
    return state


def _load_fixed_cb_state(cb: FixedCapacityCodebook, state: Dict[str, Any]) -> None:
    protos = state.get('protos', None)
    if protos is None:
        cb.clear()
        return
    cb.protos = protos.to(device=cb.device, dtype=cb.dtype).clone()
    if hasattr(cb, '_empty_meta'):
        cb._empty_meta(int(cb.protos.shape[0]))
        for name in ['meta_count', 'meta_utility', 'meta_boundary', 'meta_fp_rescue', 'meta_coverage',
                     'meta_first_step', 'meta_last_step', 'meta_last_event', 'meta_distinct_events']:
            if name in state and getattr(cb, name).shape[0] == int(cb.protos.shape[0]):
                tensor = state[name]
                target = getattr(cb, name)
                setattr(cb, name, tensor.to(device=cb.device, dtype=target.dtype).clone())
        if 'retention_counters' in state and hasattr(cb, 'retention_counters'):
            cb.retention_counters.update({str(k): int(v) for k, v in dict(state['retention_counters']).items()})


def _signed_cb_state(cb: SignedCorrectionCodebook) -> Dict[str, Any]:
    n = int(cb.n)
    return {
        'n': n,
        'protos': _tensor_cpu(cb.protos[:n]),
        'beta': _tensor_cpu(cb.beta[:n]),
        'radius': _tensor_cpu(cb.radius[:n]),
        'kappa': _tensor_cpu(cb.kappa[:n]),
        'count': _tensor_cpu(cb.count[:n]),
    }


def _load_signed_cb_state(cb: SignedCorrectionCodebook, state: Dict[str, Any]) -> None:
    n = int(state.get('n', 0))
    cb.n = min(n, cb.K)
    cb.protos.zero_()
    cb.beta.zero_()
    cb.radius.fill_(float(cb.cfg.radius0))
    cb.kappa.fill_(float(cb.cfg.kappa0))
    cb.count.zero_()
    if cb.n <= 0:
        return
    cb.protos[:cb.n] = state['protos'][:cb.n].to(cb.device).float()
    cb.beta[:cb.n] = state['beta'][:cb.n].to(cb.device).float()
    cb.radius[:cb.n] = state['radius'][:cb.n].to(cb.device).float()
    cb.kappa[:cb.n] = state['kappa'][:cb.n].to(cb.device).float()
    cb.count[:cb.n] = state['count'][:cb.n].to(cb.device).long()


def _tracker_state(tr: TemporalPrototypeTracker) -> Dict[str, Any]:
    n = int(tr.n)
    return {
        'n': n,
        'protos': _tensor_cpu(tr.protos[:n]),
        'count': _tensor_cpu(tr.count[:n]),
        'distinct_events': _tensor_cpu(tr.distinct_events[:n]),
        'usefulness_sum': _tensor_cpu(tr.usefulness_sum[:n]),
        'boundary_sum': _tensor_cpu(getattr(tr, 'boundary_sum', torch.zeros_like(tr.usefulness_sum))[:n]),
        'fp_rescue_sum': _tensor_cpu(getattr(tr, 'fp_rescue_sum', torch.zeros_like(tr.usefulness_sum))[:n]),
        'coverage_sum': _tensor_cpu(getattr(tr, 'coverage_sum', torch.zeros_like(tr.usefulness_sum))[:n]),
        'last_event': _tensor_cpu(tr.last_event[:n]),
        'first_step': _tensor_cpu(getattr(tr, 'first_step', tr.last_step)[:n]),
        'last_step': _tensor_cpu(tr.last_step[:n]),
        'promoted_count': _tensor_cpu(getattr(tr, 'promoted_count', torch.zeros_like(tr.count))[:n]),
    }


def _load_tracker_state(tr: TemporalPrototypeTracker, state: Dict[str, Any]) -> None:
    n = int(state.get('n', 0))
    tr.n = min(n, tr.K)
    tr.protos.zero_()
    tr.count.zero_()
    tr.distinct_events.zero_()
    tr.usefulness_sum.zero_()
    if hasattr(tr, 'boundary_sum'):
        tr.boundary_sum.zero_()
    if hasattr(tr, 'fp_rescue_sum'):
        tr.fp_rescue_sum.zero_()
    if hasattr(tr, 'coverage_sum'):
        tr.coverage_sum.zero_()
    tr.last_event.fill_(-1)
    if hasattr(tr, 'first_step'):
        tr.first_step.fill_(-1)
    tr.last_step.fill_(-1)
    if hasattr(tr, 'promoted_count'):
        tr.promoted_count.zero_()
    if tr.n <= 0:
        return
    tr.protos[:tr.n] = state['protos'][:tr.n].to(tr.device).float()
    tr.count[:tr.n] = state['count'][:tr.n].to(tr.device).long()
    tr.distinct_events[:tr.n] = state['distinct_events'][:tr.n].to(tr.device).long()
    tr.usefulness_sum[:tr.n] = state['usefulness_sum'][:tr.n].to(tr.device).float()
    if hasattr(tr, 'boundary_sum'):
        tr.boundary_sum[:tr.n] = state.get('boundary_sum', torch.zeros((tr.n,)))[:tr.n].to(tr.device).float()
    if hasattr(tr, 'fp_rescue_sum'):
        tr.fp_rescue_sum[:tr.n] = state.get('fp_rescue_sum', torch.zeros((tr.n,)))[:tr.n].to(tr.device).float()
    if hasattr(tr, 'coverage_sum'):
        tr.coverage_sum[:tr.n] = state.get('coverage_sum', torch.zeros((tr.n,)))[:tr.n].to(tr.device).float()
    tr.last_event[:tr.n] = state['last_event'][:tr.n].to(tr.device).long()
    if hasattr(tr, 'first_step'):
        tr.first_step[:tr.n] = state.get('first_step', state.get('last_step'))[:tr.n].to(tr.device).long()
    tr.last_step[:tr.n] = state['last_step'][:tr.n].to(tr.device).long()
    if hasattr(tr, 'promoted_count'):
        tr.promoted_count[:tr.n] = state.get('promoted_count', torch.zeros((tr.n,), dtype=torch.long))[:tr.n].to(tr.device).long()


def _class_stats_state(stats_obj: ClassStats) -> Dict[str, Any]:
    return {
        'mu_N': float(stats_obj.mu_N), 'm2_N': float(stats_obj.m2_N), 'n_N': int(stats_obj.n_N),
        'mu_A': float(stats_obj.mu_A), 'm2_A': float(stats_obj.m2_A), 'n_A': int(stats_obj.n_A),
        'lambda_z': float(stats_obj.lambda_z), 'theta_dyn': float(stats_obj.theta_dyn),
    }


def _load_class_stats_state(stats_obj: ClassStats, state: Dict[str, Any]) -> None:
    for k in ['mu_N', 'm2_N', 'n_N', 'mu_A', 'm2_A', 'n_A', 'lambda_z', 'theta_dyn']:
        if k in state:
            setattr(stats_obj, k, state[k])


def _rb_state_for_class(rb: ReasoningBank, class_name: str) -> Dict[str, Any]:
    concept = {}
    for layer, cmap in rb.concept_stats.get(class_name, {}).items():
        concept[str(layer)] = {
            str(cid): {'tp': int(st.tp), 'tn': int(st.tn), 'fp': int(st.fp), 'fn': int(st.fn)}
            for cid, st in cmap.items()
        }
    return {
        'class_stats': _class_stats_state(rb.get_or_create_stats(class_name)),
        'concept_stats': concept,
    }


def _load_rb_state_for_class(rb: ReasoningBank, class_name: str, state: Dict[str, Any]) -> ClassStats:
    stats_obj = rb.get_or_create_stats(class_name)
    _load_class_stats_state(stats_obj, state.get('class_stats', {}))
    rb.concept_stats[class_name] = {}
    for layer_s, cmap in state.get('concept_stats', {}).items():
        layer = int(layer_s)
        rb.concept_stats[class_name][layer] = {}
        for cid_s, st in cmap.items():
            rb.concept_stats[class_name][layer][int(cid_s)] = ConceptStats(
                tp=int(st.get('tp', 0)), tn=int(st.get('tn', 0)), fp=int(st.get('fp', 0)), fn=int(st.get('fn', 0))
            )
    return stats_obj


def _serialize_vmb_class_state(vmb: PatchVMB, class_name: str) -> Dict[str, Any]:
    def ser_normal(tree):
        out = {}
        for mode, view_dict in tree.get(class_name, {}).items():
            out[str(mode)] = {view: [_fixed_cb_state(cb) for cb in cbs] for view, cbs in view_dict.items()}
        return out
    def ser_trackers(tree):
        out = {}
        for mode, view_dict in tree.get(class_name, {}).items():
            out[str(mode)] = {view: [_tracker_state(tr) for tr in trs] for view, trs in view_dict.items()}
        return out
    defect = {view: [_fixed_cb_state(cb) for cb in cbs] for view, cbs in vmb.defect_cb.get(class_name, {}).items()}
    corr = {view: [_signed_cb_state(cb) for cb in cbs] for view, cbs in vmb.corr_cb.get(class_name, {}).items()}
    return {
        'normal_dyn': ser_normal(vmb.normal_dyn),
        'normal_dyn_ltm': ser_normal(vmb.normal_dyn_ltm),
        'normal_dyn_candidates': ser_trackers(vmb.normal_dyn_candidates),
        'defect_cb': defect,
        'corr_cb': corr,
        'ltm_pending_updates': int(vmb._ltm_pending_updates.get(class_name, 0)),
        'ltm_last_event': int(vmb._ltm_last_event.get(class_name, -1)),
        'shadow_ltm': bool(vmb.shadow_ltm),
        'enable_ltm_retrieval': bool(vmb.enable_ltm_retrieval),
        'enable_ltm_promotion': bool(vmb.enable_ltm_promotion),
        'ltm_pressure_ratio': float(vmb.ltm_pressure_ratio),
        'ltm_min_pending_updates': int(vmb.ltm_min_pending_updates),
        'ltm_min_usefulness': float(vmb.ltm_min_usefulness),
        'ltm_score_threshold': float(vmb.ltm_score_threshold),
        'enable_maturation_buffer': bool(getattr(vmb, 'enable_maturation_buffer', False)),
        'mature_merge_radius': float(getattr(vmb, 'mature_merge_radius', getattr(vmb, 'ltm_candidate_merge_eps', 0.06))),
        'mature_max_event_protos': int(getattr(vmb, 'mature_max_event_protos', getattr(vmb, 'mature_event_prototypes_per_layer', 8))),
        'mature_max_candidates': int(getattr(vmb, 'mature_max_candidates', getattr(vmb, 'ltm_candidate_capacity', 0))),
        'mature_min_events': int(getattr(vmb, 'mature_min_events', 2)),
        'mature_min_usefulness': float(getattr(vmb, 'mature_min_usefulness', 0.15)),
        'mature_score_threshold': float(getattr(vmb, 'mature_score_threshold', getattr(vmb, 'mature_score_thr', 0.45))),
        'mature_promote_every': int(getattr(vmb, 'mature_promote_every', 32)),
        'mature_promote_max_per_layer': int(getattr(vmb, 'mature_promote_max_per_layer', max(4, int(vmb.K_normal_ltm) // 8))),
        'mature_rescue_min_fp': float(getattr(vmb, 'mature_rescue_min_fp', 0.15)),
        'enable_stm_retirement': bool(getattr(vmb, 'enable_stm_retirement', False)),
        'stm_retire_eps': float(getattr(vmb, 'stm_retire_eps', 0.025)),
        'stm_retire_ltm_eps': float(getattr(vmb, 'stm_retire_ltm_eps', getattr(vmb, 'stm_retire_eps', 0.025))),
        'ltm_retrieval_k': int(getattr(vmb, 'ltm_retrieval_k', 1)),
        'enable_maturation_buffer_retrieval': bool(getattr(vmb, 'enable_maturation_buffer_retrieval', False)),
        'enable_utility_aware_stm_retention': bool(getattr(vmb, 'enable_utility_aware_stm_retention', False)),
        'stm_retention_local_k': int(getattr(vmb, 'stm_retention_local_k', 32)),
        'stm_retention_recency_tau': float(getattr(vmb, 'stm_retention_recency_tau', 1024.0)),
        'stm_retention_ltm_cover_eps': float(getattr(vmb, 'stm_retention_ltm_cover_eps', 0.025)),
        'stm_retention_replace_margin': float(getattr(vmb, 'stm_retention_replace_margin', 0.0)),
        'stm_retention_ema': float(getattr(vmb, 'stm_retention_ema', 0.10)),
        'mature_retrieval_min_events': int(getattr(vmb, 'mature_retrieval_min_events', 2)),
        'mature_retrieval_min_usefulness': float(getattr(vmb, 'mature_retrieval_min_usefulness', 0.10)),
        'mature_retrieval_score_thr': float(getattr(vmb, 'mature_retrieval_score_thr', 0.25)),
        'mature_retrieval_penalty': float(getattr(vmb, 'mature_retrieval_penalty', 0.01)),
        'mature_retrieval_max_protos': int(getattr(vmb, 'mature_retrieval_max_protos', 256)),
        'mature_retrieval_k': int(getattr(vmb, 'mature_retrieval_k', 1)),
    }


def _load_vmb_class_state(vmb: PatchVMB, class_name: str, state: Dict[str, Any]) -> None:
    for mode_s, view_dict in state.get('normal_dyn', {}).items():
        mode = int(mode_s)
        for view, layers in view_dict.items():
            for i, cb_state in enumerate(layers):
                _load_fixed_cb_state(vmb.normal_dyn[class_name][mode][view][i], cb_state)
    for mode_s, view_dict in state.get('normal_dyn_ltm', {}).items():
        mode = int(mode_s)
        for view, layers in view_dict.items():
            for i, cb_state in enumerate(layers):
                _load_fixed_cb_state(vmb.normal_dyn_ltm[class_name][mode][view][i], cb_state)
    for mode_s, view_dict in state.get('normal_dyn_candidates', {}).items():
        mode = int(mode_s)
        for view, layers in view_dict.items():
            for i, tr_state in enumerate(layers):
                _load_tracker_state(vmb.normal_dyn_candidates[class_name][mode][view][i], tr_state)
    for view, layers in state.get('defect_cb', {}).items():
        for i, cb_state in enumerate(layers):
            _load_fixed_cb_state(vmb.defect_cb[class_name][view][i], cb_state)
    for view, layers in state.get('corr_cb', {}).items():
        for i, cb_state in enumerate(layers):
            _load_signed_cb_state(vmb.corr_cb[class_name][view][i], cb_state)
    vmb._ltm_pending_updates[class_name] = int(state.get('ltm_pending_updates', 0))
    vmb._ltm_last_event[class_name] = int(state.get('ltm_last_event', -1))
    vmb.shadow_ltm = bool(state.get('shadow_ltm', vmb.shadow_ltm))
    vmb.enable_ltm_retrieval = bool(state.get('enable_ltm_retrieval', vmb.enable_ltm_retrieval))
    vmb.enable_ltm_promotion = bool(state.get('enable_ltm_promotion', vmb.enable_ltm_promotion))
    vmb.ltm_pressure_ratio = float(state.get('ltm_pressure_ratio', vmb.ltm_pressure_ratio))
    vmb.ltm_min_pending_updates = int(state.get('ltm_min_pending_updates', vmb.ltm_min_pending_updates))
    vmb.ltm_min_usefulness = float(state.get('ltm_min_usefulness', vmb.ltm_min_usefulness))
    vmb.ltm_score_threshold = float(state.get('ltm_score_threshold', vmb.ltm_score_threshold))
    vmb.enable_maturation_buffer = bool(state.get('enable_maturation_buffer', getattr(vmb, 'enable_maturation_buffer', False)))
    vmb.mature_min_events = int(state.get('mature_min_events', getattr(vmb, 'mature_min_events', 2)))
    vmb.mature_min_usefulness = float(state.get('mature_min_usefulness', getattr(vmb, 'mature_min_usefulness', 0.15)))
    vmb.mature_score_threshold = float(state.get('mature_score_threshold', getattr(vmb, 'mature_score_threshold', 0.45)))
    vmb.mature_score_thr = float(vmb.mature_score_threshold)
    vmb.mature_max_event_protos = int(state.get('mature_max_event_protos', getattr(vmb, 'mature_max_event_protos', getattr(vmb, 'mature_event_prototypes_per_layer', 8))))
    vmb.mature_event_prototypes_per_layer = int(vmb.mature_max_event_protos)
    vmb.mature_max_candidates = int(state.get('mature_max_candidates', getattr(vmb, 'mature_max_candidates', getattr(vmb, 'ltm_candidate_capacity', 0))))
    vmb.mature_merge_radius = float(state.get('mature_merge_radius', getattr(vmb, 'mature_merge_radius', getattr(vmb, 'ltm_candidate_merge_eps', 0.06))))
    vmb.mature_promote_every = int(state.get('mature_promote_every', getattr(vmb, 'mature_promote_every', 32)))
    vmb.mature_promote_max_per_layer = int(state.get('mature_promote_max_per_layer', getattr(vmb, 'mature_promote_max_per_layer', max(4, int(vmb.K_normal_ltm) // 8))))
    vmb.mature_rescue_min_fp = float(state.get('mature_rescue_min_fp', getattr(vmb, 'mature_rescue_min_fp', 0.15)))
    vmb.enable_stm_retirement = bool(state.get('enable_stm_retirement', getattr(vmb, 'enable_stm_retirement', False)))
    vmb.stm_retire_eps = float(state.get('stm_retire_eps', getattr(vmb, 'stm_retire_eps', 0.025)))
    vmb.stm_retire_ltm_eps = float(state.get('stm_retire_ltm_eps', getattr(vmb, 'stm_retire_ltm_eps', vmb.stm_retire_eps)))
    vmb.ltm_retrieval_k = int(state.get('ltm_retrieval_k', getattr(vmb, 'ltm_retrieval_k', 1)))
    vmb.enable_maturation_buffer_retrieval = bool(state.get('enable_maturation_buffer_retrieval', getattr(vmb, 'enable_maturation_buffer_retrieval', False)))
    vmb.enable_utility_aware_stm_retention = bool(state.get('enable_utility_aware_stm_retention', getattr(vmb, 'enable_utility_aware_stm_retention', False)))
    vmb.stm_retention_local_k = int(state.get('stm_retention_local_k', getattr(vmb, 'stm_retention_local_k', 32)))
    vmb.stm_retention_recency_tau = float(state.get('stm_retention_recency_tau', getattr(vmb, 'stm_retention_recency_tau', 1024.0)))
    vmb.stm_retention_ltm_cover_eps = float(state.get('stm_retention_ltm_cover_eps', getattr(vmb, 'stm_retention_ltm_cover_eps', 0.025)))
    vmb.stm_retention_replace_margin = float(state.get('stm_retention_replace_margin', getattr(vmb, 'stm_retention_replace_margin', 0.0)))
    vmb.stm_retention_ema = float(state.get('stm_retention_ema', getattr(vmb, 'stm_retention_ema', 0.10)))
    vmb.mature_retrieval_min_events = int(state.get('mature_retrieval_min_events', getattr(vmb, 'mature_retrieval_min_events', 2)))
    vmb.mature_retrieval_min_usefulness = float(state.get('mature_retrieval_min_usefulness', getattr(vmb, 'mature_retrieval_min_usefulness', 0.10)))
    vmb.mature_retrieval_score_thr = float(state.get('mature_retrieval_score_thr', getattr(vmb, 'mature_retrieval_score_thr', 0.25)))
    vmb.mature_retrieval_penalty = float(state.get('mature_retrieval_penalty', getattr(vmb, 'mature_retrieval_penalty', 0.01)))
    vmb.mature_retrieval_max_protos = int(state.get('mature_retrieval_max_protos', getattr(vmb, 'mature_retrieval_max_protos', 256)))
    vmb.mature_retrieval_k = int(state.get('mature_retrieval_k', getattr(vmb, 'mature_retrieval_k', 1)))


def _coverage_state(mem: NormalCoverageMemory) -> Dict[str, Any]:
    return {
        'dynamic_feats': _tensor_cpu(mem.dynamic_feats),
        'dynamic_ref_dists': _tensor_cpu(mem.dynamic_ref_dists),
        'max_dynamic': int(mem.max_dynamic),
    }


def _load_coverage_state(mem: NormalCoverageMemory, state: Dict[str, Any]) -> None:
    mem.max_dynamic = int(state.get('max_dynamic', mem.max_dynamic))
    dyn = state.get('dynamic_feats', None)
    dyn_ref = state.get('dynamic_ref_dists', None)
    if dyn is not None:
        mem.dynamic_feats = dyn.float().cpu().clone()
    if dyn_ref is not None:
        mem.dynamic_ref_dists = dyn_ref.float().cpu().clone()


def _threshold_state(th: Any) -> Dict[str, Any]:
    if isinstance(th, GuardedDualAnchorThresholdController):
        return {
            'kind': 'guarded_dual_anchor',
            'bufN': list(th.bufN), 'bufA': list(th.bufA), 'recent_labeled': list(th.recent_labeled),
            'muN0': float(th.muN0), 'sigmaN0': float(th.sigmaN0), 'theta0': float(th.theta0), 'theta_current': float(th.theta_current),
            'n_normals_total': int(th.n_normals_total), 'n_normals_support': int(th.n_normals_support),
            'n_normals_labeled': int(th.n_normals_labeled), 'n_normals_unlabeled': int(th.n_normals_unlabeled),
            'n_defects_labeled': int(th.n_defects_labeled), 'labeled_since_update': int(th.labeled_since_update),
            'last_move': float(th.last_move), 'last_theta_star': float(th.last_theta_star),
            'last_accept': bool(th.last_accept), 'last_eval': copy.deepcopy(th.last_eval), 'last_mode': str(th.last_mode),
        }
    return {
        'kind': 'legacy_normal_only',
        'bufN': list(th.bufN), 'muN0': float(th.muN0), 'sigmaN0': float(th.sigmaN0), 'theta0': float(th.theta0),
        'n_normals_total': int(th.n_normals_total), 'n_normals_support': int(th.n_normals_support),
        'n_normals_labeled': int(th.n_normals_labeled), 'n_normals_unlabeled': int(th.n_normals_unlabeled),
    }


def _load_threshold_state(th: Any, state: Dict[str, Any]) -> None:
    if isinstance(th, GuardedDualAnchorThresholdController):
        th.bufN = [float(x) for x in state.get('bufN', [])]
        th.bufA = [float(x) for x in state.get('bufA', [])]
        th.recent_labeled = [(float(s), int(y)) for s, y in state.get('recent_labeled', [])]
        th.muN0 = float(state.get('muN0', th.muN0))
        th.sigmaN0 = float(state.get('sigmaN0', th.sigmaN0))
        th.theta0 = float(state.get('theta0', th.theta0))
        th.theta_current = float(state.get('theta_current', th.theta_current))
        th.n_normals_total = int(state.get('n_normals_total', th.n_normals_total))
        th.n_normals_support = int(state.get('n_normals_support', th.n_normals_support))
        th.n_normals_labeled = int(state.get('n_normals_labeled', th.n_normals_labeled))
        th.n_normals_unlabeled = int(state.get('n_normals_unlabeled', th.n_normals_unlabeled))
        th.n_defects_labeled = int(state.get('n_defects_labeled', th.n_defects_labeled))
        th.labeled_since_update = int(state.get('labeled_since_update', th.labeled_since_update))
        th.last_move = float(state.get('last_move', th.last_move))
        th.last_theta_star = float(state.get('last_theta_star', th.last_theta_star))
        th.last_accept = bool(state.get('last_accept', th.last_accept))
        th.last_eval = copy.deepcopy(state.get('last_eval', th.last_eval))
        th.last_mode = str(state.get('last_mode', th.last_mode))
    else:
        th.bufN = [float(x) for x in state.get('bufN', [])]
        th.muN0 = float(state.get('muN0', th.muN0))
        th.sigmaN0 = float(state.get('sigmaN0', th.sigmaN0))
        th.theta0 = float(state.get('theta0', th.theta0))
        th.n_normals_total = int(state.get('n_normals_total', th.n_normals_total))
        th.n_normals_support = int(state.get('n_normals_support', th.n_normals_support))
        th.n_normals_labeled = int(state.get('n_normals_labeled', th.n_normals_labeled))
        th.n_normals_unlabeled = int(state.get('n_normals_unlabeled', th.n_normals_unlabeled))


def _simple_aif_state_dict(st: SimpleAIFState) -> Dict[str, Any]:
    # v20 restores the v16 SimpleAIFState: no stream-prior tracker is serialized.
    return {
        'query_cost_runtime': float(st.query_cost_runtime),
        'steps': int(st.steps),
        'queries': int(st.queries),
        'recent_query_flags': list(st.recent_query_flags),
    }


def _load_simple_aif_state(st: SimpleAIFState, state: Dict[str, Any]) -> None:
    st.query_cost_runtime = float(state.get('query_cost_runtime', st.query_cost_runtime))
    st.steps = int(state.get('steps', st.steps))
    st.queries = int(state.get('queries', st.queries))
    st.recent_query_flags = [int(x) for x in state.get('recent_query_flags', st.recent_query_flags)]


def _query_state_dict(qs: QueryState) -> Dict[str, Any]:
    return {
        'used': int(qs.used),
        'recent_scores': list(qs.recent_scores),
        'calib_used': int(qs.calib_used),
        'baseline_steps': int(getattr(qs, 'baseline_steps', 0)),
        'baseline_used': int(getattr(qs, 'baseline_used', 0)),
        'recent_policy_scores': list(getattr(qs, 'recent_policy_scores', [])),
        'recent_policy_queries': list(getattr(qs, 'recent_policy_queries', [])),
    }


def _load_query_state(qs: QueryState, state: Dict[str, Any]) -> None:
    qs.used = int(state.get('used', qs.used))
    qs.recent_scores = [float(x) for x in state.get('recent_scores', qs.recent_scores)]
    qs.calib_used = int(state.get('calib_used', qs.calib_used))
    qs.baseline_steps = int(state.get('baseline_steps', getattr(qs, 'baseline_steps', 0)))
    qs.baseline_used = int(state.get('baseline_used', getattr(qs, 'baseline_used', 0)))
    qs.recent_policy_scores = [float(x) for x in state.get('recent_policy_scores', getattr(qs, 'recent_policy_scores', []))]
    qs.recent_policy_queries = [int(x) for x in state.get('recent_policy_queries', getattr(qs, 'recent_policy_queries', []))]


def build_resume_state(
    *,
    class_name: str,
    vmb: PatchVMB,
    rb: ReasoningBank,
    stats_obj: ClassStats,
    th: Any,
    coverage_mem: NormalCoverageMemory,
    simple_aif_state: SimpleAIFState,
    qs: QueryState,
    global_steps: int,
    policy: str,
    ablation: str,
) -> Dict[str, Any]:
    return {
        'meta': {
            'class_name': str(class_name),
            'global_steps': int(global_steps),
            'policy': str(policy),
            'ablation': str(ablation),
        },
        'vmb': _serialize_vmb_class_state(vmb, class_name),
        'rb': _rb_state_for_class(rb, class_name),
        'stats': _class_stats_state(stats_obj),
        'threshold': _threshold_state(th),
        'coverage': _coverage_state(coverage_mem),
        'simple_aif_state': _simple_aif_state_dict(simple_aif_state),
        'query_state': _query_state_dict(qs),
    }


def apply_resume_state(
    *,
    resume_state: Dict[str, Any],
    class_name: str,
    vmb: PatchVMB,
    rb: ReasoningBank,
    stats_obj: ClassStats,
    th: Any,
    coverage_mem: NormalCoverageMemory,
    simple_aif_state: SimpleAIFState,
    qs: QueryState,
) -> int:
    meta = resume_state.get('meta', {})
    if str(meta.get('class_name', class_name)) != str(class_name):
        raise ValueError(f"Resume-state class mismatch: expected {class_name}, got {meta.get('class_name')}")
    _load_vmb_class_state(vmb, class_name, resume_state.get('vmb', {}))
    _load_rb_state_for_class(rb, class_name, resume_state.get('rb', {}))
    _load_class_stats_state(stats_obj, resume_state.get('stats', {}))
    _load_threshold_state(th, resume_state.get('threshold', {}))
    _load_coverage_state(coverage_mem, resume_state.get('coverage', {}))
    _load_simple_aif_state(simple_aif_state, resume_state.get('simple_aif_state', {}))
    _load_query_state(qs, resume_state.get('query_state', {}))
    return int(meta.get('global_steps', 0))


def run_conventional_panel_experiment(
    *,
    classes: Sequence[str],
    methods: Sequence[str],
    out_json: str,
    base_run_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    prefix = str(Path(out_json).with_suffix(''))
    panel = {'classes': list(classes), 'methods': list(methods), 'per_class': {}, 'aggregate': {}}

    # Unified panel aliases.
    # Use hardcode_* for module attribution under the same ambiguity-band policy.
    # Use aif_* / simple_aif for controller comparison under the same full v21 module stack.
    method_overrides = {
        # Full-stack controller comparison
        'simple_aif': {'policy': 'simple_aif', 'ablation': 'normal_dyn_plus_correction'},
        'aif_full': {'policy': 'simple_aif', 'ablation': 'normal_dyn_plus_correction'},
        'hardcode_band': {'policy': 'hardcode_band', 'ablation': 'normal_dyn_plus_correction'},
        'hardcode_full': {'policy': 'hardcode_band', 'ablation': 'normal_dyn_plus_correction'},
        'entropy': {'policy': 'entropy', 'ablation': 'normal_dyn_plus_correction'},
        'entropy_full': {'policy': 'entropy', 'ablation': 'normal_dyn_plus_correction'},
        'margin': {'policy': 'margin', 'ablation': 'normal_dyn_plus_correction'},
        'margin_full': {'policy': 'margin', 'ablation': 'normal_dyn_plus_correction'},
        'novelty': {'policy': 'novelty', 'ablation': 'normal_dyn_plus_correction'},
        'novelty_full': {'policy': 'novelty', 'ablation': 'normal_dyn_plus_correction'},
        'random': {'policy': 'random', 'ablation': 'normal_dyn_plus_correction'},
        'random_full': {'policy': 'random', 'ablation': 'normal_dyn_plus_correction'},
        'periodic': {'policy': 'periodic', 'ablation': 'normal_dyn_plus_correction'},
        'periodic_full': {'policy': 'periodic', 'ablation': 'normal_dyn_plus_correction'},

        # AIF-side ablations, kept for backward compatibility
        'no_correction': {'policy': 'simple_aif', 'ablation': 'normal_dyn_only'},
        'no_dynamic': {'policy': 'simple_aif', 'ablation': 'correction_only'},
        'static_baseline': {'policy': 'simple_aif', 'ablation': 'static_baseline'},

        # Hardcode-band module attribution aliases
        'hardcode_static': {'policy': 'hardcode_band', 'ablation': 'static_baseline'},
        'hardcode_fixed': {'policy': 'hardcode_band', 'ablation': 'static_baseline'},
        'hardcode_dyn': {'policy': 'hardcode_band', 'ablation': 'normal_dyn_only'},
        'hardcode_dynamic': {'policy': 'hardcode_band', 'ablation': 'normal_dyn_only'},
        # Dynamic + correction without the guarded-dual-anchor sidecar; uses legacy normal-only theta.
        'hardcode_dyn_corr': {'policy': 'hardcode_band', 'ablation': 'normal_dyn_plus_correction', 'theta_controller_kind': 'legacy_normal_only'},
        'hardcode_dynamic_correction': {'policy': 'hardcode_band', 'ablation': 'normal_dyn_plus_correction', 'theta_controller_kind': 'legacy_normal_only'},
        # Full hardcode stack with guarded sidecar.
        'hardcode_dyn_corr_sidecar': {'policy': 'hardcode_band', 'ablation': 'normal_dyn_plus_correction', 'theta_controller_kind': 'guarded_dual_anchor'},
    }

    for cls_name in classes:
        panel['per_class'][cls_name] = {}
        compare_runs = []
        for method in methods:
            if method not in method_overrides:
                raise ValueError(f'Unknown panel method: {method}')
            run_kwargs = dict(base_run_kwargs)
            run_kwargs.update(method_overrides[method])
            run_kwargs['class_name'] = cls_name
            run_kwargs['return_state'] = False
            run_kwargs['save_temporal_artifacts_flag'] = True
            run_kwargs['out_json'] = f'{prefix}__{cls_name}__{method}.json'
            summ = run_phase3_streaming_aif(**run_kwargs)
            panel['per_class'][cls_name][method] = {
                'metrics': summ['metrics'],
                'temporal': summ.get('temporal', {}),
                'summary_path': summ.get('summary_path', None),
                'record_path': summ.get('record_path', None),
            }
            recs = parse_stream_record(summ.get('record_path', ''))
            if recs:
                compare_runs.append((method, recs))
            del summ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if compare_runs:
            cmp_prefix = f'{prefix}__{cls_name}'
            panel['per_class'][cls_name]['comparison_plots'] = save_temporal_comparison_plot(
                runs=compare_runs, out_prefix=cmp_prefix, class_name=cls_name
            )

    for method in methods:
        accs, qrates, aurocs = [], [], []
        for cls_name in classes:
            if method not in panel['per_class'][cls_name]:
                continue
            m = panel['per_class'][cls_name][method]['metrics']
            accs.append(float(m.get('acc', 0.0)))
            qrates.append(float(m.get('qrate', 0.0)))
            aurocs.append(float(m.get('AUROC_score', m.get('auroc', 0.0))))
        if accs:
            panel['aggregate'][method] = {
                'mean_acc': float(np.mean(accs)),
                'mean_qrate': float(np.mean(qrates)),
                'mean_auroc': float(np.mean(aurocs)),
                'acc_per_qrate': float(np.mean(accs) / max(1e-8, np.mean(qrates))),
                'auroc_per_qrate': float(np.mean(aurocs) / max(1e-8, np.mean(qrates))),
            }

    panel_path = prefix + '_panel_summary.json'
    safe_makedirs(panel_path)
    with open(panel_path, 'w', encoding='utf-8') as f:
        json.dump(panel, f, indent=2, ensure_ascii=False)
    panel['panel_summary_path'] = panel_path
    return panel


def run_ab_revisit_experiment(
    *,
    class_a: str,
    class_b: str,
    a1_frac: float,
    out_json: str,
    base_run_kwargs: Dict[str, Any],
    compare_ltm: bool = False,
    early_n: int = 200,
) -> Dict[str, Any]:
    prefix = str(Path(out_json).with_suffix(''))
    common = dict(base_run_kwargs)
    common['policy'] = 'simple_aif'
    common['ablation'] = 'normal_dyn_plus_correction'
    common['save_temporal_artifacts_flag'] = True

    a1_kwargs = dict(common)
    a1_kwargs.update({'class_name': class_a, 'out_json': prefix + '__A1.json', 'stream_start_frac': 0.0, 'stream_end_frac': float(a1_frac), 'return_state': True})
    a1 = run_phase3_streaming_aif(**a1_kwargs)
    state_a = a1.pop('_resume_state', None)

    b_kwargs = dict(common)
    b_kwargs.update({'class_name': class_b, 'out_json': prefix + '__B.json', 'stream_start_frac': 0.0, 'stream_end_frac': 1.0, 'return_state': False})
    b_run = run_phase3_streaming_aif(**b_kwargs)

    a2_reset_kwargs = dict(common)
    a2_reset_kwargs.update({'class_name': class_a, 'out_json': prefix + '__A2_reset.json', 'stream_start_frac': float(a1_frac), 'stream_end_frac': 1.0, 'return_state': False})
    a2_reset = run_phase3_streaming_aif(**a2_reset_kwargs)

    a2_resume_kwargs = dict(common)
    a2_resume_kwargs.update({'class_name': class_a, 'out_json': prefix + '__A2_resume.json', 'stream_start_frac': float(a1_frac), 'stream_end_frac': 1.0, 'resume_state': state_a, 'return_state': False})
    a2_resume = run_phase3_streaming_aif(**a2_resume_kwargs)

    compare_runs = [
        ('reset', parse_stream_record(a2_reset.get('record_path', ''))),
        ('resume', parse_stream_record(a2_resume.get('record_path', ''))),
    ]

    ltm_block = None
    if compare_ltm:
        a1_ltm_kwargs = dict(common)
        a1_ltm_kwargs.update({'class_name': class_a, 'out_json': prefix + '__A1_ltm.json', 'stream_start_frac': 0.0, 'stream_end_frac': float(a1_frac), 'return_state': True, 'enable_ltm_lite': True})
        a1_ltm = run_phase3_streaming_aif(**a1_ltm_kwargs)
        state_a_ltm = a1_ltm.pop('_resume_state', None)
        a2_resume_ltm_kwargs = dict(common)
        a2_resume_ltm_kwargs.update({'class_name': class_a, 'out_json': prefix + '__A2_resume_ltm.json', 'stream_start_frac': float(a1_frac), 'stream_end_frac': 1.0, 'resume_state': state_a_ltm, 'return_state': False, 'enable_ltm_lite': True})
        a2_resume_ltm = run_phase3_streaming_aif(**a2_resume_ltm_kwargs)
        compare_runs.append(('resume_ltm', parse_stream_record(a2_resume_ltm.get('record_path', ''))))
        ltm_block = {'A1_ltm': a1_ltm, 'A2_resume_ltm': a2_resume_ltm}

    cmp_paths = save_temporal_comparison_plot(runs=compare_runs, out_prefix=prefix + '__A2', class_name=class_a)

    revisit = {
        'class_a': class_a,
        'class_b': class_b,
        'a1_frac': float(a1_frac),
        'A1': {'metrics': a1['metrics'], 'temporal': a1.get('temporal', {})},
        'B': {'metrics': b_run['metrics'], 'temporal': b_run.get('temporal', {})},
        'A2_reset': {'metrics': a2_reset['metrics'], 'temporal': a2_reset.get('temporal', {})},
        'A2_resume': {'metrics': a2_resume['metrics'], 'temporal': a2_resume.get('temporal', {})},
        'comparison_plots': cmp_paths,
        'resume_minus_reset': {
            'full_acc': float(a2_resume['metrics']['acc'] - a2_reset['metrics']['acc']),
            'full_qrate': float(a2_resume['metrics']['qrate'] - a2_reset['metrics']['qrate']),
            'full_auroc': float(a2_resume['metrics']['AUROC_score'] - a2_reset['metrics']['AUROC_score']),
            'first_window_acc': float(a2_resume.get('temporal', {}).get('first_window', {}).get('acc', 0.0) - a2_reset.get('temporal', {}).get('first_window', {}).get('acc', 0.0)),
            'first_window_qrate': float(a2_resume.get('temporal', {}).get('first_window', {}).get('qrate', 0.0) - a2_reset.get('temporal', {}).get('first_window', {}).get('qrate', 0.0)),
        },
    }
    if ltm_block is not None:
        revisit['A1_ltm'] = {'metrics': ltm_block['A1_ltm']['metrics'], 'temporal': ltm_block['A1_ltm'].get('temporal', {})}
        revisit['A2_resume_ltm'] = {'metrics': ltm_block['A2_resume_ltm']['metrics'], 'temporal': ltm_block['A2_resume_ltm'].get('temporal', {})}
        revisit['resume_ltm_minus_reset'] = {
            'full_acc': float(ltm_block['A2_resume_ltm']['metrics']['acc'] - a2_reset['metrics']['acc']),
            'full_qrate': float(ltm_block['A2_resume_ltm']['metrics']['qrate'] - a2_reset['metrics']['qrate']),
            'full_auroc': float(ltm_block['A2_resume_ltm']['metrics']['AUROC_score'] - a2_reset['metrics']['AUROC_score']),
            'first_window_acc': float(ltm_block['A2_resume_ltm'].get('temporal', {}).get('first_window', {}).get('acc', 0.0) - a2_reset.get('temporal', {}).get('first_window', {}).get('acc', 0.0)),
            'first_window_qrate': float(ltm_block['A2_resume_ltm'].get('temporal', {}).get('first_window', {}).get('qrate', 0.0) - a2_reset.get('temporal', {}).get('first_window', {}).get('qrate', 0.0)),
        }

    revisit_path = prefix + '_revisit_summary.json'
    safe_makedirs(revisit_path)
    with open(revisit_path, 'w', encoding='utf-8') as f:
        json.dump(revisit, f, indent=2, ensure_ascii=False)
    revisit['revisit_summary_path'] = revisit_path
    return revisit


@torch.no_grad()
def compute_normal_train_calibration_scores(
    *,
    class_name: str,
    train_good_paths: List[str],
    backbone: Any,
    vmb: Any,
    centroids_cpu: torch.Tensor,
    n_modes: int,
    use_modes: bool,
    feature_fuse: str,
    views_by_name: Dict[str, ViewSpec],
    topk_patches: int,
    score_mode: str,
    lam: float,
    tau_close: float,
    gamma: float,
    layer_fusion: str,
    view_fusion: str,
    img_agg: str,
    img_topk: int,
    img_quantile: float,
    empty_cache_every: int = 50,
) -> List[float]:
    """Score all normal training images for threshold calibration only.

    This function does not write into fixed/dynamic/correction memories.  It
    uses the already-built few-shot fixed VMB to obtain scalar anomaly scores
    for normal training images, then the caller can set theta0 from the
    (1-target_fpr) quantile of these scores.
    """
    scores: List[float] = []
    for i, p in enumerate(sorted(list(train_good_paths)), start=1):
        img = load_rgb(p)
        feats_id, g = encode_pil(backbone, img, feature_fuse)
        feats_id = [l2_normalize(x) for x in feats_id]
        mode = choose_mode(g, centroids_cpu) if use_modes and centroids_cpu is not None and centroids_cpu.numel() else 0
        mode = int(max(0, min(int(mode), int(max(1, n_modes)) - 1)))

        view_to_feats: Dict[str, List[torch.Tensor]] = {}
        for vname, vspec in views_by_name.items():
            img_v = apply_view_pil(img, vspec)
            feats_v, _ = encode_pil(backbone, img_v, feature_fuse)
            view_to_feats[vname] = [l2_normalize(x) for x in feats_v]

        score_img, _, _ = vmb.score_sample_multiview(
            cls_name=class_name,
            view_to_feats=view_to_feats,
            mode=mode,
            views=views_by_name,
            topk_patches=topk_patches,
            mode_scoring=score_mode,
            lam=lam,
            tau_close=tau_close,
            gamma=gamma,
            layer_fusion=layer_fusion,
            view_fusion=view_fusion,
            img_agg=img_agg,
            img_topk=img_topk,
            img_quantile=img_quantile,
            concept_assign_view="id",
            img_hw=(img.height, img.width),
        )
        scores.append(float(score_img))
        if torch.cuda.is_available() and empty_cache_every > 0 and (i % int(empty_cache_every) == 0):
            torch.cuda.empty_cache()
    return scores


# ---------------------------------------------------------------------------
#  v22++ coverage-aware support selection
# ---------------------------------------------------------------------------

def _coverage_tail_make_run_cache_dir(
    cache_dir: Optional[str],
    *,
    class_name: str,
    seed: int,
    support_seed: Optional[int],
    stream_seed: Optional[int],
) -> Optional[str]:
    """Create and clear a per-run cache/diagnostic folder.

    The root cache directory is not used as state.  Each class/seed run gets a
    deterministic subfolder which is deleted before use, preventing stale
    candidate diagnostics from contaminating future runs while preserving the
    user-supplied cache root for batch experiments.
    """
    cache_dir = None if cache_dir is None else str(cache_dir).strip()
    if not cache_dir:
        return None
    import shutil
    def _tok(x: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))
    run_name = f"{_tok(class_name)}__seed{int(seed)}__sseed{_tok('none' if support_seed is None else support_seed)}__stream{_tok('none' if stream_seed is None else stream_seed)}"
    run_dir = os.path.join(cache_dir, run_name)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


@torch.no_grad()
def _coverage_tail_global_features(
    backbone: Any,
    train_good_paths: List[str],
    *,
    feature_fuse: str,
    empty_cache_every: int = 50,
) -> torch.Tensor:
    """Return L2-normalized global image features for normal train paths."""
    feats: List[torch.Tensor] = []
    for i, pth in enumerate(sorted(list(train_good_paths)), start=1):
        try:
            # encode_path is already used by build_modes_from_train_good and is
            # the lightest available path-level DINO feature call in this script.
            _, g = backbone.encode_path(pth)
            feats.append(l2_normalize(g.detach().float().cpu().view(-1)))
        except Exception:
            img = load_rgb(pth)
            _, g = encode_pil(backbone, img, feature_fuse)
            feats.append(l2_normalize(g.detach().float().cpu().view(-1)))
        if torch.cuda.is_available() and empty_cache_every > 0 and (i % int(empty_cache_every) == 0):
            torch.cuda.empty_cache()
    if len(feats) == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.stack(feats, dim=0).float()


def _coverage_tail_mode_indices(labels: Optional[List[int]], n_paths: int, n_modes: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {int(k): [] for k in range(int(max(1, n_modes)))}
    if labels is None or len(labels) != int(n_paths):
        out[0] = list(range(int(n_paths)))
        return out
    for i, lab in enumerate(labels):
        k = int(max(0, min(int(lab), int(max(1, n_modes)) - 1)))
        out.setdefault(k, []).append(int(i))
    return out


def _coverage_tail_greedy_kcenter_indices(
    X: torch.Tensor,
    candidate_indices: List[int],
    *,
    k: int,
    centroid: torch.Tensor,
    rng: random.Random,
    inlier_q: float,
    outlier_lambda: float,
    random_start: bool = False,
    random_jitter: float = 0.0,
) -> List[int]:
    """Density-filtered, outlier-penalized greedy k-center in global feature space."""
    idxs = [int(i) for i in candidate_indices]
    k = int(max(1, k))
    if len(idxs) <= k:
        return sorted(idxs)
    X_cpu = l2_normalize(X.float().cpu())
    c = l2_normalize(centroid.float().cpu().view(-1))
    Xi = X_cpu[idxs]
    d_cent = 1.0 - torch.clamp(Xi @ c.view(-1, 1), -1.0, 1.0).view(-1)
    q = float(np.clip(float(inlier_q), 0.50, 1.00))
    thr = float(torch.quantile(d_cent, q).item()) if d_cent.numel() > 1 else float(d_cent.max().item())
    keep_local = torch.nonzero(d_cent <= thr, as_tuple=False).view(-1).tolist()
    if len(keep_local) < k:
        keep_local = list(range(len(idxs)))
    pool = [idxs[int(j)] for j in keep_local]
    d_cent_pool = d_cent[torch.tensor(keep_local, dtype=torch.long)]

    if random_start:
        # Prefer central starts, but randomize among the central half so repeated
        # support seeds still explore different coverage candidates.
        order = torch.argsort(d_cent_pool).tolist()
        top = order[: max(1, min(len(order), max(k, len(order) // 2)))]
        first = pool[int(rng.choice(top))]
    else:
        first = pool[int(torch.argmin(d_cent_pool).item())]
    selected = [int(first)]

    # Maintain min distance to selected set.  Cosine distance because X is normalized.
    Xp = X_cpu[pool]
    x_first = X_cpu[first:first+1]
    min_d = (1.0 - torch.clamp(Xp @ x_first.T, -1.0, 1.0).view(-1)).clone()
    selected_set = {int(first)}
    while len(selected) < k and len(selected) < len(pool):
        score = min_d - float(outlier_lambda) * d_cent_pool
        if random_jitter > 0.0:
            noise = torch.tensor([rng.random() for _ in range(score.numel())], dtype=score.dtype) * float(random_jitter)
            score = score + noise
        for j, orig_idx in enumerate(pool):
            if int(orig_idx) in selected_set:
                score[j] = -1e9
        j_best = int(torch.argmax(score).item())
        new_idx = int(pool[j_best])
        selected.append(new_idx)
        selected_set.add(new_idx)
        x_new = X_cpu[new_idx:new_idx+1]
        d_new = 1.0 - torch.clamp(Xp @ x_new.T, -1.0, 1.0).view(-1)
        min_d = torch.minimum(min_d, d_new)
    return sorted(selected)


def _coverage_tail_closest_indices(
    X: torch.Tensor,
    candidate_indices: List[int],
    *,
    k: int,
    centroid: torch.Tensor,
) -> List[int]:
    idxs = [int(i) for i in candidate_indices]
    if len(idxs) <= int(k):
        return sorted(idxs)
    X_cpu = l2_normalize(X.float().cpu())
    c = l2_normalize(centroid.float().cpu().view(-1))
    d = 1.0 - torch.clamp(X_cpu[idxs] @ c.view(-1, 1), -1.0, 1.0).view(-1)
    order = torch.argsort(d).tolist()[: int(max(1, k))]
    return sorted([idxs[int(j)] for j in order])


def _coverage_tail_make_candidates(
    *,
    train_good_paths: List[str],
    X: torch.Tensor,
    labels: Optional[List[int]],
    centroids_cpu: Optional[torch.Tensor],
    n_modes: int,
    shots_per_mode: int,
    num_candidates: int,
    seed: int,
    support_seed: Optional[int],
    inlier_q: float,
    outlier_lambda: float,
    legacy_mode_to_paths: Dict[int, List[str]],
) -> List[Dict[str, Any]]:
    """Generate support-set candidates as {name, mode_to_paths, selected_indices}."""
    paths = list(train_good_paths)
    n = len(paths)
    n_modes = int(max(1, n_modes))
    k_each = int(max(1, shots_per_mode))
    X_cpu = l2_normalize(X.float().cpu()) if X.numel() else torch.empty((0, 0))
    if centroids_cpu is not None and getattr(centroids_cpu, "numel", lambda: 0)() and int(centroids_cpu.shape[0]) >= n_modes:
        C = l2_normalize(centroids_cpu.float().cpu())
    elif X_cpu.numel() > 0:
        C = l2_normalize(X_cpu.mean(dim=0, keepdim=True)).repeat(n_modes, 1)
    else:
        C = torch.empty((0, 0), dtype=torch.float32)
    mode_idxs = _coverage_tail_mode_indices(labels, n, n_modes)
    rng_base = int(seed if support_seed is None else support_seed)
    rng = random.Random(rng_base)
    out: List[Dict[str, Any]] = []

    def _mk_record(name: str, mode_to_indices: Dict[int, List[int]]) -> Dict[str, Any]:
        m2p: Dict[int, List[str]] = {}
        selected: List[int] = []
        for m in range(n_modes):
            inds = [int(i) for i in mode_to_indices.get(int(m), [])]
            if not inds:
                # fallback: central sample from the whole mode or whole pool
                pool = mode_idxs.get(int(m), []) or list(range(n))
                inds = _coverage_tail_closest_indices(X_cpu, pool, k=k_each, centroid=C[min(m, C.shape[0]-1)] if C.numel() else X_cpu[pool[0]]) if X_cpu.numel() else pool[:k_each]
            inds = sorted(inds[:k_each])
            selected.extend(inds)
            m2p[int(m)] = [paths[i] for i in inds]
        return {"name": str(name), "mode_to_paths": m2p, "selected_indices": sorted(set(selected))}

    # Candidate 0: current v22 draw/legacy support, kept as a reference.
    legacy_indices: Dict[int, List[int]] = {}
    path_to_i = {p: i for i, p in enumerate(paths)}
    for m, ps in legacy_mode_to_paths.items():
        legacy_indices[int(m)] = [path_to_i[p] for p in ps if p in path_to_i]
    out.append(_mk_record("legacy_or_seeded_draw", legacy_indices))

    # Candidate 1: central cluster medoids/closest-to-mode-centroid samples.
    central: Dict[int, List[int]] = {}
    for m in range(n_modes):
        pool = mode_idxs.get(int(m), []) or list(range(n))
        cent = C[min(m, C.shape[0]-1)] if C.numel() else X_cpu[pool].mean(dim=0)
        central[int(m)] = _coverage_tail_closest_indices(X_cpu, pool, k=k_each, centroid=cent) if X_cpu.numel() else pool[:k_each]
    out.append(_mk_record("cluster_central", central))

    # Candidate 2: pure-ish density-filtered k-center.
    kc: Dict[int, List[int]] = {}
    for m in range(n_modes):
        pool = mode_idxs.get(int(m), []) or list(range(n))
        cent = C[min(m, C.shape[0]-1)] if C.numel() else X_cpu[pool].mean(dim=0)
        kc[int(m)] = _coverage_tail_greedy_kcenter_indices(
            X_cpu, pool, k=k_each, centroid=cent, rng=rng, inlier_q=inlier_q,
            outlier_lambda=0.0, random_start=False, random_jitter=0.0,
        ) if X_cpu.numel() else pool[:k_each]
    out.append(_mk_record("density_filtered_kcenter", kc))

    # Candidate 3: outlier-penalized k-center.
    dkc: Dict[int, List[int]] = {}
    for m in range(n_modes):
        pool = mode_idxs.get(int(m), []) or list(range(n))
        cent = C[min(m, C.shape[0]-1)] if C.numel() else X_cpu[pool].mean(dim=0)
        dkc[int(m)] = _coverage_tail_greedy_kcenter_indices(
            X_cpu, pool, k=k_each, centroid=cent, rng=rng, inlier_q=inlier_q,
            outlier_lambda=outlier_lambda, random_start=False, random_jitter=0.0,
        ) if X_cpu.numel() else pool[:k_each]
    out.append(_mk_record("density_weighted_kcenter", dkc))

    # Remaining: randomized density-weighted k-center variants.
    target = int(max(1, num_candidates))
    cidx = 0
    while len(out) < target:
        rnd: Dict[int, List[int]] = {}
        rng_i = random.Random(rng_base + 1009 * (cidx + 1))
        for m in range(n_modes):
            pool = mode_idxs.get(int(m), []) or list(range(n))
            cent = C[min(m, C.shape[0]-1)] if C.numel() else X_cpu[pool].mean(dim=0)
            rnd[int(m)] = _coverage_tail_greedy_kcenter_indices(
                X_cpu, pool, k=k_each, centroid=cent, rng=rng_i, inlier_q=inlier_q,
                outlier_lambda=outlier_lambda, random_start=True, random_jitter=1e-4,
            ) if X_cpu.numel() else pool[:k_each]
        out.append(_mk_record(f"randomized_density_weighted_kcenter_{cidx:02d}", rnd))
        cidx += 1

    # Deduplicate by selected path set while preserving order.
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for rec in out:
        key = tuple(sorted([p for ps in rec["mode_to_paths"].values() for p in ps]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(rec)
        if len(uniq) >= target:
            break
    return uniq


@torch.no_grad()
def _coverage_tail_build_temp_vmb_and_score(
    *,
    class_name: str,
    mode_to_paths: Dict[int, List[str]],
    calib_paths: List[str],
    backbone: Any,
    centroids_cpu: torch.Tensor,
    n_modes: int,
    use_modes: bool,
    feature_fuse: str,
    views_by_name: Dict[str, ViewSpec],
    support_aug_cfg: Any,
    aug_per_image: int,
    baseline_compat: bool,
    support_include_identity: bool,
    K_normal_fixed: int,
    K_normal_dyn: int,
    K_normal_ltm: int,
    K_defect: int,
    K_concept: int,
    knn_k_normal: int,
    fixed_per_img_cap: int,
    bank_dtype: str,
    seed: int,
    topk_patches: int,
    score_mode: str,
    lam: float,
    tau_close: float,
    gamma: float,
    layer_fusion: str,
    view_fusion: str,
    img_agg: str,
    img_topk: int,
    img_quantile: float,
    empty_cache_every: int,
) -> List[float]:
    """Build a temporary fixed bank from a candidate support set and score normal calibration images."""
    temp_vmb = PatchVMB(
        device=str(backbone.device) if hasattr(backbone, "device") else "cuda",
        K_normal_fixed=int(K_normal_fixed),
        K_normal_dyn=max(1, int(K_normal_dyn)),
        K_normal_ltm=max(1, int(K_normal_ltm)),
        K_defect=max(1, int(K_defect)),
        K_concept=max(1, int(K_concept)),
        knn_k_normal=int(knn_k_normal),
        corr_cfg=CorrectionConfig(enabled=False),
        use_defect_bank=False,
        ltm_min_stm_size=16,
        ltm_min_repeat=2,
    )
    temp_vmb.bank_dtype = torch.float16 if str(bank_dtype).lower() == "fp16" else torch.float32
    temp_vmb.set_mode_centroids(class_name, centroids_cpu.detach().cpu() if centroids_cpu is not None and centroids_cpu.numel() else torch.empty((0, 0)))
    rng_np = np.random.RandomState(int(seed))

    # Initialize concept codebooks with the first available support feature.  The
    # concept ids are not used in candidate scoring, but score_sample_multiview
    # expects the per-class codebook object to exist.
    first_path = None
    for _m, _ps in mode_to_paths.items():
        if _ps:
            first_path = _ps[0]
            break
    if first_path is None:
        return []
    temp_vmb._init_concepts_if_needed(class_name, encode_pil(backbone, load_rgb(first_path), feature_fuse)[0])

    for mode, paths in mode_to_paths.items():
        for vname, vspec in views_by_name.items():
            support_feats_mv: List[List[torch.Tensor]] = []
            for pth in paths:
                img = load_rgb(pth)
                refs = make_support_refs(
                    img,
                    backbone,
                    support_aug_cfg,
                    aug_per_image,
                    baseline_compat=baseline_compat,
                    include_identity=support_include_identity,
                )
                for img_aug in refs:
                    img_v = apply_view_pil(img_aug, vspec)
                    feats_v, _ = encode_pil(backbone, img_v, feature_fuse)
                    slim: List[torch.Tensor] = []
                    for l in range(len(feats_v)):
                        t = l2_normalize(feats_v[l].to(temp_vmb.device))
                        m = int(t.shape[0])
                        if m <= 0:
                            slim.append(t)
                            continue
                        k = min(int(fixed_per_img_cap), m)
                        idx = rng_np.choice(m, size=k, replace=False)
                        slim.append(t[torch.from_numpy(idx).to(device=temp_vmb.device, dtype=torch.long)])
                    support_feats_mv.append(slim)
            temp_vmb.build_fixed_normal_from_support(
                cls_name=class_name,
                support_feats=support_feats_mv,
                mode=int(mode),
                view=str(vname),
                normal_seed_patches=8192,
                seed=int(seed),
            )
            if torch.cuda.is_available() and empty_cache_every > 0:
                torch.cuda.empty_cache()

    scores = compute_normal_train_calibration_scores(
        class_name=class_name,
        train_good_paths=calib_paths,
        backbone=backbone,
        vmb=temp_vmb,
        centroids_cpu=temp_vmb.get_mode_centroids(class_name),
        n_modes=int(n_modes),
        use_modes=bool(use_modes),
        feature_fuse=feature_fuse,
        views_by_name=views_by_name,
        topk_patches=topk_patches,
        score_mode=score_mode,
        lam=lam,
        tau_close=tau_close,
        gamma=gamma,
        layer_fusion=layer_fusion,
        view_fusion=view_fusion,
        img_agg=img_agg,
        img_topk=img_topk,
        img_quantile=img_quantile,
        empty_cache_every=empty_cache_every,
    )
    try:
        del temp_vmb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return scores


@torch.no_grad()
def select_coverage_tail_support_paths(
    *,
    class_name: str,
    train_good_paths: List[str],
    backbone: Any,
    labels: Optional[List[int]],
    centroids_cpu: Optional[torch.Tensor],
    n_modes: int,
    use_modes: bool,
    feature_fuse: str,
    views_by_name: Dict[str, ViewSpec],
    support_aug_cfg: Any,
    aug_per_image: int,
    baseline_compat: bool,
    support_include_identity: bool,
    legacy_mode_to_paths: Dict[int, List[str]],
    seed: int,
    support_seed: Optional[int],
    stream_seed: Optional[int],
    num_candidates: int,
    calib_max: int,
    inlier_q: float,
    tail_q: float,
    tail_q_hi: float,
    theta_q: float,
    lambda_tail_gap: float,
    lambda_std: float,
    lambda_outlier: float,
    cache_dir: Optional[str],
    K_normal_fixed: int,
    K_normal_dyn: int,
    K_normal_ltm: int,
    K_defect: int,
    K_concept: int,
    knn_k_normal: int,
    fixed_per_img_cap: int,
    bank_dtype: str,
    topk_patches: int,
    score_mode: str,
    lam: float,
    tau_close: float,
    gamma: float,
    layer_fusion: str,
    view_fusion: str,
    img_agg: str,
    img_topk: int,
    img_quantile: float,
    empty_cache_every: int = 50,
) -> Tuple[Dict[int, List[str]], Dict[str, Any]]:
    """Select few-shot support images by normal-tail validation.

    It uses all normal training images only to choose a small support set and to
    compute scalar calibration scores; the final fixed visual bank remains built
    from the selected few-shot supports.
    """
    paths = sorted(list(train_good_paths))
    if len(paths) == 0:
        return legacy_mode_to_paths, {"enabled": False, "reason": "empty_train_good"}
    run_cache = _coverage_tail_make_run_cache_dir(
        cache_dir, class_name=class_name, seed=seed, support_seed=support_seed, stream_seed=stream_seed
    )
    print(f"[support-select] coverage_tail class={class_name} candidates={int(num_candidates)} calib_max={int(calib_max)} cache={run_cache}")

    X = _coverage_tail_global_features(backbone, paths, feature_fuse=feature_fuse, empty_cache_every=empty_cache_every)
    candidates = _coverage_tail_make_candidates(
        train_good_paths=paths,
        X=X,
        labels=labels,
        centroids_cpu=centroids_cpu,
        n_modes=int(n_modes),
        shots_per_mode=int(max(1, max(len(v) for v in legacy_mode_to_paths.values()) if legacy_mode_to_paths else 1)),
        num_candidates=int(max(1, num_candidates)),
        seed=seed,
        support_seed=support_seed,
        inlier_q=float(inlier_q),
        outlier_lambda=float(lambda_outlier),
        legacy_mode_to_paths=legacy_mode_to_paths,
    )

    # Deterministic calibration subset from the normal train pool.  For each
    # candidate, selected supports are excluded before scoring to reduce optimistic bias.
    all_sorted = sorted(paths)
    calib_max = int(max(1, calib_max))
    results: List[Dict[str, Any]] = []
    best_rec: Optional[Dict[str, Any]] = None
    for ci, cand in enumerate(candidates):
        selected_paths = sorted([p for ps in cand["mode_to_paths"].values() for p in ps])
        selected_set = set(selected_paths)
        pool = [p for p in all_sorted if p not in selected_set]
        if len(pool) == 0:
            pool = list(all_sorted)
        if len(pool) > calib_max:
            idx = np.unique(np.round(np.linspace(0, len(pool) - 1, num=calib_max)).astype(int))
            calib_paths = [pool[int(i)] for i in idx]
        else:
            calib_paths = pool
        print(f"[support-select] eval {ci+1}/{len(candidates)} name={cand['name']} support={len(selected_paths)} calib={len(calib_paths)}")
        scores = _coverage_tail_build_temp_vmb_and_score(
            class_name=class_name,
            mode_to_paths=cand["mode_to_paths"],
            calib_paths=calib_paths,
            backbone=backbone,
            centroids_cpu=centroids_cpu if centroids_cpu is not None else torch.empty((0, 0)),
            n_modes=int(n_modes),
            use_modes=bool(use_modes),
            feature_fuse=feature_fuse,
            views_by_name=views_by_name,
            support_aug_cfg=support_aug_cfg,
            aug_per_image=int(aug_per_image),
            baseline_compat=bool(baseline_compat),
            support_include_identity=bool(support_include_identity),
            K_normal_fixed=int(K_normal_fixed),
            K_normal_dyn=int(K_normal_dyn),
            K_normal_ltm=int(K_normal_ltm),
            K_defect=int(K_defect),
            K_concept=int(K_concept),
            knn_k_normal=int(knn_k_normal),
            fixed_per_img_cap=int(fixed_per_img_cap),
            bank_dtype=str(bank_dtype),
            seed=int(seed) + 17 * int(ci),
            topk_patches=int(topk_patches),
            score_mode=str(score_mode),
            lam=float(lam),
            tau_close=float(tau_close),
            gamma=float(gamma),
            layer_fusion=str(layer_fusion),
            view_fusion=str(view_fusion),
            img_agg=str(img_agg),
            img_topk=int(img_topk),
            img_quantile=float(img_quantile),
            empty_cache_every=int(empty_cache_every),
        )
        arr = np.asarray(scores, dtype=np.float32)
        if arr.size == 0:
            obj = float("inf")
            stats = {"n": 0, "objective": obj}
        else:
            q_tail = float(np.quantile(arr, float(np.clip(tail_q, 0.0, 1.0))))
            q_hi = float(np.quantile(arr, float(np.clip(tail_q_hi, 0.0, 1.0))))
            std = float(arr.std(ddof=1)) if arr.shape[0] > 1 else float(arr.std())
            theta_sel = float(np.quantile(arr, float(np.clip(theta_q, 0.0, 1.0))))
            # Global-feature outlier penalty of selected supports.
            path_to_i = {p: i for i, p in enumerate(paths)}
            sel_idx = [path_to_i[p] for p in selected_paths if p in path_to_i]
            if X.numel() and sel_idx:
                selected_X = X[sel_idx]
                center = l2_normalize(X.mean(dim=0, keepdim=True)).squeeze(0)
                out_pen = float((1.0 - torch.clamp(selected_X @ center.view(-1, 1), -1.0, 1.0).view(-1)).mean().item())
            else:
                out_pen = 0.0
            obj = float(q_tail + float(lambda_tail_gap) * max(0.0, q_hi - q_tail) + float(lambda_std) * std + float(lambda_outlier) * out_pen)
            stats = {
                "n": int(arr.shape[0]),
                "mean": float(arr.mean()),
                "std": float(std),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "q_tail": float(q_tail),
                "q_hi": float(q_hi),
                "theta_q": float(theta_sel),
                "outlier_penalty": float(out_pen),
                "objective": float(obj),
            }
        rec = {
            "candidate_index": int(ci),
            "name": str(cand["name"]),
            "mode_to_paths": {str(k): list(v) for k, v in cand["mode_to_paths"].items()},
            "selected_paths": selected_paths,
            "calib_paths_n": int(len(calib_paths)),
            "score_stats": stats,
            # Keep scores only for the best record later; avoid massive diagnostics.
            "_calibration_scores": [float(x) for x in scores],
        }
        results.append(rec)
        if best_rec is None or float(stats.get("objective", float("inf"))) < float(best_rec["score_stats"].get("objective", float("inf"))):
            best_rec = rec

    if best_rec is None:
        return legacy_mode_to_paths, {"enabled": True, "failed": True, "reason": "no_candidate_scores"}
    best_scores = list(best_rec.pop("_calibration_scores", []))
    for rec in results:
        rec.pop("_calibration_scores", None)
    info = {
        "enabled": True,
        "selector": "coverage_tail",
        "class_name": str(class_name),
        "seed": int(seed),
        "support_seed": None if support_seed is None else int(support_seed),
        "stream_seed": None if stream_seed is None else int(stream_seed),
        "num_candidates_requested": int(num_candidates),
        "num_candidates_evaluated": int(len(results)),
        "calib_max": int(calib_max),
        "inlier_q": float(inlier_q),
        "tail_q": float(tail_q),
        "tail_q_hi": float(tail_q_hi),
        "theta_q_requested": float(theta_q),
        "lambda_tail_gap": float(lambda_tail_gap),
        "lambda_std": float(lambda_std),
        "lambda_outlier": float(lambda_outlier),
        "cache_dir": None if run_cache is None else str(run_cache),
        "best_candidate_index": int(best_rec["candidate_index"]),
        "best_candidate_name": str(best_rec["name"]),
        "best_score_stats": best_rec["score_stats"],
        "best_mode_to_paths": best_rec["mode_to_paths"],
        "candidates": results,
        "calibration_scores": best_scores,
    }
    if run_cache:
        diag_path = os.path.join(run_cache, "coverage_tail_support_selection.json")
        try:
            diag_save = dict(info)
            # Do not write full scalar calibration scores in the candidate list; keep
            # only the selected candidate's scores for reproducible theta diagnostics.
            diag_save["selected_calibration_scores"] = best_scores
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diag_save, f, indent=2, ensure_ascii=False)
            info["diagnostic_path"] = diag_path
        except Exception as e:
            info["diagnostic_error"] = repr(e)
    print(f"[support-select] selected name={info['best_candidate_name']} objective={float(info['best_score_stats'].get('objective', float('nan'))):.6f} theta_q={float(info['best_score_stats'].get('theta_q', float('nan'))):.6f}")
    best_mode_to_paths = {int(k): list(v) for k, v in info["best_mode_to_paths"].items()}
    return best_mode_to_paths, info


def run_phase3_streaming_aif(
    *,
    dataset: str,
    root: str,
    class_name: str,
    feature_fuse: str = "concat",
    csv: Optional[str],
    ablation: str = "full",
    baseline_compat: bool = False,
    support_include_identity: bool = False,
    dynonly_freeze_theta: bool = False,
    dynonly_freeze_normal_dyn: bool = False,
    device: str,
    max_samples: Optional[int],
    out_json: str,
    seed: int = 0,
    query_seed: Optional[int] = None,
    support_seed: Optional[int] = None,
    stream_seed: Optional[int] = None,
    init_dump_json: Optional[str] = None,
    use_global_threshold_initialization: bool = False,
    global_threshold_init_mode: str = "zscore_gaussian",
    support_select: str = "legacy",
    support_select_num_candidates: int = 8,
    support_select_calib_max: int = 256,
    support_select_inlier_q: float = 0.95,
    support_select_tail_q: float = 0.95,
    support_select_tail_q_hi: float = 0.99,
    support_select_theta_q: float = 0.925,
    support_select_lambda_tail_gap: float = 0.50,
    support_select_lambda_std: float = 0.05,
    support_select_lambda_outlier: float = 0.20,
    support_select_cache_dir: Optional[str] = None,

    # Phase-3 policy
    policy: str = "simple_aif",
    aif_cfg: "AIFConfig" = None,
    simple_aif_cfg: "SimpleAIFConfig" = None,

    # Category/mode index
    use_modes: bool = True,
    K_modes: int = 3,
    shots_per_mode: int = 4,

    # Support augmentation + pseudo views
    support_aug_cfg: SupportAugConfig = SupportAugConfig(),
    aug_per_image: int = 4,
    views: Optional[List[ViewSpec]] = None,
    use_rotations: bool = True,

    # VMB params
    K_normal_fixed: int = 4096,
    K_normal_dyn: int = 1024,
    K_normal_ltm: int = 512,
    ltm_min_stm_size: int = 16,
    ltm_min_repeat: int = 3,
    K_defect: int = 256,
    K_concept: int = 4096,
    knn_k_normal: int = 5,

    # correction bank + set-membership core normals
    corr_cfg: CorrectionConfig = CorrectionConfig(),
    sm_cfg: SetMembershipConfig = SetMembershipConfig(),
    use_defect_bank: bool = True,
    disable_defect_bank: bool = False,

    # scoring params
    topk_patches: int = 32,
    score_mode: str = "boost",     # boost or contrast
    lam: float = 0.8,
    tau_close: float = 0.20,
    gamma: float = 1.0,

    # fusion/aggregation
    layer_fusion: str = "mean",    # mean or max
    view_fusion: str = "mean",     # mean or max
    img_agg: str = "topk_mean",    # topk_mean | spiky_quantile | max
    img_topk: int = 64,
    img_quantile: float = 0.95,

    # bank update params
    tauN_insert: float = 0.15,
    tauN_replace: float = 0.25,
    tauD_insert: float = 0.10,
    tauD_replace: float = 0.20,

    # RB gating
    gating: RBGatingConfig = RBGatingConfig(),

    # query config
    qcfg: QueryConfig = QueryConfig(),

    # thresholding config (v4)
    thcfg: ThresholdConfig = ThresholdConfig(),

    theta_controller_kind: str = "default",
    theta_sidecar_cfg: GuardedThetaSidecarConfig = GuardedThetaSidecarConfig(),

    store_only_queried_cards: bool = True,

    concept_seed_patches: int = 2048,
    concept_per_img_cap: int = 256,
    fixed_per_img_cap: int = 512,
    bank_dtype: str = "fp16",
    empty_cache_every: int = 50,
    debug_defect_writes: bool = False,

    # v16 experiment controls
    stream_start_idx: Optional[int] = None,
    stream_end_idx: Optional[int] = None,
    stream_start_frac: Optional[float] = None,
    stream_end_frac: Optional[float] = None,
    no_shuffle_test: bool = False,
    save_state_path: Optional[str] = None,
    load_state_path: Optional[str] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    return_state: bool = False,
    save_temporal_artifacts_flag: bool = False,
    temporal_later_frac: float = 0.5,
    temporal_first_n: int = 200,
    enable_deployment_cost: bool = False,
    deployment_cost_cuda_sync: bool = False,
    enable_hidden_fn_analysis: bool = False,
    hidden_fn_near_sigma: float = 1.0,
    hidden_fn_low_p: float = 0.10,
    hidden_fn_topk: int = 20,
    enable_ltm_lite: bool = False,
    ltm_lite_retrieval: bool = True,
    ltm_lite_promotion: bool = True,
    # v20 memory-management controls extracted from v19
    enable_maturation_buffer: bool = False,
    maturation_ltm_retrieval: bool = True,
    maturation_ltm_promotion: bool = True,
    ltm_retrieval_k: int = 1,
    enable_maturation_buffer_retrieval: bool = False,
    mature_retrieval_min_events: int = 2,
    mature_retrieval_min_usefulness: float = 0.10,
    mature_retrieval_score_thr: float = 0.25,
    mature_retrieval_penalty: float = 0.01,
    mature_retrieval_max_protos: int = 256,
    mature_retrieval_k: int = 1,
    mature_merge_radius: float = 0.06,
    mature_event_merge_eps: float = 0.03,
    mature_max_event_protos: int = 8,
    mature_min_events: int = 2,
    mature_min_usefulness: float = 0.15,
    mature_score_thr: float = 0.45,
    mature_promote_every: int = 32,
    mature_max_candidates: int = 0,
    enable_stm_retirement: bool = False,
    stm_retire_ltm_eps: float = 0.025,
    enable_utility_aware_stm_retention: bool = False,
    stm_retention_local_k: int = 32,
    stm_retention_recency_tau: float = 1024.0,
    stm_retention_ltm_cover_eps: float = 0.025,
    stm_retention_replace_margin: float = 0.03,
    stm_retention_ema: float = 0.10,

    # Optional qualitative correction-bank diagnostics.
    # These are passive by default and only activate when --qual_corr_dump_dir is set.
    qual_corr_dump_dir: Optional[str] = None,
    qual_corr_classes: str = "",
    qual_corr_min_abs: float = 0.01,
    qual_corr_max_per_bucket: int = 2,
    qual_corr_save_flips_only: bool = False,
) -> Dict[str, Any]:

    set_seed(seed)
    ablation = str(ablation).lower()
    if ablation not in {"full", "static_baseline", "normal_dyn_only", "correction_only", "defect_only", "normal_dyn_plus_correction"}:
        raise ValueError(f"Unknown --ablation: {ablation}")
    # Unified policy alias: keep backward-compatible hardcode_band but also accept hardcode.
    policy = str(policy).lower().strip()
    if policy == "hardcode":
        policy = "hardcode_band"
    supported_policies = {"simple_aif", "hardcode_band", "selftrain_core"} | QUERY_BASELINE_POLICIES
    if policy not in supported_policies:
        raise ValueError(f"Unsupported --policy: {policy}; use one of {sorted(supported_policies)}")
    if disable_defect_bank:
        use_defect_bank = False
    if simple_aif_cfg is None:
        simple_aif_cfg = SimpleAIFConfig()
    global_threshold_init_mode = str(global_threshold_init_mode or "zscore_gaussian").lower().strip()
    if global_threshold_init_mode not in {"zscore_gaussian", "raw_quantile"}:
        raise ValueError(f"Unknown --global_threshold_init_mode: {global_threshold_init_mode}")
    support_select = str(support_select or "legacy").lower().strip()
    if support_select in {"default", "none"}:
        support_select = "legacy"
    if support_select not in {"legacy", "coverage_tail"}:
        raise ValueError(f"Unknown --support_select: {support_select}")

    # Protocol-compatible unsupervised comparator.
    # selftrain_core performs no human queries. It enriches normal_dyn from
    # confident predicted-normal samples accepted by the core-normal feasible-set
    # gate. The gate is intentionally conservative but not so strict that the
    # comparator degenerates into a static fixed-bank detector.
    if policy == "selftrain_core":
        sm_cfg.enabled = True

    # views default (VisionAD: identity + PosClamp + YFlip is a strong small set)
    if views is None:
        views = [
            ViewSpec(name="id", kind="id"),
            ViewSpec(name="posclamp", kind="posclamp", posclamp_low=64),
            ViewSpec(name="yflip", kind="yflip"),
        ]

    # Rotation pseudo-views (query + support). Enabled by default.
    if use_rotations:
        for vs in [
            ViewSpec(name="rot90", kind="rot90"),
            ViewSpec(name="rot180", kind="rot180"),
            ViewSpec(name="rot270", kind="rot270"),
        ]:
            if all(v.name != vs.name for v in views):
                views.append(vs)
    views_by_name: Dict[str, ViewSpec] = {v.name: v for v in views} 

    # 1) dataset index
    if dataset.lower() == "mvtec":
        index = IndustrialDatasetIndex(root=root)
        all_train = index.train_samples
        all_test = index.test_samples
    elif dataset.lower() == "visa":
        if csv is None:
            raise ValueError("--csv is required for VisA")
        index = VisACSVIndex(root=root, csv_path=csv)
        all_train = index.train_samples
        all_test = index.test_samples
    elif dataset.lower() == "realiad":
        # Real-IAD index via JSON variant folder under root/jsons/
        # We use the 'csv' argument to pass json_variant for Real-IAD (e.g. 'realiad_jsons_sv', 'realiad_jsons_fuiad_0.1').
        if csv is not None and os.path.isdir(csv):
            index = RealIADJSONIndex(root=root, json_dir=csv)
        else:
            # fallback: expect root/jsons/realiad_jsons
            cand = os.path.join(root, "jsons")
            index = RealIADJSONIndex(root=root, json_dir=cand)
        all_train = index.train_samples
        all_test = index.test_samples
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # 2) backbone
    backbone = DINOv2MultiLayerBackbone(device=device)
    vmb = PatchVMB(
        device=device,
        K_normal_fixed=K_normal_fixed,
        K_normal_dyn=K_normal_dyn,
        K_normal_ltm=K_normal_ltm,
        K_defect=K_defect,
        K_concept=K_concept,
        knn_k_normal=knn_k_normal,
        corr_cfg=corr_cfg,
        use_defect_bank=use_defect_bank,
        ltm_min_stm_size=ltm_min_stm_size,
        ltm_min_repeat=ltm_min_repeat,
    )
    vmb.bank_dtype = torch.float16 if bank_dtype == "fp16" else torch.float32
    if enable_ltm_lite:
        # Minimal v10+LTM ablation: keep the v10 controller and STM writes intact,
        # only enable bounded LTM consolidation / retrieval sidecar.
        vmb.shadow_ltm = False
        vmb.enable_ltm_retrieval = bool(ltm_lite_retrieval)
        vmb.enable_ltm_promotion = bool(ltm_lite_promotion)
        vmb.ltm_pressure_ratio = 0.50
        vmb.ltm_min_pending_updates = 16
        vmb.ltm_min_usefulness = 0.20
        vmb.ltm_score_threshold = 0.50

    # v20 memory-management module: keep the v16 controller/query loop stable,
    # but enable bounded memory consolidation and utility-aware STM retention.
    if bool(enable_maturation_buffer):
        vmb.shadow_ltm = False
        vmb.enable_maturation_buffer = True
        vmb.enable_ltm_retrieval = bool(maturation_ltm_retrieval)
        vmb.enable_ltm_promotion = bool(maturation_ltm_promotion)
        vmb.ltm_retrieval_k = int(max(1, ltm_retrieval_k))
        vmb.enable_maturation_buffer_retrieval = bool(enable_maturation_buffer_retrieval)
        vmb.mature_retrieval_min_events = int(max(1, mature_retrieval_min_events))
        vmb.mature_retrieval_min_usefulness = float(mature_retrieval_min_usefulness)
        vmb.mature_retrieval_score_thr = float(mature_retrieval_score_thr)
        vmb.mature_retrieval_penalty = float(max(0.0, mature_retrieval_penalty))
        vmb.mature_retrieval_max_protos = int(max(1, mature_retrieval_max_protos))
        vmb.mature_retrieval_k = int(max(1, mature_retrieval_k))
        vmb.mature_merge_radius = float(mature_merge_radius)
        vmb.ltm_candidate_merge_eps = float(mature_merge_radius)
        vmb.mature_event_merge_eps = float(mature_event_merge_eps)
        vmb.mature_max_event_protos = int(max(1, mature_max_event_protos))
        vmb.mature_event_prototypes_per_layer = int(max(1, mature_max_event_protos))
        vmb.mature_min_events = int(max(1, mature_min_events))
        vmb.mature_min_usefulness = float(mature_min_usefulness)
        vmb.mature_score_threshold = float(mature_score_thr)
        vmb.mature_score_thr = float(mature_score_thr)
        vmb.mature_promote_every = int(max(1, mature_promote_every))
        if int(mature_max_candidates) > 0:
            vmb.ltm_candidate_capacity = int(mature_max_candidates)
            vmb.mature_max_candidates = int(mature_max_candidates)
        vmb.enable_stm_retirement = bool(enable_stm_retirement)
        vmb.stm_retire_ltm_eps = float(stm_retire_ltm_eps)
        vmb.stm_retire_eps = float(stm_retire_ltm_eps)

    vmb.enable_utility_aware_stm_retention = bool(enable_utility_aware_stm_retention)
    vmb.stm_retention_local_k = int(max(1, stm_retention_local_k))
    vmb.stm_retention_recency_tau = float(stm_retention_recency_tau)
    vmb.stm_retention_ltm_cover_eps = float(stm_retention_ltm_cover_eps)
    vmb.stm_retention_replace_margin = float(stm_retention_replace_margin)
    vmb.stm_retention_ema = float(stm_retention_ema)

    rb = ReasoningBank()
    qs = QueryState()


    # Phase-3 AIF controller
    if aif_cfg is None:
        aif_cfg = AIFConfig()
    aif = AIFController(aif_cfg)

    # Support pool for QUERY_SUPPORT (normal images not used as initial support shots)
    support_pool = []
    used_support_paths = set()
    support_used = 0

    # 3) train good paths for class
    # If class_name is empty, evaluate all classes sequentially (memory-safe).
    if class_name == "":
        overall = {"dataset": dataset, "root": root, "classes": [], "per_class": {}, "metrics": {}}
        for cn in sorted(index.classes):
            out_base = out_json
            # write per-class outputs
            if out_base.lower().endswith(".json"):
                out_cls = out_base[:-5] + f"__{cn}.json"
            else:
                out_cls = out_base + f"__{cn}.json"
            summ = run_phase3_streaming_aif(    # recursive call itself
                dataset=dataset,
                root=root,
                class_name=cn,
                csv=csv,
                ablation=ablation,
                baseline_compat=baseline_compat,
                support_include_identity=support_include_identity,
                policy=policy,
                aif_cfg=aif_cfg,
                device=device,
                max_samples=max_samples,
                out_json=out_cls,
                seed=seed,
                query_seed=query_seed,
                support_seed=support_seed,
                stream_seed=stream_seed,
                feature_fuse=feature_fuse,
                use_global_threshold_initialization=bool(use_global_threshold_initialization),
                global_threshold_init_mode=str(global_threshold_init_mode),
                support_select=str(support_select),
                support_select_num_candidates=int(support_select_num_candidates),
                support_select_calib_max=int(support_select_calib_max),
                support_select_inlier_q=float(support_select_inlier_q),
                support_select_tail_q=float(support_select_tail_q),
                support_select_tail_q_hi=float(support_select_tail_q_hi),
                support_select_theta_q=float(support_select_theta_q),
                support_select_lambda_tail_gap=float(support_select_lambda_tail_gap),
                support_select_lambda_std=float(support_select_lambda_std),
                support_select_lambda_outlier=float(support_select_lambda_outlier),
                support_select_cache_dir=support_select_cache_dir,
                # pass through the rest via locals() pattern below
                shots_per_mode=shots_per_mode,
                K_modes=K_modes,
                use_modes=use_modes,
                K_normal_fixed=K_normal_fixed,
                K_normal_dyn=K_normal_dyn,
                K_defect=K_defect,
                K_concept=K_concept,
                knn_k_normal=knn_k_normal,
                topk_patches=topk_patches,
                score_mode=score_mode,
                lam=lam,
                tau_close=tau_close,
                gamma=gamma,
                views=views,
                use_rotations=use_rotations,
                support_aug_cfg=support_aug_cfg,
                corr_cfg=corr_cfg,
                use_defect_bank=use_defect_bank,
                disable_defect_bank=disable_defect_bank,
                enable_maturation_buffer=enable_maturation_buffer,
                maturation_ltm_retrieval=maturation_ltm_retrieval,
                maturation_ltm_promotion=maturation_ltm_promotion,
                ltm_retrieval_k=ltm_retrieval_k,
                enable_maturation_buffer_retrieval=enable_maturation_buffer_retrieval,
                mature_retrieval_min_events=mature_retrieval_min_events,
                mature_retrieval_min_usefulness=mature_retrieval_min_usefulness,
                mature_retrieval_score_thr=mature_retrieval_score_thr,
                mature_retrieval_penalty=mature_retrieval_penalty,
                mature_retrieval_max_protos=mature_retrieval_max_protos,
                mature_retrieval_k=mature_retrieval_k,
                mature_merge_radius=mature_merge_radius,
                mature_event_merge_eps=mature_event_merge_eps,
                mature_max_event_protos=mature_max_event_protos,
                mature_min_events=mature_min_events,
                mature_min_usefulness=mature_min_usefulness,
                mature_score_thr=mature_score_thr,
                mature_promote_every=mature_promote_every,
                mature_max_candidates=mature_max_candidates,
                enable_stm_retirement=enable_stm_retirement,
                stm_retire_ltm_eps=stm_retire_ltm_eps,
                enable_utility_aware_stm_retention=enable_utility_aware_stm_retention,
                stm_retention_local_k=stm_retention_local_k,
                stm_retention_recency_tau=stm_retention_recency_tau,
                stm_retention_ltm_cover_eps=stm_retention_ltm_cover_eps,
                stm_retention_replace_margin=stm_retention_replace_margin,
                stm_retention_ema=stm_retention_ema,
            )
            overall["classes"].append(cn)
            overall["per_class"][cn] = summ["metrics"]
        # save overall summary
        overall_path = out_json[:-5] + "__ALL_summary.json" if out_json.lower().endswith(".json") else out_json + "__ALL_summary.json"
        safe_makedirs(overall_path)
        with open(overall_path, "w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2)
        print("\n=== DATASET-WIDE SUMMARY ===")
        print(json.dumps(overall["per_class"], indent=2))
        return overall

    train_good = sorted([s for s in all_train if s.cls_name == class_name and s.is_good], key=lambda s: s.path)
    if len(train_good) == 0:
        raise RuntimeError(f"No normal train samples for class {class_name}")
    train_good_paths = [s.path for s in train_good]

    # 4) mode building (VisionAD category-index idea inside a class)
    if use_modes:
        centroids_cpu, labels = build_modes_from_train_good(
            backbone, train_good_paths, K_modes=K_modes, seed=seed
        )
        vmb.set_mode_centroids(class_name, centroids_cpu)
        n_modes = int(centroids_cpu.shape[0])
        # pick support paths per mode (cluster-conditional inlier filtering)
        mode_to_paths = select_clean_support_paths_per_mode(
            backbone, train_good_paths, labels, centroids_cpu,
            shots_per_mode=shots_per_mode, seed=seed,
        )
        # If --support_seed is explicitly provided, convert the deterministic
        # inlier-ranked candidates into a repeated support draw by sampling from
        # each mode's clean candidate pool. This isolates support-set identity
        # from stream order and query randomness.
        if support_seed is not None:
            rng_support = random.Random(int(support_seed))
            for k in range(n_modes):
                # True repeated-support draw: sample support image identities from
                # the eligible normal images assigned to this mode. This is more
                # appropriate for robustness testing than the legacy closest-inlier
                # deterministic selection.
                candidates = [train_good_paths[i] for i, lab in enumerate(labels) if int(lab) == int(k)]
                if candidates:
                    idxs = list(range(len(candidates)))
                    rng_support.shuffle(idxs)
                    chosen = sorted(idxs[:max(1, int(shots_per_mode))])
                    mode_to_paths[int(k)] = [candidates[i] for i in chosen]
        # fallback: ensure each mode has at least 1 support
        rng = random.Random(seed if support_seed is None else int(support_seed))
        for k in range(n_modes):
            if not mode_to_paths.get(int(k), []):
                idxs = [i for i, lab in enumerate(labels) if int(lab) == int(k)]
                rng.shuffle(idxs)
                chosen = idxs[:max(1, shots_per_mode)]
                mode_to_paths[int(k)] = [train_good_paths[i] for i in chosen]

    else:
        n_modes = 1
        mode_to_paths = {0: _sample_paths_deterministic_or_seeded(train_good_paths, max(1, shots_per_mode), support_seed)}
        vmb.set_mode_centroids(class_name, torch.empty((0, 0)))


    # 4a++) Optional coverage-aware support selection.
    # This replaces the final support draw before fixed-bank construction.  It
    # directly targets the failure mode observed in support-seed robustness runs:
    # poor support coverage creates unstable normal-score tails and bad theta0.
    support_selection_info: Dict[str, Any] = {"enabled": False, "selector": str(support_select)}
    if str(support_select).lower().strip() == "coverage_tail":
        mode_to_paths, support_selection_info = select_coverage_tail_support_paths(
            class_name=class_name,
            train_good_paths=train_good_paths,
            backbone=backbone,
            labels=labels if use_modes else None,
            centroids_cpu=centroids_cpu if use_modes else torch.empty((0, 0)),
            n_modes=int(n_modes),
            use_modes=bool(use_modes),
            feature_fuse=feature_fuse,
            views_by_name=views_by_name,
            support_aug_cfg=support_aug_cfg,
            aug_per_image=int(aug_per_image),
            baseline_compat=bool(baseline_compat),
            support_include_identity=bool(support_include_identity),
            legacy_mode_to_paths=mode_to_paths,
            seed=int(seed),
            support_seed=support_seed,
            stream_seed=stream_seed,
            num_candidates=int(support_select_num_candidates),
            calib_max=int(support_select_calib_max),
            inlier_q=float(support_select_inlier_q),
            tail_q=float(support_select_tail_q),
            tail_q_hi=float(support_select_tail_q_hi),
            theta_q=float(support_select_theta_q),
            lambda_tail_gap=float(support_select_lambda_tail_gap),
            lambda_std=float(support_select_lambda_std),
            lambda_outlier=float(support_select_lambda_outlier),
            cache_dir=support_select_cache_dir,
            K_normal_fixed=int(K_normal_fixed),
            K_normal_dyn=int(K_normal_dyn),
            K_normal_ltm=int(K_normal_ltm),
            K_defect=int(K_defect),
            K_concept=int(K_concept),
            knn_k_normal=int(knn_k_normal),
            fixed_per_img_cap=int(fixed_per_img_cap),
            bank_dtype=str(bank_dtype),
            topk_patches=int(topk_patches),
            score_mode=str(score_mode),
            lam=float(lam),
            tau_close=float(tau_close),
            gamma=float(gamma),
            layer_fusion=str(layer_fusion),
            view_fusion=str(view_fusion),
            img_agg=str(img_agg),
            img_topk=int(img_topk),
            img_quantile=float(img_quantile),
            empty_cache_every=int(empty_cache_every),
        )
        print(f"[support-select] final_support_paths={json.dumps({str(k): v for k, v in mode_to_paths.items()}, ensure_ascii=False)}")


    # 4b) Set-membership core-normal feasible sets (seeded from per-mode support globals)
    core_sets: Dict[int, CoreNormalFeasibleSet] = {}
    if sm_cfg.enabled:
        for m_id, paths in mode_to_paths.items():
            seed_globals: List[torch.Tensor] = []
            for p in paths:
                img0 = load_rgb(p)
                _, g0 = encode_pil(backbone, img0, feature_fuse)
                seed_globals.append(g0)
            core_sets[int(m_id)] = CoreNormalFeasibleSet(seed_globals, sm_cfg)


     # ---- Memory-safe support seeding: stream  per-image sampling ----
     # Seed concept cb incrementally without storing support_feats_identity_all.
     # We push sampled patch vectors directly into concept cb update.
    vmb._init_concepts_if_needed(class_name, encode_pil(backbone, load_rgb(mode_to_paths[list(mode_to_paths.keys())[0]][0]), feature_fuse)[0])
    rng_np = np.random.RandomState(seed)
    concept_seed_patches = concept_seed_patches #2048
    concept_per_img_cap = concept_per_img_cap #256

    seen = 0
    for mode, paths in mode_to_paths.items():
        for p in paths:
            seen +=1
            img = load_rgb(p)
            refs = make_support_refs(
                    img,
                    backbone,
                    support_aug_cfg,
                    aug_per_image,
                    baseline_compat=baseline_compat,
                    include_identity=support_include_identity,
                )
            for img_aug in refs:   # include identity
                feats_id, _ = encode_pil(backbone, img_aug, feature_fuse)
                for l in range(len(feats_id)):
                    t = l2_normalize(feats_id[l].to(vmb.device))
                    m = int(t.shape[0])
                    if m <= 0: 
                        continue
                    k = min(concept_per_img_cap, m)
                    idx = rng_np.choice(m, size=k, replace=False)
                    vmb.concept_cb[class_name][l].update(
                         t[torch.from_numpy(idx).to(device=vmb.device, dtype=torch.long)],
                         tau_insert=0.10, tau_replace=0.30
                     )
            if torch.cuda.is_available() and empty_cache_every > 0 and (seen % empty_cache_every == 0):
                torch.cuda.empty_cache()

    seen = 0
    # seed fixed normal banks per (mode, view) in a memory-safe way:
    for mode, paths in mode_to_paths.items():
        for vname, vspec in views_by_name.items():
            support_feats_mv: List[List[torch.Tensor]] = []
            for p in paths:
                seen +=1
                img = load_rgb(p)
                refs = make_support_refs(
                    img,
                    backbone,
                    support_aug_cfg,
                    aug_per_image,
                    baseline_compat=baseline_compat,
                    include_identity=support_include_identity,
                )
                for img_aug in refs:     # include identity
                    img_v = apply_view_pil(img_aug, vspec)
                    feats_v, _ = encode_pil(backbone, img_v, feature_fuse)
                    # keep only per-image subsample so support_feats_mv stays small
                    slim = []
                    for l in range(len(feats_v)):
                        t = l2_normalize(feats_v[l].to(vmb.device))
                        m = int(t.shape[0])
                        if m <= 0:
                            slim.append(t)
                            continue
                        k = min(fixed_per_img_cap, m)
                        idx = rng_np.choice(m, size=k, replace=False)
                        slim.append(t[torch.from_numpy(idx).to(device=vmb.device, dtype=torch.long)])
                    support_feats_mv.append(slim)
            if torch.cuda.is_available() and empty_cache_every > 0 and (seen % empty_cache_every == 0):
                 torch.cuda.empty_cache()

            vmb.build_fixed_normal_from_support(
                cls_name=class_name,
                support_feats=support_feats_mv,
                mode=mode,
                view=vname,
                normal_seed_patches=8192,
                seed=seed,
            )

    # support-only scores for initializing ClassStats (identity view only, per mode)
    stats = rb.get_or_create_stats(class_name)
    # build QUERY_SUPPORT pool: all good train samples for this class (we will filter out initial support shots)
    for s in all_train:
        if (s.cls_name == class_name) and s.is_good:
            support_pool.append(s)
    used_support_paths = set([p for _m, _paths in mode_to_paths.items() for p in _paths])
    support_pool = [s for s in support_pool if s.path not in used_support_paths]


    support_scores: List[float] = []
    support_global_feats: List[torch.Tensor] = []
    for mode, paths in mode_to_paths.items():  # why not to use used_support_paths here?
        for p in paths:
            img = load_rgb(p)
            feats_id, g = encode_pil(backbone, img, feature_fuse)
            feats_id = [l2_normalize(x) for x in feats_id]
            support_global_feats.append(l2_normalize(g.detach().float().cpu()).view(-1))
            score_s, _, _ = vmb.score_sample_multiview(
                cls_name=class_name,
                view_to_feats={"id": feats_id},
                mode=mode,
                views={"id": ViewSpec("id", "id")},
                topk_patches=topk_patches,
                mode_scoring="boost",
                lam=lam,
                tau_close=tau_close,
                gamma=gamma,
                layer_fusion=layer_fusion,
                view_fusion="mean",
                img_agg=img_agg,
                img_topk=img_topk,
                img_quantile=img_quantile,
                concept_assign_view="id",
                img_hw=(img.height, img.width)
            )
            support_scores.append(float(score_s))
    threshold_init_scores: List[float] = list(support_scores)
    threshold_calibration_scores: List[float] = []
    threshold_init_source = "support"
    threshold_init_info: Dict[str, Any] = {
        "enabled": bool(use_global_threshold_initialization),
        "source": "support",
        "n_support_scores": int(len(support_scores)),
        "n_calibration_scores": 0,
        "target_fpr": float(thcfg.target_fpr),
        "mode": str(global_threshold_init_mode),
    }
    # coverage_tail provides a normal-train calibration score set from the
    # selected support candidate.  Use it for theta0 before falling back to the
    # optional all-normal global threshold initializer.
    if str(support_select).lower().strip() == "coverage_tail" and support_selection_info.get("calibration_scores"):
        threshold_calibration_scores = [float(x) for x in support_selection_info.get("calibration_scores", [])]
        threshold_init_scores = list(threshold_calibration_scores)
        threshold_init_source = "support_select_coverage_tail"
        arr_cal = np.asarray(threshold_calibration_scores, dtype=np.float32)
        q_cal = float(np.clip(float(support_select_theta_q), 0.0, 1.0))
        mu_cal = float(arr_cal.mean())
        sig_cal = float(arr_cal.std(ddof=1)) if arr_cal.shape[0] > 1 else float(arr_cal.std())
        sig_cal = float(max(sig_cal, 1e-6))
        theta_q_sel = float(np.quantile(arr_cal, q_cal))
        threshold_init_info = {
            "enabled": True,
            "source": "support_select_coverage_tail",
            "mode": "raw_quantile",
            "n_support_scores": int(len(support_scores)),
            "n_calibration_scores": int(len(threshold_calibration_scores)),
            "target_fpr": float(thcfg.target_fpr),
            "support_select_theta_q": float(q_cal),
            "theta_selected": float(theta_q_sel),
            "mu": float(mu_cal),
            "sigma": float(sig_cal),
            "min": float(arr_cal.min()),
            "max": float(arr_cal.max()),
            "support_selection": {k: v for k, v in support_selection_info.items() if k != "calibration_scores"},
        }
        print(f"[support-select-threshold] n={len(threshold_calibration_scores)} theta_q{q_cal:.3f}={theta_q_sel:.6f}")
    elif bool(use_global_threshold_initialization):
        print(f"[global-threshold-init] scoring normal train pool for class={class_name} (n={len(train_good_paths)})")
        threshold_calibration_scores = compute_normal_train_calibration_scores(
            class_name=class_name,
            train_good_paths=train_good_paths,
            backbone=backbone,
            vmb=vmb,
            centroids_cpu=vmb.get_mode_centroids(class_name),
            n_modes=int(n_modes),
            use_modes=bool(use_modes),
            feature_fuse=feature_fuse,
            views_by_name=views_by_name,
            topk_patches=topk_patches,
            score_mode=score_mode,
            lam=lam,
            tau_close=tau_close,
            gamma=gamma,
            layer_fusion=layer_fusion,
            view_fusion=view_fusion,
            img_agg=img_agg,
            img_topk=img_topk,
            img_quantile=img_quantile,
            empty_cache_every=empty_cache_every,
        )
        if len(threshold_calibration_scores) > 0:
            threshold_init_scores = list(threshold_calibration_scores)
            threshold_init_source = "normal_train_pool"
            arr_cal = np.asarray(threshold_calibration_scores, dtype=np.float32)
            q_cal = float(np.clip(1.0 - float(thcfg.target_fpr), 0.0, 1.0))
            mu_cal = float(arr_cal.mean())
            sig_cal = float(arr_cal.std(ddof=1)) if arr_cal.shape[0] > 1 else float(arr_cal.std())
            sig_cal = float(max(sig_cal, 1e-6))
            z_cal = _safe_normal_ppf(q_cal)
            theta_q = float(np.quantile(arr_cal, q_cal))
            theta_z = float(mu_cal + sig_cal * z_cal)
            theta_selected = theta_z if global_threshold_init_mode == "zscore_gaussian" else theta_q
            threshold_init_info = {
                "enabled": True,
                "source": "normal_train_pool",
                "mode": str(global_threshold_init_mode),
                "n_support_scores": int(len(support_scores)),
                "n_calibration_scores": int(len(threshold_calibration_scores)),
                "target_fpr": float(thcfg.target_fpr),
                "quantile": float(q_cal),
                "z_quantile": float(z_cal),
                "theta_quantile": float(theta_q),
                "theta_gaussian": float(theta_z),
                "theta_selected": float(theta_selected),
                "mu": float(mu_cal),
                "sigma": float(sig_cal),
                "min": float(arr_cal.min()),
                "max": float(arr_cal.max()),
            }
            print(f"[global-threshold-init] mode={global_threshold_init_mode} n={len(threshold_calibration_scores)} theta_selected={theta_selected:.6f} theta_q={theta_q:.6f} theta_z={theta_z:.6f} target_fpr={float(thcfg.target_fpr):.4f}")
        else:
            print("[global-threshold-init] no calibration scores found; falling back to support-only theta initialization")

    stats.init_from_support(threshold_init_scores)

    # v4 robust thresholding (legacy or guarded dual-anchor sidecar)
    if theta_sidecar_cfg is None:
        theta_sidecar_cfg = GuardedThetaSidecarConfig()
    if str(theta_controller_kind).lower().strip() == "guarded_dual_anchor":
        th = GuardedDualAnchorThresholdController(theta_sidecar_cfg, thcfg)
    else:
        th = NormalThresholdController(thcfg)
    if threshold_init_source in {"normal_train_pool", "support_select_coverage_tail"}:
        th.init_from_normal_calibration(
            support_scores=support_scores,
            calibration_scores=threshold_calibration_scores,
            mode=("raw_quantile" if threshold_init_source == "support_select_coverage_tail" else str(global_threshold_init_mode)),
        )
        if threshold_init_source == "support_select_coverage_tail" and len(threshold_calibration_scores) > 0:
            # Existing init_from_normal_calibration uses target_fpr -> q95 by default.
            # For coverage-aware support, we expose an independent theta quantile
            # because industrial AD may prefer a lower-FNR operating point.
            theta_override = float(np.quantile(np.asarray(threshold_calibration_scores, dtype=np.float32), float(np.clip(support_select_theta_q, 0.0, 1.0))))
            try:
                th.theta0 = float(theta_override)
                if hasattr(th, "theta_current"):
                    th.theta_current = float(theta_override)
                # For the legacy normal-only controller, theta() is computed from
                # bufN once populated.  Align its target quantile with the requested
                # coverage-tail theta quantile.
                if hasattr(th, "cfg") and hasattr(th.cfg, "target_fpr") and not hasattr(th, "theta_current"):
                    th.cfg.target_fpr = float(max(0.0, min(1.0, 1.0 - float(support_select_theta_q))))
                if hasattr(th, "support_guard_info") and isinstance(th.support_guard_info, dict):
                    th.support_guard_info["coverage_tail_theta_override"] = float(theta_override)
                    th.support_guard_info["coverage_tail_theta_q"] = float(support_select_theta_q)
            except Exception:
                pass
    else:
        th.init_from_support(support_scores)
    coverage_mem = NormalCoverageMemory.from_support_feats(
        support_global_feats,
        max_dynamic=int(K_normal_dyn),
    )
    simple_aif_state = SimpleAIFState(simple_aif_cfg)

    resume_step_offset = 0
    loaded_resume_state = None
    if load_state_path is not None:
        loaded_resume_state = torch.load(load_state_path, map_location="cpu")
    elif resume_state is not None:
        loaded_resume_state = resume_state
    if loaded_resume_state is not None:
        resume_step_offset = apply_resume_state(
            resume_state=loaded_resume_state,
            class_name=class_name,
            vmb=vmb,
            rb=rb,
            stats_obj=stats,
            th=th,
            coverage_mem=coverage_mem,
            simple_aif_state=simple_aif_state,
            qs=qs,
        )
        print(f"[resume] loaded state for class={class_name} with step_offset={resume_step_offset}")

    if init_dump_json:
        safe_makedirs(init_dump_json)
        init_dump = {
            "script_name": os.path.basename(__file__),
            "dataset": dataset,
            "class_name": class_name,
            "seed": int(seed),
            "support_seed": None if support_seed is None else int(support_seed),
            "stream_seed": None if stream_seed is None else int(stream_seed),
            "feature_fuse": feature_fuse,
            "ablation": ablation,
            "baseline_compat": bool(baseline_compat),
            "support_include_identity": bool(support_include_identity),
            "K_modes": int(K_modes),
            "shots_per_mode": int(shots_per_mode),
            "support_paths_per_mode": {str(k): list(v) for k, v in mode_to_paths.items()},
            "support_scores": [float(x) for x in support_scores],
            "threshold_initialization": threshold_init_info,
            "support_selection": {k: v for k, v in support_selection_info.items() if k != "calibration_scores"},
            "theta0": float(th.theta()),
            "muN0": float(th.muN0),
            "sigmaN0": float(th.sigmaN0),
            "theta_controller_kind": str(theta_controller_kind),
            "theta_controller_state": th.debug_state() if hasattr(th, "debug_state") else {},
            "centroids": centroids_cpu.tolist() if (use_modes and centroids_cpu.numel()) else [],
            "mode_labels_preview": labels[: min(len(labels), 200)] if use_modes else [],
        }
        with open(init_dump_json, "w", encoding="utf-8") as f:
            json.dump(init_dump, f, indent=2)

    # 6) build test stream
    test_samples = sorted([s for s in all_test if s.cls_name == class_name], key=lambda s: s.path)
    if len(test_samples) == 0:
        raise RuntimeError(f"No test samples for class {class_name}")
    if not no_shuffle_test:
        if stream_seed is None:
            random.shuffle(test_samples)
        else:
            rng_stream = random.Random(int(stream_seed))
            rng_stream.shuffle(test_samples)

    total_available = len(test_samples)
    if stream_start_frac is not None or stream_end_frac is not None:
        s_frac = 0.0 if stream_start_frac is None else float(stream_start_frac)
        e_frac = 1.0 if stream_end_frac is None else float(stream_end_frac)
        s_frac = max(0.0, min(1.0, s_frac))
        e_frac = max(0.0, min(1.0, e_frac))
        if e_frac < s_frac:
            raise ValueError(f"Invalid stream fraction slice: start={s_frac}, end={e_frac}")
        frac_s_idx = int(math.floor(float(total_available) * s_frac))
        frac_e_idx = int(math.floor(float(total_available) * e_frac))
        stream_start_idx = frac_s_idx if stream_start_idx is None else int(stream_start_idx)
        stream_end_idx = frac_e_idx if stream_end_idx is None else int(stream_end_idx)

    s_idx = 0 if stream_start_idx is None else max(0, int(stream_start_idx))
    e_idx = total_available if stream_end_idx is None else min(total_available, int(stream_end_idx))
    if e_idx < s_idx:
        raise ValueError(f"Invalid stream slice: start_idx={s_idx}, end_idx={e_idx}")
    test_samples = test_samples[s_idx:e_idx]

    if max_samples is not None:
        test_samples = test_samples[:max_samples]

    # 7) streaming loop
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_q: List[int] = []
    tp = tn = fp = fn = 0
    n_queries = 0
    v22_forced_warmup_queries = 0
    v22_audit_queries = 0
    v22_fn_audit_queries = 0
    v22_fn_audit_last_step = -10**9
    v22_budget_suppressed_queries = 0

    centroids_cpu = vmb.get_mode_centroids(class_name)
    print(len(test_samples))

    record_txt_path = str(Path(out_json).with_suffix("")) + "_record.txt"
    record_handle = open(record_txt_path, "w", encoding="utf-8")
    print(f"[record] writing stream log to: {record_txt_path}")

    qual_corr_records: List[Dict[str, Any]] = []
    qual_corr_counts: Dict[Tuple[str, str], int] = {}
    qual_corr_class_set = set(parse_comma_list(qual_corr_classes)) if str(qual_corr_classes or "").strip() else set()
    qual_corr_enabled = bool(qual_corr_dump_dir) and ((not qual_corr_class_set) or (str(class_name) in qual_corr_class_set))
    if qual_corr_enabled:
        os.makedirs(str(qual_corr_dump_dir), exist_ok=True)
        print(f"[qual-corr] enabled class={class_name} dump_dir={qual_corr_dump_dir}")

    # v22_diagnostics: optional deployment-cost and hidden-FN diagnostics.
    # These are passive by default and do not change model behavior.
    deployment_cost_enabled = bool(enable_deployment_cost)
    hidden_fn_enabled = bool(enable_hidden_fn_analysis)

    if deployment_cost_enabled and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    deployment_cost: Dict[str, Any] = {
        "enabled": bool(deployment_cost_enabled),
        "cuda_sync": bool(deployment_cost_cuda_sync),
        "n_total": 0,
        "n_query": 0,
        "n_no_query": 0,
        "total_step_sec": 0.0,
        "query_step_sec": 0.0,
        "no_query_step_sec": 0.0,
        "stage_sec": {"load": 0.0, "feature": 0.0, "score": 0.0, "policy": 0.0, "update": 0.0},
    }

    hidden_fn_records: List[Dict[str, Any]] = []
    hidden_fn_summary_work: Dict[str, Any] = {
        "enabled": bool(hidden_fn_enabled),
        "near_sigma": float(hidden_fn_near_sigma),
        "low_p_threshold": float(hidden_fn_low_p),
        "total_defects": 0,
        "pred_normal_defects": 0,
        "hidden_unqueried_fns": 0,
        "queried_pred_normal_defects": 0,
        "near_threshold_pred_normal_defects": 0,
        "near_threshold_hidden_unqueried_fns": 0,
        "low_conf_hidden_unqueried_fns": 0,
        "far_below_threshold_hidden_unqueried_fns": 0,
    }

    core_selftrain_diag: Dict[str, Any] = {
        "enabled": bool(str(policy) == "selftrain_core"),
        "pred_normal_seen": 0,
        "global_accepts": 0,
        "patch_update_events": 0,
        "patch_added": 0,
        "patch_replaced": 0,
        "view_update_calls": 0,
    }

    def _sum_core_update_stats(update_stats: Dict[str, Any]) -> Tuple[int, int]:
        added = 0
        replaced = 0
        if not isinstance(update_stats, dict):
            return 0, 0
        layers = update_stats.get("layers", {})
        if isinstance(layers, dict):
            for st in layers.values():
                if isinstance(st, dict):
                    added += int(st.get("n_added", 0))
                    replaced += int(st.get("n_replaced", 0))
        return int(added), int(replaced)

    def _core_feasible_set_diagnostics() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for m_id, fs in core_sets.items():
            try:
                out[str(int(m_id))] = fs.diagnostics()
            except Exception as _e:
                out[str(m_id)] = {"error": repr(_e)}
        return out

    def _diag_cuda_sync() -> None:
        if deployment_cost_enabled and deployment_cost_cuda_sync and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def _diag_add_stage(stage: str, seconds: float) -> None:
        if not deployment_cost_enabled:
            return
        deployment_cost["stage_sec"][stage] = float(deployment_cost["stage_sec"].get(stage, 0.0) + max(0.0, float(seconds)))

    def _diag_finish_step(step_start: float, update_start: float, queried_flag: bool) -> None:
        if not deployment_cost_enabled:
            return
        _diag_cuda_sync()
        end_time = time.perf_counter()
        update_sec = max(0.0, float(end_time - update_start))
        step_sec = max(0.0, float(end_time - step_start))
        deployment_cost["stage_sec"]["update"] = float(deployment_cost["stage_sec"].get("update", 0.0) + update_sec)
        deployment_cost["total_step_sec"] = float(deployment_cost.get("total_step_sec", 0.0) + step_sec)
        deployment_cost["n_total"] = int(deployment_cost.get("n_total", 0)) + 1
        if bool(queried_flag):
            deployment_cost["n_query"] = int(deployment_cost.get("n_query", 0)) + 1
            deployment_cost["query_step_sec"] = float(deployment_cost.get("query_step_sec", 0.0) + step_sec)
        else:
            deployment_cost["n_no_query"] = int(deployment_cost.get("n_no_query", 0)) + 1
            deployment_cost["no_query_step_sec"] = float(deployment_cost.get("no_query_step_sec", 0.0) + step_sec)

    def _hidden_fn_observe(*, t_i: int, global_t_i: int, sample_path_i: str, y_true_i: int, pred_i: bool,
                           queried_i: bool, score_i: float, theta_i: float, sigmaN_i: float,
                           p_i: float, novelty_i: float, outcome_i: str) -> None:
        if not hidden_fn_enabled:
            return
        if int(y_true_i) == 1:
            hidden_fn_summary_work["total_defects"] = int(hidden_fn_summary_work.get("total_defects", 0)) + 1
        if int(y_true_i) != 1 or bool(pred_i):
            return
        sigma = max(float(sigmaN_i), 1e-6)
        gap = float(theta_i) - float(score_i)
        gap_sigma = float(gap / sigma)
        near = bool(0.0 <= gap_sigma <= float(hidden_fn_near_sigma))
        low_conf = bool(float(p_i) <= float(hidden_fn_low_p))
        far_below = bool(gap_sigma > 2.0)
        hidden_fn_summary_work["pred_normal_defects"] = int(hidden_fn_summary_work.get("pred_normal_defects", 0)) + 1
        if near:
            hidden_fn_summary_work["near_threshold_pred_normal_defects"] = int(hidden_fn_summary_work.get("near_threshold_pred_normal_defects", 0)) + 1
        if bool(queried_i):
            hidden_fn_summary_work["queried_pred_normal_defects"] = int(hidden_fn_summary_work.get("queried_pred_normal_defects", 0)) + 1
        else:
            hidden_fn_summary_work["hidden_unqueried_fns"] = int(hidden_fn_summary_work.get("hidden_unqueried_fns", 0)) + 1
            if near:
                hidden_fn_summary_work["near_threshold_hidden_unqueried_fns"] = int(hidden_fn_summary_work.get("near_threshold_hidden_unqueried_fns", 0)) + 1
            if low_conf:
                hidden_fn_summary_work["low_conf_hidden_unqueried_fns"] = int(hidden_fn_summary_work.get("low_conf_hidden_unqueried_fns", 0)) + 1
            if far_below:
                hidden_fn_summary_work["far_below_threshold_hidden_unqueried_fns"] = int(hidden_fn_summary_work.get("far_below_threshold_hidden_unqueried_fns", 0)) + 1
        hidden_fn_records.append({
            "t": int(t_i),
            "global_t": int(global_t_i),
            "path": str(sample_path_i),
            "score_img": float(score_i),
            "theta": float(theta_i),
            "sigmaN": float(sigma),
            "theta_minus_score": float(gap),
            "theta_minus_score_sigma": float(gap_sigma),
            "p_defect": float(p_i),
            "novelty": float(novelty_i),
            "queried": bool(queried_i),
            "outcome": str(outcome_i),
            "near_threshold": bool(near),
            "low_confidence": bool(low_conf),
            "far_below_threshold": bool(far_below),
        })

    for t, sample in enumerate(test_samples, start=1):
        if deployment_cost_enabled:
            _diag_cuda_sync()
            _diag_step_start = time.perf_counter()
            _diag_stage_start = _diag_step_start
        else:
            _diag_step_start = 0.0
            _diag_stage_start = 0.0

        global_t = int(resume_step_offset) + int(t)
        img = load_rgb(sample.path)
        if deployment_cost_enabled:
            _diag_add_stage("load", time.perf_counter() - _diag_stage_start)
            _diag_cuda_sync()
            _diag_stage_start = time.perf_counter()

        # identity encoding for mode routing + RB concepts
        feats_id, g = encode_pil(backbone, img, feature_fuse)
        feats_id = [l2_normalize(x) for x in feats_id]
        mode = choose_mode(g, centroids_cpu) if use_modes and centroids_cpu.numel() else 0
        mode = int(max(0, min(mode, n_modes - 1)))

        # encode all views
        view_to_feats: Dict[str, List[torch.Tensor]] = {}
        for vname, vspec in views_by_name.items():
            img_v = apply_view_pil(img, vspec)
            feats_v, _ = encode_pil(backbone, img_v, feature_fuse)
            view_to_feats[vname] = [l2_normalize(x) for x in feats_v]

        if deployment_cost_enabled:
            _diag_cuda_sync()
            _diag_add_stage("feature", time.perf_counter() - _diag_stage_start)
            _diag_stage_start = time.perf_counter()

        score_img, A_map_stats, view_to_feats = vmb.score_sample_multiview(   
            cls_name=class_name,
            view_to_feats=view_to_feats,
            mode=mode,
            views=views_by_name,
            topk_patches=topk_patches,
            mode_scoring=score_mode,
            lam=lam,
            tau_close=tau_close,
            gamma=gamma,
            layer_fusion=layer_fusion,
            view_fusion=view_fusion,
            img_agg=img_agg,
            img_topk=img_topk,
            img_quantile=img_quantile,
            concept_assign_view="id",
            img_hw=(img.height, img.width)
        )

        # v4: final decision uses normal-only, FPR-controlled theta; p_defect used for querying
        theta = th.theta()
        p_defect = th.posterior(score_img)
        pred_defect = (float(score_img) >= float(theta))
        if deployment_cost_enabled:
            _diag_cuda_sync()
            _diag_add_stage("score", time.perf_counter() - _diag_stage_start)
            _diag_stage_start = time.perf_counter()

        # boundary evidence seeded? (defect prototypes OR correction anchors)
        defect_seeded = False
        if vmb.use_defect_bank:
            defect_seeded = any(
                vmb.defect_cb[class_name][vname][l].size() > 0
                for vname in views_by_name
                for l in range(len(feats_id))
            )

        corr_seeded = False
        if vmb.corr_cfg.enabled:
            corr_seeded = any(
                vmb.corr_cb[class_name][vname][l].size() > 0
                for vname in views_by_name
                for l in range(len(feats_id))
            )

        boundary_seeded = bool(defect_seeded or corr_seeded)

        #novelty = compute_novelty_percentile(score_img, qs.recent_scores)
        update_query_state_score(qs, qcfg, score_img)
        novelty, novelty_info = coverage_mem.novelty_percentile(g)
        novelty = float(np.clip(
            (float(novelty) - float(simple_aif_cfg.coverage_ref_clip_low))
            / max(float(simple_aif_cfg.coverage_ref_clip_high) - float(simple_aif_cfg.coverage_ref_clip_low), 1e-6),
            0.0,
            1.0,
        ))
        rb_error = compute_rb_error_rate(rb, class_name, A_map_stats, pred_defect=pred_defect, min_count=gating.min_count)

        # --- Phase-3 policy selection (AIF or heuristic) ---
        action: AIFAction = AIFAction.CLASSIFY
        regime: AIFRegime = AIFRegime.STABLE
        aif_info: Dict[str, Any] = {}
        evidence = compute_two_channel_evidence(
            score=float(score_img), theta=float(theta), sigmaN=float(th.sigmaN()), A_map_stats=A_map_stats
        )
        theta_eff_n = th.effective_normal_count()
        theta_eff_n_next = th.effective_normal_count(extra_verified_normals=1)

        if ablation == "static_baseline":
            queried = False
            action = AIFAction.CLASSIFY
            q_policy = "static_baseline"
            aif_info = {"ablation": ablation, "action": action.value, "feasible_actions": [AIFAction.CLASSIFY.value]}
        elif ablation in {"normal_dyn_only", "correction_only", "normal_dyn_plus_correction"}:
            if str(policy) == "hardcode_band":
                queried = bool(float(qcfg.ambig_low) <= float(p_defect) <= float(qcfg.ambig_high))
                action = AIFAction.QUERY_LABEL if queried else AIFAction.CLASSIFY
                aif_info = {
                    "controller": "hardcode_band",
                    "action": action.value,
                    "ablation": ablation,
                    "novelty": float(novelty),
                    "novelty_info": novelty_info,
                    "band": [float(qcfg.ambig_low), float(qcfg.ambig_high)],
                    "G": {},
                    "G_val": None,
                }
                q_policy = f"{ablation}::hardcode_band::{float(qcfg.ambig_low):.3f}_{float(qcfg.ambig_high):.3f}::{action.value}"
            elif str(policy) == "simple_aif":
                dynamic_normal_count = int(coverage_mem.dynamic_feats.shape[0]) if hasattr(coverage_mem, "dynamic_feats") else 0
                boot_action, boot_reason, boot_info = choose_simple_aif_bootstrap_action(
                    step=global_t,
                    score=float(score_img),
                    p_defect=float(p_defect),
                    theta=float(theta),
                    sigmaN=float(th.sigmaN()),
                    cfg=simple_aif_cfg,
                    dynamic_normal_count=int(dynamic_normal_count),
                )
                if boot_action is not None:
                    action = boot_action
                    queried = bool(action == AIFAction.QUERY_LABEL)
                    aif_info = dict(boot_info)
                    aif_info["controller"] = "simple_aif"
                    aif_info["version"] = "v8"
                    aif_info["action"] = action.value
                    aif_info["ablation"] = ablation
                    aif_info["novelty"] = float(novelty)
                    aif_info["novelty_info"] = novelty_info
                    aif_info["G"] = {}
                    aif_info["G_val"] = None
                    q_policy = f"{ablation}::simple_aif::v8::bootstrap::{boot_reason}"
                else:
                    action, aif_info = select_simple_aif_action(
                        cfg=simple_aif_cfg,
                        state=simple_aif_state,
                        ablation=ablation,
                        p_defect=float(p_defect),
                        score=float(score_img),
                        theta=float(theta),
                        sigmaN=float(th.sigmaN()),
                        pred_defect=bool(pred_defect),
                        novelty=float(novelty),
                        novelty_info=novelty_info,
                        evidence=evidence,
                        A_map_stats=A_map_stats,
                    )
                    queried = bool(action == AIFAction.QUERY_LABEL)
                    q_policy = f"{ablation}::simple_aif::v8::{action.value}"
                # v22 optional warm-up/audit override before updating AIF budget state.
                if bool(getattr(simple_aif_cfg, "v22_enable_warmup_budget", False)) or bool(getattr(simple_aif_cfg, "v22_enable_defect_audit", False)) or bool(getattr(simple_aif_cfg, "v22_enable_fn_audit", False)):
                    N_stream = max(1, int(len(test_samples)))
                    warm_T = int(max(1, min(N_stream, int(getattr(simple_aif_cfg, "v22_warmup_steps", 1000)))))
                    target_q = float(np.clip(float(simple_aif_cfg.target_qrate), 0.0, 1.0))
                    warm_q = float(np.clip(float(getattr(simple_aif_cfg, "v22_warmup_qrate", 0.06)), target_q, 1.0))
                    total_target = float(target_q) * float(N_stream)
                    warm_target_total = min(float(warm_q) * float(warm_T), total_target)
                    post_q = max(0.0, (total_target - warm_target_total) / float(max(1, N_stream - warm_T))) if N_stream > warm_T else target_q
                    desired_now = warm_q * float(t) if int(t) <= warm_T else warm_target_total + post_q * float(int(t) - warm_T)
                    desired_floor = int(math.floor(desired_now)); desired_ceil = int(math.ceil(desired_now))
                    if bool(getattr(simple_aif_cfg, "v22_budget_guard", True)) and bool(queried) and int(n_queries) >= desired_ceil:
                        queried = False; action = AIFAction.CLASSIFY; v22_budget_suppressed_queries += 1
                        q_policy = f"{q_policy}::v22_budget_suppress"; aif_info["v22_budget_suppressed"] = True
                    forced_reason = None
                    audit_eligible = (bool(getattr(simple_aif_cfg, "v22_enable_defect_audit", False)) and int(t) <= warm_T and bool(pred_defect)
                                      and float(p_defect) >= float(getattr(simple_aif_cfg, "v22_audit_min_p", 0.80))
                                      and float(score_img) >= float(theta) + float(getattr(simple_aif_cfg, "v22_audit_score_sigma", 1.0)) * float(max(th.sigmaN(), 1e-6)))
                    audit_target = int(math.floor(float(getattr(simple_aif_cfg, "v22_audit_frac", 0.25)) * desired_now))

                    # Balanced-rescue FN audit: predicted-normal, close below theta.
                    # This exposes hidden false negatives without turning the whole policy into a high-query audit.
                    fn_audit_warm_T = int(max(1, min(N_stream, int(getattr(simple_aif_cfg, "v22_fn_audit_warmup_steps", 1500)))))
                    fn_audit_band = float(max(0.0, getattr(simple_aif_cfg, "v22_fn_audit_band", 0.040)))
                    fn_audit_extra_q = float(max(0.0, getattr(simple_aif_cfg, "v22_fn_audit_max_extra_qrate", 0.005)))
                    fn_audit_target = int(max(1, math.floor(fn_audit_extra_q * float(min(int(t), fn_audit_warm_T))))) if fn_audit_extra_q > 0 else 0
                    fn_gap_ok = (int(t) - int(v22_fn_audit_last_step)) >= int(max(1, getattr(simple_aif_cfg, "v22_fn_audit_min_gap", 20)))
                    fn_audit_margin = float(theta) - float(score_img)
                    fn_audit_eligible = (
                        bool(getattr(simple_aif_cfg, "v22_enable_fn_audit", False))
                        and int(t) <= fn_audit_warm_T
                        and (not bool(pred_defect))
                        and fn_gap_ok
                        and float(fn_audit_margin) >= 0.0
                        and float(fn_audit_margin) <= fn_audit_band
                        and float(p_defect) >= float(getattr(simple_aif_cfg, "v22_fn_audit_min_p", 0.05))
                    )

                    if (not queried) and fn_audit_eligible and int(v22_fn_audit_queries) < int(fn_audit_target):
                        forced_reason = "fn_audit_near_theta"
                    if (not queried) and forced_reason is None and audit_eligible and int(v22_audit_queries) < audit_target:
                        forced_reason = "defect_side_audit"
                    if (not queried) and forced_reason is None and bool(getattr(simple_aif_cfg, "v22_enable_warmup_budget", False)) and int(t) <= warm_T and int(n_queries) < desired_floor:
                        forced_reason = "warmup_budget_catchup"
                    if forced_reason is not None:
                        queried = True; action = AIFAction.QUERY_LABEL
                        if forced_reason == "defect_side_audit":
                            v22_audit_queries += 1
                        elif forced_reason == "fn_audit_near_theta":
                            v22_fn_audit_queries += 1
                            v22_fn_audit_last_step = int(t)
                        else:
                            v22_forced_warmup_queries += 1
                        q_policy = f"{q_policy}::v22_force::{forced_reason}"
                        aif_info["v22_forced_query"] = True; aif_info["v22_force_reason"] = str(forced_reason)
                    aif_info["v22"] = {"warmup_steps": int(warm_T), "warmup_qrate": float(warm_q), "post_warmup_qrate": float(post_q), "desired_now": float(desired_now), "desired_floor": int(desired_floor), "desired_ceil": int(desired_ceil), "audit_eligible": bool(audit_eligible), "audit_target": int(audit_target), "audit_queries_so_far": int(v22_audit_queries), "fn_audit_eligible": bool(fn_audit_eligible), "fn_audit_target": int(fn_audit_target), "fn_audit_queries_so_far": int(v22_fn_audit_queries), "fn_audit_margin": float(fn_audit_margin), "forced_warmup_queries_so_far": int(v22_forced_warmup_queries), "budget_suppressed_so_far": int(v22_budget_suppressed_queries)}
                simple_aif_state.observe(bool(queried))
            elif str(policy) == "selftrain_core":
                queried = False
                action = AIFAction.CLASSIFY
                aif_info = {
                    "controller": "selftrain_core",
                    "action": action.value,
                    "ablation": ablation,
                    "novelty": float(novelty),
                    "novelty_info": novelty_info,
                    "G": {},
                    "G_val": None,
                    "note": "No human query; only conservative core-normal self-training may update normal_dyn.",
                }
                q_policy = f"{ablation}::selftrain_core::classify"
            elif str(policy) in QUERY_BASELINE_POLICIES:
                action, aif_info = select_query_baseline_action(
                    policy=str(policy),
                    qs=qs,
                    simple_cfg=simple_aif_cfg,
                    seed=int(seed),
                    query_seed=query_seed,
                    step=int(global_t),
                    p_defect=float(p_defect),
                    score=float(score_img),
                    theta=float(theta),
                    sigmaN=float(th.sigmaN()),
                    novelty=float(novelty),
                    novelty_info=novelty_info,
                    evidence=evidence,
                )
                queried = bool(action == AIFAction.QUERY_LABEL)
                aif_info["ablation"] = str(ablation)
                q_policy = f"{ablation}::{str(policy)}::target_qrate_{float(simple_aif_cfg.target_qrate):.4f}::{action.value}"
            else:
                raise ValueError(f"Unsupported policy in v21-query-baselines: {policy}")
        else:
            raise ValueError(f"Unsupported ablation/policy combination in v16: {ablation} / {policy}")

        if deployment_cost_enabled:
            _diag_cuda_sync()
            _diag_add_stage("policy", time.perf_counter() - _diag_stage_start)

        core_added = False
        core_update_stats: Dict[str, Any] = {}

        y_true = 0 if sample.is_good else 1

        # simple_aif in v10 only uses classify vs query_label.
        region_stats = None
        has_direct_feedback = bool(queried)
        y_user = y_true if has_direct_feedback else None

        # outcome based on prediction vs truth
        if pred_defect and y_true == 1:
            outcome = "TP"; tp += 1
        elif (not pred_defect) and y_true == 0:
            outcome = "TN"; tn += 1
        elif pred_defect and y_true == 0:
            outcome = "FP"; fp += 1
        else:
            outcome = "FN"; fn += 1

        _hidden_fn_observe(
            t_i=int(t),
            global_t_i=int(global_t),
            sample_path_i=str(sample.path),
            y_true_i=int(y_true),
            pred_i=bool(pred_defect),
            queried_i=bool(queried),
            score_i=float(score_img),
            theta_i=float(theta),
            sigmaN_i=float(th.sigmaN()),
            p_i=float(p_defect),
            novelty_i=float(novelty),
            outcome_i=str(outcome),
        )

        if deployment_cost_enabled:
            _diag_cuda_sync()
            _diag_update_start = time.perf_counter()
        else:
            _diag_update_start = 0.0

        if qual_corr_enabled:
            try:
                rough_corr = _qual_corr_extract_rough_stats(A_map_stats)
                rough_has_corr = (
                    float(rough_corr.get("corr_top_max", 0.0)) >= abs(float(qual_corr_min_abs))
                    or float(rough_corr.get("corr_top_min", 0.0)) <= -abs(float(qual_corr_min_abs))
                )
                if rough_has_corr:
                    debug_maps = vmb.score_sample_correction_debug_maps(
                        cls_name=class_name,
                        view_to_feats=view_to_feats,
                        mode=mode,
                        views=views_by_name,
                        topk_patches=topk_patches,
                        mode_scoring=score_mode,
                        lam=lam,
                        tau_close=tau_close,
                        gamma=gamma,
                        layer_fusion=layer_fusion,
                        view_fusion=view_fusion,
                        img_agg=img_agg,
                        img_topk=img_topk,
                        img_quantile=img_quantile,
                        concept_assign_view="id",
                        img_hw=(img.height, img.width),
                    )
                    maybe_dump_correction_qual_example(
                        qual_corr_dump_dir=qual_corr_dump_dir,
                        qual_corr_counts=qual_corr_counts,
                        qual_corr_records=qual_corr_records,
                        qual_corr_max_per_bucket=int(qual_corr_max_per_bucket),
                        qual_corr_min_abs=float(qual_corr_min_abs),
                        qual_corr_save_flips_only=bool(qual_corr_save_flips_only),
                        class_name=class_name,
                        sample_path=sample.path,
                        img=img,
                        t=int(t),
                        global_t=int(global_t),
                        y_true=int(y_true),
                        outcome=str(outcome),
                        theta=float(theta),
                        score_img=float(score_img),
                        pred_defect=bool(pred_defect),
                        maps=debug_maps,
                    )
            except Exception as _e:
                if int(t) <= 5 or bool(debug_defect_writes):
                    print(f"[qual-corr] warning: failed to dump example at class={class_name} t={t}: {_e}")

        if queried:
            n_queries += 1
            if ablation == "normal_dyn_only":
                suspicious = False
                r_user = 1.0
                inlier_mode = True
                allow_updates = True
                allow_normal_writes = False
                allow_defect_writes = False
                allow_correction_writes = False
                allow_bank_updates = False
                added_normal = False
                added_defect = False
                update_stats_all: Dict[str, Any] = {"normal_dyn": {}, "defect": {}, "correction": {}}
                if int(y_true) == 0:
                    if not dynonly_freeze_theta:
                        th.observe_labeled(float(score_img), 0)

                    stats.update(score_img, is_anomaly=False)

                    if not dynonly_freeze_normal_dyn:
                        n_patches = int(feats_id[0].shape[0])
                        _allow_write, _topk_keep, write_reason, write_info = select_informative_normal_write(
                            score=float(score_img),
                            theta=float(theta),
                            sigmaN=float(th.sigmaN()),
                            pred_defect=bool(pred_defect),
                            novelty=float(novelty),
                        )
                        update_stats_all["normal_dyn_shadow_write"] = {"reason": str(write_reason), **write_info}
                        layer_to_idx_norm = layer_to_topk_from_amap(A_map_stats)
                        for vname, vspec in views_by_name.items():
                            layer_to_idx_v = map_layer_patch_idx_to_view(layer_to_idx_norm, vspec, n_patches)
                            usN = vmb.update_from_topk(
                                cls_name=class_name,
                                feats_per_layer=view_to_feats[vname],
                                which="normal_dyn",
                                mode=mode,
                                view=vname,
                                layer_to_patch_idx=layer_to_idx_v,
                                tau_insert=tauN_insert,
                                tau_replace=tauN_replace,
                                event_id=global_t,
                                write_value=float(write_info.get("write_value", 1.0)),
                                outcome=outcome,
                                boundary_value=float(write_info.get("boundary", 0.0)),
                                coverage_gain=float(novelty),
                            )
                            update_stats_all["normal_dyn"][vname] = usN
                        added_normal = True
                    coverage_mem.observe_normal(g)
                else:
                    stats.update(score_img, is_anomaly=True)
                card = {
                    "t": t,
                    "class": class_name,
                    "path": sample.path,
                    "mode": mode,
                    "score_img": float(score_img),
                    "p_defect": float(p_defect),
                    "pred": int(pred_defect),
                    "y_true": int(y_true),
                    "queried": True,
                    "query_policy": q_policy,
                    "regime": regime.value if isinstance(regime, AIFRegime) else str(regime),
                    "novelty": float(novelty),
                    "rb_error_rate": float(rb_error),
                    "outcome": outcome,
                    "A_map_stats": A_map_stats,
                    "region_stats": region_stats,
                    "contam": {
                        "evidence": evidence,
                        "suspicious": False,
                        "inlier_mode": True,
                        "r_user": 1.0,
                        "has_direct_feedback": True,
                        "allow_updates": True,
                        "allow_bank_updates": False,
                        "allow_normal_writes": False,
                        "allow_defect_writes": False,
                        "allow_correction_writes": False,
                    },
                    "defect_write_debug": None,
                    "aif": aif_info,
                    "stats": {
                        "mu_N": stats.mu_N, "var_N": stats.var_N(), "n_N": stats.n_N,
                        "mu_A": stats.mu_A, "var_A": stats.var_A(), "n_A": stats.n_A,
                        "theta_dyn": stats.theta_dyn,
                        "theta": float(theta),
                        "sigmaN": float(th.sigmaN()),
                        "bufN_len": int(len(getattr(th, "bufN", []))),
                        "bufA_len": int(len(getattr(th, "bufA", []))),
                        "theta_sidecar_last_mode": str(getattr(th, "last_mode", "none")),
                        "theta_sidecar_last_move": float(getattr(th, "last_move", 0.0)),
                        "theta_sidecar_last_accept": bool(getattr(th, "last_accept", False)),
                        "coverage_support_size": int(coverage_mem.support_feats.shape[0]) if coverage_mem.support_feats.ndim == 2 else 0,
                        "coverage_dynamic_size": int(coverage_mem.dynamic_feats.shape[0]) if coverage_mem.dynamic_feats.ndim == 2 else 0,
                    },
                    "bank_updates": {
                        "added_normal_dyn": bool(added_normal),
                        "added_defect": False,
                        "added_correction": False,
                        "core_added": False,
                        "core_update_stats": {},
                        "update_stats": update_stats_all,
                    },
                    "bank_sizes": {
                        "normal_fixed": {
                            vname: [vmb.normal_fixed[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "normal_dyn": {
                            vname: [vmb.normal_dyn[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "normal_dyn_ltm": {
                            vname: [vmb.normal_dyn_ltm[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "defect": {
                            vname: [vmb.defect_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "correction": {
                            vname: [vmb.corr_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "use_defect_bank": bool(vmb.use_defect_bank),
                        "corr_enabled": bool(vmb.corr_cfg.enabled),
                    },
                }
                rb.add_card(card)
                qrate = n_queries / max(1, t)
                all_scores.append(float(score_img))
                all_labels.append(int(y_true))

                acc = (tp + tn) / max(1, (tp + tn + fp + fn))
                show_content = (
                    f"[t={t:03d}] mode={mode} score={score_img:.4f} theta={theta:.4f} p={p_defect:.3f} pred={int(pred_defect)} "
                    f"true={y_true} outcome={outcome} acc={acc:.3f} qrate={qrate:.3f} "
                    f"regime={regime.value if isinstance(regime, AIFRegime) else str(regime)} "
                    f"action={action.value if isinstance(action, AIFAction) else q_policy} "
                    f"EFE_value={aif_info.get('G_val', None)} "
                    f"queried=1 dyn_only=1 added_normal_dyn={int(added_normal)}"
                )

                print(show_content)
                record_handle.write(show_content + "\n")
                query_debug = {
                "t": int(t),
                "mode": int(mode),
                "queried": True,
                "ablation": "normal_dyn_only",
                "score_img": float(score_img),
                "theta": float(theta),
                "p_defect": float(p_defect),
                "pred": int(pred_defect),
                "y_true": int(y_true),
                "outcome": str(outcome),
                "added_normal_dyn": bool(added_normal),
                "bufN_len": int(len(getattr(th, "bufN", []))),
                        "bufA_len": int(len(getattr(th, "bufA", []))),
                        "theta_sidecar_last_mode": str(getattr(th, "last_mode", "none")),
                        "theta_sidecar_last_move": float(getattr(th, "last_move", 0.0)),
                        "theta_sidecar_last_accept": bool(getattr(th, "last_accept", False)),
                "normal_dyn_sizes": {
                    vname: [vmb.normal_dyn[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                    for vname in views_by_name
                },
            }
                record_handle.write("[QDEBUG] " + json.dumps(query_debug) + "\n")
                record_handle.flush()

                if empty_cache_every and (t % int(empty_cache_every) == 0) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _diag_finish_step(_diag_step_start, _diag_update_start, True)
                continue
            
            if ablation == "correction_only":
                suspicious = False
                r_user = 1.0
                inlier_mode = True
                allow_updates = True
                allow_normal_writes = False
                allow_defect_writes = False
                allow_correction_writes = True
                allow_bank_updates = False

                added_normal = False
                added_defect = False
                update_stats_all: Dict[str, Any] = {"normal_dyn": {}, "defect": {}, "correction": {}}

                # IMPORTANT: threshold controller disabled here
                # so do NOT call:
                #   th.observe_labeled(...)
                #   stats.update(...)
                #   th.observe_unlabeled(...)

                n_patches = int(feats_id[0].shape[0])
                raw_layer_to_idx = get_raw_topk_patches(A_map_stats, topk_keep=16)
                layer_to_idx_corr = raw_layer_to_idx

                corr_sign = +1 if int(y_true) == 1 else -1
                corr_strength = 1.0

                for vname, vspec in views_by_name.items():
                    layer_to_idx_v = map_layer_patch_idx_to_view(layer_to_idx_corr, vspec, n_patches)
                    cs = vmb.update_corrections_from_topk(
                        cls_name=class_name,
                        feats_per_layer=view_to_feats[vname],
                        mode=mode,
                        view=vname,
                        layer_to_patch_idx=layer_to_idx_v,
                        sign=corr_sign,
                        score_img=float(score_img),
                        theta=float(theta),
                        strength_scale=float(corr_strength),
                        outcome=outcome,
                        step=global_t,
                    )
                    update_stats_all["correction"][vname] = cs

                card = {
                    "t": t,
                    "class": class_name,
                    "path": sample.path,
                    "mode": mode,
                    "score_img": float(score_img),
                    "p_defect": float(p_defect),
                    "pred": int(pred_defect),
                    "y_true": int(y_true),
                    "queried": True,
                    "query_policy": q_policy,
                    "regime": regime.value if isinstance(regime, AIFRegime) else str(regime),
                    "novelty": float(novelty),
                    "rb_error_rate": float(rb_error),
                    "outcome": outcome,
                    "A_map_stats": A_map_stats,
                    "region_stats": region_stats,
                    "contam": {
                        "evidence": evidence,
                        "suspicious": False,
                        "inlier_mode": True,
                        "r_user": 1.0,
                        "has_direct_feedback": True,
                        "allow_updates": True,
                        "allow_bank_updates": False,
                        "allow_normal_writes": False,
                        "allow_defect_writes": False,
                        "allow_correction_writes": True,
                    },
                    "defect_write_debug": None,
                    "aif": aif_info,
                    "stats": {
                        "mu_N": stats.mu_N, "var_N": stats.var_N(), "n_N": stats.n_N,
                        "mu_A": stats.mu_A, "var_A": stats.var_A(), "n_A": stats.n_A,
                        "theta_dyn": stats.theta_dyn,
                        "theta": float(theta),
                        "sigmaN": float(th.sigmaN()),
                        "bufN_len": int(len(getattr(th, "bufN", []))),
                        "bufA_len": int(len(getattr(th, "bufA", []))),
                        "theta_sidecar_last_mode": str(getattr(th, "last_mode", "none")),
                        "theta_sidecar_last_move": float(getattr(th, "last_move", 0.0)),
                        "theta_sidecar_last_accept": bool(getattr(th, "last_accept", False)),
                        "coverage_support_size": int(coverage_mem.support_feats.shape[0]) if coverage_mem.support_feats.ndim == 2 else 0,
                        "coverage_dynamic_size": int(coverage_mem.dynamic_feats.shape[0]) if coverage_mem.dynamic_feats.ndim == 2 else 0,
                    },
                    "bank_updates": {
                        "added_normal_dyn": False,
                        "added_defect": False,
                        "added_correction": bool(any(len(v.get("layers", {})) > 0 for v in update_stats_all.get("correction", {}).values() if isinstance(v, dict))),
                        "core_added": False,
                        "core_update_stats": {},
                        "update_stats": update_stats_all,
                    },
                    "bank_sizes": {
                        "normal_fixed": {
                            vname: [vmb.normal_fixed[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "normal_dyn": {
                            vname: [vmb.normal_dyn[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "normal_dyn_ltm": {
                            vname: [vmb.normal_dyn_ltm[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "defect": {
                            vname: [vmb.defect_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "correction": {
                            vname: [vmb.corr_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "use_defect_bank": bool(vmb.use_defect_bank),
                        "corr_enabled": bool(vmb.corr_cfg.enabled),
                    },
                }

                rb.add_card(card)

                qrate = n_queries / max(1, t)
                all_scores.append(float(score_img))
                all_labels.append(int(y_true))

                acc = (tp + tn) / max(1, (tp + tn + fp + fn))
                show_content = (
                    f"[t={t:03d}] mode={mode} score={score_img:.4f} theta={theta:.4f} p={p_defect:.3f} pred={int(pred_defect)} "
                    f"true={y_true} outcome={outcome} acc={acc:.3f} qrate={qrate:.3f} "
                    f"regime={regime.value if isinstance(regime, AIFRegime) else str(regime)} "
                    f"action={action.value if isinstance(action, AIFAction) else q_policy} "
                    f"EFE_value={aif_info.get('G_val', None)} queried=1 correction_only=1"
                )
                print(show_content)
                record_handle.write(show_content + "\n")
                record_handle.flush()

                if empty_cache_every and (t % int(empty_cache_every) == 0) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _diag_finish_step(_diag_step_start, _diag_update_start, True)
                continue
            
            if ablation == "normal_dyn_plus_correction":
                suspicious = False
                r_user = 1.0
                inlier_mode = True
                allow_updates = True
                allow_normal_writes = False
                allow_defect_writes = False
                allow_correction_writes = True
                allow_bank_updates = False

                added_normal = False
                added_defect = False
                update_stats_all: Dict[str, Any] = {"normal_dyn": {}, "defect": {}, "correction": {}}

                # IMPORTANT:
                # threshold controller disabled here:
                #   do NOT call th.observe_labeled(...)
                #   do NOT call th.observe_unlabeled(...)
                #
                # but keep stats.update(...) if you want the same semantics as your
                # theta-frozen normal_dyn ablation.
                #
                # Patch: the guarded theta sidecar must observe BOTH queried normals and defects.
                th.observe_labeled(float(score_img), int(y_true))

                if int(y_true) == 0:
                    stats.update(score_img, is_anomaly=False)

                    # -------- normal_dyn write (v5 STM, shadow LTM) --------
                    n_patches = int(feats_id[0].shape[0])
                    _allow_write, _topk_keep, write_reason, write_info = select_informative_normal_write(
                        score=float(score_img),
                        theta=float(theta),
                        sigmaN=float(th.sigmaN()),
                        pred_defect=bool(pred_defect),
                        novelty=float(novelty),
                    )
                    update_stats_all["normal_dyn_shadow_write"] = {"reason": str(write_reason), **write_info}
                    layer_to_idx_norm = layer_to_topk_from_amap(A_map_stats)

                    for vname, vspec in views_by_name.items():
                        layer_to_idx_v = map_layer_patch_idx_to_view(layer_to_idx_norm, vspec, n_patches)
                        usN = vmb.update_from_topk(
                            cls_name=class_name,
                            feats_per_layer=view_to_feats[vname],
                            which="normal_dyn",
                            mode=mode,
                            view=vname,
                            layer_to_patch_idx=layer_to_idx_v,
                            tau_insert=tauN_insert,
                            tau_replace=tauN_replace,
                            event_id=global_t,
                            write_value=float(write_info.get("write_value", 1.0)),
                            outcome=outcome,
                            boundary_value=float(write_info.get("boundary", 0.0)),
                            coverage_gain=float(novelty),
                        )
                        update_stats_all["normal_dyn"][vname] = usN

                    added_normal = True
                    coverage_mem.observe_normal(g)
                else:
                    stats.update(score_img, is_anomaly=True)

                # -------- correction-bank write --------
                n_patches = int(feats_id[0].shape[0])
                raw_layer_to_idx = get_raw_topk_patches(A_map_stats, topk_keep=16)
                corr_sign = +1 if int(y_true) == 1 else -1
                corr_strength = 1.0

                for vname, vspec in views_by_name.items():
                    layer_to_idx_v = map_layer_patch_idx_to_view(raw_layer_to_idx, vspec, n_patches)
                    cs = vmb.update_corrections_from_topk(
                        cls_name=class_name,
                        feats_per_layer=view_to_feats[vname],
                        mode=mode,
                        view=vname,
                        layer_to_patch_idx=layer_to_idx_v,
                        sign=corr_sign,
                        score_img=float(score_img),
                        theta=float(theta),
                        strength_scale=float(corr_strength),
                        outcome=outcome,
                        step=global_t,
                    )
                    update_stats_all["correction"][vname] = cs

                card = {
                    "t": t,
                    "class": class_name,
                    "path": sample.path,
                    "mode": mode,
                    "score_img": float(score_img),
                    "p_defect": float(p_defect),
                    "pred": int(pred_defect),
                    "y_true": int(y_true),
                    "queried": True,
                    "query_policy": q_policy,
                    "regime": regime.value if isinstance(regime, AIFRegime) else str(regime),
                    "novelty": float(novelty),
                    "rb_error_rate": float(rb_error),
                    "outcome": outcome,
                    "A_map_stats": A_map_stats,
                    "region_stats": region_stats,
                    "contam": {
                        "evidence": evidence,
                        "suspicious": False,
                        "inlier_mode": True,
                        "r_user": 1.0,
                        "has_direct_feedback": True,
                        "allow_updates": True,
                        "allow_bank_updates": False,
                        "allow_normal_writes": False,
                        "allow_defect_writes": False,
                        "allow_correction_writes": True,
                    },
                    "defect_write_debug": None,
                    "aif": aif_info,
                    "stats": {
                        "mu_N": stats.mu_N, "var_N": stats.var_N(), "n_N": stats.n_N,
                        "mu_A": stats.mu_A, "var_A": stats.var_A(), "n_A": stats.n_A,
                        "theta_dyn": stats.theta_dyn,
                        "theta": float(theta),
                        "sigmaN": float(th.sigmaN()),
                        "bufN_len": int(len(getattr(th, "bufN", []))),
                        "bufA_len": int(len(getattr(th, "bufA", []))),
                        "theta_sidecar_last_mode": str(getattr(th, "last_mode", "none")),
                        "theta_sidecar_last_move": float(getattr(th, "last_move", 0.0)),
                        "theta_sidecar_last_accept": bool(getattr(th, "last_accept", False)),
                        "coverage_support_size": int(coverage_mem.support_feats.shape[0]) if coverage_mem.support_feats.ndim == 2 else 0,
                        "coverage_dynamic_size": int(coverage_mem.dynamic_feats.shape[0]) if coverage_mem.dynamic_feats.ndim == 2 else 0,
                    },
                    "bank_updates": {
                        "added_normal_dyn": bool(added_normal),
                        "added_defect": False,
                        "added_correction": bool(any(len(v.get("layers", {})) > 0 for v in update_stats_all.get("correction", {}).values() if isinstance(v, dict))),
                        "core_added": False,
                        "core_update_stats": {},
                        "update_stats": update_stats_all,
                    },
                    "bank_sizes": {
                        "normal_fixed": {
                            vname: [vmb.normal_fixed[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "normal_dyn": {
                            vname: [vmb.normal_dyn[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "normal_dyn_ltm": {
                            vname: [vmb.normal_dyn_ltm[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "defect": {
                            vname: [vmb.defect_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "correction": {
                            vname: [vmb.corr_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "use_defect_bank": bool(vmb.use_defect_bank),
                        "corr_enabled": bool(vmb.corr_cfg.enabled),
                    },
                }

                rb.add_card(card)

                qrate = n_queries / max(1, t)
                all_scores.append(float(score_img))
                all_labels.append(int(y_true))

                acc = (tp + tn) / max(1, (tp + tn + fp + fn))
                show_content = (
                    f"[t={t:03d}] mode={mode} score={score_img:.4f} theta={theta:.4f} p={p_defect:.3f} pred={int(pred_defect)} "
                    f"true={y_true} outcome={outcome} acc={acc:.3f} qrate={qrate:.3f} "
                    f"regime={regime.value if isinstance(regime, AIFRegime) else str(regime)} "
                    f"action={action.value if isinstance(action, AIFAction) else q_policy} "
                    f"EFE_value={aif_info.get('G_val', None)} queried=1 normal_dyn_plus_correction=1 "
                    f"added_normal_dyn={int(added_normal)}"
                )
                print(show_content)
                record_handle.write(show_content + "\n")
                record_handle.flush()

                if empty_cache_every and (t % int(empty_cache_every) == 0) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _diag_finish_step(_diag_step_start, _diag_update_start, True)
                continue 

            # --- Phase-3 contamination-aware feedback handling ---
            suspicious = False
            r_user = 1.0

            # Bayesian reliability updates only apply when we truly asked for the sample label/region.
            if has_direct_feedback:
                suspicious = is_label_suspicious(y_user=int(y_true), evidence=evidence, cfg=aif.cfg)

                # Feedback-channel reliability (r_user): update from self-consistency with two channels
                w_fb = min(3.0, 1.0 + abs(float(evidence.get('margin_z', 0.0))) / 2.0)
                aif.update_user_reliability(class_name, consistent=(not suspicious), w=w_fb)
                r_user = aif.belief(class_name).r_user.mean()

                # Model reliability (r_vmb): discounted by current feedback trust
                aif.update_vmb_reliability(class_name, correct=(pred_defect == bool(y_true)), w=float(r_user))

            # Cluster-conditional inlier gate (mode-specific) for NORMAL-side updates
            fs = core_sets.get(int(mode), None)
            inlier_mode = fs.is_inlier(g, slack=1.05) if fs is not None else True

            # Decide whether we trust this feedback enough to update memory/calibration.
            # QUERY_SUPPORT enriches support-derived normal memory only; it does not expose the current sample label.
            allow_updates = bool(has_direct_feedback and (not suspicious) and (float(r_user) >= float(aif.cfg.trust_min_write)))
            allow_normal_writes = bool(allow_updates and bool(inlier_mode))
            allow_defect_writes = bool(allow_updates)
            allow_correction_writes = bool(allow_updates)
            allow_bank_updates = bool(allow_normal_writes if int(y_true) == 0 else allow_defect_writes)

            # Experiment A: explicit diagnostics for queried defect-side writes (FN / TP).
            defect_debug_active = bool(debug_defect_writes and queried and (int(y_true) == 1) and (outcome in ["FN", "TP"]))
            defect_write_debug: Optional[Dict[str, Any]] = None
            if defect_debug_active:
                defect_write_debug = {
                    "outcome": str(outcome),
                    "queried": bool(queried),
                    "policy": str(policy),
                    "query_policy": str(q_policy),
                    "action": action.value if isinstance(action, AIFAction) else str(action),
                    "pred_defect": bool(pred_defect),
                    "y_true": int(y_true),
                    "score_img": float(score_img),
                    "theta": float(theta),
                    "p_defect": float(p_defect),
                    "sigmaN": float(th.sigmaN()),
                    "bufN_len": int(len(getattr(th, "bufN", []))),
                        "bufA_len": int(len(getattr(th, "bufA", []))),
                        "theta_sidecar_last_mode": str(getattr(th, "last_mode", "none")),
                        "theta_sidecar_last_move": float(getattr(th, "last_move", 0.0)),
                        "theta_sidecar_last_accept": bool(getattr(th, "last_accept", False)),
                    "boundary_seeded": bool(boundary_seeded),
                    "support_left": int(aif.support_left(class_name)) if (policy != "heuristic") else None,
                    "has_direct_feedback": bool(has_direct_feedback),
                    "allow_updates": bool(allow_updates),
                    "allow_bank_updates": bool(allow_bank_updates),
                    "suspicious": bool(suspicious),
                    "inlier_mode": bool(inlier_mode),
                    "r_user": float(r_user),
                    "trust_min_write": float(aif.cfg.trust_min_write),
                    "tau_insert": {"defect": float(tauD_insert), "correction": None},
                    "tau_replace": {"defect": float(tauD_replace), "correction": None},
                    "mask_grid_present": bool(mask_grid is not None),
                    "region_stats": region_stats,
                    "pre_sizes": {
                        "defect": {
                            vname: [vmb.defect_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                        "correction": {
                            vname: [vmb.corr_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                            for vname in views_by_name
                        },
                    },
                    "selected_patch_counts": {},
                    "selected_patch_total": 0,
                    "post_sizes": None,
                    "write_results": {},
                    "tp_note": None,
                }

            # Prepare update outputs (always defined for logging)
            added_normal = False
            added_defect = False
            update_stats_all: Dict[str, Any] = {"normal_dyn": {}, "defect": {}, "correction": {}}

            if allow_updates:
                # Always-on trusted global updates for the sidecar threshold controller.
                # Feed BOTH normal and defect queried labels; do not gate on inlier_mode here,
                # because the sidecar needs defect anchors as well.
                th.observe_labeled(score_img, y_true)

                stats.update(score_img, is_anomaly=(y_true == 1))

                per_layer = A_map_stats.get("per_layer_summary", {})
                for lk, info in per_layer.items():
                    layer = int(lk.replace("layer", ""))
                    rb.update_concepts(class_name, layer, info.get("topk_concept_ids", []), outcome)

                n_patches = int(feats_id[0].shape[0])
                raw_layer_to_idx = get_raw_topk_patches(A_map_stats, topk_keep=16)
                layer_to_idx_pos = raw_layer_to_idx
                layer_to_idx_neg = raw_layer_to_idx

                # In the simplified v10 path, queried label feedback always shapes the correction bank.
                if allow_correction_writes and queried:
                    corr_sign = +1 if int(y_true) == 1 else -1
                    layer_to_idx_corr = layer_to_idx_pos if corr_sign > 0 else layer_to_idx_neg
                    corr_strength = compute_correction_strength(
                        score=float(score_img),
                        theta=float(theta),
                        sigmaN=float(th.sigmaN()),
                        novelty=float(novelty),
                        rb_error=float(rb_error),
                        theta_unc=float(th.sigmaN()) / math.sqrt(max(1.0, float(theta_eff_n))),
                        cfg=aif.cfg,
                    )
                    if outcome in {"FP", "FN"}:
                        corr_strength *= float(aif.cfg.correction_mistake_bonus)

                    if defect_debug_active and (defect_write_debug is not None):
                        sel_counts = {f"layer{int(l)}": int(len(idx)) for l, idx in layer_to_idx_corr.items()}
                        defect_write_debug["selected_patch_counts"] = sel_counts
                        defect_write_debug["selected_patch_total"] = int(sum(sel_counts.values()))
                        defect_write_debug["region_stats"] = region_stats

                    for vname, vspec in views_by_name.items():
                        layer_to_idx_v = map_layer_patch_idx_to_view(layer_to_idx_corr, vspec, n_patches)
                        cs = vmb.update_corrections_from_topk(
                            cls_name=class_name,
                            feats_per_layer=view_to_feats[vname],
                            mode=mode,
                            view=vname,
                            layer_to_patch_idx=layer_to_idx_v,
                            sign=corr_sign,
                            score_img=float(score_img),
                            theta=float(theta),
                            strength_scale=float(corr_strength),
                            outcome=outcome,
                            step=global_t,
                        )
                        update_stats_all["correction"][vname] = cs

                # Positive side: TP and FN both enrich defect memory.
                if allow_defect_writes and int(y_true) == 1 and vmb.use_defect_bank:
                    layer_to_idx_def = layer_to_idx_pos
                    for vname, vspec in views_by_name.items():
                        layer_to_idx_v = map_layer_patch_idx_to_view(layer_to_idx_def, vspec, n_patches)
                        usD = vmb.update_from_topk(
                            cls_name=class_name,
                            feats_per_layer=view_to_feats[vname],
                            which="defect",
                            mode=mode,
                            view=vname,
                            layer_to_patch_idx=layer_to_idx_v,
                            tau_insert=tauD_insert,
                            tau_replace=tauD_replace,
                        )
                        update_stats_all["defect"][vname] = usD
                    added_defect = True

                # Negative side: TN and FP may enrich normal memory, but more conservatively.
                if allow_normal_writes and int(y_true) == 0 and should_write_normal_from_feedback(
                    outcome=outcome,
                    novelty=float(novelty),
                    score=float(score_img),
                    theta=float(theta),
                    sigmaN=float(th.sigmaN()),
                    regime=regime,
                    cfg=aif.cfg,
                ):
                    _allow_write, _topk_keep, write_reason, write_info = select_informative_normal_write(
                        score=float(score_img),
                        theta=float(theta),
                        sigmaN=float(th.sigmaN()),
                        pred_defect=bool(pred_defect),
                        novelty=float(novelty),
                    )
                    update_stats_all["normal_dyn_shadow_write"] = {"reason": str(write_reason), **write_info}
                    layer_to_idx_norm = layer_to_idx_neg
                    for vname, vspec in views_by_name.items():
                        layer_to_idx_v = map_layer_patch_idx_to_view(layer_to_idx_norm, vspec, n_patches)
                        usN = vmb.update_from_topk(
                            cls_name=class_name,
                            feats_per_layer=view_to_feats[vname],
                            which="normal_dyn",
                            mode=mode,
                            view=vname,
                            layer_to_patch_idx=layer_to_idx_v,
                            tau_insert=tauN_insert,
                            tau_replace=tauN_replace,
                            event_id=global_t,
                            write_value=float(write_info.get("write_value", 1.0)),
                            outcome=outcome,
                            boundary_value=float(write_info.get("boundary", 0.0)),
                            coverage_gain=float(novelty),
                        )
                        update_stats_all["normal_dyn"][vname] = usN
                    added_normal = True
                    coverage_mem.observe_normal(g)

                if defect_debug_active and (defect_write_debug is not None):
                    defect_write_debug["write_results"] = {
                        "defect": update_stats_all.get("defect", {}),
                        "correction": update_stats_all.get("correction", {}),
                        "normal_dyn": update_stats_all.get("normal_dyn", {}),
                    }

            else:
                # Feedback considered suspicious / low-trust: skip calibration and memory writes.
                update_stats_all["skipped"] = True
                update_stats_all["reason"] = "suspicious_or_low_trust"
            
            if defect_debug_active and (defect_write_debug is not None):
                if outcome == "TP":
                    defect_write_debug["tp_note"] = "Queried TP now writes positive-side correction anchors and defect-bank prototypes under trusted feedback."
                defect_write_debug["post_sizes"] = {
                    "defect": {
                        vname: [vmb.defect_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                        for vname in views_by_name
                    },
                    "correction": {
                        vname: [vmb.corr_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                        for vname in views_by_name
                    },
                }
                if not defect_write_debug.get("write_results"):
                    defect_write_debug["write_results"] = {
                        "defect": update_stats_all.get("defect", {}),
                        "correction": update_stats_all.get("correction", {}),
                    }
                print(
                    f"[DEFECT_WRITE_DEBUG] t={t:03d} cls={class_name} outcome={outcome} action={defect_write_debug['action']} "
                    f"allow_updates={int(allow_updates)} allow_bank_updates={int(allow_bank_updates)} suspicious={int(suspicious)} "
                    f"sel_total={int(defect_write_debug.get('selected_patch_total', 0))} defect_pre={defect_write_debug['pre_sizes']['defect']} defect_post={defect_write_debug['post_sizes']['defect']}"
                )

            # store RB card
            card = {
                "t": t,
                "class": class_name,
                "path": sample.path,
                "mode": mode,
                "score_img": float(score_img),
                "p_defect": float(p_defect),
                "pred": int(pred_defect),
                "y_true": int(y_true),
                "queried": True,
                "query_policy": q_policy,
                "regime": regime.value if isinstance(regime, AIFRegime) else str(regime),
                "novelty": float(novelty),
                "novelty_info": novelty_info,
                "rb_error_rate": float(rb_error),
                "outcome": outcome,
                "A_map_stats": A_map_stats,
                "region_stats": region_stats,
                "contam": {
                    "evidence": evidence,
                    "suspicious": bool(suspicious),
                    "inlier_mode": bool(inlier_mode),
                    "r_user": float(r_user),
                    "has_direct_feedback": bool(has_direct_feedback),
                    "allow_updates": bool(allow_updates),
                    "allow_bank_updates": bool(allow_bank_updates),
                    "allow_normal_writes": bool(allow_normal_writes),
                    "allow_defect_writes": bool(allow_defect_writes),
                    "allow_correction_writes": bool(allow_correction_writes),
                },
                "defect_write_debug": defect_write_debug,
                "aif": aif_info,
                "stats": {
                    "mu_N": stats.mu_N, "var_N": stats.var_N(), "n_N": stats.n_N,
                    "mu_A": stats.mu_A, "var_A": stats.var_A(), "n_A": stats.n_A,
                    "theta_dyn": stats.theta_dyn,
                    "theta": float(theta),
                    "sigmaN": float(th.sigmaN()),
                    "bufN_len": int(len(getattr(th, "bufN", []))),
                        "bufA_len": int(len(getattr(th, "bufA", []))),
                        "theta_sidecar_last_mode": str(getattr(th, "last_mode", "none")),
                        "theta_sidecar_last_move": float(getattr(th, "last_move", 0.0)),
                        "theta_sidecar_last_accept": bool(getattr(th, "last_accept", False)),
                    "dynonly_freeze_theta": bool(dynonly_freeze_theta),
                    "dynonly_freeze_normal_dyn": bool(dynonly_freeze_normal_dyn)
                },
                "bank_updates": {
                    "added_normal_dyn": bool(added_normal),
                    "added_defect": bool(added_defect),
                    "added_correction": bool(any(len(v.get("layers", {})) > 0 for v in update_stats_all.get("correction", {}).values() if isinstance(v, dict))),
                    "core_added": bool(core_added),
                    "core_update_stats": core_update_stats,
                    "update_stats": update_stats_all,
                },
                "bank_sizes": {
                    "normal_fixed": {
                        vname: [vmb.normal_fixed[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                        for vname in views_by_name
                    },
                    "normal_dyn": {
                        vname: [vmb.normal_dyn[class_name][mode][vname][l].size() for l in range(len(feats_id))]
                        for vname in views_by_name
                    },
                    "defect": {
                        vname: [vmb.defect_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                        for vname in views_by_name
                    },
                    "correction": {
                        vname: [vmb.corr_cb[class_name][vname][l].size() for l in range(len(feats_id))]
                        for vname in views_by_name
                    },
                    "use_defect_bank": bool(vmb.use_defect_bank),
                    "corr_enabled": bool(vmb.corr_cfg.enabled),
                },
            }
            rb.add_card(card)
        else:
            if ablation == "static_baseline":
                pass
            elif ablation == "correction_only":
                pass
            elif ablation == "normal_dyn_plus_correction":
                pass
            elif (ablation == "normal_dyn_only") and dynonly_freeze_theta:
                pass
            elif str(policy) == "selftrain_core":
                # The self-training comparator isolates unsupervised normal-memory
                # enrichment.  It intentionally does not use unlabeled scores to
                # update the threshold sidecar.
                pass
            else:
                # Only now, after deciding not to query, may a very confident self-labeled normal tighten theta.
                th.observe_unlabeled(score_img, p_defect, pred_defect)

            # Set-membership core-normal update (no human): enrich normal_dyn with typical predicted normals.
            if sm_cfg.enabled and (not pred_defect):
                core_selftrain_diag["pred_normal_seen"] = int(core_selftrain_diag.get("pred_normal_seen", 0)) + 1
                fs = core_sets.get(int(mode), None)
                if fs is not None and fs.accept(g, score=float(score_img), theta=float(theta), sigmaN=float(th.sigmaN()), p_defect=float(p_defect)):
                    core_selftrain_diag["global_accepts"] = int(core_selftrain_diag.get("global_accepts", 0)) + 1
                    n_patches = int(feats_id[0].shape[0])
                    layer_to_idx_core = vmb.select_core_patches(
                        cls_name=class_name,
                        feats_per_layer=view_to_feats["id"],
                        mode=mode,
                        view="id",
                        k_core=int(sm_cfg.core_patches_per_layer),
                    )
                    # apply across all views for view-consistent enrichment
                    total_added_core = 0
                    total_replaced_core = 0
                    for vname, vspec in views_by_name.items():
                        layer_to_idx_v = map_layer_patch_idx_to_view(layer_to_idx_core, vspec, n_patches)
                        us = vmb.update_from_topk(
                            cls_name=class_name,
                            feats_per_layer=view_to_feats[vname],
                            which="normal_dyn",
                            mode=mode,
                            view=vname,
                            layer_to_patch_idx=layer_to_idx_v,
                            tau_insert=float(sm_cfg.core_tau_insert),
                            tau_replace=float(sm_cfg.core_tau_replace),
                            event_id=int(global_t),
                            write_value=0.5,
                            outcome="SELFTRAIN_CORE",
                        )
                        core_update_stats[vname] = us
                        core_selftrain_diag["view_update_calls"] = int(core_selftrain_diag.get("view_update_calls", 0)) + 1
                        add_i, repl_i = _sum_core_update_stats(us)
                        total_added_core += int(add_i)
                        total_replaced_core += int(repl_i)
                    core_selftrain_diag["patch_update_events"] = int(core_selftrain_diag.get("patch_update_events", 0)) + 1
                    core_selftrain_diag["patch_added"] = int(core_selftrain_diag.get("patch_added", 0)) + int(total_added_core)
                    core_selftrain_diag["patch_replaced"] = int(core_selftrain_diag.get("patch_replaced", 0)) + int(total_replaced_core)
                    core_added = bool((int(total_added_core) + int(total_replaced_core)) > 0)

            if not store_only_queried_cards:
                rb.add_card({
                    "t": t, "class": class_name, "path": sample.path, "mode": mode,
                    "score_img": float(score_img), "p_defect": float(p_defect),
                    "pred": int(pred_defect), "y_true": int(y_true),
                    "queried": False, "query_policy": None,
                    "regime": regime.value if isinstance(regime, AIFRegime) else str(regime),
                    "outcome": outcome,
                    "core_added": bool(core_added),
                    "core_update_stats": core_update_stats,
                })

        _diag_finish_step(_diag_step_start, _diag_update_start, bool(queried))

        all_scores.append(float(score_img))
        all_labels.append(int(y_true))

        # progress print
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        qrate = n_queries / max(1, t)
        show_content = (
            f"[t={t:03d}] mode={mode} score={score_img:.4f} theta={theta:.4f} p={p_defect:.3f} pred={int(pred_defect)} "
            f"true={y_true} outcome={outcome} acc={acc:.3f} qrate={qrate:.3f} "
            f"regime={regime.value if policy != 'heuristic' else 'heuristic'} "
            f"action={action.value if policy != 'heuristic' else q_policy} "
            f"EFE_value={aif_info.get('G_val', None) if policy != 'heuristic' else None} "
        )
        print(show_content)
        record_handle.write(show_content + "\n")

    # final metrics
    total = tp + tn + fp + fn
    acc = (tp + tn) / max(1, total)
    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))
    auroc = auc_roc(all_scores, all_labels)
    qrate = n_queries / max(1, total)

    def _deployment_cost_summary() -> Dict[str, Any]:
        n_total_diag = max(1, int(deployment_cost.get("n_total", 0)))
        n_query_diag = max(1, int(deployment_cost.get("n_query", 0)))
        n_no_query_diag = max(1, int(deployment_cost.get("n_no_query", 0)))
        stage_sec = {str(k): float(v) for k, v in deployment_cost.get("stage_sec", {}).items()}
        stage_ms_per_image = {str(k): float(1000.0 * v / n_total_diag) for k, v in stage_sec.items()}
        out = {
            "enabled": bool(deployment_cost_enabled),
            "cuda_sync": bool(deployment_cost_cuda_sync),
            "n_total": int(deployment_cost.get("n_total", 0)),
            "n_query": int(deployment_cost.get("n_query", 0)),
            "n_no_query": int(deployment_cost.get("n_no_query", 0)),
            "total_wall_sec": float(deployment_cost.get("total_step_sec", 0.0)),
            "mean_ms_per_image": float(1000.0 * deployment_cost.get("total_step_sec", 0.0) / n_total_diag),
            "mean_ms_queried_image": float(1000.0 * deployment_cost.get("query_step_sec", 0.0) / n_query_diag) if int(deployment_cost.get("n_query", 0)) > 0 else None,
            "mean_ms_unqueried_image": float(1000.0 * deployment_cost.get("no_query_step_sec", 0.0) / n_no_query_diag) if int(deployment_cost.get("n_no_query", 0)) > 0 else None,
            "stage_sec": stage_sec,
            "stage_ms_per_image": stage_ms_per_image,
        }
        if torch.cuda.is_available():
            try:
                out["cuda_max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                out["cuda_max_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024.0 ** 2))
            except Exception as _e:
                out["cuda_memory_error"] = repr(_e)
        return out

    def _hidden_fn_summary() -> Dict[str, Any]:
        total_defects_h = max(1, int(hidden_fn_summary_work.get("total_defects", 0)))
        pred_normal_h = max(1, int(hidden_fn_summary_work.get("pred_normal_defects", 0)))
        hidden_h = max(1, int(hidden_fn_summary_work.get("hidden_unqueried_fns", 0)))
        recs = list(hidden_fn_records)
        out: Dict[str, Any] = dict(hidden_fn_summary_work)
        out["pred_normal_defect_rate_over_defects"] = float(hidden_fn_summary_work.get("pred_normal_defects", 0) / total_defects_h)
        out["hidden_unqueried_fn_rate_over_defects"] = float(hidden_fn_summary_work.get("hidden_unqueried_fns", 0) / total_defects_h)
        out["queried_fraction_among_pred_normal_defects"] = float(hidden_fn_summary_work.get("queried_pred_normal_defects", 0) / pred_normal_h)
        out["near_threshold_fraction_among_pred_normal_defects"] = float(hidden_fn_summary_work.get("near_threshold_pred_normal_defects", 0) / pred_normal_h)
        out["near_threshold_fraction_among_hidden_unqueried_fns"] = float(hidden_fn_summary_work.get("near_threshold_hidden_unqueried_fns", 0) / hidden_h)
        out["low_conf_fraction_among_hidden_unqueried_fns"] = float(hidden_fn_summary_work.get("low_conf_hidden_unqueried_fns", 0) / hidden_h)
        out["far_below_threshold_fraction_among_hidden_unqueried_fns"] = float(hidden_fn_summary_work.get("far_below_threshold_hidden_unqueried_fns", 0) / hidden_h)
        if recs:
            for key in ["theta_minus_score", "theta_minus_score_sigma", "p_defect", "novelty"]:
                vals = [float(r[key]) for r in recs if key in r and r[key] is not None]
                if vals:
                    arr = np.asarray(vals, dtype=np.float64)
                    out[f"{key}_mean"] = float(arr.mean())
                    out[f"{key}_median"] = float(np.median(arr))
                    out[f"{key}_min"] = float(arr.min())
                    out[f"{key}_max"] = float(arr.max())
            hidden_recs = [r for r in recs if not bool(r.get("queried", False))]
            top_by_p = sorted(hidden_recs, key=lambda r: float(r.get("p_defect", 0.0)), reverse=True)[:max(0, int(hidden_fn_topk))]
            near_recs = sorted([r for r in hidden_recs if bool(r.get("near_threshold", False))], key=lambda r: abs(float(r.get("theta_minus_score_sigma", 0.0))))[:max(0, int(hidden_fn_topk))]
            out["examples_top_p_hidden_unqueried"] = top_by_p
            out["examples_near_threshold_hidden_unqueried"] = near_recs
        else:
            out["examples_top_p_hidden_unqueried"] = []
            out["examples_near_threshold_hidden_unqueried"] = []
        return out

    summary = {
        "dataset": dataset,
        "class_name": class_name,
        "seed": int(seed),
        "query_seed": None if query_seed is None else int(query_seed),
        "support_seed": None if support_seed is None else int(support_seed),
        "stream_seed": None if stream_seed is None else int(stream_seed),
        "modes": {"use_modes": use_modes, "K_modes": K_modes, "shots_per_mode": shots_per_mode},
        "support_aug": support_aug_cfg.__dict__,
        "views": [v.__dict__ for v in views],
        "max_samples": max_samples,
        "threshold_initialization": threshold_init_info,
        "support_selection": {k: v for k, v in support_selection_info.items() if k != "calibration_scores"},
        "params": {
            "K_normal_fixed": K_normal_fixed, "K_normal_dyn": K_normal_dyn, "K_normal_ltm": K_normal_ltm, "ltm_min_repeat": ltm_min_repeat,
            "K_defect": K_defect, "K_concept": K_concept,
            "knn_k_normal": knn_k_normal,
            "topk_patches": topk_patches,
            "score_mode": score_mode,
            "lam": lam, "tau_close": tau_close, "gamma": gamma,
            "fusion": {"layer_fusion": layer_fusion, "view_fusion": view_fusion},
            "img_agg": {"kind": img_agg, "topk": img_topk, "quantile": img_quantile},
            "tauN_insert": tauN_insert, "tauN_replace": tauN_replace,
            "tauD_insert": tauD_insert, "tauD_replace": tauD_replace,
            "gating": gating.__dict__,
            "query": qcfg.__dict__,
            "simple_aif": simple_aif_cfg.__dict__ if simple_aif_cfg is not None else {},
            "v22_counters": {"forced_warmup_queries": int(v22_forced_warmup_queries), "audit_queries": int(v22_audit_queries), "fn_audit_queries": int(v22_fn_audit_queries), "budget_suppressed_queries": int(v22_budget_suppressed_queries)},
            "threshold": thcfg.__dict__,
            "theta_controller_kind": str(theta_controller_kind),
            "theta_sidecar": (theta_sidecar_cfg.__dict__ if theta_sidecar_cfg is not None else {}),
            "correction": corr_cfg.__dict__,
            "set_membership": sm_cfg.__dict__,
            "use_defect_bank_effective": bool(vmb.use_defect_bank),
        },
        "selftrain_core_diagnostics": {
            **core_selftrain_diag,
            "feasible_sets": _core_feasible_set_diagnostics(),
        },
        "metrics": {
            "acc": acc,
            "FPR": fpr,
            "FNR": fnr,
            "qrate": qrate,
            "AUROC_score": auroc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        },
        "rb_cards": len(rb.cards),
    }

    if deployment_cost_enabled:
        summary["deployment_cost"] = _deployment_cost_summary()
    if hidden_fn_enabled:
        summary["hidden_fn_analysis"] = _hidden_fn_summary()
    
    record_handle.close()

    summary["record_path"] = record_txt_path
    summary["stream_slice"] = {
        "start_idx": None if stream_start_idx is None else int(stream_start_idx),
        "end_idx": None if stream_end_idx is None else int(stream_end_idx),
        "start_frac": None if stream_start_frac is None else float(stream_start_frac),
        "end_frac": None if stream_end_frac is None else float(stream_end_frac),
        "resume_step_offset": int(resume_step_offset),
        "no_shuffle_test": bool(no_shuffle_test),
        "stream_seed": None if stream_seed is None else int(stream_seed),
    }
    if save_temporal_artifacts_flag:
        summary["temporal"] = save_temporal_artifacts_for_run(
            record_path=record_txt_path,
            out_json=out_json,
            class_name=class_name,
            policy=str(policy),
            later_frac=float(temporal_later_frac),
            early_n=int(temporal_first_n),
        )

    if bool(qual_corr_dump_dir):
        summary["correction_qualitative"] = save_correction_qualitative_outputs(
            qual_corr_dump_dir=qual_corr_dump_dir,
            class_name=class_name,
            qual_corr_records=qual_corr_records,
            vmb=vmb,
        )

    try:
        summary["memory"] = vmb.memory_summary(class_name)
        if deployment_cost_enabled and isinstance(summary.get("deployment_cost"), dict):
            mem = summary.get("memory", {})
            summary["deployment_cost"]["memory_counts"] = {
                "normal_dyn_stm_total": int(mem.get("normal_dyn_stm_total", 0)) if isinstance(mem, dict) else 0,
                "normal_dyn_ltm_total": int(mem.get("normal_dyn_ltm_total", 0)) if isinstance(mem, dict) else 0,
                "maturation_candidate_total": int(mem.get("maturation_candidate_total", 0)) if isinstance(mem, dict) else 0,
                "correction_total": int(mem.get("correction_bank", {}).get("total", 0)) if isinstance(mem, dict) and isinstance(mem.get("correction_bank", {}), dict) else 0,
            }
    except Exception as _e:
        summary["memory_error"] = str(_e)

    resume_state_out = None
    if save_state_path is not None or return_state:
        resume_state_out = build_resume_state(
            class_name=class_name,
            vmb=vmb,
            rb=rb,
            stats_obj=stats,
            th=th,
            coverage_mem=coverage_mem,
            simple_aif_state=simple_aif_state,
            qs=qs,
            global_steps=int(resume_step_offset + total),
            policy=str(policy),
            ablation=str(ablation),
        )
    if save_state_path is not None and resume_state_out is not None:
        safe_makedirs(save_state_path)
        torch.save(resume_state_out, save_state_path)
        summary["state_path"] = save_state_path

    rb.save_json(out_json)
    summary_path = os.path.splitext(out_json)[0] + "_summary.json"
    summary["summary_path"] = summary_path
    safe_makedirs(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== PHASE-3 v21 SUMMARY ===")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"RB cards saved to: {out_json}")
    print(f"Summary saved to: {summary_path}")
    if summary.get("temporal"):
        print(f"Temporal artifacts saved to: {summary['temporal'].get('temporal_json')}")
    if summary.get("correction_qualitative"):
        print(f"Correction qualitative artifacts saved to: {summary['correction_qualitative'].get('qual_corr_dump_dir')}")
    if summary.get("state_path"):
        print(f"State checkpoint saved to: {summary['state_path']}")

    if return_state and resume_state_out is not None:
        summary["_resume_state"] = resume_state_out
    return summary




# ---------------------------------------------------------------------------
#  Batch-mode helpers for final-submission experiments
# ---------------------------------------------------------------------------

def parse_batch_class_list(s: Optional[str]) -> List[str]:
    """Parse comma/list-form class names for batch mode.

    Accepts strings like:
      "audiojack,bottle_cap,*end_cap"
      "[audiojack, bottle_cap, *end_cap]"
    Leading '*' markers are treated as human notes and stripped, so *end_cap
    becomes end_cap.
    """
    if s is None:
        return []
    text = str(s).strip()
    if not text:
        return []
    text = text.replace("\n", ",").replace(";", ",")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("'", "").replace('"', "")
    out: List[str] = []
    for raw in text.split(','):
        x = raw.strip()
        if not x:
            continue
        while x.startswith('*'):
            x = x[1:].strip()
        if x:
            out.append(x)
    # Keep user order but remove duplicates.
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def parse_int_list_loose(s: Optional[str]) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    text = str(s).strip().replace("[", "").replace("]", "").replace(";", ",")
    vals: List[int] = []
    for raw in text.split(','):
        x = raw.strip()
        if x:
            vals.append(int(x))
    return vals


def _support_stream_seed_token(name: str, value: Optional[int]) -> str:
    """Filename token for support/stream robustness sweeps."""
    if value is None:
        return ""
    return f"_{name}{int(value)}"


def _sample_paths_deterministic_or_seeded(paths: List[str], n: int, seed_value: Optional[int]) -> List[str]:
    """Return deterministic first-n paths unless a support seed is explicitly supplied.

    This preserves the legacy v21 behavior when --support_seed is omitted, while
    enabling true repeated-support-draw experiments when --support_seed is set.
    """
    paths = list(paths)
    n = max(1, int(n))
    if len(paths) <= n:
        return paths
    if seed_value is None:
        return paths[:n]
    rng = random.Random(int(seed_value))
    idxs = list(range(len(paths)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:n])
    return [paths[i] for i in chosen]


def _safe_filename_token(s: Any) -> str:
    token = str(s).strip().replace(os.sep, "_")
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    token = token.strip("_")
    return token or "run"


def _qrate_tag_for_policy(policy: str, simple_cfg: "SimpleAIFConfig", args: argparse.Namespace) -> str:
    pol = str(policy).lower().strip()
    if pol in QUERY_BASELINE_POLICIES:
        q = float(simple_cfg.target_qrate)
        return f"q{int(round(q * 1000)):03d}"
    if pol == "simple_aif":
        q = float(args.simple_aif_target_qrate)
        return f"q{int(round(q * 1000)):03d}"
    if pol == "selftrain_core":
        return "q000"
    if pol in {"hardcode", "hardcode_band"}:
        lo = int(round(float(args.ambig_low) * 100))
        hi = int(round(float(args.ambig_high) * 100))
        return f"band{lo:02d}_{hi:02d}"
    return "run"


class _Tee:
    """Write stream output to both console and a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for st in self.streams:
            try:
                st.write(data)
                st.flush()
            except Exception:
                pass
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def _query_seeds_for_batch_policy(policy: str, args: argparse.Namespace) -> List[Optional[int]]:
    pol = str(policy).lower().strip()
    if pol == "random":
        explicit = parse_int_list_loose(getattr(args, "batch_query_seeds", ""))
        if explicit:
            return [int(x) for x in explicit]
        n = int(max(1, getattr(args, "batch_num_query_seeds", 3)))
        base = int(getattr(args, "batch_query_seed_base", 0))
        return [base + i for i in range(n)]
    # Non-random policies are deterministic under fixed support/stream seed.
    qseed = getattr(args, "query_seed", None)
    return [None if qseed is None else int(qseed)]


def _support_seeds_for_batch(args: argparse.Namespace) -> List[Optional[int]]:
    explicit = parse_int_list_loose(getattr(args, "batch_support_seeds", ""))
    if explicit:
        return [int(x) for x in explicit]
    n = int(max(1, getattr(args, "batch_num_support_seeds", 1)))
    if n > 1:
        base = int(getattr(args, "batch_support_seed_base", 0))
        return [base + i for i in range(n)]
    sseed = getattr(args, "support_seed", None)
    return [None if sseed is None else int(sseed)]


def _stream_seeds_for_batch(args: argparse.Namespace) -> List[Optional[int]]:
    explicit = parse_int_list_loose(getattr(args, "batch_stream_seeds", ""))
    if explicit:
        return [int(x) for x in explicit]
    n = int(max(1, getattr(args, "batch_num_stream_seeds", 1)))
    if n > 1:
        base = int(getattr(args, "batch_stream_seed_base", 0))
        return [base + i for i in range(n)]
    sseed = getattr(args, "stream_seed", None)
    return [None if sseed is None else int(sseed)]


def _batch_output_path(out_dir: str, cls_name: str, policy: str, qtag: str, run_tag: str, qseed: Optional[int], support_seed: Optional[int] = None, stream_seed: Optional[int] = None) -> str:
    cls_tok = _safe_filename_token(cls_name)
    pol_tok = _safe_filename_token(policy)
    run_tok = _safe_filename_token(run_tag)
    seed_part = "" if qseed is None else f"_qseed{int(qseed)}"
    seed_part += _support_stream_seed_token("sseed", support_seed)
    seed_part += _support_stream_seed_token("stream", stream_seed)
    filename = f"{cls_tok}_{run_tok}_{pol_tok}_{qtag}{seed_part}.json"
    return os.path.join(out_dir, filename)


def run_batch_experiment(
    *,
    args: argparse.Namespace,
    base_run_kwargs: Dict[str, Any],
    simple_aif_cfg: "SimpleAIFConfig",
) -> Dict[str, Any]:
    """Run one or more policies over a user-provided class list.

    This is intended for final-submission sweeps where repeatedly typing one
    command per class/policy is error-prone.  It writes each run's normal JSON
    artifacts plus a captured console log (.txt) into --batch_out_dir, and also
    writes a batch-level summary JSON.
    """
    classes = parse_batch_class_list(getattr(args, "batch_classes", ""))
    if not classes:
        raise ValueError("--batch_classes is required in --experiment_mode batch")
    out_dir = str(getattr(args, "batch_out_dir", "")).strip()
    if not out_dir:
        # Fall back to the directory of --out_json, but explicit --batch_out_dir
        # is strongly preferred for large final sweeps.
        out_dir = os.path.dirname(str(getattr(args, "out_json", ""))) or "."
    os.makedirs(out_dir, exist_ok=True)

    policies = parse_comma_list(getattr(args, "batch_policies", ""))
    if not policies:
        policies = [str(getattr(args, "policy", "simple_aif"))]
    policies = [str(p).strip().lower() for p in policies if str(p).strip()]

    batch = {
        "mode": "batch",
        "classes": classes,
        "policies": policies,
        "batch_out_dir": out_dir,
        "base_seed": int(getattr(args, "seed", 0)),
        "support_seed_arg": None if getattr(args, "support_seed", None) is None else int(getattr(args, "support_seed")),
        "stream_seed_arg": None if getattr(args, "stream_seed", None) is None else int(getattr(args, "stream_seed")),
        "query_seed_arg": None if getattr(args, "query_seed", None) is None else int(getattr(args, "query_seed")),
        "runs": [],
        "aggregate": {},
    }

    support_seeds = _support_seeds_for_batch(args)
    stream_seeds = _stream_seeds_for_batch(args)
    batch["support_seeds"] = support_seeds
    batch["stream_seeds"] = stream_seeds

    for pol in policies:
        qtag = _qrate_tag_for_policy(pol, simple_aif_cfg, args)
        qseeds = _query_seeds_for_batch_policy(pol, args)
        for cls_name in classes:
            for support_seed_i in support_seeds:
                for stream_seed_i in stream_seeds:
                    for qseed in qseeds:
                        out_json = _batch_output_path(
                            out_dir=out_dir,
                            cls_name=cls_name,
                            policy=pol,
                            qtag=qtag,
                            run_tag=str(getattr(args, "batch_run_tag", "E01_v21")),
                            qseed=qseed,
                            support_seed=support_seed_i,
                            stream_seed=stream_seed_i,
                        )
                        log_path = os.path.splitext(out_json)[0] + ".txt"
                        run_rec = {
                            "class_name": cls_name,
                            "policy": pol,
                            "query_seed": qseed,
                            "support_seed": support_seed_i,
                            "stream_seed": stream_seed_i,
                            "out_json": out_json,
                            "log_path": log_path,
                            "status": "pending",
                        }
                        print(f"\n[batch] START class={cls_name} policy={pol} support_seed={support_seed_i} stream_seed={stream_seed_i} query_seed={qseed} -> {out_json}")
                        try:
                            kwargs = dict(base_run_kwargs)
                            kwargs["policy"] = pol
                            kwargs["query_seed"] = qseed
                            kwargs["support_seed"] = support_seed_i
                            kwargs["stream_seed"] = stream_seed_i
                            # The user-visible --class_name / --out_json are ignored in
                            # batch mode; class/output are generated per run.
                            with open(log_path, "w", encoding="utf-8") as log_f:
                                tee_out = _Tee(sys.__stdout__, log_f)
                                tee_err = _Tee(sys.__stderr__, log_f)
                                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                                    print(f"[batch] class={cls_name} policy={pol} support_seed={support_seed_i} stream_seed={stream_seed_i} query_seed={qseed}")
                                    print(f"[batch] out_json={out_json}")
                                    summary = run_phase3_streaming_aif(
                                        class_name=cls_name,
                                        out_json=out_json,
                                        **kwargs,
                                    )
                            run_rec["status"] = "ok"
                            run_rec["metrics"] = summary.get("metrics", {})
                            if isinstance(summary.get("deployment_cost"), dict):
                                run_rec["deployment_cost"] = summary.get("deployment_cost")
                            if isinstance(summary.get("hidden_fn_analysis"), dict):
                                run_rec["hidden_fn_analysis"] = summary.get("hidden_fn_analysis")
                            if isinstance(summary.get("selftrain_core_diagnostics"), dict):
                                run_rec["selftrain_core_diagnostics"] = summary.get("selftrain_core_diagnostics")
                            if isinstance(summary.get("memory"), dict):
                                run_rec["memory"] = summary.get("memory")
                            run_rec["summary_path"] = summary.get("summary_path")
                            run_rec["record_path"] = summary.get("record_path")
                            print(f"[batch] DONE  class={cls_name} policy={pol} support_seed={support_seed_i} stream_seed={stream_seed_i} query_seed={qseed}")
                        except Exception as e:
                            run_rec["status"] = "error"
                            run_rec["error"] = repr(e)
                            run_rec["traceback"] = traceback.format_exc()
                            # Ensure the traceback is also available in the run log.
                            try:
                                with open(log_path, "a", encoding="utf-8") as log_f:
                                    log_f.write("\n[batch-error]\n")
                                    log_f.write(run_rec["traceback"])
                            except Exception:
                                pass
                            print(f"[batch] ERROR class={cls_name} policy={pol} support_seed={support_seed_i} stream_seed={stream_seed_i} query_seed={qseed}: {e}")
                            if bool(getattr(args, "batch_fail_fast", False)):
                                batch["runs"].append(run_rec)
                                raise
                        batch["runs"].append(run_rec)

                        # Write an incremental summary so long batches survive interruption.
                        batch_path_tmp = os.path.join(out_dir, f"{_safe_filename_token(getattr(args, 'batch_run_tag', 'E01_v21'))}_batch_summary_partial.json")
                        try:
                            with open(batch_path_tmp, "w", encoding="utf-8") as f:
                                json.dump(batch, f, indent=2, ensure_ascii=False)
                        except Exception:
                            pass

    # Aggregate successful metrics by policy.
    metric_keys = ["acc", "AUROC_score", "qrate", "FPR", "FNR", "tp", "tn", "fp", "fn"]
    for pol in policies:
        rows = [r for r in batch["runs"] if r.get("status") == "ok" and r.get("policy") == pol and isinstance(r.get("metrics"), dict)]
        agg: Dict[str, Any] = {"n_ok": len(rows)}
        for k in metric_keys:
            vals = [float(r["metrics"][k]) for r in rows if k in r.get("metrics", {}) and r["metrics"][k] is not None]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                agg[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "min": float(arr.min()), "max": float(arr.max())}
        # Optional deployment-cost aggregation.
        cost_rows = [r.get("deployment_cost", {}) for r in rows if isinstance(r.get("deployment_cost"), dict)]
        if cost_rows:
            cost_agg: Dict[str, Any] = {"n_ok": len(cost_rows)}
            for k in ["mean_ms_per_image", "mean_ms_queried_image", "mean_ms_unqueried_image", "cuda_max_memory_allocated_mb", "cuda_max_memory_reserved_mb"]:
                vals = [float(x[k]) for x in cost_rows if x.get(k) is not None]
                if vals:
                    arr = np.asarray(vals, dtype=np.float64)
                    cost_agg[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "min": float(arr.min()), "max": float(arr.max())}
            stage_keys = set()
            for x in cost_rows:
                if isinstance(x.get("stage_ms_per_image"), dict):
                    stage_keys.update(str(k) for k in x["stage_ms_per_image"].keys())
            if stage_keys:
                cost_agg["stage_ms_per_image"] = {}
                for sk in sorted(stage_keys):
                    vals = [float(x.get("stage_ms_per_image", {}).get(sk)) for x in cost_rows if x.get("stage_ms_per_image", {}).get(sk) is not None]
                    if vals:
                        arr = np.asarray(vals, dtype=np.float64)
                        cost_agg["stage_ms_per_image"][sk] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "min": float(arr.min()), "max": float(arr.max())}
            agg["deployment_cost"] = cost_agg

        # Optional hidden-FN aggregation.
        hfn_rows = [r.get("hidden_fn_analysis", {}) for r in rows if isinstance(r.get("hidden_fn_analysis"), dict)]
        if hfn_rows:
            total_defects_h = int(sum(int(x.get("total_defects", 0)) for x in hfn_rows))
            pred_normal_h = int(sum(int(x.get("pred_normal_defects", 0)) for x in hfn_rows))
            hidden_h = int(sum(int(x.get("hidden_unqueried_fns", 0)) for x in hfn_rows))
            queried_h = int(sum(int(x.get("queried_pred_normal_defects", 0)) for x in hfn_rows))
            near_pred_h = int(sum(int(x.get("near_threshold_pred_normal_defects", 0)) for x in hfn_rows))
            near_hidden_h = int(sum(int(x.get("near_threshold_hidden_unqueried_fns", 0)) for x in hfn_rows))
            low_conf_hidden_h = int(sum(int(x.get("low_conf_hidden_unqueried_fns", 0)) for x in hfn_rows))
            far_hidden_h = int(sum(int(x.get("far_below_threshold_hidden_unqueried_fns", 0)) for x in hfn_rows))
            hfn_agg = {
                "n_ok": len(hfn_rows),
                "total_defects": total_defects_h,
                "pred_normal_defects": pred_normal_h,
                "hidden_unqueried_fns": hidden_h,
                "queried_pred_normal_defects": queried_h,
                "near_threshold_pred_normal_defects": near_pred_h,
                "near_threshold_hidden_unqueried_fns": near_hidden_h,
                "low_conf_hidden_unqueried_fns": low_conf_hidden_h,
                "far_below_threshold_hidden_unqueried_fns": far_hidden_h,
                "pred_normal_defect_rate_over_defects": float(pred_normal_h / max(1, total_defects_h)),
                "hidden_unqueried_fn_rate_over_defects": float(hidden_h / max(1, total_defects_h)),
                "queried_fraction_among_pred_normal_defects": float(queried_h / max(1, pred_normal_h)),
                "near_threshold_fraction_among_pred_normal_defects": float(near_pred_h / max(1, pred_normal_h)),
                "near_threshold_fraction_among_hidden_unqueried_fns": float(near_hidden_h / max(1, hidden_h)),
                "low_conf_fraction_among_hidden_unqueried_fns": float(low_conf_hidden_h / max(1, hidden_h)),
                "far_below_threshold_fraction_among_hidden_unqueried_fns": float(far_hidden_h / max(1, hidden_h)),
            }
            agg["hidden_fn_analysis"] = hfn_agg

        # Optional selftrain_core diagnostics aggregation.
        st_rows = [r.get("selftrain_core_diagnostics", {}) for r in rows if isinstance(r.get("selftrain_core_diagnostics"), dict)]
        if st_rows:
            st_agg: Dict[str, Any] = {"n_ok": len(st_rows)}
            for k in ["pred_normal_seen", "global_accepts", "patch_update_events", "patch_added", "patch_replaced", "view_update_calls"]:
                st_agg[k] = int(sum(int(x.get(k, 0)) for x in st_rows))
            st_agg["global_accept_rate_over_pred_normal"] = float(st_agg.get("global_accepts", 0) / max(1, st_agg.get("pred_normal_seen", 0)))
            # Also aggregate final memory sizes if available in run records.
            mem_rows = [r.get("memory", {}) for r in rows if isinstance(r.get("memory"), dict)]
            if mem_rows:
                st_agg["final_memory_counts"] = {
                    "normal_dyn_stm_total_sum": int(sum(int(x.get("normal_dyn_stm_total", 0)) for x in mem_rows)),
                    "normal_dyn_ltm_total_sum": int(sum(int(x.get("normal_dyn_ltm_total", 0)) for x in mem_rows)),
                    "maturation_candidate_total_sum": int(sum(int(x.get("maturation_candidate_total", 0)) for x in mem_rows)),
                }
            agg["selftrain_core_diagnostics"] = st_agg

        batch["aggregate"][pol] = agg

    batch_path = os.path.join(out_dir, f"{_safe_filename_token(getattr(args, 'batch_run_tag', 'E01_v21'))}_batch_summary.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)
    batch["batch_summary_path"] = batch_path
    print(f"\n[batch] Summary saved to: {batch_path}")
    print(json.dumps(batch.get("aggregate", {}), indent=2))
    return batch

# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase-3 online memory-governed anomaly detection: BEA/simple-AIF querying, correction bank, set-membership core normals, and self-training comparator"
    )
    parser.add_argument("--dataset", type=str, required=True, help="mvtec, visa, or realiad")
    parser.add_argument("--root", type=str, required=True, help="dataset root")
    parser.add_argument("--class_name", type=str, default="", help="If empty, evaluate all classes in the dataset")
    parser.add_argument("--csv", type=str, default=None, help="VisA csv path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_fuse", type=str, default="concat", choices=["concat","sum","none"], help="VisionAD-style token-level feature fusion before NN search")
    parser.add_argument("--ablation", type=str, default="full",
                    choices=["full", "static_baseline", "normal_dyn_only", "correction_only", "normal_dyn_plus_correction"])
    parser.add_argument("--baseline_compat", action="store_true")
    parser.add_argument("--support_include_identity", action="store_true")
    parser.add_argument("--dynonly_freeze_theta", action="store_true",
                    help="For --ablation normal_dyn_only: disable theta updates from both queried and unqueried samples")
    parser.add_argument("--dynonly_freeze_normal_dyn", action="store_true",
                    help="For --ablation normal_dyn_only: disable normal_dyn memory writes while keeping theta adaptation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query_seed", type=int, default=None,
                        help="Seed used only by independent query baselines that need randomness (random, and cold-start scoring baselines). Keeps support/stream seed fixed.")
    parser.add_argument("--support_seed", type=int, default=None,
                        help="When set, randomly draws the initial few-shot support images using this seed. Omit to preserve the legacy deterministic support selection.")
    parser.add_argument("--support_select", type=str, default="legacy", choices=["legacy", "coverage_tail"],
                        help="Support selection strategy. coverage_tail chooses few-shot normal supports by normal-train tail validation before fixed-bank construction.")
    parser.add_argument("--support_select_num_candidates", type=int, default=8,
                        help="Number of candidate support sets evaluated by --support_select coverage_tail.")
    parser.add_argument("--support_select_calib_max", type=int, default=256,
                        help="Max normal-train images used to validate each coverage_tail candidate.")
    parser.add_argument("--support_select_inlier_q", type=float, default=0.95,
                        help="Global-feature inlier quantile for density-filtered k-center support selection.")
    parser.add_argument("--support_select_tail_q", type=float, default=0.95,
                        help="Normal-score tail quantile minimized by coverage_tail candidate selection.")
    parser.add_argument("--support_select_tail_q_hi", type=float, default=0.99,
                        help="Higher tail quantile used to penalize long normal-score tails.")
    parser.add_argument("--support_select_theta_q", type=float, default=0.925,
                        help="Normal calibration quantile used to initialize theta after coverage_tail support selection.")
    parser.add_argument("--support_select_lambda_tail_gap", type=float, default=0.50)
    parser.add_argument("--support_select_lambda_std", type=float, default=0.05)
    parser.add_argument("--support_select_lambda_outlier", type=float, default=0.20)
    parser.add_argument("--support_select_cache_dir", type=str, default=None,
                        help="Optional diagnostics/cache root. A per-run subfolder is cleared automatically before each run.")
    parser.add_argument("--stream_seed", type=int, default=None,
                        help="When set, shuffles the test stream with this seed. Omit to preserve the legacy global-seed stream shuffle.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_json", type=str, default="rb_phase3_sm.json")

    # Mode routing (VisionAD clustered normal modes)
    parser.add_argument("--use_modes", action="store_true", help="Enable CLS-space k-means modes (recommended)")
    parser.add_argument("--K_modes", type=int, default=3)
    parser.add_argument("--shots_per_mode", type=int, default=4)

    # Support augmentation
    parser.add_argument("--aug_per_image", type=int, default=4)
    parser.add_argument("--translate", type=float, default=0.08)
    parser.add_argument("--hflip_p", type=float, default=0.5)
    parser.add_argument("--rotation_deg", type=float, default=15.0, help="Support rotation degrees (0 disables)")
    parser.add_argument("--rotation_p", type=float, default=0.7, help="Prob to apply support rotation")

    # Banks
    parser.add_argument("--K_normal_fixed", type=int, default=4096)
    parser.add_argument("--K_normal_dyn", type=int, default=1024)
    parser.add_argument("--K_normal_ltm", type=int, default=512)
    parser.add_argument("--ltm_min_stm_size", type=int, default=16)
    parser.add_argument("--ltm_min_repeat", type=int, default=2)
    parser.add_argument("--K_defect", type=int, default=256)
    parser.add_argument("--K_concept", type=int, default=4096)
    parser.add_argument("--knn_k_normal", type=int, default=5)

    # Correction bank (boundary refinement)
    parser.add_argument("--no_correction_bank", action="store_true", help="Disable correction bank boundary refinement")
    parser.add_argument("--K_correction", type=int, default=256)
    parser.add_argument("--corr_weight", type=float, default=1.0)
    parser.add_argument("--beta_max", type=float, default=0.20)
    # v21 correction-bank write/retention controls. Defaults are enabled and conservative.
    parser.add_argument("--disable_sign_aware_correction_write", action="store_true", help="Disable v21 outcome-aware correction write strength")
    parser.add_argument("--disable_patch_weighted_correction_beta", action="store_true", help="Disable v21 patch-local beta weighting")
    parser.add_argument("--disable_sign_aware_correction_retention", action="store_true", help="Disable v21 sign-aware correction-bank replacement")
    parser.add_argument("--corr_outcome_weight_fp", type=float, default=1.30)
    parser.add_argument("--corr_outcome_weight_fn", type=float, default=1.20)
    parser.add_argument("--corr_outcome_weight_tn", type=float, default=0.60)
    parser.add_argument("--corr_outcome_weight_tp", type=float, default=0.80)
    parser.add_argument("--corr_patch_beta_floor", type=float, default=0.75)
    parser.add_argument("--corr_patch_beta_cap", type=float, default=1.25)
    parser.add_argument("--corr_patch_beta_power", type=float, default=1.0)
    parser.add_argument("--corr_replace_margin", type=float, default=0.0)
    parser.add_argument("--keep_defect_bank", action="store_true", help="Keep defect bank even when correction bank is enabled")
    parser.add_argument("--use_defect_bank", action="store_true", help="Force defect bank scoring/updates (overrides correction default)")
    parser.add_argument("--disable_defect_bank", action="store_true", help="Force-disable defect bank for clean baseline/ablation runs")

    # Set-membership core-normal self-labeling (conservative)
    parser.add_argument("--no_set_membership", action="store_true", help="Disable core-normal self-label enrichment")
    parser.add_argument("--core_p_defect_max", type=float, default=0.30)
    parser.add_argument("--core_margin_sigma", type=float, default=0.5)
    parser.add_argument("--core_radius_slack", type=float, default=2.0)
    parser.add_argument("--core_global_novelty_min", type=float, default=0.005)
    parser.add_argument("--core_max_buf", type=int, default=256)
    parser.add_argument("--core_patches_per_layer", type=int, default=16)
    parser.add_argument("--core_tau_insert", type=float, default=0.25)
    parser.add_argument("--core_tau_replace", type=float, default=0.40)

    # Scoring
    parser.add_argument("--topk_patches", type=int, default=32)
    parser.add_argument("--score_mode", type=str, default="boost", choices=["boost", "contrast"])
    parser.add_argument("--lam", type=float, default=0.8)
    parser.add_argument("--tau_close", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=1.0)

    # Fusion + aggregation
    parser.add_argument("--layer_fusion", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--view_fusion", type=str, default="mean", choices=["sum", "mean", "max"])
    parser.add_argument("--img_agg", type=str, default="topk_mean", choices=["topk_mean", "spiky_quantile", "max"])
    parser.add_argument("--img_topk", type=int, default=64)
    parser.add_argument("--img_quantile", type=float, default=0.95)

    # Updates
    parser.add_argument("--tauN_insert", type=float, default=0.15)
    parser.add_argument("--tauN_replace", type=float, default=0.25)
    parser.add_argument("--tauD_insert", type=float, default=0.10)
    parser.add_argument("--tauD_replace", type=float, default=0.20)

    # RB gating
    parser.add_argument("--min_count", type=int, default=3)
    parser.add_argument("--min_fp_rate", type=float, default=0.6)
    parser.add_argument("--min_fn_rate", type=float, default=0.6)

    # Querying
    parser.add_argument("--ambig_low", type=float, default=0.4)
    parser.add_argument("--ambig_high", type=float, default=0.6)
    parser.add_argument("--warmup_queries", type=int, default=15)
    parser.add_argument("--warmup_steps", type=int, default=25)
    parser.add_argument("--topq_quantile", type=float, default=0.90)
    parser.add_argument("--store_all_cards", action="store_true")
    parser.add_argument("--init_dump_json", type=str, default=None,
                        help="Optional JSON path to dump deterministic initialization artifacts")

    # Thresholding (v4)
    parser.add_argument("--use_global_threshold_initialization", "--use_global_threshold_init", dest="use_global_threshold_initialization", action="store_true",
                        help="Initialize theta from scalar scores over all normal training images; fixed visual memory remains few-shot support-only.")
    parser.add_argument("--global_threshold_init_mode", type=str, default="zscore_gaussian", choices=["zscore_gaussian", "raw_quantile"],
                        help="Global normal-threshold initializer: zscore_gaussian uses mean+std*Phi^{-1}(1-target_fpr); raw_quantile reproduces the previous empirical normal quantile behavior.")
    parser.add_argument("--target_fpr", type=float, default=0.05)
    parser.add_argument("--theta_min_bufN", type=int, default=30)
    parser.add_argument("--theta_bufN_max", type=int, default=500)
    parser.add_argument("--p_low", type=float, default=0.05)
    parser.add_argument("--z_prior_anom", type=float, default=5.0)
    parser.add_argument("--posterior_slope_c", type=float, default=1.0)
    parser.add_argument("--theta_controller", type=str, default="guarded_dual_anchor", choices=["legacy_normal_only", "guarded_dual_anchor"],
                        help="Threshold controller: legacy normal-only quantile controller, or decoupled guarded dual-anchor sidecar.")
    parser.add_argument("--theta_sidecar_normal_capacity", type=int, default=512)
    parser.add_argument("--theta_sidecar_defect_capacity", type=int, default=512)
    parser.add_argument("--theta_sidecar_recent_capacity", type=int, default=96)
    parser.add_argument("--theta_sidecar_min_normal_anchor", type=int, default=32)
    parser.add_argument("--theta_sidecar_min_defect_anchor", type=int, default=16)
    parser.add_argument("--theta_sidecar_min_recent_total", type=int, default=24)
    parser.add_argument("--theta_sidecar_min_recent_per_class", type=int, default=6)
    parser.add_argument("--theta_sidecar_update_every_labeled", type=int, default=8)
    parser.add_argument("--theta_sidecar_q_normal", type=float, default=0.95)
    parser.add_argument("--theta_sidecar_q_defect", type=float, default=0.10)
    parser.add_argument("--theta_sidecar_delta_normal", type=float, default=0.005)
    parser.add_argument("--theta_sidecar_delta_defect", type=float, default=0.005)
    parser.add_argument("--theta_sidecar_sep_margin", type=float, default=0.02)
    parser.add_argument("--theta_sidecar_lambda_fp", type=float, default=1.0)
    parser.add_argument("--theta_sidecar_lambda_fn", type=float, default=1.0)
    parser.add_argument("--theta_sidecar_lambda_move", type=float, default=0.2)
    parser.add_argument("--theta_sidecar_accept_margin", type=float, default=1e-4)
    parser.add_argument("--theta_sidecar_guard_delta_fp", type=float, default=0.01)
    parser.add_argument("--theta_sidecar_guard_delta_fn", type=float, default=0.01)
    parser.add_argument("--theta_sidecar_step_eta", type=float, default=0.3)
    parser.add_argument("--theta_sidecar_step_up", type=float, default=0.01)
    parser.add_argument("--theta_sidecar_step_down", type=float, default=0.01)
    parser.add_argument("--theta_sidecar_candidate_radius", type=float, default=0.02)
    parser.add_argument("--theta_sidecar_candidate_step", type=float, default=0.005)
    parser.add_argument("--theta_sidecar_no_support_seed", action="store_true")
    parser.add_argument("--theta_sidecar_disable_normal_fallback", action="store_true")
    parser.add_argument("--theta_sidecar_min_recent_normals_fallback", type=int, default=16)
    parser.add_argument("--theta_sidecar_min_defect_anchor_fallback", type=int, default=2)
    parser.add_argument("--theta_sidecar_normal_fallback_fpr_trigger", type=float, default=0.25)
    parser.add_argument("--theta_sidecar_normal_fallback_guard_delta_fn", type=float, default=0.01)

    # v22 support-robust cold-start and early-rescue controls. Defaults are disabled.
    parser.add_argument("--v22_enable_support_guard", action="store_true")
    parser.add_argument("--v22_support_sigma_floor", type=float, default=0.035)
    parser.add_argument("--v22_support_theta_z", type=float, default=2.5)
    parser.add_argument("--v22_support_cold_sigma_thr", type=float, default=0.020)
    parser.add_argument("--v22_enable_warmup_budget", action="store_true")
    parser.add_argument("--v22_warmup_steps", type=int, default=1000)
    parser.add_argument("--v22_warmup_qrate", type=float, default=0.06)
    parser.add_argument("--v22_disable_budget_guard", action="store_true")
    parser.add_argument("--v22_enable_defect_audit", action="store_true")
    parser.add_argument("--v22_audit_frac", type=float, default=0.25)
    parser.add_argument("--v22_audit_min_p", type=float, default=0.80)
    parser.add_argument("--v22_audit_score_sigma", type=float, default=1.0)
    parser.add_argument("--v22_enable_fn_audit", action="store_true",
                        help="Enable bounded FN audit for predicted-normal samples just below theta.")
    parser.add_argument("--v22_fn_audit_band", type=float, default=0.040)
    parser.add_argument("--v22_fn_audit_max_extra_qrate", type=float, default=0.005)
    parser.add_argument("--v22_fn_audit_min_gap", type=int, default=20)
    parser.add_argument("--v22_fn_audit_warmup_steps", type=int, default=1500)
    parser.add_argument("--v22_fn_audit_min_p", type=float, default=0.05)
    parser.add_argument("--v22_enable_fast_upward_rescue", action="store_true")
    parser.add_argument("--v22_enable_dual_anchor_balanced_rescue", action="store_true",
                        help="Enable guarded_dual_anchor_v2_balanced_rescue: two-sided threshold rescue with FN audit support.")
    parser.add_argument("--v22_fast_rescue_min_fp_normals", type=int, default=2)
    parser.add_argument("--v22_fast_rescue_fpr_trigger", type=float, default=0.35)
    parser.add_argument("--v22_fast_rescue_q_normal", type=float, default=0.95)
    parser.add_argument("--v22_fast_rescue_delta_normal", type=float, default=0.005)
    parser.add_argument("--v22_fast_rescue_step_up", type=float, default=0.030)
    parser.add_argument("--v22_fast_rescue_guard_delta_fn", type=float, default=0.020)
    parser.add_argument("--v22_fast_rescue_max_recent", type=int, default=64)
    parser.add_argument("--v22_dual_anchor_min_normals", type=int, default=8)
    parser.add_argument("--v22_dual_anchor_min_defects", type=int, default=3)
    parser.add_argument("--v22_dual_anchor_defect_q", type=float, default=0.15)
    parser.add_argument("--v22_dual_anchor_defect_margin", type=float, default=0.010)
    parser.add_argument("--v22_dual_anchor_normal_q", type=float, default=0.95)
    parser.add_argument("--v22_dual_anchor_up_trigger", type=float, default=0.35)
    parser.add_argument("--v22_dual_anchor_down_trigger", type=float, default=0.25)
    parser.add_argument("--v22_dual_anchor_exit_fpr", type=float, default=0.20)
    parser.add_argument("--v22_dual_anchor_exit_fnr", type=float, default=0.15)
    parser.add_argument("--v22_dual_anchor_anchor_gap", type=float, default=0.015)
    parser.add_argument("--v22_dual_anchor_up_defect_slack", type=float, default=0.020)
    parser.add_argument("--v22_dual_anchor_step_up", type=float, default=0.020)
    parser.add_argument("--v22_dual_anchor_step_down", type=float, default=0.020)
    parser.add_argument("--v22_dual_anchor_normal_step", type=float, default=0.005)
    parser.add_argument("--v22_dual_anchor_rescue_eta", type=float, default=1.0)
    parser.add_argument("--v22_dual_anchor_candidate_radius_normal", type=float, default=0.020)
    parser.add_argument("--v22_dual_anchor_candidate_radius_rescue", type=float, default=0.060)
    parser.add_argument("--v22_dual_anchor_candidate_step", type=float, default=0.005)
    parser.add_argument("--v22_dual_anchor_guard_delta_fp_normal", type=float, default=0.010)
    parser.add_argument("--v22_dual_anchor_guard_delta_fn_normal", type=float, default=0.010)
    parser.add_argument("--v22_dual_anchor_guard_delta_fn_up", type=float, default=0.050)
    parser.add_argument("--v22_dual_anchor_guard_delta_fp_down", type=float, default=0.200)
    parser.add_argument("--v22_dual_anchor_lambda_fp", type=float, default=1.0)
    parser.add_argument("--v22_dual_anchor_lambda_fn", type=float, default=1.5)
    parser.add_argument("--v22_dual_anchor_lambda_move", type=float, default=0.10)
    parser.add_argument("--v22_dual_anchor_lambda_q", type=float, default=0.10)
    parser.add_argument("--v22_dual_anchor_lambda_osc", type=float, default=0.05)
    parser.add_argument("--v22_dual_anchor_cooldown_steps", type=int, default=150)

    # Views: keep simple via flags (id is always included)
    parser.add_argument("--use_posclamp", action="store_true", help="Add posclamp view")
    parser.add_argument("--use_yflip", action="store_true", help="Add vertical flip view")
    parser.add_argument("--use_xflip", action="store_true", help="Add horizontal flip view")
    parser.add_argument("--posclamp_low", type=int, default=64)
    parser.add_argument("--no_rotations", action="store_true", help="Disable rotation pseudo-views")

    # -------- Memory control knobs --------
    parser.add_argument("--concept_seed_patches", type=int, default=2048,
                    help="Total target patches to seed concept codebook (upper bound).")
    parser.add_argument("--concept_per_img_cap", type=int, default=256,
                    help="Max patches sampled per support image (per augmentation) for concept seeding.")
    parser.add_argument("--fixed_per_img_cap", type=int, default=512,
                    help="Max patches sampled per support image (per augmentation) for fixed normal bank building.")
    parser.add_argument("--bank_dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                    help="Datatype for storing memory banks (fp16 saves VRAM). Similarity still computed in fp32.")
    parser.add_argument("--empty_cache_every", type=int, default=50,
                    help="Call torch.cuda.empty_cache() every N support images processed (0 disables).")
    
    

    # --- Phase-3 AIF policy ---
    parser.add_argument("--policy", type=str, default="simple_aif", choices=["simple_aif", "hardcode_band", "hardcode", "selftrain_core", "entropy", "margin", "novelty", "random", "periodic"],
                        help="Unified query policy: simple_aif, hardcode_band/hardcode, selftrain_core, or independent baselines entropy/margin/novelty/random/periodic. selftrain_core never queries and only uses conservative core-normal self-training.")
    parser.add_argument("--simple_aif_cost_fp", type=float, default=1.0)
    parser.add_argument("--simple_aif_cost_fn", type=float, default=2.5)
    parser.add_argument("--simple_aif_query_cost", type=float, default=0.8)
    parser.add_argument("--simple_aif_beta_entropy", type=float, default=0.3)
    parser.add_argument("--simple_aif_lambda_normal", type=float, default=0.35)
    parser.add_argument("--simple_aif_lambda_correction", type=float, default=0.15)
    parser.add_argument("--simple_aif_lambda_defect_side", type=float, default=0.35)
    parser.add_argument("--simple_aif_target_qrate", type=float, default=0.06)
    parser.add_argument("--query_baseline_target_qrate", type=float, default=None,
                        help="Optional target qrate override for entropy/margin/novelty/random/periodic baselines. If omitted, they use --simple_aif_target_qrate.")
    parser.add_argument("--simple_aif_query_cost_lr", type=float, default=0.02)
    parser.add_argument("--simple_aif_min_query_cost", type=float, default=0.05)
    parser.add_argument("--simple_aif_max_query_cost", type=float, default=3.0)
    parser.add_argument("--simple_aif_local_qrate_window", type=int, default=256)
    parser.add_argument("--simple_aif_no_adapt_query_cost", action="store_true")
    parser.add_argument("--simple_aif_use_local_saliency", action="store_true")
    parser.add_argument("--simple_aif_saliency_weight", type=float, default=1.0)
    parser.add_argument("--simple_aif_correction_gap_scale", type=float, default=0.05)
    parser.add_argument("--simple_aif_defect_weak_corr_scale", type=float, default=0.05)
    parser.add_argument("--simple_aif_defect_near_sigma", type=float, default=0.5)
    parser.add_argument("--simple_aif_defect_far_sigma", type=float, default=3.0)
    parser.add_argument("--simple_aif_defect_tau_sigma", type=float, default=0.5)
    parser.add_argument("--simple_aif_defect_near_min_abs", type=float, default=0.005)
    parser.add_argument("--simple_aif_defect_far_min_abs", type=float, default=0.02)
    parser.add_argument("--simple_aif_defect_tau_min_abs", type=float, default=0.005)
    parser.add_argument("--simple_aif_coverage_ref_clip_low", type=float, default=0.05)
    parser.add_argument("--simple_aif_coverage_ref_clip_high", type=float, default=0.95)
    parser.add_argument("--simple_aif_bootstrap_dynamic_target", type=int, default=16)
    parser.add_argument("--simple_aif_bootstrap_max_steps", type=int, default=400)
    parser.add_argument("--simple_aif_bootstrap_band_low_sigma", type=float, default=1.0)
    parser.add_argument("--simple_aif_bootstrap_band_high_sigma", type=float, default=1.0)
    parser.add_argument("--simple_aif_bootstrap_band_min_abs", type=float, default=0.02)
    parser.add_argument("--simple_aif_bootstrap_top_tail_sigma", type=float, default=2.5)
    parser.add_argument("--simple_aif_bootstrap_top_tail_min_abs", type=float, default=0.08)
    parser.add_argument("--aif_cost_fp", type=float, default=1.0)
    parser.add_argument("--aif_cost_fn", type=float, default=5.0)
    parser.add_argument("--aif_cost_query_label", type=float, default=1.0)
    parser.add_argument("--aif_cost_query_region", type=float, default=2.0)
    parser.add_argument("--aif_cost_query_support", type=float, default=4.0)
    parser.add_argument("--aif_region_k", type=int, default=64)
    parser.add_argument("--aif_support_budget_per_class", type=int, default=8)
    parser.add_argument("--debug_defect_writes", action="store_true", help="Log detailed FN/TP defect-write diagnostics into RB cards and stdout")
    parser.add_argument("--qual_corr_dump_dir", type=str, default=None,
                        help="Optional output directory for correction-bank qualitative before/after maps.")
    parser.add_argument("--qual_corr_classes", type=str, default="",
                        help="Comma-separated classes to dump correction qualitative examples for. Empty means all classes in the current run.")
    parser.add_argument("--qual_corr_min_abs", type=float, default=0.01,
                        help="Minimum absolute signed correction residual required before saving qualitative examples.")
    parser.add_argument("--qual_corr_max_per_bucket", type=int, default=2,
                        help="Maximum qualitative examples saved per class/bucket.")
    parser.add_argument("--qual_corr_save_flips_only", action="store_true",
                        help="If set, save only correction examples that flip the decision under the current theta.")
    parser.add_argument("--experiment_mode", type=str, default="single", choices=["single", "panel", "revisit", "batch"], help="single run, conventional panel, A→B→A revisit protocol, or batch over specified classes/policies")
    parser.add_argument("--panel_classes", type=str, default="", help="Comma-separated representative classes for conventional panel mode")
    parser.add_argument("--panel_methods", type=str, default="hardcode_static,hardcode_dyn,hardcode_dyn_corr,hardcode_full,simple_aif", help="Comma-separated panel methods; supports hardcode_static, hardcode_dyn, hardcode_dyn_corr, hardcode_full, simple_aif, entropy, margin, novelty, random, periodic, etc.")
    parser.add_argument("--batch_classes", type=str, default="",
                        help="Comma/list-form classes for batch mode, e.g. 'audiojack,bottle_cap,*end_cap' or '[audiojack, bottle_cap]'. Leading '*' markers are stripped.")
    parser.add_argument("--batch_out_dir", type=str, default="",
                        help="Destination directory for batch-mode JSON summaries and captured console logs.")
    parser.add_argument("--batch_policies", type=str, default="",
                        help="Optional comma-separated policies for batch mode. If empty, uses --policy only.")
    parser.add_argument("--batch_run_tag", type=str, default="E01_v21",
                        help="Filename tag used in batch mode, e.g. E01_v21.")
    parser.add_argument("--batch_num_query_seeds", type=int, default=3,
                        help="For random policy in batch mode, number of query seeds to run when --batch_query_seeds is not provided. Non-random policies run once.")
    parser.add_argument("--batch_query_seed_base", type=int, default=0,
                        help="First auto-generated query seed for random policy in batch mode.")
    parser.add_argument("--batch_query_seeds", type=str, default="",
                        help="Optional explicit query seeds for random policy, e.g. '0,1,2'. Overrides --batch_num_query_seeds.")
    parser.add_argument("--batch_support_seeds", type=str, default="",
                        help="Optional explicit support seeds for repeated support draws, e.g. '0,1,2'. Runs all listed support draws.")
    parser.add_argument("--batch_num_support_seeds", type=int, default=1,
                        help="Auto-generate this many support seeds from --batch_support_seed_base when >1. Default preserves legacy deterministic support if --support_seed is also omitted.")
    parser.add_argument("--batch_support_seed_base", type=int, default=0,
                        help="First auto-generated support seed for batch repeated-support experiments.")
    parser.add_argument("--batch_stream_seeds", type=str, default="",
                        help="Optional explicit stream seeds for stream-order robustness, e.g. '0,1,2'. Runs all listed stream permutations.")
    parser.add_argument("--batch_num_stream_seeds", type=int, default=1,
                        help="Auto-generate this many stream seeds from --batch_stream_seed_base when >1. Default preserves legacy stream behavior if --stream_seed is also omitted.")
    parser.add_argument("--batch_stream_seed_base", type=int, default=0,
                        help="First auto-generated stream seed for batch stream-order experiments.")
    parser.add_argument("--batch_fail_fast", action="store_true",
                        help="Abort batch mode after the first failed run.")
    parser.add_argument("--stream_start_idx", type=int, default=None)
    parser.add_argument("--stream_end_idx", type=int, default=None)
    parser.add_argument("--stream_start_frac", type=float, default=None)
    parser.add_argument("--stream_end_frac", type=float, default=None)
    parser.add_argument("--no_shuffle_test", action="store_true", help="Disable test-stream shuffling before slicing")
    parser.add_argument("--save_temporal_artifacts", action="store_true", help="Save per-run temporal JSON and plots")
    parser.add_argument("--temporal_later_frac", type=float, default=0.5)
    parser.add_argument("--enable_deployment_cost", action="store_true",
                        help="Record per-run deployment-cost timing and CUDA memory diagnostics without changing model behavior")
    parser.add_argument("--deployment_cost_cuda_sync", action="store_true",
                        help="Synchronize CUDA around timing stages for more accurate but slower GPU timing")
    parser.add_argument("--enable_hidden_fn_analysis", action="store_true",
                        help="Record hidden false-negative diagnostics for predicted-normal defect samples")
    parser.add_argument("--hidden_fn_near_sigma", type=float, default=1.0,
                        help="Near-threshold hidden-FN band in units of sigma_N")
    parser.add_argument("--hidden_fn_low_p", type=float, default=0.10,
                        help="Low-confidence threshold for hidden-FN diagnostics")
    parser.add_argument("--hidden_fn_topk", type=int, default=20,
                        help="Number of hidden-FN examples saved in summary JSON")
    parser.add_argument("--temporal_first_n", type=int, default=200)
    parser.add_argument("--save_state_path", type=str, default=None, help="Optional torch checkpoint path for resume state")
    parser.add_argument("--load_state_path", type=str, default=None, help="Load a saved resume-state checkpoint before streaming")
    parser.add_argument("--enable_ltm_lite", action="store_true", help="Enable minimal v10+LTM sidecar without changing the query controller")
    parser.add_argument("--disable_ltm_lite_retrieval", action="store_true")
    parser.add_argument("--disable_ltm_lite_promotion", action="store_true")
    parser.add_argument("--revisit_class_a", type=str, default="", help="Class A for A→B→A revisit protocol")
    parser.add_argument("--revisit_class_b", type=str, default="", help="Class B for A→B→A revisit protocol")
    parser.add_argument("--revisit_a1_frac", type=float, default=0.5, help="Fraction of class-A stream used for session A1")
    parser.add_argument("--revisit_compare_ltm", action="store_true", help="Also run v10+LTM-lite in revisit mode")

    # v20 memory-management flags: extracted from v19, but controller/sidecar remain v16-core by default.
    parser.add_argument("--enable_maturation_buffer", action="store_true")
    parser.add_argument("--disable_maturation_ltm_retrieval", action="store_true")
    parser.add_argument("--disable_maturation_ltm_promotion", action="store_true")
    parser.add_argument("--ltm_retrieval_k", type=int, default=1)
    parser.add_argument("--enable_maturation_buffer_retrieval", action="store_true")
    parser.add_argument("--mature_retrieval_min_events", type=int, default=2)
    parser.add_argument("--mature_retrieval_min_usefulness", type=float, default=0.10)
    parser.add_argument("--mature_retrieval_score_thr", type=float, default=0.25)
    parser.add_argument("--mature_retrieval_penalty", type=float, default=0.01)
    parser.add_argument("--mature_retrieval_max_protos", type=int, default=256)
    parser.add_argument("--mature_retrieval_k", type=int, default=1)
    parser.add_argument("--mature_merge_radius", type=float, default=0.06)
    parser.add_argument("--mature_event_merge_eps", type=float, default=0.03)
    parser.add_argument("--mature_max_event_protos", type=int, default=8)
    parser.add_argument("--mature_min_events", type=int, default=2)
    parser.add_argument("--mature_min_usefulness", type=float, default=0.15)
    parser.add_argument("--mature_score_thr", type=float, default=0.45)
    parser.add_argument("--mature_promote_every", type=int, default=32)
    parser.add_argument("--mature_max_candidates", type=int, default=0)
    parser.add_argument("--enable_stm_retirement", action="store_true")
    parser.add_argument("--stm_retire_ltm_eps", type=float, default=0.025)
    parser.add_argument("--enable_utility_aware_stm_retention", action="store_true")
    parser.add_argument("--stm_retention_local_k", type=int, default=32)
    parser.add_argument("--stm_retention_recency_tau", type=float, default=1024.0)
    parser.add_argument("--stm_retention_ltm_cover_eps", type=float, default=0.025)
    parser.add_argument("--stm_retention_replace_margin", type=float, default=0.03)
    parser.add_argument("--stm_retention_ema", type=float, default=0.10)

    args = parser.parse_args()


    aif_cfg = AIFConfig(
        cost_fp=float(args.aif_cost_fp),
        cost_fn=float(args.aif_cost_fn),
        cost_query_label=float(args.aif_cost_query_label),
        cost_query_region=float(args.aif_cost_query_region),
        cost_query_support=float(args.aif_cost_query_support),
        region_k=int(args.aif_region_k),
        support_budget_per_class=int(args.aif_support_budget_per_class),
    )
    simple_aif_cfg = SimpleAIFConfig(
        cost_fp=float(args.simple_aif_cost_fp),
        cost_fn=float(args.simple_aif_cost_fn),
        query_cost=float(args.simple_aif_query_cost),
        beta_entropy=float(args.simple_aif_beta_entropy),
        lambda_normal=float(args.simple_aif_lambda_normal),
        lambda_correction=float(args.simple_aif_lambda_correction),
        lambda_defect_side=float(args.simple_aif_lambda_defect_side),
        adapt_query_cost=(not bool(args.simple_aif_no_adapt_query_cost)),
        target_qrate=float(args.simple_aif_target_qrate),
        query_cost_lr=float(args.simple_aif_query_cost_lr),
        min_query_cost=float(args.simple_aif_min_query_cost),
        max_query_cost=float(args.simple_aif_max_query_cost),
        local_qrate_window=int(args.simple_aif_local_qrate_window),
        use_local_saliency=bool(args.simple_aif_use_local_saliency),
        saliency_weight=float(args.simple_aif_saliency_weight),
        correction_gap_scale=float(args.simple_aif_correction_gap_scale),
        defect_weak_corr_scale=float(args.simple_aif_defect_weak_corr_scale),
        defect_near_sigma=float(args.simple_aif_defect_near_sigma),
        defect_far_sigma=float(args.simple_aif_defect_far_sigma),
        defect_tau_sigma=float(args.simple_aif_defect_tau_sigma),
        defect_near_min_abs=float(args.simple_aif_defect_near_min_abs),
        defect_far_min_abs=float(args.simple_aif_defect_far_min_abs),
        defect_tau_min_abs=float(args.simple_aif_defect_tau_min_abs),
        coverage_ref_clip_low=float(args.simple_aif_coverage_ref_clip_low),
        coverage_ref_clip_high=float(args.simple_aif_coverage_ref_clip_high),
        bootstrap_dynamic_target=int(args.simple_aif_bootstrap_dynamic_target),
        bootstrap_max_steps=int(args.simple_aif_bootstrap_max_steps),
        bootstrap_band_low_sigma=float(args.simple_aif_bootstrap_band_low_sigma),
        bootstrap_band_high_sigma=float(args.simple_aif_bootstrap_band_high_sigma),
        bootstrap_band_min_abs=float(args.simple_aif_bootstrap_band_min_abs),
        bootstrap_top_tail_sigma=float(args.simple_aif_bootstrap_top_tail_sigma),
        bootstrap_top_tail_min_abs=float(args.simple_aif_bootstrap_top_tail_min_abs),
        v22_enable_warmup_budget=bool(args.v22_enable_warmup_budget),
        v22_warmup_steps=int(args.v22_warmup_steps),
        v22_warmup_qrate=float(args.v22_warmup_qrate),
        v22_budget_guard=(not bool(args.v22_disable_budget_guard)),
        v22_enable_defect_audit=bool(args.v22_enable_defect_audit),
        v22_audit_frac=float(args.v22_audit_frac),
        v22_audit_min_p=float(args.v22_audit_min_p),
        v22_audit_score_sigma=float(args.v22_audit_score_sigma),
        v22_enable_fn_audit=bool(args.v22_enable_fn_audit),
        v22_fn_audit_band=float(args.v22_fn_audit_band),
        v22_fn_audit_max_extra_qrate=float(args.v22_fn_audit_max_extra_qrate),
        v22_fn_audit_min_gap=int(args.v22_fn_audit_min_gap),
        v22_fn_audit_warmup_steps=int(args.v22_fn_audit_warmup_steps),
        v22_fn_audit_min_p=float(args.v22_fn_audit_min_p),
    )
    if args.query_baseline_target_qrate is not None:
        # Baseline policies use the same target-qrate field as SimpleAIF so panel
        # code can sweep one value consistently across controllers.
        simple_aif_cfg.target_qrate = float(args.query_baseline_target_qrate)

    gating = RBGatingConfig(
        min_count=args.min_count,
        min_fp_rate=args.min_fp_rate,
        min_fn_rate=args.min_fn_rate,
    )
    qcfg = QueryConfig(
        ambig_low=args.ambig_low,
        ambig_high=args.ambig_high,
        warmup_queries=args.warmup_queries,
        warmup_steps=args.warmup_steps,
        topq_quantile=args.topq_quantile,
    )

    thcfg = ThresholdConfig(
        target_fpr=float(args.target_fpr),
        min_bufN=int(args.theta_min_bufN),
        bufN_max=int(args.theta_bufN_max),
        p_low=float(args.p_low),
        z_prior_anom=float(args.z_prior_anom),
        slope_c=float(args.posterior_slope_c),
        enable_support_guard=bool(args.v22_enable_support_guard),
        support_sigma_floor=float(args.v22_support_sigma_floor),
        support_theta_z=float(args.v22_support_theta_z),
        support_cold_sigma_thr=float(args.v22_support_cold_sigma_thr),
    )
    theta_sidecar_cfg = GuardedThetaSidecarConfig(
        normal_capacity=int(args.theta_sidecar_normal_capacity),
        defect_capacity=int(args.theta_sidecar_defect_capacity),
        recent_capacity=int(args.theta_sidecar_recent_capacity),
        min_normal_anchor=int(args.theta_sidecar_min_normal_anchor),
        min_defect_anchor=int(args.theta_sidecar_min_defect_anchor),
        min_recent_total=int(args.theta_sidecar_min_recent_total),
        min_recent_per_class=int(args.theta_sidecar_min_recent_per_class),
        update_every_labeled=int(args.theta_sidecar_update_every_labeled),
        q_normal=float(args.theta_sidecar_q_normal),
        q_defect=float(args.theta_sidecar_q_defect),
        delta_normal=float(args.theta_sidecar_delta_normal),
        delta_defect=float(args.theta_sidecar_delta_defect),
        sep_margin=float(args.theta_sidecar_sep_margin),
        lambda_fp=float(args.theta_sidecar_lambda_fp),
        lambda_fn=float(args.theta_sidecar_lambda_fn),
        lambda_move=float(args.theta_sidecar_lambda_move),
        accept_margin=float(args.theta_sidecar_accept_margin),
        guard_delta_fp=float(args.theta_sidecar_guard_delta_fp),
        guard_delta_fn=float(args.theta_sidecar_guard_delta_fn),
        step_eta=float(args.theta_sidecar_step_eta),
        step_up=float(args.theta_sidecar_step_up),
        step_down=float(args.theta_sidecar_step_down),
        candidate_radius=float(args.theta_sidecar_candidate_radius),
        candidate_step=float(args.theta_sidecar_candidate_step),
        support_seed=(not bool(args.theta_sidecar_no_support_seed)),
        enable_normal_fallback=(not bool(args.theta_sidecar_disable_normal_fallback)),
        min_recent_normals_fallback=int(args.theta_sidecar_min_recent_normals_fallback),
        min_defect_anchor_fallback=int(args.theta_sidecar_min_defect_anchor_fallback),
        normal_fallback_fpr_trigger=float(args.theta_sidecar_normal_fallback_fpr_trigger),
        normal_fallback_guard_delta_fn=float(args.theta_sidecar_normal_fallback_guard_delta_fn),
        enable_fast_upward_rescue=bool(args.v22_enable_fast_upward_rescue),
        enable_balanced_rescue=bool(args.v22_enable_dual_anchor_balanced_rescue),
        fast_rescue_min_fp_normals=int(args.v22_fast_rescue_min_fp_normals),
        fast_rescue_fpr_trigger=float(args.v22_fast_rescue_fpr_trigger),
        fast_rescue_q_normal=float(args.v22_fast_rescue_q_normal),
        fast_rescue_delta_normal=float(args.v22_fast_rescue_delta_normal),
        fast_rescue_step_up=float(args.v22_fast_rescue_step_up),
        fast_rescue_guard_delta_fn=float(args.v22_fast_rescue_guard_delta_fn),
        fast_rescue_max_recent=int(args.v22_fast_rescue_max_recent),
        balanced_min_normals=int(args.v22_dual_anchor_min_normals),
        balanced_min_defects=int(args.v22_dual_anchor_min_defects),
        balanced_defect_q=float(args.v22_dual_anchor_defect_q),
        balanced_defect_margin=float(args.v22_dual_anchor_defect_margin),
        balanced_normal_q=float(args.v22_dual_anchor_normal_q),
        balanced_up_trigger=float(args.v22_dual_anchor_up_trigger),
        balanced_down_trigger=float(args.v22_dual_anchor_down_trigger),
        balanced_exit_fpr=float(args.v22_dual_anchor_exit_fpr),
        balanced_exit_fnr=float(args.v22_dual_anchor_exit_fnr),
        balanced_anchor_gap=float(args.v22_dual_anchor_anchor_gap),
        balanced_up_defect_slack=float(args.v22_dual_anchor_up_defect_slack),
        balanced_step_up=float(args.v22_dual_anchor_step_up),
        balanced_step_down=float(args.v22_dual_anchor_step_down),
        balanced_normal_step=float(args.v22_dual_anchor_normal_step),
        balanced_rescue_eta=float(args.v22_dual_anchor_rescue_eta),
        balanced_candidate_radius_normal=float(args.v22_dual_anchor_candidate_radius_normal),
        balanced_candidate_radius_rescue=float(args.v22_dual_anchor_candidate_radius_rescue),
        balanced_candidate_step=float(args.v22_dual_anchor_candidate_step),
        balanced_guard_delta_fp_normal=float(args.v22_dual_anchor_guard_delta_fp_normal),
        balanced_guard_delta_fn_normal=float(args.v22_dual_anchor_guard_delta_fn_normal),
        balanced_guard_delta_fn_up=float(args.v22_dual_anchor_guard_delta_fn_up),
        balanced_guard_delta_fp_down=float(args.v22_dual_anchor_guard_delta_fp_down),
        balanced_lambda_fp=float(args.v22_dual_anchor_lambda_fp),
        balanced_lambda_fn=float(args.v22_dual_anchor_lambda_fn),
        balanced_lambda_move=float(args.v22_dual_anchor_lambda_move),
        balanced_lambda_q=float(args.v22_dual_anchor_lambda_q),
        balanced_lambda_osc=float(args.v22_dual_anchor_lambda_osc),
        balanced_cooldown_steps=int(args.v22_dual_anchor_cooldown_steps),
    )
    support_aug_cfg = VisionADSupportAugConfig(
        rotation_deg=float(args.rotation_deg),
        rotation_p=float(args.rotation_p),
        translate=float(args.translate),
        hflip_p=float(args.hflip_p)
    )

    # Build correction + set-membership configs from CLI
    corr_cfg = CorrectionConfig(
        enabled=(not args.no_correction_bank),
        K_correction=int(args.K_correction),
        corr_weight=float(args.corr_weight),
        beta_max=float(args.beta_max),
        keep_defect_bank=bool(args.keep_defect_bank or args.use_defect_bank),
        enable_sign_aware_write=(not bool(args.disable_sign_aware_correction_write)),
        enable_patch_weighted_beta=(not bool(args.disable_patch_weighted_correction_beta)),
        enable_sign_aware_retention=(not bool(args.disable_sign_aware_correction_retention)),
        outcome_weight_fp=float(args.corr_outcome_weight_fp),
        outcome_weight_fn=float(args.corr_outcome_weight_fn),
        outcome_weight_tn=float(args.corr_outcome_weight_tn),
        outcome_weight_tp=float(args.corr_outcome_weight_tp),
        patch_beta_floor=float(args.corr_patch_beta_floor),
        patch_beta_cap=float(args.corr_patch_beta_cap),
        patch_beta_power=float(args.corr_patch_beta_power),
        corr_replace_margin=float(args.corr_replace_margin),
    )

    sm_cfg = SetMembershipConfig(
        enabled=(not args.no_set_membership),
        core_p_defect_max=float(args.core_p_defect_max),
        core_margin_sigma=float(args.core_margin_sigma),
        radius_slack=float(args.core_radius_slack),
        core_global_novelty_min=float(args.core_global_novelty_min),
        core_max_buf=int(args.core_max_buf),
        core_patches_per_layer=int(args.core_patches_per_layer),
        core_tau_insert=float(args.core_tau_insert),
        core_tau_replace=float(args.core_tau_replace),
    )

    use_defect_bank = bool(args.use_defect_bank or args.keep_defect_bank)
    if args.disable_defect_bank:
        use_defect_bank = False

    views = [ViewSpec(name="id", kind="id")]
    if args.use_posclamp:
        views.append(ViewSpec(name="posclamp", kind="posclamp", posclamp_low=int(args.posclamp_low)))
    if args.use_yflip:
        views.append(ViewSpec(name="yflip", kind="yflip"))
    if args.use_xflip:
        views.append(ViewSpec(name="xflip", kind="xflip"))

    base_run_kwargs = dict(
        dataset=args.dataset,
        root=args.root,
        csv=args.csv,
        init_dump_json=args.init_dump_json,
        use_global_threshold_initialization=bool(args.use_global_threshold_initialization),
        global_threshold_init_mode=str(args.global_threshold_init_mode),
        support_select=str(args.support_select),
        support_select_num_candidates=int(args.support_select_num_candidates),
        support_select_calib_max=int(args.support_select_calib_max),
        support_select_inlier_q=float(args.support_select_inlier_q),
        support_select_tail_q=float(args.support_select_tail_q),
        support_select_tail_q_hi=float(args.support_select_tail_q_hi),
        support_select_theta_q=float(args.support_select_theta_q),
        support_select_lambda_tail_gap=float(args.support_select_lambda_tail_gap),
        support_select_lambda_std=float(args.support_select_lambda_std),
        support_select_lambda_outlier=float(args.support_select_lambda_outlier),
        support_select_cache_dir=args.support_select_cache_dir,
        baseline_compat=bool(args.baseline_compat),
        support_include_identity=bool(args.support_include_identity),
        dynonly_freeze_theta=bool(args.dynonly_freeze_theta),
        dynonly_freeze_normal_dyn=bool(args.dynonly_freeze_normal_dyn),
        device=args.device,
        max_samples=args.max_samples,
        seed=args.seed,
        query_seed=args.query_seed,
        support_seed=args.support_seed,
        stream_seed=args.stream_seed,
        #feature_fuse=args.feature_fuse,
        policy=str(args.policy),
        aif_cfg=aif_cfg,
        simple_aif_cfg=simple_aif_cfg,
        use_modes=args.use_modes,
        K_modes=args.K_modes,
        shots_per_mode=args.shots_per_mode,
        support_aug_cfg=support_aug_cfg,
        aug_per_image=int(args.aug_per_image),
        views=views,
        use_rotations=(not args.no_rotations),
        K_normal_fixed=args.K_normal_fixed,
        K_normal_dyn=args.K_normal_dyn,
        K_normal_ltm=args.K_normal_ltm,
        ltm_min_stm_size=int(args.ltm_min_stm_size),
        ltm_min_repeat=int(args.ltm_min_repeat),
        K_defect=args.K_defect,
        K_concept=args.K_concept,
        knn_k_normal=args.knn_k_normal,
        corr_cfg=corr_cfg,
        sm_cfg=sm_cfg,
        use_defect_bank=use_defect_bank,
        disable_defect_bank=bool(args.disable_defect_bank),
        topk_patches=args.topk_patches,
        score_mode=args.score_mode,
        lam=args.lam,
        tau_close=args.tau_close,
        gamma=args.gamma,
        layer_fusion=args.layer_fusion,
        view_fusion=args.view_fusion,
        img_agg=args.img_agg,
        img_topk=args.img_topk,
        img_quantile=args.img_quantile,
        tauN_insert=args.tauN_insert,
        tauN_replace=args.tauN_replace,
        tauD_insert=args.tauD_insert,
        tauD_replace=args.tauD_replace,
        gating=gating,
        qcfg=qcfg,
        thcfg=thcfg,
        theta_controller_kind=str(args.theta_controller),
        theta_sidecar_cfg=theta_sidecar_cfg,
        store_only_queried_cards=(not args.store_all_cards),
        concept_seed_patches=args.concept_seed_patches,
        concept_per_img_cap=args.concept_per_img_cap,
        fixed_per_img_cap=args.fixed_per_img_cap,
        bank_dtype=args.bank_dtype,
        empty_cache_every=args.empty_cache_every,
        debug_defect_writes=bool(args.debug_defect_writes),
        stream_start_idx=args.stream_start_idx,
        stream_end_idx=args.stream_end_idx,
        stream_start_frac=args.stream_start_frac,
        stream_end_frac=args.stream_end_frac,
        no_shuffle_test=bool(args.no_shuffle_test),
        save_state_path=args.save_state_path,
        load_state_path=args.load_state_path,
        save_temporal_artifacts_flag=bool(args.save_temporal_artifacts),
        temporal_later_frac=float(args.temporal_later_frac),
        temporal_first_n=int(args.temporal_first_n),
        enable_deployment_cost=bool(args.enable_deployment_cost),
        deployment_cost_cuda_sync=bool(args.deployment_cost_cuda_sync),
        enable_hidden_fn_analysis=bool(args.enable_hidden_fn_analysis),
        hidden_fn_near_sigma=float(args.hidden_fn_near_sigma),
        hidden_fn_low_p=float(args.hidden_fn_low_p),
        hidden_fn_topk=int(args.hidden_fn_topk),
        enable_ltm_lite=bool(args.enable_ltm_lite),
        ltm_lite_retrieval=(not bool(args.disable_ltm_lite_retrieval)),
        ltm_lite_promotion=(not bool(args.disable_ltm_lite_promotion)),
        enable_maturation_buffer=bool(args.enable_maturation_buffer),
        maturation_ltm_retrieval=(not bool(args.disable_maturation_ltm_retrieval)),
        maturation_ltm_promotion=(not bool(args.disable_maturation_ltm_promotion)),
        ltm_retrieval_k=int(args.ltm_retrieval_k),
        enable_maturation_buffer_retrieval=bool(args.enable_maturation_buffer_retrieval),
        mature_retrieval_min_events=int(args.mature_retrieval_min_events),
        mature_retrieval_min_usefulness=float(args.mature_retrieval_min_usefulness),
        mature_retrieval_score_thr=float(args.mature_retrieval_score_thr),
        mature_retrieval_penalty=float(args.mature_retrieval_penalty),
        mature_retrieval_max_protos=int(args.mature_retrieval_max_protos),
        mature_retrieval_k=int(args.mature_retrieval_k),
        mature_merge_radius=float(args.mature_merge_radius),
        mature_event_merge_eps=float(args.mature_event_merge_eps),
        mature_max_event_protos=int(args.mature_max_event_protos),
        mature_min_events=int(args.mature_min_events),
        mature_min_usefulness=float(args.mature_min_usefulness),
        mature_score_thr=float(args.mature_score_thr),
        mature_promote_every=int(args.mature_promote_every),
        mature_max_candidates=int(args.mature_max_candidates),
        enable_stm_retirement=bool(args.enable_stm_retirement),
        stm_retire_ltm_eps=float(args.stm_retire_ltm_eps),
        enable_utility_aware_stm_retention=bool(args.enable_utility_aware_stm_retention),
        stm_retention_local_k=int(args.stm_retention_local_k),
        stm_retention_recency_tau=float(args.stm_retention_recency_tau),
        stm_retention_ltm_cover_eps=float(args.stm_retention_ltm_cover_eps),
        stm_retention_replace_margin=float(args.stm_retention_replace_margin),
        stm_retention_ema=float(args.stm_retention_ema),
        qual_corr_dump_dir=args.qual_corr_dump_dir,
        qual_corr_classes=str(args.qual_corr_classes),
        qual_corr_min_abs=float(args.qual_corr_min_abs),
        qual_corr_max_per_bucket=int(args.qual_corr_max_per_bucket),
        qual_corr_save_flips_only=bool(args.qual_corr_save_flips_only),
        ablation=args.ablation,
    )

    if str(args.experiment_mode) == 'single':
        run_phase3_streaming_aif(
            class_name=args.class_name,
            out_json=args.out_json,
            **base_run_kwargs,
        )
    elif str(args.experiment_mode) == 'panel':
        classes = parse_comma_list(args.panel_classes)
        if not classes:
            raise ValueError('--panel_classes is required in panel mode')
        panel = run_conventional_panel_experiment(
            classes=classes,
            methods=parse_comma_list(args.panel_methods),
            out_json=args.out_json,
            base_run_kwargs=base_run_kwargs,
        )
        print(json.dumps(panel.get('aggregate', {}), indent=2))
        print(f"Panel summary saved to: {panel.get('panel_summary_path')}")
    elif str(args.experiment_mode) == 'batch':
        batch = run_batch_experiment(
            args=args,
            base_run_kwargs=base_run_kwargs,
            simple_aif_cfg=simple_aif_cfg,
        )
        print(f"Batch summary saved to: {batch.get('batch_summary_path')}")
    elif str(args.experiment_mode) == 'revisit':
        if (not args.revisit_class_a) or (not args.revisit_class_b):
            raise ValueError('--revisit_class_a and --revisit_class_b are required in revisit mode')
        revisit = run_ab_revisit_experiment(
            class_a=str(args.revisit_class_a),
            class_b=str(args.revisit_class_b),
            a1_frac=float(args.revisit_a1_frac),
            out_json=args.out_json,
            base_run_kwargs=base_run_kwargs,
            compare_ltm=bool(args.revisit_compare_ltm),
            early_n=int(args.temporal_first_n),
        )
        print(json.dumps({
            'resume_minus_reset': revisit.get('resume_minus_reset', {}),
            'resume_ltm_minus_reset': revisit.get('resume_ltm_minus_reset', {}),
        }, indent=2))
        print(f"Revisit summary saved to: {revisit.get('revisit_summary_path')}")
    else:
        raise ValueError(f'Unknown experiment_mode: {args.experiment_mode}')





 