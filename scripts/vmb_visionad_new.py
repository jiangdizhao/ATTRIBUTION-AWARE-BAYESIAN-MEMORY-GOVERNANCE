# -*- coding: utf-8 -*-
"""
vmb_visionad.py — VisionAD faithful reproduction baseline

Implements the key components described in:
  "Search is All You Need for Few-shot Anomaly Detection" (VisionAD)

Reproduced mechanics:
  - Category-indexed global & patch memory banks (Eq.1–2)
  - Category retrieval via global (CLS) similarity (Eq.3)
  - Patch-level NN anomaly map (Eq.4)
  - Pseudo multi-view: apply identical view transforms to query & reference, then
    fuse aligned anomaly maps (Eq.5)
  - Image score = mean of top 1% pixels (Eq.6)
  - Support enhancement via rotation / translation / flipping

This file is self-contained and intentionally avoids importing torchvision
(some environments have torch/torchvision binary mismatches).
"""
import os
import gc
import json
import math
import random
import argparse
from typing import List, Dict, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from sklearn.metrics import roc_auc_score

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip() != ""]

def format_float(x: float, nd: int = 4) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
#  Simple Torch KMeans (kept for Phase-2 compatibility)
# ---------------------------------------------------------------------------

@torch.no_grad()
def kmeans_torch(
    x: torch.Tensor,
    K: int,
    num_iters: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [N, C] L2-normalized
    Returns:
        centroids: [K, C]
        labels:    [N] long (0..K-1)
    """
    device = x.device
    N, C = x.shape
    if K <= 0:
        raise ValueError("K must be positive")
    if N < K:
        raise ValueError(f"N ({N}) must be >= K ({K})")

    perm = torch.randperm(N, device=device)
    centroids = x[perm[:K]].clone()  # [K, C]

    for _ in range(int(num_iters)):
        sims = x @ centroids.t()                      # [N, K]
        labels = sims.argmax(dim=1)                   # [N]
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = labels == k
            if mask.any():
                new_centroids[k] = l2_normalize(x[mask].mean(dim=0))
            else:
                new_centroids[k] = centroids[k]
        centroids = new_centroids

    sims = x @ centroids.t()
    labels = sims.argmax(dim=1)
    return centroids, labels


# ---------------------------------------------------------------------------
#  Dataset indexing helpers (MVTec / VisA / Real-IAD)
# ---------------------------------------------------------------------------

class ImageSample:
    def __init__(self, path: str, cls_name: str, split: str, is_good: bool, mask_path: str = None):
        self.path = path
        self.cls_name = cls_name
        self.split = split      # "train" or "test"
        self.is_good = is_good  # True for normal, False for anomaly
        self.mask_path = mask_path


class RealIADJSONIndex:
    """
    Real-IAD index from per-category JSON protocol.
    See your modified Real-IAD directory structure in earlier scripts.
    """
    def __init__(self, root: str, json_dir: str, only_classes: Optional[Sequence[str]] = None):
        self.root = root
        self.json_dir = json_dir
        self.only_classes = set(only_classes) if only_classes else None

        self.classes: List[str] = []
        self.train_samples: List[ImageSample] = []
        self.test_samples: List[ImageSample] = []
        self._build()

    def _build(self):
        if not os.path.isdir(self.json_dir):
            raise FileNotFoundError(f"Real-IAD json_dir not found: {self.json_dir}")

        json_files = sorted([f for f in os.listdir(self.json_dir) if f.lower().endswith(".json")])
        for jf in json_files:
            cls_name = os.path.splitext(jf)[0]
            if self.only_classes is not None and cls_name not in self.only_classes:
                continue

            fp = os.path.join(self.json_dir, jf)
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("meta", {})
            prefix = meta.get("prefix", "")           # e.g. "audiojack/"
            normal_class = meta.get("normal_class", "OK")

            base = os.path.join(self.root, "images", prefix).rstrip("/")

            def _rec_to_sample(rec: dict, split: str) -> ImageSample:
                cat = rec.get("category", cls_name)
                anom_cls = rec.get("anomaly_class", normal_class)
                img_rel = rec.get("image_path", "")
                mask_rel = rec.get("mask_path", None)
                img_path = os.path.join(base, img_rel)
                mask_path = os.path.join(base, mask_rel) if mask_rel else None
                is_good = (anom_cls == normal_class)
                return ImageSample(img_path, cat, split, is_good, mask_path)

            for rec in data.get("train", []):
                self.train_samples.append(_rec_to_sample(rec, "train"))
            for rec in data.get("test", []):
                self.test_samples.append(_rec_to_sample(rec, "test"))

            if cls_name not in self.classes:
                self.classes.append(cls_name)

        self.classes.sort()


class VisACSVIndex:
    """
    VisA index based on a *single CSV* (e.g., 1cls.csv / 2cls_*.csv),
    where the split (train/test) is already specified per row.

    Expected columns (as in your original vmb_visionad.py):
      - object : class/category name
      - split  : "train" or "test"
      - label  : "normal" or "anomaly"
      - image  : relative image path (joined with root)
      - mask   : (optional) relative mask path (joined with root)

    Produces:
      - self.classes
      - self.train_samples : List[ImageSample] with split == "train"
      - self.test_samples  : List[ImageSample] with split == "test"
    """

    def __init__(self, root: str, csv_path: str, only_classes: Optional[Sequence[str]] = None):
        self.root = root
        self.csv_path = csv_path
        self.only_classes = set(only_classes) if only_classes else None

        self.classes: List[str] = []
        self.train_samples: List[ImageSample] = []
        self.test_samples: List[ImageSample] = []
        self._load()

    def _load(self) -> None:
        df = pd.read_csv(self.csv_path)

        # required columns
        for col in ["object", "split", "label", "image"]:
            if col not in df.columns:
                raise ValueError(f"VisA CSV missing required column: '{col}' (found: {list(df.columns)})")

        df["is_good"] = df["label"].map({"normal": True, "anomaly": False})
        if df["is_good"].isna().any():
            bad = df[df["is_good"].isna()]["label"].unique().tolist()
            raise ValueError(f"Unexpected VisA label values in CSV: {bad} (expected 'normal'/'anomaly')")

        if self.only_classes is None:
            self.classes = sorted(df["object"].unique().tolist())
        else:
            self.classes = sorted([c for c in df["object"].unique().tolist() if c in self.only_classes])

        has_mask = ("mask" in df.columns)

        for _, row in df.iterrows():
            cls_name = str(row["object"])
            if self.only_classes is not None and cls_name not in self.only_classes:
                continue

            split = str(row["split"]).strip().lower()
            if split not in ["train", "test"]:
                continue

            is_good = bool(row["is_good"])
            rel_img = str(row["image"])
            img_path = os.path.join(self.root, rel_img)

            mask_path = None
            if has_mask:
                mask_rel = row["mask"]
                if isinstance(mask_rel, str) and (mask_rel.strip() != "") and (mask_rel.strip().lower() != "nan"):
                    mask_path = os.path.join(self.root, mask_rel)

            s = ImageSample(path=img_path, cls_name=cls_name, split=split, is_good=is_good, mask_path=mask_path)
            if split == "train":
                self.train_samples.append(s)
            else:
                self.test_samples.append(s)


class IndustrialDatasetIndex:
    """
    Minimal MVTec-AD style index: root/class/train/good and root/class/test/*.
    """
    def __init__(self, root: str, only_classes: Optional[Sequence[str]] = None):
        self.root = root
        self.only_classes = set(only_classes) if only_classes else None

        self.classes: List[str] = []
        self.train_samples: List[ImageSample] = []
        self.test_samples: List[ImageSample] = []
        self._build()

    def _is_img(self, p: str) -> bool:
        return p.lower().endswith(IMG_EXTS)

    def _build(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        if self.only_classes is not None:
            classes = [c for c in classes if c in self.only_classes]

        for cls in classes:
            cls_dir = os.path.join(self.root, cls)
            train_good = os.path.join(cls_dir, "train", "good")
            test_dir = os.path.join(cls_dir, "test")
            gt_dir = os.path.join(cls_dir, "ground_truth")

            if os.path.isdir(train_good):
                for r, _, fs in os.walk(train_good):
                    for f in fs:
                        if self._is_img(f):
                            self.train_samples.append(ImageSample(os.path.join(r, f), cls, "train", True))

            if os.path.isdir(test_dir):
                for defect in sorted(os.listdir(test_dir)):
                    defect_dir = os.path.join(test_dir, defect)
                    if not os.path.isdir(defect_dir):
                        continue
                    is_good = (defect == "good")
                    for r, _, fs in os.walk(defect_dir):
                        for f in fs:
                            if not self._is_img(f):
                                continue
                            img_path = os.path.join(r, f)
                            mask_path = None
                            if (not is_good) and os.path.isdir(gt_dir):
                                rel = os.path.relpath(img_path, defect_dir)
                                base = os.path.splitext(os.path.basename(rel))[0]
                                cand = os.path.join(gt_dir, defect, base + "_mask.png")
                                if os.path.isfile(cand):
                                    mask_path = cand
                            self.test_samples.append(ImageSample(img_path, cls, "test", is_good, mask_path))

            self.classes.append(cls)

        self.classes.sort()


# ---------------------------------------------------------------------------
#  Image preprocessing (no torchvision)
# ---------------------------------------------------------------------------

def resize_and_center_crop(img: Image.Image, resize_size: int, crop_size: int) -> Image.Image:
    img = img.resize((resize_size, resize_size), resample=Image.BICUBIC)
    if crop_size == resize_size:
        return img
    left = (resize_size - crop_size) // 2
    top = (resize_size - crop_size) // 2
    return img.crop((left, top, left + crop_size, top + crop_size))

def pil_to_normalized_tensor(img: Image.Image, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))               # [3,H,W]
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
#  DINOv2 backbone: multi-layer patch features + CLS global
# ---------------------------------------------------------------------------

class DINOv2MultiLayerBackbone(nn.Module):
    """
    Multi-layer DINOv2 ViT backbone with CLS global feature.

    - Loads DINOv2 via torch.hub (facebookresearch/dinov2).
    - Uses hooks on transformer blocks to grab token features.
    - encode_batch returns:
        patch_feats_per_layer: list of [B, N_patches, C]
        global_feats:          [B, C] CLS token from deepest hooked layer
    """
    def __init__(
        self,
        model_name: str = "dinov2_vitl14_reg",
        layers: Optional[Sequence[int]] = None,
        img_size: int = 392,
        resize_size: int = 448,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        self.img_size = int(img_size)
        self.resize_size = int(resize_size)
        self.use_fp16 = bool(use_fp16)

        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device)
        if self.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()

        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError("DINOv2 model has no .blocks attribute. Can't register hooks.")
        n_blocks = len(blocks)

        if layers is None or len(list(layers)) == 0:
            # End-of-stage sampling: 4 stages
            self.layers = [max(0, int(round(n_blocks * p)) - 1) for p in (0.25, 0.50, 0.75, 1.0)]
            self.layers = sorted(list(dict.fromkeys(self.layers)))
        else:
            self.layers = sorted(list(layers))

        cfg = getattr(self.model, "default_cfg", None)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        if cfg is not None:
            self.mean = tuple(cfg.get("mean", self.mean))
            self.std = tuple(cfg.get("std", self.std))

        self.num_register_tokens = int(getattr(self.model, "num_register_tokens", 0))

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._feat_dict: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        blocks = getattr(self.model, "blocks")
        for idx in self.layers:
            if idx < 0 or idx >= len(blocks):
                raise ValueError(f"Layer index {idx} outside [0, {len(blocks)-1}]")
            block = blocks[idx]

            def _hook_closure(i):
                def _hook(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    self._feat_dict[i] = out.detach()
                return _hook

            self._hooks.append(block.register_forward_hook(_hook_closure(idx)))

    def _clear_feats(self):
        self._feat_dict = {}

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        img = resize_and_center_crop(img, self.resize_size, self.img_size)
        x = pil_to_normalized_tensor(img, self.mean, self.std)  # [3,H,W] float32
        return x

    @torch.no_grad()
    def encode_batch(self, imgs: List[Image.Image]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        self._clear_feats()
        x = torch.stack([self._preprocess(im) for im in imgs], dim=0).to(self.device)  # [B,3,H,W]
        if self.use_fp16 and self.device.type == "cuda":
            x = x.half()
        _ = self.model(x)

        for idx in self.layers:
            if idx not in self._feat_dict:
                raise RuntimeError(f"Hook for layer {idx} did not fire. Collected: {list(self._feat_dict.keys())}")

        start_patch = 1 + self.num_register_tokens
        patch_feats_per_layer: List[torch.Tensor] = []
        for idx in self.layers:
            tokens = self._feat_dict[idx]                          # [B,1+R+N,C]
            patch_tokens = tokens[:, start_patch:, :].float()      # [B,N,C]
            patch_tokens = F.normalize(patch_tokens, dim=-1)
            patch_feats_per_layer.append(patch_tokens.contiguous())

        last_layer_idx = max(self.layers)
        tokens_last = self._feat_dict[last_layer_idx]              # [B,1+R+N,C]
        cls = tokens_last[:, 0, :].float()                         # [B,C]
        global_feats = F.normalize(cls, dim=-1)
        return patch_feats_per_layer, global_feats

    @torch.no_grad()
    def encode_path(self, img_path: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        img = Image.open(img_path).convert("RGB")
        patches, g = self.encode_batch([img])
        return [p[0] for p in patches], g[0]


def fuse_layers_one_group(patch_feats_per_layer: List[torch.Tensor],mode="concat") -> torch.Tensor:
    """
    VisionAD 'group-to-group' (1 group) feature fusion:
      concatenate all selected layer patch features into a single vector per patch,
      then L2-normalize.

    Input: list of [B, N, C_l] (all same N)
    Output: [B, N, sum(C_l)] L2-normalized
    """
    if len(patch_feats_per_layer) == 0:
        raise ValueError("No layer features to fuse")
    if mode == "concat":
        x = torch.cat(patch_feats_per_layer, dim=-1)  # [B,N,D]
        return F.normalize(x, dim=-1)
    elif mode == "sum":
        z = torch.stack(patch_feats_per_layer, dim=0).sum(dim=0)  # [B,N,C]
        return F.normalize(z, dim=-1)
    else:
        raise ValueError(mode)
    


# ---------------------------------------------------------------------------
#  Views (Pseudo Multi-View)
# ---------------------------------------------------------------------------

class ViewSpec:
    """
    VisionAD pseudo multi-view transforms: thresholding (PosClamp) and flipping.
    """
    def __init__(self, name: str, kind: str, posclamp_low: int = 64):
        self.name = name
        self.kind = kind
        self.posclamp_low = int(posclamp_low)

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
        arr = np.asarray(img).astype(np.uint8)
        low = int(max(0, min(255, v.posclamp_low)))
        arr = np.maximum(arr, low).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if v.kind == "rot90":  return img.transpose(Image.ROTATE_90)
    if v.kind == "rot180": return img.transpose(Image.ROTATE_180)
    if v.kind == "rot270": return img.transpose(Image.ROTATE_270)
    raise ValueError(f"Unknown view kind: {v.kind}")

def align_map_to_identity(map_hw: torch.Tensor, v: ViewSpec) -> torch.Tensor:
    if v.kind in ["id", "posclamp", "rbswap"]:
        return map_hw
    if v.kind == "xflip":
        return torch.flip(map_hw, dims=[1])
    if v.kind == "yflip":
        return torch.flip(map_hw, dims=[0])
    if v.kind == "rot90":   # inverse of rot90 is rot270
        return torch.rot90(map_hw, k=1, dims=(0, 1))  # check orientation; may need k=3 depending on PIL direction
    if v.kind == "rot180":
        return torch.rot90(map_hw, k=2, dims=(0, 1))
    if v.kind == "rot270":  # inverse of rot270 is rot90
        return torch.rot90(map_hw, k=3, dims=(0, 1))
    return map_hw


# ---------------------------------------------------------------------------
#  Support augmentation (Support Enhancement)
# ---------------------------------------------------------------------------

class SupportAugConfig:
    def __init__(
        self,
        rotation_deg: float = 15.0,
        rotation_p: float = 0.7,
        translate: float = 0.08,
        hflip_p: float = 0.5,
    ):
        self.rotation_deg = float(rotation_deg)
        self.rotation_p = float(rotation_p)
        self.translate = float(translate)
        self.hflip_p = float(hflip_p)

def augment_support(img: Image.Image, cfg: SupportAugConfig) -> Image.Image:
    out = img
    ROT_SET = [0, 45, 90, 135, 180, 225, 270]  # deterministic pose coverage
    if getattr(cfg, "rot90_set", True):
        ang = random.choice(ROT_SET)
        out = out.rotate(ang, resample=Image.BICUBIC, fillcolor=(0,0,0))
    if cfg.rotation_deg > 0 and random.random() < cfg.rotation_p:
        ang = random.uniform(-cfg.rotation_deg, cfg.rotation_deg)
        out = out.rotate(ang, resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    if cfg.translate > 0:
        w, h = out.size
        dx = random.uniform(-cfg.translate, cfg.translate) * w
        dy = random.uniform(-cfg.translate, cfg.translate) * h
        out = out.transform(
            (w, h),
            Image.AFFINE,
            (1.0, 0.0, dx, 0.0, 1.0, dy),
            resample=Image.BICUBIC,
            fillcolor=(0, 0, 0),
        )
    if cfg.hflip_p > 0 and random.random() < cfg.hflip_p:
        out = ImageOps.mirror(out)
    return out


# ---------------------------------------------------------------------------
#  Memory banks (Category-indexed)
# ---------------------------------------------------------------------------

class VisionADMemory:
    """
    Stores category-indexed patch memory bank M_P and global bank M_G (per view).

    For each view:
      - global_feats: [N_global, Cg]
      - global_labels: [N_global] (class index)
      - patch_feats_by_class: dict[class_idx] -> [N_patches_total, D]
    """
    def __init__(self, n_classes: int, views: List[ViewSpec], device: torch.device, dtype: torch.dtype):
        self.n_classes = int(n_classes)
        self.views = views
        self.device = device
        self.dtype = dtype
        self.global_feats: Dict[str, torch.Tensor] = {}
        self.global_labels: Dict[str, torch.Tensor] = {}
        self.patch_feats_by_class: Dict[str, Dict[int, torch.Tensor]] = {}

    def save(self, path: str, class_names: List[str], meta: dict):
        safe_makedirs(os.path.dirname(path) or ".")
        payload = {
            "class_names": class_names,
            "views": [(v.name, v.kind, v.posclamp_low) for v in self.views],
            "meta": meta,
            "global_feats": {k: v.cpu() for k, v in self.global_feats.items()},
            "global_labels": {k: v.cpu() for k, v in self.global_labels.items()},
            "patch_feats_by_class": {
                vn: {int(ci): t.cpu() for ci, t in d.items()}
                for vn, d in self.patch_feats_by_class.items()
            },
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, device: torch.device) -> Tuple["VisionADMemory", List[str], dict]:
        payload = torch.load(path, map_location="cpu")
        class_names = payload["class_names"]
        views = [ViewSpec(name=n, kind=k, posclamp_low=int(pl)) for (n, k, pl) in payload["views"]]
        meta = payload.get("meta", {})
        mem = VisionADMemory(n_classes=len(class_names), views=views, device=device, dtype=torch.float16)
        mem.global_feats = {k: v.to(device) for k, v in payload["global_feats"].items()}
        mem.global_labels = {k: v.to(device) for k, v in payload["global_labels"].items()}
        mem.patch_feats_by_class = {
            vn: {int(ci): t.to(device) for ci, t in d.items()}
            for vn, d in payload["patch_feats_by_class"].items()
        }
        return mem, class_names, meta


# ---------------------------------------------------------------------------
#  VisionAD inference (Eq.3–6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def retrieve_category(global_q: torch.Tensor, global_bank: torch.Tensor, labels_bank: torch.Tensor) -> int:
    """
    Eq.(3): category retrieval by nearest global token.
    """
    sims = global_bank @ global_q  # [N]
    idx = int(torch.argmax(sims).item())
    return int(labels_bank[idx].item())

@torch.no_grad()
def patch_nn_anomaly_map(
    patches_q: torch.Tensor,     # [Nq, D] normalized
    patches_ref: torch.Tensor,   # [Nr, D] normalized
    grid_hw: Tuple[int, int],
    out_hw: Tuple[int, int],
    knn: int = 1,
    chunk: int = 4096,
) -> torch.Tensor:
    """
    Eq.(4): score per patch: 1 - max_j sim(q_i, m_j), then upsample to pixel map.

    Returns: [H,W] float32 on current device.
    """
    Nq, D = patches_q.shape
    Nr = patches_ref.shape[0]

    max_sim = torch.full((Nq,), -1e9, device=patches_q.device, dtype=torch.float32)
    q = patches_q.float()

    for j0 in range(0, Nr, chunk):
        ref = patches_ref[j0:j0+chunk].float()   # [c,D]
        sims = q @ ref.t()                       # [Nq,c]
        max_sim = torch.maximum(max_sim, sims.max(dim=1).values)

    if knn <= 1:
        score_patch = 1.0 - torch.clamp(max_sim, -1.0, 1.0)
    else:
        # maintain running top-k sims per query patch
        topk = torch.full((Nq, knn), -1e9, device=patches_q.device, dtype=torch.float32)
        q = patches_q.float()
        for j0 in range(0, Nr, chunk):
            ref = patches_ref[j0:j0+chunk].float()
            sims = q @ ref.t()  # [Nq, c]
            cand = torch.topk(sims, k=min(knn, sims.shape[1]), dim=1).values
            topk = torch.topk(torch.cat([topk, cand], dim=1), k=knn, dim=1).values
        sim_agg = topk.mean(dim=1)
        score_patch = 1.0 - torch.clamp(sim_agg, -1.0, 1.0)

    Ht, Wt = grid_hw
    if Ht * Wt != Nq:
        Ht, Wt = 1, Nq

    # if Ht * Wt != Nq:
    #     t = int(round(math.sqrt(Nq)))
    #     if t * t == Nq:
    #         Ht = Wt = t
    #     else:
    #         Ht, Wt = 1, Nq
    score_map = score_patch.view(1, 1, Ht, Wt)
    score_up = F.interpolate(score_map, size=out_hw, mode="bilinear", align_corners=False)
    return score_up[0, 0].contiguous()

@torch.no_grad()
def image_score_top1pct(anom_map: torch.Tensor) -> float:
    """
    Eq.(6): mean of top 1% pixels.
    """
    v = anom_map.flatten()
    k = max(1, int(round(0.01 * v.numel())))
    return float(torch.topk(v, k=k, largest=True).values.mean().item())


class VisionADRunner:
    def __init__(
        self,
        backbone: DINOv2MultiLayerBackbone,
        class_names: List[str],
        views: List[ViewSpec],
        support_aug_cfg: SupportAugConfig,
        shots: int = 4,
        aug_per_image: int = 4,
        device: str = "cuda",
        bank_dtype: torch.dtype = torch.float16,
        oracle_class: bool = False,
        nn_chunk: int = 4096,
        layer_fuse: str = "concat"
    ):
        self.backbone = backbone
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.views = views
        self.support_aug_cfg = support_aug_cfg
        self.shots = int(shots)
        self.aug_per_image = int(aug_per_image)
        self.device = torch.device(device)
        self.bank_dtype = bank_dtype if self.device.type == "cuda" else torch.float32
        self.oracle_class = bool(oracle_class)
        self.nn_chunk = int(nn_chunk)

        self.out_hw = (self.backbone.img_size, self.backbone.img_size)
        ps = 14
        self.grid_hw = (self.backbone.img_size // ps, self.backbone.img_size // ps)
        self.layer_fuse = layer_fuse

    @torch.no_grad()
    def build_memory(
        self,
        support_samples: List[ImageSample],
        cache_path: Optional[str] = None,
        rebuild: bool = False,
    ) -> VisionADMemory:
        if cache_path and (not rebuild) and os.path.isfile(cache_path):
            mem, cls_names, _ = VisionADMemory.load(cache_path, self.device)
            if cls_names == self.class_names:
                return mem

        mem = VisionADMemory(n_classes=len(self.class_names), views=self.views, device=self.device, dtype=self.bank_dtype)

        # select K shots per class (deterministic)
        by_class: Dict[str, List[ImageSample]] = {c: [] for c in self.class_names}
        for s in support_samples:
            if s.is_good and s.cls_name in by_class:
                by_class[s.cls_name].append(s)
        for c in by_class:
            by_class[c] = sorted(by_class[c], key=lambda x: x.path)[:self.shots]

        # accumulators on CPU first to avoid GPU fragmentation during bank build
        g_feats: Dict[str, List[torch.Tensor]] = {v.name: [] for v in self.views}
        g_labs: Dict[str, List[int]] = {v.name: [] for v in self.views}
        p_feats: Dict[str, Dict[int, List[torch.Tensor]]] = {v.name: {i: [] for i in range(len(self.class_names))}
                                                             for v in self.views}

        for cls in self.class_names:
            cls_idx = self.class_to_idx[cls]
            shots = by_class[cls]
            if len(shots) == 0:
                continue

            for shot in shots:
                base = Image.open(shot.path).convert("RGB")
                refs: List[Image.Image] = [base]
                for _ in range(max(0, self.aug_per_image)):
                    refs.append(augment_support(base, self.support_aug_cfg))

                for v in self.views:
                    v_imgs = [apply_view_pil(im, v) for im in refs]
                    patch_layers, g = self.backbone.encode_batch(v_imgs)  # list[B,N,C], [B,C]
                    fused = fuse_layers_one_group(patch_layers, mode=self.layer_fuse)           # [B,N,D]

                    g_feats[v.name].append(g.cpu())
                    g_labs[v.name].extend([cls_idx] * g.shape[0])
                    p_feats[v.name][cls_idx].append(fused.cpu())

        for v in self.views:
            vn = v.name
            if len(g_feats[vn]) == 0:
                raise RuntimeError(f"No support features collected for view={vn}")

            G = torch.cat(g_feats[vn], dim=0)                      # [Ng,C]
            L = torch.tensor(g_labs[vn], dtype=torch.long)

            mem.global_feats[vn] = F.normalize(G, dim=-1).to(self.device, dtype=torch.float32)
            mem.global_labels[vn] = L.to(self.device)

            mem.patch_feats_by_class[vn] = {}
            for ci in range(len(self.class_names)):
                if len(p_feats[vn][ci]) == 0:
                    continue
                P = torch.cat(p_feats[vn][ci], dim=0)              # [Nr,N,D]
                P = P.reshape(-1, P.shape[-1])                     # [Nr*N,D]
                mem.patch_feats_by_class[vn][ci] = F.normalize(P, dim=-1).to(self.device, dtype=self.bank_dtype)

        if cache_path:
            meta = {
                "shots": self.shots,
                "aug_per_image": self.aug_per_image,
                "views": [(v.name, v.kind, v.posclamp_low) for v in self.views],
                "model_name": self.backbone.model_name,
                "layers": self.backbone.layers,
                "img_size": self.backbone.img_size,
                "resize_size": self.backbone.resize_size,
            }
            mem.save(cache_path, self.class_names, meta)

        return mem

    @torch.no_grad()
    def infer_one(self, mem: VisionADMemory, img_path: str, cls_oracle: Optional[str] = None) -> Tuple[float, torch.Tensor, int]:
        img = Image.open(img_path).convert("RGB")

        total_map = None
        pred_class = -1

        for v in self.views:
            img_v = apply_view_pil(img, v)
            patch_layers, g = self.backbone.encode_batch([img_v])
            fused = fuse_layers_one_group(patch_layers, mode=self.layer_fuse)[0].to(self.device)  # [N,D]
            gq = g[0].to(self.device)

            if self.oracle_class and cls_oracle is not None:
                ci = self.class_to_idx[cls_oracle]
            else:
                ci = retrieve_category(gq, mem.global_feats[v.name], mem.global_labels[v.name])

            if pred_class < 0:
                pred_class = ci

            ref = mem.patch_feats_by_class[v.name].get(ci, None)
            if ref is None or ref.numel() == 0:
                anom_map = torch.zeros(self.out_hw, device=self.device)
            else:
                knn = int(getattr(self, "knn", 1))
                anom_map = patch_nn_anomaly_map(fused, ref, self.grid_hw, self.out_hw, knn=knn, chunk=self.nn_chunk)
                #anom_map = patch_nn_anomaly_map(fused, ref, self.grid_hw, self.out_hw, chunk=self.nn_chunk)

            anom_map = align_map_to_identity(anom_map, v)
            total_map = anom_map if total_map is None else (total_map + anom_map)  # Eq.(5): sum fusion

        return image_score_top1pct(total_map), total_map.detach().cpu(), int(pred_class)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def build_views(spec: str, posclamp_low: int) -> List[ViewSpec]:
    kinds = [s.strip() for s in str(spec).split(",") if s.strip()]
    out: List[ViewSpec] = []
    for k in kinds:
        if k == "id":
            out.append(ViewSpec("id", "id"))
        elif k == "posclamp":
            out.append(ViewSpec("posclamp", "posclamp", posclamp_low=posclamp_low))
        elif k == "xflip":
            out.append(ViewSpec("xflip", "xflip"))
        elif k == "yflip":
            out.append(ViewSpec("yflip", "yflip"))
        elif k == "rbswap":
            out.append(ViewSpec("rbswap", "rbswap"))
        elif k == "rot90":  out.append(ViewSpec("rot90", "rot90"))
        elif k == "rot180": out.append(ViewSpec("rot180", "rot180"))
        elif k == "rot270": out.append(ViewSpec("rot270", "rot270"))
        else:
            raise ValueError(f"Unknown view: {k}")
    out = sorted(out, key=lambda v: 0 if v.kind == "id" else 1)
    return out


def main():
    p = argparse.ArgumentParser("VisionAD faithful baseline (few-shot anomaly detection)")
    p.add_argument("--dataset", type=str, required=True, choices=["mvtec", "visa", "realiad"])
    p.add_argument("--root", type=str, required=True)

    # VisA
    p.add_argument("--csv", type=str, default="", help="VisA single CSV path (contains split column)")
    # Real-IAD
    p.add_argument("--realiad_json_dir", type=str, default="")

    # common
    p.add_argument("--class_name", type=str, default="", help="Run only this class/category")
    p.add_argument("--shots", type=int, default=1)
    p.add_argument("--aug_per_image", type=int, default=4)
    p.add_argument("--views", type=str, default="id,posclamp,yflip")
    p.add_argument("--posclamp_low", type=int, default=64)
    p.add_argument("--knn", type=int, default=1, help="kNN for patch matching (1 = max sim)")
    p.add_argument("--num_seeds", type=int, default=1, help="repeat evaluation with different support sampling seeds")
    p.add_argument("--layer_fuse", type=str, default="concat", choices=["concat", "sum"])

    # backbone
    p.add_argument("--model_name", type=str, default="dinov2_vitl14_reg")
    p.add_argument("--layers", type=str, default="", help="Comma list of block indices; empty -> auto")
    p.add_argument("--img_size", type=int, default=392)
    p.add_argument("--resize_size", type=int, default=448)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_fp16", action="store_true")

    # support enhancement
    p.add_argument("--rotation_deg", type=float, default=15.0)
    p.add_argument("--rotation_p", type=float, default=0.7)
    p.add_argument("--translate", type=float, default=0.08)
    p.add_argument("--hflip_p", type=float, default=0.5)

    # inference
    p.add_argument("--oracle_class", action="store_true", help="Use known class label (debug only)")
    p.add_argument("--nn_chunk", type=int, default=4096)

    # caching / output
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default="./cache_visionad")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--out_csv", type=str, default="", help="Optional path to write per-sample scores CSV")

    args = p.parse_args()
    set_seed(int(args.seed))

    only_cls = [args.class_name] if args.class_name else None

    # dataset index
    if args.dataset == "mvtec":
        idx = IndustrialDatasetIndex(args.root, only_classes=only_cls)
    elif args.dataset == "visa":
        if not args.csv:
            raise ValueError("For --dataset visa, provide --csv (single VisA CSV with split column)")
        idx = VisACSVIndex(args.root, args.csv, only_classes=only_cls)
    else:
        if not args.realiad_json_dir:
            raise ValueError("For --dataset realiad, provide --realiad_json_dir")
        idx = RealIADJSONIndex(args.root, args.realiad_json_dir, only_classes=only_cls)

    class_names = idx.classes
    if len(class_names) == 0:
        raise RuntimeError("No classes found. Check dataset paths.")

    layers = parse_int_list(args.layers)
    backbone = DINOv2MultiLayerBackbone(
        model_name=args.model_name,
        layers=layers,
        img_size=int(args.img_size),
        resize_size=int(args.resize_size),
        device=args.device,
        use_fp16=(not args.no_fp16),
    )

    views = build_views(args.views, posclamp_low=int(args.posclamp_low))
    aug_cfg = SupportAugConfig(
        rotation_deg=float(args.rotation_deg),
        rotation_p=float(args.rotation_p),
        translate=float(args.translate),
        hflip_p=float(args.hflip_p),
    )

    runner = VisionADRunner(
        backbone=backbone,
        class_names=class_names,
        views=views,
        support_aug_cfg=aug_cfg,
        shots=int(args.shots),
        aug_per_image=int(args.aug_per_image),
        device=args.device,
        bank_dtype=torch.float16,
        oracle_class=bool(args.oracle_class),
        nn_chunk=int(args.nn_chunk),
        layer_fuse=args.layer_fuse
    )

    runner.knn = int(args.knn)
    runner.seed = int(args.seed)
    cache_dir = args.cache_dir
    safe_makedirs(cache_dir)
    cache_key = (
        f"{args.dataset}_{args.model_name}_R{args.resize_size}_C{args.img_size}_"
        f"layers{'-'.join(map(str, backbone.layers))}_K{args.shots}_aug{args.aug_per_image}_"
        f"views{args.views.replace(',','-')}"
    )
    cache_path = os.path.join(cache_dir, cache_key + ".pt")

    support_normals = [s for s in idx.train_samples if s.is_good]

    seed_aurocs = []          # mean AUROC over classes per seed (or the single class if --class_name is set)
    seed_class_auroc = {}     # optional: per-class AUROC trajectories

    # iterate seeds
    base_seed = int(args.seed)
    num_seeds = int(args.num_seeds)
    if num_seeds <= 0:
        num_seeds = 1

    for si in range(base_seed, base_seed + num_seeds):
        print(f"\n================ Seed {si} ================\n")
        set_seed(si)
        runner.seed = int(si)

        # IMPORTANT: use a seed-specific cache key so we don't accidentally reuse banks
        cache_key_seed = cache_key + f"_seed{si}"
        cache_path_seed = os.path.join(cache_dir, cache_key_seed + ".pt")

        # rebuild if user asks, OR if num_seeds>1 (because we want different support draws)
        rebuild_this = bool(args.rebuild_cache) or (num_seeds > 1)

        mem = runner.build_memory(
            support_normals,
            cache_path=cache_path_seed,
            rebuild=rebuild_this,
        )

        per_class_scores: Dict[str, List[float]] = {c: [] for c in class_names}
        per_class_labels: Dict[str, List[int]] = {c: [] for c in class_names}

        rows = []
        for s in idx.test_samples:
            if only_cls and s.cls_name not in only_cls:
                continue
            y = 0 if s.is_good else 1
            score, _, pred_ci = runner.infer_one(mem, s.path, cls_oracle=s.cls_name)
            per_class_scores[s.cls_name].append(float(score))
            per_class_labels[s.cls_name].append(int(y))
            if args.out_csv:
                rows.append({
                    "seed": int(si),
                    "path": s.path,
                    "cls_name": s.cls_name,
                    "label": int(y),
                    "score": float(score),
                    "pred_cls": class_names[pred_ci] if pred_ci >= 0 else "",
                })

        # AUROC per class (image-level)
        aurocs = []
        print("\nPer-class image AUROC:")
        for c in class_names:
            ys = per_class_labels[c]
            ss = per_class_scores[c]
            if len(ys) == 0 or len(set(ys)) < 2:
                au = float("nan")
            else:
                au = float(roc_auc_score(ys, ss))
                aurocs.append(au)

            seed_class_auroc.setdefault(c, []).append(au)
            print(f"  {c:>16s}: {format_float(au)}  (N={len(ys)})")

        mean_au = float(np.nanmean(np.asarray(aurocs, dtype=np.float64))) if len(aurocs) > 0 else float("nan")
        seed_aurocs.append(mean_au)
        print(f"\nSeed {si} mean AUROC over classes: {format_float(mean_au)}")

        # write per-sample scores (append per seed)
        if args.out_csv:
            safe_makedirs(os.path.dirname(args.out_csv) or ".")
            # append mode: if file exists, append without header
            df = pd.DataFrame(rows)
            if (si == base_seed) or (not os.path.isfile(args.out_csv)):
                df.to_csv(args.out_csv, index=False)
            else:
                df.to_csv(args.out_csv, index=False, mode="a", header=False)
            print(f"Appended per-sample scores to: {args.out_csv}")

    # Summary across seeds
    arr = np.asarray(seed_aurocs, dtype=np.float64)
    mean_all = float(np.nanmean(arr)) if arr.size > 0 else float("nan")
    std_all = float(np.nanstd(arr)) if arr.size > 0 else float("nan")

    print("\n================ Multi-seed Summary ================\n")
    print(f"Seeds: {list(range(base_seed, base_seed + num_seeds))}")
    print(f"Mean AUROC over classes: {format_float(mean_all)} ± {format_float(std_all)}")

    # If user runs a single class, also print that class's mean±std explicitly
    if args.class_name:
        c = args.class_name
        traj = np.asarray(seed_class_auroc.get(c, []), dtype=np.float64)
        m = float(np.nanmean(traj)) if traj.size > 0 else float("nan")
        s = float(np.nanstd(traj)) if traj.size > 0 else float("nan")
        print(f"Class '{c}' AUROC: {format_float(m)} ± {format_float(s)}")

    backbone.cleanup()


if __name__ == "__main__":
    main()
