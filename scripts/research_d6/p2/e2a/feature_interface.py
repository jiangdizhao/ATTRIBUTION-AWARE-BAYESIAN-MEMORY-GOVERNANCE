"""Shared final-layer spatial-token interface for the two E2 backbones.

The caller supplies tokens from the official model output and the registered
patch-center mask. This module does not load a model or access image data.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _unit_per_token(value: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norms <= np.float32(1e-12)):
        raise ValueError(f"{name} contains a zero or near-zero token")
    return np.ascontiguousarray(value / norms, dtype=np.float32)


def build_spatial_feature_record(
    cls_tokens: Any,
    register_tokens: Any,
    patch_tokens: Any,
    valid_patch_mask: Any,
    *,
    backbone: str,
) -> dict[str, Any]:
    """Return the shared FP32, per-token L2-normalized representation.

    Shapes are batch-first: CLS ``(B,D)``, registers ``(B,R,D)``, patches
    ``(B,N,D)``. A one-dimensional mask ``(N,)`` is shared across the batch;
    a two-dimensional mask must be ``(B,N)``. Masking is for downstream
    pooling only; it does not undo padding-token influence during attention.
    """
    cls = _array(cls_tokens, "cls_tokens", 2)
    registers = _array(register_tokens, "register_tokens", 3)
    patches = _array(patch_tokens, "patch_tokens", 3)
    mask = np.asarray(valid_patch_mask, dtype=bool)
    if cls.shape[0] != registers.shape[0] or cls.shape[0] != patches.shape[0]:
        raise ValueError("CLS, register and patch batch dimensions must match")
    if cls.shape[1] != registers.shape[2] or cls.shape[1] != patches.shape[2]:
        raise ValueError("CLS, register and patch feature dimensions must match")
    if registers.shape[1] != 4:
        raise ValueError("the registered E2 interface expects exactly four register tokens")
    if mask.ndim == 1:
        if mask.shape != (patches.shape[1],):
            raise ValueError("one-dimensional valid_patch_mask must match patch count")
        mask = np.broadcast_to(mask[None, :], (patches.shape[0], patches.shape[1])).copy()
    elif mask.ndim == 2:
        if mask.shape != patches.shape[:2]:
            raise ValueError("two-dimensional valid_patch_mask must match batch and patch count")
    else:
        raise ValueError("valid_patch_mask must have one or two dimensions")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("each image must have at least one valid patch")

    return {
        "schema": "d6.e2.shared_spatial_tokens.v1",
        "backbone": backbone,
        "source_layer": "official final-layer normalized outputs (model LayerNorm); DINOv3 last_hidden_state",
        "token_l2_normalization": "independent per CLS/register/patch token in FP32",
        "cls_unit": _unit_per_token(cls, "cls_tokens"),
        "register_unit": _unit_per_token(registers, "register_tokens"),
        "patch_unit": _unit_per_token(patches, "patch_tokens"),
        "valid_patch_mask": np.ascontiguousarray(mask, dtype=bool),
        "padding_attention_note": "valid_patch_mask only gates downstream spatial pooling; padding tokens may already influence attention",
    }


def dinov2_to_shared_interface(output: dict[str, Any], valid_patch_mask: Any) -> dict[str, Any]:
    """Adapt official ``forward_features`` output from DINOv2 ViT-L/14 reg."""
    required = ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens")
    missing = [key for key in required if key not in output]
    if missing:
        raise KeyError(f"DINOv2 forward_features output is missing: {missing}")
    return build_spatial_feature_record(
        output[required[0]], output[required[1]], output[required[2]], valid_patch_mask,
        backbone="dinov2_vitl14_reg",
    )


def dinov3_to_shared_interface(last_hidden_state: Any, valid_patch_mask: Any) -> dict[str, Any]:
    """Adapt HF DINOv3 ``last_hidden_state`` ordered CLS, 4 regs, patches."""
    tokens = _array(last_hidden_state, "last_hidden_state", 3)
    if tokens.shape[1] < 6:
        raise ValueError("DINOv3 sequence must contain CLS, four registers and patches")
    return build_spatial_feature_record(
        tokens[:, :1, :].reshape(tokens.shape[0], tokens.shape[2]),
        tokens[:, 1:5, :], tokens[:, 5:, :], valid_patch_mask,
        backbone="dinov3_vitl16_pretrain_lvd1689m",
    )
