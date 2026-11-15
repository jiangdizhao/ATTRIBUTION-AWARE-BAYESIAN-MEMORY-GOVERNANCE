"""Deterministic RGB EXIF-aware 448 letterbox preprocessing for E2-A1."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from contracts import (EXIF_POLICY, INPUT_SIZE, MEAN, PATCH_SIZES, PILLOW_POLICY, STD,
                       PREPROCESS_SCHEMA, PREPROCESS_VERSION, ROUNDING_POLICY)


def round_half_up_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("round_half_up_ratio requires numerator >= 0 and denominator > 0")
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder >= denominator)


def patch_center_mask(content_rect_xyxy: tuple[int, int, int, int], patch_size: int) -> np.ndarray:
    left, top, right, bottom = content_rect_xyxy
    if patch_size <= 0 or INPUT_SIZE % patch_size:
        raise ValueError("input size must be divisible by patch size")
    if not (0 <= left < right <= INPUT_SIZE and 0 <= top < bottom <= INPUT_SIZE):
        raise ValueError(f"Invalid content rectangle: {content_rect_xyxy}")
    grid = INPUT_SIZE // patch_size
    centers = (np.arange(grid, dtype=np.float64) + 0.5) * patch_size
    mask = ((centers[:, None] >= top) & (centers[:, None] < bottom)
            & (centers[None, :] >= left) & (centers[None, :] < right))
    flat = np.ascontiguousarray(mask.reshape(-1), dtype=np.bool_)
    if not flat.any():
        raise ValueError(f"No valid patch centers for patch size {patch_size} and content {content_rect_xyxy}")
    return flat


def image_to_canonical(rgb_image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
    """Resize an already EXIF-transposed RGB Pillow image to the shared tensor."""
    if rgb_image.mode != "RGB":
        raise ValueError(f"Unsupported color mode {rgb_image.mode!r}; only RGB is registered")
    width, height = rgb_image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid source dimensions: {rgb_image.size}")
    if width >= height:
        resized_width = INPUT_SIZE
        resized_height = round_half_up_ratio(height * INPUT_SIZE, width)
    else:
        resized_height = INPUT_SIZE
        resized_width = round_half_up_ratio(width * INPUT_SIZE, height)
    if not (1 <= resized_width <= INPUT_SIZE and 1 <= resized_height <= INPUT_SIZE):
        raise ValueError(f"Rounded resized dimensions are invalid: {(resized_width, resized_height)}")
    resized = rgb_image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
        reducing_gap=None,
    )
    resized_rgb = np.asarray(resized, dtype=np.uint8)
    left = (INPUT_SIZE - resized_width) // 2
    top = (INPUT_SIZE - resized_height) // 2
    right_pad = INPUT_SIZE - resized_width - left
    bottom_pad = INPUT_SIZE - resized_height - top
    rect = (left, top, left + resized_width, top + resized_height)
    rgb = resized_rgb.astype(np.float32) / np.float32(255.0)
    mean = np.asarray(MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (rgb - mean) / std
    tensor = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    tensor[0, :, top:top + resized_height, left:left + resized_width] = np.transpose(normalized, (2, 0, 1))
    masks = {f"mask_{name}": patch_center_mask(rect, patch) for name, patch in PATCH_SIZES.items()}
    metadata = {
        "schema": PREPROCESS_SCHEMA,
        "preprocess_version": PREPROCESS_VERSION,
        "source_width_after_exif": width,
        "source_height_after_exif": height,
        "source_mode_after_exif": rgb_image.mode,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "content_rect_xyxy": list(rect),
        "padding_ltrb": [left, top, right_pad, bottom_pad],
        "rounding_policy": ROUNDING_POLICY,
        "exif_policy": EXIF_POLICY,
        "resize_policy": PILLOW_POLICY,
        "resize_crop": "no crop",
        "normalization": "RGB float32 [0,1]; (x-ImageNet mean)/std; FP32",
        "mean_rgb": list(MEAN),
        "std_rgb": list(STD),
        "padding_after_normalization": 0.0,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": tensor.dtype.name,
        "valid_patch_counts": {key: int(value.sum()) for key, value in masks.items()},
        "mask_shapes": {key: list(value.shape) for key, value in masks.items()},
    }
    return np.ascontiguousarray(tensor), {**metadata, **masks}


def decode_and_preprocess(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode exactly one authorized file, applying its EXIF orientation first."""
    try:
        with Image.open(path) as source:
            original_mode = source.mode
            original_size = list(source.size)
            exif_orientation = int(source.getexif().get(274, 1))
            oriented = ImageOps.exif_transpose(source)
            if oriented.mode != "RGB":
                raise ValueError(f"Unsupported color mode {oriented.mode!r}; only RGB is registered")
            tensor, metadata = image_to_canonical(oriented)
    except Exception as exc:
        raise RuntimeError(f"Image decode/preprocess blocked for registered source: {type(exc).__name__}: {exc}") from exc
    metadata.update({
        "source_width_before_exif": int(original_size[0]),
        "source_height_before_exif": int(original_size[1]),
        "source_mode_before_exif": original_mode,
        "exif_orientation_tag": exif_orientation,
    })
    return tensor, metadata


def canonical_array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()
