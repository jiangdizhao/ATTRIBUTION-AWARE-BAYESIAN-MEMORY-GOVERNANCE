"""Create one synthetic RGB example and the shared E2-A0 canonical tensor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "outputs/research_d6/p2/e2a0"
INPUT_SIZE = 448
SOURCE_WIDTH = 320
SOURCE_HEIGHT = 240
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode("ascii"))
    h.update(repr(value.shape).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def synthetic_rgb() -> np.ndarray:
    """Generate a fixed background with clear, deterministic line structures."""
    rgb = np.empty((SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)
    rgb[:] = (225, 219, 202)
    rgb[34:39, 20:300] = (31, 76, 133)
    rgb[92:96, 14:308] = (172, 53, 41)
    rgb[158:164, 28:291] = (45, 121, 79)
    rgb[204:208, 42:278] = (93, 62, 132)
    rgb[20:220, 56:60] = (28, 28, 28)
    rgb[25:214, 251:256] = (20, 126, 145)
    yy, xx = np.mgrid[0:SOURCE_HEIGHT, 0:SOURCE_WIDTH]
    diagonal_1 = np.abs((yy - 28) - 0.52 * (xx - 36)) <= 1.5
    diagonal_2 = np.abs((yy - 212) + 0.34 * (xx - 40)) <= 1.5
    rgb[diagonal_1] = (183, 104, 20)
    rgb[diagonal_2] = (35, 35, 35)
    return rgb


def patch_center_mask(patch_size: int, content_rect_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = content_rect_xyxy
    rows = np.arange(INPUT_SIZE // patch_size, dtype=np.float64)
    cols = np.arange(INPUT_SIZE // patch_size, dtype=np.float64)
    cy = (rows + 0.5) * patch_size
    cx = (cols + 0.5) * patch_size
    return ((cy[:, None] >= top) & (cy[:, None] < bottom) &
            (cx[None, :] >= left) & (cx[None, :] < right))


def build_canonical() -> dict[str, Any]:
    source = synthetic_rgb()
    source_image = Image.fromarray(source, mode="RGB")
    resized_width = INPUT_SIZE
    resized_height = SOURCE_HEIGHT * INPUT_SIZE // SOURCE_WIDTH
    resized = source_image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    resized_rgb = np.asarray(resized, dtype=np.uint8)
    top = (INPUT_SIZE - resized_height) // 2
    bottom_pad = INPUT_SIZE - resized_height - top
    content_rect = (0, top, INPUT_SIZE, top + resized_height)

    values = resized_rgb.astype(np.float32) / np.float32(255.0)
    values = (values - MEAN[None, None, :]) / STD[None, None, :]
    tensor = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    tensor[0, :, top:top + resized_height, :] = np.transpose(values, (2, 0, 1))

    mask_v2 = patch_center_mask(14, content_rect)
    mask_v3 = patch_center_mask(16, content_rect)
    if mask_v2.shape != (32, 32) or int(mask_v2.sum()) != 768:
        raise AssertionError("DINOv2 patch-center mask did not produce registered 768 valid cells")
    if mask_v3.shape != (28, 28) or int(mask_v3.sum()) != 588:
        raise AssertionError("DINOv3 patch-center mask did not produce registered 588 valid cells")

    return {
        "source_rgb": source,
        "source_png_image": source_image,
        "tensor": np.ascontiguousarray(tensor),
        "mask_v2": np.ascontiguousarray(mask_v2.reshape(-1)),
        "mask_v3": np.ascontiguousarray(mask_v3.reshape(-1)),
        "metadata": {
            "schema": "d6.e2a0.canonical_input.v1",
            "input_kind": "synthetic_only_no_research_images",
            "source_width": SOURCE_WIDTH,
            "source_height": SOURCE_HEIGHT,
            "source_mode": "RGB",
            "resized_width": resized_width,
            "resized_height": resized_height,
            "valid_content_rect_xyxy": list(content_rect),
            "padding_pixels_ltrb": [0, top, 0, bottom_pad],
            "resize": "Pillow Image.Resampling.BILINEAR; longest side 448; antialias as provided by Pillow resize",
            "crop": "none",
            "normalization": "RGB float32 [0,1], then (x-ImageNet_mean)/ImageNet_std",
            "mean_rgb": MEAN.tolist(),
            "std_rgb": STD.tolist(),
            "padding_after_normalization": 0.0,
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": tensor.dtype.name,
            "canonical_array_sha256": array_sha256(tensor),
            "dino_v2_patch_grid": [32, 32],
            "dino_v2_patch_mask_count": int(mask_v2.sum()),
            "dino_v3_patch_grid": [28, 28],
            "dino_v3_patch_mask_count": int(mask_v3.sum()),
            "patch_center_rule": "center x/y lies in half-open resized content rectangle; row-major flattening",
            "pillow_version": Image.__version__ if hasattr(Image, "__version__") else __import__("PIL").__version__,
            "numpy_version": np.__version__,
            "preprocessing_code_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_names = [
        "canonical_source.png", "canonical_tensor.npy", "dino_v2_patch_mask.npy",
        "dino_v3_patch_mask.npy", "canonical_manifest.json",
    ]
    if any((OUT_DIR / name).exists() for name in expected_names):
        raise FileExistsError("Canonical artifacts already exist; refusing to overwrite E2-A0 output")
    result = build_canonical()
    png_path = OUT_DIR / "canonical_source.png"
    result["source_png_image"].save(png_path, format="PNG", optimize=False)
    tensor_path = OUT_DIR / "canonical_tensor.npy"
    np.save(tensor_path, result["tensor"], allow_pickle=False)
    v2_path = OUT_DIR / "dino_v2_patch_mask.npy"
    np.save(v2_path, result["mask_v2"], allow_pickle=False)
    v3_path = OUT_DIR / "dino_v3_patch_mask.npy"
    np.save(v3_path, result["mask_v3"], allow_pickle=False)
    manifest = dict(result["metadata"])
    manifest["source_png_sha256"] = sha256_file(png_path)
    manifest["canonical_tensor_file_sha256"] = sha256_file(tensor_path)
    manifest["dino_v2_mask_file_sha256"] = sha256_file(v2_path)
    manifest["dino_v3_mask_file_sha256"] = sha256_file(v3_path)
    (OUT_DIR / "canonical_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
