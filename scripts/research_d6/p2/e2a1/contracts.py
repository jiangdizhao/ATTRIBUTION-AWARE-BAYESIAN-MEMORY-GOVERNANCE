"""Frozen identities and paths for LP-D6-P2-E2-A1."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/research_d6/p2/e2a1"
EXPECTED_HEAD = "6da731f2ecf93fd23e6c51e6f66639f2528b8d57"
EXPECTED_BRANCH = "stage3-eg-famar-execution"

INPUT_MANIFEST_SHA256 = "820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63"
P1A_FEATURE_MANIFEST_SHA256 = "ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d"
WX_TAPE_SHA256 = "13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1"
FEATURE_INTERFACE_SCHEMA = "d6.e2.shared_spatial_tokens.v1"
PREPROCESS_SCHEMA = "d6.e2a1.canonical_rgb_letterbox.v1"
PREPROCESS_VERSION = "LP-D6-P2-E2-A1-PP-v1"

DINO2_INTERPRETER = Path(r"D:\anaconda3\envs\torch\python.exe")
DINO2_MODEL = "dinov2_vitl14_reg"
DINO2_CHECKPOINT_SHA256 = "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
DINO2_SOURCE_TREE_SHA256 = "7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c"

DINO3_INTERPRETER = Path(r"D:\anaconda3\envs\dinov3\python.exe")
DINO3_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO3_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
DINO3_WEIGHTS_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
DINO3_CONFIG_SHA256 = "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
DINO3_PROCESSOR_SHA256 = "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7"

INPUT_SIZE = 448
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PATCH_SIZES = {"dinov2": 14, "dinov3": 16}
GPU_MIN_FREE_BYTES = 10 * 1024**3
ALLOCATOR_CAP_BYTES = 12 * 1024**3
MAX_RUN_SECONDS = 2 * 60 * 60
OUTPUT_CAP_BYTES = int(1.5 * 1024**3)

PILLOW_POLICY = (
    "Pillow 9.3.0; RGB Image.resize with Resampling.BILINEAR and reducing_gap=None "
    "(single pass, fixed Pillow filter behavior; no additional resize or crop)"
)
ROUNDING_POLICY = "positive rational dimensions use nearest integer, exact halves round up"
EXIF_POLICY = "apply Pillow ImageOps.exif_transpose before requiring RGB mode and measuring dimensions"
