"""Frozen LP-D6-P2-E2-B identities and registered numerical settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPECTED_HEAD = "1e6a602cb8b788bca03a61773ef442a55f3fa7e2"
EXPECTED_BRANCH = "stage3-eg-famar-execution"

E2A1_ROOT = ROOT / "outputs/research_d6/p2/e2a1"
OUTPUT_ROOT = ROOT / "outputs/research_d6/p2/e2b"
RUN_DIR = OUTPUT_ROOT / "run_20261116_01"

EXPECTED = {
    "canonical_manifest_sha256": "0fca669b4f69ed2071b4f4c0ce8f60728c6bd0e6efe126be580f23037b748133",
    "dinov2_feature_manifest_sha256": "3588bab021a9cd8043d663c1d6ce46f366f89f89802fb1ac928dfae3360efddf",
    "dinov3_feature_manifest_sha256": "20e100671344422779d848f7664cdb6290d7fe4cd0e1ed1ea09f108fe0a01887",
    "p1a_input_manifest_sha256": "820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63",
    "p1a_feature_manifest_sha256": "ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d",
    "wx_tape_sha256": "13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1",
    "p1b_joined_source_metadata_sha256": "c814ee3fb8a91ae845c6d7d5d43bdf102b8f30712ff41a0c0fa57d3a0fe96f4b",
}

TAPE_PATH = ROOT / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/p1_gate0_draft/D6_P1_WX_role_map_feedback_tape.csv"
P1A_INPUT_MANIFEST = ROOT / "outputs/research_d6/p1a_features/input_manifest.csv"
P1A_FEATURE_MANIFEST = ROOT / "outputs/research_d6/p1a_features/feature_manifest.csv"
P1B_JOINED_METADATA = ROOT / "outputs/research_d6/p1b/run_20261113_01/joined_human_annotations.csv"

ANNOTATION_SOURCES = {
    "D4-Lite": (ROOT / "research_pilot/LP_W1_D4LITE/annotation_queue.csv",
                ROOT / "research_pilot/LP_W1_D4LITE/annotations.csv"),
    "D4B": (ROOT / "research_pilot/LP_W1_D4B/annotation_queue.csv",
            ROOT / "research_pilot/LP_W1_D4B/annotations.csv"),
    "D-dev": (ROOT / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/annotation_queue.csv",
              ROOT / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/annotations.csv"),
}

FEATURE_SCHEMA = "d6.e2.shared_spatial_tokens.v1"
VALID_STATES = {
    "present": 1,
    "absent_assessable": 0,
    "unobservable_or_uncertain": None,
    "not_applicable": None,
}

BACKBONES = ("dinov2", "dinov3")
GLOBAL_READOUTS = ("G-CLS", "G-Mean", "G-Concat")
MIL_READOUTS = ("M-BCE", "M-Rank-Softplus")
MIL_SEEDS = (0, 1)

LAMBDA = 1e-4
TAU = 1.0
MARGIN = 1.0
KAPPA = 0.1
RHO = 0.1
MIL_INIT_WEIGHT_STD = "1/sqrt(D)"
MIL_INIT_BIAS = 0.0
SOLVER = {
    "method": "L-BFGS-B",
    "ftol": 1e-12,
    "gtol": 1e-8,
    "maxiter": 10_000,
    "maxls": 50,
    "maxcor": 10,
    "accept_gradient_inf": 1e-6,
}
THREAD_LIMIT = 2
PROFILE_BACKBONE = "dinov2"
PROFILE_METHOD = "M-BCE"
PROFILE_SEED = 0

D4LITE = ROOT / "research_pilot/LP_W1_D4LITE"
D4B = ROOT / "research_pilot/LP_W1_D4B"
DDEV = ROOT / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V"

PROTECTED_PATHS = (
    "outputs/research_d6/p1a_features",
    "outputs/research_d6/p1b/run_20261113_01",
    "outputs/research_d6/p1b_r1",
    "outputs/research_d6/p2/e1/run_20261115_r1_01",
    "outputs/research_d6/p2/e1_r2",
    "outputs/research_d6/p2/e2a0",
    "outputs/research_d6/p2/e2a1",
    "docs/research_d6/p1a",
    "docs/research_d6/p1b",
    "docs/research_d6/p1b_r1",
    "docs/research_d6/p2/g0",
    "docs/research_d6/p2/e1_r1",
    "docs/research_d6/p2/e1_r2",
    "docs/research_d6/p2/e2a0",
    "docs/research_d6/p2/e2a1",
    "scripts/research_d6/p1a",
    "scripts/research_d6/p1b",
    "scripts/research_d6/p2/e1",
    "scripts/research_d6/p2/e2a",
    "scripts/research_d6/p2/e2a1",
    "tests/research_d6/test_p2_e1.py",
    "tests/research_d6/test_e2a0_canonical_preprocess.py",
    "tests/research_d6/test_e2a1_preprocessing_storage.py",
)
