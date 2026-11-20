"""Shared G1A identity, resource, hashing and private-output helpers."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
R2C = ROOT / "outputs/research_d6/p3_g0_r2c/run_20261121_01"
R2B = ROOT / "outputs/research_d6/p3_g0_r2b/run_20261121_01"
OUT = ROOT / "outputs/research_d6/p3_g1a/run_20261121_01"
HEAD = "e2c0bd0113327173bf5d131f7e0cd8a5a826c8ea"
BRANCH = "stage3-eg-famar-execution"
AUTHORIZATION = "RESEARCH_LEAD_CHAT_APPROVAL_20261121_P3_G1A"
WHITELIST_SHA = "e61fa61e3646258fa58173a0f7708b630d2e1d38fe76f350fdc1126eea519c6f"
SELECTION_SHA = "cb5e3eb303444a67e22f47a6cc146cc72ece58a3bb0bb612bee389006728d9e6"
MAX_OUTPUT_BYTES = 3 * 1024**3
MAX_RUNTIME_SECONDS = 2 * 60 * 60


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_array(array) -> str:
    import numpy as np
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode("ascii"))
    h.update(repr(value.shape).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def self_sha(record: dict) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in record.items() if k != "record_sha256"})).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_once(path: Path, rows: list[dict], fields: list[str] | None = None) -> str:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path.name}")
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha_file(path)


def write_json_once(path: Path, value: dict) -> str:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path.name}")
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    return sha_file(path)


def restrict_acl(path: Path) -> None:
    if os.name != "nt":
        raise RuntimeError("WINDOWS_ACL_REQUIRED")
    user = subprocess.check_output(["whoami"], text=True).strip()
    cmd = ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F",
           "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"]
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise RuntimeError("ACL_SETUP_FAILED")


def check_output_budget(path: Path) -> int:
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    if size > MAX_OUTPUT_BYTES:
        raise RuntimeError("OUTPUT_BUDGET_EXCEEDED")
    return size


def current_git() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "rev-parse", f"refs/remotes/origin/{BRANCH}"], cwd=ROOT, text=True).strip()
    if (head, branch, remote) != (HEAD, BRANCH, HEAD):
        raise RuntimeError("GIT_IDENTITY_MISMATCH")
    return {"branch": branch, "head": head, "remote_head": remote}


def validate_r2c() -> dict:
    from scripts.research_d6.p3_g0_r2d.r2d_core import validate_input_bundle
    bundle = validate_input_bundle(R2C, R2B)
    if len(bundle["whitelist"]) != 216 or bundle["counts"] != {
        "image_count": 216, "group_count": 108, "product_count": 6,
        "representative_groups": 36, "stress_pool_groups": 72,
    }:
        raise RuntimeError("R2C_FIXED_SCOPE_MISMATCH")
    # New-study authorization B permits prior-HOLD-unknown rows as development
    # candidates; only a documented, specific prohibited identity is excluded.
    prohibited = [r for r in bundle["whitelist"] if r.get("protected_exclusion_status") in
                  {"KNOWN_PROTECTED_CONFLICT", "KNOWN_PROTECTED"}]
    allowed = [r for r in bundle["whitelist"] if r not in prohibited]
    return {"bundle": bundle, "allowed": allowed, "known_protected_excluded": prohibited}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

