"""Run the bounded R2d preflight; hashes candidate bytes only after item CLEAR."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research_d6.p3_g0_r2d import r2d_core as core

R2C = ROOT / "outputs/research_d6/p3_g0_r2c/run_20261121_01"
R2B = ROOT / "outputs/research_d6/p3_g0_r2b/run_20261121_01"
OUT = ROOT / "outputs/research_d6/p3_g0_r2d/run_20261121_01"
BRANCH = "stage3-eg-famar-execution"
HEAD = "4f197b2fb439917ecdb0c1ff67a0a9f0cfb4e6e9"


def restrict_acl(path: Path) -> None:
    if os.name != "nt":
        raise core.R2DError("ACL_PLATFORM_UNSUPPORTED")
    user = subprocess.check_output(["whoami"], text=True).strip()
    cmd = ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F",
           "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"]
    if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise core.R2DError("ACL_SETUP_FAILED")


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    fields = list(rows[0]) if rows else []
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    core.assert_output_absent(OUT)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "rev-parse", "refs/remotes/origin/" + BRANCH], cwd=ROOT, text=True).strip()
    if (head, branch, remote) != (HEAD, BRANCH, HEAD):
        raise core.R2DError("GIT_IDENTITY_MISMATCH")
    if subprocess.run(["git", "check-ignore", "-q", str(OUT)], cwd=ROOT).returncode:
        raise core.R2DError("OUTPUT_NOT_IGNORED")
    bundle = core.validate_input_bundle(R2C, R2B)

    # The authorized, non-sensitive records contain no item-level protected
    # exclusion proof. In particular, path inequality is not a clearance.
    results = core.protected_preflight(bundle["whitelist"], evidence_by_id={})
    summary = core.aggregate_preflight(results)
    if summary["cleared_for_sha_read"] != 0:
        # Fail closed: no evidence provider is configured by this bounded run.
        raise core.R2DError("UNEXPECTED_CLEAR_EVIDENCE_SOURCE")
    OUT.mkdir(parents=True, exist_ok=False)
    restrict_acl(OUT)
    write_csv(OUT / "protection_preflight.csv", results)
    receipt = {
        "schema": "lp.d6.p3.r2d.blocked-protection.v1",
        "status": "R2D_BLOCKED_PROTECTION",
        "authorization": core.AUTHORIZATION,
        "authorization_status": "SHA_READ_AUTHORIZED",
        "protection_status": "PROTECTION_UNKNOWN_FOR_ALL_CANDIDATES",
        "git": {"branch": branch, "head": head, "remote_head": remote},
        "inputs": bundle["input_hashes"],
        "r2c_selection_receipt_sha256": core.SELECTION_RECEIPT_SHA,
        "candidate_counts": bundle["counts"],
        "preflight_summary": summary,
        "reason": "NO_ITEM_LEVEL_NON_SENSITIVE_PROTECTED_EXCLUSION_EVIDENCE",
        "image_byte_reads": 0,
        "image_decode": "NO",
        "protected_data_access": "NO",
        "model_or_annotation_use": "NO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["runner_sha256"] = core.sha_file(Path(__file__))
    receipt["core_sha256"] = core.sha_file(Path(core.__file__))
    with (OUT / "run_receipt.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(json.dumps(core.public_summary(summary), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (core.R2DError, OSError, KeyError, ValueError, FileNotFoundError) as exc:
        print(json.dumps({"status": "R2D_BLOCKED_INTEGRITY", "reason": type(exc).__name__ + ":" + str(exc)}))
        raise SystemExit(2)

