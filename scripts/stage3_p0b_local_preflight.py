#!/usr/bin/env python3
"""Stage-3 P0-B local, read-only provenance preflight.

Hashes Real-IAD metadata JSON bytes (NOT image pixels), checks the prior
inventory and Git blob of v22, optionally hashes a *specified* checkpoint.
Does not import DINO, execute the v22 experiment, or expose evaluator labels
to online learning. Writes only a compact local JSON provenance receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "stage3" / "experiment_manifest_v1.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_index_hash(directory: Path) -> tuple[str, list[str]]:
    files = sorted(directory.glob("*.json"), key=lambda p: p.name)
    if len(files) != 30:
        raise ValueError(f"Expected exactly 30 product JSONs, found {len(files)}")
    h = hashlib.sha256()
    for p in files:
        rawname = p.name.encode("utf-8")
        h.update(len(rawname).to_bytes(4, "big"))
        h.update(rawname)
        length = p.stat().st_size
        h.update(length.to_bytes(8, "big"))
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest(), [p.name for p in files]


def git_blob_sha(relpath: str) -> str | None:
    proc = subprocess.run(["git", "-C", str(REPO), "hash-object", relpath],
                          capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def run(root: Path, json_dir: Path, inventory_path: Path, checkpoint: Path | None) -> dict:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not root.is_dir() or not json_dir.is_dir():
        raise ValueError("Real-IAD --root or --json-dir does not exist")
    if not inventory_path.is_file():
        raise ValueError("Run the previously supplied inventory command first; report not found")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    digest, filenames = json_index_hash(json_dir)
    problems = []
    if inventory.get("verify_files") is not True:
        problems.append("Inventory report did not verify image and mask file existence")
    if inventory.get("issues", {}).get("counts"):
        problems.append("Inventory still has nonempty warning counts")
    if inventory.get("n_products") != 30 or inventory.get("n_distinct_defect_types") != 8:
        problems.append("Inventory product or defect type count differs from P0-A")
    if sorted(inventory.get("source_json_files", [])) != filenames:
        problems.append("JSON filenames differ from the prior inventory")
    o = inventory.get("overall", {})
    if any(o.get(k) != v for k, v in {
        "train_normal": 36465, "train_defect": 0, "test_normal": 63256,
        "test_defect": 51329, "total": 151050,
    }.items()):
        problems.append("Inventory totals differ from the P0-A uploaded audit")

    v22_sha = git_blob_sha(m["source"]["v22_path"])
    visionad_sha = git_blob_sha(m["source"]["visionad_path"])
    if v22_sha != m["source"]["v22_blob_sha"]:
        problems.append("Local v22 source Git blob differs from frozen Stage3 source")
    if visionad_sha != m["source"]["visionad_blob_sha"]:
        problems.append("Local VisionAD source Git blob differs from frozen Stage3 source")

    checkpoint_hash = None
    if checkpoint is not None:
        if not checkpoint.is_file():
            problems.append("Explicit --checkpoint file does not exist")
        else:
            checkpoint_hash = file_sha256(checkpoint)
    report = {
        "schema": "eg-famar.p0b.local_provenance.v1",
        "status": "P0B_LOCAL_AUDIT_NOT_G0_EVALUATION",
        "data_root": str(root),
        "metadata_dir": str(json_dir),
        "dataset_index_sha256": digest,
        "dataset_index_scope": "30 JSON filenames and exact bytes only; images/masks were checked for existence in earlier inventory, NOT hashed or pixel-validated",
        "json_files_count": len(filenames),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_flags": {"verify_files": inventory.get("verify_files"),
                             "warning_counts": inventory.get("issues", {}).get("counts", {})},
        "code": {"v22_git_blob_sha": v22_sha, "visionad_git_blob_sha": visionad_sha},
        "checkpoint": {"explicit_path": str(checkpoint) if checkpoint else None,
                       "sha256": checkpoint_hash,
                       "loaded_by_v22_verified": False},
        "missing_still": [
            "Verify which checkpoint is actually loaded by the v22 runtime; a hash of a guessed file is insufficient",
            "Verify exact original preprocessing and support-derived theta0 numeric parity",
            "Confirm historical runs succeeded; command.txt is CLI evidence only",
        ],
        "errors": problems,
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--inventory", type=Path,
                   default=Path("outputs/stage3_p0/inventory/realiad_inventory.json"))
    p.add_argument("--checkpoint", type=Path,
                   help="Optional exact checkpoint FILE path, not just a hub model ID")
    p.add_argument("--out", type=Path,
                   default=Path("outputs/stage3_p0/p0b_local_provenance.json"))
    a = p.parse_args()
    try:
        report = run(a.root, a.json_dir, a.inventory, a.checkpoint)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"P0-B preflight error: {exc}")
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    print(json.dumps({
        "data_index_sha256": report["dataset_index_sha256"],
        "inventory_sha256": report["inventory_sha256"],
        "v22_git_blob_sha": report["code"]["v22_git_blob_sha"],
        "visionad_git_blob_sha": report["code"]["visionad_git_blob_sha"],
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "errors": report["errors"],
        "saved": str(a.out),
        "g0_experiment_executed": False,
    }, indent=2))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
