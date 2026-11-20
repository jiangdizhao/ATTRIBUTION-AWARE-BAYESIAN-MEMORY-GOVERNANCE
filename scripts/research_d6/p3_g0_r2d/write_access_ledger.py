"""Write an append-only private per-candidate record from the blocked R2d run."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/research_d6/p3_g0_r2d/run_20261121_01"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ledger_path = OUT / "file_access_ledger.csv"
    receipt_path = OUT / "file_access_ledger_receipt.json"
    if ledger_path.exists() or receipt_path.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")
    receipt = json.loads((OUT / "run_receipt.json").read_text(encoding="utf-8"))
    with (OUT / "protection_preflight.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fields = ["opaque_image_id", "opaque_group_id", "conditional_role", "candidate_kind", "read_type",
              "read_performed", "read_success", "sha_status", "file_sha256", "bytes_read", "decoded",
              "used_for_model", "whitelist_version", "whitelist_sha256", "authorization_record",
              "audit_time_utc", "audit_code_sha256", "protection_status", "reason_code"]
    with ledger_path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "opaque_image_id": row["opaque_image_id"],
                "opaque_group_id": row["opaque_group_id"],
                "conditional_role": row["conditional_role"],
                "candidate_kind": row["candidate_kind"],
                "read_type": "BYTE_HASH_ONLY",
                "read_performed": "NO",
                "read_success": "NO",
                "sha_status": row["sha_status"],
                "file_sha256": "",
                "bytes_read": "0",
                "decoded": "NO",
                "used_for_model": "NO",
                "whitelist_version": "LP-D6-P3-G0-R2C-v1",
                "whitelist_sha256": receipt["inputs"]["targeted_sha_whitelist.csv"],
                "authorization_record": receipt["authorization"],
                "audit_time_utc": receipt["created_utc"],
                "audit_code_sha256": receipt["runner_sha256"],
                "protection_status": row["protection_status"],
                "reason_code": row["reason_code"],
            })
    code_sha = sha256(Path(__file__))
    payload = {"schema": "lp.d6.p3.r2d.private-access-ledger.v1", "row_count": len(rows),
               "ledger_sha256": sha256(ledger_path), "ledger_writer_sha256": code_sha,
               "r2d_runner_sha256": receipt["runner_sha256"], "authorization": receipt["authorization"],
               "read_performed_count": 0, "sha_verified_count": 0, "protection_unknown_count": len(rows),
               "status": "R2D_BLOCKED_PROTECTION"}
    with receipt_path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"rows": len(rows), "read_performed": 0, "ledger_sha256": payload["ledger_sha256"],
                      "writer_sha256": code_sha}, indent=2))


if __name__ == "__main__":
    main()
