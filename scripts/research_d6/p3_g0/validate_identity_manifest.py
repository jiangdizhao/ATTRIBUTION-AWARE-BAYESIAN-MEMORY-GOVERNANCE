"""Validate one explicitly supplied, custodian-authorized identity-only manifest.

No source discovery, image reading, model execution, or output file creation.
The caller is responsible for authorization and authenticating the custodian's
records; this validator checks the manifest's documented assertions only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "p3-g0-r1-identity-validator-v1.0"
MAX_BYTES = 20_000_000
FIELDS = (
    "opaque_image_id", "product", "opaque_group_id", "view", "source_split",
    "source_sha256", "exposure_status", "allowed_research_role",
    "permission_authority", "permission_record_version", "permission_status",
    "exclusion_status", "exclusion_check_scope", "exclusion_authority",
    "exclusion_record_version", "near_duplicate_status", "group_conflict_status",
    "object_identity_confidence", "source_manifest_version",
)
ID_IMAGE = re.compile(r"^IMG_[0-9a-f]{32}$")
ID_GROUP = re.compile(r"^GRP_[0-9a-f]{32}$")
PRODUCT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
VIEW = re.compile(r"^C[1-9][0-9]?$" )
SHA = re.compile(r"^[0-9a-f]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
ENUM = {
    "source_split": {"train", "test", "UNKNOWN"},
    "exposure_status": {"METADATA_EXPOSED_ONLY", "NO_PRIOR_PIXEL_EXPOSURE_WITHIN_SCOPE", "PIXEL_EXPOSED_DEV", "NOT_KNOWN_EXPOSED", "UNKNOWN"},
    "allowed_research_role": {"D_CAL", "D_EVAL", "D_CAL_OR_D_EVAL", "NONE", "UNKNOWN"},
    "permission_status": {"ACTIVE", "REVOKED", "EXPIRED", "UNKNOWN"},
    "exclusion_status": {"CLEAR", "CONFLICT", "UNKNOWN"},
    "exclusion_check_scope": {"DFIT_DDEV_P2_AND_PROTECTED", "PARTIAL", "UNKNOWN"},
    "near_duplicate_status": {"CLEAR_WITHIN_SCOPE", "CONFLICT", "UNKNOWN"},
    "group_conflict_status": {"CLEAR_WITHIN_SCOPE", "CONFLICT", "UNKNOWN"},
    "object_identity_confidence": {"VERIFIED", "INFERRED", "UNKNOWN"},
}
P2_PRODUCTS = {"porcelain_doll", "sim_card_set", "usb_adaptor", "u_block", "toothbrush", "transistor1"}
GROUP_FIELDS = ("product", "source_split", "exposure_status", "allowed_research_role",
                "permission_authority", "permission_record_version", "permission_status",
                "exclusion_check_scope", "exclusion_authority", "exclusion_record_version",
                "object_identity_confidence", "source_manifest_version")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def result(status: str, **extra: object) -> dict:
    return {"validator_version": VERSION, "status": status, **extra}


def _parse(data: bytes, fmt: str) -> list[dict]:
    text = data.decode("utf-8-sig")
    if fmt == "csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        header = reader.fieldnames
        if header is None or len(header) != len(set(header)) or set(header) != set(FIELDS):
            raise ValueError("FIELD_SCHEMA_MISMATCH")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError("CSV_ROW_WIDTH_MISMATCH")
        return rows
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"schema", "rows"}:
        raise ValueError("JSON_ENVELOPE_MISMATCH")
    if obj["schema"] != "lp.p3.identity-only.v1" or not isinstance(obj["rows"], list):
        raise ValueError("JSON_ENVELOPE_MISMATCH")
    if any(not isinstance(row, dict) or set(row) != set(FIELDS) for row in obj["rows"]):
        raise ValueError("FIELD_SCHEMA_MISMATCH")
    return obj["rows"]


def validate_rows(rows: list[dict], requested_role: str) -> dict:
    if requested_role not in {"D_CAL", "D_EVAL"}:
        return result("BLOCKED", reasons={"INVALID_REQUESTED_ROLE": 1})
    if not rows:
        return result("BLOCKED", reasons={"EMPTY_MANIFEST": 1})
    structural = Counter()
    for row in rows:
        if set(row) != set(FIELDS):
            structural["FIELD_SCHEMA_MISMATCH"] += 1
            continue
        if any(not isinstance(row[k], str) for k in FIELDS):
            structural["NON_STRING_FIELD"] += 1
            continue
        if not ID_IMAGE.fullmatch(row["opaque_image_id"]): structural["INVALID_OPAQUE_IMAGE_ID"] += 1
        if not ID_GROUP.fullmatch(row["opaque_group_id"]): structural["INVALID_OPAQUE_GROUP_ID"] += 1
        if not PRODUCT.fullmatch(row["product"]): structural["INVALID_PRODUCT"] += 1
        if not VIEW.fullmatch(row["view"]): structural["INVALID_VIEW"] += 1
        if row["source_sha256"] and not SHA.fullmatch(row["source_sha256"]): structural["INVALID_SOURCE_SHA"] += 1
        for key in ("permission_authority", "permission_record_version", "source_manifest_version"):
            if not TOKEN.fullmatch(row[key]): structural["INVALID_REQUIRED_ATTESTATION"] += 1
        for key in ("exclusion_authority", "exclusion_record_version"):
            if row[key] and not TOKEN.fullmatch(row[key]): structural["INVALID_EXCLUSION_ATTESTATION"] += 1
        for key, allowed in ENUM.items():
            if row[key] not in allowed: structural["INVALID_ENUM"] += 1
    if structural:
        return result("BLOCKED", row_count=len(rows), reasons=dict(sorted(structural.items())))

    ids = Counter(r["opaque_image_id"] for r in rows)
    shas = Counter(r["source_sha256"] for r in rows if r["source_sha256"])
    groups = defaultdict(list)
    group_products = defaultdict(set)
    for row in rows:
        groups[(row["product"], row["opaque_group_id"])].append(row)
        group_products[row["opaque_group_id"]].add(row["product"])
    group_counts = Counter()
    reason_counts = Counter()
    by_product = defaultdict(Counter)
    split_counts = Counter()
    confidence_counts = Counter()

    for (product, group), members in sorted(groups.items()):
        bad, unknown, block = set(), set(), set()
        if len(group_products[group]) != 1: block.add("GROUP_ID_ACROSS_PRODUCTS")
        if any(len({row[f] for row in members}) != 1 for f in GROUP_FIELDS): block.add("INCONSISTENT_GROUP_METADATA")
        if any(ids[r["opaque_image_id"]] > 1 for r in members): block.add("DUPLICATE_IMAGE_ID")
        if any(r["source_sha256"] and shas[r["source_sha256"]] > 1 for r in members): bad.add("DUPLICATE_SOURCE_SHA")
        views = [r["view"] for r in members]
        if len(views) != len(set(views)): block.add("DUPLICATE_VIEW_IN_GROUP")
        if len(set(views)) < 2: bad.add("FEWER_THAN_TWO_VIEWS")
        if product in P2_PRODUCTS: bad.add("P2_DEVELOPMENT_PRODUCT")
        for row in members:
            if not row["source_sha256"]: unknown.add("MISSING_SOURCE_SHA")
            if row["source_split"] == "UNKNOWN": unknown.add("SOURCE_SPLIT_UNKNOWN")
            if row["exposure_status"] in {"UNKNOWN", "NOT_KNOWN_EXPOSED"}: unknown.add("EXPOSURE_UNKNOWN")
            if row["exposure_status"] == "PIXEL_EXPOSED_DEV": bad.add("ALREADY_PIXEL_EXPOSED")
            if row["permission_status"] in {"REVOKED", "EXPIRED"}: bad.add("PERMISSION_INACTIVE")
            if row["permission_status"] == "UNKNOWN": unknown.add("PERMISSION_UNKNOWN")
            if row["allowed_research_role"] == "NONE": bad.add("ROLE_UNAUTHORIZED")
            elif row["allowed_research_role"] == "UNKNOWN": unknown.add("ROLE_UNKNOWN")
            elif row["allowed_research_role"] not in {requested_role, "D_CAL_OR_D_EVAL"}: bad.add("ROLE_UNAUTHORIZED")
            if row["exclusion_status"] == "CONFLICT": bad.add("PROTECTED_EXCLUSION_CONFLICT")
            if row["near_duplicate_status"] == "CONFLICT": bad.add("NEAR_DUPLICATE_CONFLICT")
            if row["group_conflict_status"] == "CONFLICT": bad.add("GROUP_IDENTITY_CONFLICT")
            if row["exclusion_status"] == "UNKNOWN": unknown.add("PROTECTED_EXCLUSION_UNKNOWN")
            if row["near_duplicate_status"] == "UNKNOWN": unknown.add("NEAR_DUPLICATE_UNKNOWN")
            if row["group_conflict_status"] == "UNKNOWN": unknown.add("GROUP_IDENTITY_UNKNOWN")
            if row["exclusion_check_scope"] != "DFIT_DDEV_P2_AND_PROTECTED": unknown.add("EXCLUSION_SCOPE_INCOMPLETE")
            if not row["exclusion_authority"] or not row["exclusion_record_version"]: unknown.add("EXCLUSION_ATTESTATION_MISSING")
        status = "BLOCKED" if block else "INELIGIBLE" if bad else "UNKNOWN" if unknown else "PASS"
        group_counts[status] += 1
        by_product[product][status] += 1
        split_counts[members[0]["source_split"]] += 1
        confidence_counts[members[0]["object_identity_confidence"]] += 1
        for reason in block | bad | unknown: reason_counts[reason] += 1
    products = []
    for product, counts in sorted(by_product.items()):
        eligible = counts["PASS"]
        potential = eligible + counts["UNKNOWN"]
        status = "PASS" if eligible >= 10 else "UNKNOWN" if potential >= 10 else "INELIGIBLE"
        products.append({"product": product, "groups_pass": eligible, "groups_unknown": counts["UNKNOWN"],
                         "groups_ineligible": counts["INELIGIBLE"], "groups_blocked": counts["BLOCKED"],
                         "nominal_groups": sum(counts.values()), "capacity_status": status})
    confirmed = sum(p["capacity_status"] == "PASS" for p in products)
    potential = confirmed + sum(p["capacity_status"] == "UNKNOWN" for p in products)
    if group_counts["BLOCKED"]:
        overall = "BLOCKED"
    elif confirmed >= 6:
        overall = "PASS"
    elif potential >= 6:
        overall = "UNKNOWN"
    else:
        overall = "INELIGIBLE"
    return result(overall, requested_role=requested_role, row_count=len(rows),
                  nominal_product_count=len(products), nominal_group_count=len(groups),
                  confirmed_product_count=confirmed, conditional_product_count=potential-confirmed,
                  group_status_counts=dict(sorted(group_counts.items())),
                  reason_group_counts=dict(sorted(reason_counts.items())),
                  group_counts_by_split=dict(sorted(split_counts.items())),
                  object_identity_confidence_group_counts=dict(sorted(confidence_counts.items())),
                  products=products,
                  warning="Manifest claims are not independent proof of authorization, exclusion coverage, or physical-object independence.")


def validate_file(path: Path | None, requested_role: str, authorization_attested: bool,
                  source_name: str, expected_sha: str) -> dict:
    if path is None:
        return result("MISSING_AUTHORIZED_MANIFEST")
    if not authorization_attested:
        return result("BLOCKED", reasons={"AUTHORIZATION_NOT_ATTESTED": 1})
    if not TOKEN.fullmatch(source_name) or not SHA.fullmatch(expected_sha):
        return result("BLOCKED", reasons={"SOURCE_OR_EXPECTED_SHA_INVALID": 1})
    if not path.name.endswith((".identity-only.csv", ".identity-only.json")):
        return result("BLOCKED", reasons={"IDENTITY_ONLY_FILENAME_REQUIRED": 1})
    if not path.is_file():
        return result("MISSING_AUTHORIZED_MANIFEST")
    if path.stat().st_size > MAX_BYTES:
        return result("BLOCKED", reasons={"MANIFEST_TOO_LARGE": 1})
    data = path.read_bytes()
    actual = digest(data)
    if actual != expected_sha:
        return result("BLOCKED", manifest_sha256=actual, reasons={"MANIFEST_SHA_MISMATCH": 1})
    try:
        rows = _parse(data, "csv" if path.name.endswith(".csv") else "json")
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return result("BLOCKED", manifest_sha256=actual, reasons={"INVALID_IDENTITY_ONLY_SCHEMA": 1})
    report = validate_rows(rows, requested_role)
    report.update({"manifest_sha256": actual, "source_name": source_name,
                   "validator_sha256": digest(Path(__file__).read_bytes()),
                   "authorization_attested_by_caller": True})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="explicit authorized *.identity-only.csv/json")
    parser.add_argument("--role", choices=("D_CAL", "D_EVAL"), default="D_EVAL")
    parser.add_argument("--authorized-identity-only", action="store_true", help="caller attests read authorization")
    parser.add_argument("--source-name", default="", help="non-sensitive custodian record identifier")
    parser.add_argument("--expected-sha256", default="", help="independently supplied SHA-256 of manifest")
    args = parser.parse_args()
    report = validate_file(args.manifest, args.role, args.authorized_identity_only,
                           args.source_name, args.expected_sha256)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
