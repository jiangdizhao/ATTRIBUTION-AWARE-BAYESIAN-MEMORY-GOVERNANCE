"""Fail-closed R2d SHA-only candidate audit helpers.

The runner validates frozen metadata first and requires item-level protection
clearance before any target-file open. No image decoding dependencies exist.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import BinaryIO, Iterable

AUTHORIZATION = "RESEARCH_LEAD_CHAT_APPROVAL_20261121_R2D_SHA_ONLY"
WHITELIST_SHA = "e61fa61e3646258fa58173a0f7708b630d2e1d38fe76f350fdc1126eea519c6f"
SELECTION_RECEIPT_SHA = "cb5e3eb303444a67e22f47a6cc146cc72ece58a3bb0bb612bee389006728d9e6"
FALLBACK_SHA = "65e14488a0900fa818d32a380275103a8be5655f9094434888af906507f85c5c"
EXPOSURE_COMPARE_SHA = "7db55289a884cc851e01da89d9931b7e010a791ec2d7b0658b1a844d6b6bad9e"
R2B_INPUT_SHA = {
    "extraction_receipt.json": "3aa6d1e7a8b10c426e462adb0f5085281acf62d4cc480dd654328f07fc5f49d6",
    "independent_structural_check.json": "08348e5378e2ba3fde51b8b8cf5597e4c36fcaabbad1b4f110ca762a34430ecd",
    "conditional_identity_order.csv": "38abc314eac0b4d79405e947a87f9cbe5b1fbe619efe7c5961cbdf8d345fe50d",
    "custodian_source_mapping.csv": "6b0754dd466a6a29dff43004a4853489eb80871fa84e3ec1ae06324b9d83df78",
}
P2_SELECTION_SHA = "c0def1f598d38ea68c78da028958f7ee030a7bdf21528d5a249fe8f8a6f81b7f"


class R2DError(RuntimeError):
    pass


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_sha(path: Path, expected: str, *, hash_fn=sha_file) -> None:
    if not path.is_file() or hash_fn(path) != expected:
        raise R2DError("PINNED_FILE_SHA_MISMATCH")


def verify_recorded_output_hash(actual: str, expected: str) -> None:
    if actual != expected:
        raise R2DError("RECORDED_OUTPUT_HASH_MISMATCH")


def assert_output_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError("OUTPUT_ALREADY_EXISTS")


def public_summary(summary: dict) -> dict:
    allowed = {"status", "whitelist_count", "known_protected_conflict", "cleared_for_sha_read",
               "protection_unknown", "sha_verified", "sha_skipped_protection_unknown", "sha_failed",
               "image_bytes_read", "image_decoded", "protected_data_accessed", "model_used"}
    return {key: summary[key] for key in allowed if key in summary}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_whitelist_rows(rows: list[dict[str, str]], products: list[dict[str, str]],
                            groups: list[dict[str, str]]) -> dict:
    if len(rows) != 216:
        raise R2DError("WHITELIST_ROW_COUNT")
    required = {"opaque_image_id", "product", "opaque_group_id", "view", "conditional_role",
                "candidate_kind", "eligibility", "source_image_path", "source_sha256_status",
                "future_file_access_permission", "historical_exposure_status", "protected_exclusion_status"}
    if any(not required.issubset(row) for row in rows):
        raise R2DError("WHITELIST_SCHEMA")
    if len({r["opaque_image_id"] for r in rows}) != 216:
        raise R2DError("DUPLICATE_OPAQUE_IMAGE_ID")
    if len({r["product"] for r in rows}) != 6:
        raise R2DError("PRODUCT_COUNT")
    if len({r["opaque_group_id"] for r in rows}) != 108:
        raise R2DError("GROUP_COUNT")
    if len(products) != 12 or len(groups) != 108:
        raise R2DError("R2C_ROLE_FRAME_COUNTS")
    selected_products = {r["product"]: r["conditional_role"] for r in products if r["conditional_role"] != "BACKUP_PRODUCT"}
    if len(selected_products) != 6 or sum(v == "TENTATIVE_D_CAL" for v in selected_products.values()) != 2 or sum(v == "TENTATIVE_D_EVAL" for v in selected_products.values()) != 4:
        raise R2DError("TENTATIVE_ROLE_COUNTS")
    if set(selected_products.values()) != {"TENTATIVE_D_CAL", "TENTATIVE_D_EVAL"}:
        raise R2DError("FORMAL_OR_INVALID_ROLE")
    group_by_id = {g["opaque_group_id"]: g for g in groups}
    if len(group_by_id) != 108:
        raise R2DError("DUPLICATE_GROUP_ID")
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["product"] not in selected_products or row["conditional_role"] != selected_products[row["product"]]:
            raise R2DError("NON_CANDIDATE_PRODUCT_OR_ROLE")
        if row["eligibility"] != "UNKNOWN" or row["source_sha256_status"] != "UNKNOWN":
            raise R2DError("UNKNOWN_STATUS_PROMOTED")
        if row["candidate_kind"] == "STRESS_SELECTED" or row["conditional_role"] in {"D_CAL", "D_EVAL"}:
            raise R2DError("FORMAL_ROLE_PROMOTION")
        group = group_by_id.get(row["opaque_group_id"])
        if not group or any(group[k] != row[k] for k in ("product", "conditional_role", "candidate_kind")):
            raise R2DError("GROUP_IDENTITY_MISMATCH")
        by_group[row["opaque_group_id"]].append(row)
        validate_lexical_path(row["source_image_path"], row["product"])
    if any(len(v) != 2 or len({r["view"] for r in v}) != 2 for v in by_group.values()):
        raise R2DError("GROUP_VIEW_SELECTION")
    kinds = Counter(g["candidate_kind"] for g in groups)
    if kinds != {"REPRESENTATIVE_CANDIDATE": 36, "STRESS_POOL_CANDIDATE": 72}:
        raise R2DError("CANDIDATE_KIND_COUNTS")
    return {"image_count": len(rows), "group_count": len(by_group), "product_count": len(selected_products),
            "representative_groups": kinds["REPRESENTATIVE_CANDIDATE"],
            "stress_pool_groups": kinds["STRESS_POOL_CANDIDATE"]}


def validate_lexical_path(path_text: str, product: str, approved_root: str = r"G:\dataset\realiad") -> PureWindowsPath:
    path = PureWindowsPath(path_text)
    root = PureWindowsPath(approved_root)
    if not path.is_absolute() or path.drive.casefold() != root.drive.casefold() or path.anchor.startswith("\\\\"):
        raise R2DError("PATH_DRIVE_OR_UNC")
    if any(p in {".", ".."} or ":" in p for p in path.parts[1:]):
        raise R2DError("PATH_TRAVERSAL")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise R2DError("PATH_OUTSIDE_ROOT") from exc
    if not relative.parts or product.casefold() not in {part.casefold() for part in relative.parts[:-1]}:
        raise R2DError("PATH_PRODUCT_MISMATCH")
    if path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
        raise R2DError("PATH_FILE_TYPE")
    return path


def classify_protection(row: dict[str, str], evidence: dict | None = None) -> tuple[str, str]:
    """Only an item-scoped, versioned non-sensitive custodian clearance can clear.

    The real run passes no evidence because no such R2d exclusion record exists.
    Synthetic test evidence is never loaded by the production runner.
    """
    recorded = row.get("protected_exclusion_status", "UNKNOWN")
    if recorded in {"KNOWN_PROTECTED_CONFLICT", "PROTECTED"}:
        return "KNOWN_PROTECTED_CONFLICT", "WHITELIST_MARKED_PROTECTED"
    if evidence and evidence.get("opaque_image_id") == row.get("opaque_image_id"):
        required = {"scope", "source_version", "authorization", "evidence_id", "status"}
        if required.issubset(evidence) and evidence["status"] == "CLEARED_FOR_SHA_READ" and evidence["scope"] == "ITEM_LEVEL_PROTECTED_EXCLUSION":
            return "CLEARED_FOR_SHA_READ", "ITEM_LEVEL_NON_SENSITIVE_EXCLUSION_RECORD"
    return "PROTECTION_UNKNOWN", "NO_ITEM_LEVEL_PROTECTED_EXCLUSION_EVIDENCE"


def gated_hash_stream(stream: BinaryIO, *, authorization: str, protection_status: str,
                      path_safe: bool, before_token: tuple | None = None,
                      after_metadata=None, block_size: int = 1024 * 1024) -> tuple[str, int]:
    if authorization != AUTHORIZATION:
        raise R2DError("SHA_AUTHORIZATION_REQUIRED")
    if protection_status != "CLEARED_FOR_SHA_READ":
        raise R2DError("PROTECTION_NOT_CLEARED")
    if not path_safe:
        raise R2DError("PATH_NOT_SAFE")
    if block_size < 1 or block_size > 1024 * 1024:
        raise R2DError("INVALID_BLOCK_SIZE")
    digest = hashlib.sha256()
    total = 0
    while True:
        block = stream.read(block_size)
        if not block:
            break
        digest.update(block)
        total += len(block)
    if after_metadata is not None and before_token != after_metadata():
        raise R2DError("FILE_CHANGED_DURING_READ")
    return digest.hexdigest(), total


def exact_duplicate_groups(rows: list[dict[str, str]]) -> dict:
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get("sha256", "")
        if value:
            by_sha[value].append(row)
    duplicates = {key: value for key, value in by_sha.items() if len(value) > 1}
    return {"duplicate_file_count": sum(len(v) for v in duplicates.values()),
            "affected_group_count": len({r["opaque_group_id"] for v in duplicates.values() for r in v}),
            "duplicate_sha_count": len(duplicates)}


def compare_sha(candidate_rows: list[dict[str, str]], historical_sha: Iterable[str]) -> dict:
    candidate = {r["sha256"] for r in candidate_rows if r.get("sha256")}
    history = {x for x in historical_sha if x}
    return {"exact_matches": len(candidate & history), "comparison_status": "COMPARED" if candidate else "NO_CANDIDATE_SHA"}


def eligibility_after_sha(protection_status: str, other_requirements: dict | None = None) -> str:
    if protection_status != "CLEARED_FOR_SHA_READ":
        return "UNKNOWN"
    if not other_requirements or any(v != "CLEAR" for v in other_requirements.values()):
        return "UNKNOWN"
    return "PASS"


def validate_no_reparse_components(component_reparse_flags: Iterable[bool]) -> bool:
    return not any(component_reparse_flags)


def validate_input_bundle(r2c: Path, r2b: Path, *, hash_fn=sha_file) -> dict:
    pinned_files = {
        "targeted_sha_whitelist.csv": WHITELIST_SHA,
        "selection_receipt.json": SELECTION_RECEIPT_SHA,
        "fallback_policy.json": FALLBACK_SHA,
        "exposure_metadata_comparison.json": EXPOSURE_COMPARE_SHA,
    }
    for name, expected in pinned_files.items():
        try:
            verify_file_sha(r2c / name, expected, hash_fn=hash_fn)
        except R2DError as exc:
            raise R2DError("R2C_PINNED_INPUT_MISMATCH") from exc
    receipt = json.loads((r2c / "selection_receipt.json").read_text(encoding="utf-8"))
    for name, expected in (("conditional_products.csv", receipt["outputs_sha256"]["conditional_products.csv"]),
                           ("conditional_groups.csv", receipt["outputs_sha256"]["conditional_groups.csv"]),
                           ("targeted_sha_whitelist.csv", WHITELIST_SHA),
                           ("full_order_reference.json", receipt["order_reference_sha256"])):
        if hash_fn(r2c / name) != expected:
            raise R2DError("R2C_OUTPUT_RECEIPT_MISMATCH")
    if receipt.get("formal_partition") is not False or receipt.get("stress_selected") is not False:
        raise R2DError("R2C_ROLE_PROMOTION")
    for filename, expected in R2B_INPUT_SHA.items():
        try:
            verify_file_sha(r2b / filename, expected, hash_fn=hash_fn)
        except R2DError as exc:
            raise R2DError("R2B_INPUT_MISMATCH") from exc
    r2b_receipt = json.loads((r2b / "extraction_receipt.json").read_text(encoding="utf-8"))
    structural = json.loads((r2b / "independent_structural_check.json").read_text(encoding="utf-8"))
    expected_shards = {entry["product"]: entry["sha256"] for entry in structural["shards"]}
    expected_products = set(expected_shards)
    if (len(expected_shards) != 12 or expected_products != {x["product"] for x in r2b_receipt["source_outcomes"]}
            or receipt.get("shard_sha256") != expected_shards):
        raise R2DError("R2B_PRODUCT_SET_MISMATCH")
    for product, expected in expected_shards.items():
        shard = r2b / f"{product}.identity-only.csv"
        if not shard.is_file() or hash_fn(shard) != expected:
            raise R2DError("R2B_SHARD_MISMATCH")
        source = next((x for x in r2b_receipt["source_outcomes"] if x["product"] == product), None)
        structural_shard = next((x for x in structural["shards"] if x["product"] == product), None)
        if source is None or structural_shard is None or source.get("manifest_sha256") != expected or structural_shard.get("sha256") != expected:
            raise R2DError("R2B_SHARD_RECEIPT_MISMATCH")
    p2_manifest = r2c.parents[3] / "research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/selection_manifest.csv"
    try:
        verify_file_sha(p2_manifest, P2_SELECTION_SHA, hash_fn=hash_fn)
    except R2DError as exc:
        raise R2DError("P2_MANIFEST_MISMATCH") from exc
    p2_rows = read_csv(p2_manifest)
    if len(p2_rows) != 48:
        raise R2DError("P2_MANIFEST_COUNT")
    p2_parents = set()
    for row in p2_rows:
        absolute, relative = PureWindowsPath(row["image_path"]), PureWindowsPath(row["image_relpath"])
        if (not absolute.is_absolute() or relative.is_absolute() or relative.drive or ".." in relative.parts or
                not relative.parts or absolute.parts[-len(relative.parts):] != relative.parts):
            raise R2DError("P2_PATH_LAYOUT_INVALID")
        product_root = PureWindowsPath(*absolute.parts[:-len(relative.parts)])
        if product_root.name != row["product"] or PureWindowsPath(r"G:\dataset\realiad") not in product_root.parents:
            raise R2DError("P2_ROOT_INVALID")
        p2_parents.add(product_root.parent)
    if len(p2_parents) != 1:
        raise R2DError("P2_BASE_AMBIGUOUS")
    image_base = p2_parents.pop()
    mapping_rows = read_csv(r2b / "custodian_source_mapping.csv")
    mapping_by_id = {r["opaque_image_id"]: r for r in mapping_rows}
    if len(mapping_by_id) != len(mapping_rows):
        raise R2DError("R2B_MAPPING_DUPLICATE_ID")
    wl = read_csv(r2c / "targeted_sha_whitelist.csv")
    prods = read_csv(r2c / "conditional_products.csv")
    groups = read_csv(r2c / "conditional_groups.csv")
    counts = validate_whitelist_rows(wl, prods, groups)
    for row in wl:
        mapped = mapping_by_id.get(row["opaque_image_id"])
        if mapped is None or any(mapped[k] != row[v] for k, v in (("product", "product"), ("opaque_group_id", "opaque_group_id"), ("view", "view"))):
            raise R2DError("WHITELIST_MAPPING_IDENTITY_MISMATCH")
        relative = PureWindowsPath(mapped["source_image_identity"])
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise R2DError("MAPPING_RELATIVE_PATH_INVALID")
        expected_path = image_base / row["product"] / relative
        if PureWindowsPath(row["source_image_path"]) != expected_path:
            raise R2DError("WHITELIST_SOURCE_PATH_MISMATCH")
    # Candidate opaque identities are matched against identity-only metadata;
    # image and mask paths are never opened here.
    chosen_ids = {r["opaque_image_id"] for r in wl}
    source_rows = {}
    for product in {r["product"] for r in wl}:
        for item in read_csv(r2b / f"{product}.identity-only.csv"):
            if item["opaque_image_id"] in chosen_ids:
                source_rows[item["opaque_image_id"]] = item
    if set(source_rows) != chosen_ids:
        raise R2DError("WHITELIST_NOT_IN_R2B_FRAME")
    for row in wl:
        source = source_rows[row["opaque_image_id"]]
        if any(source[k] != row[v] for k, v in (("product", "product"), ("opaque_group_id", "opaque_group_id"), ("view", "view"))):
            raise R2DError("WHITELIST_FRAME_MISMATCH")
    return {"receipt": receipt, "whitelist": wl, "products": prods, "groups": groups,
            "counts": counts, "input_hashes": {**pinned_files,
            "conditional_products.csv": hash_fn(r2c / "conditional_products.csv"),
            "conditional_groups.csv": hash_fn(r2c / "conditional_groups.csv"),
            "full_order_reference.json": hash_fn(r2c / "full_order_reference.json")}}


def protected_preflight(rows: list[dict[str, str]], evidence_by_id: dict[str, dict] | None = None) -> list[dict]:
    evidence_by_id = evidence_by_id or {}
    results = []
    for row in rows:
        state, reason = classify_protection(row, evidence_by_id.get(row["opaque_image_id"]))
        results.append({"opaque_image_id": row["opaque_image_id"], "opaque_group_id": row["opaque_group_id"],
                        "product": row["product"], "conditional_role": row["conditional_role"],
                        "candidate_kind": row["candidate_kind"], "protection_status": state,
                        "reason_code": reason, "sha_status": "SHA_SKIPPED_PROTECTION_UNKNOWN" if state == "PROTECTION_UNKNOWN" else "NOT_RUN",
                        "sha256": "", "bytes_read": 0, "decoded": "NO", "used_for_model": "NO"})
    return results


def aggregate_preflight(results: list[dict]) -> dict:
    states = Counter(r["protection_status"] for r in results)
    return {"whitelist_count": len(results), "known_protected_conflict": states["KNOWN_PROTECTED_CONFLICT"],
            "cleared_for_sha_read": states["CLEARED_FOR_SHA_READ"], "protection_unknown": states["PROTECTION_UNKNOWN"],
            "sha_verified": sum(r["sha_status"] == "SHA_VERIFIED" for r in results),
            "sha_skipped_protection_unknown": sum(r["sha_status"] == "SHA_SKIPPED_PROTECTION_UNKNOWN" for r in results),
            "sha_failed": sum(r["sha_status"] in {"SHA_FILE_MISSING", "SHA_FILE_CHANGED", "SHA_READ_ERROR", "SHA_SKIPPED_PATH_SECURITY"} for r in results),
            "image_bytes_read": sum(r["bytes_read"] for r in results),
            "image_decoded": "NO", "protected_data_accessed": "NO", "model_used": "NO",
            "status": "R2D_BLOCKED_PROTECTION" if not any(r["protection_status"] == "CLEARED_FOR_SHA_READ" for r in results) else "R2D_SHA_AUDIT_PARTIAL"}

