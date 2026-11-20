"""Private, fail-closed preparation and annotation I/O for LP-D6-P3-G1B.

This module reads G1A metadata only during preparation. Real image bytes are
read only by the separately authorized annotation launcher.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

SALT = "LP-D6-P3-G0-v0.2-20261120"
SEP = "\x1f"
QUEUE_SEED = "LP-D6-P3-G1B-v1-20261121"
EXPECTED_G1A_ROLES_SHA256 = "7e402d017ac6795f0a5c4ed6bf7493244e872e6e288fd88a3f11ecdf031be443"
G1A_ROOT_REL = Path("outputs/research_d6/p3_g1a/run_20261121_01")
RUN_REL = Path("outputs/research_d6/p3_g1b/run_20261121_01")
VALID_STATES = {
    "present", "absent_assessable", "unobservable_or_uncertain", "not_applicable"
}
ANNOTATION_FIELDS = [
    "presentation_id", "opaque_image_id", "opaque_group_id", "product",
    "study_role", "view", "attribute_name", "attribute_state", "confidence",
    "location_kind", "point_x_normalized", "point_y_normalized", "notes",
    "first_displayed_utc", "last_saved_utc", "cumulative_active_seconds",
    "annotator_id", "annotation_revision",
]


class IntegrityError(RuntimeError):
    """A frozen input or identity relationship did not validate."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def canonical_field(value: str) -> str:
    if SEP in value:
        raise IntegrityError("CONTROL_SEPARATOR_IN_KEY")
    return unicodedata.normalize("NFC", value)


def order_key(domain: str, key: str) -> tuple[bytes, bytes]:
    canonical = canonical_text(key)
    digest = hashlib.sha256((SALT + SEP + domain + SEP + canonical).encode("utf-8")).digest()
    return digest, canonical.encode("utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_once(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> str:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def write_json_once(path: Path, value: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path.name}")
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def atomic_write_bytes(path: Path, payload: bytes, *, replace: Callable = os.replace) -> None:
    """Replace one file atomically; a failed replacement leaves the prior file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        replace(str(temp_path), str(path))
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = ANNOTATION_FIELDS) -> None:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def require_git_ignored(repo: Path, relpath: Path) -> None:
    import subprocess
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(relpath.as_posix())],
        cwd=repo, check=False,
    )
    if result.returncode != 0:
        raise IntegrityError("PRIVATE_OUTPUT_NOT_GIT_IGNORED")


def restrict_windows_acl(path: Path) -> None:
    """Remove inherited ACLs and grant access only to the runner, SYSTEM and admins."""
    import subprocess
    if os.name != "nt":
        raise IntegrityError("WINDOWS_ACL_REQUIRED")
    user = subprocess.check_output(["whoami"], text=True).strip()
    command = [
        "icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F",
        "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F",
    ]
    if subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise IntegrityError("PRIVATE_OUTPUT_ACL_SETUP_FAILED")


def _role_for_conditional(value: str) -> str:
    mapping = {"TENTATIVE_D_CAL": "P3_G1A_D_CAL", "TENTATIVE_D_EVAL": "P3_G1A_D_EVAL"}
    try:
        return mapping[value]
    except KeyError as exc:
        raise IntegrityError("CONDITIONAL_ROLE_INVALID") from exc


def load_g1a_metadata(g1a_root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]], dict]:
    """Read only selected roles, authorized identity metadata, and G1A exposure ledger."""
    role_path = g1a_root / "development_roles.csv"
    if sha256_file(role_path) != EXPECTED_G1A_ROLES_SHA256:
        raise IntegrityError("G1A_ROLES_SHA_MISMATCH")
    roles = read_csv(role_path)
    authorized_rows = read_csv(g1a_root / "authorized_images.csv")
    exposure_rows = read_csv(g1a_root / "exposure_ledger.csv")
    authorized = {row["opaque_image_id"]: row for row in authorized_rows}
    exposure = {row["opaque_image_id"]: row for row in exposure_rows}
    if len(authorized) != len(authorized_rows) or len(exposure) != len(exposure_rows):
        raise IntegrityError("G1A_DUPLICATE_IDENTITY")
    if len(roles) != 120 or len({r["opaque_image_id"] for r in roles}) != 120:
        raise IntegrityError("G1A_ROLE_CARDINALITY")
    if set(r["opaque_image_id"] for r in roles) - (authorized.keys() & exposure.keys()):
        raise IntegrityError("G1A_IDENTITY_NOT_AUTHORIZED")
    group_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    slot_products: dict[str, set[str]] = defaultdict(set)
    mapping: dict[str, dict[str, str]] = {}
    for role in roles:
        image_id = role["opaque_image_id"]
        source = authorized[image_id]
        ledger = exposure[image_id]
        expected_role = role["study_role"]
        if expected_role not in {"P3_G1A_D_CAL", "P3_G1A_D_EVAL"}:
            raise IntegrityError("G1A_ROLE_INVALID")
        if (role["opaque_group_id"] != source["opaque_group_id"]
                or role["opaque_group_id"] != ledger["opaque_group_id"]
                or role["view"] != source["view"] or role["view"] != ledger["view"]
                or role["candidate_kind"] != source["candidate_kind"]
                or _role_for_conditional(source["conditional_role"]) != expected_role):
            raise IntegrityError("G1A_ROLE_IDENTITY_ASSOCIATION_MISMATCH")
        if (ledger["read_success"] != "YES" or ledger["decoded"] != "YES"
                or ledger["final_path_verified"] != "YES" or ledger["reparse_check"] != "PASS"
                or ledger["used_for_model"] != "NO"):
            raise IntegrityError("G1A_EXPOSURE_STATUS_INVALID")
        source_sha = ledger["source_sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            raise IntegrityError("G1A_SOURCE_SHA_INVALID")
        source_path = source["source_image_path"]
        normalized_path = source_path.replace("/", "\\").casefold()
        if not source_path or not normalized_path.startswith("g:\\dataset\\realiad\\"):
            raise IntegrityError("G1A_SOURCE_PATH_OUTSIDE_WHITELIST")
        slot_products[role["product_slot"]].add(source["product"])
        if source["product"] != ledger["product"]:
            raise IntegrityError("G1A_PRODUCT_ASSOCIATION_MISMATCH")
        group_rows[(source["product"], role["opaque_group_id"])].append(role)
        mapping[image_id] = {
            "opaque_image_id": image_id,
            "opaque_group_id": role["opaque_group_id"],
            "product": source["product"],
            "product_slot": role["product_slot"],
            "study_role": expected_role,
            "view": role["view"],
            "candidate_kind": role["candidate_kind"],
            "source_image_path": source_path,
            "source_sha256": source_sha,
            "source_size_bytes": ledger["source_size_bytes"],
            "exposure_status": "NEW_STUDY_DEV_EXPOSED",
            "g1a_read_success": ledger["read_success"],
            "g1a_final_path_verified": ledger["final_path_verified"],
        }
    if len(slot_products) != 6 or any(len(products) != 1 for products in slot_products.values()):
        raise IntegrityError("G1A_PRODUCT_SLOT_MAP_INVALID")
    product_role = Counter((row["product_slot"], row["study_role"]) for row in roles)
    expected_counts = {
        ("1", "P3_G1A_D_CAL"): 20, ("2", "P3_G1A_D_CAL"): 20,
        ("3", "P3_G1A_D_EVAL"): 20, ("4", "P3_G1A_D_EVAL"): 20,
        ("5", "P3_G1A_D_EVAL"): 20, ("6", "P3_G1A_D_EVAL"): 20,
    }
    if dict(product_role) != expected_counts:
        raise IntegrityError("G1A_PRODUCT_ROLE_CARDINALITY")
    if len(group_rows) != 60 or any(
        len(rows) != 2 or len({row["view"] for row in rows}) != 2
        or len({row["study_role"] for row in rows}) != 1
        for rows in group_rows.values()
    ):
        raise IntegrityError("G1A_GROUP_INTEGRITY")
    summary = {
        "development_roles_sha256": sha256_file(role_path),
        "authorized_images_sha256": sha256_file(g1a_root / "authorized_images.csv"),
        "exposure_ledger_sha256": sha256_file(g1a_root / "exposure_ledger.csv"),
        "unlabeled_scores_sha256": sha256_file(g1a_root / "unlabeled_scores.csv"),
        "stress_selection_sha256": sha256_file(g1a_root / "stress_selection.csv"),
        "final_execution_receipt_sha256": sha256_file(g1a_root / "final_execution_receipt.json"),
        "selection_summary_sha256": sha256_file(g1a_root / "selection_summary.json"),
        "run_summary_sha256": sha256_file(g1a_root / "run_summary.json"),
        "image_count": len(mapping),
        "group_count": len(group_rows),
        "calibration_views": 40,
        "calibration_groups": 20,
        "evaluation_views": 80,
        "evaluation_groups": 40,
        "product_count": len(slot_products),
        "source_image_bytes_read": 0,
    }
    return roles, mapping, exposure, summary


def build_queue(roles: list[dict[str, str]], mapping: dict[str, dict[str, str]], *, max_attempts: int = 10000) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Freeze 120 first views and 24 blind repeats using G0's REPEAT_GROUP rule."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for role in roles:
        record = mapping[role["opaque_image_id"]]
        groups[(record["product"], role["opaque_group_id"])].append(role)
    ordered_groups = sorted(
        groups,
        key=lambda group: order_key("REPEAT_GROUP", canonical_field(group[0]) + SEP + canonical_field(group[1])),
    )
    repeated_groups = ordered_groups[:12]
    repeat_ids = {row["opaque_image_id"] for key in repeated_groups for row in groups[key]}
    base_events: list[dict[str, Any]] = []
    sequence = 0
    for role in roles:
        image_id = role["opaque_image_id"]
        base_events.append({
            "event_index": sequence,
            "opaque_image_id": image_id,
            "opaque_group_id": role["opaque_group_id"],
            "product": mapping[image_id]["product"],
            "product_slot": role["product_slot"],
            "study_role": role["study_role"],
            "view": role["view"],
            "candidate_kind": role["candidate_kind"],
            "presentation_kind": "FIRST",
        })
        sequence += 1
    for role in roles:
        image_id = role["opaque_image_id"]
        if image_id in repeat_ids:
            base_events.append({
                "event_index": sequence,
                "opaque_image_id": image_id,
                "opaque_group_id": role["opaque_group_id"],
                "product": mapping[image_id]["product"],
                "product_slot": role["product_slot"],
                "study_role": role["study_role"],
                "view": role["view"],
                "candidate_kind": role["candidate_kind"],
                "presentation_kind": "REPEAT",
            })
            sequence += 1
    if len(base_events) != 144 or len(repeat_ids) != 24:
        raise IntegrityError("QUEUE_CARDINALITY_INVALID")

    # The fixed nonce stream contains no identity or label. If a particular
    # deterministic permutation places a repeat beside its first presentation,
    # advance to the next pre-registered nonce block.
    ordered_events = None
    chosen_attempt = None
    for attempt in range(max_attempts):
        nonce_rows = []
        for event in base_events:
            nonce = sha256_bytes(f"{QUEUE_SEED}|block={attempt}|index={event['event_index']}".encode("utf-8"))
            pid = "P-" + sha256_bytes(f"{QUEUE_SEED}|presentation|{event['event_index']}".encode("utf-8"))[:20]
            nonce_rows.append((order_key("PRESENTATION_ORDER", nonce), {**event, "queue_nonce": nonce, "presentation_id": pid}))
        candidate = [row for _, row in sorted(nonce_rows, key=lambda item: item[0])]
        positions = defaultdict(list)
        for index, event in enumerate(candidate):
            positions[event["opaque_image_id"]].append(index)
        if all(len(indices) == 1 or abs(indices[0] - indices[1]) > 1 for indices in positions.values()):
            ordered_events, chosen_attempt = candidate, attempt
            break
    if ordered_events is None:
        raise IntegrityError("QUEUE_NONADJACENCY_CONSTRAINT_UNSATISFIED")

    first_by_image = {row["opaque_image_id"]: row["presentation_id"] for row in ordered_events if row["presentation_kind"] == "FIRST"}
    queue = []
    for index, event in enumerate(ordered_events, 1):
        queue.append({
            "presentation_order": index,
            "presentation_id": event["presentation_id"],
            "opaque_image_id": event["opaque_image_id"],
            "opaque_group_id": event["opaque_group_id"],
            "product": event["product"],
            "product_slot": event["product_slot"],
            "study_role": event["study_role"],
            "view": event["view"],
            "candidate_kind": event["candidate_kind"],
            "presentation_kind": event["presentation_kind"],
            "first_presentation_id": first_by_image[event["opaque_image_id"]],
            "queue_nonce": event["queue_nonce"],
        })
    repeats = []
    for image_id in sorted(repeat_ids):
        pair = [row for row in queue if row["opaque_image_id"] == image_id]
        first = next(row for row in pair if row["presentation_kind"] == "FIRST")
        repeat = next(row for row in pair if row["presentation_kind"] == "REPEAT")
        repeats.append({
            "opaque_image_id": image_id,
            "opaque_group_id": first["opaque_group_id"],
            "product": first["product"],
            "product_slot": first["product_slot"],
            "study_role": first["study_role"],
            "view": first["view"],
            "first_presentation_id": first["presentation_id"],
            "repeat_presentation_id": repeat["presentation_id"],
            "first_order": first["presentation_order"],
            "repeat_order": repeat["presentation_order"],
        })
    repeat_group_rows = []
    for product, group_id in repeated_groups:
        rows = groups[(product, group_id)]
        repeat_group_rows.append({
            "product": product,
            "opaque_group_id": group_id,
            "study_role": rows[0]["study_role"],
            "view_count": len(rows),
            "selection_domain": "REPEAT_GROUP",
        })
    diagnostics = {
        "unique_images": 120,
        "presentation_count": len(queue),
        "repeat_image_count": len(repeat_ids),
        "repeat_group_count": len(repeated_groups),
        "repeat_group_product_role_counts": {
            f"{slot}:{role}": count
            for (slot, role), count in Counter((groups[k][0]["product_slot"], groups[k][0]["study_role"]) for k in repeated_groups).items()
        },
        "queue_nonce_block": chosen_attempt,
        "min_first_repeat_distance_positions": min(abs(r["first_order"] - r["repeat_order"]) for r in repeats),
        "presentation_ids_unique": len({r["presentation_id"] for r in queue}) == 144,
        "queue_order_hash": "sha256(G0_ORDER_KEY(PRESENTATION_ORDER, nonce))",
        "repeat_group_order_hash": "sha256(G0_ORDER_KEY(REPEAT_GROUP, product+U+001F+opaque_group_id))",
    }
    return queue, repeats, {**diagnostics, "repeated_groups": repeat_group_rows}


def validate_queue(queue: list[dict[str, str]], mapping: dict[str, dict[str, str]]) -> dict[str, int]:
    if len(queue) != 144 or len({row["presentation_id"] for row in queue}) != 144:
        raise IntegrityError("QUEUE_PRESENTATION_COUNT_OR_ID_INVALID")
    images: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in queue:
        image_id = row["opaque_image_id"]
        expected = mapping.get(image_id)
        if expected is None:
            raise IntegrityError("QUEUE_IMAGE_NOT_IN_FROZEN_120")
        for field in ("opaque_group_id", "product", "study_role", "view", "candidate_kind"):
            if row[field] != expected[field]:
                raise IntegrityError("QUEUE_ID_ROLE_ASSOCIATION_MISMATCH")
        if row["presentation_kind"] not in {"FIRST", "REPEAT"}:
            raise IntegrityError("QUEUE_PRESENTATION_KIND_INVALID")
        images[image_id].append(row)
        groups[(row["product"], row["opaque_group_id"])].append(row)
    if len(images) != 120 or sum(len(v) == 2 for v in images.values()) != 24:
        raise IntegrityError("QUEUE_UNIQUE_IMAGE_REPEAT_CARDINALITY")
    if any(len(v) not in {1, 2} or Counter(row["presentation_kind"] for row in v) != Counter({"FIRST": 1, **({"REPEAT": 1} if len(v) == 2 else {})}) for v in images.values()):
        raise IntegrityError("QUEUE_DUPLICATE_PRESENTATION_STRUCTURE")
    if len(groups) != 60 or sum(len(v) == 4 for v in groups.values()) != 12 or any(
        len(v) not in {2, 4} or len({row["study_role"] for row in v}) != 1
        or len({row["product"] for row in v}) != 1
        for v in groups.values()
    ):
        raise IntegrityError("QUEUE_GROUP_PRESENTATION_STRUCTURE")
    return {"images": len(images), "groups": len(groups), "repeated_images": 24, "presentations": len(queue)}


def verify_image_hash(actual_sha256: str, expected_sha256: str) -> None:
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise IntegrityError("SOURCE_IMAGE_SHA_MISMATCH")


def load_dcal_annotations(path: Path, private_root: Path) -> list[dict[str, str]]:
    """D-cal-only loader; refuses paths outside its one frozen CSV."""
    expected = (private_root / "D-cal" / "annotations.csv").resolve(strict=False)
    candidate = path.resolve(strict=False)
    if candidate != expected:
        raise PermissionError("D_CAL_LOADER_PATH_NOT_ALLOWLISTED")
    if path.is_symlink():
        raise PermissionError("D_CAL_LOADER_REPARSE_BLOCKED")
    rows = read_csv(path)
    for row in rows:
        if row.get("attribute_state", "") and row["attribute_state"] not in VALID_STATES:
            raise IntegrityError("ANNOTATION_STATE_INVALID")
    return rows


def assert_output_within_run(path: Path, run_root: Path) -> None:
    resolved_root = run_root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise IntegrityError("OUTPUT_PATH_OUTSIDE_G1B_RUN") from exc
    if "LP_W1_D5R_IMAGE_PILOT_V2_2V".casefold() in str(resolved_path).casefold():
        raise IntegrityError("OLD_V2_OUTPUT_FORBIDDEN")


class SingleWriterLock:
    """OS-level nonblocking single-writer lock requiring no third-party package."""
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError("ANNOTATION_SESSION_ALREADY_OPEN") from exc

    def release(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
