#!/usr/bin/env python3
"""Stage-3 P0 contract linter. Read-only, stdlib only; no dataset images or GT loaded.

A successful default structural check does NOT constitute G0 approval.
Run with --strict to demand that documented unresolved P0 evidence is filled.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "configs" / "stage3" / "experiment_manifest_v1.json"
FORBIDDEN_AGENT_FIELDS = frozenset({
    "ground_truth", "offline_gt", "target_gt", "target_label", "gt_mask",
    "evaluator_label", "oracle_address", "unqueried_mask", "unqueried_type",
})
ACTIONS = frozenset({"classify", "query_binary", "query_rich"})


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object: {path}")
    return data


def field(doc: dict[str, Any], dotted: str) -> Any:
    value: Any = doc
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_folds(folds: dict[str, Any], outer_count: int = 5,
                   targets_each: int = 6) -> list[dict[str, Any]]:
    """Returns four source-CV partitions per outer fold, never sealed targets."""
    if folds.get("schema") != "abmg.realiad.category_folds.v1":
        raise ValueError("Unexpected existing fold schema")
    cats = folds.get("all_categories")
    outer = folds.get("folds")
    if not isinstance(cats, list) or len(cats) != outer_count * targets_each:
        raise ValueError("Category universe must have exactly 30 entries")
    if len(set(cats)) != len(cats):
        raise ValueError("Duplicate categories in category universe")
    if not isinstance(outer, list) or len(outer) != outer_count:
        raise ValueError("Expected five outer folds")
    seen = Counter()
    blocks = []
    for i, f in enumerate(outer):
        if f.get("fold") != i:
            raise ValueError("Outer folds must be numbered 0..4 in order")
        targets = f.get("target_categories")
        if not isinstance(targets, list) or len(targets) != targets_each:
            raise ValueError(f"Fold {i} has wrong target count")
        if len(set(targets)) != targets_each or not set(targets).issubset(cats):
            raise ValueError(f"Fold {i} contains duplicate/unknown categories")
        seen.update(targets)
        blocks.append(set(targets))
    if any(seen[c] != 1 for c in cats):
        raise ValueError("Each product must be sealed as target exactly once")
    partitions = []
    for i, sealed in enumerate(blocks):
        sources = set(cats) - sealed
        inner = []
        for j, val in enumerate(blocks):
            if j == i:
                continue
            develop = sources - val
            if len(develop) != 18 or len(val) != 6 or (develop & val) or (val & sealed):
                raise ValueError(f"Invalid source CV for outer {i}, held-out block {j}")
            inner.append({"source_validation_fold": j,
                          "development": sorted(develop),
                          "validation": sorted(val)})
        if len(inner) != 4 or set.union(*(set(x["validation"]) for x in inner)) != sources:
            raise ValueError("Four source-CV folds must partition the 24 sources")
        partitions.append({"outer_fold": i,
                           "sealed_target": sorted(sealed),
                           "source_cv": inner})
    return partitions


def unresolved_fields(manifest: dict[str, Any]) -> list[str]:
    pending = []
    for name in manifest.get("required_before_g0", []):
        val = field(manifest, name)
        if val is None or val is False or val is True and name.endswith("unverified") or val == "":
            pending.append(name)
    return pending


def check_manifest(path: Path = DEFAULT_MANIFEST, *, strict: bool = False) -> dict[str, Any]:
    m = load_json(path)
    problems = []
    if m.get("schema") != "eg-famar.experiment_manifest.v1":
        problems.append("Incorrect manifest schema")
    if m.get("phase") != "P0":
        problems.append("Incorrect current phase")
    try:
        folded = load_json(REPO / m["dataset"]["outer_folds_path"])
        partitions = validate_folds(folded, m["dataset"]["outer_folds"],
                                    m["dataset"]["target_categories_per_fold"])
    except (KeyError, ValueError, OSError) as exc:
        problems.append(f"Fold contract: {exc}")
        partitions = []
    try:
        source_cv = load_json(REPO / m["dataset"]["source_cv_file"])
        if source_cv.get("schema") != "eg-famar.source_cv.v1":
            raise ValueError("Source CV schema mismatch")
        declared = source_cv.get("outer_folds", [])
        if len(declared) != len(partitions):
            raise ValueError("Source CV outer fold count mismatch")
        for expected, actual in zip(partitions, declared):
            if actual.get("outer_fold") != expected["outer_fold"]:
                raise ValueError("Source CV outer fold index mismatch")
            if set(actual.get("sealed_target", [])) != set(expected["sealed_target"]):
                raise ValueError("Source CV sealed target mismatch")
            if len(actual.get("source_cv", [])) != 4:
                raise ValueError("Source CV requires four groups")
            for e, a in zip(expected["source_cv"], actual["source_cv"]):
                if (a.get("held_out_original_block") != e["source_validation_fold"]
                    or set(a.get("development_categories", [])) != set(e["development"])
                    or set(a.get("validation_categories", [])) != set(e["validation"])):
                    raise ValueError("Frozen Source CV partition mismatch")
    except (KeyError, ValueError, OSError) as exc:
        problems.append(f"Frozen source-CV contract: {exc}")
    for path_key in ("v22_path", "visionad_path", "inventory_path"):
        src = m.get("source", {}).get(path_key)
        if not isinstance(src, str) or not (REPO / src).is_file():
            problems.append(f"Missing existing source file: {path_key}")
    tiers = m.get("feedback", {})
    if tiers.get("operator_tiers") != ["B", "R"]:
        problems.append("Only B and R human interaction tiers are authorized")
    if tiers.get("B", {}).get("fields") != ["binary"]:
        problems.append("B grants image-level binary only")
    if tiers.get("R", {}).get("fields") != ["binary", "defect_type", "mask"]:
        problems.append("R must grant full Binary+Type+Mask")
    if tiers.get("unqueried_gt_visible") is not False:
        problems.append("Unqueried GT must never reach agent")
    if m.get("sensor", {}).get("frozen") is not True:
        problems.append("Online visual backbone must be frozen")
    if m.get("dataset", {}).get("sealed_targets_evaluator_only") is not True:
        problems.append("Sealed targets must be evaluation-only")
    if not m.get("evaluation", {}).get("prequential"):
        problems.append("Prequential scoring contract missing")
    if m.get("query", {}).get("identical_ids_for_mechanism_isolation") is not True:
        problems.append("Matched-replay query IDs must be fixed")
    pending = unresolved_fields(m)
    if strict and (pending or m.get("status") != "READY_FOR_G0_REVIEW"):
        problems.append("G0 readiness is BLOCKED: unresolved P0 evidence or status")
    return {"schema": "eg-famar.p0.validation.v1",
            "structurally_valid": not problems if not strict else not problems,
            "g0_ready": not problems and not pending and m.get("status") == "READY_FOR_G0_REVIEW",
            "outer_folds_validated": len(partitions),
            "source_cv_partitions_validated": sum(len(p["source_cv"]) for p in partitions),
            "pending_evidence": pending, "errors": problems}


def validate_agent_event(event: dict[str, Any]) -> None:
    """Fail closed if evaluator-only evidence appears in any agent-view event.

    Payload shape: time, image_id, action, pre (scores/state), revealed (fields
    actually acquired), paid_binary, paid_rich_incremental, prior_binary_query.
    This validates a trace contract; it is not wired into legacy v22 runtime.
    """
    if not isinstance(event, dict):
        raise ValueError("Expected an event object")
    if not isinstance(event.get("time"), int) or event["time"] < 0:
        raise ValueError("Nonnegative time required")
    if not isinstance(event.get("image_id"), str) or not event["image_id"]:
        raise ValueError("Stable image_id required")
    act = event.get("action")
    if act not in ACTIONS:
        raise ValueError("Unsupported action")
    pre = event.get("pre")
    if not isinstance(pre, dict) or "score" not in pre or "threshold" not in pre:
        raise ValueError("Pre-feedback score and threshold required")
    revealed = event.get("revealed")
    if not isinstance(revealed, dict):
        raise ValueError("revealed must be a dictionary (empty when not queried)")
    def scan(value: Any) -> None:
        if isinstance(value, dict):
            if FORBIDDEN_AGENT_FIELDS.intersection(value):
                raise ValueError("Evaluator-only field present in agent event")
            for v in value.values():
                scan(v)
        elif isinstance(value, list):
            for v in value:
                scan(v)
    scan(event)
    if act == "classify":
        if revealed or event.get("paid_binary") or event.get("paid_rich_incremental"):
            raise ValueError("No feedback can be revealed without a query")
    elif act == "query_binary":
        if set(revealed) != {"binary"} or type(revealed["binary"]) is not int or revealed["binary"] not in (0, 1):
            raise ValueError("Binary query only authorizes {binary:0|1}")
        if event.get("paid_binary") is not True or event.get("paid_rich_incremental"):
            raise ValueError("Binary payment contract violated")
    else:
        if set(revealed) != {"binary", "defect_type", "mask_ref"}:
            raise ValueError("Rich defect query requires binary/type/mask_ref")
        if (type(revealed["binary"]) is not int or revealed["binary"] != 1
                or not isinstance(revealed["defect_type"], str)
                or not revealed["defect_type"]
                or not isinstance(revealed["mask_ref"], str)
                or not revealed["mask_ref"]):
            raise ValueError("Rich requires real verified defect type and mask")
        if (event.get("paid_binary") is not True
                or event.get("paid_rich_incremental") is not True
                or event.get("prior_binary_query") is not True):
            raise ValueError("Rich must follow a paid Binary defect verification")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--strict", action="store_true",
                        help="Block until all G0 prerequisites and review status are resolved")
    parser.add_argument("--trace", type=Path,
                        help="Optional JSONL agent-visible trace to validate without loading evaluator GT")
    args = parser.parse_args()
    report = check_manifest(args.manifest, strict=args.strict)
    if args.trace:
        checked = 0
        with args.trace.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    validate_agent_event(json.loads(line))
                    checked += 1
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    report["errors"].append(f"trace line {checked+1}: {exc}")
        report["events_validated"] = checked
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
