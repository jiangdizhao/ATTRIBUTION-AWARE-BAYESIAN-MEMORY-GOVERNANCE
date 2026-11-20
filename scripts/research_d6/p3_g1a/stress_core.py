"""Pure label-free H12 and spatial-distance stress scoring for LP-D6-P3-G1A."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import numpy as np

SALT = "LP-D6-P3-G0-v0.2-20261120"
SEP = "\x1f"
STRESS_FIELDS = {"opaque_image_id", "opaque_group_id", "product_slot", "study_role", "view",
                 "candidate_kind", "stress_tie_key"}


def validate_stress_manifest(rows: list[dict], *, expected_count: int = 216) -> None:
    if len(rows) != expected_count or len(rows) > 216:
        raise ValueError("STRESS_INPUT_CARDINALITY")
    if any(set(row) != STRESS_FIELDS for row in rows):
        raise ValueError("STRESS_INPUT_SCHEMA_NOT_LABEL_FREE")
    ids = [r["opaque_image_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("STRESS_INPUT_DUPLICATE_IMAGE")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["study_role"] not in {"P3_G1A_D_CAL", "P3_G1A_D_EVAL"}:
            raise ValueError("STRESS_INPUT_ROLE_INVALID")
        if row["candidate_kind"] not in {"REPRESENTATIVE_CANDIDATE", "STRESS_POOL_CANDIDATE"}:
            raise ValueError("STRESS_INPUT_KIND_INVALID")
        if row["view"] not in {"C1", "C2", "C3", "C4", "C5"}:
            raise ValueError("STRESS_INPUT_VIEW_INVALID")
        groups[row["opaque_group_id"]].append(row)
    if len(groups) * 2 != len(rows):
        raise ValueError("STRESS_INPUT_GROUP_COUNT")
    for items in groups.values():
        if (len(items) != 2 or len({r["view"] for r in items}) != 2
                or len({r["study_role"] for r in items}) != 1
                or len({r["product_slot"] for r in items}) != 1
                or len({r["candidate_kind"] for r in items}) != 1
                or len({r["stress_tie_key"] for r in items}) != 1):
            raise ValueError("STRESS_INPUT_GROUP_ROLE_OR_VIEW_CROSSING")


def h12_score(global_feature: np.ndarray, theta: np.ndarray) -> float:
    x = np.asarray(global_feature, dtype=np.float64)
    w = np.asarray(theta, dtype=np.float64)
    if x.ndim != 1 or w.shape != (len(x) + 1,) or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("H12_FEATURE_OR_PARAMETER_INVALID")
    result = float(x @ w[:-1] + w[-1])
    if not np.isfinite(result):
        raise ValueError("H12_SCORE_NONFINITE")
    return result


def order_key(domain: str, key: str) -> tuple[bytes, bytes]:
    payload = (SALT + SEP + domain + SEP + key).encode("utf-8")
    return hashlib.sha256(payload).digest(), key.encode("utf-8")


def unit_mean(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if x.ndim != 2 or len(valid) != len(x) or not valid.any() or not np.isfinite(x).all():
        raise ValueError("INVALID_GLOBAL_FEATURE_INPUT")
    mean = x[valid].mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("INVALID_GLOBAL_MEAN_NORM")
    return mean / norm


def local_cells(patches: np.ndarray, mask: np.ndarray, bins: int = 4) -> np.ndarray:
    z = np.asarray(patches, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.bool_).reshape(-1)
    side = math.isqrt(len(valid))
    if z.ndim != 2 or len(z) != len(valid) or side * side != len(valid):
        raise ValueError("PATCH_GRID_SHAPE_INVALID")
    if not np.isfinite(z).all() or not valid.any():
        raise ValueError("PATCH_GRID_NONFINITE_OR_EMPTY")
    ids = np.flatnonzero(valid)
    cells = []
    for i in range(bins):
        for j in range(bins):
            keep = ((ids // side * bins // side == i)
                    & (ids % side * bins // side == j))
            if keep.any():
                mean = z[ids[keep]].mean(axis=0)
                norm = float(np.linalg.norm(mean))
                if np.isfinite(norm) and norm > 1e-12:
                    cells.append(mean / norm)
    if not cells:
        raise ValueError("NO_NONZERO_LOCAL_CELLS")
    return np.stack(cells)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("DISTANCE_INPUT_INVALID")
    val = float(np.clip((1.0 - float(np.dot(x, y))) / 2.0, 0.0, 1.0))
    if not np.isfinite(val):
        raise ValueError("DISTANCE_NONFINITE")
    return val


def local_distance(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("LOCAL_DISTANCE_SHAPE_INVALID")
    if not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("LOCAL_DISTANCE_EMPTY_OR_NONFINITE")
    d = np.clip((1.0 - x @ y.T) / 2.0, 0.0, 1.0)
    value = float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2.0)
    if not np.isfinite(value):
        raise ValueError("LOCAL_DISTANCE_NONFINITE")
    return value


def nearest(query: dict, anchors: list[dict], features: dict[str, dict]) -> tuple[dict, float, float, float]:
    if not anchors:
        raise ValueError("EMPTY_ANCHOR_SET")
    candidates = []
    for anchor in anchors:
        qid, aid = query["image_id"], anchor["image_id"]
        if qid == aid:
            raise ValueError("SELF_MATCH_FORBIDDEN")
        q, a = features[qid], features[aid]
        dg = cosine_distance(q["global"], a["global"])
        dl = local_distance(q["local"], a["local"])
        dm = 0.5 * dg + 0.5 * dl
        candidates.append((dm, int(anchor["event_index"]), aid, anchor, dg, dl))
    dm, _, _, anchor, dg, dl = min(candidates, key=lambda v: (v[0], v[1], v[2]))
    return anchor, float(dm), float(dg), float(dl)


def group_medians(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        group_id = row.get("group_id", row.get("opaque_group_id"))
        if not group_id:
            raise ValueError("STRESS_GROUP_ID_MISSING")
        groups[(row["product_slot"], group_id)].append(row)
    output = []
    for (slot, group), items in groups.items():
        image_ids = [x.get("image_id", x.get("opaque_image_id")) for x in items]
        if len(items) != 2 or len(set(image_ids)) != 2 or any(not x for x in image_ids):
            raise ValueError("STRESS_GROUP_VIEW_CARDINALITY")
        if len({x["candidate_kind"] for x in items}) != 1:
            raise ValueError("STRESS_GROUP_KIND_MISMATCH")
        output.append({"product_slot": slot, "group_id": group,
                       "candidate_kind": items[0]["candidate_kind"],
                       "c": float(np.median([abs(float(x["h_logit"])) for x in items])),
                       "d": float(np.median([float(x["mixed_distance"]) for x in items])),
                       "tie_key": items[0]["tie_key"], "image_ids": image_ids})
    return output


def select_stress(groups: list[dict], expected_product_slots: list[str]) -> dict[str, Any]:
    """Select one group in each frozen low/high confidence × distance bin.

    Median ties are assigned to the low side. No label, source type, mask path,
    filename, or model outcome is accepted by this function.
    """
    selected, diagnostics = [], []
    for slot in expected_product_slots:
        pool = [g for g in groups if g["product_slot"] == slot and g["candidate_kind"] == "STRESS_POOL_CANDIDATE"]
        if len(pool) != 12:
            diagnostics.append({"product_slot": slot, "status": "STRESS_POOL_INCOMPLETE", "pool_groups": len(pool)})
            continue
        cm = float(np.median([g["c"] for g in pool]))
        dm = float(np.median([g["d"] for g in pool]))
        bins: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for g in pool:
            cb = "LOW" if g["c"] <= cm else "HIGH"
            db = "LOW" if g["d"] <= dm else "HIGH"
            bins[(cb, db)].append(g)
        coverage = {f"{c}_{d}": len(bins[(c, d)]) for c in ("LOW", "HIGH") for d in ("LOW", "HIGH")}
        if any(value == 0 for value in coverage.values()):
            diagnostics.append({"product_slot": slot, "status": "STRESS_BIN_COVERAGE_INSUFFICIENT",
                                "pool_groups": len(pool), "confidence_median": cm,
                                "distance_median": dm, "bin_counts": coverage})
            continue
        for cb in ("LOW", "HIGH"):
            for db in ("LOW", "HIGH"):
                candidates = bins[(cb, db)]
                winner = min(candidates, key=lambda g: (bytes.fromhex(g["tie_key"]), g["group_id"].encode("utf-8")))
                selected.append({**winner, "confidence_bin": cb, "distance_bin": db,
                                 "confidence_median": cm, "distance_median": dm,
                                 "bin_candidate_count": len(candidates)})
        diagnostics.append({"product_slot": slot, "status": "PASS", "pool_groups": len(pool),
                            "confidence_median": cm, "distance_median": dm, "bin_counts": coverage})
    return {"selected": selected, "diagnostics": diagnostics,
            "status": "PASS" if len(selected) == 4 * len(expected_product_slots) else "PARTIAL"}
