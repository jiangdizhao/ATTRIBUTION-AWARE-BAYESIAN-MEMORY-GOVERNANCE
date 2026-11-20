"""Isolated, label-free H12 scoring and pre-registered G1A stress selection."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1a_common as c
from stress_core import (group_medians, nearest, select_stress, unit_mean, local_cells,
                         cosine_distance, validate_stress_manifest, h12_score)


def checked_feature(output: Path, image_id: str, manifest_row: dict) -> dict:
    base = output / "dinov2" / "items" / image_id
    recpath = base / "record.json"
    if not recpath.is_file() or c.sha_file(recpath) != manifest_row["record_sha256"]:
        raise RuntimeError("FEATURE_RECORD_MISSING_OR_SHA_MISMATCH")
    record = json.loads(recpath.read_text(encoding="utf-8"))
    identity = record.get("identity", {})
    if identity.get("opaque_image_id") != image_id or identity.get("source_sha256") != manifest_row["source_sha256"]:
        raise RuntimeError("FEATURE_IDENTITY_MISMATCH")
    arrays = {}
    for name in ("patch_unit", "valid_patch_mask"):
        spec = record["arrays"][name]
        path = base / spec["file"]
        if c.sha_file(path) != spec["file_sha256"]:
            raise RuntimeError("FEATURE_ARRAY_FILE_SHA_MISMATCH")
        arr = np.load(path, allow_pickle=False)
        if c.sha_array(arr) != spec["array_sha256"]:
            raise RuntimeError("FEATURE_ARRAY_CONTENT_SHA_MISMATCH")
        arrays[name] = arr
    patches = np.asarray(arrays["patch_unit"], dtype=np.float32)
    mask = np.asarray(arrays["valid_patch_mask"], dtype=np.bool_)
    if patches.shape != (1024, 1024) or mask.shape != (1024,) or not mask.any() or not np.isfinite(patches).all():
        raise RuntimeError("FEATURE_ARRAY_SHAPE_OR_VALUE_INVALID")
    return {"patches": patches, "mask": mask, "global": unit_mean(patches, mask),
            "local": local_cells(patches, mask)}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> str:
    return c.write_csv_once(path, rows, fields)


def run(output: Path, feature_manifest_name: str = "feature_manifest.csv") -> dict:
    # The selector's only tabular input is a deliberately stripped manifest.
    input_rows = c.read_csv(output / "stress_input_manifest.csv")
    if not input_rows:
        raise RuntimeError("STRESS_INPUT_EMPTY")
    validate_stress_manifest(input_rows, expected_count=len(input_rows))
    feature_rows = c.read_csv(output / feature_manifest_name)
    fmap = {r["opaque_image_id"]: r for r in feature_rows}
    if len(fmap) != len(feature_rows):
        raise RuntimeError("DUPLICATE_FEATURE_MANIFEST_ID")
    all_ids = {r["opaque_image_id"] for r in input_rows}
    if not set(fmap).issubset(all_ids):
        raise RuntimeError("FEATURE_SCOPE_OUTSIDE_ALLOWLIST")
    theta_path = output / "history" / "h12_theta.npy"
    theta = np.load(theta_path, allow_pickle=False).astype(np.float64, copy=False)
    if theta.shape != (1025,) or not np.isfinite(theta).all():
        raise RuntimeError("H12_PARAMETER_SHAPE_INVALID")
    anchor_npz = np.load(output / "history" / "anchor_features.npz", allow_pickle=False)
    anchor_ids = anchor_npz["image_ids"].astype(str).tolist()
    event_indices = anchor_npz["event_indices"].astype(np.int64).tolist()
    anchor_patches = anchor_npz["patches"]
    anchor_masks = anchor_npz["masks"]
    if (len(anchor_ids) != 5 or len(set(anchor_ids)) != 5 or len(event_indices) != 5
            or anchor_patches.shape != (5, 1024, 1024) or anchor_masks.shape != (5, 1024)
            or sorted(event_indices) != event_indices):
        raise RuntimeError("HISTORICAL_ANCHOR_FEATURE_SCOPE_INVALID")

    features: dict[str, dict] = {}
    available_rows = [r for r in input_rows if r["opaque_image_id"] in fmap]
    for row in available_rows:
        features[row["opaque_image_id"]] = checked_feature(output, row["opaque_image_id"], fmap[row["opaque_image_id"]])
    anchor_records = []
    for image_id, event_index, patches, mask in zip(anchor_ids, event_indices, anchor_patches, anchor_masks):
        f = {"patches": patches, "mask": mask.astype(bool)}
        f["global"] = unit_mean(f["patches"], f["mask"])
        f["local"] = local_cells(f["patches"], f["mask"])
        features[image_id] = f
        anchor_records.append({"image_id": image_id, "event_index": int(event_index)})

    score_rows = []
    by_image = {r["opaque_image_id"]: r for r in input_rows}
    for image_id in sorted(by_image):
        row = by_image[image_id]
        if image_id not in features:
            continue
        f = features[image_id]
        logit = h12_score(f["global"], theta)
        g_anchor, g_distance = min(
            ((a, cosine_distance(f["global"], features[a["image_id"]]["global"])) for a in anchor_records),
            key=lambda v: (v[1], v[0]["event_index"], v[0]["image_id"]))
        m_anchor, m_distance, m_global, m_local = nearest(
            {"image_id": image_id}, anchor_records, features)
        score_rows.append({"opaque_image_id": image_id, "opaque_group_id": row["opaque_group_id"],
                           "product_slot": row["product_slot"], "study_role": row["study_role"],
                           "candidate_kind": row["candidate_kind"], "view": row["view"],
                           "h_logit": logit, "h_prediction": int(logit >= 0.0),
                           "global_anchor_id": g_anchor["image_id"], "global_anchor_event": g_anchor["event_index"],
                           "global_distance": float(g_distance), "mixed_anchor_id": m_anchor["image_id"],
                           "mixed_anchor_event": m_anchor["event_index"], "mixed_distance": m_distance,
                           "mixed_global_component": m_global, "mixed_local_component": m_local,
                           "local_cell_count": int(len(f["local"])),
                           "tie_key": row["stress_tie_key"]})
    counts: dict[tuple[str, str], int] = {}
    for row in score_rows:
        key = (row["product_slot"], row["opaque_group_id"])
        counts[key] = counts.get(key, 0) + 1
    incomplete_group_keys = {key for key, count in counts.items() if count != 2}
    complete_rows = [r for r in score_rows if (r["product_slot"], r["opaque_group_id"]) not in incomplete_group_keys]
    group_rows = group_medians(complete_rows)
    selection = select_stress(group_rows, sorted({r["product_slot"] for r in input_rows}, key=int))
    selected_keys = {(r["product_slot"], r["group_id"]) for r in selection["selected"]}
    role_rows = []
    for group in group_rows:
        if group["candidate_kind"] == "REPRESENTATIVE_CANDIDATE":
            chosen = True
            role_reason = "FROZEN_REPRESENTATIVE"
        else:
            chosen = (group["product_slot"], group["group_id"]) in selected_keys
            role_reason = "PRE_REGISTERED_STRESS_BIN" if chosen else "STRESS_POOL_NOT_SELECTED"
        if chosen:
            members = [r for r in available_rows if r["product_slot"] == group["product_slot"]
                       and r["opaque_group_id"] == group["group_id"]]
            if len(members) != 2:
                raise RuntimeError("ROLE_GROUP_VIEW_CARDINALITY")
            role_rows.extend({"opaque_image_id": m["opaque_image_id"], "opaque_group_id": m["opaque_group_id"],
                              "product_slot": m["product_slot"], "study_role": m["study_role"],
                              "view": m["view"], "candidate_kind": m["candidate_kind"],
                              "role_reason": role_reason} for m in members)

    scores_sha = write_csv(output / "unlabeled_scores.csv", score_rows)
    group_sha = write_csv(output / "group_stress_scores.csv", group_rows)
    stress_sha = write_csv(output / "stress_selection.csv", selection["selected"])
    roles_sha = write_csv(output / "development_roles.csv", role_rows)
    summary = {"status": selection["status"], "candidate_views_authorized": len(input_rows),
               "candidate_views_scored": len(score_rows), "candidate_views_missing_features": len(input_rows) - len(available_rows),
               "incomplete_groups_after_feature_failures": len(incomplete_group_keys),
               "representative_groups": sum(r["candidate_kind"] == "REPRESENTATIVE_CANDIDATE" for r in group_rows),
               "stress_pool_groups": sum(r["candidate_kind"] == "STRESS_POOL_CANDIDATE" for r in group_rows),
               "stress_selected_groups": len(selection["selected"]), "development_groups": len(role_rows) // 2,
               "development_views": len(role_rows), "per_product_bin_diagnostics": selection["diagnostics"],
               "unlabeled_scores_sha256": scores_sha, "group_stress_scores_sha256": group_sha,
               "stress_selection_sha256": stress_sha, "development_roles_sha256": roles_sha,
               "new_attribute_labels_read": False, "defect_type_read": False,
               "gt_masks_read": False, "future_truth_read": False}
    c.write_json_once(output / "selection_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "candidate_views_scored", "representative_groups",
                                               "stress_pool_groups", "stress_selected_groups",
                                               "development_groups", "development_views")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-manifest", default="feature_manifest.csv")
    args = parser.parse_args()
    run(Path(args.output), args.feature_manifest)
