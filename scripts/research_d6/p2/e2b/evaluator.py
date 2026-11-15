"""Evaluator-only product metrics for the fixed W/X development views."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import config


def binary_target(state: str) -> int | None:
    if state not in config.VALID_STATES:
        raise ValueError(f"unknown attribute state {state!r}")
    return config.VALID_STATES[state]


def summarize(scores: np.ndarray, future_rows: list[dict[str, Any]],
              state_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(future_rows) or not np.isfinite(values).all():
        raise ValueError("future score count or finiteness error")
    y_states = [state_by_id[r["image_id"]]["state"] for r in future_rows]
    products = sorted({r["product"] for r in future_rows})
    pred = (values >= 0.0).astype(np.int8)
    product_rows = []
    ba_values: list[float] = []
    ap_paired_values: list[float] = []
    ap_all_values: list[float] = []
    auc_values: list[float] = []
    per_item = []
    for i, row in enumerate(future_rows):
        y = binary_target(y_states[i])
        per_item.append({
            "image_id": row["image_id"], "product": row["product"],
            "sample_key": row["sample_key"], "view": row["view"],
            "source_type_evaluator_only": row.get("source_type_evaluator_only", "UNKNOWN"),
            "attribute_state_evaluator_only": y_states[i],
            "binary_target_evaluator_only": y,
            "score": float(values[i]), "threshold": 0.0,
            "prediction": int(pred[i]),
        })
    for product in products:
        idx = [i for i, row in enumerate(future_rows)
               if row["product"] == product and binary_target(y_states[i]) is not None]
        labels = np.asarray([binary_target(y_states[i]) for i in idx], dtype=np.int8)
        local_scores = values[idx]
        local_pred = pred[idx]
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        n_binary = len(labels)
        ba = None
        ba_status = "undefined_missing_class"
        if positives and negatives:
            tpr = float(np.mean(local_pred[labels == 1] == 1))
            tnr = float(np.mean(local_pred[labels == 0] == 0))
            ba = 0.5 * (tpr + tnr)
            ba_status = "defined"
            ba_values.append(ba)
        else:
            tpr = float(np.mean(local_pred[labels == 1] == 1)) if positives else None
            tnr = float(np.mean(local_pred[labels == 0] == 0)) if negatives else None
        ap = None
        ap_status = "undefined_no_positive"
        if positives:
            ap = float(average_precision_score(labels, local_scores))
            if negatives == 0:
                ap_status = "degenerate_all_positive"
                ap_all_values.append(ap)
            else:
                ap_status = "defined_both_classes"
                ap_all_values.append(ap)
                ap_paired_values.append(ap)
        auc = None
        auc_status = "undefined_missing_class"
        if positives and negatives:
            auc = float(roc_auc_score(labels, local_scores))
            auc_status = "defined"
            auc_values.append(auc)
        product_rows.append({
            "product": product,
            "n_views": sum(r["product"] == product for r in future_rows),
            "n_binary": n_binary,
            "present": positives,
            "absent": negatives,
            "uncertain": sum(r["product"] == product and y_states[i] == "unobservable_or_uncertain"
                             for i, r in enumerate(future_rows)),
            "not_applicable": sum(r["product"] == product and y_states[i] == "not_applicable"
                                   for i, r in enumerate(future_rows)),
            "ba": ba, "ba_status": ba_status, "tpr": tpr, "tnr": tnr,
            "ap": ap, "ap_status": ap_status,
            "roc_auc": auc, "roc_auc_status": auc_status,
        })
    binary_count = sum(binary_target(s) is not None for s in y_states)
    summary = {
        "future_views": len(future_rows),
        "future_binary_views": binary_count,
        "present": sum(binary_target(s) == 1 for s in y_states),
        "absent": sum(binary_target(s) == 0 for s in y_states),
        "uncertain": sum(s == "unobservable_or_uncertain" for s in y_states),
        "not_applicable": sum(s == "not_applicable" for s in y_states),
        "product_count": len(products),
        "ba_product_macro": float(np.mean(ba_values)) if ba_values else None,
        "ba_product_count": len(ba_values),
        "ap_product_macro_paired": float(np.mean(ap_paired_values)) if ap_paired_values else None,
        "ap_paired_product_count": len(ap_paired_values),
        "ap_product_macro_including_degenerate": float(np.mean(ap_all_values)) if ap_all_values else None,
        "ap_product_count_including_degenerate": len(ap_all_values),
        "roc_auc_product_macro": float(np.mean(auc_values)) if auc_values else None,
        "roc_auc_product_count": len(auc_values),
        "positive_prediction_rate_binary": float(np.mean(pred[[binary_target(s) is not None for s in y_states]] == 1))
        if binary_count else None,
    }
    return summary, product_rows, per_item


def source_type_state_counts(future_rows: list[dict[str, Any]],
                             state_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counts = defaultdict(int)
    for row in future_rows:
        state = state_by_id[row["image_id"]]["state"]
        counts[(row.get("source_type_evaluator_only", "UNKNOWN"), state)] += 1
    return [{"source_type_evaluator_only": k[0], "attribute_state_evaluator_only": k[1],
             "count": v} for k, v in sorted(counts.items())]
