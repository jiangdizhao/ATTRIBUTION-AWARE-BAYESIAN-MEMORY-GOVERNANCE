"""Causal predict-before-reveal online tape execution, independent of data labels."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from model import parameter_digest


def run_trajectory(*, protocol: str, trajectory_id: str,
                   initial_theta: np.ndarray,
                   initial_features: list[Any], initial_labels: list[int],
                   initial_image_ids: list[str], feedback_events: list[dict[str, Any]],
                   reveal_state: Callable[[str], str],
                   fit_history: Callable[[list[Any], np.ndarray, np.ndarray, int], dict[str, Any]],
                   predict_one: Callable[[np.ndarray, Any], float],
                   predict_future: Callable[[np.ndarray, list[Any]], np.ndarray],
                   future_features: list[Any], future_rows: list[dict[str, Any]],
                   evaluate: Callable[[np.ndarray], tuple[dict, list[dict], list[dict]]],
                   update_enabled: bool) -> dict[str, Any]:
    """Run one model/control; labels are revealed only after each query prediction."""
    theta0 = np.asarray(initial_theta, dtype=np.float64).copy()
    theta = theta0.copy()
    base_digest = parameter_digest(theta0)
    history_features = list(initial_features)
    history_labels = [int(y) for y in initial_labels]
    history_image_ids = list(initial_image_ids)
    if len(history_features) != len(history_labels) or len(history_labels) != len(history_image_ids):
        raise ValueError("initial training history mismatch")
    if len(set(history_image_ids)) != len(history_image_ids):
        raise ValueError("initial training image IDs are not unique")
    if set(history_labels) != {0, 1}:
        raise ValueError("initial training history must contain both classes")
    seen_events: set[str] = set()
    seen_images = set(history_image_ids)
    checkpoints: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    def checkpoint(index: int, event_count: int, binary_updates: int,
                   fit_seconds: float, solver_record: dict | None, status: str = "PASS"):
        scores = np.asarray(predict_future(theta, future_features), dtype=np.float64).reshape(-1)
        metrics, products, items = evaluate(scores)
        checkpoints.append({
            "protocol": protocol, "trajectory_id": trajectory_id,
            "checkpoint": index, "events_seen": event_count,
            "binary_updates": binary_updates,
            "training_rows": len(history_labels),
            "state_sha256": parameter_digest(theta),
            "status": status,
            "fit_seconds": fit_seconds,
            "solver_record": solver_record or {},
            "metrics": metrics, "product_metrics": products,
            "item_predictions": items,
            "scores": scores,
            "theta": theta.copy(),
        })

    checkpoint(0, 0, 0, 0.0, None)
    binary_updates = 0
    stopped = False
    for event_index, event in enumerate(sorted(feedback_events, key=lambda x: int(x["stream_order"])), start=1):
        event_id = str(event["event_id"])
        image_id = str(event["image_id"])
        if event_id in seen_events:
            raise ValueError("duplicate feedback event would repeat an annotation")
        if image_id in seen_images:
            raise ValueError("a feedback image cannot be inserted twice or overlap D-fit")
        if int(event["stream_order"]) != event_index:
            raise ValueError("feedback stream order must be exactly 1..12")

        # Save the query score and model-state summary before the reveal callback runs.
        pre_state = parameter_digest(theta)
        query_score = float(predict_one(theta, event["features"]))
        if not np.isfinite(query_score):
            raise FloatingPointError("nonfinite query prediction")
        seen_events.add(event_id)
        seen_images.add(image_id)
        revealed_state = reveal_state(event_id)
        if revealed_state not in {"present", "absent_assessable", "unobservable_or_uncertain", "not_applicable"}:
            raise ValueError("feedback reveal returned an invalid state")
        y = {"present": 1, "absent_assessable": 0,
             "unobservable_or_uncertain": None, "not_applicable": None}[revealed_state]
        event_row = {
            "protocol": protocol, "trajectory_id": trajectory_id,
            "event_id": event_id, "image_id": image_id,
            "stream_order": event_index,
            "feedback_product_evaluator_only": event.get("product", ""),
            "feedback_sample_key_evaluator_only": event.get("sample_key", ""),
            "feedback_view_evaluator_only": event.get("view", ""),
            "score_before_reveal": query_score,
            "prediction_before_reveal": int(query_score >= 0.0),
            "state_sha256_before_reveal": pre_state,
            "revealed_state_after_prediction": revealed_state,
            "binary_target_added_after_reveal": y,
            "updated": False,
            "status": "NO_BINARY_UPDATE" if y is None else "PENDING_UPDATE",
        }
        fit_seconds = 0.0
        solver_record = None
        if update_enabled and y is not None:
            history_features.append(event["features"])
            history_labels.append(int(y))
            history_image_ids.append(image_id)
            fit_result = fit_history(
                history_features, np.asarray(history_labels, dtype=np.int8), theta.copy(), event_index
            )
            fit_seconds = float(fit_result.get("elapsed_seconds", 0.0))
            solver_record = {k: v for k, v in fit_result.items() if k not in {"theta"}}
            if not fit_result.get("accepted", False):
                # The failed candidate is never installed as a valid model state.
                history_features.pop(); history_labels.pop(); history_image_ids.pop()
                event_row["status"] = fit_result.get("status", "SOLVER_FAILURE")
                event_rows.append(event_row)
                stopped = True
                break
            theta = np.asarray(fit_result["theta"], dtype=np.float64).copy()
            binary_updates += 1
            event_row["updated"] = True
            event_row["status"] = "PASS"
            event_row["fit_objective"] = fit_result["objective"]
            event_row["fit_gradient_inf_norm"] = fit_result["gradient_inf_norm"]
        elif y is None:
            # Uncertain/N-A consumes this event index but contributes no binary sample.
            event_row["status"] = "NO_BINARY_UPDATE"
        else:
            # Frozen B0 observes the event only for matched cost/ordering accounting.
            event_row["status"] = "FROZEN_CONTROL_NO_UPDATE"

        if not update_enabled and parameter_digest(theta) != base_digest:
            raise AssertionError("B0 frozen state changed")
        if not update_enabled and y is not None:
            event_row["binary_target_added_after_reveal"] = "not_used_by_frozen_control"
        event_rows.append(event_row)
        checkpoint(event_index, event_index, binary_updates, fit_seconds, solver_record)

    return {
        "protocol": protocol,
        "trajectory_id": trajectory_id,
        "status": "SOLVER_STOPPED" if stopped else "PASS",
        "theta_initial": theta0,
        "theta_final": theta.copy(),
        "initial_state_sha256": base_digest,
        "final_state_sha256": parameter_digest(theta),
        "binary_updates": binary_updates,
        "training_rows_final": len(history_labels),
        "checkpoints": checkpoints,
        "events": event_rows,
        "update_enabled": update_enabled,
    }
