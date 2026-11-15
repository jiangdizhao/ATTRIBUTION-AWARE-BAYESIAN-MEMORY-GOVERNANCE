"""Atomic, resumable per-checkpoint execution primitives for E2-B-R1."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np


class IntegrityError(RuntimeError):
    pass


class ExecutionStop(RuntimeError):
    pass


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_exclusive(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    payload = canonical_bytes(value) + b"\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return sha256_file(path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> str:
    """Atomic replacement for mutable ledgers; scientific records use exclusive writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return sha256_file(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IntegrityError(f"invalid JSON record: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"expected object JSON: {path}")
    return value


def _record_digest(metadata: dict[str, Any], state_sha: str) -> str:
    body = {k: v for k, v in metadata.items() if k != "record_sha256"}
    return sha256_bytes(canonical_bytes({"metadata": body, "state_npz_sha256": state_sha}))


def write_bundle(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Write a complete checkpoint as a temporary directory then atomically publish it."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.pending-{os.getpid()}-{next(tempfile._get_candidate_names())}"
    temp.mkdir()
    try:
        state_path = temp / "state.npz"
        with state_path.open("xb") as stream:
            np.savez_compressed(stream, **{k: np.asarray(v) for k, v in arrays.items()})
            stream.flush()
            os.fsync(stream.fileno())
        state_sha = sha256_file(state_path)
        record = jsonable(dict(metadata))
        record["state_file"] = state_path.name
        record["state_npz_sha256"] = state_sha
        record["record_sha256"] = _record_digest(record, state_sha)
        record_path = temp / "record.json"
        with record_path.open("xb") as stream:
            stream.write(canonical_bytes(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temp, path)
        return verify_bundle(path)
    except Exception:
        # Preserve a pending directory as forensic evidence; never silently repair it.
        raise


def verify_bundle(path: Path, expected_identity: dict[str, Any] | None = None,
                  expected_checkpoint: int | None = None) -> dict[str, Any]:
    if not path.is_dir():
        raise IntegrityError(f"checkpoint bundle is not a directory: {path}")
    record = read_json(path / "record.json")
    state_path = path / str(record.get("state_file", "state.npz"))
    if not state_path.is_file() or sha256_file(state_path) != record.get("state_npz_sha256"):
        raise IntegrityError(f"checkpoint state SHA mismatch: {path}")
    if _record_digest(record, record["state_npz_sha256"]) != record.get("record_sha256"):
        raise IntegrityError(f"checkpoint record digest mismatch: {path}")
    if expected_identity is not None and record.get("identity") != jsonable(expected_identity):
        raise IntegrityError(f"checkpoint identity mismatch: {path}")
    if expected_checkpoint is not None and record.get("checkpoint") != expected_checkpoint:
        raise IntegrityError(f"checkpoint number mismatch: {path}")
    try:
        with np.load(state_path, allow_pickle=False) as payload:
            arrays = {key: payload[key].copy() for key in payload.files}
    except Exception as exc:
        raise IntegrityError(f"checkpoint array load failed: {path}: {exc}") from exc
    record["_arrays"] = arrays
    return record


def event_record(path: Path, expected_identity: dict[str, Any], event_id: str,
                 stream_order: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = read_json(path)
    if record.get("identity") != jsonable(expected_identity):
        raise IntegrityError(f"event identity mismatch: {path}")
    if record.get("event_id") != event_id or record.get("stream_order") != stream_order:
        raise IntegrityError(f"event order/ID mismatch: {path}")
    supplied_sha = record.get("record_sha256")
    if supplied_sha != sha256_bytes(canonical_bytes({k: v for k, v in record.items() if k != "record_sha256"})):
        raise IntegrityError(f"event record SHA mismatch: {path}")
    return record


def write_event_exclusive(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    value = jsonable(dict(record))
    value["record_sha256"] = sha256_bytes(canonical_bytes(value))
    write_json_exclusive(path, value)
    return value


def _history_digest(ids: list[str], labels: list[int]) -> str:
    return sha256_bytes(canonical_bytes({"image_ids": ids, "labels": labels}))


def run_checkpointed_trajectory(*, output_dir: Path, identity: dict[str, Any],
                                trajectory_id: str, side: str, protocol: str,
                                initial_theta: np.ndarray, initial_features: list[Any],
                                initial_labels: list[int], initial_image_ids: list[str],
                                feedback_events: list[dict[str, Any]],
                                reveal_state: Callable[[str], str],
                                fit_history: Callable[[list[Any], np.ndarray, np.ndarray, int], dict[str, Any]],
                                predict_one: Callable[[np.ndarray, Any], float],
                                predict_future: Callable[[np.ndarray, list[Any]], np.ndarray],
                                future_features: list[Any], future_rows: list[dict[str, Any]],
                                evaluate: Callable[[np.ndarray], tuple[dict, list[dict], list[dict]]],
                                update_enabled: bool,
                                resource_check: Callable[[str], None] | None = None,
                                stop_after_checkpoint: int | None = None) -> dict[str, Any]:
    """Execute/resume one trajectory; only committed checkpoints advance the state."""
    if side not in {"learner", "control"}:
        raise ValueError("side must be learner or control")
    if len(initial_features) != len(initial_labels) or len(initial_labels) != len(initial_image_ids):
        raise ValueError("initial training history mismatch")
    if len(set(initial_image_ids)) != len(initial_image_ids):
        raise ValueError("initial training IDs are not unique")
    if set(int(x) for x in initial_labels) != {0, 1}:
        raise ValueError("initial training labels must contain both classes")
    if len(feedback_events) != 12 or len(future_rows) != 24:
        raise ValueError("registered protocol must contain 12 feedback and 24 future rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    events_dir = output_dir / "events"
    events_dir.mkdir(exist_ok=True)
    completion_path = output_dir / "complete.json"
    failure_path = output_dir / "terminal_failure.json"
    ordered_events = sorted(feedback_events, key=lambda row: int(row["stream_order"]))
    if [int(row["stream_order"]) for row in ordered_events] != list(range(1, 13)):
        raise ValueError("feedback order must be exactly 1..12")

    if failure_path.exists():
        failure = read_json(failure_path)
        if failure.get("identity") != jsonable(identity) or failure.get("trajectory_id") != trajectory_id:
            raise IntegrityError(f"terminal failure identity mismatch: {failure_path}")
        if failure.get("record_sha256") != sha256_bytes(
                canonical_bytes({k: v for k, v in failure.items() if k != "record_sha256"})):
            raise IntegrityError(f"terminal failure record SHA mismatch: {failure_path}")
        return {"status": "SOLVER_FAILED", "trajectory_id": trajectory_id,
                "last_checkpoint": int(failure.get("last_valid_checkpoint", 0)),
                "terminal_failure": failure}

    checkpoint_paths = sorted(output_dir.glob("checkpoint_[0-9][0-9]"))
    checkpoint_numbers = [int(p.name[-2:]) for p in checkpoint_paths]
    if checkpoint_numbers != list(range(len(checkpoint_numbers))):
        raise IntegrityError(f"checkpoint sequence has a gap: {output_dir}")
    checkpoints = [verify_bundle(path, identity, i) for i, path in enumerate(checkpoint_paths)]
    if len(checkpoints) > 13:
        raise IntegrityError(f"too many checkpoints: {output_dir}")
    if completion_path.exists():
        complete = read_json(completion_path)
        if complete.get("identity") != jsonable(identity) or complete.get("trajectory_id") != trajectory_id:
            raise IntegrityError(f"completion identity mismatch: {completion_path}")
        if complete.get("record_sha256") != sha256_bytes(
                canonical_bytes({k: v for k, v in complete.items() if k != "record_sha256"})):
            raise IntegrityError(f"completion record SHA mismatch: {completion_path}")
        if len(checkpoints) != 13 or complete.get("checkpoint_record_sha256") != [
                cp["record_sha256"] for cp in checkpoints]:
            raise IntegrityError(f"completed trajectory does not match its checkpoints: {output_dir}")
        return {"status": "PASS", "trajectory_id": trajectory_id, "last_checkpoint": 12,
                "resumed_complete": True}

    initial = np.asarray(initial_theta, dtype=np.float64).copy()
    base_digest = hashlib.sha256(initial.tobytes()).hexdigest()
    history_features = list(initial_features)
    history_labels = [int(y) for y in initial_labels]
    history_ids = list(initial_image_ids)
    theta = initial.copy()
    binary_updates = 0
    last_checkpoint = 0

    if checkpoints:
        first = checkpoints[0]
        if first.get("initial_theta_sha256") != base_digest or first.get("side") != side:
            raise IntegrityError(f"initial trajectory state mismatch: {output_dir}")
        theta = np.asarray(first["_arrays"]["theta"], dtype=np.float64)
        if hashlib.sha256(theta.tobytes()).hexdigest() != first.get("theta_sha256"):
            raise IntegrityError(f"checkpoint theta digest mismatch: {output_dir}")
        last_checkpoint = len(checkpoints) - 1
        for cp_index in range(1, len(checkpoints)):
            cp = checkpoints[cp_index]
            if cp.get("side") != side or cp.get("initial_theta_sha256") != base_digest:
                raise IntegrityError(f"checkpoint lineage mismatch: {output_dir}/{cp_index}")
            event_info = cp.get("event")
            if not event_info or int(event_info.get("stream_order", -1)) != cp_index:
                raise IntegrityError(f"checkpoint event lineage missing: {output_dir}/{cp_index}")
            expected_event = ordered_events[cp_index - 1]
            if event_info.get("event_id") != expected_event["event_id"] or event_info.get("image_id") != expected_event["image_id"]:
                raise IntegrityError(f"checkpoint is attached to a different feedback item: {output_dir}/{cp_index}")
            event_path = events_dir / f"event_{cp_index:02d}.json"
            persisted_event = event_record(event_path, identity, expected_event["event_id"], cp_index)
            if persisted_event is None or persisted_event.get("record_sha256") != cp.get("event_record_sha256"):
                raise IntegrityError(f"checkpoint feedback event receipt mismatch: {event_path}")
            state = persisted_event["revealed_state"]
            target = {"present": 1, "absent_assessable": 0,
                      "unobservable_or_uncertain": None, "not_applicable": None}.get(state, "INVALID")
            if target == "INVALID":
                raise IntegrityError(f"invalid persisted feedback state: {event_path}")
            if update_enabled and target is not None:
                if expected_event["image_id"] in history_ids:
                    raise IntegrityError(f"replayed feedback would duplicate training item: {event_path}")
                history_features.append(expected_event["features"])
                history_labels.append(int(target))
                history_ids.append(expected_event["image_id"])
                if not cp.get("fit_record", {}).get("accepted", False):
                    raise IntegrityError(f"checkpoint contains an unaccepted fit: {output_dir}/{cp_index}")
                theta = np.asarray(cp["_arrays"]["theta"], dtype=np.float64)
                binary_updates += 1
            else:
                if update_enabled and target is None:
                    theta = np.asarray(cp["_arrays"]["theta"], dtype=np.float64)
                elif not update_enabled:
                    theta = np.asarray(cp["_arrays"]["theta"], dtype=np.float64)
            if hashlib.sha256(theta.tobytes()).hexdigest() != cp.get("theta_sha256"):
                raise IntegrityError(f"restored checkpoint theta mismatch: {output_dir}/{cp_index}")
            if cp.get("training_history_sha256") != _history_digest(history_ids, history_labels):
                raise IntegrityError(f"restored training history digest mismatch: {output_dir}/{cp_index}")
            last_checkpoint = cp_index
    else:
        # A finalized output directory with an orphan record is not silently repaired.
        unexpected = [p for p in output_dir.iterdir()
                      if p.name != "events" and not p.name.startswith(".checkpoint_")]
        if unexpected:
            raise IntegrityError(f"orphan scientific output requires review: {unexpected[0]}")
        scores = np.asarray(predict_future(theta, future_features), dtype=np.float64).reshape(-1)
        metrics, products, items = evaluate(scores)
        metadata = {
            "schema": "lp.d6.p2.e2b.r1.checkpoint.v1", "identity": identity,
            "trajectory_id": trajectory_id, "side": side, "protocol": protocol,
            "checkpoint": 0, "events_seen": 0, "binary_updates": 0,
            "training_history_rows": len(history_labels),
            "training_history_sha256": _history_digest(history_ids, history_labels),
            "initial_theta_sha256": base_digest,
            "theta_sha256": hashlib.sha256(theta.tobytes()).hexdigest(),
            "last_successful_event": 0, "fit_record": {"status": "INITIAL_STATE", "accepted": True},
            "metrics": metrics, "product_metrics": products, "item_predictions": items,
            "event": None, "event_record_sha256": "", "fit_seconds": 0.0,
        }
        write_bundle(output_dir / "checkpoint_00", metadata,
                     {"theta": theta, "future_scores": scores})
        checkpoints = [verify_bundle(output_dir / "checkpoint_00", identity, 0)]
        last_checkpoint = 0
        if resource_check is not None:
            resource_check("after_checkpoint_0")

    if last_checkpoint == 12:
        record_complete(output_dir, identity, trajectory_id, checkpoints)
        return {"status": "PASS", "trajectory_id": trajectory_id, "last_checkpoint": 12}

    for event_index in range(last_checkpoint + 1, 13):
        if resource_check is not None:
            resource_check(f"before_event_{event_index}")
        event = ordered_events[event_index - 1]
        event_id = str(event["event_id"])
        image_id = str(event["image_id"])
        if image_id in history_ids:
            raise IntegrityError(f"feedback item overlaps current training history: {image_id}")
        prepared_path = events_dir / f"prepared_{event_index:02d}.json"
        prepared = load_event_stage(prepared_path, identity, event_id, event_index)
        if prepared is None:
            score = float(predict_one(theta, event["features"]))
            if not np.isfinite(score):
                raise FloatingPointError(f"nonfinite query prediction at event {event_index}")
            prepared = write_event_stage(prepared_path, {
                "schema": "lp.d6.p2.e2b.r1.query.v1", "identity": identity,
                "event_id": event_id, "image_id": image_id, "stream_order": event_index,
                "score_before_reveal": score, "prediction_before_reveal": int(score >= 0.0),
                "state_sha256_before_reveal": hashlib.sha256(theta.tobytes()).hexdigest(),
                "feedback_cost_status": event.get("feedback_cost_status", "UNKNOWN"),
                "revealed_fields": event.get("revealed_fields", ""),
                "product_evaluator_only": event.get("product", ""),
                "sample_key_evaluator_only": event.get("sample_key", ""),
                "view_evaluator_only": event.get("view", ""),
            })
        if prepared.get("state_sha256_before_reveal") != hashlib.sha256(theta.tobytes()).hexdigest():
            raise IntegrityError(f"pending query was prepared from a different model state: {prepared_path}")
        revealed_path = events_dir / f"event_{event_index:02d}.json"
        persisted = event_record(revealed_path, identity, event_id, event_index)
        if persisted is None:
            state = reveal_state(event_id)
            if state not in {"present", "absent_assessable", "unobservable_or_uncertain", "not_applicable"}:
                raise ValueError(f"invalid revealed state at {event_id}")
            persisted = write_event_exclusive(revealed_path, {
                "schema": "lp.d6.p2.e2b.r1.revealed-event.v1", "identity": identity,
                "event_id": event_id, "image_id": image_id, "stream_order": event_index,
                "query_record_sha256": prepared["record_sha256"],
                "score_before_reveal": prepared["score_before_reveal"],
                "prediction_before_reveal": prepared["prediction_before_reveal"],
                "state_sha256_before_reveal": prepared["state_sha256_before_reveal"],
                "revealed_state": state,
                "binary_target_after_reveal": {"present": 1, "absent_assessable": 0,
                    "unobservable_or_uncertain": None, "not_applicable": None}[state],
            })
        elif persisted.get("query_record_sha256") != prepared.get("record_sha256"):
            raise IntegrityError(f"pending event query and reveal do not match: {revealed_path}")

        state = persisted["revealed_state"]
        target = persisted["binary_target_after_reveal"]
        fit_record: dict[str, Any] = {"status": "NO_BINARY_UPDATE", "accepted": True}
        fit_seconds = 0.0
        if update_enabled and target is not None:
            if image_id in history_ids:
                raise IntegrityError(f"feedback item would enter training twice: {image_id}")
            history_features.append(event["features"])
            history_labels.append(int(target))
            history_ids.append(image_id)
            if resource_check is not None:
                resource_check(f"before_fit_event_{event_index}")
            fit_result = fit_history(history_features, np.asarray(history_labels, dtype=np.int8),
                                     theta.copy(), event_index)
            fit_record = {k: jsonable(v) for k, v in fit_result.items() if k != "theta"}
            fit_seconds = float(fit_result.get("elapsed_seconds", 0.0))
            if not fit_result.get("accepted", False):
                history_features.pop(); history_labels.pop(); history_ids.pop()
                failure = {
                    "schema": "lp.d6.p2.e2b.r1.terminal-failure.v1", "identity": identity,
                    "trajectory_id": trajectory_id, "side": side,
                    "failure_event": event_index, "event_id": event_id,
                    "revealed_event_sha256": persisted["record_sha256"],
                    "last_valid_checkpoint": last_checkpoint,
                    "fit_record": fit_record,
                    "last_valid_theta_sha256": hashlib.sha256(theta.tobytes()).hexdigest(),
                    "reason": "registered optimizer acceptance gate rejected candidate",
                }
                failure["record_sha256"] = sha256_bytes(canonical_bytes(failure))
                write_json_exclusive(failure_path, failure)
                return {"status": "SOLVER_FAILED", "trajectory_id": trajectory_id,
                        "last_checkpoint": last_checkpoint, "terminal_failure": failure}
            theta = np.asarray(fit_result["theta"], dtype=np.float64).copy()
            binary_updates += 1
        elif not update_enabled:
            fit_record = {"status": "FROZEN_CONTROL_NO_UPDATE", "accepted": True}
        elif target is None:
            fit_record = {"status": "NO_BINARY_UPDATE", "accepted": True}

        scores = np.asarray(predict_future(theta, future_features), dtype=np.float64).reshape(-1)
        if len(scores) != len(future_rows) or not np.isfinite(scores).all():
            raise FloatingPointError(f"invalid future score vector at checkpoint {event_index}")
        metrics, products, items = evaluate(scores)
        cp_event = {
            "event_id": event_id, "image_id": image_id, "stream_order": event_index,
            "feedback_state_after_prediction": state,
            "binary_target_after_prediction": target,
            "query_score_before_reveal": persisted["score_before_reveal"],
            "query_state_sha256_before_reveal": persisted["state_sha256_before_reveal"],
            "revealed_event_sha256": persisted["record_sha256"],
            "updated": bool(update_enabled and target is not None),
        }
        metadata = {
            "schema": "lp.d6.p2.e2b.r1.checkpoint.v1", "identity": identity,
            "trajectory_id": trajectory_id, "side": side, "protocol": protocol,
            "checkpoint": event_index, "events_seen": event_index,
            "binary_updates": binary_updates, "training_history_rows": len(history_labels),
            "training_history_sha256": _history_digest(history_ids, history_labels),
            "initial_theta_sha256": base_digest,
            "theta_sha256": hashlib.sha256(theta.tobytes()).hexdigest(),
            "last_successful_event": event_index if fit_record.get("accepted") else last_checkpoint,
            "fit_record": fit_record, "fit_seconds": fit_seconds,
            "metrics": metrics, "product_metrics": products, "item_predictions": items,
            "event": cp_event, "event_record_sha256": persisted["record_sha256"],
        }
        bundle = write_bundle(output_dir / f"checkpoint_{event_index:02d}", metadata,
                              {"theta": theta, "future_scores": scores})
        checkpoints.append(bundle)
        last_checkpoint = event_index
        if resource_check is not None:
            resource_check(f"after_checkpoint_{event_index}")
        if stop_after_checkpoint is not None and event_index >= stop_after_checkpoint:
            return {"status": "INTERRUPTED_FOR_TEST", "trajectory_id": trajectory_id,
                    "last_checkpoint": last_checkpoint}

    record_complete(output_dir, identity, trajectory_id, checkpoints)
    return {"status": "PASS", "trajectory_id": trajectory_id, "last_checkpoint": 12}


def load_event_stage(path: Path, identity: dict[str, Any], event_id: str,
                     stream_order: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = read_json(path)
    if (record.get("identity") != jsonable(identity) or record.get("event_id") != event_id
            or record.get("stream_order") != stream_order):
        raise IntegrityError(f"query event identity mismatch: {path}")
    if record.get("record_sha256") != sha256_bytes(
            canonical_bytes({k: v for k, v in record.items() if k != "record_sha256"})):
        raise IntegrityError(f"query event SHA mismatch: {path}")
    return record


def write_event_stage(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    record = jsonable(dict(value))
    record["record_sha256"] = sha256_bytes(canonical_bytes(record))
    write_json_exclusive(path, record)
    return record


def record_complete(output_dir: Path, identity: dict[str, Any], trajectory_id: str,
                    checkpoints: list[dict[str, Any]]) -> None:
    if len(checkpoints) != 13 or [cp.get("checkpoint") for cp in checkpoints] != list(range(13)):
        raise IntegrityError(f"cannot mark incomplete trajectory as complete: {trajectory_id}")
    path = output_dir / "complete.json"
    record = {
        "schema": "lp.d6.p2.e2b.r1.complete.v1", "identity": identity,
        "trajectory_id": trajectory_id, "checkpoint_count": len(checkpoints),
        "checkpoint_record_sha256": [cp["record_sha256"] for cp in checkpoints],
        "last_checkpoint": 12,
    }
    record["record_sha256"] = sha256_bytes(canonical_bytes(record))
    write_json_exclusive(path, record)
