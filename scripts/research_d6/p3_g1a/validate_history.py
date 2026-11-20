"""Verify the archived X H12 head, five old anchors and score reconstruction."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1a_common as c

E2A1 = c.ROOT / "outputs/research_d6/p2/e2a1"
RUN = c.ROOT / "outputs/research_d6/p2/e2b/r1_run_20261116_01"
REPLAY = c.ROOT / "outputs/research_d6/p2/correction_memory_replay/run_20261119_01"
PAIR = "dinov2__G-Mean__X"
EXPECTED_FEATURE_MANIFEST = "3588bab021a9cd8043d663c1d6ce46f366f89f89802fb1ac928dfae3360efddf"
EXPECTED_INPUT_MANIFEST = "820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63"
EXPECTED_H12_STATE = "8c81a8c61c9dae934ce071827d27b51e42a7dd25def233454341fe81c2cbaa76"
EXPECTED_H12_THETA = "bb8232f44907a9f063e11d3dd8ff3ed1ea4c207876326396b057591abab03dc4"
EXPECTED_H12_RECORD = "226c31697373541278ffb826cb49ef011964d0295af2b66af78bebc3085da3e7"


def checkpoint(index: int):
    path = RUN / "trajectories" / PAIR / "learner" / f"checkpoint_{index:02d}"
    record_path = path / "record.json"
    state_path = path / "state.npz"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    state_sha = c.sha_file(state_path)
    if state_sha != record.get("state_npz_sha256"):
        raise RuntimeError("HISTORICAL_CHECKPOINT_STATE_SHA_MISMATCH")
    expected_record = __import__("hashlib").sha256(c.canonical_json({
        "metadata": {k: v for k, v in record.items() if k != "record_sha256"},
        "state_npz_sha256": record.get("state_npz_sha256"),
    })).hexdigest()
    if expected_record != record.get("record_sha256"):
        raise RuntimeError("HISTORICAL_CHECKPOINT_RECORD_SHA_MISMATCH")
    with np.load(state_path, allow_pickle=False) as data:
        theta = np.ascontiguousarray(data["theta"], dtype=np.float64)
    theta_sha = __import__("hashlib").sha256(theta.tobytes()).hexdigest()
    if theta_sha != record.get("theta_sha256"):
        raise RuntimeError("HISTORICAL_THETA_SHA_MISMATCH")
    return record, theta, state_sha


def feature_for(image_id: str, manifest: dict[str, dict[str, str]]):
    row = manifest.get(image_id)
    if row is None:
        raise RuntimeError("ANCHOR_FEATURE_ID_MISSING")
    # E2A1's frozen feature manifest intentionally stores no path column; the
    # registered item layout is dinov2/items/<opaque image_id>.
    base = E2A1 / "dinov2" / "items" / image_id
    record_path = base / "record.json"
    if c.sha_file(record_path) != row["record_sha256"]:
        raise RuntimeError("ARCHIVED_FEATURE_RECORD_SHA_MISMATCH")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("identity", {}).get("image_id") != image_id or record.get("identity", {}).get("source_sha256") != row["source_sha256"]:
        raise RuntimeError("ARCHIVED_FEATURE_IDENTITY_MISMATCH")
    arrays = {}
    for name in ("patch_unit", "valid_patch_mask"):
        spec = record["arrays"][name]
        array_path = base / spec["file"]
        if c.sha_file(array_path) != spec["file_sha256"]:
            raise RuntimeError("ARCHIVED_FEATURE_ARRAY_FILE_SHA_MISMATCH")
        array = np.load(array_path, allow_pickle=False)
        if c.sha_array(array) != spec["array_sha256"]:
            raise RuntimeError("ARCHIVED_FEATURE_ARRAY_SHA_MISMATCH")
        arrays[name] = array
    patches = np.asarray(arrays["patch_unit"], dtype=np.float32)
    mask = np.asarray(arrays["valid_patch_mask"], dtype=np.bool_)
    if patches.shape != (1024, 1024) or mask.shape != (1024,) or not mask.any() or not np.isfinite(patches).all():
        raise RuntimeError("ARCHIVED_FEATURE_SHAPE_OR_VALUE_MISMATCH")
    return {"patches": patches, "mask": mask, "source_sha256": row["source_sha256"]}


def global_gmean(feature: dict) -> np.ndarray:
    # Match the archived replay helper: cast each FP32 token to FP64 before
    # accumulating the valid-patch mean.
    patches = np.asarray(feature["patches"], dtype=np.float64)
    mean = patches[np.asarray(feature["mask"], dtype=np.bool_)].mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(mean).all() or norm <= 1e-12:
        raise RuntimeError("INVALID_GMEAN")
    return mean / norm


def check_event_digest(record: dict) -> bool:
    return c.self_sha(record) == record.get("record_sha256")


def run(output: Path) -> dict:
    if output.exists() and not output.is_dir():
        raise RuntimeError("OUTPUT_PATH_INVALID")
    fmanifest_path = E2A1 / "dinov2" / "feature_manifest.csv"
    im_manifest = E2A1 / "dinov2" / "feature_manifest.csv"
    if c.sha_file(fmanifest_path) != EXPECTED_FEATURE_MANIFEST:
        raise RuntimeError("E2A1_FEATURE_MANIFEST_SHA_MISMATCH")
    input_manifest = E2A1 / "registered_input_manifest.csv"
    if not input_manifest.is_file():
        input_manifest = E2A1 / "input_manifest.csv"
    if not input_manifest.is_file() or c.sha_file(input_manifest) != EXPECTED_INPUT_MANIFEST:
        raise RuntimeError("E2A1_REGISTERED_INPUT_MANIFEST_SHA_MISMATCH")
    feature_rows = c.read_csv(fmanifest_path)
    if len(feature_rows) != 102 or len({r["image_id"] for r in feature_rows}) != 102:
        raise RuntimeError("E2A1_FEATURE_MANIFEST_COUNT")
    feature_map = {r["image_id"]: r for r in feature_rows}

    cp12, theta12, h12_state_sha = checkpoint(12)
    if (h12_state_sha != EXPECTED_H12_STATE or cp12.get("theta_sha256") != EXPECTED_H12_THETA
            or cp12.get("record_sha256") != EXPECTED_H12_RECORD or cp12.get("trajectory_id") != PAIR + "__learner"
            or cp12.get("checkpoint") != 12 or cp12.get("events_seen") != 12
            or theta12.shape != (1025,) or not np.isfinite(theta12).all()):
        raise RuntimeError("H12_IDENTITY_MISMATCH")
    if cp12.get("identity", {}).get("feature_manifest_sha256") != EXPECTED_FEATURE_MANIFEST:
        raise RuntimeError("H12_FEATURE_MANIFEST_BINDING_MISMATCH")

    # Rebuild the old scores against the archived pre-evaluator record. This
    # table contains IDs and logits only; do not consume item_predictions
    # embedded in checkpoint record.json because that object also carries
    # evaluator fields.
    replay_scores = c.read_csv(REPLAY / "unscored_predictions.csv")
    predictions = [r for r in replay_scores if r["protocol"] == "X" and r["checkpoint"] == "12"]
    if len(predictions) != 24 or len({r["image_id"] for r in predictions}) != 24:
        raise RuntimeError("H12_FUTURE_SCORE_CARDINALITY")
    diffs = []
    for pred in predictions:
        feature = feature_for(pred["image_id"], feature_map)
        score = float(global_gmean(feature) @ theta12[:-1] + theta12[-1])
        archived = float(pred["h_score"])
        diff = abs(score - archived)
        if not np.isfinite(score) or diff > 1e-8:
            raise RuntimeError("H12_SCORE_RECONSTRUCTION_FAILED")
        diffs.append(diff)

    with (REPLAY / "event_anchors.csv").open(encoding="utf-8-sig", newline="") as f:
        replay_rows = list(csv.DictReader(f))
    anchors = [r for r in replay_rows if r["protocol"] == "X" and r["write_anchor"] == "1"]
    if len(anchors) != 5 or len({r["image_id"] for r in anchors}) != 5:
        raise RuntimeError("HISTORICAL_ANCHOR_COUNT_OR_IDENTITY")
    if Counter(r["label"] for r in anchors) != Counter({"1": 2, "0": 3}):
        raise RuntimeError("HISTORICAL_ANCHOR_LABEL_COUNTS")

    anchor_features = []
    anchor_checks = []
    for row in anchors:
        event_index = int(row["event_index"])
        if not 1 <= event_index <= 12:
            raise RuntimeError("ANCHOR_EVENT_ORDER_INVALID")
        prior, prior_theta, prior_state_sha = checkpoint(event_index - 1)
        prior_theta_sha = __import__("hashlib").sha256(prior_theta.tobytes()).hexdigest()
        event_path = RUN / "trajectories" / PAIR / "learner" / "events" / f"event_{event_index:02d}.json"
        prepared_path = RUN / "trajectories" / PAIR / "learner" / "events" / f"prepared_{event_index:02d}.json"
        event = json.loads(event_path.read_text(encoding="utf-8"))
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        if not check_event_digest(event) or not check_event_digest(prepared):
            raise RuntimeError("HISTORICAL_EVENT_DIGEST_MISMATCH")
        feature = feature_for(row["image_id"], feature_map)
        score = float(global_gmean(feature) @ prior_theta[:-1] + prior_theta[-1])
        y = int(row["label"])
        state = "present" if y == 1 else "absent_assessable"
        predicted = int(score >= 0.0)
        if (event_index != int(row["event_index"]) or event["event_id"] != row["event_id"]
                or event["image_id"] != row["image_id"] or event["stream_order"] != event_index
                or event["state_sha256_before_reveal"] != prior_theta_sha
                or prepared["state_sha256_before_reveal"] != prior_theta_sha
                or row["state_sha256"] != prior_theta_sha
                or prepared["image_id"] != row["image_id"] or prepared["event_id"] != row["event_id"]
                or abs(score - float(row["query_score_before_reveal"])) > 1e-8
                or abs(score - float(event["score_before_reveal"])) > 1e-8
                or abs(score - float(prepared["score_before_reveal"])) > 1e-8
                or int(event["prediction_before_reveal"]) != predicted
                or int(prepared["prediction_before_reveal"]) != predicted
                or event["revealed_state"] != state or int(event["binary_target_after_reveal"]) != y
                or predicted == y or int(row["write_anchor"]) != 1):
            raise RuntimeError("HISTORICAL_ANCHOR_PRE_REVEAL_VALIDATION_FAILED")
        anchor_features.append(feature)
        anchor_checks.append({"event_index": event_index, "opaque_image_id": row["image_id"],
                              "source_sha256": feature["source_sha256"], "pre_reveal_theta_sha256": prior_theta_sha,
                              "pre_reveal_score_verified": True,
                              "stored_as_correction": True})

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "h12_theta.npy", theta12, allow_pickle=False)
    np.savez_compressed(output / "anchor_features.npz",
                        image_ids=np.asarray([r["opaque_image_id"] for r in anchor_checks]),
                        event_indices=np.asarray([r["event_index"] for r in anchor_checks], dtype=np.int64),
                        patches=np.stack([x["patches"] for x in anchor_features]),
                        masks=np.stack([x["mask"] for x in anchor_features]))
    payload = {"schema": "lp.d6.p3.g1a.history-validation.v1", "status": "PASS",
               "h12_trajectory_id": PAIR + "__learner", "checkpoint": 12,
               "h12_state_npz_sha256": h12_state_sha, "h12_record_sha256": cp12["record_sha256"],
               "h12_theta_sha256": cp12["theta_sha256"], "feature_manifest_sha256": EXPECTED_FEATURE_MANIFEST,
               "archive_future_scores_rebuilt": 24, "reference_source": "unscored_predictions.csv (no evaluator labels)",
               "max_abs_score_delta": max(diffs),
               "historical_anchor_count": len(anchors), "anchor_positive": 2, "anchor_negative": 3,
               "anchor_pre_reveal_checks": anchor_checks,
               "new_attribute_labels_read": False, "protected_storage_accessed": False}
    (output / "history_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(Path(args.output))
    print(json.dumps({k: result[k] for k in ("status", "checkpoint", "archive_future_scores_rebuilt",
                                             "max_abs_score_delta", "historical_anchor_count",
                                             "anchor_positive", "anchor_negative")}, indent=2))
