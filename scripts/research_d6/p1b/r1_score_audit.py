"""Post-hoc, score-level diagnostics over the archived LP-D6-P1-B outputs.

This module does not initialize or update a learner. It reads only the existing
ignored P1-B result directory and writes new audit artifacts elsewhere.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "outputs/research_d6/p1b/run_20261113_01"
AUDIT_DIR = ROOT / "outputs/research_d6/p1b_r1"
PUBLIC_AGGREGATE = ROOT / "docs/research_d6/p1b/D6_P1B_PRODUCT_METRICS_AGGREGATE.csv"
CHECKPOINTS = (0, 3, 6, 9, 12)
SGD_METHOD = "B1-SGD-Replay"

# Immutable hashes captured before this audit touched any P1-B output.
ORIGINAL_RUN_FILE_SHA256 = {
    "attempted_configurations.csv": "61a4551239f89dc6a322cb3dbf8075dc25ff17e832de0c6f0eadd1bc62f7959e",
    "checkpoint_metrics.csv": "68bfa1df2f627a1a0ec797f78ea7a86a3b778a527d6f3fa6d216749981ec1e1b",
    "feedback_event_log.csv": "ef35b495d7012514188adecb97eacc412a0f51ee2f41e30661ee8a87f2ca086f",
    "future_and_query_predictions.csv": "c3882100850e6d14024b7dcb8fd6a7fbcbe189a043c6979e419c2deae63f929a",
    "group_metrics.csv": "6b429b8982a81641d399de410caa73e0b8c58269ecfd73632c39478da88b5712",
    "input_integrity_details.json": "dba13512b324cd0329a653e02df0009e79e1524f6ec21dde1b8e598d701c6666",
    "joined_human_annotations.csv": "c814ee3fb8a91ae845c6d7d5d43bdf102b8f30712ff41a0c0fa57d3a0fe96f4b",
    "model_state_index.csv": "cef7597d199d359c3ffc9142d2e1a27876bccf1f13f026347d7feccda0b5fcc2",
    "model_states.npz": "cdc16eb1caae1e2dbdc5b5c81048dc3ae41f08f55b4ae64f01945d27d7c46dfa",
    "paired_gains.csv": "0117bdce3b33ce72156355199d2c2fe21467ec87d2a86cad0f7859c238fac9c4",
    "preflight_integrity_report.json": "c6c5fa5e98b1abb01e3761c53057632ea779980e122f80a1b36d97fae77a1545",
    "product_metrics.csv": "a5dba54976298fb0540039326ddd2fe65b63b5f1a9ee2b0e347cf12b3e3410c6",
    "refit_convergence_audit.csv": "a8a32430ff406ea80a91397a5c6ae208b91d83206ba94736963da1740df84475",
    "run_provenance.json": "b58638059dbe81413e8b6a10a4ddbadfb3fd7aa15343533af1c7a0920337afdc",
    "source_attribute_diagnostic.csv": "341810071f924683b6edb57ed64756cb0b00b0db31b5b86ec4db7ee3b6486114",
}

PUBLIC_COLUMNS = (
    "protocol", "method", "configuration_id", "product", "checkpoint",
    "n_views", "n_binary", "present", "absent", "uncertain", "not_applicable",
    "ba", "ba_status", "tpr", "tnr", "ap", "ap_status",
)
PRODUCT_FIELD_MAP = {
    "configuration_id": "config_id",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_original_run_artifacts() -> dict[str, str]:
    if not RUN_DIR.is_dir():
        raise RuntimeError(f"original P1-B run directory is missing: {RUN_DIR}")
    found = {p.name for p in RUN_DIR.iterdir() if p.is_file()}
    expected_names = set(ORIGINAL_RUN_FILE_SHA256)
    if found != expected_names:
        raise RuntimeError(
            f"original P1-B file set changed; missing={sorted(expected_names-found)}, "
            f"extra={sorted(found-expected_names)}"
        )
    actual = {name: sha256_file(RUN_DIR / name) for name in sorted(expected_names)}
    mismatched = [name for name, digest in actual.items()
                  if digest != ORIGINAL_RUN_FILE_SHA256[name]]
    if mismatched:
        raise RuntimeError(f"original P1-B result files changed: {mismatched}")
    return actual


def _public_aggregate_rows(metrics: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for source in metrics:
        if source["checkpoint"] not in {"0", "12"}:
            continue
        if not source["method"].startswith(("B0-", "B1-", "B2-")):
            continue
        row = {}
        for col in PUBLIC_COLUMNS:
            source_col = PRODUCT_FIELD_MAP.get(col, col)
            row[col] = source.get(source_col, "")
        rows.append(row)
    rows.sort(key=lambda r: (r["protocol"], r["method"], r["configuration_id"],
                             int(r["checkpoint"]), r["product"]))
    return rows


def build_public_aggregate(metrics_path: Path | None = None) -> list[dict[str, str]]:
    return _public_aggregate_rows(read_csv(metrics_path or RUN_DIR / "product_metrics.csv"))


def verify_published_aggregate() -> int:
    expected = build_public_aggregate()
    published = read_csv(PUBLIC_AGGREGATE)
    if published != expected:
        raise RuntimeError("published product aggregate differs from original product_metrics.csv")
    if len(expected) != 432:
        raise RuntimeError(f"expected 432 public aggregate rows, got {len(expected)}")
    if any(set(row) != set(PUBLIC_COLUMNS) for row in published):
        raise RuntimeError("published aggregate schema includes missing or extra fields")
    forbidden = {"image_id", "sample_key", "attribute_state", "binary_target",
                 "score", "prediction", "path", "source_path"}
    if forbidden & set(PUBLIC_COLUMNS):
        raise RuntimeError("private per-image fields are forbidden in the public aggregate")
    return len(published)


def write_public_aggregate() -> int:
    rows = build_public_aggregate()
    PUBLIC_AGGREGATE.parent.mkdir(parents=True, exist_ok=True)
    with PUBLIC_AGGREGATE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PUBLIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return verify_published_aggregate()


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _rank_pair_counts(base: np.ndarray, current: np.ndarray) -> tuple[int, int, int, int]:
    retained = reversed_pairs = tied_now = compared = 0
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            old_sign = int(np.sign(base[i] - base[j]))
            new_sign = int(np.sign(current[i] - current[j]))
            if old_sign == 0:
                continue
            compared += 1
            if new_sign == old_sign:
                retained += 1
            elif new_sign == 0:
                tied_now += 1
            else:
                reversed_pairs += 1
    return compared, retained, reversed_pairs, tied_now


def _score_rows() -> list[dict[str, str]]:
    rows = read_csv(RUN_DIR / "future_and_query_predictions.csv")
    selected = [r for r in rows
                if r["method"] == SGD_METHOD and r["role"] == "future_eval"
                and r["binary_target"] in {"0", "1"}]
    if not selected:
        raise RuntimeError("no binary future rows for B1-SGD-Replay were found")
    return selected


def build_score_diagnostics() -> dict[str, list[dict[str, Any]]]:
    predictions = _score_rows()
    product_metrics = [r for r in read_csv(RUN_DIR / "product_metrics.csv")
                       if r["method"] == SGD_METHOD and r["checkpoint"] in {str(c) for c in CHECKPOINTS}]
    by_key: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in predictions:
        key = (row["protocol"], row["config_id"], int(row["checkpoint"]))
        by_key.setdefault(key, []).append(row)
    expected_groups = {p: sorted({c for pp, c, _ in by_key if pp == p}) for p in ("W", "X")}
    if any(len(expected_groups[p]) != 6 for p in expected_groups):
        raise RuntimeError(f"expected six SGD configurations per protocol: {expected_groups}")

    summary: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []
    config_consistency: list[dict[str, Any]] = []
    state_diagnostics: list[dict[str, Any]] = []
    metric_by_key = {(r["protocol"], r["config_id"], int(r["checkpoint"]), r["product"]): r
                     for r in product_metrics}

    for protocol in ("W", "X"):
        config_ids = expected_groups[protocol]
        baseline_scores: dict[str, np.ndarray] = {}
        score_vectors: dict[tuple[str, int], np.ndarray] = {}
        prediction_maps: dict[tuple[str, int], dict[str, int]] = {}
        for config_id in config_ids:
            previous: dict[str, dict[str, str]] | None = None
            for checkpoint in CHECKPOINTS:
                group = sorted(by_key.get((protocol, config_id, checkpoint), []),
                               key=lambda r: r["image_id"])
                if len(group) != 21 or len({r["image_id"] for r in group}) != 21:
                    raise RuntimeError(f"expected 21 unique binary future views: {protocol}/{config_id}/{checkpoint}")
                y = np.asarray([int(r["binary_target"]) for r in group], dtype=np.int8)
                scores = np.asarray([float(r["score"]) for r in group], dtype=np.float64)
                thresholds = {float(r["threshold"]) for r in group}
                if thresholds != {0.0}:
                    raise RuntimeError(f"registered SGD threshold changed: {thresholds}")
                pred = np.asarray([int(r["prediction"]) for r in group], dtype=np.int8)
                if not np.array_equal(pred, (scores >= 0.0).astype(np.int8)):
                    raise RuntimeError("recorded predictions disagree with the fixed zero-logit threshold")
                correct = pred == y
                pred_pos = int(pred.sum())
                pred_neg = int(len(pred) - pred_pos)
                positives = int(y.sum())
                negatives = int(len(y) - positives)
                tp = int(np.sum((pred == 1) & (y == 1)))
                tn = int(np.sum((pred == 0) & (y == 0)))
                below = int(np.sum(scores < 0.0))
                equal = int(np.sum(scores == 0.0))
                above = int(np.sum(scores > 0.0))
                old_by_id = {r["image_id"]: r for r in previous} if previous else {}
                ctow = sum(_bool(old_by_id[r["image_id"]]["correct"]) and not _bool(r["correct"])
                           for r in group if r["image_id"] in old_by_id)
                wtoc = sum(not _bool(old_by_id[r["image_id"]]["correct"]) and _bool(r["correct"])
                           for r in group if r["image_id"] in old_by_id)
                summary.append({
                    "protocol": protocol, "config_id": config_id, "checkpoint": checkpoint,
                    "n_binary": len(group), "present": positives, "absent": negatives,
                    "predicted_positive": pred_pos, "predicted_negative": pred_neg,
                    "predicted_positive_fraction": pred_pos / len(group),
                    "tpr": tp / positives if positives else "", "tnr": tn / negatives if negatives else "",
                    "true_positive": tp, "true_negative": tn,
                    "score_below_zero": below, "score_equal_zero": equal, "score_above_zero": above,
                    "score_min": float(np.min(scores)), "score_median": float(np.median(scores)),
                    "score_max": float(np.max(scores)), "roc_auc_posthoc": float(roc_auc_score(y, scores)),
                    "correct_to_wrong_from_previous_selected_checkpoint": ctow if previous else "",
                    "wrong_to_correct_from_previous_selected_checkpoint": wtoc if previous else "",
                })
                score_vectors[(config_id, checkpoint)] = scores
                prediction_maps[(config_id, checkpoint)] = {r["image_id"]: int(r["prediction"]) for r in group}
                if checkpoint == 0:
                    baseline_scores[config_id] = scores
                base = baseline_scores[config_id]
                compared, retained, reversed_pairs, tied_now = _rank_pair_counts(base, scores)
                ranks.append({
                    "protocol": protocol, "config_id": config_id, "checkpoint": checkpoint,
                    "strict_pairs_at_checkpoint0": compared, "order_retained": retained,
                    "order_reversed": reversed_pairs, "current_ties_from_initial_strict_pairs": tied_now,
                    "order_retained_fraction": retained / compared if compared else "",
                    "spearman_vs_checkpoint0": float(spearmanr(base, scores).statistic),
                })
                for product in sorted({r["product"] for r in group}):
                    metric = metric_by_key[(protocol, config_id, checkpoint, product)]
                    products.append({
                        "protocol": protocol, "config_id": config_id, "product": product,
                        "checkpoint": checkpoint, "n_views": metric["n_views"],
                        "n_binary": metric["n_binary"], "present": metric["present"],
                        "absent": metric["absent"], "uncertain": metric["uncertain"],
                        "not_applicable": metric["not_applicable"], "ba": metric["ba"],
                        "ba_status": metric["ba_status"], "tpr": metric["tpr"],
                        "tnr": metric["tnr"], "ap": metric["ap"], "ap_status": metric["ap_status"],
                    })
                previous = group

        for checkpoint in CHECKPOINTS:
            ids_per_config = [prediction_maps[(cid, checkpoint)] for cid in config_ids]
            ids = sorted(ids_per_config[0])
            pair_agreements = []
            for i in range(len(ids_per_config)):
                for j in range(i + 1, len(ids_per_config)):
                    pair_agreements.append(sum(ids_per_config[i][x] == ids_per_config[j][x] for x in ids) / len(ids))
            per_item_score_ranges = [max(score_vectors[(cid, checkpoint)][k] for cid in config_ids)
                                     - min(score_vectors[(cid, checkpoint)][k] for cid in config_ids)
                                     for k in range(len(ids))]
            prediction_patterns = {
                tuple(ids_per_config[c][image_id] for c in range(len(config_ids))) for image_id in ids
            }
            config_consistency.append({
                "protocol": protocol, "checkpoint": checkpoint, "n_configs": len(config_ids),
                "pairwise_prediction_agreement_mean": float(np.mean(pair_agreements)),
                "fraction_views_identical_across_all_configs": sum(
                    len({m[image_id] for m in ids_per_config}) == 1 for image_id in ids
                ) / len(ids),
                "unique_per_view_prediction_patterns": len(prediction_patterns),
                "score_range_across_configs_mean_per_view": float(np.mean(per_item_score_ranges)),
                "score_range_across_configs_max_per_view": float(np.max(per_item_score_ranges)),
            })

    # All logged per-image predictions for the selected checkpoints, kept local/ignored.
    by_image = {(r["protocol"], r["config_id"], int(r["checkpoint"]), r["image_id"]): r
                for r in predictions if int(r["checkpoint"]) in CHECKPOINTS}
    per_image = []
    for key, row in sorted(by_image.items()):
        per_image.append({
            "protocol": row["protocol"], "config_id": row["config_id"],
            "checkpoint": int(row["checkpoint"]), "image_id": row["image_id"],
            "sample_key": row["sample_key"], "product": row["product"],
            "binary_target": int(row["binary_target"]), "score": float(row["score"]),
            "threshold": float(row["threshold"]), "prediction": int(row["prediction"]),
            "correct": _bool(row["correct"]),
        })

    state_index = read_csv(RUN_DIR / "model_state_index.csv")
    with np.load(RUN_DIR / "model_states.npz", allow_pickle=False) as archive:
        for index_row in state_index:
            if index_row["method"] != SGD_METHOD:
                continue
            protocol, config_id = index_row["protocol"], index_row["pair_id"]
            prefix = f"{protocol}__{config_id}__"
            w0 = archive[prefix + "initial__coef"].astype(np.float64).reshape(-1)
            w1 = archive[prefix + "final__coef"].astype(np.float64).reshape(-1)
            b0 = float(archive[prefix + "initial__intercept"].reshape(-1)[0])
            b1 = float(archive[prefix + "final__intercept"].reshape(-1)[0])
            norm0, norm1 = float(np.linalg.norm(w0)), float(np.linalg.norm(w1))
            denom = norm0 * norm1
            cosine = float(np.dot(w0, w1) / denom) if denom else ""
            state_diagnostics.append({
                "protocol": protocol, "config_id": config_id,
                "initial_state_sha256": index_row["initial_state_sha256"],
                "final_state_sha256": index_row["final_state_sha256"],
                "state_changed": index_row["initial_state_sha256"] != index_row["final_state_sha256"],
                "initial_weight_l2": norm0, "final_weight_l2": norm1,
                "weight_delta_l2": float(np.linalg.norm(w1 - w0)),
                "weight_delta_relative_to_initial": float(np.linalg.norm(w1 - w0) / max(norm0, 1e-30)),
                "weight_cosine_initial_final": cosine,
                "initial_intercept": b0, "final_intercept": b1,
                "intercept_delta": b1 - b0,
            })

    return {
        "summary": summary, "per_image": per_image, "products": products,
        "rank": ranks, "config_consistency": config_consistency,
        "state": state_diagnostics,
    }


def _write_local_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty diagnostic file: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_local_diagnostics(diagnostics: dict[str, list[dict[str, Any]]]) -> None:
    paths = {
        "summary": "sgd_threshold_summary.csv",
        "per_image": "sgd_threshold_per_image.csv",
        "products": "sgd_product_metrics_selected_checkpoints.csv",
        "rank": "sgd_rank_diagnostics.csv",
        "config_consistency": "sgd_configuration_consistency.csv",
        "state": "sgd_state_diagnostics.csv",
    }
    for key, filename in paths.items():
        _write_local_csv(AUDIT_DIR / filename, diagnostics[key])


def main() -> int:
    before = verify_original_run_artifacts()
    aggregate_rows = write_public_aggregate()
    diagnostics = build_score_diagnostics()
    write_local_diagnostics(diagnostics)
    after = verify_original_run_artifacts()
    if before != after:
        raise RuntimeError("original P1-B artifacts changed during the R1 audit")
    summary = {
        "schema": "lp.d6.p1b.r1.audit_provenance.v1",
        "status": "PASS",
        "original_run_artifact_sha256_manifest": ORIGINAL_RUN_FILE_SHA256,
        "public_aggregate_sha256": sha256_file(PUBLIC_AGGREGATE),
        "public_aggregate_rows": aggregate_rows,
        "posthoc_diagnostic_rows": {k: len(v) for k, v in diagnostics.items()},
        "original_run_artifacts_unchanged": True,
        "analysis_scope": "existing logged predictions, product metrics, and serialized initial/final SGD states; no learner execution",
    }
    (AUDIT_DIR / "r1_audit_provenance.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "original_run_artifact_sha256_manifest"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
