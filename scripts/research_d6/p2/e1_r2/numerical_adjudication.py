"""Limited post-hoc numerical adjudication for archived LP-D6-P2-E1-R1.

This script never calls the R1 runner's main entry point and refuses to overwrite
an existing R2 output directory. It performs exactly two preselected checkpoint
Newton-CG re-solves and one registered X/event-8 Newton-CG diagnostic.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
E1_DIR = ROOT / "scripts/research_d6/p2/e1"
if str(E1_DIR) not in sys.path:
    sys.path.insert(0, str(E1_DIR))
P1B_DIR = ROOT / "scripts/research_d6/p1b"
if str(P1B_DIR) not in sys.path:
    sys.path.insert(0, str(P1B_DIR))

import core as p1b_core  # noqa: E402
import run_experiment as p1b_runner  # noqa: E402
import run_e1 as r1  # noqa: E402
import proximal_logistic as proximal  # noqa: E402

EXPECTED_HEAD = "88868d4b0b18ffd6bb03e3d94b49e67904df044a"
EXPECTED_BRANCH = "stage3-eg-famar-execution"
R1_DIR = ROOT / "outputs/research_d6/p2/e1/run_20261115_r1_01"
OUT_DIR = ROOT / "outputs/research_d6/p2/e1_r2"
NEWTON_OPTIONS = {"xtol": 1e-12, "maxiter": 10000}
LAMBDA = 1e-4
GAMMAS = (0.0, 1e-4, 1e-3)
GRAD_TOL = 1e-6
OBJECTIVE_ATOL = 1e-8
SCORE_ATOL = 1e-5


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write an empty result CSV: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def output_tree_manifest(path: Path) -> dict[str, dict[str, Any]]:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return {
        p.relative_to(path).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in files
    }


def load_first_states_for_ids(
    requested: set[str], integrity: dict[str, Any]
) -> dict[str, str]:
    """Read only labels for requested IDs; evaluator IDs are loaded after solves."""
    if not requested:
        return {}
    queue_paths = [
        p1b_runner.D4LITE / "annotation_queue.csv",
        p1b_runner.D4B / "annotation_queue.csv",
        p1b_runner.V2 / "annotation_queue.csv",
    ]
    annotation_paths = [
        p1b_runner.D4LITE / "annotations.csv",
        p1b_runner.D4B / "annotations.csv",
        p1b_runner.V2 / "annotations.csv",
    ]
    found: dict[str, str] = {}
    for queue_path, annotation_path in zip(queue_paths, annotation_paths):
        queue = read_csv(queue_path)
        first_pid: dict[str, str] = {}
        for row in sorted(queue, key=lambda x: int(x["presentation_order"])):
            item_id = row["item_id"]
            if item_id in requested and item_id not in first_pid:
                first_pid[item_id] = row["presentation_id"]
        if not first_pid:
            continue
        pid_to_item = {pid: iid for iid, pid in first_pid.items()}
        with annotation_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                item_id = pid_to_item.get(row["presentation_id"])
                if item_id is None:
                    continue
                state = row["state"]
                require(state in p1b_core.VALID_STATES, f"invalid registered label state for {item_id}")
                found[item_id] = state
    require(set(found) == requested,
            f"requested permitted labels missing: {len(requested - set(found))}")
    return found


def load_feature_vectors(integrity: dict[str, Any]) -> dict[str, np.ndarray]:
    feature_rows = {r["image_id"]: r for r in integrity["feature_rows"]}
    features: dict[str, np.ndarray] = {}
    for image_id, row in feature_rows.items():
        rel = Path(row["feature_relpath"])
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe feature path in registered manifest")
        values = np.load(r1.P1A_DIR / rel, allow_pickle=False)
        require(values.dtype == np.float32 and values.shape == (1024,), "registered feature shape/dtype mismatch")
        require(np.isfinite(values).all(), "nonfinite cached feature")
        features[image_id] = values
    require(len(features) == 102, "expected 102 registered feature vectors")
    return features


def fit_vector(theta: np.ndarray, X: np.ndarray, y: np.ndarray, w0: np.ndarray, gamma: float) -> dict[str, float]:
    value, grad = proximal.objective_and_gradient(
        theta, X, y, w0, lam=LAMBDA, gamma=gamma
    )
    return {"objective": float(value), "gradient_inf_norm": float(np.max(np.abs(grad)))}


def solve_newton_cg(
    X: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    gamma: float,
    initial_theta: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Independent solver call with the registered R1 Newton-CG objective/options."""
    X64 = np.asarray(X, dtype=np.float64)
    y8 = np.asarray(y, dtype=np.int8)
    result = minimize(
        lambda theta, xx, yy, anchor: proximal.objective_and_gradient(
            theta, xx, yy, anchor, lam=LAMBDA, gamma=gamma
        ),
        np.asarray(initial_theta, dtype=np.float64).copy(),
        args=(X64, y8, np.asarray(w0, dtype=np.float64)),
        method="Newton-CG",
        jac=True,
        hessp=lambda theta, p, xx, yy, anchor: proximal.hessian_vector_product(
            theta, p, xx, yy, lam=LAMBDA, gamma=gamma
        ),
        options=dict(NEWTON_OPTIONS),
    )
    theta = np.asarray(result.x, dtype=np.float64)
    diag = fit_vector(theta, X64, y8, w0, gamma)
    diag.update({
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
    })
    return theta, diag


def gamma_prefix(gamma: float) -> str:
    return f"gamma_{gamma:.0e}"


def metric_bundle(scores: np.ndarray, future: list[Any]) -> dict[str, Any]:
    evaluated = r1.evaluate_scores(scores, future)
    metric = evaluated["metrics"]
    return {
        "ba_product_macro": metric["ba_product_macro"],
        "ba_eligible_product_count": metric["ba_eligible_product_count"],
        "ap_product_macro_paired_class": metric["ap_product_macro_paired_class"],
        "ap_paired_class_product_count": metric["ap_paired_class_product_count"],
        "roc_auc_product_macro": evaluated["roc_auc_product_macro"],
        "roc_auc_product_count": evaluated["roc_auc_product_count"],
        "products": evaluated["products"],
        "predictions": evaluated["predictions"],
        "binary_truth": evaluated["truth"],
    }


def main() -> int:
    require(not OUT_DIR.exists(), f"refusing to overwrite R2 output directory: {OUT_DIR}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    require(head == EXPECTED_HEAD, f"unexpected starting HEAD: {head}")
    require(branch == EXPECTED_BRANCH, f"unexpected branch: {branch}")
    require("torch" not in sys.modules, "Torch imported before R2 execution")
    require(Path(sys.executable).resolve() == Path(r"D:\anaconda3\envs\torch\python.exe").resolve(),
            "use the recorded R1 interpreter D:\\anaconda3\\envs\\torch\\python.exe")
    require(OUT_DIR.parent.exists(), "R1 output parent missing")

    provenance = json.loads((R1_DIR / "run_provenance.json").read_text(encoding="utf-8"))
    require(provenance.get("status") == "PARTIAL", "R1 output is not the archived PARTIAL run")
    require(provenance.get("execution_head") == "4c4319397de9496526248214a0ebe9e678f5ede7",
            "R1 output execution identity differs from reviewed record")
    expected_source_hashes = provenance["e1_source_code_sha256"]
    observed_source_hashes = r1.source_code_hashes()
    require(expected_source_hashes == observed_source_hashes, "R1 execution sources differ from archived provenance")

    r1_before = output_tree_manifest(R1_DIR)
    require(len(r1_before) >= 10, "R1 output inventory unexpectedly incomplete")
    original_p1b = r1.validate_original_p1b_hashes()
    require(len(original_p1b) == 15, "original P1-B artifact verification incomplete")

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    with threadpool_limits(limits=2):
        # Integrity preflight hashes only the already-authorized R1 source images and
        # frozen cache payloads; it never decodes pixels or opens masks/checkpoints.
        integrity = p1b_runner.capture_preflight()
        expected_cache = provenance["feature_cache"]
        require(integrity["snapshot_sha256"] == expected_cache["snapshot_sha256"],
                "R1 input/cache snapshot changed")
        require(integrity["feedback_tape_sha256"] == expected_cache["feedback_tape_sha256"],
                "R1 W/X feedback tape hash changed")
        require(integrity["input_manifest_sha256"] == expected_cache["input_manifest_sha256"],
                "R1 input manifest hash changed")
        require(integrity["feature_manifest_sha256"] == expected_cache["feature_index_sha256"],
                "R1 feature manifest hash changed")
        require(integrity["n_dfit"] == 54 and integrity["n_ddev"] == 48,
                "R1 source-role counts changed")

        tape_rows = integrity["tape_rows"]
        feedback_rows = {
            p: sorted((r for r in tape_rows if r["protocol"] == p and
                       r["feedback_or_future_role"] == "feedback_query"),
                      key=lambda r: int(r["stream_order"]))
            for p in ("W", "X")
        }
        fit_ids = [r["image_id"] for r in integrity["input_rows"] if r["data_role"] == "D-fit"]
        query_ids = {r["image_id"] for rows in feedback_rows.values() for r in rows}
        fit_and_query_states = load_first_states_for_ids(set(fit_ids) | query_ids, integrity)
        fit_states = {i: fit_and_query_states[i] for i in fit_ids}
        fit_targets = {i: p1b_core.binary_target(s) for i, s in fit_states.items()}
        binary_fit_ids = [i for i in fit_ids if fit_targets[i] is not None]
        require(len(binary_fit_ids) == 52 and
                sum(fit_targets[i] == 1 for i in binary_fit_ids) == 27 and
                sum(fit_targets[i] == 0 for i in binary_fit_ids) == 25,
                "D-fit binary supervision differs from the R1 registered set")
        features = load_feature_vectors(integrity)
        Xfit = np.stack([features[i] for i in binary_fit_ids]).astype(np.float64)
        yfit = np.asarray([fit_targets[i] for i in binary_fit_ids], dtype=np.int8)

        feedback: dict[str, list[Any]] = {}
        for protocol, rows in feedback_rows.items():
            items = []
            for row in rows:
                iid = row["image_id"]
                state = fit_and_query_states[iid]
                require(row["revealed_fields"] == "surface_abrasion_visible_state_after_prediction",
                        f"unexpected revealed field in {protocol} tape")
                items.append(p1b_core.FeedbackItem(
                    event_id=row["event_id"], image_id=iid, product=row["product"],
                    sample_key=row["sample_key"], stream_order=int(row["stream_order"]),
                    state=state, x=features[iid],
                ))
            feedback[protocol] = items
            require(len(items) == 12, f"unexpected {protocol} feedback event count")

        r1_checks = read_csv(R1_DIR / "newton_cg_verification.csv")
        require(len(r1_checks) == 15, "expected exactly 15 archived Newton-CG checks")
        # The predeclared rule uses only archived numerical score disagreement.
        # Ties are resolved by protocol, numeric gamma, then checkpoint.
        ordered = sorted(
            r1_checks,
            key=lambda row: (-float(row["max_future_score_abs_difference"]),
                             row["protocol"], float(row["gamma"]), int(row["checkpoint"])),
        )
        selected = ordered[:2]
        require([(r["protocol"], float(r["gamma"]), int(r["checkpoint"])) for r in selected] ==
                [("W", 1e-4, 0), ("W", 1e-4, 3)],
                "predeclared top-two checkpoint selection differs from archived numerical rows")

        # Persist a call ledger before the first new solver invocation so a crash
        # cannot lead to an accidental second run of the same authorized solve.
        OUT_DIR.mkdir(parents=True, exist_ok=False)
        solver_attempts: list[dict[str, Any]] = []
        attempts_path = OUT_DIR / "solver_attempt_ledger.json"

        def persist_attempts() -> None:
            tmp = attempts_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(json_ready(solver_attempts), indent=2), encoding="utf-8")
            os.replace(tmp, attempts_path)

        def counted_solve(label: str, X: np.ndarray, y: np.ndarray, w0: np.ndarray,
                          gamma: float, initial: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            require(len(solver_attempts) < 3, "R2 maximum of three Newton-CG calls would be exceeded")
            entry = {"solve_index": len(solver_attempts) + 1, "label": label,
                     "gamma": gamma, "training_rows": int(len(y)), "status": "STARTED"}
            solver_attempts.append(entry)
            persist_attempts()
            try:
                theta_value, diagnostics = solve_newton_cg(X, y, w0, gamma, initial)
            except BaseException as exc:
                entry.update({"status": "RAISED", "exception_type": type(exc).__name__,
                              "exception": str(exc)})
                persist_attempts()
                raise
            entry.update({"status": "RETURNED", "success": diagnostics["success"],
                          "message": diagnostics["message"],
                          "gradient_inf_norm": diagnostics["gradient_inf_norm"],
                          "objective": diagnostics["objective"],
                          "iterations": diagnostics["iterations"],
                          "function_evaluations": diagnostics["function_evaluations"]})
            persist_attempts()
            return theta_value, diagnostics

        # Fit exactly two preselected existing checkpoints plus the registered event-8
        # point; no future evaluator labels are loaded before all three solves finish.
        with np.load(R1_DIR / "model_states.npz", allow_pickle=False) as states:
            w0 = states["registered_anchor_w0"].astype(np.float64, copy=True)
            b0 = float(states["registered_anchor_b0"].reshape(-1)[0])
            require(p1b_core.array_digest(w0.reshape(1, -1), np.asarray([b0], dtype=np.float64)) ==
                    provenance["anchor"]["initial_learner_state_sha256"],
                    "frozen R1 anchor state hash mismatch")
            selected_solves: list[dict[str, Any]] = []
            for row in selected:
                protocol = row["protocol"]
                gamma = float(row["gamma"])
                cp = int(row["checkpoint"])
                prefix = f"{protocol}__{gamma_prefix(gamma)}__checkpoint_{cp:02d}__"
                theta_lbfgs = np.concatenate([
                    states[prefix + "coef"].reshape(-1).astype(np.float64),
                    states[prefix + "intercept"].reshape(-1).astype(np.float64),
                ])
                saved_fx = states[prefix + "feedback_x"].astype(np.float32, copy=True)
                saved_fy = states[prefix + "feedback_y"].astype(np.int8, copy=True)
                used_events = [e for e in feedback[protocol]
                               if e.stream_order <= cp and p1b_core.binary_target(e.state) is not None]
                event_fx = (np.stack([e.x for e in used_events]).astype(np.float32)
                            if used_events else np.empty((0, 1024), dtype=np.float32))
                event_fy = np.asarray([p1b_core.binary_target(e.state) for e in used_events], dtype=np.int8)
                require(np.array_equal(saved_fx, event_fx) and np.array_equal(saved_fy, event_fy),
                        f"saved feedback history differs from legal {protocol} tape through checkpoint {cp}")
                X = np.vstack([Xfit, event_fx.astype(np.float64)])
                y = np.concatenate([yfit, event_fy])
                lbfgs_diag = fit_vector(theta_lbfgs, X, y, w0, gamma)
                theta_newton, diag = counted_solve(
                    f"checkpoint:{protocol}:gamma={gamma}:cp={cp}",
                    X, y, w0, gamma, np.zeros(X.shape[1] + 1, dtype=np.float64)
                )
                selected_solves.append({
                    "protocol": protocol, "gamma": gamma, "checkpoint": cp,
                    "recorded_score_discrepancy": float(row["max_future_score_abs_difference"]),
                    "training_rows": int(len(y)), "event_count": len(used_events),
                    "theta_lbfgs": theta_lbfgs, "theta_newton": theta_newton,
                    "newton_diag": diag, "lbfgs_diag": lbfgs_diag,
                })

            # The failure diagnostic uses exactly X feedback events 1..8 and accepted
            # checkpoint-7 parameters as the sole permitted nonzero Newton start.
            x_events = feedback["X"]
            first_eight = x_events[:8]
            require([e.stream_order for e in first_eight] == list(range(1, 9)),
                    "X feedback tape does not contain registered events 1-8")
            require(all(p1b_core.binary_target(e.state) is not None for e in first_eight),
                    "X events 1-8 include a nonbinary state")
            event8_prefix = np.stack([e.x for e in first_eight]).astype(np.float32)
            event8_y = np.asarray([p1b_core.binary_target(e.state) for e in first_eight], dtype=np.int8)
            cp7 = "X__gamma_1e-03__checkpoint_07__"
            require(np.array_equal(states[cp7 + "feedback_x"], event8_prefix[:7]) and
                    np.array_equal(states[cp7 + "feedback_y"], event8_y[:7]),
                    "X gamma=1e-3 checkpoint-7 state does not match events 1-7")
            theta7 = np.concatenate([
                states[cp7 + "coef"].reshape(-1).astype(np.float64),
                states[cp7 + "intercept"].reshape(-1).astype(np.float64),
            ])
            X8 = np.vstack([Xfit, event8_prefix.astype(np.float64)])
            y8 = np.concatenate([yfit, event8_y])
            theta8, diag8 = counted_solve("event:X:gamma=0.001:event=8", X8, y8, w0, 1e-3, theta7)

        # Only after the three solves have returned do evaluator-only labels become
        # available for margins and score/metric comparisons.
        future_rows = {
            p: [r for r in tape_rows if r["protocol"] == p and
                r["feedback_or_future_role"] == "future_eval_candidate"]
            for p in ("W", "X")
        }
        future_ids = {r["image_id"] for rows in future_rows.values() for r in rows}
        future_states = load_first_states_for_ids(future_ids, integrity)
        futures: dict[str, list[Any]] = {
            p: [p1b_core.FutureItem(
                image_id=row["image_id"], product=row["product"],
                sample_key=row["sample_key"], state=future_states[row["image_id"]],
                x=features[row["image_id"]],
            ) for row in rows]
            for p, rows in future_rows.items()
        }

        prediction_rows = read_csv(R1_DIR / "future_predictions.csv")
        bound_rows: list[dict[str, Any]] = []
        for row in r1_checks:
            protocol = row["protocol"]
            gamma = float(row["gamma"])
            cp = int(row["checkpoint"])
            matches = [p for p in prediction_rows if p["protocol"] == protocol and
                       float(p["gamma"]) == gamma and int(p["checkpoint"]) == cp]
            binary_rows = [p for p in matches if p["binary_target_evaluator_only"] in ("0", "1")]
            require(len(matches) == 24 and len(binary_rows) == 21,
                    f"unexpected R1 future prediction support at {protocol}/{gamma}/{cp}")
            scores = np.asarray([float(p["score"]) for p in binary_rows], dtype=np.float64)
            labels = np.asarray([int(p["binary_target_evaluator_only"]) for p in binary_rows], dtype=np.int8)
            epsilon = float(row["max_future_score_abs_difference"])
            min_boundary = float(np.min(np.abs(scores)))
            class_status = "STRICTLY_EXCLUDED" if min_boundary > epsilon else "UNRESOLVED_BY_BOUND"
            min_pair_gap = float("inf")
            paired_products = 0
            for product in sorted({p["product"] for p in binary_rows}):
                idx = [i for i, p in enumerate(binary_rows) if p["product"] == product]
                ys = labels[idx]
                pos = scores[idx][ys == 1]
                neg = scores[idx][ys == 0]
                if len(pos) and len(neg):
                    paired_products += 1
                    gap = float(np.min(np.abs(pos[:, None] - neg[None, :])))
                    min_pair_gap = min(min_pair_gap, gap)
            if paired_products == 0:
                ranking_status = "NO_PAIRED_CLASS_PRODUCT"
                min_pair_gap_value: float | None = None
            else:
                min_pair_gap_value = min_pair_gap
                ranking_status = "STRICTLY_EXCLUDED" if min_pair_gap > 2.0 * epsilon else "UNRESOLVED_BY_BOUND"
            bound_rows.append({
                "protocol": protocol, "gamma": gamma, "checkpoint": cp,
                "archived_max_score_difference_epsilon": epsilon,
                "binary_future_views": len(binary_rows),
                "minimum_absolute_zero_threshold_margin": min_boundary,
                "classification_change_excluded_by_bound": class_status,
                "minimum_within_product_positive_negative_score_gap": min_pair_gap_value,
                "paired_class_products": paired_products,
                "ranking_change_excluded_by_bound": ranking_status,
                "ba_impact_bound": "UNCHANGED" if class_status == "STRICTLY_EXCLUDED" else "UNRESOLVED_BY_BOUND",
                "paired_ap_auc_impact_bound": "UNCHANGED" if ranking_status == "STRICTLY_EXCLUDED" else "UNRESOLVED_BY_BOUND",
            })

        selected_rows: list[dict[str, Any]] = []
        product_metric_rows: list[dict[str, Any]] = []
        for solve_rank, solve in enumerate(selected_solves, start=1):
            protocol, gamma, cp = solve["protocol"], solve["gamma"], solve["checkpoint"]
            future = futures[protocol]
            XF = np.stack([item.x for item in future]).astype(np.float64)
            theta_lbfgs, theta_newton = solve["theta_lbfgs"], solve["theta_newton"]
            score_lbfgs = XF @ theta_lbfgs[:-1] + theta_lbfgs[-1]
            score_newton = XF @ theta_newton[:-1] + theta_newton[-1]
            stored = [p for p in prediction_rows if p["protocol"] == protocol and
                      float(p["gamma"]) == gamma and int(p["checkpoint"]) == cp]
            stored_by_id = {p["image_id"]: float(p["score"]) for p in stored}
            max_saved_score_reconstruction = max(
                abs(float(score_lbfgs[i]) - stored_by_id[item.image_id])
                for i, item in enumerate(future)
            )
            require(max_saved_score_reconstruction <= 1e-8,
                    f"R1 checkpoint state does not reconstruct archived logits at {protocol}/{gamma}/{cp}")
            m_lb = metric_bundle(score_lbfgs, future)
            m_new = metric_bundle(score_newton, future)
            truth = m_lb["binary_truth"]
            ids = sorted(truth)
            agreement = float(np.mean([
                m_lb["predictions"][iid] == m_new["predictions"][iid] for iid in ids
            ]))
            grad_lb = solve["lbfgs_diag"]["gradient_inf_norm"]
            grad_new = solve["newton_diag"]["gradient_inf_norm"]
            obj_diff = abs(solve["newton_diag"]["objective"] - solve["lbfgs_diag"]["objective"])
            max_score_diff = float(np.max(np.abs(score_newton - score_lbfgs)))
            diag = solve["newton_diag"]
            selected_rows.append({
                "protocol": protocol, "gamma": gamma, "checkpoint": cp,
                "selection_rule_rank": solve_rank,
                "recorded_r1_max_score_difference": solve["recorded_score_discrepancy"],
                "training_rows": solve["training_rows"], "binary_future_views": len(ids),
                "newton_success": diag["success"], "newton_message": diag["message"],
                "newton_iterations": diag["iterations"],
                "newton_function_evaluations": diag["function_evaluations"],
                "newton_objective": diag["objective"],
                "lbfgsb_objective": solve["lbfgs_diag"]["objective"],
                "objective_abs_difference": obj_diff,
                "newton_gradient_inf_norm": grad_new,
                "lbfgsb_gradient_inf_norm": grad_lb,
                "max_future_score_abs_difference": max_score_diff,
                "future_prediction_agreement": agreement,
                "ba_lbfgsb": m_lb["ba_product_macro"], "ba_newton": m_new["ba_product_macro"],
                "ba_difference_newton_minus_lbfgsb": m_new["ba_product_macro"] - m_lb["ba_product_macro"],
                "paired_ap_lbfgsb": m_lb["ap_product_macro_paired_class"],
                "paired_ap_newton": m_new["ap_product_macro_paired_class"],
                "paired_ap_difference_newton_minus_lbfgsb":
                    m_new["ap_product_macro_paired_class"] - m_lb["ap_product_macro_paired_class"],
                "roc_auc_lbfgsb": m_lb["roc_auc_product_macro"],
                "roc_auc_newton": m_new["roc_auc_product_macro"],
                "roc_auc_difference_newton_minus_lbfgsb":
                    m_new["roc_auc_product_macro"] - m_lb["roc_auc_product_macro"],
                "weight_l2_difference": float(np.linalg.norm(theta_newton[:-1] - theta_lbfgs[:-1])),
                "intercept_difference": float(theta_newton[-1] - theta_lbfgs[-1]),
                "r1_logits_reconstructed_max_difference": max_saved_score_reconstruction,
                "original_registered_numeric_gates": json.dumps({
                    "gradient_inf_le_1e-6": diag["success"] and grad_new <= GRAD_TOL and grad_lb <= GRAD_TOL,
                    "objective_diff_le_1e-8": obj_diff <= OBJECTIVE_ATOL,
                    "score_diff_le_1e-5": max_score_diff <= SCORE_ATOL,
                }, sort_keys=True),
            })
            lb_eval = r1.evaluate_scores(score_lbfgs, future)
            new_eval = r1.evaluate_scores(score_newton, future)
            for lb_product, new_product in zip(m_lb["products"], m_new["products"]):
                p = lb_product["product"]
                lb_auc = lb_eval["roc_auc_product_values"].get(p)
                new_auc = new_eval["roc_auc_product_values"].get(p)
                product_metric_rows.append({
                    "protocol": protocol, "gamma": gamma, "checkpoint": cp,
                    "product": p,
                    "n_binary": lb_product["n_binary"],
                    "ba_lbfgsb": lb_product["ba"], "ba_newton": new_product["ba"],
                    "ba_difference_newton_minus_lbfgsb":
                        None if lb_product["ba"] is None else new_product["ba"] - lb_product["ba"],
                    "ap_lbfgsb": lb_product["ap"], "ap_newton": new_product["ap"],
                    "ap_difference_newton_minus_lbfgsb":
                        None if lb_product["ap"] is None else new_product["ap"] - lb_product["ap"],
                    "roc_auc_lbfgsb": lb_auc, "roc_auc_newton": new_auc,
                    "roc_auc_difference_newton_minus_lbfgsb":
                        None if lb_auc is None else new_auc - lb_auc,
                })

        failure_rows = [r for r in read_csv(R1_DIR / "feedback_event_log.csv")
                        if r["protocol"] == "X" and float(r["gamma"]) == 1e-3 and
                        r["event_id"] == "X-FB-008"]
        require(len(failure_rows) == 1, "registered X/gamma=1e-3 event-8 failure record missing/duplicated")
        failure = failure_rows[0]
        event8_obj_diff_signed = diag8["objective"] - float(failure["objective_after_update"])
        event8 = {
            "protocol": "X", "gamma": 1e-3, "event_id": "X-FB-008", "stream_order": 8,
            "training_rows": int(len(y8)), "binary_feedback_events_used": 8,
            "initial_guess": "accepted checkpoint 7 parameters",
            "newton_options": NEWTON_OPTIONS,
            "newton_success": diag8["success"], "newton_message": diag8["message"],
            "newton_iterations": diag8["iterations"],
            "newton_function_evaluations": diag8["function_evaluations"],
            "newton_objective": diag8["objective"],
            "newton_gradient_inf_norm": diag8["gradient_inf_norm"],
            "original_gradient_gate": GRAD_TOL,
            "newton_meets_original_gradient_gate": bool(diag8["success"] and diag8["gradient_inf_norm"] <= GRAD_TOL),
            "rejected_lbfgsb_candidate_objective": float(failure["objective_after_update"]),
            "rejected_lbfgsb_candidate_gradient_inf_norm": float(failure["gradient_inf_after_update"]),
            "newton_minus_rejected_candidate_objective": event8_obj_diff_signed,
            "absolute_objective_difference": abs(event8_obj_diff_signed),
            "rejected_candidate_parameters_available": False,
            "rejected_candidate_parameter_difference": "UNKNOWN_NOT_ARCHIVED",
            "interpretation": (
                "Newton-CG reached the registered gradient gate; this is consistent with an L-BFGS-B "
                "stopping-condition issue for this single event, but does not repair or pass the R1 track."
                if diag8["success"] and diag8["gradient_inf_norm"] <= GRAD_TOL
                else "Newton-CG did not meet the registered gate; this single-point numerical issue remains unresolved."
            ),
        }

        # Postflight verifies no source/cache or R1 artifact changed during diagnosis.
        post_integrity = p1b_runner.capture_preflight()
        require(post_integrity["snapshot_sha256"] == integrity["snapshot_sha256"],
                "registered input/cache snapshot changed during R2")
        require(output_tree_manifest(R1_DIR) == r1_before, "archived R1 output tree changed during R2")
        require(r1.validate_original_p1b_hashes() == original_p1b,
                "original P1-B output hashes changed during R2")

    require(len(solver_attempts) == 3 and all(x["status"] == "RETURNED" for x in solver_attempts),
            "the authorized three solver calls did not all return; preserve directory and report blocker")
    write_csv(OUT_DIR / "checkpoint_bound_analysis.csv", bound_rows)
    write_csv(OUT_DIR / "selected_checkpoint_newton_rechecks.csv", selected_rows)
    write_csv(OUT_DIR / "selected_checkpoint_product_metrics.csv", product_metric_rows)
    (OUT_DIR / "event8_newton_diagnostic.json").write_text(
        json.dumps(json_ready(event8), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "r1_output_sha256_manifest.json").write_text(
        json.dumps(r1_before, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    elapsed_wall = time.perf_counter() - wall_start
    elapsed_cpu = time.process_time() - cpu_start
    provenance_out = {
        "task": "LP-D6-P2-E1-R2 limited numerical adjudication",
        "status": "DIAGNOSTIC_ONLY_R1_PARTIAL_UNCHANGED",
        "execution_head": head, "branch": branch,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "torch_imported": "torch" in sys.modules,
        "device": "CPU only", "thread_limit": 2,
        "original_input_snapshot_sha256": integrity["snapshot_sha256"],
        "feedback_tape_sha256": integrity["feedback_tape_sha256"],
        "input_manifest_sha256": integrity["input_manifest_sha256"],
        "feature_manifest_sha256": integrity["feature_manifest_sha256"],
        "anchor_state_sha256": provenance["anchor"]["initial_learner_state_sha256"],
        "r1_source_sha256_verified": observed_source_hashes,
        "r1_provenance_sha256": sha256_file(R1_DIR / "run_provenance.json"),
        "r1_output_inventory_sha256_manifest": "r1_output_sha256_manifest.json",
        "original_p1b_files_verified": len(original_p1b),
        "newton_cg_solves": 3,
        "selected_checkpoint_rule": "descending archived max_future_score_abs_difference; ties protocol, numeric gamma, checkpoint",
        "selected_checkpoints": [
            {"protocol": s["protocol"], "gamma": s["gamma"], "checkpoint": s["checkpoint"],
             "archived_score_difference": s["recorded_score_discrepancy"]}
            for s in selected_solves
        ],
        "event8_newton_gradient_gate_pass": event8["newton_meets_original_gradient_gate"],
        "lambda": LAMBDA, "gamma_registered": list(GAMMAS),
        "gradient_inf_gate": GRAD_TOL,
        "newton_objective_atol": OBJECTIVE_ATOL,
        "future_score_atol": SCORE_ATOL,
        "newton_options": NEWTON_OPTIONS,
        "elapsed_wall_seconds": elapsed_wall,
        "elapsed_process_cpu_seconds": elapsed_cpu,
        "future_labels_loaded_after_all_solves": True,
        "r1_artifacts_unchanged": True,
        "p1b_original_artifacts_unchanged": True,
        "input_snapshot_unchanged": True,
    }
    (OUT_DIR / "run_provenance.json").write_text(
        json.dumps(json_ready(provenance_out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("R2_STATUS=COMPLETE_DIAGNOSTIC_ONLY")
    print("newton_cg_solves=3")
    print("selected=" + ",".join(f"{s['protocol']}/g={s['gamma']}/cp={s['checkpoint']}" for s in selected_solves))
    print("event8_gradient_inf=" + f"{diag8['gradient_inf_norm']:.12g}")
    print("event8_gate=" + str(event8["newton_meets_original_gradient_gate"]))
    print("r1_partial_preserved=True")
    print("output=" + str(OUT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
