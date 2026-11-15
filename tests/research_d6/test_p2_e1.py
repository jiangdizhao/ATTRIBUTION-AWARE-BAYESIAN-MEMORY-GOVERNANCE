"""Synthetic and integrity tests for LP-D6-P2-E1; no model/data fitting on D-dev."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import sys
import time
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
E1_DIR = ROOT / "scripts/research_d6/p2/e1"
if str(E1_DIR) not in sys.path:
    sys.path.insert(0, str(E1_DIR))

from proximal_logistic import (  # noqa: E402
    GRAD_INF_TOL,
    LAMBDA,
    ProximalLogistic,
    SolverFailure,
    fit_lbfgsb,
    fit_lbfgsb,
    hessian_vector_product,
    objective_and_gradient,
)
from run_e1 import (  # noqa: E402
    APPROVED_G0_COMMIT,
    directional_gradient_check,
    is_ancestor,
    protected_snapshot,
    replay_proximal,
    validate_feedback_future_roles,
    validate_original_p1b_hashes,
)
import core as p1b_core  # noqa: E402
import run_experiment as p1b_runner  # noqa: E402


@dataclass(frozen=True)
class TinyItem:
    image_id: str
    product: str
    sample_key: str
    state: str
    x: np.ndarray
    event_id: str = ""
    stream_order: int = 0


def tiny_data():
    rng = np.random.default_rng(79)
    base_x = rng.normal(size=(10, 5)).astype(np.float32)
    base_y = np.asarray([0, 1] * 5, dtype=np.int8)
    anchor_w = np.asarray([0.1, -0.2, 0.05, 0.08, -0.1], dtype=np.float64)
    anchor_b = -0.03
    feedback = [
        TinyItem("Q1", "p", "query-1", "present", rng.normal(size=5).astype(np.float32), "E1", 1),
        TinyItem("Q2", "p", "query-2", "absent_assessable", rng.normal(size=5).astype(np.float32), "E2", 2),
        TinyItem("Q3", "p", "query-3", "unobservable_or_uncertain", rng.normal(size=5).astype(np.float32), "E3", 3),
        TinyItem("Q4", "p", "query-4", "not_applicable", rng.normal(size=5).astype(np.float32), "E4", 4),
    ]
    future = [
        TinyItem("F1", "p", "future-1", "present", rng.normal(size=5).astype(np.float32)),
        TinyItem("F2", "p", "future-2", "absent_assessable", rng.normal(size=5).astype(np.float32)),
        TinyItem("F3", "p", "future-3", "present", rng.normal(size=5).astype(np.float32)),
        TinyItem("F4", "p", "future-4", "absent_assessable", rng.normal(size=5).astype(np.float32)),
    ]
    return base_x, base_y, anchor_w, anchor_b, feedback, future


class ObjectiveTests(unittest.TestCase):
    def test_new_style_gamma_zero_initializer_is_numerically_converged(self):
        rng = np.random.default_rng(20261115)
        X = rng.normal(size=(52, 12)).astype(np.float32)
        y = np.asarray([1] * 27 + [0] * 25, dtype=np.int8)
        rng.shuffle(y)
        old_w = rng.normal(size=12) * 0.03
        old_b = 0.12
        theta, diagnostics = fit_lbfgsb(
            X, y, old_w, np.concatenate([old_w, [old_b]]), lam=LAMBDA, gamma=0.0
        )
        self.assertEqual(diagnostics.status, "PASS")
        self.assertLessEqual(diagnostics.gradient_inf_norm, GRAD_INF_TOL)
        check = directional_gradient_check(theta, X, y, theta[:-1])
        self.assertLessEqual(check["max_abs_directional_error"], 1e-7)

    def test_gradient_matches_central_finite_differences(self):
        rng = np.random.default_rng(20261115)
        X = rng.normal(size=(11, 7)).astype(np.float64)
        y = np.asarray([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.float64)
        w0 = rng.normal(size=7)
        theta = rng.normal(size=8) * 0.2
        value, gradient = objective_and_gradient(theta, X, y, w0, lam=LAMBDA, gamma=1e-3)
        numerical = np.zeros_like(theta)
        eps = 1e-6
        for i in range(theta.size):
            step = np.zeros_like(theta)
            step[i] = eps
            plus = objective_and_gradient(theta + step, X, y, w0, lam=LAMBDA, gamma=1e-3)[0]
            minus = objective_and_gradient(theta - step, X, y, w0, lam=LAMBDA, gamma=1e-3)[0]
            numerical[i] = (plus - minus) / (2 * eps)
        self.assertTrue(np.isfinite(value))
        self.assertLess(float(np.max(np.abs(gradient - numerical))), 2e-8)

    def test_gamma_zero_is_exact_ordinary_full_history_logistic_objective(self):
        rng = np.random.default_rng(44)
        X = rng.normal(size=(9, 4))
        y = np.asarray([0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.float64)
        w0 = rng.normal(size=4)
        theta = rng.normal(size=5)
        w, b = theta[:-1], theta[-1]
        scores = X @ w + b
        ordinary = np.mean(np.logaddexp(0.0, scores) - y * scores) + 0.5 * LAMBDA * np.dot(w, w)
        value, _ = objective_and_gradient(theta, X, y, w0, lam=LAMBDA, gamma=0.0)
        self.assertAlmostEqual(value, float(ordinary), places=14)

    def test_hessian_vector_product_matches_gradient_difference(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=(8, 4))
        y = np.asarray([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.float64)
        w0 = rng.normal(size=4)
        theta = rng.normal(size=5) * 0.2
        vector = rng.normal(size=5)
        eps = 1e-6
        grad_plus = objective_and_gradient(theta + eps * vector, X, y, w0, gamma=1e-3)[1]
        grad_minus = objective_and_gradient(theta - eps * vector, X, y, w0, gamma=1e-3)[1]
        numerical = (grad_plus - grad_minus) / (2 * eps)
        analytic = hessian_vector_product(theta, vector, X, y, gamma=1e-3)
        self.assertLess(float(np.max(np.abs(numerical - analytic))), 2e-8)

    def test_solver_failure_and_ftol_only_are_not_accepted(self):
        X = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        y = np.asarray([1, 1, 0, 0], dtype=np.int8)
        w0 = np.zeros(2, dtype=np.float64)
        theta0 = np.zeros(3, dtype=np.float64)
        failed = SimpleNamespace(success=False, x=theta0, message="line search failed", nit=2, nfev=3)
        with self.assertRaises(SolverFailure) as ctx:
            fit_lbfgsb(X, y, w0, theta0, gamma=1e-4, minimize_fn=lambda *a, **k: failed)
        self.assertEqual(ctx.exception.status, "SOLVER_FAILED")
        ftol = SimpleNamespace(success=True, x=theta0, message="CONVERGENCE: RELATIVE REDUCTION OF F", nit=1, nfev=2)
        with self.assertRaises(SolverFailure) as ctx:
            fit_lbfgsb(X, y, w0, theta0, gamma=1e-4, minimize_fn=lambda *a, **k: ftol)
        self.assertEqual(ctx.exception.status, "FTOL_ONLY_NOT_CONVERGED")


class ReplayTests(unittest.TestCase):
    def test_failed_optimizer_stops_only_its_trajectory_without_state_mutation(self):
        base_x, base_y, w0, b0, feedback, future = tiny_data()
        failed_result = SimpleNamespace(
            codex_objective=0.4,
            codex_gradient_inf_norm=1.2e-6,
            codex_elapsed_seconds=0.01,
            nit=7,
            nfev=9,
            message="CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH",
        )

        def reject_update(_model, _x, _y, _event_key):
            raise SolverFailure("FTOL_ONLY_NOT_CONVERGED", "synthetic strict-gate failure",
                                result=failed_result)

        with patch.object(ProximalLogistic, "observe", reject_update):
            result = replay_proximal(
                protocol="X", gamma=1e-3, x_fit=base_x, y_fit=base_y,
                w0=w0, b0=b0, feedback=feedback, future=future,
                cpu_start=time.process_time(),
            )
        self.assertEqual(result["status"], "FTOL_ONLY_NOT_CONVERGED")
        self.assertEqual(result["fit_count"], 0)
        self.assertEqual(sorted(result["states"]), [0])
        self.assertEqual(len(result["rows"]["events"]), 1)
        row = result["rows"]["events"][0]
        self.assertTrue(row["update_attempted"])
        self.assertFalse(row["update_applied"])
        self.assertEqual(row["state_sha256_before"], row["state_sha256_after"])

    def test_all_gamma_tracks_share_exact_initial_state(self):
        base_x, base_y, w0, b0, feedback, future = tiny_data()
        initial = []
        for gamma in (0.0, 1e-4, 1e-3):
            result = replay_proximal(protocol="X", gamma=gamma, x_fit=base_x, y_fit=base_y,
                                     w0=w0, b0=b0, feedback=feedback, future=future,
                                     cpu_start=time.process_time())
            initial.append(result["states"][0])
        for state in initial[1:]:
            np.testing.assert_array_equal(state["coef"], initial[0]["coef"])
            np.testing.assert_array_equal(state["intercept"], initial[0]["intercept"])

    def test_prediction_precedes_reveal_and_unknown_states_do_not_update(self):
        base_x, base_y, w0, b0, feedback, future = tiny_data()
        revealed = []
        score_calls = {"single": 0}
        original_scores = ProximalLogistic.scores

        def instrumented_scores(model, x):
            values = np.asarray(x)
            result = original_scores(model, values)
            if values.ndim == 1 or (values.ndim == 2 and values.shape[0] == 1):
                score_calls["single"] += 1
            return result

        def reveal(event):
            self.assertGreater(score_calls["single"], len(revealed))
            revealed.append(event.event_id)
            return event.state

        with patch.object(ProximalLogistic, "scores", instrumented_scores):
            result = replay_proximal(
                protocol="X", gamma=1e-4, x_fit=base_x, y_fit=base_y,
                w0=w0, b0=b0, feedback=feedback, future=future,
                cpu_start=time.process_time(), reveal_state=reveal,
            )
        self.assertEqual(result["fit_count"], 2)
        self.assertEqual(result["learner"].event_keys, ["E1", "E2"])
        unknown = [e for e in result["rows"]["events"] if e["event_id"] in {"E3", "E4"}]
        self.assertEqual(len(unknown), 2)
        for row in unknown:
            self.assertFalse(row["update_applied"])
            self.assertEqual(row["state_sha256_before"], row["state_sha256_after"])
            self.assertEqual(row["solver_status"], "NO_BINARY_UPDATE")
        self.assertTrue(all(row["prediction_timing"] == "before_reveal" for row in result["rows"]["events"]))

    def test_future_label_counterfactual_cannot_change_predictions_or_states(self):
        base_x, base_y, w0, b0, feedback, future = tiny_data()
        changed = [replace(item, state=("absent_assessable" if item.state == "present" else "present"))
                   for item in future]
        first = replay_proximal(protocol="X", gamma=1e-4, x_fit=base_x, y_fit=base_y,
                                w0=w0, b0=b0, feedback=feedback, future=future,
                                cpu_start=time.process_time())
        second = replay_proximal(protocol="X", gamma=1e-4, x_fit=base_x, y_fit=base_y,
                                 w0=w0, b0=b0, feedback=feedback, future=changed,
                                 cpu_start=time.process_time())
        self.assertEqual([r["state_sha256_after"] for r in first["rows"]["events"]],
                         [r["state_sha256_after"] for r in second["rows"]["events"]])
        self.assertEqual([r["pre_reveal_score"] for r in first["rows"]["events"]],
                         [r["pre_reveal_score"] for r in second["rows"]["events"]])
        self.assertEqual([r["prediction"] for r in first["rows"]["predictions"]],
                         [r["prediction"] for r in second["rows"]["predictions"]])

    def test_feedback_future_sample_key_crossover_is_rejected(self):
        base_x, _, _, _, feedback, future = tiny_data()
        overlapping_future = [replace(future[0], product=feedback[0].product,
                                      sample_key=feedback[0].sample_key)] + future[1:]
        with self.assertRaises(RuntimeError):
            validate_feedback_future_roles(feedback, overlapping_future)

    def test_anchor_and_source_arrays_remain_immutable_across_updates(self):
        base_x, base_y, w0, b0, feedback, _future = tiny_data()
        base_before = base_x.copy()
        anchor_before = w0.copy()
        model = ProximalLogistic(base_x, base_y, w0, b0, gamma=1e-3)
        self.assertFalse(model.anchor_w.flags.writeable)
        model.observe(feedback[0].x, 1, feedback[0].event_id)
        np.testing.assert_array_equal(model.anchor_w, anchor_before)
        np.testing.assert_array_equal(base_x, base_before)
        np.testing.assert_array_equal(w0, anchor_before)

    def test_deterministic_repeatability_with_identical_inputs(self):
        base_x, base_y, w0, b0, feedback, future = tiny_data()
        outputs = []
        for _ in range(2):
            result = replay_proximal(protocol="X", gamma=1e-4, x_fit=base_x, y_fit=base_y,
                                     w0=w0, b0=b0, feedback=feedback, future=future,
                                     cpu_start=time.process_time())
            outputs.append((result["learner"].state_digest(), result["learner"].w.copy(),
                            result["learner"].b))
        self.assertEqual(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        self.assertEqual(outputs[0][2], outputs[1][2])


class InputPreservationTests(unittest.TestCase):
    def test_approved_g0_commit_is_in_current_head_ancestry(self):
        self.assertTrue(is_ancestor(APPROVED_G0_COMMIT, "HEAD"))

    def test_registered_preflight_and_original_p1b_hashes(self):
        before = protected_snapshot()
        integrity = p1b_runner.capture_preflight()
        hashes = validate_original_p1b_hashes()
        after = protected_snapshot()
        self.assertEqual(integrity["n_dfit"], 54)
        self.assertEqual(integrity["n_ddev"], 48)
        self.assertEqual(integrity["feedback_tape_sha256"],
                         "13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1")
        self.assertEqual(len(hashes), 15)
        self.assertEqual(before, after)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        unittest.main()

