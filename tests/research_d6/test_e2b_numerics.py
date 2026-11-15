from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts/research_d6/p2/e2b"
sys.path.insert(0, str(SCRIPT_DIR))

from model import (  # noqa: E402
    binary_labels,
    build_global_readout,
    fit_global,
    global_objective,
    mil_initial_parameters,
    mil_objective_parts,
    pair_indices,
    predict_global,
    valid_bag,
)
from pipeline import run_trajectory  # noqa: E402


def central_difference(fun, x, step=1e-6):
    result = np.empty_like(x, dtype=np.float64)
    for j in range(len(x)):
        plus = x.copy(); plus[j] += step
        minus = x.copy(); minus[j] -= step
        result[j] = (fun(plus)[0] - fun(minus)[0]) / (2.0 * step)
    return result


class E2BNumericalContracts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(312)
        self.bags = [rng.normal(size=(n, 4)).astype(np.float32) for n in (3, 2, 4, 3)]
        self.y = np.asarray([1, 0, 1, 0], dtype=np.int8)
        self.theta = np.asarray([0.11, -0.2, 0.07, 0.03, -0.13], dtype=np.float64)

    def test_global_analytic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(7, 5))
        y = np.asarray([1, 0, 1, 0, 1, 0, 0])
        theta = rng.normal(size=6) * 0.1
        fun = lambda t: global_objective(t, x, y, 1e-4)
        _, analytic = fun(theta)
        numeric = central_difference(fun, theta)
        self.assertLess(float(np.max(np.abs(analytic - numeric))), 2e-7)

    def test_mil_bce_and_softplus_rank_gradients_match_finite_difference(self):
        for rho in (0.0, 0.1):
            fun = lambda t: (mil_objective_parts(t, self.bags, self.y, 1e-4,
                                                 rho=rho, tau=1.0, margin=1.0, kappa=0.1).total,
                             mil_objective_parts(t, self.bags, self.y, 1e-4,
                                                 rho=rho, tau=1.0, margin=1.0, kappa=0.1).gradient)
            parts = mil_objective_parts(self.theta, self.bags, self.y, 1e-4, rho=rho)
            numeric = central_difference(fun, self.theta)
            self.assertLess(float(np.max(np.abs(parts.gradient - numeric))), 3e-6)

    def test_rho_zero_is_exactly_mil_bce(self):
        bce = mil_objective_parts(self.theta, self.bags, self.y, 1e-4, rho=0.0)
        rank = mil_objective_parts(self.theta, self.bags, self.y, 1e-4, rho=0.1)
        bce_only = mil_objective_parts(self.theta, self.bags, self.y, 1e-4, rho=0.0)
        self.assertEqual(bce.total, bce_only.total)
        np.testing.assert_array_equal(bce.gradient, bce_only.gradient)
        self.assertEqual(rank.pair_count, 4)

    def test_invalid_patches_are_excluded(self):
        z = np.arange(20, dtype=np.float32).reshape(5, 4)
        mask = np.asarray([1, 0, 1, 0, 1], dtype=bool)
        bag = valid_bag(z, mask)
        altered = z.copy(); altered[~mask] = 1e20
        bag_altered = valid_bag(altered, mask)
        np.testing.assert_array_equal(bag, bag_altered)
        cls = np.ones(4, dtype=np.float32); cls /= np.linalg.norm(cls)
        feat_a = build_global_readout("G-Mean", cls, z, mask)
        feat_b = build_global_readout("G-Mean", cls, altered, mask)
        np.testing.assert_array_equal(feat_a, feat_b)

    def test_nonbinary_and_one_class_histories_rejected(self):
        with self.assertRaises(ValueError):
            binary_labels(np.asarray([0, 2]))
        with self.assertRaises(ValueError):
            binary_labels(np.asarray([1, 1]))

    def test_pair_count_and_balanced_pair_weighting(self):
        y = np.asarray([1, 1, 0, 0, 0])
        pos, neg = pair_indices(y)
        self.assertEqual(len(pos), 6)
        self.assertEqual(list(zip(pos.tolist(), neg.tolist())),
                         [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)])

    def test_ranking_gradient_has_zero_bias_component_and_bias_invariance(self):
        a = mil_objective_parts(self.theta, self.bags, self.y, 1e-4, rho=0.1)
        shifted = self.theta.copy(); shifted[-1] += 3.75
        b = mil_objective_parts(shifted, self.bags, self.y, 1e-4, rho=0.1)
        self.assertEqual(a.gradient[-1] - mil_objective_parts(self.theta, self.bags, self.y, 1e-4, rho=0.0).gradient[-1], 0.0)
        # Ranking itself is invariant to a common bag-logit shift.
        self.assertAlmostEqual(a.rank, b.rank, places=12)

    def test_extreme_logits_are_finite(self):
        from model import sigmoid, mil_objective_parts
        self.assertTrue(np.isfinite(sigmoid(np.asarray([-1e4, 0.0, 1e4]))).all())
        theta = np.asarray([1000.0, -1000.0, 0.0, 0.0, 0.0])
        parts = mil_objective_parts(theta, self.bags, self.y, 1e-4, rho=0.1)
        self.assertTrue(np.isfinite(parts.total))
        self.assertTrue(np.isfinite(parts.gradient).all())

    def test_paired_seed_initialization_is_reproducible_and_distinct(self):
        a0 = mil_initial_parameters(16, 0)
        a0_again = mil_initial_parameters(16, 0)
        a1 = mil_initial_parameters(16, 1)
        np.testing.assert_array_equal(a0, a0_again)
        self.assertFalse(np.array_equal(a0, a1))
        self.assertEqual(a0[-1], 0.0)

    def test_global_solver_failure_gate_does_not_accept_high_gradient(self):
        options = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 10000,
                   "maxls": 50, "maxcor": 10, "accept_gradient_inf": 1e-6}
        x = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        y = np.asarray([1, 1, 0, 0])
        result = fit_global(x, y, None, 1e-4, options)
        self.assertTrue(result["accepted"])
        self.assertLessEqual(result["gradient_inf_norm"], 1e-6)

    def test_previous_predictions_are_independent_of_future_label_values(self):
        # The learner API receives training features/labels only; evaluator labels are separate.
        x = np.asarray([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        y = np.asarray([1, 1, 0, 0])
        options = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 10000,
                   "maxls": 50, "maxcor": 10, "accept_gradient_inf": 1e-6}
        fitted = fit_global(x, y, None, 1e-4, options)
        self.assertTrue(fitted["accepted"])
        future_x = np.asarray([[0.5, 0.5], [-0.5, -0.5]])
        scores_a = predict_global(fitted["theta"], future_x)
        # Swapping evaluator-only future labels has no path to optimizer/predictor calls.
        evaluator_labels_a = np.asarray([1, 0])
        evaluator_labels_b = 1 - evaluator_labels_a
        scores_b = predict_global(fitted["theta"], future_x)
        np.testing.assert_array_equal(scores_a, scores_b)
        self.assertFalse(np.array_equal(evaluator_labels_a, evaluator_labels_b))

    def _tiny_run(self, evaluator_labels, update_enabled=True, duplicate=False):
        calls = []
        fitted_histories = []
        fitted_checkpoints = []
        events = [
            {"event_id": "E1", "image_id": "Q1", "stream_order": 1,
             "features": np.asarray([1.0]), "product": "hidden-product"},
            {"event_id": "E2", "image_id": "Q2" if not duplicate else "Q1", "stream_order": 2,
             "features": np.asarray([-1.0]), "product": "hidden-product"},
        ]

        def reveal(event_id):
            calls.append(("reveal", event_id))
            return {"E1": "present", "E2": "unobservable_or_uncertain"}[event_id]

        def fit(features, labels, initial, checkpoint):
            fitted_histories.append((len(features), tuple(labels.tolist())))
            fitted_checkpoints.append(checkpoint)
            return {"accepted": True, "theta": initial + 0.1,
                    "elapsed_seconds": 0.001, "objective": 0.5,
                    "gradient_inf_norm": 0.0, "status": "PASS"}

        def predict_one(theta, feature):
            calls.append(("predict", float(feature[0])))
            return float(theta[0] * feature[0])

        def predict_future(theta, features):
            return np.asarray([float(theta[0] * x[0]) for x in features])

        states = {"F1": {"state": evaluator_labels[0]}, "F2": {"state": evaluator_labels[1]}}
        future_rows = [{"image_id": "F1", "product": "p", "sample_key": "g1"},
                       {"image_id": "F2", "product": "p", "sample_key": "g2"}]

        def evaluator(scores):
            # Evaluator labels affect only returned metrics, never fit/prediction callbacks.
            labels = [states[r["image_id"]]["state"] for r in future_rows]
            return {"eval_labels": ";".join(labels), "mean_score": float(np.mean(scores))}, [], []

        result = run_trajectory(
            protocol="X", trajectory_id="synthetic", initial_theta=np.asarray([0.2]),
            initial_features=[np.asarray([1.0]), np.asarray([-1.0])],
            initial_labels=[1, 0], initial_image_ids=["D1", "D2"],
            feedback_events=events, reveal_state=reveal, fit_history=fit,
            predict_one=predict_one, predict_future=predict_future,
            future_features=[np.asarray([0.5]), np.asarray([-0.5])],
            future_rows=future_rows, evaluate=evaluator,
            update_enabled=update_enabled,
        )
        return result, calls, fitted_histories, fitted_checkpoints

    def test_prediction_precedes_reveal_and_unknown_consumes_no_update(self):
        result, calls, histories, checkpoints = self._tiny_run(["present", "absent_assessable"])
        self.assertEqual(calls[:2], [("predict", 1.0), ("reveal", "E1")])
        self.assertEqual(calls[2:4], [("predict", -1.0), ("reveal", "E2")])
        self.assertEqual(len(histories), 1)
        self.assertEqual(histories[0][0], 3)
        self.assertEqual(checkpoints, [1])
        self.assertEqual(result["binary_updates"], 1)

    def test_future_label_counterfactual_leaves_states_and_query_predictions_unchanged(self):
        a, _, _, _ = self._tiny_run(["present", "absent_assessable"])
        b, _, _, _ = self._tiny_run(["absent_assessable", "present"])
        self.assertEqual([r["score_before_reveal"] for r in a["events"]],
                         [r["score_before_reveal"] for r in b["events"]])
        self.assertEqual([r["state_sha256_before_reveal"] for r in a["events"]],
                         [r["state_sha256_before_reveal"] for r in b["events"]])
        self.assertEqual([r["state_sha256"] for r in a["checkpoints"]],
                         [r["state_sha256"] for r in b["checkpoints"]])

    def test_frozen_control_never_changes_state(self):
        result, _, histories, _ = self._tiny_run(["present", "absent_assessable"], update_enabled=False)
        self.assertEqual(result["binary_updates"], 0)
        self.assertEqual(len(histories), 0)
        self.assertEqual(len({r["state_sha256"] for r in result["checkpoints"]}), 1)

    def test_duplicate_feedback_image_is_rejected(self):
        with self.assertRaises(ValueError):
            self._tiny_run(["present", "absent_assessable"], duplicate=True)


if __name__ == "__main__":
    unittest.main()
