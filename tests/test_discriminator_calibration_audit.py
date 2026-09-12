import unittest

import numpy as np

from scripts.abmg_discriminator_calibration_audit import (
    CORE4,
    _nll_from_scores,
    _softmax_temperature,
    calibration_metrics,
    diag_gaussian_scores,
    fit_temperature,
)


class DiscriminatorCalibrationAuditTests(unittest.TestCase):
    def setUp(self):
        self.X_train = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.1, 0.9],
            ],
            dtype=np.float64,
        )
        self.X_test = np.asarray(
            [
                [0.95, 0.05, 0.0, 0.0],
                [0.05, 0.95, 0.0, 0.0],
                [0.0, 0.05, 0.95, 0.0],
                [0.0, 0.0, 0.05, 0.95],
            ],
            dtype=np.float64,
        )
        self.support = {
            "AK": np.asarray([0, 1], dtype=np.int64),
            "HS": np.asarray([2, 3], dtype=np.int64),
            "QS": np.asarray([4, 5], dtype=np.int64),
            "ZW": np.asarray([6, 7], dtype=np.int64),
        }
        self.shared = {
            "mu0": np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
            "var": np.asarray([0.2, 0.2, 0.2, 0.2], dtype=np.float64),
        }

    def test_equal_shot_shrunk_and_bayes_predictive_have_identical_argmax(self):
        pred_s, score_s = diag_gaussian_scores(
            self.X_train,
            self.X_test,
            self.support,
            self.shared,
            kappa0=1.0,
            posterior_predictive=False,
        )
        pred_b, score_b = diag_gaussian_scores(
            self.X_train,
            self.X_test,
            self.support,
            self.shared,
            kappa0=1.0,
            posterior_predictive=True,
        )
        np.testing.assert_array_equal(pred_s, pred_b)
        # Equal shots imply one common predictive-variance multiplier, so score
        # ordering must be preserved even though absolute scores differ.
        np.testing.assert_array_equal(np.argsort(score_s, axis=1), np.argsort(score_b, axis=1))

    def test_temperature_fit_can_reduce_nll(self):
        labels = np.asarray(["AK", "HS", "QS", "ZW"] * 10, dtype=object)
        # Correct class is usually highest, but logits are intentionally far too
        # large for the occasional ambiguity/mistake.
        scores = np.zeros((40, 4), dtype=np.float64)
        for i, y in enumerate(labels):
            c = CORE4.index(y)
            scores[i, c] = 20.0
            scores[i, (c + 1) % 4] = 18.0 if i % 5 else 22.0
        raw = _nll_from_scores(scores, labels, 1.0, CORE4)
        fit = fit_temperature(scores, labels, CORE4)
        self.assertTrue(fit["optimizer_success"])
        self.assertGreater(fit["temperature"], 1.0)
        self.assertLess(fit["calibration_nll"], raw)

    def test_calibration_metrics_perfect_predictions(self):
        labels = np.asarray(["AK", "HS", "QS", "ZW"], dtype=object)
        p = np.eye(4, dtype=np.float64) * 0.97 + (1.0 - np.eye(4)) * 0.01
        metrics = calibration_metrics(labels, p, CORE4, n_bins=5)
        self.assertLess(metrics["negative_log_likelihood"], 0.05)
        self.assertLess(metrics["brier_score"], 0.01)
        self.assertLess(metrics["ece"], 0.05)
        self.assertGreater(metrics["mean_true_responsibility"], 0.95)

    def test_temperature_does_not_change_argmax(self):
        scores = np.asarray(
            [
                [2.0, 1.0, 0.0, -1.0],
                [0.0, 3.0, 1.0, 2.0],
                [0.2, 0.1, 4.0, 3.0],
            ],
            dtype=np.float64,
        )
        raw = _softmax_temperature(scores, 1.0)
        cal = _softmax_temperature(scores, 17.0)
        np.testing.assert_array_equal(raw.argmax(axis=1), cal.argmax(axis=1))


if __name__ == "__main__":
    unittest.main()
