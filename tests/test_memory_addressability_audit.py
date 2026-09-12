import unittest

import numpy as np

from scripts.abmg_memory_addressability_audit import (
    CORE4,
    _confusion_true_normalized,
    bayes_shared_diag_predict,
    bayes_uncertainty_diagnostics,
    exemplar_max_predict,
    fit_shared_diag_prior,
)


class TestMemoryAddressabilityAudit(unittest.TestCase):
    def test_exemplar_max_routes_to_matching_factor(self):
        # Four orthogonal factor exemplars, with two examples per factor.
        x = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.1, 0.9],
            ],
            dtype=np.float64,
        )
        support = {
            "AK": np.asarray([0, 1]),
            "HS": np.asarray([2, 3]),
            "QS": np.asarray([4, 5]),
            "ZW": np.asarray([6, 7]),
        }
        q = np.asarray([[0.95, 0.05, 0.0, 0.0], [0.0, 0.0, 0.05, 0.95]])
        pred, scores = exemplar_max_predict(x, q, support, CORE4)
        self.assertEqual(pred.tolist(), ["AK", "ZW"])
        self.assertEqual(scores.shape, (2, 4))

    def test_shared_diag_bayes_predicts_separated_toy_classes(self):
        # Give each class a distinct axis.  Background stats are label-free.
        x = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0], [0.05, 0.95, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.05, 0.95, 0.0],
                [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.05, 0.95],
            ],
            dtype=np.float64,
        )
        support = {
            "AK": np.asarray([0, 1]),
            "HS": np.asarray([2, 3]),
            "QS": np.asarray([4, 5]),
            "ZW": np.asarray([6, 7]),
        }
        prior = fit_shared_diag_prior(x)
        q = np.asarray([[0.98, 0.02, 0.0, 0.0], [0.0, 0.98, 0.02, 0.0]])
        pred, scores, resp = bayes_shared_diag_predict(
            x, q, support, prior, kappa0=1.0, label_order=CORE4
        )
        self.assertEqual(pred.tolist(), ["AK", "HS"])
        self.assertEqual(scores.shape, (2, 4))
        self.assertTrue(np.allclose(resp.sum(axis=1), 1.0, atol=1e-10))

    def test_bayes_uncertainty_is_finite(self):
        y = np.asarray(["AK", "HS", "QS", "ZW"], dtype=object)
        p = np.asarray(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7],
            ],
            dtype=np.float64,
        )
        d = bayes_uncertainty_diagnostics(y, p, CORE4)
        self.assertGreater(d["mean_true_responsibility"], 0.69)
        self.assertTrue(np.isfinite(d["negative_log_likelihood"]))
        self.assertTrue(np.isfinite(d["brier_score"]))

    def test_confusion_rows_sum_to_one_when_each_class_present(self):
        y = np.asarray(["AK", "AK", "HS", "HS", "QS", "QS", "ZW", "ZW"], dtype=object)
        pred = np.asarray(["AK", "HS", "HS", "HS", "QS", "ZW", "ZW", "AK"], dtype=object)
        cm = _confusion_true_normalized(y, pred)
        self.assertEqual(cm.shape, (4, 4))
        self.assertTrue(np.allclose(cm.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
