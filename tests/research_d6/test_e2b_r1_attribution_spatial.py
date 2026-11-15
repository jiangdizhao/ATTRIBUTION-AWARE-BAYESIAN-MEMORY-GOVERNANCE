from __future__ import annotations

import unittest

import numpy as np

from scripts.research_d6.p2.e2b_r1_diagnostics import attribution_spatial as audit


class AttributionSpatialSyntheticTests(unittest.TestCase):
    def test_global_four_state_scores_and_shapley_sum(self):
        x = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        theta0 = np.asarray([0.5, -0.25, -0.1])
        theta12 = np.asarray([0.9, -0.4, 0.2])
        params = audit.four_state_parameters(theta0, theta12)
        scores = {key: audit.score_global(value, x) for key, value in params.items()}
        np.testing.assert_allclose(scores["F00"], x @ theta0[:-1] + theta0[-1])
        np.testing.assert_allclose(scores["F11"], x @ theta12[:-1] + theta12[-1])
        metrics = {key: float(np.mean(value)) for key, value in scores.items()}
        parts = audit.shapley_two_factor(metrics["F00"], metrics["F01"],
                                         metrics["F10"], metrics["F11"])
        self.assertAlmostEqual(parts["phi_weight"] + parts["phi_bias"],
                               metrics["F11"] - metrics["F00"], places=14)

    def test_mil_four_state_scores_use_same_lme_scorer(self):
        bags = [np.asarray([[0.0, 0.5], [1.0, -0.2], [0.4, 0.7]]),
                np.asarray([[0.2, 0.1], [-0.3, 1.2]])]
        theta0 = np.asarray([0.3, -0.1, -0.2])
        theta12 = np.asarray([-0.4, 0.6, 0.15])
        params = audit.four_state_parameters(theta0, theta12)
        scores = {key: audit.score_mil(bags, value) for key, value in params.items()}
        for key, theta in params.items():
            expected = np.asarray([audit.mil_spatial_components(b, theta)["score_full"] for b in bags])
            np.testing.assert_allclose(scores[key], expected, rtol=0.0, atol=1e-14)

    def test_shapley_contributions_sum_to_metric_change(self):
        result = audit.shapley_two_factor(0.4, 0.5, 0.65, 0.7)
        self.assertAlmostEqual(result["phi_weight"] + result["phi_bias"], result["total_delta"], places=15)
        self.assertAlmostEqual(result["additivity_residual"], 0.0, places=15)

    def test_intercept_shift_preserves_ap_auc_but_can_change_ba(self):
        items = [
            {"product": "p", "state": "present"},
            {"product": "p", "state": "present"},
            {"product": "p", "state": "absent_assessable"},
            {"product": "p", "state": "absent_assessable"},
        ]
        scores = np.asarray([-0.1, 0.2, -0.2, 0.1])
        shifted = scores - 0.15
        before, _ = audit.metric_summary(items, scores)
        after, _ = audit.metric_summary(items, shifted)
        self.assertAlmostEqual(before["ap_product_macro_paired"], after["ap_product_macro_paired"], places=15)
        self.assertAlmostEqual(before["roc_auc_product_macro"], after["roc_auc_product_macro"], places=15)
        self.assertNotAlmostEqual(before["ba_product_macro"], after["ba_product_macro"], places=12)

    def test_mil_spatial_identity_and_jensen_nonnegative(self):
        bag = np.asarray([[0.1, 1.0], [0.8, -0.2], [-0.3, 0.4]], dtype=np.float64)
        theta = np.asarray([0.8, -0.4, 0.15])
        result = audit.mil_spatial_components(bag, theta, tau=1.0)
        self.assertGreaterEqual(result["delta_spatial"], 0.0)
        self.assertAlmostEqual(result["score_full"], result["score_mean"] + result["delta_spatial"], places=12)

    def test_uniform_patch_scores_have_zero_spatial_correction(self):
        bag = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
        theta = np.asarray([1.0, 0.0, -0.3])
        result = audit.mil_spatial_components(bag, theta)
        self.assertAlmostEqual(result["delta_spatial"], 0.0, places=14)
        self.assertAlmostEqual(result["effective_patch_fraction"], 1.0, places=14)

    def test_valid_patch_mask_excludes_padding(self):
        padded = np.asarray([[0.0, 0.5], [1.0, -0.2], [1e6, -1e6]], dtype=np.float64)
        mask = np.asarray([True, True, False])
        theta = np.asarray([0.3, 0.2, -0.1])
        masked = audit.mil_spatial_components(padded, theta, valid_mask=mask)
        reference = audit.mil_spatial_components(padded[:2], theta)
        self.assertEqual(masked["n_valid_patches"], 2)
        self.assertAlmostEqual(masked["score_full"], reference["score_full"], places=14)
        self.assertAlmostEqual(masked["delta_spatial"], reference["delta_spatial"], places=14)

    def test_missing_checkpoint_12_is_not_synthesized(self):
        self.assertIsNone(audit.select_endpoint_pair(range(10)))
        self.assertEqual(audit.select_endpoint_pair(range(13)), (0, 12))

    def test_score_reconstruction_failure_blocks_track(self):
        with self.assertRaises(audit.IntegrityError):
            audit.require_score_match(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0 + 2e-8]))
        self.assertLessEqual(audit.require_score_match(np.asarray([0.0, 1.0]),
                                                       np.asarray([0.0, 1.0 + 1e-9])), 1e-8)

    def test_cache_identity_mismatch_blocks(self):
        with self.assertRaises(audit.IntegrityError):
            audit.verify_identity({"model": "frozen-model", "weights_sha256": "abc"},
                                  {"model": "different-model", "weights_sha256": "abc"},
                                  "synthetic-cache-item")
        with self.assertRaises(audit.IntegrityError):
            audit.verify_identity({"model": "frozen-model", "weights_sha256": "expected-sha"},
                                  {"model": "frozen-model", "weights_sha256": "different-sha"},
                                  "synthetic-cache-item")


if __name__ == "__main__":
    unittest.main()
