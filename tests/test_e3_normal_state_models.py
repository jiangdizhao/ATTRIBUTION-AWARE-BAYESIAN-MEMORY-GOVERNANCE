import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from abmg_e3_normal_state_models import (
    DiagonalNIG,
    RobustDiagonalNIG,
    build_normal_state,
    fit_hierarchical_nig_prior,
    fit_weak_nig_prior,
)


class TestE3NormalStateModels(unittest.TestCase):
    def setUp(self):
        self.dev = {
            "a": np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [1.1, -0.1], [1.0, 0.05]]
            ),
            "b": np.asarray(
                [[0.0, 1.0], [0.1, 0.9], [-0.1, 1.1], [0.05, 1.0]]
            ),
            "c": np.asarray(
                [[0.7, 0.7], [0.8, 0.6], [0.6, 0.8], [0.72, 0.68]]
            ),
        }
        self.weak = fit_weak_nig_prior(
            self.dev, kappa0=0.1, alpha0=2.5
        )
        self.hier = fit_hierarchical_nig_prior(
            self.dev, alpha0=3.0
        )

    def test_priors_are_valid(self):
        self.weak.validate()
        self.hier.validate()
        self.assertEqual(self.weak.dim, 2)
        self.assertEqual(self.hier.dim, 2)
        self.assertTrue(np.all(self.hier.kappa0 > 0))

    def test_nig_fractional_update_matches_formula(self):
        prior = self.weak
        st = DiagonalNIG(prior)
        x = np.asarray([0.5, 0.25])
        w = 0.5

        k0 = prior.kappa0.copy()
        m0 = prior.mu0.copy()
        b0 = prior.beta0.copy()
        a0 = prior.alpha0.copy()

        st.update(x, w)
        kn = k0 + w
        expected_mu = (k0 * m0 + w * x) / kn
        expected_beta = b0 + 0.5 * (k0 * w / kn) * (x - m0) ** 2

        self.assertTrue(np.allclose(st.kappa, kn))
        self.assertTrue(np.allclose(st.mu, expected_mu))
        self.assertTrue(np.allclose(st.alpha, a0 + 0.5 * w))
        self.assertTrue(np.allclose(st.beta, expected_beta))

    def test_nig_predictive_log_prob_is_finite(self):
        st = DiagonalNIG(self.weak)
        st.update(np.asarray([1.0, 0.0]))
        lp = st.predictive_log_prob(
            np.asarray([[1.0, 0.0], [4.0, 4.0]])
        )
        self.assertEqual(lp.shape, (2,))
        self.assertTrue(np.isfinite(lp).all())
        self.assertGreater(lp[0], lp[1])

    def test_robust_nig_downweights_outlier(self):
        base = RobustDiagonalNIG(
            self.weak,
            robust_df=4.0,
            min_influence=0.05,
        )
        base.update(np.asarray([1.0, 0.0]), robust=False)

        near = base.clone()
        far = base.clone()

        near.update(np.asarray([1.0, 0.02]), robust=True)
        far.update(np.asarray([20.0, 20.0]), robust=True)

        near_w = near.applied_weights[-1]
        far_w = far.applied_weights[-1]
        self.assertGreater(near_w, far_w)
        self.assertGreaterEqual(far_w, 0.05)
        self.assertLessEqual(near_w, 1.0)

    def test_all_models_share_interface(self):
        for name in (
            "deterministic_diag",
            "nig",
            "hierarchical_nig",
            "robust_nig",
        ):
            with self.subTest(name=name):
                st = build_normal_state(
                    name,
                    weak_prior=self.weak,
                    hierarchical_prior=self.hier,
                )
                st.update(np.asarray([1.0, 0.0]), robust=False)
                ref = st.clone()
                st.update(np.asarray([0.95, 0.05]), robust=True)
                lp = st.predictive_log_prob(
                    np.asarray([[1.0, 0.0], [0.9, 0.1]])
                )
                coverage = st.predictive_interval_coverage(
                    np.asarray([[1.0, 0.0], [0.9, 0.1]])
                )
                drift = st.state_drift(ref)
                self.assertTrue(np.isfinite(lp).all())
                self.assertTrue(0.0 <= coverage <= 1.0)
                self.assertIn("mean_l2", drift)
                self.assertIn("predictive_variance_mean", st.uncertainty())


if __name__ == "__main__":
    unittest.main()
