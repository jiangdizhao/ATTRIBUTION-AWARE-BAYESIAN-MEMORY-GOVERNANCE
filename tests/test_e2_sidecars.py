import sys
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from abmg_e2_sidecars import (
    SidecarTrainConfig,
    build_sidecar,
    coordinate_participation_ratio,
    cross_group_covariance_penalty,
    group_l2_sparsity,
    group_participation_ratio,
)


class TestE2Sidecars(unittest.TestCase):
    def test_corr_config_requires_exact_groups(self):
        cfg = SidecarTrainConfig(address_dim=8, corr_groups=2)
        cfg.validate()
        with self.assertRaises(ValueError):
            SidecarTrainConfig(address_dim=7, corr_groups=2).validate()

    def test_cross_group_penalty_ignores_within_group_correlation(self):
        # Group 0 uses dims 0:2, group 1 uses dims 2:4.
        # Each group has internal structure, but the two groups are arranged
        # to have zero cross-covariance.
        z = torch.tensor(
            [
                [1.0, 1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0, 1.0],
            ]
        )
        penalty = cross_group_covariance_penalty(z, groups=2)
        self.assertLess(float(penalty), 1e-7)

    def test_cross_group_penalty_detects_coupling(self):
        z = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-2.0, -2.0, -2.0, -2.0],
            ]
        )
        penalty = cross_group_covariance_penalty(z, groups=2)
        self.assertGreater(float(penalty), 0.1)

    def test_group_sparsity_prefers_one_active_group(self):
        one = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        two = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        self.assertLess(
            float(group_l2_sparsity(one, groups=2)),
            float(group_l2_sparsity(two, groups=2)),
        )

    def test_participation_ratio(self):
        sparse = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        dense = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        self.assertAlmostEqual(
            coordinate_participation_ratio(sparse), 1.0, places=5
        )
        self.assertAlmostEqual(
            coordinate_participation_ratio(dense), 4.0, places=5
        )
        self.assertAlmostEqual(
            group_participation_ratio(dense, groups=2), 2.0, places=5
        )

    def test_raw_pca_ica_shapes(self):
        torch.manual_seed(0)
        train = torch.randn(80, 12)
        val = torch.randn(20, 12)
        cfg = SidecarTrainConfig(address_dim=4, corr_groups=2, seed=0)
        device = torch.device("cpu")

        raw = build_sidecar("raw", 12, cfg)
        raw.fit(train, val, cfg, device)
        self.assertEqual(tuple(raw.transform(val, device).shape), (20, 12))

        pca = build_sidecar("pca", 12, cfg)
        pca.fit(train, val, cfg, device)
        self.assertEqual(tuple(pca.transform(val, device).shape), (20, 4))

        ica = build_sidecar("ica", 12, cfg)
        ica.fit(train, val, cfg, device)
        self.assertEqual(tuple(ica.transform(val, device).shape), (20, 4))

    def test_all_neural_sidecars_fit_and_transform(self):
        torch.manual_seed(1)
        train = torch.randn(48, 12)
        val = torch.randn(16, 12)
        cfg = SidecarTrainConfig(
            address_dim=4,
            hidden_dim=16,
            bottleneck_dim=8,
            residual_dim=4,
            corr_groups=2,
            batch_size=16,
            epochs=1,
            patience=1,
            seed=1,
        )
        device = torch.device("cpu")
        for name in (
            "beta_tcvae",
            "factor_vae",
            "corrvae",
            "address_residual",
            "sparse_ae",
        ):
            with self.subTest(name=name):
                sidecar = build_sidecar(name, 12, cfg)
                report = sidecar.fit(train, val, cfg, device)
                z = sidecar.transform(val, device)
                self.assertEqual(tuple(z.shape), (16, 4))
                self.assertTrue(torch.isfinite(z).all())
                self.assertEqual(report["output_dim"], 4)


if __name__ == "__main__":
    unittest.main()
