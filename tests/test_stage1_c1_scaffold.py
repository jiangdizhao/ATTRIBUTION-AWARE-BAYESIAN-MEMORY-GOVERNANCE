import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import abmg_stage1_c1_beta_vae as c1  # noqa: E402


class C1ProtocolTests(unittest.TestCase):
    def test_frozen_folds_cover_each_category_once(self):
        folds = c1.load_folds(REPO_ROOT / "configs" / "realiad_folds_v0.json")
        targets = []
        for fold in range(5):
            source, target = c1.fold_categories(folds, fold)
            self.assertEqual(len(source), 24)
            self.assertEqual(len(target), 6)
            self.assertTrue(set(source).isdisjoint(target))
            targets.extend(target)
        self.assertEqual(len(targets), 30)
        self.assertEqual(len(set(targets)), 30)

    def test_beta_warmup(self):
        self.assertEqual(c1.beta_at_epoch(4.0, 0, 5), 0.0)
        self.assertAlmostEqual(c1.beta_at_epoch(4.0, 1, 5), 0.8)
        self.assertAlmostEqual(c1.beta_at_epoch(4.0, 4, 5), 3.2)
        self.assertEqual(c1.beta_at_epoch(4.0, 5, 5), 4.0)
        self.assertEqual(c1.beta_at_epoch(4.0, 20, 5), 4.0)

    def test_validation_objective_uses_requested_beta(self):
        rec = 100.0
        kl = 10.0
        self.assertEqual(c1.beta_objective(rec, kl, 0.0), 100.0)
        self.assertEqual(c1.beta_objective(rec, kl, 0.8), 108.0)
        self.assertEqual(c1.beta_objective(rec, kl, 4.0), 140.0)

    def test_checkpoint_selection_waits_for_target_beta(self):
        target = 4.0
        warmup = 5
        for epoch in range(5):
            beta_eff = c1.beta_at_epoch(target, epoch, warmup)
            self.assertFalse(
                c1.checkpoint_selection_eligible(target, beta_eff, warmup),
                msg=f"epoch={epoch} beta_eff={beta_eff} must still be warm-up only",
            )
        self.assertTrue(
            c1.checkpoint_selection_eligible(
                target, c1.beta_at_epoch(target, 5, warmup), warmup
            )
        )
        self.assertTrue(c1.checkpoint_selection_eligible(target, target, 0))

    def test_image_split_is_deterministic(self):
        a = c1.image_split("same-image", fold=2, val_fraction=0.1)
        b = c1.image_split("same-image", fold=2, val_fraction=0.1)
        self.assertEqual(a, b)
        self.assertIn(a, {"train", "val"})

    def test_grouping_shapes(self):
        canonical = c1.make_groups(32, "canonical", 0)
        coordinate = c1.make_groups(32, "coordinate", 0)
        random_groups = c1.make_groups(32, "random", 0)
        self.assertEqual(len(canonical), 8)
        self.assertTrue(all(len(g) == 4 for g in canonical))
        self.assertEqual(len(coordinate), 32)
        self.assertTrue(all(len(g) == 1 for g in coordinate))
        self.assertEqual(sorted(x for g in random_groups for x in g), list(range(32)))
        self.assertNotEqual(random_groups, canonical)

    def test_factor_surprise_and_responsibility_shapes(self):
        z = torch.randn(3, 10, 32)
        mean = torch.zeros(32)
        var = torch.ones(32)
        groups = c1.make_groups(32, "canonical", 0)
        patch = c1.factor_patch_surprise(z, mean, var, groups)
        self.assertEqual(tuple(patch.shape), (3, 10, 8))
        image = c1.topmean_factor(patch, 0.2)
        self.assertEqual(tuple(image.shape), (3, 8))
        r = c1.responsibility(image, torch.zeros(8), torch.ones(8), 1.0)
        self.assertEqual(tuple(r.shape), (3, 8))
        self.assertTrue(torch.allclose(r.sum(dim=1), torch.ones(3), atol=1e-6))

    def test_stability_and_centroid_separability_toy_case(self):
        r = np.asarray([
            [0.95, 0.05],
            [0.90, 0.10],
            [0.05, 0.95],
            [0.10, 0.90],
        ])
        labels = ["a", "a", "b", "b"]
        within, between, delta, _, _ = c1.stability_fast(r, labels)
        self.assertGreater(within, between)
        self.assertGreater(delta, 0.0)
        result = c1.source_separability(r, ["i0", "i1", "i2", "i3"], labels, 0.5, 0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
