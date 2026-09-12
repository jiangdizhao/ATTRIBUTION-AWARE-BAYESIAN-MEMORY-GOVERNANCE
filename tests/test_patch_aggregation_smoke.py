import unittest

import torch

from scripts.abmg_patch_aggregation_smoke import parse_k_values, pool_selected_raw_patches


class TestPatchAggregationSmoke(unittest.TestCase):
    def test_parse_k_values_sorted_unique(self):
        self.assertEqual(parse_k_values("16,1,4,4,2,8"), (1, 2, 4, 8, 16))

    def test_k1_uniform_equals_softmax(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        s = torch.tensor([0.1, 0.2, 0.9])
        idx = torch.tensor([2])
        a = pool_selected_raw_patches(x, s, idx, mode="uniform", temperature=20.0)
        b = pool_selected_raw_patches(x, s, idx, mode="score_softmax", temperature=20.0)
        self.assertTrue(torch.allclose(a, b, atol=0.0, rtol=0.0))

    def test_softmax_pooling_moves_toward_high_score_patch(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        s = torch.tensor([0.9, 0.1])
        idx = torch.tensor([0, 1])
        uniform = pool_selected_raw_patches(x, s, idx, mode="uniform", temperature=20.0)
        weighted = pool_selected_raw_patches(x, s, idx, mode="score_softmax", temperature=20.0)
        self.assertTrue(weighted[0] > uniform[0])
        self.assertTrue(weighted[1] < uniform[1])
        self.assertAlmostEqual(float(weighted.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
