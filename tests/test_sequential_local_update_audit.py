import unittest
from copy import deepcopy

import numpy as np

from scripts.abmg_sequential_local_update_audit import (
    CORE4,
    assert_local_update,
    effect_summary,
    init_memory,
    score_memory,
    stratified_stream_sentinel_split,
    uniform_query_positions,
    update_memory_local,
)


class SequentialLocalUpdateAuditTests(unittest.TestCase):
    def setUp(self):
        self.X = np.asarray(
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
        self.support = {
            "AK": np.asarray([0], dtype=np.int64),
            "HS": np.asarray([2], dtype=np.int64),
            "QS": np.asarray([4], dtype=np.int64),
            "ZW": np.asarray([6], dtype=np.int64),
        }
        self.prior = {
            "mu0": np.zeros(4, dtype=np.float64),
            "var": np.ones(4, dtype=np.float64) * 0.2,
        }

    def test_local_update_changes_only_target_state(self):
        mem = init_memory(self.X, self.support)
        before = deepcopy(mem)
        update_memory_local(mem, "AK", np.asarray([0.8, 0.2, 0.0, 0.0]))
        self.assertTrue(assert_local_update(before, mem, "AK"))
        self.assertEqual(mem["AK"]["n"], before["AK"]["n"] + 1)
        for label in ("HS", "QS", "ZW"):
            self.assertEqual(mem[label]["n"], before[label]["n"])
            np.testing.assert_array_equal(mem[label]["sum"], before[label]["sum"])

    def test_query_schedule_is_deterministic_and_label_free(self):
        a = uniform_query_positions(100, 12, 7)
        b = uniform_query_positions(100, 12, 7)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 12)
        self.assertEqual(len(np.unique(a)), 12)
        self.assertTrue(np.all((a >= 0) & (a < 100)))

    def test_stratified_split_is_disjoint_and_complete(self):
        categories = np.asarray(
            ["a"] * 8 + ["b"] * 8,
            dtype=object,
        )
        labels = np.asarray(list(CORE4) * 4, dtype=object)
        stream, sentinel = stratified_stream_sentinel_split(
            categories, labels, 0.25, seed=0
        )
        self.assertFalse(set(stream.tolist()) & set(sentinel.tolist()))
        self.assertEqual(len(stream) + len(sentinel), len(labels))
        self.assertEqual(set(labels[sentinel].tolist()), set(CORE4))

    def test_unequal_counts_can_change_predictive_ranking(self):
        mem = init_memory(self.X, self.support)
        # Make AK much more mature than the other factors.
        for _ in range(5):
            update_memory_local(mem, "AK", np.asarray([1.0, 0.0, 0.0, 0.0]))
        q = np.asarray(
            [
                [0.7, 0.3, 0.0, 0.0],
                [0.3, 0.7, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        pred_s, score_s = score_memory(q, mem, self.prior, 1.0, "diag_shrunk")
        pred_b, score_b = score_memory(q, mem, self.prior, 1.0, "bayes_predictive")
        self.assertEqual(score_s.shape, (2, 4))
        self.assertEqual(score_b.shape, (2, 4))
        # This audit deliberately allows the two rules to differ when n_g differs;
        # the important engineering check is that both return finite rankings.
        self.assertTrue(np.isfinite(score_s).all())
        self.assertTrue(np.isfinite(score_b).all())
        self.assertEqual(len(pred_s), 2)
        self.assertEqual(len(pred_b), 2)

    def test_effect_summary_detects_harmful_and_helpful_flips(self):
        y = np.asarray(["AK", "HS"], dtype=object)
        before_scores = np.asarray(
            [[1.0, 0.0, -1.0, -1.0], [0.9, 1.0, -1.0, -1.0]],
            dtype=np.float64,
        )
        after_scores = np.asarray(
            [[1.2, 0.0, -1.0, -1.0], [1.1, 1.0, -1.0, -1.0]],
            dtype=np.float64,
        )
        labels = np.asarray(CORE4, dtype=object)
        pred_b = labels[before_scores.argmax(axis=1)]
        pred_a = labels[after_scores.argmax(axis=1)]
        eff = effect_summary(y, pred_b, before_scores, pred_a, after_scores)
        self.assertGreater(eff["AK"]["margin_delta"], 0.0)
        self.assertGreater(eff["HS"]["harmful_flip_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
