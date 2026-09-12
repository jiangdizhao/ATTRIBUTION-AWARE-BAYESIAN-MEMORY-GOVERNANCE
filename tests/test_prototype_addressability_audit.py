import unittest

import numpy as np

from scripts.abmg_prototype_addressability_audit import (
    CORE4,
    best_cluster_label_mapping,
    category_diverse_support_indices,
    prototype_predict,
    routing_diagnostics,
    within_category_label_shuffle,
)


class TestPrototypeAddressabilityAudit(unittest.TestCase):
    def test_category_diverse_support_selection_uses_distinct_categories_when_possible(self):
        labels = np.asarray(["AK"] * 6 + ["HS"] * 6, dtype=object)
        categories = np.asarray(
            ["a", "a", "b", "b", "c", "c"] * 2, dtype=object
        )
        out = category_diverse_support_indices(
            labels, categories, ("AK", "HS"), shots_per_label=3, seed=7
        )
        self.assertEqual(len(out["AK"]), 3)
        self.assertEqual(len(out["HS"]), 3)
        self.assertEqual(len(set(categories[out["AK"]].tolist())), 3)
        self.assertEqual(len(set(categories[out["HS"]].tolist())), 3)

    def test_within_category_shuffle_preserves_each_category_label_multiset(self):
        labels = np.asarray(["AK", "AK", "HS", "QS", "AK", "ZW"], dtype=object)
        categories = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
        shuffled = within_category_label_shuffle(labels, categories, seed=3)
        for cat in ("a", "b"):
            idx = np.flatnonzero(categories == cat)
            self.assertEqual(
                sorted(labels[idx].tolist()), sorted(shuffled[idx].tolist())
            )

    def test_cosine_prototype_routing_on_separable_toy_data(self):
        X_train = np.asarray(
            [[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64
        )
        support = {label: np.asarray([i]) for i, label in enumerate(CORE4)}
        X_test = np.asarray(
            [[2, 0.1], [0.1, 2], [-2, 0.1], [0.1, -2]], dtype=np.float64
        )
        pred, sims = prototype_predict(X_train, X_test, support, CORE4)
        self.assertEqual(pred.tolist(), list(CORE4))
        diag = routing_diagnostics(np.asarray(CORE4, dtype=object), sims, CORE4)
        self.assertEqual(diag["top2_recall"], 1.0)
        self.assertGreater(diag["mean_true_minus_best_wrong_margin"], 0.0)

    def test_cluster_mapping_recovers_permuted_cluster_ids(self):
        labels = np.asarray(["AK", "AK", "HS", "HS", "QS", "QS", "ZW", "ZW"], dtype=object)
        clusters = np.asarray([2, 2, 0, 0, 3, 3, 1, 1], dtype=np.int64)
        mapping = best_cluster_label_mapping(clusters, labels, CORE4)
        self.assertEqual(mapping[2], "AK")
        self.assertEqual(mapping[0], "HS")
        self.assertEqual(mapping[3], "QS")
        self.assertEqual(mapping[1], "ZW")


if __name__ == "__main__":
    unittest.main()
