from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P1B = ROOT / "scripts/research_d6/p1b"
sys.path.insert(0, str(P1B))

from core import (  # noqa: E402
    FeedbackItem, FutureItem, FrozenLinearPredictor, LinearPredictor,
    RecursiveRidgeLearner, SGDReplayLearner, binary_target, run_prequential,
)
from run_experiment import (  # noqa: E402
    build_items, build_role_items, capture_preflight, first_annotation_states,
    load_human_labels,
)


class TraceLearner(LinearPredictor):
    family = "test_trace"

    def __init__(self):
        self.w = np.zeros(3, dtype=np.float64)
        self.updates: list[tuple[str, int]] = []
        self.calls: list[str] = []

    @property
    def threshold(self):
        return 0.0

    def predict_scores(self, x):
        values = np.asarray(x, dtype=np.float64)
        self.calls.append(f"predict:{len(values)}")
        return values @ self.w[:2] + self.w[-1]

    def observe(self, x, y, event_key):
        self.calls.append(f"observe:{event_key}")
        self.updates.append((event_key, int(y)))
        self.w[:2] += (1 if y else -1) * np.asarray(x, dtype=np.float64).reshape(-1) * 0.25
        self.w[-1] += (1 if y else -1) * 0.1
        return 0.0

    def state_digest(self):
        from core import array_digest
        return array_digest(self.w)

    def state_payload_bytes(self):
        return self.w.nbytes

    def export_state(self):
        return {"w": self.w.copy()}


def tiny_stream(feedback_state="present", future_state="present"):
    event = FeedbackItem("E1", "I-Q", "p", "group-q", 1, feedback_state,
                         np.asarray([1, 0], dtype=np.float32))
    future = [
        FutureItem("I-F1", "p", "group-f1", future_state,
                   np.asarray([1, 0], dtype=np.float32)),
        FutureItem("I-F2", "p", "group-f2", "absent_assessable",
                   np.asarray([-1, 0], dtype=np.float32)),
    ]
    return [event], future


class P1BTests(unittest.TestCase):
    def test_unknown_and_not_applicable_are_distinct_nonbinary_states(self):
        self.assertIsNone(binary_target("unobservable_or_uncertain"))
        self.assertIsNone(binary_target("not_applicable"))
        self.assertNotEqual("unobservable_or_uncertain", "not_applicable")
        self.assertEqual(binary_target("present"), 1)
        self.assertEqual(binary_target("absent_assessable"), 0)

    def test_query_prediction_occurs_before_feedback_reveal(self):
        events, future = tiny_stream("present", "present")
        learner = TraceLearner()
        def reveal(_event_id):
            self.assertEqual(learner.calls[-1], "predict:1")
            learner.calls.append("reveal")
            return "present"
        _, _, _, _, event_rows = run_prequential(
            learner, events, future, reveal, protocol="X", method="test",
            config_id="c", seed=0, pair_id="p",
        )
        self.assertEqual(event_rows[0]["prediction_timing"], "before_reveal")
        self.assertLess(learner.calls.index("predict:1"), learner.calls.index("reveal"))
        self.assertLess(learner.calls.index("reveal"), learner.calls.index("observe:E1"))

    def test_unqueried_future_truth_cannot_change_query_prediction_or_state(self):
        events_a, future_a = tiny_stream("present", "present")
        events_b, future_b = tiny_stream("present", "absent_assessable")
        outputs = []
        for events, future in ((events_a, future_a), (events_b, future_b)):
            learner = TraceLearner()
            outputs.append(run_prequential(
                learner, events, future, lambda _eid: "present", protocol="X",
                method="test", config_id="c", seed=0, pair_id="p",
            ))
        ev_a, ev_b = outputs[0][4], outputs[1][4]
        self.assertEqual(ev_a[0]["pre_reveal_score"], ev_b[0]["pre_reveal_score"])
        self.assertEqual(ev_a[0]["state_sha256_before"], ev_b[0]["state_sha256_before"])
        self.assertEqual(ev_a[0]["state_sha256_after"], ev_b[0]["state_sha256_after"])
        self.assertEqual(outputs[0][1][1]["state_sha256"], outputs[1][1][1]["state_sha256"])

    def test_feedback_counterfactual_changes_state_only_after_reveal(self):
        outcomes = []
        for state in ("present", "absent_assessable"):
            events, future = tiny_stream(state, "present")
            learner = TraceLearner()
            outcomes.append(run_prequential(
                learner, events, future, lambda _eid, s=state: s, protocol="X",
                method="test", config_id="c", seed=0, pair_id="p",
            ))
        a, b = outcomes[0][4][0], outcomes[1][4][0]
        self.assertEqual(a["pre_reveal_score"], b["pre_reveal_score"])
        self.assertEqual(a["state_sha256_before"], b["state_sha256_before"])
        self.assertNotEqual(a["state_sha256_after"], b["state_sha256_after"])

    def test_unknown_and_na_feedback_consume_event_without_update(self):
        for state in ("unobservable_or_uncertain", "not_applicable"):
            events, future = tiny_stream(state, "present")
            learner = TraceLearner()
            _, _, _, _, rows = run_prequential(
                learner, events, future, lambda _eid, s=state: s, protocol="W",
                method="test", config_id="c", seed=0, pair_id="p",
            )
            event = rows[0]
            self.assertFalse(event["update_applied"])
            self.assertEqual(event["state_sha256_before"], event["state_sha256_after"])
            self.assertEqual(learner.updates, [])

    def test_frozen_b0_never_changes(self):
        frozen = FrozenLinearPredictor(np.asarray([0.3, -0.2]), 0.1, 0.0,
                                       "test", "config")
        before = frozen.state_digest()
        events, future = tiny_stream("present", "present")
        _, _, _, _, rows = run_prequential(
            frozen, events, future, lambda _eid: "present", protocol="X",
            method="B0", config_id="c", seed=0, pair_id="p",
        )
        self.assertEqual(before, frozen.state_digest())
        self.assertEqual(rows[0]["state_sha256_before"], rows[0]["state_sha256_after"])
        self.assertEqual(rows[0]["update_applied"], False)

    def test_repeat_presentations_collapse_to_first_item_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "queue.csv").write_text(
                "presentation_id,presentation_order,item_id\nP1,1,I1\nP2,2,I1\nP3,3,I2\n",
                encoding="utf-8",
            )
            (root / "annotations.csv").write_text(
                "presentation_id,item_id,state\nP1,I1,present\nP2,I1,absent_assessable\nP3,I2,not_applicable\n",
                encoding="utf-8",
            )
            states, counts = first_annotation_states(root / "queue.csv", root / "annotations.csv")
        self.assertEqual(set(states), {"I1", "I2"})
        self.assertEqual(states["I1"]["state"], "present")
        self.assertEqual(states["I1"]["presentations_for_item"], 2)
        self.assertEqual(counts["I1"], 2)
        self.assertEqual(binary_target(states["I2"]["state"]), None)

    def test_learner_api_accepts_no_product_or_source_fields(self):
        predict_params = list(inspect.signature(LinearPredictor.predict_scores).parameters)
        observe_params = list(inspect.signature(LinearPredictor.observe).parameters)
        forbidden = {"product", "sample_key", "view", "source_type", "defect_type", "mask"}
        self.assertEqual(predict_params, ["self", "x"])
        self.assertTrue(forbidden.isdisjoint(observe_params))
        self.assertEqual(set(observe_params), {"self", "x", "y", "event_key"})

    def test_recursive_ridge_matches_direct_batch_refit(self):
        rng = np.random.default_rng(41)
        x0 = rng.normal(size=(12, 7))
        y0 = np.asarray([0, 1] * 6, dtype=np.float64)
        new_x = rng.normal(size=(4, 7))
        new_y = np.asarray([1, 0, 1, 1], dtype=np.float64)
        lam = 0.7
        learner = RecursiveRidgeLearner(lam)
        learner.initialize(x0, y0)
        for n, (x, y) in enumerate(zip(new_x, new_y)):
            learner.observe(x, int(y), f"e{n}")
        all_x = np.concatenate([x0, new_x])
        all_y = np.concatenate([y0, new_y])
        z = np.column_stack([all_x, np.ones(len(all_x))])
        gram = z.T @ z + lam * np.eye(z.shape[1])
        direct_w = np.linalg.solve(gram, z.T @ all_y)
        direct_P = np.linalg.solve(gram, np.eye(gram.shape[0]))
        np.testing.assert_allclose(learner.w, direct_w, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(learner.P, direct_P, atol=1e-10, rtol=1e-9)

    def test_same_sgd_settings_and_tape_are_deterministic(self):
        rng = np.random.default_rng(8)
        x_fit = rng.normal(size=(16, 2)).astype(np.float32)
        y_fit = np.asarray([0, 1] * 8, dtype=np.int64)
        events, future = tiny_stream("present", "present")
        outputs = []
        for _ in range(2):
            learner = SGDReplayLearner(alpha=1e-3, seed=0, eta0=0.01, replay_capacity=12)
            learner.initialize(x_fit, y_fit)
            outputs.append(run_prequential(
                learner, events, future, lambda _eid: "present", protocol="X",
                method="B1", config_id="fixed", seed=0, pair_id="fixed",
            ))
        np.testing.assert_array_equal(
            [r["score"] for r in outputs[0][0]], [r["score"] for r in outputs[1][0]]
        )
        self.assertEqual(
            [r["state_sha256"] for r in outputs[0][1]],
            [r["state_sha256"] for r in outputs[1][1]],
        )

    def test_registered_wx_tape_preflight_has_separate_feature_roles(self):
        audit = capture_preflight()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual((audit["n_dfit"], audit["n_ddev"], audit["n_total"]), (54, 48, 102))
        self.assertEqual((audit["w_query_ids"], audit["w_future_ids"]), (12, 24))
        self.assertEqual((audit["x_query_ids"], audit["x_future_ids"]), (12, 24))

    def test_exact_registered_x_tape_replay_is_repeatable(self):
        integrity = capture_preflight()
        labels, _ = load_human_labels(integrity)
        features, _ = build_items(integrity, labels)
        fit_ids = [r["image_id"] for r in integrity["input_rows"]
                   if r["data_role"] == "D-fit" and binary_target(labels[r["image_id"]]["state"]) is not None]
        x_fit = np.stack([features[i] for i in fit_ids]).astype(np.float32)
        y_fit = np.asarray([binary_target(labels[i]["state"]) for i in fit_ids], dtype=np.int64)
        events, future = build_role_items("X", integrity["tape_rows"], features, labels)
        outputs = []
        for _ in range(2):
            learner = SGDReplayLearner(alpha=1e-4, seed=0, eta0=0.01, replay_capacity=12)
            learner.initialize(x_fit, y_fit)
            outputs.append(run_prequential(
                learner, events, future, lambda event_id: next(e.state for e in events if e.event_id == event_id),
                protocol="X", method="B1-SGD-Replay", config_id="registered", seed=0,
                pair_id="registered",
            ))
        p0, m0, _, _, e0 = outputs[0]
        p1, m1, _, _, e1 = outputs[1]
        self.assertEqual([(r["role"], r["image_id"], r["checkpoint"], r["prediction"])
                          for r in p0],
                         [(r["role"], r["image_id"], r["checkpoint"], r["prediction"])
                          for r in p1])
        np.testing.assert_array_equal([r["score"] for r in p0], [r["score"] for r in p1])
        self.assertEqual([r["state_sha256"] for r in m0], [r["state_sha256"] for r in m1])
        self.assertEqual([r["ba_product_macro"] for r in m0], [r["ba_product_macro"] for r in m1])
        self.assertEqual([r["ap_product_macro_including_degenerate"] for r in m0],
                         [r["ap_product_macro_including_degenerate"] for r in m1])
        self.assertEqual([r["state_sha256_after"] for r in e0], [r["state_sha256_after"] for r in e1])

    def test_zz_assets_unchanged_by_validation_and_synthetic_runs(self):
        before = capture_preflight()["snapshot_sha256"]
        events, future = tiny_stream("present", "present")
        learner = TraceLearner()
        run_prequential(learner, events, future, lambda _eid: "present", protocol="X",
                        method="test", config_id="c", seed=0, pair_id="p")
        after = capture_preflight()["snapshot_sha256"]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
