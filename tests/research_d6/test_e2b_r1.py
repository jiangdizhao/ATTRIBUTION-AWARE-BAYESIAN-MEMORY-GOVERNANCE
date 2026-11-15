from __future__ import annotations

import hashlib
import pathlib
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
E2B = ROOT / "scripts/research_d6/p2/e2b"
R1 = ROOT / "scripts/research_d6/p2/e2b_r1"
import sys
sys.path.insert(0, str(E2B))
sys.path.insert(0, str(R1))

from model import fit_global, global_objective, parameter_digest  # noqa: E402
from pipeline import run_trajectory  # noqa: E402
from r1_core import ExecutionStop, IntegrityError, read_json, run_checkpointed_trajectory, verify_bundle  # noqa: E402


def _fixture():
    rng = np.random.default_rng(19)
    x0 = [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
          np.asarray([-1.0, 0.0]), np.asarray([0.0, -1.0])]
    y0 = [1, 1, 0, 0]
    ids0 = [f"D{i}" for i in range(4)]
    states = ["present", "unobservable_or_uncertain", "absent_assessable",
              "present", "absent_assessable", "present", "absent_assessable",
              "unobservable_or_uncertain", "present", "absent_assessable",
              "present", "absent_assessable"]
    events = [{"event_id": f"E{i:02d}", "image_id": f"Q{i:02d}",
               "stream_order": i, "features": rng.normal(size=2),
               "product": "synthetic", "sample_key": f"g{i}", "view": f"v{i}"}
              for i in range(1, 13)]
    future = [{"image_id": f"F{i:02d}", "product": "synthetic",
               "sample_key": f"f{i}", "view": "v1"} for i in range(24)]
    future_states = {r["image_id"]: ("present" if i % 2 == 0 else "absent_assessable")
                     for i, r in enumerate(future)}
    return x0, y0, ids0, events, states, future, future_states


def _fitter(features, labels, theta, checkpoint):
    # Deterministic synthetic update used to compare event orchestration only.
    target = float(np.sum(labels) - (len(labels) / 2.0))
    fitted = np.asarray(theta, dtype=np.float64).copy()
    fitted[0] += target * 0.001
    return {"theta": fitted, "accepted": True, "status": "PASS",
            "solver_success": True, "solver_status": 0, "finite": True,
            "gradient_inf_norm": 0.0, "objective": 0.25,
            "iterations": 1, "function_evaluations": 1,
            "elapsed_seconds": 0.001}


def _predict_one(theta, feature):
    return float(np.dot(theta, feature))


def _predict_future(theta, features):
    return np.asarray([_predict_one(theta, x) for x in features], dtype=np.float64)


def _eval_factory(future, future_states):
    def evaluate(scores):
        binary = np.asarray([future_states[r["image_id"]] == "present" for r in future])
        pred = np.asarray(scores) >= 0
        items = [{"image_id": r["image_id"], "prediction": int(pred[i]),
                  "score": float(scores[i]), "binary_target_evaluator_only": int(binary[i]),
                  "attribute_state_evaluator_only": future_states[r["image_id"]],
                  "product": r["product"], "sample_key": r["sample_key"], "view": r["view"]}
                 for i, r in enumerate(future)]
        return {"accuracy": float(np.mean(pred == binary))}, [], items
    return evaluate


class E2BR1PersistenceTests(unittest.TestCase):
    def test_synthetic_tape_order_and_checkpoints_match_frozen_pipeline(self):
        x0, y0, ids0, events, states, future, future_states = _fixture()
        reveal_calls = []
        state_by_event = {f"E{i:02d}": states[i - 1] for i in range(1, 13)}
        old_fit_checkpoints = []
        new_fit_checkpoints = []

        def old_fit(features, labels, theta, checkpoint):
            old_fit_checkpoints.append(checkpoint)
            return _fitter(features, labels, theta, checkpoint)

        def new_fit(features, labels, theta, checkpoint):
            new_fit_checkpoints.append(checkpoint)
            return _fitter(features, labels, theta, checkpoint)

        def reveal(event_id):
            reveal_calls.append(event_id)
            return state_by_event[event_id]

        old = run_trajectory(
            protocol="X", trajectory_id="synthetic", initial_theta=np.zeros(2),
            initial_features=x0, initial_labels=y0, initial_image_ids=ids0,
            feedback_events=events, reveal_state=reveal, fit_history=old_fit,
            predict_one=_predict_one, predict_future=_predict_future,
            future_features=[np.asarray([1.0, 0.1]) for _ in future],
            future_rows=future, evaluate=_eval_factory(future, future_states), update_enabled=True,
        )
        self.assertEqual(len(reveal_calls), 12)
        expected_binary_events = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12]
        self.assertEqual(old_fit_checkpoints, expected_binary_events)
        reveal_calls.clear()
        features = [np.asarray([1.0, 0.1]) for _ in future]
        with tempfile.TemporaryDirectory() as td:
            identity = {"code_fingerprint": "synthetic-code", "tape_sha256": "synthetic-tape",
                        "feature_manifest_sha256": "synthetic-features", "trajectory_id": "synthetic",
                        "side": "learner"}
            first = run_checkpointed_trajectory(
                output_dir=pathlib.Path(td) / "trajectory", identity=identity,
                trajectory_id="synthetic", side="learner", protocol="X",
                initial_theta=np.zeros(2), initial_features=x0, initial_labels=y0,
                initial_image_ids=ids0, feedback_events=events, reveal_state=reveal,
                fit_history=new_fit, predict_one=_predict_one, predict_future=_predict_future,
                future_features=features, future_rows=future,
                evaluate=_eval_factory(future, future_states), update_enabled=True,
                stop_after_checkpoint=5,
            )
            self.assertEqual(first["status"], "INTERRUPTED_FOR_TEST")
            resumed = run_checkpointed_trajectory(
                output_dir=pathlib.Path(td) / "trajectory", identity=identity,
                trajectory_id="synthetic", side="learner", protocol="X",
                initial_theta=np.zeros(2), initial_features=x0, initial_labels=y0,
                initial_image_ids=ids0, feedback_events=events, reveal_state=reveal,
                fit_history=new_fit, predict_one=_predict_one, predict_future=_predict_future,
                future_features=features, future_rows=future,
                evaluate=_eval_factory(future, future_states), update_enabled=True,
            )
            self.assertEqual(resumed["status"], "PASS")
            self.assertEqual(reveal_calls, [f"E{i:02d}" for i in range(1, 13)])
            self.assertEqual(new_fit_checkpoints, expected_binary_events)
            for i, cp in enumerate(old["checkpoints"]):
                stored = verify_bundle(pathlib.Path(td) / "trajectory" / f"checkpoint_{i:02d}",
                                       identity, i)
                np.testing.assert_array_equal(stored["_arrays"]["theta"], cp["theta"])
                np.testing.assert_array_equal(stored["_arrays"]["future_scores"], cp["scores"])
                self.assertEqual(stored["metrics"], cp["metrics"])
            self.assertEqual(old["checkpoints"][-1]["state_sha256"],
                             parameter_digest(verify_bundle(pathlib.Path(td) / "trajectory" / "checkpoint_12",
                                                            identity, 12)["_arrays"]["theta"]))

    def test_uncertain_event_does_not_call_fitter_and_b0_is_frozen(self):
        x0, y0, ids0, events, states, future, future_states = _fixture()
        state_by_event = {f"E{i:02d}": states[i - 1] for i in range(1, 13)}
        fit_checkpoints = []

        def fit(*args):
            fit_checkpoints.append(args[-1])
            return _fitter(*args)

        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "b0"
            result = run_checkpointed_trajectory(
                output_dir=out, identity={"code_fingerprint": "test", "side": "control"},
                trajectory_id="b0", side="control", protocol="W",
                initial_theta=np.asarray([0.3, -0.2]), initial_features=x0,
                initial_labels=y0, initial_image_ids=ids0, feedback_events=events,
                reveal_state=lambda event_id: state_by_event[event_id], fit_history=fit,
                predict_one=_predict_one, predict_future=_predict_future,
                future_features=[np.asarray([0.5, -0.5]) for _ in future],
                future_rows=future, evaluate=_eval_factory(future, future_states), update_enabled=False,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(fit_checkpoints, [])
            initial_sha = None
            for i in range(13):
                record = verify_bundle(out / f"checkpoint_{i:02d}",
                                       {"code_fingerprint": "test", "side": "control"}, i)
                initial_sha = initial_sha or record["theta_sha256"]
                self.assertEqual(record["theta_sha256"], initial_sha)
                self.assertEqual(record["binary_updates"], 0)

    def test_model_reuse_is_deterministic_for_identical_synthetic_inputs(self):
        x = np.asarray([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        y = np.asarray([1, 1, 0, 0])
        start = np.zeros(3)
        opts = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 10000,
                "maxls": 50, "maxcor": 10, "accept_gradient_inf": 1e-6}
        a = fit_global(x, y, start, 1e-4, opts)
        b = fit_global(x, y, start, 1e-4, opts)
        self.assertTrue(a["accepted"] and b["accepted"])
        np.testing.assert_array_equal(a["theta"], b["theta"])
        self.assertEqual(a["objective"], b["objective"])
        np.testing.assert_array_equal(a["theta"], b["theta"])

    def test_paired_mil_losses_share_the_exact_seed_initialization(self):
        from model import mil_initial_parameters
        for dimension in (16, 1024):
            for seed in (0, 1):
                mbce_start = mil_initial_parameters(dimension, seed)
                rank_start = mil_initial_parameters(dimension, seed)
                np.testing.assert_array_equal(mbce_start, rank_start)
                self.assertEqual(hashlib.sha256(mbce_start.tobytes()).hexdigest(),
                                 hashlib.sha256(rank_start.tobytes()).hexdigest())

    def test_future_label_counterfactual_does_not_change_learner_state(self):
        x0, y0, ids0, events, states, future, truth_a = _fixture()
        truth_b = dict(truth_a)
        for key, value in truth_b.items():
            truth_b[key] = "absent_assessable" if value == "present" else "present"
        feedback = {f"E{i:02d}": states[i - 1] for i in range(1, 13)}
        with tempfile.TemporaryDirectory() as td:
            outputs = []
            for name, truth in (("a", truth_a), ("b", truth_b)):
                run_checkpointed_trajectory(
                    output_dir=pathlib.Path(td) / name,
                    identity={"code_fingerprint": "same", "tape_sha256": "same"},
                    trajectory_id=name, side="learner", protocol="X",
                    initial_theta=np.zeros(2), initial_features=x0, initial_labels=y0,
                    initial_image_ids=ids0, feedback_events=events,
                    reveal_state=lambda event_id: feedback[event_id], fit_history=_fitter,
                    predict_one=_predict_one, predict_future=_predict_future,
                    future_features=[np.asarray([1.0, 0.1]) for _ in future],
                    future_rows=future, evaluate=_eval_factory(future, truth), update_enabled=True,
                )
                outputs.append([verify_bundle(pathlib.Path(td) / name / f"checkpoint_{i:02d}")
                                for i in range(13)])
            for a, b in zip(*outputs):
                self.assertEqual(a["theta_sha256"], b["theta_sha256"])
                np.testing.assert_array_equal(a["_arrays"]["theta"], b["_arrays"]["theta"])
                np.testing.assert_array_equal(a["_arrays"]["future_scores"], b["_arrays"]["future_scores"])

    def test_checkpoint_corruption_and_wrong_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "checkpoint_00"
            identity = {"code_fingerprint": "x"}
            from r1_core import write_bundle
            write_bundle(root, {"identity": identity, "checkpoint": 0},
                         {"theta": np.asarray([1.0]), "future_scores": np.asarray([0.5])})
            with self.assertRaises(IntegrityError):
                verify_bundle(root, {"code_fingerprint": "different"}, 0)
            with (root / "state.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(IntegrityError):
                verify_bundle(root, identity, 0)

    def test_revealed_pending_event_is_reused_after_interruption(self):
        x0, y0, ids0, events, states, future, future_states = _fixture()
        state_by_event = {f"E{i:02d}": states[i - 1] for i in range(1, 13)}
        reveal_calls = []

        def reveal(event_id):
            reveal_calls.append(event_id)
            return state_by_event[event_id]

        def stop_before_third_fit(stage):
            if stage == "before_fit_event_3":
                raise ExecutionStop("synthetic pause after durable reveal")

        identity = {"code_fingerprint": "synthetic-code", "tape_sha256": "synthetic-tape"}
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "trajectory"
            args = dict(output_dir=out, identity=identity, trajectory_id="synthetic",
                side="learner", protocol="X", initial_theta=np.zeros(2),
                initial_features=x0, initial_labels=y0, initial_image_ids=ids0,
                feedback_events=events, reveal_state=reveal, fit_history=_fitter,
                predict_one=_predict_one, predict_future=_predict_future,
                future_features=[np.asarray([1.0, 0.1]) for _ in future],
                future_rows=future, evaluate=_eval_factory(future, future_states),
                update_enabled=True)
            with self.assertRaises(ExecutionStop):
                run_checkpointed_trajectory(**args, resource_check=stop_before_third_fit)
            self.assertEqual(reveal_calls, ["E01", "E02", "E03"])
            pending = read_json(out / "events" / "event_03.json")
            self.assertEqual(pending["revealed_state"], "absent_assessable")
            resumed = run_checkpointed_trajectory(**args)
            self.assertEqual(resumed["status"], "PASS")
            self.assertEqual(reveal_calls, [f"E{i:02d}" for i in range(1, 13)])


if __name__ == "__main__":
    unittest.main()
