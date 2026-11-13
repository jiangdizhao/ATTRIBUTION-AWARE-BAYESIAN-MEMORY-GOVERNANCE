"""CPU-only learner and prequential evaluation primitives for LP-D6-P1-B.

This module deliberately accepts feature arrays and revealed binary labels only.
Product, group, source-type, annotation-state, and evaluator labels stay in the
runner/evaluator layer and are never model inputs.
"""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score


VALID_STATES = {
    "present": 1,
    "absent_assessable": 0,
    "unobservable_or_uncertain": None,
    "not_applicable": None,
}


def binary_target(state: str) -> int | None:
    """Map only assessable states to binary targets; unknown/N-A stay masked."""
    if state not in VALID_STATES:
        raise ValueError(f"invalid attribute state: {state!r}")
    return VALID_STATES[state]


def array_digest(*arrays: np.ndarray, extra: bytes = b"") -> str:
    h = hashlib.sha256(extra)
    for arr in arrays:
        value = np.ascontiguousarray(arr)
        h.update(str(value.dtype).encode("ascii"))
        h.update(repr(value.shape).encode("ascii"))
        h.update(value.tobytes())
    return h.hexdigest()


class LinearPredictor:
    """Small common interface; predict receives features, observe receives a paid label."""

    family = "abstract"
    update_enabled = True

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def observe(self, x: np.ndarray, y: int, event_key: str) -> float:
        raise NotImplementedError

    def state_digest(self) -> str:
        raise NotImplementedError

    def state_payload_bytes(self) -> int:
        raise NotImplementedError

    def export_state(self) -> dict[str, np.ndarray]:
        raise NotImplementedError


class FrozenLinearPredictor(LinearPredictor):
    """Immutable paired B0 linear head."""

    family = "frozen"
    update_enabled = False

    def __init__(self, coef: np.ndarray, intercept: float, threshold: float,
                 source_family: str, source_config: str):
        self.coef = np.asarray(coef, dtype=np.float64).reshape(-1).copy()
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        self.source_family = source_family
        self.source_config = source_config
        self.coef.setflags(write=False)

    @classmethod
    def from_linear(cls, coef: np.ndarray, intercept: float, threshold: float,
                    source_family: str, source_config: str) -> "FrozenLinearPredictor":
        return cls(coef, intercept, threshold, source_family, source_config)

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        return values @ self.coef + self.intercept

    def observe(self, x: np.ndarray, y: int, event_key: str) -> float:
        raise RuntimeError("B0 no-update state cannot accept feedback")

    def state_digest(self) -> str:
        return array_digest(self.coef, np.asarray([self.intercept, self.threshold]))

    def state_payload_bytes(self) -> int:
        return int(self.coef.nbytes + 16)

    def export_state(self) -> dict[str, np.ndarray]:
        return {
            "coef": self.coef.copy(),
            "intercept": np.asarray([self.intercept], dtype=np.float64),
            "threshold": np.asarray([self.threshold], dtype=np.float64),
        }


class SGDReplayLearner(LinearPredictor):
    """One-pass deterministic SGD logistic learner with bounded paid replay."""

    family = "sgd_logistic_replay"

    def __init__(self, alpha: float, seed: int, eta0: float = 0.01,
                 replay_capacity: int = 12):
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.eta0 = float(eta0)
        self.replay_capacity = int(replay_capacity)
        self.model = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=self.alpha,
            learning_rate="constant", eta0=self.eta0,
            fit_intercept=True, shuffle=False, tol=None,
            random_state=self.seed, average=False,
        )
        self.buffer: list[tuple[str, np.ndarray, int]] = []
        self.seen_event_keys: set[str] = set()
        self._initialized = False

    def initialize(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or len(x) != len(y) or set(np.unique(y)) != {0, 1}:
            raise ValueError("SGD initialization needs a 2D matrix with both binary classes")
        order = np.random.default_rng(self.seed).permutation(len(y))
        for n, idx in enumerate(order):
            kwargs = {"classes": np.asarray([0, 1], dtype=np.int64)} if n == 0 else {}
            self.model.partial_fit(x[idx:idx + 1], y[idx:idx + 1], **kwargs)
        self._initialized = True

    @property
    def threshold(self) -> float:
        return 0.0

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("SGD learner has not been initialized")
        values = np.asarray(x, dtype=np.float32)
        return np.asarray(self.model.decision_function(values), dtype=np.float64).reshape(-1)

    def observe(self, x: np.ndarray, y: int, event_key: str) -> float:
        if not self._initialized:
            raise RuntimeError("SGD learner has not been initialized")
        if y not in (0, 1):
            raise ValueError("only a revealed binary label may update SGD")
        if event_key in self.seen_event_keys:
            raise ValueError("a feedback event may be inserted only once")
        start = time.perf_counter()
        self.seen_event_keys.add(event_key)
        self.buffer.append((event_key, np.asarray(x, dtype=np.float32).reshape(-1).copy(), int(y)))
        if len(self.buffer) > self.replay_capacity:
            self.buffer.pop(0)
        # The list is paid-event insertion order. One ordered pass follows every valid reveal.
        for _, bx, by in self.buffer:
            self.model.partial_fit(bx.reshape(1, -1), np.asarray([by], dtype=np.int64))
        return time.perf_counter() - start

    def state_digest(self) -> str:
        extra = b"".join(k.encode("utf-8") + b"\0" + bytes([y]) for k, _, y in self.buffer)
        arrays = [np.asarray(self.model.coef_), np.asarray(self.model.intercept_)]
        arrays.extend(x for _, x, _ in self.buffer)
        return array_digest(*arrays, extra=extra)

    def state_payload_bytes(self) -> int:
        return int(self.model.coef_.nbytes + self.model.intercept_.nbytes +
                   sum(x.nbytes + 1 + len(k.encode("utf-8")) for k, x, _ in self.buffer))

    def export_state(self) -> dict[str, np.ndarray]:
        state = {
            "coef": np.asarray(self.model.coef_, dtype=np.float64).copy(),
            "intercept": np.asarray(self.model.intercept_, dtype=np.float64).copy(),
            "buffer_X": (np.stack([x for _, x, _ in self.buffer]).astype(np.float32)
                         if self.buffer else np.empty((0, self.model.coef_.shape[1]), dtype=np.float32)),
            "buffer_y": np.asarray([y for _, _, y in self.buffer], dtype=np.int8),
            "buffer_event_keys_utf8_hex": np.asarray(
                [k.encode("utf-8").hex() for k, _, _ in self.buffer], dtype="U"
            ),
        }
        return state


class LogisticRefitLearner(LinearPredictor):
    """Full-history deterministic LBFGS logistic refit after each valid reveal."""

    family = "logistic_full_history_refit"

    def __init__(self, alpha: float, max_iter: int = 10000, tol: float = 1e-10):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.model: LogisticRegression | None = None
        self.base_x: np.ndarray | None = None
        self.base_y: np.ndarray | None = None
        self.paid_x: list[np.ndarray] = []
        self.paid_y: list[int] = []
        self.paid_keys: list[str] = []
        self.n_iter = 0
        self.converged = True

    def _refit(self) -> float:
        assert self.base_x is not None and self.base_y is not None
        if self.paid_x:
            x = np.concatenate([self.base_x, np.stack(self.paid_x).astype(np.float32)], axis=0)
            y = np.concatenate([self.base_y, np.asarray(self.paid_y, dtype=np.int64)])
        else:
            x, y = self.base_x, self.base_y
        # Under mean log-loss + alpha/2*||w||^2, C = 1/(alpha*n).
        # Intercept is unpenalized by lbfgs; this is documented as an objective mapping,
        # not an assertion that the two optimizers yield numerically identical solutions.
        c_value = 1.0 / (self.alpha * len(y))
        model = LogisticRegression(
            penalty="l2", C=c_value, solver="lbfgs", fit_intercept=True,
            max_iter=self.max_iter, tol=self.tol, random_state=0,
        )
        start = time.perf_counter()
        model.fit(x, y)
        elapsed = time.perf_counter() - start
        self.model = model
        self.n_iter = int(np.max(model.n_iter_))
        self.converged = self.n_iter < self.max_iter
        return elapsed

    def initialize(self, x: np.ndarray, y: np.ndarray) -> float:
        self.base_x = np.asarray(x, dtype=np.float32).copy()
        self.base_y = np.asarray(y, dtype=np.int64).copy()
        if set(np.unique(self.base_y)) != {0, 1}:
            raise ValueError("Logistic refit initialization needs both binary classes")
        return self._refit()

    @property
    def threshold(self) -> float:
        return 0.0

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Logistic refit learner has not been initialized")
        return np.asarray(self.model.decision_function(np.asarray(x, dtype=np.float32)), dtype=np.float64).reshape(-1)

    def observe(self, x: np.ndarray, y: int, event_key: str) -> float:
        if self.model is None:
            raise RuntimeError("Logistic refit learner has not been initialized")
        if y not in (0, 1):
            raise ValueError("only a revealed binary label may update full-history logistic")
        if event_key in self.paid_keys:
            raise ValueError("a feedback event may be inserted only once")
        self.paid_keys.append(event_key)
        self.paid_x.append(np.asarray(x, dtype=np.float32).reshape(-1).copy())
        self.paid_y.append(int(y))
        return self._refit()

    def state_digest(self) -> str:
        if self.model is None:
            raise RuntimeError("Logistic refit learner has not been initialized")
        arrays = [self.model.coef_, self.model.intercept_]
        if self.paid_x:
            arrays += [np.stack(self.paid_x), np.asarray(self.paid_y, dtype=np.int8)]
        extra = b"\0".join(k.encode("utf-8") for k in self.paid_keys)
        return array_digest(*arrays, extra=extra)

    def state_payload_bytes(self) -> int:
        if self.model is None or self.base_x is None or self.base_y is None:
            return 0
        return int(self.model.coef_.nbytes + self.model.intercept_.nbytes +
                   self.base_x.nbytes + self.base_y.nbytes +
                   sum(x.nbytes + 1 + len(k.encode("utf-8"))
                       for x, k in zip(self.paid_x, self.paid_keys)))

    def export_state(self) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Logistic refit learner has not been initialized")
        assert self.base_x is not None and self.base_y is not None
        return {
            "coef": np.asarray(self.model.coef_, dtype=np.float64).copy(),
            "intercept": np.asarray(self.model.intercept_, dtype=np.float64).copy(),
            "base_X": self.base_x.copy(),
            "base_y": self.base_y.astype(np.int8, copy=True),
            "paid_X": (np.stack(self.paid_x).astype(np.float32)
                       if self.paid_x else np.empty((0, self.base_x.shape[1]), dtype=np.float32)),
            "paid_y": np.asarray(self.paid_y, dtype=np.int8),
            "paid_event_keys_utf8_hex": np.asarray(
                [k.encode("utf-8").hex() for k in self.paid_keys], dtype="U"
            ),
        }


class RecursiveRidgeLearner(LinearPredictor):
    """Exact Sherman-Morrison recursive least-squares head with penalized bias."""

    family = "recursive_ridge"

    def __init__(self, regularization: float):
        self.regularization = float(regularization)
        self.P: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.seen_event_keys: set[str] = set()
        self.n_features: int | None = None

    def initialize(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("ridge initialization shape mismatch")
        z = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
        gram = z.T @ z + self.regularization * np.eye(z.shape[1], dtype=np.float64)
        self.P = np.linalg.solve(gram, np.eye(gram.shape[0], dtype=np.float64))
        self.w = self.P @ (z.T @ y)
        self.n_features = x.shape[1]

    @property
    def threshold(self) -> float:
        return 0.5

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        if self.w is None or self.n_features is None:
            raise RuntimeError("ridge learner has not been initialized")
        values = np.asarray(x, dtype=np.float64)
        return values @ self.w[:-1] + self.w[-1]

    def observe(self, x: np.ndarray, y: int, event_key: str) -> float:
        if self.P is None or self.w is None or self.n_features is None:
            raise RuntimeError("ridge learner has not been initialized")
        if y not in (0, 1):
            raise ValueError("only a revealed binary label may update ridge")
        if event_key in self.seen_event_keys:
            raise ValueError("a feedback event may be inserted only once")
        start = time.perf_counter()
        self.seen_event_keys.add(event_key)
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(xv) != self.n_features:
            raise ValueError("ridge feature dimension mismatch")
        z = np.concatenate([xv, np.ones(1, dtype=np.float64)])
        pz = self.P @ z
        denom = 1.0 + float(z @ pz)
        if not np.isfinite(denom) or denom <= 0.0:
            raise FloatingPointError("recursive ridge denominator is nonpositive or nonfinite")
        residual = float(y) - float(z @ self.w)
        self.w = self.w + pz * (residual / denom)
        self.P = self.P - np.outer(pz, pz) / denom
        return time.perf_counter() - start

    def state_digest(self) -> str:
        if self.P is None or self.w is None:
            raise RuntimeError("ridge learner has not been initialized")
        extra = b"\0".join(k.encode("utf-8") for k in sorted(self.seen_event_keys))
        return array_digest(self.P, self.w, extra=extra)

    def state_payload_bytes(self) -> int:
        if self.P is None or self.w is None:
            return 0
        return int(self.P.nbytes + self.w.nbytes)

    def export_state(self) -> dict[str, np.ndarray]:
        if self.P is None or self.w is None:
            raise RuntimeError("ridge learner has not been initialized")
        return {"P": self.P.copy(), "w": self.w.copy(),
                "regularization": np.asarray([self.regularization], dtype=np.float64)}


@dataclass(frozen=True)
class FutureItem:
    image_id: str
    product: str
    sample_key: str
    state: str
    x: np.ndarray


@dataclass(frozen=True)
class FeedbackItem:
    event_id: str
    image_id: str
    product: str
    sample_key: str
    stream_order: int
    state: str
    x: np.ndarray


def summarize_future(scores: np.ndarray, items: Sequence[FutureItem], threshold: float) -> tuple[dict, list[dict], list[dict]]:
    """Evaluator-only metrics; labels/product/group never enter a predictor."""
    if len(scores) != len(items):
        raise ValueError("future prediction/item count mismatch")
    products = sorted({i.product for i in items})
    product_rows: list[dict] = []
    ba_values: list[float] = []
    ap_all_values: list[float] = []
    ap_paired_values: list[float] = []
    total_binary = 0
    group_patterns: dict[tuple[str, str], list[int]] = {}
    pred = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int8)
    for item, label in zip(items, pred):
        group_patterns.setdefault((item.product, item.sample_key), []).append(
            binary_target(item.state) if binary_target(item.state) is not None else -1
        )
    for product in products:
        idx = [n for n, item in enumerate(items)
               if item.product == product and binary_target(item.state) is not None]
        y = np.asarray([binary_target(items[n].state) for n in idx], dtype=np.int8)
        s = np.asarray([scores[n] for n in idx], dtype=np.float64)
        yp = pred[idx]
        positives = int(np.sum(y == 1)); negatives = int(np.sum(y == 0)); n_binary = len(y)
        total_binary += n_binary
        if positives and negatives:
            tpr = float(np.mean(yp[y == 1] == 1))
            tnr = float(np.mean(yp[y == 0] == 0))
            ba = (tpr + tnr) / 2.0
            ba_status = "defined"
            ba_values.append(ba)
        else:
            tpr = float(np.mean(yp[y == 1] == 1)) if positives else None
            tnr = float(np.mean(yp[y == 0] == 0)) if negatives else None
            ba = None
            ba_status = "undefined_missing_class"
        if positives:
            ap = float(average_precision_score(y, s))
            if negatives == 0:
                ap_status = "degenerate_all_positive"
                ap_all_values.append(ap)
            elif positives == 0:
                ap_status = "undefined_no_positive"
            else:
                ap_status = "defined_both_classes"
                ap_all_values.append(ap)
                ap_paired_values.append(ap)
        else:
            ap = None
            ap_status = "undefined_no_positive"
        group_idxs: dict[str, list[int]] = {}
        for n, item in enumerate(items):
            if item.product == product:
                group_idxs.setdefault(item.sample_key, []).append(n)
        eligible_group_acc = []
        pattern_counts = {"present_only": 0, "absent_only": 0, "mixed": 0, "no_binary": 0}
        for gidx in group_idxs.values():
            gs = [binary_target(items[n].state) for n in gidx]
            gy = [v for v in gs if v is not None]
            if not gy:
                pattern_counts["no_binary"] += 1
            elif all(v == 1 for v in gy):
                pattern_counts["present_only"] += 1
            elif all(v == 0 for v in gy):
                pattern_counts["absent_only"] += 1
            else:
                pattern_counts["mixed"] += 1
            valid = [n for n in gidx if binary_target(items[n].state) is not None]
            if valid:
                eligible_group_acc.append(float(np.mean([
                    pred[n] == binary_target(items[n].state) for n in valid
                ])))
        product_rows.append({
            "product": product, "n_views": sum(i.product == product for i in items),
            "n_binary": n_binary, "present": positives, "absent": negatives,
            "uncertain": sum(i.product == product and i.state == "unobservable_or_uncertain" for i in items),
            "not_applicable": sum(i.product == product and i.state == "not_applicable" for i in items),
            "ba": ba, "ba_status": ba_status, "tpr": tpr, "tnr": tnr,
            "ap": ap, "ap_status": ap_status,
            "groups": len(group_idxs),
            "group_present_only": pattern_counts["present_only"],
            "group_absent_only": pattern_counts["absent_only"],
            "group_mixed": pattern_counts["mixed"],
            "group_no_binary": pattern_counts["no_binary"],
            "group_accuracy_mean": float(np.mean(eligible_group_acc)) if eligible_group_acc else None,
            "group_accuracy_min": float(np.min(eligible_group_acc)) if eligible_group_acc else None,
            "group_accuracy_max": float(np.max(eligible_group_acc)) if eligible_group_acc else None,
        })
    metrics = {
        "future_views": len(items), "future_binary_views": total_binary,
        "future_groups": len({(i.product, i.sample_key) for i in items}),
        "ba_product_macro": float(np.mean(ba_values)) if ba_values else None,
        "ba_eligible_product_count": len(ba_values),
        "ap_product_macro_including_degenerate": float(np.mean(ap_all_values)) if ap_all_values else None,
        "ap_products_including_degenerate": len(ap_all_values),
        "ap_product_macro_paired_class": float(np.mean(ap_paired_values)) if ap_paired_values else None,
        "ap_paired_class_product_count": len(ap_paired_values),
    }
    return metrics, product_rows, [
        {"sample_key": key, "product": product,
         "binary_views": sum(binary_target(i.state) is not None for i in items if i.product == product and i.sample_key == key),
         "state_pattern": "/".join("P" if v == 1 else "A" if v == 0 else "U" for v in sorted(
             group_patterns[(product, key)])),
         "group_accuracy": float(np.mean([
             pred[n] == binary_target(items[n].state)
             for n, it in enumerate(items) if it.product == product and it.sample_key == key
             and binary_target(it.state) is not None
         ])) if any(it.product == product and it.sample_key == key and binary_target(it.state) is not None for it in items) else None}
        for product, key in sorted(group_patterns)
    ]


def run_prequential(
    learner: LinearPredictor,
    feedback_events: Sequence[FeedbackItem],
    future_items: Sequence[FutureItem],
    reveal: Callable[[str], str],
    *,
    protocol: str,
    method: str,
    config_id: str,
    seed: int | None,
    pair_id: str,
    state_sink: Callable[[str, int, LinearPredictor], None] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Run predict -> reveal -> update -> fixed-future-evaluation checkpoints."""
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    product_metric_rows: list[dict] = []
    group_metric_rows: list[dict] = []
    event_rows: list[dict] = []
    prior_correct: dict[str, bool] = {}

    def evaluate(checkpoint: int) -> None:
        start = time.perf_counter()
        scores = learner.predict_scores(np.stack([i.x for i in future_items]))
        elapsed = time.perf_counter() - start
        metrics, products, groups = summarize_future(scores, future_items, learner.threshold)
        pred_start = len(prediction_rows)
        metrics.update({
            "protocol": protocol, "method": method, "config_id": config_id,
            "seed": seed, "pair_id": pair_id, "checkpoint": checkpoint,
            "prediction_batch_latency_ms": elapsed * 1000.0,
            "prediction_latency_ms_per_view": elapsed * 1000.0 / max(1, len(future_items)),
            "state_sha256": learner.state_digest(),
            "state_payload_bytes": learner.state_payload_bytes(),
        })
        metric_rows.append(metrics)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        for item, score in zip(future_items, scores):
            y = binary_target(item.state)
            pred = int(score >= learner.threshold)
            correct = None if y is None else bool(pred == y)
            previous = prior_correct.get(item.image_id)
            harmful = bool(previous is True and correct is False)
            prediction_rows.append({
                "protocol": protocol, "method": method, "config_id": config_id,
                "seed": seed, "pair_id": pair_id, "checkpoint": checkpoint,
                "role": "future_eval", "image_id": item.image_id,
                "sample_key": item.sample_key, "product": item.product,
                "attribute_state": item.state, "binary_target": y,
                "score": float(score), "threshold": learner.threshold,
                "prediction": pred, "correct": correct,
                "previous_checkpoint_correct": previous,
                "harmful_flip_from_previous": harmful,
            })
            if correct is not None:
                prior_correct[item.image_id] = correct
        step_predictions = prediction_rows[pred_start:]
        harmful_total = sum(bool(row["harmful_flip_from_previous"]) for row in step_predictions)
        metrics["harmful_flips_from_previous_checkpoint"] = harmful_total
        harmful_by_product: dict[str, int] = {}
        for row in step_predictions:
            if row["harmful_flip_from_previous"]:
                harmful_by_product[row["product"]] = harmful_by_product.get(row["product"], 0) + 1
        for row in products:
            row.update({"protocol": protocol, "method": method, "config_id": config_id,
                        "seed": seed, "pair_id": pair_id, "checkpoint": checkpoint})
            row["harmful_flips_from_previous_checkpoint"] = harmful_by_product.get(row["product"], 0)
            product_metric_rows.append(row)
        for row in groups:
            row.update({"protocol": protocol, "method": method, "config_id": config_id,
                        "seed": seed, "pair_id": pair_id, "checkpoint": checkpoint})
            group_metric_rows.append(row)
        if state_sink is not None:
            state_sink(protocol, checkpoint, learner)

    evaluate(0)
    seen_events: set[str] = set()
    seen_images: set[str] = set()
    for event in sorted(feedback_events, key=lambda x: x.stream_order):
        if event.event_id in seen_events or event.image_id in seen_images:
            raise ValueError("feedback tape repeats an event/image; annotation repeats are not training events")
        seen_events.add(event.event_id)
        seen_images.add(event.image_id)
        before = learner.state_digest()
        start = time.perf_counter()
        query_score = float(learner.predict_scores(event.x.reshape(1, -1))[0])
        predict_elapsed = time.perf_counter() - start
        query_prediction = int(query_score >= learner.threshold)
        # The evaluator reveals only after the pre-reveal score has been generated and logged.
        state = reveal(event.event_id)
        target = binary_target(state)
        update_elapsed = 0.0
        update_applied = False
        if learner.update_enabled and target is not None:
            update_elapsed = learner.observe(event.x, target, event.event_id)
            update_applied = True
        after = learner.state_digest()
        event_rows.append({
            "protocol": protocol, "method": method, "config_id": config_id,
            "seed": seed, "pair_id": pair_id, "event_id": event.event_id,
            "image_id": event.image_id, "sample_key": event.sample_key,
            "product": event.product, "stream_order": event.stream_order,
            "prediction_timing": "before_reveal", "pre_reveal_score": query_score,
            "pre_reveal_prediction": query_prediction, "threshold": learner.threshold,
            "revealed_state_evaluator_only": state, "binary_target_if_revealed": target,
            "feedback_cost_unit": 1, "update_applied": update_applied,
            "update_latency_ms": update_elapsed * 1000.0,
            "pre_reveal_prediction_latency_ms": predict_elapsed * 1000.0,
            "state_sha256_before": before, "state_sha256_after": after,
            "state_payload_bytes_after": learner.state_payload_bytes(),
        })
        prediction_rows.append({
            "protocol": protocol, "method": method, "config_id": config_id,
            "seed": seed, "pair_id": pair_id, "checkpoint": event.stream_order,
            "role": "feedback_query_pre_reveal", "image_id": event.image_id,
            "sample_key": event.sample_key, "product": event.product,
            "attribute_state": state, "binary_target": target,
            "score": query_score, "threshold": learner.threshold,
            "prediction": query_prediction, "correct": None if target is None else bool(query_prediction == target),
            "previous_checkpoint_correct": None, "harmful_flip_from_previous": False,
        })
        evaluate(event.stream_order)
    return prediction_rows, metric_rows, product_metric_rows, group_metric_rows, event_rows


def instantiate_sgd(alpha: float, seed: int, x_fit: np.ndarray, y_fit: np.ndarray) -> SGDReplayLearner:
    learner = SGDReplayLearner(alpha=alpha, seed=seed)
    learner.initialize(x_fit, y_fit)
    return learner


def instantiate_refit(alpha: float, x_fit: np.ndarray, y_fit: np.ndarray) -> tuple[LogisticRefitLearner, float]:
    learner = LogisticRefitLearner(alpha=alpha)
    elapsed = learner.initialize(x_fit, y_fit)
    return learner, elapsed


def instantiate_ridge(lam: float, x_fit: np.ndarray, y_fit: np.ndarray) -> RecursiveRidgeLearner:
    learner = RecursiveRidgeLearner(regularization=lam)
    learner.initialize(x_fit, y_fit)
    return learner

