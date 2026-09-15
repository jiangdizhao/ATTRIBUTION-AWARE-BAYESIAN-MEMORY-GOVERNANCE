#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential local-update / interference audit for ABMG.

Consumes an existing ``abmg.factorizability.features.v1`` artifact and does NOT
rerun DINO or open Real-IAD images.

Scientific question
-------------------
Can sparse verified feedback update only the addressed factor memory, improve
future routing, and avoid harmful cross-factor interference in a sequential
held-out-product stream?

Protocol
--------
* Core-4 labels: AK / HS / QS / ZW.
* Source-CV training product categories provide:
  - one initial verified support per factor by default;
  - a label-free shared diagonal background mean/variance.
* Each held-out source-CV product group is split, offline only, into:
  - adaptation stream: predictions are made before any feedback/update;
  - sentinel set: labels are never used for memory updates and are used only to
    measure learning/interference at checkpoints.
* Query locations are sampled uniformly from stream positions, independently of
  defect labels and model predictions.  Only queried examples reveal a label.
* Updating factor g changes only ``n_g`` and ``sum_g``.  Other factor states are
  checked for exact invariance after every update.
* Two routing rules use the same memory state:
  - ``diag_shrunk``: shrunk factor mean + shared diagonal covariance;
  - ``bayes_predictive``: same mean plus count-dependent posterior-predictive
    variance.  Unlike the earlier equal-shot audit, factor counts may differ
    here, so these two rules can now rank classes differently.

Immediate structural-credit audit
---------------------------------
Before and after every queried update to factor g, a fixed sentinel probe bank is
rescored.  For every evaluation factor h we record the change in true-vs-best-
wrong score margin, accuracy, helpful flips, and harmful flips.  Aggregating
these effects gives a 4x4 credit-assignment matrix:

    row g = factor whose memory was updated
    col h = factor whose sentinel behavior was measured

A desirable local learner has positive diagonal effects and small/off-diagonal
harmful effects.

Interpretation boundary
-----------------------
This is an offline source-side simulation of sequential memory updates.  It does
not establish semantic/causal factor discovery, deployable query policy,
calibrated online responsibilities, final C1/C2, or end-to-end anomaly-detection
improvement.  Outer ABMG target categories remain untouched.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

try:  # direct execution: python scripts/...
    from abmg_prototype_addressability_audit import (
        CORE4,
        EXPECTED_SCHEMA,
        _finite_mean_std,
        _l2,
        _metrics,
        category_diverse_support_indices,
        parse_int_list,
        parse_str_list,
    )
    from abmg_memory_addressability_audit import fit_shared_diag_prior
except ImportError:  # module import from repo root / unit tests
    from scripts.abmg_prototype_addressability_audit import (
        CORE4,
        EXPECTED_SCHEMA,
        _finite_mean_std,
        _l2,
        _metrics,
        category_diverse_support_indices,
        parse_int_list,
        parse_str_list,
    )
    from scripts.abmg_memory_addressability_audit import fit_shared_diag_prior

MODELS: Tuple[str, ...] = ("diag_shrunk", "bayes_predictive")


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    vals = f1_score(
        y_true,
        y_pred,
        labels=list(CORE4),
        average=None,
        zero_division=0,
    )
    return {label: float(vals[i]) for i, label in enumerate(CORE4)}


def stratified_stream_sentinel_split(
    categories: np.ndarray,
    labels: np.ndarray,
    sentinel_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split held-out examples by (product category, defect label).

    Labels are used only to make the OFFLINE diagnostic split stable and to keep
    each factor represented in the never-updated sentinel set.  They do not
    affect stream order, query positions, routing, or memory updates unless that
    stream item is explicitly queried later.
    """
    if not 0.0 < float(sentinel_fraction) < 1.0:
        raise ValueError("sentinel_fraction must be in (0,1)")
    categories = np.asarray(categories, dtype=object)
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(int(seed))
    stream: List[int] = []
    sentinel: List[int] = []
    cells: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, (c, y) in enumerate(zip(categories, labels)):
        cells[(str(c), str(y))].append(i)

    for key in sorted(cells):
        idx = np.asarray(cells[key], dtype=np.int64)
        rng.shuffle(idx)
        if len(idx) < 2:
            stream.extend(idx.tolist())
            continue
        n_sentinel = int(round(len(idx) * float(sentinel_fraction)))
        n_sentinel = min(max(n_sentinel, 1), len(idx) - 1)
        sentinel.extend(idx[:n_sentinel].tolist())
        stream.extend(idx[n_sentinel:].tolist())

    stream_arr = np.asarray(sorted(stream), dtype=np.int64)
    sentinel_arr = np.asarray(sorted(sentinel), dtype=np.int64)
    if set(stream_arr.tolist()) & set(sentinel_arr.tolist()):
        raise AssertionError("stream and sentinel overlap")
    if len(stream_arr) + len(sentinel_arr) != len(labels):
        raise AssertionError("split does not cover all examples")
    missing = [y for y in CORE4 if y not in set(labels[sentinel_arr].tolist())]
    if missing:
        raise RuntimeError(f"sentinel split lost Core-4 labels: {missing}")
    return stream_arr, sentinel_arr


def uniform_query_positions(n_stream: int, budget: int, seed: int) -> np.ndarray:
    """Choose exact stream positions uniformly, with no label/model input."""
    n_stream = int(n_stream)
    budget = min(max(int(budget), 0), n_stream)
    if budget == 0:
        return np.empty((0,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_stream, size=budget, replace=False).astype(np.int64))


def select_probe_indices(labels: np.ndarray, max_per_label: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(int(seed))
    out: List[int] = []
    for y in CORE4:
        idx = np.flatnonzero(labels == y)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        k = len(idx) if int(max_per_label) <= 0 else min(len(idx), int(max_per_label))
        out.extend(idx[:k].tolist())
    return np.asarray(sorted(out), dtype=np.int64)


def init_memory(
    X_train: np.ndarray,
    support_by_label: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    x = _l2(X_train)
    mem: Dict[str, Dict[str, Any]] = {}
    for y in CORE4:
        idx = np.asarray(support_by_label[y], dtype=np.int64)
        if len(idx) == 0:
            raise ValueError(f"initial support missing label={y}")
        mem[y] = {
            "n": int(len(idx)),
            "sum": x[idx].sum(axis=0).astype(np.float64),
        }
    return mem


def update_memory_local(
    memory: Dict[str, Dict[str, Any]],
    label: str,
    vector: np.ndarray,
) -> None:
    """Update exactly one factor's sufficient statistics in-place."""
    label = str(label)
    if label not in CORE4:
        raise ValueError(f"unknown factor={label!r}")
    z = np.asarray(vector, dtype=np.float64).reshape(1, -1)
    z = _l2(z)[0]
    memory[label]["n"] = int(memory[label]["n"]) + 1
    memory[label]["sum"] = np.asarray(memory[label]["sum"], dtype=np.float64) + z


def assert_local_update(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    updated_label: str,
) -> bool:
    for y in CORE4:
        if y == updated_label:
            continue
        if int(before[y]["n"]) != int(after[y]["n"]):
            return False
        if not np.array_equal(np.asarray(before[y]["sum"]), np.asarray(after[y]["sum"])):
            return False
    return True


def score_memory(
    X_eval: np.ndarray,
    memory: Mapping[str, Mapping[str, Any]],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
    model: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score using shrunk shared-diagonal memory or posterior predictive."""
    if model not in MODELS:
        raise ValueError(f"unknown model={model!r}")
    if float(kappa0) <= 0:
        raise ValueError("kappa0 must be > 0")
    xv = _l2(X_eval)
    mu0 = np.asarray(shared_prior["mu0"], dtype=np.float64)
    var = np.asarray(shared_prior["var"], dtype=np.float64)
    cols = []
    for y in CORE4:
        n = int(memory[y]["n"])
        sum_x = np.asarray(memory[y]["sum"], dtype=np.float64)
        kappa_n = float(kappa0) + float(n)
        mu = (float(kappa0) * mu0 + sum_x) / kappa_n
        score_var = var
        if model == "bayes_predictive":
            score_var = var * (1.0 + 1.0 / kappa_n)
        diff = xv - mu[None, :]
        logp = -0.5 * np.sum(
            (diff * diff) / score_var[None, :] + np.log(score_var[None, :]),
            axis=1,
        )
        cols.append(logp)
    scores = np.stack(cols, axis=1)
    pred = np.asarray(CORE4, dtype=object)[scores.argmax(axis=1)]
    return pred, scores


def true_margin(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    lut = {y: i for i, y in enumerate(CORE4)}
    cols = np.asarray([lut[str(y)] for y in y_true], dtype=np.int64)
    rows = np.arange(len(cols))
    t = scores[rows, cols]
    other = np.array(scores, copy=True)
    other[rows, cols] = -np.inf
    return t - other.max(axis=1)


def evaluate_state(
    X: np.ndarray,
    y: np.ndarray,
    memory: Mapping[str, Mapping[str, Any]],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
    model: str,
) -> Dict[str, Any]:
    pred, scores = score_memory(X, memory, shared_prior, kappa0, model)
    margins = true_margin(y, scores)
    per_margin = {}
    for label in CORE4:
        m = labels_mask = np.asarray(y == label, dtype=bool)
        per_margin[label] = float(margins[m].mean()) if np.any(m) else float("nan")
    return {
        "metrics": _metrics(y, pred),
        "per_class_f1": _per_class_f1(y, pred),
        "mean_true_minus_best_wrong_margin": float(margins.mean()),
        "per_class_margin": per_margin,
        "pred": pred,
        "scores": scores,
    }


def effect_summary(
    y_probe: np.ndarray,
    pred_before: np.ndarray,
    scores_before: np.ndarray,
    pred_after: np.ndarray,
    scores_after: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    mb = true_margin(y_probe, scores_before)
    ma = true_margin(y_probe, scores_after)
    out: Dict[str, Dict[str, float]] = {}
    for h in CORE4:
        mask = np.asarray(y_probe == h, dtype=bool)
        if not np.any(mask):
            continue
        cb = pred_before[mask] == y_probe[mask]
        ca = pred_after[mask] == y_probe[mask]
        delta = ma[mask] - mb[mask]
        out[h] = {
            "margin_delta": float(delta.mean()),
            "abs_margin_delta": float(np.abs(delta).mean()),
            "accuracy_delta": float(ca.mean() - cb.mean()),
            "helpful_flip_rate": float(np.mean((~cb) & ca)),
            "harmful_flip_rate": float(np.mean(cb & (~ca))),
        }
    return out


def _new_effect_accumulator() -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        g: {
            h: {
                "n": 0.0,
                "margin_delta_sum": 0.0,
                "margin_delta_sq_sum": 0.0,
                "abs_margin_delta_sum": 0.0,
                "accuracy_delta_sum": 0.0,
                "helpful_flip_rate_sum": 0.0,
                "harmful_flip_rate_sum": 0.0,
            }
            for h in CORE4
        }
        for g in CORE4
    }


def _accumulate_effect(
    acc: Dict[str, Dict[str, Dict[str, float]]],
    update_factor: str,
    effects: Mapping[str, Mapping[str, float]],
) -> None:
    for h, e in effects.items():
        a = acc[update_factor][h]
        d = float(e["margin_delta"])
        a["n"] += 1.0
        a["margin_delta_sum"] += d
        a["margin_delta_sq_sum"] += d * d
        a["abs_margin_delta_sum"] += float(e["abs_margin_delta"])
        a["accuracy_delta_sum"] += float(e["accuracy_delta"])
        a["helpful_flip_rate_sum"] += float(e["helpful_flip_rate"])
        a["harmful_flip_rate_sum"] += float(e["harmful_flip_rate"])


def _checkpoint_record(
    X_sentinel: np.ndarray,
    y_sentinel: np.ndarray,
    memory: Mapping[str, Mapping[str, Any]],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
    n_feedback: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_feedback": int(n_feedback),
        "factor_counts": {y: int(memory[y]["n"]) for y in CORE4},
        "models": {},
    }
    preds = {}
    for model in MODELS:
        ev = evaluate_state(X_sentinel, y_sentinel, memory, shared_prior, kappa0, model)
        preds[model] = ev.pop("pred")
        ev.pop("scores")
        out["models"][model] = ev
    out["shrunk_vs_bayes_prediction_agreement"] = float(
        np.mean(preds["diag_shrunk"] == preds["bayes_predictive"])
    )
    return out


def run_one_sequence(
    X: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    cv_group: Sequence[str],
    source_cv_fold: int,
    repeat: int,
    seed: int,
    initial_shots: int,
    query_budget: int,
    checkpoints: Sequence[int],
    sentinel_fraction: float,
    probe_per_label: int,
    kappa0: float,
) -> Dict[str, Any]:
    wanted = set(CORE4)
    all_categories = set(categories.tolist())
    test_cats = set(str(x) for x in cv_group)
    train_cats = all_categories - test_cats
    train_all = np.asarray([c in train_cats for c in categories], dtype=bool)
    train_core = np.asarray(
        [(c in train_cats) and (y in wanted) for c, y in zip(categories, labels)],
        dtype=bool,
    )
    test_core = np.asarray(
        [(c in test_cats) and (y in wanted) for c, y in zip(categories, labels)],
        dtype=bool,
    )
    xt, yt, ct = X[train_core], labels[train_core], categories[train_core]
    xv, yv, cv = X[test_core], labels[test_core], categories[test_core]
    if len(set(yt.tolist())) != len(CORE4) or len(set(yv.tolist())) != len(CORE4):
        raise RuntimeError("Core-4 coverage missing in train/test split")

    split_seed = int(seed) + 9_000_000 + 10_000 * int(source_cv_fold)
    stream_idx, sentinel_idx = stratified_stream_sentinel_split(
        cv, yv, float(sentinel_fraction), split_seed
    )
    X_stream0, y_stream0 = xv[stream_idx], yv[stream_idx]
    X_sentinel, y_sentinel = xv[sentinel_idx], yv[sentinel_idx]
    probe_idx = select_probe_indices(
        y_sentinel, int(probe_per_label), split_seed + 123
    )
    X_probe, y_probe = X_sentinel[probe_idx], y_sentinel[probe_idx]

    run_seed = int(seed) + 100_000 * int(source_cv_fold) + int(repeat)
    support = category_diverse_support_indices(
        yt, ct, CORE4, int(initial_shots), run_seed + 17
    )
    memory = init_memory(xt, support)
    shared_prior = fit_shared_diag_prior(X[train_all])

    rng = np.random.default_rng(run_seed + 101)
    order = rng.permutation(len(X_stream0)).astype(np.int64)
    X_stream, y_stream = X_stream0[order], y_stream0[order]
    query_pos = uniform_query_positions(len(X_stream), int(query_budget), run_seed + 202)
    query_set = set(query_pos.tolist())

    effect_acc = {m: _new_effect_accumulator() for m in MODELS}
    state_locality_checks: List[bool] = []
    feedback_label_counts = {y: 0 for y in CORE4}
    preq_true: Dict[str, List[str]] = {m: [] for m in MODELS}
    preq_pred: Dict[str, List[str]] = {m: [] for m in MODELS}
    checkpoint_set = set(int(x) for x in checkpoints)
    checkpoint_rows: List[Dict[str, Any]] = []
    if 0 in checkpoint_set:
        checkpoint_rows.append(
            _checkpoint_record(
                X_sentinel, y_sentinel, memory, shared_prior, kappa0, 0
            )
        )

    n_feedback = 0
    for pos in range(len(X_stream)):
        z = X_stream[pos : pos + 1]
        y = str(y_stream[pos])
        for model in MODELS:
            pred, _ = score_memory(z, memory, shared_prior, kappa0, model)
            preq_true[model].append(y)
            preq_pred[model].append(str(pred[0]))

        if pos not in query_set:
            continue

        probe_before = {}
        for model in MODELS:
            pb, sb = score_memory(X_probe, memory, shared_prior, kappa0, model)
            probe_before[model] = (pb, sb)

        before = deepcopy(memory)
        update_memory_local(memory, y, z[0])
        state_locality_checks.append(assert_local_update(before, memory, y))
        feedback_label_counts[y] += 1
        n_feedback += 1

        for model in MODELS:
            pa, sa = score_memory(X_probe, memory, shared_prior, kappa0, model)
            pb, sb = probe_before[model]
            eff = effect_summary(y_probe, pb, sb, pa, sa)
            _accumulate_effect(effect_acc[model], y, eff)

        if n_feedback in checkpoint_set:
            checkpoint_rows.append(
                _checkpoint_record(
                    X_sentinel,
                    y_sentinel,
                    memory,
                    shared_prior,
                    kappa0,
                    n_feedback,
                )
            )

    prequential = {}
    for model in MODELS:
        yt_arr = np.asarray(preq_true[model], dtype=object)
        yp_arr = np.asarray(preq_pred[model], dtype=object)
        prequential[model] = {
            "metrics": _metrics(yt_arr, yp_arr),
            "per_class_f1": _per_class_f1(yt_arr, yp_arr),
        }

    return {
        "source_cv_fold": int(source_cv_fold),
        "repeat": int(repeat),
        "seed": int(run_seed),
        "test_categories": sorted(test_cats),
        "n_train_core4": int(train_core.sum()),
        "n_train_background": int(train_all.sum()),
        "n_stream": int(len(X_stream)),
        "n_sentinel": int(len(X_sentinel)),
        "n_probe": int(len(X_probe)),
        "query_budget_requested": int(query_budget),
        "n_feedback_realized": int(n_feedback),
        "query_positions_label_independent": True,
        "feedback_label_counts": feedback_label_counts,
        "initial_support_categories": {
            y: sorted(set(ct[idx].tolist())) for y, idx in support.items()
        },
        "state_locality_pass_fraction": float(np.mean(state_locality_checks))
        if state_locality_checks
        else float("nan"),
        "checkpoints": checkpoint_rows,
        "prequential": prequential,
        "effect_accumulators": effect_acc,
    }


def _aggregate_effects(runs: Sequence[Dict[str, Any]], model: str) -> Dict[str, Any]:
    matrix: Dict[str, Dict[str, Any]] = {g: {} for g in CORE4}
    diag_signed: List[Tuple[float, float]] = []
    off_abs: List[Tuple[float, float]] = []
    off_harm: List[Tuple[float, float]] = []
    for g in CORE4:
        for h in CORE4:
            sums = {
                "n": 0.0,
                "margin_delta_sum": 0.0,
                "margin_delta_sq_sum": 0.0,
                "abs_margin_delta_sum": 0.0,
                "accuracy_delta_sum": 0.0,
                "helpful_flip_rate_sum": 0.0,
                "harmful_flip_rate_sum": 0.0,
            }
            for r in runs:
                cell = r["effect_accumulators"][model][g][h]
                for k in sums:
                    sums[k] += float(cell[k])
            n = sums["n"]
            if n > 0:
                mean = sums["margin_delta_sum"] / n
                var = max(sums["margin_delta_sq_sum"] / n - mean * mean, 0.0)
                row = {
                    "n_update_events": int(n),
                    "margin_delta_mean": float(mean),
                    "margin_delta_std": float(np.sqrt(var)),
                    "abs_margin_delta_mean": float(sums["abs_margin_delta_sum"] / n),
                    "accuracy_delta_mean": float(sums["accuracy_delta_sum"] / n),
                    "helpful_flip_rate_mean": float(sums["helpful_flip_rate_sum"] / n),
                    "harmful_flip_rate_mean": float(sums["harmful_flip_rate_sum"] / n),
                }
                if g == h:
                    diag_signed.append((mean, n))
                else:
                    off_abs.append((row["abs_margin_delta_mean"], n))
                    off_harm.append((row["harmful_flip_rate_mean"], n))
            else:
                row = {"n_update_events": 0}
            matrix[g][h] = row

    def weighted(vals: Sequence[Tuple[float, float]]) -> float:
        den = sum(w for _, w in vals)
        return float(sum(v * w for v, w in vals) / den) if den else float("nan")

    diag = weighted(diag_signed)
    off = weighted(off_abs)
    return {
        "matrix": matrix,
        "diagonal_margin_delta_mean": diag,
        "offdiagonal_abs_margin_delta_mean": off,
        "offdiagonal_harmful_flip_rate_mean": weighted(off_harm),
        "diagonal_to_offdiagonal_abs_ratio": float(abs(diag) / max(off, 1e-12))
        if np.isfinite(diag) and np.isfinite(off)
        else float("nan"),
    }


def _aggregate_checkpoints(
    runs: Sequence[Dict[str, Any]], checkpoints: Sequence[int]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cp in checkpoints:
        cp_rows = []
        for r in runs:
            match = [x for x in r["checkpoints"] if int(x["n_feedback"]) == int(cp)]
            if match:
                cp_rows.append((r, match[0]))
        row: Dict[str, Any] = {"n_runs": len(cp_rows), "models": {}}
        for model in MODELS:
            mrow: Dict[str, Any] = {}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                m, s = _finite_mean_std(
                    x[1]["models"][model]["metrics"][metric] for x in cp_rows
                )
                mrow[f"{metric}_mean"] = m
                mrow[f"{metric}_std"] = s
            per_class = {}
            for label in CORE4:
                m, s = _finite_mean_std(
                    x[1]["models"][model]["per_class_f1"][label] for x in cp_rows
                )
                per_class[label] = {"f1_mean": m, "f1_std": s}
            mrow["per_class"] = per_class
            mm, ms = _finite_mean_std(
                x[1]["models"][model]["mean_true_minus_best_wrong_margin"]
                for x in cp_rows
            )
            mrow["margin_mean"] = mm
            mrow["margin_std"] = ms

            deltas = []
            for r, current in cp_rows:
                base = [x for x in r["checkpoints"] if int(x["n_feedback"]) == 0]
                if base:
                    deltas.append(
                        current["models"][model]["metrics"]["macro_f1"]
                        - base[0]["models"][model]["metrics"]["macro_f1"]
                    )
            dm, ds = _finite_mean_std(deltas)
            mrow["macro_f1_delta_from_initial_mean"] = dm
            mrow["macro_f1_delta_from_initial_std"] = ds
            row["models"][model] = mrow

        am, ast = _finite_mean_std(
            x[1]["shrunk_vs_bayes_prediction_agreement"] for x in cp_rows
        )
        row["shrunk_vs_bayes_prediction_agreement_mean"] = am
        row["shrunk_vs_bayes_prediction_agreement_std"] = ast
        for label in CORE4:
            cm, cs = _finite_mean_std(
                x[1]["factor_counts"][label] for x in cp_rows
            )
            row.setdefault("factor_counts", {})[label] = {
                "mean": cm,
                "std": cs,
            }
        out[str(cp)] = row
    return out


def aggregate_runs(
    runs: Sequence[Dict[str, Any]], checkpoints: Sequence[int]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_runs": len(runs),
        "checkpoints": _aggregate_checkpoints(runs, checkpoints),
        "interference": {m: _aggregate_effects(runs, m) for m in MODELS},
        "prequential": {},
    }
    lm, ls = _finite_mean_std(r["state_locality_pass_fraction"] for r in runs)
    out["state_locality_pass_fraction_mean"] = lm
    out["state_locality_pass_fraction_std"] = ls
    for model in MODELS:
        mrow = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            m, s = _finite_mean_std(r["prequential"][model]["metrics"][metric] for r in runs)
            mrow[f"{metric}_mean"] = m
            mrow[f"{metric}_std"] = s
        out["prequential"][model] = mrow
    for label in CORE4:
        m, s = _finite_mean_std(r["feedback_label_counts"][label] for r in runs)
        out.setdefault("feedback_label_counts", {})[label] = {"mean": m, "std": s}
    return out


def _load_artifact(path: Path) -> Dict[str, Any]:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if artifact.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected artifact schema={artifact.get('schema')!r}")
    return artifact


def run(args: argparse.Namespace) -> int:
    artifact_path = Path(args.features)
    artifact = _load_artifact(artifact_path)
    if int(artifact.get("outer_fold", -1)) != int(args.fold):
        raise ValueError(
            f"artifact outer_fold={artifact.get('outer_fold')} != requested fold={args.fold}"
        )
    requested = parse_str_list(args.representations)
    missing = [x for x in requested if x not in artifact["representations"]]
    if missing:
        raise KeyError(f"Representation(s) not found: {missing}")
    checkpoints = tuple(sorted(set(int(x) for x in parse_int_list(args.checkpoints)) | {0}))
    if max(checkpoints) > int(args.query_budget):
        raise ValueError("max checkpoint cannot exceed query budget")
    if int(args.initial_shots) <= 0:
        raise ValueError("initial_shots must be > 0")
    if float(args.bayes_kappa0) <= 0:
        raise ValueError("bayes_kappa0 must be > 0")

    items = artifact["items"]
    cv_groups = artifact["source_cv_groups"]
    result: Dict[str, Any] = {
        "schema": "abmg.sequential_local_update_interference_audit.v1",
        "outer_fold": int(args.fold),
        "source_artifact": str(artifact_path),
        "target_categories_untouched": artifact["target_categories_untouched"],
        "core4_labels": list(CORE4),
        "representations_requested": list(requested),
        "protocol": {
            "initial_shots_per_factor_from_training_products": int(args.initial_shots),
            "query_budget": int(args.query_budget),
            "checkpoints": list(checkpoints),
            "repeats_per_source_cv_fold": int(args.repeats),
            "sentinel_fraction": float(args.sentinel_fraction),
            "probe_per_label": int(args.probe_per_label),
            "query_policy": "uniform random stream positions; label/model independent",
            "prequential_rule": "predict before feedback/update on every stream item",
            "factor_state": ["n_g", "sum_g"],
            "shared_state": "label-free source-training-product mean and diagonal variance; frozen during stream",
            "models": list(MODELS),
            "bayes_kappa0": float(args.bayes_kappa0),
            "sentinel_labels_never_used_for_updates": True,
        },
        "interpretation_boundary": (
            "Sequential source-side diagnostic only. It tests local external-memory update behavior and "
            "cross-factor interference, not automatic factor discovery, deployable query policy, calibrated "
            "online responsibilities, final C1/C2, or end-to-end anomaly-detection gain."
        ),
        "representations": {},
    }

    for key in requested:
        slot = artifact["representations"][key]
        idx = slot["item_indices"].numpy().astype(np.int64)
        X = slot["features"].float().numpy().astype(np.float64)
        categories = np.asarray([items[i]["category"] for i in idx], dtype=object)
        labels = np.asarray(
            [items[i]["defect_source_offline_only"] for i in idx], dtype=object
        )
        runs = []
        print(f"Evaluating {key}: n={len(X)} d={X.shape[1]}")
        for fi, group in enumerate(cv_groups):
            for ri in range(int(args.repeats)):
                r = run_one_sequence(
                    X,
                    categories,
                    labels,
                    group,
                    fi,
                    ri,
                    int(args.seed),
                    int(args.initial_shots),
                    int(args.query_budget),
                    checkpoints,
                    float(args.sentinel_fraction),
                    int(args.probe_per_label),
                    float(args.bayes_kappa0),
                )
                runs.append(r)
        agg = aggregate_runs(runs, checkpoints)
        result["representations"][key] = {"aggregate": agg, "runs": runs}

        print(f"  state locality pass={agg['state_locality_pass_fraction_mean']:.4f}")
        for cp in checkpoints:
            row = agg["checkpoints"][str(cp)]
            parts = []
            for model in MODELS:
                m = row["models"][model]
                parts.append(
                    f"{model} F1={m['macro_f1_mean']:.4f} "
                    f"delta={m['macro_f1_delta_from_initial_mean']:+.4f}"
                )
            print(f"  feedback={cp}: " + " | ".join(parts))
        for model in MODELS:
            inter = agg["interference"][model]
            print(
                f"  {model} interference: diag_margin={inter['diagonal_margin_delta_mean']:+.4f}, "
                f"offdiag_abs={inter['offdiagonal_abs_margin_delta_mean']:.4f}, "
                f"offdiag_harm_flip={inter['offdiagonal_harmful_flip_rate_mean']:.4f}, "
                f"locality_ratio={inter['diagonal_to_offdiagonal_abs_ratio']:.3f}"
            )

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_path = out_dir / "sequential_local_update_interference_audit.json"
    _json_dump(result, out_path)
    print(f"Saved {out_path}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--features", type=str, required=True)
    p.add_argument(
        "--representations",
        type=str,
        default="oracle/raw_patch,sensor/raw_patch/k8/score_softmax",
    )
    p.add_argument("--initial-shots", type=int, default=1)
    p.add_argument("--query-budget", type=int, default=32)
    p.add_argument("--checkpoints", type=str, default="4,8,16,32")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--sentinel-fraction", type=float, default=0.25)
    p.add_argument("--probe-per-label", type=int, default=16)
    p.add_argument("--bayes-kappa0", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/sequential_local_update_interference_audit",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
