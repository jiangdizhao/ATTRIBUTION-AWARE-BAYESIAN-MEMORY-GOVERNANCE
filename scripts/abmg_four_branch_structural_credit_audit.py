#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Four-branch structural-credit audit for ABMG.

This audit is the causal control that follows the sequential local-update /
interference audit. It consumes an existing ``abmg.factorizability.features.v1``
artifact and does NOT rerun DINO or open Real-IAD images.

Scientific question
-------------------
Given the same sparse verified feedback, does writing the observation to the
correct persistent factor address improve future same-factor routing while
causing less collateral interference than non-selective or deliberately wrong
credit assignment?

Matched branches
----------------
Every branch uses the same frozen evidence vectors, source-CV split, initial
support, stream order, sentinel set, query positions, revealed labels, shared
diagonal prior, and routing rule. Only the write-credit vector differs:

* ``no_update``:         alpha_g = 0 for every factor.
* ``global_broadcast``:  alpha_g = 1/G for every factor.
* ``correct_address``:   alpha_y = 1, all other alpha_g = 0.
* ``shuffled_address``:  alpha_pi(y) = 1 for a fixed seeded derangement pi.

The global branch deliberately uses 1/G per address, rather than a full unit at
every address, so the total effective write mass is matched to the one-hot
correct and shuffled branches. Thus any difference is attributable more cleanly
to where feedback is written rather than to a G-fold difference in update mass.

Interpretation boundary
-----------------------
This remains an offline source-side structural-credit diagnostic. It does not
establish automatic factor discovery, semantic/causal factors, a deployable
query policy, calibrated online probabilities, final C1/C2, or end-to-end
anomaly-detection improvement. Outer ABMG target categories remain untouched.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    from abmg_sequential_local_update_audit import (
        _accumulate_effect,
        _aggregate_effects,
        _new_effect_accumulator,
        effect_summary,
        init_memory,
        select_probe_indices,
        stratified_stream_sentinel_split,
        true_margin,
        uniform_query_positions,
    )
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
    from scripts.abmg_sequential_local_update_audit import (
        _accumulate_effect,
        _aggregate_effects,
        _new_effect_accumulator,
        effect_summary,
        init_memory,
        select_probe_indices,
        stratified_stream_sentinel_split,
        true_margin,
        uniform_query_positions,
    )

MODELS: Tuple[str, ...] = ("diag_shrunk", "bayes_predictive")
BRANCHES: Tuple[str, ...] = (
    "no_update",
    "global_broadcast",
    "correct_address",
    "shuffled_address",
)
GLOBAL_CREDIT_WEIGHT = 1.0 / float(len(CORE4))


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


def _load_artifact(path: Path) -> Dict[str, Any]:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if artifact.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected artifact schema={artifact.get('schema')!r}")
    return artifact


def make_derangement(labels: Sequence[str], seed: int) -> Dict[str, str]:
    """Return a deterministic seeded permutation with no fixed points."""
    labels = tuple(str(x) for x in labels)
    if len(labels) < 2:
        raise ValueError("derangement requires at least two labels")
    rng = np.random.default_rng(int(seed))
    base = np.asarray(labels, dtype=object)
    for _ in range(10_000):
        perm = rng.permutation(base)
        if all(str(a) != str(b) for a, b in zip(base, perm)):
            return {str(a): str(b) for a, b in zip(base, perm)}
    raise RuntimeError("failed to generate derangement")


def _normalized_vector(vector: np.ndarray) -> np.ndarray:
    z = np.asarray(vector, dtype=np.float64).reshape(1, -1)
    return _l2(z)[0]


def update_memory_weighted(
    memory: Dict[str, Dict[str, Any]],
    credit: Mapping[str, float],
    vector: np.ndarray,
) -> None:
    """Apply one normalized observation with an arbitrary non-negative credit vector.

    ``n`` is treated as an effective sample count and may therefore be fractional.
    The update is the sufficient-statistic analogue of a fractional Bayesian write:

        n_g   <- n_g + alpha_g
        sum_g <- sum_g + alpha_g * z
    """
    z = _normalized_vector(vector)
    for label in CORE4:
        w = float(credit.get(label, 0.0))
        if w < 0.0 or not np.isfinite(w):
            raise ValueError(f"invalid credit weight for {label}: {w}")
        if w == 0.0:
            continue
        memory[label]["n"] = float(memory[label]["n"]) + w
        memory[label]["sum"] = (
            np.asarray(memory[label]["sum"], dtype=np.float64) + w * z
        )


def credit_vector(
    branch: str,
    true_label: str,
    shuffle_map: Mapping[str, str],
) -> Dict[str, float]:
    true_label = str(true_label)
    if true_label not in CORE4:
        raise ValueError(f"unknown factor={true_label!r}")
    if branch == "no_update":
        return {g: 0.0 for g in CORE4}
    if branch == "global_broadcast":
        return {g: GLOBAL_CREDIT_WEIGHT for g in CORE4}
    if branch == "correct_address":
        return {g: 1.0 if g == true_label else 0.0 for g in CORE4}
    if branch == "shuffled_address":
        dst = str(shuffle_map[true_label])
        if dst == true_label:
            raise ValueError("shuffled_address requires a derangement")
        return {g: 1.0 if g == dst else 0.0 for g in CORE4}
    raise ValueError(f"unknown branch={branch!r}")


def changed_factor_states(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    changed: List[str] = []
    for label in CORE4:
        n_before = float(before[label]["n"])
        n_after = float(after[label]["n"])
        s_before = np.asarray(before[label]["sum"])
        s_after = np.asarray(after[label]["sum"])
        if n_before != n_after or not np.array_equal(s_before, s_after):
            changed.append(label)
    return tuple(changed)


def expected_changed_factors(
    branch: str,
    true_label: str,
    shuffle_map: Mapping[str, str],
) -> Tuple[str, ...]:
    if branch == "no_update":
        return tuple()
    if branch == "global_broadcast":
        return tuple(CORE4)
    if branch == "correct_address":
        return (str(true_label),)
    if branch == "shuffled_address":
        return (str(shuffle_map[str(true_label)]),)
    raise ValueError(f"unknown branch={branch!r}")


def apply_branch_update(
    memory: Dict[str, Dict[str, Any]],
    branch: str,
    true_label: str,
    vector: np.ndarray,
    shuffle_map: Mapping[str, str],
) -> Tuple[str, ...]:
    before = deepcopy(memory)
    update_memory_weighted(memory, credit_vector(branch, true_label, shuffle_map), vector)
    changed = changed_factor_states(before, memory)
    expected = expected_changed_factors(branch, true_label, shuffle_map)
    if changed != expected:
        raise AssertionError(
            f"branch={branch} true={true_label}: changed={changed}, expected={expected}"
        )
    return changed


def score_weighted_memory(
    X_eval: np.ndarray,
    memory: Mapping[str, Mapping[str, Any]],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
    model: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score memory while preserving fractional effective counts."""
    if model not in MODELS:
        raise ValueError(f"unknown model={model!r}")
    if float(kappa0) <= 0.0:
        raise ValueError("kappa0 must be > 0")
    xv = _l2(X_eval)
    mu0 = np.asarray(shared_prior["mu0"], dtype=np.float64)
    var = np.asarray(shared_prior["var"], dtype=np.float64)
    cols = []
    for label in CORE4:
        n = float(memory[label]["n"])
        sum_x = np.asarray(memory[label]["sum"], dtype=np.float64)
        kappa_n = float(kappa0) + n
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


def evaluate_state(
    X: np.ndarray,
    y: np.ndarray,
    memory: Mapping[str, Mapping[str, Any]],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
    model: str,
) -> Dict[str, Any]:
    pred, scores = score_weighted_memory(X, memory, shared_prior, kappa0, model)
    margins = true_margin(y, scores)
    per_margin = {}
    for label in CORE4:
        mask = np.asarray(y == label, dtype=bool)
        per_margin[label] = float(margins[mask].mean()) if np.any(mask) else float("nan")
    return {
        "metrics": _metrics(y, pred),
        "per_class_f1": _per_class_f1(y, pred),
        "mean_true_minus_best_wrong_margin": float(margins.mean()),
        "per_class_margin": per_margin,
        "pred": pred,
        "scores": scores,
    }


def _checkpoint_record(
    X_sentinel: np.ndarray,
    y_sentinel: np.ndarray,
    memories: Mapping[str, Mapping[str, Mapping[str, Any]]],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
    n_feedback: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_feedback": int(n_feedback), "branches": {}}
    for branch in BRANCHES:
        brow: Dict[str, Any] = {
            "factor_effective_counts": {
                y: float(memories[branch][y]["n"]) for y in CORE4
            },
            "models": {},
        }
        preds: Dict[str, np.ndarray] = {}
        for model in MODELS:
            ev = evaluate_state(
                X_sentinel,
                y_sentinel,
                memories[branch],
                shared_prior,
                kappa0,
                model,
            )
            preds[model] = ev.pop("pred")
            ev.pop("scores")
            brow["models"][model] = ev
        brow["shrunk_vs_bayes_prediction_agreement"] = float(
            np.mean(preds["diag_shrunk"] == preds["bayes_predictive"])
        )
        out["branches"][branch] = brow
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
    probe_idx = select_probe_indices(y_sentinel, int(probe_per_label), split_seed + 123)
    X_probe, y_probe = X_sentinel[probe_idx], y_sentinel[probe_idx]

    run_seed = int(seed) + 100_000 * int(source_cv_fold) + int(repeat)
    support = category_diverse_support_indices(
        yt, ct, CORE4, int(initial_shots), run_seed + 17
    )
    base_memory = init_memory(xt, support)
    memories = {branch: deepcopy(base_memory) for branch in BRANCHES}
    shared_prior = fit_shared_diag_prior(X[train_all])
    shuffle_map = make_derangement(CORE4, run_seed + 303)

    rng = np.random.default_rng(run_seed + 101)
    order = rng.permutation(len(X_stream0)).astype(np.int64)
    X_stream, y_stream = X_stream0[order], y_stream0[order]
    query_pos = uniform_query_positions(len(X_stream), int(query_budget), run_seed + 202)
    query_set = set(query_pos.tolist())

    effect_acc = {
        branch: {model: _new_effect_accumulator() for model in MODELS}
        for branch in BRANCHES
    }
    update_integrity: Dict[str, List[bool]] = {branch: [] for branch in BRANCHES}
    feedback_label_counts = {y: 0 for y in CORE4}
    preq_true: Dict[str, Dict[str, List[str]]] = {
        branch: {model: [] for model in MODELS} for branch in BRANCHES
    }
    preq_pred: Dict[str, Dict[str, List[str]]] = {
        branch: {model: [] for model in MODELS} for branch in BRANCHES
    }
    checkpoint_set = set(int(x) for x in checkpoints)
    checkpoint_rows: List[Dict[str, Any]] = []
    if 0 in checkpoint_set:
        checkpoint_rows.append(
            _checkpoint_record(
                X_sentinel, y_sentinel, memories, shared_prior, kappa0, 0
            )
        )

    update_events: List[Dict[str, Any]] = []
    n_feedback = 0
    for pos in range(len(X_stream)):
        z = X_stream[pos : pos + 1]
        y = str(y_stream[pos])

        for branch in BRANCHES:
            for model in MODELS:
                pred, _ = score_weighted_memory(
                    z, memories[branch], shared_prior, kappa0, model
                )
                preq_true[branch][model].append(y)
                preq_pred[branch][model].append(str(pred[0]))

        if pos not in query_set:
            continue

        event: Dict[str, Any] = {
            "feedback_index": int(n_feedback + 1),
            "stream_position": int(pos),
            "true_factor": y,
            "shuffled_destination": str(shuffle_map[y]),
            "branches": {},
        }

        for branch in BRANCHES:
            probe_before = {}
            for model in MODELS:
                pb, sb = score_weighted_memory(
                    X_probe, memories[branch], shared_prior, kappa0, model
                )
                probe_before[model] = (pb, sb)

            before = deepcopy(memories[branch])
            changed = apply_branch_update(
                memories[branch], branch, y, z[0], shuffle_map
            )
            expected = expected_changed_factors(branch, y, shuffle_map)
            integrity = changed == expected
            update_integrity[branch].append(bool(integrity))
            event["branches"][branch] = {
                "credit": credit_vector(branch, y, shuffle_map),
                "changed_factors": list(changed),
                "update_integrity": bool(integrity),
                "total_effective_count_delta": float(
                    sum(float(memories[branch][g]["n"]) - float(before[g]["n"]) for g in CORE4)
                ),
            }

            for model in MODELS:
                pa, sa = score_weighted_memory(
                    X_probe, memories[branch], shared_prior, kappa0, model
                )
                pb, sb = probe_before[model]
                eff = effect_summary(y_probe, pb, sb, pa, sa)
                # Row key is the TRUE feedback factor in every branch. This keeps
                # diagonal/off-diagonal effects directly comparable across branches,
                # including shuffled writes whose destination is intentionally wrong.
                _accumulate_effect(effect_acc[branch][model], y, eff)

        feedback_label_counts[y] += 1
        n_feedback += 1
        update_events.append(event)

        if n_feedback in checkpoint_set:
            checkpoint_rows.append(
                _checkpoint_record(
                    X_sentinel,
                    y_sentinel,
                    memories,
                    shared_prior,
                    kappa0,
                    n_feedback,
                )
            )

    prequential: Dict[str, Dict[str, Any]] = {}
    for branch in BRANCHES:
        prequential[branch] = {}
        for model in MODELS:
            yt_arr = np.asarray(preq_true[branch][model], dtype=object)
            yp_arr = np.asarray(preq_pred[branch][model], dtype=object)
            prequential[branch][model] = {
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
        "query_positions": [int(x) for x in query_pos.tolist()],
        "query_positions_label_independent": True,
        "feedback_label_counts": feedback_label_counts,
        "shuffle_map": shuffle_map,
        "initial_support_categories": {
            y: sorted(set(ct[idx].tolist())) for y, idx in support.items()
        },
        "update_integrity_pass_fraction": {
            branch: float(np.mean(update_integrity[branch]))
            if update_integrity[branch]
            else float("nan")
            for branch in BRANCHES
        },
        "checkpoints": checkpoint_rows,
        "prequential": prequential,
        "effect_accumulators": effect_acc,
        "update_events": update_events,
    }


def _aggregate_checkpoints(
    runs: Sequence[Dict[str, Any]], checkpoints: Sequence[int]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cp in checkpoints:
        cp_rows = []
        for run in runs:
            match = [x for x in run["checkpoints"] if int(x["n_feedback"]) == int(cp)]
            if match:
                cp_rows.append((run, match[0]))
        row: Dict[str, Any] = {"n_runs": len(cp_rows), "branches": {}}

        for branch in BRANCHES:
            brow: Dict[str, Any] = {"models": {}}
            for model in MODELS:
                mrow: Dict[str, Any] = {}
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    m, s = _finite_mean_std(
                        x[1]["branches"][branch]["models"][model]["metrics"][metric]
                        for x in cp_rows
                    )
                    mrow[f"{metric}_mean"] = m
                    mrow[f"{metric}_std"] = s

                per_class = {}
                for label in CORE4:
                    m, s = _finite_mean_std(
                        x[1]["branches"][branch]["models"][model]["per_class_f1"][label]
                        for x in cp_rows
                    )
                    per_class[label] = {"f1_mean": m, "f1_std": s}
                mrow["per_class"] = per_class

                mm, ms = _finite_mean_std(
                    x[1]["branches"][branch]["models"][model][
                        "mean_true_minus_best_wrong_margin"
                    ]
                    for x in cp_rows
                )
                mrow["margin_mean"] = mm
                mrow["margin_std"] = ms

                own_deltas = []
                vs_no_update = []
                for run, current in cp_rows:
                    base = [
                        x for x in run["checkpoints"] if int(x["n_feedback"]) == 0
                    ]
                    if base:
                        own_deltas.append(
                            current["branches"][branch]["models"][model]["metrics"]["macro_f1"]
                            - base[0]["branches"][branch]["models"][model]["metrics"]["macro_f1"]
                        )
                    vs_no_update.append(
                        current["branches"][branch]["models"][model]["metrics"]["macro_f1"]
                        - current["branches"]["no_update"]["models"][model]["metrics"]["macro_f1"]
                    )
                dm, ds = _finite_mean_std(own_deltas)
                nm, ns = _finite_mean_std(vs_no_update)
                mrow["macro_f1_delta_from_initial_mean"] = dm
                mrow["macro_f1_delta_from_initial_std"] = ds
                mrow["macro_f1_delta_vs_no_update_mean"] = nm
                mrow["macro_f1_delta_vs_no_update_std"] = ns
                brow["models"][model] = mrow

            for label in CORE4:
                cm, cs = _finite_mean_std(
                    x[1]["branches"][branch]["factor_effective_counts"][label]
                    for x in cp_rows
                )
                brow.setdefault("factor_effective_counts", {})[label] = {
                    "mean": cm,
                    "std": cs,
                }
            row["branches"][branch] = brow
        out[str(cp)] = row
    return out


def _aggregate_interference(
    runs: Sequence[Dict[str, Any]], branch: str, model: str
) -> Dict[str, Any]:
    # Reuse the previous audit's tested aggregation implementation by presenting
    # the selected branch in its expected shape.
    shim = [
        {"effect_accumulators": run["effect_accumulators"][branch]}
        for run in runs
    ]
    return _aggregate_effects(shim, model)


def _paired_final_contrasts(
    runs: Sequence[Dict[str, Any]], final_checkpoint: int
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pairs = (
        ("correct_address", "no_update"),
        ("correct_address", "global_broadcast"),
        ("correct_address", "shuffled_address"),
        ("global_broadcast", "no_update"),
        ("shuffled_address", "no_update"),
    )
    for model in MODELS:
        mrow: Dict[str, Any] = {}
        for lhs, rhs in pairs:
            vals = []
            for run in runs:
                match = [
                    x for x in run["checkpoints"]
                    if int(x["n_feedback"]) == int(final_checkpoint)
                ]
                if not match:
                    continue
                cp = match[0]
                lv = cp["branches"][lhs]["models"][model]["metrics"]["macro_f1"]
                rv = cp["branches"][rhs]["models"][model]["metrics"]["macro_f1"]
                vals.append(float(lv) - float(rv))
            mean, std = _finite_mean_std(vals)
            mrow[f"{lhs}_minus_{rhs}"] = {
                "macro_f1_mean": mean,
                "macro_f1_std": std,
                "n_runs": len(vals),
            }
        out[model] = mrow
    return out


def aggregate_runs(
    runs: Sequence[Dict[str, Any]], checkpoints: Sequence[int]
) -> Dict[str, Any]:
    final_cp = int(max(checkpoints))
    out: Dict[str, Any] = {
        "n_runs": len(runs),
        "checkpoints": _aggregate_checkpoints(runs, checkpoints),
        "interference": {
            branch: {
                model: _aggregate_interference(runs, branch, model)
                for model in MODELS
            }
            for branch in BRANCHES
        },
        "prequential": {},
        "update_integrity": {},
        "paired_final_contrasts": _paired_final_contrasts(runs, final_cp),
    }

    for branch in BRANCHES:
        m, s = _finite_mean_std(
            run["update_integrity_pass_fraction"][branch] for run in runs
        )
        out["update_integrity"][branch] = {"pass_fraction_mean": m, "std": s}
        out["prequential"][branch] = {}
        for model in MODELS:
            mrow = {}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                mm, ms = _finite_mean_std(
                    run["prequential"][branch][model]["metrics"][metric]
                    for run in runs
                )
                mrow[f"{metric}_mean"] = mm
                mrow[f"{metric}_std"] = ms
            out["prequential"][branch][model] = mrow

    for label in CORE4:
        m, s = _finite_mean_std(run["feedback_label_counts"][label] for run in runs)
        out.setdefault("feedback_label_counts", {})[label] = {"mean": m, "std": s}
    return out


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
    if float(args.bayes_kappa0) <= 0.0:
        raise ValueError("bayes_kappa0 must be > 0")

    items = artifact["items"]
    cv_groups = artifact["source_cv_groups"]
    result: Dict[str, Any] = {
        "schema": "abmg.four_branch_structural_credit_audit.v1",
        "outer_fold": int(args.fold),
        "source_artifact": str(artifact_path),
        "target_categories_untouched": artifact["target_categories_untouched"],
        "core4_labels": list(CORE4),
        "branches": list(BRANCHES),
        "representations_requested": list(requested),
        "protocol": {
            "initial_shots_per_factor_from_training_products": int(args.initial_shots),
            "query_budget": int(args.query_budget),
            "checkpoints": list(checkpoints),
            "repeats_per_source_cv_fold": int(args.repeats),
            "sentinel_fraction": float(args.sentinel_fraction),
            "probe_per_label": int(args.probe_per_label),
            "query_policy": "uniform random stream positions; label/model independent and shared by all branches",
            "prequential_rule": "every branch predicts before shared feedback/update on every stream item",
            "shared_initial_state": True,
            "shared_stream_and_queries": True,
            "global_credit_rule": f"uniform alpha_g=1/G={GLOBAL_CREDIT_WEIGHT:.6f}; total effective write mass=1",
            "correct_credit_rule": "one-hot at verified factor; total effective write mass=1",
            "shuffled_credit_rule": "one-hot at fixed seeded derangement of verified factor; total effective write mass=1",
            "no_update_rule": "feedback revealed at identical query positions but persistent factor state is unchanged",
            "factor_state": ["effective_n_g", "weighted_sum_g"],
            "shared_state": "label-free source-training-product mean and diagonal variance; frozen during stream",
            "models": list(MODELS),
            "bayes_kappa0": float(args.bayes_kappa0),
            "sentinel_labels_never_used_for_updates": True,
            "effect_matrix_rows": "verified true factor, even in shuffled branch",
            "effect_matrix_columns": "sentinel evaluation factor",
        },
        "primary_reading": (
            "Compare correct_address against global_broadcast and shuffled_address under identical supervision. "
            "Support for structural credit requires retained/improved same-factor benefit with lower collateral "
            "interference; shuffled writes should weaken or reverse that advantage."
        ),
        "interpretation_boundary": (
            "Source-side causal control over write location only. It does not establish automatic factor discovery, "
            "semantic/causal factors, deployable querying, calibrated online probabilities, final C1/C2, or "
            "end-to-end anomaly-detection improvement."
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
                runs.append(
                    run_one_sequence(
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
                )
        agg = aggregate_runs(runs, checkpoints)
        result["representations"][key] = {"aggregate": agg, "runs": runs}

        print("  update-integrity pass fractions:")
        for branch in BRANCHES:
            val = agg["update_integrity"][branch]["pass_fraction_mean"]
            print(f"    {branch}: {val:.4f}")

        for cp in checkpoints:
            row = agg["checkpoints"][str(cp)]
            parts = []
            for branch in BRANCHES:
                m = row["branches"][branch]["models"]["diag_shrunk"]
                parts.append(
                    f"{branch} F1={m['macro_f1_mean']:.4f} "
                    f"vs-no={m['macro_f1_delta_vs_no_update_mean']:+.4f}"
                )
            print(f"  feedback={cp} diag_shrunk: " + " | ".join(parts))

        print("  diag_shrunk structural effects:")
        for branch in BRANCHES:
            inter = agg["interference"][branch]["diag_shrunk"]
            print(
                f"    {branch}: diag_margin={inter['diagonal_margin_delta_mean']:+.4f}, "
                f"offdiag_abs={inter['offdiagonal_abs_margin_delta_mean']:.4f}, "
                f"offdiag_harm_flip={inter['offdiagonal_harmful_flip_rate_mean']:.4f}, "
                f"locality_ratio={inter['diagonal_to_offdiagonal_abs_ratio']:.3f}"
            )

        contrasts = agg["paired_final_contrasts"]["diag_shrunk"]
        print(f"  paired final contrasts at feedback={max(checkpoints)}:")
        for name, vals in contrasts.items():
            print(
                f"    {name}: macro-F1 {vals['macro_f1_mean']:+.4f} "
                f"+- {vals['macro_f1_std']:.4f}"
            )

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_path = out_dir / "four_branch_structural_credit_audit.json"
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
        default="outputs/four_branch_structural_credit_audit",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
