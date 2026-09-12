#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse-feedback memory/addressability audit for ABMG.

This is a follow-up to ``abmg_prototype_addressability_audit.py``.  It consumes
an existing ``abmg.factorizability.features.v1`` artifact and does NOT rerun
DINO or open Real-IAD images.

The experiment asks a deliberately narrow question:

    Given the same 1/2/4/8 verified Core-4 feedback examples, does a richer
    external memory route held-out product categories better than one centroid
    per defect family?

Three memory mechanisms are compared with exactly the same support examples:

1. single_centroid
   One L2-normalized mean prototype per factor.  This is the previous baseline.

2. exemplar_max
   Every verified example remains an address prototype.  The class score is the
   maximum cosine similarity to any verified exemplar of that factor.  This is
   a minimal multi-prototype memory: no fitting, no SGD, and no new latent model.

3. bayes_shared_diag
   A Normal-Normal empirical-Bayes class-conditional memory in L2-normalized
   frozen evidence space.  A label-free shared diagonal observation variance
   and prior mean are estimated from the source-CV training-category evidence
   pool.  For each factor g, sparse feedback updates only two sufficient stats:

       n_g <- n_g + 1
       sum_g <- sum_g + z

   with prior

       mu_g ~ N(mu0, Sigma / kappa0)

   and likelihood

       z | mu_g ~ N(mu_g, Sigma).

   Hence

       kappa_g = kappa0 + n_g
       mu_g*   = (kappa0*mu0 + sum_g) / kappa_g

   and the posterior-predictive routing score uses

       z_new | D_g ~ N(mu_g*, Sigma * (1 + 1/kappa_g)).

   Equal factor priors are used.  The softmax of predictive log scores is also
   recorded as an uncertainty-aware responsibility vector.  ``kappa0`` is fixed
   before evaluation and must not be tuned on held-out products.

Protocol
--------
* Core-4 labels: AK / HS / QS / ZW.
* Product-category-held-out source CV is inherited from the artifact.
* Category-diverse few-shot supports are sampled from training products only.
* The same support examples are used for all three real-memory mechanisms.
* A within-training-category label-shuffle control is evaluated for every model.
* Per-class F1 and true-normalized confusion matrices are reported at every shot.
* Outer ABMG target categories remain untouched.

Interpretation boundary
-----------------------
This is an offline diagnostic of external-memory addressability.  It does not
establish semantic/causal factors, final C1/C2, or end-to-end online benefit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

try:  # direct script execution: python scripts/...
    from abmg_prototype_addressability_audit import (
        CORE4,
        EXPECTED_SCHEMA,
        _finite_mean_std,
        _l2,
        _metrics,
        category_diverse_support_indices,
        parse_int_list,
        parse_str_list,
        prototype_predict,
        within_category_label_shuffle,
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
        prototype_predict,
        within_category_label_shuffle,
    )


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def _softmax(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.exp(np.clip(s, -700.0, 700.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    vals = f1_score(
        y_true,
        y_pred,
        labels=list(CORE4),
        average=None,
        zero_division=0,
    )
    return {str(label): float(vals[i]) for i, label in enumerate(CORE4)}


def _confusion_true_normalized(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(CORE4),
        normalize="true",
    ).astype(np.float64)


def score_routing_diagnostics(
    y_true: np.ndarray,
    scores: np.ndarray,
    label_order: Sequence[str] = CORE4,
) -> Dict[str, float]:
    """Rank/margin diagnostics valid for cosine or log-predictive scores."""
    labels = tuple(str(x) for x in label_order)
    label_to_col = {y: i for i, y in enumerate(labels)}
    true_cols = np.asarray([label_to_col[str(y)] for y in y_true], dtype=np.int64)
    rows = np.arange(len(y_true))
    true_score = scores[rows, true_cols]
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[rows, true_cols] = -np.inf
    best_wrong = masked.max(axis=1)
    margin = true_score - best_wrong
    k = min(2, scores.shape[1])
    topk = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top2 = np.asarray(
        [true_cols[i] in set(topk[i].tolist()) for i in range(len(y_true))],
        dtype=np.float64,
    )
    return {
        "mean_true_minus_best_wrong_margin": float(np.mean(margin)),
        "median_true_minus_best_wrong_margin": float(np.median(margin)),
        "top2_recall": float(np.mean(top2)),
    }


def exemplar_max_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    support_by_label: Mapping[str, np.ndarray],
    label_order: Sequence[str] = CORE4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-prototype routing: class score = max cosine to its exemplars."""
    xt = _l2(X_train)
    xv = _l2(X_test)
    scores = []
    for label in label_order:
        idx = np.asarray(support_by_label[str(label)], dtype=np.int64)
        bank = xt[idx]
        scores.append((xv @ bank.T).max(axis=1))
    score_mat = np.stack(scores, axis=1)
    pred = np.asarray(label_order, dtype=object)[score_mat.argmax(axis=1)]
    return pred, score_mat


def fit_shared_diag_prior(X_pool: np.ndarray, variance_floor: float = 1e-6) -> Dict[str, np.ndarray]:
    """Fit label-free empirical-Bayes background statistics on train products."""
    x = _l2(X_pool)
    mu0 = x.mean(axis=0)
    var = x.var(axis=0, ddof=1 if len(x) > 1 else 0)
    # Relative floor prevents numerically tiny coordinates dominating likelihood.
    positive = var[var > 0]
    ref = float(np.median(positive)) if positive.size else 1.0
    floor = max(float(variance_floor), ref * 1e-3)
    var = np.maximum(var, floor)
    return {"mu0": mu0.astype(np.float64), "var": var.astype(np.float64)}


def bayes_shared_diag_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    support_by_label: Mapping[str, np.ndarray],
    shared_prior: Mapping[str, np.ndarray],
    *,
    kappa0: float,
    label_order: Sequence[str] = CORE4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal-Normal posterior-predictive routing with shared diagonal Sigma."""
    if float(kappa0) <= 0:
        raise ValueError("kappa0 must be > 0")
    xt = _l2(X_train)
    xv = _l2(X_test)
    mu0 = np.asarray(shared_prior["mu0"], dtype=np.float64)
    var = np.asarray(shared_prior["var"], dtype=np.float64)
    scores = []

    for label in label_order:
        idx = np.asarray(support_by_label[str(label)], dtype=np.int64)
        n = int(len(idx))
        if n <= 0:
            raise ValueError(f"No support for label={label}")
        sum_x = xt[idx].sum(axis=0)
        kappa_n = float(kappa0) + float(n)
        mu_n = (float(kappa0) * mu0 + sum_x) / kappa_n
        pred_var = var * (1.0 + 1.0 / kappa_n)
        diff = xv - mu_n[None, :]
        # Equal class priors. Constants common to every class are harmless, but
        # the predictive log-determinant is retained for correctness.
        logp = -0.5 * np.sum(
            (diff * diff) / pred_var[None, :] + np.log(pred_var[None, :]),
            axis=1,
        )
        scores.append(logp)

    score_mat = np.stack(scores, axis=1)
    responsibilities = _softmax(score_mat)
    pred = np.asarray(label_order, dtype=object)[score_mat.argmax(axis=1)]
    return pred, score_mat, responsibilities


def bayes_uncertainty_diagnostics(
    y_true: np.ndarray,
    responsibilities: np.ndarray,
    label_order: Sequence[str] = CORE4,
) -> Dict[str, float]:
    labels = tuple(str(x) for x in label_order)
    label_to_col = {y: i for i, y in enumerate(labels)}
    true_cols = np.asarray([label_to_col[str(y)] for y in y_true], dtype=np.int64)
    rows = np.arange(len(y_true))
    p = np.clip(np.asarray(responsibilities, dtype=np.float64), 1e-12, 1.0)
    true_p = p[rows, true_cols]
    entropy = -np.sum(p * np.log(p), axis=1)
    onehot = np.zeros_like(p)
    onehot[rows, true_cols] = 1.0
    brier = np.sum((p - onehot) ** 2, axis=1)
    pred_cols = p.argmax(axis=1)
    correct = pred_cols == true_cols
    return {
        "mean_true_responsibility": float(np.mean(true_p)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_entropy_correct": float(np.mean(entropy[correct])) if np.any(correct) else float("nan"),
        "mean_entropy_incorrect": float(np.mean(entropy[~correct])) if np.any(~correct) else float("nan"),
        "negative_log_likelihood": float(np.mean(-np.log(true_p))),
        "brier_score": float(np.mean(brier)),
    }


def _evaluate_one(
    model: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    support: Mapping[str, np.ndarray],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
) -> Dict[str, Any]:
    if model == "single_centroid":
        pred, scores = prototype_predict(X_train, X_test, support, CORE4)
        uncertainty = None
    elif model == "exemplar_max":
        pred, scores = exemplar_max_predict(X_train, X_test, support, CORE4)
        uncertainty = None
    elif model == "bayes_shared_diag":
        pred, scores, resp = bayes_shared_diag_predict(
            X_train,
            X_test,
            support,
            shared_prior,
            kappa0=float(kappa0),
            label_order=CORE4,
        )
        uncertainty = bayes_uncertainty_diagnostics(y_test, resp, CORE4)
    else:
        raise ValueError(f"Unknown model={model!r}")

    out: Dict[str, Any] = {
        "metrics": _metrics(y_test, pred),
        "per_class_f1": _per_class_f1(y_test, pred),
        "confusion_true_normalized": _confusion_true_normalized(y_test, pred).tolist(),
        "routing": score_routing_diagnostics(y_test, scores, CORE4),
    }
    if uncertainty is not None:
        out["uncertainty"] = uncertainty
    return out


def _aggregate_model_runs(runs: Sequence[Dict[str, Any]], model: str) -> Dict[str, Any]:
    agg: Dict[str, Any] = {"n_runs": len(runs)}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        m, s = _finite_mean_std(r[model]["real"]["metrics"][metric] for r in runs)
        agg[f"real_{metric}_mean"] = m
        agg[f"real_{metric}_std"] = s
        sm, ss = _finite_mean_std(r[model]["shuffle"]["metrics"][metric] for r in runs)
        agg[f"shuffle_{metric}_mean"] = sm
        agg[f"shuffle_{metric}_std"] = ss
    agg["macro_f1_gain_over_within_category_shuffle"] = float(
        agg["real_macro_f1_mean"] - agg["shuffle_macro_f1_mean"]
    )

    per_class: Dict[str, Any] = {}
    for label in CORE4:
        m, s = _finite_mean_std(r[model]["real"]["per_class_f1"][label] for r in runs)
        sm, ss = _finite_mean_std(r[model]["shuffle"]["per_class_f1"][label] for r in runs)
        per_class[label] = {
            "real_f1_mean": m,
            "real_f1_std": s,
            "shuffle_f1_mean": sm,
            "shuffle_f1_std": ss,
            "gain_over_shuffle": float(m - sm),
        }
    agg["per_class"] = per_class

    mats = np.asarray(
        [r[model]["real"]["confusion_true_normalized"] for r in runs],
        dtype=np.float64,
    )
    agg["confusion_labels"] = list(CORE4)
    agg["confusion_true_normalized_mean"] = mats.mean(axis=0).tolist()
    agg["confusion_true_normalized_std"] = mats.std(axis=0).tolist()

    for metric in (
        "mean_true_minus_best_wrong_margin",
        "median_true_minus_best_wrong_margin",
        "top2_recall",
    ):
        m, s = _finite_mean_std(r[model]["real"]["routing"][metric] for r in runs)
        agg[f"routing_{metric}_mean"] = m
        agg[f"routing_{metric}_std"] = s

    if model == "bayes_shared_diag":
        for metric in (
            "mean_true_responsibility",
            "mean_entropy",
            "mean_entropy_correct",
            "mean_entropy_incorrect",
            "negative_log_likelihood",
            "brier_score",
        ):
            m, s = _finite_mean_std(r[model]["real"]["uncertainty"][metric] for r in runs)
            agg[f"uncertainty_{metric}_mean"] = m
            agg[f"uncertainty_{metric}_std"] = s
    return agg


def evaluate_memory_models(
    X: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    cv_groups: Sequence[Sequence[str]],
    shot_values: Sequence[int],
    repeats: int,
    seed: int,
    kappa0: float,
) -> Dict[str, Any]:
    models = ("single_centroid", "exemplar_max", "bayes_shared_diag")
    all_categories = set(categories.tolist())
    wanted = set(CORE4)
    by_shot: Dict[str, Any] = {}

    for shots in shot_values:
        runs = []
        for fi, group in enumerate(cv_groups):
            test_cats = set(str(x) for x in group)
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
            xv, yv = X[test_core], labels[test_core]
            if len(set(yt.tolist())) != len(CORE4) or len(set(yv.tolist())) != len(CORE4):
                continue

            # Label-free background state.  It sees train-category evidence only
            # and is fixed across repeats/memory mechanisms for this CV fold.
            shared_prior = fit_shared_diag_prior(X[train_all])

            for ri in range(int(repeats)):
                run_seed = int(seed) + 100_000 * int(shots) + 10_000 * fi + ri
                support = category_diverse_support_indices(
                    yt, ct, CORE4, int(shots), run_seed
                )
                shuffled_yt = within_category_label_shuffle(
                    yt, ct, run_seed + 50_000_000
                )
                shuffled_support = category_diverse_support_indices(
                    shuffled_yt,
                    ct,
                    CORE4,
                    int(shots),
                    run_seed + 60_000_000,
                )

                row: Dict[str, Any] = {
                    "source_cv_fold": fi,
                    "repeat": ri,
                    "seed": run_seed,
                    "test_categories": sorted(test_cats),
                    "n_train_core4": int(train_core.sum()),
                    "n_train_background": int(train_all.sum()),
                    "n_test_core4": int(test_core.sum()),
                    "n_feedback_examples": int(shots) * len(CORE4),
                    "support_categories": {
                        label: sorted(set(ct[idx].tolist()))
                        for label, idx in support.items()
                    },
                }
                for model in models:
                    row[model] = {
                        "real": _evaluate_one(
                            model, xt, xv, yv, support, shared_prior, float(kappa0)
                        ),
                        "shuffle": _evaluate_one(
                            model,
                            xt,
                            xv,
                            yv,
                            shuffled_support,
                            shared_prior,
                            float(kappa0),
                        ),
                    }
                runs.append(row)

        model_agg = {model: _aggregate_model_runs(runs, model) for model in models}
        baseline = model_agg["single_centroid"]["real_macro_f1_mean"]
        paired = {
            model: float(model_agg[model]["real_macro_f1_mean"] - baseline)
            for model in models
            if model != "single_centroid"
        }
        by_shot[str(shots)] = {
            "shots_per_label": int(shots),
            "n_feedback_examples": int(shots) * len(CORE4),
            "n_runs": len(runs),
            "models": model_agg,
            "macro_f1_delta_vs_single_centroid": paired,
            "runs": runs,
        }
    return by_shot


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
        raise KeyError(
            f"Representation(s) not found: {missing}. Available={sorted(artifact['representations'])}"
        )
    if float(args.bayes_kappa0) <= 0:
        raise ValueError("--bayes-kappa0 must be > 0")

    shots = parse_int_list(args.shots)
    items = artifact["items"]
    cv_groups = artifact["source_cv_groups"]
    result: Dict[str, Any] = {
        "schema": "abmg.memory_addressability_audit.v1",
        "outer_fold": int(args.fold),
        "source_artifact": str(artifact_path),
        "target_categories_untouched": artifact["target_categories_untouched"],
        "core4_labels": list(CORE4),
        "representations_requested": list(requested),
        "protocol": {
            "shots_per_label": list(shots),
            "repeats_per_cv_fold": int(args.repeats),
            "support_selection": "same category-diverse real supports for all memory mechanisms",
            "shuffle_control": "within-training-category label shuffle",
            "models": {
                "single_centroid": "one normalized mean per factor",
                "exemplar_max": "all verified examples retained; max cosine per factor",
                "bayes_shared_diag": "Normal-Normal posterior predictive with label-free shared diagonal Sigma",
            },
            "bayes_kappa0": float(args.bayes_kappa0),
            "bayes_shared_background": "all evidence vectors from source-CV training product categories; labels unused",
            "bayes_factor_state": ["n_g", "sum_g"],
        },
        "interpretation_boundary": (
            "Offline source-side diagnostic only. Positive results support sparse-feedback external-memory "
            "addressability; they do not prove semantic/causal factors, final C1/C2, or end-to-end online gain."
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
        core_count = int(np.sum([y in set(CORE4) for y in labels]))
        print(f"Evaluating {key}: n={len(X)} d={X.shape[1]} core4={core_count}")
        by_shot = evaluate_memory_models(
            X,
            categories,
            labels,
            cv_groups,
            shots,
            int(args.repeats),
            int(args.seed),
            float(args.bayes_kappa0),
        )
        result["representations"][key] = {"by_shot": by_shot}

        for s in shots:
            row = by_shot[str(s)]
            parts = []
            for model in ("single_centroid", "exemplar_max", "bayes_shared_diag"):
                a = row["models"][model]
                parts.append(
                    f"{model} F1={a['real_macro_f1_mean']:.4f} "
                    f"gain={a['macro_f1_gain_over_within_category_shuffle']:+.4f}"
                )
            print(f"  {s}-shot ({row['n_feedback_examples']} feedback): " + " | ".join(parts))
            if int(s) == max(shots):
                for model in ("single_centroid", "exemplar_max", "bayes_shared_diag"):
                    pc = row["models"][model]["per_class"]
                    print(
                        "    " + model + " per-class F1: "
                        + ", ".join(f"{lab}={pc[lab]['real_f1_mean']:.3f}" for lab in CORE4)
                    )

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_path = out_dir / "memory_addressability_audit.json"
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
    p.add_argument("--shots", type=str, default="1,2,4,8")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--bayes-kappa0",
        type=float,
        default=1.0,
        help="Fixed prior strength in pseudo-observations; predeclared, do not tune on held-out products.",
    )
    p.add_argument("--out-dir", type=str, default="outputs/memory_addressability_audit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
