#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused discriminator/calibration audit for ABMG sparse factor memory.

Consumes an existing ``abmg.factorizability.features.v1`` artifact.  It does
NOT rerun DINO and does NOT open Real-IAD images.

Scientific question
-------------------
The previous memory audit found that ``bayes_shared_diag`` routed Core-4 defect
factors substantially better than a cosine centroid, but its softmax
responsibilities were severely overconfident.  This audit decomposes that gain
under exactly the same few-shot supports and held-out product-category protocol.

Four matched discriminators are evaluated:

1. cosine_centroid
   Previous single-centroid cosine baseline.

2. diag_gaussian_mle
   Deterministic class means from the verified supports, scored with the same
   label-free shared diagonal covariance.  This isolates the effect of a
   variance-normalized (Mahalanobis-like) metric from cosine geometry.

3. diag_gaussian_shrunk
   Same deterministic shared-diagonal score, but with the Normal-Normal
   posterior mean

       mu_g* = (kappa0 * mu0 + sum_g) / (kappa0 + n_g).

   This isolates shrinkage toward the label-free source background mean.

4. bayes_predictive
   Current Normal-Normal posterior-predictive score with

       Sigma_pred,g = Sigma * (1 + 1/(kappa0+n_g)).

   With the equal-shot protocol used here, n_g is identical for every factor.
   Therefore ``diag_gaussian_shrunk`` and ``bayes_predictive`` should have the
   same class ranking/argmax; this is an explicit mechanistic sanity check.

Calibration
-----------
For each outer source-CV fold / shot / repeat / model, a single positive
softmax temperature is fitted by NLL on Core-4 examples from SOURCE TRAINING
PRODUCT CATEGORIES ONLY, excluding the sparse support examples.  It is then
applied to the held-out source-CV product categories.

This dense-label temperature fit is an OFFLINE DIAGNOSTIC OF CALIBRATABILITY,
not a deployable sparse-feedback mechanism.  It cannot establish that calibrated
responsibilities are available online without an additional calibration design.

Reported probability diagnostics include NLL, multiclass Brier score, ECE,
mean entropy, entropy on correct/incorrect predictions, and mean true-class
responsibility, before and after temperature scaling.

Hard boundary
-------------
Outer ABMG target categories stay untouched.  No held-out source-CV test label
is used to fit memory state, shared covariance, or temperature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
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
        prototype_predict,
    )
    from abmg_memory_addressability_audit import fit_shared_diag_prior
except ImportError:  # module import from repo root / tests
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
    )
    from scripts.abmg_memory_addressability_audit import fit_shared_diag_prior


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def _softmax_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("temperature must be finite and > 0")
    z = np.asarray(scores, dtype=np.float64) / t
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -700.0, 700.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)


def _label_cols(labels: np.ndarray, label_order: Sequence[str] = CORE4) -> np.ndarray:
    lut = {str(y): i for i, y in enumerate(label_order)}
    return np.asarray([lut[str(y)] for y in labels], dtype=np.int64)


def _nll_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    label_order: Sequence[str] = CORE4,
) -> float:
    y = _label_cols(labels, label_order)
    z = np.asarray(scores, dtype=np.float64) / float(temperature)
    logp = z - logsumexp(z, axis=1, keepdims=True)
    return float(-np.mean(logp[np.arange(len(y)), y]))


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    label_order: Sequence[str] = CORE4,
) -> Dict[str, float]:
    """Fit one positive softmax temperature by calibration-set NLL.

    Optimization is performed in log-temperature space over a deliberately wide
    bounded range.  Gaussian scores can be thousands of times larger than cosine
    scores, so the range must accommodate substantial down-scaling.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    if scores.ndim != 2 or scores.shape[0] != len(labels):
        raise ValueError("scores must be [N,C] and align with labels")

    def objective(log_t: float) -> float:
        return _nll_from_scores(scores, labels, float(np.exp(log_t)), label_order)

    opt = minimize_scalar(
        objective,
        bounds=(-8.0, 20.0),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 300},
    )
    t = float(np.exp(float(opt.x)))
    return {
        "temperature": t,
        "calibration_nll": float(opt.fun),
        "optimizer_success": bool(opt.success),
    }


def calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_order: Sequence[str] = CORE4,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Top-label ECE plus proper multiclass probability scores."""
    y = _label_cols(np.asarray(y_true, dtype=object), label_order)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    rows = np.arange(len(y))
    pred = p.argmax(axis=1)
    correct = pred == y
    conf = p.max(axis=1)
    true_p = p[rows, y]
    entropy = -np.sum(p * np.log(p), axis=1)
    onehot = np.zeros_like(p)
    onehot[rows, y] = 1.0
    brier = np.sum((p - onehot) ** 2, axis=1)

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    reliability = []
    ece = 0.0
    for bi in range(int(n_bins)):
        lo, hi = float(edges[bi]), float(edges[bi + 1])
        if bi == int(n_bins) - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n:
            mean_conf = float(conf[mask].mean())
            acc = float(correct[mask].mean())
            frac = float(n / len(y))
            ece += frac * abs(acc - mean_conf)
        else:
            mean_conf = float("nan")
            acc = float("nan")
            frac = 0.0
        reliability.append(
            {
                "bin": bi,
                "lo": lo,
                "hi": hi,
                "n": n,
                "fraction": frac,
                "mean_confidence": mean_conf,
                "accuracy": acc,
            }
        )

    return {
        "negative_log_likelihood": float(np.mean(-np.log(true_p))),
        "brier_score": float(np.mean(brier)),
        "ece": float(ece),
        "mean_confidence": float(np.mean(conf)),
        "mean_true_responsibility": float(np.mean(true_p)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_entropy_correct": float(np.mean(entropy[correct])) if np.any(correct) else float("nan"),
        "mean_entropy_incorrect": float(np.mean(entropy[~correct])) if np.any(~correct) else float("nan"),
        "reliability_bins": reliability,
    }


def cosine_centroid_scores(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    support: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    pred, scores = prototype_predict(X_train, X_eval, support, CORE4)
    return pred, np.asarray(scores, dtype=np.float64)


def diag_gaussian_scores(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    support: Mapping[str, np.ndarray],
    shared_prior: Mapping[str, np.ndarray],
    *,
    kappa0: Optional[float] = None,
    posterior_predictive: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared-diagonal Gaussian scores with optional Normal-Normal shrinkage.

    kappa0=None -> deterministic support sample mean (MLE).
    kappa0>0    -> shrunk posterior mean.
    posterior_predictive=True additionally inflates variance by
    (1 + 1/(kappa0+n_g)).
    """
    xt = _l2(X_train)
    xv = _l2(X_eval)
    mu0 = np.asarray(shared_prior["mu0"], dtype=np.float64)
    var = np.asarray(shared_prior["var"], dtype=np.float64)
    scores = []

    if posterior_predictive and (kappa0 is None or float(kappa0) <= 0):
        raise ValueError("posterior_predictive requires kappa0 > 0")
    if kappa0 is not None and float(kappa0) <= 0:
        raise ValueError("kappa0 must be > 0")

    for label in CORE4:
        idx = np.asarray(support[str(label)], dtype=np.int64)
        n = int(len(idx))
        if n <= 0:
            raise ValueError(f"No support for label={label}")
        sum_x = xt[idx].sum(axis=0)
        if kappa0 is None:
            mu = sum_x / float(n)
            score_var = var
        else:
            kappa_n = float(kappa0) + float(n)
            mu = (float(kappa0) * mu0 + sum_x) / kappa_n
            score_var = var * (1.0 + 1.0 / kappa_n) if posterior_predictive else var

        diff = xv - mu[None, :]
        logp = -0.5 * np.sum(
            (diff * diff) / score_var[None, :] + np.log(score_var[None, :]),
            axis=1,
        )
        scores.append(logp)

    score_mat = np.stack(scores, axis=1)
    pred = np.asarray(CORE4, dtype=object)[score_mat.argmax(axis=1)]
    return pred, score_mat


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    vals = f1_score(
        y_true,
        y_pred,
        labels=list(CORE4),
        average=None,
        zero_division=0,
    )
    return {str(label): float(vals[i]) for i, label in enumerate(CORE4)}


def _support_union(support: Mapping[str, np.ndarray]) -> np.ndarray:
    vals = [np.asarray(support[label], dtype=np.int64) for label in CORE4]
    return np.unique(np.concatenate(vals, axis=0))


def _score_model(
    model: str,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    support: Mapping[str, np.ndarray],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if model == "cosine_centroid":
        return cosine_centroid_scores(X_train, X_eval, support)
    if model == "diag_gaussian_mle":
        return diag_gaussian_scores(
            X_train, X_eval, support, shared_prior, kappa0=None, posterior_predictive=False
        )
    if model == "diag_gaussian_shrunk":
        return diag_gaussian_scores(
            X_train,
            X_eval,
            support,
            shared_prior,
            kappa0=float(kappa0),
            posterior_predictive=False,
        )
    if model == "bayes_predictive":
        return diag_gaussian_scores(
            X_train,
            X_eval,
            support,
            shared_prior,
            kappa0=float(kappa0),
            posterior_predictive=True,
        )
    raise ValueError(f"Unknown model={model!r}")


def _evaluate_run_model(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    support: Mapping[str, np.ndarray],
    shared_prior: Mapping[str, np.ndarray],
    kappa0: float,
    n_bins: int,
) -> Dict[str, Any]:
    pred_test, scores_test = _score_model(
        model, X_train, X_test, support, shared_prior, float(kappa0)
    )

    support_idx = _support_union(support)
    cal_mask = np.ones(len(X_train), dtype=bool)
    cal_mask[support_idx] = False
    # Calibration labels are offline-only and source-training-category only.
    X_cal = X_train[cal_mask]
    y_cal = y_train[cal_mask]
    _, scores_cal = _score_model(
        model, X_train, X_cal, support, shared_prior, float(kappa0)
    )
    temp = fit_temperature(scores_cal, y_cal, CORE4)

    raw_p = _softmax_temperature(scores_test, 1.0)
    cal_p = _softmax_temperature(scores_test, temp["temperature"])

    return {
        "classification": _metrics(y_test, pred_test),
        "per_class_f1": _per_class_f1(y_test, pred_test),
        "temperature_fit": temp,
        "raw_calibration": calibration_metrics(y_test, raw_p, CORE4, n_bins),
        "temperature_calibrated": calibration_metrics(y_test, cal_p, CORE4, n_bins),
    }


def _aggregate_reliability(runs: Sequence[Dict[str, Any]], model: str, section: str) -> list:
    bins = [r[model][section]["reliability_bins"] for r in runs]
    if not bins:
        return []
    n_bins = len(bins[0])
    out = []
    for bi in range(n_bins):
        row: Dict[str, Any] = {
            "bin": bi,
            "lo": float(bins[0][bi]["lo"]),
            "hi": float(bins[0][bi]["hi"]),
        }
        for key in ("n", "fraction", "mean_confidence", "accuracy"):
            vals = [b[bi][key] for b in bins]
            m, s = _finite_mean_std(vals)
            row[f"{key}_mean"] = m
            row[f"{key}_std"] = s
        out.append(row)
    return out


def _aggregate_model(runs: Sequence[Dict[str, Any]], model: str) -> Dict[str, Any]:
    agg: Dict[str, Any] = {"n_runs": len(runs)}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        m, s = _finite_mean_std(r[model]["classification"][metric] for r in runs)
        agg[f"{metric}_mean"] = m
        agg[f"{metric}_std"] = s

    per_class: Dict[str, Any] = {}
    for label in CORE4:
        m, s = _finite_mean_std(r[model]["per_class_f1"][label] for r in runs)
        per_class[label] = {"f1_mean": m, "f1_std": s}
    agg["per_class"] = per_class

    tm, ts = _finite_mean_std(r[model]["temperature_fit"]["temperature"] for r in runs)
    agg["temperature_mean"] = tm
    agg["temperature_std"] = ts
    agg["temperature_optimizer_success_fraction"] = float(
        np.mean([1.0 if r[model]["temperature_fit"]["optimizer_success"] else 0.0 for r in runs])
    ) if runs else float("nan")

    cal_metrics = (
        "negative_log_likelihood",
        "brier_score",
        "ece",
        "mean_confidence",
        "mean_true_responsibility",
        "mean_entropy",
        "mean_entropy_correct",
        "mean_entropy_incorrect",
    )
    for section in ("raw_calibration", "temperature_calibrated"):
        for metric in cal_metrics:
            m, s = _finite_mean_std(r[model][section][metric] for r in runs)
            agg[f"{section}_{metric}_mean"] = m
            agg[f"{section}_{metric}_std"] = s
        agg[f"{section}_reliability_bins"] = _aggregate_reliability(runs, model, section)
    return agg


def _paired_comparison(
    runs: Sequence[Dict[str, Any]],
    a: str,
    b: str,
) -> Dict[str, float]:
    """Paired delta b-a on macro-F1 across identical support/test runs."""
    d = np.asarray(
        [
            float(r[b]["classification"]["macro_f1"])
            - float(r[a]["classification"]["macro_f1"])
            for r in runs
        ],
        dtype=np.float64,
    )
    return {
        "mean_delta_macro_f1": float(d.mean()) if d.size else float("nan"),
        "std_delta_macro_f1": float(d.std(ddof=0)) if d.size else float("nan"),
        "win_fraction_b_gt_a": float(np.mean(d > 0)) if d.size else float("nan"),
        "tie_fraction": float(np.mean(np.isclose(d, 0.0, atol=1e-12))) if d.size else float("nan"),
    }


def evaluate_discriminators(
    X: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    cv_groups: Sequence[Sequence[str]],
    shot_values: Sequence[int],
    repeats: int,
    seed: int,
    kappa0: float,
    n_bins: int,
) -> Dict[str, Any]:
    models = (
        "cosine_centroid",
        "diag_gaussian_mle",
        "diag_gaussian_shrunk",
        "bayes_predictive",
    )
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

            # Same label-free shared background state as the prior memory audit.
            shared_prior = fit_shared_diag_prior(X[train_all])

            for ri in range(int(repeats)):
                run_seed = int(seed) + 100_000 * int(shots) + 10_000 * fi + ri
                support = category_diverse_support_indices(
                    yt, ct, CORE4, int(shots), run_seed
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
                    "n_calibration_examples": int(len(xt) - len(_support_union(support))),
                    "support_categories": {
                        label: sorted(set(ct[idx].tolist()))
                        for label, idx in support.items()
                    },
                }
                for model in models:
                    row[model] = _evaluate_run_model(
                        model,
                        xt,
                        yt,
                        xv,
                        yv,
                        support,
                        shared_prior,
                        float(kappa0),
                        int(n_bins),
                    )

                pred_shrunk, _ = _score_model(
                    "diag_gaussian_shrunk", xt, xv, support, shared_prior, float(kappa0)
                )
                pred_bayes, _ = _score_model(
                    "bayes_predictive", xt, xv, support, shared_prior, float(kappa0)
                )
                row["shrunk_vs_bayes_prediction_agreement"] = float(np.mean(pred_shrunk == pred_bayes))
                runs.append(row)

        model_agg = {m: _aggregate_model(runs, m) for m in models}
        agreement_m, agreement_s = _finite_mean_std(
            r["shrunk_vs_bayes_prediction_agreement"] for r in runs
        )
        by_shot[str(shots)] = {
            "shots_per_label": int(shots),
            "n_feedback_examples": int(shots) * len(CORE4),
            "n_runs": len(runs),
            "models": model_agg,
            "paired_mechanism_deltas": {
                "whitening_diag_mle_minus_cosine": _paired_comparison(
                    runs, "cosine_centroid", "diag_gaussian_mle"
                ),
                "shrinkage_minus_diag_mle": _paired_comparison(
                    runs, "diag_gaussian_mle", "diag_gaussian_shrunk"
                ),
                "bayes_predictive_minus_shrunk": _paired_comparison(
                    runs, "diag_gaussian_shrunk", "bayes_predictive"
                ),
            },
            "shrunk_vs_bayes_prediction_agreement_mean": agreement_m,
            "shrunk_vs_bayes_prediction_agreement_std": agreement_s,
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
    if int(args.calibration_bins) < 2:
        raise ValueError("--calibration-bins must be >= 2")

    shots = parse_int_list(args.shots)
    items = artifact["items"]
    cv_groups = artifact["source_cv_groups"]
    result: Dict[str, Any] = {
        "schema": "abmg.discriminator_calibration_audit.v1",
        "outer_fold": int(args.fold),
        "source_artifact": str(artifact_path),
        "target_categories_untouched": artifact["target_categories_untouched"],
        "core4_labels": list(CORE4),
        "representations_requested": list(requested),
        "protocol": {
            "shots_per_label": list(shots),
            "repeats_per_cv_fold": int(args.repeats),
            "support_selection": "same category-diverse supports for every discriminator",
            "models": {
                "cosine_centroid": "normalized support mean + cosine",
                "diag_gaussian_mle": "support sample mean + label-free shared diagonal covariance",
                "diag_gaussian_shrunk": "Normal-Normal posterior mean + deterministic shared diagonal covariance",
                "bayes_predictive": "Normal-Normal posterior mean + posterior-predictive shared diagonal covariance",
            },
            "bayes_kappa0": float(args.bayes_kappa0),
            "calibration": {
                "method": "single scalar softmax temperature fitted by NLL",
                "fit_data": "Core-4 source-CV training-category examples excluding sparse supports",
                "held_out_test_products_never_used": True,
                "n_bins_ece": int(args.calibration_bins),
                "deployment_eligibility": "offline diagnostic only; dense calibration labels are not part of sparse-feedback deployment",
            },
            "mechanism_decomposition": [
                "diag_gaussian_mle - cosine_centroid isolates shared diagonal whitening/metric",
                "diag_gaussian_shrunk - diag_gaussian_mle isolates prior-mean shrinkage",
                "bayes_predictive - diag_gaussian_shrunk isolates posterior-predictive variance",
            ],
        },
        "interpretation_boundary": (
            "Source-side diagnostic only. Classification gain and calibratability do not by themselves "
            "establish deployable Bayesian responsibilities, final C1/C2, or end-to-end online benefit."
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
        by_shot = evaluate_discriminators(
            X,
            categories,
            labels,
            cv_groups,
            shots,
            int(args.repeats),
            int(args.seed),
            float(args.bayes_kappa0),
            int(args.calibration_bins),
        )
        result["representations"][key] = {"by_shot": by_shot}

        for s in shots:
            row = by_shot[str(s)]
            print(f"  {s}-shot ({row['n_feedback_examples']} feedback):")
            for model in (
                "cosine_centroid",
                "diag_gaussian_mle",
                "diag_gaussian_shrunk",
                "bayes_predictive",
            ):
                a = row["models"][model]
                print(
                    f"    {model}: F1={a['macro_f1_mean']:.4f} "
                    f"raw NLL={a['raw_calibration_negative_log_likelihood_mean']:.3f} "
                    f"-> cal NLL={a['temperature_calibrated_negative_log_likelihood_mean']:.3f}; "
                    f"ECE={a['raw_calibration_ece_mean']:.3f}"
                    f"->{a['temperature_calibrated_ece_mean']:.3f}; T={a['temperature_mean']:.3g}"
                )
            print(
                "    mechanism deltas: "
                f"whitening={row['paired_mechanism_deltas']['whitening_diag_mle_minus_cosine']['mean_delta_macro_f1']:+.4f}, "
                f"shrinkage={row['paired_mechanism_deltas']['shrinkage_minus_diag_mle']['mean_delta_macro_f1']:+.4f}, "
                f"predictive={row['paired_mechanism_deltas']['bayes_predictive_minus_shrunk']['mean_delta_macro_f1']:+.4f}, "
                f"shrunk/bayes agreement={row['shrunk_vs_bayes_prediction_agreement_mean']:.4f}"
            )

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_path = out_dir / "discriminator_calibration_audit.json"
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
        help="Fixed prior strength inherited from the previous audit; do not tune on held-out products.",
    )
    p.add_argument("--calibration-bins", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="outputs/discriminator_calibration_audit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
