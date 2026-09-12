#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stricter evaluator for the ABMG factorizability audit.

This evaluator consumes ``factorizability_features.pt`` produced by
``abmg_factorizability_audit.py extract``.  It does not run DINO and does not
open Real-IAD images/JSON files, so an existing extraction artifact can be
re-used after changing evaluation controls.

Why v2 exists
-------------
The first source inventory showed that four Real-IAD defect codes have broad
cross-product coverage in outer fold 0:

    AK, HS, QS, ZW

while BX/CH/PS/YW occur in relatively few product categories.  Therefore v2
reports two scopes:

* ``core4``: fixed AK/HS/QS/ZW labels in every source-category CV fold.  This is
  the primary, apples-to-apples diagnostic.
* ``shared``: the broader fold-specific shared-label diagnostic retained from
  v1 as a secondary analysis.

It also adds a *within-training-category label permutation* control.  Labels
are shuffled only among samples from the same product category.  This preserves
category-specific label frequencies while destroying the association between a
particular visual defect sample and its defect code.  A useful representation
should beat this stricter control across held-out product categories.

Nothing in this file establishes C1 by itself and nothing here changes the
supervisor-approved proposal.  This is a representation factorizability audit.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

CORE4 = ("AK", "HS", "QS", "ZW")
EXPECTED_SCHEMA = "abmg.factorizability.features.v1"


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def _finite_mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([float(x) for x in xs if np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _l2_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _fit_centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(x_train)
    a = _l2_np(scaler.transform(x_train).astype(np.float64))
    b = _l2_np(scaler.transform(x_test).astype(np.float64))
    classes = np.array(sorted(set(y_train.tolist())), dtype=object)
    centroids: List[np.ndarray] = []
    for c in classes:
        z = a[y_train == c].mean(axis=0, keepdims=True)
        centroids.append(_l2_np(z)[0])
    C = np.stack(centroids, axis=0)
    return classes[(b @ C.T).argmax(axis=1)]


def _fit_linear_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(x_train)
    a = scaler.transform(x_train)
    b = scaler.transform(x_test)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(a, y_train)
    return clf.predict(b)


def global_label_shuffle(labels: np.ndarray, seed: int) -> np.ndarray:
    """Global permutation retained as a weak/null baseline."""
    rng = np.random.default_rng(int(seed))
    out = np.array(labels, copy=True)
    rng.shuffle(out)
    return out


def within_category_label_shuffle(
    labels: np.ndarray,
    categories: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Shuffle labels independently inside each training product category.

    The exact label multiset of every category is preserved.  Consequently the
    control retains category/label frequency structure but destroys the mapping
    from an individual visual sample to its defect code.
    """
    rng = np.random.default_rng(int(seed))
    out = np.array(labels, copy=True)
    cats = np.asarray(categories, dtype=object)
    for cat in sorted(set(cats.tolist())):
        idx = np.flatnonzero(cats == cat)
        vals = np.array(out[idx], copy=True)
        rng.shuffle(vals)
        out[idx] = vals
    return out


def _eligible_labels(
    scope: str,
    labels: np.ndarray,
    train0: np.ndarray,
    test0: np.ndarray,
    min_train_per_label: int,
    min_test_per_label: int,
) -> List[str]:
    tr_counts = Counter(labels[train0].tolist())
    te_counts = Counter(labels[test0].tolist())
    if scope == "core4":
        # Fixed across folds.  If one core label is absent, mark the fold invalid
        # rather than silently changing the task.
        wanted = list(CORE4)
        ok = all(
            tr_counts[c] >= int(min_train_per_label) and te_counts[c] >= int(min_test_per_label)
            for c in wanted
        )
        return wanted if ok else []
    if scope == "shared":
        return sorted(
            c
            for c in set(tr_counts) & set(te_counts)
            if tr_counts[c] >= int(min_train_per_label)
            and te_counts[c] >= int(min_test_per_label)
        )
    raise ValueError(f"Unknown scope={scope!r}")


def evaluate_scope(
    X: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    cv_groups: Sequence[Sequence[str]],
    *,
    scope: str,
    min_train_per_label: int,
    min_test_per_label: int,
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    folds_out: List[Dict[str, Any]] = []
    all_source = set(categories.tolist())

    for fi, group in enumerate(cv_groups):
        test_cats = set(group)
        train_cats = all_source - test_cats
        train0 = np.asarray([c in train_cats for c in categories], dtype=bool)
        test0 = np.asarray([c in test_cats for c in categories], dtype=bool)
        selected_labels = _eligible_labels(
            scope,
            labels,
            train0,
            test0,
            min_train_per_label,
            min_test_per_label,
        )
        selected_set = set(selected_labels)
        train = train0 & np.asarray([y in selected_set for y in labels], dtype=bool)
        test = test0 & np.asarray([y in selected_set for y in labels], dtype=bool)
        total_test = int(test0.sum())

        if len(selected_labels) < 2 or int(train.sum()) == 0 or int(test.sum()) == 0:
            folds_out.append(
                {
                    "source_cv_fold": fi,
                    "test_categories": sorted(test_cats),
                    "status": "insufficient_labels",
                    "labels": selected_labels,
                    "test_label_coverage": float(test.sum() / max(total_test, 1)),
                }
            )
            continue

        xt, yt = X[train], labels[train]
        ct = categories[train]
        xv, yv = X[test], labels[test]

        pred_centroid = _fit_centroid_predict(xt, yt, xv)
        pred_linear = _fit_linear_predict(xt, yt, xv, seed + fi)

        perm_global: List[float] = []
        perm_within_cat: List[float] = []
        for pi in range(int(permutations)):
            base_seed = int(seed) + 100_000 * (fi + 1) + pi

            yg = global_label_shuffle(yt, base_seed)
            if len(set(yg.tolist())) >= 2:
                pg = _fit_linear_predict(xt, yg, xv, base_seed + 10_000_000)
                perm_global.append(
                    float(f1_score(yv, pg, average="macro", zero_division=0))
                )

            yw = within_category_label_shuffle(yt, ct, base_seed + 50_000_000)
            if len(set(yw.tolist())) >= 2:
                pw = _fit_linear_predict(xt, yw, xv, base_seed + 60_000_000)
                perm_within_cat.append(
                    float(f1_score(yv, pw, average="macro", zero_division=0))
                )

        g_mean, g_std = _finite_mean_std(perm_global)
        w_mean, w_std = _finite_mean_std(perm_within_cat)
        real = _metrics(yv, pred_linear)

        folds_out.append(
            {
                "source_cv_fold": fi,
                "test_categories": sorted(test_cats),
                "status": "ok",
                "labels": selected_labels,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "n_total_test_before_label_filter": total_test,
                "test_label_coverage": float(test.sum() / max(total_test, 1)),
                "centroid": _metrics(yv, pred_centroid),
                "linear_probe": real,
                "linear_probe_global_shuffle_macro_f1": {
                    "n": len(perm_global),
                    "mean": g_mean,
                    "std": g_std,
                },
                "linear_probe_within_category_shuffle_macro_f1": {
                    "n": len(perm_within_cat),
                    "mean": w_mean,
                    "std": w_std,
                },
                "linear_probe_macro_f1_gain_over_global_shuffle": (
                    float(real["macro_f1"] - g_mean) if np.isfinite(g_mean) else float("nan")
                ),
                "linear_probe_macro_f1_gain_over_within_category_shuffle": (
                    float(real["macro_f1"] - w_mean) if np.isfinite(w_mean) else float("nan")
                ),
            }
        )

    ok = [x for x in folds_out if x.get("status") == "ok"]
    aggregate: Dict[str, Any] = {
        "scope": scope,
        "fixed_labels": list(CORE4) if scope == "core4" else None,
        "n_valid_cv_folds": len(ok),
    }
    for method in ("centroid", "linear_probe"):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            m, s = _finite_mean_std([x[method][metric] for x in ok])
            aggregate[f"{method}_{metric}_mean"] = m
            aggregate[f"{method}_{metric}_std"] = s

    aggregate["test_label_coverage_mean"] = _finite_mean_std(
        [x["test_label_coverage"] for x in ok]
    )[0]
    aggregate["global_shuffle_macro_f1_mean"] = _finite_mean_std(
        [x["linear_probe_global_shuffle_macro_f1"]["mean"] for x in ok]
    )[0]
    aggregate["within_category_shuffle_macro_f1_mean"] = _finite_mean_std(
        [x["linear_probe_within_category_shuffle_macro_f1"]["mean"] for x in ok]
    )[0]

    real = aggregate.get("linear_probe_macro_f1_mean", float("nan"))
    glob = aggregate.get("global_shuffle_macro_f1_mean", float("nan"))
    within = aggregate.get("within_category_shuffle_macro_f1_mean", float("nan"))
    aggregate["linear_probe_macro_f1_gain_over_global_shuffle"] = (
        float(real - glob) if np.isfinite(real) and np.isfinite(glob) else float("nan")
    )
    aggregate["linear_probe_macro_f1_gain_over_within_category_shuffle"] = (
        float(real - within) if np.isfinite(real) and np.isfinite(within) else float("nan")
    )

    return {"folds": folds_out, "aggregate": aggregate}


def summarise_localisation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    masked = [x for x in rows if x.get("mask_available")]
    nonempty = [x for x in masked if int(x.get("n_positive_patches", 0)) > 0]
    empty = [x for x in masked if int(x.get("n_positive_patches", 0)) == 0]

    def mean_key(xs: Sequence[Dict[str, Any]], key: str) -> float:
        vals = [float(x.get(key, float("nan"))) for x in xs]
        vals = [x for x in vals if np.isfinite(x)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_declared_masks": len(masked),
        "n_nonempty_after_sensor_geometry": len(nonempty),
        "n_empty_after_sensor_geometry": len(empty),
        "empty_after_sensor_geometry_fraction": float(len(empty) / max(len(masked), 1)),
        "hit_at_k_all_declared_masks": (
            float(np.mean([1.0 if x.get("sensor_hit_any") else 0.0 for x in masked]))
            if masked
            else float("nan")
        ),
        "hit_at_k_nonempty_geometry": (
            float(np.mean([1.0 if x.get("sensor_hit_any") else 0.0 for x in nonempty]))
            if nonempty
            else float("nan")
        ),
        "mean_patch_recall_at_k_nonempty_geometry": mean_key(nonempty, "sensor_patch_recall"),
        "mean_per_image_patch_auroc_nonempty_geometry": mean_key(nonempty, "sensor_patch_auroc"),
        "note": (
            "An empty mask after sensor geometry means the annotated defect vanished under the "
            "448-resize/392-center-crop/28x28 patch transform. Such cases cannot contribute to "
            "oracle patch representations and are reported explicitly rather than silently dropped."
        ),
    }


def run(args: argparse.Namespace) -> int:
    path = Path(args.features)
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")

    if artifact.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected artifact schema: {artifact.get('schema')!r}")
    if int(artifact.get("outer_fold", -1)) != int(args.fold):
        raise ValueError(
            f"Artifact outer_fold={artifact.get('outer_fold')} does not match --fold={args.fold}"
        )

    items = artifact["items"]
    cv_groups = artifact["source_cv_groups"]

    result: Dict[str, Any] = {
        "schema": "abmg.factorizability.evaluation.v2",
        "outer_fold": int(args.fold),
        "target_categories_untouched": artifact["target_categories_untouched"],
        "source_cv_groups": cv_groups,
        "core4_labels": list(CORE4),
        "decision_scope": (
            "Diagnostic only. core4 is primary because AK/HS/QS/ZW have broad source-category "
            "coverage. shared is secondary. A representation is promising only if structure "
            "generalises across held-out product categories and materially exceeds both global "
            "and within-training-category label-shuffle controls. This does not establish C1."
        ),
        "representations": {},
        "sensor_localisation": summarise_localisation(artifact.get("localisation", [])),
    }

    for key, slot in sorted(artifact["representations"].items()):
        idx = slot["item_indices"].numpy().astype(np.int64)
        X = slot["features"].float().numpy().astype(np.float64)
        cats = np.asarray([items[i]["category"] for i in idx], dtype=object)
        labels = np.asarray([items[i]["defect_source_offline_only"] for i in idx], dtype=object)
        print(f"Evaluating {key}: n={len(idx)} d={X.shape[1]}")

        core4 = evaluate_scope(
            X,
            cats,
            labels,
            cv_groups,
            scope="core4",
            min_train_per_label=args.min_train_per_label,
            min_test_per_label=args.min_test_per_label,
            permutations=args.permutations,
            seed=args.seed,
        )
        shared = evaluate_scope(
            X,
            cats,
            labels,
            cv_groups,
            scope="shared",
            min_train_per_label=args.min_train_per_label,
            min_test_per_label=args.min_test_per_label,
            permutations=args.permutations,
            seed=args.seed + 7_000_000,
        )
        result["representations"][key] = {"core4": core4, "shared": shared}

    ranking: List[Dict[str, Any]] = []
    for key, obj in result["representations"].items():
        agg = obj["core4"]["aggregate"]
        ranking.append(
            {
                "representation": key,
                "core4_linear_probe_macro_f1": agg.get("linear_probe_macro_f1_mean"),
                "core4_centroid_macro_f1": agg.get("centroid_macro_f1_mean"),
                "core4_gain_over_within_category_shuffle": agg.get(
                    "linear_probe_macro_f1_gain_over_within_category_shuffle"
                ),
                "core4_gain_over_global_shuffle": agg.get(
                    "linear_probe_macro_f1_gain_over_global_shuffle"
                ),
                "core4_test_label_coverage": agg.get("test_label_coverage_mean"),
            }
        )

    ranking.sort(
        key=lambda x: (
            -999.0
            if not np.isfinite(float(x["core4_linear_probe_macro_f1"]))
            else float(x["core4_linear_probe_macro_f1"])
        ),
        reverse=True,
    )
    result["ranking_by_core4_linear_probe_macro_f1"] = ranking

    out = Path(args.out_dir) / f"fold_{args.fold}" / "evaluation_v2.json"
    _json_dump(result, out)
    print(
        json.dumps(
            {
                "ranking": ranking,
                "sensor_localisation": result["sensor_localisation"],
            },
            indent=2,
            allow_nan=True,
        )
    )
    print(f"Wrote {out}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--features", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="outputs/factorizability_audit")
    p.add_argument("--min-train-per-label", type=int, default=8)
    p.add_argument("--min-test-per-label", type=int, default=2)
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
