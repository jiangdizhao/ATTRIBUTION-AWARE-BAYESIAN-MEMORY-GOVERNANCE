#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prototype/addressability audit for ABMG source-side defect evidence.

This script consumes an existing ``abmg.factorizability.features.v1`` artifact
(e.g. ``patch_aggregation_features.pt``) and does NOT rerun DINO or open Real-IAD
images.  It asks two diagnostic questions before any change to the supervisor-
approved proposal:

1. Unsupervised structure:
   Do frozen evidence vectors form reusable defect-type structure across held-out
   product categories when clustered without defect labels?

2. Sparse-feedback prototype routing:
   If only 1/2/4/8 human-confirmed examples per defect factor are available from
   source product categories, can a tiny cosine-prototype memory route defects
   from unseen product categories to a stable factor address?

The primary label scope is the fixed Real-IAD Core-4 defect set AK/HS/QS/ZW,
chosen earlier because these labels have broad source-category coverage.  Labels
are used ONLY for this offline diagnostic and for simulating sparse human
feedback.  Outer-fold target categories are never introduced by this evaluator.

Important interpretation boundary
---------------------------------
A strong result would show that a simple sparse memory can exploit factor
information already present in frozen sensor evidence.  It would NOT establish
causal/semantic disentanglement and would NOT by itself establish the full C1
or C2 claims.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
)

CORE4: Tuple[str, ...] = ("AK", "HS", "QS", "ZW")
EXPECTED_SCHEMA = "abmg.factorizability.features.v1"


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def _finite_mean_std(xs: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray([float(x) for x in xs if np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def parse_int_list(spec: str) -> Tuple[int, ...]:
    vals = sorted({int(x.strip()) for x in str(spec).split(",") if x.strip()})
    if not vals or any(v <= 0 for v in vals):
        raise ValueError(f"Expected positive comma-separated integers, got {spec!r}")
    return tuple(vals)


def parse_str_list(spec: str) -> Tuple[str, ...]:
    vals = tuple(x.strip() for x in str(spec).split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one comma-separated value")
    return vals


def within_category_label_shuffle(
    labels: np.ndarray,
    categories: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Shuffle labels independently inside each product category.

    This preserves each category's exact label-frequency profile while destroying
    the association between an individual evidence vector and its defect code.
    """
    rng = np.random.default_rng(int(seed))
    out = np.array(labels, dtype=object, copy=True)
    categories = np.asarray(categories, dtype=object)
    for cat in sorted(set(categories.tolist())):
        idx = np.flatnonzero(categories == cat)
        vals = np.array(out[idx], dtype=object, copy=True)
        rng.shuffle(vals)
        out[idx] = vals
    return out


def category_diverse_support_indices(
    labels: np.ndarray,
    categories: np.ndarray,
    wanted_labels: Sequence[str],
    shots_per_label: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Select few-shot supports while maximising product-category diversity.

    Selection is deterministic for a given seed.  Each round takes at most one
    example from each available category before reusing a category, avoiding a
    few-shot prototype being dominated by one product when cross-product routing
    is the scientific question.
    """
    labels = np.asarray(labels, dtype=object)
    categories = np.asarray(categories, dtype=object)
    rng = np.random.default_rng(int(seed))
    out: Dict[str, np.ndarray] = {}

    for label in wanted_labels:
        cand = np.flatnonzero(labels == label)
        if cand.size < int(shots_per_label):
            raise ValueError(
                f"label={label} has only {cand.size} candidates; need {shots_per_label}"
            )
        by_cat: Dict[str, List[int]] = defaultdict(list)
        for i in cand.tolist():
            by_cat[str(categories[i])].append(int(i))
        cat_names = sorted(by_cat)
        # Independent shuffles inside categories and of category traversal order.
        queues: Dict[str, List[int]] = {}
        for cat in cat_names:
            q = np.asarray(by_cat[cat], dtype=np.int64)
            rng.shuffle(q)
            queues[cat] = q.tolist()
        cat_order = np.asarray(cat_names, dtype=object)
        rng.shuffle(cat_order)

        chosen: List[int] = []
        while len(chosen) < int(shots_per_label):
            made_progress = False
            for cat_obj in cat_order.tolist():
                cat = str(cat_obj)
                if queues[cat]:
                    chosen.append(queues[cat].pop())
                    made_progress = True
                    if len(chosen) >= int(shots_per_label):
                        break
            if not made_progress:
                raise RuntimeError(f"Unable to select {shots_per_label} supports for {label}")
            rng.shuffle(cat_order)
        out[str(label)] = np.asarray(chosen, dtype=np.int64)
    return out


def build_cosine_prototypes(
    X: np.ndarray,
    support_by_label: Mapping[str, np.ndarray],
    label_order: Sequence[str],
) -> np.ndarray:
    Xn = _l2(X)
    protos: List[np.ndarray] = []
    for label in label_order:
        idx = np.asarray(support_by_label[str(label)], dtype=np.int64)
        p = Xn[idx].mean(axis=0, keepdims=True)
        protos.append(_l2(p)[0])
    return np.stack(protos, axis=0)


def prototype_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    support_by_label: Mapping[str, np.ndarray],
    label_order: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    protos = build_cosine_prototypes(X_train, support_by_label, label_order)
    sims = _l2(X_test) @ protos.T
    pred = np.asarray(label_order, dtype=object)[sims.argmax(axis=1)]
    return pred, sims


def routing_diagnostics(
    y_true: np.ndarray,
    sims: np.ndarray,
    label_order: Sequence[str],
) -> Dict[str, float]:
    label_to_col = {str(y): i for i, y in enumerate(label_order)}
    true_cols = np.asarray([label_to_col[str(y)] for y in y_true], dtype=np.int64)
    true_sim = sims[np.arange(len(y_true)), true_cols]
    masked = np.array(sims, copy=True)
    masked[np.arange(len(y_true)), true_cols] = -np.inf
    best_wrong = masked.max(axis=1)
    top2 = np.argpartition(-sims, kth=min(1, sims.shape[1] - 1), axis=1)[:, :2]
    top2_hit = np.asarray(
        [true_cols[i] in set(top2[i].tolist()) for i in range(len(y_true))], dtype=np.float64
    )
    margin = true_sim - best_wrong
    return {
        "mean_true_similarity": float(np.mean(true_sim)),
        "mean_best_wrong_similarity": float(np.mean(best_wrong)),
        "mean_true_minus_best_wrong_margin": float(np.mean(margin)),
        "median_true_minus_best_wrong_margin": float(np.median(margin)),
        "top2_recall": float(np.mean(top2_hit)),
    }


def best_cluster_label_mapping(
    train_clusters: np.ndarray,
    train_labels: np.ndarray,
    label_order: Sequence[str],
) -> Dict[int, str]:
    """Map K=4 cluster IDs to Core-4 names using training labels only.

    Exhaustive 4! assignment avoids an extra scipy dependency.  The clustering
    itself remains label-free; this mapping is only for interpretable matched-F1.
    ARI/AMI on held-out categories remain permutation-invariant diagnostics.
    """
    cluster_ids = sorted(set(int(x) for x in train_clusters.tolist()))
    labels = tuple(str(x) for x in label_order)
    if len(cluster_ids) != len(labels):
        raise ValueError("cluster count must equal number of labels for matched mapping")
    best_score = -1
    best: Optional[Dict[int, str]] = None
    for perm in itertools.permutations(labels):
        mapping = {cluster_ids[i]: perm[i] for i in range(len(cluster_ids))}
        pred = np.asarray([mapping[int(c)] for c in train_clusters], dtype=object)
        score = int(np.sum(pred == train_labels))
        if score > best_score:
            best_score = score
            best = mapping
    assert best is not None
    return best


def evaluate_unsupervised_structure(
    X: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    cv_groups: Sequence[Sequence[str]],
    seed: int,
) -> Dict[str, Any]:
    folds: List[Dict[str, Any]] = []
    all_categories = set(categories.tolist())
    wanted = set(CORE4)

    for fi, group in enumerate(cv_groups):
        test_cats = set(str(x) for x in group)
        train_cats = all_categories - test_cats
        train = np.asarray(
            [(c in train_cats) and (y in wanted) for c, y in zip(categories, labels)],
            dtype=bool,
        )
        test = np.asarray(
            [(c in test_cats) and (y in wanted) for c, y in zip(categories, labels)],
            dtype=bool,
        )
        xt, yt, ct = X[train], labels[train], categories[train]
        xv, yv = X[test], labels[test]
        if len(set(yt.tolist())) != len(CORE4) or len(set(yv.tolist())) != len(CORE4):
            folds.append({"source_cv_fold": fi, "status": "insufficient_core4"})
            continue

        model = KMeans(n_clusters=len(CORE4), n_init=20, random_state=int(seed) + fi)
        train_cluster = model.fit_predict(_l2(xt))
        test_cluster = model.predict(_l2(xv))

        mapping = best_cluster_label_mapping(train_cluster, yt, CORE4)
        mapped_test = np.asarray([mapping[int(c)] for c in test_cluster], dtype=object)

        shuffled = within_category_label_shuffle(yt, ct, int(seed) + 1_000_000 + fi)
        shuffled_mapping = best_cluster_label_mapping(train_cluster, shuffled, CORE4)
        shuffled_mapped_test = np.asarray(
            [shuffled_mapping[int(c)] for c in test_cluster], dtype=object
        )

        real_metrics = _metrics(yv, mapped_test)
        shuffle_metrics = _metrics(yv, shuffled_mapped_test)
        folds.append(
            {
                "source_cv_fold": fi,
                "status": "ok",
                "test_categories": sorted(test_cats),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "adjusted_rand_index": float(adjusted_rand_score(yv, test_cluster)),
                "adjusted_mutual_info": float(adjusted_mutual_info_score(yv, test_cluster)),
                "matched": real_metrics,
                "within_category_shuffle_mapping": shuffle_metrics,
                "matched_macro_f1_gain_over_shuffle_mapping": float(
                    real_metrics["macro_f1"] - shuffle_metrics["macro_f1"]
                ),
                "train_cluster_to_label_mapping_offline_only": {
                    str(k): v for k, v in sorted(mapping.items())
                },
            }
        )

    ok = [x for x in folds if x.get("status") == "ok"]
    agg: Dict[str, Any] = {"n_valid_cv_folds": len(ok)}
    for key in ("adjusted_rand_index", "adjusted_mutual_info"):
        m, s = _finite_mean_std(x[key] for x in ok)
        agg[f"{key}_mean"] = m
        agg[f"{key}_std"] = s
    for prefix, path in (
        ("matched", "matched"),
        ("shuffle_mapping", "within_category_shuffle_mapping"),
    ):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            m, s = _finite_mean_std(x[path][metric] for x in ok)
            agg[f"{prefix}_{metric}_mean"] = m
            agg[f"{prefix}_{metric}_std"] = s
    agg["matched_macro_f1_gain_over_shuffle_mapping"] = float(
        agg.get("matched_macro_f1_mean", float("nan"))
        - agg.get("shuffle_mapping_macro_f1_mean", float("nan"))
    )
    return {"folds": folds, "aggregate": agg}


def evaluate_fewshot_prototypes(
    X: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    cv_groups: Sequence[Sequence[str]],
    shot_values: Sequence[int],
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    all_categories = set(categories.tolist())
    wanted = set(CORE4)
    by_shot: Dict[str, Any] = {}

    for shots in shot_values:
        runs: List[Dict[str, Any]] = []
        for fi, group in enumerate(cv_groups):
            test_cats = set(str(x) for x in group)
            train_cats = all_categories - test_cats
            train_mask = np.asarray(
                [(c in train_cats) and (y in wanted) for c, y in zip(categories, labels)],
                dtype=bool,
            )
            test_mask = np.asarray(
                [(c in test_cats) and (y in wanted) for c, y in zip(categories, labels)],
                dtype=bool,
            )
            xt, yt, ct = X[train_mask], labels[train_mask], categories[train_mask]
            xv, yv = X[test_mask], labels[test_mask]
            if len(set(yt.tolist())) != len(CORE4) or len(set(yv.tolist())) != len(CORE4):
                continue

            for ri in range(int(repeats)):
                run_seed = int(seed) + 100_000 * int(shots) + 10_000 * fi + ri
                support = category_diverse_support_indices(
                    yt, ct, CORE4, int(shots), run_seed
                )
                pred, sims = prototype_predict(xt, xv, support, CORE4)
                real = _metrics(yv, pred)
                route = routing_diagnostics(yv, sims, CORE4)

                shuffled_yt = within_category_label_shuffle(
                    yt, ct, run_seed + 50_000_000
                )
                shuffled_support = category_diverse_support_indices(
                    shuffled_yt, ct, CORE4, int(shots), run_seed + 60_000_000
                )
                shuffled_pred, _ = prototype_predict(
                    xt, xv, shuffled_support, CORE4
                )
                shuffled_metrics = _metrics(yv, shuffled_pred)

                support_categories = {
                    label: sorted(set(ct[idx].tolist()))
                    for label, idx in support.items()
                }
                runs.append(
                    {
                        "source_cv_fold": fi,
                        "repeat": ri,
                        "seed": run_seed,
                        "test_categories": sorted(test_cats),
                        "n_train_pool": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "n_feedback_examples": int(shots) * len(CORE4),
                        "support_categories": support_categories,
                        "real": real,
                        "routing": route,
                        "within_category_shuffle": shuffled_metrics,
                        "macro_f1_gain_over_within_category_shuffle": float(
                            real["macro_f1"] - shuffled_metrics["macro_f1"]
                        ),
                    }
                )

        agg: Dict[str, Any] = {
            "shots_per_label": int(shots),
            "n_feedback_examples": int(shots) * len(CORE4),
            "n_runs": len(runs),
            "repeats_per_cv_fold": int(repeats),
        }
        for section in ("real", "within_category_shuffle"):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                m, s = _finite_mean_std(r[section][metric] for r in runs)
                agg[f"{section}_{metric}_mean"] = m
                agg[f"{section}_{metric}_std"] = s
        for metric in (
            "mean_true_similarity",
            "mean_best_wrong_similarity",
            "mean_true_minus_best_wrong_margin",
            "median_true_minus_best_wrong_margin",
            "top2_recall",
        ):
            m, s = _finite_mean_std(r["routing"][metric] for r in runs)
            agg[f"routing_{metric}_mean"] = m
            agg[f"routing_{metric}_std"] = s
        agg["macro_f1_gain_over_within_category_shuffle"] = float(
            agg["real_macro_f1_mean"] - agg["within_category_shuffle_macro_f1_mean"]
        )
        by_shot[str(shots)] = {"aggregate": agg, "runs": runs}

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
    shots = parse_int_list(args.shots)
    items = artifact["items"]
    cv_groups = artifact["source_cv_groups"]

    result: Dict[str, Any] = {
        "schema": "abmg.prototype_addressability_audit.v1",
        "outer_fold": int(args.fold),
        "source_artifact": str(artifact_path),
        "target_categories_untouched": artifact["target_categories_untouched"],
        "core4_labels": list(CORE4),
        "representations_requested": list(requested),
        "fewshot_protocol": {
            "shots_per_label": list(shots),
            "repeats_per_cv_fold": int(args.repeats),
            "support_selection": "category_diverse_round_robin",
            "distance": "cosine",
            "prototype": "L2-normalized mean of L2-normalized support vectors",
            "shuffle_control": "within-training-category label shuffle",
        },
        "unsupervised_protocol": {
            "algorithm": "spherical-like KMeans on L2-normalized evidence",
            "n_clusters": len(CORE4),
            "n_init": 20,
            "primary_permutation_invariant_metrics": ["ARI", "AMI"],
            "matched_label_mapping": "training-fold labels only; offline interpretability diagnostic",
        },
        "interpretation_boundary": (
            "Diagnostic only. Strong results support reusable address structure in frozen evidence; "
            "they do not prove semantic disentanglement, causal factors, C1, or C2."
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
        core = np.asarray([y in set(CORE4) for y in labels], dtype=bool)
        print(
            f"Evaluating {key}: n={len(X)} d={X.shape[1]} core4={int(core.sum())}"
        )

        unsup = evaluate_unsupervised_structure(
            X,
            categories,
            labels,
            cv_groups,
            seed=int(args.seed),
        )
        fewshot = evaluate_fewshot_prototypes(
            X,
            categories,
            labels,
            cv_groups,
            shot_values=shots,
            repeats=int(args.repeats),
            seed=int(args.seed),
        )
        result["representations"][key] = {
            "n": int(len(X)),
            "dim": int(X.shape[1]),
            "n_core4": int(core.sum()),
            "unsupervised_structure": unsup,
            "fewshot_prototype_routing": fewshot,
        }

    summary: List[Dict[str, Any]] = []
    for key, obj in result["representations"].items():
        uns = obj["unsupervised_structure"]["aggregate"]
        row: Dict[str, Any] = {
            "representation": key,
            "unsupervised_ARI": uns.get("adjusted_rand_index_mean"),
            "unsupervised_AMI": uns.get("adjusted_mutual_info_mean"),
            "unsupervised_matched_macro_f1": uns.get("matched_macro_f1_mean"),
        }
        for shot in shots:
            agg = obj["fewshot_prototype_routing"][str(shot)]["aggregate"]
            row[f"prototype_{shot}shot_macro_f1"] = agg.get("real_macro_f1_mean")
            row[f"prototype_{shot}shot_top2_recall"] = agg.get("routing_top2_recall_mean")
            row[f"prototype_{shot}shot_gain_over_shuffle"] = agg.get(
                "macro_f1_gain_over_within_category_shuffle"
            )
        summary.append(row)
    result["summary"] = summary

    out = Path(args.out_dir) / f"fold_{args.fold}" / "prototype_addressability_audit.json"
    _json_dump(result, out)
    print(json.dumps({"summary": summary}, indent=2, allow_nan=True))
    print(f"Wrote {out}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--features", type=str, required=True)
    p.add_argument(
        "--representations",
        type=str,
        default="oracle/raw_patch,sensor/raw_patch/k8/score_softmax",
        help="Comma-separated artifact representation keys.",
    )
    p.add_argument("--shots", type=str, default="1,2,4,8")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/prototype_addressability_audit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
