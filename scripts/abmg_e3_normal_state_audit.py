#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched ABMG E3 sparse normal-state audit.

This script consumes the frozen E3 evidence cache.  It does not run DINO.

For every source-CV fold, development-product normal TRAIN vectors fit only the
source priors.  Held-out products then receive identical sparse state supports,
verified-normal update streams, contamination positions, and evaluation probes.

Default stationary phase:
  deterministic_diag,nig,hierarchical_nig
  shots=1,2,4
  support seeds=0,1,2
  checkpoints=0,4,8,16,32
  contamination=0

Default contamination phase should be requested explicitly:
  deterministic_diag,nig,hierarchical_nig,robust_nig
  shots=2,4
  checkpoints=32
  contamination=0,0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from abmg_e3_normal_state_models import (
    MODEL_NAMES,
    RobustDiagonalNIG,
    build_normal_state,
    fit_hierarchical_nig_prior,
    fit_weak_nig_prior,
    l2_normalize_rows,
)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")


def parse_csv(spec: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in str(spec).split(",") if x.strip())


def parse_int_csv(spec: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if not vals:
        raise ValueError("integer list is empty")
    return vals


def parse_float_csv(spec: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in str(spec).split(",") if x.strip())
    if not vals:
        raise ValueError("float list is empty")
    return vals


def parse_cv_folds(spec: str, n_folds: int) -> Tuple[int, ...]:
    s = str(spec).strip().lower()
    if s == "all":
        return tuple(range(int(n_folds)))
    vals = tuple(sorted(set(int(x) for x in parse_csv(s))))
    if not vals:
        raise ValueError("no source-CV folds requested")
    if min(vals) < 0 or max(vals) >= int(n_folds):
        raise ValueError(f"source CV folds must be in [0,{n_folds-1}]")
    return vals


def _hash_order(seed: int, key: str) -> str:
    return hashlib.sha1(f"{int(seed)}|{key}".encode("utf-8")).hexdigest()


def _load_artifact(path: str | Path) -> Dict[str, Any]:
    try:
        a = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        a = torch.load(Path(path), map_location="cpu")
    if a.get("schema") != "abmg.e3.evidence.v1":
        raise ValueError(f"unexpected E3 evidence schema={a.get('schema')!r}")
    return a


def _build_index(artifact: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = artifact["items"]
    x = artifact["evidence"].float().numpy().astype(np.float64)
    x = l2_normalize_rows(x)

    out: Dict[str, Dict[str, Any]] = {}
    by_cat_role: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i, item in enumerate(items):
        by_cat_role[str(item["category"])][str(item["role"])].append(i)

    for cat, roles in by_cat_role.items():
        slot: Dict[str, Any] = {}
        for role in ("train_normal", "test_normal", "defect"):
            idx = np.asarray(roles.get(role, []), dtype=np.int64)
            slot[role] = x[idx] if len(idx) else np.empty((0, x.shape[1]))
            slot[f"{role}_ids"] = [
                str(items[int(i)]["image_id"]) for i in idx.tolist()
            ]
            if role == "defect":
                slot["defect_sources"] = [
                    str(items[int(i)]["defect_source_offline_only"])
                    for i in idx.tolist()
                ]
        out[cat] = slot
    return out


def _select_state_support(
    x: np.ndarray,
    ids: Sequence[str],
    shots: int,
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    if len(x) != len(ids):
        raise ValueError("state support ids/vectors mismatch")
    if len(x) < int(shots):
        raise ValueError(f"need {shots} train normals, have {len(x)}")
    order = sorted(
        range(len(ids)),
        key=lambda i: _hash_order(int(seed), str(ids[i])),
    )
    chosen = order[: int(shots)]
    return x[chosen], [str(ids[i]) for i in chosen]


def _fixed_normal_stream_split(
    x: np.ndarray,
    ids: Sequence[str],
    *,
    max_updates: int,
    stream_seed: int,
    min_sentinel: int,
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    if len(x) != len(ids):
        raise ValueError("normal test ids/vectors mismatch")
    need = int(max_updates) + int(min_sentinel)
    if len(x) < need:
        raise ValueError(
            f"need >= {need} test normals for update+sentinel, have {len(x)}"
        )
    order = sorted(
        range(len(ids)),
        key=lambda i: _hash_order(int(stream_seed), str(ids[i])),
    )
    q_idx = order[: int(max_updates)]
    s_idx = order[int(max_updates) :]
    return (
        x[q_idx],
        [str(ids[i]) for i in q_idx],
        x[s_idx],
        [str(ids[i]) for i in s_idx],
    )


def _contamination_plan(
    n_updates: int,
    noise_rate: float,
    n_defects: int,
    seed: int,
) -> Dict[int, int]:
    if not (0.0 <= float(noise_rate) < 1.0):
        raise ValueError("noise_rate must be in [0,1)")
    n = int(round(float(noise_rate) * int(n_updates)))
    if n <= 0:
        return {}
    if n_defects <= 0:
        raise ValueError("contamination requested but no defect probes exist")
    rng = np.random.default_rng(int(seed))
    positions = np.sort(
        rng.choice(int(n_updates), size=n, replace=False)
    ).tolist()
    defect_order = rng.permutation(int(n_defects)).tolist()
    return {
        int(pos): int(defect_order[i % len(defect_order)])
        for i, pos in enumerate(positions)
    }


def _fpr_at_95_tpr(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    mask = tpr >= 0.95
    return float(np.min(fpr[mask])) if np.any(mask) else 1.0


def evaluate_state(
    state: Any,
    sentinel_normal: np.ndarray,
    defects: np.ndarray,
    reference_state: Any,
) -> Dict[str, Any]:
    lp_n = state.predictive_log_prob(sentinel_normal)
    lp_d = state.predictive_log_prob(defects)
    s_n = -lp_n
    s_d = -lp_d
    y = np.concatenate(
        [
            np.zeros(len(s_n), dtype=np.int64),
            np.ones(len(s_d), dtype=np.int64),
        ]
    )
    score = np.concatenate([s_n, s_d])
    auroc = (
        float(roc_auc_score(y, score))
        if len(np.unique(y)) == 2
        else float("nan")
    )

    coverage = state.predictive_interval_coverage(
        sentinel_normal,
        level=0.90,
    )
    return {
        "auroc": auroc,
        "fpr_at_95_tpr": _fpr_at_95_tpr(y, score),
        "normal_surprise_mean": float(np.mean(s_n)),
        "normal_surprise_std": float(np.std(s_n)),
        "defect_surprise_mean": float(np.mean(s_d)),
        "defect_minus_normal_surprise_gap": float(
            np.mean(s_d) - np.mean(s_n)
        ),
        "coverage_90": float(coverage),
        "coverage_90_abs_error": float(abs(coverage - 0.90)),
        "uncertainty": state.uncertainty(),
        "state_drift_from_initial": state.state_drift(reference_state),
        "state_summary": state.state_summary(),
        "n_sentinel_normals": int(len(s_n)),
        "n_defects": int(len(s_d)),
    }


def _finite_mean_std(xs: Iterable[float]) -> Tuple[float, float]:
    a = np.asarray([float(x) for x in xs], dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def _aggregate_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_category_seed_units": int(len(rows))}
    for key in (
        "auroc",
        "fpr_at_95_tpr",
        "normal_surprise_mean",
        "normal_surprise_std",
        "defect_minus_normal_surprise_gap",
        "coverage_90",
        "coverage_90_abs_error",
    ):
        m, s = _finite_mean_std(r["metrics"][key] for r in rows)
        out[f"{key}_mean"] = m
        out[f"{key}_std"] = s

    m, s = _finite_mean_std(
        r["metrics"]["state_drift_from_initial"]["mean_l2"]
        for r in rows
    )
    out["state_mean_l2_drift_mean"] = m
    out["state_mean_l2_drift_std"] = s

    m, s = _finite_mean_std(
        r["metrics"]["uncertainty"]["epistemic_mean_variance"]
        for r in rows
    )
    out["epistemic_mean_variance_mean"] = m
    out["epistemic_mean_variance_std"] = s

    robust_clean = []
    robust_bad = []
    for r in rows:
        ss = r["metrics"]["state_summary"]
        if np.isfinite(float(ss.get("mean_clean_influence", float("nan")))):
            robust_clean.append(float(ss["mean_clean_influence"]))
        if np.isfinite(
            float(ss.get("mean_contaminated_influence", float("nan")))
        ):
            robust_bad.append(float(ss["mean_contaminated_influence"]))
    if robust_clean:
        out["robust_clean_influence_mean"] = float(np.mean(robust_clean))
    if robust_bad:
        out["robust_contaminated_influence_mean"] = float(
            np.mean(robust_bad)
        )
    return out


def aggregate_runs(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[
        Tuple[str, int, float, int],
        List[Dict[str, Any]],
    ] = defaultdict(list)
    for r in rows:
        key = (
            str(r["model"]),
            int(r["shots"]),
            float(r["noise_rate"]),
            int(r["checkpoint"]),
        )
        grouped[key].append(r)

    cells: Dict[str, Any] = {}
    for (model, shots, noise, checkpoint), xs in sorted(grouped.items()):
        key = (
            f"{model}__shot{shots}__noise{noise:g}"
            f"__q{checkpoint}"
        )
        cells[key] = {
            "model": model,
            "shots": shots,
            "noise_rate": noise,
            "checkpoint": checkpoint,
            **_aggregate_group(xs),
        }

    # Matched clean/noisy deltas for each model/shots/checkpoint when both exist.
    noise_effects: Dict[str, Any] = {}
    index = {
        (
            v["model"],
            int(v["shots"]),
            float(v["noise_rate"]),
            int(v["checkpoint"]),
        ): v
        for v in cells.values()
    }
    for key, clean in list(index.items()):
        model, shots, noise, checkpoint = key
        if abs(noise) > 1e-12:
            continue
        noisy_keys = [
            k
            for k in index
            if k[0] == model
            and k[1] == shots
            and k[3] == checkpoint
            and k[2] > 0
        ]
        for nk in noisy_keys:
            noisy = index[nk]
            tag = (
                f"{model}__shot{shots}__noise{nk[2]:g}"
                f"__q{checkpoint}"
            )
            noise_effects[tag] = {
                "auroc_delta_noisy_minus_clean": float(
                    noisy["auroc_mean"] - clean["auroc_mean"]
                ),
                "coverage_error_delta_noisy_minus_clean": float(
                    noisy["coverage_90_abs_error_mean"]
                    - clean["coverage_90_abs_error_mean"]
                ),
                "state_drift_delta_noisy_minus_clean": float(
                    noisy["state_mean_l2_drift_mean"]
                    - clean["state_mean_l2_drift_mean"]
                ),
                "fpr95_delta_noisy_minus_clean": float(
                    noisy["fpr_at_95_tpr_mean"]
                    - clean["fpr_at_95_tpr_mean"]
                ),
            }
    return {"cells": cells, "noise_effects": noise_effects}


def _support_seed_variability(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Average within-category AUROC std across support seeds."""
    groups: Dict[
        Tuple[str, int, float, int, int, str],
        List[float],
    ] = defaultdict(list)
    for r in rows:
        groups[
            (
                str(r["model"]),
                int(r["shots"]),
                float(r["noise_rate"]),
                int(r["checkpoint"]),
                int(r["source_cv_fold"]),
                str(r["category"]),
            )
        ].append(float(r["metrics"]["auroc"]))

    by_cell: Dict[Tuple[str, int, float, int], List[float]] = defaultdict(list)
    for key, vals in groups.items():
        if len(vals) <= 1:
            continue
        model, shots, noise, checkpoint, _, _ = key
        by_cell[(model, shots, noise, checkpoint)].append(
            float(np.std(vals, ddof=0))
        )

    out: Dict[str, Any] = {}
    for (model, shots, noise, checkpoint), vals in sorted(by_cell.items()):
        out[
            f"{model}__shot{shots}__noise{noise:g}__q{checkpoint}"
        ] = {
            "mean_within_category_support_seed_auroc_std": float(
                np.mean(vals)
            ),
            "n_categories": int(len(vals)),
        }
    return out


def run_cv_fold(
    *,
    artifact: Mapping[str, Any],
    data: Mapping[str, Mapping[str, Any]],
    source_cv_fold: int,
    models: Sequence[str],
    shots_list: Sequence[int],
    support_seeds: Sequence[int],
    checkpoints: Sequence[int],
    noise_rates: Sequence[float],
    stream_seed: int,
    contamination_seed: int,
    min_sentinel_normals: int,
    weak_kappa0: float,
    weak_alpha0: float,
    hierarchical_alpha0: float,
    robust_df: float,
    robust_min_influence: float,
    deterministic_floor_ratio: float,
) -> Dict[str, Any]:
    source_categories = tuple(str(x) for x in artifact["source_categories"])
    test_categories = tuple(
        str(x) for x in artifact["source_cv_groups"][int(source_cv_fold)]
    )
    train_categories = tuple(
        c for c in source_categories if c not in set(test_categories)
    )

    dev_normal = {
        c: np.asarray(data[c]["train_normal"], dtype=np.float64)
        for c in train_categories
        if c in data and len(data[c]["train_normal"]) >= 2
    }
    weak_prior = fit_weak_nig_prior(
        dev_normal,
        kappa0=float(weak_kappa0),
        alpha0=float(weak_alpha0),
    )
    hierarchical_prior = fit_hierarchical_nig_prior(
        dev_normal,
        alpha0=float(hierarchical_alpha0),
    )

    max_checkpoint = max(int(x) for x in checkpoints)
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for category in test_categories:
        d = data.get(category)
        if d is None:
            skipped.append({"category": category, "reason": "missing_category"})
            continue
        train_x = np.asarray(d["train_normal"], dtype=np.float64)
        train_ids = list(d["train_normal_ids"])
        test_n = np.asarray(d["test_normal"], dtype=np.float64)
        test_n_ids = list(d["test_normal_ids"])
        defects = np.asarray(d["defect"], dtype=np.float64)

        if len(defects) == 0:
            skipped.append({"category": category, "reason": "no_defects"})
            continue
        try:
            query_x, query_ids, sentinel_x, sentinel_ids = (
                _fixed_normal_stream_split(
                    test_n,
                    test_n_ids,
                    max_updates=max_checkpoint,
                    stream_seed=int(stream_seed),
                    min_sentinel=int(min_sentinel_normals),
                )
            )
        except ValueError as exc:
            skipped.append({"category": category, "reason": str(exc)})
            continue

        for shots in shots_list:
            for support_seed in support_seeds:
                try:
                    init_x, init_ids = _select_state_support(
                        train_x,
                        train_ids,
                        int(shots),
                        int(support_seed),
                    )
                except ValueError as exc:
                    skipped.append(
                        {
                            "category": category,
                            "shots": int(shots),
                            "support_seed": int(support_seed),
                            "reason": str(exc),
                        }
                    )
                    continue

                for noise_rate in noise_rates:
                    contam = _contamination_plan(
                        max_checkpoint,
                        float(noise_rate),
                        len(defects),
                        int(contamination_seed)
                        + 100_000 * int(source_cv_fold),
                    )

                    states: Dict[str, Any] = {}
                    refs: Dict[str, Any] = {}
                    for model_name in models:
                        st = build_normal_state(
                            model_name,
                            weak_prior=weak_prior,
                            hierarchical_prior=hierarchical_prior,
                            deterministic_floor_ratio=float(
                                deterministic_floor_ratio
                            ),
                            robust_df=float(robust_df),
                            robust_min_influence=float(
                                robust_min_influence
                            ),
                        )
                        # Identical initialization evidence; robust influence is
                        # disabled so B3 starts from the same sparse support.
                        for x0 in init_x:
                            st.update(x0, 1.0, robust=False)
                        states[model_name] = st
                        refs[model_name] = st.clone()

                    if 0 in checkpoints:
                        for model_name, st in states.items():
                            rows.append(
                                {
                                    "source_cv_fold": int(source_cv_fold),
                                    "category": category,
                                    "model": model_name,
                                    "shots": int(shots),
                                    "support_seed": int(support_seed),
                                    "noise_rate": float(noise_rate),
                                    "checkpoint": 0,
                                    "n_contaminated_updates": 0,
                                    "init_image_ids": init_ids,
                                    "query_stream_ids": query_ids,
                                    "sentinel_image_ids": sentinel_ids,
                                    "metrics": evaluate_state(
                                        st,
                                        sentinel_x,
                                        defects,
                                        refs[model_name],
                                    ),
                                }
                            )

                    n_bad = 0
                    for t in range(1, max_checkpoint + 1):
                        pos = t - 1
                        is_bad = pos in contam
                        x_update = (
                            defects[int(contam[pos])]
                            if is_bad
                            else query_x[pos]
                        )
                        if is_bad:
                            n_bad += 1

                        for model_name, st in states.items():
                            st.update(
                                x_update,
                                1.0,
                                robust=True,
                            )
                            if isinstance(st, RobustDiagonalNIG):
                                st.record_last_weight(
                                    contaminated=bool(is_bad)
                                )

                        if t in checkpoints:
                            for model_name, st in states.items():
                                rows.append(
                                    {
                                        "source_cv_fold": int(source_cv_fold),
                                        "category": category,
                                        "model": model_name,
                                        "shots": int(shots),
                                        "support_seed": int(support_seed),
                                        "noise_rate": float(noise_rate),
                                        "checkpoint": int(t),
                                        "n_contaminated_updates": int(n_bad),
                                        "init_image_ids": init_ids,
                                        "query_stream_ids": query_ids,
                                        "sentinel_image_ids": sentinel_ids,
                                        "metrics": evaluate_state(
                                            st,
                                            sentinel_x,
                                            defects,
                                            refs[model_name],
                                        ),
                                    }
                                )

    return {
        "schema": "abmg.e3.source_cv_fold.v1",
        "source_cv_fold": int(source_cv_fold),
        "train_categories": list(train_categories),
        "test_categories": list(test_categories),
        "weak_prior_metadata": weak_prior.metadata,
        "hierarchical_prior_metadata": hierarchical_prior.metadata,
        "runs": rows,
        "aggregate": aggregate_runs(rows),
        "support_seed_variability": _support_seed_variability(rows),
        "skipped": skipped,
    }


def _all_rows(cv_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cv in cv_results:
        out.extend(cv["runs"])
    return out


def _decision_diagnostics(
    aggregate: Mapping[str, Any],
    support_var: Mapping[str, Any],
) -> Dict[str, Any]:
    cells = aggregate["cells"]
    out: Dict[str, Any] = {
        "canonical_c4": [],
        "hierarchical_vs_nig": [],
        "robust_under_noise": [],
    }

    def get(model: str, shots: int, noise: float, q: int):
        return cells.get(f"{model}__shot{shots}__noise{noise:g}__q{q}")

    checkpoints = sorted(
        {
            int(v["checkpoint"])
            for v in cells.values()
        }
    )
    shots_vals = sorted({int(v["shots"]) for v in cells.values()})
    noise_vals = sorted({float(v["noise_rate"]) for v in cells.values()})

    for shots in shots_vals:
        for q in checkpoints:
            d = get("deterministic_diag", shots, 0.0, q)
            n = get("nig", shots, 0.0, q)
            if d and n:
                key_d = f"deterministic_diag__shot{shots}__noise0__q{q}"
                key_n = f"nig__shot{shots}__noise0__q{q}"
                sd = support_var.get(key_d, {}).get(
                    "mean_within_category_support_seed_auroc_std",
                    float("nan"),
                )
                sn = support_var.get(key_n, {}).get(
                    "mean_within_category_support_seed_auroc_std",
                    float("nan"),
                )
                out["canonical_c4"].append(
                    {
                        "shots": shots,
                        "checkpoint": q,
                        "nig_minus_deterministic_auroc": float(
                            n["auroc_mean"] - d["auroc_mean"]
                        ),
                        "nig_minus_deterministic_coverage_error": float(
                            n["coverage_90_abs_error_mean"]
                            - d["coverage_90_abs_error_mean"]
                        ),
                        "support_seed_auroc_std_reduction_fraction": (
                            float((sd - sn) / sd)
                            if np.isfinite(sd)
                            and sd > 0
                            and np.isfinite(sn)
                            else float("nan")
                        ),
                        "nig_operationally_noninferior": bool(
                            n["auroc_mean"] - d["auroc_mean"] >= -0.01
                        ),
                    }
                )

            n = get("nig", shots, 0.0, q)
            h = get("hierarchical_nig", shots, 0.0, q)
            if n and h:
                out["hierarchical_vs_nig"].append(
                    {
                        "shots": shots,
                        "checkpoint": q,
                        "hierarchical_minus_nig_auroc": float(
                            h["auroc_mean"] - n["auroc_mean"]
                        ),
                        "hierarchical_minus_nig_coverage_error": float(
                            h["coverage_90_abs_error_mean"]
                            - n["coverage_90_abs_error_mean"]
                        ),
                    }
                )

    for shots in shots_vals:
        for q in checkpoints:
            for noise in noise_vals:
                if noise <= 0:
                    continue
                r0 = get("robust_nig", shots, 0.0, q)
                rn = get("robust_nig", shots, noise, q)
                n0 = get("nig", shots, 0.0, q)
                nn = get("nig", shots, noise, q)
                if r0 and rn and n0 and nn:
                    out["robust_under_noise"].append(
                        {
                            "shots": shots,
                            "checkpoint": q,
                            "noise_rate": noise,
                            "nig_auroc_damage": float(
                                nn["auroc_mean"] - n0["auroc_mean"]
                            ),
                            "robust_auroc_damage": float(
                                rn["auroc_mean"] - r0["auroc_mean"]
                            ),
                            "robust_clean_minus_nig_clean_auroc": float(
                                r0["auroc_mean"] - n0["auroc_mean"]
                            ),
                            "robust_clean_influence_mean": rn.get(
                                "robust_clean_influence_mean",
                                float("nan"),
                            ),
                            "robust_contaminated_influence_mean": rn.get(
                                "robust_contaminated_influence_mean",
                                float("nan"),
                            ),
                        }
                    )
    return out


def run(args: argparse.Namespace) -> int:
    artifact = _load_artifact(args.evidence)
    if int(artifact["outer_fold"]) != int(args.fold):
        raise ValueError("evidence outer fold does not match --fold")

    models = parse_csv(args.models)
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown models={unknown}; allowed={MODEL_NAMES}")

    shots_list = tuple(sorted(set(parse_int_csv(args.shots))))
    support_seeds = parse_int_csv(args.support_seeds)
    checkpoints = tuple(sorted(set(parse_int_csv(args.checkpoints))))
    if checkpoints[0] < 0:
        raise ValueError("checkpoints must be >=0")
    noise_rates = tuple(sorted(set(parse_float_csv(args.noise_rates))))
    if any((x < 0 or x >= 1) for x in noise_rates):
        raise ValueError("noise rates must be in [0,1)")

    cv_folds = parse_cv_folds(
        args.source_cv_folds,
        len(artifact["source_cv_groups"]),
    )
    data = _build_index(artifact)

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "schema": "abmg.e3.audit_config.v1",
        "outer_fold": int(args.fold),
        "evidence": str(args.evidence),
        "target_categories_untouched": list(
            artifact["target_categories_untouched"]
        ),
        "source_cv_folds": list(cv_folds),
        "models": list(models),
        "shots": list(shots_list),
        "support_seeds": list(support_seeds),
        "checkpoints": list(checkpoints),
        "noise_rates": list(noise_rates),
        "stream_seed": int(args.stream_seed),
        "contamination_seed": int(args.contamination_seed),
        "min_sentinel_normals": int(args.min_sentinel_normals),
        "weak_prior": {
            "kappa0": float(args.weak_kappa0),
            "alpha0": float(args.weak_alpha0),
        },
        "hierarchical_prior": {
            "alpha0": float(args.hierarchical_alpha0),
        },
        "robust_nig": {
            "df": float(args.robust_df),
            "min_influence": float(args.robust_min_influence),
        },
        "deterministic_floor_ratio": float(
            args.deterministic_floor_ratio
        ),
        "invariants": {
            "representation": "raw DINO",
            "evidence": (
                "fixed 4-shot detector top-8 score-softmax pooled vector"
            ),
            "online_update_weight": 1.0,
            "factor_addressed_credit": False,
            "defect_updates_normal_state": (
                "only evaluator-injected contamination stress"
            ),
        },
    }
    _json_dump(resolved, out_dir / "resolved_config.json")

    cv_results = []
    for cv_fold in cv_folds:
        print(f"=== E3 source-CV fold {cv_fold} ===")
        result = run_cv_fold(
            artifact=artifact,
            data=data,
            source_cv_fold=int(cv_fold),
            models=models,
            shots_list=shots_list,
            support_seeds=support_seeds,
            checkpoints=checkpoints,
            noise_rates=noise_rates,
            stream_seed=int(args.stream_seed),
            contamination_seed=int(args.contamination_seed),
            min_sentinel_normals=int(args.min_sentinel_normals),
            weak_kappa0=float(args.weak_kappa0),
            weak_alpha0=float(args.weak_alpha0),
            hierarchical_alpha0=float(args.hierarchical_alpha0),
            robust_df=float(args.robust_df),
            robust_min_influence=float(args.robust_min_influence),
            deterministic_floor_ratio=float(
                args.deterministic_floor_ratio
            ),
        )
        cv_results.append(result)
        cv_dir = out_dir / f"cv_{cv_fold}"
        cv_dir.mkdir(parents=True, exist_ok=True)
        _json_dump(result, cv_dir / "e3_cv_summary.json")
        print(f"Wrote {cv_dir / 'e3_cv_summary.json'}")

    rows = _all_rows(cv_results)
    aggregate = aggregate_runs(rows)
    support_var = _support_seed_variability(rows)
    summary = {
        "schema": "abmg.e3.normal_state_audit.summary.v1",
        "outer_fold": int(args.fold),
        "target_categories_untouched": list(
            artifact["target_categories_untouched"]
        ),
        "n_cv_folds_completed": len(cv_results),
        "aggregate": aggregate,
        "support_seed_variability": support_var,
        "decision_diagnostics": _decision_diagnostics(
            aggregate,
            support_var,
        ),
        "scientific_question": (
            "Which sparse normal-state model gives stable surprise and "
            "uncertainty under identical verified-normal evidence?"
        ),
    }
    _json_dump(summary, out_dir / "e3_summary.json")
    print(f"Wrote E3 summary to: {out_dir / 'e3_summary.json'}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--evidence", type=str, required=True)
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/e3_normal_state",
    )
    p.add_argument("--source-cv-folds", type=str, default="all")

    p.add_argument(
        "--models",
        type=str,
        default="deterministic_diag,nig,hierarchical_nig",
    )
    p.add_argument("--shots", type=str, default="1,2,4")
    p.add_argument("--support-seeds", type=str, default="0,1,2")
    p.add_argument("--checkpoints", type=str, default="0,4,8,16,32")
    p.add_argument("--noise-rates", type=str, default="0")

    p.add_argument("--stream-seed", type=int, default=20261005)
    p.add_argument("--contamination-seed", type=int, default=314159)
    p.add_argument("--min-sentinel-normals", type=int, default=8)

    p.add_argument("--weak-kappa0", type=float, default=0.01)
    p.add_argument("--weak-alpha0", type=float, default=2.5)
    p.add_argument("--hierarchical-alpha0", type=float, default=3.0)
    p.add_argument("--robust-df", type=float, default=4.0)
    p.add_argument("--robust-min-influence", type=float, default=0.05)
    p.add_argument(
        "--deterministic-floor-ratio",
        type=float,
        default=0.05,
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
