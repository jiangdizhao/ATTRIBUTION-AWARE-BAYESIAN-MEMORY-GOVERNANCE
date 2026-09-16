#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-branch confirmatory structural-credit audit for ABMG.

This experiment extends ``abmg_four_branch_structural_credit_audit.py`` with a
proposal-faithful full-broadcast control while reusing the same tested stream,
scoring, checkpoint, and interference machinery.

Branches
--------
* no_update:        alpha_g = 0 for all g.
* global_broadcast: alpha_g = 1/G for all g (mass-matched broadcast).
* full_broadcast:   alpha_g = 1 for all g (original C2 global control).
* correct_address:  alpha_y = 1, all other alpha_g = 0.
* shuffled_address: alpha_pi(y) = 1 for a fixed seeded derangement pi.

All branches share the same source-CV split, initial supports, stream order,
sentinel set, query indices, revealed labels, shared diagonal prior, and routing
rule. Only write credit differs. Outer ABMG target categories remain untouched.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # direct execution: python scripts/...
    import abmg_four_branch_structural_credit_audit as four
    from abmg_prototype_addressability_audit import (
        CORE4,
        _finite_mean_std,
        parse_int_list,
        parse_str_list,
    )
except ImportError:  # module import from repo root / unit tests
    from scripts import abmg_four_branch_structural_credit_audit as four
    from scripts.abmg_prototype_addressability_audit import (
        CORE4,
        _finite_mean_std,
        parse_int_list,
        parse_str_list,
    )

MODELS: Tuple[str, ...] = four.MODELS
BRANCHES: Tuple[str, ...] = (
    "no_update",
    "global_broadcast",
    "full_broadcast",
    "correct_address",
    "shuffled_address",
)
MASS_MATCHED_CREDIT_WEIGHT = four.GLOBAL_CREDIT_WEIGHT
FULL_BROADCAST_CREDIT_WEIGHT = 1.0

# Keep immutable references to the original four-branch helpers. This matters
# because the run context temporarily patches the imported module globals.
_FOUR_CREDIT_VECTOR = four.credit_vector
_FOUR_EXPECTED_CHANGED_FACTORS = four.expected_changed_factors


def credit_vector(
    branch: str,
    true_label: str,
    shuffle_map: Mapping[str, str],
) -> Dict[str, float]:
    """Return the write-credit vector for one verified feedback event."""
    true_label = str(true_label)
    if true_label not in CORE4:
        raise ValueError(f"unknown factor={true_label!r}")
    if branch == "full_broadcast":
        return {g: FULL_BROADCAST_CREDIT_WEIGHT for g in CORE4}
    if branch in four.BRANCHES or branch in (
        "no_update",
        "global_broadcast",
        "correct_address",
        "shuffled_address",
    ):
        return _FOUR_CREDIT_VECTOR(branch, true_label, shuffle_map)
    raise ValueError(f"unknown branch={branch!r}")


def expected_changed_factors(
    branch: str,
    true_label: str,
    shuffle_map: Mapping[str, str],
) -> Tuple[str, ...]:
    if branch == "full_broadcast":
        return tuple(CORE4)
    if branch in four.BRANCHES or branch in (
        "no_update",
        "global_broadcast",
        "correct_address",
        "shuffled_address",
    ):
        return _FOUR_EXPECTED_CHANGED_FACTORS(branch, true_label, shuffle_map)
    raise ValueError(f"unknown branch={branch!r}")


def apply_branch_update(
    memory: Dict[str, Dict[str, Any]],
    branch: str,
    true_label: str,
    vector: np.ndarray,
    shuffle_map: Mapping[str, str],
) -> Tuple[str, ...]:
    """Apply exactly one branch update and verify its write footprint."""
    before = deepcopy(memory)
    four.update_memory_weighted(
        memory,
        credit_vector(branch, true_label, shuffle_map),
        vector,
    )
    changed = four.changed_factor_states(before, memory)
    expected = expected_changed_factors(branch, true_label, shuffle_map)
    if changed != expected:
        raise AssertionError(
            f"branch={branch} true={true_label}: changed={changed}, expected={expected}"
        )
    return changed


def _paired_final_contrasts(
    runs: Sequence[Dict[str, Any]], final_checkpoint: int
) -> Dict[str, Any]:
    """Predeclared paired final comparisons, including the fifth branch."""
    out: Dict[str, Any] = {}
    pairs = (
        ("correct_address", "no_update"),
        ("correct_address", "global_broadcast"),
        ("correct_address", "full_broadcast"),
        ("correct_address", "shuffled_address"),
        ("global_broadcast", "no_update"),
        ("full_broadcast", "no_update"),
        ("full_broadcast", "global_broadcast"),
        ("shuffled_address", "no_update"),
    )
    for model in MODELS:
        mrow: Dict[str, Any] = {}
        for lhs, rhs in pairs:
            vals = []
            for run in runs:
                matches = [
                    x
                    for x in run["checkpoints"]
                    if int(x["n_feedback"]) == int(final_checkpoint)
                ]
                if not matches:
                    continue
                cp = matches[0]
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


@contextmanager
def _five_branch_hooks() -> Iterator[None]:
    """Temporarily extend the already-tested four-branch machinery to five.

    The patch is restored in ``finally`` so importing this module does not alter
    behavior of the original four-branch script or its unit tests.
    """
    originals = {
        "BRANCHES": four.BRANCHES,
        "credit_vector": four.credit_vector,
        "expected_changed_factors": four.expected_changed_factors,
        "apply_branch_update": four.apply_branch_update,
        "_paired_final_contrasts": four._paired_final_contrasts,
    }
    four.BRANCHES = BRANCHES
    four.credit_vector = credit_vector
    four.expected_changed_factors = expected_changed_factors
    four.apply_branch_update = apply_branch_update
    four._paired_final_contrasts = _paired_final_contrasts
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(four, name, value)


def run(args: argparse.Namespace) -> int:
    artifact_path = Path(args.features)
    artifact = four._load_artifact(artifact_path)
    if int(artifact.get("outer_fold", -1)) != int(args.fold):
        raise ValueError(
            f"artifact outer_fold={artifact.get('outer_fold')} != requested fold={args.fold}"
        )
    requested = parse_str_list(args.representations)
    missing = [x for x in requested if x not in artifact["representations"]]
    if missing:
        raise KeyError(f"Representation(s) not found: {missing}")
    checkpoints = tuple(
        sorted(set(int(x) for x in parse_int_list(args.checkpoints)) | {0})
    )
    if max(checkpoints) > int(args.query_budget):
        raise ValueError("max checkpoint cannot exceed query budget")
    if int(args.initial_shots) <= 0:
        raise ValueError("initial_shots must be > 0")
    if float(args.bayes_kappa0) <= 0.0:
        raise ValueError("bayes_kappa0 must be > 0")

    items = artifact["items"]
    cv_groups = artifact["source_cv_groups"]
    result: Dict[str, Any] = {
        "schema": "abmg.five_branch_structural_credit_audit.v1",
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
            "query_policy": (
                "uniform random stream positions; label/model independent and "
                "shared by all five branches"
            ),
            "prequential_rule": (
                "every branch predicts before shared feedback/update on every stream item"
            ),
            "shared_initial_state": True,
            "shared_stream_and_queries": True,
            "mass_matched_global_credit_rule": (
                f"global_broadcast: alpha_g=1/G={MASS_MATCHED_CREDIT_WEIGHT:.6f}; "
                "total effective write mass=1"
            ),
            "proposal_faithful_full_broadcast_rule": (
                f"full_broadcast: alpha_g={FULL_BROADCAST_CREDIT_WEIGHT:.1f} for every factor; "
                f"total effective write mass=G={len(CORE4)}"
            ),
            "correct_credit_rule": (
                "one-hot at verified factor; total effective write mass=1"
            ),
            "shuffled_credit_rule": (
                "one-hot at fixed seeded derangement; total effective write mass=1"
            ),
            "no_update_rule": (
                "feedback revealed at identical query positions but persistent state unchanged"
            ),
            "factor_state": ["effective_n_g", "weighted_sum_g"],
            "shared_state": (
                "label-free source-training-product mean and diagonal variance; frozen during stream"
            ),
            "models": list(MODELS),
            "bayes_kappa0": float(args.bayes_kappa0),
            "sentinel_labels_never_used_for_updates": True,
            "effect_matrix_rows": "verified true factor in every branch",
            "effect_matrix_columns": "sentinel evaluation factor",
        },
        "primary_reading": (
            "Correct addressing must be compared with BOTH mass-matched broadcast and "
            "proposal-faithful full broadcast. Shuffled addressing remains the matched "
            "wrong-destination negative control."
        ),
        "interpretation_boundary": (
            "Source-side causal control over write location/broadcast strength only; not "
            "automatic factor discovery, semantic/causal factorization, deployable querying, "
            "online calibration, final C1/C2, or end-to-end anomaly-detection improvement."
        ),
        "representations": {},
    }

    with _five_branch_hooks():
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
                        four.run_one_sequence(
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
            agg = four.aggregate_runs(runs, checkpoints)
            result["representations"][key] = {"aggregate": agg, "runs": runs}

            print("  update-integrity pass fractions:")
            for branch in BRANCHES:
                value = agg["update_integrity"][branch]["pass_fraction_mean"]
                print(f"    {branch}: {value:.4f}")

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
            for name, values in contrasts.items():
                print(
                    f"    {name}: macro-F1 {values['macro_f1_mean']:+.4f} "
                    f"+- {values['macro_f1_std']:.4f}"
                )

    out_dir = Path(args.out_dir) / f"fold_{args.fold}"
    out_path = out_dir / "five_branch_structural_credit_audit.json"
    four._json_dump(result, out_path)
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
        default="outputs/five_branch_structural_credit_audit",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
