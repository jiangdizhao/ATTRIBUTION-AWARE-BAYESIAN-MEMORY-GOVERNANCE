# Five-branch structural-credit audit

## Purpose

The four-branch audit showed that, under identical sparse feedback, correct factor-addressed writes strongly outperform no-update, mass-matched global/broadcast writes, and deliberately shuffled writes. One methodological ambiguity remained: the four-branch `global_broadcast` condition used `alpha_g = 1/G`, whereas the original proposal's undifferentiated C2 control uses `alpha_g = 1` for every factor.

This confirmatory audit adds that proposal-faithful control without changing the frozen evidence, stream, query, scoring, or evaluation machinery.

## Five matched branches

For a verified feedback event with true factor `y` and `G=4` Core-4 addresses:

```text
no_update:
    alpha_g = 0                 for all g

global_broadcast:
    alpha_g = 1/G               for all g
    total effective write mass = 1

full_broadcast:
    alpha_g = 1                 for all g
    total effective write mass = G

correct_address:
    alpha_y = 1; alpha_g = 0    for g != y
    total effective write mass = 1

shuffled_address:
    alpha_pi(y) = 1             for a fixed seeded derangement pi
    total effective write mass = 1
```

The two broadcast branches answer different questions. `global_broadcast` controls total update mass and isolates write destination. `full_broadcast` matches the original proposal's undifferentiated update, where every memory component receives the full verified observation.

## What remains exactly matched

All five branches use the same frozen DINO-derived evidence vectors, source-CV fold, initial verified supports, shared diagonal background prior, stream/sentinel split, randomized stream order, query indices, revealed feedback labels, routing model, sentinel probes, and checkpoint schedule. Every prediction remains prequential: predict first, then reveal feedback only at the preselected query position, then update.

Outer ABMG target categories remain untouched.

## Primary comparison

The confirmatory C2 question is now:

> Does `correct_address` outperform both `global_broadcast` and `full_broadcast` while producing a stronger same-factor benefit and less harmful collateral interference?

The shuffled branch remains the wrong-destination negative control.

Primary final paired contrasts are:

```text
correct_address - no_update
correct_address - global_broadcast
correct_address - full_broadcast
correct_address - shuffled_address
global_broadcast - no_update
full_broadcast - no_update
full_broadcast - global_broadcast
shuffled_address - no_update
```

Mechanism metrics remain the same as the four-branch audit: macro-F1 learning curves, same-factor true-vs-best-wrong margin change, off-factor absolute margin movement, off-factor harmful-flip rate, and update-integrity checks.

## Interpretation

A particularly strong structural-credit result would be:

1. `correct_address` improves future routing substantially over no update;
2. `correct_address` beats mass-matched broadcast, showing that selective write destination matters when total write mass is controlled;
3. `correct_address` also beats full broadcast, showing that the result is not an artifact of giving each broadcast address only a fractional observation;
4. `shuffled_address` loses the advantage or becomes harmful, showing that local updating is useful only when the local destination is structurally correct.

If full broadcast matches or exceeds correct addressing, the selective-memory C2 claim must be narrowed or revised rather than hidden.

## Implementation

Script:

```text
scripts/abmg_five_branch_structural_credit_audit.py
```

The script reuses the tested implementation in `abmg_four_branch_structural_credit_audit.py` through a temporary runtime hook. The hook adds `full_broadcast` only while the five-branch experiment runs and restores the original four-branch module afterward. This prevents the confirmatory experiment from silently changing the historical four-branch implementation.

Focused tests:

```text
tests/test_five_branch_structural_credit_audit.py
```

## Recommended command

```powershell
python scripts/abmg_five_branch_structural_credit_audit.py --fold 0 --features "outputs\patch_aggregation_confirm\fold_0\patch_aggregation_features.pt" --representations "oracle/raw_patch,sensor/raw_patch/k8/score_softmax" --initial-shots 1 --query-budget 32 --checkpoints "4,8,16,32" --repeats 10 --sentinel-fraction 0.25 --probe-per-label 16 --bayes-kappa0 1.0 --seed 0 --out-dir outputs/five_branch_structural_credit_audit
```

Output:

```text
outputs/five_branch_structural_credit_audit/fold_0/five_branch_structural_credit_audit.json
```

## Interpretation boundary

This remains a source-side causal control over where verified evidence is written and how broadly it is broadcast. It does not establish automatic factor discovery, semantic or causal factors, deployable query policy, calibrated online probabilities, final end-to-end C1/C2, or end-to-end anomaly-detection improvement.
