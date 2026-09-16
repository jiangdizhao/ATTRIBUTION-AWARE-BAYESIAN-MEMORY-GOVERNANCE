# Four-branch structural-credit audit

## Purpose

This audit is the immediate causal control after the sequential local-update / interference audit.

The previous experiment established two different facts:

1. factor-specific persistent state can be updated with exact **state locality**; and
2. the resulting routing still has some **behavioral interference** because factor scores compete.

The missing question is whether the *correct write address itself* is responsible for the useful learning.

> Under the exact same stream, support, query locations, and revealed feedback, does writing a verified observation to the correct factor address improve future same-factor routing more cleanly than either broadcasting the observation across memory or writing it to an intentionally wrong address?

This remains a source-side audit over the existing cached frozen-DINO evidence artifact. It does not rerun DINO and it does not use the outer held-out target categories.

## Four matched branches

Every branch receives exactly the same stream item and, at a queried position, the same revealed Core-4 label. The only manipulated variable is the credit vector used for the persistent memory write.

Let `G=4` and let `alpha_g` denote the effective credit assigned to factor memory `g`.

### A. `no_update`

```text
alpha_g = 0  for every g
```

The query still occurs, so supervision cost is matched, but the feedback is discarded. This measures what would happen without persistent learning.

### B. `global_broadcast`

```text
alpha_g = 1/G  for every g
```

Every factor memory changes. The `1/G` weight is deliberate: the total effective write mass is one observation, matching the correct and shuffled one-hot branches. This prevents a trivial `G`-fold update-magnitude confound.

### C. `correct_address`

```text
alpha_y = 1
alpha_g = 0 for g != y
```

Only the persistent memory address corresponding to the verified factor is updated.

### D. `shuffled_address`

```text
alpha_pi(y) = 1
alpha_g = 0 otherwise
```

`pi` is a fixed seeded derangement of `AK/HS/QS/ZW` within each run, so no true label maps to itself. This branch receives exactly one local write per query, just like the correct branch, but the write is intentionally routed to the wrong address.

## Weighted sufficient-statistic update

For one normalized evidence vector `z`, every branch uses the same generic update:

```text
n_g   <- n_g + alpha_g
sum_g <- sum_g + alpha_g * z
```

`n_g` is therefore an effective sample count and may be fractional in the global branch.

The routing model is the same shared-diagonal shrunk-memory model used in the previous audit:

```text
kappa_g = kappa0 + n_g
mu_g* = (kappa0 * mu0 + sum_g) / kappa_g
```

with the same `diag_shrunk` and `bayes_predictive` score variants. The weighted scorer is algebraically identical to the previous scorer when counts are integers; a unit test checks that equivalence.

## Strictly matched protocol

For a given source-CV fold and repeat, all four branches share:

* the same held-out product categories;
* the same adaptation-stream / sentinel split;
* the same initial verified support per Core-4 factor;
* the same frozen label-free shared diagonal prior;
* the same randomized stream order;
* the same uniformly sampled, label/model-independent query positions;
* the same revealed feedback label at each queried position;
* the same routing rule and `kappa0`;
* the same sentinel probe bank and checkpoints.

Every stream item is predicted before any possible feedback/update, preserving the prequential test-then-train rule.

## Update-integrity checks

After every queried event the script verifies the exact state footprint:

```text
no_update         -> no factor state changes
global_broadcast  -> all four factor states change
correct_address   -> only the verified factor changes
shuffled_address  -> only pi(verified factor) changes
```

For every learning branch, the logged total effective-count increase is exactly `1.0` per queried event.

## Primary measurements

### 1. Checkpoint learning curves

At `0, 4, 8, 16, 32` feedback events, evaluate the never-updated sentinel set and report:

* macro-F1;
* per-factor F1;
* true-vs-best-wrong margin;
* change from the branch's initial state;
* paired macro-F1 difference versus `no_update`.

### 2. Structural-credit / interference matrix

Before and after every queried event, rescore a fixed sentinel probe bank.

Rows are always indexed by the **true verified factor**, including the shuffled branch. Columns are the factor identity of the sentinel probes. This makes branch effects directly comparable.

Report:

* diagonal true-factor margin change;
* off-diagonal absolute margin movement;
* off-diagonal harmful-flip rate;
* diagonal-to-off-diagonal locality ratio.

### 3. Paired final contrasts

At the last feedback checkpoint, compute run-paired macro-F1 differences:

```text
correct_address - no_update
correct_address - global_broadcast
correct_address - shuffled_address
global_broadcast - no_update
shuffled_address - no_update
```

These are paired within the same source-CV fold/repeat, so support, stream, and queries cancel as nuisance variation.

## Scientific decision

Evidence for structural credit assignment should not be declared from final F1 alone.

The desired pattern is:

```text
correct_address:
    positive future same-factor benefit
    competitive or best checkpoint learning
    limited off-factor harmful movement

shuffled_address:
    weaker same-factor benefit and/or more harmful interference

global_broadcast:
    learning benefit no better than correct addressing and greater collateral movement

no_update:
    no persistent learning effect
```

A particularly important causal contrast is `correct_address` versus `shuffled_address`, because both perform exactly one local write with the same total update mass; only the write destination differs.

If shuffled writes perform as well as correct writes, the current evidence does not support the claim that the factor address is carrying useful structural credit.

If global broadcast performs equally well with no extra collateral cost, selective addressing is not yet justified.

## Command

From the repository root on the existing `stage1-factorizability-audit` branch:

```powershell
python scripts/abmg_four_branch_structural_credit_audit.py --fold 0 --features "outputs\patch_aggregation_confirm\fold_0\patch_aggregation_features.pt" --representations "oracle/raw_patch,sensor/raw_patch/k8/score_softmax" --initial-shots 1 --query-budget 32 --checkpoints "4,8,16,32" --repeats 10 --sentinel-fraction 0.25 --probe-per-label 16 --bayes-kappa0 1.0 --seed 0 --out-dir outputs/four_branch_structural_credit_audit
```

Linux path form:

```bash
python scripts/abmg_four_branch_structural_credit_audit.py \
  --fold 0 \
  --features "outputs/patch_aggregation_confirm/fold_0/patch_aggregation_features.pt" \
  --representations "oracle/raw_patch,sensor/raw_patch/k8/score_softmax" \
  --initial-shots 1 \
  --query-budget 32 \
  --checkpoints "4,8,16,32" \
  --repeats 10 \
  --sentinel-fraction 0.25 \
  --probe-per-label 16 \
  --bayes-kappa0 1.0 \
  --seed 0 \
  --out-dir outputs/four_branch_structural_credit_audit
```

Output:

```text
outputs/four_branch_structural_credit_audit/fold_0/four_branch_structural_credit_audit.json
```

## Tests

Run the new focused tests first:

```bash
python -m unittest discover -s tests -p "test_four_branch_structural_credit_audit.py" -v
```

Then run the existing sequential-audit tests as a regression check:

```bash
python -m unittest discover -s tests -p "test_sequential_local_update_audit.py" -v
```

## Interpretation boundary

A positive result would support a specific external-memory primitive:

```text
frozen evidence
-> sparse verification
-> correct persistent write address
-> localized sufficient-statistic update
-> better future routing with bounded collateral competition
```

It would still not prove automatic factor discovery, semantic/causal disentanglement, a deployable query policy, calibrated online responsibility, final C1/C2, or end-to-end anomaly-detection improvement.
