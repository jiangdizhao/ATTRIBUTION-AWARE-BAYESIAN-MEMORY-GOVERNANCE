# Sequential local-update / interference audit

## Purpose

This audit follows the discriminator/calibration result.  The previous static experiments established that:

1. frozen DINO evidence contains reusable Core-4 defect structure;
2. shared-diagonal geometry substantially improves sparse factor routing;
3. Normal-Normal prior shrinkage is useful mainly in the ultra-few-shot regime;
4. posterior-predictive variance does not change argmax under equal factor counts;
5. raw Gaussian score softmax is badly overconfident, although the score family is calibratable offline.

The next question is sequential rather than static:

> When a verified stream observation belongs to factor `g`, can we update only `g`'s external memory, improve future routing, and avoid harmful changes to the other factors?

No DINO inference or Real-IAD image loading is performed.  The audit consumes the existing `abmg.factorizability.features.v1` artifact.

## Fixed data boundary

* Core-4 factors: `AK`, `HS`, `QS`, `ZW`.
* Outer ABMG fold-0 target categories remain untouched.
* The four source-CV groups are inherited from the cached feature artifact.
* For each source-CV fold, the 18 training product categories provide:
  * one initial verified support per factor by default;
  * a label-free shared background mean and diagonal variance.
* The six held-out product categories are split into:
  * an adaptation stream;
  * a sentinel set whose labels are never used for updates.
* The stream/sentinel split is stratified by `(product category, defect code)` only to keep the offline diagnostic stable.  This split is evaluator-only.
* Stream order is randomized independently for each repeat.
* Query positions are sampled uniformly from stream positions and use neither labels nor model predictions.
* Every stream item is predicted **before** any possible feedback/update.

Thus the online part follows a test-then-train/prequential protocol.

## Memory state

For each factor `g`, the persistent state is only

```text
n_g
sum_g = sum_i z_i
```

where `z_i` is the L2-normalized frozen evidence vector of a verified example.

The label-free shared background state is

```text
mu0
Sigma = diag(sigma^2)
```

and is frozen during the stream.

A verified observation for factor `g` performs only

```text
n_g   <- n_g + 1
sum_g <- sum_g + z
```

The implementation checks after every update that all `h != g` factor states are bitwise unchanged.

## Routing models

The two routing models share exactly the same memory updates.

### A. `diag_shrunk`

The factor mean is

```text
kappa_g = kappa0 + n_g
mu_g* = (kappa0 * mu0 + sum_g) / kappa_g
```

and routing uses the shared diagonal Gaussian score

```text
s_g(z) = -0.5 * sum_j [ (z_j - mu_gj*)^2 / sigma_j^2 + log sigma_j^2 ]
```

with fixed `kappa0=1.0`.

### B. `bayes_predictive`

The same posterior mean is used, but the predictive variance is

```text
Sigma_pred,g = Sigma * (1 + 1/kappa_g)
```

In the earlier equal-shot audit all `n_g` were identical, so this multiplier could not change ranking.  In the present random-query stream the factor counts generally differ, so posterior-predictive variance can now affect routing.  This is the first audit where that distinction is operational rather than algebraically cancelled.

## Sparse-feedback schedule

Defaults:

```text
initial support: 1 verified example per factor from training products
query budget:    32 stream queries total
checkpoints:     0, 4, 8, 16, 32 feedback events
sentinel split:  25% of held-out Core-4 examples
immediate probe: up to 16 sentinel examples per factor
repeats:         10 per source-CV fold
```

The 32 query positions are sampled uniformly without replacement from the stream and are independent of labels.  Consequently the number of updates received by AK/HS/QS/ZW will usually be unequal.  The result reports these realized counts rather than forcing an oracle-balanced schedule.

## Learning metric

At feedback checkpoints the complete sentinel set is evaluated.  Primary quantities are:

* macro-F1;
* per-factor F1;
* true-vs-best-wrong score margin;
* macro-F1 change relative to the `0`-feedback memory;
* agreement between `diag_shrunk` and `bayes_predictive`.

A positive sequential result should show that sentinel performance improves as verified observations accumulate.  The sentinel set is never used for memory updates.

The audit also reports prequential stream metrics, where each stream sample is scored before any update on that sample.

## Structural-credit / interference matrix

Before and after every queried update to factor `g`, a fixed sentinel probe bank is rescored.  For each evaluation factor `h`, the script records:

* change in true-vs-best-wrong margin;
* absolute margin change;
* accuracy change;
* wrong -> correct flip rate;
* correct -> wrong flip rate.

These are aggregated into a 4x4 matrix:

```text
rows    = updated factor g
columns = evaluated factor h
```

The diagonal asks whether updating `g` helps future `g` examples.

The off-diagonal entries ask how much the same local update perturbs other factors.

A desirable structural-credit pattern is therefore:

```text
positive diagonal margin change
small off-diagonal absolute margin change
low off-diagonal harmful-flip rate
state_locality_pass_fraction = 1.0
```

The script also reports a diagnostic ratio

```text
abs(mean diagonal margin gain) / mean absolute off-diagonal margin change
```

Higher is more localized, but this ratio is descriptive rather than a predeclared pass/fail threshold.

## Command

```powershell
python scripts/abmg_sequential_local_update_audit.py --fold 0 --features "outputs\patch_aggregation_confirm\fold_0\patch_aggregation_features.pt" --representations "oracle/raw_patch,sensor/raw_patch/k8/score_softmax" --initial-shots 1 --query-budget 32 --checkpoints "4,8,16,32" --repeats 10 --sentinel-fraction 0.25 --probe-per-label 16 --bayes-kappa0 1.0 --seed 0 --out-dir outputs/sequential_local_update_interference_audit
```

Output:

```text
outputs/sequential_local_update_interference_audit/fold_0/sequential_local_update_interference_audit.json
```

## Primary scientific reading

The audit should answer four distinct questions:

1. **State locality:** does feedback for `g` literally modify only `M_g`?
2. **Sequential learning:** does sentinel routing improve from feedback checkpoint 0 -> 4 -> 8 -> 16 -> 32?
3. **Behavioral interference:** when `M_g` changes, how much do predictions for `h != g` move or flip incorrectly?
4. **Unequal-count Bayesian effect:** once `n_g` differs across factors, does posterior-predictive variance help or hurt relative to the deterministic shrunk-memory score?

A positive result would support a concrete structural-credit primitive:

```text
observation -> factor prediction -> queried verification -> local sufficient-statistic update -> better future routing
```

It would still not establish automatic factor discovery, a deployable query policy, online probability calibration, semantic/causal factorization, final C1/C2, or end-to-end anomaly-detection improvement.
