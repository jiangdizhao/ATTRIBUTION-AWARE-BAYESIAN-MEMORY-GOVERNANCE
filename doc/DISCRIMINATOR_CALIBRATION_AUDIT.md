# Discriminator / calibration audit

## Purpose

This source-side diagnostic follows the sparse memory/addressability audit.  The
previous result showed that a shared-diagonal Normal-Normal memory substantially
outperformed a cosine centroid, but its raw posterior responsibilities were
severely overconfident.  This audit isolates *why* classification improved and
whether the score family is calibratable without touching held-out product
categories.

No DINO inference or Real-IAD image loading is performed.  The audit consumes an
existing `abmg.factorizability.features.v1` artifact.

## Fixed data protocol

* Core-4 defect labels: `AK`, `HS`, `QS`, `ZW`.
* Outer ABMG fold targets remain untouched.
* Source-category CV groups are inherited from the feature artifact.
* For each source-CV fold, the six test product categories are untouched until
  evaluation.
* Few-shot supports are selected from source-CV training categories only, using
  the same category-diverse selector as the previous prototype/memory audits.
* Default feedback budgets remain 1/2/4/8 verified examples per Core-4 factor.
* The same support indices are used by every discriminator in a run.
* Shared diagonal background statistics use training-product evidence only and
  do not use labels.
* `kappa0=1.0` is inherited from the previous audit and is not tuned here.

## Matched discriminator decomposition

### A. Cosine centroid

One normalized mean per factor and cosine routing.  This is the previous
single-prototype baseline.

### B. Deterministic shared-diagonal Gaussian (MLE mean)

For factor `g`, the class mean is the sample mean of the verified normalized
support vectors.  A label-free shared diagonal covariance is estimated from the
source-CV training-product evidence pool.  The class score is the shared-
Mahalanobis Gaussian log score.

`B - A` isolates the effect of coordinate-wise variance normalization / shared
Mahalanobis geometry.

### C. Deterministic shared-diagonal Gaussian with shrunk mean

The same deterministic shared covariance is used, but the class mean is

`mu_g* = (kappa0 * mu0 + sum_g) / (kappa0 + n_g)`.

`C - B` isolates prior-mean shrinkage.

### D. Normal-Normal posterior predictive

The current Bayesian score retains the shrunk mean and uses

`Sigma_pred,g = Sigma * (1 + 1/(kappa0+n_g))`.

`D - C` isolates posterior-predictive variance.

Because the audit gives every factor exactly the same number of supports,
`n_g` is identical across factors.  Therefore the predictive-variance multiplier
is common to every class.  C and D should have identical class ranking and
argmax.  The audit reports their prediction agreement explicitly.  If they do
not agree, treat that as an implementation problem before interpreting results.

## Calibration diagnostic

Raw softmax probabilities are evaluated first.  Then one positive scalar
softmax temperature is fitted by NLL for every fold/shot/repeat/model.

The temperature fit uses Core-4 examples from **source-CV training product
categories only**, excluding the sparse support examples.  Held-out source-CV
test products are never used to fit temperature.

This is intentionally an **offline calibratability diagnostic**.  It uses many
training labels and therefore is *not* evidence that the deployed sparse-
feedback agent can obtain calibrated responsibilities without additional
machinery.

The held-out-product metrics are:

* NLL,
* multiclass Brier score,
* top-label ECE (10 bins by default),
* mean confidence,
* mean true-factor responsibility,
* predictive entropy,
* entropy on correct predictions,
* entropy on incorrect predictions,
* reliability-bin confidence and accuracy.

Temperature scaling does not change the argmax.  Therefore any F1 difference is
caused by discriminator geometry, not calibration.

## Primary scientific reading

1. If `diag_gaussian_mle > cosine_centroid`, the major gain comes from the
   shared diagonal metric / whitening.
2. If `diag_gaussian_shrunk > diag_gaussian_mle`, shrinkage is additionally
   useful under sparse feedback.
3. Under equal shots, `bayes_predictive` should not improve F1 over
   `diag_gaussian_shrunk`; identical ranking would show that the previous
   classification gain was not caused by predictive uncertainty itself.
4. If temperature scaling sharply reduces NLL/ECE/Brier on held-out products,
   the score family is calibratable, but the fitted temperature remains an
   offline source-side calibration aid, not yet a deployable responsibility
   mechanism.
5. If calibrated entropy is consistently larger on incorrect routes than on
   correct routes, uncertainty may be useful for evidence seeking / write
   gating, subject to a later sparse/online calibration design.

## Command

```powershell
python scripts/abmg_discriminator_calibration_audit.py --fold 0 --features "outputs\patch_aggregation_confirm\fold_0\patch_aggregation_features.pt" --representations "oracle/raw_patch,sensor/raw_patch/k8/score_softmax" --shots "1,2,4,8" --repeats 20 --seed 0 --bayes-kappa0 1.0 --calibration-bins 10 --out-dir outputs/discriminator_calibration_audit
```

Output:

`outputs/discriminator_calibration_audit/fold_0/discriminator_calibration_audit.json`

## Interpretation boundary

This audit can establish which score geometry produces the routing gain and
whether those scores can be calibrated with source-side labelled data.  It does
not establish semantic or causal factorization, deployable online calibration,
full C1/C2, or end-to-end anomaly-detection improvement.
