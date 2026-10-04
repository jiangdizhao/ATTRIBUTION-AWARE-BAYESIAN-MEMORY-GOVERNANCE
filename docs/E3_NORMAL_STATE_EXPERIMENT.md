# E3 Sparse Normal-State Model Audit

**Status:** IMPLEMENTED / awaiting local execution  
**Branch:** \`stage2-e3-normal-state-audit\`  
**Scientific scope:** choose a defensible sparse online normal-state model while keeping the frozen detector, representation, query schedule, and evidence stream fixed.

## 1. Question

E3 asks:

> How should sparse verified-normal evidence represent normality and uncertainty?

The causal comparison is:

\`\`\`text
same frozen DINO evidence
same detector / top-8 selection
same support images
same verified-normal update events
same held-out normal/defect probes
            |
            v
   change ONLY normal-state model
\`\`\`

E3 does **not** introduce the query controller, source-enriched defect feedback, hard-normal correction bank, or factor-addressed write policy. Those remain later experiments.

## 2. Representation frozen from E2

E2 did not justify a sidecar. Therefore E3 uses raw frozen DINO evidence.

For every image:

\`\`\`text
frozen DINO patch features
  -> existing 4-shot normal-NN detector
  -> fixed top-8 suspicious patches
  -> score-softmax pooling (temperature 20)
  -> one 1024-D image evidence vector
  -> L2 normalization
  -> E3 normal-state model
\`\`\`

The 4-shot detector support is held fixed. E3's 1/2/4-shot stress refers to the number of **normal-state initialization items**, not the detector support.

## 3. Primary candidate models

### D0 -- deterministic diagonal Gaussian

Running point estimates:

\[
x_j \sim N(\mu_j,\sigma_j^2)
\]

with plug-in mean/variance. It carries no posterior uncertainty. Source-derived variance is used only as a numerical/few-shot fallback.

Scientific role: mandatory simple comparator.

### B1 -- diagonal NIG / Student-t predictive

For every coordinate:

\[
x_j\mid\mu_j,\sigma_j^2 \sim N(\mu_j,\sigma_j^2)
\]

\[
(\mu_j,\sigma_j^2)\sim NIG(\mu_{0,j},\kappa_{0,j},\alpha_{0,j},\beta_{0,j}).
\]

Verified-normal updates use the exact conjugate fractional-weight update. The posterior predictive is Student-t.

Scientific role: canonical proposal-aligned Bayesian C4 model.

### B2 -- hierarchical empirical-Bayes NIG

Same online NIG posterior update as B1, but the prior is fitted from the 18 source-development product categories in each source-CV fold.

The hierarchy estimates:

* global normal mean;
* pooled within-product variance;
* between-product mean variance;
* coordinate-wise prior strength \(\kappa_0\).

This tests whether source-derived transfer stabilizes ultra-few-shot product normality.

Scientific role: structured prior transfer.

### B3 -- robust NIG weighted update

Uses the same weak prior as B1, but an incoming verified-normal update receives an additional robust influence weight determined from its current predictive residual.

The robust factor is bounded in \((0,1]\). Therefore an extreme mislabeled or contaminated observation cannot move the state as strongly as an ordinary NIG update.

This is a **robust generalized-Bayes update**, not a second exact conjugate probabilistic model.

Scientific role: controlled contamination stress only.

## 4. Candidates intentionally deferred

### NIW

NIW is not a primary E3 candidate after E2 because the retained representation is raw 1024-D DINO. Estimating a full \(1024\times1024\) covariance from 1--4 items is not a meaningful few-shot comparison.

A bounded low-dimensional NIW diagnostic may be opened later only if a specific correlation failure is demonstrated.

### Discounted / dynamic NIG

Conceptually useful for gradual factory drift, but Real-IAD does not provide a trustworthy physical time/drift sequence. It is deferred rather than tested on an arbitrary synthetic temporal process.

### Mixture / DPMM normality

Deferred until E3/E4 shows persistent multimodal normal modes that a single-state model cannot handle.

### Free-energy / active-inference steady-state model

Not treated as a competing density estimator. E3 estimates the normal-state density. A later controller may use surprise/uncertainty from that state under an active-inference or expected-free-energy policy.

## 5. What remains unchanged

* DINOv2-Register ViT-L/14 frozen.
* DINO layers 4--18, mean fusion.
* 448 resize, 392 center crop, 28x28 patch grid.
* Raw-DINO cosine normal-NN detector.
* Detector support fixed at 4 normal images/category.
* Sensor evidence fixed at top-8 + score-softmax, temperature 20.
* Raw DINO is the representation.
* Outer target categories remain sealed.
* No E1B correction memory.
* No E2 sidecar.
* No adaptive query controller.
* No defect-driven normal-state update.
* No factor-addressed write rule yet.
* Every model sees the same initialization support, normal update stream, checkpoints, contamination locations, and evaluation probes.

## 6. Data protocol

For outer fold 0:

* 24 source categories are available.
* 6 outer target categories remain unopened.
* The 24 sources are partitioned into four six-category source-CV groups.

For one source-CV fold:

### Development products: 18 categories

Used only to estimate source normal prior statistics.

No defect label is used to fit any prior.

### Held-out products: 6 categories

For each held-out product:

1. The frozen detector uses its ordinary 4-shot normal support.
2. Up to **32** additional normal TRAIN images not used by the detector are selected deterministically for source-prior/state-support evidence.
3. Up to **64** normal TEST images are selected deterministically and split into:
   * 32 verified-normal update items;
   * at least 32 fixed normal sentinel items when available.
4. Defect TEST probes are deterministically capped at **16 per (product, defect-source)** for the E3 audit.
5. Defect labels/masks are evaluator-only and never update the normal state.

These bounds are deliberate. E3 is a matched normal-state mechanism audit, not a full detector benchmark. Using every Real-IAD normal image only repeats expensive frozen-backbone extraction without strengthening the causal comparison.

Every source category is held out exactly once across the four source-CV folds.

## 7. Sparse-update stress matrix

### E3-A -- stationary sparse normality

Models:

\`\`\`text
deterministic_diag,nig,hierarchical_nig
\`\`\`

State initialization:

\`\`\`text
1,2,4 normal items
\`\`\`

Support seeds:

\`\`\`text
0,1,2
\`\`\`

Verified-normal update checkpoints:

\`\`\`text
0,4,8,16,32
\`\`\`

No contaminated feedback.

### E3-B -- controlled contamination

Models:

\`\`\`text
deterministic_diag,nig,hierarchical_nig,robust_nig
\`\`\`

Use 2- and 4-shot initialization and final 32-update checkpoint.

Matched contamination rates:

\`\`\`text
0%,10%
\`\`\`

At contaminated update positions, the same held-out defect evidence vector is deliberately presented to every model as if it were verified normal.

This is evaluator-controlled label noise; it is never called genuine human feedback.

## 8. Model interface

Every state exposes the same conceptual operations:

\`\`\`python
update(x, weight)
predictive_log_prob(x)
uncertainty()
state_summary()
state_drift(reference_state)
\`\`\`

All item evidence vectors are L2-normalized before entering the state model.

## 9. Primary outputs

At every checkpoint:

### Downstream anomaly utility

Using negative predictive log probability as surprise:

* image AUROC: normal sentinel vs held-out defects;
* FPR at 95% defect TPR;
* mean normal surprise;
* mean defect surprise;
* defect-minus-normal surprise gap.

### Predictive calibration

Marginal 90% predictive-interval coverage on clean normal sentinel vectors:

\[
\text{calibration error}=|\text{coverage}_{90}-0.90|.
\]

This is coordinate-marginal calibration; it is not a claim of joint 1024-D coverage.

### Sparse-state stability

* uncertainty magnitude;
* mean state drift from post-initialization state;
* support-seed variability;
* checkpoint-to-checkpoint normal-surprise variability.

### Safe write behaviour

Under E3-B contamination:

* anomaly-AUROC degradation relative to the matched clean stream;
* extra state drift caused by contamination;
* robust-NIG mean influence weight on clean vs contaminated updates.

## 10. Decision rules

### Canonical C4 question: D0 vs B1

NIG is justified only if it improves a predeclared sparse/noisy calibration or stability endpoint while remaining operationally competitive.

Operational non-inferiority:

\[
\Delta AUROC_{\text{NIG-D0}} \ge -0.01
\]

at the matched condition.

Evidence for Bayesian value requires at least one of:

* lower 90% coverage calibration error by at least 0.02 in 1/2-shot settings;
* lower support-seed AUROC variability by at least 20%;
* lower contamination-induced state drift / utility loss.

If these do not occur, the deterministic state remains preferred even though NIG is mathematically richer.

### Hierarchical NIG

Retain only if it improves 1/2-shot held-out-product calibration or anomaly utility beyond ordinary NIG without harming 4-shot performance materially.

### Robust NIG

Retain only if it is approximately neutral on the clean stream and materially reduces damage under 10% contaminated updates.

## 11. Implementation

* \`scripts/abmg_e3_normal_state_models.py\`
  * deterministic diagonal Gaussian;
  * diagonal NIG;
  * hierarchical-prior fitting;
  * robust NIG update;
  * predictive intervals, uncertainty, drift.

* \`scripts/abmg_e3_extract_evidence.py\`
  * one-time frozen-DINO E3 evidence extraction;
  * detector support exclusion from state-support pool;
  * source-only cache and manifest.

* \`scripts/abmg_e3_normal_state_audit.py\`
  * four-fold source-CV replay;
  * matched state supports / normal streams;
  * clean E3-A and contamination E3-B conditions;
  * aggregation and decision metrics.

* \`tests/test_e3_normal_state_models.py\`
* \`tests/test_e3_normal_state_audit.py\`

## 12. Expected outputs

Extraction:

\`\`\`text
outputs/e3_evidence/fold_0/
  e3_evidence.pt
  e3_evidence_manifest.json
  resolved_config.json
\`\`\`

E3-A:

\`\`\`text
outputs/e3_stationary/fold_0/
  e3_summary.json
  cv_0/e3_cv_summary.json
  ...
  cv_3/e3_cv_summary.json
  resolved_config.json
\`\`\`

E3-B:

\`\`\`text
outputs/e3_contamination/fold_0/
  e3_summary.json
  cv_0/e3_cv_summary.json
  ...
  cv_3/e3_cv_summary.json
  resolved_config.json
\`\`\`

## 13. Status log

| Date | Status | Evidence / decision |
|---|---|---|
| 2026-10-05 | E3 protocol frozen | Raw DINO retained from E2; four primary normal-state conditions defined |
| 2026-10-05 | Extraction runtime correction | Initial extractor processed all normal images, making even the defect-capped smoke run close to full cost. Extraction is now deterministically bounded to 32 train normals/product, 64 test normals/product, and 16 defects/(product, source), with batched patch-NN evaluation. |
| pending | Unit tests | Not yet run locally |
| pending | Evidence extraction | Not yet run |
| pending | E3-A stationary sparse audit | Not yet run |
| pending | E3-B contamination audit | Not yet run |
| pending | E3 result review | Results will be appended here before E4 |
