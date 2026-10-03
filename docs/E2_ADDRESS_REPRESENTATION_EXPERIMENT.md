# E2 Address-Representation Candidate Audit

**Status:** E2-A/E2-B SCREENING COMPLETE / raw DINO retained provisionally  
**Branch:** \`stage2-e2-address-sidecar-benchmark\`  
**Scientific scope:** choose the simplest stable representation for **memory addressing** while keeping anomaly detection in frozen raw-DINO space.

## 1. Question

E2 asks:

> Does an optional downstream address representation make recurring defect evidence easier to route to the correct memory address than raw DINO itself?

E2 does **not** replace the detector and does **not** test online memory updates.

The invariant architecture is:

\`\`\`text
image
  |
  v
frozen DINO patch h
  |-------------------------------> raw-DINO normal-NN anomaly score
  |                                  -> fixed Sensor top-8 + score-softmax
  |
  +--> optional sidecar f(h)=z_addr
         -> SAME selected patches / SAME spatial weights
         -> address vector
         -> SAME diag_shrunk factor memory
         -> Core-4 factor responsibility / routing
\`\`\`

Later ABMG integration may use \`z_addr\` to decide **where** to write while retaining raw DINO evidence \`h\` as the rich observation being written. E2 itself does not perform that write.

## 2. Candidates

| Candidate | Address representation | E2 role |
|---|---|---|
| \`raw\` | identity \`z=h\` | mandatory baseline |
| \`pca\` | 32-D PCA | linear decorrelation/compression sanity check |
| \`ica\` | 32-D ICA | linear independent-component sanity check |
| \`beta_tcvae\` | 32-D posterior mean | selective total-correlation penalty |
| \`factor_vae\` | 32-D posterior mean | discriminator-estimated total-correlation penalty |
| \`corrvae\` | 32-D grouped posterior mean | structured groups: within-group correlation allowed, cross-group coupling penalized |
| \`address_residual\` | 32-D address channel | address code separated from residual visual detail |
| \`sparse_ae\` | 32-D nonnegative sparse code | sparse activation as an address mechanism |

### CorrVAE-inspired candidate

This implementation follows the project requirement:

> **strong structure within a factor, weak coupling between factors**

The 32-D address is split into 8 groups of 4 coordinates.

* No independence penalty is applied **inside** a group.
* Cross-group covariance is penalized.
* Group-L2 sparsity encourages a small number of groups to become active for an observation.

Therefore E2 does **not** claim that scalar latent coordinates are independent or semantic. The candidate tests whether a grouped operational address is more stable than scalar factorization.

## 3. What remains unchanged

The only causal change is the address representation.

* DINOv2-Register ViT-L/14 remains frozen.
* DINO layers 4--18 and mean fusion remain fixed.
* Preprocessing remains 448 resize -> 392 center crop -> 28x28 patch grid.
* Normal anomaly detector remains raw-DINO cosine nearest neighbour.
* Normal support remains 4-shot with the existing augmentation protocol.
* Sensor selection remains top-8.
* Sensor pooling weights remain score-softmax with temperature 20.
* Oracle mask localization is offline diagnostic only.
* Core-4 remains AK / HS / QS / ZW.
* Factor router remains \`diag_shrunk\`, \`kappa0=1\`.
* The same category-diverse defect support indices are reused across candidates for a support seed.
* E1B correction memory is OFF.
* Online updates are OFF.
* NIG/E3 mechanisms are OFF.
* Query/controller mechanisms are OFF.
* Outer target categories remain sealed.

## 4. Data protocol

For outer fold 0:

* 30 total Real-IAD categories.
* 6 outer target categories remain unopened.
* 24 source categories are partitioned into four predeclared six-product source-CV groups.

For each source-CV fold:

### 4.1 Sidecar training

* 18 source product categories = development categories.
* Use **normal TRAIN images only**.
* Split normal training images deterministically by product into train/validation.
* Default screening cap:
  * max 32 train images/product,
  * max 8 validation images/product,
  * 64 DINO patches/image.
* Defect-source labels and masks are never used to fit or select a sidecar.
* Validation is used for early stopping/convergence only.

### 4.2 Factor-memory support

From defect images in the 18 development products:

* Core-4 labels are evaluator-only.
* Select exactly 8 category-diverse verified examples per Core-4 factor in the screening run.
* The exact same support indices are used for every candidate.

### 4.3 Held-out evaluation

The remaining six source products are never used to fit the sidecar.

Evaluate both:

* **Sensor:** frozen raw-DINO top-8 + score-softmax, then transform selected patch features to address space.
* **Oracle:** transform the GT-mask-positive patches to address space and average them.

Rotate through all four source-CV groups.

The outer six target categories are untouched throughout E2.

## 5. How raw and sidecar features are combined

They are **not concatenated**.

For patch \(p\):

\`\`\`text
h_p = frozen DINO feature
s_p = raw-DINO anomaly score
z_p = f(h_p)  # optional address sidecar
\`\`\`

Raw DINO controls spatial selection:

\`\`\`text
top8 = rank(s_p)
q_p  = softmax(20 * s_p), p in top8
\`\`\`

The same weights pool the address code:

\`\`\`text
z_item = sum_p q_p z_p
\`\`\`

For the raw baseline, \`z_p=h_p\`.

This deliberately implements:

> **raw DINO = what was observed; address sidecar = where the evidence belongs**

No \`[h,z]\` concatenation is used in E2 because it would make the causal contribution of the sidecar ambiguous.

## 6. Success standard

### Primary endpoint

Held-out-product **Sensor Core-4 macro-F1** under the unchanged \`diag_shrunk\` router.

A learned sidecar passes the predeclared primary screening gate only if, across all four source-CV folds:

1. mean Sensor macro-F1 improves over raw DINO by at least **+0.02**;
2. Sensor F1 improves in at least **3/4 source-CV folds**;
3. Oracle macro-F1 does not fall by more than **0.02** relative to raw DINO.

This gate is intentionally practical: a sidecar must justify extra architecture, not merely produce a tiny numerical change.

### Secondary diagnostics

Recorded but not independently treated as "wins":

* balanced accuracy;
* top-2 routing recall;
* true-vs-best-wrong routing margin;
* source routing-signature stability;
* Oracle routing retention;
* factor-address spatial consistency with offline masks;
* coordinate participation ratio / group participation ratio;
* support-seed stability.

If raw DINO is competitive, **no sidecar** is the preferred E2 outcome.

## 7. Experiment stages

### E2-A -- linear sanity check

Candidates:

\`\`\`text
raw,pca,ica
\`\`\`

Screening data cap:

\`\`\`text
64 examples per (product, defect-source) cell
\`\`\`

One factor-support seed initially.

### E2-B -- learned sidecars

Candidates:

\`\`\`text
raw,beta_tcvae,factor_vae,corrvae,address_residual,sparse_ae
\`\`\`

Use the same screening data cap and the same factor-support seed.

### E2-C -- finalist confirmation

Do **not** run automatically.

After E2-A/E2-B results are reviewed, retain raw DINO plus at most two justified sidecars. Then run:

* all source defects (\`--max-per-type-per-category 0\`);
* support seeds 0,1,2,3,4;
* 1/2/4/8 factor supports if needed;
* multiple neural training seeds for surviving learned sidecars.

E2-C is intentionally delayed until screening evidence exists.

## 8. Implementation

* \`scripts/abmg_e2_sidecars.py\`
  * all sidecar models and training losses.
* \`scripts/abmg_e2_address_sidecar_benchmark.py\`
  * source-CV data construction, DINO extraction, sidecar fitting, matched Sensor/Oracle address extraction, fixed-router evaluation, metrics, result aggregation.
* \`tests/test_e2_sidecars.py\`
  * focused model/loss/invariant tests.
* \`tests/test_e2_address_sidecar_benchmark.py\`
  * protocol/helper tests.

## 9. Expected outputs

\`\`\`text
outputs/e2_address_sidecar/fold_0/
  resolved_config.json
  e2_summary.json
  cv_0/e2_cv_summary.json
  cv_0/e2_eval_manifest.json
  ...
  cv_3/e2_cv_summary.json
  cv_3/e2_eval_manifest.json
\`\`\`

The JSON files are intended to be small enough to upload for review.

## 10. Status log

| Date | Status | Evidence / decision |
|---|---|---|
| 2026-10-03 | E2 protocol implemented | Awaiting local unit tests and E2-A smoke run |
| 2026-10-04 | E2-A linear screening complete | Raw DINO Sensor macro-F1 0.6060. PCA 0.4347, ICA 0.4395. Neither passed the primary gate. |
| 2026-10-04 | E2-B learned-sidecar screening complete | Raw DINO Sensor macro-F1 0.6062. Best learned candidate CorrVAE-inspired 0.4791; no learned candidate improved raw in any source-CV fold. |
| 2026-10-04 | E2 screening decision | No sidecar qualifies for finalist confirmation under the predeclared gate. Raw DINO is the provisional address representation. E2-C is not triggered for the tested 32-D sidecars. |

## 11. Result-entry template

After results are reviewed, append a new dated entry here containing:

* exact Git commit;
* exact resolved config;
* E2-A table;
* E2-B table;
* sidecars passing/failing the primary gate;
* failure diagnostics where relevant;
* decision on which representation is frozen for E3/E4;
* whether E2-C is necessary.


## 12. E2-A / E2-B screening results (2026-10-04)

Uploaded result files were reviewed as two result families:

* `*_linear_*`: raw DINO, PCA, ICA.
* `*_learned_*`: raw DINO, Beta-TCVAE, FactorVAE, CorrVAE-inspired, address-plus-residual AE, sparse AE.
* `*_cv0.json` ... `*_cv3.json`: the four source-CV held-out-product folds.
* `e2_summary_*.json`: the corresponding four-fold aggregate summaries.

The branch HEAD before this result-record commit was
`c3544b9f84b20436b247fd7f574280e736573aa1`. The uploaded JSON files do not
embed the Git commit SHA, so the exact local run commit is not independently
recoverable from the outputs themselves.

### 12.1 Aggregate routing result

| Candidate | Sensor macro-F1 | Oracle macro-F1 | Sensor source-signature gap | Spatial true-factor patch AUROC | Primary gate |
|---|---:|---:|---:|---:|---|
| Raw DINO | 0.6062 | 0.7598 | +0.1913 | 0.7347 | Baseline |
| CorrVAE-inspired | 0.4791 | 0.5988 | -0.0635 | 0.9302 | Fail |
| Sparse AE | 0.4776 | 0.5962 | -0.0373 | 0.9187 | Fail |
| FactorVAE | 0.4775 | 0.5975 | -0.0145 | 0.9191 | Fail |
| ICA | 0.4395 | 0.5237 | -0.0224 | 0.9097 | Fail |
| Address+residual AE | 0.4361 | 0.5548 | -0.0880 | 0.9250 | Fail |
| PCA | 0.4347 | 0.5188 | -0.0267 | 0.9295 | Fail |
| Beta-TCVAE | 0.4098 | 0.5452 | -0.1639 | 0.9055 | Fail |

The raw baseline repeated almost identically in the independent linear and
learned screening runs (mean Sensor macro-F1 0.6060 vs 0.6062; maximum
per-fold difference < 0.001), which is a useful internal reproducibility check.

### 12.2 Main scientific result

No tested 32-D sidecar improved raw DINO in even one of the four source-CV folds.
All sidecars also suffered substantial Oracle-routing degradation. Therefore the
Sensor deficit cannot be explained only by noisy top-8 spatial selection: even
when GT-mask-positive patches are supplied, the tested sidecar address spaces
discard or distort cross-product Core-4 routing information.

At the same time, most sidecars produced much higher true-factor patch AUROC
than raw DINO. This does **not** mean that the deployed Sensor became better:
Sensor top-8 selection is fixed in raw-DINO space. It means that, inside a known
factor column, sidecar scores often separate mask-positive from mask-negative
patches well, while the different factor identities remain poorly separated.
The negative mean source-signature gaps support this interpretation.

### 12.3 Candidate-specific diagnostics

**PCA / ICA.** 32-D PCA retained only about 63% of normal-patch variance and
routing fell sharply. ICA was also poor and failed its convergence criterion in
three of the four folds. Linear re-coordinate/compression therefore did not
produce a useful compact address under this setting.

**Beta-TCVAE.** This was the weakest learned candidate. Its strongly negative
source-signature gap and large Sensor/Oracle routing losses show that selective
TC regularization did not recover the cross-product routing structure lost by
the earlier beta-VAE-style bottleneck.

**FactorVAE.** Better than Beta-TCVAE, but still far below raw DINO and below the
primary gate. Independence pressure did not create a superior address space.

**CorrVAE-inspired.** It was the highest-mean learned candidate, but only
marginally ahead of FactorVAE/Sparse AE and far below raw DINO. More importantly,
its group participation ratio was about 7.1 of 8 groups in every fold. Thus the
desired "few active groups" address behavior did not emerge. The objective
penalized cross-group covariance, but the current result files do not report the
final measured cross-group covariance, so weak inter-group coupling itself
cannot yet be claimed empirically.

**Address-plus-residual AE.** The separate address channel did not preserve
enough Core-4 identity for routing, despite reasonable reconstruction training.

**Sparse AE.** The measured fraction of address coordinates with
`|z| < 1e-3` was 0.0 in all four folds, so this configuration did not actually
learn a sparse code. Its failure should be interpreted as a failure of the
tested sparse-AE configuration, not as evidence that sparse addressing is
impossible in general.

### 12.4 Decision

For the current ABMG pipeline:

```text
raw DINO remains the best available address representation.
```

No tested sidecar qualifies for the planned E2-C finalist confirmation.
Therefore E3/E4 should not depend on any of these 32-D sidecars unless a new,
explicitly motivated representation diagnostic is opened.

A remaining interpretation caveat is dimensionality: raw DINO is 1024-D while
all sidecars were forced to a 32-D address. The present result supports the
narrow claim "none of the tested 32-D sidecars improves raw DINO"; it does **not**
prove that factorized/structured sidecars can never work. If that causal question
becomes important, the clean next diagnostic is a small dimensionality control
(e.g. PCA 64/128/256/512) rather than a full E2-C rerun.
