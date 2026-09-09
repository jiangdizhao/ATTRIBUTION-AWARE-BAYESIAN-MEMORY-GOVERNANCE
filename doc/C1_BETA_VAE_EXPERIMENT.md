# C1 beta-VAE / factor-addressability experiment

This document describes how to run `scripts/abmg_stage1_c1_beta_vae.py` and how to decide whether each experimental stage has produced usable evidence.

The experiment implements **Stage 1 (representation feasibility)** and **Stage 2 (C1 factor addressability)** only. It deliberately does **not** implement Bayesian NIG memory updates, source-enriched correction, or query-policy learning. Those belong to later claims and should not be added unless C1 survives.

## 0. Protocol that must not change

The script enforces the frozen Real-IAD five-fold protocol in `configs/realiad_folds_v0.json`:

- 30 categories total;
- five outer folds;
- six held-out target categories per fold;
- the other 24 categories are source categories;
- every category is a target exactly once across the five folds;
- VAE/beta-VAE training uses only **normal training images from the 24 source categories**;
- target categories never train the factorizer and never choose its hyperparameters;
- the four target normal supports are permitted at target evaluation, as in the few-shot protocol;
- hidden target defect labels and masks are evaluator-only information.

The compact Stage-0B cache is an image-level source-normal cache. It contains 64 deterministic DINO patches per training image. The number 64 is a storage/training sample choice only; it is not the VAE latent dimension and it is not the number of factors. During target evaluation the script runs DINO on demand and uses the complete 28 x 28 = 784 patch grid.

The principal representation conditions are:

- `R0`: original frozen DINO representation;
- `R1`: feature-space VAE with beta = 1;
- `Rbeta`: same architecture and latent dimension with beta = 4.

The default latent dimension is 32. The principal grouped C1 test partitions it into eight consecutive groups of four coordinates. This grouping is an implementation hypothesis, not a semantic claim. The script therefore also supports a 32-coordinate baseline and a random regrouping negative control.

## 1. Checkout and static check

A new branch is used so Stage 0 remains frozen:

```powershell
git fetch origin
git checkout stage1-c1-beta-vae
git pull
python -m py_compile scripts/abmg_stage1_c1_beta_vae.py
python scripts/abmg_stage1_c1_beta_vae.py --help
```

Expected evidence of success:

- `py_compile` exits silently;
- `--help` lists `scale-audit`, `train`, `retention`, `addressability`, `compare`, and `aggregate`;
- no Stage-0 files need to be changed locally.

Define local paths for convenience:

```powershell
$ROOT="E:\PATH\TO\Real-IAD"
$JSON="E:\PATH\TO\realiad_jsons_sv"
$CACHE="cache\realiad_dino_source_normal_compact"
```

Use your actual paths.

---

## 2. Step A — source-only loss/gradient scale audit

### Why this step exists

We do **not** assume that centering/scalar normalization is beneficial. Before training the real VAE, the script compares raw DINO descriptors with the proposed source-only scalar normalization using identical initial VAE weights and identical source-normal image samples.

For one patch, the training objective is

```text
L = L_rec + beta * L_KL
```

where reconstruction is the summed feature-space squared error and KL is summed over the 32 latent coordinates.

The audit reports both:

```text
(beta * KL) / reconstruction
```

and the corresponding **encoder gradient-norm ratio**:

```text
||grad(beta * KL)|| / ||grad(reconstruction)||
```

The default numerical compatibility band is `[0.1, 10]`. This means neither term is more than roughly one order of magnitude larger than the other under this preflight probe. This band is an engineering diagnostic, not a C1 scientific result.

### Run fold 0 first

```powershell
python scripts/abmg_stage1_c1_beta_vae.py scale-audit --fold 0 --cache-dir "$CACHE" --device cuda --out-dir outputs/stage1_scale_audit
```

Outputs:

```text
outputs/stage1_scale_audit/
  fold_0_scale_audit.json
  fold_0_scale_audit.csv
```

### Evidence that Step A succeeded technically

Open `fold_0_scale_audit.json`. Check:

1. `source_categories` contains 24 categories.
2. `target_categories_hidden_from_audit` contains the six fold-0 target categories.
3. `cache_fingerprint` equals the Stage-0B fingerprint.
4. `image_level_split` reports nonzero train and validation image counts.
5. Both `raw` and `scalar` conditions contain finite reconstruction, KL, and gradient values.

### Decision about normalization

For the current fold:

- if `raw.preflight_compatible == true`, use `--normalization raw`;
- if raw fails but `scalar.preflight_compatible == true`, use `--normalization scalar`;
- if both fail, **stop before primary VAE training** and inspect the audit rather than silently rescaling the loss;
- if both pass, prefer `raw` because it introduces fewer transformations.

Do not choose normalization using target AUROC or target defect labels.

Before training all five folds, repeat this audit for folds 0-4. The decision for a fold must use only that fold's source-side audit.

---

## 3. Step B — engineering smoke training

This is only to verify that the VAE training machinery runs. It is not a scientific result.

Assume the fold-0 audit selected `raw` below; replace it with `scalar` if the audit requires that.

```powershell
python scripts/abmg_stage1_c1_beta_vae.py train --fold 0 --cache-dir "$CACHE" --beta 1 --normalization raw --epochs 2 --image-batch-size 32 --seed 0 --device cuda --out-dir outputs/stage1_factorizers_smoke

python scripts/abmg_stage1_c1_beta_vae.py train --fold 0 --cache-dir "$CACHE" --beta 4 --normalization raw --epochs 2 --image-batch-size 32 --seed 0 --device cuda --out-dir outputs/stage1_factorizers_smoke
```

Expected outputs under each condition:

```text
resolved_config.json
train.jsonl
checkpoint_last.pt
checkpoint_best.pt
summary.json
```

Evidence that the smoke test succeeded:

- training and validation losses are finite;
- no NaN/Inf appears;
- `checkpoint_best.pt` is written;
- `summary.json` reports `target_categories_untouched` with six categories;
- train/validation counts are image-level, not patch-level;
- `validation.per_coordinate_kl` and `validation.per_coordinate_var_mu` each contain 32 values.

Do not interpret a two-epoch smoke model scientifically. With a five-epoch beta warm-up, it has not even reached the final beta.

---

## 4. Step C — primary R1 and Rbeta training

### Training unit

The compact cache stores one image as `[64, 1024]` DINO patch descriptors. An image is the minimum split/loading unit:

```text
image i -> all 64 patches -> train OR validation
```

Patches from the same image can never be split across train and validation. Inside an optimizer batch, several images are stacked as `[B_image, 64, 1024]` and only then flattened to `[B_image * 64, 1024]` because the MLP VAE models each patch descriptor independently.

### Beta warm-up

For target beta `beta*`, effective beta increases linearly from zero and reaches the target at epoch 5. For beta=4 the sequence begins approximately:

```text
epoch 0: 0.0
epoch 1: 0.8
epoch 2: 1.6
epoch 3: 2.4
epoch 4: 3.2
epoch 5+: 4.0
```

### Train fold 0

Use exactly the same normalization choice, seed, architecture, optimizer settings, and image split for R1 and Rbeta. Only beta should differ.

```powershell
python scripts/abmg_stage1_c1_beta_vae.py train --fold 0 --cache-dir "$CACHE" --beta 1 --normalization raw --epochs 30 --beta-warmup-epochs 5 --image-batch-size 64 --seed 0 --device cuda --out-dir outputs/stage1_factorizers

python scripts/abmg_stage1_c1_beta_vae.py train --fold 0 --cache-dir "$CACHE" --beta 4 --normalization raw --epochs 30 --beta-warmup-epochs 5 --image-batch-size 64 --seed 0 --device cuda --out-dir outputs/stage1_factorizers
```

### Evidence that factorizer training is healthy

Inspect each `summary.json`.

The essential checks are:

- `validation.recon_cosine_original_space` is finite and the reconstruction is not degenerate;
- `validation.kl_loss` is finite;
- `validation.per_coordinate_kl` is not effectively zero for every coordinate;
- `validation.per_coordinate_var_mu` is not effectively zero for every coordinate;
- `canonical_8_group_kl` and `canonical_8_group_var_mu` are not concentrated almost entirely into one or two groups unless the data genuinely produce that result;
- `validation.mean_abs_offdiag_mu_corr` is recorded as a descriptive latent-factorization diagnostic.

There is intentionally no magic threshold saying `KL > x = active`. Posterior collapse should be diagnosed jointly from near-zero per-coordinate KL and near-zero variance of posterior means across the validation set.

A lower latent correlation for beta=4 is interesting but **does not prove C1**. Operational addressability is tested later.

### Repeat for all folds

After fold 0 is healthy, train beta=1 and beta=4 for folds 1, 2, 3, and 4 using each fold's predeclared normalization result from Step A.

At the end there should be ten primary checkpoints:

```text
5 folds x 2 beta conditions = 10 checkpoints
```

Do not run beta=2 or beta=8 on held-out target categories to choose beta. They are optional source-side sensitivity conditions only.

---

## 5. Step D — held-out anomaly-information retention

This step asks whether the factorized representation destroyed useful anomaly information.

The evaluator runs frozen DINO on the six held-out target categories of the current fold. DINO is executed once per target batch; the same raw patch features are supplied to all compared conditions.

`R0` uses original DINO patch features. `R1` and `Rbeta` use deterministic posterior means followed by the VAE decoder to reconstruct DINO-space features. The same VisionAD-style support bank, patch nearest-neighbour scorer, and top-1% image aggregation are then used for all three conditions. This makes the representation the main changed component.

For fold 0, assuming standard output paths:

```powershell
python scripts/abmg_stage1_c1_beta_vae.py retention --fold 0 --root "$ROOT" --json-dir "$JSON" --checkpoint "R1=outputs/stage1_factorizers/fold_0/beta_1/seed_0/checkpoint_best.pt" --checkpoint "Rbeta=outputs/stage1_factorizers/fold_0/beta_4/seed_0/checkpoint_best.pt" --support-seed 0 --target-batch-size 2 --device cuda --out-dir outputs/stage1_retention
```

Outputs:

```text
outputs/stage1_retention/fold_0/
  per_item_scores.csv
  retention_summary.json
```

### Evidence that retention evaluation succeeded

Check:

- exactly the six fold target categories are present;
- `R0`, `R1`, and `Rbeta` have AUROC/AP for every target category;
- R0 per-category values should be numerically consistent with the corresponding Stage-0A run because the sensor and support protocol are the same;
- `delta_macro_auroc_vs_R0` is reported for R1 and Rbeta;
- AUROC is the primary retention statistic; AP is secondary.

Do **not** infer an acceptable beta-VAE loss from R0 support variability. R0 variation is only a noise/reference scale. This implementation deliberately reports the actual paired utility loss and leaves the acceptable scientific/practical tolerance to be predeclared rather than invented after seeing target results.

Repeat the retention command for folds 1-4 with their corresponding checkpoints.

---

## 6. Step E — how factor surprise and responsibility are constructed

C1 needs an operational answer to: **which latent coordinates/groups carried the abnormal evidence?**

For a target category, the four normal supports are encoded by the frozen factorizer. The support augmentation used in Stage 0 provides multiple support feature grids. For each latent coordinate `j`, the script estimates a fixed Gaussian normal reference:

```text
z_j ~ Normal(mu_j, var_j)
```

This is deliberately a deterministic Gaussian reference for C1. The later Normal-Inverse-Gamma Bayesian memory is not introduced here because C1 should isolate representation/addressability before C4 Bayesian governance.

For a query patch, coordinate surprise is Gaussian negative log likelihood:

```text
s_j = 0.5 * [ log(2*pi*var_j) + (z_j-mu_j)^2 / var_j ]
```

For group `g`, factor surprise is the sum over coordinates in that group:

```text
a_g,p = sum_{j in g} s_j,p
```

For each factor, the image-level evidence is the mean of the highest 1% of patch surprises. Support-derived factor means/stds standardize these image-level values, then responsibility is:

```text
r_g = softmax(lambda * standardized_factor_surprise)_g
```

with default `lambda = 1`.

`r` is an address/credit pattern. It does not mean that factor 3 literally equals a scratch or any physical cause.

---

## 7. Step F — principal C1 addressability experiment

Run the canonical eight groups of four coordinates first.

```powershell
python scripts/abmg_stage1_c1_beta_vae.py addressability --fold 0 --root "$ROOT" --json-dir "$JSON" --checkpoint "R1=outputs/stage1_factorizers/fold_0/beta_1/seed_0/checkpoint_best.pt" --checkpoint "Rbeta=outputs/stage1_factorizers/fold_0/beta_4/seed_0/checkpoint_best.pt" --grouping canonical --grouping-seed 0 --factor-top-fraction 0.01 --responsibility-lambda 1.0 --centroid-train-fraction 0.5 --centroid-seed 0 --target-batch-size 2 --device cuda --out-dir outputs/stage2_c1
```

Outputs:

```text
outputs/stage2_c1/fold_0/canonical/
  R1_per_item_responsibility.csv
  Rbeta_per_item_responsibility.csv
  addressability_summary.json
```

The evaluator computes three C1 measurements.

### 7.1 Defect-source separability

Within each target category, defect-source labels are used only offline after responsibilities are computed. Each defect source is deterministically split so approximately 50% of its examples construct responsibility centroids and the remaining examples test nearest-centroid classification.

Success evidence:

- `source_separability.accuracy` and `macro_f1` are finite for categories with enough repeated defect sources;
- the main comparison is `Rbeta` versus `R1`, not the absolute value alone;
- categories with fewer than two usable defect sources can legitimately be unscored (`NaN`) and are excluded from finite macro means.

This centroid split is an **offline representation diagnostic**, not an online C3 source-memory experiment.

### 7.2 Responsibility stability

Responsibilities are L2-normalized for cosine comparison. The script computes all same-source pairs and all different-source pairs efficiently within each category:

```text
S_within  = mean cosine(r_i, r_j | same defect source)
S_between = mean cosine(r_i, r_j | different defect source)
DeltaS    = S_within - S_between
```

Desired evidence:

```text
DeltaS > 0
```

and, for the C1 comparison,

```text
DeltaS(Rbeta) > DeltaS(R1)
```

across held-out categories rather than only one favourable category.

### 7.3 Spatial consistency

For each defective image, the script selects the group with maximum image responsibility and uses that group's patch-surprise map. The ground-truth anomaly mask is resized/cropped with the same Stage-0 geometry and is used **offline only** to compute pixel AUROC.

Desired evidence:

- `n_masks_scored > 0` for applicable categories;
- pixel AUROC above random (`0.5`) is evidence that the credited factor is spatially expressed at the annotated defect;
- the stronger C1 result is better spatial consistency for Rbeta than R1 across categories.

Repeat the canonical addressability experiment for folds 1-4.

---

## 8. Step G — grouping controls

The eight groups of four coordinates are not theoretically guaranteed to be natural factors. C1 therefore needs explicit controls.

### Coordinate-level baseline

Treat every latent coordinate as one address:

```powershell
python scripts/abmg_stage1_c1_beta_vae.py addressability --fold 0 --root "$ROOT" --json-dir "$JSON" --checkpoint "R1=outputs/stage1_factorizers/fold_0/beta_1/seed_0/checkpoint_best.pt" --checkpoint "Rbeta=outputs/stage1_factorizers/fold_0/beta_4/seed_0/checkpoint_best.pt" --grouping coordinate --target-batch-size 2 --device cuda --out-dir outputs/stage2_c1
```

This produces 32 responsibility entries rather than eight.

Interpretation:

- if coordinate-level addressability is consistently stronger, individual coordinates may be a better update granularity than arbitrary 4-D groups;
- this does not invalidate beta-VAE automatically, but it weakens the case for the 8 x 4 grouping.

### Random-regrouping negative control

Randomly permute the 32 coordinates and then form eight groups of four:

```powershell
python scripts/abmg_stage1_c1_beta_vae.py addressability --fold 0 --root "$ROOT" --json-dir "$JSON" --checkpoint "R1=outputs/stage1_factorizers/fold_0/beta_1/seed_0/checkpoint_best.pt" --checkpoint "Rbeta=outputs/stage1_factorizers/fold_0/beta_4/seed_0/checkpoint_best.pt" --grouping random --grouping-seed 0 --target-batch-size 2 --device cuda --out-dir outputs/stage2_c1
```

Interpretation:

- if canonical grouping clearly outperforms random regrouping, the original grouping carries some stable structure;
- if canonical and random are essentially indistinguishable, do not claim that consecutive 4-D groups are special;
- if both are useful but coordinate-level is best, later memory design should consider coordinate-level or learned grouping instead.

Run the same coordinate and random controls for all five folds before the final aggregate report.

---

## 9. Step H — optional one-fold comparison

After fold 0 retention and canonical C1 results are available:

```powershell
python scripts/abmg_stage1_c1_beta_vae.py compare --retention-summary outputs/stage1_retention/fold_0/retention_summary.json --addressability-summary outputs/stage2_c1/fold_0/canonical/addressability_summary.json --out outputs/stage2_c1/fold_0/c1_comparison.json
```

This is useful for debugging the interpretation before spending time on all five folds.

It reports R0/R1/Rbeta utility and Rbeta-minus-R1 addressability deltas. It deliberately does not issue an automatic `C1=PASS` verdict.

---

## 10. Step I — five-fold aggregation

After all five retention runs and all five canonical/coordinate/random addressability runs are complete:

```powershell
python scripts/abmg_stage1_c1_beta_vae.py aggregate --retention-root outputs/stage1_retention --addressability-root outputs/stage2_c1 --bootstrap 5000 --bootstrap-seed 12345 --out-dir outputs/c1_aggregate
```

Outputs:

```text
outputs/c1_aggregate/
  retention_30_category.csv
  c1_cross_fold_summary.json
```

### Evidence that the cross-fold experiment is complete

`c1_cross_fold_summary.json` must report:

```text
target_rotation_verified = true
n_unique_target_categories = 30
```

This is the critical proof that every Real-IAD category contributed exactly once as a held-out target.

The aggregate report contains:

- R0, R1, Rbeta macro AUROC and AP over 30 held-out category results;
- paired bootstrap confidence intervals for R1-vs-R0 and Rbeta-vs-R0 AUROC changes;
- paired Rbeta-vs-R1 AUROC/AP differences;
- R1 and Rbeta source-separability accuracy/F1;
- R1 and Rbeta responsibility-stability delta;
- R1 and Rbeta spatial pixel AUROC;
- category-level paired bootstrap intervals for Rbeta-minus-R1 addressability changes;
- canonical-vs-random and canonical-vs-coordinate grouping controls when those runs are present.

The bootstrap unit is the **held-out target category**, not individual images. Images inside one industrial category are not treated as independent scientific replicates.

---

## 11. How to decide whether C1 is supported

Do not use latent independence by itself as the conclusion. Lower off-diagonal correlation or a visually neat latent space is only a mechanism diagnostic.

C1 needs two pieces of evidence together:

### A. Anomaly information is retained

Look at the 30-category held-out differences:

```text
AUROC(R1)    - AUROC(R0)
AUROC(Rbeta) - AUROC(R0)
```

and their paired bootstrap intervals. AP is secondary.

The script does not invent a post-hoc non-inferiority tolerance. Decide what magnitude of anomaly-utility loss is practically/scientifically acceptable before using the final target result as a claim boundary.

### B. Operational addressability improves

The principal comparison is beta=4 against beta=1 on independent evidence:

```text
macro defect-source F1 / accuracy
within-source minus between-source responsibility cosine
spatial pixel AUROC of the max-responsibility factor
```

The strongest support is a coherent improvement across several of these measures rather than a single isolated metric.

### C. The grouping negative control must be interpreted

If random regrouping is as good as canonical grouping, the result may support coordinate-level/address-pattern structure but **not** the claim that the chosen 8 x 4 grouping is meaningful.

If the coordinate baseline is better than 8 x 4 grouping, the next memory stage should reconsider its update granularity instead of forcing the grouped design.

### Stop/go rule

Proceed to factor-addressed memory (C2) only if beta regularisation shows measurable operational addressability beyond beta=1 **and** anomaly utility remains acceptable.

If beta=4 collapses the latent representation, loses too much anomaly ranking, or fails to improve addressability over beta=1, C1 is not supported in its present form. Do not add Bayesian memory/query complexity to hide that failure.

---

## 12. What to send for review after each step

You do not need to upload large checkpoints or CSVs first. The compact review artifacts are:

1. after Step A: `fold_0_scale_audit.json`;
2. after primary fold-0 training: the two `summary.json` files for beta=1 and beta=4;
3. after fold-0 retention: `retention_summary.json`;
4. after fold-0 canonical C1: `addressability_summary.json`;
5. after all folds: `c1_cross_fold_summary.json`.

Review these gates sequentially. If an earlier gate fails, do not spend time on the later full experiment until the failure is understood.
