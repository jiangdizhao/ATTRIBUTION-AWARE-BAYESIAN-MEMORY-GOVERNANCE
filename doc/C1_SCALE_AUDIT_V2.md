# Revised Stage-1 scale audit (v2)

This note supersedes the original one-shot `scale-audit` decision rule in `doc/C1_BETA_VAE_EXPERIMENT.md`.

The old audit compared reconstruction and beta-weighted KL gradients at random initialization with beta already fixed at 4. That is not representative of the actual beta-VAE training process because the primary experiment uses beta warm-up from 0 to 4. The v2 audit therefore performs a short, paired source-only calibration run and observes the optimization trajectory under the same warm-up schedule that will be used later.

The revised command is implemented in:

```text
scripts/abmg_stage1_c1_beta_vae_v2.py
```

For commands other than `scale-audit`, the v2 wrapper delegates unchanged to `abmg_stage1_c1_beta_vae.py`.

## 1. What the revised audit tests

For each normalization condition (`raw` and `scalar`), the audit:

1. uses only the 24 source categories for the selected fold;
2. keeps the six target categories completely hidden;
3. samples a deterministic, approximately category-balanced image subset;
4. keeps each image and all of its 64 cached patches together;
5. initializes raw and scalar models from identical network weights;
6. uses the same train images, validation images, epoch image order, and latent-noise seeds for both conditions;
7. trains for six epochs by default so the existing warm-up schedule reaches beta=4 exactly at epoch 5;
8. records reconstruction, KL, beta-weighted KL, encoder gradient-component ratios, deterministic reconstruction cosine, per-coordinate KL, posterior-mean variance, and continuous latent participation ratios.

The objective remains:

```text
L = 0.5 * sum_feature_squared_error + beta_effective * KL
```

The audit deliberately has **no automatic 0.1-10 pass/fail gradient band**. Gradient ratios are descriptive diagnostics after the model has started adapting.

## 2. Why six epochs are used

The existing warm-up function is:

```text
beta_effective = beta_target * min(1, epoch / warmup_epochs)
```

With beta target 4 and warm-up 5 epochs:

```text
epoch 0 -> 0.0
epoch 1 -> 0.8
epoch 2 -> 1.6
epoch 3 -> 2.4
epoch 4 -> 3.2
epoch 5 -> 4.0
```

Therefore a six-epoch calibration is the shortest default run that includes one epoch at the final beta=4.

## 3. Run the revised fold-0 audit

First update the branch:

```powershell
git fetch origin
git checkout stage1-c1-beta-vae
git pull
```

Compile and run the unit test:

```powershell
python -m py_compile scripts/abmg_stage1_c1_beta_vae_v2.py
python -m unittest tests.test_stage1_scale_audit_v2 -v
```

Then define the compact cache path if it is not already defined:

```powershell
$CACHE="cache\realiad_dino_source_normal_compact"
```

Run the paired raw/scalar calibration:

```powershell
python scripts/abmg_stage1_c1_beta_vae_v2.py scale-audit --fold 0 --cache-dir "$CACHE" --device cuda --out-dir outputs/stage1_scale_audit_v2
```

Default calibration size:

```text
1024 source-train images
256 source-validation images
64 cached patches per image
32 image batch size
6 epochs
beta target = 4
beta warm-up = 5 epochs
modes = raw,scalar
```

The subset is image-level and balanced across the 24 source categories as closely as integer quotas permit.

## 4. Outputs

The command writes:

```text
outputs/stage1_scale_audit_v2/
  fold_0_scale_audit_v2.json
  fold_0_scale_audit_v2_trajectory.csv
```

The JSON is the main artifact to inspect.

## 5. Evidence that the audit ran correctly

The audit is technically successful when all of the following are true:

- `source_categories` contains exactly 24 categories;
- `target_categories_hidden_from_audit` contains exactly the six fold-0 targets;
- `cache_fingerprint` matches the Stage-0B cache fingerprint;
- the calibration train subset contains 1024 complete images and the validation subset contains 256 complete images;
- `paired_design.same_initial_weights`, `same_train_images`, `same_validation_images`, `same_epoch_image_order`, and `same_stochastic_latent_seeds_per_step` are all `true`;
- both conditions report finite values through the final epoch;
- `reached_target_beta` is `true` for each condition;
- no NaN or Inf appears in reconstruction, KL, reconstruction cosine, gradient norms, or latent diagnostics.

This only proves that the calibration experiment is numerically well formed. It does not prove that beta-VAE is useful.

## 6. How to interpret the trajectory

For each condition, inspect the sequence from epoch 0 to epoch 5.

### Reconstruction

`val_recon_cosine_original_space` should improve substantially from the untrained value. Reconstruction loss should generally decrease rather than diverge.

A condition is suspect if reconstruction remains essentially unlearned or becomes unstable as beta increases.

### KL

`val_kl_loss` should remain finite. A decreasing KL by itself is not a failure; beta regularization is expected to reduce unnecessary latent information.

A potential posterior-collapse pattern is supported only when several signals agree, for example:

```text
KL -> almost zero
sum Var(mu) -> almost zero
per-coordinate KL -> almost zero broadly
per-coordinate Var(mu) -> almost zero broadly
reconstruction no longer carries useful input variation
```

Do not diagnose collapse from one scalar alone.

### Gradient ratio

`encoder_grad_ratio_betaKL_over_rec` is now measured at the current trained model and current `beta_effective`.

It is descriptive only. A ratio above 10 is not automatically a failure, and a ratio below 10 is not automatically a success. The important question is whether the resulting optimization remains stable and retains useful reconstruction/latent variation while beta increases.

### Participation ratios

The audit reports:

```text
kl_participation_ratio
var_mu_participation_ratio
```

For nonnegative coordinate statistics x_j, participation ratio is:

```text
(sum_j x_j)^2 / sum_j x_j^2
```

It can be interpreted as a continuous effective number of coordinates carrying the measured quantity. A value near 1 means strong concentration into very few coordinates; a value distributed toward the latent dimension (32) means broader usage.

This is a latent-usage diagnostic, not a semantic disentanglement score.

## 7. Normalization decision rule

The v2 audit does not automatically choose a winner.

Use this policy:

- prefer `raw` if raw training is finite, reconstruction learns, and there is no clear posterior-collapse pattern;
- choose `scalar` only if it provides a material optimization advantage without worse reconstruction or latent collapse;
- if both are unstable or degenerate, stop and revise the VAE objective/architecture before the primary beta=1 versus beta=4 experiment;
- never use held-out target AUROC, defect labels, or masks to choose normalization.

The default scientific preference remains `raw` because it changes the frozen DINO representation least. The audit exists to test whether that preference is numerically defensible.

## 8. What to send for review

After the fold-0 run, inspect or share:

```text
outputs/stage1_scale_audit_v2/fold_0_scale_audit_v2.json
```

Do not begin the 30-epoch primary VAE/beta-VAE training until the fold-0 dynamic audit has been interpreted.
