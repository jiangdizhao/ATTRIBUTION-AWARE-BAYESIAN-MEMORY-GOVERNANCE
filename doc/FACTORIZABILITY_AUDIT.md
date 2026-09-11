# Factorizability Audit

## Purpose

This is a diagnostic workstream before changing the supervisor-approved ABMG proposal. It does **not** replace C1 and does **not** change the frozen detector. It asks a simpler question first:

> Where does repeatable defect-type structure live in the frozen DINO/v22 evidence?

The current beta-VAE reconstruction experiment showed that whole-feature reconstruction can lose anomaly-ranking information on fold 0. That negative result is sufficient to motivate an audit, but not sufficient to reject factor addressability.

## Data-access boundary

For an outer fold `k`, the audit reads only the 24 source categories. The six target categories are named from `configs/realiad_folds_v0.json` but their JSON files, images, labels, and masks are not opened during source extraction/evaluation.

The four remaining frozen six-category folds are used as internal source-side category-held-out CV groups. Therefore a defect-source probe is trained on products that are different from the products used for evaluation.

Ground-truth defect-source labels and masks are offline evaluator information only.

## Candidate evidence representations

The same frozen Stage-0 DINOv2-Register ViT-L/14 sensor is used throughout.

1. `raw_patch`: mean fused DINO descriptor over selected suspicious patches.
2. `nn_residual`: signed residual `h - n*` from the nearest normal support patch.
3. `abs_nn_residual`: absolute nearest-normal residual.
4. `whitened_nn_residual`: residual divided by normal-support coordinate scale.
5. `subspace_residual`: component of `h` outside a normal PCA subspace fitted from the four-shot support bank only.
6. `layer_distance_profile`: per-layer mean/max nearest-normal cosine distance across DINO blocks 4..18.

No candidate is assumed to be correct in advance.

## Two localisation conditions

### Oracle localisation

The Real-IAD pixel mask is transformed by the same 448-resize / 392-center-crop geometry as the sensor and max-pooled to the 28x28 patch grid. Any patch touched by the mask is retained.

This is an offline diagnostic only. It answers:

> If the defect is localised correctly, does the representation contain reusable defect-type information?

### Sensor localisation

The unchanged frozen nearest-normal sensor selects the top-K suspicious patches; default `K=16`, matching the predecessor v22 correction-write scale.

This answers:

> Does the deployed sensor expose the same reusable structure without mask access?

Interpretation:

- oracle strong, sensor weak -> localisation/evidence selection is the bottleneck;
- oracle weak, sensor weak -> the candidate representation itself is weak;
- oracle and sensor strong -> factor routing is plausible without changing detection.

## Evaluation

For each representation/localiser pair, evaluate across category-held-out source CV folds using:

- nearest-centroid defect-source classification (closest to the proposed source-to-factor mechanism);
- balanced multinomial linear probe as a secondary upper-bound diagnostic;
- label-shuffle controls for the linear probe;
- macro-F1, balanced accuracy, accuracy, and test-label coverage.

The linear probe never enters the online system. It is only a diagnostic of whether the candidate evidence contains linearly accessible defect-source structure.

Sensor localisation is also measured against the offline masks using hit@K, patch recall@K, and per-image patch AUROC.

## Commands

### 1. Inventory

CPU only:

```powershell
python scripts/abmg_factorizability_audit.py inventory --fold 0 --root "$ROOT" --json-dir "$JSON" --out-dir outputs/factorizability_audit
```

Inspect defect-type/category coverage before running the backbone.

### 2. Small extraction smoke test

```powershell
python scripts/abmg_factorizability_audit.py extract --fold 0 --root "$ROOT" --json-dir "$JSON" --max-per-type-per-category 2 --query-batch-size 2 --device cuda --out-dir outputs/factorizability_audit_smoke
```

This is engineering-only.

### 3. Smoke evaluation

```powershell
python scripts/abmg_factorizability_audit.py evaluate --fold 0 --features outputs/factorizability_audit_smoke/fold_0/factorizability_features.pt --permutations 2 --out-dir outputs/factorizability_audit_smoke
```

Do not interpret the smoke metrics scientifically.

### 4. Principal source-side audit

Default cap is 32 defect images per `(product category, defect type)` cell:

```powershell
python scripts/abmg_factorizability_audit.py extract --fold 0 --root "$ROOT" --json-dir "$JSON" --max-per-type-per-category 32 --query-batch-size 2 --device cuda --out-dir outputs/factorizability_audit
```

Then:

```powershell
python scripts/abmg_factorizability_audit.py evaluate --fold 0 --features outputs/factorizability_audit/fold_0/factorizability_features.pt --permutations 10 --out-dir outputs/factorizability_audit
```

The audit should be interpreted before exposing the six fold-0 target categories.

## Decision discipline

This audit is not a new proposal claim. A promising representation should satisfy all of the following qualitatively before it deserves further work:

1. category-held-out source macro-F1 is clearly above label-shuffle controls;
2. the effect is not restricted to one source CV fold;
3. label coverage is high enough that the metric is meaningful;
4. oracle-vs-sensor comparison identifies whether the bottleneck is representation or localisation;
5. nearest-centroid results are directionally consistent with the linear-probe diagnostic.

Only after this evidence should a new factor-address mechanism be proposed or the supervisor proposal be revised.