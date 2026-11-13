# D6-P1-A Feature Integrity Report

**Status: PASS.** The authorized frozen global-CLS cache is complete. This is feature preparation only; no attribute learner was trained or evaluated.

## Scope and result

| Item | Result |
|---|---|
| D-fit | 54/54 approved D4-Lite/D4B original images |
| D-dev | 48/48 approved D5R-P2 V2 Real-IAD selected views |
| Total | 102 unique source images; no other image or mask accessed |
| Device | CPU only; two CPU threads; FP32; batch size 1 |
| Feature | 1024-dimensional L2-normalized global CLS from raw transformer block 18 |
| Completeness | 102/102 feature records and per-item output hashes verified |
| Numerical validity | All values finite; maximum norm error from 1 was 1.79e-7 |
| Source identity | All 102 source SHA-256 values matched manifests before extraction and during final audit |
| ID/role mismatches | None found |

The input manifest SHA-256 is `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63`. The deterministic output feature index SHA-256 is `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d`. The private index contains one row per image and per-item feature SHA-256 values. The `.npy` feature payloads total 430,848 bytes; the full ignored output directory also contains restricted manifests and per-item integrity sidecars.

## Canonical CLS equivalence

Before full extraction, one authorized D-fit image was processed on the same CPU, with the same checkpoint and FP32 preprocessing, through:

1. The existing `PaperAlignedFrozenSensor.encode_pil_batch()` reference, which retains selected patch-layer outputs and returns the global CLS vector.
2. A separate CLS-only adapter that uses the same preprocessor and same model, captures only the block-18 output, selects token 0, and applies the same `torch.nn.functional.normalize(dim=-1)` operation.

Predeclared tolerance was `atol=1e-6`, `rtol=1e-5`. The outputs matched: maximum absolute difference `5.9604645e-8`, relative L2 difference `8.4991e-8`, vector norm `0.99999994`, dimension 1024, all finite. The lighter path therefore passed the canonical global-CLS equivalence check. It does not retain patch feature maps or alter model weights, input preprocessing, block selection, token selection, normalization, or precision.

Preprocessing was RGB, resize 448, center crop 392, ImageNet mean `(0.485, 0.456, 0.406)`, standard deviation `(0.229, 0.224, 0.225)`, and block indices 4–18 available to the reference implementation. The persisted feature is the normalized CLS output from block 18 only. Storage is FP32; no FP16, AMP, PCA, or compression was used.

## Runtime and recovery evidence

- CPU smoke measured 4.31 seconds for the CLS-only path, estimating about 7.3 minutes for 102 images.
- The full extraction computed the remaining 100 images in 458.29 seconds after verifying the two already-complete items. CPU extraction RSS peaked at 1,770,688,512 bytes; the earlier reference-plus-adapter smoke process peaked at 2,797,170,688 bytes.
- Resume behavior was exercised: the first smoke feature was validated and skipped; a bounded two-item run produced one additional atomic item; the subsequent full run verified/skipped those two and computed the other 100. No valid feature was overwritten.
- Each feature is written to a temporary per-item directory and atomically renamed. The deterministic 102-row index is rebuilt from verified records. All 102 output hashes, dimensions, dtypes, norms, and finite-value checks passed.
- The output is under the ignored local path `outputs/research_d6/p1a_features/`. It is not in public Git. The input-path/ID mapping and features must remain local.

No annotations, confidence values, source defect types, official good/defect labels, masks, or VLM fields are included in the feature payload or read by the extraction script. The feature cache is not a training result and does not establish attribute-learning headroom.
