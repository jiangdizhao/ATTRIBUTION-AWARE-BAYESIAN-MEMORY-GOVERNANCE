# D6-P2-E2-A1 Protocol and Data Audit

**Status:** `E2A1_PASS`  
**Execution date:** 2026-11-16  
**Execution identity:** branch `stage3-eg-famar-execution`, HEAD `6da731f2ecf93fd23e6c51e6f66639f2528b8d57`  
**Scope:** frozen-feature engineering only; no learner training or E2-B evaluation.

## Authorized input and identity

The run used only the 102 rows in the registered P1-A whitelist. The registered input manifest SHA-256 is `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63`; the P1-A feature index SHA-256 is `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d`; the W/X tape SHA-256 is `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1`.

The whitelist audit verified 102 unique opaque image IDs and 102 unique source-file SHA-256 values. Every source hash matched the registered source, and every path was within an already registered source root. The composition was 54 D-fit images from the CONVAD/MVTec-concepts source and 48 D-dev views from Real-IAD. This records the already authorized development split; it does not establish independent physical-object counts for the Real-IAD sample groups.

Exposure and authorization fields were checked before decoding. The 54 D-fit images retain their prior pilot exposure status. The 48 D-dev images retain `NEW_STUDY_PILOT_EXPOSED` status. Pilot-exposed material and identifiable near-duplicates remain excluded from future D-conf. Historical sealed/HOLD file-level overlap remains `UNKNOWN`; no protected image or label was opened to resolve it.

## Access boundaries observed

- Only the registered 102 source images were decoded. No directory discovery was used to find additional samples.
- No source defect labels, attribute labels, masks, or future evaluation labels were read by preprocessing or either model process (`labels_read=false`).
- No D-conf, sealed/HOLD, Probe, Mint/PCB, or other protected data were accessed.
- DINOv2 and DINOv3 ran in separate registered Conda environments and read the same persisted canonical tensors. Both ran offline with FP32, batch size one, `eval()` and inference mode; TF32 and autocast were disabled.
- The cache contains frozen visual representations and identity metadata only. It contains no learned attribute head, pooled MIL score, or model evaluation result.

## Canonical input protocol

Schema: `d6.e2a1.canonical_rgb_letterbox.v1`. Images were EXIF-transposed, required to decode as RGB, resized in one Pillow 9.3.0 BILINEAR pass so the longest edge is 448 pixels, and centered on a 448×448 canvas without cropping. Resized dimensions use round-half-up; odd padding remainder is placed on the right or bottom. RGB values are converted to FP32 and ImageNet-normalized; normalized padding is zero. The persisted input tensor shape is `(1, 3, 448, 448)`.

The DINOv2 and DINOv3 patch-center masks were generated from each recorded content rectangle. The independent post-run verifier confirmed each stored mask equals the geometry-derived mask. All 102 inputs had 1,024 valid DINOv2 patch locations and 784 valid DINOv3 patch locations.

## Decision boundary

These checks establish provenance and reproducibility of the authorized feature cache. They do not show that either representation recognizes surface abrasion, that spatial/MIL learning works, or that cross-product transfer succeeds. E2-B remains a separate, not-yet-authorized experiment.
