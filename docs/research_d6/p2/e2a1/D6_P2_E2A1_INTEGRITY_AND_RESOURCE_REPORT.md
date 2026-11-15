# D6-P2-E2-A1 Integrity and Resource Report

**Status:** `E2A1_PASS`  
**Execution date:** 2026-11-16  
**Run identity:** `stage3-eg-famar-execution` / `6da731f2ecf93fd23e6c51e6f66639f2528b8d57`  
**Output:** Git-ignored `outputs/research_d6/p2/e2a1/`.

## Verification results

| Check | Result |
|---|---|
| Registered inputs | 102 unique rows; 54 D-fit + 48 D-dev; source SHA/path audit PASS |
| Canonical tensors | 102/102; `(1,3,448,448)` FP32, finite, content and file/array hashes verified |
| DINOv2 feature bundles | 102/102; exact whitelist join, dimensions, FP32 finite arrays, per-array hashes and unit norms verified |
| DINOv3 feature bundles | 102/102; exact whitelist join, dimensions, FP32 finite arrays, per-array hashes and unit norms verified |
| Masks | Both backbone masks match canonical content geometry; all counts are the full grid (1,024 and 784) |
| Cross-backbone identity | Ordered image sets equal; both join to the same canonical tensors |
| Protected assets | Before/after snapshots equal across 18 registered protected paths |
| Backbone identities | Before/after checkpoint, source-tree, snapshot, weights and config identities equal |
| Label/data boundary | No source labels read; no protected data opened; no training or E2-B |
| Output placement | Git-ignored; no output files staged |

An independent invocation of the cache verifier reloaded all arrays from disk and passed the same whitelist joins, hashes, shapes, dtypes, finite-value, normalization, and mask checks.

## Tests and engineering history

The runner executed the six E2-A0 canonical-preprocessing tests and eleven E2-A1 preprocessing/storage tests; both suites returned code 0. The E2-A0 test file is present locally with SHA-256 `0c10a6b451d5988fdd026eb6a5620f5808742d2f18e4d08a9b53a6b7b3496984`, but it is not tracked at the registered starting HEAD. It was retained unmodified and not staged; this archival gap should be included in the next planned code-archival review. The E2-A1 test file also remains untracked with the new implementation.

Two earlier engineering attempts stopped before loading DINOv2 because of a Python import-path issue and then a feature-interface hash-path issue. Their logs and two-image canonical smoke artifacts were preserved in separate ignored attempt directories. Neither attempt produced model feature caches. The path/hash handling was corrected; the final run completed successfully. No historical report or output was overwritten.

## Runtime and storage

GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB total. The recorded free memory at each backbone start was 21,452 MiB. Peak framework-allocated memory was 1,275,088,896 bytes for DINOv2 and 1,264,507,904 bytes for DINOv3. Existing GPU processes were not terminated.

| Stage | Processed | Worker wall time including model load | Feature-generation time sum | Total throughput |
|---|---:|---:|---:|---:|
| DINOv2 | 102 | 31.75 s | 8.35 s | 3.21 images/s |
| DINOv3 | 102 | 27.10 s | 6.69 s | 3.76 images/s |

The complete guarded task took 119.39 seconds, including tests, smoke checks, preprocessing, model subprocesses, and final integrity checks. This is below the two-hour budget. The output cap was 1,610,612,736 bytes (1.5 GiB); the recorded cache/task output is 1,006,956,552 bytes including the final task-result JSON, also below the cap.

## Reproducibility limits

The cache is reproducible against the recorded local code, model identities, runtime environments, and P1-A whitelist. The untracked E2-A0 test archival gap means a clean checkout at the starting HEAD cannot yet reproduce that test suite without separately archiving the already present local test file. The current workspace has multiple unrelated pre-existing untracked research files; none were staged or committed in this task.
