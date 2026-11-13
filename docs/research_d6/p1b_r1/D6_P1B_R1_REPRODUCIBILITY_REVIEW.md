# D6-P1-B-R1 Reproducibility Review

**Date:** 2026-11-13
**Decision:** Reproducibility blocker repaired for subsequent preflight/reproduction. The original P1-B execution and its scientific outputs were not rerun or overwritten.

## Repair to the Git revision gate

The old gate compared current `HEAD` to `815e5c801be408d08807089d9b48eb372e9258a6` exactly. That value is the P1-B run's historical starting HEAD; the reviewed implementation was committed later, so the check rejected the archived branch after commit. The revised policy now requires:

1. current branch exactly `stage3-eg-famar-execution`;
2. approved baseline `815e5c801be408d08807089d9b48eb372e9258a6` to be an ancestor of the actual execution `HEAD`;
3. exact input, feature-index, feedback-tape, checkpoint, Hub-source-tree, preprocessing, source-image, and cached-feature fingerprints to continue matching.

A future run's provenance records its actual `execution_head`, the approved baseline commit, current P1-B implementation SHA-256 values, and the reviewed archived implementation identity separately. `capture_preflight()` returns the same identities. The exact data/fingerprint checks were not weakened.

## Historical source identity and limits

The original run provenance still records historical starting `git_sha=815e5c801be408d08807089d9b48eb372e9258a6`. At experiment execution, the new P1-B files were uncommitted, so that HEAD alone does not identify the full source. Its existing provenance records execution-time code hashes: `run_experiment.py=2c48e84345345921c9ec1bacac7f384082c093d837268f5a7ada91dd28f8e855` and `core.py=f45aafa175dc4691ea505a2b130f01dd20d76f9987d676772d47f2ab656f9f58`.

The subsequently archived/reviewed P1-B implementation is commit `743ee24854465c3f4c8079aaaa74a6e3dd93ad44` with `run_experiment.py=af3f221e885a51aa14d9adbdbde8c8ddf30b163713730e58329460cf7fbce54e` and `core.py=f45aafa175dc4691ea505a2b130f01dd20d76f9987d676772d47f2ab656f9f58`. The runner hash differs because the post-run archive correction materialized explicit B0 coefficient/intercept/threshold arrays and verified their digests; it did not rerun learners or change predictions/metrics. This archived implementation is used as the reviewed source identity. The R1 working implementation hashes are `run_experiment.py=395d15ac5dbcc5ddc3811365f705ad3c1e9cde1ba140f0eddc663ec96100681f`, `core.py=f45aafa175dc4691ea505a2b130f01dd20d76f9987d676772d47f2ab656f9f58`, and `r1_score_audit.py=afaf8438f8da1a5d9056ce81713082fc30c483d82f03a4072be76596ab4f8086`; the R1 test file SHA-256 is `5ab582246542d8dd2a240f87c6c32a67677df988b012ddd225ec4d9d1ca83154`. These hashes describe the amended working tree and are not attributed to the original execution.

## Product aggregate restored

`D6_P1B_PRODUCT_METRICS_AGGREGATE.csv` has been regenerated directly from the original ignored `product_metrics.csv`, restricted to checkpoints 0 and 12. It contains 432 product-level rows. Fields are configuration/method, product, checkpoint, aggregate support counts, BA/AP, TPR/TNR, and metric-defined status. It contains no image IDs, sample keys, per-image labels, predictions, scores, or paths.

- Original metrics SHA-256: `a5dba54976298fb0540039326ddd2fe65b63b5f1a9ee2b0e347cf12b3e3410c6`
- Published aggregate SHA-256: `38239adad11bc72feef44924cdbbefede89afc05a4195e79a5ee497efdf2b0d5`
- Local original-run hash manifest SHA-256: `21300b8dca1f2d13524f0d970022161cd7189746395c121c00c66f9fbc5593bd`
- P1-A input manifest / feature manifest / tape SHA-256 remain `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63` / `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d` / `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1`.

The existing link in `D6_P1B_BASELINE_RESULTS.md` now resolves to this regenerated aggregate. The aggregate equality check compares every exported field and row against the original product metric source.

## Tests and preservation check

- `py_compile` passed for the amended runner, score audit, and R1 tests.
- Six R1-specific tests passed: actual committed HEAD ancestry and full data preflight, wrong branch/non-ancestor rejection, incompatible fingerprint rejection, exact public aggregate comparison, logged W threshold transition recomputation, and pinned hashes of all 15 original run files.
- Tests exercised no learner initialization, training, or update. They read/hash authorized existing cache and P1-B outputs only.
- The original run's 15 files still match their pinned SHA-256 values; preflight input snapshot SHA remains `580940b5bc9b204d6bded6a548e1cc76fdba65e8b05be1f8bfbbb8fc35d8b0c7` before and after the original run. No original provenance, prediction, model-state, or scientific output file was changed.

## R1 diagnostic artifacts

Full per-image diagnostics remain local under the Git-ignored `outputs/research_d6/p1b_r1/` directory. It contains the 60-row per-configuration checkpoint table, 1,260 per-image score rows, 270 product/checkpoint rows, rank diagnostics, cross-configuration prediction agreement, state comparisons, and an audit provenance file. The public report is [D6_P1B_R1_SCORE_AND_THRESHOLD_AUDIT.md](D6_P1B_R1_SCORE_AND_THRESHOLD_AUDIT.md).

## Scope and next step

No D6-P2/P3, A1/A2, ABMG, feature extraction, data access, annotation, or model training occurred. The original P1-B numerical findings do not change. Interpretation is narrower: X AP gain is concentrated in toothbrush; W's 14/7 hard-label transition is an all-positive to all-negative threshold crossing accompanied by marked rank changes, not a pure intercept-only explanation. This does not establish a new performance claim.

**Recommended next step:** research lead reviews this R1 amendment and decides separately whether to accept it into the repository. D6-P2 remains unauthorized pending a separate decision.
