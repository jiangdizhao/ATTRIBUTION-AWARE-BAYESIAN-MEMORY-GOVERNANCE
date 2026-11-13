# D6-P1-A Review and Next Action

## Decision

**P1-A: PASS.** The 102 authorized global CLS feature records are complete, identity-checked, FP32, numerically equivalent to the reference feature, and restartable. CPU-only mode was used throughout; no D6 GPU work or model learning occurred.

## Acceptance checks

- Exact input scope: 54 D-fit + 48 D-dev; no other image, mask, or annotation value read by the extractor.
- Input/source SHA checks: 102/102 pass before/during extraction and again in final audit.
- Reference equivalence: pass at `atol=1e-6`, `rtol=1e-5`; max absolute difference `5.96e-8`.
- Output checks: 102 unique rows, shape `[1024]`, dtype float32, device CPU, all finite, normalized, and output SHA verified.
- Resume: pass. One smoke item and one bounded partial item were verified and skipped on continuation; the remaining 100 were computed once.
- Original annotation CSVs, pilot manifests, W/X tape, and historical cache were not modified.
- No Git commit/push. The output cache is under the ignored `outputs/` path.

## Boundary and next action

The feature cache is ready for **D6-P1-B contract review**. P1-B execution remains unauthorized. The researcher must separately approve the amended B0/B1/B1-Refit/B2 contract, exact W/X readout, tuning policy, and compute limits before any attribute learner is implemented or run. D6-P2/P3 and D-conf remain outside this authorization.

Do not interpret successful extraction as evidence for online learning, cross-product transfer, headroom, or a publishable contribution. A CLS-only failure in a later task would not by itself test spatial DINO features or establish failure of Direction A.
