# E4: Asymmetric Verified-Feedback Write Audit (implementation v1)

Status: **implementation ready for local source-side validation; no Real-IAD outcome claimed.**

Canonical basis: `doc/ABMG_extension_v2.pdf`, 7 Oct 2026 handoff, the E3 ordinary diagonal NIG (Student-t) decision, and earlier structural-credit controls.

## Scientific question and locked scope

Does verified-normal feedback justify a bounded hard-normal correction bank above normal-state NIG updating (E4A)? Does **past-only** source-enriched credit improve future Core-4 factor routing versus binary-only and shuffled-source alternatives at exactly the same query positions (E4B)? If both survive separately, do they coexist without negative interaction (E4C)?

No active query policy, no VLM, no target-category tuning, no mask feedback, no learned sidecar, no DINO fine-tuning, no semantic/cause-factor claim. Keep the frozen raw-DINO detector, top-8 Sensor evidence, score-softmax T=20, and E3 diagonal NIG. This is a **source-side mechanism audit**, not end-to-end deployment validation.

## Why E4 needs a new cache

E3's existing `e3_evidence.pt` has only one score-weighted pooled 1024-D vector per image. Reproducing post-feedback hard-normal correction requires patch-level features. `abmg_e4_extract_evidence.py` reuses the E3 data sampling and frozen Stage-0 Sensor and saves:

- same baseline raw top-8 score-softmax pooled descriptor;
- frozen top-**64** Sensor candidate patch descriptors + original scores;
- the standard deviation of **all** original patch anomaly scores;
- source-only item metadata, with defect-source labels tagged offline-only.

E4 correction re-ranks **only the cached top 64**; patches outside this candidate pool are unavailable. This is an explicit, scientifically meaningful approximation to E1B's full 784-patch rescoring, not a claim of exact E1B reproduction. No masks are read. Source-only evidence generation opens the 24 source categories and never loads outer target JSONs.

## Matched design

For each of four E3 source-CV groups, six source products are held out and 18 development products provide only initial routing support, weak NIG prior, and a **fixed operating threshold** (95th percentile of the development-product 4-shot normal surprise). Use held-out-product 4-shot TRAIN normals to initialize category NIG. Divide held-out TEST images into a 25% **offline stratified** never-updated sentinel and a streaming set; the offline labels only stabilize the evaluation split and cannot influence query decisions. Shuffle the stream deterministically. Sample exactly **32** uniformly random query positions independent of predictions/labels and shared by every branch. Reuse checkpoint values `0,4,8,16,32`. Support seeds 0,1,2; stream/query seed fixed across support seeds.

Prequential order: score current image before feedback; if its fixed query position is selected, reveal its label (and Core-4 source only if defective) **after scoring**; update allowed external states; later evaluate never-updated sentinels. Normal-state NIG updates are identical across all branches. A defective query must leave `mu,kappa,alpha,beta` bitwise unchanged. No sentinel item is ever a query.

### E4A primary comparisons

- **N0**: verified-normal item updates category-local NIG with the original (uncorrected) pooled vector; no hard-normal correction bank.
- **N1**: the identical NIG write; on a verified false positive only, write all original frozen top-8 patches to a category-local negative correction prototype bank (8 entries, deterministic fill and nearest-prototype running-mean consolidation). Read: corrected candidate anomaly scores = frozen scores minus original full-patch score std times nearest nonnegative prototype cosine. Rerank top-8 inside the frozen top-64 candidates; score the corrected pooled vector using the unchanged NIG state. A true negative updates NIG only.
- N2 is **not implemented**, pending specific evidence that recurring hard-normal modes exist and N1 adds benefit.

Primary evaluation: sentinel normal FPR at frozen development-only threshold; simultaneous constraints on sentinel defect recall and AUROC. Secondary: AP, routing F1, number of genuine FP writes, fraction of changed candidate top-8 selections, effects by checkpoint. N0/N1 NIG state identity is asserted.

**Proposed preregistered E4A gate**, not a result: at Q=32, mean FPR improvement ≥1 pp, positive FPR improvement in ≥3/4 independent source-CV groups, recall loss ≤1 pp, AUROC loss ≤0.01. Repeated support seeds within a group are correlated, so count **source-CV groups**, not individual seed runs, when checking the 3-of-4 criterion. If insufficient verified FPs are queried, report "inconclusive at Q=32"; do not silently increase the confirmatory budget after seeing results.

### E4B primary comparisons

Freeze the E4A policy (`--normal-policy n0` unless N1 passes its independent gate). The Core-4 routing baseline is source-development product 8-shot/category-diverse `diag_shrunk` with shared variance. All D branches use the same prior, initial means, raw feature vector, current **uncalibrated** responsibility proxy `softmax(zscore(initial diag_shrunk logits))`, stream and queries.

- **D0**: binary-only credit `r_t`.
- **D1**: `alpha_t=0.5*r_t+0.5*q_{d,t}` where `q_{d,t}` is the centroid of responsibilities from **previously queried** defects of the same human-revealed source; when none exists use `r_t`.
- **D2**: same but `q_{pi(d),t}` for a fixed seeded derangement of the four sources. Same cold-start rule.
- **D3**: diagnostic, non-deployable one-hot true source credit.

A **false negative** verified defect from Core-4 updates the corresponding factor-routing sufficient statistics by fractional credit (total effective mass=1), using the identical original pooled descriptor. A **true positive** contributes to the future source centroid but produces no factor correction in this primary audit. This deliberately leaves outcome-dependent strength and spatially positive-mask writes for later mechanisms. Defects outside Core-4 produce no factor update; report their query count. For every defect, the NIG normal state is strictly read-only. The E4 scorer preserves fractional counts (earlier `score_memory` truncates them to `int`).

Primary outcome: sentinel Core-4 source-routing macro-F1; secondary true-source margins, per-source margin deltas, paired harmful/helpful routing flips, detector AUROC/recall, exact factor write weights, and write counts. **Important**: this primary factor-routing correction is not yet a demonstrated positive improvement to binary anomaly scores; the NIG binary detector is unmodified by a defect write. Do not assert end-to-end detector benefit from a routing-only result.

**Proposed E4B gate**: D1 outperforms both D0 and D2 by ≥0.02 macro-F1 at Q=32, positive differences in ≥3/4 source-CV groups, and no more than +1 pp off-source harmful flips. If any contrast fails, narrow/drop the C3 claim rather than rescue with new unregistered mechanisms.

### E4C factorial integration (only if the independent gates pass)

- C00: N0+D0.
- C10: N1+D0.
- C01: N0+D1.
- C11: N1+D1.

Exactly the same support, stream, query events, metadata, and thresholds; only permitted writes differ. Compare both separate mechanisms with the combined mechanism to detect interactions. The joint spatial-structural responsibility `q(t,p)*r(t,g)`, rare adaptive mask actions, and EFE/BED controller remain future work.

## Commands (from repository root)

```bash
# 0. Establish code version
 git fetch origin
 git switch stage2-e4-asymmetric-feedback-audit
 git pull --ff-only origin stage2-e4-asymmetric-feedback-audit
 git rev-parse HEAD

# 1. Validate CPU-only code before GPU extraction
 python -m py_compile scripts/abmg_e4_extract_evidence.py scripts/abmg_e4_feedback_write_audit.py
 python -m unittest discover -s tests -p 'test_e4*.py' -v

# 2. Source-only feature extraction, GPU required; replace paths
 python scripts/abmg_e4_extract_evidence.py --fold 0 \
   --root /PATH/TO/Real-IAD --json-dir /PATH/TO/realiad_jsons_sv \
   --out-dir outputs/e4_evidence --device cuda --query-batch-size 4

# 3. E4A source-CV: only after validating the manifest
 python scripts/abmg_e4_feedback_write_audit.py --fold 0 \
   --evidence outputs/e4_evidence/fold_0/e4_evidence.pt \
   --mode e4a --source-cv-folds all --support-seeds 0,1,2

# 4. E4B: choose n0/n1 based on the INDEPENDENT E4A outcome
 python scripts/abmg_e4_feedback_write_audit.py --fold 0 \
   --evidence outputs/e4_evidence/fold_0/e4_evidence.pt \
   --mode e4b --normal-policy n0 --source-cv-folds all --support-seeds 0,1,2

# 5. ONLY after both independent gates: factorial E4C
 python scripts/abmg_e4_feedback_write_audit.py --fold 0 \
   --evidence outputs/e4_evidence/fold_0/e4_evidence.pt \
   --mode e4c --source-cv-folds all --support-seeds 0,1,2
```

## Artifacts

- `outputs/e4_evidence/fold_0/e4_evidence.pt`, `e4_evidence_manifest.json`;
- `outputs/e4_asymmetric_feedback/fold_0/e4_e4a_n0_audit.json`;
- `outputs/e4_asymmetric_feedback/fold_0/e4_e4b_n0_audit.json`;
- `outputs/e4_asymmetric_feedback/fold_0/e4_e4c_n0_audit.json`.

Keep bulky caches/results untracked. Compare all four independent source-CV groups before advancing. Verify the per-run `n_core4_queries`, `n_hard_normal_obs`, `n_positive_factor_writes`; zero useful feedback makes the proposed gate **unassessable**, not a positive result.

## Out-of-scope / unfinished

The sealed outer-target five-fold confirmatory study is **not implemented here**. This branch is a source-side E4A/B/C mechanism implementation; after the gates are decided and protocol fully frozen, create a separately reviewed, leak-safe target-evaluation runner. The first actual Real-IAD run is mandatory before claiming runtime success or scientific outcomes. This document records design commitments, not observed results.
