# Stage 3 EG-FAMAR — Current Research Handoff

**Updated:** 2026-10-23  
**Branch:** `stage3-eg-famar-execution`  
**Base Stage-2 HEAD:** `3b772a0737cce8d1f11897db7064441a476e9cf1`  
**Stage-3 status:** Planning/recording only. No G0/G1/G2/G3/G4 implementation or validation is claimed.

**Canonical Stage-3 execution authority:** [STAGE3_EG_FAMAR_RESEARCH_EXECUTION_PLAN.md](STAGE3_EG_FAMAR_RESEARCH_EXECUTION_PLAN.md). The 2026-10-22 EG-FAMAR PDF is the latest scientific proposal; if older 2026-10-16/20 handoffs conflict, use 10-22 staging/numbering and document any adjustment explicitly.

## Research identity

Frozen DINOv2 + primary open-set Normal NN + governed bounded mutable memory and score correction. The question is whether evidence-authorized, locally addressed writing AND compatible reading give persistent future utility with less off-address interference and without absorbing unknown defects. Scheme B is active; Scheme C human-grounded schema transfer remains optional/unvalidated.

## Decisions from 2026-10-23 review

1. **Support matters**: preserve original deterministic four-shot selection and exact four selected IDs, training candidate pool provenance, augmentations, fixed-bank and initial support-derived threshold. Previous random-support results show large FPR/FNR changes. Never compare methods using different support IDs.
2. **Sidecar matters**: retain the full legacy guarded dual-anchor controller; previous ablations show a real FPR/FNR trade-off, not guaranteed accuracy improvement. Use fixed-threshold/fixed-query replay for memory-causal attribution and normal live Sidecar for end-to-end operating results.
3. **Only two feedback tiers initially**: B = binary Normal/Defect; R = after verified Defect, additional *Type + Mask* with explicit incremental time/cost. R requested on Normal remains normal-only; do not fabricate a mask. No Point/Scribble/Box workflow in G1-G3. Four feature-hiding views of already-paid R evidence are permitted for *offline* contribution ablations, not distinct human-query tiers.
4. **Source discipline**: the previous paper's DINO/features/scoring/aggregation might differ from later Stage-0 defaults. Verify actual commands/preprocessing/checkpoints and `test_phase3_AIF_simplified_v22.py` before G0 parity claims.
5. **No leakage**: prequential `score→predict→query→reveal→update`. Hidden target GT/type/mask only for external evaluator, not memory, router or controller. Image-level binary FN does not identify defective patches.
6. **Stage-gated research**: P0 first; G0 only after user approval. Do not batch-implement G1-G4. Simple baselines must be allowed to win; don't force factorization, NIG or AIF.

## Current task: P0 protocol freeze ONLY

- Gather legacy source paper/supplement, v22 source, true run configs and logs; define exact DINO/sensor/score and Support+Sidecar interface.
- Draft `Canonical Memory I/O v1`, `Experiment Manifest v1` with folds/hashes/stream/support/query/cost/capacity, permission matrix and source-frozen stop criteria.
- Freeze five outer folds of six sealed Real-IAD target categories; remaining 24 source categories undergo 18/6 source-CV four-way evaluation; each of 30 products target exactly once.
- Establish two-tier feedback accounting and oracle-mask physical availability audit.
- Prepare G0 reproducibility acceptance criteria. **Do not start G0 implementation without an explicit next-stage request.**

## Next milestones (not yet executed)

- **G0** exact full legacy v22 + deterministic matched replay, query/memory/sidecar/no-leak audit.
- **G1** independent addressed Normal writes (legacy, global, clusters, raw DINO, shuffled) and addressed retrieval (global, v22 mode, hard local, soft gated), then joint read/write.
- **G2** B vs R, ledger/correction/type/risk readouts; comparison to simple separate stores and masks/types-shuffled controls.
- **G3** shadow complete-write risk audit; deterministic gate vs optional NIG; track harmful TP→FN and unknown recall.
- **G4 optional** at matched annotation minutes compare B-only BEA, random R escalation, heuristic and optional AIF/BED.
- **Final** sealed all 30 categories, paired product-level uncertainty, fixed/random Support robustness, under matched resource budgets.

## Historical results not to reinterpret

- Previous paper (designated 4-shot, 30 Real-IAD): Static Acc 0.5522/AUROC 0.8324; v22 full Acc 0.7837/AUROC 0.8801, qrate 0.0303; Stage-3 not yet evaluated.
- Previous structural credit F1 demonstrates known defect-source routing, NOT automatic normal-context addressability or end-to-end gains.
- Tested 32-D sidecar underperformed raw 1024-D DINO; NIG showed interval coverage gains, NOT demonstrated safety.
- Mask availability in JSON does not prove physical mask integrity; do an inventory before R conditions.

## Collaboration / Git discipline

Canonical current planning document is the linked roadmap, not old README status. For each coding-agent task: read this handoff, roadway and latest source; inspect current remote HEAD/working tree; implement only the approved stage; run reproducible tests; report diffs/commands/metrics/limitations; commit and push; verify remote SHA; then update this handoff and request the next gate review.

**No code changed by branch creation and Stage-3 documentation setup.** Do not merge Stage-3 planning into `main` automatically.
