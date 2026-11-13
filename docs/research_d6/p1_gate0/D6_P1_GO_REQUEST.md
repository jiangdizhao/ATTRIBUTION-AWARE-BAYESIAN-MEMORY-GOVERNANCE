# LP-D6-P1 — Limited Execution Request (Prepared by Gate0)

**Requested decision:** approve or decline a narrowly bounded, exploratory D6-P1 baseline replay.
**Current state:** NOT AUTHORIZED. This file is a request, not approval.
**Gate0 recommendation:** a limited request is supportable with Protocol X as the primary cross-product diagnostic and Protocol W as a secondary same-product diagnostic. This would test feasibility/headroom only; the present sample does not support confirmatory claims.

## Proposed authorized scope

1. **Data roles:** accept the fixed W/X role map in `D6_P1_DATA_SPLIT_AND_TAPE_DRAFT.md`. X feedback products are porcelain_doll, u_block, usb_adaptor; X future products are sim_card_set, toothbrush, transistor1. W is two feedback and two future sample-key groups per product. Do not change roles after seeing model outputs.
2. **Historical supervision:** use only the existing first human annotation per registered image from the authorized D4-Lite/D4B D-fit and D5R-P2 V2 D-dev records. Repeats do not create extra items. No new label purchase is requested. Treat cost as assumed event units; actual cost remains unknown.
3. **Feature access:** separately authorize loading the frozen DINO encoder and extracting global features for exactly 54 D-fit plus 48 selected D-dev images (102 total). Do not access masks, other 72 Real-IAD views, other product JSON, D-conf, or sealed/HOLD material. First record the DINO Hub source identity, checkpoint SHA-256, environment versions, feature width, and current GPU memory. If the checkpoint cannot be pinned without a new download, stop and return for a separate decision.
4. **Methods:** implement and run B0 paired no-update, B1 online logistic regression with a ≤12-event replay buffer, and B2 exact recursive ridge using the same initial D-fit information, features, tape, evaluation set, and prediction-before-reveal order. B3/B4 are not in this request. Use only the fixed candidate settings listed in the baseline contract; report every setting, with no selection on D-dev labels.
5. **Learner/evaluator boundary:** no true product, defect type, source normal/defect code, masks, VLM labels, or future labels enter learner state, features, queries, or hyperparameter selection. Keep future labels in a separately joined evaluator path and log pre-reveal predictions and post-reveal states.
6. **Evaluation:** Protocol X product-macro BA/AP is primary, with all three future products shown. Protocol W is secondary; explicitly mark BA undefined for porcelain_doll and sim_card_set and report AP as all-positive/degenerate there. Report support, uncertain coverage, harmful flips, event count, state bytes, and latency. Cluster interpretation by sample_key and label physical independence INFERRED.
7. **Exclusions:** no D-conf claim; exclude all 54 D-fit images and all 120 Real-IAD P2 exposure images plus known/identifiable near-duplicates from any future D-conf. Historical sealed/HOLD overlap remains UNKNOWN and is not to be investigated by opening protected material.

## Conditions to resolve before start

- Researcher signs the retrospective, already-label-visible status of both W/X splits and accepts the small group/product support.
- Researcher approves the exact 102-image feature extraction scope, DINO source/checkpoint pin procedure, baseline implementation work, fixed settings, and a compute/storage ceiling.
- The actual DINO checkpoint/source identity and feature dimension are recorded. No package installation or checkpoint download is included in the current proposal.
- A static leakage review confirms that all five views inherit a single sample-key role, repeats are collapsed, and the evaluator cannot affect earlier predictions or updates.

If any condition cannot be met, keep P1 on HOLD/INCONCLUSIVE; do not replace the split, add data, or relax D-conf exclusions to make the run appear successful.

## Decision record (to be completed by researcher)

- [ ] Approve limited D6-P1 exactly within the scope above
- [ ] Decline / hold for a named blocker: ______________________________
- [ ] Amend the scope before approval (must produce a versioned contract): ______________________________

**This request does not authorize D6-P2/P3, new data, new annotation, D-conf access, or a scientific claim beyond exploratory feasibility.**
