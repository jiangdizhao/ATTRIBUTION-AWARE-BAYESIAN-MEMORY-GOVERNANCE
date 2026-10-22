# Stage 3 P0 — Canonical Memory I/O v1 (Draft Contract)

**Status:** P0 contract drafted, interfaces NOT yet wired into v22; no G0 or EG-FAMAR mechanism implemented.  
**Scope:** Design authority, invariants, replay behavior, and test obligations before Stage-3 G0 restoration.

## 0. Existing code is authoritative for behavior, not script defaults

Reuse rather than rewrite:
- \`scripts/test_phase3_AIF_simplified_v22.py\`: frozen scorer integration; support mode selection \`build_modes_from_train_good\`, \`select_clean_support_paths_per_mode\`; \`PatchVMB\`; \`FixedCapacityCodebook\`; \`MaturationBuffer\`; \`SignedCorrectionCodebook\`; \`GuardedDualAnchorThresholdController\`; \`SimpleAIF\`; conservative \`is_label_suspicious\` / \`should_write_normal_from_feedback\`.
- \`scripts/vmb_visionad_new.py\`: pretrained DINOv2 multi-layer backbone and patch nearest-normal scorer.
- \`scripts/abmg_realiad_dataset_inventory.py\`: *read-only* metadata, split, defect-type and physical-mask checks.
- \`configs/realiad_folds_v0.json\`: fixed five outer folds. Source CV is derived without reading any target labels.

**Critical replication caveat:** CLI defaults in the copied v22 do not establish the main-paper run. For example its argparse defaults include \`--K_normal_fixed 4096\`, \`--K_correction 256\`, \`--img_agg topk_mean\`, \`--simple_aif_target_qrate 0.06\`, while reported 30-class paper configuration uses 25,000, 1,024, max, and approximately 0.03. Also Stage-0's mean-fused top-1%-pixel sensor is NOT by default numerically equivalent to the legacy paper's concatenation + max setup. Before G0 restore exact historical commands, checkpoints, logs and per-item traces. No invented overrides.

## 1. Inference/feedback/commit temporal boundary

At time t, immutable DINO and fixed support extract evidence; read approved \`NormalReferenceStore\` views; score with raw normal nearest-neighbor + bounded \`SignedCorrection\`; use current \`ThresholdSidecar\` to produce \`prediction\`; select query with current \`QueryController\`; *then* reveal only purchased supervision. \`AddressRouter\` proposes destinations/abstains, \`GovernancePolicy\` may approve/defer/quarantine/reject. Only approved writes produce new state visible at t+1. Evaluator labels and masks never enter this online chain otherwise.

For fixed-event mechanism isolation, replay identical support IDs, B/R revelations, query IDs, ordering, capacities, correction/threshold settings. In closed-loop tests, allow Sidecar and BEA to evolve causally under the same preregistered policy/cost, and report realized query differences rather than claiming perfectly matched actions.

## 2. Canonical immutable inputs (per sample)

\`SensorRecordV1\`:
\`\`\`json
{
  "schema": "eg-famar.sensor.v1",
  "time": 0,
  "image_id": "opaque_public_id",
  "product": "audiojack",
  "preprocessing_profile_id": "resolved_in_P0",
  "frozen_checkpoint_sha256": "resolved_in_P0",
  "feature_ref": "feature-cache ID without labels",
  "patch_grid_hw": [28, 28]
}
\`\`\`
\`SensorRecordV1\` must not expose \`ground_truth\`, \`defect_type\`, \`mask\`, \`mask_path\` or oracle addresses. Ground-truth identity remains held in a physically separated evaluator-only manifest. The grid dimensions are illustrative of an existing Stage-0 profile, not a final verified v22 profile.

\`PredictionV1\`:
\`\`\`json
{
  "schema": "eg-famar.prediction.v1",
  "time": 0,
  "image_id": "opaque_public_id",
  "score_pre_feedback": 0.1,
  "threshold_pre_feedback": 0.15,
  "binary_prediction": 0,
  "state_version": "opaque_hash",
  "normal_memory_version": "opaque_hash",
  "query_action": "classify"
}
\`\`\`
Scoring must include the immutable Fixed bank irrespective of dynamic memory. \`score_pre_feedback\` is immutable in the log even if an action later updates memory. \`classify\`, \`query_binary\`, and \`query_rich\` are **trace outcomes**, not an authorization for a direct Rich request before obtaining Binary. Binary/Type/Mask authority is specified in the separate feedback document.

## 3. Mutable interfaces (each product-specific unless explicitly justified)

| Component | Immutable input | Reads / output | Authorized writes |
|---|---|---|---|
| \`NormalReferenceStore\` | frozen feature + selected Fixed supports | Fixed / STM / mature buffer / LTM; nearest-normal distance and provenance | verified NORMAL candidate only, after governance acceptance; Fixed cannot change |
| \`EvidenceCorrectionLedger\` | actual purchased B/R event, provenance, cost, reliability | immutable feedback event + typed views (signed correction, known type, normal evidence, risk probes) | append authorized evidence once; never overwrite label authority |
| \`AddressRouter\` | sensor evidence, previously authorized memory, legally revealed feedback | candidate operational address, soft responsibilities, confidence or \`abstain\` | no direct writes to Normal bank; address map changes only via approved proposals |
| \`GovernancePolicy\` | proposed write, current state, already purchased proof, risk-probe availability | \`approve/defer/quarantine/reject\`, bounded magnitude, reason, risk status | approval records; candidate may not mutate state until approval |
| \`SignedCorrectionView\` | labeled local event + allowed localization quality | bounded signed patch residual and anchor metadata | normal-side negative from verified normal; defect-side positive patch **trusted** only if R mask reveals patch; B defect remains weak/uncertain |
| \`ThresholdSidecar\` | verified \`(score_pre_feedback,binary)\` only | operating theta and normal-score scale | labeled calibration reservoirs; independently guarded movement |
| \`QueryController\` | pre-feedback score, uncertainty, budget, store state | B or optional Rich escalation request | query accounting; labels never revealed without real action |
| \`ReplayHarness\` | approved manifest and stream/query schedule | deterministic before/after events + compare report | evaluator-only access segregated from agent view |

An **address** is an operational routing key, not an intrinsically semantic/cause-related factor. Human-confirmed defect type is not automatically a normal-appearance address. \`null/abstain\` is a valid Router result and may trigger safe global Fixed NN fallback.

## 4. Canonical evidence event and governance proposal

\`EvidenceEventV1\` fields: \`event_id\`, \`image_id\`, \`time\`, \`product\`, \`reveal_tier\`, \`binary\`, optional \`defect_type\`, optional \`mask_ref\`, \`reveal_time\`, \`provenance\`, \`reliability\`, \`paid_binary_seconds\`, \`paid_rich_incremental_seconds\`, \`source_scope\`, \`event_hash\`. Type/mask are illegal unless queried, paid, and Binary is verified Defect. Mask is only trustworthy within the annotated region and with validated on-disk mask coverage. Missing/invalid mask must fail closed (stay B), never silently become predicted location truth.

\`WriteProposalV1\`: \`proposal_id\`, \`evidence_event_id\`, \`pre_state_hash\`, \`operation\` (\`insert/merge/replace/consolidate/correct\`), \`target_memory\`, \`address\`, \`responsibility_vector_or_null\`, \`patch_refs\`, \`proposed_magnitude\`, \`risk_probe_ids\`. All patch references must be from frozen sensor and available evidence; no evaluator mask is a patch selector except after paid R query.

\`WriteDecisionV1\`: \`proposal_id\`, \`outcome\` (\`approve/defer/quarantine/reject\`), \`risk_status\` (\`clear/flagged/unknown\`), \`authorized_magnitude\`, \`reason_code\`, \`source_data_version\`. With zero acquired defect probes, \`risk_status=unknown\`, not \`clear\`; no claim about unseen defect safety.

\`WriteReceiptV1\`: \`proposal_id\`, \`pre_state_hash\`, \`post_state_hash\`, \`affected_addresses\`, \`affected_prototypes\`, \`evictions\`, \`bytes_delta\`, \`replay_step\`, \`rollback_handle\`. Required for later G3 counterfactual audit; P0 defines contract only.

## 5. Specific invariants to test before each mechanism stage

1. **Invariant I1 / Support:** exactly the same four selected support images, augmentation and resulting immutable bank within paired branches; record predeployment candidate-pool access. Validate Fixed hash unchanged throughout stream.
2. **I2 / Read:** score always includes eligible Fixed NN; dynamic address-gated retrieval cannot disable Fixed evidence. Distinguish \`legacy mode\`, \`global NN\`, \`hard addressed\` and \`soft addressed\`.
3. **I3 / Label:** online code never reads unqueried image GT, type or mask; \`query_binary\` reveals only binary; \`query_rich\` requires prior paid B with Defect, valid Type+Mask and incremental cost. Agent trace excludes evaluator labels even when current item is correctly predicted.
4. **I4 / Sequence:** record score/prediction/query before reveal; no same-item retrospective score rewrite. Updates only affect t+1 onwards.
5. **I5 / Normality:** verified defect never authorizes Normal NN prototype admission. Positive defect anchors without paid spatial label cannot be high-trust patch GT.
6. **I6 / Budget:** all memories have explicit capped entries+bytes; no hidden duplicate evidence counts; amount of privileged information and annotation time matched per comparison.
7. **I7 / Risk:** unknown/absent defect probes are not scored as certified safe; measure harmful TP→FN and unknown defect recall on sealed *evaluator* after execution.
8. **I8 / Experimental fold:** 5 outer folds × 6 sealed targets, each product target once; four source-CV groups of 18/6 per outer, never use sealed target results for model selection.
9. **I9 / Threshold:** fixed threshold/query in isolation; identical rules but causal divergence in closed-loop; report FPR/FNR and AUROC separately.
10. **I10 / Baseline:** exact v22 behavior, run CLI and source identity documented; no accidental migration to Stage-0 sensor protocol.

## 6. P0 implementation status / interface binding plan

P0 **implemented** a read-only manifest linter \`scripts/stage3_p0_validate.py\`, and a manifest draft \`configs/stage3/experiment_manifest_v1.json\`. They do not import the old heavyweight v22 or visionad script or read any data. Canonical memory classes, runtime wrappers, and replay patches remain future G0/1 work after protocol review. Do not annotate them as already working.

Checks:
\`\`\`powershell
python scripts/stage3_p0_validate.py
python scripts/stage3_p0_validate.py --strict
python -m unittest discover -s tests -p "test_stage3_p0*.py" -v
\`\`\`
Normal P0 validation can pass while \`--strict\` **must fail** until unresolved source run CLI, hashes, support IDs, budget and safety parameters are reconciled. A failing strict gate is expected and protective. See the handoff for remaining review tasks.
