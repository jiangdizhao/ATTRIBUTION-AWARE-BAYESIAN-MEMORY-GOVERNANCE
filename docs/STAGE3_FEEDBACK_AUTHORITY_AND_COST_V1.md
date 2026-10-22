# Stage 3 P0 — Feedback Authority and Annotation Cost v1

**State:** Draft contract; two human interaction tiers only. The experimental query policy and cost parameters are NOT calibrated. This document defines permissible information access, not a new trained controller.

## 1. Two operator-facing tiers

**B / Binary:** query result \`binary ∈ {0 normal, 1 defect}\` for one image, with measured/instrumented \`paid_binary_seconds\`. A B=0 can propose normal prototypes (with admission checks), negative signed correction and a labeled sidecar update. A B=1 supports image-level defect verification, sidecar calibration and uncertain candidate defect evidence/bag; it cannot assert any individual patch is defective or disclose a Type/Mask.

**R / Rich:** **after** a paid B query whose result was \`binary=1\`, the operator can upgrade to confirmed \`defect_type\` and an image-aligned \`mask\` (both together). It additionally incurs \`paid_rich_incremental_seconds\`; no payment implies no upgrade. The mask's geometry, nonempty foreground, image alignment and physical file availability must be validated. The initial prototype authority is restricted to truly masked defect patches. The outside-mask region is only a *candidate* normal area and needs boundary/contamination checks. A normal image has no Defect Type or defect Mask, so no R escalation occurs.

**Cost accounting:** \`cost(B)=measured_binary_seconds\`, \`cost(R)=measured_binary_seconds + measured_rich_incremental_seconds\`. No numerical seconds or cost ratios are fabricated in P0; they are mandatory unresolved manifest fields. A \`B→R\` escalation on the same image counts as one Binary event plus its paid incremental annotation, not two independent defect exposures.

## 2. Authority matrix

| True queried outcome | Permissible fields given to online method | Safe proposed usage | Forbidden |
|---|---|---|---|
| Unqueried or query rejected | no label, no type, no mask | score + choose actions from public features | self-inventing GT from evaluator |
| B Normal | binary=0 | guarded normal write, negative correction, sidecar normal anchor | assuming universal normality in every image region without scope checks |
| B Defect | binary=1 | image defect history, weak positive correction candidates in quarantine, sidecar defect anchor | interpreting top-K patches as true defect positions, known type or mask |
| R Defect with verified type+mask | binary=1, paid type, paid mask | verified masked patch correction, known-type retrieval (with abstain), acquired defect risk probe | dropping global NN open-set scoring or assigning masks to unrelated images |
| Invalid/incomplete R payload | B result only; mark R failure without inventing data | defer escalation and retain B evidence | filling missing Mask/Type from unqueried evaluator data |

**Human authority ≠ provenance of predictions.** A derived sensor saliency map, an automatic top-K or pseudo-segmentation is not equivalent to a paid ground-truth mask.

## 3. Separation of evaluator vs agent

The dataset index and GT mask metadata are maintained by evaluator-only code. Agent-visible \`SensorRecordV1\` contains only public image, product/stream identity and cached features. \`EvidenceEventV1\` is created **only after** the query decision. G0/Stage-3 runtime should use separate immutable evaluator manifests and a narrow reveal API; do not inject a private GT dictionary into a global object that arbitrary Router/Sidecar callbacks can read. Any feature cache with embedded labels is rejected.

The provided \`scripts/abmg_realiad_dataset_inventory.py\` audits metadata and can optionally verify physical files; it must be called as an offline preflight by an operator, never from the online agent. The Real-IAD JSON inventory from earlier work counted defects with mask paths; path declarations alone do not certify actual files, nonempty masks, or alignment.

## 4. Four offline information-hiding ablations, NOT four query tiers

When an R event has already been purchased and legitimately revealed in a fixed replay, we can mask its fields for downstream mechanism attribution:

| Offline condition | Passed to algorithm | Legitimate conclusion |
|---|---|---|
| F0 | Binary only | binary baseline |
| F1 | Binary + Type, hide Mask | incremental type-field use, *not a separate human type-only cost estimate* |
| F2 | Binary + Mask, hide Type | incremental mask-field use, *not separate mask-only cost estimate* |
| F3 | Binary + Type + Mask | full joint information benefit |

These are diagnostic information ablations under the same queried items. A Type/Mask improvement is not by itself cost-effective: G4 must additionally compare at **matched measured total annotation minutes** against B-only and random-upgrade strategies. The online operator interface remains two-tier B/R.

## 5. Unit, cost and availability rules

- Before G0: collect empirical annotation time samples (or a preregistered justified cost model) on source data and save immutable evidence. Null \`binary_seconds\` and \`rich_incremental_seconds\` in the P0 manifest are deliberate stop flags.
- No user-requested query to an unavailable/mismatched mask may silently receive a synthetic Mask. Log an invalid-R event and charge actual work spent; the algorithm receives only B.
- Distinguish **number of events**, **number of B labels**, **number of successful R upgrades**, **realized query rate**, **total human seconds**, and **acquired known/unknown defect types**.
- Do not count one rich label repeatedly if used by signed correction, type identification and risk audit.
- Calibrate query budget, escalation and any extra pilot masks exclusively within valid source-CV. Targets remain sealed for final evaluation.
- Every paired comparison uses the same pre-feedback score/query timing, exact support IDs and capacity constraints; prospective policy differences require separate closed-loop evaluation rather than forced identical event counts.

## 6. Minimal contract test fixture

\`\`\`json
{"time":0,"image_id":"sample0","action":"classify","pre":{"score":0.12,"threshold":0.15},"revealed":{},"paid_binary":false,"paid_rich_incremental":false}
\`\`\`

\`\`\`json
{"time":1,"image_id":"sample1","action":"query_binary","pre":{"score":0.22,"threshold":0.15},"revealed":{"binary":1},"paid_binary":true,"paid_rich_incremental":false}
\`\`\`

\`\`\`json
{"time":2,"image_id":"sample2","action":"query_rich","pre":{"score":0.22,"threshold":0.15},"revealed":{"binary":1,"defect_type":"scratch","mask_ref":"paid-mask-sample2"},"paid_binary":true,"paid_rich_incremental":true,"prior_binary_query":true}
\`\`\`

The read-only \`scripts/stage3_p0_validate.py --trace TRACE.jsonl\` checks the trace shape and prohibits obvious evaluator-only fields. It is a preliminary contract linter, NOT a proof of full runtime noninterference; at G0 use executable strict data-layer boundaries and adversarial leakage tests.
