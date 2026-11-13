# D6-P0 Experiment Contract Draft

- **Status:** Draft for researcher review; no split, feedback tape, method or threshold is frozen by this document.
- **Plan:** Last Paper Four-Month Research Execution Plan v1.1.
- **Current route:** A_CONTINUE_FEASIBILITY. ABMG_ALTERNATIVE_PLANNED is retained but not run. A1 remains HOLD, A2 remains SCREEN, and A3 remains a baseline/organization principle rather than a new algorithm claim.

## 1. Question and claim boundary

Primary feasibility question:

> Given the same frozen visual features and the same sparse, human-confirmed surface_abrasion_visible feedback, do persistent online updates improve predictions on later, disjoint sample-key groups across products, and how often do updates harm initially correct future predictions?

The first D6 screen concerns one view-level attribute only. It does not establish industrial anomaly-detection improvement, whole-object abrasion status, causal material knowledge, a reduction in real closed-loop query count, or independent generalization. A feature- or label-level improvement is not a detector contribution by itself.

Attribute v1.1 is fixed. Legal states remain Present, Absent assessable, Unobservable/uncertain and Not applicable. Only Present/Absent assessable are binary metric targets. Unknown, unqueried, uncertain and not applicable are never recoded as negative.

## 2. Candidate data roles

Roles below are candidates for a later approved P1 screen, not an executed or frozen partition.

| Role | Candidate contents | Permitted use after separate P1 GO | Boundary |
|---|---|---|---|
| D-fit | 54 unique CONVAD/MVTec pilot images from D4-Lite and D4B: 27 Present, 25 Absent assessable, 2 Uncertain across capsule, metal_nut and wood | Initial image-level attribute head or common starting state | Already PILOT_EXPOSED; source-stratum enriched; physical IDs UNKNOWN; D4B has four identified near-pairs; do not use these images for evaluation |
| D-dev | 24 Real-IAD sample-key groups, 48 manually labeled views across six products; all 120 images from these groups are already NEW_STUDY_PILOT_EXPOSED | Fixed prequential replay, within-product and cross-product development diagnostics, baseline/headroom selection | All labels are already researcher-visible; original split is test; development only; not D-conf |
| D-conf | None identified or opened | One final confirmation after all methods, definitions and metrics are frozen | Requires a separately authorized new source/product batch, group-level overlap record and blinded human labels |

D-fit and D-dev are separate datasets with zero exact pilot-manifest path/SHA overlap. That reduces direct duplication risk but does not establish that the source tasks or label distributions are exchangeable. The 54 D-fit images are still historical annotation exposures, not newly purchased feedback.

### Group and product rules

- The indivisible split unit is a registered sample-key group. All five source views from one sample key must receive one role, even though only two views were manually labeled in V2.
- Physical-object independence remains INFERRED. Use “independent registered sample-key groups” unless source provenance later verifies physical objects.
- Never split views, crops, patches or augmentations across roles. A feedback event on a group disqualifies every view of that group from future evaluation.
- Same-product future adaptation means feedback and later evaluation groups share product identity but not sample key.
- Cross-product transfer means feedback products and future-evaluation products are disjoint by a frozen product assignment. It is not strict zero-shot product generalization if initialization, preprocessing or metadata exposed those products.
- Existing V2 labels have been viewed. Any split is a D-dev protocol, not a blind confirmation split. Apply a deterministic label-independent assignment and report missing class support rather than searching for a favorable split.
- All 54 CONVAD images, all 120 Real-IAD P2 exposures, their registered groups, and confirmed near-duplicates are excluded from D-conf. Historical sealed/HOLD file overlap remains UNKNOWN.

An exploratory layout could use D4-Lite/D4B as the D-fit candidate and divide the 24 D5R groups into fixed feedback and future roles. For a three-product/three-product cross-product diagnostic, the absolute ceiling is 12 groups on each side. A same-product layout has only four groups per product total; a two/two split is merely a candidate and may leave an evaluation product without both classes. These are arithmetic bounds, not preselected identities or sufficient sample sizes.

## 3. Supervision and access contract

### Learner-visible at prediction time

- Frozen DINO-derived global and patch features, once a separately approved feature bridge exists.
- The current view and only context fields registered as available in deployment. Product identity may select a shared fallback/context rule but cannot encode the target label.
- The learner's current persistent state and prior paid feedback events.

### Feedback fields that may be revealed after a prediction

Primary track:

- One human v1.1 state for surface_abrasion_visible on the queried view.
- Confidence may be retained for evaluator audit; whether a method may weight by confidence must be fixed before P1 and shared by every baseline.
- A present-location point may be used only in a separately declared point-enriched track. The same point field and its cost must be available to every method. No point is imputed for a missing field.
- Feedback costs are logged as event count plus an explicit ASSUMED cost sensitivity. Existing elapsed_seconds is not a valid actual-time estimate.

Never reveal to a learner unless a separate task grants and matches that exact field:

- Future attribute labels, unqueried labels or labels for another view in the same sample-key group.
- Official OK/defect/type codes, defect masks, VLM labels, candidate strata or evaluator-only splits.
- Product ID as a hidden attribute answer.

Defect type, original split, official normal/defect and mask fields are evaluator-only diagnostics. An optional type-prediction model must be trained only with explicitly permitted source-side type supervision and reported as a separate extra-supervision control. It cannot receive true target defect type.

The single-attribute record schema must remain extensible to multiple co-occurring attributes: per-field state, authority, reveal time, region binding and cost are separate. A missing abrasion annotation says nothing about contamination or another property.

## 4. Event sequence and causal invariants

For every stream event:

1. Load only registered features and the current state.
2. Predict the current view/group and write the pre-update score, attributes, state hash and event ID.
3. Consult a feedback tape fixed from registered identities, not hidden labels or candidate model outcomes.
4. Reveal only the purchased attribute field, if any; leave other fields unknown.
5. Update the learner and record changed state, memory size and update time.
6. Score only later, disjoint sample-key groups. Do not score another view from the just-queried group as independent future evidence.
7. Evaluate with an offline evaluator that owns the future labels. Its values never enter learner state, query order or hyperparameter selection.

Before P1, an independent reviewer should verify that replacing any unqueried future truth leaves the earlier event tape, prediction and learner state unchanged. A fixed-tape replay is not a live query-policy trial and cannot establish actual query savings.

## 5. Minimum strong baseline contract

All baselines use the same feature cache, D-fit initialization, D-dev group/product roles, feedback IDs/order/fields, unknown handling, prediction contexts and future labels. Every learner receives prediction before reveal. Report memory bytes, update latency, inference latency, training/tuning budget and feedback cost.

| ID | Baseline | Required protocol adaptation | Current local implementation status |
|---|---|---|---|
| B0 | No-update | Common initialized attribute head; no post-start writes | Trivial control; no D6 runner |
| B1 | Online Logistic + bounded Replay; lightweight MLP + Replay as a capacity sensitivity | Train only revealed binary attribute fields; replay only already paid events; identical replay capacity, event order and tuning opportunities | sklearn is installed; local logistic code exists for factorization diagnostics, not this attribute stream. No D6 online replay implementation. MLP can use scikit-learn, but current DINO feature input is absent |
| B2 | Recursive Ridge / CONCIL-style analytical update | Exact recursive sufficient-statistic update on the same features and revealed states; unknown/N/A masked; fixed regularization from source-side development only | No direct local recursive attribute implementation located |
| B3 | Context-aware Region Prototype | Use the same frozen patch/global features, context and revealed annotation fields. Primary comparison must not receive extra point supervision. A point-enriched secondary arm gives identical point visibility and cost to all methods | Existing prototype code is for factor/anomaly diagnostics. No D6 attribute adapter |
| B4 | CB2M-style persistent concept correction | Separate the paper-faithful/original baseline from a sparse-feedback protocol-adapted baseline. Match features, concepts, revealed fields, event order and capacity; report any extra offline concept labels required by the original method | No local CB2M implementation or pinned code revision. Do not call an adapter an exact reproduction |

B1's logistic model is the primary ordinary online attribute learner; the MLP is a bounded sensitivity. B2 must not be disadvantaged by forced replay when its sufficient statistics retain the same paid information. B3's region access must not become an uncounted annotation advantage. B4 must not be weakened by silently removing required concept supervision; if the original cannot be matched, label the comparison protocol-adapted and document the mismatch.

Capacity fairness means equal candidate feature information, feedback labels, tuning slots and reasonable state budget, not identical parameter counts across fundamentally different update rules. Freeze a common maximum of three source-side configurations and two seeds as a planning ceiling; actual P1 compute/time limits must be approved before running. Do not tune on D-conf.

## 6. Defect-type shortcut control

1. Publish the evaluator-only product × source-type × human-attribute table from P0.
2. Report attribute BA/AP within source-type strata only where both states and enough independent groups exist.
3. Include an evaluator-only source-type oracle as a confounding diagnostic if useful; label it ILLEGAL_FOR_LEARNER and do not rank it as a deployable baseline.
4. A visual type-predictor is optional and uses only registered source-side type labels. It is extra supervision, reported separately; its predicted type may not be replaced by target truth.
5. Preserve all sampled strata and report enriched sampling. Never infer Present from CH/HS or Absent from OK.

The observed D5R pilot crosses labels with source type, but the pilot is too small and metadata-enriched to rule out a strong shortcut.

## 7. Outcomes and Headroom

### H1 — Feedback Headroom

Compare future attribute performance under the same initialized state at cumulative paid-feedback cost nodes. The event count is bounded by eligible independent groups; do not force 16 or 32 events. Attribute BA and AP are reported by product and overall macro average. For BA, both Present and Absent assessable must be represented in the evaluation role. AP must also report positive prevalence and group support. Use sample-key cluster intervals; with the current six products/four groups each, intervals are exploratory.

Same-product and cross-product future gains are separate endpoints. A harmful flip is an independently evaluated future item that was correct before an update and incorrect after it; already-wrong cases do not count as newly harmed. Report update-by-update performance, field reliability, coverage, memory growth and latency.

Historical pilot labels can support only retrospective replay. Their logged durations do not establish real paid-query time. Use feedback-event counts and a preregistered assumed-cost sensitivity until reliable prospective timing exists.

### H2 — Task Readout Headroom

NOT_READY. D5R V2 supplies view-level attribute labels and optional points, not verified attribute masks, a frozen industrial acceptance rule or an authorized detection evaluator. No mask was opened. Keep attribute learning separate from anomaly-detection claims. Any oracle attribute labels remain evaluator-only and are not an upper bound unless the readout, threshold and support are fixed independently.

### Metrics and decision values

- Primary candidate: future, product-macro balanced accuracy; companion metric: average precision, with per-product support and undefined cases shown.
- Report within-product future gain, cross-product future gain, harmful flips, coverage of uncertain/unobservable states and uncertainty by sample-key group.
- TBA may summarize future BA across cumulative cost nodes only after baseline-only feasibility and cost nodes are fixed.
- δ_task = 0.02 TBA gain and ε_FN = 0.01 unknown-defect FNR tolerance remain provisional v1.0 candidates, not frozen gates. Before candidate comparison, select one contribution path, estimate baseline-only variation, document task risk and freeze the effect/risk margins. If the current groups cannot support interpretable uncertainty, mark the result INCONCLUSIVE instead of relaxing the threshold.
- A fixed feedback replay cannot prove fewer real queries. A query-saving claim requires a later closed-loop, equally reliable and costed evaluation.

## 8. Gate and stop rules

**GO to a bounded baseline screen** only after the researcher signs the candidate D-fit/D-dev split, feedback fields/tape, metric path, evaluator boundary, resource ceiling and feature/runtime source.

**SIMPLER_BASELINE_SUFFICIENT** if a strong ordinary baseline explains the observed usable gain under the same feedback, feature and cost budget. Accept this result; do not add A1/A2 complexity to rescue a novelty claim.

**INCONCLUSIVE** if too few independent groups, missing class support, weak label reliability or wide group-level uncertainty prevents a decision. Propose one bounded data supplement, not an open-ended annotation loop.

**BLOCKED** for absent feature bridge, missing runtime dependency, permission conflict, group/identity mismatch or any label leakage.

**NO_GO for a candidate** only after a valid, sufficiently activated, fair comparison falsifies its registered contribution. A blocked or underpowered pilot is not an algorithm failure and does not terminate A.

D6-P1/P2/P3 remain NOT_AUTHORIZED. This document is a draft contract, not an execution command.
