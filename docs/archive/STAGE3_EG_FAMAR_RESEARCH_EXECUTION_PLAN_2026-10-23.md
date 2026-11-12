# Stage 3 — EG-FAMAR Research Execution Plan (P0 / G0–G4 / Final)

**Date:** 2026-10-23  
**Status:** PROPOSED / DOCUMENTATION BASELINE — NO NEW STAGE-3 MECHANISM VALIDATED.  
**Branch:** `stage3-eg-famar-execution`  
**Fork point:** `stage2-patch-e3-e4-mechanism-validation` at `3b772a0737cce8d1f11897db7064441a476e9cf1`  
**Working title:** Evidence-Governed Factor-Addressed Memory Adaptation and Retrieval (EG-FAMAR), Scheme B.

> This is the execution record for the agreed Stage-3 research sequence. It records decisions and proposed experiments, NOT their completion or authorization to start G0 coding. P0 protocol review comes first, and each later gate needs a separate result review and user authorization. Do not merge previous Stage-2 F1/AUROC/NIG findings with new-system results.

## 1. Evidence authority and research scope

1. **Primary scientific plan:** `EG_FAMAR_Research_Plan_English_2026-10-22.pdf` (user-provided, currently outside this Git tree).
2. **Implementation/decision handoff:** `ABMG_Scheme_B_Research_Execution_Handoff_2026-10-20.docx` and 2026-10-16 handoff (user-provided).
3. **Legacy experimental authority:** `pattern_recognition_main_elsarticle_v_10.pdf`, `pattern_recognition_supplementary_v_10.pdf`, `scripts/test_phase3_AIF_simplified_v22.py`, exact earlier experiment commands/logs/artifacts.
4. **Repository Stage-2 scripts/docs:** mechanism audit evidence; branch history is not an integrated Stage-3 implementation.

Keep the DINOv2 visual backbone frozen at deployment. Keep the legacy VisionAD-style Normal NN as the primary open-set anomaly scorer and bounded signed correction as a residual. Work on external memory, attribution/address proposals, supervision authorization, readout, and governance. Operational appearance address ≠ ground-truth human-semantic factor ≠ defect-source type ≠ patch coordinate. Scheme C transferable human-grounded attribute/schema learning is **out of scope**, optional later.

Historical observations: earlier correct-address Five-Branch routing was stronger than broadcast/shuffled, 32-D sidecars were worse than tested raw 1024-D DINO routing, rare masks had modest localization benefit, and NIG improved some normal predictive interval calibration without established online write safety. These results motivate Stage 3 but do not validate it.

## 2. Decisions frozen for initial Stage-3 development

### 2.1 Support and initial operating-point integrity

- Preserve **the actual v22 non-mode legacy 4-shot support path** for the historical full-system anchor. The recovered user command (`command.txt` line 191, recorded in `configs/stage3/legacy_v22_recorded_cli_v1.json`) passes `--K_modes 1` but **not** `--use_modes`: K-means is NOT active. With no support seed the current source selects the first four sorted normal-train paths; with `--batch_support_seeds 0,1,2`, v22 makes three seeded random four-shot draws from the normal-train candidate pool. Treat these as distinct experimental conditions; the 30-class recovered reference uses the latter. Record selected support IDs, eligible candidate-pool access, augmentations, fixed bank, checkpoint, fusion, kNN and theta initializer before claiming exact parity. Four selected supports from a large candidate pool are NOT equivalent to seeing only four total predeployment normal images.
- Preserve `Fixed` support bank immutable during the online stream. The four support IDs influence both reference geometry and initial score/threshold scale; keep those mechanisms separately identifiable.
- Hold support IDs constant **within each paired method comparison**. Predeclare multiple alternative random four-shot support seeds as a separate robustness stress test; no target-set cherry-picking.
- Reproduce the legacy **guarded dual-anchor Threshold Sidecar**, including its normal/defect/recent reservoirs, FPR/FNR guards, update cadence and damping. Do not assume sidecar universally improves every metric: earlier staged ablation shifted the FPR/FNR trade-off.
- During *memory mechanism isolation*, hold threshold fixed (or use precisely frozen threshold replay), identical query IDs, correction policy and capacity. For *end-to-end closed-loop tests*, allow each branch's sidecar/BEA to evolve causally under the **same source-frozen rules and matched realized budgets**. Never combine fixed-query effects with adaptive-query effects into one causal claim.
- First check feature/scorer differences across `test_phase3_AIF_simplified_v22.py`, paper commands and Stage-0 feature cache: block fusion, 448→392 / 28×28 patches, layer/view reductions, 1024-D vectors, preprocessing, 4-shot augmentation, image max aggregation, and score normalization. File names/architecture labels alone do not prove numerical equivalence.

### 2.2 Exactly two *human interaction* feedback tiers for initial experiments

**B / Binary:** queried image receives only `Normal` or `Defect`. A verified normal may authorize normal-coverage proposals and negative correction (subject to gates); image-level Defect proves at least one defect exists, NOT which patch is defective. Binary FN top-K remains uncertain/bag evidence; no invented mask, type or high-confidence positive localized anchor.

**R / Rich:** first obtain the Binary result. If it is Defect, optional escalation returns **Defect Type + Grounded Mask** together, charging incremental annotation time. For Normal, no defect mask/type exists; the result stays a normal Binary event. Dataset GT/type/mask remain evaluator-only until the corresponding explicit paid query.

**Important:** only two operator-facing query tiers, but rich-record **offline feature-hiding ablations** are permitted: Binary only; Binary+Type (mask hidden); Binary+Mask (type hidden); Binary+Type+Mask. These are NOT four real query tiers and cannot by themselves establish cost-efficiency of independently requesting Type or Mask. Measure or predeclare actual per-event human time and charging rules, and distinguish *equal-event mechanical comparisons* from *equal-human-time utility comparisons*. Do not automatically add Point/Scribble/Box during G1–G3. Defer other granularity to G4 only if evidence justifies it.

### 2.3 Immutable supervision boundary and online order

For each stream image:
1. Read only state available before time t; extract frozen features.
2. Retrieve Fixed and eligible adaptive normal prototypes; add authorized signed residual; aggregate; score; compare with current threshold.
3. Record pre-feedback score, prediction, threshold, candidate address, and memory version.
4. Execute approved query policy; if no query, **never** reveal hidden GT/type/mask to the agent.
5. If queried, reveal only B (or, when enabled/paid, R) fields; log cost and immutable evidence provenance.
6. Router proposes address/responsibilities and may abstain; governance authorizes proposal/defer/quarantine/reject.
7. Apply permitted updates atomically with before/after hashes, receipts, budget and rollback support.
8. Evaluate learning only on subsequent samples (prequential). The current queried image becoming correct is not persistent adaptation evidence.

The evaluator may use full hidden labels/masks **only to score outcomes** after the action boundary.

## 3. P0 — protocol freeze **(NEXT ACTION; no G0 algorithm code yet)**

Deliverables: `Experiment Manifest v1`, `Canonical Memory I/O v1`, feedback authority matrix, fixed-support and threshold contracts, reproducibility record and explicit source-only stop thresholds.

P0 checklist:
- Record immutable DINO checkpoint/hash, exact transformations and feature cache contract; restore legacy `v22` runtime knobs from paper+code+actual run manifests, not assumed defaults.
- Define fixed support provenance, candidate-pool eligibility, four selected IDs per product, augmentations, initial threshold rule; separate optional normal-only commissioning as another explicitly costed condition.
- Record dataset/split hashes and image IDs, query visibility API, labels/type/mask physical availability, no-leak asserts.
- Preregister five **outer product-category folds** (6 sealed targets each, every Real-IAD category target once). Within each outer fold: 24 sources; four source-CV partitions (18 development / 6 source-held-out). All knobs and gates selected on outer sources only; sealed targets cannot guide retuning.
- Specify what qualifies as `known` versus `unknown` defect types with respect to only legitimate, acquired source information; do not make an unknown class known by inspecting target GT.
- Define fixed-query matched replay IDs, query types/costs, fallback behavior when Rich mask/type is invalid/missing, stream order, support/stream seeds and capacity/accounting across normal/correction/type/risk stores.
- Predefine primary metrics, paired product-level statistical unit, acceptable FN/unknown-recall degradation, minimum future-normal benefit, compute/storage budgets, and audit log schemas.
- Preserve BEA's historical source settings for the initial isolation; no retroactive tuning with sealed target labels.
- Audit local vs repository code availability and exact baseline log/CSV provenance before declaring G0 reproducible.

**P0 GO:** documented protocols reviewed and explicitly approved by user. Otherwise STOP.

## 4. G0 — full legacy restoration and matched replay

Implement/run only after P0 approval.

- Restore and match `Fixed + STM + Maturation Buffer + LTM + bidirectional Signed Correction + Guarded Threshold Sidecar + BEA`, including support and threshold initialization, layer/view fusion, distance scoring, capacities, `is_label_suspicious`/feedback reliability behavior.
- Reproduce recorded legacy prequential category/image traces and source reported full-30 Real-IAD macro reference (designated support): Static Acc 0.5522 / AUROC 0.8324 / FPR 0.7908 / FNR 0.0328; legacy full stack Acc 0.7837 / AUROC 0.8801 / FPR 0.3144 / FNR 0.1180 / realized qrate 0.0303. Numbers are **historical reference**, not success claims for this new branch; differences must be explained rather than hidden.
- Verify immutable Support and source-only threshold, query-before-reveal, no GT/mask leak, exact event order, memory sizes/evictions, raw score snapshots and reproducible matched replay.
- Separate static/fixed-threshold mechanism tests from live closed-loop guarded-sidecar tests; record when queries cannot remain identical under changing scores.
- Include matched Support random-seed stress tests only as robustness controls; report coupled support-bank/threshold effects, not solely representation effects.

**G0 GO:** key metrics and item-level trajectories reproduced (or discrepancies logged/approved), no-leak and capacity assertions pass. Otherwise STOP and repair.

## 5. G1 — local normal write address + addressed read consistency (B feedback only)

**H1 / write experiment:** shared B labels, same queried images, fixed memory budget, identical selected normal candidates and correction/threshold conditions. Compare `No Update`, `Legacy v22`, `Global/Broadcast`, `Appearance-cluster address`, `deployable Raw-DINO local address`, `Shuffled/wrong address`. Distinguish a mere address tag from a consequential selective write; do not use oracle normal attributes.

**H1-R / read experiment:** hold *exactly the same approved memory snapshot*, altering only read policy: `Global eligible NN`, `Legacy v22 mode retrieval`, `Hard local gate`, `Stable-global Fixed + soft-gated adaptive NN`. Fixed base bank always available. Validate address-router abstention and normal-mode transitions. A finite soft penalty cannot guarantee zero off-address interference.

**Joint test:** only after independent write/read effects: combine winning writing and retrieval to test their interaction.

Metrics: downstream future normal coverage, FPR@95% defect TPR, FPR/FNR at source-locked threshold, overall and unknown defect recall, image AUROC/AP, off-address score/state drift, prototype redundancy/eviction, compute cost, per-product paired deltas.

**G1 GO:** consistent future utility over several source-CV groups without unacceptable unknown-defect harm; better than simple strong controls. If no improvement over legacy/appearance clusters or read gating loses legitimate normal coverage, retain simpler component and narrow claim. Do not claim semantically disentangled factors without evidence.

## 6. G2 — evidence ledger, signed correction and sparse defect reuse (B vs R)

- Add append-only evidence/provenance with ID, image, product, time, actual revealed fields, reliability, authority, costs, address proposal, maturity, state hashes and write receipts. Four **logical** readouts: signed correction, known-type retrieval (abstain), normal-evidence view, risk-probe view. Physical unification is not assumed a contribution.
- Evaluate B-only vs small R upgrades on identical IDs as **mechanical/representation ablations**; later compare at matched total human minutes. For R, Type and Mask are exposed together only after a paid defect escalation. Offline mask-hidden/type-hidden variants answer attribution, not acquisition cost.
- Against `Legacy correction-only`, `simple separate stores`, `unified ledger`, `read-only evidence`, `type-shuffled`, `patch-location-shuffled`. Frozen Normal NN remains primary open-set detector; a known-type prediction cannot declare everything else normal.
- Keep image-level FN events uncertain until Mask is explicitly revealed. Weak/unlocalized anchors must not silently become high-confidence positive patch evidence.

Metrics: *future* FP→TN and FN→TP gains/regressions, known-type macro-F1 with abstention, pixel localization where legal, unknown-defect recall, permission violations, store bytes and human cost.

**G2 GO:** authority-aware reuse has attributable benefit over simpler matched-budget alternatives. Otherwise preserve a simpler logical/physical implementation. Rich feedback need not win when expensive.

## 7. G3 — counterfactual normal-write risk audit + deterministic vs NIG

- For **verified Normal** candidates only, simulate *entire* proposed insertion/merge/eviction/retrieval effect in shadow memory before commit. Test normal coverage and known defect risks: TP→FN at current/predeclared threshold, score lowering on existing FN, masked positive-patch degradation, and off-address interference. Historical probe events must have been legitimately acquired before t.
- Gate outcomes: `APPROVE`, `DEFER`, `QUARANTINE`, `REJECT`; keep atomic receipts and rollback handles. Without defect probes, status is `UNKNOWN` not `SAFE`. Historical/commissioned sentinels cannot certify unknown-defect safety.
- Compare `No gate`, `simple quota/threshold gate`, `deterministic full-shadow risk gate`, `NIG-assisted gate`, all on matched candidate proposals, replay, capacity and available evidence. Use source-frozen hyperparameters.
- Measure coverage–recall frontiers, refused useful normal writes, known TP→FN flips, masked defect-region degradation, unknown recall, regret, gate cost and marginal NIG normal-predictive calibration separately.

**G3 GO:** net future normal benefit subject to source-locked defect-recall tolerance; NIG remains only if it adds **decision** utility over deterministic gate. NIG interval coverage alone is not a safety win.

## 8. G4 — optional binary/rich feedback-cost escalation

**Conditional only if earlier mechanisms justify increased query complexity.** Retain only two operator-facing tiers: B and R.

Compare `Legacy B-only BEA`, `random rich escalation`, `heuristic value-based escalation`, optionally `AIF/BED expected information/decision value`. Rich includes actual paid Type+Mask only on defective queried samples. Predefine equal-human-minute comparisons plus equal-query-ID diagnostic replays. Report acquired confirmed defects, type/mask coverage, FPR/FNR, future utility per minute, and inference/runtime overhead.

**G4 GO:** clearly beats simple/random escalation at matched annotation time and acceptable recall; otherwise retain B-only or simple escalation. No presumption that AIF/BED must win.

## 9. Optional representation line R (non-blocking)

If G1 requires better addresses, separately test dimension-matched (e.g., 64/128/256/512-D) PCA/AE/factorizing representations against Raw DINO, with identical frozen sensor, support, queries, capacities and retrieval. Past 32-D sidecar failures do not refute every factorizer. Do not claim semantic disentanglement from low correlation or reconstruction performance.

## 10. Final 30-category sealed evaluation

- Perform all five product-level outer folds: exactly six **sealed** categories per fold; every category evaluated as target once. Tune thresholds, query, router, gate and hyperparameters only on each fold's outer-source development/validation categories.
- Comparator stack: `Static`, `full legacy v22`, `G1 only`, `G1+G2`, `G1+G2+G3`, `Full G4` only if it passed. Match support and supervision/cost/memory budgets appropriately; keep fixed-query causal ablations separate from actual closed-loop policy comparisons.
- Add planned paired random-support and stream-order stress tests. Across conditions: compare same Support IDs and seed/order; report macro across *product categories* with product-level paired differences/confidence intervals, failure categories and worst-case harm. A frame, queried event, or random seed is NOT an independent product.
- Primary: prequential image AUROC/AP, FPR, FNR and defect recall under source-locked threshold, FPR@95% defect TPR, known-vs-unknown recall, future gain on later samples, harmful TP→FN writes. Report human minutes/events, memory bytes/prototypes, p50/p95 retrieval latency and shadow cost.
- Do not optimize a single scalar utility if it conceals unknown-defect FNR. Preserve all negative findings and registered exits.

## 11. Milestone table and explicit approval boundary

| Milestone | Main artifact | Gate |
|---|---|---|
| **P0 — CURRENT** | Source-locked manifest, Canonical Memory I/O v1, authority and budget contracts | User approves G0 implementation |
| G0 | Legacy numerical + item-level replay, tests and discrepancy report | Baseline/no-leak restored |
| G1 | Paired addressed write/read study; H1/H1-R independent and joint | Future detection/coverage gain without unacceptable FN harm |
| G2 | Binary/Rich authority ledger and fair reuse ablations | Better than simple stores, attributable field value |
| G3 | Auditable shadow gate, deterministic/NIG comparison | Positive safe consolidation frontier |
| G4 (optional) | Cost-controlled B/R controller comparison | Higher future utility per actual human minute |
| Final | Sealed five-fold 30-class report and archived experiments | Claim only supported contributions |

**Execution discipline:** for each stage, (1) review+freeze design and manifest; (2) implement one bounded task; (3) run local test and capture exact commands/hashes/results; (4) external result review; (5) Git commit/push; (6) obtain explicit approval for next stage. Do not continuously implement G1–G4 in one coding-agent task. Keep the Stage-2 mechanism audits and prior paper unchanged and separately cited.

## 12. Minimum upcoming P0 files to create after approval

- `configs/stage3/experiment_manifest_v1.json` — freeze dataset/hash/folds/support/seed/score/query/capacity rules.
- `docs/STAGE3_CANONICAL_MEMORY_IO_V1.md` — exact Sensor, Normal Store, Evidence Ledger, Router, Correction, Sidecar, Query, Replay contracts.
- `docs/STAGE3_FEEDBACK_AUTHORITY_AND_COST_V1.md` — B/R reveal API, no-leak permission table, cost accounting and mask/type validity.
- `docs/STAGE3_G0_REPRODUCTION_REPORT.md` — only after G0 execution.
- `docs/STAGE3_RESULT_LEDGER.md` — eventually append all matched results, failures, code SHAs and go/no-go decisions.

**Present status:** only this Stage-3 plan/handoff has been committed. No claims of implemented G0–G4. User has not yet authorized G0 coding under the 2026-10-22 plan.
