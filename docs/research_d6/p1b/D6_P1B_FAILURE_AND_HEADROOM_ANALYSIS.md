# D6-P1-B Failure and Headroom Analysis

## Interpretation boundary

The replay shows that simple linear learners can consume the registered historical attribute feedback and change future-view scores under frozen global CLS features. It does **not** establish independent cross-product generalization, a human-query saving, or a publishable contribution. Each protocol has only 12 future sample-key groups and 21 binary-evaluable views. A sample key is operational grouping, not proven physical-object independence. The Gate0 roles were assigned retrospectively after the team had seen the pilot labels; this is not a prospectively blind test.

All D4/D4B and D5R-V2 images remain pilot-exposed and excluded from D-conf. D-conf, sealed targets, Probe, Mint/PCB and masks were not opened. Historical sealed/HOLD file-level overlap remains UNKNOWN.

## Attribute and source-type confounding

The following are evaluator-only cross-tabs. Source types were not converted into attribute targets and were never model inputs.

| D-fit source stratum (evaluator only) | Present | Absent | Uncertain | Total |
| --- | --- | --- | --- | --- |
| scratch | 18 | 0 | 0 | 18 |
| good | 2 | 21 | 1 | 24 |
| other source categories | 7 | 4 | 1 | 12 |

| D-dev source class (evaluator only) | Present | Absent | Uncertain | Total |
| --- | --- | --- | --- | --- |
| CH | 5 | 1 | 2 | 8 |
| HS | 5 | 0 | 0 | 5 |
| OK | 4 | 15 | 1 | 20 |
| other codes | 11 | 3 | 1 | 15 |

All 18 source-scratch D-fit images were human Present; the source-good stratum was mostly Absent, with two Present and one Uncertain. D-dev does show partial decoupling: source-OK includes four Present, and source-CH includes one Absent and two Uncertain. However all five HS examples were Present. This enriched pilot does not establish full-data deconfounding; the visual feature may encode cues correlated with source type or product even though those fields are absent from the model.

Gate0 records four known low-dHash pairs within D4B, all in D-fit and therefore in the same initialization role. Real-IAD sample-key groups were kept indivisible across roles. No cross-source perceptual-hash comparison exists, so cross-source near-duplicate status remains UNKNOWN. The role/annotation audit found no same-group feedback/future crossing, no repeated annotation used as another event, and no extra feature records.

## Paired outcomes and failures

- **Primary X balanced accuracy:** B1-SGD-Replay changed no product-macro BA for any alpha/seed. B1-Refit gains were `+0.111, 0, 0`; Ridge gains were `-0.083, 0, 0`. X contains only 21 binary future views across three products.
- **X average precision:** mean paired-class AP gain was approximately `+0.103` for SGD+Replay, `+0.064` for full-history Refit, and `+0.031` for Ridge. SGD's final AP gain was consistent across its six runs, while BA stayed fixed. Ranking changes therefore did not resolve the fixed-threshold errors in this sample.
- **Secondary W balanced accuracy:** SGD+Replay had zero BA gain. Refit gains were `+0.063, +0.125, +0.250`; Ridge gains were `+0.188, +0.250, 0`. W has only four BA-eligible products; two all-positive products have undefined BA.
- **W AP caveat:** all-positive products have AP=1 by definition. The all-product AP macro is optimistic; the paired-class macro is shown separately.
- **Harmful flips:** transition count means correct-to-wrong changes across consecutive checkpoints; the same image may recur. Net harm compares final prediction correctness with the same learner's own initial state.

| Protocol | Learner | Correct-to-wrong transitions/run, mean (range) | Unique views harmed/run, mean (range) | Final net harmed vs init, mean (range) | Final net corrected vs init, mean (range) |
| --- | --- | --- | --- | --- | --- |
| X | B1-SGD-Replay | 0.00 (0-0) | 0.00 (0-0) | 0.00 (0-0) | 0.00 (0-0) |
| X | B1-Refit | 5.33 (4-7) | 3.33 (3-4) | 0.00 (0-0) | 1.33 (0-4) |
| X | B2-Recursive-Ridge | 6.33 (4-9) | 5.00 (2-9) | 3.33 (0-9) | 4.00 (0-12) |
| W | B1-SGD-Replay | 14.00 (14-14) | 14.00 (14-14) | 14.00 (14-14) | 7.00 (7-7) |
| W | B1-Refit | 16.00 (11-21) | 13.67 (10-17) | 3.00 (2-4) | 3.33 (3-4) |
| W | B2-Recursive-Ridge | 14.67 (0-26) | 10.33 (0-17) | 1.67 (0-3) | 3.00 (0-6) |

In W, SGD+Replay ended with a mean of 14 initially-correct future views turned incorrect, offset by 7 corrected views; its product-macro BA did not change. Refit and Ridge had transient harmful transitions too, though their mean final net harm was lower. In X, Ridge averaged 3.33 initially-correct views made wrong at the final checkpoint while also correcting some views. These counts expose instability that a final macro score alone would hide.

The sample-key descriptive summaries also show substantial between-group spread: X SGD group accuracy ranges from 0 to 1 before and after updates, while W SGD's mean across group-level accuracies moves from 0.682 to 0.318. These group summaries average fixed settings within each group and are not inferential uncertainty estimates; one of twelve groups in each protocol had only Uncertain views and no group accuracy.

Every query prediction preceded its reveal. Valid labels updated only after reveal; Uncertain consumed an event unit without an update. Counterfactual tests changed unqueried future truth and found earlier predictions/states unchanged. Recursive Ridge matched direct batch coefficients and inverse statistics on synthetic data (`atol=1e-10`, `rtol=1e-9`). All 78 refit checkpoint states reproduced exactly under the fixed thread limit, and none reached `max_iter`.

## Headroom reading

There is **limited, metric-dependent headroom** for online adaptation in this global-CLS screen: some X ranking metrics and several W BA settings improve; X BA is mostly unchanged; W SGD causes net harmful future-view changes; and outcomes depend on estimator/configuration. This justifies retaining ordinary online attribute learning as a serious baseline. It does not show that a more complex mechanism is needed, or that improvements persist on unseen products or a deconfounded future set.

The main unresolved issue is evaluation identifiability rather than compute: few products/groups, retrospectively visible labels, source-type correlations, unproven physical-object independence, and unknown cross-source near duplicates. The CLS-only screen does not test spatial patch features. No query-cost reduction was measured because these are historical annotations with unknown real costs.
