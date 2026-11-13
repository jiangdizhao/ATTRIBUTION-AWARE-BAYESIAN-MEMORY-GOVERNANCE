# D6-P1-B Baseline Contract Amendment after P1-A

**Status:** Contract amendment for review only. No P1-B learner was trained, updated, or evaluated.

## Feature contract amendment

P1-A has produced the Gate0 global CLS representation as **1024-dimensional L2-normalized FP32**. This supersedes the Gate0 draft's tentative FP16 on-disk suggestion for this cache. All P1-B methods must read the same FP32 feature values, source IDs, D-fit/D-dev roles, and W/X feedback tapes. Do not recast to FP16, apply AMP, PCA, or alter the feature definition for one method. The feature cache contains no labels.

Only feature identity and integrity are established by P1-A. The 54 D-fit labels and 48 D-dev labels remain in their separate approved annotation records; the learner must receive labels only through the registered D-fit initialization and predict-then-reveal events. Product, sample key, source type, original split, good/defect, mask, confidence, and future truth remain evaluator-only unless a separate protocol explicitly approves a field.

## Minimum baseline amendment

| Method | Contract |
|---|---|
| B0 — No-update | Frozen copy of the D-fit initialized head; no post-initialization writes. Keep paired no-update copies for logistic and ridge families. |
| B1 — Online SGD Logistic + Replay | Binary logistic-loss online updates after each reveal; replay only from a bounded buffer of at most 12 unique revealed D-dev event records. The event is predicted before its label is revealed. Report all fixed settings and update/memory costs. |
| B1-Refit — Full-history Logistic refit | After each feedback reveal, fit the logistic head from scratch on all 52 binary-labeled D-fit examples plus all prior binary D-dev feedback events. It receives no additional labels, features, or annotations. Use regularization matched to the B1 candidate settings, selected before evaluation; report refit compute cost. This control prevents a replay-retention choice from being mistaken for an inherent limitation of ordinary logistic learning. |
| B2 — Recursive Ridge | Initialize on the same 52 binary-labeled D-fit examples and recursively update sufficient statistics for each revealed binary event, with no forgetting. Uncertain/not-applicable/unqueried states cause no binary update. |

Across methods, the two uncertain D-fit labels are masked, not converted to negatives. D-dev uncertain or not-applicable feedback consumes an event unit but produces no binary target update. Unknown/unqueried labels remain unknown. No official defect type or normal/defect label is an attribute target.

## Comparison and reporting rules

- Use the frozen D-fit initialization information and the exact same W/X feedback IDs, event order, reveal fields, future groups, and FP32 global features.
- At every query event, generate and record the prediction before revealing the legal state; update only afterward. Replacing an unrevealed future label must not alter any prior query, prediction, or learner state.
- Report W and X as distinct diagnostics. W has undefined per-product BA for porcelain_doll and sim_card_set. X has three future products with P/A support but only four registered groups each. Show class and group support, undefined metrics, AP degeneracy, harmful flips, uncertain coverage, memory, update latency, and inference latency.
- No future-label hyperparameter selection. Report the prespecified settings; do not select a winner on D-dev. Actual human time/money cost remains unknown; event counts are assumed equal-cost units only.
- B3 Context-aware Region Prototype and B4 CB2M remain deferred. P1-A contains no patch cache or region supervision and authorizes no extra location/mask input.

This amendment does not authorize P1-B execution. Separate researcher approval is required before any learner implementation/run, and P1-A success is not evidence that an online update improves future attribute predictions.
