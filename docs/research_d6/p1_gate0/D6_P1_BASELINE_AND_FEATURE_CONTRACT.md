# D6-P1 Gate0 — Baseline and Feature Contract

**Status:** Draft for approval only. No feature extraction, model training, inference, or P1 baseline run occurred.

## 1. Common learner contract

| Contract item | Frozen draft |
|---|---|
| Target | View-level `surface_abrasion_visible` under frozen Attribute Definition v1.1 |
| Binary encoding | Present = 1; Absent assessable = 0 |
| Excluded from binary loss/metric | Unobservable/uncertain, not_applicable, unqueried, and unlabeled views; keep each state distinct in coverage output |
| Initial data | Same 54 D-fit images for every method: use the 27 Present and 25 Absent labels; mask the 2 Uncertain labels from binary fitting |
| Online data | Exactly the same registered feedback events, stream order and reveal timing from the W or X tape |
| Evaluation | Exactly the same future-view set per protocol; first annotation per image only; report product and sample-key support |
| Input visibility | One shared frozen global image feature per view. No product ID, sample key, view ID, source defect type, official good/defect field, mask, VLM label, or future label reaches a learner |
| Update timing | Score the queried view before reveal; reveal only its authorized human state; then update. Future items are scored at checkpoints and never update state |
| Cost | One equal event unit as an assumption; actual monetary/time cost is UNKNOWN. Uncertain/N-A queries still incur the event unit but produce no binary update |

The three learner methods share the same D-fit source examples and the same feature schema, but their parameterizations differ by method. To isolate the effect of updating, B0 is a frozen copy of each initialized learner family (B0-Logistic paired with B1; B0-Ridge paired with B2). These paired no-update controls receive no post-initialization writes.

## 2. Minimum baseline definitions

### B0 — No-update

Initialize from D-fit, then freeze. Keep one paired frozen state for the logistic learner and one for recursive ridge so that B1/B2 comparisons do not conflate update effects with model-family differences. Report initial future metrics at feedback cost 0.

### B1 — Online Logistic Regression + bounded replay

- Use `sklearn.linear_model.SGDClassifier(loss="log_loss")` with a fixed L2 penalty, initialized on the same 52 binary-labeled D-fit examples. The installed scikit-learn is 1.3.0 in the recommended environment.
- Draft settings: `alpha ∈ {1e-5, 1e-4, 1e-3}`, constant `eta0=0.01`, deterministic one-pass replay, no future-label selection. Report all three settings; do not pick a winner on D-dev. If two seeds are used, freeze them as 0 and 1 and report both.
- Keep a bounded buffer of at most 12 unique D-dev feedback-event records (the full tape ceiling). After reveal, add a binary event and make one ordered pass through the currently paid binary buffer. Never replay unqueried or future examples. A nonbinary state consumes the event but is not inserted as a binary target.
- Report weight, intercept, replay-buffer bytes, update latency, and prediction latency.

### B2 — Recursive Ridge

- Initialize a ridge least-squares head from the same 52 D-fit binary-labeled examples, with identical normalized global features and a bias term.
- Draft regularization settings: `lambda ∈ {0.1, 1.0, 10.0}`. Report all three, with no D-dev model selection.
- Maintain exact regularized sufficient statistics and apply a recursive Sherman–Morrison update for each revealed binary feedback event; no forgetting and no replay are required because the sufficient statistics retain the same paid binary information. With augmented feature `z=[x;1]`, initialize `P=(lambda*I + sum(z*z^T))^-1` and `w=P*sum(z*y)`. For a new binary `(z,y)`, set `v=Pz`, `d=1+z^T v`, `w <- w + v*(y-z^T w)/d`, `P <- P - vv^T/d`. Uncertain/N-A states do not update the binary head.
- Report the matrix/vector state bytes, update latency, and prediction latency. Use the raw ridge score for AP ranking and a fixed 0.5 threshold for balanced accuracy; do not calibrate or threshold-tune on future labels.

### Deferred methods

B3 Context-aware Region Prototype and B4 CB2M remain in the full baseline matrix but are not part of this Gate0 minimum execution contract. Do not give either method point marks, masks, source defect labels, or additional annotation fields in the primary B0–B2 comparison. If later approved, the original and protocol-adapted CB2M implementations must be named separately.

## 3. Evaluation contract

- Primary endpoint: product-macro balanced accuracy over products with both Present and Absent future views, with the number of eligible products displayed. Protocol X has three eligible future products; Protocol W has four of six. Never fill undefined BA with zero or choose a new split to fix it.
- Companion endpoint: AP per product and product-macro AP, with positive prevalence and binary support. Mark all-positive AP as degenerate; do not interpret it as discrimination.
- Report each feedback checkpoint `k=0..12` on the fixed future set, update-by-update changes and harmful flips. Treat future views clustered by `(product, sample_key)`; 24 views are not 24 independent objects.
- Report uncertain/N-A coverage separately. Do not recode them as negatives. Repeated presentations are not extra items; use the first queue presentation for one item-level target.
- W and X are separate estimands and use different product support. Do not compare their macro numbers as if evaluated on the same product population.
- Official source type/good-defect fields stay evaluator-only. A shortcut diagnostic may stratify after predictions; true defect type is never a feature, query selector, target attribute, or update input.

## 4. Feature schema and extraction boundary

### Shared representation

Use one frozen global DINOv2 ViT-L/14 Register CLS representation per image view, L2-normalized by the existing Stage0 encoder and stored as one float16 vector per image. Cast to float32 for B0/B1/B2 input. Width `D` must be verified by an authorized encoder smoke step; do not claim the existing cache's unverified `feature_dim` is confirmed. The config names `dinov2_vitl14_reg`, 448 resize, 392 crop, patch size 14, and mean fusion over zero-indexed blocks 4–18. The current encoder's global output is the CLS feature from the deepest hooked block; patch outputs are fused spatial features.

Full-image/global features support the view-level abrasion classifier required for B0–B2. Patch/region features retain local spatial information and could support a later region prototype, but there is no dense abrasion-region truth. Existing optional points are not masks and are excluded from this primary comparison.

### What exists and what does not

- The P0 compact cache `realiad_dino_source_normal_compact` has 36,465 Real-IAD train-normal records over 30 categories, 285 shards, and 64 sampled patches per image, with global/patch fields declared. It is not a D6 cache: it contains neither the 54 D-fit pilot images nor the 48 selected V2 D-dev images. Tensor shards and true feature width were not read.
- `scripts/abmg_stage0_foundation.py` has a frozen DINO sensor with `encode_pil_batch()` returning patch and global features. It is a reusable encoder interface, not a D6 dataset/attribute cache writer. Its current record loader is Real-IAD-specific; D-fit needs a separate audited manifest adapter. No D6 item-to-feature cache or B0/B1/B2 stream runner exists.
- Therefore the 54 D-fit and 48 D-dev items require new feature extraction for the proposed comparison. Minimum pixel access is those exact 102 previously authorized source-side images only. Do not open masks or the other 72 exposed-but-unannotated Real-IAD views. This extraction and any model load require a separate explicit P1 authorization.

### Artifact and separation rules for a later authorized run

- Save features in a local non-public/non-Git artifact. Use opaque `image_id` keys; one row/vector per source image. Record source dataset/revision, original relative path in a restricted manifest, source SHA-256, feature schema/dtype/shape, preprocessing, model/repository revision, checkpoint SHA-256, environment versions, extraction time, and output shard/manifest SHA-256.
- Keep human labels and evaluator-only source metadata outside the learner feature payload. The learner reads feature rows by registered identity; an offline evaluator joins labels only after predictions.
- Do not overwrite the existing Real-IAD normal cache or historical receipts. Do not use unselected views or D-conf. Keep exact paths/labels/features out of public Git unless separately approved.

## 5. Remaining implementation work before P1 execution

1. Freeze the actual DINO hub-source revision and checkpoint SHA-256; the current cache filename alone is not an integrity pin.
2. Create a manifest adapter for the 54 D-fit and 48 D-dev images and verify feature IDs/hashes without opening any other image.
3. Implement and test the causal event runner: all predictions before reveal, update after reveal, no future truth access, fixed order, event-state snapshots.
4. Implement B0/B1/B2 and the separate evaluator, then conduct static schedule/ID checks before any scientific run.
5. Obtain researcher approval of the exact split, cost assumption, settings, storage budget, image scope, and P1-only execution boundary.

No model, feature, or image operation from this list was run for Gate0.
