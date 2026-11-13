# D6-P1 Gate0 — Development Split and Feedback Tape Draft

**Status:** Draft for researcher decision. D6-P1 remains **NOT AUTHORIZED**.
**Prepared:** 2026-11-13
**Purpose:** Freeze the permitted development identities, event order, role boundary, and metric feasibility before any feature extraction or model run.

## 1. Evidence and scope

| Status | Finding |
|---|---|
| VERIFIED | The authorized D5R-P2 V2 manifest has 48 selected Real-IAD views in 24 `(product, sample_key)` groups, six products with four groups each. The exposure ledger has all 120 C1–C5 views and marks every row `NEW_STUDY_PILOT_EXPOSED`. All 24 groups have five distinct view IDs. |
| RECALCULATED | The annotation queue has 58 presentations for 48 unique image IDs. Using only the first queue presentation per image gives 25 `present`, 19 `absent_assessable`, 4 `unobservable_or_uncertain`, 0 `not_applicable`. Repeat presentations do not add evaluation instances. |
| VERIFIED | All 48 selected image IDs match exposure-ledger records by image ID, product, sample key, view, and SHA-256. All are in the recorded Real-IAD `test` split. No image pixels or masks were read for this Gate0 audit. |
| VERIFIED (manifest scope) | No augmentation-derived rows are listed in the selected/exposure manifests. Repeat annotation presentations are collapsed to one image-level item. This is a metadata check, not a pixel-based duplicate audit. |
| INFERRED | A sample key is the operational indivisible group. Physical-object independence is not proven; report “registered sample-key groups,” not 24 verified independent objects. |
| UNKNOWN | Historical sealed/HOLD file-level overlap remains unknown. No protected material was opened. D-conf is not identified, opened, or authorized. |

P0 already establishes 54 D4-Lite/D4B CONVAD/MVTec images as D-fit candidates: 27 Present, 25 Absent, 2 Uncertain, with physical IDs unknown. Exact pilot-manifest path/SHA overlap with the 120 Real-IAD exposures was zero. Four D4B dHash pairs at distance ≤4 are within D-fit and must remain clustered for any D-fit sensitivity analysis. No cross-source perceptual-hash comparison is available, so cross-source near-duplicate status is unknown. D-fit and D-dev are historical exposures, not new paid feedback or confirmatory data.

The current branch is `stage3-eg-famar-execution` at `39fddc870e3cce2804150322641137b93e763360`; this is the archive commit whose parent is the P0 audit-time HEAD `4f265d89a9a29b871d0f1ff5c0f66b35a5ad3c37`. No Git files were modified by Gate0.

## 2. Frozen assignment rules

The two assignments are deterministic and use opaque identities only. A group's ranking token is the two selected `item_id` values sorted and joined with `|`; `sample_key` is used only to bind all views to one role. Hash function is lowercase SHA-256 of UTF-8 bytes. No state, confidence, defect code, image appearance, model output, or future score enters assignment.

- **Protocol W — same-product future adaptation:** within each product, sort its four group tokens by `SHA256("LP-D6-P1-Gate0-W-v1|" + group_token)`. First two groups are feedback; remaining two are future evaluation.
- **Protocol X — cross-product transfer:** sort the six product strings by `SHA256("LP-D6-P1-Gate0-X-product-v1|" + product)`. First three are feedback products; remaining three are future products. The resulting fixed roles are **feedback:** `porcelain_doll`, `u_block`, `usb_adaptor`; **future:** `sim_card_set`, `toothbrush`, `transistor1`.
- Within a feedback group, select exactly one of its two V2-labeled views as the query view by the minimum `SHA256("LP-D6-P1-Gate0-{W|X}-query-view|" + item_id)`. The other selected view and all three unselected views from that group are excluded from future evaluation.
- A future group contributes its two V2-labeled views to evaluation. Its other three exposed but unannotated views remain unknown and are not evaluated. No sample key crosses roles; all five source views inherit their group's role.
- Order feedback groups by `SHA256("LP-D6-P1-Gate0-{W|X}-event-group|" + group_token)` and order future views by `SHA256("LP-D6-P1-Gate0-{W|X}-future-view|" + item_id)`. The tape lists 12 feedback events first, then the 24 future-view identities. At each feedback event, score the query view **before** revealing its existing legal attribute state, then update only on Present/Absent. Uncertain/Not applicable still consumes an event unit but produces no binary update. Future labels never enter the learner or event order. Score the fixed future set at the initial state and after each update checkpoint; do not update on future items.

Because the team has already seen all D4/D4B and D5R-P2 labels, these role assignments are retrospective development splits. The assignment algorithm is label-blind, but the split-selection process cannot be represented as a blinded or prospectively label-independent test. Do not tune on these future labels or claim independent generalization.

The D-fit/D-dev source distributions also differ. P0 records a strong D4B shortcut risk: all six scratch-stratum images were labeled Present and all six good-stratum images Absent. Real-IAD source defect codes are not abrasion labels; P0's small enriched V2 pilot includes Present outside CH/HS and Absent within CH/HS. These observations do not establish full-dataset deconfounding. Source type may only be used in evaluator-only diagnostics.

## 3. Protocol W support

Counts are first-presentation, view-level counts. `BA` is defined only with at least one Present and one Absent. AP is mathematically available with positives, but an all-positive product has degenerate AP and no negative discrimination. Report AP per product with support; show both the all-positive-inclusive product macro and the paired-class product macro so the degenerate cases cannot inflate the only headline number.

| Product | Feedback groups / events | Feedback-event P/A/U | Future groups | Future-view P/A/U/N-A | Future BA | Future AP interpretation |
|---|---:|---:|---:|---:|---|---|
| porcelain_doll | 2 / 2 | 2/0/0 | 2 | 4/0/0/0 | Undefined | Degenerate, all positive |
| sim_card_set | 2 / 2 | 1/1/0 | 2 | 4/0/0/0 | Undefined | Degenerate, all positive |
| toothbrush | 2 / 2 | 1/1/0 | 2 | 1/1/2/0 | Defined on 2 binary views only | Defined; very small support |
| transistor1 | 2 / 2 | 0/2/0 | 2 | 2/2/0/0 | Defined | Defined; very small support |
| u_block | 2 / 2 | 1/1/0 | 2 | 2/2/0/0 | Defined | Defined; very small support |
| usb_adaptor | 2 / 2 | 1/1/0 | 2 | 1/2/1/0 | Defined | Defined; very small support |

Across W future groups, 6 are Present-only, 2 Absent-only, 3 mixed, and 1 has no binary-labeled view. Thus per-product BA is undefined for two of six products; the eligible-product macro BA has only four products and must state its denominator. Do not replace missing classes by resampling a more favorable split.

## 4. Protocol X support

| Product | Role | Groups | Labeled views | First-label P/A/U/N-A | BA/AP status |
|---|---|---:|---:|---:|---|
| porcelain_doll | Feedback | 4 | 8 | 7/1/0/0 | Not evaluated as future |
| u_block | Feedback | 4 | 8 | 3/5/0/0 | Not evaluated as future |
| usb_adaptor | Feedback | 4 | 8 | 3/4/1/0 | Not evaluated as future |
| sim_card_set | Future | 4 | 8 | 6/2/0/0 | BA and AP defined |
| toothbrush | Future | 4 | 8 | 3/2/3/0 | BA and AP defined on 5 binary views |
| transistor1 | Future | 4 | 8 | 3/5/0/0 | BA and AP defined |

The 12 X feedback events contain 6 Present and 6 Absent states: porcelain_doll 3/1, u_block 1/3, usb_adaptor 2/2. The three future products each have both binary states, so an X product-macro BA/AP readout is definable. Future group patterns are: sim_card_set 3 Present-only and 1 Absent-only; toothbrush 1 each Present-only, Absent-only, mixed, and no-binary; transistor1 1 Present-only, 2 Absent-only, and 1 mixed. These are four groups per product, not verified independent physical objects. The small cluster count makes uncertainty intervals exploratory and weak.

X is the most interpretable cross-product feasibility diagnostic in this pool. It is not strict zero-shot product generalization: all six products had prior metadata exposure, all pilot labels are known to the team, the Real-IAD pilot uses the original test split, and the development assignment is made after those exposures. Product identity is evaluator-only; never provide true product or defect-type labels to the learner.

## 5. Tape artifact, costs, and exclusions

The controlled local file is `research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/p1_gate0_draft/D6_P1_WX_role_map_feedback_tape.csv` (240 rows: 120 exposed views × two protocols; SHA-256 `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1`). It contains exact IDs and roles but no annotation state values or image paths. Public docs omit sample-key/image-ID rows. Each protocol has 12 query events, 24 future-view candidates, 12 feedback groups, and 12 future groups. All five views of each group are assigned one role; same-group alternate views are explicitly excluded.

The attribute feedback already exists in the approved pilot annotations. The event cost is therefore **historical cost unknown; assumed one equal event unit** for sensitivity accounting. Existing elapsed time is not treated as paid query cost. No new human labels are authorized or required by this tape.

All 54 D-fit images, their known D4B dHash clusters, all 120 Real-IAD P2 exposure images and all views in their 24 groups are excluded from D-conf. The 72 unannotated V2 views are not silently treated as Absent. Historical sealed/HOLD overlap remains UNKNOWN. Different dataset names and exact path/SHA checks are not proof that all near-duplicates are absent.

## 6. Gate0 judgment

- **Protocol X:** legal identity separation and both-class future metrics are definable for three future products; only a small exploratory replay is supportable.
- **Protocol W:** group separation is valid, but BA is undefined for two product-specific future sets. Keep W as a secondary diagnostic with those undefined cases reported.
- **Inference limits:** the labels are visible, groups are few, physical independence is inferred, source strata are enriched, and actual feedback cost is unknown. No confirmatory claim is supported.
- **Decision:** a narrowly bounded D6-P1 request can be submitted for global-feature B0/B1/B2 development replay only. Approval must separately cover feature extraction and code execution. No D6-P1 action was run here.

### Input fingerprints

| Input | SHA-256 |
|---|---|
| P0 data audit | `2a47947ab6e8194a2dcf18fcd741e09abcfb8817c31174bb435706c717c0433f` |
| P0 experiment contract | `3f87ad3374422ec90c2c9949dbd95e113e09259b03e0c2b3894d295d4297d517` |
| P0 readiness manifest | `6385db5319643960259d0fb60b45bc1cf66c80be2d33f86b29de0188bbbbf57d` |
| V2 selection manifest | `c0def1f598d38ea68c78da028958f7ee030a7bdf21528d5a249fe8f8a6f81b7f` |
| V2 annotation queue | `23f3d1084006fd4f1e5f221225ec77862be3facc8713e3bf3592ce448ff3476e` |
| V2 annotations | `1d138ff1a420d6134ad7fc1f2830074cc3d41fda5a8d789a81dd55b0f0e4d9a4` |
| V2 exposure ledger | `a274bbbdae3a1e4c1822c2968458d9c3878a1f466e82177632be1c5b449d3306` |
