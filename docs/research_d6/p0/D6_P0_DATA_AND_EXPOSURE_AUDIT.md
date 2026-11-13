# D6-P0 Data and Exposure Audit

- **Study:** Sparse Human Feedback for Persistent Cross-Product Attribute Learning
- **Audit date:** 2026-11-13
- **Plan authority:** Four-Month Research Execution Plan v1.1
- **P0 status:** DOCUMENTATION_READY
- **Scientific/data readiness:** INCONCLUSIVE
- **D6-P1/P2/P3:** NOT_AUTHORIZED

## Scope and evidence labels

This is a read-only audit of existing documents, approved pilot CSVs, exposure ledgers, cache metadata, source code interfaces and Git metadata. No image pixels, masks, new Real-IAD JSON files, hidden labels or feature tensors were read. No model was trained, loaded or run. Existing annotations and manifests were not changed.

- **VERIFIED** means directly present in an authorized local document, manifest or ledger.
- **RECALCULATED** means recomputed from the named local CSV/JSON metadata.
- **INFERRED** means an operational grouping or interpretation lacks physical-object proof.
- **UNKNOWN** means the permitted evidence cannot settle the question.
- **BLOCKED** means this task did not authorize the required access or execution.

## Repository and authority

- Local branch: stage3-eg-famar-execution.
- Local HEAD: 4f265d89a9a29b871d0f1ff5c0f66b35a5ad3c37.
- Remote branch HEAD read from origin on 2026-11-13: 4f265d89a9a29b871d0f1ff5c0f66b35a5ad3c37. Local branch was neither ahead nor behind.
- The canonical v1.1 Markdown plan is present and tracked. The local 2026-11-12 D6 handoff DOCX is present and was read.
- No tracked working-tree or staged modifications existed at task start. Existing untracked user material included the v1.0/v1.1 plan DOCX files, the D6 handoff DOCX, research_triage/h1 and research_pilot/. They were preserved.
- The v1.1 plan and current Stage-3 handoff agree that the next work is D6-P0 preparation and that experiments remain on hold. Historical Stage-3 protocols do not grant new study permissions or override v1.1.

## Recalculated pilot records

### D4-Lite and D4B: CONVAD/MVTec source-side human pilots

| Pilot | First-label images | Present | Absent assessable | Uncertain | Repeat agreement | Image kind |
|---|---:|---:|---:|---:|---:|---|
| D4-Lite | 36 | 18 | 17 | 1 | 5/8 | 36 original real images |
| D4B | 18 | 9 | 8 | 1 | 7/8 | 18 original real images |

The 54 images have distinct recorded SHA-256 values. D4B's source manifest records zero path and SHA matches against D4-Lite. D4B's recorded dHash values independently yield four within-pilot pairs at Hamming distance at most four (one pair at distance three, three at distance four). These are development exposures, not D-conf candidates.

D4B source-category diagnostic, computed from its first presentations: all six scratch-stratum images were labeled Present and all six good-stratum images Absent. The remaining six images span bent, color, crack, faulty_imprint, flip and hole; their labels do not uniformly follow those source names. This is a strong shortcut risk in the enriched pilot, not proof that the full datasets are perfectly confounded.

The D4-Lite directory also contains annotations_export.csv with 26 D4B-P presentation IDs. Its state distribution matches the canonical D4B annotations.csv, 25/26 complete rows are identical, and the one changed row differs only in elapsed_seconds and last_saved_at_utc. It has no presentation-ID overlap with the D4-Lite raw annotations or queue. It was therefore not counted as D4-Lite data; it appears to be a misplaced D4B export and should remain untouched pending owner confirmation.

The D4-Lite and D4B manifests mark physical_object_id as UNKNOWN. Their image counts are not verified independent physical-object counts.

### D5R-P1 metadata exposure

The existing P1 exposure ledger has 18 product-JSON records, all marked NEW_STUDY_DEV_EXPOSED. Summing its authorized ledger fields gives 22,100 train records, 68,415 test records, 90,515 total metadata records and 18,103 sample-key groups. This was metadata exposure only; pixels, masks and feature tensors are recorded as unread. This exposure does not open any old Stage-3 item-level HOLD set.

The local P1 decision record reports that these 18 product categories also served as target products in old outer folds 2–4. Treat this as inherited P1 evidence: category-level metadata/defect-code blindness for those old fold roles is compromised for this new study exposure, while old fold files, results and separately protected Probe/Mint/PCB/Calibration items remain unchanged and sealed. Whether any specific unseen protected item overlaps remains UNKNOWN. Any of these 18 products used later with previously unseen pixels must still be described as metadata-exposed, not fully product-unseen.

### D5R-P2 V2: Real-IAD image-level human annotations

Recalculation from annotations.csv, annotation_queue.csv and selection_manifest.csv:

- 58 presentation rows, all with one of the four allowed states; no blank state, duplicate presentation ID, unknown queued presentation, or queue-to-item mismatch.
- 48 unique image items, 24 sample-key groups, 6 products with 4 groups each, and 10 blind repeats.
- Repeat agreement: 8/10. Disagreements: one Present/Absent assessable and one Present/Uncertain. No label was edited or adjudicated here.
- First presentation distribution: 25 Present, 19 Absent assessable, 4 Unobservable/uncertain, 0 Not applicable.
- All-presentation distribution including repeats: 28 Present, 25 Absent assessable, 5 Unobservable/uncertain, 0 Not applicable.
- All 25 first-presentation Present labels have a point mark. Among first Absent labels, one has a point; among first Uncertain labels, two have a point. These points are not masks or pixel-accurate region ground truth.
- Confidence among first labels: Present 14 high, 10 medium, 1 missing; Absent 17 high, 2 medium; Uncertain 4 high. Confidence is not calibrated reliability.
- All 48 annotated views are from the original Real-IAD test split. The full 120-row P2 exposure ledger has 120 unique item IDs and SHA-256 values; every row remains NEW_STUDY_PILOT_EXPOSED. The V2 selection manifest is a 48-image subset; the other 72 views remain exposed although unannotated.

First-label Product × Attribute counts:

| Product | Present | Absent | Uncertain | Not applicable |
|---|---:|---:|---:|---:|
| porcelain_doll | 7 | 1 | 0 | 0 |
| sim_card_set | 6 | 2 | 0 | 0 |
| toothbrush | 3 | 2 | 3 | 0 |
| transistor1 | 3 | 5 | 0 | 0 |
| u_block | 3 | 5 | 0 | 0 |
| usb_adaptor | 3 | 4 | 1 | 0 |

Sample-key group state patterns across the two selected views:

| Pattern | Groups |
|---|---:|
| Present / Present | 8 |
| Present / Absent assessable | 8 |
| Absent assessable / Absent assessable | 5 |
| Absent assessable / Uncertain | 1 |
| Present / Uncertain | 1 |
| Uncertain / Uncertain | 1 |

Ten of 24 groups have different states across the two views. This is view-level visibility evidence within operational sample-key groups. It does not prove that each sample key is a physically independent object or justify converting a negative view into object-level absence.

### Product × source group type × human attribute

Source type is evaluator-only metadata. Counts below use one first annotation per image; they do not derive or replace attribute labels.

| Product | Source group type | Present | Absent | Uncertain |
|---|---|---:|---:|---:|
| porcelain_doll | CH | 1 | 1 | 0 |
| porcelain_doll | HS | 2 | 0 | 0 |
| porcelain_doll | OK | 2 | 0 | 0 |
| porcelain_doll | ZW | 2 | 0 | 0 |
| sim_card_set | CH | 2 | 0 | 0 |
| sim_card_set | HS | 2 | 0 | 0 |
| sim_card_set | OK | 0 | 2 | 0 |
| sim_card_set | ZW | 2 | 0 | 0 |
| toothbrush | CH | 1 | 1 | 2 |
| toothbrush | OK | 0 | 1 | 1 |
| toothbrush | ZW | 2 | 0 | 0 |
| transistor1 | BX | 2 | 2 | 0 |
| transistor1 | OK | 0 | 2 | 0 |
| transistor1 | ZW | 1 | 1 | 0 |
| u_block | CH | 1 | 1 | 0 |
| u_block | HS | 1 | 1 | 0 |
| u_block | OK | 0 | 2 | 0 |
| u_block | QS | 1 | 1 | 0 |
| usb_adaptor | AK | 1 | 0 | 1 |
| usb_adaptor | CH | 0 | 2 | 0 |
| usb_adaptor | HS | 1 | 1 | 0 |
| usb_adaptor | OK | 1 | 1 | 0 |

Across sample-key source groups: CH = 5/5/2, HS = 6/2/0, OK = 3/8/1, AK = 1/0/1, BX = 2/2/0, QS = 1/1/0 and ZW = 7/1/0 (Present/Absent/Uncertain). There are Present labels outside CH/HS and Absent labels within CH/HS. At per-view source class level, OK has 4/15/1 and non-OK classes combined have 21/4/3. This is evidence against a literal one-to-one mapping in this selected pilot, but the source-stratified sample is too small and enriched to establish general deconfounding.

The authorized P1 selection record maps AK to pit, BX to deformation, CH to abrasion, HS to scratch, QS to missing parts and ZW to contamination. These are source defect categories, not human attribute values.

## Exposure, source identity and D-conf exclusion

- D4-Lite and D4B together contribute 54 recorded original real CONVAD/MVTec images. Their combined recorded paths and SHA-256 values have zero exact overlap with the 120 Real-IAD P2 exposures.
- The 48 V2 images match 48 rows of the 120-row Real-IAD exposure ledger by identity and SHA-256. All 120, not just the 48 labeled images, are excluded from any future D-conf claim. All views in their 24 sample-key groups are also excluded.
- D4-Lite's 36 and D4B's 18 images, plus the four identified D4B dHash near-pairs, are excluded from D-conf.
- Exact path/SHA source separation is verified only over the available pilot manifests. D4-Lite and Real-IAD lack a common, audited perceptual-hash comparison here; cross-dataset near-duplicate status is therefore UNKNOWN. Historical sealed/HOLD item-level overlap remains UNKNOWN because no approved non-sensitive denylist was present. No protected file was opened.
- The D4B audit's 4 within-pilot dHash pairs should be clustered for any D4B diagnostic and excluded together from D-conf. Do not treat different dataset names as proof that all near-duplicates are absent.

## Attribute, defect, mask and context supervision

The pilot annotations provide a single image/view-level surface_abrasion_visible state, confidence, optional point and free-text note. They do not provide complete attribute masks or a reliable attribute-to-region segmentation. Uncertain, not applicable and unqueried remain distinct; only Present and Absent assessable enter a binary attribute metric.

Official normal/defect/source-type metadata is evaluator-only. It may stratify results after predictions; it is not an attribute target or learner input. A combined anomaly mask, if separately authorized later, would not identify which region belongs to abrasion. No masks were opened here. Product context may be declared as a visible inference-time field, but product identity cannot be used as a hidden answer.

## Existing feature substrate and code interfaces

Static inspection only:

- cache/realiad_dino_source_normal_compact metadata declares 36,465 train-normal records over 30 categories, 285 shards and 64 sampled patch features per image. Its purpose is Stage-1 source-normal factorizer training; it excludes defects and is not a cache for the 54 CONVAD pilot images or the 48 Real-IAD V2 annotated images.
- The cache writer schema exposes patch_features, global_features, patch_indices, grid_hw and feature_dim. The manifest provides image/category/split identity. The code therefore has a global/patch feature interface with recoverable sampled patch positions, but the tensor shards and actual feature_dim were not read. Full spatial coverage of every patch is not present in this compact 64-patch cache.
- scripts/abmg_stage0_foundation.py describes full frozen patch and global feature payloads, but this is not a D6 attribute cache. No V2 item-to-feature cache was found in the permitted static inspection.
- No D6-specific Logistic/Replay, Recursive Ridge/CONCIL, Context-aware Attribute Prototype or CB2M protocol implementation exists in the repository. Existing sklearn logistic code is for factorization audits; existing prototypes are for ABMG/factor/anomaly diagnostics, not a matched surface-abrasion learner. They are possible engineering components, not completed baselines.
- requirements-stage0.txt lists numpy, pandas, Pillow, scikit-learn and torch without pins. The current Python is 3.11.9; installed distribution metadata reports numpy 1.26.4, pandas 1.5.3, Pillow 12.0.0, scikit-learn 1.3.0 and torch NOT_INSTALLED. No dependency was installed. DINO extraction and loading these .pt shards are blocked in this interpreter.

## Feasibility judgment

**Exploratory D-dev replay: conditionally supportable, not authorized yet.** A bounded candidate is D4-Lite/D4B's 54 previously labeled CONVAD images as initial D-fit and the Real-IAD V2's 24 sample-key groups as development feedback/future candidates. These sources have no exact path/SHA overlap, and the Real-IAD pilot has positive and assessable-negative views across all six selected products. A same-product group holdout can be attempted with a fixed group-level split; a cross-product diagnostic can reserve complete products for feedback versus future evaluation. The group split, event order and exact product roles are not frozen, and all pilot labels are already visible to the research team. Any result would be exploratory development evidence, not a blinded test.

If a three-product/three-product product split were selected, the current V2 pool supplies at most 12 operational groups on each side. That is an upper-bound planning count, not a selected split or an independent-object count. A within-product two/two group split supplies only four operational groups per product in total. Per-product metrics may be undefined in a particular deterministic split if one side lacks positive or negative labels; do not choose the split after seeing model outputs.

**Formal headroom and confirmation: INCONCLUSIVE / not ready.** Twenty-four groups and six products are too few for stable product- and group-level uncertainty estimates; physical independence is INFERRED; source types and pilot selection are enriched; timing does not measure paid feedback cost; no D6 features exist for the annotations; and no authorized D-conf partition is available. Attribute BA/AP can be defined for eligible development items but cannot yet support a confirmatory claim. Detection/TBA readout is NOT_READY: no fixed authorized industrial readout, verified attribute-region masks or approved evaluator-only mask access were established.

Minimum bounded supplement before asking for D6-P1 GO:

1. Researcher signs a D-fit/D-dev assignment and label-independent group/product split, including the treatment of the 54 CONVAD and 24 Real-IAD groups. Mark all as exposed development data.
2. Separate authorization covers feature extraction or a compatible existing feature source for at most the 54 D-fit and 48 selected D-dev images; no masks, other Real-IAD views or sealed targets are needed for the attribute-only screen.
3. Use an already compatible runtime or report the torch blocker. Do not install dependencies under this P0.
4. Pre-register the feedback tape, cost sensitivity, primary attribute metric and evaluator separation. Existing elapsed_seconds cannot be used as actual cost.
5. Before any confirmatory claim, obtain a separately authorized, group-disjoint D-conf source or product batch with blinded human attribute labels and a non-sensitive overlap procedure. This is not part of the current P1 screen.

## Source checksums

SHA-256 values are for the exact local audit inputs, not for images themselves.

| Relative source path | SHA-256 |
|---|---|
| docs/Last_Paper_Four_Month_Research_Execution_Plan_2026-11_to_2027-02_v1_1.md | 1a3b83fa46dcc315963ad2a17deb019a6d01cc596193f5de067db13723fd0844 |
| docs/Last_Paper_D6_Research_Handoff_2026-11-12.docx | a26221d118c6b1d1d643181d28ac5f4ce6e83c707b57b606ba3b8a59a86cf1b5 |
| research_pilot/LP_W1_D4LITE/annotations.csv | 004e616ac4fcc8ca1a25c89c9a1a70b8ef52d0497410002ecbaee2ac06033fa0 |
| research_pilot/LP_W1_D4LITE/pilot_manifest.csv | 08deb61712e952c93dc41ae5c169bd1c9697b9fd43f3365ffe1986dd9b83871d |
| research_pilot/LP_W1_D4B/annotations.csv | e3b35d25ef5fef4bae130dd9fb99a4a77c17fa88410ac56d149fa54adfe464d1 |
| research_pilot/LP_W1_D4B/source_manifest.csv | 0fcbbf35fd733cea75b4029d60d156042a0f39d2e3df5950cfcb1886d6814925 |
| research_pilot/LP_W1_D5R_SOURCE_METADATA/P1_Source_Metadata_Exposure_Ledger.csv | d16dfb81f414b5197e80e955b50e636cc2eb69f5b39a3fdb5bad1fa3486f8c90 |
| research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/annotations.csv | 1d138ff1a420d6134ad7fc1f2830074cc3d41fda5a8d789a81dd55b0f0e4d9a4 |
| research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/annotation_queue.csv | 23f3d1084006fd4f1e5f221225ec77862be3facc8713e3bf3592ce448ff3476e |
| research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/selection_manifest.csv | c0def1f598d38ea68c78da028958f7ee030a7bdf21528d5a249fe8f8a6f81b7f |
| research_pilot/LP_W1_D5R_IMAGE_PILOT_V2_2V/exposure_ledger.csv | a274bbbdae3a1e4c1822c2968458d9c3878a1f466e82177632be1c5b449d3306 |
