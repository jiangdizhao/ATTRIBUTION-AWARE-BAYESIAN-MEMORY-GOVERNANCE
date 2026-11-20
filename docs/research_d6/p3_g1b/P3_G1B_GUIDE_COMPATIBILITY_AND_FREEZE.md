# P3-G1B Guide Compatibility and Freeze Record

**Decision:** `NO_MATERIAL_CONFLICT_FOUND`  
**Guide status:** frozen as the operational interpretation accompanying Attribute Definition v1.1  
**Freeze date:** 2026-11-21  
**Real G1B annotation:** not authorized or started

## Frozen artifact

The researcher-supplied guide was copied verbatim to [P3_G1B_ATTRIBUTE_ANNOTATION_GUIDE_v1_1.md](P3_G1B_ATTRIBUTE_ANNOTATION_GUIDE_v1_1.md) and to the Git-ignored G1B private run directory. The original “操作性解释草案” heading is retained; this freeze does not rename or replace the historical Attribute Definition v1.1.

- Guide SHA-256: `7b75bfeda392523655fffc72a93ad74ff178c05836dd96b66e9d08dbe750549e`
- Historical Attribute Definition v1.1 SHA-256: `41aed5ae2a8c59d82ff222085fa816808a1c0504a393d257c8ba2c28a5678367`
- G1B input-freeze receipt SHA-256 (unchanged): `dfab27d75bd6460f9a11bd50714d1f088946335fcf4c32436ecd9e2c2c27c100`
- A supplemental private protocol addendum binds the guide hash to that receipt. The original G1B receipt and queue are unchanged.

## Compatibility review

The comparison used the frozen v1.1 definition card and the already archived D4B final quality audit. It did not read the D4-Lite or D4B raw annotation CSVs or open any image. No G1B image, label, D-eval truth, mask, or model output was read.

| Guide statement | Existing v1.1 rule and prior audit evidence | Finding |
|---|---|---|
| A clear, non-design shallow line or localized scuff can be Present, even if only one line is visible. | v1.1 includes scoring and shallow linear wear, but requires clear, localized surface evidence beyond ordinary texture/geometry. | Compatible. “One line” is not an automatic positive; the existing clarity and surface-evidence threshold still applies. |
| Crack, pit, contamination, missing part, or breakage alone is not Present; separate abrasion evidence may coexist. | v1.1 excludes holes, cracks, punctures, deformation, residue, contamination, isolated color change, and missing components unless separate abrasion-like evidence is clear. | Compatible and non-exclusive across defect types. |
| Reflection, shadow, exposure, blur, or an ambiguous line may be Uncertain; if enough exposed surface is assessable and no abrasion is visible, use Absent. | v1.1 defines the same distinction between `unobservable_or_uncertain` and `absent_assessable`. D4B’s only repeat disagreement was Absent versus Uncertain with a reflection-related note; no Present/Absent repeat conflict was reported. | Compatible; the guide makes the prior boundary explicit. |
| `not_applicable` means no relevant exposed surface/object, not blur or hidden surface. | v1.1 says no exposed surface makes the attribute inapplicable; hidden, cropped, or poorly resolved surfaces are uncertain/unobservable. | Compatible. |
| Each view and blind repeat is judged independently; source labels and other-view answers cannot replace the visible evidence. | v1.1 prohibits inference from source split, anomaly/defect labels, masks, VLM scores, or other images. The G1B queue assigns independent presentation IDs and blank repeat states. | Compatible with the frozen G1B protocol. |
| Notes may record likely confusions such as pit, crack, glare, or blur. | v1.1 permits notes and does not make the appearance-only attribute a mutually exclusive defect taxonomy. | Compatible; notes remain descriptive and are not extra labels. |

The prior D4B audit also reported some human Present labels in source strata named `bent` and `hole`, consistent with the guide’s rule that abrasion can coexist with another defect. The archived aggregate cannot establish whether those individual visual judgments were correct, and this review makes no such claim.

### Narrow interpretation guard

The example “塑料表面出现方向一致的局部磨花纹理” is subordinate to the v1.1 evidence threshold. Directional texture by itself, regular processing lines, seams, mold marks, or repeating texture do not establish abrasion. The local appearance must still clearly show a disturbed/rubbed surface finish beyond ordinary structure. This records the existing v1.1 boundary; it does not revise the guide or create a product-specific rule.

## Freeze and use boundary

The supplied wording is frozen as an operational explanation, not as a new attribute definition version. No substantive conflict requiring a split or relabel was found. Use it together with the historical v1.1 card; do not change either after the first G1B presentation. If a later change is needed, issue a separately versioned protocol amendment before any new confirmation labels are viewed.

The G1B annotation launcher remains unauthorized. The D-cal and D-eval templates remain empty, and the guide freeze does not grant annotation, model-training, D6-evaluation, or D-conf permission.
