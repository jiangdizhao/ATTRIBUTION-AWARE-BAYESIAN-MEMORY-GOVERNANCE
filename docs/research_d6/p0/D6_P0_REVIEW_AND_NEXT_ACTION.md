# D6-P0 Review and Next Action

- **P0 completion:** DOCUMENTATION_READY
- **Data/headroom finding:** INCONCLUSIVE
- **D6-P1:** NOT_AUTHORIZED

## What was checked

The v1.1 plan, 2026-11-12 D6 handoff, current Stage-3 handoff, D5R source protocol, D4/D4B pilot records, V2 annotation queue/manifest/exposure ledger, existing cache metadata and relevant baseline code interfaces were reviewed. The local branch and origin branch both resolve to 4f265d89a9a29b871d0f1ff5c0f66b35a5ad3c37.

## Independently recalculated

- D4-Lite: 44 presentations, 36 unique images, 18 Present / 17 Absent / 1 Uncertain, 5/8 repeated labels agree.
- D4B: 26 presentations, 18 unique images, 9 Present / 8 Absent / 1 Uncertain, 7/8 repeated labels agree. Six scratch-stratum images were Present and six good-stratum images Absent.
- D5R-P2 V2: 58/58 valid presentations, 48 unique images, 24 sample-key groups, 6 products × 4 groups, 10 repeats and 8/10 agreement. First labels are 25/19/4/0; all 25 first Present items have a point.
- The 120-row Real-IAD exposure ledger is 120 unique Real-IAD test-origin images. Forty-eight are V2 annotated; the other 72 remain exposed. P1's 18 metadata-ledger rows sum to 90,515 records and 18,103 sample-key groups.
- Exact recorded SHA overlap is zero between the 54 CONVAD pilot images and the 120 Real-IAD exposures. D4B has four recorded dHash near-pairs at distance ≤4.

## Inherited or unresolved

Physical-object independence remains INFERRED for Real-IAD sample keys and UNKNOWN for the two CONVAD pilots. Historical sealed/HOLD item-level overlap and cross-dataset near-duplicate status remain UNKNOWN. V2 elapsed_seconds has a 43.3-second median, 159.5-second mean and 3,986.29-second maximum; it is not valid actual annotation-cost evidence.

The D4-Lite folder's 26-row annotations_export.csv uses D4B presentation IDs and is not part of the D4-Lite raw record. It matches D4B annotations on 25 complete rows and differs on timing fields in one row. It was not altered or merged.

## Three highest-impact issues

1. Only 24 Real-IAD sample-key groups and six products are manually labeled; physical independence is not verified and intervals at product/group level will be weak.
2. All D4/D4B and Real-IAD pilot labels are researcher-visible development exposures. No independent D-conf source is identified or authorized.
3. The existing cache has train-normal features, not features for the 54 D-fit or 48 D-dev annotated images. Current Python 3.11.9 lacks torch, and no matched D6 B0–B4 runners exist.

## Readiness

The existing annotations can support a bounded development replay design: candidate D-fit is the 54 previously labeled CONVAD images; candidate D-dev is the 24 Real-IAD groups. The six Real-IAD products each contain both Present and Absent view labels in the complete pilot. A label-independent group split could support exploratory same-product and whole-product-held-out diagnostics after a separate P1 authorization.

That is not enough to claim statistical headroom or independent generalization. No features were generated, no method was run, H2 task-readout headroom lacks an authorized fixed detector/mask evaluation, and P1 remains unauthorized. The v1.1 candidate margins δ_task=0.02 and ε_FN=0.01 are not frozen.

## One recommended next action

Researcher review and, if desired, issue a separate bounded D6-P1 authorization for an attribute-only development screen using at most 54 CONVAD D-fit images and 48 already selected Real-IAD D-dev images. The authorization should first freeze group/product roles, event tape, label visibility, assumed-cost sensitivity, metrics, runtime/feature source and resource ceiling. It must explicitly preserve the no-D-conf status and must not include mask access or sealed data. If the current machine is the runtime, resolve the missing-torch constraint through an already available compatible environment or keep feature extraction BLOCKED; do not install dependencies as part of this P0.

No P1/P2/P3 work was started, no historical file or annotation was edited, and no Git commit or push was made.
