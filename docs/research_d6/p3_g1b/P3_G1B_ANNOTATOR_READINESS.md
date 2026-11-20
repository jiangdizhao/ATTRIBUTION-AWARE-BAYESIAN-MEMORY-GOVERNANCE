# P3-G1B Annotator Readiness

**Status:** `P3_G1B_ANNOTATOR_PREP_READY`  
**Scope:** annotation software, frozen queue, and synthetic-only verification. This report does not authorize a real annotation session.

## Input identity and exposure

- Branch / HEAD: `stage3-eg-famar-execution` / `8108ba33c3107224e1aa7bb203a584132b5d9241`.
- G1A `development_roles.csv` SHA-256 matched the registered value `7e402d017ac6795f0a5c4ed6bf7493244e872e6e288fd88a3f11ecdf031be443`.
- G1A status, role associations, product slots, view membership, exposure ledger, source SHA records, and registered selection/score hashes passed metadata checks. G1A source paths and SHA values are frozen in the ignored private run directory.
- G1A had already source-read and SHA-verified its selected candidates. G1B preparation read metadata and manifests only: **0 image bytes opened** and no attribute labels, defect labels, masks, or model outputs were read as annotation answers.
- All 120 selected identities remain `NEW_STUDY_DEV_EXPOSED`. Historical sealed/HOLD file-level overlap remains `UNKNOWN`; these items and identifiable near duplicates must be excluded from future D-conf.
- The old GUI SHA-256 matched the expected value `00059f52334eae9e49d476d380130a5df2ba2d5e77c4c9d87cfec2eed2853b05`. It was not modified or run. The new adapter reuses its four-state, confidence, optional-point, navigation, and atomic-save behavior while supporting G1A’s queue and fail-closed source checks.

## Frozen queue

| Check | Result |
|---|---:|
| Unique selected views | 120 |
| Operational sample-key groups | 60 |
| D-cal | 40 views / 20 groups / 2 products |
| D-eval | 80 views / 40 groups / 4 products |
| First presentations | 120 |
| Blind-repeat groups | 12 |
| Blind-repeat presentations | 24 |
| Total queue | 144 |
| Duplicate image IDs | 0 |
| Minimum first-to-repeat separation | 6 queue positions |

The repeated groups are the first 12 under the G0 `REPEAT_GROUP` SHA-256 ordering over `product + U+001F + opaque_group_id`; no role balancing was applied. The resulting repeat allocation is 9 D-cal groups and 3 D-eval groups (by product slot: 4, 5, 1, 1, 0, 1). Every selected group stays within one role, and each repeated view receives an independent presentation ID and blank annotation state.

Frozen queue SHA-256: `95263d09fd7eb9e676e0d5f96fc6be491a27b10113f650e2d8ff9966900e6703`  
Blind-repeat mapping SHA-256: `ad1fc983d3fd5429d43291e6b24e3f0f60d4cfe558186157fcb7da376fc4b186`  
Input-freeze receipt SHA-256: `dfab27d75bd6460f9a11bd50714d1f088946335fcf4c32436ecd9e2c2c27c100`

## Annotation controls

- The UI presents only the image, its queue position, the frozen `surface_abrasion_visible` v1.1 guidance, four state choices, confidence, optional point, and notes. Product, sample-key, view, source label/type, filename/path, source split, H12 score, anchor, stress bin, and repeat linkage are hidden.
- Blank remains blank. The four stored states are `present`, `absent_assessable`, `unobservable_or_uncertain`, and `not_applicable`; no state is copied across views or blind repeats.
- Image IDs must belong to the frozen 120 and match the frozen role mapping. Before any real display, the approved launcher verifies the G1A role-file hash, queue hash, ID-role associations, and all registered source SHA values through G1A’s exact-path secure reader. It rechecks the image SHA when displaying each image.
- D-cal and D-eval CSVs are separate. The D-cal loader accepts only the allowlisted D-cal CSV. A single-writer OS lock, atomic replacement, timing state, and correction history protect saved work.
- The private run directory is Git-ignored and its Windows ACL grants access to the current account, SYSTEM, and Administrators. Both annotation CSVs contain headers only. No `annotation_authorization.json` exists; the private template says `NOT_AUTHORIZED`.

## Verification

- Python syntax compilation: passed.
- `python -m unittest tests.research_d6.test_p3_g1b_annotator -v`: **25 passed**.
- `python -m scripts.research_d6.p3_g1b.annotate --self-test`: passed in a separate process using two generated synthetic images; Tk 8.6 rendered the canvas and temporary CSV save/close/reopen/restore succeeded.
- Synthetic coverage includes 120/144 queue structure, G0 repeat selection, role separation, blank and all four states, independent repeat state, EXIF orientation, hash mismatch rejection, output-path guard, concurrent writer rejection, failed atomic replacement, D-eval rejection by the D-cal loader, and hidden GUI metadata.
- Real-image GUI startup and live annotation were deliberately **not** run. They remain gated on a separate research-lead authorization file.

Runtime used for preparation and GUI smoke test: `D:\anaconda3\python.exe`, Python 3.11.9, Tk 8.6, Pillow 12.0.0.

## Start command after separate authorization

From the repository root, after the research lead separately authorizes the session and creates the private `annotation_authorization.json` matching the freeze receipt:

```powershell
Set-Location E:\ATTRIBUTION-AWARE-BAYESIAN-MEMORY-GOVERNANCE
D:\anaconda3\python.exe -m scripts.research_d6.p3_g1b.annotate --run-dir outputs/research_d6/p3_g1b/run_20261121_01
```

Without that approval file, the launcher stops before reading any Real-IAD image bytes. This preparation does not authorize model training, D6 evaluation, or D-conf access.

## Operational guide freeze

Before any later authorized first presentation, the annotator must read the frozen [P3-G1B operational interpretation guide](P3_G1B_ATTRIBUTE_ANNOTATION_GUIDE_v1_1.md). Compatibility review found no material conflict with the historical v1.1 definition or archived D4B audit; the guide SHA-256 and decision are recorded in [P3_G1B_GUIDE_COMPATIBILITY_AND_FREEZE.md](P3_G1B_GUIDE_COMPATIBILITY_AND_FREEZE.md) and the private protocol addendum. The original definition card and input-freeze receipt remain unchanged.
