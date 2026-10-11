# Patch-level E3 and E4 mechanism validation (v2)

This is a new **source-side mechanism-validation protocol**. Earlier E3/E4 files and failed gates remain unchanged. It is NOT a deployment stress result.

## E3: patch normality
- The frozen Sensor evidence cache MUST include ALL 784 patches (28 x 28). A 64-patch shortlist is rejected.
- Source-CV development training normals define empirical prior moments; evaluation products use two normal supports, high-surprise verified-normal query examples, and disjoint test normal/defect sentinels.
- Compare deterministic shrinkage diagonal Gaussian against diagonal Normal-Inverse-Gamma (Student-t) **per patch and per group**, aggregating surprise AFTER patch scoring.
- All compared models see identical supports and query examples. Evaluate future sentinel normal surprise, defect surprise and AUROC at 0, 4, 8, 16 and 32 updates where available.

## E4A: factor-addressed normal memory
Hold the NIG model fixed; compare no-update, mass-matched broadcast (1/8 to each address), predicted-surprise selective (one-hot), and deranged wrong-address credit (one-hot). All branches see identical verified normal queries. Updates have one total effective sample mass per feedback image, distributed across selected patches and factor addresses. Record per-address mass and future sentinel metrics.

This experiment deliberately makes informative feedback available to validate the mechanism. The known-normal high-surprise query schedule is **offline-only** and not a deployable selection policy. It is not a claim of proof of C2/C4. A score drop in both normals and defects is not automatically an anomaly detection improvement.

## Representation limits
The eight disjoint DINO raw-coordinate groups are operational addresses, NOT demonstrated causal or disentangled factors. Patch order in the cache is sorted by frozen anomaly score, not image position, so this audits patchwise density but not spatially indexed localization. The fractional intra-image NIG update is a composite-likelihood approximation for correlated patches. There is no proof of independent samples within an image.

## Single-line Windows PowerShell sequence
1. On the user's GPU, extract full patches (several GB possible):

python scripts/abmg_e4_extract_evidence.py --fold 0 --root "YOUR_REALIAD_ROOT" --json-dir "YOUR_REALIAD_JSON_DIR" --candidates 784 --out-dir outputs/patch_evidence --device cuda

2. Run E3 source-CV fold zero:

python scripts/abmg_patch_e3_e4_v2.py --mode e3 --fold 0 --source-cv-fold 0 --evidence outputs/patch_evidence/fold_0/e4_evidence.pt --budget 16

3. Inspect E3 before E4A:

python scripts/abmg_patch_e3_e4_v2.py --mode e4a --fold 0 --source-cv-fold 0 --evidence outputs/patch_evidence/fold_0/e4_evidence.pt --budget 16

Results: outputs/patch_e3_e4_v2/fold_0/cv_0/patch_e3_audit.json and patch_e4a_audit.json

## Safeguards and future work
Only source categories participate; the outer-target products are not loaded or evaluated. Development priors use TRAIN normals, while test-normal and defect sentinels never write. Query identities are offline-known normals, never sentinel observations. E4B source-enriched defect credit is **a separate subsequent experiment**, not validated by this code. After checking this initial CV fold, test four source-CV folds and seeds before any target evaluation.
