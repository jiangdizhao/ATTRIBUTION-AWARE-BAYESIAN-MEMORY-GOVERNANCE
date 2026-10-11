# E3 v3 — Multimodal patch-local normal-memory audit

This is a **new E3 mechanism-validation experiment**, not a revision of the previously recorded E3 v2 result. Existing E3 v2 and E4A/B scripts are intentionally unchanged.

## What changed

**v2:** Eight coordinate groups, but one diagonal normality distribution per coordinate group; updates replace patches by pooled sufficient statistics. Even though patches are scored individually, all unrelated appearance modes share one normal distribution.

**v3:** Every one of the **784** cached DINO patch descriptors receives its own anomaly surprise before image-level top-8 pooling. A product starts with **two support-normal images**, from which fixed cosine farthest-first feature anchors form an optional **8-mode** normal-memory bank. Each appearance mode has independent online Gaussian or Normal-Inverse-Gamma sufficient statistics. A new verified-normal image updates only appearance modes matched by its selected high-surprise patch descriptors. The frozen support-derived keys are shared across model branches.

Four controls use identical supports and feedback: global Gaussian (one mode), global NIG (one mode), local Gaussian (eight modes), and local NIG (eight modes). The global controls distinguish the benefit of local/multimodal memory from the benefit of Bayesian uncertainty.

Source-CV development **training normals only** define common empirical prior moments. Held-out product **training normals** supply supports and candidate feedback. Half their TEST normals may be used for offline known-normal feedback; the other half form disjoint sentinel test-normal images; defect test images are evaluation-only sentinels. All feedback events and selected patches are fixed once using the common cold-start local NIG and reused across models. Each support or feedback image contributes at most 1 effective update unit across all its selected patches and modes. NIG is a composite-likelihood approximation because patches within an image are correlated.

**Important limitations:** Fixed appearance modes are not discovered semantic factors. Cached patches were sorted by frozen anomaly score, so this is **appearance-local patch density, not spatial-grid conditional NIG**; original 2D patch positions cannot be reconstructed from this cache. No GT mask enters the scorer. Controlled queried normals are an offline validation device, not a deployable query policy. We do not expect gains automatically; examine both normal/defect surprise and their separate score changes alongside AUROC.

## How to run (Windows PowerShell, one-line commands)

1. Synchronize branch:

git pull --ff-only origin stage2-patch-e3-e4-mechanism-validation

2. Run targeted tests:

python -m unittest discover -s tests -p "test_patch_e3_local_modes_v3.py" -v

3. First source-CV fold only, reuse previous full-patch evidence, no DINO extraction:

python scripts/abmg_patch_e3_local_modes_v3.py --fold 0 --source-cv-fold 0 --evidence outputs/patch_evidence/fold_0/e4_evidence.pt --support 2 --modes 8 --budget 16 --patch-write 32

4. Inspect output and send the JSON before running E4:

outputs/patch_e3_v3/fold_0/cv_0/patch_e3_local_modes_audit.json

Outputs include per-category AUROC, normal/defect surprise, paired changes from initial memory and local-mode effective mass at checkpoints 0,4,8,16. After the first fold is evaluated, further source-CV folds and multiple seeds are required before any general conclusion. Outer target categories remain sealed.

Do not proceed to existing E4A/E4B without explicitly deciding whether and how their memory APIs should adopt the new local-mode design.
