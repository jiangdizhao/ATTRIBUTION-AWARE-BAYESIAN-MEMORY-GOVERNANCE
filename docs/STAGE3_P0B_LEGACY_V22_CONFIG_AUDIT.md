# P0-B — Legacy v22 Configuration Recovery and Evidence Audit

**Date:** 2026-10-23  
**Status:** historical command RECOVERED, execution and numeric parity UNVERIFIED.  
**Machine-readable original argument list:** [legacy_v22_recorded_cli_v1.json](../configs/stage3/legacy_v22_recorded_cli_v1.json)  
**Provenance:** user-supplied command.txt line 191 (uploaded in chat, not copied into this Git tree).

## 1. Primary v22 anchor command (line 191)

The record describes a full v22 / v21-equivalent Real-IAD batch with all 30 product names, three support seeds (0,1,2), and one stream seed (0). If every category/seed combination ran successfully that is 90 expected runs, NOT a proven result. The machine-readable profile preserves 73 explicit argument tokens, excludes shell tee/redirection, and tags unverified claims.

| Component | Explicit historical command settings |
|---|---|
| Sensor | baseline_compat; feature_fuse=concat; layer_fusion=mean; view_fusion=mean; img_agg=max; no_rotations |
| Support | K_modes=1; shots_per_mode=4; aug_per_image=4; batch_support_seeds=0,1,2 |
| Fixed Normal | K_normal_fixed=25000; fixed_per_img_cap=5000 |
| STM/LTM | K_normal_dyn=1024; K_normal_ltm=512; ltm_min_stm_size=16; ltm_min_repeat=2; ltm_retrieval_k=1 |
| Maturation / retention | enable_maturation_buffer; enable_maturation_buffer_retrieval; enable_utility_aware_stm_retention; stm_retention_replace_margin=0.03 |
| Correction | ablation=normal_dyn_plus_correction; K_correction=1024; corr_weight=1.0 |
| Active querying | policy=simple_aif; simple_aif_use_local_saliency; simple_aif_query_cost=0.8; simple_aif_beta_entropy=0.3; simple_aif_target_qrate=0.03 |
| Operating threshold | theta_controller=guarded_dual_anchor |
| Stream/inference | no_set_membership; save_temporal_artifacts; debug_defect_writes; experiment_mode=batch |

**Support selection correction:** K_modes=1 alone does NOT activate clustering; the command omits the independent boolean flag --use_modes. The v22 source therefore uses _sample_paths_deterministic_or_seeded from the non-mode branch: without support_seed it takes the first four sorted normal-train paths; with --batch_support_seeds 0,1,2 it draws seeded random sets of four from eligible normal-train images. These are not automatically mode-central support inliers. For paired Stage-3 mechanisms freeze actual class-by-seed Support IDs and the coupled initial theta, not merely the seed number. Preselection may access a large normal-train pool; do not claim a strict only-four-predeployment-images setting.

**Default-vs-recorded distinction:** v22 copied in this Git tree has default normal/defect/recent reservoir capacities 512/512/96; updates every 8 labeled events, quantiles 0.95/0.10, guards 0.01/0.01, and eta 0.3. The historical command did not explicitly override those values. Treat these as current-code defaults, not verified values of the original executable. Algorithmic simple_aif_query_cost 0.8 is NOT 0.8 human seconds. --no_rotations disables query pseudo-rotation, not support augmentation. The original CLI uses --csv for Real-IAD JSON directory, unlike E2's --json-dir.

## 2. Other command.txt entries: different hypotheses, NOT primary anchor

| Source line | Purpose | Limitation |
|---|---|---|
| 184 | Four-class v21-equivalent panel | not all 30 classes |
| 188 | legacy selector with support guard and warmup | bundled v22 additions; uses PowerShell variable $classes30 |
| 195 | coverage-tail selector with threshold quantile 0.75 | support guard/warmup also active; confounded |
| 200 | stream order seeds 0,1,2 | distinct from support-seed robustness |
| 204 | AIF target qrate 0.10 | misleading q001 run-tag text |
| 207 | hardcode ambiguity band 0.3–0.7 | different query controller |
| 220 | deployment-cost and hidden FN audits | separate diagnostics |
| 223 | selftrain_core without human queries | different supervision regime |

## 3. P0-A data evidence

User locally ran Real-IAD inventory with --verify-files at G:/dataset/realiad and G:/dataset/realiad/jsons/realiad_jsons. It reports 30 products, 8 type codes, 36,465 train Normal; 63,256 test Normal; 51,329 test Defect; no warnings; 51,329 existing declared mask files. This is a user-local file-existence audit, NOT a checksum of all images, mask pixel content test or validation of geometric registration. Test-stream defect prevalence: 51,329 / 114,585 ≈ 44.80%; overall Train+Test ≈ 33.98%.

## 4. Correct stage-scoped evidence gates

- **Before G0 smoke:** recorded CLI consistency, local JSON index digest and frozen model/preprocessing identity, plus separate explicit user stage approval. G0 output traces and Support IDs are generated during G0 and must NOT gate their own creation.
- **Before G1:** actual G0 scores, support paths, theta0, query IDs, sidecar evolution, store capacities and exact logged discrepancy/acceptance.
- **Before G2 Rich:** Mask decode/foreground/size checks and purchased-feedback authorization; dataset-wide masks remain evaluator-only until purchased.
- **Before G4:** source-measured or preregistered Binary and incremental Rich annotation times; match total human minutes.
- **Before final:** pre-registered source-only risk thresholds, utility criteria and compute/storage budgets.

P0 structural lint and --strict G0-smoke readiness are distinct. Neither constitutes permission to run G0; stage approval is required.

## 5. Next local read-only command (PowerShell, one line)

python scripts/stage3_p0b_local_preflight.py --root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --inventory "outputs/stage3_p0/inventory/realiad_inventory.json" --out "outputs/stage3_p0/p0b_local_provenance.json"

This command hashes the 30 JSON metadata files in stable filename+length+bytes order, audits the existing P0-A inventory and verifies local v22/VisionAD Git blob identity. It does not hash all image pixels, evaluate masks, load a model or run a baseline. Optional --checkpoint PATH hashes an explicitly identified weight file but does not by itself prove v22 loaded that file.

The old E:/ output location from command.txt is only historical; do not reuse it for new experiments. Use outputs/stage3_g0/... once G0 is separately approved.
