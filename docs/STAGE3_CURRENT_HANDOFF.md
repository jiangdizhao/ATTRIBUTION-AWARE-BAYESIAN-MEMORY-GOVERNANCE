# Stage 3 EG-FAMAR — Current Research Handoff

**Updated:** 2026-10-23  
**Branch:** `stage3-eg-famar-execution`  
**Base Stage-2 HEAD:** `3b772a0737cce8d1f11897db7064441a476e9cf1`  
**Stage-3 status:** P0-A inventory audit completed locally, P0-B historical command recovered; G0/G1/G2/G3/G4 have NOT been run or validated.

**Canonical Stage-3 execution authority:** [STAGE3_EG_FAMAR_RESEARCH_EXECUTION_PLAN.md](STAGE3_EG_FAMAR_RESEARCH_EXECUTION_PLAN.md). The 2026-10-22 EG-FAMAR PDF is the latest scientific proposal; if older 2026-10-16/20 handoffs conflict, use 10-22 staging/numbering and document any adjustment explicitly.

## Research identity

Frozen DINOv2 + primary open-set Normal NN + governed bounded mutable memory and score correction. The question is whether evidence-authorized, locally addressed writing AND compatible reading give persistent future utility with less off-address interference and without absorbing unknown defects. Scheme B is active; Scheme C human-grounded schema transfer remains optional/unvalidated.

## Decisions from 2026-10-23 review

1. **Support matters**: historical v22 command line 191 specifies `--K_modes 1` but omits `--use_modes`; so there is **no K-means mode-central selection**. Unseeded legacy non-mode selection takes the first four sorted normal-train paths; `--batch_support_seeds 0,1,2` instead requests seeded random four-shot draws. Freeze actual four IDs **per class and support seed**, eligible normal-train pool, augmentation, immutable bank and theta0. Never compare paired methods with different IDs. Coverage-tail is a separately confounded exploratory condition, NOT the mainline baseline.
2. **Sidecar matters**: retain the full legacy guarded dual-anchor controller; previous ablations show a real FPR/FNR trade-off, not guaranteed accuracy improvement. Use fixed-threshold/fixed-query replay for memory-causal attribution and normal live Sidecar for end-to-end operating results.
3. **Only two feedback tiers initially**: B = binary Normal/Defect; R = after verified Defect, additional *Type + Mask* with explicit incremental time/cost. R requested on Normal remains normal-only; do not fabricate a mask. No Point/Scribble/Box workflow in G1-G3. Four feature-hiding views of already-paid R evidence are permitted for *offline* contribution ablations, not distinct human-query tiers.
4. **Source discipline**: the previous paper's DINO/features/scoring/aggregation might differ from later Stage-0 defaults. Verify actual commands/preprocessing/checkpoints and `test_phase3_AIF_simplified_v22.py` before G0 parity claims.
5. **No leakage**: prequential `score→predict→query→reveal→update`. Hidden target GT/type/mask only for external evaluator, not memory, router or controller. Image-level binary FN does not identify defective patches.
6. **Stage-gated research**: P0 first; G0 only after user approval. Do not batch-implement G1-G4. Simple baselines must be allowed to win; don't force factorization, NIG or AIF.

## P0 implementation checkpoint (2026-10-23)

**P0 draft files committed; NOT YET REVIEW-CLOSED / G0 NOT AUTHORIZED.** No legacy inference or full G0 replay was run.

- `configs/stage3/experiment_manifest_v1.json` — source-grounded manifest, legacy paper vs argparse differences and explicit unresolved readiness fields.
- `configs/stage3/source_cv_v1.json` — all 5 sealed outer folds and 20 explicit 18/6 source-CV groups.
- `docs/STAGE3_CANONICAL_MEMORY_IO_V1.md` — immutable sensor, mutable state, router, governance, sidecar, feedback, replay I/O and invariants.
- `docs/STAGE3_FEEDBACK_AUTHORITY_AND_COST_V1.md` — B/R authority, paid rich upgrade, field-hiding ablations and missing-mask behavior.
- `scripts/stage3_p0_validate.py` — stdlib-only, read-only manifest/fold/source-CV checks and preliminary agent-view event visibility lint.
- `tests/test_stage3_p0_contract.py` — synthetic validation tests. `.github/workflows/stage3-p0.yml` is a narrow CI check (when GitHub Actions is enabled).
- Existing v22/visionad/inventory/Stage-2 experiment code remains unchanged. Do NOT substitute the uploaded local copy of v22 for the exact Git blob until its differences are audited.

The historical v22 primary CLI has now been recovered from uploaded command.txt **line 191**, archived as `configs/stage3/legacy_v22_recorded_cli_v1.json`; see [P0-B audit](STAGE3_P0B_LEGACY_V22_CONFIG_AUDIT.md). The historical CLI's Fixed 25,000, Correction 1,024, Max image aggregation and 3% query target differ from the current v22 defaults 4,096 / 256 / topk_mean / 6%. The recovered command is **not proof that a prior run succeeded** or that current code shares the old historical source SHA.

Run after local sync:
```powershell
python -m unittest discover -s tests -p "test_stage3_p0*.py" -v
python scripts/stage3_p0_validate.py
python scripts/stage3_p0_validate.py --strict
```
The first two should pass; `--strict` is designed to **fail** until local JSON index and model/preprocessing fingerprints are reconciled and the G0-smoke gate is explicitly reviewed. The validator now separates future G1/G2/G4/final evidence from G0-smoke blockers. A structural pass does not mean G0 ready.

**Remaining evidence by stage:** Before G0 smoke, local JSON-index checksum and actually loaded frozen DINO/preprocessing identity. G0 will generate support-image IDs, theta0, item-level scores/query traces and actual memory/Sidecar capacities; these do not circularly block G0. Rich Mask pixel validation belongs before G2; B/R annotation time before G4; source-frozen safety/cost bounds before final comparison. Historical run-result corroboration is still outstanding.

## P0-B checkpoint (2026-10-23) — documentation+preflight, NOT a model run

- P0-A local user inventory: 30 products, 8 defect codes, 36,465 train Normal, 63,256 test Normal, 51,329 test Defect; zero file/path warnings, declared mask paths exist. No mask pixel validation or full image hash.
- `configs/stage3/legacy_v22_recorded_cli_v1.json`: complete primary v22 30-class command flag list from command.txt line 191, source attribution preserved.
- `docs/STAGE3_P0B_LEGACY_V22_CONFIG_AUDIT.md`: command alternatives, source-vs-default distinctions and correct non-mode support selection.
- `scripts/stage3_p0b_local_preflight.py`: local-only JSON-index digest + source Git blob audit, optional explicit model weight SHA256. Output lives under ignored `outputs/`.
- `scripts/stage3_p0_validate.py`: legacy CLI/fold consistency check and separate `pending_by_gate` (g0/g1/g2/g4/final). Tests/CI extended.
- **Next user action:** pull branch and run `python scripts/stage3_p0b_local_preflight.py --root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --inventory "outputs/stage3_p0/inventory/realiad_inventory.json" --out "outputs/stage3_p0/p0b_local_provenance.json"` (one line), then share summary. Do not run a G0 baseline yet.

## Current task: P0 protocol freeze ONLY

- Gather legacy source paper/supplement, v22 source, true run configs and logs; define exact DINO/sensor/score and Support+Sidecar interface.
- Draft `Canonical Memory I/O v1`, `Experiment Manifest v1` with folds/hashes/stream/support/query/cost/capacity, permission matrix and source-frozen stop criteria.
- Freeze five outer folds of six sealed Real-IAD target categories; remaining 24 source categories undergo 18/6 source-CV four-way evaluation; each of 30 products target exactly once.
- Establish two-tier feedback accounting and oracle-mask physical availability audit.
- Prepare G0 reproducibility acceptance criteria. **Do not start G0 implementation without an explicit next-stage request.**

## Next milestones (not yet executed)

- **G0** exact full legacy v22 + deterministic matched replay, query/memory/sidecar/no-leak audit.
- **G1** independent addressed Normal writes (legacy, global, clusters, raw DINO, shuffled) and addressed retrieval (global, v22 mode, hard local, soft gated), then joint read/write.
- **G2** B vs R, ledger/correction/type/risk readouts; comparison to simple separate stores and masks/types-shuffled controls.
- **G3** shadow complete-write risk audit; deterministic gate vs optional NIG; track harmful TP→FN and unknown recall.
- **G4 optional** at matched annotation minutes compare B-only BEA, random R escalation, heuristic and optional AIF/BED.
- **Final** sealed all 30 categories, paired product-level uncertainty, fixed/random Support robustness, under matched resource budgets.

## Historical results not to reinterpret

- Previous paper (designated 4-shot, 30 Real-IAD): Static Acc 0.5522/AUROC 0.8324; v22 full Acc 0.7837/AUROC 0.8801, qrate 0.0303; Stage-3 not yet evaluated.
- Previous structural credit F1 demonstrates known defect-source routing, NOT automatic normal-context addressability or end-to-end gains.
- Tested 32-D sidecar underperformed raw 1024-D DINO; NIG showed interval coverage gains, NOT demonstrated safety.
- Mask availability in JSON does not prove physical mask integrity; do an inventory before R conditions.

## Collaboration / Git discipline

Canonical current planning document is the linked roadmap, not old README status. For each coding-agent task: read this handoff, roadway and latest source; inspect current remote HEAD/working tree; implement only the approved stage; run reproducible tests; report diffs/commands/metrics/limitations; commit and push; verify remote SHA; then update this handoff and request the next gate review.

**No code changed by branch creation and Stage-3 documentation setup.** Do not merge Stage-3 planning into `main` automatically.
