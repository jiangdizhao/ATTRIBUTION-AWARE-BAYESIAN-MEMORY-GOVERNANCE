# ATTRIBUTION-AWARE-BAYESIAN-MEMORY-GOVERNANCE

A sequence of mechanisms for online few-shot anomaly detection: keep the deployed visual backbone frozen, but make its downstream representation factor-addressable so sparse human feedback can update only the relevant parts of Bayesian memory rather than modifying the whole memory indiscriminately.

## Current Stage-3 research planning (2026-10-23)

The working research direction has advanced to **Scheme B / EG-FAMAR** (Evidence-Governed Factor-Addressed Memory Adaptation and Retrieval). Stage 3 is currently **P0 protocol freeze only**: no G0-G4 mechanism is claimed to be implemented or validated.

- **Execution plan:** [Stage-3 EG-FAMAR P0/G0-G4 and Final protocol](docs/STAGE3_EG_FAMAR_RESEARCH_EXECUTION_PLAN.md)
- **Current handoff:** [Stage-3 research status and next P0 task](docs/STAGE3_CURRENT_HANDOFF.md)
- **P0 contracts:** [Canonical Memory I/O](docs/STAGE3_CANONICAL_MEMORY_IO_V1.md) · [Binary/Rich authority and cost](docs/STAGE3_FEEDBACK_AUTHORITY_AND_COST_V1.md)
- **P0 machine-readable artifacts:** `configs/stage3/experiment_manifest_v1.json`, `configs/stage3/source_cv_v1.json`, and the [recovered legacy v22 CLI](configs/stage3/legacy_v22_recorded_cli_v1.json). Read-only check: `python scripts/stage3_p0_validate.py`; tests: `python -m unittest discover -s tests -p "test_stage3_p0*.py" -v`. The `--strict` check is the G0-SMOKE prerequisite check and intentionally fails until local index/model identity + review are supplied; G1/G2/G4 prerequisites are separate.
- **Branch:** `stage3-eg-famar-execution`, created from `stage2-patch-e3-e4-mechanism-validation` at `3b772a0`.
- **Major protocol decisions:** preserve legacy four-shot Normal Support selection and guarded Threshold Sidecar as first-class controls; initially allow only Binary and (on queried Defect) Binary + Type + Mask feedback; perform matched-support/query/capacity ablations and prohibit unqueried GT access.
- **P0-B record:** [legacy experiment configuration audit](docs/STAGE3_P0B_LEGACY_V22_CONFIG_AUDIT.md). Uploaded command.txt line 191 records the full v22 30-class AIF stack. Historical support uses NO `--use_modes` (non-mode first-four or seeded random four-shot), not mode-central inliers. P0-A user-local inventory checked 30 classes/8 defect codes with no filename/path warnings; mask content not validated.
- **Next task:** locally run read-only `scripts/stage3_p0b_local_preflight.py` to hash 30 JSON metadata files and verify source identity, then inspect DINO checkpoint/preprocessing before any separate G0 smoke approval.

The older Stage-0 instructions below are preserved as historical implementation guidance, not as a statement of current research status.

## Historical Stage-0 implementation checkpoint

Branch: `stage0-foundation-cache`

The first implementation milestone contains:

- **Stage 0A** — a thin VisionAD-aligned, single-view Real-IAD baseline for numerical continuity;
- **Stage 0B** — a restartable frozen DINOv2 patch-feature cache for later `R0 / R1 / R_beta` experiments;
- frozen five-fold Real-IAD category assignment;
- a strict separation between deployable feature/cache metadata and offline-only labels, defect-source labels, and masks.

The two predecessor scripts in `scripts/` remain reference code and are not modified by Stage 0.

## Stage-0 entry point

```bash
python scripts/abmg_stage0_foundation.py --help
```

Recommended local validation order:

```bash
python -m py_compile scripts/abmg_stage0_foundation.py
python -m unittest discover -s tests -p 'test_stage0*.py' -v

python scripts/abmg_stage0_foundation.py inspect \
  --root /PATH/TO/Real-IAD \
  --json-dir /PATH/TO/realiad_jsons_sv
```

Then run a small Stage-0A continuity smoke test and Stage-0B cache smoke test before launching the full Real-IAD job.

See [`doc/STAGE0_FOUNDATION_AND_CACHE.md`](doc/STAGE0_FOUNDATION_AND_CACHE.md) for exact commands, artifact structure, the paper-aligned sensor configuration, and the data-access boundary.

## Source documents

- `doc/attribution_aware_bayesian_memory_governance_step_by_step_guide.pdf`
- `doc/Search is All You Need for Few-shot Anomaly Detection.pdf`

## Research-stage discipline

The Stage-0 implementation intentionally does **not** add beta-VAE training, NIG memory, factor responsibility, source-centroid memory, factor correction, or a new query controller. Those mechanisms are introduced only after the frozen sensor/cache boundary is validated locally.
