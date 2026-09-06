# ATTRIBUTION-AWARE-BAYESIAN-MEMORY-GOVERNANCE

A sequence of mechanisms for online few-shot anomaly detection: keep the deployed visual backbone frozen, but make its downstream representation factor-addressable so sparse human feedback can update only the relevant parts of Bayesian memory rather than modifying the whole memory indiscriminately.

## Current implementation checkpoint

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
