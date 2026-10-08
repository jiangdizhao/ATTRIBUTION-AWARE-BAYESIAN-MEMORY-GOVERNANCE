# E4: Asymmetric verified-feedback audit

Status: implementation in progress, real-data validation pending.

## Frozen protocol

The E3 branch `stage2-e3-normal-state-audit` at `fb02116ac4acaf0c5d5c2212ea9f670c04d3e465` is the source of truth for the frozen raw-DINO sensor and diagonal NIG/Student-t normal-state model. Outer target categories remain sealed; E4 is a source-side mechanism audit.

- E4A: N0 (identical raw-evidence NIG updates) versus N1 (+ category-local bounded hard-normal negative correction from verified FP only). Compare future sentinel FPR with defect-recall/AUROC non-inferiority.
- E4B: freeze independently selected E4A normal policy; compare D0 binary responsibility, D1 correct past-only source centroid, D2 seeded shuffled-source centroid, and D3 offline one-hot oracle address. Defect writes never modify NIG; source centroids cannot see the current example before its write. Only verified FN Core-4 events write positive routing memory.
- E4C: only if A and B separately pass, use C00/C10/C01/C11 factorial integration to detect interaction.
- Fixed source-CV, identical support/query/stream/sentinel for all branches, 32 uniform label-independent queries, prequential prediction before revealed feedback, and checkpoints 0/4/8/16/32.
- E1B rare masks and adaptive C5 queries remain deferred. A new E4 evidence cache is needed to retain frozen top-64 patch candidates; E3 pooled-only evidence cannot support post-feedback spatial reranking.
- Unsuccessful components are not rescued by tuning on held-out target categories. The complete five-fold outer target rotation is a separate confirmatory stage.

## Current deliverable boundary

This documentation commit **does not represent the code implementation**; integration code and tests must be committed and source-side synthetic/local checks pass before the GPU experiment is authorized. Do not treat this branch as experiment-ready until the full E4 runner and tests are present.
