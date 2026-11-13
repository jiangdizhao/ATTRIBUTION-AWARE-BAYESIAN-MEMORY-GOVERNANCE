# D6-P1-B Review and Next Action

## Decision

**P1-B: PASS as an exploratory matched-baseline screen.** The independent cache gate passed; B0/B1/B1-Refit/B2 ran under the fixed W/X tapes; all 48 paired trajectories passed; the required leakage, no-update, repeat, and recursive-Ridge checks passed. This is not an independent confirmation and is not authorization for D6-P2.

## Audit summary

- Exact feature whitelist: 102 records (54 D-fit, 48 D-dev); source and feature hashes verified before and after; fixed W/X tape SHA verified.
- D-fit used 52 binary labels (27 Present, 25 Absent); two Uncertain labels were masked. D-dev used only first human states, revealed at registered events.
- X: 12 events (7 P/5 A), 24 future views, 21 binary labels, 12 sample-key groups, three future products. W: 12 events (5 P/6 A/1 U), 24 future views, 21 binary labels, 12 sample-key groups, six products.
- X BA gains are mostly zero across the simple online controls, while AP gains are positive for SGD+Replay and modest on average for the other families. W BA improves for some Refit/Ridge settings but has transient and final harmful flips. No candidate was selected as a winner.
- Runtime was 9.21s wall / 9.28s process CPU; peak RSS 232,546,304 bytes; local artifacts 105,269,545 bytes (<200 MiB). CPU-only; no install, GPU, DINO, or image-pixel access.
- Tests: 13 passed. No original annotation, manifest, tape, feature file, or historical result was changed. No commit/push.

## Limits

D4/D4B and Real-IAD V2 are development-exposed. D-conf remains unauthorized and untouched. Historical sealed/HOLD file-level overlap remains UNKNOWN. Source-type correlations remain; sample-key grouping does not certify object independence. This screen tests global CLS only and does not reject spatial DINO features or Direction A. Fixed replay does not measure actual human cost or query savings.

## Recommended single next action

**Research lead: review and freeze this P1-B package, then separately decide whether to authorize a preregistered D6-P2/headroom follow-up.** No additional learner, baseline, data access, or confirmation-set work starts automatically.
