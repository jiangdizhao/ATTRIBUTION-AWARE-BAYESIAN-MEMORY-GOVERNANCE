# D6-P1-B Experiment Protocol and Execution Record

**Status: PASS.** The bounded frozen-CLS online baseline replay completed on 2026-11-13. This is a development screen only; it does not authorize D6-P2/P3, A1/A2, B3/B4, ABMG, or D-conf access.

## Authorization and integrity gate

- Branch and starting HEAD: `stage3-eg-famar-execution` / `815e5c801be408d08807089d9b48eb372e9258a6`.
- Before any learner ran, the approved D4-Lite, D4B, and D5R-V2 manifests were independently reconstructed and matched the P1-A whitelist: **102/102 IDs** (54 D-fit, 48 D-dev). Source-byte hashes and feature-file hashes matched 102/102. Features were finite, 1024-D FP32, L2-normalized; maximum norm error was 1.76e-07.
- Input-manifest SHA-256: `820947acd834aad337f1c46587051f3f4f9bdee6b174dcc2c89de0bb43830e63`. Feature-index SHA-256: `ee0232b29f64f9fd4f10198ba3ef20da89fa85b2083a389b774f6a9723bbb99d`. Controlled W/X tape SHA-256: `13561809074aad72e2f13289723ccba8addc4768395480c50d3173abf66e8cc1`.
- DINO checkpoint, cached Hub source tree, and preprocessing fingerprints matched P1-A: `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`, `7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c`, `4d3a5906bacafde057f806bbe53ccf24e4960574837e9c47bd273233223ec56e`. P1-B did not load DINO, decode image pixels, extract features, or use GPU.
- Every W/X query and future ID resolved to its exact authorized feature record. Each protocol had 12 unique feedback events, 24 future views, and 12 sample-key groups on each side; no sample-key crossed roles. The tape was not edited.
- Post-run source, annotation, queue, tape, feature-manifest, and feature-file digests matched the pre-run snapshot. No input asset changed.

## Annotation states and grouping

The canonical human records were the tool-written `annotations.csv` files, ordered by their frozen annotation queues. Repeated presentations were collapsed to the first presentation per image: D4-Lite 36 images from 44 presentations; D4B 18 from 26; D-dev 48 from 58. Repeats were never treated as additional training or feedback events.

- D-fit: 27 Present, 25 Absent assessable, 2 Uncertain. Only the 52 binary states initialized heads; N/A was not converted to a target.
- D-dev first states: 25 Present, 19 Absent assessable, 4 Uncertain, 0 N/A.
- Protocol X feedback: 7 Present, 5 Absent (12 binary updates). Future: 24 views, 21 binary-evaluable, 12 sample-key groups across 3 future products.
- Protocol W feedback: 5 Present, 6 Absent, 1 Uncertain (12 event units, 11 binary updates). Future: 24 views, 21 binary-evaluable, 12 sample-key groups across 6 products.
- Sample keys are operational grouping only; they do not prove independent physical objects. All pilot-exposed images remain excluded from D-conf. Historical sealed/HOLD overlap remains UNKNOWN and was not inspected.

## Learner contracts used

All methods used the same 1024-D P1-A FP32 vectors. No additional scaling or data-dependent preprocessing was fit. The learner API received feature vectors; only a registered binary human label arrived after reveal. Product, sample key, view, source type, source split, official normal/defect labels, masks, confidence, and future attributes remained evaluator-only.

- **B0:** paired immutable copy of each estimator/configuration's D-fit initialization. B0 state did not change through any event.
- **B1 SGD logistic + replay:** `SGDClassifier(loss="log_loss", penalty="l2")`, alpha `(1e-05, 0.0001, 0.001)`, constant `eta0=0.01`, seeds 0 and 1. Initialization was one seeded permutation and one pass over the 52 binary D-fit items. After a valid reveal, the unique event was appended and one insertion-order pass was made over the bounded 12-item buffer. Uncertain/N-A labels were not inserted or used as negative.
- **B1-Refit:** full-history `LogisticRegression(solver="lbfgs", penalty="l2")` with the same alpha values mapped to `C=1/(alpha*n_binary_training_rows)` under the mean-log-loss plus `alpha/2 ||w||^2` convention. LBFGS does not penalize the intercept. This maps objective strength; it does not claim numerical equivalence between SGD updates and full-history optimization. Fixed `max_iter=10000`, `tol=1e-10`; all 78 initial/update fits converged (maximum 211 iterations). An exact audit replay matched all 78 checkpoint state hashes under two threads.
- **B2 recursive Ridge:** lambda `(0.1, 1.0, 10.0)`, float64 sufficient statistics, exact Sherman-Morrison update. The augmented intercept is regularized by the same lambda. BA threshold fixed at 0.5; AP uses raw continuous scores.
- Logistic classification uses a zero logit threshold; AP uses raw decision scores. There was no future-label tuning. Every candidate used its matched frozen B0. The historical tape is not a live query-policy or actual human-cost experiment.

## Prequential order and metrics

Checkpoint 0 evaluates the initialized state. At event `k`, predict the registered query, then reveal its first human state, update only on Present/Absent, then evaluate the same fixed future set at checkpoint `k`. Uncertain consumes one assumed event-cost unit and causes no binary update. N/A remains distinct. Other views from a queried group cannot enter future evaluation.

Primary metric: Protocol X product-macro balanced accuracy with all three product supports. Protocol W is secondary and reports eligible-product macro BA; two all-positive W products have undefined BA, retained as missing. AP is reported by product, with both all-positive-inclusive product macro and paired-class product macro. All-positive AP=1 is marked degenerate. Group summaries are descriptive; future views are not counted as independent physical objects. No confidence interval over only three X products is presented as robust generalization.

## Runtime, tests, and artifacts

- Runtime: `D:\anaconda3\envs\torch\python.exe`, Python 3.11.14; NumPy 1.24.1, scikit-learn 1.3.0, SciPy 1.11.2, threadpoolctl 3.2.0, joblib 1.3.2, psutil 5.9.0. CPU-only, CUDA masked, PyTorch not imported, thread limit 2. No packages installed.
- Main run: 9.21s wall / 9.28s process CPU; peak observed RSS 232,546,304 bytes. Artifacts: 105,269,545 bytes, below 200 MiB.
- Attempts: 24 learner trajectories plus 24 paired B0 trajectories; 48/48 PASS, no failed candidate settings.
- `py_compile`: PASS; unit tests: 13 PASS, including ordering/leakage counterfactuals, W/X metadata gate, registered-X replay repeatability, recursive Ridge vs direct batch, repeat collapse, and input immutability.
- Run command: `D:\anaconda3\envs\torch\python.exe scripts\research_d6\p1b\run_experiment.py --run-dir outputs\research_d6\p1b\run_20261113_01 --max-runtime-seconds 3600`.
- Test command: `D:\anaconda3\envs\torch\python.exe -m unittest discover -s tests\research_d6 -p 'test_p1b_*.py' -v`.
- Raw predictions, evaluator-only joined annotations, model states, event logs, and per-ID integrity hashes are local under `outputs/research_d6/p1b/run_20261113_01/` (Git-ignored). Public docs contain only aggregates. After the run, explicit B0 coefficient/intercept/threshold arrays were added to the state archive and their digests matched the recorded B0 states; no learner was rerun and no result or input changed.
