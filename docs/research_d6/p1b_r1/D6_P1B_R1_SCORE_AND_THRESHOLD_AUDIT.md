# D6-P1-B-R1 Score and Threshold Audit

**Date:** 2026-11-13
**Scope:** Post-hoc analysis of the immutable P1-B prediction logs, product metrics, and serialized initial/final SGD states. No learner was rerun or trained. All ROC-AUC values below are post-hoc evaluator-only diagnostics; the registered zero-logit Balanced Accuracy was not changed or threshold-tuned.

## Evidence classes

- **Original registered results:** P1-B checkpoints and zero-logit predictions, preserved in `outputs/research_d6/p1b/run_20261113_01/`.
- **Recomputed aggregate:** the public product CSV is a projection of the original `product_metrics.csv` at checkpoints 0 and 12.
- **Post-hoc diagnostics:** additional score distribution, ROC-AUC, rank-order, and state comparisons computed from those saved results.
- **Unresolved:** why the online updates cause the observed rank changes; whether the behavior persists beyond this small, retrospectively selected development screen.

## Q1 - Protocol W crosses from all Present to all Absent

Yes. Across all six SGD configurations (three alpha values and two seeds), the 21 binary-evaluable future views are all predicted Present at checkpoint 0. After the first three feedback events (checkpoint 3), all 21 are predicted Absent; checkpoints 6, 9, and 12 remain all Absent. The three Uncertain views are excluded from binary metrics. At checkpoint 0, TPR/TNR are 1.0/0.0; after checkpoint 3 they are 0.0/1.0. The resulting product-macro BA is 0.5 in both states.

### Checkpoint summary across the six fixed SGD configurations

Score minima, medians, and maxima are shown as ranges across configurations. Score counts are below / equal to / above the registered threshold 0.0. The full 60-row per-configuration table is local at `outputs/research_d6/p1b_r1/sgd_threshold_summary.csv`.

| Protocol | Checkpoint | Predicted positive / negative (of 21) | TPR / TNR | Scores below / equal / above 0 | Score minimum range | Score median range | Score maximum range | Post-hoc ROC-AUC mean [range] | Correct-to-wrong / wrong-to-correct since prior shown checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 0 | 21 / 0 | 1.000 / 0.000 | 0 / 0 / 21 | 0.018999 to 0.021482 | 0.019647 to 0.022192 | 0.020007 to 0.022546 | 0.893 [0.888 to 0.898] | N/A |
| W | 3 | 0 / 21 | 0.000 / 1.000 | 21 / 0 / 0 | -0.019555 to -0.017085 | -0.017763 to -0.015314 | -0.016721 to -0.014244 | 0.383 [0.378 to 0.388] | 14 / 7 |
| W | 6 | 0 / 21 | 0.000 / 1.000 | 21 / 0 / 0 | -0.046266 to -0.043965 | -0.043820 to -0.041511 | -0.040592 to -0.038269 | 0.265 [0.265 to 0.265] | 0 / 0 |
| W | 9 | 0 / 21 | 0.000 / 1.000 | 21 / 0 / 0 | -0.058370 to -0.056309 | -0.055258 to -0.053189 | -0.050228 to -0.048145 | 0.347 [0.347 to 0.347] | 0 / 0 |
| W | 12 | 0 / 21 | 0.000 / 1.000 | 21 / 0 / 0 | -0.061849 to -0.059977 | -0.057855 to -0.055982 | -0.051342 to -0.049449 | 0.536 [0.531 to 0.541] | 0 / 0 |
| X | 0 | 21 / 0 | 1.000 / 0.000 | 0 / 0 / 21 | 0.019152 to 0.021701 | 0.019622 to 0.022142 | 0.020098 to 0.022640 | 0.653 [0.648 to 0.657] | N/A |
| X | 3 | 21 / 0 | 1.000 / 0.000 | 0 / 0 / 21 | 0.018493 to 0.020975 | 0.019432 to 0.021916 | 0.019994 to 0.022468 | 0.769 [0.769 to 0.769] | 0 / 0 |
| X | 6 | 21 / 0 | 1.000 / 0.000 | 0 / 0 / 21 | 0.026551 to 0.028871 | 0.027900 to 0.030202 | 0.029169 to 0.031505 | 0.769 [0.769 to 0.769] | 0 / 0 |
| X | 9 | 21 / 0 | 1.000 / 0.000 | 0 / 0 / 21 | 0.024665 to 0.026728 | 0.026321 to 0.028401 | 0.028303 to 0.030400 | 0.764 [0.759 to 0.769] | 0 / 0 |
| X | 12 | 21 / 0 | 1.000 / 0.000 | 0 / 0 / 21 | 0.082922 to 0.084734 | 0.085398 to 0.087197 | 0.088454 to 0.090245 | 0.782 [0.778 to 0.787] | 0 / 0 |

## Q2 - The W flip is both a threshold crossing and a rank change

The initial-to-checkpoint-3 confusion pattern is exactly what the zero threshold produces when every score changes sign: 14 initially correct Present views become incorrect and seven initially incorrect Absent views become correct. These are the 14/7 view changes reported in the original summary; no further hard predictions change at the later listed checkpoints.

However, the score ordering also changes substantially, so the effect is **not only** a common offset crossing zero. On the 21 views (210 initially strict score pairs), at checkpoint 3 the mean Spearman correlation with checkpoint 0 is -0.329, with about 41.4% of initial pair orders retained and 123/210 reversed on average across configurations. Global post-hoc ROC-AUC falls from 0.893 at checkpoint 0 to 0.383 at checkpoint 3. By checkpoint 12, mean ROC-AUC is 0.536 and only 53.8% of initial pair orders remain. The fixed threshold explains the all-positive/all-negative predictions and the BA symmetry; the large rank reversal shows additional score-order changes.

## Q3 - Protocol X AP gain is concentrated in toothbrush

The per-product metrics are identical across the six SGD configurations for these listed checkpoints. AP uses binary-evaluable views only; the toothbrush has three Uncertain views that are excluded.

| Product | Binary support (Present / Absent / Uncertain) | AP at 0 | AP at 12 | Delta AP | BA at checkpoints 0 and 12 | Contribution to equal-weight product-macro Delta AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sim_card_set | 6 / 2 / 0 | 0.976190 | 0.958333 | -0.017857 | 0.500 to 0.500 | -0.005952 |
| toothbrush | 3 / 2 / 3 | 0.588889 | 0.916667 | +0.327778 | 0.500 to 0.500 | +0.109259 |
| transistor1 | 3 / 5 / 0 | 1.000000 | 1.000000 | +0.000000 | 0.500 to 0.500 | +0.000000 |

Toothbrush contributes +0.109259 to the product-macro AP change; sim_card_set contributes -0.005952 and transistor1 contributes 0. The net macro gain is +0.103307. Thus the X AP gain is driven by toothbrush and is partly offset by sim_card_set; it is not a uniform three-product improvement. All three products remain BA=0.5 because all 21 binary views stay on the Present side of the fixed threshold at every checkpoint.

## Q4 - Alpha/seed insensitivity is at the hard-label level

For both protocols, the six alpha/seed configurations make exactly the same binary prediction for every one of the 21 views at each of the five displayed checkpoints (100% pairwise prediction agreement). Across-configuration mean per-view score ranges are small but nonzero: W 0.002536 at checkpoint 0 and 0.001871 at checkpoint 12; X 0.002546 and 0.001797. Continuous ranking is not identical in every product/configuration: for example W USB-adaptor AP ranges from 0.5 to 1.0 at checkpoint 0, while at checkpoint 12 its AP is 1.0 for all six.

The archived initial/final SGD states changed for all 12 protocol/configuration trajectories. W weight-vector cosine between initial and final states ranges from -0.207 to -0.176, with intercept shift around -0.0425; X cosine ranges 0.595 to 0.615, with intercept shift around +0.0329. All 12 initial-to-final state digests changed, and each protocol has six distinct final state digests. This confirms updates occurred and the settings did not produce byte-identical models. It does **not** establish convergence: the saved experiment did not record a convergence objective for the online SGD sequence. The evidence supports "same thresholded labels across configurations," not a claim that optimization was sufficient or insufficient.

## Registered W product metric caveat

At W checkpoints 0 and 12, the four mixed-class products each have BA=0.5 under the all-positive and all-negative predictions. `porcelain_doll` and `sim_card_set` contain only Present labels in their future support, so BA remains undefined there; their AP=1 is degenerate. Product-level values for every configuration at checkpoints 0, 3, 6, 9, and 12 are in `outputs/research_d6/p1b_r1/sgd_product_metrics_selected_checkpoints.csv`.

## Limits

- ROC-AUC, rank correlations, and score-distribution comparisons are post-hoc diagnostics, not replacement endpoints or threshold selection.
- The fixed future support is only 21 binary views; W has 12 operational sample-key groups, not proven independent physical objects.
- The diagnostic cannot identify why the update changes score ordering, nor support an explanation based on defect type or product cues without additional authorized experiments.
- No new model was run, and the original P1-B result files remain hash-identical.
