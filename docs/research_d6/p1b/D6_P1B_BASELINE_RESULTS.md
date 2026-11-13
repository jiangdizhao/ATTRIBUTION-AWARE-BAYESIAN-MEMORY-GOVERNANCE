# D6-P1-B Baseline Results

**Execution status: PASS.** Every fixed candidate configuration ran; no setting was selected using D-dev future labels. The result is exploratory, not independent confirmation.

## Complete matched result table

The table reports all paired B0 and updated trajectories at checkpoint 12, with each trajectory's checkpoint-0 BA and exact paired gains. `AP all` includes degenerate all-positive products; `AP paired` includes only products with both classes. Harmful transitions count a future view that moves from correct at one checkpoint to wrong at the next, summed over all 12 updates; a view can count more than once. Exact per-product values at checkpoints 0 and 12 are in [D6_P1B_PRODUCT_METRICS_AGGREGATE.csv](D6_P1B_PRODUCT_METRICS_AGGREGATE.csv).

| Protocol | Method | Setting | BA@0 | BA@12 | AP-all@12 | AP-paired@12 | Delta BA vs B0 | Delta AP-paired vs B0 | Harmful transitions | BA/AP products |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X | B0-Logistic-SGD | alpha=1e-03; seed=0 | 0.500 | 0.500 | 0.855 | 0.855 | NA | NA | 0 | 3/3 |
| X | B0-Logistic-SGD | alpha=1e-03; seed=1 | 0.500 | 0.500 | 0.855 | 0.855 | NA | NA | 0 | 3/3 |
| X | B0-Logistic-SGD | alpha=1e-04; seed=0 | 0.500 | 0.500 | 0.855 | 0.855 | NA | NA | 0 | 3/3 |
| X | B0-Logistic-SGD | alpha=1e-04; seed=1 | 0.500 | 0.500 | 0.855 | 0.855 | NA | NA | 0 | 3/3 |
| X | B0-Logistic-SGD | alpha=1e-05; seed=0 | 0.500 | 0.500 | 0.855 | 0.855 | NA | NA | 0 | 3/3 |
| X | B0-Logistic-SGD | alpha=1e-05; seed=1 | 0.500 | 0.500 | 0.855 | 0.855 | NA | NA | 0 | 3/3 |
| X | B1-SGD-Replay | alpha=1e-03; seed=0 | 0.500 | 0.500 | 0.958 | 0.958 | 0.000 | 0.103 | 0 | 3/3 |
| X | B1-SGD-Replay | alpha=1e-03; seed=1 | 0.500 | 0.500 | 0.958 | 0.958 | 0.000 | 0.103 | 0 | 3/3 |
| X | B1-SGD-Replay | alpha=1e-04; seed=0 | 0.500 | 0.500 | 0.958 | 0.958 | 0.000 | 0.103 | 0 | 3/3 |
| X | B1-SGD-Replay | alpha=1e-04; seed=1 | 0.500 | 0.500 | 0.958 | 0.958 | 0.000 | 0.103 | 0 | 3/3 |
| X | B1-SGD-Replay | alpha=1e-05; seed=0 | 0.500 | 0.500 | 0.958 | 0.958 | 0.000 | 0.103 | 0 | 3/3 |
| X | B1-SGD-Replay | alpha=1e-05; seed=1 | 0.500 | 0.500 | 0.958 | 0.958 | 0.000 | 0.103 | 0 | 3/3 |
| X | B0-Logistic-Refit | alpha=1e-03 | 0.500 | 0.500 | 0.844 | 0.844 | NA | NA | 0 | 3/3 |
| X | B0-Logistic-Refit | alpha=1e-04 | 0.667 | 0.667 | 0.689 | 0.689 | NA | NA | 0 | 3/3 |
| X | B0-Logistic-Refit | alpha=1e-05 | 0.556 | 0.556 | 0.603 | 0.603 | NA | NA | 0 | 3/3 |
| X | B1-Refit | alpha=1e-03 | 0.500 | 0.500 | 0.826 | 0.826 | 0.000 | -0.019 | 5 | 3/3 |
| X | B1-Refit | alpha=1e-04 | 0.667 | 0.667 | 0.826 | 0.826 | 0.000 | 0.137 | 7 | 3/3 |
| X | B1-Refit | alpha=1e-05 | 0.556 | 0.667 | 0.678 | 0.678 | 0.111 | 0.075 | 4 | 3/3 |
| X | B0-Ridge | lambda=0.1 | 0.583 | 0.583 | 0.780 | 0.780 | NA | NA | 0 | 3/3 |
| X | B0-Ridge | lambda=1 | 0.500 | 0.500 | 0.826 | 0.826 | NA | NA | 0 | 3/3 |
| X | B0-Ridge | lambda=10 | 0.500 | 0.500 | 0.844 | 0.844 | NA | NA | 0 | 3/3 |
| X | B2-Recursive-Ridge | lambda=0.1 | 0.583 | 0.500 | 0.826 | 0.826 | -0.083 | 0.046 | 6 | 3/3 |
| X | B2-Recursive-Ridge | lambda=1 | 0.500 | 0.500 | 0.826 | 0.826 | 0.000 | 0.000 | 4 | 3/3 |
| X | B2-Recursive-Ridge | lambda=10 | 0.500 | 0.500 | 0.892 | 0.892 | 0.000 | 0.048 | 9 | 3/3 |
| W | B0-Logistic-SGD | alpha=1e-03; seed=0 | 0.500 | 0.500 | 1.000 | 1.000 | NA | NA | 0 | 4/4 |
| W | B0-Logistic-SGD | alpha=1e-03; seed=1 | 0.500 | 0.500 | 0.917 | 0.875 | NA | NA | 0 | 4/4 |
| W | B0-Logistic-SGD | alpha=1e-04; seed=0 | 0.500 | 0.500 | 1.000 | 1.000 | NA | NA | 0 | 4/4 |
| W | B0-Logistic-SGD | alpha=1e-04; seed=1 | 0.500 | 0.500 | 0.917 | 0.875 | NA | NA | 0 | 4/4 |
| W | B0-Logistic-SGD | alpha=1e-05; seed=0 | 0.500 | 0.500 | 1.000 | 1.000 | NA | NA | 0 | 4/4 |
| W | B0-Logistic-SGD | alpha=1e-05; seed=1 | 0.500 | 0.500 | 0.917 | 0.875 | NA | NA | 0 | 4/4 |
| W | B1-SGD-Replay | alpha=1e-03; seed=0 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.000 | 14 | 4/4 |
| W | B1-SGD-Replay | alpha=1e-03; seed=1 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.125 | 14 | 4/4 |
| W | B1-SGD-Replay | alpha=1e-04; seed=0 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.000 | 14 | 4/4 |
| W | B1-SGD-Replay | alpha=1e-04; seed=1 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.125 | 14 | 4/4 |
| W | B1-SGD-Replay | alpha=1e-05; seed=0 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.000 | 14 | 4/4 |
| W | B1-SGD-Replay | alpha=1e-05; seed=1 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.125 | 14 | 4/4 |
| W | B0-Logistic-Refit | alpha=1e-03 | 0.562 | 0.562 | 0.917 | 0.875 | NA | NA | 0 | 4/4 |
| W | B0-Logistic-Refit | alpha=1e-04 | 0.688 | 0.688 | 0.833 | 0.750 | NA | NA | 0 | 4/4 |
| W | B0-Logistic-Refit | alpha=1e-05 | 0.625 | 0.625 | 0.736 | 0.604 | NA | NA | 0 | 4/4 |
| W | B1-Refit | alpha=1e-03 | 0.562 | 0.812 | 1.000 | 1.000 | 0.250 | 0.125 | 21 | 4/4 |
| W | B1-Refit | alpha=1e-04 | 0.688 | 0.812 | 0.917 | 0.875 | 0.125 | 0.125 | 16 | 4/4 |
| W | B1-Refit | alpha=1e-05 | 0.625 | 0.688 | 0.917 | 0.875 | 0.062 | 0.271 | 11 | 4/4 |
| W | B0-Ridge | lambda=0.1 | 0.625 | 0.625 | 0.917 | 0.875 | NA | NA | 0 | 4/4 |
| W | B0-Ridge | lambda=1 | 0.500 | 0.500 | 0.917 | 0.875 | NA | NA | 0 | 4/4 |
| W | B0-Ridge | lambda=10 | 0.500 | 0.500 | 0.806 | 0.708 | NA | NA | 0 | 4/4 |
| W | B2-Recursive-Ridge | lambda=0.1 | 0.625 | 0.812 | 0.917 | 0.875 | 0.188 | 0.000 | 18 | 4/4 |
| W | B2-Recursive-Ridge | lambda=1 | 0.500 | 0.750 | 0.917 | 0.875 | 0.250 | 0.000 | 26 | 4/4 |
| W | B2-Recursive-Ridge | lambda=10 | 0.500 | 0.500 | 0.917 | 0.875 | 0.000 | 0.167 | 0 | 4/4 |

## Fixed future support by product

| Protocol | Product | P | A | U | N/A | Views | Groups | BA status | AP status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X | sim_card_set | 6 | 2 | 0 | 0 | 8 | 4 | defined | defined_both_classes |
| X | toothbrush | 3 | 2 | 3 | 0 | 8 | 4 | defined | defined_both_classes |
| X | transistor1 | 3 | 5 | 0 | 0 | 8 | 4 | defined | defined_both_classes |
| W | porcelain_doll | 4 | 0 | 0 | 0 | 4 | 2 | undefined_missing_class | degenerate_all_positive |
| W | sim_card_set | 4 | 0 | 0 | 0 | 4 | 2 | undefined_missing_class | degenerate_all_positive |
| W | toothbrush | 1 | 1 | 2 | 0 | 4 | 2 | defined | defined_both_classes |
| W | transistor1 | 2 | 2 | 0 | 0 | 4 | 2 | defined | defined_both_classes |
| W | u_block | 2 | 2 | 0 | 0 | 4 | 2 | defined | defined_both_classes |
| W | usb_adaptor | 1 | 2 | 1 | 0 | 4 | 2 | defined | defined_both_classes |

Product supports are first-presentation view counts. Groups are registered sample keys, not proof of independent objects. Product-level BA and AP vary by model/configuration; the aggregate CSV contains every candidate's product values.

## Per-product outcome ranges across candidate settings

Ranges cover all prespecified alpha/lambda settings and both SGD seeds. They avoid presenting one favorable configuration as the full result.

| Protocol | Product | P/A/U | Views | Groups | Learner | B0 BA range | Updated BA range | B0 AP range | Updated AP range | AP status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X | sim_card_set | 6/2/0 | 8 | 4 | B1-SGD-Replay | 0.500-0.500 | 0.500-0.500 | 0.976-0.976 | 0.958-0.958 | defined_both_classes |
| X | sim_card_set | 6/2/0 | 8 | 4 | B1-Refit | 0.500-1.000 | 0.500-1.000 | 1.000-1.000 | 1.000-1.000 | defined_both_classes |
| X | sim_card_set | 6/2/0 | 8 | 4 | B2-Recursive-Ridge | 0.500-0.750 | 0.500-0.500 | 1.000-1.000 | 0.976-1.000 | defined_both_classes |
| X | toothbrush | 3/2/3 | 8 | 4 | B1-SGD-Replay | 0.500-0.500 | 0.500-0.500 | 0.589-0.589 | 0.917-0.917 | defined_both_classes |
| X | toothbrush | 3/2/3 | 8 | 4 | B1-Refit | 0.500-0.500 | 0.500-0.500 | 0.533-0.533 | 0.478-0.533 | defined_both_classes |
| X | toothbrush | 3/2/3 | 8 | 4 | B2-Recursive-Ridge | 0.500-0.500 | 0.500-0.500 | 0.478-0.533 | 0.478-0.700 | defined_both_classes |
| X | transistor1 | 3/5/0 | 8 | 4 | B1-SGD-Replay | 0.500-0.500 | 0.500-0.500 | 1.000-1.000 | 1.000-1.000 | defined_both_classes |
| X | transistor1 | 3/5/0 | 8 | 4 | B1-Refit | 0.500-0.500 | 0.500-0.500 | 0.276-1.000 | 0.500-1.000 | defined_both_classes |
| X | transistor1 | 3/5/0 | 8 | 4 | B2-Recursive-Ridge | 0.500-0.500 | 0.500-0.500 | 0.806-1.000 | 1.000-1.000 | defined_both_classes |
| W | porcelain_doll | 4/0/0 | 4 | 2 | B1-SGD-Replay | NA | NA | 1.000-1.000 | 1.000-1.000 | degenerate_all_positive |
| W | porcelain_doll | 4/0/0 | 4 | 2 | B1-Refit | NA | NA | 1.000-1.000 | 1.000-1.000 | degenerate_all_positive |
| W | porcelain_doll | 4/0/0 | 4 | 2 | B2-Recursive-Ridge | NA | NA | 1.000-1.000 | 1.000-1.000 | degenerate_all_positive |
| W | sim_card_set | 4/0/0 | 4 | 2 | B1-SGD-Replay | NA | NA | 1.000-1.000 | 1.000-1.000 | degenerate_all_positive |
| W | sim_card_set | 4/0/0 | 4 | 2 | B1-Refit | NA | NA | 1.000-1.000 | 1.000-1.000 | degenerate_all_positive |
| W | sim_card_set | 4/0/0 | 4 | 2 | B2-Recursive-Ridge | NA | NA | 1.000-1.000 | 1.000-1.000 | degenerate_all_positive |
| W | toothbrush | 1/1/2 | 4 | 2 | B1-SGD-Replay | 0.500-0.500 | 0.500-0.500 | 1.000-1.000 | 1.000-1.000 | defined_both_classes |
| W | toothbrush | 1/1/2 | 4 | 2 | B1-Refit | 0.500-0.500 | 0.500-0.500 | 0.500-0.500 | 0.500-1.000 | defined_both_classes |
| W | toothbrush | 1/1/2 | 4 | 2 | B2-Recursive-Ridge | 0.500-0.500 | 0.500-0.500 | 0.500-0.500 | 0.500-1.000 | defined_both_classes |
| W | transistor1 | 2/2/0 | 4 | 2 | B1-SGD-Replay | 0.500-0.500 | 0.500-0.500 | 1.000-1.000 | 1.000-1.000 | defined_both_classes |
| W | transistor1 | 2/2/0 | 4 | 2 | B1-Refit | 0.500-0.500 | 0.750-1.000 | 0.417-1.000 | 1.000-1.000 | defined_both_classes |
| W | transistor1 | 2/2/0 | 4 | 2 | B2-Recursive-Ridge | 0.500-0.500 | 0.500-1.000 | 1.000-1.000 | 1.000-1.000 | defined_both_classes |
| W | u_block | 2/2/0 | 4 | 2 | B1-SGD-Replay | 0.500-0.500 | 0.500-0.500 | 1.000-1.000 | 1.000-1.000 | defined_both_classes |
| W | u_block | 2/2/0 | 4 | 2 | B1-Refit | 0.750-1.000 | 0.750-1.000 | 1.000-1.000 | 1.000-1.000 | defined_both_classes |
| W | u_block | 2/2/0 | 4 | 2 | B2-Recursive-Ridge | 0.500-1.000 | 0.500-1.000 | 1.000-1.000 | 1.000-1.000 | defined_both_classes |
| W | usb_adaptor | 1/2/1 | 4 | 2 | B1-SGD-Replay | 0.500-0.500 | 0.500-0.500 | 0.500-1.000 | 1.000-1.000 | defined_both_classes |
| W | usb_adaptor | 1/2/1 | 4 | 2 | B1-Refit | 0.500-0.750 | 0.750-0.750 | 0.500-1.000 | 1.000-1.000 | defined_both_classes |
| W | usb_adaptor | 1/2/1 | 4 | 2 | B2-Recursive-Ridge | 0.500-0.500 | 0.500-0.750 | 0.333-1.000 | 0.500-1.000 | defined_both_classes |

## Performance versus feedback events

Means are across the family configurations; brackets are min-max. Checkpoint 0 deltas are zero by construction.

| Protocol | Learner | Checkpoint | Mean Delta BA [range] | Mean Delta AP-paired [range] | Configs |
| --- | --- | --- | --- | --- | --- |
| X | B1-SGD-Replay | 0 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 6 |
| X | B1-SGD-Replay | 3 | 0.000 [0.000,0.000] | 0.050 [0.050,0.050] | 6 |
| X | B1-SGD-Replay | 6 | 0.000 [0.000,0.000] | 0.103 [0.103,0.103] | 6 |
| X | B1-SGD-Replay | 9 | 0.000 [0.000,0.000] | 0.109 [0.109,0.109] | 6 |
| X | B1-SGD-Replay | 12 | 0.000 [0.000,0.000] | 0.103 [0.103,0.103] | 6 |
| X | B1-Refit | 0 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 3 |
| X | B1-Refit | 3 | 0.000 [-0.111,0.139] | 0.024 [0.000,0.036] | 3 |
| X | B1-Refit | 6 | 0.028 [-0.083,0.167] | 0.005 [-0.019,0.019] | 3 |
| X | B1-Refit | 9 | 0.056 [-0.028,0.167] | 0.065 [-0.019,0.128] | 3 |
| X | B1-Refit | 12 | 0.037 [0.000,0.111] | 0.064 [-0.019,0.137] | 3 |
| X | B2-Recursive-Ridge | 0 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 3 |
| X | B2-Recursive-Ridge | 3 | 0.065 [0.000,0.167] | 0.034 [0.019,0.065] | 3 |
| X | B2-Recursive-Ridge | 6 | 0.028 [0.000,0.083] | 0.015 [0.000,0.046] | 3 |
| X | B2-Recursive-Ridge | 9 | 0.028 [0.000,0.083] | 0.028 [0.019,0.046] | 3 |
| X | B2-Recursive-Ridge | 12 | -0.028 [-0.083,0.000] | 0.031 [0.000,0.048] | 3 |
| W | B1-SGD-Replay | 0 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 6 |
| W | B1-SGD-Replay | 3 | 0.000 [0.000,0.000] | -0.208 [-0.271,-0.146] | 6 |
| W | B1-SGD-Replay | 6 | 0.000 [0.000,0.000] | -0.271 [-0.333,-0.208] | 6 |
| W | B1-SGD-Replay | 9 | 0.000 [0.000,0.000] | -0.062 [-0.125,0.000] | 6 |
| W | B1-SGD-Replay | 12 | 0.000 [0.000,0.000] | 0.062 [0.000,0.125] | 6 |
| W | B1-Refit | 0 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 3 |
| W | B1-Refit | 3 | -0.146 [-0.188,-0.062] | -0.153 [-0.250,-0.104] | 3 |
| W | B1-Refit | 6 | -0.125 [-0.188,-0.062] | 0.132 [0.000,0.271] | 3 |
| W | B1-Refit | 9 | 0.062 [-0.062,0.188] | 0.132 [0.000,0.271] | 3 |
| W | B1-Refit | 12 | 0.146 [0.062,0.250] | 0.174 [0.125,0.271] | 3 |
| W | B2-Recursive-Ridge | 0 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 3 |
| W | B2-Recursive-Ridge | 3 | -0.042 [-0.125,0.000] | -0.111 [-0.229,0.125] | 3 |
| W | B2-Recursive-Ridge | 6 | -0.021 [-0.125,0.062] | 0.014 [0.000,0.042] | 3 |
| W | B2-Recursive-Ridge | 9 | 0.146 [0.000,0.312] | 0.056 [0.000,0.167] | 3 |
| W | B2-Recursive-Ridge | 12 | 0.146 [0.000,0.250] | 0.056 [0.000,0.167] | 3 |

## Descriptive sample-key group outcomes

The table summarizes group accuracy at checkpoints 0 and 12. Within each fixed sample-key, accuracy was first averaged across that learner's prespecified settings/seeds; the table then reports the mean, population SD, and range over evaluable sample-key groups. Eleven of twelve groups per protocol had at least one binary-evaluable view; one group contained only Uncertain views and is omitted from accuracy summaries. These are descriptive group-level spreads, not confidence intervals, and the settings/seeds are not independent replications. A sample-key is operational grouping, not proof of a distinct physical object.

| Protocol | Learner state | Evaluable groups / total | Mean group accuracy, checkpoint 0 → 12 | Between-group SD, checkpoint 0 → 12 | Group accuracy range, checkpoint 0 → 12 |
| --- | --- | --- | --- | --- | --- |
| X | B0 Logistic-SGD | 11/12 | 0.545 → 0.545 | 0.450 → 0.450 | 0.000–1.000 → 0.000–1.000 |
| X | B1 SGD-Replay | 11/12 | 0.545 → 0.545 | 0.450 → 0.450 | 0.000–1.000 → 0.000–1.000 |
| X | B0 Logistic-Refit | 11/12 | 0.545 → 0.545 | 0.370 → 0.370 | 0.000–1.000 → 0.000–1.000 |
| X | B1-Refit | 11/12 | 0.545 → 0.606 | 0.370 → 0.416 | 0.000–1.000 → 0.000–1.000 |
| X | B0 Ridge | 11/12 | 0.530 → 0.530 | 0.139 → 0.139 | 0.333–0.667 → 0.333–0.667 |
| X | B2 Recursive-Ridge | 11/12 | 0.530 → 0.545 | 0.139 → 0.450 | 0.333–0.667 → 0.000–1.000 |
| W | B0 Logistic-SGD | 11/12 | 0.682 → 0.682 | 0.386 → 0.386 | 0.000–1.000 → 0.000–1.000 |
| W | B1 SGD-Replay | 11/12 | 0.682 → 0.318 | 0.386 → 0.386 | 0.000–1.000 → 0.000–1.000 |
| W | B0 Logistic-Refit | 11/12 | 0.697 → 0.697 | 0.308 → 0.308 | 0.000–1.000 → 0.000–1.000 |
| W | B1-Refit | 11/12 | 0.697 → 0.712 | 0.308 → 0.302 | 0.000–1.000 → 0.000–1.000 |
| W | B0 Ridge | 11/12 | 0.576 → 0.576 | 0.130 → 0.130 | 0.333–0.667 → 0.333–0.667 |
| W | B2 Recursive-Ridge | 11/12 | 0.576 → 0.621 | 0.130 → 0.247 | 0.333–0.667 → 0.167–1.000 |

## Feedback, optimization, and cost

- X revealed 7 Present and 5 Absent states (12 binary updates). W revealed 5 Present, 6 Absent, 1 Uncertain; all 12 event units counted, 11 binary updates. No N/A state occurred.
- D-fit initialization was exactly 52 binary views: 27 Present / 25 Absent; the 2 Uncertain labels were masked.
- Three SGD alpha candidates ran with seeds 0 and 1 in both protocols. Three full-history Logistic settings and three Ridge lambda settings ran in both protocols. Each had its own frozen B0.
- All 78 LBFGS fits converged with the fixed iteration/tolerance contract; replaying the registered feedback histories reproduced all 78 model-state hashes.
- Event units are assumed equal cost. Actual human time/money cost and query savings were not measured.

## Runtime and memory

Update times below sum only valid feedback updates per trajectory, excluding future-set prediction. Full-history refit time is reported separately from the initial fit.

| Protocol | Learner | Mean init fit ms | Total valid-update ms/run: mean [range] | Max numeric state bytes | Mean future inference ms/view |
| --- | --- | --- | --- | --- | --- |
| X | B1-SGD-Replay | 13.16 | 22.92 [18.84,40.86] | 53364 | 0.005954 |
| X | B1-Refit | 16.30 | 199.69 [72.75,334.84] | 270868 | 0.006930 |
| X | B2-Recursive-Ridge | 86.69 | 68.74 [68.13,69.47] | 8413200 | 0.005049 |
| W | B1-SGD-Replay | 13.81 | 17.66 [16.18,21.72] | 49259 | 0.005454 |
| W | B1-Refit | 14.47 | 177.83 [59.20,321.56] | 266763 | 0.006343 |
| W | B2-Recursive-Ridge | 93.46 | 65.51 [63.27,66.86] | 8413200 | 0.005051 |

Total process peak RSS was 232,546,304 bytes. P1-B artifacts were 105,269,545 bytes. Runner time was 9.21s wall / 9.28s process CPU. No GPU or DINO execution occurred.
