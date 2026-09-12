# Prototype / Addressability Audit

## Purpose

This is a source-side diagnostic that follows the factorizability and patch-aggregation audits. It does **not** alter the deployed detector, rerun DINO, or modify the supervisor-approved proposal.

The input is an existing `abmg.factorizability.features.v1` artifact. The default comparison is:

- `oracle/raw_patch`: offline mask-localised ceiling;
- `sensor/raw_patch/k8/score_softmax`: practical frozen-sensor evidence selected before this audit.

The primary labels are the fixed cross-product Core-4 defect set `AK, HS, QS, ZW`.

## Question A — unsupervised structure

For each source-category CV fold:

1. hold out one six-product group;
2. L2-normalise training evidence from the remaining source products;
3. fit four-cluster KMeans without using defect labels;
4. assign held-out-product evidence to the learned clusters;
5. report held-out ARI and AMI;
6. for interpretability only, map training clusters to defect labels using the best 4! assignment and report held-out matched macro-F1;
7. repeat the label mapping after shuffling labels *within each training product category* as a control.

ARI/AMI are the cleaner unsupervised diagnostics because they are permutation invariant and do not require assigning semantic names to clusters.

## Question B — sparse-feedback prototype routing

For each source-category CV fold and each shot count `1,2,4,8`:

1. keep the same six product categories fully held out;
2. simulate sparse human feedback by revealing only `shots` confirmed examples per Core-4 defect type from the remaining source categories;
3. choose feedback examples with a deterministic category-diverse round-robin rule so one product cannot dominate a prototype when diverse products are available;
4. L2-normalise each support vector;
5. form one prototype per defect type from the normalised support vectors and re-normalise the prototype;
6. route every held-out-product query to the prototype with maximum cosine similarity;
7. repeat with multiple support seeds;
8. compare to a within-training-category label-shuffle control.

The total feedback budget is therefore `4 × shots` labelled examples: 4, 8, 16, or 32 examples.

Primary outputs include macro-F1, balanced accuracy, top-2 routing recall, and the true-prototype minus best-wrong-prototype cosine margin.

## Interpretation boundary

This audit tests whether frozen evidence already supports a reusable address under sparse feedback. It does not establish semantic/causal disentanglement and it does not by itself establish C1 or C2. A positive result would justify studying factor-routed online memory updates; a negative result would indicate that a richer routing representation or memory structure is needed before selective updating.

## Run

```powershell
python -m unittest tests.test_prototype_addressability_audit -v
```

Then, using the larger 32-per-type/category patch-aggregation artifact:

```powershell
python scripts/abmg_prototype_addressability_audit.py --fold 0 --features "outputs\patch_aggregation_confirm\fold_0\patch_aggregation_features.pt" --representations "oracle/raw_patch,sensor/raw_patch/k8/score_softmax" --shots "1,2,4,8" --repeats 20 --seed 0 --out-dir outputs/prototype_addressability_audit
```

Output:

```text
outputs/prototype_addressability_audit/fold_0/prototype_addressability_audit.json
```

Do not tune K, score-softmax temperature, or the Core-4 label set from this experiment; those choices were fixed before this audit.
