# Memory Addressability Audit

## Purpose

This experiment follows the source-side factorizability and prototype audits.
Those audits established two provisional facts on outer fold 0:

1. frozen DINO defect-region evidence contains reusable cross-product defect-type information;
2. a single cosine centroid per Core-4 defect type improves steadily as sparse verified feedback increases, but remains well below the full-data oracle probe ceiling.

The next question is therefore not whether to train another representation model.  It is whether a slightly richer **external memory** can exploit the same frozen evidence more effectively under the same sparse-feedback budget.

This audit is diagnostic only.  It does not change the deployed detector, DINO backbone, supervisor-approved proposal, or outer-fold target categories.

## Fixed evaluation scope

Primary factors:

- AK
- HS
- QS
- ZW

Representations:

- `oracle/raw_patch` — offline localization ceiling;
- `sensor/raw_patch/k8/score_softmax` — predeclared deployable evidence representation from the preceding patch-aggregation audit.

Few-shot feedback budgets:

- 1 example per factor = 4 total labels;
- 2 per factor = 8 total;
- 4 per factor = 16 total;
- 8 per factor = 32 total.

Supports are category-diverse and are drawn only from the 18 source-CV training product categories.  Evaluation occurs on the six held-out source product categories.  The outer ABMG target categories remain untouched.

## Compared memory mechanisms

### 1. Single centroid

The previous baseline stores one normalized mean per factor:

\[
c_g = \operatorname{Norm}\left(\frac{1}{N_g}\sum_{i:y_i=g}\operatorname{Norm}(z_i)\right).
\]

Routing uses cosine similarity.

### 2. Exemplar-max multi-prototype memory

Every verified example is retained as a prototype:

\[
M_g=\{z_{g,1},\ldots,z_{g,N_g}\}.
\]

The factor score is

\[
s_g(z)=\max_{m\in M_g}\cos(z,m).
\]

This is intentionally primitive: no K-means, no learned router, no SGD, and no tuned temperature.  It directly tests whether the weakness of the single-centroid baseline is caused by multimodality within each factor family.

### 3. Bayesian shared-diagonal memory

All source-CV **training-category** evidence vectors are used without labels to estimate a background mean \(\mu_0\) and shared diagonal covariance \(\Sigma\).  This background state is fixed for the CV fold.

For each factor \(g\):

\[
\mu_g\sim\mathcal N(\mu_0,\Sigma/\kappa_0),
\]

\[
z\mid\mu_g\sim\mathcal N(\mu_g,\Sigma).
\]

The predeclared prior strength is

\[
\kappa_0=1.
\]

Sparse verified feedback updates only

\[
n_g\leftarrow n_g+1,
\qquad
s_g\leftarrow s_g+z.
\]

The posterior mean is

\[
\mu_g^*=\frac{\kappa_0\mu_0+s_g}{\kappa_0+n_g}.
\]

The posterior predictive distribution is

\[
z_{new}\mid D_g
\sim
\mathcal N\left(
\mu_g^*,
\Sigma\left(1+\frac{1}{\kappa_0+n_g}\right)
\right).
\]

Equal factor priors are used.  Normalized predictive probabilities are recorded as candidate write-responsibility values.

`kappa0` must not be tuned on held-out product categories in this audit.

## Controls and diagnostics

Every memory mechanism is evaluated with the same real support examples.  A within-training-category label-shuffle control destroys the association between evidence and defect label while preserving product-category label frequencies.

At every shot count the script reports:

- macro-F1;
- gain over the within-category shuffle control;
- per-class F1 for AK/HS/QS/ZW;
- true-normalized 4x4 confusion matrix;
- top-2 routing recall;
- true-vs-best-wrong score margin.

For the Bayesian memory it additionally reports:

- mean responsibility assigned to the true factor;
- predictive entropy;
- entropy on correct and incorrect routes;
- negative log likelihood;
- Brier score.

These uncertainty quantities are diagnostics only; no threshold or abstention rule is tuned here.

## Decision logic

The experiment is useful even if no richer memory wins.

- If `exemplar_max` materially beats `single_centroid`, the evidence supports a multimodal external-memory interpretation: one factor should retain multiple local prototypes rather than collapse to one centroid.
- If `bayes_shared_diag` improves routing and gives sensible uncertainty separation, a compact Bayesian sufficient-statistic memory is a plausible next mechanism for responsibility-weighted write governance.
- If neither mechanism improves the single centroid, the present frozen evidence may be decodable but not easily exploitable by these simple sparse-memory forms; a learned/lightweight router or different evidence representation would then need justification.
- Per-class confusion determines whether the bottleneck is concentrated in specific defect families rather than global.

No result from this audit alone establishes final C1/C2 or causal/semantic factorization.

## Command

Using the existing large patch-aggregation artifact:

```powershell
python scripts/abmg_memory_addressability_audit.py --fold 0 --features "outputs\patch_aggregation_confirm\fold_0\patch_aggregation_features.pt" --representations "oracle/raw_patch,sensor/raw_patch/k8/score_softmax" --shots "1,2,4,8" --repeats 20 --seed 0 --bayes-kappa0 1.0 --out-dir outputs/memory_addressability_audit
```

Output:

```text
outputs\memory_addressability_audit\fold_0\memory_addressability_audit.json
```
