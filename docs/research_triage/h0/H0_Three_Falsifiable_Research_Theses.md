# H0 — Three Competing, Falsifiable Research Theses

**Audit:** LASTPAPER-H0-20261031  
**Status:** Proposed hypotheses only. No thesis below is a proven result, experiment authorization, or scientific GO.

## Candidate A — Human-grounded structural credit and persistent schema revision

### 1. Precise question

When similar frozen visual evidence is compatible with multiple expert-defined interpretations, can sparse expert feedback identify which reusable schema object and applicability scope should be revised, so future instances improve without online foundation-model fine-tuning?

### 2. Variables and observations

For inspection event t, let:

- x_t be the image or frozen visual features;
- c_t be observed deployment context, such as product/configuration, component, camera, batch, or operating condition;
- z_t be a latent interpretation/schema assignment;
- y_t be the operational Normal/Defect label;
- a_t be human-authored attribute values with states present, absent, or unobservable;
- r_t be relations among attributes/components;
- e_t be optional discriminating evidence, such as a second view or verified process field;
- q_t be the query/action and its inclusion probability;
- k_t be separately measured costs for y, type, attributes, relations, explanation, and extra evidence.

Let persistent state G_t be a graph of human-grounded properties, relations, applicability conditions, exceptions, provenance, and uncertainty. An update action J_t may revise a node or relation, add/split/merge a schema, defer, quarantine, or abstain. The primary outcome is prequential future utility on later instances, not the corrected current image.

### 3. Human feedback required

At minimum: operational label, an expert-defined observable attribute value, and the context in which it applies. To test structural revision, include a relation or correction to an existing rule. A defect type may help select examples but cannot stand in for a cause. A causal explanation requires independent process/physical evidence.

### 4. Persistent state that changes

G_t and the mapping from context and observed attributes to schema applicability change. The backbone remains frozen. A method that only stores x_t and replays a correction is a memory baseline, not evidence of schema revision.

### 5. Strongest existing method

The closest direct pair is iCSN for human-interactive concept revision/addition and CB2M for persistent reuse of concept corrections. A strong combined baseline is context-tagged CB2M/intervention memory plus an editable concept dictionary and ordinary Bayesian mixture or rule revision. Dynamic-V2C is a further high-overlap threat, but its full paper and code were not public on the project page at cutoff. [P03–P06]

### 6. Exact predicted baseline failure

The narrow prediction is that a context-blind nearest-feature memory or fixed-topology CB2M will reuse the wrong correction when two contexts have overlapping visual features but require different schema scopes. It should fail by negative transfer or wrong structural-target selection.

This prediction does **not** establish that a context-aware Bayesian mixture, per-context prototype, or simple split rule will fail. If one of those equal-cost baselines solves the cases, the novel mechanism is unnecessary.

### 7. Minimal theoretical counterexample or empirical falsifier

Counterexample: two latent explanations z=1 and z=2 have the same observable distribution over all features/context/feedback available to the learner:

P(x,c,y,a,r,e | z=1) = P(x,c,y,a,r,e | z=2).

Then posterior odds over z remain equal to prior odds; no structural-credit algorithm can identify the correct edit from those observations. A human-authored attribute, context field, or extra view helps only if it changes the likelihood, e.g. has positive conditional information for distinguishing z.

Empirical falsifier: on a preselected matched-feature/context-reversal set, compare structural revision with feature NN, centroid/prototype, greedy/replay memory, Bayesian mixture with split/merge, MIL/patch assignment where relevant, CB2M-style correction memory, and a simple expert-authored rule dictionary. If feature/context memory or rule baselines achieve equal future utility and transfer at matched annotations and state capacity, stop.

### 8. Expected future learning benefit

Fewer repeated annotations and lower wrong-scope reuse on future objects; the clearest possible benefit is a corrected applicability condition transferring to new visual appearances or configurations with no near-exemplar in memory.

### 9. Main identifiability assumptions

- The context fields used online are observable and recorded before feedback.
- Expert concepts and relations are operationally defined and reliably annotated.
- At least one feedback field or evidence action distinguishes the candidate schemas.
- Both competing structures occur with nonzero sampling probability; no relevant cell has zero query propensity.
- Train/test objects and views are grouped by physical object, with future instances withheld.
- Similarity or prediction scores do not leak evaluator-only labels.

### 10. Negative result that kills the hypothesis

Kill if annotators cannot agree on the property/scope; if context/evidence leaves schemas observationally indistinguishable; if corrections do not change future state; if transfer vanishes on held-out objects/configurations; if a context-tagged prototype, CB2M, or rule dictionary ties at equal cost; or if success depends on hidden masks/types or online backbone fine-tuning.

### 11. Four-month feasibility

**Medium to High.** A small annotation-feasibility pilot and a frozen-feature comparison are feasible within four months. A publication-quality general claim requires multiple product/configuration contexts, expert data, strong concept-memory baselines, and independent validation; the current repository does not supply those annotations.

### 12. Evidence needed before GO

An expert-reviewed annotation contract; evidence that the selected ambiguity pair recurs in real visual inspection; cost and agreement measurements; a formal target estimand and non-identifiability conditions; and a matched-cost baseline comparison. The novelty boundary must be checked against iCSN, CB2M, editable/continual CBMs, Bayesian mixtures, and rule revision.

**Status:** HOLD. Potentially important; not presently independently defensible.

## Candidate B — Active evidence acquisition for reusable structural knowledge

### 1. Precise question

Can a learner choose a second view, crop, modality, process field, or expert question because the answer is expected to resolve structural ambiguity and improve later schema reuse, rather than only improve the current image decision?

### 2. Variables and observations

Use Candidate A’s (x_t,c_t,G_t,z_t). Let u_t be an available evidence action, o_t its result, and K(u_t) its measured action/annotation cost. The relevant quantity is expected future-state utility:

VOI_future(u) = E[U_future(G_{t+1}) | u, current state] - U_future(G_t) - K(u).

Current-instance accuracy and future reusable-knowledge gain must be reported separately.

### 3. Human feedback required

Only the action result that disambiguates a candidate property/schema, with separately costed human labels or evidence access. A second camera view or tool reading may be nonhuman evidence; its acquisition cost and reliability still need measurement.

### 4. Persistent state that changes

The schema graph, its applicability posterior, or an explicit defer/abstain decision must change after observing o_t. If the action affects only the current prediction, this is ordinary interactive prediction, not persistent learning.

### 5. Strongest existing method

Interactive CBMs already select concepts to query for current prediction; CB2M identifies likely mistakes, requests targeted interventions, and reuses interventions; standard targeted/budgeted active learning and VOI cover cost-aware sequential acquisition. ActiveInspect from O0 covers budgeted sequential view/modality/crop/reference/stop selection for current-object diagnosis. [P02, P04, P20]

### 6. Exact predicted baseline failure

The proposed failure is that current-image acquisition policies spend their budget on evidence with immediate predictive value but fail to value an answer that updates a schema reused across later objects. However, a finite-horizon VOI or Bayes-adaptive policy can include that future value. The distinction survives only if the standard future-utility formulation or a strong persistent-memory query baseline fails under matched cost.

### 7. Minimal theoretical counterexample or empirical falsifier

Counterexample: an extra view resolves the current instance but the resolved property never recurs; its only utility is immediate classification. A current-image VOI policy is sufficient and there is no persistent-learning contribution.

Empirical falsifier: on repeated schema instances, compare a future-utility policy with random evidence, current-instance CooP/VOI, one-step Bayesian VOI, and CB2M-targeted query selection at matched human minutes and action access. If no policy changes held-out future schema utility beyond the simple baselines, stop.

### 8. Expected future learning benefit

Lower repeated query cost if one costly evidence action establishes a reusable property/scope relation that transfers to later instances.

### 9. Main identifiability assumptions

The action outcome must be observable, its cost/reliability known, and its distribution conditional on each candidate schema estimable. There must be recurrent later instances that benefit, and evaluation must separate current-image correction from future-state effects.

### 10. Negative result that kills the hypothesis

Kill if action results only change current predictions; if future-aware standard VOI or CB2M matches it; if the action does not alter the schema state; or if repeated future use is absent.

### 11. Four-month feasibility

**High** if the required extra view/action data already exist; otherwise **High** for data acquisition. The current dataset documentation does not provide action logs, costs, or operator-choice histories.

### 12. Evidence needed before GO

Demonstrate that an acquired observation changes a reusable schema and helps independent future objects; then compare against standard future-utility VOI, not only entropy or random query.

**Status:** NO_GO_CANDIDATE as an independent thesis for now. It is too close to active learning/VOI plus CB2M unless persistent transfer is first demonstrated.

## Candidate C — Safe persistent adaptation under selective feedback and normality shift

### 1. Precise question

Under normal appearance shift and rare novel defects, when does selectively returned, delayed, or noisy feedback justify a persistent normal-memory update, and when should the proposal be quarantined or reversible?

### 2. Variables and observations

Let x_t be frozen features, d_t an operating context/shift, Q_t the query indicator with propensity π_t, Y_t the queried Normal/Defect label, η_t the label error process, δ_t the feedback delay, and S_t the mutable detector/memory state. Primary outcome is future sentinel risk and harmful defect absorption under matched query and annotation budgets.

### 3. Human feedback required

Binary Normal/Defect is sufficient to test the narrow update-safety question; query timing, nonresponse, delay, and error must be recorded. Rich Type/Mask can be a separate tier, not an implicit upgrade.

### 4. Persistent state that changes

Normal prototype/bank state and update status: approve, defer/quarantine, reject, or rollback. This does not require semantic schema learning.

### 5. Strongest existing method

HITL Memory Correction directly modifies PatchCore memory from binary feedback; SH-HIL-PCF combines drift triggers and human updates; OWAD/RoDA cover open-world shift and adaptation; selective-label, nonresponse, delay, random audit, and propensity methods are broader baselines. [P16–P18, P26–P31]

### 6. Exact predicted baseline failure

An irreversible greedy normal insertion can absorb a defect when a query is missing, delayed, or wrong, creating future false negatives. A query-dependent stream can also leave zero-propensity regions where inverse-propensity estimation is not identified. The hypothesis is that reversible quarantine plus explicit random audit/coverage and a simple risk gate improves future safety at equal cost.

### 7. Minimal theoretical counterexample or empirical falsifier

Counterexample: a region with π_t=0 contains an unseen defect but is observationally identical to new normality in the observed features. No propensity estimator can identify its risk; a defensible system must abstain or obtain new evidence. Empirical falsifier: compare no-update, greedy insertion, fixed gate, random audit + standard IPS/AIPW where positivity holds, reversible quarantine, and existing drift/HITL baselines on a no-leak future-sentinel stream.

### 8. Expected future learning benefit

More normal coverage under shift without increasing harmful defect absorption or unknown-defect false negatives.

### 9. Main identifiability assumptions

Positive query probability in regions where risk is estimated; recorded query propensity and response process; bounded or estimable feedback error/delay; independent future sentinels; and a known definition of accepted quality.

### 10. Negative result that kills the hypothesis

Kill if fixed gates/random audits/standard estimators match the reversible mechanism; if required response/noise assumptions cannot be checked; if performance depends on test labels or fabricated operator behavior; or if the dataset has no defensible sequential/shift protocol.

### 11. Four-month feasibility

**Medium** for a controlled offline design; **High** for deployment claims because query propensities, response delays, and evolving quality norms are absent from the available benchmark documentation.

### 12. Evidence needed before GO

Access to or collection of real event-order/query/response metadata, a verifiable operating-context shift, and a clean separation between public evaluator labels and online feedback. O0 recommended a risk-identifiability audit before any experiment.

**Status:** INFORMATION_MISSING. It is the strongest O0 alternative but not a substitute for the structural objective.

## Decision summary

No candidate meets all advance criteria today. Candidate A merits a small annotation-identifiability pilot only if humans can provide actual scope-bearing attributes and a recurring ambiguous pair. Candidate B is not independent of current interactive prediction/VOI methods. Candidate C needs deployment-feedback data. These are agent recommendations, not research-stage authorizations.
