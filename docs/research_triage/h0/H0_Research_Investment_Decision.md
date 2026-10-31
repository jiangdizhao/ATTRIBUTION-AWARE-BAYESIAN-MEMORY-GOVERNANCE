# H0 — Research Investment Decision

**Audit:** LASTPAPER-H0-20261031  
**Recommendation:** No scientific GO and no algorithm development recommendation.  
**Human research approval:** PENDING.

## Direct answer

The current evidence does **not** establish something the learner can acquire, retain, revise, or transfer from sparse human-grounded feedback that a strong feature/prototype/concept-memory baseline cannot already do.

The only plausible residual is a narrower problem: a human correction that includes an applicability scope or relation may identify which reusable schema object should be revised, and that revision may help future objects with similar appearance but different context. This remains a hypothesis. iCSN already revises and adds interactive concepts; CB2M already stores and reuses concept corrections; concept-incremental CBMs expand vocabularies; theory revision adds/deletes rules; standard Bayesian mixtures infer latent component assignments. A combination of these methods could already be sufficient.

The scientific difference must be demonstrated by matched-feature cases where:

1. the correct structural update target is genuinely ambiguous under Binary-only or feature-only evidence;
2. expert-authored attribute/context/relation feedback changes that target;
3. the revised scope transfers to independent future objects;
4. a context-aware CB2M, reversible mixture, prototype, and explicit rule dictionary fail under the same labels, cost, and capacity.

No existing project report or reviewed paper establishes these four facts for this project. If the strongest simple structured baseline ties, the complex mechanism should not be developed.

## Comparative recommendation

| Candidate | Specific gap | Strongest simple competitor and predicted failure | Supervision/data status | Decision |
|---|---|---|---|---|
| A. Human-grounded structural credit and persistent schema revision | Unknown whether sparse scope/relation correction identifies the correct reusable update target and transfers to future contexts | Context-blind NN/CB2M may misapply corrections across contexts; context-keyed CB2M, Bayesian mixture, or simple rule split may solve it | No project-authored attribute/scope/relation set; tiny expert pilot is feasible but uncollected | **HOLD** |
| B. Active evidence acquisition for reusable structure | Unknown whether an action’s value lies in future schema reuse beyond current diagnosis | ICBM/CB2M and standard future-utility VOI may already solve it; if utility includes future state, the formulation may be ordinary sequential decision theory | No action logs, tool costs, or persistent feedback traces | **NO_GO_CANDIDATE** as a separate line until Candidate A demonstrates a reusable state effect |
| C. Safe persistent adaptation under selective feedback and shift | Important O0 candidate: whether noisy, delayed, selective feedback can support safe reversible updates | Fixed gates, random audit, IPS/AIPW, greedy memory, HITL Memory Correction, and drift baselines may suffice | Real query propensities, response delays, label errors, and policy changes are missing | **INFORMATION_MISSING** |

These are investment recommendations only. They do not unlock S1, Q1, S2, Q2, G0, or any other research stage.

## Required decision criteria

| Advance criterion | Candidate A today |
|---|---|
| Specific unsolved problem | Partly: target is narrowed to scope-aware structural revision; prior-art boundary remains close |
| Credible theoretical/method gap | Unknown; may reduce to Bayesian mixture or theory revision |
| Strong simple baseline expected to fail | Only context-blind memory has a stated failure; context-aware baselines may not fail |
| Minimal falsifier feasible | Yes as a small human annotation identifiability pilot, followed by a matched-feature memory comparison |
| Supervision realistic | Unknown until attribute, scope, relation, and cost are measured |
| General claim beyond one benchmark | Not established; needs independent products/contexts and another dataset |

Candidate A does not yet satisfy the advance criteria. No candidate receives GO_CANDIDATE.

## Recommended next action

After human review, consider only a **24-object human annotation identifiability pilot** as specified in H0_Minimal_Human_Attribute_Annotation_Plan.md. Its purpose is to determine whether an expert-defined, recurring ambiguity has usable attribute/context/relation evidence and a feasible annotation cost. It is not an algorithm experiment and cannot establish publication-level performance.

Do not implement structural credit assignment until the pilot shows that:

- humans agree on the property and scope;
- the distinction is observable at deployment time;
- at least one realistic pair defeats context-blind feature memory;
- a simple context-aware rule or mixture does not already solve it.

The first empirical comparison, if later separately authorized, is the no-leak matched-cost design in H0_Decisive_Micro_Experiment_Design.md.

## EISCA and SCAEA

- **EISCA:** remains HOLD. Binary-only patch/address responsibility is still not shown identifiable; broad weak localization, patch-memory updates, and binary HITL memory correction have close precedents. Rich Type/Mask or causal-source labels cannot be silently counted as Binary supervision.
- **SCAEA:** remains HOLD. Cost-sensitive concept querying and decision-aware VOI are established. A query policy only merits separate consideration if it improves persistent future schema state beyond standard future-utility VOI and CB2M.
- Neither H0 nor O0 unlocks a held research stage or authorizes experiments.

## Governance

This is an agent recommendation. Human research approval remains PENDING. No source/configuration/experiment files were modified; no data, hidden labels, training, inference, or experiments were accessed or run. Existing O0 files were preserved.
