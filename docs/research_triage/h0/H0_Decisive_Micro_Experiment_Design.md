# H0 — Decisive Micro-Experiment Design for Candidate A

**Purpose:** Determine whether sparse human-grounded structural revision adds future inspection value beyond strong retrieval, prototype, mixture, and concept-memory baselines.  
**Status:** Design only. No code, data, annotations, training, or experiments were run.

## Hypothesis under test

On real industrial images where visually similar evidence supports competing interpretations, human-authored attribute/context/relation feedback can identify a reusable schema update that improves future-instance explanation and inspection outcomes at matched human cost. This is a conditional hypothesis, not a claim of novelty or a proven result.

The experiment is not about whether a model can replay a corrected concept value. CB2M already studies that. The question is whether explicit revision of a schema object and its applicability scope helps beyond the same human feedback stored in simpler ways.

## Data and sample design

### Smallest decision experiment

- **Stream:** 24 unique physical objects from the annotation pilot. Reveal only the authorized fields at each simulated/real query event.
- **Future sentinel:** 24 additional unique physical objects with human-authored attribute/scope labels sealed from the methods until scoring. Existing public defect labels and masks may be used by the evaluator only.
- **Total:** 48 independent physical objects; every view, crop, frame, or patch of one object remains in the same group.
- **Coverage:** two expert-selected interpretations, two observable contexts, and two distinct appearance regimes. Keep each product/configuration, batch, physical item, and synchronized-view group in a single split.
- **Cross-context condition:** hold out a configuration/context from feedback while providing only context fields that would actually be available to the deployed method. If the context itself is unavailable, do not claim transfer into it.
- **Exemplar-resistant condition:** split by object/configuration/capture sequence and appearance regime; exclude duplicate views and near-identical acquisitions from feedback. The primary held-out result must include similar frozen features with different expert interpretations, not only easy far-apart classes.

These counts are a cheap go/no-go screen, not statistical power. A positive result would require expansion across independent product groups and a second dataset before a paper-level claim.

### Candidate source sets

After expert screening, use only a small number of public, non-sealed image groups from Real-IAD, MVTec LOCO AD, MVTec AD/AD 2, VisA, or Real-IAD D3. Choose the dataset that actually contains the ambiguity and useful context fields. Public type labels and masks are evaluator-only unless explicitly re-annotated and authorized. No dataset-wide scan is part of this design. [P21–P25]

## Feedback and event protocol

1. Freeze the feature extractor, initial support, candidate schemas, and test split before querying.
2. At each stream event, score and record the current prediction first.
3. Reveal exactly the contracted paid fields: Binary, type if authorized, each attribute/relation/context field, and optional additional evidence. Record field-level cost and authority.
4. Allow each method to update only after the field is revealed. Log its proposed target object, action (revise/add/split/merge/defer), state before/after, and provenance.
5. Never give any method evaluator-only type/mask/attribute labels before the event authorizes them. Keep future sentinel labels sealed until all outputs are frozen.
6. Apply contradictory evidence as a new ordered event that cites the earlier event and any changed context/evidence; do not overwrite the audit history.

The primary mechanism comparison uses identical query IDs and identical fields for every method. Query selection is held fixed because Candidate A is a write/structure claim, not a query-policy claim. An active query comparison belongs to Candidate B and is not added unless separately justified.

## Baselines

All baselines receive identical feedback authority, query positions, cost, frozen representation, support, context fields, and comparable persistent-state capacity.

1. Frozen-feature nearest neighbour.
2. Online nearest centroid/prototype.
3. Greedy insertion/replay memory with bounded capacity.
4. Ordinary Bayesian mixture or class-conditional memory with latent component assignment.
5. MIL/top-K patch assignment if the hypothesis requires spatial responsibility.
6. CB2M-style memory that stores the corrected example encoding and intervention.
7. Fixed structured concept dictionary with explicit expert-authored context rules.
8. Reversible Bayesian mixture/rule baseline with standard component split/merge selection.
9. Proposed structural-credit/schema-revision method.

CBM-style interventions must receive the same human concept labels. Do not compare a rich attribute/rule method against a Binary-only baseline and attribute the difference to the algorithm. If the proposed method uses a causal or physical variable unavailable to the baselines, that supervision must be separately costed and analyzed.

## Outcomes, separated by learning claim

| Outcome | What it tests | What must not count as success |
|---|---|---|
| Same-instance correction | Immediate response to an authorized correction | Improvement to the image that was just labeled |
| Future-instance reuse | Utility on later independent objects in the same context | Reclassification of stored examples or duplicate camera views |
| Cross-context transfer | Use of a scope/relation on a held-out configuration | A context ID or label leaked from the evaluator |
| Knowledge interference | Damage to previously learned contexts/defect interpretations after a new update | Reporting only the new context’s gain |
| Revision after contradiction | Correctly changing scope or schema after new evidence | Deleting history or counting the contradictory label as an ordinary new class |

Primary outputs: future object-level defect decision at fixed operating points; future schema/interpretation assignment against expert-held-out labels; harmful cross-context reuse; performance retained on previously covered contexts; annotation minutes; memory bytes; and the fraction of corrections that change a future state. Report per-object and per-product/context paired results, not patches as independent samples.

## Expected baseline failure and required check

The expected failure is specifically **wrong-scope reuse**: a context-blind nearest-neighbour or CB2M memory may apply an old correction to a visually similar item whose expert-defined applicability differs. The primary analysis must show that this error occurs and that the new method corrects it without damaging the original context.

If a context-keyed CB2M, ordinary Bayesian mixture with reversible split/merge, or the simple rule dictionary resolves the same cases, then there is no demonstrated algorithmic headroom. Do not weaken those baselines to manufacture a win.

## Stop rules

Stop or narrow Candidate A if:

- experts cannot define and consistently label the two interpretations;
- the differing context or evidence is unavailable before the decision;
- the feature/context/feedback distribution is non-identifying;
- any strong simple baseline ties at equal supervision, cost, and state capacity;
- gains are limited to same-instance correction or near-duplicate retrieval;
- cross-context benefit comes from richer labels rather than structural update;
- new updates damage previously learned contexts without a reliable revision mechanism;
- human annotation cost makes repeated use implausible.

## Later independent validation

A positive micro result is not sufficient for scientific novelty. Independently reproduce it on a second industrial dataset or product family with object-level grouping and a different appearance/context pattern. Use public labels only as evaluator information; newly authored attributes, scopes, and relations must be separately documented. If the mechanism depends on a specific assembly ontology, validate transfer to an independently defined ontology or explicitly narrow the claim.

## Execution boundary

This document proposes a design only. It does not authorize data access, image review, annotation collection, implementation, training, GPU inference, or any Stage-3/HOLD transition.
