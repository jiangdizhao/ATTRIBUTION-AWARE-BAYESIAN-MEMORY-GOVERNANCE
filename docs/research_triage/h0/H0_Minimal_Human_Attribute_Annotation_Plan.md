# H0 — Minimal Human Attribute and Schema Annotation Pilot

**Purpose:** Check whether an industrial visual ambiguity is human-identifiable, repeatable, and useful for future schema revision before any algorithm is developed.  
**Status:** Design only. No images were inspected or annotated for this audit.

## 1. What the pilot must establish

The pilot must establish all three:

1. Experts can distinguish two recurring interpretations that overlap in frozen visual appearance.
2. They can state which observable property, relation, or context makes the interpretation applicable.
3. The resulting correction could plausibly alter future cases, not merely explain the current image.

Do not start by declaring a dataset defect class to be a cause. First ask an inspection expert which recurring ambiguous pattern is meaningful and which additional evidence, if any, resolves it.

## 2. Minimal ontology and annotation record

Use an event record with separate fields. Each field has value, confidence, provenance, time, and an explicit unobservable/unknown value. “Not annotated” must not be encoded as “absent.”

| Field | Minimal allowed content | Meaning and boundary |
|---|---|---|
| Operational label y | normal / defect / indeterminate | Acceptance decision for this object under the current quality rule. Does not name a concept or cause. |
| Defect type d | Existing category label, or expert-authored short controlled term | Descriptive taxonomy only. Do not interpret as causal. |
| Visual attribute a | At most two expert-defined predicates for the selected ambiguity; each present / absent / unobservable, with component/region reference | Observable property, such as a surface discontinuity or residue. It is not a defect class or mechanism. |
| Relation r | One controlled relation among component, attribute, or zone, such as attached-to, crosses-boundary, inside/outside expected region, or adjacent-to | States how properties/components relate. It is not a causal edge unless separately evidenced. |
| Applicability context c | Only fields available to the deployed inspector: product/configuration, component/zone, camera/view, acquisition condition, and relevant process state if recorded | States where a schema applies. A field unavailable at inference cannot be used as a hidden selector. |
| Competing interpretation z | One of two expert-reviewed explanations, both allowed to be noncausal visual interpretations | Records the ambiguity to resolve. Use “cause” only when external physical/process evidence supports it. |
| Discriminating evidence e | Named visual cue, alternate view, tool result, or verified process field, plus source and time | Describes evidence that changes the relative support for z. A model saliency map is not human evidence. |
| Schema correction ΔG | revise scope / add / split / merge / defer / quarantine / retire / no change; affected node/relation and reason | Human instruction about the reusable knowledge structure, not just current-image prediction. |
| Cost and authority | Measured seconds by field, annotator role, source, confidence, and whether the information was paid/authorized | Prevents treating all annotations as free or equally authoritative. |

### Attribute, label, and cause

- **Label:** “This item is acceptable/defective.”
- **Type:** “This item belongs to a defect taxonomy category.”
- **Attribute:** “This visible property is present/absent/unobservable.”
- **Causal explanation:** “This physical process produced the defect.” This requires process history, physical inspection, or other independent evidence; appearance alone is insufficient.
- **Schema:** A reusable bundle of attributes, relations, applicability conditions, exceptions, evidence requirements, and provenance.

### Illustrative ambiguity only

An expert might identify a repeated line-like feature that is a normal joint within a defined component zone but a defect when it crosses that zone. The visual attribute might be “surface discontinuity”; the relation is “crosses expected joint boundary”; the context defines the component/configuration. This illustration is not an annotation claim about any named dataset and is not itself a causal explanation.

## 3. Smallest informative pilot

### Independent examples

Start with **24 independent physical objects**, not 24 views, patches, crops, or frames:

- Select two competing interpretations only after expert screening.
- Cover two observable contexts and two materially different visual appearance regimes.
- Use two independent objects in each of the 2 interpretation × 2 context × 2 appearance cells: 16 positive/ambiguous objects.
- Add 8 matched controls where the key property is absent or genuinely unobservable, balanced across context and appearance.
- Include at least four staged correction events within those objects: record an initial judgment, reveal an authorized alternative view or verified context field, then record whether the expert revises the value or applicability scope. These are ordered evidence events, not fabricated contradictory labels.

This is the smallest pilot proposed for checking vocabulary, cell coverage, repeatability, and whether added evidence can distinguish the pair. One or two examples per cell cannot estimate model performance, statistical power, or population prevalence. The pilot is not powered evidence for a paper. If experts cannot populate these cells from real cases, that is a useful feasibility stop.

Every multi-view observation from one physical object remains one independent object. Real-IAD’s synchronized views, where applicable, must not be counted as separate examples.

### Initial image sources

Use public, non-sealed images only for case finding: Real-IAD, Real-IAD D3, MVTec AD/AD 2, MVTec LOCO AD, or VisA, depending on the expert-identified ambiguity. Public image labels, defect types, and masks may screen examples or serve as evaluator-only outcomes. They do not supply the new human-authored attributes, context rules, relations, or causal explanations. [P21–P25]

Do not run a dataset-wide scan. Curate a small object list through documentation and later human review. Keep all objects from the same physical item, configuration, batch, or capture sequence in one split.

## 4. Human-authored versus dataset-derived information

| Information | Permitted source in the pilot | Role |
|---|---|---|
| Image-level normal/defect label | Existing public dataset annotation for screening/evaluator only, or new expert response | Online feedback only after the authorized query event |
| Defect category/type | Existing public category/type label for screening or evaluator, or expert correction | Descriptive type; never treated as cause |
| Pixel mask | Existing public mask for evaluator-only localization, or a separately paid expert mask if needed | Spatial supervision only; not a schema |
| Attribute definition and per-object value | Must be authored or explicitly verified by an industrial domain expert | Required human-grounded concept supervision |
| Context applicability and exceptions | Must be authored by an expert and based on available operating records | Required to test scope-aware schema revision |
| Component/property relation | Must be authored or explicitly verified by an expert | Required for a structural rule claim |
| Cause or process mechanism | Must be linked to traceable process/physical evidence | Do not infer from image texture or defect type |
| Conflicting correction and reason | Must be captured from the expert with event order and evidence provenance | Tests whether a prior reusable rule should be revised |
| Alternative view/tool result | Actual source observation and acquisition cost | Tests whether more evidence resolves structure |

No current project document records actual human-authored attributes, applicability conditions, relations, or correction history. The Stage-3 Binary/Type/Mask contract is a draft access policy and must not be represented as an existing attribute dataset.

## 5. Annotation procedure, time, and quality control

1. A domain expert proposes two interpretations and writes one operational definition per attribute before seeing model outputs.
2. Annotators first see the image and currently observable context only. They record y, d, attribute values, relation, confidence, and whether cause is actually known.
3. On designated staged cases, reveal the authorized alternate view or process/context evidence only after the initial entry. Record the update as a separate event with the new evidence source.
4. Two qualified annotators independently code all 24 cases. A third expert or joint adjudication resolves disagreements, preserving both original records.
5. Measure time separately for binary judgment, type, each attribute, context, relation, explanation, and correction. Proposed planning cap: five minutes per object packet per annotator; this is a pilot budget assumption, not a measured cost or guaranteed workflow. Record any overrun rather than truncating difficult cases.
6. Keep the annotator blind to method identity, predictions, held-out labels, and future split. Record confidence and “cannot tell” instead of forcing a label.

For the 24-case screen, two independent passes at the five-minute cap imply up to four expert-hours total, excluding adjudication. This budget is only a feasibility target. Report exact per-field medians, ranges, agreement, and adjudication counts; do not convert the small pilot into a population reliability estimate.

## 6. What makes the pilot inconclusive

Mark the pilot INCONCLUSIVE if any of the following holds:

- No real pair of recurring interpretations can be found in allowed public images.
- The distinction depends on a hidden process field unavailable to the deployed learner.
- Annotators cannot agree on attribute definitions or applicability rules after adjudication.
- The two interpretations have no distinguishing observable evidence, and no affordable action can obtain it.
- Context cells are missing, all examples are near-duplicates, or the same physical object leaks across cells.
- The “cause” label is only a defect-type label or a post hoc visual guess.
- Required annotations exceed the measured cost budget or require detailed root-cause investigation for most objects.

If the feature and available context distributions are identical under two explanations, the target is not identifiable from them. More posterior mass, a larger schema graph, or repeated updates cannot reconstruct absent information. An extra view, process record, or expert field is necessary only when it has a likelihood that differs between the candidate explanations and is available before the decision.
