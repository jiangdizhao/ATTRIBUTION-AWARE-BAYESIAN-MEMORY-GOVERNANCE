# Last Paper plan v1.1 — Release and provenance record

**Release date:** 2026-11-13 (Australia/Sydney)  
**Branch:** `stage3-eg-famar-execution`  
**Pre-revision remote parent:** `561dfdf091e9b7b80da5688c88ed85583bef852a`  
**Scope:** documentation only; no model experiments, dataset reads, new annotations or runtime/configuration changes.

## Canonical source and rendered deliverable

The source is [Last_Paper_Four_Month_Research_Execution_Plan_2026-11_to_2027-02_v1_1.md](Last_Paper_Four_Month_Research_Execution_Plan_2026-11_to_2027-02_v1_1.md).

| Artifact | Role | Size / checksum |
|---|---|---|
| Full Markdown plan | Repository-maintained complete editable text | 56,392 UTF-8 bytes |
| Markdown SHA-256 | Exact source identity | `96ccdf6ccdc25ad1a1c58e154668f9d9e1c2442d8c9bcfd287bee14c7ee6f9d9` |
| Markdown Git blob SHA-1 | Repository source identity | `e7fa32291ccd1cae322e6a5cdd90e7f4722e6b43` |
| `Last_Paper_Four_Month_Research_Execution_Plan_2026-11_to_2027-02_v1_1.docx` | 36-page rendered editable Word document, delivered separately in the conversation for Project upload | 74,520 bytes |
| DOCX SHA-256 | Exact delivered Word file identity | `72a110c597d7e33e2d188fb7c5f84d5f46648415f642641178a70a247a28b954` |

The DOCX was generated from the complete Markdown body, with a formatted cover, tables, page fields and reference hyperlinks. Its 36 rendered pages were visually reviewed; the DOCX ZIP integrity check passed. These are document-quality checks, not research experiment results. Rendering intermediates are not release deliverables.

The pre-revision Git tree did not contain the user-provided four-month v1.0 DOCX. This revision adds the corresponding full v1.1 Markdown plan and updates existing planning entry points. **The binary DOCX is a separate conversation attachment, not a file uploaded in this Git commit.** The user can upload it to the Project folder as the new current version. The original v1.0 project document remains a historical reference.

## Research decisions carried into the release

1. A continues its existing feasibility validation. New discussion, ABMG's inclusion, one failed implementation or inadequate evidence at a calendar checkpoint does not automatically terminate A. Continuation is not a promise of positive findings.
2. ABMG is an independent conditional alternative. Chapters 12–17 define hypotheses, minimal state/addresses, bounded correction, Bayesian/deterministic comparisons, three compact validation packages and resource/claim limits. Prior implementation failures are retained without treating them as a universal refutation.
3. A now explicitly handles rich feedback authority, region-attribute-context binding, multi-anomaly coexistence, shared/local interference and fixed acceptance rules that cannot suppress unrelated anomalies.
4. Human dependence is assessed at comparable verified reliability; query-rate targets and spending ceilings are different. Budget exhaustion, forced spending and overconfidence cannot count as learned query reduction.
5. Small credible improvements, noninferiority plus cost savings, or selective-update principles may be meaningful. Their primary endpoints must be prospective; no post-hoc rescue of failed registered targets.
6. D6 remains protocol preparation / experiment HOLD. No old data or stage restriction is automatically removed. The recent signed-correction contribution remains USER_REPORTED until exact output/configuration/code audit.

## Changed current entries and preserved history

Current entries: `README.md`, `docs/STAGE3_EG_FAMAR_RESEARCH_EXECUTION_PLAN.md`, `docs/STAGE3_CURRENT_HANDOFF.md`, the full v1.1 plan, and this release record.

Historical documents are stored using their original blobs, unchanged:

| Archive path | Original blob SHA |
|---|---|
| `docs/archive/README_2026-10-23.md` | `e603aec5e5e1f00931443c88653df55e1135c52e` |
| `docs/archive/STAGE3_EG_FAMAR_RESEARCH_EXECUTION_PLAN_2026-10-23.md` | `bd5b06182458a273f5ec1723e96cffa2e998e4a9` |
| `docs/archive/STAGE3_CURRENT_HANDOFF_2026-10-23.md` | `ba73c3e3838e1882394bf6cda297af67e4ab307f` |

Archive-relative links and dated commands are preserved as historical text; follow the current entry points for current instructions. No historical result has been rewritten. Existing scripts, tests, folds, manifests, data and model files are untouched.

## Local synchronization boundary

Fetch and fast-forward the named branch only after checking the Windows working tree. Do not discard local uncommitted work, untracked pilot files or local commits. If a fast-forward is impossible, inspect divergence rather than using `reset --hard` or force push. The remote branch is `stage3-eg-famar-execution`; `origin/stage3-eg-famar-execution` is its local remote-tracking reference.
