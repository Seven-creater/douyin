# R2-C.4 Creative Structure Minimal-Sufficiency Experiment

Status: `IMPLEMENTED, NOT EXECUTED`

This directory is an offline human-evaluation experiment. It is deliberately
isolated from `src/agentic_video/creative_structure_v1`: it does not import the
runtime, call Omni/LLM, modify production artifacts, or generate stories.

## Research questions

- RQ1: Which candidate structures can express the frozen reference mechanism?
- RQ2: Which components of a maximum candidate are necessary under deletion?
- RQ3: Does the smallest sufficient structure transfer across domains while
  rejecting surface-similar but mechanism-wrong cases?

## Frozen source

The reference representation is the R2-B output used after commit `adfd863`:

- Narrative SHA: `2496bc6be5e00280311c73abcf5b4b18f4ef707cad5c11af0ac050429f695d34`
- Editing SHA: `f4211e31bf1e0c1e387e5055633ff40d96bb22288a7150de54624c93451892a3`

Neither artifact is copied into this directory or exposed to transfer-panel
reviewers. Candidate labels and human judgments must never enter an extraction
prompt.

## Experiment 0 — Candidate screening pilot

`candidates/screening_candidates.json` freezes A/B/C as three competing
theoretical directions:

- A: minimal information update;
- B: information update plus capability specialization and qualification;
- C: social contradiction and transformation.

This is only a pilot to check whether reviewers can apply the criteria
consistently. It is not the primary minimality experiment and cannot by itself
identify indispensable components.

## Experiment 1 — Full Structure deletion ablation

`candidates/full_structure.json` contains the union of the reasonable generic,
capability/scope, and social/transformation components. It has 11 frozen
components.

`deletion_tests/ablation_plan.json` contains one first-level deletion per
component. Deleting a component also removes every dependent component so that
reviewers never evaluate an invalid dangling graph. Each variant records:

- the deletion root;
- the complete dependency closure removed;
- retained components;
- structural compression ratio.

The first-level plan is a pilot-sized ablation, not an exhaustive powerset. If
a deletion passes, a later versioned experiment may use that retained variant
as a new parent. This experiment does not silently grow a combinatorial tree.

Structural compression ratio is:

```text
removed components / full candidate components
```

A dependency-root ablation tests the necessity of the root plus its forced
closure. It must not be misreported as isolated causal attribution for every
removed dependent.

## Experiment 2 — Cross-domain transfer

`bindings/cases.json` freezes three positive domains:

- software incident diagnosis;
- learning assessment;
- cooking process diagnosis.

Reviewers bind variables and relations only; they do not write stories. A
positive passes only when the structure is retained without adding, deleting,
merging, or renaming a core relation.

## Experiment 3 — Surface-lure controls

The same file freezes two negative cases:

- `N1_SURFACE_LURE`: athletic victory with neither a prior interpretation nor
  an information-driven update;
- `N2_CONTROLLED_MECHANISM`: the same athletic surface with a stable prior
  interpretation, but the result changes no interpretation.

The second case isolates information update more closely: rejection cannot be
explained by rejecting the sports surface.

## Human panels

- Panel S sees the frozen reference representation and one anonymized
  structure. It judges source sufficiency only.
- Panel T sees one anonymized structure and transfer cases, but not the
  reference. It records bindings and negative-case rejection.
- Panel M sees a valid parent and one deletion variant. It may not repair the
  variant or add a relation.

At least two reviewers work independently. Disagreements are recorded in
`evaluation_forms/reviewer_disagreements.json`; they are not overwritten by a
silent majority vote.

## Gates and selection

A structure is eligible only when:

1. source sufficiency passes;
2. all three positive cases pass without structural edits;
3. both negative cases are rejected;
4. reference-bound commitments equal zero.

Among eligible structures, `analysis.py` selects lexicographically by:

1. higher structural compression ratio;
2. fewer binding additions;
3. fewer unresolved binding decisions;
4. fewer retained components;
5. stable variant ID as a deterministic final tie-break.

If forms are missing or contain `null`, status is `NOT_READY`. If all forms are
complete but no structure passes, status is `INCONCLUSIVE`. The tool never
fills a judgment or changes a threshold.

## Files

```text
candidates/screening_candidates.json   Experiment 0 candidates
candidates/full_structure.json         Experiment 1 maximum candidate
bindings/cases.json                    Positive and negative cases
deletion_tests/ablation_plan.json      Generated first-level ablations
evaluation_forms/                      Blank human records
analysis.py                            Validation, ablation, and analysis only
```

## Commands

```text
python -m experiments.creative_structure_minimality.analysis validate
python -m experiments.creative_structure_minimality.analysis generate-deletions
python -m experiments.creative_structure_minimality.analysis analyze
```

The checked-in evaluation form is intentionally empty, so the last command
must return `NOT_READY` until real human judgments are recorded. Execution of
the human experiment requires a separate explicit instruction.
