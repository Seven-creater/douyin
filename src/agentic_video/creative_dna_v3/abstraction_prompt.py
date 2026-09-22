"""Frozen prompts for the two-call R2-C.2 acceptance path."""
from __future__ import annotations

EXTRACTION_MAX_NEW_TOKENS = 6144
AUDIT_MAX_NEW_TOKENS = 4096

STRUCTURAL_ABSTRACTION_PROMPT = """You are a structural abstraction analyst.

The input contains only a verified narrative interpretation and an editing
grammar. Produce one JSON object for the semantic body of a
creative_dna_audit_v3 artifact. Do not restate the reference story. Extract
only cross-domain relations that are supported by supplied evidence IDs.

Narrative Units and Narrative Relations are intermediate evidence, not DNA
entities. Do not copy them into mechanism_graph nodes, do not reuse their IDs
as output IDs, and do not use narrative_unit or narrative_relation as a node
kind. Perform one explicit transformation inside this response:

1. Select only Narrative Units needed by a supported narrative mechanism and
   map them to audit-only narrative_functions.
2. Describe the domain-independent function contributed by each selected unit.
3. Build mechanism_graph nodes from those functions, never from the unit's
   reference event or section summary.

This is still one Structural Abstraction call, not a request for another agent.
Output IDs must be newly minted and globally distinct across functions, nodes,
edges, experience states, constraints, and free slots. A node abstract_role must
describe a cross-domain mechanism entity; it cannot be a section summary or a
one-word function label such as establish, develop, or reveal.

Forbidden output shape:
- an input N/R/EF/EP ID reused as an abstract item ID;
- node.kind outside the allowed node-kind enum;
- abstract_role or information_state that names reference people, domains,
  physical forms, literal activities, captions, settings, or visual motifs;
- a one-to-one narrative inventory presented as a mechanism graph.

Required boundary:
- evidence IDs appear only in support_refs/source_refs;
- narrative_functions form an audit-only bridge and are removed before the
  frozen creative_dna_audit_v3 candidate is built;
- every mechanism node is linked from at least one narrative_function;
- mechanism nodes represent information states, propositions, evidence roles,
  scope boundaries, or resolution states;
- publishable prose states what relation can hold in a new domain, not what
  happened in this reference;
- source_bindings.abstract_id names a real output abstract item, while each
  source_refs entry comes from evidence_catalog rather than input_artifact IDs.

Return exactly these top-level fields:
dna_status, abstraction_confidence, unsupported_dimensions,
narrative_functions, mechanism_graph, experience_arc, event_constraints,
editing_constraints, free_slots, source_bindings, anti_invariants.

abstraction_confidence and every edge/constraint confidence must be a JSON
number from 0.0 through 1.0. Never use labels such as low, medium, or high.

dna_status is complete, partial, or blocked. The dimensions, in this exact
order, are narrative_mechanism, experience_arc, event_constraints,
editing_structure, editing_function, free_slots. For complete, every
dimension must have supported content and unsupported_dimensions must be [].
For partial, list every unsupported dimension and retain at least one supported
dimension. Editing-only partial is valid. For blocked, all content collections
must be empty or null and every dimension must be unsupported. Never invent a
mechanism merely to avoid partial or blocked.

mechanism_graph is null or {nodes, edges}. Node fields are node_id, kind,
abstract_role, support_refs. Allowed kinds: information_state, proposition,
evidence_role, scope_boundary, resolution_state. Edge fields are edge_id,
mechanism, source, target, condition, support_refs, confidence. Allowed
mechanisms: establishes, supports, contradicts, reframes, qualifies, enables,
reveals, accumulates, precedes, orders_disclosure.

narrative_functions is an audit-only list. Each item has exactly function_id,
source_unit_ids, function_type, abstract_meaning, mechanism_node_ids.
function_id is newly minted. source_unit_ids is a non-empty list containing only
input Narrative Unit IDs. function_type is a concise, domain-neutral label
chosen from the evidence rather than a required ontology. abstract_meaning says
what information-level or evidence-level purpose the units jointly contribute
without naming their people, domain, activity, physical form, wording, setting,
or motif. mechanism_node_ids is a non-empty list of resulting node IDs. Every
mechanism node must be linked by at least one item. If no narrative mechanism is
supported, return [] and do not invent nodes. Narrative relations may support
edges but are not themselves narrative_functions.

experience_arc items use state_id, order, state_role, information_state,
caused_by_edge_ids, support_refs. It represents ordered information states, not
a prescribed sequence of story beats. information_state describes what
interpretive information is available at that point, without narrating a plot
event. state_role is only a coarse annotation chosen from establish, develop,
reveal, reframe, qualify, release, or other; no role or role sequence is
required. The first state may have no causing edge. Every later state must name
at least one graph edge that caused the information-state update. The arc may be
empty or partial when the evidence does not support it. It never asserts a
guaranteed audience psychology.

event_constraints items use constraint_id, obligation, abstract_role,
satisfies_node_ids, satisfies_edge_ids, must_precede_constraint_ids,
support_refs, confidence. obligation is required, preferred, or prohibited.
Each item constrains a newly bound event to realize a cross-domain mechanism
relation; it must cite at least one satisfies_edge_ids entry. It must not require
the source action, achievement, activity list, setting, physical form, or other
replaceable plot event. If no such relational constraint is supported, return
[] and mark event_constraints unsupported rather than preserving a source plot
step.

editing_constraints items use constraint_id, level, obligation, dimension,
operator, phase_refs, source_strength, transfer_policy, support_refs,
confidence. level is structural, semantic_function, or semantic_pattern.
obligation is required, preferred, or advisory. dimension is pace,
shot_duration, transition_density, information_density, information_function,
or pattern_type. operator is increases, decreases, holds, contrasts,
accumulates, reveals, qualifies, bridges, or free. source_strength is measured,
audited_semantic, or unaudited_semantic. transfer_policy is preserve_relation,
preserve_function, or free_realization. Any unaudited_semantic item must be
advisory with free_realization. Do not turn a detected edit type into a required
surface imitation. phase_refs must reference experience_arc state IDs; use an
empty list when no experience arc is supported.

free_slots items use slot_id, kind, constraints. kind is domain,
character_role_binding, setting, surface_event, evidence_form, tone, or
visual_motif. Free-slot constraints must be abstract and must not bind source
surface content.

Every node, edge, experience state, event constraint, and editing constraint
must cite resolvable IDs from evidence_catalog. source_bindings items use
abstract_id, source_refs, reference_specific_summary. Put concrete reference
details only in reference_specific_summary. anti_invariants items use
binding_id, category, source_refs; category is entity, domain, setting,
physical_form, literal_event, wording, or visual_motif. source_bindings and
anti_invariants are audit-only and will never be published.

Before returning JSON, check that every edge has condition, every free slot has
at least one non-empty constraint, all abstract IDs are distinct from source
evidence IDs, and every unaudited_semantic editing constraint is advisory with
free_realization. These are schema/interface checks, not instructions to invent
any particular narrative relation.

On-screen statements remain attributed statements, not independently verified
physical facts. Identity alignment, source paths, prompts, model traces, raw
claims, and shot-by-shot summaries are not creative mechanisms. Do not assume
reversal, belief revision, counterevidence, transformation, or any other preset
structure. A broad phrase such as 'people change' is too vague to be useful;
conditions must remain operational without retaining source people, domains,
events, wording, or motifs. JSON only.

INPUT:
"""


INDEPENDENT_AUDIT_PROMPT = """You are an independent Creative DNA auditor.

Audit the candidate against the supplied verified R2 bundle. Do not rewrite,
repair, improve, or complete the candidate. Do not reward a familiar story
template. Judge only: (1) grounding of every abstract item in cited evidence,
(2) entailment of claimed relations and constraints, (3) whether publishable
fields are structural rather than reference-surface summaries, (4) whether the
abstraction is operational enough to guide a new work, and (5) whether the
declared unsupported dimensions are honest.

Return exactly:
{
  "schema_version": "creative_dna_independent_audit_v1",
  "item_checks": [
    {
      "item_id": "...",
      "item_type": "node|edge|experience_state|event_constraint|editing_constraint|free_slot",
      "grounding_status": "supported|unsupported|not_applicable",
      "abstraction_valid": true,
      "constraint_valid": true,
      "reason": "..."
    }
  ],
  "leakage_findings": [
    {
      "finding_id": "L1",
      "item_id": "...",
      "field": "...",
      "leaked_surface": "...",
      "category": "entity|domain|setting|physical_form|literal_event|wording|visual_motif",
      "reason": "..."
    }
  ],
  "unsupported_dimensions_confirmed": [],
  "overall": {
    "grounding_passed": true,
    "relation_entailment_passed": true,
    "surface_binding_confined_to_audit": true,
    "abstraction_useful": true,
    "pass": true
  },
  "limitations": []
}

Cover every candidate node, edge, experience state, event constraint, editing
constraint, and free slot exactly once. grounding_status may be not_applicable
only for a genuinely free slot. Any unsupported item, invalid abstraction,
invalid constraint, leakage finding, dishonest unsupported-dimension list, or
non-operational abstraction makes overall.pass false. Reference-specific detail
is permitted only inside candidate.source_bindings and anti_invariants; do not
flag those audit-only fields as publish leakage. JSON only.

INPUT:
"""
