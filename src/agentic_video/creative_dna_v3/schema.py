"""Frozen schema constants for the R2-C.1 Creative DNA boundary."""
from __future__ import annotations

AUDIT_SCHEMA_VERSION = "creative_dna_audit_v3"
SPEC_SCHEMA_VERSION = "creative_spec_v1"
PROVENANCE_SCHEMA_VERSION = "artifact_provenance_v1"

DNA_STATUSES = frozenset({"complete", "partial", "blocked"})
PUBLISHABLE_STATUSES = frozenset({"complete", "partial"})

DNA_DIMENSIONS = (
    "narrative_mechanism",
    "experience_arc",
    "event_constraints",
    "editing_structure",
    "editing_function",
    "free_slots",
)

NODE_KINDS = frozenset({
    "information_state", "proposition", "evidence_role", "scope_boundary",
    "resolution_state",
})
MECHANISMS = frozenset({
    "establishes", "supports", "contradicts", "reframes", "qualifies",
    "enables", "reveals", "accumulates", "precedes", "orders_disclosure",
})
EXPERIENCE_STATE_ROLES = frozenset({
    "establish", "develop", "reveal", "reframe", "qualify", "release",
    "other",
})
EVENT_OBLIGATIONS = frozenset({"required", "preferred", "prohibited"})
EDITING_LEVELS = frozenset({
    "structural", "semantic_function", "semantic_pattern",
})
EDITING_OBLIGATIONS = frozenset({"required", "preferred", "advisory"})
EDITING_DIMENSIONS = frozenset({
    "pace", "shot_duration", "transition_density", "information_density",
    "information_function", "pattern_type",
})
EDITING_OPERATORS = frozenset({
    "increases", "decreases", "holds", "contrasts", "accumulates",
    "reveals", "qualifies", "bridges", "free",
})
SOURCE_STRENGTHS = frozenset({
    "measured", "audited_semantic", "unaudited_semantic",
})
TRANSFER_POLICIES = frozenset({
    "preserve_relation", "preserve_function", "free_realization",
})
FREE_SLOT_KINDS = frozenset({
    "domain", "character_role_binding", "setting", "surface_event",
    "evidence_form", "tone", "visual_motif",
})
ANTI_INVARIANT_CATEGORIES = frozenset({
    "entity", "domain", "setting", "physical_form", "literal_event",
    "wording", "visual_motif",
})

AUDIT_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "dna_status", "abstraction_confidence",
    "unsupported_dimensions", "input_artifacts", "mechanism_graph",
    "experience_arc", "event_constraints", "editing_constraints",
    "free_slots", "source_bindings", "anti_invariants",
    "validation_record", "artifact_sha",
})

SPEC_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "spec_id", "dna_status", "available_dimensions",
    "unsupported_dimensions", "narrative_control", "editing_control",
    "free_slots", "downstream_contract", "artifact_sha",
})

VALIDATION_RECORD_KEYS = frozenset({
    "grounding_passed", "relation_entailment_passed",
    "surface_binding_confined_to_audit", "editing_promotion_policy_passed",
    "publish_whitelist_passed",
})
