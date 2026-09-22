"""Frozen constants for the Creative Structure Plan v1 IR."""
from __future__ import annotations

AUDIT_SCHEMA_VERSION = "creative_structure_audit_v1"
SPEC_SCHEMA_VERSION = "creative_structure_spec_v1"
INDEPENDENT_AUDIT_SCHEMA_VERSION = "creative_structure_independent_audit_v1"
TRANSFER_TEST_SCHEMA_VERSION = "creative_structure_transfer_test_v1"

PLAN_STATUSES = frozenset({"complete", "partial", "blocked"})
DIMENSION_STATES = frozenset({"supported", "insufficient", "not_applicable"})
DIMENSIONS = (
    "structural_schema",
    "information_state_arc",
    "generation_constraints",
    "editing_structure",
    "editing_function",
    "binding_slots",
)

ELEMENT_KINDS = frozenset({
    "information_state",
    "proposition",
    "evidence_role",
    "scope_boundary",
    "resolution_state",
})
RELATION_KINDS = frozenset({
    "causal", "evidential", "temporal", "scope", "disclosure",
})
GENERATION_OBLIGATIONS = frozenset({"required", "preferred", "prohibited"})
GENERATION_CONSTRAINT_TYPES = frozenset({
    "relational", "ordering", "scope", "exclusion",
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

BINDING_SLOT_TYPES = frozenset({
    "domain", "entity_role", "setting", "surface_realization",
    "evidence_form", "tone", "visual_motif",
})
ANTI_INVARIANT_CATEGORIES = frozenset({
    "entity", "domain", "setting", "physical_form", "literal_event",
    "wording", "visual_motif",
})

VALIDATION_RECORD_KEYS = frozenset({
    "grounding_passed",
    "relation_entailment_passed",
    "abstraction_boundary_passed",
    "transferability_passed",
    "editing_promotion_policy_passed",
    "publish_whitelist_passed",
})

AUDIT_TOP_LEVEL_KEYS = frozenset({
    "schema_version",
    "plan_status",
    "abstraction_confidence",
    "dimension_status",
    "input_artifacts",
    "structural_schema",
    "information_state_arc",
    "generation_constraints",
    "editing_schema",
    "binding_slots",
    "audit_annex",
    "validation_record",
    "artifact_sha",
})

SPEC_TOP_LEVEL_KEYS = frozenset({
    "schema_version",
    "spec_id",
    "plan_status",
    "available_dimensions",
    "dimension_status",
    "structural_schema",
    "information_state_arc",
    "generation_constraints",
    "editing_schema",
    "binding_slots",
    "downstream_contract",
    "artifact_sha",
})
