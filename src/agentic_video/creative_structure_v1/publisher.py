"""Whitelist publisher for the Creative Structure Plan reference boundary."""
from __future__ import annotations

import re
from typing import Any

from src.agentic_video.creative_structure_v1.schema import (
    DIMENSIONS,
    SPEC_SCHEMA_VERSION,
    VALIDATION_RECORD_KEYS,
)
from src.agentic_video.creative_structure_v1.validator import (
    StructureValidationError,
    available_dimensions,
    validate_structure_audit,
    validate_structure_spec,
)
from src.agentic_video.manifest import json_hash

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{4,}|[\u4e00-\u9fff]{2,}")
_SURFACE_STOPWORDS = {
    "abstract", "about", "after", "again", "available", "bears", "becomes",
    "before", "binding",
    "constraint", "domain", "earlier", "effect", "events", "evidence",
    "example", "increases", "information", "interpretation", "later", "provides",
    "reference", "relation", "replaceable", "sequence", "source", "state",
    "story", "structure", "their", "there", "these", "using", "video",
    "which", "作品", "信息", "参考", "来源", "示例",
}


def _public_structural(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "elements": [
            {key: row[key] for key in ("element_id", "kind", "abstract_role")}
            for row in value["elements"]
        ],
        "relations": [
            {key: row[key] for key in (
                "relation_id", "relation_kind", "predicate", "arguments",
                "preconditions", "effects",
            )}
            for row in value["relations"]
        ],
    }


def _public_states(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in (
            "state_id", "order", "available_element_ids",
            "transition_from_state_id", "trigger_relation_ids",
        )}
        for row in rows
    ]


def _public_generation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in (
            "constraint_id", "obligation", "constraint_type",
            "requires_relation_ids", "rule",
        )}
        for row in rows
    ]


def _public_editing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in (
            "constraint_id", "obligation", "dimension", "operator",
            "applies_to_state_ids", "applies_to_relation_ids", "transfer_policy",
        )}
        for row in rows
    ]


def _public_slots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in (
            "slot_id", "slot_type", "bound_by", "constraints",
        )}
        for row in rows
    ]


def _surface_terms(audit: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for binding in audit["audit_annex"]["source_bindings"]:
        for match in _WORD_RE.findall(binding["reference_specific_summary"]):
            lowered = match.lower()
            if lowered not in _SURFACE_STOPWORDS:
                terms.add(lowered)
    return terms


def _assert_no_reference_surface(spec: dict[str, Any],
                                 audit: dict[str, Any]) -> None:
    text = repr(spec).lower()
    leaked = sorted(term for term in _surface_terms(audit) if term in text)
    if leaked:
        raise StructureValidationError("reference_surface_leak", ",".join(leaked))


def publish_structure_spec(audit: dict[str, Any], *, spec_id: str) -> dict[str, Any]:
    """Publish only the domain-independent IR fields; never rewrite them."""
    validate_structure_audit(audit)
    if audit["plan_status"] == "blocked":
        raise StructureValidationError("plan_blocked")
    if not isinstance(spec_id, str) or not spec_id.strip():
        raise StructureValidationError("spec_id_invalid")
    failed = sorted(
        key for key in VALIDATION_RECORD_KEYS
        if not audit["validation_record"][key]
    )
    if failed:
        raise StructureValidationError("audit_validation_incomplete", ",".join(failed))

    available = available_dimensions(audit)
    generation = _public_generation(audit["generation_constraints"])
    editing = _public_editing(audit["editing_schema"]["constraints"])
    slots = _public_slots(audit["binding_slots"])
    story_authorized = all(
        audit["dimension_status"][dimension] == "supported"
        for dimension in ("structural_schema", "generation_constraints")
    )
    mandatory = [
        row["constraint_id"] for row in generation + editing
        if row["obligation"] in {"required", "prohibited"}
    ]
    spec: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "spec_id": spec_id,
        "plan_status": audit["plan_status"],
        "available_dimensions": available,
        "dimension_status": {
            dimension: audit["dimension_status"][dimension]
            for dimension in DIMENSIONS
        },
        "structural_schema": _public_structural(audit["structural_schema"]),
        "information_state_arc": _public_states(audit["information_state_arc"]),
        "generation_constraints": generation,
        "editing_schema": {"constraints": editing},
        "binding_slots": slots,
        "downstream_contract": {
            "required_constraint_ids": mandatory,
            "variable_slot_ids": [row["slot_id"] for row in slots],
            "authorized_stages": (
                ["theme", "story", "screenplay"] if story_authorized else []),
            "reference_context_allowed": False,
        },
    }
    _assert_no_reference_surface(spec, audit)
    spec["artifact_sha"] = json_hash(spec)
    validate_structure_spec(spec)
    return spec
