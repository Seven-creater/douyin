"""Static cross-domain transfer cases for Creative Structure Plan review."""
from __future__ import annotations

from typing import Any

from src.agentic_video.creative_structure_v1.schema import (
    TRANSFER_TEST_SCHEMA_VERSION,
)
from src.agentic_video.creative_structure_v1.validator import validate_structure_spec
from src.agentic_video.manifest import json_hash

TRANSFER_PROFILES = (
    {
        "case_id": "TRANSFER_A",
        "case_type": "cross_domain_positive",
        "binding_profile": {
            "domain": "software incident diagnosis",
            "entity_role": "on-call engineer",
            "evidence_form": "system telemetry",
        },
        "expected": "preserve",
    },
    {
        "case_id": "TRANSFER_B",
        "case_type": "cross_domain_positive",
        "binding_profile": {
            "domain": "ecological field research",
            "entity_role": "field researcher",
            "evidence_form": "repeatable observation",
        },
        "expected": "preserve",
    },
    {
        "case_id": "TRANSFER_C",
        "case_type": "cross_domain_positive",
        "binding_profile": {
            "domain": "music ensemble rehearsal",
            "entity_role": "section leader",
            "evidence_form": "audible performance evidence",
        },
        "expected": "preserve",
    },
    {
        "case_id": "SURFACE_LURE_NEGATIVE",
        "case_type": "surface_lure_negative",
        "binding_profile": {
            "domain": "public competition",
            "entity_role": "competitor",
            "evidence_form": "unrelated result",
        },
        "expected": "reject",
        "violation": "surface similarity without the required structural relation",
    },
)


def audit_transfer_profiles() -> list[dict[str, Any]]:
    """Return profile copies for the independent audit payload."""
    return [dict(row) for row in TRANSFER_PROFILES]


def build_transfer_test_plan(spec: dict[str, Any]) -> dict[str, Any]:
    """Create review cases without generating or judging a new story."""
    validate_structure_spec(spec)
    relation_ids = [
        row["relation_id"]
        for row in (spec.get("structural_schema") or {}).get("relations", [])
    ]
    mandatory = spec["downstream_contract"]["required_constraint_ids"]
    cases = []
    for template in TRANSFER_PROFILES:
        case = {
            **template,
            "binding_profile": dict(template["binding_profile"]),
            "relation_ids_to_hold_constant": relation_ids,
            "constraint_ids_to_check": list(mandatory),
            "decision": None,
            "reason": None,
        }
        cases.append(case)
    result = {
        "schema_version": TRANSFER_TEST_SCHEMA_VERSION,
        "spec_sha": spec["artifact_sha"],
        "status": "PENDING_INDEPENDENT_REVIEW",
        "cases": cases,
    }
    result["artifact_sha"] = json_hash(result)
    return result
