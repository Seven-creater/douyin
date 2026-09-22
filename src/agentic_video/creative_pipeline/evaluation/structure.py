"""Hard structure and reference-boundary validators for creative planning."""
from __future__ import annotations

import re
from typing import Any

from src.agentic_video.creative_pipeline.contracts import ContractError
from src.agentic_video.creative_pipeline.planning.story import (
    validate_story_blueprint_schema,
)
from src.agentic_video.creative_pipeline.planning.theme import (
    validate_theme_candidate_schema,
)


_FORBIDDEN_KEYS = frozenset({
    "reference_video", "reference_media", "accepted_claims",
    "accepted_events", "claim_ledger", "narrative_interpretation",
    "editing_grammar", "audit_annex", "source_bindings", "creative_dna",
    "raw_response", "model_trace",
})
_REFERENCE_SOURCE_ID = re.compile(
    r"\b(?:S\d+_[VAT]\d+|TEXT_[A-Za-z0-9_]+|SHOT_\d+|CLAIM_\d+)\b",
    re.IGNORECASE,
)


def validate_creative_boundary(value: object) -> None:
    """Reject known reference-zone fields and source IDs recursively."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ContractError("reference_boundary_field", key)
            validate_creative_boundary(item)
    elif isinstance(value, list):
        for item in value:
            validate_creative_boundary(item)
    elif isinstance(value, str) and _REFERENCE_SOURCE_ID.search(value):
        raise ContractError("reference_source_id_leak", value)


def validate_theme_structure(value: dict[str, Any], *,
                             structure_sha: str) -> None:
    validate_creative_boundary(value)
    validate_theme_candidate_schema(value, structure_sha=structure_sha)
    bindings = value["structure_bindings"]
    state_values = [
        bindings["I0_PRIOR_INTERPRETATION"],
        bindings["E1_NEW_INFORMATION"],
        bindings["I1_UPDATED_INTERPRETATION"],
    ]
    if len(set(item.strip().lower() for item in state_values)) != 3:
        raise ContractError("theme_information_roles_not_distinct")


def validate_story_structure(value: dict[str, Any], *,
                             structure_sha: str,
                             theme_id: str) -> None:
    validate_creative_boundary(value)
    validate_story_blueprint_schema(
        value, structure_sha=structure_sha, theme_id=theme_id)
