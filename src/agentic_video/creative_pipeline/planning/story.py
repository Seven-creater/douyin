"""Story Blueprint v1 fixtures and schema entry point."""
from __future__ import annotations

from typing import Any

from src.agentic_video.creative_pipeline.contracts import (
    validate_story_blueprint,
)


STORY_PER_THEME_N = 5
STORY_SELECT_K = 3


def _relation(statement: str) -> dict[str, str]:
    return {
        "prior": "I0_PRIOR_INTERPRETATION",
        "evidence": "E1_NEW_INFORMATION",
        "updated": "I1_UPDATED_INTERPRETATION",
        "statement": statement,
    }


def build_fake_story_blueprints(
        structure_sha: str, theme: dict[str, Any]) -> list[dict[str, Any]]:
    """Build five deterministic blueprints for one selected Theme fixture."""
    candidates = []
    for index in range(1, STORY_PER_THEME_N + 1):
        suffix = f"{theme['theme_id']}_{index:02d}"
        prior = f"EVENT_PRIOR_{suffix}"
        evidence = f"EVENT_EVIDENCE_{suffix}"
        updated = f"EVENT_UPDATED_{suffix}"
        relation_id = f"REL_UPDATE_{suffix}"
        candidates.append({
            "schema_version": "story_blueprint_v1",
            "blueprint_id": f"FAKE_BLUEPRINT_{suffix}",
            "theme_id": theme["theme_id"],
            "parent_structure_sha": structure_sha,
            "logline": (
                "A deterministic fixture instantiates an information update."),
            "characters": [
                {"character_id": "ENTITY_01", "role": "observer"}],
            "setting": theme["binding_slots"]["B_SETTING"],
            "goal": "resolve the meaning of the available evidence",
            "stakes": "the initial interpretation may remain inaccurate",
            "events": [
                {"event_id": prior, "role": "prior_interpretation"},
                {"event_id": evidence, "role": "new_information"},
                {"event_id": updated, "role": "updated_interpretation"},
            ],
            "event_relations": [{
                "relation_id": relation_id,
                "type": "information_update",
                "prior_event_id": prior,
                "evidence_event_id": evidence,
                "updated_event_id": updated,
            }],
            "structure_bindings": {
                "I0_PRIOR_INTERPRETATION": prior,
                "E1_NEW_INFORMATION": evidence,
                "I1_UPDATED_INTERPRETATION": updated,
                "R1_INFORMATION_UPDATE": _relation(relation_id),
            },
            "production_assumptions": [
                "contract fixture; no media production"],
        })
    return candidates


def validate_story_blueprint_schema(value: dict[str, Any], *,
                                    structure_sha: str,
                                    theme_id: str) -> None:
    validate_story_blueprint(
        value, structure_sha=structure_sha, theme_id=theme_id)
